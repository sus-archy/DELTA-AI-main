from __future__ import annotations

import argparse
import json
from pathlib import Path

from .settings import bootstrap_environment

import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .calibration import apply_temperature, fit_temperature, logits_to_proba, write_prediction_artifact
from .class_balance import (
    compute_class_prior_array,
    compute_class_weight_array,
    compute_effective_number_weight_array,
    summarize_class_balance,
)
from .labels import SEVERITY_ORDER
from .losses import balanced_softmax_loss, class_balanced_focal_loss, focal_loss
from .metrics import summarize_classification, write_metrics
from .model_registry import get_encoder_spec, load_sequence_classifier, load_tokenizer
from .seed import set_global_seed

LABEL_NAMES = SEVERITY_ORDER


class WeightedClassificationTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: torch.Tensor | None = None,
        class_priors: torch.Tensor | None = None,
        loss_name: str = "ce",
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.class_priors = class_priors
        self.loss_name = loss_name
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        class_weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        class_priors = self.class_priors.to(logits.device) if self.class_priors is not None else None
        if self.loss_name == "weighted_ce":
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
        elif self.loss_name == "balanced_softmax":
            if class_priors is None:
                raise RuntimeError("balanced_softmax requires class priors.")
            loss = balanced_softmax_loss(logits, labels, class_priors)
        elif self.loss_name == "focal":
            loss = focal_loss(logits, labels, gamma=self.focal_gamma, class_weights=None)
        elif self.loss_name == "cb_focal":
            if class_weights is None:
                raise RuntimeError("cb_focal requires class weights.")
            loss = class_balanced_focal_loss(logits, labels, class_weights, gamma=self.focal_gamma)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        if return_outputs:
            return loss, outputs
        return loss


def _tokenize_dataset(frame: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame[["text_input", "severity_id"]], preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(batch["text_input"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.rename_column("severity_id", "labels")
    return dataset


def _compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = proba.argmax(axis=1)
    summary = summarize_classification(
        labels,
        pred,
        proba,
        label_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        proba_labels=list(range(len(LABEL_NAMES))),
    )
    return {
        "macro_f1": summary["macro_f1"],
        "weighted_f1": summary["weighted_f1"],
        "balanced_accuracy": summary["balanced_accuracy"],
        "ece": summary.get("ece", 0.0),
        "brier": summary.get("brier", 0.0),
    }


def _normalize_balance_strategy(class_balance: str) -> str:
    return "inverse" if class_balance == "balanced" else class_balance


def _resolve_run_dir(root: Path, default_name: str, run_name: str | None) -> Path:
    run_dir = root / (run_name or default_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _sample_frame_stratified(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows >= len(frame):
        return frame.copy()
    grouped = []
    total = len(frame)
    for _, group in frame.groupby("severity_label", dropna=False):
        n = max(1, round(max_rows * len(group) / total))
        grouped.append(group.sample(n=min(n, len(group)), random_state=seed))
    sampled = pd.concat(grouped, axis=0)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    elif len(sampled) < max_rows:
        remaining = frame.drop(sampled.index, errors="ignore")
        needed = min(max_rows - len(sampled), len(remaining))
        if needed > 0:
            sampled = pd.concat([sampled, remaining.sample(n=needed, random_state=seed)], axis=0)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _infer_lora_target_modules(model) -> list[str]:
    candidates = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if "classifier" in name or "score" in name or "pooler" in name:
                continue
            candidates.add(leaf)
    preferred = ["query", "key", "value", "dense", "Wqkv", "Wo", "q_proj", "k_proj", "v_proj", "o_proj"]
    selected = [item for item in preferred if item in candidates]
    if selected:
        return selected
    return sorted(candidates)


def fine_tune(
    config_path: str | Path | None,
    encoder_name: str,
    peft_mode: str,
    max_train_rows: int | None,
    num_epochs: int,
    class_balance: str,
    loss_name: str,
    focal_gamma: float,
    seed: int,
    run_name: str | None,
    calibration_method: str = "none",
    max_length_override: int | None = None,
) -> dict:
    set_global_seed(seed)
    paths = bootstrap_environment(config_path)
    spec = get_encoder_spec(encoder_name)
    class_balance = _normalize_balance_strategy(class_balance)
    train_frame = pd.read_parquet(paths.artifacts_dir / "datasets" / "train.parquet")
    validation_frame = pd.read_parquet(paths.artifacts_dir / "datasets" / "validation.parquet")
    test_frame = pd.read_parquet(paths.artifacts_dir / "datasets" / "test.parquet")
    if max_train_rows:
        train_frame = _sample_frame_stratified(train_frame, max_train_rows, seed)
        validation_frame = _sample_frame_stratified(
            validation_frame,
            max(1000, min(max_train_rows // 5, len(validation_frame))),
            seed,
        )
        test_frame = _sample_frame_stratified(
            test_frame,
            max(1000, min(max_train_rows // 5, len(test_frame))),
            seed,
        )

    tokenizer = load_tokenizer(encoder_name, paths.models_dir)
    model = load_sequence_classifier(encoder_name, paths.models_dir, num_labels=len(LABEL_NAMES))
    if peft_mode == "lora":
        if get_peft_model is None:
            raise RuntimeError("peft is not installed but --peft lora was requested.")
        target_modules = _infer_lora_target_modules(model)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    effective_max_length = int(max_length_override or spec.default_max_length)
    train_dataset = _tokenize_dataset(train_frame, tokenizer, effective_max_length)
    validation_dataset = _tokenize_dataset(validation_frame, tokenizer, effective_max_length)
    test_dataset = _tokenize_dataset(test_frame, tokenizer, effective_max_length)
    class_weights = None
    if loss_name in {"weighted_ce", "cb_focal"} and class_balance != "none":
        if class_balance == "effective":
            weight_array = compute_effective_number_weight_array(
                train_frame["severity_id"],
                num_classes=len(LABEL_NAMES),
            )
        else:
            weight_array = compute_class_weight_array(train_frame["severity_id"], num_classes=len(LABEL_NAMES))
        class_weights = torch.tensor(
            weight_array,
            dtype=torch.float32,
        )
    class_priors = torch.tensor(
        compute_class_prior_array(train_frame["severity_id"], num_classes=len(LABEL_NAMES)),
        dtype=torch.float32,
    )

    run_dir = _resolve_run_dir(
        paths.models_dir / "finetune",
        f"{encoder_name}_{peft_mode}_{loss_name}_{class_balance}",
        run_name,
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5 if peft_mode == "none" else 5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
    )

    trainer = WeightedClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
        class_priors=class_priors,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
    )
    trainer.train()
    validation_predictions = trainer.predict(validation_dataset)
    predictions = trainer.predict(test_dataset)
    validation_logits = validation_predictions.predictions
    logits = predictions.predictions
    proba = logits_to_proba(logits)
    uncalibrated_summary = summarize_classification(
        test_frame["severity_id"].to_numpy(),
        proba.argmax(axis=1),
        proba,
        label_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        proba_labels=list(range(len(LABEL_NAMES))),
    )
    temperature = None
    if calibration_method == "temperature":
        temperature = fit_temperature(validation_logits, validation_frame["severity_id"].to_numpy())
        proba = apply_temperature(logits, temperature)
    pred = proba.argmax(axis=1)
    metrics = summarize_classification(
        test_frame["severity_id"].to_numpy(),
        pred,
        proba,
        label_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        proba_labels=list(range(len(LABEL_NAMES))),
    )
    metrics["train_class_balance"] = summarize_class_balance(
        train_frame["severity_id"],
        num_classes=len(LABEL_NAMES),
        label_names=LABEL_NAMES,
        strategy=class_balance,
    )
    metrics["loss_name"] = loss_name
    metrics["class_balance_strategy"] = class_balance
    metrics["focal_gamma"] = focal_gamma
    metrics["seed"] = seed
    metrics["calibration_method"] = calibration_method
    metrics["max_length"] = effective_max_length
    if temperature is not None:
        metrics["temperature"] = float(temperature)
        metrics["uncalibrated_ece"] = uncalibrated_summary.get("ece")
        metrics["uncalibrated_brier"] = uncalibrated_summary.get("brier")
    write_prediction_artifact(
        run_dir,
        "test",
        test_frame["severity_id"].to_numpy(),
        proba,
        logits=logits,
    )
    validation_proba = (
        apply_temperature(validation_logits, temperature)
        if temperature is not None
        else logits_to_proba(validation_logits)
    )
    write_prediction_artifact(
        run_dir,
        "validation",
        validation_frame["severity_id"].to_numpy(),
        validation_proba,
        logits=validation_logits,
    )
    write_metrics(run_dir / "metrics.json", metrics)
    trainer.save_model(str(run_dir / "best_model"))
    return {"run_dir": str(run_dir), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a sequence classifier for Delta IOC severity.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoder", default="modernbert-base")
    parser.add_argument("--peft", choices=["none", "lora"], default="lora")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--class-balance", choices=["none", "balanced", "inverse", "effective"], default="effective")
    parser.add_argument(
        "--loss",
        choices=["ce", "weighted_ce", "balanced_softmax", "focal", "cb_focal"],
        default="balanced_softmax",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--calibration", choices=["none", "temperature"], default="none")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    result = fine_tune(
        args.config,
        args.encoder,
        args.peft,
        args.max_train_rows,
        args.epochs,
        args.class_balance,
        args.loss,
        args.focal_gamma,
        args.seed,
        args.run_name,
        args.calibration,
        args.max_length,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
