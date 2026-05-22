from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment, misc]

from .calibration import (
    apply_temperature,
    fit_temperature,
    proba_to_logits,
    write_prediction_artifact,
)
from .class_balance import (
    build_sample_weights,
    compute_class_weight_map,
    compute_effective_number_weight_map,
    summarize_class_balance,
)
from .data import SEVERITY_ORDER
from .labels import SEVERITY_ORDER as LABEL_NAMES
from .features import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC
from .metrics import summarize_classification, write_metrics
from .seed import set_global_seed
from .settings import bootstrap_environment

try:
    from .viz import generate_all_plots
except ImportError:
    generate_all_plots = None  # type: ignore[assignment]

LABEL_NAMES = SEVERITY_ORDER


def _load_split_frame(paths, split_name: str) -> pd.DataFrame:
    return pd.read_parquet(paths.artifacts_dir / "datasets" / f"{split_name}.parquet")


def _resolve_run_dir(root: Path, default_name: str, run_name: str | None) -> Path:
    run_dir = root / (run_name or default_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalize_balance_strategy(class_balance: str) -> str:
    return "inverse" if class_balance == "balanced" else class_balance


def _build_logreg_preprocessor(feature_variant: str) -> ColumnTransformer:
    transformers = [
        (
            "text_word",
            TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True),
            "text_input",
        ),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS_CATEGORICAL),
        ("numeric", Pipeline([("scale", StandardScaler(with_mean=False))]), FEATURE_COLUMNS_NUMERIC),
    ]
    if feature_variant == "value_char":
        transformers.insert(
            1,
            (
                "value_char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 6),
                    min_df=2,
                    max_features=40000,
                    sublinear_tf=True,
                ),
                "value",
            ),
        )
    elif feature_variant != "default":
        raise KeyError(f"Unsupported logreg feature variant '{feature_variant}'.")
    return ColumnTransformer(transformers=transformers)


def _build_logreg_pipeline(
    class_weight: dict[int, float] | None,
    seed: int,
    feature_variant: str,
    c_value: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", _build_logreg_preprocessor(feature_variant)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight=class_weight,
                    random_state=seed,
                    C=c_value,
                ),
            ),
        ]
    )


def _load_hybrid_matrix(paths, encoder_name: str):
    train = _load_split_frame(paths, "train")
    validation = _load_split_frame(paths, "validation")
    test = _load_split_frame(paths, "test")
    retrieval_dir = paths.artifacts_dir / "retrieval" / encoder_name
    embeddings_dir = paths.artifacts_dir / "embeddings" / encoder_name

    train_base = pd.get_dummies(train[FEATURE_COLUMNS_CATEGORICAL], dummy_na=True)

    def build(frame: pd.DataFrame, split_name: str):
        base = pd.get_dummies(frame[FEATURE_COLUMNS_CATEGORICAL], dummy_na=True)
        base = base.reindex(columns=train_base.columns, fill_value=0)
        base = pd.concat([base, frame[FEATURE_COLUMNS_NUMERIC].fillna(0)], axis=1)
        embeddings = pd.DataFrame(np.load(embeddings_dir / f"{split_name}.npy"))
        retrieval = pd.read_parquet(retrieval_dir / f"{split_name}.parquet")
        if len(embeddings) != len(frame):
            raise ValueError(
                f"Embedding row count mismatch for split '{split_name}' and encoder '{encoder_name}': "
                f"{len(embeddings)} vs {len(frame)}"
            )
        if len(retrieval) != len(frame):
            raise ValueError(
                f"Retrieval row count mismatch for split '{split_name}' and encoder '{encoder_name}': "
                f"{len(retrieval)} vs {len(frame)}"
            )
        matrix = pd.concat([base.reset_index(drop=True), embeddings, retrieval.reset_index(drop=True)], axis=1)
        return matrix, frame["severity_id"].to_numpy()

    return build(train, "train"), build(validation, "validation"), build(test, "test")


def _attach_balance_summary(metrics: dict, y_train, class_balance: str) -> dict:
    metrics["class_balance_strategy"] = class_balance
    metrics["train_class_balance"] = summarize_class_balance(
        y_train,
        num_classes=len(LABEL_NAMES),
        label_names=LABEL_NAMES,
        strategy=class_balance,
    )
    return metrics


def _majority_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    global_proba = (
        train["severity_id"].value_counts(normalize=True).reindex(range(len(LABEL_NAMES)), fill_value=0.0).to_numpy(dtype=float)
    )
    lookup: dict[tuple, np.ndarray] = {}
    for key, group in train.groupby(columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        lookup[key] = (
            group["severity_id"].value_counts(normalize=True).reindex(range(len(LABEL_NAMES)), fill_value=0.0).to_numpy(dtype=float)
        )
    proba_rows = []
    for _, row in test.iterrows():
        key = tuple(row[column] for column in columns)
        proba_rows.append(lookup.get(key, global_proba))
    proba = np.vstack(proba_rows)
    pred = proba.argmax(axis=1)
    return pred, proba


def train_lookup_majority(
    paths,
    columns: list[str],
    seed: int = 42,
    run_name: str | None = None,
) -> tuple[dict, str]:
    set_global_seed(seed)
    train = _load_split_frame(paths, "train")
    test = _load_split_frame(paths, "test")
    pred, proba = _majority_predict(train, test, columns)
    metrics = summarize_classification(
        y_true=test["severity_id"].to_numpy(),
        y_pred=pred,
        y_proba=proba,
        label_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        proba_labels=list(range(len(LABEL_NAMES))),
    )
    metrics["seed"] = seed
    metrics["baseline_columns"] = columns
    metrics = _attach_balance_summary(metrics, train["severity_id"], "none")
    default_name = "_".join(columns) + "_majority"
    run_dir = _resolve_run_dir(paths.artifacts_dir / "runs", default_name, run_name)
    write_metrics(run_dir / "metrics.json", metrics)
    return metrics, str(run_dir)


def train_logreg_tfidf(
    paths,
    class_balance: str,
    seed: int = 42,
    run_name: str | None = None,
    feature_variant: str = "default",
    tune_c: bool = False,
    c_grid: list[float] | None = None,
    calibration_method: str = "none",
) -> tuple[dict, str]:
    set_global_seed(seed)
    train = _load_split_frame(paths, "train")
    validation = _load_split_frame(paths, "validation")
    test = _load_split_frame(paths, "test")
    class_balance = _normalize_balance_strategy(class_balance)
    class_weight = None
    if class_balance == "inverse":
        class_weight = compute_class_weight_map(train["severity_id"], num_classes=len(LABEL_NAMES))
    elif class_balance == "effective":
        class_weight = compute_effective_number_weight_map(train["severity_id"], num_classes=len(LABEL_NAMES))

    selected_c = 1.0
    c_sweep = []
    if tune_c:
        candidate_grid = c_grid or [0.5, 1.0, 2.0, 4.0, 8.0]
        best_score = -1.0
        best_ece = float("inf")
        for candidate_c in candidate_grid:
            candidate_pipeline = _build_logreg_pipeline(class_weight, seed, feature_variant, candidate_c)
            candidate_pipeline.fit(train, train["severity_id"])
            validation_proba = candidate_pipeline.predict_proba(validation)
            validation_pred = candidate_pipeline.predict(validation)
            validation_metrics = summarize_classification(
                y_true=validation["severity_id"].to_numpy(),
                y_pred=validation_pred,
                y_proba=validation_proba,
                label_names=LABEL_NAMES,
                labels=list(range(len(LABEL_NAMES))),
                proba_labels=candidate_pipeline.named_steps["classifier"].classes_.tolist(),
            )
            c_sweep.append(
                {
                    "c": float(candidate_c),
                    "macro_f1": validation_metrics["macro_f1"],
                    "balanced_accuracy": validation_metrics["balanced_accuracy"],
                    "ece": validation_metrics.get("ece"),
                    "brier": validation_metrics.get("brier"),
                }
            )
            score = validation_metrics["macro_f1"]
            ece = validation_metrics.get("ece", float("inf"))
            if (score > best_score) or (score == best_score and ece < best_ece):
                best_score = score
                best_ece = ece
                selected_c = float(candidate_c)

    fit_frame = pd.concat([train, validation], ignore_index=True) if tune_c else train
    pipeline = _build_logreg_pipeline(class_weight, seed, feature_variant, selected_c)
    pipeline.fit(fit_frame, fit_frame["severity_id"])
    validation_proba_output = None
    test_proba = pipeline.predict_proba(test)
    test_logits = proba_to_logits(test_proba)
    validation_logits = None
    temperature = None
    uncalibrated_metrics = summarize_classification(
        y_true=test["severity_id"].to_numpy(),
        y_pred=pipeline.predict(test),
        y_proba=test_proba,
        label_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        proba_labels=pipeline.named_steps["classifier"].classes_.tolist(),
    )
    if calibration_method == "temperature":
        calibration_pipeline = pipeline if not tune_c else _build_logreg_pipeline(class_weight, seed, feature_variant, selected_c)
        if tune_c:
            calibration_pipeline.fit(train, train["severity_id"])
        validation_proba = calibration_pipeline.predict_proba(validation)
        validation_proba_output = validation_proba
        validation_logits = proba_to_logits(validation_proba)
        temperature = fit_temperature(validation_logits, validation["severity_id"].to_numpy())
        test_proba = apply_temperature(test_logits, temperature)
    else:
        validation_proba_output = pipeline.predict_proba(validation)
        validation_logits = proba_to_logits(validation_proba_output)
    test_pred = test_proba.argmax(axis=1)
    classifier_classes = pipeline.named_steps["classifier"].classes_.tolist()
    metrics = summarize_classification(
        y_true=test["severity_id"].to_numpy(),
        y_pred=test_pred,
        y_proba=test_proba,
        label_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        proba_labels=classifier_classes,
    )
    metrics["seed"] = seed
    metrics["feature_variant"] = feature_variant
    metrics["selected_c"] = float(selected_c)
    metrics["tune_c"] = bool(tune_c)
    metrics["calibration_method"] = calibration_method
    if c_sweep:
        metrics["c_sweep"] = c_sweep
    if temperature is not None:
        metrics["temperature"] = float(temperature)
        metrics["uncalibrated_ece"] = uncalibrated_metrics.get("ece")
        metrics["uncalibrated_brier"] = uncalibrated_metrics.get("brier")
    metrics = _attach_balance_summary(metrics, train["severity_id"], class_balance)
    run_suffix = f"logreg_tfidf_{feature_variant}_{class_balance}"
    run_dir = _resolve_run_dir(paths.artifacts_dir / "runs", run_suffix, run_name)
    joblib.dump(pipeline, run_dir / "model.joblib")
    write_prediction_artifact(
        run_dir,
        "test",
        test["severity_id"].to_numpy(),
        test_proba,
        logits=test_logits,
    )
    if validation_logits is not None and validation_proba_output is not None:
        stored_validation_proba = (
            apply_temperature(validation_logits, temperature)
            if calibration_method == "temperature"
            else validation_proba_output
        )
        write_prediction_artifact(
            run_dir,
            "validation",
            validation["severity_id"].to_numpy(),
            stored_validation_proba,
            logits=validation_logits,
        )
    write_metrics(run_dir / "metrics.json", metrics)

    if generate_all_plots is not None:
        try:
            generate_all_plots(
                run_dir,
                test["severity_id"].to_numpy(),
                test_pred,
                test_proba,
                metrics,
                model=pipeline,
            )
        except Exception:
            import traceback
            traceback.print_exc()

    return metrics, str(run_dir)


def train_xgboost_hybrid(
    paths,
    encoder_name: str,
    class_balance: str,
    seed: int = 42,
    run_name: str | None = None,
) -> tuple[dict, str]:
    set_global_seed(seed)
    (x_train, y_train), (_, _), (x_test, y_test) = _load_hybrid_matrix(paths, encoder_name)
    class_balance = _normalize_balance_strategy(class_balance)
    present_labels = sorted(np.unique(y_train).tolist())
    label_to_index = {label: index for index, label in enumerate(present_labels)}
    index_to_label = np.asarray(present_labels, dtype=np.int64)
    y_train_dense = np.asarray([label_to_index[label] for label in y_train], dtype=np.int64)
    sample_weight = None
    if class_balance != "none":
        sample_weight = build_sample_weights(
            y_train_dense,
            num_classes=len(present_labels),
            strategy=class_balance,
        )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(present_labels),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=seed,
    )
    model.fit(x_train, y_train_dense, sample_weight=sample_weight,
              eval_set=[(x_train, y_train_dense), (x_test, y_test)],
              verbose=False)
    proba = model.predict_proba(x_test)
    pred_dense = model.predict(x_test).astype(np.int64)
    pred = index_to_label[pred_dense]
    metrics = summarize_classification(
        y_true=y_test,
        y_pred=pred,
        y_proba=proba,
        label_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        proba_labels=present_labels,
    )
    metrics["seed"] = seed
    metrics = _attach_balance_summary(metrics, y_train, class_balance)
    run_dir = _resolve_run_dir(paths.artifacts_dir / "runs", f"xgboost_hybrid_{encoder_name}_{class_balance}", run_name)
    model.save_model(run_dir / "model.json")
    write_metrics(run_dir / "metrics.json", metrics)

    if generate_all_plots is not None:
        try:
            from .viz import load_training_history, save_training_history
            history = []
            eval_results = model.evals_result()
            n_rounds = len(eval_results.get("validation_0", {}).get("mlogloss", []))
            for i in range(n_rounds):
                entry = {"epoch": i}
                for split_key, split_name in [("validation_0", "train"), ("validation_1", "test")]:
                    if split_key in eval_results:
                        for metric_key in eval_results[split_key]:
                            if i < len(eval_results[split_key][metric_key]):
                                entry[f"{split_name}_{metric_key}"] = float(eval_results[split_key][metric_key][i])
                history.append(entry)
            save_training_history(run_dir, history)
            generate_all_plots(
                run_dir,
                y_test,
                pred,
                proba,
                metrics,
                model=model,
            )
        except Exception:
            import traceback
            traceback.print_exc()

    return metrics, str(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical Delta IOC models.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--model",
        choices=["source_majority", "type_majority", "source_type_majority", "logreg_tfidf", "xgboost_hybrid"],
        default="xgboost_hybrid",
    )
    parser.add_argument("--encoder", default="modernbert-base")
    parser.add_argument(
        "--class-balance",
        choices=["none", "balanced", "inverse", "effective"],
        default="inverse",
    )
    parser.add_argument("--feature-variant", choices=["default", "value_char"], default="default")
    parser.add_argument("--tune-c", action="store_true")
    parser.add_argument("--c-grid", type=float, nargs="*", default=None)
    parser.add_argument("--calibration", choices=["none", "temperature"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    paths = bootstrap_environment(args.config)
    if args.model == "source_majority":
        metrics, run_dir = train_lookup_majority(paths, ["source"], seed=args.seed, run_name=args.run_name)
    elif args.model == "type_majority":
        metrics, run_dir = train_lookup_majority(paths, ["type"], seed=args.seed, run_name=args.run_name)
    elif args.model == "source_type_majority":
        metrics, run_dir = train_lookup_majority(paths, ["source", "type"], seed=args.seed, run_name=args.run_name)
    elif args.model == "logreg_tfidf":
        metrics, run_dir = train_logreg_tfidf(
            paths,
            args.class_balance,
            seed=args.seed,
            run_name=args.run_name,
            feature_variant=args.feature_variant,
            tune_c=args.tune_c,
            c_grid=args.c_grid,
            calibration_method=args.calibration,
        )
    else:
        metrics, run_dir = train_xgboost_hybrid(
            paths,
            args.encoder,
            args.class_balance,
            seed=args.seed,
            run_name=args.run_name,
        )

    print(json.dumps({"run_dir": run_dir, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
