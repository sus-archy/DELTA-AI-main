from __future__ import annotations

import argparse
import json
import torch
import numpy as np
from pathlib import Path


def build_embeddings(
    config_path: str | Path | None = None,
    encoders: list[str] | None = None,
    max_rows: int | None = None,
) -> dict:
    import pandas as pd
    from tqdm.auto import tqdm

    from .model_registry import get_encoder_spec, load_encoder_model, load_tokenizer
    from .settings import bootstrap_environment, load_config

    paths = bootstrap_environment(config_path)
    config = load_config(config_path)
    encoders = encoders or config["models"]["candidate_encoders"]

    dataset_dir = paths.artifacts_dir / "datasets"
    dataset_manifest_path = dataset_dir / "manifest.json"
    with dataset_manifest_path.open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    output_root = paths.artifacts_dir / "embeddings"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest: dict[str, dict[str, str]] = json.load(handle)
    else:
        manifest = {}

    for encoder_name in encoders:
        encoder_dir = output_root / encoder_name
        encoder_dir.mkdir(parents=True, exist_ok=True)
        manifest[encoder_name] = {}
        for split_name in ["train", "validation", "test"]:
            frame = pd.read_parquet(dataset_dir / f"{split_name}.parquet")
            if max_rows:
                frame = frame.head(max_rows)
            embeddings = encode_texts(frame["text_input"].fillna("").tolist(), encoder_name, paths.models_dir)
            if embeddings.shape[0] != len(frame):
                raise ValueError(
                    f"Embedding row count mismatch for {encoder_name}/{split_name}: "
                    f"{embeddings.shape[0]} vs {len(frame)}"
                )
            split_path = encoder_dir / f"{split_name}.npy"
            np.save(split_path, embeddings)
            manifest[encoder_name][split_name] = {
                "path": str(split_path),
                "rows": int(len(frame)),
                "dim": int(embeddings.shape[1]),
            }

        manifest[encoder_name]["dataset_manifest"] = {
            "requested_limit": dataset_manifest.get("requested_limit"),
            "source_row_count": dataset_manifest.get("source_row_count"),
            "rows": dataset_manifest.get("rows", {}),
        }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(
    texts: list[str],
    encoder_name: str,
    cache_dir: str | Path,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> "np.ndarray":
    import numpy as np
    import torch

    from .model_registry import get_encoder_spec, load_encoder_model, load_tokenizer

    spec = get_encoder_spec(encoder_name)
    tokenizer = load_tokenizer(encoder_name, cache_dir)
    model = load_encoder_model(encoder_name, cache_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = max_length or spec.default_max_length
    batch_size = batch_size or spec.default_batch_size
    embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline embeddings for Delta IOC datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoders", nargs="*", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    manifest = build_embeddings(args.config, args.encoders, args.max_rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()