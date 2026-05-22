from __future__ import annotations

import argparse
import json
from pathlib import Path


def _neighbor_features(
    query_embeddings,
    index,
    train_labels,
    train_scores,
    k: int,
):
    import numpy as np
    import pandas as pd

    distances, neighbors = index.search(query_embeddings.astype(np.float32), k)
    rows = []
    for dist_row, idx_row in zip(distances, neighbors):
        valid = idx_row >= 0
        idx_row = idx_row[valid]
        dist_row = dist_row[valid]
        label_values = train_labels[idx_row]
        score_values = train_scores[idx_row]
        rows.append(
            {
                "knn_mean_distance": float(dist_row.mean()) if len(dist_row) else 0.0,
                "knn_mean_teacher_score": float(score_values.mean()) if len(score_values) else 0.0,
                "knn_medium_ratio": float((label_values == "medium").mean()) if len(label_values) else 0.0,
                "knn_high_ratio": float((label_values == "high").mean()) if len(label_values) else 0.0,
                "knn_critical_ratio": float((label_values == "critical").mean()) if len(label_values) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_retrieval(config_path: str | Path | None = None, encoder_name: str = "modernbert-base", k: int = 20) -> dict:
    import faiss
    import numpy as np
    import pandas as pd

    from .settings import bootstrap_environment

    paths = bootstrap_environment(config_path)
    dataset_dir = paths.artifacts_dir / "datasets"
    dataset_manifest_path = dataset_dir / "manifest.json"
    with dataset_manifest_path.open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    embeddings_dir = paths.artifacts_dir / "embeddings" / encoder_name
    retrieval_dir = paths.artifacts_dir / "retrieval" / encoder_name
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_parquet(dataset_dir / "train.parquet")
    validation_frame = pd.read_parquet(dataset_dir / "validation.parquet")
    test_frame = pd.read_parquet(dataset_dir / "test.parquet")
    train_embeddings = np.load(embeddings_dir / "train.npy")
    validation_embeddings = np.load(embeddings_dir / "validation.npy")
    test_embeddings = np.load(embeddings_dir / "test.npy")
    if len(train_frame) != len(train_embeddings):
        raise ValueError(f"Train retrieval input mismatch for {encoder_name}: {len(train_frame)} vs {len(train_embeddings)}")
    if len(validation_frame) != len(validation_embeddings):
        raise ValueError(
            f"Validation retrieval input mismatch for {encoder_name}: {len(validation_frame)} vs {len(validation_embeddings)}"
        )
    if len(test_frame) != len(test_embeddings):
        raise ValueError(f"Test retrieval input mismatch for {encoder_name}: {len(test_frame)} vs {len(test_embeddings)}")

    index = faiss.IndexFlatL2(train_embeddings.shape[1])
    index.add(train_embeddings.astype(np.float32))

    outputs = {
        "dataset_manifest": {
            "requested_limit": dataset_manifest.get("requested_limit"),
            "source_row_count": dataset_manifest.get("source_row_count"),
            "rows": dataset_manifest.get("rows", {}),
        }
    }
    for split_name, frame, embeddings in [
        ("train", train_frame, train_embeddings),
        ("validation", validation_frame, validation_embeddings),
        ("test", test_frame, test_embeddings),
    ]:
        features = _neighbor_features(
            embeddings,
            index,
            train_frame["severity_label"].to_numpy(),
            train_frame["teacher_score_replayed"].to_numpy(),
            k=k,
        )
        output_path = retrieval_dir / f"{split_name}.parquet"
        features.to_parquet(output_path, index=False)
        outputs[split_name] = {
            "path": str(output_path),
            "rows": int(len(features)),
        }

    with (retrieval_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval features from offline embeddings.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoder", default="modernbert-base")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    manifest = build_retrieval(args.config, args.encoder, args.k)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()