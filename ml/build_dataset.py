from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import pandas as pd

from .data import (
    apply_group_stratified_split,
    apply_temporal_split,
    load_db_frame,
    load_ip_sidecar,
    load_phishing_sidecar,
    iter_split_frames,
)
from .features import augment_features
from .settings import bootstrap_environment, load_config
from .labels import SEVERITY_ORDER


def _sample_group_rows(
    group: pd.DataFrame,
    n: int,
    seed: int,
    selected_indices: set[int] | None = None,
) -> pd.DataFrame:
    if n <= 0 or group.empty:
        return group.iloc[0:0].copy()

    selected_indices = selected_indices or set()
    remaining = group.loc[~group.index.isin(selected_indices)].copy()
    if remaining.empty:
        return remaining
    if n >= len(remaining):
        return remaining

    rng = seed
    key_heads = pd.concat(
        [
            item.sample(n=1, random_state=rng)
            for _, item in remaining.groupby("indicator_key", dropna=False, sort=False)
        ],
        axis=0,
    )
    if n <= len(key_heads):
        return key_heads.sample(n=n, random_state=rng)

    picked = key_heads
    need = n - len(picked)
    leftovers = remaining.loc[~remaining.index.isin(picked.index)]
    if need > 0 and not leftovers.empty:
        extra = leftovers.sample(n=min(need, len(leftovers)), random_state=rng)
        picked = pd.concat([picked, extra], axis=0)
    return picked


def _sample_frame_by_strata(
    frame: pd.DataFrame,
    limit: int,
    strat_cols: list[str],
    seed: int = 42,
    coverage_floor: int = 25,
    small_cell_threshold: int = 25,
) -> pd.DataFrame:
    if limit >= len(frame):
        return frame.copy()

    grouped = list(frame.groupby(strat_cols, dropna=False, sort=False))
    selected_indices: set[int] = set()
    sampled_parts: list[pd.DataFrame] = []
    coverage_counts: dict[tuple, int] = {}

    for cell_key, group in grouped:
        quota = len(group) if len(group) <= small_cell_threshold else min(coverage_floor, len(group))
        picked = _sample_group_rows(group, quota, seed, selected_indices)
        if not picked.empty:
            selected_indices.update(picked.index.tolist())
            sampled_parts.append(picked)
        coverage_counts[cell_key] = int(len(picked))

    current_rows = sum(len(part) for part in sampled_parts)
    if current_rows > limit:
        coverage = pd.concat(sampled_parts, axis=0)
        return coverage.sample(n=limit, random_state=seed).reset_index(drop=True)

    remaining_budget = limit - current_rows
    residual_sizes = {
        cell_key: max(0, len(group) - coverage_counts.get(cell_key, 0))
        for cell_key, group in grouped
    }
    total_residual = sum(residual_sizes.values())

    if remaining_budget > 0 and total_residual > 0:
        raw_allocations: dict[tuple, float] = {
            cell_key: remaining_budget * residual / total_residual
            for cell_key, residual in residual_sizes.items()
        }
        fill_allocations: dict[tuple, int] = {
            cell_key: min(residual_sizes[cell_key], int(math.floor(amount)))
            for cell_key, amount in raw_allocations.items()
        }
        assigned = sum(fill_allocations.values())
        remaining_fill = remaining_budget - assigned

        ranked_cells = sorted(
            raw_allocations,
            key=lambda key: (raw_allocations[key] - math.floor(raw_allocations[key]), residual_sizes[key]),
            reverse=True,
        )
        for cell_key in ranked_cells:
            if remaining_fill <= 0:
                break
            if fill_allocations[cell_key] >= residual_sizes[cell_key]:
                continue
            fill_allocations[cell_key] += 1
            remaining_fill -= 1

        for cell_key, group in grouped:
            quota = fill_allocations.get(cell_key, 0)
            if quota <= 0:
                continue
            picked = _sample_group_rows(group, quota, seed, selected_indices)
            if not picked.empty:
                selected_indices.update(picked.index.tolist())
                sampled_parts.append(picked)

    sampled = pd.concat(sampled_parts, axis=0)
    if len(sampled) < limit:
        remaining = frame.loc[~frame.index.isin(selected_indices)]
        if not remaining.empty:
            extra = remaining.sample(n=min(limit - len(sampled), len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, extra], axis=0)
    if len(sampled) > limit:
        sampled = sampled.sample(n=limit, random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def sample_frame_for_benchmark(
    frame: pd.DataFrame,
    limit: int,
    seed: int = 42,
    coverage_floor: int = 25,
    small_cell_threshold: int = 25,
) -> pd.DataFrame:
    if limit >= len(frame):
        return frame.copy()

    severity_counts = frame["severity_label"].value_counts()
    present_severities = [label for label in SEVERITY_ORDER if label in severity_counts.index]
    if not present_severities:
        raise ValueError("No supported severity labels found for balanced sampling.")

    base_quota = limit // len(present_severities)
    remainder = limit % len(present_severities)
    sampled_parts: list[pd.DataFrame] = []

    for index, severity_label in enumerate(present_severities):
        severity_limit = base_quota + (1 if index < remainder else 0)
        severity_frame = frame.loc[frame["severity_label"] == severity_label].copy()
        picked = _sample_frame_by_strata(
            severity_frame,
            severity_limit,
            strat_cols=["source", "type"],
            seed=seed + index,
            coverage_floor=coverage_floor,
            small_cell_threshold=small_cell_threshold,
        )
        sampled_parts.append(picked)

    sampled = pd.concat(sampled_parts, axis=0)
    if len(sampled) > limit:
        sampled = sampled.groupby("severity_label", group_keys=False, dropna=False).apply(
            lambda item: item.sample(
                n=min(len(item), limit // len(present_severities)),
                random_state=seed,
            )
        )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_dataset(config_path: str | Path | None = None, limit: int | None = None) -> dict:
    paths = bootstrap_environment(config_path)
    config = load_config(config_path)

    if paths.db_csv is None or not paths.db_csv.exists():
        raise FileNotFoundError(
            f"DB CSV not found: {paths.db_csv}. "
            "Set data.db_csv in your config or place Data/DB-ThreatIndicators.csv in the project root."
        )

    frame = load_db_frame(paths.db_csv, limit=None)
    source_row_count = int(len(frame))

    ip_sidecar = load_ip_sidecar(paths.merged_ip_txt)
    phishing_sidecar = load_phishing_sidecar(paths.merged_phishing_csv)
    frame = augment_features(frame, ip_sidecar, phishing_sidecar)

    try:
        from .teacher import classify_ioc
        teacher_records = frame.apply(
            lambda row: classify_ioc({
                "type": row["type"],
                "source": row["source"],
                "description": row.get("description", ""),
                "observedCount": row.get("observedCount", 0),
                "tags": row.get("tags_list", []),
                "confidence": row.get("confidence"),
            }),
            axis=1,
        ).tolist()
        frame["teacher_severity_replayed"] = [item["severity"] for item in teacher_records]
        frame["teacher_score_replayed"] = [item["severityScore"] for item in teacher_records]
        frame["teacher_confidence_replayed"] = [item["confidence"] for item in teacher_records]
        frame["teacher_concepts_json"] = [json.dumps(item["concepts"]) for item in teacher_records]
    except Exception:
        pass

    splits_cfg = config.get("splits", {})
    split_mode = str(splits_cfg.get("mode", "temporal")).lower()
    seed = int(config.get("runtime", {}).get("seed", 42))

    if limit is not None:
        frame = sample_frame_for_benchmark(frame, limit, seed=seed)

    if split_mode in {"group_stratified", "ratio", "stratified"}:
        bundle = apply_group_stratified_split(
            frame,
            seed=seed,
            train_frac=float(splits_cfg.get("train_frac", 0.8)),
            validation_frac=float(splits_cfg.get("validation_frac", 0.1)),
        )
    elif split_mode == "temporal":
        bundle = apply_temporal_split(
            frame,
            test_days=int(splits_cfg.get("test_days", 90)),
            validation_days=int(splits_cfg.get("validation_days", 60)),
        )
    else:
        raise ValueError(
            f"Unsupported split mode: {split_mode}. "
            "Use 'temporal' or 'group_stratified'."
        )

    dataset_dir = paths.artifacts_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_row_count": source_row_count,
        "requested_limit": limit,
        "rows": {},
        "paths": {},
    }
    for split_name, split_frame in iter_split_frames(bundle):
        split_path = dataset_dir / f"{split_name}.parquet"
        split_frame.to_parquet(split_path, index=False)
        manifest["rows"][split_name] = int(len(split_frame))
        manifest["paths"][split_name] = str(split_path)

    manifest_path = dataset_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature-rich Delta IOC datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = build_dataset(args.config, args.limit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()