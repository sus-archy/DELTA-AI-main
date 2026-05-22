from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .build_dataset import build_dataset
from .build_embeddings import build_embeddings
from .build_retrieval import build_retrieval
from .finetune import fine_tune
from .report import build_benchmark_visual_report
from .settings import bootstrap_environment, load_config
from .train import train_logreg_tfidf, train_lookup_majority, train_xgboost_hybrid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _log_event(log_path: Path, event_type: str, **extra: Any) -> None:
    payload = {"ts": _utc_now(), "event": event_type}
    payload.update(extra)
    _append_jsonl(log_path, payload)


def _metrics_keys() -> list[str]:
    return [
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "ece",
        "brier",
    ]


def _load_benchmark_config(config_path: str | Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    benchmark = config.get("benchmark")
    if benchmark is None:
        raise KeyError("Config does not define a benchmark section.")
    return config, benchmark


def _enabled_experiments(benchmark: dict[str, Any], only_names: set[str] | None) -> list[dict[str, Any]]:
    experiments = []
    for experiment in benchmark.get("experiments", []):
        if not experiment.get("enabled", True):
            continue
        if only_names and experiment["name"] not in only_names:
            continue
        experiments.append(experiment)
    return experiments


def _ensure_dataset(config_path: str | Path | None, benchmark: dict[str, Any], log_path: Path, force: bool) -> None:
    paths = bootstrap_environment(config_path)
    manifest_path = paths.artifacts_dir / "datasets" / "manifest.json"
    requested_limit = benchmark.get("dataset_limit")
    if manifest_path.exists() and not force and not benchmark.get("rebuild_dataset", False):
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("requested_limit", "__missing__") == requested_limit:
            _log_event(log_path, "dataset_skip", reason="existing_manifest", requested_limit=requested_limit)
            return
        _log_event(
            log_path,
            "dataset_rebuild_required",
            reason="manifest_limit_mismatch",
            existing_limit=manifest.get("requested_limit", "__missing__"),
            requested_limit=requested_limit,
        )
    _log_event(log_path, "dataset_build_start", limit=benchmark.get("dataset_limit"))
    build_dataset(config_path, limit=benchmark.get("dataset_limit"))
    _log_event(log_path, "dataset_build_complete", manifest=str(manifest_path))


def _needs_embeddings(experiment: dict[str, Any]) -> bool:
    return experiment.get("runner") == "classical" and experiment.get("model") == "xgboost_hybrid"


def _ensure_embeddings_and_retrieval(
    config_path: str | Path | None,
    experiments: list[dict[str, Any]],
    log_path: Path,
    force: bool,
) -> None:
    paths = bootstrap_environment(config_path)
    dataset_manifest_path = paths.artifacts_dir / "datasets" / "manifest.json"
    with dataset_manifest_path.open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    dataset_signature = {
        "requested_limit": dataset_manifest.get("requested_limit"),
        "source_row_count": dataset_manifest.get("source_row_count"),
        "rows": dataset_manifest.get("rows", {}),
    }
    encoders = sorted({exp["encoder"] for exp in experiments if _needs_embeddings(exp)})
    for encoder in encoders:
        embeddings_dir = paths.artifacts_dir / "embeddings" / encoder
        retrieval_dir = paths.artifacts_dir / "retrieval" / encoder
        embedding_files = [embeddings_dir / f"{split}.npy" for split in ["train", "validation", "test"]]
        retrieval_files = [retrieval_dir / f"{split}.parquet" for split in ["train", "validation", "test"]]
        embeddings_manifest_path = paths.artifacts_dir / "embeddings" / "manifest.json"
        retrieval_manifest_path = retrieval_dir / "manifest.json"

        embeddings_manifest = _read_json(embeddings_manifest_path, {})
        encoder_embedding_manifest = embeddings_manifest.get(encoder, {})
        embeddings_match_dataset = encoder_embedding_manifest.get("dataset_manifest") == dataset_signature

        if force or not all(path.exists() for path in embedding_files) or not embeddings_match_dataset:
            _log_event(log_path, "embeddings_build_start", encoder=encoder)
            build_embeddings(config_path, encoders=[encoder], max_rows=benchmark_limit(config_path))
            _log_event(log_path, "embeddings_build_complete", encoder=encoder)
        else:
            _log_event(log_path, "embeddings_skip", encoder=encoder, reason="existing_files")

        retrieval_manifest = _read_json(retrieval_manifest_path, {})
        retrieval_match_dataset = retrieval_manifest.get("dataset_manifest") == dataset_signature

        if force or not all(path.exists() for path in retrieval_files) or not retrieval_match_dataset:
            _log_event(log_path, "retrieval_build_start", encoder=encoder)
            build_retrieval(config_path, encoder_name=encoder)
            _log_event(log_path, "retrieval_build_complete", encoder=encoder)
        else:
            _log_event(log_path, "retrieval_skip", encoder=encoder, reason="existing_files")


def benchmark_limit(config_path: str | Path | None) -> int | None:
    _, benchmark = _load_benchmark_config(config_path)
    return benchmark.get("dataset_limit")


def _experiment_id(name: str, seed: int) -> str:
    return f"{name}__seed{seed}"


def _run_experiment(experiment: dict[str, Any], config_path: str | Path | None, seed: int) -> tuple[dict[str, Any], str]:
    paths = bootstrap_environment(config_path)
    run_name = _experiment_id(experiment["name"], seed)
    runner = experiment["runner"]
    if runner == "classical":
        model = experiment["model"]
        if model == "source_majority":
            return train_lookup_majority(paths, ["source"], seed=seed, run_name=run_name)
        if model == "type_majority":
            return train_lookup_majority(paths, ["type"], seed=seed, run_name=run_name)
        if model == "source_type_majority":
            return train_lookup_majority(paths, ["source", "type"], seed=seed, run_name=run_name)
        if model == "logreg_tfidf":
            return train_logreg_tfidf(
                paths,
                class_balance=experiment.get("class_balance", "inverse"),
                seed=seed,
                run_name=run_name,
                feature_variant=experiment.get("feature_variant", "default"),
                tune_c=bool(experiment.get("tune_c", False)),
                c_grid=experiment.get("c_grid"),
                calibration_method=experiment.get("calibration", "none"),
            )
        if model == "xgboost_hybrid":
            return train_xgboost_hybrid(
                paths,
                encoder_name=experiment["encoder"],
                class_balance=experiment.get("class_balance", "effective"),
                seed=seed,
                run_name=run_name,
            )
        raise KeyError(f"Unsupported classical model '{model}'.")

    if runner == "finetune":
        result = fine_tune(
            config_path=config_path,
            encoder_name=experiment["encoder"],
            peft_mode=experiment.get("peft", "lora"),
            max_train_rows=experiment.get("max_train_rows"),
            num_epochs=experiment.get("epochs", 2),
            class_balance=experiment.get("class_balance", "effective"),
            loss_name=experiment.get("loss", "balanced_softmax"),
            focal_gamma=float(experiment.get("focal_gamma", 1.5)),
            seed=seed,
            run_name=run_name,
            calibration_method=experiment.get("calibration", "none"),
            max_length_override=experiment.get("max_length"),
        )
        return result["metrics"], result["run_dir"]

    raise KeyError(f"Unsupported runner '{runner}'.")


def _aggregate_completed(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for record in state.get("experiments", {}).values():
        if record.get("status") != "completed":
            continue
        row = {
            "experiment": record["experiment_name"],
            "seed": record["seed"],
            "runner": record["runner"],
            "model": record["model"],
            "encoder": record.get("encoder"),
            "group": record.get("group"),
            "ablation": record.get("ablation"),
            "run_dir": record.get("run_dir"),
        }
        row.update({key: record.get("metrics", {}).get(key) for key in _metrics_keys()})
        rows.append(row)

    if not rows:
        return [], []

    frame = pd.DataFrame(rows)
    aggregates = []
    for experiment_name, group in frame.groupby("experiment"):
        record = {"experiment": experiment_name}
        first = group.iloc[0]
        for field in ["runner", "model", "encoder", "group", "ablation"]:
            record[field] = first.get(field)
        record["num_seeds"] = int(group["seed"].nunique())
        for key in _metrics_keys():
            record[f"{key}_mean"] = float(group[key].mean())
            record[f"{key}_std"] = float(group[key].std(ddof=0))
        aggregates.append(record)
    aggregates = sorted(aggregates, key=lambda item: item.get("macro_f1_mean", -1.0), reverse=True)
    return rows, aggregates


def _write_benchmark_reports(benchmark_root: Path, state: dict[str, Any]) -> None:
    rows, aggregates = _aggregate_completed(state)
    _write_json(benchmark_root / "results.json", {"runs": rows, "aggregate": aggregates})
    if rows:
        pd.DataFrame(rows).sort_values(["experiment", "seed"]).to_csv(benchmark_root / "results.csv", index=False)
    if aggregates:
        pd.DataFrame(aggregates).to_csv(benchmark_root / "aggregate.csv", index=False)
        lines = [
            "# Benchmark Leaderboard",
            "",
            "| Experiment | Seeds | Macro-F1 | Balanced Acc | ECE | Brier |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in aggregates:
            lines.append(
                f"| {item['experiment']} | {item['num_seeds']} | "
                f"{item['macro_f1_mean']:.4f} ± {item['macro_f1_std']:.4f} | "
                f"{item['balanced_accuracy_mean']:.4f} ± {item['balanced_accuracy_std']:.4f} | "
                f"{item['ece_mean']:.4f} ± {item['ece_std']:.4f} | "
                f"{item['brier_mean']:.4f} ± {item['brier_std']:.4f} |"
            )
        (benchmark_root / "leaderboard.md").write_text("\n".join(lines), encoding="utf-8")
    build_benchmark_visual_report(benchmark_root)


def run_benchmark(
    config_path: str | Path | None = None,
    benchmark_name: str = "final_db_label_benchmark",
    only_experiments: list[str] | None = None,
    force: bool = False,
    prepare_only: bool = False,
    dataset_limit: int | None = None,
) -> dict[str, Any]:
    config, benchmark = _load_benchmark_config(config_path)
    if dataset_limit is not None:
        benchmark["dataset_limit"] = dataset_limit
    paths = bootstrap_environment(config_path)
    benchmark_root = paths.artifacts_dir / "benchmarks" / benchmark_name
    benchmark_root.mkdir(parents=True, exist_ok=True)
    log_path = benchmark_root / "events.jsonl"
    state_path = benchmark_root / "state.json"

    state = _read_json(
        state_path,
        {
            "benchmark_name": benchmark_name,
            "target": benchmark.get("target", "db_severity"),
            "created_at": _utc_now(),
            "config_path": str(config_path) if config_path else None,
            "planned_experiment_names": [],
            "planned_seeds": [],
            "planned_total_runs": 0,
            "experiments": {},
        },
    )
    state["updated_at"] = _utc_now()
    _write_json(state_path, state)

    only_names = set(only_experiments or [])
    experiments = _enabled_experiments(benchmark, only_names if only_names else None)
    seeds = benchmark.get("seeds", [config["runtime"]["seed"]])
    resume = bool(benchmark.get("resume", True))
    stop_on_error = bool(benchmark.get("stop_on_error", False))
    state["planned_experiment_names"] = [experiment["name"] for experiment in experiments]
    planned_seed_values: list[int] = []
    for experiment in experiments:
        planned_seed_values.extend(experiment.get("seeds", seeds))
    state["planned_seeds"] = sorted(set(planned_seed_values))
    state["planned_total_runs"] = int(sum(len(experiment.get("seeds", seeds)) for experiment in experiments))
    state["updated_at"] = _utc_now()
    _write_json(state_path, state)

    _log_event(log_path, "benchmark_start", benchmark=benchmark_name, seeds=state["planned_seeds"], experiment_count=len(experiments))
    _ensure_dataset(config_path, benchmark, log_path, force=force)
    _ensure_embeddings_and_retrieval(config_path, experiments, log_path, force=force)

    if prepare_only:
        state["updated_at"] = _utc_now()
        _write_json(state_path, state)
        _write_benchmark_reports(benchmark_root, state)
        _log_event(log_path, "benchmark_prepare_only_complete", benchmark=benchmark_name)
        return state

    for experiment in experiments:
        experiment_seeds = experiment.get("seeds", seeds)
        for seed in experiment_seeds:
            experiment_id = _experiment_id(experiment["name"], seed)
            previous = state["experiments"].get(experiment_id)
            if resume and previous and previous.get("status") == "completed" and not force:
                _log_event(log_path, "experiment_skip", experiment_id=experiment_id, reason="already_completed")
                continue

            record = {
                "experiment_name": experiment["name"],
                "seed": seed,
                "status": "running",
                "runner": experiment["runner"],
                "model": experiment.get("model", "sequence_classifier"),
                "encoder": experiment.get("encoder"),
                "group": experiment.get("group"),
                "ablation": experiment.get("ablation"),
                "started_at": _utc_now(),
                "attempts": int(previous.get("attempts", 0) + 1) if previous else 1,
            }
            state["experiments"][experiment_id] = record
            state["updated_at"] = _utc_now()
            _write_json(state_path, state)
            _log_event(log_path, "experiment_start", experiment_id=experiment_id, experiment=experiment)

            try:
                metrics, run_dir = _run_experiment(experiment, config_path, seed)
                record["status"] = "completed"
                record["completed_at"] = _utc_now()
                started_at = datetime.fromisoformat(record["started_at"])
                completed_at = datetime.fromisoformat(record["completed_at"])
                record["duration_seconds"] = (completed_at - started_at).total_seconds()
                record["run_dir"] = run_dir
                record["metrics"] = metrics
                _log_event(
                    log_path,
                    "experiment_complete",
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    duration_seconds=record["duration_seconds"],
                    metrics={key: metrics.get(key) for key in _metrics_keys()},
                )
            except Exception as exc:  # pragma: no cover - orchestration safety
                record["status"] = "failed"
                record["completed_at"] = _utc_now()
                started_at = datetime.fromisoformat(record["started_at"])
                completed_at = datetime.fromisoformat(record["completed_at"])
                record["duration_seconds"] = (completed_at - started_at).total_seconds()
                record["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _log_event(
                    log_path,
                    "experiment_failed",
                    experiment_id=experiment_id,
                    duration_seconds=record["duration_seconds"],
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                if stop_on_error:
                    state["experiments"][experiment_id] = record
                    state["updated_at"] = _utc_now()
                    _write_json(state_path, state)
                    _write_benchmark_reports(benchmark_root, state)
                    raise

            state["experiments"][experiment_id] = record
            state["updated_at"] = _utc_now()
            _write_json(state_path, state)
            _write_benchmark_reports(benchmark_root, state)

    _log_event(log_path, "benchmark_complete", benchmark=benchmark_name)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable seeded benchmark suite for Delta IOC severity models.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark-name", default="final_db_label_benchmark")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Dataset row limit. Omit to use config value or full dataset.")
    args = parser.parse_args()
    state = run_benchmark(
        config_path=args.config,
        benchmark_name=args.benchmark_name,
        only_experiments=args.experiments,
        force=args.force,
        prepare_only=args.prepare_only,
        dataset_limit=args.limit,
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
