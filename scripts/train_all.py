#!/usr/bin/env python3
"""
train_all.py — Train all models and get ranked leaderboard.

Usage:
    # Train ALL models (16 experiments)
    python scripts/train_all.py

    # Train only classical models (skip neural/GPU)
    python scripts/train_all.py --classical-only

    # Train only neural models (LoRA fine-tuning)
    python scripts/train_all.py --neural-only

    # Train specific models
    python scripts/train_all.py --experiments logreg_tfidf_effective xgboost_hybrid_modernbert_effective

    # Re-run even completed experiments (ignore cached results)
    python scripts/train_all.py --force

    # Check current benchmark state without running
    python scripts/train_all.py --prepare-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.benchmark import run_benchmark
from ml.settings import bootstrap_environment


CLASSICAL_EXPERIMENTS = [
    "logreg_tfidf_valuechar_effective_tuned",
    "logreg_tfidf_valuechar_effective",
    "logreg_tfidf_effective",
    "logreg_tfidf_none",
    "logreg_tfidf_balanced",
    "xgboost_hybrid_modernbert_effective",
    "xgboost_hybrid_securebert2_effective",
]

NEURAL_EXPERIMENTS = [
    "modernbert_lora_balanced_softmax_effective",
    "modernbert_lora_weighted_ce_effective",
    "modernbert_lora_ce_none",
    "modernbert_lora_cb_focal_effective",
    "securebert2_lora_balanced_softmax_effective",
    "securebert2_lora_balanced_softmax_effective_epoch3",
    "securebert2_lora_cb_focal_effective",
    "cysecbert_lora_balanced_softmax_effective",
    "secbert_lora_balanced_softmax_effective",
]

ALL_EXPERIMENTS = CLASSICAL_EXPERIMENTS + NEURAL_EXPERIMENTS

BENCHMARK_NAME = "full_model_competition"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all DELTA-AI models and rank them by macro-F1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_full.yaml",
        help="Path to benchmark config (default: configs/benchmark_full.yaml)",
    )
    parser.add_argument(
        "--benchmark-name",
        default=BENCHMARK_NAME,
        help="Benchmark run name (default: full_model_competition)",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Train only classical models (LogReg, XGBoost). Skips neural experiments.",
    )
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Train only neural models (LoRA fine-tuning). Skips classical experiments.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Train specific experiment names (e.g. logreg_tfidf_effective securebert2_lora_...)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even already-completed experiments (ignore cached results).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build dataset and embeddings but skip training. Shows what would run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Dataset size limit (default: 20000). Use --no-limit for full dataset.",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Use full dataset (no limit). Overrides --limit.",
    )
    args = parser.parse_args()

    if args.classical_only:
        only_experiments = CLASSICAL_EXPERIMENTS
        print("[INFO] Classical-only mode: skipping neural experiments.")
    elif args.neural_only:
        only_experiments = NEURAL_EXPERIMENTS
        print("[INFO] Neural-only mode: skipping classical experiments.")
    elif only_experiments:
        print(f"[INFO] Training specific experiments: {only_experiments}")
    else:
        print("[INFO] Training ALL 16 models...")

    limit = None if args.no_limit else args.limit
    print(f"[INFO] Dataset limit: {'FULL DATASET' if limit is None else limit}")
    print(f"[INFO] Benchmark name: {args.benchmark_name}")
    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Force re-run: {force}")
    print()

    if prepare_only:
        print("[INFO] --prepare-only: building dataset and reports only (no training)")
        state = run_benchmark(
            config_path=config_path,
            benchmark_name=args.benchmark_name,
            only_experiments=only_experiments,
            force=force,
            prepare_only=True,
            dataset_limit=limit,
        )
        benchmark_root = Path("artifacts") / "benchmarks" / args.benchmark_name
        status_file = benchmark_root / "status.json"
        if status_file.exists():
            with status_file.open() as f:
                status = json.load(f)
            counts = status.get("counts", {})
            print()
            print("=== Benchmark Status ===")
            print(f"  Total:    {counts.get('total', 0)}")
            print(f"  Done:     {counts.get('completed', 0)}")
            print(f"  Running:  {counts.get('running', 0)}")
            print(f"  Failed:   {counts.get('failed', 0)}")
            print(f"  Pending:  {counts.get('not_started', 0)}")
        return

    state = run_benchmark(
        config_path=config_path,
        benchmark_name=args.benchmark_name,
        only_experiments=only_experiments,
        force=force,
        prepare_only=False,
        dataset_limit=limit,
    )

    benchmark_root = Path("artifacts") / "benchmarks" / args.benchmark_name
    leaderboard_path = benchmark_root / "leaderboard.md"
    results_csv = benchmark_root / "aggregate.csv"
    results_json = benchmark_root / "results.json"

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    if leaderboard_path.exists():
        print()
        print("--- Leaderboard (ranked by macro-F1) ---")
        print(leaderboard_path.read_text())

    if results_csv.exists():
        print()
        print(f"Full results CSV: {results_csv}")

    if results_json.exists():
        print(f"Full results JSON: {results_json}")

    print()
    print("Best model artifacts:")
    runs_dir = Path("artifacts/runs")
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir())[:5]:
            metrics = run_dir / "metrics.json"
            if metrics.exists():
                with metrics.open() as f:
                    m = json.load(f)
                f1 = m.get("macro_f1", "N/A")
                print(f"  {run_dir.name}: macro_f1={f1}")

    print()
    print(f"Next: See leaderboard at artifacts/benchmarks/{args.benchmark_name}/leaderboard.md")


if __name__ == "__main__":
    main()