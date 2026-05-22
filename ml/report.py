from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .settings import bootstrap_environment


def build_summary_report(config_path: str | Path | None = None) -> dict:
    paths = bootstrap_environment(config_path)
    report = {
        "artifacts_root": str(paths.artifacts_dir),
        "runs": [],
    }
    runs_root = paths.artifacts_dir / "runs"
    if runs_root.exists():
        for metrics_path in runs_root.glob("*/metrics.json"):
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            run_dir = metrics_path.parent
            has_plots = (run_dir / "plots").exists()
            report["runs"].append(
                {
                    "run": run_dir.name,
                    "run_dir": str(run_dir),
                    "plots_dir": str(run_dir / "plots") if has_plots else None,
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "ece": metrics.get("ece"),
                    "brier": metrics.get("brier"),
                }
            )
    report["runs"] = sorted(report["runs"], key=lambda item: (item["macro_f1"] is not None, item["macro_f1"]), reverse=True)

    output_dir = paths.reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if report["runs"]:
        df = pd.DataFrame(report["runs"])
        cols = [c for c in df.columns if c != "run_dir"]
        df[cols].to_csv(output_dir / "summary.csv", index=False)
    return report


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def build_benchmark_visual_report(benchmark_root: str | Path) -> dict:
    benchmark_root = Path(benchmark_root)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    visuals_dir = benchmark_root / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    state = _read_json(benchmark_root / "state.json", {"experiments": {}})
    results = _read_json(benchmark_root / "results.json", {"runs": [], "aggregate": []})
    aggregate = pd.DataFrame(results.get("aggregate", []))
    runs = pd.DataFrame(results.get("runs", []))

    records = list(state.get("experiments", {}).values())
    planned_total = int(state.get("planned_total_runs") or len(records))
    completed = sum(1 for record in records if record.get("status") == "completed")
    running = sum(1 for record in records if record.get("status") == "running")
    failed = sum(1 for record in records if record.get("status") == "failed")
    total = planned_total
    progress_pct = (100.0 * completed / total) if total else 0.0

    durations_hours = []
    for record in records:
        started = _parse_iso(record.get("started_at"))
        completed_at = _parse_iso(record.get("completed_at"))
        if started and completed_at:
            durations_hours.append((completed_at - started).total_seconds() / 3600.0)

    status = {
        "benchmark_name": state.get("benchmark_name"),
        "target": state.get("target"),
        "config_path": state.get("config_path"),
        "updated_at": state.get("updated_at"),
        "counts": {
            "total": total,
            "completed": completed,
            "running": running,
            "failed": failed,
            "not_started": max(total - completed - running - failed, 0),
        },
        "progress_pct": progress_pct,
        "duration_hours_mean": (sum(durations_hours) / len(durations_hours)) if durations_hours else None,
    }
    with (benchmark_root / "status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)

    lines = [
        "# Benchmark Status",
        "",
        f"- Benchmark: `{status['benchmark_name']}`",
        f"- Progress: `{completed}/{total}` completed ({progress_pct:.1f}%)",
        f"- Running: `{running}`",
        f"- Failed: `{failed}`",
        f"- Not started: `{status['counts']['not_started']}`",
    ]
    if status["duration_hours_mean"] is not None:
        lines.append(f"- Mean completed run duration: `{status['duration_hours_mean']:.2f}` hours")
    if not aggregate.empty:
        lines.extend(
            [
                "",
                "## Leaderboard",
                "",
                "| Experiment | Seeds | Macro-F1 | Balanced Acc | ECE | Brier | Plots |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in aggregate.iterrows():
            run_dirs = list(benchmark_root.glob(f"*/{row['experiment']}*"))
            plot_links = []
            for d in run_dirs:
                plots_dir = d / "plots"
                if plots_dir.exists():
                    rel = plots_dir.relative_to(benchmark_root.parent)
                    plot_links.append(f"[{d.name}]({rel})")
            plots_cell = "; ".join(plot_links) if plot_links else ""
            lines.append(
                f"| {row['experiment']} | {int(row['num_seeds'])} | "
                f"{row['macro_f1_mean']:.4f} | {row['balanced_accuracy_mean']:.4f} | "
                f"{row['ece_mean']:.4f} | {row['brier_mean']:.4f} | {plots_cell} |"
            )
    (benchmark_root / "status.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return status

    if not aggregate.empty:
        chart_frame = aggregate.sort_values("macro_f1_mean", ascending=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(chart_frame["experiment"], chart_frame["macro_f1_mean"], color="#2f6bff")
        axes[0].set_title("Macro-F1 Mean")
        axes[0].set_xlabel("Macro-F1")

        axes[1].barh(chart_frame["experiment"], chart_frame["balanced_accuracy_mean"], color="#159f6b")
        axes[1].set_title("Balanced Accuracy Mean")
        axes[1].set_xlabel("Balanced Accuracy")
        fig.tight_layout()
        fig.savefig(visuals_dir / "leaderboard.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if not runs.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = runs["experiment"].astype("category").cat.codes
        scatter = ax.scatter(
            runs["ece"],
            runs["macro_f1"],
            c=colors,
            cmap="tab10",
            alpha=0.85,
            s=60,
        )
        ax.set_title("Run Tradeoff: Calibration vs Macro-F1")
        ax.set_xlabel("ECE (lower is better)")
        ax.set_ylabel("Macro-F1 (higher is better)")
        fig.tight_layout()
        fig.savefig(visuals_dir / "tradeoff.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple run summary report.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark-root", default=None)
    args = parser.parse_args()
    if args.benchmark_root:
        report = build_benchmark_visual_report(args.benchmark_root)
    else:
        report = build_summary_report(args.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
