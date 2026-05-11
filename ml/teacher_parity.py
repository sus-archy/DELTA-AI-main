from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .data import load_db_frame
from .settings import bootstrap_environment, load_config


def run_teacher_parity(
    config_path: str | Path | None = None,
    sample_rows: int = 5000,
    script_path: str | Path | None = None,
) -> dict:
    paths = bootstrap_environment(config_path)
    if script_path is None:
        script_path = paths.project_root / "scripts" / "severityClassifier.js"
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy Node script not found: {script_path}. Provide --script-path or place it at scripts/severityClassifier.js relative to project root.")
    frame = load_db_frame(paths.db_csv, limit=sample_rows)

    payload = []
    for _, row in frame.iterrows():
        raw = dict(row["raw_obj"]) if isinstance(row["raw_obj"], dict) else {}
        raw.setdefault("type", row["type"])
        raw.setdefault("source", row["source"])
        raw.setdefault("description", row["description"])
        raw["observedCount"] = row["observedCount"]
        raw["tags"] = row["tags_list"]
        raw["confidence"] = row["confidence"]
        payload.append(raw)

    predicted = classify_many_with_node(payload, script_path)
    predicted_frame = pd.DataFrame(predicted)
    severity_match = float((predicted_frame["severity"].astype(str).str.lower() == frame["severity_label"]).mean())
    confidence_match = float((predicted_frame["confidence"].astype(str) == frame["confidence"].astype(str)).mean())

    comparison = frame[["type", "value", "source", "severity_label", "confidence"]].copy()
    comparison["node_severity"] = predicted_frame["severity"].astype(str).str.lower()
    comparison["node_score"] = predicted_frame["severityScore"]
    comparison["node_confidence"] = predicted_frame["confidence"].astype(str)
    mismatches = comparison[comparison["severity_label"] != comparison["node_severity"]].head(50)

    report = {
        "rows_checked": int(len(frame)),
        "severity_match_rate": severity_match,
        "confidence_match_rate": confidence_match,
        "mismatch_examples": mismatches.to_dict(orient="records"),
    }

    output_dir = paths.artifacts_dir / "teacher_parity"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "latest.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Node teacher outputs with stored DB labels.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--script-path", default=None)
    args = parser.parse_args()
    try:
        report = run_teacher_parity(args.config, args.sample_rows, args.script_path)
        print(json.dumps(report, indent=2))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=__import__('sys').stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
