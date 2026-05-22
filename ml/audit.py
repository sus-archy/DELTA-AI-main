from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_db_frame, load_ip_sidecar, load_phishing_sidecar
from .settings import bootstrap_environment, load_config


def run_audit(config_path: str | Path | None = None, limit: int | None = None) -> dict:
    paths = bootstrap_environment(config_path)
    config = load_config(config_path)
    frame = load_db_frame(paths.db_csv, limit=limit)
    ip_sidecar = load_ip_sidecar(paths.merged_ip_txt)
    phishing_sidecar = load_phishing_sidecar(paths.merged_phishing_csv)

    null_counts = {
        column: int(frame[column].isna().sum() + frame[column].astype(str).str.strip().eq("").sum())
        for column in ["type", "value", "description", "source", "fingerprint", "observedCount", "raw", "severity", "confidence", "tags"]
    }
    duplicates = (
        frame.groupby(["type", "value"])
        .agg(rows=("value", "size"), sources=("source", "nunique"), severities=("severity_label", "nunique"))
        .reset_index()
    )
    monthly = (
        frame.assign(month=frame["lastSeen_dt"].dt.to_period("M").astype(str))
        .groupby(["month", "severity_label"])
        .size()
        .reset_index(name="count")
    )
    raw_source_mismatch = int(
        sum(
            str(raw.get("source", "")) != str(source)
            for raw, source in zip(frame["raw_obj"], frame["source"])
        )
    )
    phishing_domains = set(phishing_sidecar["domain_norm"])
    phishing_overlap = int(frame["value"].astype(str).str.lower().isin(phishing_domains).sum())

    report = {
        "project_root": str(paths.project_root),
        "db_rows": int(len(frame)),
        "null_counts": null_counts,
        "severity_counts": frame["severity_label"].value_counts().to_dict(),
        "type_counts": frame["type"].astype(str).str.lower().value_counts().to_dict(),
        "source_counts": frame["source"].astype(str).value_counts().to_dict(),
        "confidence_counts": frame["confidence"].astype(str).value_counts().to_dict(),
        "first_seen_min": str(frame["firstSeen_dt"].min()),
        "last_seen_max": str(frame["lastSeen_dt"].max()),
        "fingerprint_duplicates": int(frame["fingerprint"].duplicated().sum()),
        "duplicate_indicator_keys": int((duplicates["rows"] > 1).sum()),
        "multi_source_duplicate_keys": int((duplicates["sources"] > 1).sum()),
        "multi_severity_duplicate_keys": int((duplicates["severities"] > 1).sum()),
        "raw_source_mismatches": raw_source_mismatch,
        "ip_sidecar_exact_count": int(len(ip_sidecar["exact_ips"])),
        "ip_sidecar_cidr_count": int(len(ip_sidecar["cidr_prefix_by_base"])),
        "phishing_rows": int(len(phishing_sidecar)),
        "phishing_overlap_on_db_value": phishing_overlap,
        "monthly_distribution": monthly.to_dict(orient="records"),
        "assumptions": {
            "test_days": config.get("splits", {}).get("test_days", 90),
            "validation_days": config.get("splits", {}).get("validation_days", 60),
        },
    }

    audit_dir = paths.artifacts_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_path = audit_dir / "latest_audit.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Delta IOC severity datasets.")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    args = parser.parse_args()
    report = run_audit(args.config, args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
