from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_registry import list_encoder_specs
from .settings import bootstrap_environment


def summarize_encoders(config_path: str | Path | None = None) -> dict:
    paths = bootstrap_environment(config_path)
    encoders = []
    for spec in list_encoder_specs():
        encoders.append(
            {
                "name": spec.name,
                "model_id": spec.model_id,
                "family": spec.family,
                "domain": spec.domain,
                "default_max_length": spec.default_max_length,
                "default_batch_size": spec.default_batch_size,
                "default_pooling": spec.default_pooling,
                "notes": spec.notes,
                "download_root": str(paths.models_dir),
                "recommended_mode": "frozen-embeddings+hybrid" if spec.family in {"modernbert", "securebert2"} else "domain-ablation",
                "finetune_opportunity": "lora" if spec.domain == "cybersecurity" or spec.family == "modernbert" else "probe-only",
            }
        )

    output = {"encoders": encoders}
    out_dir = paths.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "encoders.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize configured encoder options and fine-tuning opportunities.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    report = summarize_encoders(args.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
