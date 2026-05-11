from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@dataclass(slots=True)
class ProjectPaths:
    project_root: Path
    db_csv: Path | None
    merged_ip_txt: Path | None
    merged_phishing_csv: Path | None
    artifacts_dir: Path
    models_dir: Path
    reports_dir: Path
    cache_root: Path
    hf_home: Path
    transformers_cache: Path
    torch_home: Path
    sentence_transformers_home: Path


def prime_cache_environment(project_root: str | Path | None = None) -> dict[str, str]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    cache_root = root / ".cache"
    hf_home = cache_root / "hf"
    hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    torch_home = cache_root / "torch"
    sentence_transformers_home = cache_root / "sentence_transformers"
    for path in [cache_root, hf_home, hub_cache, transformers_cache, torch_home, sentence_transformers_home]:
        path.mkdir(parents=True, exist_ok=True)
    env_updates = {
        "XDG_CACHE_HOME": str(cache_root),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hub_cache),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache),
        "TRANSFORMERS_CACHE": str(transformers_cache),
        "TORCH_HOME": str(torch_home),
        "SENTENCE_TRANSFORMERS_HOME": str(sentence_transformers_home),
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    return env_updates


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def build_paths(config_path: str | Path | None = None) -> ProjectPaths:
    config_file = Path(config_path).resolve() if config_path else DEFAULT_CONFIG.resolve()
    config = load_config(config_file)
    config_dir = config_file.parent
    project_root = _resolve_path(config["project_root"], config_dir) or config_dir.parent

    data_config = config.get("data", {})

    cache_root = _resolve_path(config["paths"]["cache_root"], project_root) if config.get("paths", {}).get("cache_root") else project_root / ".cache"

    return ProjectPaths(
        project_root=project_root,
        db_csv=_resolve_path(data_config.get("db_csv"), project_root),
        merged_ip_txt=_resolve_path(data_config.get("merged_ip_txt"), project_root),
        merged_phishing_csv=_resolve_path(data_config.get("merged_phishing_csv"), project_root),
        artifacts_dir=_resolve_path(config["paths"]["artifacts_dir"], project_root) or project_root / "artifacts",
        models_dir=_resolve_path(config["paths"]["models_dir"], project_root) or project_root / "models",
        reports_dir=_resolve_path(config["paths"]["reports_dir"], project_root) or project_root / "reports",
        cache_root=cache_root,
        hf_home=cache_root / "hf",
        transformers_cache=cache_root / "hf" / "transformers",
        torch_home=cache_root / "torch",
        sentence_transformers_home=cache_root / "sentence_transformers",
    )


def bootstrap_environment(config_path: str | Path | None = None) -> ProjectPaths:
    paths = build_paths(config_path)
    prime_cache_environment(paths.project_root)
    required = [
        paths.artifacts_dir,
        paths.models_dir,
        paths.reports_dir,
        paths.cache_root,
        paths.hf_home,
        paths.transformers_cache,
        paths.torch_home,
        paths.sentence_transformers_home,
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)

    return paths