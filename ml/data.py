from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .labels import SEVERITY_ORDER, SEVERITY_TO_ID


@dataclass(slots=True)
class DatasetBundle:
    full: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def parse_tags(value: str | list[str] | None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_raw(value: str | None) -> dict:
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def canonicalize_value(ioc_type: str, value: str) -> str:
    value = str(value).strip().lower()
    if ioc_type in {"domain", "hostname"}:
        return value.rstrip(".")
    if ioc_type == "url":
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path or ''}".rstrip("/")
    return value


def build_indicator_key(ioc_type: str, value: str) -> str:
    return f"{str(ioc_type).lower()}||{canonicalize_value(str(ioc_type), value)}"


REQUIRED_COLUMNS = {"type", "value", "severity", "source", "description", "tags", "firstSeen", "lastSeen", "observedCount", "raw"}


def load_data_frame(data_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Load IOC data from CSV, TXT (TSV), or JSON (array or JSONL) format.

    Auto-detects format by file extension:
      - .csv  → pd.read_csv
      - .txt  → pd.read_csv with tab/auto separator detection
      - .json → tries JSON array first, falls back to JSONL (line-delimited)
    """
    data_path = Path(data_path)
    ext = data_path.suffix.lower()

    if ext == ".json":
        raw = data_path.read_text(encoding="utf-8").strip()
        if raw.startswith("["):
            records = json.loads(raw)
        else:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if limit is not None:
            records = records[:limit]
        frame = pd.DataFrame(records)
    elif ext == ".txt":
        frame = pd.read_csv(data_path, sep=None, engine="python", nrows=limit)
    else:
        frame = pd.read_csv(data_path, nrows=limit)

    present = set(frame.columns)
    missing = REQUIRED_COLUMNS - present
    if missing:
        raise ValueError(
            f"Data file '{data_path}' is missing required columns: {', '.join(sorted(missing))}. "
            f"Got columns: {', '.join(sorted(present))}"
        )
    return frame


def load_db_frame(csv_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    frame = load_data_frame(csv_path, limit=limit)
    frame["tags_list"] = frame["tags"].map(parse_tags)
    frame["raw_obj"] = frame["raw"].map(parse_raw)
    frame["indicator_key"] = [
        build_indicator_key(ioc_type, value)
        for ioc_type, value in zip(frame["type"], frame["value"])
    ]
    frame["firstSeen_dt"] = pd.to_datetime(frame["firstSeen"], errors="coerce", utc=True)
    frame["lastSeen_dt"] = pd.to_datetime(frame["lastSeen"], errors="coerce", utc=True)
    frame["severity_label"] = frame["severity"].astype(str).str.lower()
    unknown_labels = sorted(set(frame["severity_label"].dropna()) - set(SEVERITY_TO_ID))
    if unknown_labels:
        raise ValueError(
            "Unexpected severity labels found in DB: "
            + ", ".join(unknown_labels)
            + ". Expected only: "
            + ", ".join(SEVERITY_ORDER)
        )
    frame["severity_id"] = frame["severity_label"].map(SEVERITY_TO_ID).astype(int)
    return frame


def apply_temporal_split(
    frame: pd.DataFrame,
    test_days: int,
    validation_days: int,
) -> DatasetBundle:
    max_time = frame["lastSeen_dt"].max()
    test_cutoff = max_time - pd.Timedelta(days=test_days)
    validation_cutoff = test_cutoff - pd.Timedelta(days=validation_days)

    split_labels = []
    seen_train_keys: set[str] = set()
    seen_validation_keys: set[str] = set()

    for _, row in frame.sort_values("lastSeen_dt").iterrows():
        key = row["indicator_key"]
        last_seen = row["lastSeen_dt"]
        if pd.isna(last_seen):
            split = "train"
        elif key in seen_train_keys:
            split = "train"
        elif key in seen_validation_keys:
            split = "validation"
        elif last_seen > test_cutoff:
            split = "test"
        elif last_seen > validation_cutoff:
            split = "validation"
            seen_validation_keys.add(key)
        else:
            split = "train"
            seen_train_keys.add(key)
        split_labels.append(split)

    split_frame = frame.copy()
    split_frame["split"] = split_labels
    return DatasetBundle(
        full=split_frame,
        train=split_frame[split_frame["split"] == "train"].reset_index(drop=True),
        validation=split_frame[split_frame["split"] == "validation"].reset_index(drop=True),
        test=split_frame[split_frame["split"] == "test"].reset_index(drop=True),
    )


def apply_group_smoke_split(frame: pd.DataFrame) -> DatasetBundle:
    unique_keys = sorted(frame["indicator_key"].dropna().unique().tolist())
    train_cut = int(len(unique_keys) * 0.7)
    validation_cut = int(len(unique_keys) * 0.85)
    train_keys = set(unique_keys[:train_cut])
    validation_keys = set(unique_keys[train_cut:validation_cut])

    split_frame = frame.copy()
    split_frame["split"] = "test"
    split_frame.loc[split_frame["indicator_key"].isin(train_keys), "split"] = "train"
    split_frame.loc[split_frame["indicator_key"].isin(validation_keys), "split"] = "validation"
    return DatasetBundle(
        full=split_frame,
        train=split_frame[split_frame["split"] == "train"].reset_index(drop=True),
        validation=split_frame[split_frame["split"] == "validation"].reset_index(drop=True),
        test=split_frame[split_frame["split"] == "test"].reset_index(drop=True),
    )


def apply_group_random_split(
    frame: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
) -> DatasetBundle:
    if train_frac <= 0 or validation_frac <= 0 or (train_frac + validation_frac) >= 1:
        raise ValueError("train_frac and validation_frac must be positive and sum to less than 1.")

    unique_keys = frame["indicator_key"].dropna().unique()
    keys = np.asarray(unique_keys, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    train_cut = int(len(keys) * train_frac)
    validation_cut = int(len(keys) * (train_frac + validation_frac))
    train_keys = set(keys[:train_cut].tolist())
    validation_keys = set(keys[train_cut:validation_cut].tolist())

    split_frame = frame.copy()
    split_frame["split"] = "test"
    split_frame.loc[split_frame["indicator_key"].isin(train_keys), "split"] = "train"
    split_frame.loc[split_frame["indicator_key"].isin(validation_keys), "split"] = "validation"
    return DatasetBundle(
        full=split_frame,
        train=split_frame[split_frame["split"] == "train"].reset_index(drop=True),
        validation=split_frame[split_frame["split"] == "validation"].reset_index(drop=True),
        test=split_frame[split_frame["split"] == "test"].reset_index(drop=True),
    )


def apply_group_stratified_split(
    frame: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
    label_col: str = "severity_label",
) -> DatasetBundle:
    if train_frac <= 0 or validation_frac <= 0 or (train_frac + validation_frac) >= 1:
        raise ValueError("train_frac and validation_frac must be positive and sum to less than 1.")

    grouped = (
        frame.groupby("indicator_key", dropna=False)[label_col]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0])
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    train_keys: set[str] = set()
    validation_keys: set[str] = set()

    for _, severity_group in grouped.groupby(label_col, dropna=False, sort=False):
        keys = severity_group["indicator_key"].to_numpy(dtype=object)
        rng.shuffle(keys)
        train_cut = int(len(keys) * train_frac)
        validation_cut = int(len(keys) * (train_frac + validation_frac))
        train_keys.update(keys[:train_cut].tolist())
        validation_keys.update(keys[train_cut:validation_cut].tolist())

    split_frame = frame.copy()
    split_frame["split"] = "test"
    split_frame.loc[split_frame["indicator_key"].isin(train_keys), "split"] = "train"
    split_frame.loc[split_frame["indicator_key"].isin(validation_keys), "split"] = "validation"
    return DatasetBundle(
        full=split_frame,
        train=split_frame[split_frame["split"] == "train"].reset_index(drop=True),
        validation=split_frame[split_frame["split"] == "validation"].reset_index(drop=True),
        test=split_frame[split_frame["split"] == "test"].reset_index(drop=True),
    )


def load_ip_sidecar(path: str | Path | None) -> dict[str, object]:
    if path is None or not Path(path).exists():
        return {"exact_ips": set(), "cidr_prefix_by_base": {}}
    cidr_prefix_by_base: dict[str, int] = {}
    exact_ips: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            if "/" in value:
                base, prefix = value.rsplit("/", 1)
                base = base.strip()
                if not base:
                    continue
                try:
                    prefix_int = int(prefix)
                    if "." in base:
                        pass
                    else:
                        base = f"{int(base)}"
                except ValueError:
                    exact_ips.add(value)
                    continue
                try:
                    import ipaddress
                    network = ipaddress.ip_network(f"{base}/{prefix_int}", strict=False)
                    network_addr = str(network.network_address)
                    cidr_prefix_by_base[network_addr] = prefix_int
                except ValueError:
                    exact_ips.add(value)
            else:
                exact_ips.add(value)
    return {"exact_ips": exact_ips, "cidr_prefix_by_base": cidr_prefix_by_base}


def load_phishing_sidecar(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["domain_norm", "target", "host_norm", "date_dt"])
    frame = pd.read_csv(path)
    frame["domain_norm"] = frame["domain"].astype(str).str.lower().str.strip(".")
    frame["host_norm"] = frame["host"].astype(str).str.lower().str.strip(".")
    frame["date_dt"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    return frame


def make_text_input(row: pd.Series) -> str:
    tags = " ".join(row.get("tags_list", []))
    description = str(row.get("description", ""))
    return " | ".join(
        part
        for part in [
            f"type: {row.get('type', '')}",
            f"source: {row.get('source', '')}",
            f"value: {row.get('value', '')}",
            f"description: {description}",
            f"tags: {tags}",
        ]
        if part
    )


def iter_split_frames(bundle: DatasetBundle) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "train", bundle.train
    yield "validation", bundle.validation
    yield "test", bundle.test