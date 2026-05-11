from __future__ import annotations

import ipaddress
import math
from urllib.parse import urlparse

import pandas as pd

from .data import make_text_input


def _safe_len(value: object) -> int:
    return len(str(value or ""))


def _extract_domain_features(value: str) -> tuple[int, str]:
    parts = str(value).lower().split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    depth = max(0, len(parts) - 1)
    return depth, tld


def _extract_url_features(value: str) -> tuple[str, str, int]:
    parsed = urlparse(str(value))
    path_depth = len([part for part in parsed.path.split("/") if part])
    return parsed.scheme.lower(), parsed.netloc.lower(), path_depth


def _ip_matches_cidr(ip_str: str, cidr_dict: dict[str, int]) -> tuple[int, int]:
    try:
        ip = ipaddress.ip_address(ip_str)
        for base, prefix in cidr_dict.items():
            try:
                network = ipaddress.ip_network(f"{base}/{prefix}", strict=False)
                if ip in network:
                    return 1, prefix
            except ValueError:
                continue
    except ValueError:
        pass
    return 0, 0


def _compute_ip_features(
    values: pd.Series,
    ip_sidecar: dict[str, object],
) -> tuple[list[int], list[int], list[int], list[int]]:
    exact_ips = ip_sidecar.get("exact_ips", set())
    cidr_dict = ip_sidecar.get("cidr_prefix_by_base", {})

    is_in_ip_txt = []
    is_in_spamhaus_cidr = []
    cidr_prefix_len = []
    cidr_base_match = []

    for value in values:
        val_str = str(value)
        is_in_ip_txt.append(1 if val_str in exact_ips else 0)

        if val_str in cidr_dict:
            is_in_spamhaus_cidr.append(1)
            cidr_prefix_len.append(cidr_dict[val_str])
            cidr_base_match.append(1)
        else:
            matched, prefix = _ip_matches_cidr(val_str, cidr_dict)
            is_in_spamhaus_cidr.append(matched)
            cidr_prefix_len.append(prefix)
            cidr_base_match.append(1 if matched else 0)

    return is_in_ip_txt, is_in_spamhaus_cidr, cidr_prefix_len, cidr_base_match


def _compute_phishing_features(
    frame: pd.DataFrame,
    phishing_sidecar: pd.DataFrame,
) -> tuple[list[str], list[int], list[str], list[int]]:
    phishing_by_domain: dict[str, int] = {}
    target_mode: dict[str, str] = {}

    if not phishing_sidecar.empty and "domain_norm" in phishing_sidecar.columns:
        phishing_by_domain = phishing_sidecar.groupby("domain_norm").size().to_dict()
        target_mode = (
            phishing_sidecar.groupby("domain_norm")["target"]
            .agg(lambda values: pd.Series(values).mode().iloc[0] if not pd.Series(values).mode().empty else "")
            .to_dict()
        )

    normalized_domains = []
    phish_seen = []
    phish_target = []
    phish_flag = []

    for ioc_type, value, host in zip(frame["type"], frame["value"], frame["url_host"]):
        if ioc_type in {"domain", "hostname"}:
            domain = str(value).lower().strip(".")
        elif ioc_type == "url":
            domain = str(host).lower().strip(".")
        else:
            domain = ""
        normalized_domains.append(domain)
        count = phishing_by_domain.get(domain, 0)
        phish_seen.append(count)
        phish_target.append(target_mode.get(domain, ""))
        phish_flag.append(1 if count > 0 else 0)

    return normalized_domains, phish_seen, phish_target, phish_flag


def augment_features(
    frame: pd.DataFrame,
    ip_sidecar: dict[str, object] | None = None,
    phishing_sidecar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = frame.copy()

    ip_sidecar = ip_sidecar or {"exact_ips": set(), "cidr_prefix_by_base": {}}
    phishing_sidecar = phishing_sidecar or pd.DataFrame(columns=["domain_norm", "target"])

    now = enriched["lastSeen_dt"].max()

    enriched["text_input"] = enriched.apply(make_text_input, axis=1)
    enriched["desc_len"] = enriched["description"].map(_safe_len)
    enriched["tag_count"] = enriched["tags_list"].map(len)
    enriched["value_len"] = enriched["value"].map(_safe_len)
    enriched["age_days"] = (enriched["lastSeen_dt"] - enriched["firstSeen_dt"]).dt.total_seconds().fillna(0) / 86400.0
    enriched["recency_days"] = (now - enriched["lastSeen_dt"]).dt.total_seconds().fillna(0) / 86400.0
    enriched["observedCount_num"] = pd.to_numeric(enriched["observedCount"], errors="coerce").fillna(0)
    enriched["log_observedCount"] = enriched["observedCount_num"].map(lambda value: math.log1p(value))

    enriched["is_hash_like"] = enriched["type"].isin(["sha256", "sha1", "md5"]).astype(int)
    enriched["is_url"] = enriched["type"].eq("url").astype(int)
    enriched["is_domain"] = enriched["type"].isin(["domain", "hostname"]).astype(int)
    enriched["is_ip"] = enriched["type"].isin(["ipv4", "ipv6"]).astype(int)

    domain_depths = []
    tlds = []
    url_schemes = []
    url_hosts = []
    url_path_depths = []
    for ioc_type, value in zip(enriched["type"], enriched["value"]):
        if ioc_type in {"domain", "hostname"}:
            depth, tld = _extract_domain_features(str(value))
            domain_depths.append(depth)
            tlds.append(tld)
            url_schemes.append("")
            url_hosts.append(str(value).lower())
            url_path_depths.append(0)
        elif ioc_type == "url":
            scheme, host, depth = _extract_url_features(str(value))
            url_schemes.append(scheme)
            url_hosts.append(host)
            url_path_depths.append(depth)
            if host:
                depth_host, tld = _extract_domain_features(host)
                domain_depths.append(depth_host)
                tlds.append(tld)
            else:
                domain_depths.append(0)
                tlds.append("")
        else:
            domain_depths.append(0)
            tlds.append("")
            url_schemes.append("")
            url_hosts.append("")
            url_path_depths.append(0)

    enriched["domain_depth"] = domain_depths
    enriched["tld"] = tlds
    enriched["url_scheme"] = url_schemes
    enriched["url_host"] = url_hosts
    enriched["url_path_depth"] = url_path_depths

    ip_feat = _compute_ip_features(enriched["value"], ip_sidecar)
    enriched["is_in_ip_txt"] = ip_feat[0]
    enriched["is_in_spamhaus_cidr"] = ip_feat[1]
    enriched["cidr_prefix_len"] = ip_feat[2]
    enriched["cidr_base_match"] = ip_feat[3]

    phish_feat = _compute_phishing_features(enriched, phishing_sidecar)
    enriched["normalized_domain"] = phish_feat[0]
    enriched["phish_domain_seen_count"] = phish_feat[1]
    enriched["phish_target_mode"] = phish_feat[2]
    enriched["phish_domain_seen_flag"] = phish_feat[3]
    return enriched


FEATURE_COLUMNS_NUMERIC = [
    "desc_len",
    "tag_count",
    "value_len",
    "age_days",
    "recency_days",
    "observedCount_num",
    "log_observedCount",
    "is_hash_like",
    "is_url",
    "is_domain",
    "is_ip",
    "domain_depth",
    "url_path_depth",
    "is_in_ip_txt",
    "is_in_spamhaus_cidr",
    "cidr_prefix_len",
    "cidr_base_match",
    "phish_domain_seen_count",
    "phish_domain_seen_flag",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "type",
    "source",
    "tld",
    "url_scheme",
    "phish_target_mode",
]