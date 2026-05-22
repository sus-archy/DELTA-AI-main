from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


SOURCE_SCORES = {
    "urlhaus": 90,
    "threatfox": 90,
    "otx": 85,
    "emergingthreats": 85,
    "spamhaus": 90,
    "ciarmy": 80,
    "phishtank": 75,
    "phishstats": 75,
    "bazaar": 70,
    "malshare": 75,
    "dshield_openioc": 60,
    "dshield_threatfeeds": 60,
    "bazaar_yara": 65,
    "default": 50,
}

TYPE_SEVERITY_MAP = {
    "hash": {"base": "high", "score": 80},
    "ip": {"base": "medium", "score": 60},
    "domain": {"base": "medium", "score": 65},
    "url": {"base": "high", "score": 75},
    "email": {"base": "medium", "score": 60},
    "file": {"base": "high", "score": 80},
    "default": {"base": "medium", "score": 50},
}

CRITICAL_KEYWORDS = [
    "ransomware",
    "apt",
    "advanced persistent",
    "zero-day",
    "exploit",
    "backdoor",
    "trojan",
    "rat",
    "remote access",
    "c2",
    "command and control",
    "cryptominer",
    "miner",
    "botnet",
    "ddos",
]

HIGH_KEYWORDS = [
    "malware",
    "phishing",
    "phish",
    "scam",
    "fraud",
    "stealer",
    "banking",
    "credential",
    "keylogger",
    "spyware",
    "adware",
]

MEDIUM_KEYWORDS = ["suspicious", "potentially unwanted", "pua", "pup", "unwanted"]


def _source_key(value: str | None) -> str:
    return (value or "").lower().split("_")[0]


def _tags_text(tags: list[str] | str | None) -> str:
    if isinstance(tags, list):
        return " ".join(tags).lower()
    if isinstance(tags, str):
        return tags.lower()
    return ""


def calculate_severity_score(ioc: dict[str, Any]) -> int:
    score = 50.0
    source_key = _source_key(ioc.get("source"))
    source_score = SOURCE_SCORES.get(source_key, SOURCE_SCORES["default"])
    score += (source_score - 50) * 0.3

    type_key = str(ioc.get("type", ""))
    type_info = TYPE_SEVERITY_MAP.get(type_key, TYPE_SEVERITY_MAP["default"])
    score += (type_info["score"] - 50) * 0.3

    description = str(ioc.get("description", "")).lower()
    combined_text = f"{description} {_tags_text(ioc.get('tags'))}"
    if any(keyword in combined_text for keyword in CRITICAL_KEYWORDS):
        score += 20
    elif any(keyword in combined_text for keyword in HIGH_KEYWORDS):
        score += 10
    elif any(keyword in combined_text for keyword in MEDIUM_KEYWORDS):
        score += 5

    observed_count = ioc.get("observedCount")
    if observed_count is not None:
        try:
            observed_count = float(observed_count)
            if observed_count > 1:
                score += min(10, math.log(observed_count) * 2)
        except (TypeError, ValueError):
            pass

    confidence = ioc.get("confidence")
    if confidence is not None:
        try:
            score += (float(confidence) - 50) * 0.1
        except (TypeError, ValueError):
            pass

    return max(0, min(100, round(score)))


def score_to_severity(score: int) -> str:
    if score >= 85:
        return "critical"
    if score >= 70:
        return "high"
    if score >= 50:
        return "medium"
    if score >= 30:
        return "low"
    return "info"


def calculate_confidence(ioc: dict[str, Any]) -> int:
    source_key = _source_key(ioc.get("source"))
    confidence = SOURCE_SCORES.get(source_key, SOURCE_SCORES["default"])
    description = str(ioc.get("description", ""))
    if len(description) > 20:
        confidence += 5
    tags = ioc.get("tags")
    if isinstance(tags, list) and tags:
        confidence += 5
    observed_count = ioc.get("observedCount")
    if observed_count is not None:
        try:
            observed_count = float(observed_count)
            if observed_count > 3:
                confidence += min(10, observed_count * 2)
        except (TypeError, ValueError):
            pass
    return max(0, min(100, round(confidence)))


def extract_teacher_concepts(ioc: dict[str, Any], score: int | None = None) -> dict[str, Any]:
    description = str(ioc.get("description", "")).lower()
    tags_text = _tags_text(ioc.get("tags"))
    combined_text = f"{description} {tags_text}"
    if score is None:
        score = calculate_severity_score(ioc)
    observed_bucket = "single"
    try:
        observed = float(ioc.get("observedCount", 0))
        if observed >= 100:
            observed_bucket = "very-high"
        elif observed >= 10:
            observed_bucket = "high"
        elif observed >= 2:
            observed_bucket = "multi"
    except (TypeError, ValueError):
        observed_bucket = "unknown"

    return {
        "source_reliability_bucket": SOURCE_SCORES.get(_source_key(ioc.get("source")), 50),
        "type_risk_bucket": TYPE_SEVERITY_MAP.get(str(ioc.get("type", "")), TYPE_SEVERITY_MAP["default"])["score"],
        "has_critical_keyword": any(keyword in combined_text for keyword in CRITICAL_KEYWORDS),
        "has_high_keyword": any(keyword in combined_text for keyword in HIGH_KEYWORDS),
        "has_medium_keyword": any(keyword in combined_text for keyword in MEDIUM_KEYWORDS),
        "observation_bucket": observed_bucket,
        "score_bucket": score_to_severity(score),
    }


def classify_ioc(ioc: dict[str, Any]) -> dict[str, Any]:
    severity_score = calculate_severity_score(ioc)
    severity = score_to_severity(severity_score)
    confidence = ioc.get("confidence")
    if confidence is None or confidence == "":
        confidence = calculate_confidence(ioc)
    return {
        "severity": severity,
        "severityScore": severity_score,
        "confidence": confidence,
        "concepts": extract_teacher_concepts(ioc, severity_score),
    }


def classify_many(iocs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [classify_ioc(ioc) for ioc in iocs]


def classify_many_with_node(iocs: list[dict[str, Any]], script_path: str | Path) -> list[dict[str, Any]]:
    payload = json.dumps(iocs)
    node_script = """
const fs = require('fs');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const teacher = require(process.argv[1]);
const result = data.map(item => teacher.classifyIOC(item));
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", node_script, str(script_path)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)
