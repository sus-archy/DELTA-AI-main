"""Centralized label definitions. All modules should import from here."""

from __future__ import annotations

SEVERITY_ORDER: list[str] = ["medium", "high", "critical"]
SEVERITY_TO_ID: dict[str, int] = {label: index for index, label in enumerate(SEVERITY_ORDER)}
ID_TO_SEVERITY: dict[int, str] = {index: label for index, label in enumerate(SEVERITY_ORDER)}
NUM_CLASSES: int = len(SEVERITY_ORDER)