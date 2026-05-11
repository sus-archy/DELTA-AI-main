from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_class_counts(labels: Iterable[int], num_classes: int) -> np.ndarray:
    label_array = np.asarray(list(labels), dtype=np.int64)
    return np.bincount(label_array, minlength=num_classes)


def compute_class_weight_array(labels: Iterable[int], num_classes: int) -> np.ndarray:
    counts = compute_class_counts(labels, num_classes)
    present_mask = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    if not present_mask.any():
        return weights
    total_present = counts[present_mask].sum()
    num_present_classes = int(present_mask.sum())
    weights[present_mask] = total_present / (num_present_classes * counts[present_mask])
    return weights


def compute_effective_number_weight_array(
    labels: Iterable[int],
    num_classes: int,
    beta: float = 0.999,
) -> np.ndarray:
    counts = compute_class_counts(labels, num_classes).astype(np.float64)
    present_mask = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    if not present_mask.any():
        return weights
    effective_num = (1.0 - np.power(beta, counts[present_mask])) / (1.0 - beta)
    raw_weights = 1.0 / np.maximum(effective_num, 1e-12)
    raw_weights = raw_weights / raw_weights.mean()
    weights[present_mask] = raw_weights.astype(np.float32)
    return weights


def compute_class_prior_array(labels: Iterable[int], num_classes: int) -> np.ndarray:
    counts = compute_class_counts(labels, num_classes).astype(np.float64)
    priors = np.zeros(num_classes, dtype=np.float32)
    total = counts.sum()
    if total <= 0:
        return priors
    priors = (counts / total).astype(np.float32)
    return priors


def compute_class_weight_map(labels: Iterable[int], num_classes: int) -> dict[int, float]:
    weights = compute_class_weight_array(labels, num_classes)
    return {
        class_id: float(weight)
        for class_id, weight in enumerate(weights)
        if weight > 0
    }


def compute_effective_number_weight_map(
    labels: Iterable[int],
    num_classes: int,
    beta: float = 0.999,
) -> dict[int, float]:
    weights = compute_effective_number_weight_array(labels, num_classes, beta=beta)
    return {
        class_id: float(weight)
        for class_id, weight in enumerate(weights)
        if weight > 0
    }


def build_sample_weights(
    labels: Iterable[int],
    num_classes: int,
    strategy: str = "inverse",
    beta: float = 0.999,
) -> np.ndarray:
    label_array = np.asarray(list(labels), dtype=np.int64)
    if strategy == "none":
        class_weights = np.ones(num_classes, dtype=np.float32)
    elif strategy == "effective":
        class_weights = compute_effective_number_weight_array(label_array, num_classes, beta=beta)
    else:
        class_weights = compute_class_weight_array(label_array, num_classes)
    return class_weights[label_array]


def summarize_class_balance(
    labels: Iterable[int],
    num_classes: int,
    label_names: list[str] | None = None,
    strategy: str = "inverse",
    beta: float = 0.999,
) -> dict[str, dict[str, float | int]]:
    label_array = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(label_array, minlength=num_classes)
    priors = compute_class_prior_array(label_array, num_classes)
    if strategy == "none":
        weights = np.where(counts > 0, 1.0, 0.0).astype(np.float32)
    elif strategy == "effective":
        weights = compute_effective_number_weight_array(label_array, num_classes, beta=beta)
    else:
        weights = compute_class_weight_array(label_array, num_classes)
    total = int(counts.sum())
    names = label_names or [str(index) for index in range(num_classes)]
    summary: dict[str, dict[str, float | int]] = {}
    for class_id in range(num_classes):
        label_name = names[class_id] if class_id < len(names) else str(class_id)
        count = int(counts[class_id])
        fraction = float(count / total) if total else 0.0
        summary[label_name] = {
            "count": count,
            "fraction": fraction,
            "weight": float(weights[class_id]),
            "prior": float(priors[class_id]),
        }
    return summary
