from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidences >= left) & (confidences < right if right < 1 else confidences <= right)
        if mask.any():
            accuracy = correctness[mask].mean()
            confidence = confidences[mask].mean()
            ece += abs(accuracy - confidence) * mask.mean()
    return float(ece)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    one_hot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def balanced_accuracy_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> float:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = []
    for index in range(len(labels)):
        support = matrix[index, :].sum()
        if support > 0:
            recalls.append(matrix[index, index] / support)
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def summarize_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    label_names: list[str] | None = None,
    labels: list[int] | None = None,
    proba_labels: list[int] | None = None,
    score_true: np.ndarray | None = None,
    score_pred: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    report_target_names = label_names
    if label_names is not None and len(label_names) != len(labels):
        report_target_names = [label_names[label] for label in labels]
    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "balanced_accuracy": balanced_accuracy_from_labels(y_true, y_pred, labels),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=report_target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if label_names is not None:
            aligned = np.zeros((len(y_true), len(label_names)), dtype=y_proba.dtype)
            if proba_labels is None:
                if y_proba.shape[1] == len(label_names):
                    proba_labels = list(range(len(label_names)))
                elif y_proba.shape[1] == len(labels):
                    proba_labels = list(labels)
                else:
                    raise ValueError("Unable to align probability columns with label names.")
            if len(proba_labels) != y_proba.shape[1]:
                raise ValueError("Probability column count does not match proba_labels length.")
            aligned[:, proba_labels] = y_proba
            y_proba = aligned
        summary["ece"] = expected_calibration_error(y_true, y_proba)
        summary["brier"] = multiclass_brier(y_true, y_proba)
    if score_true is not None and score_pred is not None:
        summary["teacher_score_mae"] = float(mean_absolute_error(score_true, score_pred))
    return summary


def write_metrics(path: str | Path, metrics: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
