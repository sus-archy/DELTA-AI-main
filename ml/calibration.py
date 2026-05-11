from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


EPSILON = 1e-12


def logits_to_proba(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def proba_to_logits(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=np.float64), EPSILON, 1.0)
    return np.log(proba)


def multiclass_negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    row_index = np.arange(len(labels))
    return float(-np.mean(log_probs[row_index, labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, min_temp: float = 0.05, max_temp: float = 5.0) -> float:
    result = minimize_scalar(
        lambda temperature: multiclass_negative_log_likelihood(logits / float(temperature), labels),
        bounds=(min_temp, max_temp),
        method="bounded",
    )
    return float(result.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits_to_proba(np.asarray(logits, dtype=np.float64) / float(temperature))


def write_prediction_artifact(
    run_dir: str | Path,
    split_name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    logits: np.ndarray | None = None,
) -> None:
    payload = {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "proba": np.asarray(proba, dtype=np.float32),
        "pred": np.asarray(np.argmax(proba, axis=1), dtype=np.int64),
    }
    if logits is not None:
        payload["logits"] = np.asarray(logits, dtype=np.float32)
    output_path = Path(run_dir) / f"{split_name}_predictions.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
