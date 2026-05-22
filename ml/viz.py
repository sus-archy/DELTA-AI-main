from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score, average_precision_score
except ImportError:
    RocCurveDisplay = None
    PrecisionRecallDisplay = None


def _check_imports():
    if plt is None:
        raise ImportError("matplotlib is required for visualizations. Install with: pip install matplotlib")


LABEL_COLORS = {"medium": "#2196F3", "high": "#FF9800", "critical": "#F44336"}
LABEL_NAMES = ["medium", "high", "critical"]


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def ensure_plots_dir(run_dir: str | Path) -> Path:
    plots_dir = Path(run_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def save_training_history(run_dir: str | Path, history: list[dict[str, Any]]) -> None:
    """Save per-epoch training metrics as JSON for later plotting."""
    path = Path(run_dir) / "training_history.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_training_history(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "training_history.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    _check_imports()
    if labels is None:
        labels = LABEL_NAMES
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = "d"
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], fmt), ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_classification_report_bar(
    report: dict[str, Any],
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    _check_imports()
    if labels is None:
        labels = LABEL_NAMES
    metrics_by_class = {label: {} for label in labels}
    for label in labels:
        if label in report:
            metrics_by_class[label] = {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
            }
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.25
    for i, metric in enumerate(["precision", "recall", "f1-score"]):
        values = [metrics_by_class[label].get(metric, 0) for label in labels]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, and F1 per Class")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_training_history(history: list[dict[str, Any]], save_path: str | Path | None = None) -> plt.Figure:
    _check_imports()
    if not history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No training history available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Training History")
        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
        return fig

    df = pd.DataFrame(history)
    epochs = df.get("epoch", range(len(history)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in [
        (axes[0], "loss", "Loss"),
        (axes[1], "macro_f1", "Macro-F1"),
    ]:
        for split, style, color in [("train", "-o", "#2196F3"), ("validation", "-s", "#FF9800")]:
            col = f"{split}_{metric}"
            if col in df.columns:
                ax.plot(epochs, df[col], style, label=f"{split.capitalize()}", color=color, markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{' vs '.join(s.capitalize() for s in ['train', 'validation'])} {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_class_distribution(
    y: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Class Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    _check_imports()
    if labels is None:
        labels = LABEL_NAMES
    counts = pd.Series(y).value_counts().reindex(range(len(labels)), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [LABEL_COLORS.get(labels[i], "#607D8B") for i in range(len(labels))]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(1, v * 0.02),
                str(int(v)), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticklabels(labels)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_feature_importance(
    model: Any = None,
    coefficients: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
) -> plt.Figure:
    _check_imports()
    if coefficients is None and model is not None:
        if hasattr(model, "coef_"):
            coefficients = model.coef_
        elif hasattr(model, "feature_importances_"):
            coefficients = model.feature_importances_

    if coefficients is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Model does not expose feature importance", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
        return fig

    coefficients = np.asarray(coefficients)
    if coefficients.ndim == 2:
        coefficients = np.abs(coefficients).mean(axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    indices = np.argsort(np.abs(coefficients))[-top_n:]
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(range(len(indices)), np.abs(coefficients[indices]), color="#2f6bff")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Coefficient Magnitude (|weight|)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_pr_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    _check_imports()
    if labels is None:
        labels = LABEL_NAMES
    n_classes = y_proba.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, curve_type, score_func, display_cls in [
        (axes[0], "ROC", lambda y, p: roc_auc_score(y, p) if roc_auc_score else None, RocCurveDisplay),
        (axes[1], "Precision-Recall", lambda y, p: average_precision_score(y, p) if average_precision_score else None, PrecisionRecallDisplay),
    ]:
        for i in range(n_classes):
            y_bin = (y_true == i).astype(int)
            color = list(LABEL_COLORS.values())[i] if i < len(LABEL_COLORS) else None
            label_name = labels[i] if i < len(labels) else f"class_{i}"
            if curve_type == "ROC" and display_cls is not None:
                display_cls.from_predictions(y_bin, y_proba[:, i], ax=ax, name=label_name, color=color)
            elif display_cls is not None:
                display_cls.from_predictions(y_bin, y_proba[:, i], ax=ax, name=label_name, color=color)
            score = score_func(y_bin, y_proba[:, i])
            if score is not None:
                ax.text(0.6, 0.3 - i * 0.05, f"{label_name} AUC={score:.3f}", transform=ax.transAxes,
                        fontsize=9, color=color)
        ax.set_title(f"{curve_type} Curve (One-vs-Rest)")
        ax.legend(loc="lower right" if curve_type == "ROC" else "upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_threshold_distribution(
    y_proba: np.ndarray,
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    _check_imports()
    if labels is None:
        labels = LABEL_NAMES
    n_classes = y_proba.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        label_name = labels[i] if i < len(labels) else f"class_{i}"
        color = list(LABEL_COLORS.values())[i] if i < len(LABEL_COLORS) else "#607D8B"
        ax.hist(y_proba[:, i], bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"{label_name}\nConfidence Distribution")
        ax.yaxis.set_major_formatter(PercentFormatter(ymax=len(y_proba)))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def generate_all_plots(
    run_dir: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics_dict: dict[str, Any],
    labels: list[str] | None = None,
    model: Any = None,
    feature_names: list[str] | None = None,
    training_history: list[dict[str, Any]] | None = None,
    split_name: str = "test",
) -> Path:
    """Generate every visualization and save plots into ``run_dir/plots/``.

    Returns the plots directory path.
    """
    plots_dir = ensure_plots_dir(run_dir)
    if labels is None:
        labels = LABEL_NAMES

    report = metrics_dict.get("report", {})

    cm = metrics_dict.get("confusion_matrix")
    if cm:
        plot_confusion_matrix(cm, labels=labels, save_path=str(plots_dir / f"{split_name}_confusion_matrix.png"))
        plt.close("all")

    if report:
        plot_classification_report_bar(
            report, labels=labels,
            save_path=str(plots_dir / f"{split_name}_classification_report.png"),
        )
        plt.close("all")

    plot_class_distribution(
        y_true, labels=labels,
        title=f"{split_name.capitalize()} Set Class Distribution",
        save_path=str(plots_dir / f"{split_name}_class_distribution.png"),
    )
    plt.close("all")

    if report:
        plot_class_distribution(
            y_pred, labels=labels,
            title=f"{split_name.capitalize()} Set Predicted Distribution",
            save_path=str(plots_dir / f"{split_name}_predicted_distribution.png"),
        )
        plt.close("all")

    if model is not None:
        plot_feature_importance(
            model=model, feature_names=feature_names,
            save_path=str(plots_dir / "feature_importance.png"),
        )
        plt.close("all")

    if y_proba is not None:
        plot_pr_roc_curves(
            y_true, y_proba, labels=labels,
            save_path=str(plots_dir / f"{split_name}_pr_roc_curves.png"),
        )
        plt.close("all")

        plot_threshold_distribution(
            y_proba, labels=labels,
            save_path=str(plots_dir / f"{split_name}_threshold_distribution.png"),
        )
        plt.close("all")

    history = training_history or load_training_history(run_dir)
    if history:
        save_training_history(run_dir, history)
        plot_training_history(
            history,
            save_path=str(plots_dir / "training_history.png"),
        )
        plt.close("all")

    return plots_dir
