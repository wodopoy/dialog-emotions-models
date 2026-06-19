"""Shared evaluation metrics for dialogue emotion models.

Two families:

- Quality metrics on soft distributions `y_true`, `y_pred` of shape `(n, 6)`:
  hard top-1 metrics (accuracy, F1), distribution metrics (MAE, MSE, KL, JS),
  calibration (ECE), per-class F1, and the confusion matrix.
- Deployment metrics: on-disk size, load time, single-message latency.

Everything is regenerable from cached probability arrays so a leaderboard can be
rebuilt without re-running inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support

from dialog_emo_models.schema import EMOTIONS

EPS = 1e-12


def _as_distribution(array: NDArray[np.float64] | Sequence[Sequence[float]]) -> NDArray[np.float64]:
    matrix = np.clip(np.asarray(array, dtype=float), EPS, None)
    return matrix / matrix.sum(axis=1, keepdims=True)


def _kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    return (p * (np.log(p) - np.log(q))).sum(axis=1)


def quality_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> dict[str, float]:
    """Scalar quality metrics for one (y_true, y_pred) pair of soft distributions."""
    true = _as_distribution(y_true)
    pred = _as_distribution(y_pred)
    true_primary = true.argmax(axis=1)
    pred_top = pred.argmax(axis=1)
    mixture = 0.5 * (true + pred)
    return {
        "top1_hit": float(np.mean((np.asarray(y_true, dtype=float) > 0)[
            np.arange(len(pred_top)), pred_top
        ])),
        "primary_accuracy": float(np.mean(pred_top == true_primary)),
        "macro_f1": float(f1_score(true_primary, pred_top, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(true_primary, pred_top, average="weighted", zero_division=0)
        ),
        "micro_f1": float(f1_score(true_primary, pred_top, average="micro", zero_division=0)),
        "mae": float(np.abs(true - pred).mean()),
        "mse": float(((true - pred) ** 2).mean()),
        "kl": float(_kl(true, pred).mean()),
        "js": float((0.5 * _kl(true, mixture) + 0.5 * _kl(pred, mixture)).mean()),
        "ece": expected_calibration_error(y_true, y_pred),
    }


def expected_calibration_error(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Top-label expected calibration error: |confidence - accuracy| over bins."""
    true = _as_distribution(y_true)
    pred = _as_distribution(y_pred)
    confidence = pred.max(axis=1)
    correct = (pred.argmax(axis=1) == true.argmax(axis=1)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())
        if count:
            ece += abs(confidence[in_bin].mean() - correct[in_bin].mean()) * count / len(confidence)
    return float(ece)


def per_class_f1(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> dict[str, dict[str, float]]:
    """Per-emotion precision / recall / F1 / support on the primary (argmax) label."""
    true_primary = _as_distribution(y_true).argmax(axis=1)
    pred_top = _as_distribution(y_pred).argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_primary,
        pred_top,
        labels=range(len(EMOTIONS)),
        zero_division=0,
    )
    return {
        emotion: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, emotion in enumerate(EMOTIONS)
    }


def confusion_matrix(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.int64]:
    """6x6 confusion matrix over primary (argmax) labels, rows=true, cols=pred."""
    true_primary = _as_distribution(y_true).argmax(axis=1)
    pred_top = _as_distribution(y_pred).argmax(axis=1)
    return _sk_confusion_matrix(true_primary, pred_top, labels=range(len(EMOTIONS)))


# --- deployment metrics ---------------------------------------------------


def model_size_mb(path: str | Path) -> float:
    """On-disk size in MB of a saved model (file or directory)."""
    target = Path(path)
    if target.is_dir():
        total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    else:
        total = target.stat().st_size
    return total / (1024 * 1024)


def measure_load_time_ms(
    loader: Callable[[], object],
    *,
    repeats: int = 3,
) -> float:
    """Median wall-clock time to load a saved model from disk."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        loader()
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples))


def measure_latency_ms(
    model,
    texts: Sequence[str],
    *,
    sample: int = 200,
    repeats: int = 3,
) -> dict[str, float]:
    """Single-message (batch=1) inference latency: p50 and p95 in ms.

    Keyboards score one message at a time, so we time `predict_proba([text])`
    per message rather than a large batch.
    """
    pool = [str(text) for text in list(texts)[:sample]]
    if not pool:
        return {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0}
    model.predict_proba(pool[:8])  # warmup
    per_message = []
    for _ in range(repeats):
        for text in pool:
            start = time.perf_counter()
            model.predict_proba([text])
            per_message.append((time.perf_counter() - start) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(per_message, 50)),
        "latency_p95_ms": float(np.percentile(per_message, 95)),
    }
