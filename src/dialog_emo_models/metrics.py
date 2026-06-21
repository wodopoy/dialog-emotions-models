"""Shared evaluation metrics for dialogue emotion models.

Two families:

- Quality metrics on soft distributions `y_true`, `y_pred` of shape `(n, len(EMOTIONS))`:
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


def reliability_curve(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Per-bin top-label reliability data — the source of truth for ECE and its plot.

    Bins the predictions by top-label confidence into ``n_bins`` equal-width
    ``(low, high]`` buckets and returns, per bin, the mean confidence, the empirical
    accuracy (how often the top label is the true primary), and the count. Empty
    bins carry ``nan`` for confidence/accuracy and ``0`` for count. A reliability
    diagram plots `accuracy` against `confidence`; ECE is the count-weighted mean
    gap between them.
    """
    true = _as_distribution(y_true)
    pred = _as_distribution(y_pred)
    confidence = pred.max(axis=1)
    correct = (pred.argmax(axis=1) == true.argmax(axis=1)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())
        counts[index] = count
        if count:
            conf[index] = float(confidence[in_bin].mean())
            acc[index] = float(correct[in_bin].mean())
    return {"edges": edges, "centers": centers, "confidence": conf, "accuracy": acc, "counts": counts}


def expected_calibration_error(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Top-label expected calibration error: |confidence - accuracy| over bins."""
    curve = reliability_curve(y_true, y_pred, n_bins=n_bins)
    counts = curve["counts"]
    total = int(counts.sum())
    if total == 0:
        return 0.0
    gaps = np.abs(curve["confidence"] - curve["accuracy"])
    gaps = np.where(counts > 0, gaps, 0.0)
    return float((gaps * counts).sum() / total)


# --- temperature calibration ----------------------------------------------


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def apply_temperature(
    logits: NDArray[np.float64],
    temperature: float,
) -> NDArray[np.float64]:
    """Softmax of pre-softmax `logits` divided by a scalar `temperature`.

    ``T > 1`` softens the distribution (lower confidence), ``T < 1`` sharpens it,
    ``T = 1`` is a no-op. The argmax — and therefore accuracy, F1, the confusion
    matrix — is invariant under any positive `T`; only the probabilities (and so
    KL, NLL, ECE) move. This is the single knob of temperature scaling
    (Guo et al., 2017), applied uniformly to any model via its `predict_logits`.
    """
    return _softmax(np.asarray(logits, dtype=float) / float(temperature))


def negative_log_likelihood(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Soft-label cross-entropy ``-sum p log q`` averaged over rows.

    This is the textbook objective temperature scaling minimizes; for one-hot
    labels it reduces to the usual NLL, and it differs from `kl` only by the
    (T-independent) entropy of `y_true`, so minimizing it also minimizes KL.
    """
    true = _as_distribution(y_true)
    pred = _as_distribution(y_pred)
    return float(-(true * np.log(pred)).sum(axis=1).mean())


def best_temperature(
    logits: NDArray[np.float64],
    y_true: NDArray[np.float64],
    *,
    objective: str = "ece",
    grid: NDArray[np.float64] | Sequence[float] | None = None,
    min_improve: float = 0.0,
) -> tuple[float, float]:
    """Scalar temperature minimizing `objective` of ``softmax(logits / T)`` vs `y_true`.

    `objective` is ``"ece"`` (top-label calibration error, the reported target) or
    ``"nll"`` (soft cross-entropy, the textbook smooth surrogate). The sweep runs
    over cached `logits`, so it is just a handful of softmaxes per candidate `T`.
    Ties are broken toward `T` closest to 1.0 — the least distortion of the
    original scores.

    `min_improve` is a deadband: keep ``T = 1`` unless the best candidate beats the
    objective *at* ``T = 1`` by at least this much. Binned ECE is a noisy finite-sample
    statistic, so for an already-calibrated model its validation minimum sits a hair
    below ``T = 1`` purely by sampling noise — a "win" that does not transfer to test.
    A small deadband (e.g. 0.005 on ECE) leaves such models alone and only departs
    from ``T = 1`` for corrections large enough to be real. Returns ``(T*, value@T*)``.
    """
    if objective not in ("ece", "nll"):
        raise ValueError(f"objective must be 'ece' or 'nll', got {objective!r}")
    # Include T=1 exactly so the no-op is a real candidate (else the nearest grid node
    # ~0.98 stands in for it and any reported sub-1 T could be a grid artifact).
    default_grid = np.union1d(np.geomspace(0.5, 20.0, 160), [1.0])
    candidates = default_grid if grid is None else np.asarray(grid, dtype=float)
    true = _as_distribution(y_true)

    def score_at(temperature: float) -> float:
        probs = apply_temperature(logits, temperature)
        if objective == "nll":
            return negative_log_likelihood(true, probs)
        return expected_calibration_error(true, probs)

    best_t, best_score = 1.0, float("inf")
    for temperature in candidates:
        score = score_at(float(temperature))
        better = score < best_score - 1e-12
        tie_closer = abs(score - best_score) <= 1e-12 and abs(temperature - 1.0) < abs(best_t - 1.0)
        if better or tie_closer:
            best_t, best_score = float(temperature), float(score)

    baseline = score_at(1.0)
    if baseline - best_score < min_improve:
        return 1.0, baseline
    return best_t, best_score


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
