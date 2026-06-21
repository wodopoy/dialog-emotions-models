from __future__ import annotations

import numpy as np

from dialog_emo_models.metrics import (
    apply_temperature,
    best_temperature,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)
from dialog_emo_models.schema import EMOTIONS


def _overconfident_logits(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Peaked logits whose argmax is right ~60% of the time -> badly overconfident."""
    k = len(EMOTIONS)
    true = rng.integers(0, k, size=n)
    pred = true.copy()
    flip = rng.random(n) < 0.4
    pred[flip] = (pred[flip] + rng.integers(1, k, size=int(flip.sum()))) % k
    logits = np.full((n, k), -8.0)
    logits[np.arange(n), pred] = 8.0  # near one-hot -> confidence ~1.0
    y_true = np.eye(k)[true]
    return logits, y_true


def test_temperature_one_is_identity() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, len(EMOTIONS)))
    shifted = logits - logits.max(axis=1, keepdims=True)
    softmax = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    assert np.allclose(apply_temperature(logits, 1.0), softmax)


def test_higher_temperature_softens_distribution() -> None:
    logits = np.array([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sharp = apply_temperature(logits, 1.0).max()
    soft = apply_temperature(logits, 4.0).max()
    assert soft < sharp  # less confident
    assert soft > 1.0 / len(EMOTIONS)  # but not yet uniform


def test_temperature_preserves_argmax_and_accuracy() -> None:
    rng = np.random.default_rng(1)
    logits, _ = _overconfident_logits(rng, 200)
    base = apply_temperature(logits, 1.0).argmax(axis=1)
    for t in (0.5, 2.0, 7.0):
        assert np.array_equal(apply_temperature(logits, t).argmax(axis=1), base)


def test_ece_objective_reduces_calibration_error_on_heldout() -> None:
    rng = np.random.default_rng(2)
    fit_logits, fit_y = _overconfident_logits(rng, 600)
    test_logits, test_y = _overconfident_logits(rng, 600)

    t_star, _ = best_temperature(fit_logits, fit_y, objective="ece")
    assert t_star > 1.0  # overconfident models want softening

    raw = expected_calibration_error(test_y, apply_temperature(test_logits, 1.0))
    cal = expected_calibration_error(test_y, apply_temperature(test_logits, t_star))
    assert cal < raw


def test_nll_objective_also_softens_and_lowers_nll() -> None:
    rng = np.random.default_rng(3)
    logits, y = _overconfident_logits(rng, 600)
    t_star, _ = best_temperature(logits, y, objective="nll")
    raw = negative_log_likelihood(y, apply_temperature(logits, 1.0))
    cal = negative_log_likelihood(y, apply_temperature(logits, t_star))
    assert t_star > 1.0
    assert cal < raw


def test_deadband_holds_t1_when_gain_is_below_threshold() -> None:
    # Directly exercise the deadband: measure the ECE gain the free fit achieves,
    # then a deadband above it must hold T=1, and a deadband below it must not.
    rng = np.random.default_rng(4)
    logits, y = _overconfident_logits(rng, 600)
    t_free, _ = best_temperature(logits, y, objective="ece", min_improve=0.0)
    base = expected_calibration_error(y, apply_temperature(logits, 1.0))
    gain = base - expected_calibration_error(y, apply_temperature(logits, t_free))
    assert gain > 0.0 and t_free != 1.0  # there is a real correction here

    t_hi, _ = best_temperature(logits, y, objective="ece", min_improve=gain + 0.05)
    t_lo, _ = best_temperature(logits, y, objective="ece", min_improve=gain * 0.5)
    assert t_hi == 1.0  # deadband above the gain -> stay put
    assert t_lo == t_free  # deadband below the gain -> take the correction


def test_reliability_curve_reproduces_ece() -> None:
    rng = np.random.default_rng(5)
    logits, y = _overconfident_logits(rng, 500)
    probs = apply_temperature(logits, 2.0)
    curve = reliability_curve(y, probs, n_bins=10)
    counts = curve["counts"]
    gaps = np.where(counts > 0, np.abs(curve["confidence"] - curve["accuracy"]), 0.0)
    manual = float((gaps * counts).sum() / counts.sum())
    assert manual == expected_calibration_error(y, probs)
    assert int(counts.sum()) == len(y)
