from __future__ import annotations

import numpy as np

from dialog_emo_models.metrics import (
    confusion_matrix,
    measure_latency_ms,
    model_size_mb,
    per_class_f1,
    quality_metrics,
)
from dialog_emo_models.models import DummyEmotionModel
from dialog_emo_models.schema import EMOTIONS

_IDENTITY = np.eye(len(EMOTIONS))


def test_quality_metrics_perfect_prediction() -> None:
    metrics = quality_metrics(_IDENTITY, _IDENTITY)

    assert metrics["primary_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["mae"] < 1e-3
    assert metrics["kl"] < 1e-3
    assert metrics["js"] < 1e-3
    assert metrics["ece"] < 1e-2


def test_quality_metrics_keys_and_bounds() -> None:
    rng = np.random.default_rng(0)
    pred = rng.dirichlet(np.ones(len(EMOTIONS)), size=50)
    metrics = quality_metrics(_IDENTITY[rng.integers(0, len(EMOTIONS), 50)], pred)

    assert set(metrics) == {
        "top1_hit", "primary_accuracy", "macro_f1", "weighted_f1",
        "micro_f1", "mae", "mse", "kl", "js", "ece",
    }
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["js"] >= 0.0  # JS is bounded (unlike KL)


def test_per_class_and_confusion() -> None:
    classes = per_class_f1(_IDENTITY, _IDENTITY)
    assert set(classes) == set(EMOTIONS)
    assert all(classes[e]["f1"] == 1.0 for e in EMOTIONS)

    matrix = confusion_matrix(_IDENTITY, _IDENTITY)
    assert matrix.shape == (len(EMOTIONS), len(EMOTIONS))
    assert int(matrix.trace()) == len(EMOTIONS)


def test_deployment_metrics(tmp_path) -> None:
    blob = tmp_path / "model.bin"
    blob.write_bytes(b"\x00" * (2 * 1024 * 1024))
    assert abs(model_size_mb(blob) - 2.0) < 1e-6

    latency = measure_latency_ms(DummyEmotionModel(), ["привет", "пока", "ок"], sample=3, repeats=2)
    assert latency["latency_p95_ms"] >= latency["latency_p50_ms"] >= 0.0
