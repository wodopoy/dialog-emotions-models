from __future__ import annotations

import numpy as np

from dialog_emo_models.models import LabelPriorEmotionModel, MajorityClassEmotionModel
from dialog_emo_models.schema import EMOTIONS


def _labels(rows: list[str]) -> np.ndarray:
    return np.array([[1.0 if e == r else 0.0 for e in EMOTIONS] for r in rows])


def test_label_prior_returns_training_distribution() -> None:
    rows = ["neutral", "neutral", "neutral", "joy"]
    model = LabelPriorEmotionModel().fit(rows, _labels(rows))
    proba = model.predict_proba(["whatever", "second"])

    assert proba.shape == (2, len(EMOTIONS))
    assert np.allclose(proba[0], proba[1])  # text-blind: same for every input
    assert proba[0, EMOTIONS.index("neutral")] == max(proba[0])
    assert abs(proba[0, EMOTIONS.index("neutral")] - 0.75) < 1e-6
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_majority_class_predicts_mode() -> None:
    rows = ["neutral", "neutral", "neutral", "joy", "anger"]
    model = MajorityClassEmotionModel().fit(rows, _labels(rows))
    proba = model.predict_proba(["a", "b"])

    assert (proba.argmax(axis=1) == EMOTIONS.index("neutral")).all()


def test_baselines_handle_empty_batch() -> None:
    rows = ["joy", "sadness"]
    prior = LabelPriorEmotionModel().fit(rows, _labels(rows))
    majority = MajorityClassEmotionModel().fit(rows, _labels(rows))

    assert prior.predict_proba([]).shape == (0, len(EMOTIONS))
    assert majority.predict_proba([]).shape == (0, len(EMOTIONS))
