from __future__ import annotations

import numpy as np

from dialog_emo_models.models import RuBertTiny2EmotionModel
from dialog_emo_models.models.transformer import _reorder_logits_to_emotions
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.training import (
    available_trainable_model_names,
    create_trainable_model,
)


def test_rubert_is_registered_as_trainable() -> None:
    assert "rubert-tiny2-finetune" in available_trainable_model_names()
    assert isinstance(
        create_trainable_model("rubert-tiny2-finetune"), RuBertTiny2EmotionModel
    )


def test_reorder_logits_permutes_into_emotion_order() -> None:
    # Model emits columns in reversed emotion order.
    reversed_emotions = list(reversed(EMOTIONS))
    id2label = dict(enumerate(reversed_emotions))
    logits = np.arange(len(EMOTIONS), dtype=float).reshape(1, len(EMOTIONS))

    reordered = _reorder_logits_to_emotions(logits, id2label)

    expected = np.array([[float(len(EMOTIONS) - 1 - i) for i in range(len(EMOTIONS))]])
    assert np.array_equal(reordered, expected)
    # The column now standing for EMOTIONS[0] is the one reversed-labelled as it.
    assert reordered[0, 0] == logits[0, reversed_emotions.index(EMOTIONS[0])]


def test_reorder_logits_passes_through_unknown_labels() -> None:
    id2label = {index: f"LABEL_{index}" for index in range(len(EMOTIONS))}
    logits = np.random.default_rng(0).normal(size=(3, len(EMOTIONS)))

    assert np.array_equal(_reorder_logits_to_emotions(logits, id2label), logits)
