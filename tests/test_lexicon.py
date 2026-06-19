from __future__ import annotations

import numpy as np

from dialog_emo_models.models import LexiconEmotionModel
from dialog_emo_models.registry import create_model
from dialog_emo_models.schema import EMOTIONS


def _top_emotion(model: LexiconEmotionModel, text: str) -> str:
    proba = model.predict_proba([text])
    return EMOTIONS[int(proba.argmax(axis=1)[0])]


def test_lexicon_is_registered_for_inference() -> None:
    assert isinstance(create_model("lexicon"), LexiconEmotionModel)


def test_lexicon_picks_keyword_emotion() -> None:
    model = LexiconEmotionModel()

    assert _top_emotion(model, "мне очень страшно и тревожно") == "anxiety"
    assert _top_emotion(model, "я так рада, это супер") == "joy"
    assert _top_emotion(model, "бесит, злюсь на всех") == "anger"


def test_lexicon_falls_back_to_neutral_without_keywords() -> None:
    model = LexiconEmotionModel()

    assert _top_emotion(model, "abcdef qwerty") == "neutral"


def test_lexicon_returns_valid_distribution() -> None:
    model = LexiconEmotionModel()
    proba = model.predict_proba(["спасибо за поддержку", "грустно и тоскливо"])

    assert proba.shape == (2, len(EMOTIONS))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_lexicon_handles_empty_batch() -> None:
    assert LexiconEmotionModel().predict_proba([]).shape == (0, len(EMOTIONS))
