from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.models.base import EmotionModel
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.text import normalize_text

# Stem lists carried over from the thesis dictionary baseline, remapped from the
# original Russian label names onto the project's six emotions.
LEXICON: dict[str, tuple[str, ...]] = {
    "joy": (
        "рад", "рада", "радуюсь", "счаст", "супер", "ура",
        "спасибо", "отличн", "прекрасн", "лучше",
    ),
    "warmth": (
        "спасибо", "люблю", "обнима", "поддерж", "береги", "забот", "дорог", "ценю",
    ),
    "sadness": (
        "груст", "печаль", "тоск", "скуч", "пустот", "обид", "одинок", "тяжело",
    ),
    "anger": (
        "злю", "злит", "бесит", "раздраж", "ярост", "достало", "ненавиж", "черт",
    ),
    "anxiety": (
        "боюсь", "страш", "тревож", "пережива", "волную", "паник", "нервнича", "жутко",
    ),
    "neutral": (
        "встреч", "созвон", "файл", "задач", "магазин", "календар", "письмо", "офис",
    ),
}


class LexiconEmotionModel(EmotionModel):
    """Keyword-count dictionary baseline carried over from the thesis.

    Each emotion scores how many of its stems appear in the normalized text.
    Texts with no keyword hit fall back to a single neutral count. Scores are
    returned as logits; the surrounding pipeline softmaxes them into the
    probability distribution, reproducing the original baseline's behaviour.
    """

    def __init__(self, lexicon: dict[str, Sequence[str]] | None = None) -> None:
        source = lexicon if lexicon is not None else LEXICON
        self.lexicon = {emotion: tuple(source.get(emotion, ())) for emotion in EMOTIONS}

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        neutral_index = EMOTIONS.index("neutral")
        scores = np.zeros((len(texts), len(EMOTIONS)), dtype=float)
        for row, text in enumerate(texts):
            normalized = normalize_text(text)
            for column, emotion in enumerate(EMOTIONS):
                scores[row, column] = sum(
                    1.0 for stem in self.lexicon[emotion] if stem in normalized
                )
            if scores[row].sum() == 0:
                scores[row, neutral_index] = 1.0
        return scores
