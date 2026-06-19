from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.models.base import EmotionModel, validate_logits
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.text import normalize_text

_WORD_RE = re.compile(r"[а-яёa-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(normalize_text(text)))

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


class LearnedLexiconEmotionModel(EmotionModel):
    """Data-driven lexicon: learns the most discriminative words per emotion.

    For each word it computes a smoothed log-odds of appearing in an emotion's
    documents vs the rest, keeps the top-k positive words per emotion, and scores
    a text by summing the weights of its matched words. Stays interpretable and
    tiny (k x 6 weighted keyword lists) but is fit from data, unlike the
    hand-written `LexiconEmotionModel`.
    """

    def __init__(self, *, top_k: int = 100, min_count: int = 5, alpha: float = 1.0) -> None:
        self.top_k = top_k
        self.min_count = min_count
        self.alpha = alpha
        self.weights: dict[str, dict[str, float]] | None = None

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> LearnedLexiconEmotionModel:
        if labels is None:
            raise ValueError("LearnedLexiconEmotionModel.fit() requires labels")
        primary = validate_logits(labels, expected_rows=len(texts)).argmax(axis=1)
        per_class: dict[str, Counter] = {emotion: Counter() for emotion in EMOTIONS}
        total: Counter = Counter()
        class_docs: Counter = Counter()
        for text, label_index in zip(texts, primary):
            emotion = EMOTIONS[int(label_index)]
            class_docs[emotion] += 1
            for token in _tokens(text):
                per_class[emotion][token] += 1
                total[token] += 1

        n_docs = len(texts)
        self.weights = {}
        for emotion in EMOTIONS:
            in_docs = class_docs[emotion]
            out_docs = n_docs - in_docs
            scored: dict[str, float] = {}
            for token, count_in in per_class[emotion].items():
                if total[token] < self.min_count:
                    continue
                count_out = total[token] - count_in
                p_in = (count_in + self.alpha) / (in_docs + 2 * self.alpha)
                p_out = (count_out + self.alpha) / (out_docs + 2 * self.alpha)
                log_odds = math.log(p_in) - math.log(p_out)
                if log_odds > 0:
                    scored[token] = log_odds
            top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
            self.weights[emotion] = dict(top)
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if self.weights is None:
            raise RuntimeError("LearnedLexiconEmotionModel must be fitted before prediction")
        neutral_index = EMOTIONS.index("neutral")
        scores = np.zeros((len(texts), len(EMOTIONS)), dtype=float)
        for row, text in enumerate(texts):
            tokens = _tokens(text)
            for column, emotion in enumerate(EMOTIONS):
                weights = self.weights[emotion]
                scores[row, column] = sum(weights[t] for t in tokens if t in weights)
            if scores[row].sum() == 0:
                scores[row, neutral_index] = 1.0
        return scores
