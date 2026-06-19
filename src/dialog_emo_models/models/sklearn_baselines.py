from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import FeatureUnion

from dialog_emo_models.models.base import EmotionModel, ModelT, validate_logits
from dialog_emo_models.schema import EMOTIONS

# Combined analyzer carried over from the thesis tfidf_lr model: it fuses word
# unigram/bigram features with character n-grams in a single FeatureUnion.
WORD_CHAR_ANALYZER = "word+char"


def _build_vectorizer(
    analyzer: str,
    ngram_range: tuple[int, int],
    min_df: int,
    max_features: int | None,
):
    if analyzer == WORD_CHAR_ANALYZER:
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word", ngram_range=(1, 2), min_df=1, lowercase=True
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True
                    ),
                ),
            ]
        )
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
    )


class TfidfRidgeEmotionModel(EmotionModel):
    """Char n-gram TF-IDF + Ridge regression for soft emotion labels."""

    def __init__(
        self,
        *,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 1,
        max_features: int | None = 50_000,
        alpha: float = 1.0,
    ) -> None:
        self.vectorizer = _build_vectorizer(analyzer, ngram_range, min_df, max_features)
        self.estimator = Ridge(alpha=alpha)
        self._is_fitted = False

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> TfidfRidgeEmotionModel:
        if labels is None:
            raise ValueError("TfidfRidgeEmotionModel.fit() requires soft labels")
        target = _probabilities_to_logits(validate_logits(labels, expected_rows=len(texts)))
        features = self.vectorizer.fit_transform(_clean_texts(texts))
        self.estimator.fit(features, target)
        self._is_fitted = True
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self._require_fitted()
        features = self.vectorizer.transform(_clean_texts(texts))
        return validate_logits(self.estimator.predict(features), expected_rows=len(texts))

    def save(self, path: str | Path) -> None:
        _save_joblib(self, path)

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        return _load_joblib(cls, path)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TfidfRidgeEmotionModel must be fitted before prediction")


class TfidfLogRegEmotionModel(EmotionModel):
    """Char n-gram TF-IDF + LogisticRegression on the strongest emotion label."""

    def __init__(
        self,
        *,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 1,
        max_features: int | None = 50_000,
        max_iter: int = 1_000,
        class_weight: str | None = "balanced",
    ) -> None:
        self.vectorizer = _build_vectorizer(analyzer, ngram_range, min_df, max_features)
        self.estimator = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
        )
        self._is_fitted = False
        self._constant_class: int | None = None

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> TfidfLogRegEmotionModel:
        if labels is None:
            raise ValueError("TfidfLogRegEmotionModel.fit() requires labels")
        target = validate_logits(labels, expected_rows=len(texts)).argmax(axis=1)
        features = self.vectorizer.fit_transform(_clean_texts(texts))
        classes = np.unique(target)
        if len(classes) == 1:
            self._constant_class = int(classes[0])
        else:
            self.estimator.fit(features, target)
            self._constant_class = None
        self._is_fitted = True
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self._require_fitted()
        features = self.vectorizer.transform(_clean_texts(texts))
        logits = np.full((len(texts), len(EMOTIONS)), -30.0, dtype=float)

        if self._constant_class is not None:
            logits[:, self._constant_class] = 30.0
            return logits

        decision = np.asarray(self.estimator.decision_function(features), dtype=float)
        if decision.ndim == 1:
            classes = self.estimator.classes_
            logits[:, int(classes[0])] = -decision
            logits[:, int(classes[1])] = decision
        else:
            for index, class_id in enumerate(self.estimator.classes_):
                logits[:, int(class_id)] = decision[:, index]
        return logits

    def save(self, path: str | Path) -> None:
        _save_joblib(self, path)

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        return _load_joblib(cls, path)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TfidfLogRegEmotionModel must be fitted before prediction")


def _clean_texts(texts: Sequence[str]) -> list[str]:
    return ["" if text is None else str(text) for text in texts]


def _probabilities_to_logits(labels: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(np.clip(labels, 1e-6, 1.0))


def _save_joblib(model: EmotionModel, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def _load_joblib(cls: type[ModelT], path: str | Path) -> ModelT:
    model = joblib.load(path)
    if not isinstance(model, cls):
        raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
    return model
