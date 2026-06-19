from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from dialog_emo_models.models.base import EmotionModel, ModelT, validate_logits
from dialog_emo_models.models.sklearn_baselines import _build_vectorizer
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.text import normalize_text

_LOG_FLOOR = 1e-9


def _proba_to_logits(
    proba: NDArray[np.float64],
    classes: NDArray[np.int64],
    n_rows: int,
) -> NDArray[np.float64]:
    """Place a classifier's class-probabilities into EMOTIONS-ordered logits.

    The training labels are argmax over EMOTIONS, so `classes` are EMOTIONS
    indices; absent classes get a very low logit.
    """
    logits = np.full((n_rows, len(EMOTIONS)), -30.0, dtype=float)
    for index, class_id in enumerate(classes):
        logits[:, int(class_id)] = np.log(np.clip(proba[:, index], _LOG_FLOOR, 1.0))
    return logits


def _clean(texts: Sequence[str]) -> list[str]:
    return [normalize_text(text) for text in texts]


class TfidfNaiveBayesEmotionModel(EmotionModel):
    """TF-IDF + (Complement/Multinomial) Naive Bayes — the classic tiny text baseline."""

    def __init__(
        self,
        *,
        kind: str = "complement",
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
        max_features: int | None = 50_000,
        sublinear_tf: bool = True,
        alpha: float = 0.3,
    ) -> None:
        self.kind = kind
        self.vectorizer = _build_vectorizer(analyzer, ngram_range, min_df, max_features, sublinear_tf)
        self.estimator = ComplementNB(alpha=alpha) if kind == "complement" else MultinomialNB(alpha=alpha)
        self._fitted = False

    def fit(self, texts, labels=None) -> "TfidfNaiveBayesEmotionModel":
        if labels is None:
            raise ValueError("TfidfNaiveBayesEmotionModel.fit() requires labels")
        target = validate_logits(labels, expected_rows=len(texts)).argmax(axis=1)
        self.estimator.fit(self.vectorizer.fit_transform(_clean(texts)), target)
        self._fitted = True
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("TfidfNaiveBayesEmotionModel must be fitted before prediction")
        proba = self.estimator.predict_proba(self.vectorizer.transform(_clean(texts)))
        return _proba_to_logits(proba, self.estimator.classes_, len(texts))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        return joblib.load(path)


class TfidfTreeEmotionModel(EmotionModel):
    """TF-IDF -> TruncatedSVD -> tree ensemble (random forest or HistGradientBoosting).

    Tree ensembles need dense, low-dim features, so the sparse TF-IDF is reduced
    by SVD first. Included mainly to show that trees are dominated by linear
    models on this sparse-text task (and are far larger).
    """

    def __init__(
        self,
        *,
        estimator: str = "hgb",
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
        max_features: int | None = 50_000,
        svd_components: int = 300,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        self.estimator_kind = estimator
        self.vectorizer = _build_vectorizer(analyzer, ngram_range, min_df, max_features, True)
        self.svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
        if estimator == "rf":
            self.estimator = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=-1, random_state=random_state
            )
        else:
            self.estimator = HistGradientBoostingClassifier(random_state=random_state)
        self._fitted = False

    def fit(self, texts, labels=None) -> "TfidfTreeEmotionModel":
        if labels is None:
            raise ValueError("TfidfTreeEmotionModel.fit() requires labels")
        target = validate_logits(labels, expected_rows=len(texts)).argmax(axis=1)
        features = self.svd.fit_transform(self.vectorizer.fit_transform(_clean(texts)))
        self.estimator.fit(features, target)
        self._fitted = True
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("TfidfTreeEmotionModel must be fitted before prediction")
        features = self.svd.transform(self.vectorizer.transform(_clean(texts)))
        proba = self.estimator.predict_proba(features)
        return _proba_to_logits(proba, self.estimator.classes_, len(texts))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        return joblib.load(path)
