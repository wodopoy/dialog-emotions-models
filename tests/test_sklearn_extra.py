from __future__ import annotations

import numpy as np

from dialog_emo_models.models import TfidfNaiveBayesEmotionModel, TfidfTreeEmotionModel
from dialog_emo_models.schema import EMOTIONS


def _data():
    rows = [
        ("ура супер радость классно", "joy"),
        ("люблю спасибо забота тепло", "warmth"),
        ("грустно тоска печаль одиноко", "sadness"),
        ("бесит злюсь ярость ненавижу", "anger"),
        ("страшно тревожно боюсь паника", "anxiety"),
        ("файл задача созвон встреча", "neutral"),
    ] * 6
    texts = [text for text, _ in rows]
    labels = np.array([[1.0 if e == label else 0.0 for e in EMOTIONS] for _, label in rows])
    return texts, labels


def test_naive_bayes_trains_and_scores() -> None:
    texts, labels = _data()
    for kind in ("complement", "multinomial"):
        model = TfidfNaiveBayesEmotionModel(kind=kind, min_df=1).fit(texts, labels)
        proba = model.predict_proba(texts)
        assert proba.shape == (len(texts), len(EMOTIONS))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)


def test_tree_ensemble_trains_and_scores() -> None:
    texts, labels = _data()
    for estimator in ("hgb", "rf"):
        model = TfidfTreeEmotionModel(
            estimator=estimator, min_df=1, svd_components=5, n_estimators=20
        ).fit(texts, labels)
        proba = model.predict_proba(texts)
        assert proba.shape == (len(texts), len(EMOTIONS))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
