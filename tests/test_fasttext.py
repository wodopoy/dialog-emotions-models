from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dialog_emo_models.models import FastTextSupervisedEmotionModel
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.training import (
    available_trainable_model_names,
    create_trainable_model,
)

_HAS_FASTTEXT = importlib.util.find_spec("fasttext") is not None


def _one_hot(label: str) -> list[float]:
    return [1.0 if emotion == label else 0.0 for emotion in EMOTIONS]


def test_fasttext_is_registered_as_trainable() -> None:
    assert "fasttext-supervised" in available_trainable_model_names()
    assert isinstance(
        create_trainable_model("fasttext-supervised"), FastTextSupervisedEmotionModel
    )


@pytest.mark.skipif(_HAS_FASTTEXT, reason="fasttext installed; missing-dep path unreachable")
def test_fasttext_fit_without_dependency_raises() -> None:
    model = FastTextSupervisedEmotionModel()
    with pytest.raises(RuntimeError, match="fasttext"):
        model.fit(["привет"], np.array([_one_hot("neutral")]))


@pytest.mark.skipif(not _HAS_FASTTEXT, reason="requires optional fasttext dependency")
def test_fasttext_trains_saves_loads_and_scores(tmp_path) -> None:
    rows = [
        ("ура классно радость супер отлично", "joy"),
        ("люблю спасибо тепло обнимаю забота", "warmth"),
        ("грустно больно печаль тоска одиноко", "sadness"),
        ("злюсь бесит ярость ненавижу раздражает", "anger"),
        ("страшно тревожно боюсь паника волнуюсь", "anxiety"),
        ("встреча созвон файл задача календарь", "neutral"),
    ]
    texts = [text for text, _ in rows] * 8
    labels = np.array([_one_hot(label) for _, label in rows] * 8)

    model = FastTextSupervisedEmotionModel().fit(texts, labels)
    proba = model.predict_proba(texts)
    assert proba.shape == (len(texts), len(EMOTIONS))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)

    path = tmp_path / "ft.joblib"
    model.save(path)
    reloaded = FastTextSupervisedEmotionModel.load(path)
    assert reloaded.predict_proba(["ура супер"]).shape == (1, len(EMOTIONS))


@pytest.mark.skipif(not _HAS_FASTTEXT, reason="requires optional fasttext dependency")
def test_fasttext_temperature_softens_distribution() -> None:
    rows = [
        ("ура классно радость супер отлично", "joy"),
        ("люблю спасибо тепло обнимаю забота", "warmth"),
        ("грустно больно печаль тоска одиноко", "sadness"),
        ("злюсь бесит ярость ненавижу раздражает", "anger"),
        ("страшно тревожно боюсь паника волнуюсь", "anxiety"),
        ("встреча созвон файл задача календарь", "neutral"),
    ]
    texts = [text for text, _ in rows] * 8
    labels = np.array([_one_hot(label) for _, label in rows] * 8)

    model = FastTextSupervisedEmotionModel().fit(texts, labels)
    sharp = float(model.predict_proba(["ура классно радость супер"]).max())
    model.temperature = 4.0
    soft = float(model.predict_proba(["ура классно радость супер"]).max())

    assert soft < sharp  # higher temperature flattens the peaky distribution
