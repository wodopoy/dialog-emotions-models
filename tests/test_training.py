from __future__ import annotations

import pandas as pd
import pytest

from dialog_emo_models.cli import main
from dialog_emo_models.models import TfidfLogRegEmotionModel, TfidfRidgeEmotionModel
from dialog_emo_models.pipeline import score_parsed_frame
from dialog_emo_models.schema import (
    EMOTIONS,
    load_full_csv,
    load_parsed_csv,
    write_full_csv,
    write_parsed_csv,
)
from dialog_emo_models.training import (
    available_trainable_model_names,
    create_trainable_model,
    labels_from_full_frame,
    train_from_full_frame,
)


def _training_frame() -> pd.DataFrame:
    rows = [
        ("ура классно радость", "joy"),
        ("люблю спасибо тепло", "warmth"),
        ("грустно больно печаль", "sadness"),
        ("злюсь бесит ярость", "anger"),
        ("страшно тревожно боюсь", "anxiety"),
        ("ок понятно принято", "neutral"),
    ]
    frame = pd.DataFrame(
        {
            "turn_index": range(len(rows)),
            "timestamp": ["2026-05-20T18:00:00"] * len(rows),
            "sender": ["A"] * len(rows),
            "text": [text for text, _ in rows],
        }
    )
    for emotion in EMOTIONS:
        frame[emotion] = [1.0 if label == emotion else 0.0 for _, label in rows]
    return frame


def test_trainable_registry_contains_linear_baselines() -> None:
    names = available_trainable_model_names()

    assert names == [
        "logreg-tfidf",
        "logreg-word-char-tfidf",
        "ridge-tfidf",
        "ridge-word-char-tfidf",
    ]
    assert isinstance(create_trainable_model("ridge-tfidf"), TfidfRidgeEmotionModel)


def test_word_char_tfidf_trains_saves_loads_and_scores(tmp_path) -> None:
    frame = _training_frame()
    model = train_from_full_frame(frame, create_trainable_model("logreg-word-char-tfidf"))
    model_path = tmp_path / "word-char.joblib"
    model.save(model_path)

    loaded = TfidfLogRegEmotionModel.load(model_path)
    parsed = frame.loc[:, ["turn_index", "timestamp", "sender", "text"]]
    scored = score_parsed_frame(parsed, loaded)

    assert scored["anger"].iloc[3] > scored["joy"].iloc[3]


def test_labels_from_full_frame_returns_emotion_matrix() -> None:
    labels = labels_from_full_frame(_training_frame())

    assert labels.shape == (6, len(EMOTIONS))
    assert labels[0, EMOTIONS.index("joy")] == pytest.approx(1.0)


def test_ridge_tfidf_trains_saves_loads_and_scores(tmp_path) -> None:
    frame = _training_frame()
    model = train_from_full_frame(frame, TfidfRidgeEmotionModel())
    model_path = tmp_path / "ridge.joblib"
    model.save(model_path)

    loaded = TfidfRidgeEmotionModel.load(model_path)
    parsed = frame.loc[:, ["turn_index", "timestamp", "sender", "text"]]
    scored = score_parsed_frame(parsed, loaded)

    assert scored["joy"].iloc[0] > scored["anger"].iloc[0]
    assert scored["neutral"].iloc[-1] > scored["sadness"].iloc[-1]


def test_logreg_tfidf_trains_and_scores(tmp_path) -> None:
    frame = _training_frame()
    model = train_from_full_frame(frame, TfidfLogRegEmotionModel())
    model_path = tmp_path / "logreg.joblib"
    model.save(model_path)

    loaded = TfidfLogRegEmotionModel.load(model_path)
    parsed = frame.loc[:, ["turn_index", "timestamp", "sender", "text"]]
    scored = score_parsed_frame(parsed, loaded)

    assert scored["anger"].iloc[3] > scored["joy"].iloc[3]


def test_train_cli_writes_model_and_score_can_use_model_path(tmp_path) -> None:
    full_path = write_full_csv(_training_frame(), tmp_path / "train.csv")
    parsed_path = write_parsed_csv(
        _training_frame().loc[:, ["turn_index", "timestamp", "sender", "text"]],
        tmp_path / "parsed.csv",
    )
    model_path = tmp_path / "model.joblib"
    scored_path = tmp_path / "scored.csv"

    main(["train", "--input", str(full_path), "--output", str(model_path), "--model", "ridge-tfidf"])
    main(
        [
            "score",
            "--input",
            str(parsed_path),
            "--output",
            str(scored_path),
            "--model-path",
            str(model_path),
        ]
    )

    scored = load_full_csv(scored_path)

    assert model_path.exists()
    assert len(scored) == len(load_parsed_csv(parsed_path))
