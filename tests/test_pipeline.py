from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from dialog_emo_models.models import EmotionModel, ModelOutputError
from dialog_emo_models.pipeline import parse_telegram_json, score_parsed_frame
from dialog_emo_models.registry import create_model
from dialog_emo_models.schema import EMOTIONS, load_full_csv, write_full_csv


def test_bundled_fixture_scores_with_dummy_model(tmp_path) -> None:
    parsed = parse_telegram_json(Path("data/result.json"))
    scored = score_parsed_frame(parsed, create_model("dummy"))
    output = write_full_csv(scored, tmp_path / "scored.csv")

    loaded = load_full_csv(output)

    assert len(loaded) == 20
    assert list(loaded.columns) == [
        "turn_index",
        "timestamp",
        "sender",
        "text",
        *EMOTIONS,
    ]
    for emotion in EMOTIONS:
        assert loaded[emotion].iloc[0] == pytest.approx(1 / len(EMOTIONS))


def test_score_parsed_frame_softmaxes_logits() -> None:
    class LogitModel(EmotionModel):
        def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
            row = np.arange(len(EMOTIONS), dtype=float)
            return np.tile(row, (len(texts), 1))

    frame = pd.DataFrame(
        {"turn_index": [0], "timestamp": ["t"], "sender": ["A"], "text": ["x"]}
    )

    scored = score_parsed_frame(frame, LogitModel())

    assert scored.loc[0, list(EMOTIONS)].sum() == pytest.approx(1.0)
    assert scored["neutral"].iloc[0] > scored["joy"].iloc[0]


def test_score_parsed_frame_rejects_wrong_model_shape() -> None:
    class BadModel(EmotionModel):
        def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
            return np.zeros((len(texts), len(EMOTIONS) - 1), dtype=float)

    frame = pd.DataFrame(
        {"turn_index": [0], "timestamp": ["t"], "sender": ["A"], "text": ["x"]}
    )

    with pytest.raises(ModelOutputError, match="shape"):
        score_parsed_frame(frame, BadModel())
