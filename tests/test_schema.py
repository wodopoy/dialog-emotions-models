from __future__ import annotations

import pandas as pd
import pytest

from dialog_emo_models.schema import (
    EMOTIONS,
    DialogDataError,
    validate_full_frame,
    validate_parsed_frame,
)


def _parsed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn_index": [0, 1],
            "timestamp": ["2026-05-20T18:01:00", "2026-05-20T18:02:00"],
            "sender": ["A", "B"],
            "text": ["hello", "ok"],
        }
    )


def _full_frame() -> pd.DataFrame:
    frame = _parsed_frame()
    for emotion in EMOTIONS:
        frame[emotion] = 0.0
    frame["neutral"] = 1.0
    return frame


def test_validate_parsed_frame_requires_columns() -> None:
    with pytest.raises(DialogDataError, match="Missing required columns"):
        validate_parsed_frame(pd.DataFrame({"turn_index": [0]}))


def test_validate_parsed_frame_requires_dense_turn_index() -> None:
    frame = _parsed_frame()
    frame["turn_index"] = [0, 2]

    with pytest.raises(DialogDataError, match="dense"):
        validate_parsed_frame(frame)


def test_validate_full_frame_requires_emotion_columns() -> None:
    with pytest.raises(DialogDataError, match="Missing required columns"):
        validate_full_frame(_parsed_frame())


def test_validate_full_frame_rejects_out_of_range_probabilities() -> None:
    frame = _full_frame()
    frame.loc[0, "joy"] = 1.2

    with pytest.raises(DialogDataError, match="joy"):
        validate_full_frame(frame)


def test_validate_full_frame_rejects_probabilities_that_do_not_sum_to_one() -> None:
    frame = _full_frame()
    frame.loc[0, "neutral"] = 0.5

    with pytest.raises(DialogDataError, match="sum to 1"):
        validate_full_frame(frame)


def test_validate_full_frame_sorts_metadata_and_probabilities_together() -> None:
    frame = _full_frame().iloc[[1, 0]].reset_index(drop=True)

    validated = validate_full_frame(frame)

    assert validated["turn_index"].tolist() == [0, 1]
    assert validated["text"].tolist() == ["hello", "ok"]
