from __future__ import annotations

from pathlib import Path

import pandas as pd

from dialog_emo_models.models import EmotionModel, validate_logits
from dialog_emo_models.schema import (
    EMOTIONS,
    validate_full_frame,
    validate_parsed_frame,
    write_full_csv,
    write_parsed_csv,
)
from dialog_emo_models.telegram import load_telegram_export


def parse_telegram_json(path: str | Path) -> pd.DataFrame:
    return load_telegram_export(path)


def score_parsed_frame(
    frame: pd.DataFrame,
    model: EmotionModel,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    parsed = validate_parsed_frame(frame)
    texts = parsed["text"].tolist()
    probabilities = model.predict_proba(texts, show_progress=show_progress)
    validate_logits(probabilities, expected_rows=len(parsed))

    scored = parsed.copy()
    for index, emotion in enumerate(EMOTIONS):
        scored[emotion] = probabilities[:, index]
    return validate_full_frame(scored)


__all__ = [
    "parse_telegram_json",
    "score_parsed_frame",
    "write_full_csv",
    "write_parsed_csv",
]
