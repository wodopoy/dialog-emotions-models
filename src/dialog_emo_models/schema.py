from __future__ import annotations

from pathlib import Path

import pandas as pd

EMOTIONS = ("joy", "warmth", "sadness", "anger", "anxiety", "neutral")
PARSED_COLUMNS = ("turn_index", "timestamp", "sender", "text")
FULL_COLUMNS = (*PARSED_COLUMNS, *EMOTIONS)
PROBABILITY_SUM_TOLERANCE = 1e-3


class DialogDataError(ValueError):
    """Raised when a dialogue frame does not match the expected contract."""


def load_parsed_csv(path: str | Path) -> pd.DataFrame:
    return validate_parsed_frame(pd.read_csv(path))


def load_full_csv(path: str | Path) -> pd.DataFrame:
    return validate_full_frame(pd.read_csv(path))


def validate_parsed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, PARSED_COLUMNS)
    cleaned = frame.loc[:, PARSED_COLUMNS].copy()
    if cleaned.empty:
        raise DialogDataError("CSV must contain at least one dialogue row")

    cleaned["turn_index"] = pd.to_numeric(cleaned["turn_index"], errors="raise").astype(int)
    for column in ("timestamp", "sender", "text"):
        cleaned[column] = cleaned[column].fillna("").astype(str)

    if cleaned["turn_index"].duplicated().any():
        raise DialogDataError("turn_index values must be unique")

    cleaned = cleaned.sort_values("turn_index", kind="stable").reset_index(drop=True)
    expected = list(range(len(cleaned)))
    if cleaned["turn_index"].tolist() != expected:
        raise DialogDataError("turn_index must be dense values from 0 to N-1")

    return cleaned


def validate_full_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, FULL_COLUMNS)
    cleaned = frame.loc[:, FULL_COLUMNS].copy()
    cleaned["turn_index"] = pd.to_numeric(cleaned["turn_index"], errors="raise").astype(int)
    cleaned = cleaned.sort_values("turn_index", kind="stable").reset_index(drop=True)

    parsed = validate_parsed_frame(cleaned.loc[:, PARSED_COLUMNS])
    emotions = cleaned.loc[:, EMOTIONS].copy()

    for column in EMOTIONS:
        emotions[column] = pd.to_numeric(emotions[column], errors="raise")
        out_of_range = ~emotions[column].between(0, 1)
        if out_of_range.any():
            raise DialogDataError(f"Column {column} must contain values in [0, 1]")

    row_sums = emotions.sum(axis=1)
    invalid_sums = (row_sums - 1).abs() > PROBABILITY_SUM_TOLERANCE
    if invalid_sums.any():
        turn_indexes = parsed.loc[invalid_sums, "turn_index"].head(5).tolist()
        raise DialogDataError(
            "Emotion probabilities must sum to 1 for each row; "
            f"bad turn_index values: {turn_indexes}"
        )

    return pd.concat([parsed, emotions.reset_index(drop=True)], axis=1).loc[:, FULL_COLUMNS]


def write_parsed_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    return _write_csv(validate_parsed_frame(frame), path)


def write_full_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    return _write_csv(validate_full_frame(frame), path)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DialogDataError(f"Missing required columns: {', '.join(missing)}")


def _write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output
