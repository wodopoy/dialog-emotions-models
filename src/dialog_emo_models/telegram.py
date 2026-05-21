from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dialog_emo_models.schema import validate_parsed_frame

DROPPED_MEDIA_TYPES = frozenset({"sticker", "voice_message", "video_message"})


def load_telegram_export(path: str | Path) -> pd.DataFrame:
    """Parse a Telegram Desktop JSON export into the parsed CSV contract."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: list[dict[str, Any]] = []
    for message in payload.get("messages", []):
        if message.get("type") != "message":
            continue
        if message.get("media_type") in DROPPED_MEDIA_TYPES:
            continue

        text = _flatten_text(message.get("text", ""))
        if not text.strip():
            continue

        rows.append(
            {
                "timestamp": str(message.get("date", "")),
                "sender": str(message.get("from") or message.get("from_id") or ""),
                "text": text,
            }
        )

    frame = pd.DataFrame(rows, columns=["timestamp", "sender", "text"])
    frame.insert(0, "turn_index", range(len(frame)))
    return validate_parsed_frame(frame)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for segment in value:
            if isinstance(segment, str):
                parts.append(segment)
            elif isinstance(segment, dict):
                parts.append(str(segment.get("text", "")))
        return "".join(parts)
    return ""
