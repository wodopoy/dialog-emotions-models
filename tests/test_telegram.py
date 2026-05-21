from __future__ import annotations

import json

import pytest

from dialog_emo_models.telegram import load_telegram_export


@pytest.fixture
def export_path(tmp_path):
    payload = {
        "name": "Test",
        "type": "personal_chat",
        "id": 1,
        "messages": [
            {"id": 1, "type": "service", "date": "2026-05-20T18:00:00", "text": ""},
            {"id": 2, "type": "message", "date": "2026-05-20T18:01:00", "from": "A", "text": "Hi"},
            {
                "id": 3,
                "type": "message",
                "date": "2026-05-20T18:02:00",
                "from": "A",
                "media_type": "sticker",
                "text": "",
            },
            {
                "id": 4,
                "type": "message",
                "date": "2026-05-20T18:03:00",
                "from": "B",
                "media_type": "voice_message",
                "text": "",
            },
            {
                "id": 5,
                "type": "message",
                "date": "2026-05-20T18:04:00",
                "from": "B",
                "media_type": "video_message",
                "text": "",
            },
            {
                "id": 6,
                "type": "message",
                "date": "2026-05-20T18:05:00",
                "from": "B",
                "text": ["Hello ", {"type": "mention", "text": "@a"}],
            },
            {
                "id": 7,
                "type": "message",
                "date": "2026-05-20T18:06:00",
                "from_id": "user42",
                "text": [{"type": "custom_emoji", "text": ":)"}],
            },
            {"id": 8, "type": "message", "date": "2026-05-20T18:07:00", "from": "A", "text": "   "},
        ],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_drops_service_unwanted_media_and_empty_text(export_path) -> None:
    frame = load_telegram_export(export_path)

    assert frame["text"].tolist() == ["Hi", "Hello @a", ":)"]


def test_turn_index_is_dense(export_path) -> None:
    frame = load_telegram_export(export_path)

    assert frame["turn_index"].tolist() == [0, 1, 2]


def test_keeps_sender_and_timestamp(export_path) -> None:
    frame = load_telegram_export(export_path)

    assert frame["sender"].tolist() == ["A", "B", "user42"]
    assert frame["timestamp"].iloc[0] == "2026-05-20T18:01:00"
