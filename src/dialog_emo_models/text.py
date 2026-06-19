from __future__ import annotations

import re

MAX_CHARS = 280

EMOJI_REPLACEMENTS = {
    "🙂": " радость ",
    "😊": " радость ",
    "😂": " радость ",
    "😢": " грусть ",
    "😭": " грусть ",
    "😡": " злость ",
    "😠": " злость ",
    "😱": " страх ",
    "😨": " страх ",
    "😮": " удивление ",
    "😲": " удивление ",
}


def normalize_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Lowercase, strip URLs/mentions, and map a few emojis to Russian cues.

    Carried over from the undergraduate thesis preprocessing so the lexicon and
    fastText models see the same surface form they were tuned on.
    """
    text = str(text or "")
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)
    text = text.lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " ссылка ", text)
    text = re.sub(r"@\w+", " пользователь ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
