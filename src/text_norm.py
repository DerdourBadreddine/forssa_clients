from __future__ import annotations

import re
import unicodedata
from typing import Optional

import emoji

# NOTE: Keep normalization *light* (do not over-clean).

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+", re.UNICODE)
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")

# Mild elongation reduction: keep up to 4 repeats (coooooool -> cooool)
ELONG_RE = re.compile(r"([A-Za-z\u0600-\u06FF])\1{3,}")

# Arabic ranges / chars
ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")
TATWEEL = "\u0640"


def _to_str(x: Optional[str]) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def replace_urls(text: str) -> str:
    return URL_RE.sub("<URL>", text)


def replace_mentions(text: str) -> str:
    return MENTION_RE.sub("<USER>", text)


def keep_hashtag_word(text: str) -> str:
    # Keep hashtag signal as token content (strip leading #)
    return HASHTAG_RE.sub(r"\1", text)


def replace_emojis(text: str) -> str:
    # Keep signal with a single token
    return emoji.replace_emoji(text, replace=" <EMOJI> ")


def reduce_elongation(text: str, max_repeat: int = 4) -> str:
    return ELONG_RE.sub(lambda m: m.group(1) * max_repeat, text)


def normalize_arabic_light(text: str) -> str:
    # Remove diacritics + tatweel
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace(TATWEEL, "")

    # Unify alef variants: أ/إ/آ -> ا
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    # ى -> ي
    text = text.replace("ى", "ي")
    # ؤ -> و
    text = text.replace("ؤ", "و")
    # ئ -> ي
    text = text.replace("ئ", "ي")
    return text


def normalize_whitespace(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def normalize(text: Optional[str]) -> str:
    """Competition-safe normalization.

    - Fill missing with ""
    - Replace URLs -> <URL>
    - Replace @mentions -> <USER>
    - Convert emojis -> <EMOJI>
    - Keep hashtag word (strip #)
    - Mild elongation reduction
    - Arabic light normalization
    - Normalize whitespace
    """
    t = _to_str(text)
    t = unicodedata.normalize("NFKC", t)
    t = t.strip()
    if not t:
        return ""
    t = replace_urls(t)
    t = replace_mentions(t)
    t = keep_hashtag_word(t)
    t = replace_emojis(t)
    t = normalize_arabic_light(t)
    t = reduce_elongation(t)
    t = normalize_whitespace(t)
    return t


# Backwards-compat alias (older modules may import normalize_text)
def normalize_text(text: Optional[str]) -> str:
    return normalize(text)
