from __future__ import annotations

import re
import regex as re2
import emoji
from typing import Optional

# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@[\w_]+", re.UNICODE)
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCT_SPACING_PATTERN = re.compile(r"([!?,.])")
ELONG_PATTERN = re.compile(r"([A-Za-z\u0600-\u06FF])\1{2,}")
ARABIC_DIACRITICS = re2.compile(r"[\u064b-\u0652\u0670\u0640]")
ARABIC_ALEF_VARIANTS = re2.compile(r"[\u0622\u0623\u0625]")


def replace_urls(text: str) -> str:
    return URL_PATTERN.sub("<URL>", text)


def replace_mentions(text: str) -> str:
    return MENTION_PATTERN.sub("<USER>", text)


def normalize_hashtags(text: str) -> str:
    return HASHTAG_PATTERN.sub(r"\1", text)


def replace_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace="<EMOJI>")


def normalize_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text)
    text = ARABIC_ALEF_VARIANTS.sub("ا", text)
    return text


def reduce_elongation(text: str, max_repeat: int = 3) -> str:
    return ELONG_PATTERN.sub(lambda m: m.group(1) * max_repeat, text)


def normalize_punctuation_spacing(text: str) -> str:
    # Ensure space before punctuation is trimmed and after punctuation is spaced
    text = PUNCT_SPACING_PATTERN.sub(r" \1 ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    text = replace_urls(text)
    text = replace_mentions(text)
    text = normalize_hashtags(text)
    text = replace_emojis(text)
    text = normalize_arabic(text)
    text = reduce_elongation(text)
    text = normalize_punctuation_spacing(text)
    return text
