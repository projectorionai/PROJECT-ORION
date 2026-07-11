"""
Small shared helpers used across the O.R.I.O.N. package.

Nothing here may import from other orion_core modules (except constants) —
utils sits at the bottom of the dependency graph.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientTimeout
from PIL import Image

# PIL renamed its resampling enum; resolve once at import time.
PIL_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]


def clamp_channel(value: Any) -> int:
    """Clamp any numeric-ish value into the 0-255 colour channel range."""
    try:
        return max(0, min(255, int(value)))
    except Exception:
        return 0


def now_stamp() -> str:
    """Local wall-clock timestamp for console lines."""
    return datetime.now().strftime("%H:%M:%S")


def utc_stamp() -> str:
    """UTC ISO-8601 timestamp for durable records."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_transcript(text: str) -> str:
    """Strip model control tokens and non-printable bytes from transcripts."""
    text = re.sub(r"<ctrl\d+>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


# Invisible code points that appear inside real window titles — Edge embeds a
# zero-width space in "Microsoft Edge" (U+200B between the words), which
# silently defeats substring matching against what the user (or model) types.
_INVISIBLE_CHARS = dict.fromkeys((
    0x200B,  # zero width space (Edge window titles)
    0x200C,  # zero width non-joiner
    0x200D,  # zero width joiner
    0x2060,  # word joiner
    0xFEFF,  # zero width no-break space / BOM
    0x00AD,  # soft hyphen
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
))


def fold_title(text: Any) -> str:
    """Normalise a window/process title for matching: drop invisible Unicode,
    collapse all whitespace runs to single spaces, lowercase."""
    text = str(text or "").translate(_INVISIBLE_CHARS)
    return re.sub(r"\s+", " ", text).strip().lower()


def first_line(value: Any, limit: int = 160) -> str:
    """First line of an exception/string, truncated — safe for log output."""
    try:
        return str(value).splitlines()[0][:limit]
    except Exception:
        return ""


def aiohttp_client_timeout() -> ClientTimeout:
    """Default short timeout for opportunistic network calls."""
    return ClientTimeout(total=8.0, connect=3.0)


def weather_code_label(code: int) -> str:
    """Translate an Open-Meteo weather code into a spoken-friendly label."""
    labels = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
        96: "thunderstorm with hail",
        99: "severe thunderstorm with hail",
    }
    return labels.get(int(code), f"weather code {code}")
