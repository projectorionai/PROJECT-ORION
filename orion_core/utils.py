"""
Small shared helpers used across the O.R.I.O.N. package.

Nothing here may import from other orion_core modules (except constants) —
utils sits at the bottom of the dependency graph.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Optional

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")
from PIL import Image

# PIL renamed its resampling enum; resolve once at import time.
PIL_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]


def open_camera_capture(index: int = 0, hi_res: bool = False,
                        size: "tuple[int, int] | None" = None,
                        fps: float = 0.0, prefer: str = "") -> tuple[Any, str]:
    """Open a webcam, preferring DirectShow.

    MSMF's grab path throws -1072873821 on many Windows setups, and the
    plain default backend (CAP_ANY) additionally auto-probes a depth-sensor
    ('obsensor') backend that always fails harmlessly against a regular
    webcam — that failure prints 'Camera index out of range' to stderr on
    every open even though the actual capture goes on to work fine.
    Explicitly ordering the backends tried (DSHOW, then MSMF, then whatever
    else is available) skips that spurious probe entirely. Imports cv2
    lazily so importing this module never pulls it in for callers that don't
    need a camera. Returns (capture_or_None, backend_name_used).
    """
    import cv2
    candidates: list[tuple[str, Any]] = []
    # MEASURED on this machine with a Logitech C920, three seconds of real
    # grabs per combination:
    #
    #     MSMF  1920x1080   30.7 fps
    #     DSHOW 1920x1080    5.1 fps   (stays on uncompressed YUY2)
    #     DSHOW 1280x720    10.0 fps
    #
    # DirectShow reports 60 fps and delivers five, because it ignores a
    # request to switch to MJPG and keeps sending raw YUY2, which saturates
    # the USB bus at 1080p. Media Foundation negotiates a compressed format
    # and delivers six times the frames.
    #
    # DSHOW still leads for ordinary use: MSMF's grab path throws
    # -1072873821 on many Windows setups, which is why it was second in the
    # first place. So the order flips only when the caller actually wants the
    # resolution — and the loser is still tried, so a machine where MSMF is
    # broken keeps working.
    order = (("CAP_MSMF", "CAP_DSHOW", "CAP_ANY") if hi_res
             else ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"))
    # The backend whose device list the index came from goes first. Media
    # Foundation and DirectShow number cameras differently, so index 1 may
    # be the C920 to one and a virtual camera to the other.
    if prefer in order:
        order = (prefer,) + tuple(name for name in order if name != prefer)
    for name in order:
        flag = getattr(cv2, name, None)
        if flag is not None:
            candidates.append((name, flag))
    if not candidates:
        candidates = [("default", None)]
    for name, flag in candidates:
        try:
            cap = cv2.VideoCapture(index) if flag is None else cv2.VideoCapture(index, flag)
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            # Format and size BEFORE the validation read. Media Foundation
            # renegotiates the stream when the resolution changes, and doing
            # that after a frame has already been pulled makes it fail to
            # re-select the stream — "Failed to select stream 0", followed by
            # a malformed Mat. One negotiation, then one proof.
            if size or fps:
                try:
                    if size:
                        # MJPG only on DirectShow. MEASURED on MSMF: asking
                        # for it costs another 8 seconds on top of an open
                        # that is already slow, and buys nothing — Media
                        # Foundation negotiates a compressed format by itself
                        # and delivers 30 fps at 1080p either way. On
                        # DirectShow the same request is ignored, but it is
                        # free there, and a backend that honoured it would be
                        # better for it.
                        if name == "CAP_DSHOW":
                            cap.set(cv2.CAP_PROP_FOURCC,
                                    cv2.VideoWriter_fourcc(*"MJPG"))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
                    if fps:
                        cap.set(cv2.CAP_PROP_FPS, float(fps))
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            # isOpened() is not the same as working: MSMF opens and then
            # throws on the first grab on the setups this ordering exists to
            # protect. Read one frame before committing, so a broken backend
            # falls through to the next instead of being handed back.
            try:
                if cap.read()[0]:
                    return cap, name
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass
            continue
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return None, ""


#: Where tesseract was found, cached. None = not looked for yet;
#: "" = looked for and genuinely absent.
_TESSERACT_CMD: Optional[str] = None


def locate_tesseract() -> str:
    """The tesseract binary's path, wiring pytesseract to it. "" if absent.

    pytesseract finds tesseract by looking on PATH and nowhere else, and the
    Windows installer does not put it there - it installs to Program Files and
    leaves PATH alone. So a machine with tesseract properly installed still
    reported "tesseract is not installed or it's not in your PATH", and ORION
    fell back to a slower recogniser for no reason.

    A frozen ORION.exe would not see a PATH change anyway without a restart,
    and asking the user to edit their environment to make an installed program
    findable is not a fix. Looking where the installer actually puts it is.

    **This never raises.** Its whole contract is "the path, or "" if there
    isn't one", and it sits on the path that locates a control on screen - the
    one ORION uses to click something the accessibility tree cannot see. An
    exception here does not degrade to the next OCR engine, it aborts the
    lookup, so a surprise from pytesseract's internals would cost ORION the
    ability to click things rather than costing him some speed. The first
    version reached ``pytesseract.pytesseract`` outside the guard and did
    exactly that against any build not exposing that submodule.

    The answer is cached: this is called once per screen capture, and a miss
    costs a PATH search plus three stats every time otherwise.
    """
    global _TESSERACT_CMD

    try:
        import pytesseract  # type: ignore

        settings = pytesseract.pytesseract
    except Exception:
        return ""

    if _TESSERACT_CMD is not None:
        # Re-assign rather than returning early: the cache remembers WHERE
        # tesseract is, not that some module object was told about it.
        try:
            if _TESSERACT_CMD:
                settings.tesseract_cmd = _TESSERACT_CMD
        except Exception:
            pass
        return _TESSERACT_CMD

    try:
        configured = str(getattr(settings, "tesseract_cmd", "") or "")
        if configured and Path(configured).is_file():
            _TESSERACT_CMD = configured
            return configured

        import shutil

        found = shutil.which("tesseract")
        candidates = [found] if found else []
        candidates += [
            str(Path(base) / "Tesseract-OCR" / "tesseract.exe")
            for base in (
                os.environ.get("PROGRAMFILES", "C:" + os.sep + "Program Files"),
                os.environ.get("PROGRAMFILES(X86)", ""),
                os.environ.get("LOCALAPPDATA", ""),
            )
            if base
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                settings.tesseract_cmd = candidate
                _TESSERACT_CMD = candidate
                return candidate
    except Exception:
        return ""

    _TESSERACT_CMD = ""
    return ""


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


# ──────────────────────────────────────────────────────────────────────────────
# IDENTITY CORRECTION — the name is O.R.I.O.N., never ORIN/ORIO/ORON.
#
# The mangled forms come from two places: the Gemini Live audio TRANSCRIPTION
# (the model says the name fine, the transcriber drops a letter) and, rarely,
# the model's own text.  Prompt pleading cannot fix a transcriber, so every
# transcript and every speech payload passes through this regex instead —
# a code-level guarantee rather than a request.
# ──────────────────────────────────────────────────────────────────────────────

# Dotted forms, complete or clipped ("O.R.I.O.N.", "O.R.I.O", "O.R.I.N."), and
# clipped word forms in caps or title case (ORIN, ORIO, ORON, ORN, Orin…).
_NAME_ANY_RE = re.compile(
    r"\b(?:"
    r"O\s*\.\s*R\s*\.\s*I\s*(?:\s*\.\s*O)?(?:\s*\.\s*N)?\s*\.?"
    r"|(?:ORION|ORIN|ORIO|ORON|ORN|Orin|Orio|Oron)\.?"
    r")(?=[^A-Za-z0-9]|$)"
)


def correct_identity(text: str) -> str:
    """Rewrite every mangled or clipped self-name to the official O.R.I.O.N."""
    return _NAME_ANY_RE.sub("O.R.I.O.N.", str(text or ""))


def clean_transcript(text: str) -> str:
    """Strip model control tokens and non-printable bytes from transcripts,
    and repair any mangled self-name (ORIN/ORIO/…) the transcriber produced."""
    text = re.sub(r"<ctrl\d+>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return correct_identity(text).strip()


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


def fold_control_label(text: Any) -> str:
    """Normalise a UI CONTROL's accessible name for matching (Mark XXI,
    Track D1) — deliberately separate from fold_title rather than widening
    it, so window-title matching (control.py/agents.py's window lookups)
    keeps its exact existing behaviour and only element-click matching
    (vision.py) gains this.

    On top of fold_title's invisible-Unicode/whitespace/case folding, this
    also strips Windows mnemonic ampersands ('&Save' -> 'save', '&&' -> a
    literal '&') and folds away the "opens a dialog" ellipsis convention
    ('Save As…' / 'Save As...' / 'Save As' all fold to the same string) —
    two real-world label shapes that defeated exact/substring matching
    before this."""
    text = str(text or "")
    # && is an escaped literal ampersand; a lone & is the mnemonic marker
    # and carries no matchable meaning, so it is dropped, not the letter
    # after it.
    text = text.replace("&&", "\x00").replace("&", "").replace("\x00", "&")
    text = text.replace("…", "...")
    folded = fold_title(text)
    return re.sub(r"\.{3,}$", "", folded).strip()


def first_line(value: Any, limit: int = 160) -> str:
    """First line of an exception/string, truncated — safe for log output."""
    try:
        return str(value).splitlines()[0][:limit]
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-NAME IDENTITY — the Forge names a tool once, but every layer that
# handles it (sandbox staging, self-import detection, the generated test's own
# `import`) may spell that name differently: CamelCase, snake_case, a `_tool`
# suffix.  A tool called "EnhancedResearchModule" whose test does
# `import enhanced_research_module` is the SAME artefact — but a raw string
# compare misses that, so the mismatch was misread as a missing pip package and
# the whole forge session died trying to install it.  These two helpers make
# name identity separator- and case-insensitive, and enumerate every spelling
# the module must be importable under.
# ──────────────────────────────────────────────────────────────────────────────

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def canonical_module_id(name: Any) -> str:
    """Collapse a tool/module name to a case- and separator-insensitive identity.

    'EnhancedResearchModule', 'enhanced_research_module' and
    'enhanced-research-module' all canonicalise to 'enhancedresearchmodule',
    so the Forge can tell when a 'missing module' is really the tool itself.
    """
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _to_snake(name: str) -> str:
    """CamelCase/PascalCase → snake_case ('EnhancedResearchModule' →
    'enhanced_research_module'); already-snake names pass through unchanged."""
    s1 = _CAMEL_BOUNDARY_1.sub(r"\1_\2", str(name or ""))
    snake = _CAMEL_BOUNDARY_2.sub(r"\1_\2", s1).lower()
    return re.sub(r"[^a-z0-9]+", "_", snake).strip("_")


def module_name_variants(tool_name: str) -> list[str]:
    """Every import name a forged tool may legitimately be referenced by.

    The sandbox stages the module's source under each of these so the
    generated test's `import <whatever>` resolves regardless of the casing the
    model chose — the fix for the 'No module named ...' forge failures where
    the name simply differed in case or separators.
    """
    safe = re.sub(r"[^0-9A-Za-z_]", "_", str(tool_name or "")).strip("_")
    snake = _to_snake(tool_name)
    variants: list[str] = []
    for base in (safe, snake):
        if not base:
            continue
        for candidate in (base, f"{base}_tool"):
            if candidate and candidate not in variants:
                variants.append(candidate)
    return variants


def aiohttp_client_timeout() -> ClientTimeout:
    """Default short timeout for opportunistic network calls."""
    return ClientTimeout(total=8.0, connect=3.0)


# ──────────────────────────────────────────────────────────────────────────────
# SPEECH NORMALISATION — text every voice channel can pronounce correctly.
#
# The TTS engines (Gemini native audio and local SAPI alike) stumble on digit
# clocks ("22:27" comes out as "20:27"), bare years ("2026" → "twenty-six")
# and the all-caps name ("ORION" → "ORIN"/"ORIO").  The durable fix is to hand
# every voice channel words instead of digits, so there is nothing to misread.
# ──────────────────────────────────────────────────────────────────────────────

_UNITS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety")


def _two_digit_words(n: int) -> str:
    """0–99 in words ('27' → 'twenty-seven')."""
    if n < 20:
        return _UNITS[n]
    tens, units = divmod(n, 10)
    return _TENS[tens] + (f"-{_UNITS[units]}" if units else "")


def spoken_time(hour: int, minute: int) -> str:
    """A 24-hour clock reading as natural spoken words.

    22:27 → 'ten twenty-seven in the evening'; 07:05 → 'seven oh five in the
    morning'; 13:00 → 'one o'clock in the afternoon'.  Words only — no digits
    for any voice to misread."""
    hour, minute = int(hour) % 24, int(minute) % 60
    if hour == 0 and minute == 0:
        return "midnight"
    if hour == 12 and minute == 0:
        return "midday"
    period = ("in the morning" if hour < 12 else
              "in the afternoon" if hour < 18 else
              "in the evening" if hour < 22 else "at night")
    h12 = hour % 12 or 12
    if minute == 0:
        return f"{_UNITS[h12]} o'clock {period}"
    if minute < 10:
        return f"{_UNITS[h12]} oh {_UNITS[minute]} {period}"
    return f"{_UNITS[h12]} {_two_digit_words(minute)} {period}"


def spoken_year(year: int) -> str:
    """A calendar year in spoken words: 2026 → 'twenty twenty-six';
    2007 → 'two thousand and seven'; 1999 → 'nineteen ninety-nine'."""
    year = int(year)
    century, rest = divmod(year, 100)
    if 2000 <= year < 2010:
        return f"two thousand{f' and {_UNITS[rest]}' if rest else ''}"
    if 1000 <= year < 3000:
        return f"{_two_digit_words(century)} {_two_digit_words(rest) if rest else 'hundred'}"
    return str(year)


_CLOCK_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def normalise_for_speech(text: str) -> str:
    """Rewrite text so any TTS voice pronounces it correctly.

    • clock times → words ('22:27' → 'ten twenty-seven in the evening')
    • bare years → words ('2026' → 'twenty twenty-six')
    • every self-name form — dotted, all-caps OR mangled (ORIN/ORIO/…) —
      → 'Orion' (all-caps gets spelled out and truncated by some engines;
      title-case reads as the constellation, which is the correct sound)
    """
    text = str(text or "")
    text = _CLOCK_RE.sub(lambda m: spoken_time(int(m.group(1)), int(m.group(2))), text)
    text = _YEAR_RE.sub(lambda m: spoken_year(int(m.group(1))), text)
    text = _NAME_ANY_RE.sub("Orion", text)
    return text


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
