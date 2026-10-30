"""
Which voice ORION speaks in - one small stored choice (Mark XXXVII).

ORION has two voices, and they used to be fixed in code:

  * the CLOUD voice - Gemini's prebuilt "Charon" (deep, calm), used by the Live
    channel, by the cached lines he says offline and by the phone's speech;
  * the LOCAL voice - Kokoro (voice_kokoro), used whenever Charon cannot speak
    (no internet, a spent quota): "bm_george", a calm British man.

ORION is a man and has ONE voice (owner, 26 Oct 2026: "his voice keeps
changing ... ORION is not a girl, so he should not have a girl's voice"):
Charon online and a man's Kokoro voice offline, whatever the style. A style
changes how he TALKS (voice_settings.manner, added to his persona), never who
he sounds like. Only men's voices can be chosen at all - and not Puck, which
the owner heard as a girl's. This module is where the choice lives:

    kokoro_voice   which Kokoro voice (any English voice in voices-v1.0.bin)
    kokoro_speed   Kokoro's speed factor (1.0 = normal; 0.8 - 1.25)
    cloud_voice    which Gemini prebuilt voice (Charon, Puck, Leda, ...)

Reading order is the same as appearance.py / windows_settings.py: an
environment override for a one-off (ORION_VOICE_KOKORO_VOICE,
ORION_VOICE_KOKORO_SPEED, ORION_VOICE_CLOUD_VOICE), then the stored choice
(CONFIG_DIR/voices/voice_settings.json), then the default. THE DEFAULTS ARE
UNCHANGED - Charon and bm_george - until the owner picks from the samples.

A value that is not a real voice is never stored and never used: a typo in the
file falls back to the default instead of handing the speech engine a voice
that does not exist.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR, VOICE_PROFILE

#: Tests point this at a temp file (conftest does it for every test).
SETTINGS_PATH = CONFIG_DIR / "voices" / "voice_settings.json"

# ── what exists ──────────────────────────────────────────────────────────────

#: The English voices in Kokoro v1.0's voices-v1.0.bin (54 voices in all; the
#: rest speak other languages). af_/bf_ women, am_/bm_ men; a_ American, b_ British.
KOKORO_VOICES: tuple[str, ...] = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)

#: The voices ORION may speak in: men's voices only (am_/bm_ and Gemini's
#: male voices), without Puck. A woman's voice in the file or the environment
#: is ignored like a typo.
_WOMEN_KOKORO_PREFIXES = ("af_", "bf_")
#: Gemini's women's voices, and Puck (heard by the owner as a girl's).
NOT_HIS_GEMINI_VOICES = frozenset({
    "Zephyr", "Kore", "Leda", "Aoede", "Callirrhoe", "Autonoe", "Despina", "Erinome",
    "Laomedeia", "Achernar", "Gacrux", "Pulcherrima", "Vindemiatrix", "Sulafat", "Puck",
})

#: Gemini's prebuilt voices (the documented set; remote_speech checks the same).
GEMINI_VOICES: tuple[str, ...] = (
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
)

KOKORO_DEFAULT_VOICE = "bm_george"
KOKORO_DEFAULT_SPEED = 1.0
KOKORO_SPEED_LIMITS = (0.8, 1.25)
CLOUD_DEFAULT_VOICE = VOICE_PROFILE.gemini_voice_name        # "Charon"

#: How long a pause in the owner's speech must last before ORION answers
#: (Mark XXXX.1). The live channel's own default ended the turn on the first
#: short breath - "I can be talking and he can cut off listening and
#: instantly start replying" (owner, 27 Oct 2026).
REPLY_PAUSES: dict[str, int] = {"brisk": 700, "natural": 1200, "patient": 1700}
DEFAULT_REPLY_PAUSE = "natural"
_PAUSE_WORDS = {
    "brisk": ("brisk", "short", "shorter", "quick", "quicker", "fast", "faster", "snappy"),
    "natural": ("natural", "normal", "default", "medium", "standard"),
    "patient": ("patient", "long", "longer", "slow", "slower", "more time", "relaxed"),
}

DEFAULTS: dict[str, Any] = {
    "kokoro_voice": KOKORO_DEFAULT_VOICE,
    "kokoro_speed": KOKORO_DEFAULT_SPEED,
    "cloud_voice": CLOUD_DEFAULT_VOICE,
    "reply_pause": DEFAULT_REPLY_PAUSE,
    # Affective dialog: the native-audio model answers in a tone that
    # matches the owner's own - warmer, quieter, brisker - instead of one
    # level delivery (Mark XXXX.1).
    "expressive": True,
}
ENV_NAMES: dict[str, str] = {
    "kokoro_voice": "ORION_VOICE_KOKORO_VOICE",
    "kokoro_speed": "ORION_VOICE_KOKORO_SPEED",
    "cloud_voice": "ORION_VOICE_CLOUD_VOICE",
    "reply_pause": "ORION_REPLY_PAUSE",
    "expressive": "ORION_EXPRESSIVE_VOICE",
}

# ── the candidates the owner chooses between ─────────────────────────────────
# (value, label). Labels are names, so Title Case. The first entry of each list
# is today's default.

KOKORO_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bm_george", "George (Calm British Man, Current)"),
    ("bm_lewis", "Lewis (Deep British Man)"),
    ("bm_daniel", "Daniel (Clear British Man)"),
    ("am_michael", "Michael (Warm American Man)"),
    ("am_liam", "Liam (Easy American Man)"),
)

CLOUD_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("Charon", "Charon (Deep and Calm, Current)"),
    ("Orus", "Orus (Firm)"),
    ("Iapetus", "Iapetus (Clear)"),
    ("Algenib", "Algenib (Gravelly)"),
    ("Rasalgethi", "Rasalgethi (Informative)"),
    ("Schedar", "Schedar (Even)"),
)


@dataclass(frozen=True)
class VoiceStyle:
    """A named bundle: how ORION sounds locally AND in the cloud, so one phrase
    ("use your cute voice") changes both and he is the same character online
    and off - and how he TALKS in it (*manner*, added to his persona)."""

    key: str
    label: str
    kokoro_voice: str
    kokoro_speed: float
    cloud_voice: str
    words: tuple[str, ...]            # what people say to ask for it
    manner: str = ""                  # "" = his usual (professional) manner


#: ONE way of talking (Mark XXXIX). The cute manner (26 Oct, Mark 37.4) is
#: retired - owner, 27 Oct 2026: "ORION's voice must sound professional and not
#: cute but also fit his face" - so a stored "cute" from an older build now
#: reads as professional; asking for it gets his one style (Mark XXXX.4 removed it).
STYLES: dict[str, VoiceStyle] = {
    "classic": VoiceStyle("classic", "Professional", KOKORO_DEFAULT_VOICE, KOKORO_DEFAULT_SPEED,
                          CLOUD_DEFAULT_VOICE,
                          ("professional", "formal", "serious", "proper", "posh",
                           "classic", "normal", "default", "original", "old voice",
                           "deep", "butler", "charon", "george", "usual", "back")),
}
DEFAULT_STYLE = "classic"
#: Older and other names for a style, still understood.
_STYLE_ALIASES = {"professional": "classic"}
#: Asked for a woman's voice: he keeps his own.
_WOMEN_WORDS_RE = re.compile(r"\b(?:girl|girly|female|woman|women|feminine|lady|leda|"
                             r"heart|zephyr|kore)\b", re.I)

# ── normalising what people type or say ──────────────────────────────────────


def _letters(text: Any) -> str:
    return re.sub(r"[^a-z]", "", str(text or "").lower())


def normalise_kokoro_voice(name: Any) -> str | None:
    """'af_heart', 'Heart', 'BF-Lily', 'george' -> a real Kokoro voice id, else None."""
    key = _letters(name)
    if not key:
        return None
    for voice in KOKORO_VOICES:
        if voice.startswith(_WOMEN_KOKORO_PREFIXES):
            continue
        if key in (voice.replace("_", ""), voice.split("_", 1)[1]):
            return voice
    return None


def normalise_cloud_voice(name: Any) -> str | None:
    """'leda', 'CHARON ' -> 'Leda' / 'Charon' (a real Gemini voice), else None."""
    key = _letters(name)
    for voice in GEMINI_VOICES:
        if key and key == voice.lower():
            return None if voice in NOT_HIS_GEMINI_VOICES else voice
    return None


def normalise_speed(value: Any) -> float | None:
    """A speed factor within KOKORO_SPEED_LIMITS (clamped), or None if it is no number."""
    if isinstance(value, bool):
        return None
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return None
    if speed != speed:                      # NaN
        return None
    low, high = KOKORO_SPEED_LIMITS
    return round(max(low, min(high, speed)), 3)


def normalise_reply_pause(value: Any) -> str | None:
    """'brisk', 'natural' or 'patient' from a name or a word for one."""
    text = " ".join(str(value or "").lower().replace("_", " ").split())
    if not text:
        return None
    for name, words in _PAUSE_WORDS.items():
        if text in words or any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            return name
    return None


def normalise_flag(value: Any) -> bool | None:
    """True/False from a bool or an on/off word; None for anything else."""
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "on", "yes"}:
        return True
    if text in {"0", "false", "off", "no"}:
        return False
    return None


_NORMALISERS = {
    "kokoro_voice": normalise_kokoro_voice,
    "kokoro_speed": normalise_speed,
    "cloud_voice": normalise_cloud_voice,
    "reply_pause": normalise_reply_pause,
    "expressive": normalise_flag,
}


# ── storage ──────────────────────────────────────────────────────────────────

def _load() -> dict[str, Any]:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict[str, Any]) -> bool:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(SETTINGS_PATH, json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def stored(key: str) -> Any:
    """The valid value stored for *key*, or None if nothing (valid) is stored.
    Ignores the environment and the default - for callers with their own
    fallback (voice_kokoro's older kokoro.json)."""
    if key not in DEFAULTS:
        raise KeyError(key)
    raw = _load().get(key)
    return None if raw is None else _NORMALISERS[key](raw)


def override(key: str) -> Any:
    """The valid environment override for *key*, or None."""
    if key not in DEFAULTS:
        raise KeyError(key)
    raw = os.getenv(ENV_NAMES[key], "").strip()
    return _NORMALISERS[key](raw) if raw else None


def get(key: str) -> Any:
    """The value in effect for *key*: environment, then stored, then default.
    Unknown keys raise KeyError."""
    if key not in DEFAULTS:
        raise KeyError(key)
    for found in (override(key), stored(key)):
        if found is not None:
            return found
    return DEFAULTS[key]


def kokoro_voice() -> str:
    return str(get("kokoro_voice"))


def kokoro_speed() -> float:
    return float(get("kokoro_speed"))


def cloud_voice() -> str:
    return str(get("cloud_voice"))


def reply_pause() -> str:
    return str(get("reply_pause"))


def reply_pause_ms() -> int:
    """Silence, in milliseconds, that ends the owner's turn."""
    return REPLY_PAUSES.get(reply_pause(), REPLY_PAUSES[DEFAULT_REPLY_PAUSE])


def expressive() -> bool:
    return bool(get("expressive"))


def snapshot() -> dict[str, Any]:
    return {key: get(key) for key in DEFAULTS}


def set_value(key: str, value: Any) -> bool:
    """Store *value* for *key* (normalised: 'heart' -> 'af_heart'). False, and
    nothing stored, for an unknown key or a value that names no real voice."""
    if key not in DEFAULTS:
        return False
    normalised = _NORMALISERS[key](value)
    if normalised is None:
        return False
    data = _load()
    data[key] = normalised
    data.setdefault("schema", "orion.voice_settings.v1")
    return _save(data)


def reset() -> bool:
    """Back to the defaults (Charon and bm_george), keeping nothing stored."""
    data = {key: value for key, value in _load().items()
            if key not in DEFAULTS and key != "style"}
    data.setdefault("schema", "orion.voice_settings.v1")
    return _save(data)


# ── styles ───────────────────────────────────────────────────────────────────

def style_from_words(text: Any) -> str:
    """The style a phrase asks for ('back to professional') - or a RETIRED
    one ('your cute voice' -> 'cute', which apply_style declines) - or ''.
    A woman's voice is not a style he has ('' - he keeps his own); whole
    words only, so 'duty' is not 'cute'."""
    if _WOMEN_WORDS_RE.search(str(text or "")):
        return ""
    said = " " + re.sub(r"[^a-z_ ]", " ", str(text or "").lower()) + " "
    said = re.sub(r"\s+", " ", said)
    for key, words in ((key, style.words) for key, style in STYLES.items()):
        for word in words:
            if f" {word} " in said or f" {word}s " in said:
                return key
    return ""


def _style_key(name: Any) -> str:
    text = str(name or "").strip()
    key = _STYLE_ALIASES.get(text.lower().replace(" ", "_"), text)
    return key if key in STYLES else style_from_words(text)


def chosen_style() -> str:
    """The style the owner last chose (stored by apply_style), else the one
    the voices match exactly, else 'custom'."""
    stored_key = _load().get("style")
    if stored_key in STYLES:
        return stored_key
    return current_style()


def manner() -> str:
    """How ORION should talk in the chosen style - a persona addition, or ""
    for his usual manner (Professional, or a custom mix)."""
    style = STYLES.get(chosen_style())
    return style.manner if style is not None else ""


def current_style() -> str:
    """The style the stored/effective choice matches exactly, else 'custom'."""
    snap = snapshot()
    for key, style in STYLES.items():
        if (snap["kokoro_voice"] == style.kokoro_voice
                and abs(float(snap["kokoro_speed"]) - style.kokoro_speed) < 1e-6
                and snap["cloud_voice"] == style.cloud_voice):
            return key
    return "custom"


def apply_style(name: Any) -> tuple[str, bool]:
    """Make *name* ('professional') the way ORION talks, stored for later runs
    - always in his one voice. A retired style ('cute') is declined. (what he
    should say, ok). There is one style now; the cute one was removed (Mark
    XL.4, after the owner retired it in Mark XXXIX)."""
    if _WOMEN_WORDS_RE.search(str(name or "")):
        return ("I keep my own voice - I'm not a woman, and I talk the same "
                "professional way in it.", False)
    key = _style_key(name)
    if key not in STYLES:
        return ("I talk one way, in my own voice: "
                + " or ".join(style.label.lower() for style in STYLES.values()) + ".", False)
    style = STYLES[key]
    ok = (set_value("kokoro_voice", style.kokoro_voice)
          and set_value("kokoro_speed", style.kokoro_speed)
          and set_value("cloud_voice", style.cloud_voice))
    if ok:
        data = _load()
        data["style"] = key
        ok = _save(data)
    if not ok:
        return "I could not save that voice. Nothing has changed.", False
    return (f"Done - I'll talk in my {style.label.lower()} way, in my own voice.", True)


def describe_kokoro(voice: str) -> str:
    labels = dict(KOKORO_CANDIDATES)
    return labels.get(voice, voice).split(" (")[0]


def describe() -> str:
    """One line for diagnostics and for ORION to say out loud."""
    style = STYLES.get(chosen_style())
    return (f"Voice: online {cloud_voice()}; offline Kokoro {describe_kokoro(kokoro_voice())} "
            f"({kokoro_voice()}) at speed {kokoro_speed():g}; style "
            f"{style.label.lower() if style else 'custom'}.")


__all__ = [
    "CLOUD_CANDIDATES", "CLOUD_DEFAULT_VOICE", "DEFAULTS", "DEFAULT_STYLE", "ENV_NAMES",
    "GEMINI_VOICES", "KOKORO_CANDIDATES", "KOKORO_DEFAULT_SPEED", "KOKORO_DEFAULT_VOICE",
    "KOKORO_SPEED_LIMITS", "KOKORO_VOICES", "SETTINGS_PATH",
    "REPLY_PAUSES", "STYLES", "VoiceStyle", "apply_style", "expressive", "normalise_flag",
    "normalise_reply_pause", "reply_pause", "reply_pause_ms", "chosen_style", "cloud_voice", "current_style", "describe",
    "describe_kokoro", "get", "manner",
    "kokoro_speed", "kokoro_voice", "normalise_cloud_voice", "normalise_kokoro_voice",
    "normalise_speed", "override", "reset", "set_value", "snapshot", "stored",
    "style_from_words",
]
