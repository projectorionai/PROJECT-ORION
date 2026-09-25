"""
ElevenLabsVoice — ElevenLabs text-to-speech as a real, swappable voice
provider for SpeechSynthesiser's local/fallback voice path (audio.py).

Deliberately NOT wired into the live Gemini conversational channel
(live_worker.py / AudioPlaybackThread) — that is a different, hard-realtime
system (native audio streamed directly from the model as it speaks) and
rearchitecting it is a separate, much riskier undertaking than this module.
What this gives ORION is a genuinely modern alternative for the voice used
whenever that primary channel is unavailable — today pyttsx3 or a
PowerShell SAPI call — selected automatically the moment an API key and
voice_id are configured, and falling straight back to the existing local
engines on any failure, so nothing about ORION's voice becomes a hard
dependency on a paid API.

Pure and dependency-free beyond the stdlib + numpy (no Qt, no threading of
its own), so it is trivially testable — matches the convention already
used by config_models.py/audio_devices.py/camera_devices.py.

Configuration lives inside config/api_keys.json's own "elevenlabs" section
(the established home for every secret this app holds), resolved fresh on
each call rather than cached, so a key/voice change takes effect on the
next utterance with no restart.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .constants import API_CONFIG_PATH, CONFIG_DIR
from .atomic_io import atomic_write_text

API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL_ID = "eleven_turbo_v2_5"
SAMPLE_RATE = 24000
CACHE_DIR = CONFIG_DIR / "tts_cache"
MAX_CACHE_FILES = 200
REQUEST_TIMEOUT_S = 20.0
# Bytes pulled per streaming read.  4096 bytes of 24 kHz 16-bit mono is ~85 ms
# — small enough that the first sound arrives promptly, large enough that the
# renderer's coalescing still produces one contiguous device write.
STREAM_CHUNK_BYTES = 4096

# Local pronunciation substitutions applied before every synthesis call —
# the practical equivalent of ElevenLabs' account-level pronunciation
# dictionary feature (a PLS lexicon uploaded through their dashboard, which
# ORION's own code has no API to manage); a plain text substitution table
# achieves the same practical outcome — ORION says "ORION" the way we want
# — without depending on that account feature existing.
_DEFAULT_PRONUNCIATION = {
    "ORION": "OH-ree-on",
}


# ── configuration (config/api_keys.json -> "elevenlabs") ────────────────────

def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _elevenlabs_section() -> dict[str, Any]:
    section = _load_json(API_CONFIG_PATH).get("elevenlabs")
    return section if isinstance(section, dict) else {}


def _save_elevenlabs_section(section: dict[str, Any]) -> None:
    try:
        API_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = _load_json(API_CONFIG_PATH)
        data["elevenlabs"] = section
        atomic_write_text(API_CONFIG_PATH, json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def resolve_api_key() -> str:
    return str(_elevenlabs_section().get("api_key") or "").strip()


def resolve_voice_id() -> str:
    return str(_elevenlabs_section().get("voice_id") or "").strip()


def set_voice(api_key: str = "", voice_id: str = "") -> str:
    """Persist the ElevenLabs credential/voice. A blank argument leaves
    whatever is already configured for that field untouched, so the API
    key and voice_id can be set independently."""
    section = _elevenlabs_section()
    if api_key.strip():
        section["api_key"] = api_key.strip()
    if voice_id.strip():
        section["voice_id"] = voice_id.strip()
    _save_elevenlabs_section(section)
    return describe()


def describe() -> str:
    key_state = "configured" if resolve_api_key() else "not set"
    voice = resolve_voice_id() or "not set"
    preset = active_preset()
    preset_bit = f", preset '{preset}'" if preset else ""
    return f"ElevenLabs voice — API key {key_state}, voice_id {voice}{preset_bit}."


# ── character voice presets ──────────────────────────────────────────────────
#
# Stable ElevenLabs PREMADE voice ids — they ship with every account, so a
# preset works the moment an API key is configured, without setting up a custom
# clone first. The user can still override voice_id with ANY voice from their
# ElevenLabs Voice Library (including a bespoke Ultron clone) via set_voice.
#
# "Ultron" is deliberately built on the deepest, most measured premade male
# voice, with delivery settings tuned for calm menace — steady enough to read as
# controlled and cold, expressive enough to carry the sardonic edge.
VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "ultron": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",   # "Adam" — deep, measured, authoritative
        "name": "Ultron",
        "description": "deep, calm and menacing — measured authority with an edge",
        "settings": {"stability": 0.45, "similarity_boost": 0.9, "style": 0.45},
    },
    "jarvis": {
        "voice_id": "onwK4e9ZLuTAKqWW03F9",   # "Daniel" — refined British authority
        "name": "Jarvis",
        "description": "refined, precise and British — calm and impeccably composed",
        "settings": {"stability": 0.62, "similarity_boost": 0.85, "style": 0.15},
    },
    "narrator": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",   # "Adam" — clean, deep narration
        "name": "Narrator",
        "description": "clean, deep, neutral narration",
        "settings": {"stability": 0.60, "similarity_boost": 0.80, "style": 0.00},
    },
}


def active_preset() -> str:
    return str(_elevenlabs_section().get("preset") or "").strip().lower()


def apply_preset(name: str) -> str:
    """Select a character voice preset: sets the voice_id AND remembers the
    preset so its delivery settings shape every line. Returns a human summary."""
    key = str(name or "").strip().lower()
    preset = VOICE_PRESETS.get(key)
    if preset is None:
        return (f"No voice preset called '{name}'. Available: "
                + ", ".join(VOICE_PRESETS) + ".")
    section = _elevenlabs_section()
    section["voice_id"] = preset["voice_id"]
    section["preset"] = key
    _save_elevenlabs_section(section)
    tail = ("It takes effect on the next line ORION speaks through the local/"
            "fallback voice." if resolve_api_key() else
            "Add your ElevenLabs API key to config/api_keys.json (the "
            "\"elevenlabs\" section) and it's live.")
    return f"Voice set to '{preset['name']}' — {preset['description']}. {tail}"


def pronunciation_dictionary() -> dict[str, str]:
    custom = _elevenlabs_section().get("pronunciation")
    merged = dict(_DEFAULT_PRONUNCIATION)
    if isinstance(custom, dict):
        merged.update({str(k): str(v) for k, v in custom.items()})
    return merged


def apply_pronunciation(text: str, table: dict[str, str] | None = None) -> str:
    table = table if table is not None else pronunciation_dictionary()
    if not table:
        return text
    for word, replacement in table.items():
        text = re.sub(rf"\b{re.escape(word)}\b", replacement, text, flags=re.IGNORECASE)
    return text


# ── emotion -> ElevenLabs voice_settings ─────────────────────────────────────
#
# SpeechSynthesiser already tracks the current delivery emotion (fed by the
# emotion engine) to nudge pyttsx3's rate/volume; this reuses the SAME
# label to modulate ElevenLabs' own native stability/style controls instead
# of trying to fake a pitch shift on the output audio.
_EMOTION_VOICE_SETTINGS: dict[str, dict[str, float]] = {
    "neutral":   {"stability": 0.55, "similarity_boost": 0.75, "style": 0.00},
    "excited":   {"stability": 0.35, "similarity_boost": 0.75, "style": 0.35},
    "calm":      {"stability": 0.70, "similarity_boost": 0.75, "style": 0.00},
    "urgent":    {"stability": 0.30, "similarity_boost": 0.70, "style": 0.25},
    "sad":       {"stability": 0.65, "similarity_boost": 0.75, "style": 0.10},
    "confident": {"stability": 0.55, "similarity_boost": 0.80, "style": 0.20},
}


def voice_settings_for_emotion(emotion: str) -> dict[str, float]:
    """The PURE emotion→settings mapping (no preset, no global state)."""
    return dict(_EMOTION_VOICE_SETTINGS.get(
        str(emotion or "neutral").lower(), _EMOTION_VOICE_SETTINGS["neutral"]))


def resolve_delivery_settings(emotion: str) -> dict[str, float]:
    """The settings actually used for a spoken line: the active character
    preset if one is set (the emotion only nudges within it, so Ultron stays
    Ultron whether calm or urgent — just colder or edgier), otherwise the plain
    emotion mapping. This is the global-state-aware layer over the pure
    ``voice_settings_for_emotion``, kept separate so that stays deterministic."""
    emo = str(emotion or "neutral").lower()
    preset = VOICE_PRESETS.get(active_preset())
    if preset is None:
        return voice_settings_for_emotion(emo)
    settings = dict(preset["settings"])
    if emo in {"excited", "urgent"}:
        settings["stability"] = max(0.20, settings.get("stability", 0.5) - 0.12)
        settings["style"] = min(1.0, settings.get("style", 0.0) + 0.10)
    elif emo in {"calm", "sad"}:
        settings["stability"] = min(0.90, settings.get("stability", 0.5) + 0.10)
    return settings


def list_voices() -> list[dict[str, str]]:
    """The account's available voices — [{"voice_id": ..., "name": ...}, ...]
    — so a voice can be chosen by name rather than needing to already know
    its opaque id. Returns [] (never raises) when no API key is configured
    or the request fails; this is a convenience lookup, not on ORION's
    speech-critical path."""
    api_key = resolve_api_key()
    if not api_key:
        return []
    req = urllib.request.Request(
        f"{API_BASE}/voices", headers={"xi-api-key": api_key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []
    voices = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(voices, list):
        return []
    return [
        {"voice_id": str(v.get("voice_id") or ""), "name": str(v.get("name") or "")}
        for v in voices if isinstance(v, dict) and v.get("voice_id")
    ]


# ── synthesis + caching ───────────────────────────────────────────────────────

class ElevenLabsVoice:
    """One HTTP-backed synthesis call producing playback-ready 16-bit PCM,
    with on-disk caching so repeated/common utterances skip the network
    call entirely — a real latency and cost win, since this is a fallback
    voice invoked whenever the primary Gemini Live channel is unavailable,
    and small talk repeats ("one moment", "done", "good morning")."""

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    def available(self) -> bool:
        return bool(resolve_api_key()) and bool(resolve_voice_id())

    def _cache_key(self, text: str, voice_id: str, settings: dict[str, float]) -> str:
        blob = json.dumps(
            {"t": text, "v": voice_id, "s": settings, "m": self.model_id}, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.pcm"

    def _read_cache(self, key: str) -> bytes | None:
        path = self._cache_path(key)
        try:
            return path.read_bytes() if path.is_file() else None
        except OSError:
            return None

    def _write_cache(self, key: str, pcm: bytes) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_bytes(pcm)
            self._evict_if_over_capacity()
        except OSError:
            pass

    def _evict_if_over_capacity(self) -> None:
        try:
            files = sorted(CACHE_DIR.glob("*.pcm"), key=lambda p: p.stat().st_mtime)
            excess = len(files) - MAX_CACHE_FILES
            for path in files[:max(0, excess)]:
                path.unlink(missing_ok=True)
        except OSError:
            pass

    def synthesize_pcm(self, text: str, emotion: str = "neutral") -> bytes | None:
        """Blocking: a cache hit, or one HTTP call. Returns None on ANY
        failure (missing config, network fault, bad response) — this must
        never raise, so the caller can fall back to a local engine for
        just this one utterance."""
        if not self.available():
            return None
        clean_text = apply_pronunciation(str(text or "").strip())
        if not clean_text:
            return None
        voice_id = resolve_voice_id()
        settings = resolve_delivery_settings(emotion)
        key = self._cache_key(clean_text, voice_id, settings)
        cached = self._read_cache(key)
        if cached:
            return cached
        try:
            pcm = self._request_pcm(clean_text, voice_id, settings)
        except Exception:
            return None
        if pcm:
            self._write_cache(key, pcm)
        return pcm

    def _request_pcm(self, text: str, voice_id: str, settings: dict[str, float]) -> bytes | None:
        url = f"{API_BASE}/text-to-speech/{voice_id}?output_format=pcm_{self.SAMPLE_RATE}"
        payload = json.dumps({
            "text": text,
            "model_id": self.model_id,
            "voice_settings": settings,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={
                "xi-api-key": resolve_api_key(),
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            if resp.status != 200:
                return None
            return resp.read()

    # ── streaming synthesis (the latency path) ────────────────────────────────

    def stream_pcm(self, text: str, emotion: str = "neutral", *,
                   on_chunk: Any = None, should_stop: Any = None) -> bytes | None:
        """Stream this utterance, handing each chunk to *on_chunk* as it lands.

        Why streaming matters here
        --------------------------
        ``synthesize_pcm`` waits for the ENTIRE utterance before a single
        sample can play.  For a sentence that is a beat of silence; for a
        briefing it is several seconds, and the queue then plays utterance,
        gap, utterance, gap — which is exactly what "his speech stutters"
        describes on this path.  Streaming starts the audio on the first
        chunk off the wire, so time-to-first-sound stops scaling with how much
        ORION has to say.

        *on_chunk* receives raw 16-bit little-endian PCM at
        :data:`SAMPLE_RATE`, ready to hand straight to the renderer's queue.
        *should_stop* is polled between chunks so a barge-in cuts the network
        read immediately instead of after the whole utterance has downloaded.

        Returns the complete PCM (for the cache), or None on any failure —
        never raises, so the caller can fall back to a local engine.  A cache
        hit skips the network entirely and is delivered through *on_chunk* in
        renderer-sized pieces, so cached and fresh utterances behave the same.
        """
        if not self.available():
            return None
        clean_text = apply_pronunciation(str(text or "").strip())
        if not clean_text:
            return None
        voice_id = resolve_voice_id()
        settings = resolve_delivery_settings(emotion)
        key = self._cache_key(clean_text, voice_id, settings)

        cached = self._read_cache(key)
        if cached:
            if on_chunk is not None:
                for i in range(0, len(cached), STREAM_CHUNK_BYTES):
                    if should_stop is not None and should_stop():
                        return cached
                    on_chunk(cached[i:i + STREAM_CHUNK_BYTES])
            return cached

        url = (f"{API_BASE}/text-to-speech/{voice_id}/stream"
               f"?output_format=pcm_{self.SAMPLE_RATE}")
        payload = json.dumps({
            "text": clean_text,
            "model_id": self.model_id,
            "voice_settings": settings,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={
                "xi-api-key": resolve_api_key(),
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
            },
        )
        parts: list[bytes] = []
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                if resp.status != 200:
                    return None
                while True:
                    if should_stop is not None and should_stop():
                        # Interrupted mid-download: what we already played is
                        # real, but an incomplete utterance must never be
                        # cached as if it were the whole thing.
                        return None
                    chunk = resp.read(STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    parts.append(chunk)
                    if on_chunk is not None:
                        on_chunk(chunk)
        except Exception:
            # Nothing played yet -> a clean fallback to the local engine.
            # Something already played -> report success so the caller does
            # not speak the same sentence twice in two different voices.
            return b"".join(parts) if parts else None
        pcm = b"".join(parts)
        if pcm:
            self._write_cache(key, pcm)
        return pcm


def apply_fade(samples: Any, fade_frames: int = 240) -> Any:
    """A short linear fade-in/out (~10ms at 24kHz) on an int16 PCM array so
    utterance boundaries never click — cheap, standard audio hygiene, and
    the reason this exists rather than handing raw bytes straight to the
    output device."""
    import numpy as np
    if samples.size == 0:
        return samples
    out = samples.astype(np.float32, copy=True)
    n = min(fade_frames, out.size // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    return out.astype(np.int16)


__all__ = [
    "ElevenLabsVoice", "apply_fade", "apply_pronunciation", "describe",
    "list_voices", "pronunciation_dictionary", "resolve_api_key",
    "resolve_voice_id", "set_voice", "voice_settings_for_emotion",
    "VOICE_PRESETS", "active_preset", "apply_preset",
]
