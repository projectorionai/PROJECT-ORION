"""Selectable Gemini speech for paired phones. Credentials remain on the host."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import wave
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from .constants import VOICE_PROFILE

MAX_SPEECH_CHARS = 1200
MAX_AUDIO_BYTES = 12_000_000
PCM_RATE = 24000
MAX_STREAM_EVENT_BYTES = 4_000_000
GEMINI_VOICES = frozenset({
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
})


def speech_voice(value: Any = None) -> str:
    """Accept documented voice names only; preserve the existing default."""
    if value is None:
        try:
            from .voice_settings import cloud_voice
            return cloud_voice()           # Charon unless the owner chose another
        except Exception:
            return VOICE_PROFILE.gemini_voice_name
    if not isinstance(value, str) or value not in GEMINI_VOICES:
        raise ValueError("Choose a supported Gemini voice")
    return value


class SpeechUnavailable(RuntimeError):
    """A bounded, credential-free explanation suitable for a paired client."""


def audio_to_wav(data: bytes, mime_type: str) -> bytes:
    """Validate provider audio and wrap headerless mono PCM16 when necessary."""
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise SpeechUnavailable("Speech audio is empty or too large")
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(data), "rb") as audio:
                if (audio.getnchannels() != 1 or audio.getsampwidth() != 2
                        or not 8000 <= audio.getframerate() <= 48000
                        or audio.getnframes() > audio.getframerate() * 180):
                    raise ValueError("Unsupported WAV")
                expected = audio.getnframes() * 2
                if len(audio.readframes(audio.getnframes())) != expected:
                    raise ValueError("Truncated WAV")
        except (ValueError, EOFError, wave.Error) as exc:
            raise SpeechUnavailable("Unsupported speech audio") from exc
        return data
    mime = mime_type.lower()
    if not (mime.startswith("audio/l16") or mime.startswith("audio/pcm")) or len(data) % 2:
        raise SpeechUnavailable("Unsupported speech audio")
    match = re.search(r"rate=(\d+)", mime)
    rate = int(match[1]) if match else 24000
    channels = re.search(r"channels=(\d+)", mime)
    if channels and channels[1] != "1":
        raise SpeechUnavailable("Unsupported speech channel count")
    if not 8000 <= rate <= 48000 or len(data) > rate * 2 * 180:
        raise SpeechUnavailable("Unsupported speech sample rate or duration")
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(data)
    return output.getvalue()


def _charon_request(router: Any, text: str, voice: str | None = None) -> tuple[str, str, dict[str, Any]]:
    """Keep credentials on the host and apply conversational British delivery."""
    voice = speech_voice(voice)
    # Paragraphs format the terminal. Spoken delivery uses natural sentence pauses.
    text = re.sub(r"\s+", " ", text).strip()
    settings = getattr(router, "settings", None)
    profile = getattr(settings, "providers", {}).get("gemini")
    if profile is None or not getattr(profile, "enabled", False):
        raise SpeechUnavailable("Charon needs an enabled Gemini provider on the PC")
    active_key = getattr(router, "active_key", None)
    key = active_key(profile) if callable(active_key) else getattr(profile, "api_key", "")
    if not key:
        raise SpeechUnavailable("Charon needs a Gemini API key on the PC")
    model = os.getenv("ORION_REMOTE_TTS_MODEL", "gemini-3.8-flash-tts").strip()
    if not re.fullmatch(r"gemini-[a-z0-9.-]+tts[a-z0-9.-]*", model):
        raise SpeechUnavailable("The phone speech model name is invalid")
    payload = {
        "contents": [{"parts": [{"text":
            "Read the following text verbatim in British English at a brisk, "
            "natural conversational pace, with short pauses between sentences. "
            "Do not add commentary or read these instructions aloud.\n\n" + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {
                "voiceName": voice}}},
        },
    }
    # Current models separate delivery instructions from the spoken transcript.
    if model.startswith("gemini-3.8-"):
        payload["contents"] = [{"role": "user", "parts": [{"text": text,
            "speech_metadata": {"style": "British English, brisk natural "
                                         "conversation, short sentence pauses"}}]}]
        payload["generationConfig"]["speechConfig"] = {
            "voiceConfig": {"voice": voice}}
        payload["generationConfig"]["responseFormat"] = {
            "audio": {"mimeType": "AUDIO_L16", "sampleRate": PCM_RATE}}
    return key, model, payload


@asynccontextmanager
async def _speech_client(router: Any) -> AsyncIterator[Any]:
    """Reuse the router's loop-local TLS connection without sharing credentials."""
    pool = getattr(router, "_pool", None)
    if pool is not None:
        yield await pool.get()
    else:
        import aiohttp
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:
            yield client


async def synthesise_charon(router: Any, text: str, voice: str | None = None) -> bytes:
    """Generate a complete WAV for older companion apps and web clients."""
    import aiohttp

    key, model, payload = _charon_request(router, text, voice)
    timeout = aiohttp.ClientTimeout(total=45, connect=8)
    try:
        async with _speech_client(router) as client:
            async with client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key}, json=payload, timeout=timeout,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise SpeechUnavailable(f"Charon provider returned HTTP {response.status}")
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > MAX_AUDIO_BYTES * 4 // 3 + 65536:
                        raise SpeechUnavailable("Speech response is too large")
        result = json.loads(body)
        parts = result.get("candidates", [])[0].get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData", {})
            if inline.get("data"):
                data = base64.b64decode(inline["data"], validate=True)
                return audio_to_wav(data, str(inline.get("mimeType", "")))
        raise SpeechUnavailable("Charon returned no speech audio")
    except SpeechUnavailable:
        raise
    except Exception as exc:
        # Provider exceptions can include request URLs or credentials. Never
        # forward their text to the phone or the host log.
        raise SpeechUnavailable("Charon speech is temporarily unavailable") from exc


async def _speech_events(content: Any) -> AsyncIterator[dict[str, Any]]:
    """Decode bounded SSE events even when TCP splits a line or UTF-8 character."""
    pending = bytearray()
    data: list[bytes] = []
    event_size = total = 0
    async for chunk in content.iter_chunked(32768):
        total += len(chunk)
        if total > MAX_AUDIO_BYTES * 4 // 3 + 65536:
            raise SpeechUnavailable("Speech response is too large")
        pending.extend(chunk)
        while b"\n" in pending:
            line, _, rest = pending.partition(b"\n")
            pending = bytearray(rest)
            line = line.rstrip(b"\r")
            if line.startswith(b"data:"):
                fragment = line[5:].lstrip(b" ")
                data.append(fragment)
                event_size += len(fragment)
            elif not line and data:
                raw = b"\n".join(data)
                data.clear()
                event_size = 0
                if raw == b"[DONE]":
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise SpeechUnavailable("Unsupported speech event")
                yield value
            if event_size + len(pending) > MAX_STREAM_EVENT_BYTES:
                raise SpeechUnavailable("Speech event is too large")
        if event_size + len(pending) > MAX_STREAM_EVENT_BYTES:
            raise SpeechUnavailable("Speech event is too large")
    if pending.strip() or data:
        raise SpeechUnavailable("Charon speech stream was interrupted")


async def stream_charon(router: Any, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
    """Yield validated mono PCM immediately; reject a truncated provider reply."""
    import aiohttp

    key, model, payload = _charon_request(router, text, voice)
    total = 0
    finished = False
    try:
        async with _speech_client(router) as client:
            async with client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent",
                params={"alt": "sse"}, headers={"x-goog-api-key": key}, json=payload,
                timeout=aiohttp.ClientTimeout(total=120, connect=8, sock_read=25),
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise SpeechUnavailable(f"Charon provider returned HTTP {response.status}")
                async for event in _speech_events(response.content):
                    if "error" in event:
                        raise SpeechUnavailable("Charon could not finish the speech")
                    candidates = event.get("candidates", [])
                    for candidate in candidates[:1]:
                        for part in candidate.get("content", {}).get("parts", []):
                            inline = part.get("inlineData", {})
                            if not inline.get("data"):
                                continue
                            if finished:
                                raise SpeechUnavailable("Audio arrived after speech completion")
                            pcm = base64.b64decode(inline["data"], validate=True)
                            wav = audio_to_wav(pcm, str(inline.get("mimeType", "")))
                            with wave.open(io.BytesIO(wav), "rb") as audio:
                                if audio.getframerate() != PCM_RATE:
                                    raise SpeechUnavailable("Unsupported streaming speech rate")
                                pcm = audio.readframes(audio.getnframes())
                            total += len(pcm)
                            if total > min(MAX_AUDIO_BYTES, PCM_RATE * 2 * 180):
                                raise SpeechUnavailable("Speech audio is too large")
                            for offset in range(0, len(pcm), 65536):
                                yield pcm[offset:offset + 65536]
                        reason = candidate.get("finishReason")
                        if reason:
                            if reason != "STOP":
                                raise SpeechUnavailable("Charon could not finish the speech")
                            finished = True
        if not finished or total == 0:
            raise SpeechUnavailable("Charon speech stream was interrupted")
    except SpeechUnavailable:
        raise
    except Exception as exc:
        raise SpeechUnavailable("Charon speech is temporarily unavailable") from exc
