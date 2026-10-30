"""Probe Charon's Live protocol without printing credentials or speech content."""

from __future__ import annotations

import asyncio
import argparse
import base64
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODELS = {"gemini-3.8-live", "gemini-2.5-flash-native-audio-preview-12-2025"}


async def verify(model: str = "gemini-3.8-live") -> dict[str, object]:
    import websockets

    from orion_core.providers import read_provider_settings

    profile = read_provider_settings().providers.get("gemini")
    key = str(getattr(profile, "api_key", "") or "")
    if not key:
        return {"ok": False, "error": "Gemini key unavailable"}

    url = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={quote(key, safe='')}"
    )
    setup = {"setup": {
        "model": f"models/{model}",
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": "Charon"}}},
        },
        "systemInstruction": {"parts": [{"text":
            "Read the supplied text in a low, resonant British voice. "
            "Do not add words."}]},
    }}
    turn = {"clientContent": {"turns": [{"role": "user", "parts": [
        {"text": "Hello."}]}], "turnComplete": True}}
    started = time.perf_counter()
    first_audio: float | None = None
    turn_started: float | None = None
    audio_bytes = 0
    complete = False
    try:
        async with websockets.connect(url, open_timeout=10, ping_interval=15,
                                      max_size=2_097_152) as socket:
            await socket.send(json.dumps(setup))
            ready = False
            while time.perf_counter() - started < 15:
                event = json.loads(await asyncio.wait_for(socket.recv(), 10))
                if "error" in event:
                    return {"ok": False, "error": "Live setup refused"}
                if "setupComplete" in event:
                    ready = True
                    break
            if not ready:
                return {"ok": False, "error": "Live setup timed out"}
            turn_started = time.perf_counter()
            await socket.send(json.dumps(turn))
            deadline = time.perf_counter() + 35
            while time.perf_counter() < deadline:
                event = json.loads(await asyncio.wait_for(socket.recv(), 15))
                if "error" in event or "goAway" in event:
                    return {"ok": False, "error": "Live turn interrupted",
                            "audio_bytes": audio_bytes}
                content = event.get("serverContent", {})
                for part in content.get("modelTurn", {}).get("parts", []):
                    inline = part.get("inlineData", {})
                    if str(inline.get("mimeType", "")).lower().startswith(
                            ("audio/pcm", "audio/l16")) and inline.get("data"):
                        chunk = base64.b64decode(inline["data"], validate=True)
                        audio_bytes += len(chunk)
                        if first_audio is None:
                            first_audio = round(time.perf_counter() - started, 3)
                if content.get("turnComplete"):
                    complete = True
                    break
    except Exception as exc:
        # WebSocket exception text can include the URL and its credential.
        return {"ok": False, "error": type(exc).__name__,
                "audio_bytes": audio_bytes}
    return {"ok": complete and audio_bytes > 0, "model": model,
            "setup_seconds": round(turn_started - started, 3) if turn_started else None,
            "first_audio_seconds": first_audio,
            "first_audio_from_turn_seconds": round(first_audio - (turn_started - started), 3)
            if first_audio is not None and turn_started else None,
            "audio_bytes": audio_bytes, "complete": complete}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="gemini-3.8-live")
    result = asyncio.run(verify(parser.parse_args().model))
    print(json.dumps(result))
    raise SystemExit(0 if result["ok"] else 1)
