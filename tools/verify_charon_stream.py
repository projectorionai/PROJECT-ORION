"""Measure approved Gemini speech without writing credentials or reply text."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import sys
import time
import wave
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def measure(model: str) -> dict:
    from orion_core.providers import read_provider_settings
    from orion_core.provider_transport import SessionPool
    from orion_core.remote_speech import stream_charon, synthesise_charon

    os.environ["ORION_REMOTE_TTS_MODEL"] = model
    pool = SessionPool()
    router = SimpleNamespace(settings=read_provider_settings(), _pool=pool)
    message = "Hello. I am ready to help. What would you like to do next?"
    result = {"model": model}
    try:
        started = time.perf_counter()
        pcm = bytearray()
        chunks = 0
        async for chunk in stream_charon(router, message):
            if chunks == 0:
                result["first_audio_seconds"] = round(time.perf_counter() - started, 3)
            chunks += 1
            pcm.extend(chunk)
        result.update(stream_seconds=round(time.perf_counter() - started, 3),
                      audio_seconds=round(len(pcm) / 48000, 3), chunks=chunks,
                      words_per_minute=round(len(message.split()) * 60 / (len(pcm) / 48000)))
        started = time.perf_counter()
        wav = await synthesise_charon(router, message)
        result["complete_wav_seconds"] = round(time.perf_counter() - started, 3)
        with wave.open(io.BytesIO(wav), "rb") as audio:
            result["wav_audio_seconds"] = round(audio.getnframes() / audio.getframerate(), 3)
        result["ok"] = True
    except Exception as exc:
        # SpeechUnavailable is deliberately credential-free. Never print a
        # provider exception or a response body if another failure occurs.
        from orion_core.remote_speech import SpeechUnavailable
        result.update(ok=False, error=str(exc) if isinstance(exc, SpeechUnavailable)
                      else type(exc).__name__)
    finally:
        await pool.aclose()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.8-flash-tts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(measure(args.model))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
