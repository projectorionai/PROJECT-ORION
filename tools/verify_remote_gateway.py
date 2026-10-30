"""Host-local gateway probe; credentials and device identities never leave output.

The PC already owns its gateway signing key. This read-only probe signs a
one-minute access token for an existing active pairing, then exercises speech
and telemetry through the configured HTTPS route. It creates no new pairing.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
from pathlib import Path
import struct
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def local_probe_token() -> str:
    from orion_core.remote import RemoteAuthManager
    manager = RemoteAuthManager(config_dir=ROOT / "config")
    devices = [device for device in manager.list_devices() if not device["revoked"]]
    if not devices:
        raise RuntimeError("The gateway has no active pairing")
    payload = f"{devices[0]['device_id']}:{int(time.time()) + 60}".encode()
    encoded = base64.urlsafe_b64encode(payload).decode("ascii")
    signature = hmac.new(manager._secret, payload, hashlib.sha256).hexdigest()
    token = f"{encoded}.{signature}"
    if manager.verify_access(token) is None:
        raise RuntimeError("Local probe authorisation failed")
    return token


async def verify(url: str) -> dict:
    import aiohttp
    parsed = urlsplit(url)
    local = parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port == 8000
    if parsed.username or parsed.password or (not local and (
            parsed.scheme != "https" or parsed.port not in (None, 443))):
        raise ValueError("Use the configured HTTPS gateway on port 443")
    result = {"route": url.rstrip("/"), "ok": False}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60, connect=10),
                                    cookie_jar=aiohttp.DummyCookieJar()) as client:
        async with client.get(url.rstrip("/") + "/api/health", allow_redirects=False) as response:
            health = await response.json()
            result["health_http"] = response.status
            if response.status != 200 or not health.get("ok") or not health.get("brain"):
                return result
        token = local_probe_token()
        headers = {"Authorization": "Bearer " + token}
        async with client.get(url.rstrip("/") + "/api/hardware", headers=headers,
                              allow_redirects=False) as response:
            hardware = await response.json()
            result["hardware_http"] = response.status
            result["hardware"] = hardware if response.status == 200 else {"available": False}
        started = time.perf_counter()
        async with client.post(url.rstrip("/") + "/api/speech/stream", headers=headers,
                               json={"text": "Hello. ORION is ready on your phone."},
                               allow_redirects=False) as response:
            result["speech_http"] = response.status
            if response.status != 200 or response.headers.get("X-Orion-PCM-Format") != "s16le-mono-v1":
                return result
            result["voice"] = response.headers.get("X-Orion-Voice")
            total = frames = 0
            while True:
                size = struct.unpack("!I", await response.content.readexactly(4))[0]
                if size == 0:
                    break
                if size < 2 or size > 65536 or size % 2 or total + size > 8_640_000:
                    raise RuntimeError("Invalid PCM frame")
                await response.content.readexactly(size)
                if frames == 0:
                    result["first_audio_seconds"] = round(time.perf_counter() - started, 3)
                frames += 1
                total += size
            if await response.content.read() or total == 0:
                raise RuntimeError("Invalid speech completion")
            result.update(speech_stream_seconds=round(time.perf_counter() - started, 3),
                          audio_seconds=round(total / 48000, 3), frames=frames,
                          complete=True, ok=result["hardware_http"] == 200)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = asyncio.run(verify(args.url))
    except Exception as exc:
        # Never serialise arbitrary exceptions containing an HTTP request.
        result = {"ok": False, "error_type": type(exc).__name__}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
