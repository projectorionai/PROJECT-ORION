"""
Request-size limits on the phone gateway's JSON endpoints.

The gateway raises aiohttp's body ceiling to 8 MB so a recorded voice clip can
reach /api/transcribe. Without a separate JSON limit, every JSON endpoint —
including the unauthenticated pairing and token doors — would parse a body of
that size. These tests pin the 64 KB JSON ceiling, both when the client
declares its length and when it streams a chunked body with no length.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.remote import RemoteGateway


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Memory:
    def log_episode(self, *a):
        pass

    def prompt_context(self, limit=18):
        return ""


def _gateway(tmp_path):
    return RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)


async def _client(gateway):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(gateway._build_app()))
    await client.start_server()
    return client


def _oversized_json() -> bytes:
    pad = "x" * (RemoteGateway._MAX_JSON_BYTES + 1)
    return ('{"code": "%s"}' % pad).encode()


@pytest.mark.parametrize("path", [
    "/v1/auth/pair", "/v1/auth/token", "/api/chat", "/api/confirm", "/v1/agent/tasks",
])
def test_oversized_json_body_is_refused_with_413(tmp_path, path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            r = await client.post(path, data=_oversized_json(),
                                  headers={"Content-Type": "application/json"})
            assert r.status == 413
            assert (await r.json())["error"] == "request body too large"
        finally:
            await client.close()

    asyncio.run(flow())


def test_chunked_body_without_length_is_still_capped(tmp_path):
    """A client that omits Content-Length cannot slip past the limit."""
    async def body():
        data = _oversized_json()
        for i in range(0, len(data), 4096):
            yield data[i:i + 4096]

    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            r = await client.post("/v1/auth/pair", data=body(),
                                  headers={"Content-Type": "application/json"})
            assert r.status == 413
        finally:
            await client.close()

    asyncio.run(flow())


def test_body_at_the_limit_is_parsed(tmp_path):
    """The limit is inclusive: a body of exactly the ceiling is read normally
    (and here reaches the pairing check, which rejects the bogus code)."""
    async def flow():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        prefix, suffix = b'{"code": "', b'"}'
        pad = RemoteGateway._MAX_JSON_BYTES - len(prefix) - len(suffix)
        data = prefix + b"x" * pad + suffix
        assert len(data) == RemoteGateway._MAX_JSON_BYTES
        try:
            r = await client.post("/v1/auth/pair", data=data,
                                  headers={"Content-Type": "application/json"})
            assert r.status == 401
        finally:
            await client.close()

    asyncio.run(flow())


@pytest.mark.parametrize("data", [b"[1, 2]", b"not json", b"\xff\xfe", b""])
def test_non_object_or_malformed_body_is_a_400(tmp_path, data):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            r = await client.post("/v1/auth/pair", data=data,
                                  headers={"Content-Type": "application/json"})
            assert r.status == 400
            assert (await r.json())["error"] == "invalid JSON"
        finally:
            await client.close()

    asyncio.run(flow())


def test_pairing_still_works_with_a_normal_body(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            code = gateway.begin_pairing()
            r = await client.post("/v1/auth/pair", json={"code": code, "device_name": "phone"})
            assert r.status == 200
            creds = await r.json()
            r = await client.post("/v1/auth/token", json={
                "device_id": creds["device_id"],
                "refresh_token": creds["refresh_token"]})
            assert r.status == 200
            assert (await r.json())["access_token"]
        finally:
            await client.close()

    asyncio.run(flow())
