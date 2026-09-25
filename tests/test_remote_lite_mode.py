"""
Tests for mobile-data lite mode (#12): the heavy CDN-backed voxel face is
dropped for a self-contained CSS orb on metered connections, and text/HTML is
gzip-compressed when the client accepts it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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


def _gateway(tmp_path):
    return RemoteGateway(object(), object(), _Bus(), config_dir=tmp_path)


async def _client(gateway, **kw):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(gateway._build_app()), **kw)
    await client.start_server()
    return client


def test_full_page_has_the_heavy_face(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            body = await (await client.get("/")).text()
            assert 'src="/face"' in body
        finally:
            await client.close()
    asyncio.run(flow())


def test_lite_query_drops_face_for_css_orb(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            body = await (await client.get("/?lite=1")).text()
            assert 'src="/face"' not in body      # no CDN-backed WebGL face
            assert "orionlite" in body            # ...replaced by the CSS orb
            assert 'id="face"' in body            # keeps the id so faceMsg no-ops
        finally:
            await client.close()
    asyncio.run(flow())


def test_save_data_header_triggers_lite(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            resp = await client.get("/", headers={"Save-Data": "on"})
            body = await resp.text()
            assert 'src="/face"' not in body
        finally:
            await client.close()
    asyncio.run(flow())


def test_gzip_offered_when_accepted(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path), auto_decompress=False)
        try:
            resp = await client.get("/", headers={"Accept-Encoding": "gzip"})
            assert resp.headers.get("Content-Encoding") == "gzip"
        finally:
            await client.close()
    asyncio.run(flow())
