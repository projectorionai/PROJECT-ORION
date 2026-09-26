"""
System test: a real headless ORION node, a stand-in model server, a paired phone.

Boots ``orion.py --headless`` as its own process from a temporary copy of the
source (never the working tree: a node writes config/, and only the public
knowledge packs are copied, never private settings), points it at a local
OpenAI-compatible stand-in that fails its first request with HTTP 503, then:

* reads the one-time pairing code from the node's console output;
* pairs, takes an access token and holds a chat turn, which must be answered
  despite the 503 (one overload blip must not fail the turn);
* checks request validation (a malformed field is 400, an oversized body 413);
* stops the node and, where signals allow a clean exit, checks it shut down.

This is the path that once printed nothing - so a fresh node could not be
paired - and once answered a single 503 with "no language model is reachable".
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ModelServer:
    """An OpenAI-compatible chat endpoint that fails its first request."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.calls = 0
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        from aiohttp import web

        async def chat(request):
            body = await request.json()
            self.calls += 1
            if self.calls == 1:
                return web.Response(status=503, text="overloaded")
            said = str(body["messages"][-1]["content"])[-40:]
            return web.json_response({"id": f"r{self.calls}", "choices": [
                {"message": {"content": f"Stand-in model heard: {said}"}}]})

        async def main():
            app = web.Application()
            app.router.add_post("/v1/chat/completions", chat)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", self.port).start()
            self._stop = asyncio.Event()
            self._ready.set()
            await self._stop.wait()
            await runner.cleanup()

        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(main())

    def __enter__(self) -> "_ModelServer":
        self._thread.start()
        assert self._ready.wait(10), "the stand-in model did not start"
        return self

    def __exit__(self, *exc) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)


def _source_copy(tmp_path: Path) -> Path:
    run = tmp_path / "orion"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "orion_core", run / "orion_core", ignore=ignore)
    shutil.copy2(ROOT / "orion.py", run / "orion.py")
    packs = ROOT / "config" / "knowledge_packs"
    if packs.is_dir():
        shutil.copytree(packs, run / "config" / "knowledge_packs", ignore=ignore)
    return run


def _configure(run: Path, model_port: int) -> None:
    subprocess.run([sys.executable, "-c", (
        "from orion_core.providers import _default_provider_payload, "
        "_settings_from_payload, write_provider_settings\n"
        "p = _default_provider_payload('')\n"
        "p['providers']['groq'].update(enabled=True, "
        f"base_url='http://127.0.0.1:{model_port}/v1', api_key='stand-in-key', "
        "model='stand-in')\n"
        "p['active_provider'] = 'groq'; p['provider_order'] = ['groq']\n"
        "write_provider_settings(_settings_from_payload(p))\n")],
        cwd=run, check=True, timeout=120)


def test_a_headless_node_pairs_a_phone_and_answers_through_a_blip(tmp_path):
    pytest.importorskip("aiohttp")
    pytest.importorskip("qasync")
    run = _source_copy(tmp_path)
    node_port = _free_port()
    with _ModelServer() as model:
        _configure(run, model.port)
        env = dict(os.environ, ORION_REMOTE_PORT=str(node_port),
                   ORION_REMOTE_HOST="127.0.0.1", QT_QPA_PLATFORM="offscreen",
                   PYTHONUNBUFFERED="1")
        node = subprocess.Popen([sys.executable, "orion.py", "--headless"], cwd=run,
                                env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        lines: list[str] = []
        code = None
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                line = node.stdout.readline()
                if not line:
                    break
                lines.append(line.rstrip())
                found = re.search(r"pairing code ([A-F0-9]{4}-[A-F0-9]{4})", line)
                code = found.group(1) if found else code
                if "awaiting remote turns" in line:
                    break
            assert code, "no pairing code in the node's output:\n" + "\n".join(lines[-20:])
            reader = threading.Thread(
                target=lambda: lines.extend(l.rstrip() for l in node.stdout), daemon=True)
            reader.start()

            async def phone():
                from aiohttp import ClientSession
                base = f"http://127.0.0.1:{node_port}"
                async with ClientSession() as s:
                    r = await s.post(f"{base}/v1/auth/pair", json={"code": code})
                    creds = await r.json()
                    assert r.status == 200, creds
                    r = await s.post(f"{base}/v1/auth/token", json={
                        "device_id": creds["device_id"],
                        "refresh_token": creds["refresh_token"]})
                    auth = {"Authorization": f"Bearer {(await r.json())['access_token']}"}
                    r = await s.post(f"{base}/api/chat", json={"message": "hello there"},
                                     headers=auth)
                    reply = await r.json()
                    assert r.status == 200 and "Stand-in model heard" in reply.get("reply", ""), reply
                    r = await s.post(f"{base}/api/chat", json={"message": {"x": 1}},
                                     headers=auth)
                    assert r.status == 400
                    r = await s.post(f"{base}/api/chat", data=b"{" + b"x" * 70000,
                                     headers={**auth, "Content-Type": "application/json"})
                    assert r.status == 413

            asyncio.run(phone())
            assert model.calls == 2, "the blip should be retried exactly once"
        finally:
            if os.name == "nt":
                node.terminate()                 # no SIGTERM handler to exercise
            else:
                node.send_signal(signal.SIGTERM)
            try:
                code_on_exit = node.wait(timeout=30)
            except subprocess.TimeoutExpired:
                node.kill()
                code_on_exit = None
        if os.name != "nt":
            time.sleep(0.3)
            assert code_on_exit == 0, "\n".join(lines[-20:])
            assert any("shutting down headless node" in l for l in lines)
        assert not [l for l in lines if "Traceback" in l], "\n".join(lines[-40:])
