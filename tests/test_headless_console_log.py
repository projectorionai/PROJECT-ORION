"""
The headless node prints its log to stdout.

Nothing else listens to bus.log without the desktop window, so before this
every line - including the first-run pairing code - was discarded and a fresh
cloud node could never be paired.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.bus import OrionBus
from orion_core.remote import RemoteGateway
from orion_core.server import attach_console_log


class _Memory:
    def log_episode(self, *a):
        pass

    def prompt_context(self, limit=18):
        return ""


def test_log_lines_reach_the_stream_timestamped():
    bus = OrionBus()
    out = io.StringIO()
    attach_console_log(bus, out)
    bus.log.emit("SERVER: headless node ready")
    line = out.getvalue().strip()
    assert line.endswith("SERVER: headless node ready")
    assert line[:4].isdigit() and line[4] == "-"          # 2026-09-25 12:00:00 …


def test_a_fresh_nodes_pairing_code_is_printed(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_PORT", "0")
    bus = OrionBus()
    out = io.StringIO()
    attach_console_log(bus, out)
    gateway = RemoteGateway(object(), _Memory(), bus, config_dir=tmp_path)
    monkeypatch.setattr(gateway, "port", 0, raising=False)
    monkeypatch.setattr(gateway, "host", "127.0.0.1", raising=False)

    async def flow():
        await gateway.start()
        await gateway.stop()

    asyncio.run(flow())
    assert "enter pairing code" in out.getvalue()


def test_a_broken_stream_never_raises_into_the_emitter():
    class _Broken:
        def write(self, *_a):
            raise OSError("stdout closed")

        def flush(self):
            raise OSError("stdout closed")

    bus = OrionBus()
    attach_console_log(bus, _Broken())
    bus.log.emit("still fine")
