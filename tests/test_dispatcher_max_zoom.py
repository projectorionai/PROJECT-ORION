"""
Integration test: ``max_zoom_in_globe`` is an explicit, deterministic tool
command (Section 1) — it routes through the dispatcher's handler table and
drives the globe over the bus WITHOUT consulting any text provider.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatcher import TOOL_DECLARATIONS, OrionDispatcher


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload[0] if len(payload) == 1 else payload))


class _StubBus:
    def __init__(self):
        self.emitted: list = []
        self.globe_max_zoom = _Signal(self.emitted, "globe_max_zoom")

    def __getattr__(self, name):
        # Any other signal touched becomes a recording no-op signal.
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


def _bare_dispatcher(bus):
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = bus
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    # A sentinel that fails loudly if the deterministic path ever reaches a
    # provider/router — max_zoom must never depend on text generation.
    class _Boom:
        def __getattr__(self, _n):
            raise AssertionError("deterministic tool touched a text provider")
    d.router = _Boom()
    return d


def test_schema_and_handler_registered():
    names = [d["name"] for d in TOOL_DECLARATIONS]
    assert "max_zoom_in_globe" in names
    assert hasattr(OrionDispatcher, "max_zoom_in_globe_tool")


def test_dispatch_emits_over_bus_without_provider():
    bus = _StubBus()
    disp = _bare_dispatcher(bus)
    result = asyncio.run(disp.dispatch("max_zoom_in_globe", {"max_seconds": 5, "tolerance": 0.02}))
    assert result.ok
    assert ("globe_max_zoom", {"max_seconds": 5, "tolerance": 0.02}) in bus.emitted


def test_dispatch_filters_unknown_options():
    bus = _StubBus()
    disp = _bare_dispatcher(bus)
    asyncio.run(disp.dispatch("max_zoom_in_globe", {"nonsense": 1, "max_attempts": 12}))
    payloads = [p for (n, p) in bus.emitted if n == "globe_max_zoom"]
    assert payloads and payloads[0] == {"max_attempts": 12}
