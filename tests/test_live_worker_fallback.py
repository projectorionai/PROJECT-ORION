"""Regression coverage for live-session hand-off to text providers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.live_worker import GenAILiveWorker


class _Session:
    async def send_client_content(self, **_kwargs):
        raise RuntimeError("transport closed")


def test_rejected_live_turn_is_cleared_before_text_fallback():
    """The watchdog must not re-answer a request already handed off locally."""
    worker = SimpleNamespace(
        session=_Session(),
        _send_closed=False,
        stop_event=SimpleNamespace(is_set=lambda: False),
        _send_lock=asyncio.Lock(),
        _turn_active=True,
        _pending_turn="check the lights",
        bus=SimpleNamespace(log=SimpleNamespace(emit=lambda *_args: None)),
    )
    cleared = []
    fallbacks = []
    worker._transport_ready = lambda: True
    worker._clear_turn = lambda: cleared.append(True)

    async def fallback(text, reason=""):
        fallbacks.append((text, reason))

    worker._submit_text_fallback = fallback

    asyncio.run(GenAILiveWorker._send_text_turn(worker, "check the lights"))

    assert cleared == [True]
    assert fallbacks == [("check the lights", "Native provider rejected manual command")]
