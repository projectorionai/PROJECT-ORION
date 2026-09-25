"""
Tests for the "ORION always says 'day' instead of morning/afternoon/evening"
fix. The deterministic greeting-composition code (TemporalPresence, the
STARTUP_GREETINGS fallback, local_brain's _intent_greeting) always computed
the correct period — the bug was that offer_startup_briefing()'s own
instruction to the live model said "incorporating" (paraphrase-friendly),
unlike the stricter "word for word" instruction the later briefing-delivery
path already used, so the model had licence to drop the specific word.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.identity as idm


class _Sig:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, n):
        return _Sig()


class _FakeBriefing:
    def already_briefed_today(self):
        return False


class _StopEvent:
    def is_set(self):
        return False


async def _async_return(value):
    return value


def _worker(sent: list[str]) -> types.SimpleNamespace:
    from orion_core.live_worker import GenAILiveWorker

    worker = types.SimpleNamespace()
    worker.dispatcher = types.SimpleNamespace(briefing=_FakeBriefing())
    worker.bus = _Bus()
    worker.connected = True
    worker.stop_event = _StopEvent()
    worker.awaiting_briefing = False
    worker._greeted_at = 0.0
    worker._refresh_wake_window = lambda: None
    worker._compose_greeting = lambda: _async_return("Good evening, sir.")
    worker._say = lambda text: sent.append(text)
    worker._emit_state = lambda state: None
    return worker


def test_offer_startup_briefing_instructs_word_for_word_delivery_on_live_channel():
    from orion_core.live_worker import GenAILiveWorker

    captured: list[str] = []
    sent: list[str] = []
    worker = _worker(sent)
    worker.session = object()   # live channel active — takes the instruction-text branch

    async def _fake_send_text_turn(instruction):
        captured.append(instruction)

    worker._send_text_turn = _fake_send_text_turn
    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    assert len(captured) == 1
    instruction = captured[0]
    assert "word for word" in instruction
    assert "Good evening, sir." in instruction
    assert "'day'" in instruction   # explicitly forbids the vague substitute


def test_persona_text_carries_the_time_of_day_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    persona = idm.IdentityManager(_Bus()).persona_text()
    assert "TIME-OF-DAY RULE" in persona
    assert "morning" in persona.lower() and "afternoon" in persona.lower() and "evening" in persona.lower()
    assert "'day'" in persona or '"day"' in persona
