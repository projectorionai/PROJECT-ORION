"""
Tests for the forge auto-retry-on-provider-outage fix.

Real incident this responds to: a forge session died because every text
provider was simultaneously unavailable (rate-limited, out of credit, local
model not running) — not because the generated code was wrong. Nothing
retried it, the incident message got truncated mid-word by Incident.summary()
(120-char limit), and when the user later asked "have you fixed it yet?" the
model invented "I am continuing to diagnose" on top of a session that had
already silently ended. Two independent fixes:

    1. A ProviderError-caused failure queues (tool_name, tool_plan) onto
       ForgeOrchestrationManager.pending_retries; ImprovementHeartbeat.tick()
       drains and retries it automatically, capped by MAX_AUTO_FORGES_PER_SESSION.
    2. The recorded incident message puts the retry verdict FIRST so it
       survives Incident.summary()'s 120-char truncation, giving the model
       accurate ground truth instead of inventing "still diagnosing".
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.forge import ForgeOrchestrationManager, ImprovementHeartbeat
from orion_core.providers import AllTextProvidersFailedError
from orion_core.selfrepair import Incident


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _RaisingCodeGenerator:
    """Simulates every provider being unavailable during generation."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def __call__(self, name, plan):
        self.calls += 1
        raise self.exc


class _RecordingRepair:
    def __init__(self) -> None:
        self.incidents: list[tuple[str, str]] = []

    def record_incident(self, error_type, message, module="", **kw):
        self.incidents.append((error_type, message))


def _outage_forge() -> tuple[ForgeOrchestrationManager, _RecordingRepair]:
    exc = AllTextProvidersFailedError("Server disconnected", attempts=["groq", "openrouter"])
    forge = ForgeOrchestrationManager(_StubBus(), code_generator=_RaisingCodeGenerator(exc))
    repair = _RecordingRepair()
    forge.repair = repair
    return forge, repair


# ── forge_tool queues a retry on a provider-outage failure ──────────────────

def test_provider_outage_failure_queues_a_retry():
    forge, _repair = _outage_forge()
    result = asyncio.run(forge.forge_tool("enhanced_context_recall", "recall more context"))
    assert not result.ok
    assert forge.pending_retries == [("enhanced_context_recall", "recall more context")]


def test_provider_outage_failure_result_text_says_retry_is_scheduled():
    forge, _repair = _outage_forge()
    result = asyncio.run(forge.forge_tool("enhanced_context_recall", "recall more context"))
    assert "scheduled" in result.text.lower()


def test_non_provider_failure_does_not_queue_a_retry():
    forge = ForgeOrchestrationManager(
        _StubBus(),
        code_generator=_RaisingCodeGenerator(ValueError("plan was nonsense")),
    )
    forge.repair = _RecordingRepair()
    result = asyncio.run(forge.forge_tool("bad_plan_tool", "do something incoherent"))
    assert not result.ok
    assert forge.pending_retries == []
    assert "no retry scheduled" in result.text.lower() or "none scheduled" in result.text.lower()


# ── the incident message survives Incident.summary()'s 120-char truncation ──

def test_incident_message_puts_retry_verdict_within_truncation_window():
    forge, repair = _outage_forge()
    asyncio.run(forge.forge_tool("enhanced_context_recall", "recall more context"))
    assert len(repair.incidents) == 1
    error_type, message = repair.incidents[0]
    assert error_type == "ForgeSessionFailure"
    incident = Incident(id="inc-1", at="now", error_type=error_type, message=message)
    assert "retry scheduled" in incident.summary()   # survives the [:120] truncation


def test_incident_message_for_a_dead_end_failure_says_no_retry_within_window():
    forge = ForgeOrchestrationManager(
        _StubBus(),
        code_generator=_RaisingCodeGenerator(ValueError("plan was nonsense")),
    )
    repair = _RecordingRepair()
    forge.repair = repair
    asyncio.run(forge.forge_tool("bad_plan_tool", "do something incoherent"))
    error_type, message = repair.incidents[0]
    incident = Incident(id="inc-1", at="now", error_type=error_type, message=message)
    assert "no retry scheduled" in incident.summary()


# ── ImprovementHeartbeat.tick() drains pending_retries ──────────────────────

class _NoteRouter:
    def text_available(self):
        return True

    async def generate_text(self, prompt, system_extra=""):
        import json
        return object(), json.dumps({"analysis": "nothing new to propose", "actions": []})


def test_heartbeat_tick_retries_a_queued_session(tmp_path, monkeypatch):
    import orion_core.forge as forge_mod
    monkeypatch.setattr(forge_mod, "IMPROVE_JOURNAL", tmp_path / "journal.jsonl")

    forge = ForgeOrchestrationManager(_StubBus())   # no code_generator: forge_tool always fails cleanly
    forge.pending_retries.append(("enhanced_context_recall", "recall more context"))
    heartbeat = ImprovementHeartbeat(_StubBus(), _NoteRouter(), forge)

    entry = asyncio.run(heartbeat.tick())
    assert forge.pending_retries == []            # drained
    forged = [f for f in entry["forged"] if f.get("tool") == "enhanced_context_recall"]
    assert len(forged) == 1
    assert forged[0]["retry"] is True


def test_heartbeat_tick_respects_the_session_forge_cap_for_retries(tmp_path, monkeypatch):
    import orion_core.forge as forge_mod
    monkeypatch.setattr(forge_mod, "IMPROVE_JOURNAL", tmp_path / "journal.jsonl")

    forge = ForgeOrchestrationManager(_StubBus())
    forge.pending_retries.append(("tool_a", "plan a"))
    heartbeat = ImprovementHeartbeat(_StubBus(), _NoteRouter(), forge)
    heartbeat._forged_this_session = heartbeat.MAX_AUTO_FORGES_PER_SESSION

    entry = asyncio.run(heartbeat.tick())
    assert forge.pending_retries == []             # still drained (removed from the queue)
    skipped = [f for f in entry["forged"] if f.get("skipped")]
    assert any(s.get("tool") == "tool_a" for s in skipped)
