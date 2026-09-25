"""
Tests for the self-improvement heartbeat giving up on a tool idea that keeps
failing forge verification, instead of re-proposing it every tick forever.

Real incident this responds to: a user's log showed 'synthesis_enhancer' and
'emotion_recognizer' failing forge verification with contract_violation
repeatedly, and the heartbeat's own analysis text NOTICED the recurrence
("a recurring issue... repeated failures to forge the 'synthesis_enhancer'
and 'emotion_recognizer' tools") without ever acting on it — nothing hard-
blocked re-proposing the same doomed idea tick after tick, so a session's
limited forge attempts (MAX_AUTO_FORGES_PER_SESSION) could all get burned on
the same abstract, hard-to-specify concept instead of trying something new.
The fix tallies failures per tool NAME from the persisted journal (so it
survives a restart) and skips re-attempting a name once it crosses
MAX_ATTEMPTS_PER_TOOL_NAME, telling the model plainly in the next prompt.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.forge import ForgeOrchestrationManager, ImprovementHeartbeat


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubForge:
    sessions: dict = {}


class _NoActionRouter:
    def text_available(self) -> bool:
        return True

    async def generate_text(self, prompt, system_extra=""):
        return object(), json.dumps({"analysis": "nothing new to propose", "actions": []})


class _ProposeRouter:
    """Always proposes forging one specific tool name."""

    def __init__(self, tool_name: str, plan: str = "a precise capability plan for this idea") -> None:
        self.tool_name = tool_name
        self.plan = plan

    def text_available(self) -> bool:
        return True

    async def generate_text(self, prompt, system_extra=""):
        payload = {
            "analysis": "proposing a tool for a recurring gap",
            "actions": [{"type": "forge_tool", "tool_name": self.tool_name,
                        "plan": self.plan, "reason": "recurring gap"}],
        }
        return object(), json.dumps(payload)


def _seed_journal(path: Path, tool_name: str, failures: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"at": "t", "analysis": "", "actions": [],
                   "forged": [{"tool": tool_name, "ok": False, "detail": "boom"}]})
        for _ in range(failures)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _forge_with_spy() -> tuple[ForgeOrchestrationManager, list]:
    forge = ForgeOrchestrationManager(_StubBus())
    calls: list[tuple[str, str]] = []

    async def _spy(name, plan, **_kwargs):
        calls.append((name, plan))
        return ToolResult(f"forged {name}", ok=True)

    forge.forge_tool = _spy
    return forge, calls


def _patch_dirs(monkeypatch, tmp_path: Path) -> Path:
    import orion_core.forge as forge_mod
    monkeypatch.setattr(forge_mod, "SELF_IMPROVE_DIR", tmp_path)
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(forge_mod, "IMPROVE_JOURNAL", journal)
    return journal


# ── _failed_attempt_counts() ─────────────────────────────────────────────────

def test_failed_attempt_counts_tallies_ok_false_entries(tmp_path, monkeypatch):
    journal = _patch_dirs(monkeypatch, tmp_path)
    _seed_journal(journal, "synthesis_enhancer", 3)
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"forged": [{"tool": "synthesis_enhancer", "ok": True}]}) + "\n")

    hb = ImprovementHeartbeat(_StubBus(), router=None, forge=_StubForge())
    counts = hb._failed_attempt_counts()

    assert counts["synthesis_enhancer"] == 3   # the ok:true entry does not count


def test_failed_attempt_counts_is_empty_with_no_journal(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    hb = ImprovementHeartbeat(_StubBus(), router=None, forge=_StubForge())
    assert hb._failed_attempt_counts() == {}


def test_failed_attempt_counts_skips_malformed_lines(tmp_path, monkeypatch):
    journal = _patch_dirs(monkeypatch, tmp_path)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        "not json\n" + json.dumps({"forged": [{"tool": "x", "ok": False}]}) + "\n",
        encoding="utf-8")

    hb = ImprovementHeartbeat(_StubBus(), router=None, forge=_StubForge())
    assert hb._failed_attempt_counts() == {"x": 1}


# ── _compose_context(abandoned) ──────────────────────────────────────────────

def test_compose_context_includes_abandoned_tool_ideas_section(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    hb = ImprovementHeartbeat(_StubBus(), router=None, forge=_StubForge())
    context = hb._compose_context({"synthesis_enhancer", "emotion_recognizer"})
    assert "ABANDONED TOOL IDEAS" in context
    assert "synthesis_enhancer" in context
    assert "emotion_recognizer" in context


def test_compose_context_omits_abandoned_section_when_none_given(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    hb = ImprovementHeartbeat(_StubBus(), router=None, forge=_StubForge())
    assert "ABANDONED TOOL IDEAS" not in hb._compose_context()


# ── tick(): the actual skip behaviour ────────────────────────────────────────

def test_tick_skips_a_tool_that_has_already_failed_the_max_times(tmp_path, monkeypatch):
    journal = _patch_dirs(monkeypatch, tmp_path)
    _seed_journal(journal, "synthesis_enhancer", ImprovementHeartbeat.MAX_ATTEMPTS_PER_TOOL_NAME)
    forge, calls = _forge_with_spy()
    heartbeat = ImprovementHeartbeat(_StubBus(), _ProposeRouter("synthesis_enhancer"), forge)

    entry = asyncio.run(heartbeat.tick())

    assert calls == []   # never re-attempted
    skipped = [f for f in entry["forged"] if f.get("skipped") == "repeated failures"]
    assert skipped and skipped[0]["tool"] == "synthesis_enhancer"
    assert skipped[0]["previous_failures"] == ImprovementHeartbeat.MAX_ATTEMPTS_PER_TOOL_NAME


def test_tick_still_allows_a_tool_below_the_failure_threshold(tmp_path, monkeypatch):
    journal = _patch_dirs(monkeypatch, tmp_path)
    _seed_journal(journal, "synthesis_enhancer", ImprovementHeartbeat.MAX_ATTEMPTS_PER_TOOL_NAME - 1)
    forge, calls = _forge_with_spy()
    heartbeat = ImprovementHeartbeat(_StubBus(), _ProposeRouter("synthesis_enhancer"), forge)

    asyncio.run(heartbeat.tick())

    assert calls == [("synthesis_enhancer", "a precise capability plan for this idea")]


def test_tick_skips_a_queued_retry_for_an_abandoned_tool_name(tmp_path, monkeypatch):
    journal = _patch_dirs(monkeypatch, tmp_path)
    _seed_journal(journal, "emotion_recognizer", ImprovementHeartbeat.MAX_ATTEMPTS_PER_TOOL_NAME)
    forge, calls = _forge_with_spy()
    forge.pending_retries.append(("emotion_recognizer", "recognise emotion from text"))
    heartbeat = ImprovementHeartbeat(_StubBus(), _NoActionRouter(), forge)

    entry = asyncio.run(heartbeat.tick())

    assert calls == []
    assert forge.pending_retries == []   # still drained from the queue
    skipped = [f for f in entry["forged"] if f.get("skipped") == "repeated failures"]
    assert skipped and skipped[0]["tool"] == "emotion_recognizer"


def test_tick_feeds_the_abandoned_list_into_the_next_prompt_context(tmp_path, monkeypatch):
    journal = _patch_dirs(monkeypatch, tmp_path)
    _seed_journal(journal, "synthesis_enhancer", ImprovementHeartbeat.MAX_ATTEMPTS_PER_TOOL_NAME)
    forge, _calls = _forge_with_spy()

    captured: dict[str, str] = {}

    class _CapturingRouter:
        def text_available(self) -> bool:
            return True

        async def generate_text(self, prompt, system_extra=""):
            captured["prompt"] = prompt
            return object(), json.dumps({"analysis": "ok", "actions": []})

    heartbeat = ImprovementHeartbeat(_StubBus(), _CapturingRouter(), forge)
    asyncio.run(heartbeat.tick())

    assert "synthesis_enhancer" in captured["prompt"]
    assert "ABANDONED TOOL IDEAS" in captured["prompt"]
