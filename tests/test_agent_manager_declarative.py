"""
Tests for AgentManager's declarative-agent wiring (Mark XXI, Track E3):
register_declarative() and load_declarative_agents(), and that
AgentManager.__init__ loads config/agents/*.json automatically without
ever letting a manifest shadow a built-in specialist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import asyncio
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import agent_registry as ar
from orion_core.agents import AgentManager, AgentToolbelt, BaseAgent
from orion_core.data import ToolResult


class _Signal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, text, *_extra) -> None:
        self.messages.append(text)


class _StubBus:
    def __init__(self) -> None:
        self.log = _Signal()

    def __getattr__(self, name):
        return _Signal()


class _StubRouter:
    def has_text_fallback(self):
        return False


def _write_manifest(tmp_path: Path, name: str) -> None:
    (tmp_path / f"{name}.json").write_text(json.dumps({
        "name": name,
        "title": f"{name.title()} Agent",
        "expertise": "a declarative specialist",
        "primary": [r"\b" + name + r"\b"],
        "keywords": [],
        "persona": f"You are the {name} specialist.",
    }), encoding="utf-8")


def _manager(bus=None) -> AgentManager:
    return AgentManager(_StubRouter(), bus or _StubBus())


# ── register_declarative() ──────────────────────────────────────────────────

def test_register_declarative_adds_a_new_agent():
    mgr = _manager()
    manifest = {
        "name": "legal", "title": "Legal Advisor Agent",
        "expertise": "contracts", "primary": [], "keywords": [],
        "persona": "You are the Legal Advisor.",
    }
    ok = mgr.register_declarative(manifest)
    assert ok is True
    assert "legal" in mgr.agent_names()


def test_register_declarative_refuses_to_shadow_a_built_in():
    mgr = _manager()
    assert "coding" in mgr.agent_names()   # a real built-in
    original = mgr._agents["coding"]
    manifest = {
        "name": "coding", "title": "Fake Coding Agent",
        "expertise": "x", "primary": [], "keywords": [],
        "persona": "impostor",
    }
    ok = mgr.register_declarative(manifest)
    assert ok is False
    assert mgr._agents["coding"] is original   # untouched


def test_register_declarative_refuses_a_blank_name():
    mgr = _manager()
    before = set(mgr.agent_names())
    ok = mgr.register_declarative({"name": "", "title": "x", "expertise": "x",
                                   "persona": "x", "primary": [], "keywords": []})
    assert ok is False
    assert set(mgr.agent_names()) == before


def test_register_declarative_refuses_a_duplicate_name():
    mgr = _manager()
    manifest = {"name": "legal", "title": "Legal", "expertise": "x",
               "persona": "x", "primary": [], "keywords": []}
    assert mgr.register_declarative(manifest) is True
    assert mgr.register_declarative(manifest) is False   # already registered now


# ── load_declarative_agents() ───────────────────────────────────────────────

def test_load_declarative_agents_registers_every_valid_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "legal")
    _write_manifest(tmp_path, "medical")
    mgr = _manager()   # __init__ already loads them
    assert "legal" in mgr.agent_names()
    assert "medical" in mgr.agent_names()


def test_load_declarative_agents_returns_the_count(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "legal")
    _write_manifest(tmp_path, "medical")
    mgr = _manager()
    # __init__ already consumed them; a second explicit call re-loads the
    # same manifests, all now duplicates of already-registered agents.
    assert mgr.load_declarative_agents() == 0


def test_load_declarative_agents_with_no_directory_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    mgr = _manager()
    built_ins = {"marketing", "coding", "research", "neuroscience", "aiml",
                 "security", "reliability", "integrations", "continuity", "files", "triage"}
    assert set(mgr.agent_names()) == built_ins


def test_init_logs_a_skip_for_a_manifest_colliding_with_a_built_in(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "coding")   # collides with the real built-in
    bus = _StubBus()
    mgr = _manager(bus)
    assert any("coding" in m and "already registered" in m for m in bus.log.messages)
    # the built-in must still be the one actually registered
    assert mgr._agents["coding"].persona != "You are the coding specialist."


def test_a_declarative_agent_can_be_dispatched_to(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "legal")
    mgr = _manager()
    result = asyncio.run(mgr.dispatch("Help with a legal contract", agent_name="legal"))
    assert result.ok
    # no text-fallback provider configured -> hands back the specialist brief
    assert "legal specialist" in result.text.lower()


def _contract_manifest(**limits):
    return {
        "name": "legal", "title": "Legal Review", "expertise": "contracts",
        "persona": "You are the legal specialist.",
        "work_contract": {
            "purpose": "Find clause-specific risks.",
            "inputs": ["Supplied clauses", "User question"],
            "deliverable": "A prioritised review with clause references.",
            "acceptance": ["Cite each supplied clause", "Name unknowns"],
            "owner": "legal-review", "cancellation": "propagate",
            "limits": limits,
        },
    }


def test_work_contract_reaches_brief_and_detached_describe(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    mgr = _manager()
    manifest = _contract_manifest(max_steps=2, max_readings=1)
    assert mgr.register_declarative(manifest)
    manifest["work_contract"]["inputs"].append("mutated after registration")
    result = asyncio.run(mgr.dispatch("Review clause 2", agent_name="legal"))
    assert result.ok
    for text in ("Find clause-specific risks.", "Supplied clauses",
                 "A prioritised review with clause references.", "Name unknowns",
                 "legal-review", "propagate"):
        assert text in result.text
    row = next(r for r in mgr.describe() if r["name"] == "legal")
    assert row["work_contract"]["limits"] == {"max_steps": 2, "max_readings": 1}
    assert "mutated after registration" not in row["work_contract"]["inputs"]
    row["work_contract"]["inputs"].append("mutated description")
    assert "mutated description" not in mgr.get("legal").system_extra()
    assert mgr.get("legal").persona == "You are the legal specialist."
    assert result.receipt["state"] == "briefed"
    assert result.receipt["acceptance"]["state"] == "unverified"


def test_work_contract_direct_registration_validates_and_preserves_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    mgr = _manager()
    before = mgr.agent_names()
    manifest = _contract_manifest(max_readings=-1)
    assert mgr.register_declarative(manifest) is False
    assert mgr.agent_names() == before
    assert any("max_readings" in message for message in mgr.bus.log.messages)


class _ReadingRouter:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def has_text_fallback(self):
        return True

    async def generate_text(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(name="stub"), self.replies.pop(0)


def test_work_contract_reading_limit_does_not_mutate_parent_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    reading = 'TOOL: observe\nARGS: {"item": "clause"}'
    router = _ReadingRouter([reading, reading, "One risk remains unverified."])
    mgr = AgentManager(router, _StubBus())
    assert mgr.register_declarative(_contract_manifest(max_readings=1))

    async def observe(args):
        return ToolResult("clause 2 says X")

    belt = AgentToolbelt(extras={"observe": ("Read a clause", observe)}, budget=4)
    mgr.attach_toolbelt_factory(lambda: belt)
    result = asyncio.run(mgr.dispatch("Review clause 2", agent_name="legal"))
    assert belt.spent == 1
    assert belt.budget == 4
    assert belt.remaining() == 3
    assert len(result.evidence) == 1
    assert result.receipt["state"] == "answered"
    assert result.receipt["acceptance"]["state"] == "unverified"
    assert result.receipt["observations"] == {"attempted": 1, "succeeded": 1}


def test_work_contract_zero_reads_withdraws_tools_without_mutating_parent(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    router = _ReadingRouter(['TOOL: observe\nARGS: {}', "I cannot verify the clause."])
    mgr = AgentManager(router, _StubBus())
    assert mgr.register_declarative(_contract_manifest(max_readings=0))
    invoked = []

    async def observe(args):
        invoked.append(args)
        return ToolResult("unexpected")

    belt = AgentToolbelt(extras={"observe": ("Read a clause", observe)}, budget=4)
    mgr.attach_toolbelt_factory(lambda: belt)
    result = asyncio.run(mgr.dispatch("Review", agent_name="legal"))
    assert result.ok
    assert invoked == [] and belt.spent == 0 and belt.budget == 4
    assert all("AVAILABLE INSTRUMENTS" not in call[1]["system_extra"] for call in router.calls)


@pytest.mark.parametrize("requested,expected", [(1, 2), (999, BaseAgent.MAX_INVESTIGATION_STEPS + 1)])
def test_work_contract_step_limit_only_reduces_runtime_ceiling(monkeypatch, tmp_path, requested, expected):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    router = _ReadingRouter(['TOOL: observe\nARGS: {}'] * 8)
    mgr = AgentManager(router, _StubBus())
    assert mgr.register_declarative(_contract_manifest(max_steps=requested))

    async def observe(args):
        return ToolResult("observation")

    belt = AgentToolbelt(extras={"observe": ("Read a clause", observe)}, budget=8)
    mgr.attach_toolbelt_factory(lambda: belt)
    result = asyncio.run(mgr.dispatch("Review", agent_name="legal"))
    assert not result.ok
    assert len(router.calls) == expected
    assert result.receipt["state"] == "failed"


def test_work_contract_cancellation_propagates_without_followup_or_activity(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")

    class CancelledRouter(_ReadingRouter):
        async def generate_text(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            raise asyncio.CancelledError()

    router = CancelledRouter([])
    mgr = AgentManager(router, _StubBus())
    assert mgr.register_declarative(_contract_manifest(max_steps=1, max_readings=1))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mgr.dispatch("Review", agent_name="legal"))
    assert len(router.calls) == 1
    assert next(row for row in mgr.describe() if row["name"] == "legal")["calls"] == 0


def test_work_contract_large_read_limit_cannot_increase_parent_authority(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    reading = 'TOOL: observe\nARGS: {}'
    router = _ReadingRouter([reading, reading, "A provisional review."])
    mgr = AgentManager(router, _StubBus())
    assert mgr.register_declarative(_contract_manifest(max_readings=999))

    async def observe(args):
        return ToolResult("one observation")

    belt = AgentToolbelt(extras={"observe": ("Read a clause", observe)}, budget=1)
    mgr.attach_toolbelt_factory(lambda: belt)
    result = asyncio.run(mgr.dispatch("Review", agent_name="legal"))
    assert result.ok and len(result.evidence) == 1
    assert belt.spent == 1 and belt.budget == 1


def test_work_contract_reading_cancellation_propagates_with_consumed_parent_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    router = _ReadingRouter(['TOOL: observe\nARGS: {}'])
    mgr = AgentManager(router, _StubBus())
    assert mgr.register_declarative(_contract_manifest(max_readings=1))

    async def observe(args):
        raise asyncio.CancelledError()

    belt = AgentToolbelt(extras={"observe": ("Read a clause", observe)}, budget=4)
    mgr.attach_toolbelt_factory(lambda: belt)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mgr.dispatch("Review", agent_name="legal"))
    assert belt.spent == 1 and belt.budget == 4
    assert len(router.calls) == 1
    assert next(row for row in mgr.describe() if row["name"] == "legal")["calls"] == 0


@pytest.mark.parametrize("limits", [{"max_steps": 1}, {"max_readings": 0}])
def test_work_contract_forced_close_receipt_names_the_answering_provider(monkeypatch, tmp_path, limits):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")

    class RotatingRouter(_ReadingRouter):
        async def generate_text(self, prompt, **kwargs):
            _profile, reply = await super().generate_text(prompt, **kwargs)
            provider = "investigation-provider" if len(self.calls) == 1 else "answer-provider"
            return SimpleNamespace(name=provider), reply

    router = RotatingRouter(['TOOL: observe\nARGS: {}', "The conclusion needs verification."])
    mgr = AgentManager(router, _StubBus())
    assert mgr.register_declarative(_contract_manifest(**limits))

    async def observe(args):
        return ToolResult("clause 2")

    mgr.attach_toolbelt_factory(lambda: AgentToolbelt(
        extras={"observe": ("Read a clause", observe)}, budget=4))
    result = asyncio.run(mgr.dispatch("Review", agent_name="legal"))
    assert result.ok
    assert result.receipt["provider"] == "answer-provider"
    assert len(router.calls) == 2


@pytest.mark.parametrize("response", [
    "", " \n\t", 'TOOL: observe\nARGS: {}', '{"tool": "observe", "args": {}}',
])
def test_work_contract_direct_handle_rejects_unusable_provider_answer(response):
    agent = ar.build_agent(_contract_manifest(), _ReadingRouter([response]), _StubBus())
    result = asyncio.run(agent.handle("Review"))
    assert not result.ok
    assert "no usable answer" in result.text.lower()
    assert result.receipt["state"] == "failed"
    assert result.receipt["provider"] == "stub"


class _BrokenSignal:
    def emit(self, *_):
        raise RuntimeError("observer was deleted")


class _BrokenObserverBus:
    def __getattr__(self, name):
        return _BrokenSignal()


def test_work_contract_direct_handle_preserves_answer_when_observer_throws():
    agent = ar.build_agent(_contract_manifest(), _ReadingRouter(["Clause 2 has a risk."]),
                           _BrokenObserverBus())
    result = asyncio.run(agent.handle("Review"))
    assert result.ok and "Clause 2 has a risk." in result.text
    assert result.receipt["state"] == "answered"
    assert result.receipt["provider"] == "stub"


@pytest.mark.parametrize("read_first", [False, True])
def test_work_contract_investigation_preserves_findings_when_observers_throw(read_first):
    replies = ['TOOL: observe\nARGS: {}'] if read_first else []
    replies.append("Clause 2 has a risk.")
    agent = ar.build_agent(_contract_manifest(), _ReadingRouter(replies), _BrokenObserverBus())

    async def observe(args):
        return ToolResult("clause 2")

    belt = AgentToolbelt(extras={"observe": ("Read a clause", observe)}, budget=1)
    finding = asyncio.run(agent.investigate("Review", toolbelt=belt))
    assert finding.ok and finding.answer == "Clause 2 has a risk."
    assert finding.work["state"] == "answered"
    assert finding.work["provider"] == "stub"
    assert len(finding.tools) == int(read_first)


def test_work_contract_manager_preserves_receipt_when_observers_and_factory_throw(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    mgr = AgentManager(_ReadingRouter(["Clause 2 has a risk."]), _BrokenObserverBus())
    assert mgr.register_declarative(_contract_manifest())

    def unavailable():
        raise RuntimeError("instrument tray unavailable")

    mgr.attach_toolbelt_factory(unavailable)
    result = asyncio.run(mgr.dispatch("Review", agent_name="legal"))
    assert result.ok and "Clause 2 has a risk." in result.text
    assert result.receipt["state"] == "answered"
    assert result.receipt["observations"] == {"attempted": 0, "succeeded": 0}


def test_work_contract_optional_observer_never_swallows_cancellation():
    class CancellingSignal:
        def emit(self, *_):
            raise asyncio.CancelledError()

    agent = ar.build_agent(_contract_manifest(), _ReadingRouter(["A provisional answer."]),
                           SimpleNamespace(agent_activity=CancellingSignal()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.handle("Review"))


def test_specialist_runtime_failure_releases_slot_and_resets_monitor(monkeypatch, tmp_path):
    from orion_core import admission, job_store, jobs, agents
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    policy = admission.Admission(governor=SimpleNamespace(acting=False))
    store = job_store.JobStore(None)
    monkeypatch.setattr(admission, "ADMISSION", policy)
    monkeypatch.setattr(job_store, "_STORE", store)
    monkeypatch.setattr(jobs, "CONTROLS", {})
    mgr = _manager()
    original = agents.SpecialistRun.execute

    async def fail(_run):
        assert jobs.current_control() is not None
        raise ValueError("synthetic specialist runtime error")

    monkeypatch.setattr(agents.SpecialistRun, "execute", fail)
    try:
        with pytest.raises(ValueError, match="specialist runtime error"):
            asyncio.run(mgr.dispatch("Find the cause of a freeze", agent_name="reliability"))
        assert not jobs.CONTROLS and jobs.current_control() is None
        assert policy.snapshot()["classes"]["interactive"]["running"] == 0
        [record] = store.list()
        assert record["state"] == "failed" and not record["ok"]
        monitor = next(row for row in mgr.describe() if row["name"] == "reliability")
        assert monitor["state"] == "idle" and monitor["last_outcome"] == "failed"
        monkeypatch.setattr(agents.SpecialistRun, "execute", original)
        assert asyncio.run(mgr.dispatch("Try again", agent_name="reliability")).ok
        assert not jobs.CONTROLS
    finally:
        store.close()


@pytest.mark.parametrize("error", [OSError, asyncio.CancelledError])
def test_specialist_record_failure_releases_admission_ticket(monkeypatch, tmp_path, error):
    from orion_core import admission, job_store, jobs
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    policy = admission.Admission(governor=SimpleNamespace(acting=False))
    monkeypatch.setattr(admission, "ADMISSION", policy)
    monkeypatch.setattr(jobs, "CONTROLS", {})
    store = job_store.JobStore(None)
    monkeypatch.setattr(job_store, "_STORE", store)

    def fail(**_kwargs):
        raise error("synthetic record creation failure")

    monkeypatch.setattr(store, "create", fail)
    mgr = _manager()
    try:
        with pytest.raises(error, match="record creation failure"):
            asyncio.run(mgr.dispatch("Find the cause", agent_name="reliability"))
        assert not jobs.CONTROLS and jobs.current_control() is None
        state = policy.snapshot()["classes"]["interactive"]
        assert state["running"] == state["pending"] == 0
    finally:
        store.close()


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "completed"])
def test_specialist_final_store_failure_still_releases_live_ownership(monkeypatch, tmp_path, outcome):
    from orion_core import admission, agents, job_store, jobs
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    policy = admission.Admission(governor=SimpleNamespace(acting=False))
    store = job_store.JobStore(None)
    monkeypatch.setattr(admission, "ADMISSION", policy)
    monkeypatch.setattr(job_store, "_STORE", store)
    monkeypatch.setattr(jobs, "CONTROLS", {})
    mgr = _manager()

    def fail_transition(*_args, **_kwargs):
        raise OSError("synthetic final store failure")

    async def execute(run):
        monkeypatch.setattr(store, "transition", fail_transition)
        if outcome == "failed":
            raise ValueError("original specialist failure")
        if outcome == "cancelled":
            raise asyncio.CancelledError("original specialist cancellation")
        run.report.outcome = "done"
        run.report.summary = "Synthetic completed work"
        return run.report

    monkeypatch.setattr(agents.SpecialistRun, "execute", execute)
    error = {"failed": ValueError, "cancelled": asyncio.CancelledError,
             "completed": OSError}[outcome]
    detail = "final store failure" if outcome == "completed" else "original specialist"
    try:
        with pytest.raises(error, match=detail):
            asyncio.run(mgr.dispatch("Find the cause", agent_name="reliability"))
        assert not jobs.CONTROLS and jobs.current_control() is None
        assert policy.snapshot()["classes"]["interactive"]["running"] == 0
        monitor = next(row for row in mgr.describe() if row["name"] == "reliability")
        assert monitor["state"] == "idle"
        assert monitor["last_outcome"] == ("done" if outcome == "completed" else outcome)
    finally:
        store.close()
