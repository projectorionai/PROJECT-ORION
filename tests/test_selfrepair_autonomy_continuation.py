"""Automatic repair limits apply to concurrent work and the plugin gates."""
import asyncio
import hashlib
import threading
from types import SimpleNamespace

import pytest

from orion_core import selfrepair as module
from orion_core.data import ToolResult


@pytest.fixture
def agent(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    repairs = tmp_path / "repairs"
    monkeypatch.setattr(module, "PACKAGE_DIR", package)
    monkeypatch.setattr(module, "SELF_REPAIR_DIR", repairs)
    monkeypatch.setattr(module, "JOURNAL_PATH", repairs / "journal.jsonl")
    signal = SimpleNamespace(emit=lambda *_: None)
    agent = module.SelfRepairAgent(SimpleNamespace(log=signal, banner=signal), None, None)
    return agent, package


async def test_concurrent_automatic_repairs_cannot_exceed_session_cap(agent):
    agent, package = agent
    release = threading.Event()
    entered = asyncio.Event()
    callers = 0
    applied = []
    def passing(*_):
        assert release.wait(5)
        return True, "synthetic dedicated checks passed"
    agent._verify_auto_candidate = passing
    agent._apply_repaired = lambda value, target, **_: (applied.append(value.id) or ToolResult("synthetic receipt"))
    async def attempt(index):
        nonlocal callers
        target = package / f"module_{index}.py"
        target.write_bytes(b"VALUE = 1\n")
        value = module.Incident(str(index), "now", "ValueError", "synthetic", file=str(target), repaired_content="VALUE = 2\n")
        callers += 1
        if callers == 3:
            entered.set()
        await agent.attempt_auto_repair(value)
    tasks = [asyncio.create_task(attempt(index)) for index in range(3)]
    try:
        await asyncio.wait_for(entered.wait(), 3)
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert len(applied) == agent.MAX_AUTO_REPAIRS_PER_SESSION == agent._auto_repairs


@pytest.mark.parametrize("filename", ["dispatcher.py", "dispatch_productivity.py", "plugin_manifest.py",
                                      "plugin_events.py", "plugin_vault.py", "forge_deployment.py"])
async def test_automatic_repair_cannot_edit_plugin_policy_gates(agent, filename):
    agent, package = agent
    target = package / filename
    target.write_bytes(b"VALUE = 1\n")
    applied = []
    agent._verify_auto_candidate = lambda *_: (True, "synthetic dedicated checks passed")
    agent._apply_repaired = lambda *args, **_: (applied.append(args) or ToolResult("synthetic receipt"))
    value = module.Incident(filename, "now", "ValueError", "synthetic", file=str(target), repaired_content="VALUE = 2\n")
    await agent.attempt_auto_repair(value)
    assert applied == []
    assert agent._auto_repairs == 0


async def test_declined_tests_release_capacity_for_a_later_incident(agent):
    agent, package = agent
    applied = []
    passing = False
    agent._verify_auto_candidate = lambda *_: (passing, "synthetic check result")
    agent._apply_repaired = lambda value, _, **__: (applied.append(value.id) or ToolResult("synthetic receipt"))
    for index in range(3):
        target = package / f"module_{index}.py"
        target.write_bytes(b"VALUE = 1\n")
        value = module.Incident(str(index), "now", "ValueError", "synthetic", file=str(target), repaired_content="VALUE = 2\n")
        await agent.attempt_auto_repair(value)
        passing = True
    assert applied == ["1", "2"]
    assert agent._auto_pending == 0
    assert agent._auto_repairs == 2


async def test_changed_draft_cannot_clear_verification_identity_at_publication(agent, monkeypatch):
    agent, package = agent
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    value = module.Incident("race", "now", "ValueError", "synthetic", file=str(target),
                            repaired_content="VALUE = 2\n")
    monkeypatch.setattr(agent, "_verify_auto_candidate", lambda *_: (True, "synthetic before/after proof"))
    publish = agent._apply_repaired
    def change_candidate(incident, path, **identities):
        incident.repaired_content = "VALUE = 999\n"
        incident.verified_digest = ""
        return publish(incident, path, **identities)
    monkeypatch.setattr(agent, "_apply_repaired", change_candidate)
    await agent.attempt_auto_repair(value)
    assert not value.applied
    assert target.read_bytes() == b"VALUE = 1\n"
    assert agent._auto_repairs == 0


async def test_auto_repair_preserves_source_edited_since_the_original_draft(agent, monkeypatch):
    agent, package = agent
    target = package / "module.py"
    original = b"VALUE = 1\n"
    current = original + b"# A developer's later change must be retained.\n"
    target.write_bytes(current)
    value = module.Incident("stale", "now", "ValueError", "synthetic", file=str(target),
                            repaired_content="VALUE = 2\n",
                            source_digest=hashlib.sha256(original).hexdigest())
    def no_tests(*_):
        raise AssertionError("a stale draft should be declined before test execution")
    monkeypatch.setattr(agent, "_verify_auto_candidate", no_tests)
    await agent.attempt_auto_repair(value)
    assert target.read_bytes() == current
    assert not value.applied
    assert "source changed" in agent.journal_tail()[-1]["note"]


def test_captured_verification_identity_cannot_override_draft_provenance(agent):
    agent, package = agent
    target = package / "module.py"
    original, current = b"VALUE = 1\n", b"VALUE = 1\n# newer source\n"
    target.write_bytes(current)
    candidate = "VALUE = 2\n"
    value = module.Incident("stale", "now", "ValueError", "synthetic", file=str(target),
                            repaired_content=candidate,
                            source_digest=hashlib.sha256(original).hexdigest())
    result = agent._apply_repaired(value, target,
        expected_digest=hashlib.sha256(candidate.encode()).hexdigest(),
        expected_source_digest=hashlib.sha256(current).hexdigest())
    assert not result.ok
    assert target.read_bytes() == current
