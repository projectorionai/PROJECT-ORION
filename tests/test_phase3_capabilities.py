"""
Phase 3 capability tests — the skills system, the workflow/automation engine
and the Creator Studio creator intelligence suite.

All offline: stub bus, stub dispatcher, temporary paths.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.creator_intel import (CreatorIntelSuite, CreatorManager,
                                      HookAnalyzer, ScriptEvaluator,
                                      ViralAnalysisAgent)
from orion_core.data import ToolResult
from orion_core.skills import SkillManager
from orion_core.workflow_engine import AutomationManager, WorkflowEngine


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload))


class StubBus:
    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


# ── skills ────────────────────────────────────────────────────────────────────

@pytest.fixture()
def skills(tmp_path, monkeypatch):
    # Never touch the machine's Recycle Bin: a move inside this private tree
    # represents a confirmed, recoverable recycle receipt.
    from orion_core import recycle_bin
    retired = tmp_path / "retired-skills"
    def recycle(path):
        retired.mkdir(exist_ok=True)
        target = Path(path)
        target.replace(retired / (target.name + "-" + str(len(list(retired.iterdir())))))
        return True
    monkeypatch.setattr(recycle_bin, "refusal", lambda path: None)
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", recycle)
    return SkillManager(StubBus(), root=tmp_path / "skills")


def test_builtin_skills_seed_on_first_run(skills):
    listing = skills.list_skills().text
    for name in ("research-method", "creator-management",
                 "content-strategy", "business-analysis"):
        assert name in listing


def test_install_describe_remove_cycle(skills, tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "skill.json").write_text(json.dumps({
        "name": "Test Skill", "version": "2.0.0",
        "description": "A test skill",
        "prompts": {"system": "Always test twice."},
        "tools": ["research"],
    }), encoding="utf-8")
    result = skills.install(str(package))
    assert result.ok and "test-skill" in result.text
    described = skills.describe("test-skill")
    assert described.ok and "v2.0.0" in described.text
    # Same version again is refused.
    assert not skills.install(str(package)).ok
    assert skills.remove("test-skill").ok
    assert not skills.describe("test-skill").ok


def test_prompt_context_only_from_enabled_skills(skills):
    context = skills.prompt_context()
    assert "[SKILL research-method]" in context
    skills.set_enabled("research-method", False)
    assert "[SKILL research-method]" not in skills.prompt_context()
    skills.set_enabled("research-method", True)
    assert "[SKILL research-method]" in skills.prompt_context()


def test_broken_manifest_is_rejected(skills, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "skill.json").write_text("{not json", encoding="utf-8")
    assert not skills.install(str(bad)).ok
    missing_name = tmp_path / "anon"
    missing_name.mkdir()
    (missing_name / "skill.json").write_text(json.dumps({"version": "1.0.0"}),
                                             encoding="utf-8")
    assert not skills.install(str(missing_name)).ok


def test_skill_templates_accessible(skills):
    template = skills.template("creator-management", "feedback")
    assert "{hook_score}" in template


def _xxxv_package(path, *, name="test-skill", version="1", note="original"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "skill.json").write_text(json.dumps({
        "name": name, "version": version,
        "prompts": {"system": note},
    }), encoding="utf-8")
    (path / "notes.txt").write_text(note, encoding="utf-8")
    return path


def _xxxv_bytes(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("kind", ["parent", "absolute"])
def test_xxxv_skill_enable_cannot_address_an_outside_manifest(skills, tmp_path, kind):
    outside = _xxxv_package(tmp_path / "outside")
    before = _xxxv_bytes(outside)
    name = "../outside" if kind == "parent" else str(outside)
    result = skills.set_enabled(name, False)
    assert not result.ok, "a filesystem address is not a skill identity"
    assert _xxxv_bytes(outside) == before


@pytest.mark.parametrize("target_part", ["directory", "manifest"])
def test_xxxv_skill_enable_refuses_reparse_targets(skills, monkeypatch, target_part):
    target = skills.root / "research-method"
    victim = target if target_part == "directory" else target / "skill.json"
    before = _xxxv_bytes(target)
    original = getattr(Path, "is_junction", lambda self: False)
    monkeypatch.setattr(Path, "is_junction",
                        lambda self: self == victim or original(self), raising=False)
    result = skills.set_enabled("research-method", False)
    assert not result.ok
    assert _xxxv_bytes(target) == before


@pytest.mark.parametrize("part", ["directory", "manifest"])
def test_xxxv_skill_install_refuses_reparse_sources(skills, tmp_path, monkeypatch, part):
    package = _xxxv_package(tmp_path / "linked-package")
    victim = package if part == "directory" else package / "skill.json"
    original = getattr(Path, "is_junction", lambda self: False)
    monkeypatch.setattr(Path, "is_junction",
                        lambda self: self == victim or original(self), raising=False)
    result = skills.install(str(package))
    assert not result.ok
    assert not (skills.root / "test-skill").exists()


def test_xxxv_skill_install_refuses_an_existing_reparse_destination(skills, tmp_path, monkeypatch):
    package = _xxxv_package(tmp_path / "incoming", name="research-method", version="2")
    victim = skills.root / "research-method"
    before = _xxxv_bytes(victim)
    original = getattr(Path, "is_junction", lambda self: False)
    monkeypatch.setattr(Path, "is_junction",
                        lambda self: self == victim or original(self), raising=False)
    result = skills.install(str(package))
    assert not result.ok
    assert _xxxv_bytes(victim) == before


def test_xxxv_skill_remove_respects_recycle_refusal(skills, monkeypatch):
    from orion_core import recycle_bin
    calls = []
    monkeypatch.setattr(recycle_bin, "refusal", lambda p: "controlled bin refusal")
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", lambda p: calls.append(p))
    result = skills.remove("research-method")
    assert not result.ok and "refus" in result.text.lower()
    assert (skills.root / "research-method" / "skill.json").exists()
    assert skills.describe("research-method").ok
    assert calls == []


def test_xxxv_skill_remove_does_not_claim_an_unverified_recycle(skills, monkeypatch):
    from orion_core import recycle_bin
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", lambda p: False)
    result = skills.remove("research-method")
    assert not result.ok
    assert (skills.root / "research-method" / "skill.json").exists()
    assert skills.describe("research-method").ok


def test_xxxv_skill_remove_uses_recoverable_retirement(skills, tmp_path):
    before = _xxxv_bytes(skills.root / "research-method")
    assert skills.remove("research-method").ok
    saved = [p for p in (tmp_path / "retired-skills").iterdir()
             if p.name.startswith("research-method-")]
    assert len(saved) == 1 and _xxxv_bytes(saved[0]) == before
    assert not skills.describe("research-method").ok


def test_xxxv_skill_remove_never_collapses_invalid_identity(skills, tmp_path):
    assert skills.install(str(_xxxv_package(tmp_path / "incoming", name="testskill"))).ok
    result = skills.remove("test/skill")
    assert not result.ok
    assert skills.describe("testskill").ok


@pytest.mark.parametrize("failure", ["copy", "checkpoint"])
def test_xxxv_skill_failed_upgrade_preserves_disk_and_registry(skills, tmp_path, monkeypatch, failure):
    from orion_core import skills as module
    old = _xxxv_package(tmp_path / "old")
    assert skills.install(str(old)).ok
    target = skills.root / "test-skill"
    before = _xxxv_bytes(target)
    package = _xxxv_package(tmp_path / "new", version="2", note="replacement")
    if failure == "copy":
        original = module.shutil.copy2
        calls = []
        def copy(source, destination, *args, **kwargs):
            calls.append(Path(source).name)
            if len(calls) == 2:
                raise OSError("controlled second-file copy failure")
            return original(source, destination, *args, **kwargs)
        monkeypatch.setattr(module.shutil, "copy2", copy)
    else:
        def checkpoint(*args, **kwargs):
            raise OSError("controlled checkpoint failure")
        monkeypatch.setattr(module, "atomic_write_text", checkpoint)
    result = skills.install(str(package))
    assert not result.ok
    assert _xxxv_bytes(target) == before
    assert skills.registry.get("test-skill").version == "1"
    assert skills.reload() >= 4
    assert skills.registry.get("test-skill").version == "1"


def test_xxxv_skill_manifest_input_is_bounded(skills, tmp_path, monkeypatch):
    from orion_core import skills as module
    package = _xxxv_package(tmp_path / "oversized", note="z" * 300)
    monkeypatch.setattr(module, "_MAX_MANIFEST_BYTES", 128, raising=False)
    assert not skills.install(str(package)).ok
    assert not (skills.root / "test-skill").exists()


def test_xxxv_skill_package_enumeration_is_bounded(skills, tmp_path, monkeypatch):
    from orion_core import skills as module
    package = _xxxv_package(tmp_path / "many-files")
    for index in range(4):
        (package / f"extra-{index}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(module, "_MAX_PACKAGE_FILES", 3, raising=False)
    assert not skills.install(str(package)).ok
    assert not (skills.root / "test-skill").exists()


def test_xxxv_skill_scan_has_a_finite_entry_budget(skills, monkeypatch):
    from orion_core import skills as module
    monkeypatch.setattr(module, "_MAX_SCAN_ENTRIES", 2, raising=False)
    assert len(module.SkillLoader.scan(skills.root)) <= 2


def test_xxxv_skill_context_counts_labels_and_separators(skills):
    from orion_core.skills import Skill, SkillRegistry
    skills.registry = SkillRegistry()
    for name in ("a", "b"):
        skills.registry.put(Skill(name, prompts={"system": "12345"}))
    text = skills.prompt_context(limit=12)
    assert len(text) <= 12
    assert skills.prompt_context(limit=0) == ""
    assert skills.prompt_context(limit=-1) == ""


def test_xxxv_skill_context_large_first_entry_does_not_starve_later_entries(skills):
    from orion_core.skills import Skill, SkillRegistry
    skills.registry = SkillRegistry()
    skills.registry.put(Skill("a", prompts={"system": "too long " * 30}))
    skills.registry.put(Skill("b", prompts={"system": "useful"}))
    text = skills.prompt_context(limit=40)
    assert "[SKILL b] useful" in text and len(text) <= 40


def test_xxxv_skill_reload_publishes_only_a_complete_prompt_snapshot(skills, monkeypatch):
    import threading
    from orion_core import skills as module
    original = module.SkillLoader.scan
    entered, release = threading.Event(), threading.Event()
    before = skills.prompt_context()
    errors = []
    def scan(root):
        entered.set()
        if not release.wait(2):
            raise AssertionError("bounded test reload was not released")
        return original(root)
    monkeypatch.setattr(module.SkillLoader, "scan", scan)
    def reload():
        try:
            skills.reload()
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=reload, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        assert skills.prompt_context() == before
        assert skills.describe("research-method").ok
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert skills.prompt_context() == before


def test_xxxv_skill_registry_returns_independent_prompt_snapshots(skills):
    seen = skills.registry.get("research-method")
    original = skills.prompt_context()
    seen.prompts["system"] = "not the saved prompt"
    seen.enabled = False
    assert skills.prompt_context() == original


def test_xxxv_skill_enable_refuses_an_expanded_checkpoint_without_losing_the_package(skills, tmp_path, monkeypatch):
    from orion_core import skills as module
    assert skills.install(str(_xxxv_package(tmp_path / "compact"))).ok
    manifest = skills.root / "test-skill" / "skill.json"
    data = {"name": "test-skill", "version": "1", "prompts": {"system": "original"},
            "extra": {"rows": ["abcdefgh"] * 24}}
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    data["enabled"] = False
    assert len(raw) <= 512 < len(json.dumps(data, indent=2).encode("utf-8"))
    manifest.write_bytes(raw)
    monkeypatch.setattr(module, "_MAX_MANIFEST_BYTES", 512)
    result = skills.set_enabled("test-skill", False)
    assert not result.ok
    assert manifest.read_bytes() == raw
    assert skills.registry.get("test-skill").enabled
    skills.reload()
    assert skills.registry.get("test-skill").enabled


def test_xxxv_skill_enable_failed_checkpoint_preserves_bytes_and_prompt_snapshot(skills, monkeypatch):
    from orion_core import skills as module
    manifest = skills.root / "research-method" / "skill.json"
    raw, context = manifest.read_bytes(), skills.prompt_context()
    def failed_write(*args, **kwargs):
        raise OSError("controlled toggle checkpoint failure")
    monkeypatch.setattr(module, "atomic_write_text", failed_write)
    assert not skills.set_enabled("research-method", False).ok
    assert manifest.read_bytes() == raw
    assert skills.prompt_context() == context


def test_xxxv_skill_enable_refuses_a_manifest_for_a_different_identity(skills):
    manifest = skills.root / "research-method" / "skill.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["name"] = "different-package"
    raw = json.dumps(data).encode("utf-8")
    manifest.write_bytes(raw)
    context = skills.prompt_context()
    result = skills.set_enabled("research-method", False)
    assert not result.ok
    assert manifest.read_bytes() == raw
    assert skills.prompt_context() == context


# ── workflow engine ───────────────────────────────────────────────────────────

class StubDispatcher:
    """Records dispatches; scripted failures by tool name."""

    def __init__(self):
        self.calls = []
        self.fail_first = set()
        self.always_fail = set()

    def handler_table(self):
        return {name: (lambda a: None) for name in
                ("research", "executive", "companion", "creator_intel")}

    async def dispatch(self, name, args):
        self.calls.append((name, dict(args or {})))
        if name in self.always_fail:
            return ToolResult(f"{name} broken", ok=False)
        if name in self.fail_first:
            self.fail_first.discard(name)
            return ToolResult(f"{name} transient", ok=False)
        return ToolResult(f"{name} ran with {args}")


@pytest.fixture()
def engine(tmp_path):
    return WorkflowEngine(StubBus(), dispatcher=StubDispatcher(),
                          path=tmp_path / "workflows.json")


def test_define_rejects_unknown_tools(engine):
    result = engine.define("bad", [{"tool": "not_a_tool", "args": {}}])
    assert not result.ok and "Unknown tool" in result.text


def test_define_persists_to_disk(engine, tmp_path):
    assert engine.define("mine", [{"tool": "research",
                                   "args": {"action": "agenda"}}]).ok
    reloaded = WorkflowEngine(StubBus(), dispatcher=StubDispatcher(),
                              path=tmp_path / "workflows.json")
    assert "mine" in reloaded.definitions


def _run_to_completion(engine, name, input_text=""):
    async def scenario():
        result = engine.start(name, input_text)
        assert result.ok, result.text
        run = list(engine.runs.values())[-1]
        await run["task"]
        return run
    return asyncio.run(scenario())


def test_run_resolves_placeholders_and_completes(engine):
    engine.define("chain", [
        {"tool": "research", "args": {"action": "queue", "topic": "{input}"}},
        {"tool": "executive", "args": {"action": "focus", "context": "{prev}"}},
    ])
    run = _run_to_completion(engine, "chain", "ai systems")
    assert run["status"] == "complete"
    calls = engine.dispatcher.calls
    assert calls[0] == ("research", {"action": "queue", "topic": "ai systems"})
    assert "research ran" in calls[1][1]["context"]


def test_run_retries_transient_failure(engine, monkeypatch):
    async def _instant(_secs):
        return None
    monkeypatch.setattr("orion_core.workflow_engine.asyncio.sleep", _instant)
    engine.dispatcher.fail_first.add("research")
    engine.define("retry", [{"tool": "research", "args": {"action": "status"}, "retries": 1}])
    run = _run_to_completion(engine, "retry")
    assert run["status"] == "complete"
    assert len(engine.dispatcher.calls) == 2


def test_run_stops_on_persistent_failure(engine, monkeypatch):
    async def _instant(_secs):
        return None
    monkeypatch.setattr("orion_core.workflow_engine.asyncio.sleep", _instant)
    engine.dispatcher.always_fail.add("companion")
    engine.define("fails", [
        {"tool": "companion", "args": {}, "retries": 1},
        {"tool": "executive", "args": {}},
    ])
    run = _run_to_completion(engine, "fails")
    assert run["status"].startswith("failed at step 1")
    # The second step never ran.
    assert all(name != "executive" for name, _ in engine.dispatcher.calls)
    status = engine.status().text
    assert "fails" in status and "FAILED" in status


def test_automation_manager_seeds_builtins_and_routes(engine):
    manager = AutomationManager(engine)
    listing = asyncio.run(manager.handle({"action": "list"})).text
    for name in ("research_workflow", "creator_review_workflow",
                 "product_analysis_workflow", "daily_review_workflow"):
        assert name in listing
    unknown = asyncio.run(manager.handle({"action": "run", "name": "nope"}))
    assert not unknown.ok


def test_unstarted_workflow_reports_missing(engine):
    assert not engine.start("ghost").ok


# ── workflow runs are durable, admitted jobs ─────────────────────────────────

@pytest.fixture()
def isolated_jobs(monkeypatch):
    """A private admission policy and in-memory job store per test."""
    from orion_core import admission, job_store, jobs
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission())
    monkeypatch.setattr(job_store, "_STORE", job_store.JobStore(None))
    jobs.CONTROLS.clear()
    yield
    jobs.CONTROLS.clear()


def test_every_builtin_workflow_still_runs_to_completion(engine, isolated_jobs):
    """The named library must keep working through the new job layer."""
    AutomationManager(engine)
    for name in ("research_workflow", "creator_review_workflow",
                 "product_analysis_workflow", "daily_review_workflow"):
        run = _run_to_completion(engine, name, "short-form video")
        assert run["status"] == "complete", (name, run["status"])


def test_a_run_is_recorded_as_a_job_with_a_unique_id(engine, isolated_jobs):
    from orion_core import job_store
    engine.define("twice", [{"tool": "executive", "args": {"action": "focus"}}])

    async def scenario():
        engine.start("twice")
        engine.start("twice")                       # same second: used to collide
        for run in list(engine.runs.values()):
            await run["task"]
        return list(engine.runs.values())

    runs = asyncio.run(scenario())
    assert len(runs) == 2 and runs[0]["id"] != runs[1]["id"]
    record = job_store.default_store().get(runs[0]["job_id"])
    assert record["kind"] == "workflow" and record["state"] == "completed"
    assert record["fraction"] == 1.0
    assert [s["state"] for s in job_store.default_store().steps(record["id"])] == ["completed"]


def test_a_run_beyond_the_limit_queues_visibly_then_runs(engine, monkeypatch, isolated_jobs):
    from orion_core import admission
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(
        {admission.WorkClass.USER: admission.ClassPolicy(running=1, pending=2)}))
    engine.define("one", [{"tool": "executive", "args": {"action": "focus"}}])

    async def scenario():
        first = engine.start("one")
        second = engine.start("one")
        statuses = [r["status"] for r in engine.runs.values()]
        for run in list(engine.runs.values()):
            await run["task"]
        return first, second, statuses

    first, second, statuses = asyncio.run(scenario())
    assert statuses == ["running", "queued"]
    assert "queued" in second.text and "in line" in second.text
    assert all(r["status"] == "complete" for r in engine.runs.values())


class _GatedStub(StubDispatcher):
    def __init__(self, gate_tool):
        super().__init__()
        self.gate_tool = gate_tool
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def dispatch(self, name, args):
        if name == self.gate_tool and not self.release.is_set():
            self.calls.append((name, dict(args or {})))
            self.entered.set()
            await self.release.wait()
            return ToolResult(f"{name} ran with {args}")
        return await super().dispatch(name, args)


def test_a_run_pauses_between_steps_and_resumes(tmp_path, isolated_jobs):
    from orion_core import job_store
    dispatcher = _GatedStub("executive")
    engine = WorkflowEngine(StubBus(), dispatcher=dispatcher, path=tmp_path / "w.json")
    engine.define("two", [{"tool": "executive", "args": {"action": "focus"}},
                          {"tool": "research", "args": {"action": "agenda"}}])

    async def scenario():
        engine.start("two")
        run = list(engine.runs.values())[-1]
        await dispatcher.entered.wait()
        from orion_core import jobs
        jobs.CONTROLS[run["job_id"]].pause()
        dispatcher.release.set()
        await asyncio.sleep(0.15)
        paused = (run["status"], job_store.default_store().get(run["job_id"])["state"],
                  [c[0] for c in dispatcher.calls])
        assert engine.resume(run["id"]).ok
        await run["task"]
        return run, paused

    run, paused = asyncio.run(scenario())
    assert paused == ("paused", "paused", ["executive"])
    assert run["status"] == "complete"
    assert [c[0] for c in dispatcher.calls] == ["executive", "research"]


def _crash_mid_workflow(tmp_path, steps, gate_tool):
    """Run a workflow until *gate_tool* is in flight, copy the job database as
    a crash would leave it, abandon the run; return the reopened copy."""
    import sqlite3

    from orion_core import job_store
    live = job_store.JobStore(tmp_path / "live.db")
    dispatcher = _GatedStub(gate_tool)
    engine = WorkflowEngine(StubBus(), dispatcher=dispatcher, path=tmp_path / "w.json",
                            store=live)
    engine.define("flow", steps)

    async def scenario():
        engine.start("flow", "topic")
        run = list(engine.runs.values())[-1]
        await dispatcher.entered.wait()
        crashed = sqlite3.connect(tmp_path / "crashed.db")
        live._db.backup(crashed)
        crashed.close()
        run["task"].cancel()
        await asyncio.gather(run["task"], return_exceptions=True)

    asyncio.run(scenario())
    live.close()
    first_calls = [c[0] for c in dispatcher.calls]
    return job_store.JobStore(tmp_path / "crashed.db"), engine.definitions, first_calls


def _restarted_engine(tmp_path, store, definitions):
    dispatcher = StubDispatcher()
    engine = WorkflowEngine(StubBus(), dispatcher=dispatcher, path=tmp_path / "w2.json",
                            store=store)
    engine.definitions = definitions
    return engine, dispatcher


def test_after_a_crash_a_read_only_step_resumes_where_it_stopped(tmp_path, isolated_jobs):
    steps = [{"tool": "companion", "args": {"action": "brief"}},
             {"tool": "research", "args": {"action": "agenda", "text": "{prev}"}},
             {"tool": "executive", "args": {"action": "focus"}}]
    store, definitions, first = _crash_mid_workflow(tmp_path, steps, "research")
    [record] = store.recovered
    assert record["state"] == "interrupted" and first == ["companion", "research"]
    engine, dispatcher = _restarted_engine(tmp_path, store, definitions)

    async def scenario():
        result = engine.resume(record["id"])
        run = list(engine.runs.values())[-1]
        await run["task"]
        return result, run

    result, run = asyncio.run(scenario())
    assert result.ok and run["status"] == "complete"
    assert [c[0] for c in dispatcher.calls] == ["research", "executive"], \
        "the side-effecting first step is not repeated"
    assert "companion ran" in dispatcher.calls[0][1]["text"],         "the resumed step still gets the result the step before it produced"
    store.close()


def test_after_a_crash_a_side_effecting_step_waits_for_approval(tmp_path, isolated_jobs):
    steps = [{"tool": "research", "args": {"action": "agenda"}},
             {"tool": "companion", "args": {"action": "brief"}},
             {"tool": "executive", "args": {"action": "goals"}}]
    store, definitions, _ = _crash_mid_workflow(tmp_path, steps, "companion")
    [record] = store.recovered
    engine, dispatcher = _restarted_engine(tmp_path, store, definitions)
    manager = AutomationManager(engine)

    asked = asyncio.run(manager.handle({"action": "resume", "run": record["id"]}))
    assert not asked.ok and "approval" in asked.text
    assert dispatcher.calls == [], "nothing re-runs before the user decides"
    approval = store.pending_approvals(record["id"])[0]
    assert asyncio.run(manager.handle({"action": "approve", "approval": approval["id"]})).ok
    store.transition(record["id"], "interrupted")

    async def scenario():
        result = engine.resume(record["id"])
        run = list(engine.runs.values())[-1]
        await run["task"]
        return result, run

    result, run = asyncio.run(scenario())
    assert result.ok and run["status"] == "complete"
    assert [c[0] for c in dispatcher.calls] == ["companion", "executive"]
    store.close()


def test_declining_the_replay_stops_the_run_there(tmp_path, isolated_jobs):
    steps = [{"tool": "companion", "args": {"action": "brief"}},
             {"tool": "executive", "args": {"action": "goals"}}]
    store, definitions, _ = _crash_mid_workflow(tmp_path, steps, "companion")
    [record] = store.recovered
    engine, dispatcher = _restarted_engine(tmp_path, store, definitions)
    engine.resume(record["id"])
    approval = store.pending_approvals(record["id"])[0]
    store.decide_approval(approval["id"], False)
    store.transition(record["id"], "interrupted")
    result = engine.resume(record["id"])
    assert result.ok and "will not run it again" in result.text
    assert dispatcher.calls == []
    assert store.get(record["id"])["state"] == "cancelled"
    store.close()


def test_a_changed_definition_is_never_resumed(tmp_path, isolated_jobs):
    steps = [{"tool": "research", "args": {"action": "agenda"}},
             {"tool": "executive", "args": {"action": "goals"}}]
    store, definitions, _ = _crash_mid_workflow(tmp_path, steps, "executive")
    [record] = store.recovered
    engine, dispatcher = _restarted_engine(tmp_path, store, definitions)
    engine.definitions["flow"]["steps"].reverse()
    result = engine.resume(record["id"])
    assert not result.ok and "changed" in result.text
    assert dispatcher.calls == []
    store.close()


# ── creator intelligence ──────────────────────────────────────────────────────

def test_hook_analyzer_prefers_strong_hooks():
    analyzer = HookAnalyzer()
    strong, strong_notes = analyzer.score(
        "Stop buying protein powder before you watch this")
    weak, weak_notes = analyzer.score(
        "Hey guys welcome back to my channel today")
    assert strong > weak
    assert any("Weak opener" in n for n in weak_notes)
    assert any("curiosity" in n.lower() for n in strong_notes)


def test_script_evaluator_full_breakdown():
    script = (
        "Stop scrolling — this hook doubled our views. "
        "But here's the thing, most creators bury the point. "
        "We tested five hooks last week and one clear winner emerged. "
        "Follow for part two."
    )
    review = ScriptEvaluator().evaluate(script)
    assert review["ok"]
    assert review["cta"]["score"] == 7.0
    assert review["structure"]["est_seconds"] > 0
    assert 0 <= review["overall"] <= 10


def test_script_evaluator_flags_missing_cta():
    review = ScriptEvaluator().evaluate(
        "This is a product I quite like. It does many things. "
        "It was delivered quickly and the box was fine.")
    assert review["cta"]["score"] == 2.0
    assert "action" in review["priority_fix"].lower()


def test_creator_manager_persists_reviews(tmp_path):
    manager = CreatorManager(path=tmp_path / "creators.json")
    script = "Stop — watch this before you buy. Follow for more."
    manager.review_submission("Alice", script)
    manager.review_submission("Alice", script)
    report = " ".join(manager.tracker.report("Alice"))
    assert "2 review(s)" in report
    stored = json.loads((tmp_path / "creators.json").read_text(encoding="utf-8"))
    assert len(stored["Alice"]["reviews"]) == 2


def test_viral_analysis_finds_patterns():
    lines = ViralAnalysisAgent().analyse([
        "Stop scrolling — you need to see this",
        "Stop buying this until you watch",
        "3 mistakes you make every morning",
    ])
    text = " ".join(lines)
    assert "Curiosity" in text or "Relatability" in text
    assert "'stop' x2" in text


def _suite(tmp_path):
    return CreatorIntelSuite(StubBus(),
                             creators_path=tmp_path / "creators.json")


def test_suite_actions_offline(tmp_path):
    suite = _suite(tmp_path)
    hook = asyncio.run(suite.handle(
        {"action": "hook", "hook": "Stop buying gimbals before you see this"}))
    assert hook.ok and "/10" in hook.text
    review = asyncio.run(suite.handle({
        "action": "review_script", "creator": "Bob",
        "script": "Stop — this tripod broke in a week. But here's the thing, "
                  "the replacement is half the price. Follow for the link.",
    }))
    assert review.ok and "PRIORITY FIX" in review.text
    product = asyncio.run(suite.handle(
        {"action": "product", "product": "posture corrector"}))
    assert product.ok and "Content angles" in product.text
    strategy = asyncio.run(suite.handle({"action": "strategy"}))
    assert "TikTok" in strategy.text
    fit = asyncio.run(suite.handle(
        {"action": "creator_fit",
         "profile": "120k followers, 5.2% engagement, fitness niche"}))
    assert "strong" in fit.text
    ideas = asyncio.run(suite.handle(
        {"action": "hook_ideas", "topic": "standing desk"}))
    assert ideas.text.count("standing desk") >= 3
    performance = asyncio.run(suite.handle({"action": "performance"}))
    assert "Bob" in performance.text
    unknown = asyncio.run(suite.handle({"action": "nonsense"}))
    assert not unknown.ok


def test_suite_viral_accepts_string_examples(tmp_path):
    suite = _suite(tmp_path)
    result = asyncio.run(suite.handle({
        "action": "viral",
        "examples": "Stop scrolling now\n\nStop wasting money on ads",
    }))
    assert result.ok and "2 example(s)" in result.text
