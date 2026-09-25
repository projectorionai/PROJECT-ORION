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
def skills(tmp_path):
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
    engine.define("retry", [{"tool": "research", "args": {}, "retries": 1}])
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
