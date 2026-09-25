"""
Tests for improvement #29 (predictive resource modelling) and #27 (bounded
self-repair autonomy).

* ResourceForecaster: a rising trend yields a finite time-to-threshold and a
  preventative warning inside the horizon; flat/receding trends stay silent;
  warnings are rate-limited per metric.
* ResourceMonitor.observe queues a predictive notification before any level
  engages.
* SelfRepairAgent.attempt_auto_repair: dependency installs are autonomous;
  source fixes auto-apply ONLY with green tests and a small diff outside the
  denylist; a crash loop gets one attempt; the session cap holds.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.resource_monitor import (
    ResourceForecaster,
    ResourceMonitor,
    ResourceReading,
)
from orion_core.selfrepair import Incident, SelfRepairAgent


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_rising_trend_projects_time_to_threshold():
    clock = _Clock()
    fc = ResourceForecaster(clock=clock)
    for i in range(15):                       # +1 %-point per minute from 50%
        fc.observe("ram", 50.0 + i, ts=clock.now + i * 60.0)
    eta = fc.seconds_until("ram", 95.0)
    assert eta is not None
    assert 1700 < eta < 2000                  # ~31 minutes to 95% from 64%


def test_flat_or_receding_trend_is_silent():
    clock = _Clock()
    fc = ResourceForecaster(clock=clock)
    for i in range(20):
        fc.observe("cpu", 40.0, ts=clock.now + i * 60.0)
        fc.observe("ram", 80.0 - i, ts=clock.now + i * 60.0)
    assert fc.seconds_until("cpu", 95.0) is None
    assert fc.seconds_until("ram", 95.0) is None
    assert fc.warnings({"cpu": 95.0, "ram": 95.0}) == []


def test_warning_is_rate_limited_per_metric():
    clock = _Clock()
    fc = ResourceForecaster(clock=clock)
    for i in range(15):
        fc.observe("ram", 60.0 + i, ts=clock.now + i * 60.0)
    clock.now += 15 * 60.0
    assert len(fc.warnings({"ram": 95.0})) == 1
    assert fc.warnings({"ram": 95.0}) == []               # quiet period holds
    clock.now += ResourceForecaster.REWARN_S + 1
    fc.observe("ram", 88.0, ts=clock.now)                 # climb continues
    assert len(fc.warnings({"ram": 95.0})) == 1           # re-warns after it


def test_monitor_queues_predictive_notification_before_breach():
    clock = _Clock()
    monitor = ResourceMonitor(clock=clock)
    for i in range(15):
        clock.now += 60.0
        assessment = monitor.observe(ResourceReading(
            cpu_percent=20.0, mem_percent=60.0 + i, timestamp=clock.now))
        assert assessment.level.value == "nominal"        # nothing breached yet
    notes = monitor.drain_notifications()
    predictive = [n for n in notes if n["level"] == "predictive"]
    assert predictive and predictive[0]["affected"] == ["ram"]
    assert "trending towards" in predictive[0]["message"]


# ── self-repair autonomy ──────────────────────────────────────────────────────

class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, *_a, **_k):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _agent():
    return SelfRepairAgent(_StubBus(), None, router=None)


def _incident(**overrides):
    base = dict(id="inc-1", at="now", error_type="ValueError",
                message="boom", file="", line=1)
    base.update(overrides)
    return Incident(**base)


def test_diff_lines_counts_changes():
    original = "a\nb\nc\n"
    patched = "a\nB\nc\nd\n"
    assert SelfRepairAgent._diff_lines(original, patched) == 3   # -b +B +d


def test_missing_dependency_installs_autonomously(monkeypatch, tmp_path):
    installed = []

    class _Resolver:
        def __init__(self, bus):
            pass

        async def resolve_and_install(self, requirements):
            installed.append(list(requirements))
            return type("O", (), {"succeeded": True})()

    import orion_core.dependencies as deps
    monkeypatch.setattr(deps, "DynamicPackageResolver", _Resolver)
    import orion_core.selfrepair as sr
    monkeypatch.setattr(sr, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    agent = _agent()
    incident = _incident(error_type="ModuleNotFoundError",
                         message="No module named 'lxml'")
    asyncio.run(agent.attempt_auto_repair(incident))
    assert installed == [["lxml"]]
    assert agent._auto_repairs == 1


def test_source_fix_applies_only_with_green_tests(monkeypatch, tmp_path):
    import orion_core.selfrepair as sr
    monkeypatch.setattr(sr, "PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(sr, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    target = tmp_path / "pkg" / "widget.py"
    target.parent.mkdir()
    target.write_text("x = 1\ny = 2\n", encoding="utf-8")

    agent = _agent()
    applied = []
    agent._run_patched_tests = lambda t, c: (True, "test_widget.py: 3 passed")
    from orion_core.data import ToolResult
    agent._apply_repaired = lambda inc, t: (applied.append(t), ToolResult("ok"))[1]
    incident = _incident(file=str(target), repaired_content="x = 1\ny = 3\n")
    asyncio.run(agent.attempt_auto_repair(incident))
    assert applied == [target]
    assert agent._auto_repairs == 1


def test_source_fix_declined_when_tests_fail_or_absent(monkeypatch, tmp_path):
    import orion_core.selfrepair as sr
    monkeypatch.setattr(sr, "PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(sr, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    target = tmp_path / "pkg" / "widget.py"
    target.parent.mkdir()
    target.write_text("x = 1\n", encoding="utf-8")

    for verdict in (False, None):             # failing tests OR no tests
        agent = _agent()
        agent._run_patched_tests = lambda t, c, v=verdict: (v, "detail")
        agent._apply_repaired = lambda inc, t: (_ for _ in ()).throw(
            AssertionError("must not apply"))
        incident = _incident(file=str(target), repaired_content="x = 2\n")
        asyncio.run(agent.attempt_auto_repair(incident))
        assert agent._auto_repairs == 0


def test_denylisted_and_oversized_fixes_stay_gated(monkeypatch, tmp_path):
    import orion_core.selfrepair as sr
    monkeypatch.setattr(sr, "PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(sr, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    pkg = tmp_path / "pkg"
    pkg.mkdir()

    # security.py is denylisted no matter how good the fix looks.
    sec = pkg / "security.py"
    sec.write_text("x = 1\n", encoding="utf-8")
    agent = _agent()
    agent._apply_repaired = lambda inc, t: (_ for _ in ()).throw(
        AssertionError("must not apply"))
    asyncio.run(agent.attempt_auto_repair(
        _incident(file=str(sec), repaired_content="x = 2\n")))
    assert agent._auto_repairs == 0

    # A sweeping rewrite exceeds the diff budget.
    big = pkg / "widget.py"
    big.write_text("\n".join(f"line{i} = {i}" for i in range(30)), encoding="utf-8")
    agent2 = _agent()
    agent2._run_patched_tests = lambda t, c: (True, "green")
    agent2._apply_repaired = lambda inc, t: (_ for _ in ()).throw(
        AssertionError("must not apply"))
    rewrite = "\n".join(f"line{i} = {i * 2}" for i in range(30))
    asyncio.run(agent2.attempt_auto_repair(
        _incident(file=str(big), repaired_content=rewrite)))
    assert agent2._auto_repairs == 0


def test_crash_loop_gets_one_attempt_and_cap_holds(monkeypatch, tmp_path):
    installed = []

    class _Resolver:
        def __init__(self, bus):
            pass

        async def resolve_and_install(self, requirements):
            installed.append(list(requirements))
            return type("O", (), {"succeeded": True})()

    import orion_core.dependencies as deps
    monkeypatch.setattr(deps, "DynamicPackageResolver", _Resolver)
    import orion_core.selfrepair as sr
    monkeypatch.setattr(sr, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    agent = _agent()
    incident = _incident(error_type="ModuleNotFoundError",
                         message="No module named 'lxml'")
    for _ in range(3):                        # same signature crashes again
        asyncio.run(agent.attempt_auto_repair(incident))
    assert installed == [["lxml"]]            # exactly one attempt

    agent._auto_repairs = SelfRepairAgent.MAX_AUTO_REPAIRS_PER_SESSION
    other = _incident(id="inc-2", error_type="ModuleNotFoundError",
                      message="No module named 'httpx'", line=9)
    asyncio.run(agent.attempt_auto_repair(other))
    assert installed == [["lxml"]]            # cap blocks further autonomy
