"""
Wiring for the workflow pattern detector: the dispatcher feeding it, the
proactive survey surfacing it, and the workflow_patterns tool reporting it
on demand.

The two things most worth guarding here are contract boundaries, not logic:
recent_tools (which two live GUI panels read) must keep its existing shape,
and a detected pattern must arrive as a low-key *question* — never spoken,
never something ORION claims to have already done.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import deque

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher
from orion_core.pattern_detector import PatternSequence, WorkflowPatternDetector
from orion_core.proactive import ProactiveIntelligence


class _Signal:
    def __init__(self, sink=None):
        self._sink = sink

    def emit(self, *a, **k):
        if self._sink is not None:
            self._sink.append((a, k))

    def connect(self, *a, **k):
        pass


class _RecordingBus:
    def __init__(self):
        self.log_calls = []
        self.speak_calls = []
        self.banner_calls = []
        self.dashboard_calls = []

    def __getattr__(self, name):
        sink = {
            "log": self.log_calls,
            "speak_request": self.speak_calls,
            "banner": self.banner_calls,
            "dashboard_event": self.dashboard_calls,
        }.get(name)
        signal = _Signal(sink)
        object.__setattr__(self, name, signal)
        return signal


def _dispatcher(detector=None, registries=None) -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _RecordingBus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    d.pattern_detector = detector
    d.registries = registries
    return d


def _working_dispatcher(detector=None) -> OrionDispatcher:
    """A dispatcher whose `capabilities` tool actually succeeds — the
    detector deliberately ignores failed calls, so a tool that errors would
    record nothing."""
    from orion_core.registries import SystemRegistries
    return _dispatcher(detector, registries=SystemRegistries())


# ── dispatcher feeds the detector ───────────────────────────────────────────

async def test_a_dispatched_tool_is_recorded_once():
    detector = WorkflowPatternDetector()
    d = _working_dispatcher(detector)
    await d.dispatch("capabilities", {})
    assert len(detector) == 1
    assert detector._log[0][0] == "capabilities"


async def test_dispatch_records_the_argument_keys_not_values():
    detector = WorkflowPatternDetector()
    d = _working_dispatcher(detector)
    await d.dispatch("capabilities", {"query": "something-private"})
    assert detector._log[0][1] == "query"
    assert "something-private" not in str(list(detector._log))


async def test_a_failing_tool_is_not_recorded_as_a_pattern_step():
    # A sequence that errored out isn't a workflow worth suggesting.
    detector = WorkflowPatternDetector()
    d = _dispatcher(detector)          # no registries -> capabilities fails
    result = await d.dispatch("capabilities", {})
    assert not result.ok
    assert len(detector) == 0


async def test_recent_tools_keeps_its_existing_shape():
    # Two live GUI panels read this deque — its fields are a contract.
    detector = WorkflowPatternDetector()
    d = _working_dispatcher(detector)
    await d.dispatch("capabilities", {})
    entry = d.recent_tools[-1]
    assert set(entry) == {"tool", "ok", "ms", "at"}
    assert entry["tool"] == "capabilities"


async def test_dispatch_works_with_no_detector_attached():
    d = _working_dispatcher(None)
    result = await d.dispatch("capabilities", {})
    assert isinstance(result, ToolResult)
    assert len(d.recent_tools) == 1


async def test_a_broken_detector_never_breaks_a_dispatch():
    class _Exploding:
        def record(self, *a, **k):
            raise RuntimeError("detector exploded")

    d = _working_dispatcher(_Exploding())
    result = await d.dispatch("capabilities", {})
    assert isinstance(result, ToolResult)
    assert result.ok


# ── the workflow_patterns tool ──────────────────────────────────────────────

def _tool_dispatcher(detector):
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.pattern_detector = detector
    return d


def test_tool_reports_unavailable_with_no_detector():
    result = _tool_dispatcher(None).workflow_patterns_tool({})
    assert not result.ok


def test_tool_reports_nothing_noticed_yet():
    result = _tool_dispatcher(WorkflowPatternDetector()).workflow_patterns_tool({})
    assert result.ok
    assert "no repeated tool sequences" in result.text.lower()


def test_tool_lists_a_detected_sequence():
    detector = WorkflowPatternDetector()
    for _ in range(3):
        detector.record("open_app", {})
        detector.record("window_control", {})
    result = _tool_dispatcher(detector).workflow_patterns_tool({})
    assert result.ok
    assert "open_app -> window_control" in result.text
    assert "3 times" in result.text


def test_tool_is_explicit_that_arguments_are_not_retained():
    detector = WorkflowPatternDetector()
    for _ in range(3):
        detector.record("open_app", {})
        detector.record("window_control", {})
    result = _tool_dispatcher(detector).workflow_patterns_tool({})
    assert "don't retain" in result.text


def test_tool_honours_a_custom_min_occurrences():
    detector = WorkflowPatternDetector()
    for _ in range(2):
        detector.record("a", {})
        detector.record("b", {})
    strict = _tool_dispatcher(detector).workflow_patterns_tool({})
    lenient = _tool_dispatcher(detector).workflow_patterns_tool({"min_occurrences": 2})
    assert "no repeated" in strict.text.lower()
    assert "a -> b" in lenient.text


def test_tool_survives_a_broken_detector():
    class _Exploding:
        def detect_repeated_sequences(self, **_k):
            raise RuntimeError("boom")

    result = _tool_dispatcher(_Exploding()).workflow_patterns_tool({})
    assert not result.ok


def test_tool_is_registered_in_the_handler_table():
    d = _dispatcher()
    assert d.handler_table()["workflow_patterns"] == d.workflow_patterns_tool


# ── the proactive suggestion ────────────────────────────────────────────────

def _proactive(detector, bus=None):
    return ProactiveIntelligence(
        bus or _RecordingBus(), outlook=None, notion=None, memory=None,
        pattern_detector=detector)


def test_no_suggestion_without_a_detector():
    assert asyncio.run(_proactive(None)._check_workflow_patterns()) == []


def test_no_suggestion_when_nothing_repeats():
    proactive = _proactive(WorkflowPatternDetector())
    assert asyncio.run(proactive._check_workflow_patterns()) == []


def test_a_detected_pattern_becomes_a_suggestion():
    detector = WorkflowPatternDetector()
    for _ in range(3):
        detector.record("open_app", {})
        detector.record("window_control", {})
    suggestions = asyncio.run(_proactive(detector)._check_workflow_patterns())
    assert len(suggestions) == 1
    assert "open_app -> window_control" in suggestions[0].text


def test_the_suggestion_asks_rather_than_announces_an_action():
    detector = WorkflowPatternDetector()
    for _ in range(3):
        detector.record("open_app", {})
        detector.record("window_control", {})
    text = asyncio.run(_proactive(detector)._check_workflow_patterns())[0].text
    assert text.rstrip().endswith("?"), "a suggestion must be a question"
    assert "shall i" in text.lower()
    for claim in ("i saved", "i've saved", "i created", "i've created"):
        assert claim not in text.lower()


def test_the_suggestion_is_low_salience_so_it_is_never_spoken():
    detector = WorkflowPatternDetector()
    for _ in range(3):
        detector.record("open_app", {})
        detector.record("window_control", {})
    suggestion = asyncio.run(_proactive(detector)._check_workflow_patterns())[0]
    assert suggestion.salience == 1


def test_a_full_survey_never_speaks_the_suggestion():
    detector = WorkflowPatternDetector()
    for _ in range(3):
        detector.record("open_app", {})
        detector.record("window_control", {})
    bus = _RecordingBus()
    proactive = _proactive(detector, bus)
    asyncio.run(proactive._survey(announce=True))
    assert bus.speak_calls == [], "proactive nudges are never spoken unprompted"
    assert bus.banner_calls == [], "salience 1 must stay out of the banner"
    assert bus.dashboard_calls, "it should still reach the dashboard channel"


def test_a_broken_detector_does_not_break_the_survey():
    class _Exploding:
        def detect_repeated_sequences(self, **_k):
            raise RuntimeError("boom")

    assert asyncio.run(_proactive(_Exploding())._check_workflow_patterns()) == []


def test_at_most_two_patterns_are_suggested_at_once():
    detector = WorkflowPatternDetector()
    for tools in (("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")):
        for _ in range(3):
            for tool in tools:
                detector.record(tool, {})
    suggestions = asyncio.run(_proactive(detector)._check_workflow_patterns())
    assert len(suggestions) <= 2, "don't dump every finding at once"


def test_suggestion_keys_are_stable_for_deduplication():
    detector = WorkflowPatternDetector()
    for _ in range(3):
        detector.record("open_app", {})
        detector.record("window_control", {})
    proactive = _proactive(detector)
    first = asyncio.run(proactive._check_workflow_patterns())[0]
    second = asyncio.run(proactive._check_workflow_patterns())[0]
    assert first.key == second.key
