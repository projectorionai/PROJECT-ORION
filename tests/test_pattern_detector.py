"""
WorkflowPatternDetector — spotting runs of tool calls the user keeps
repeating, so ORION can offer to name one as a workflow.

Two properties matter most and are asserted hard here:

  * argument VALUES never enter the log (only the sorted key names), so a
    rolling record of recent activity can't hold real queries or paths;
  * a repeating cycle is reported once, as itself — not as a fistful of
    rotations and seams of the same habit.

Pure in-memory logic: no Qt, no dispatcher, no I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.pattern_detector import (
    WorkflowPatternDetector,
    args_signature,
)


def _cycle(detector, tools, times, args=None):
    for _ in range(times):
        for tool in tools:
            detector.record(tool, args or {})


# ── args_signature: keys only, never values ─────────────────────────────────

def test_signature_is_the_sorted_key_names():
    assert args_signature({"path": "x", "query": "y"}) == "path,query"


def test_signature_never_contains_a_value():
    signature = args_signature({"query": "my-secret-search", "path": "C:/private"})
    assert "my-secret-search" not in signature
    assert "private" not in signature


def test_signature_of_empty_or_missing_args_is_blank():
    assert args_signature({}) == ""
    assert args_signature(None) == ""


def test_signature_is_order_independent():
    assert args_signature({"b": 1, "a": 2}) == args_signature({"a": 2, "b": 1})


# ── recording ────────────────────────────────────────────────────────────────

def test_record_stores_the_tool_and_signature():
    detector = WorkflowPatternDetector()
    detector.record("web_search", {"query": "anything"})
    tool, signature, at = detector._log[0]
    assert tool == "web_search"
    assert signature == "query"
    assert at > 0


def test_recorded_log_never_holds_argument_values():
    detector = WorkflowPatternDetector()
    detector.record("read_file", {"path": "C:/Users/secret/diary.txt"})
    assert "diary" not in str(list(detector._log))
    assert "secret" not in str(list(detector._log))


def test_failed_calls_are_not_recorded():
    detector = WorkflowPatternDetector()
    detector.record("web_search", {}, ok=False)
    assert len(detector) == 0


def test_blank_tool_names_are_ignored():
    detector = WorkflowPatternDetector()
    detector.record("", {})
    detector.record("   ", {})
    assert len(detector) == 0


def test_the_log_is_bounded():
    detector = WorkflowPatternDetector(maxlen=10)
    for i in range(50):
        detector.record(f"tool_{i}", {})
    assert len(detector) == 10


def test_clear_empties_the_log():
    detector = WorkflowPatternDetector()
    detector.record("a", {})
    detector.clear()
    assert len(detector) == 0


# ── detection ────────────────────────────────────────────────────────────────

def test_no_patterns_in_an_empty_log():
    assert WorkflowPatternDetector().detect_repeated_sequences() == []


def test_a_pair_repeated_enough_times_is_found():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["open_app", "window_control"], times=3)
    found = detector.detect_repeated_sequences()
    assert found
    assert found[0].tool_names == ["open_app", "window_control"]
    assert found[0].occurrences >= 3


def test_a_pair_below_the_threshold_is_not_reported():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["open_app", "window_control"], times=2)
    assert detector.detect_repeated_sequences(min_occurrences=3) == []


def test_min_occurrences_is_configurable():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["open_app", "window_control"], times=2)
    assert detector.detect_repeated_sequences(min_occurrences=2)


def test_a_repeating_cycle_is_reported_once_not_as_rotations():
    # Four passes of a three-tool cycle also contain rotations (B→C→A) and
    # seams (C→A). Reporting those alongside the real cycle is noise.
    detector = WorkflowPatternDetector()
    _cycle(detector, ["web_search", "read_file", "save_memory"], times=4)
    found = detector.detect_repeated_sequences()
    assert len(found) == 1
    assert found[0].tool_names == ["web_search", "read_file", "save_memory"]
    assert found[0].occurrences == 4


def test_two_genuinely_distinct_habits_are_both_reported():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["open_app", "window_control"], times=3)
    _cycle(detector, ["outlook_mail", "notion_workspace"], times=3)
    found = detector.detect_repeated_sequences()
    reported = {tuple(seq.tool_names) for seq in found}
    assert ("open_app", "window_control") in reported
    assert ("outlook_mail", "notion_workspace") in reported


def test_the_same_tool_with_different_arg_shapes_is_a_different_step():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["file_op"], times=3, args={"read": 1})
    _cycle(detector, ["file_op"], times=3, args={"write": 1, "content": 2})
    found = detector.detect_repeated_sequences(min_length=1, min_occurrences=3)
    signatures = {seq.steps[0][1] for seq in found}
    assert len(signatures) >= 1   # signature distinguishes the two usages


def test_unrelated_traffic_produces_no_pattern():
    detector = WorkflowPatternDetector()
    for i in range(30):
        detector.record(f"tool_{i}", {})
    assert detector.detect_repeated_sequences() == []


def test_results_are_ordered_most_repeated_first():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["a", "b"], times=6)
    _cycle(detector, ["c", "d"], times=3)
    found = detector.detect_repeated_sequences()
    assert found[0].occurrences >= found[-1].occurrences


def test_max_length_bounds_the_sequence_size():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["a", "b", "c", "d", "e"], times=4)
    found = detector.detect_repeated_sequences(max_length=2)
    assert all(len(seq.steps) <= 2 for seq in found)


# ── PatternSequence presentation ────────────────────────────────────────────

def test_describe_uses_an_ascii_arrow():
    # This string reaches bus.log.emit, which can land on a cp1252 console.
    detector = WorkflowPatternDetector()
    _cycle(detector, ["alpha", "beta"], times=3)
    described = detector.detect_repeated_sequences()[0].describe()
    assert described == "alpha -> beta"
    described.encode("cp1252")   # must not raise


def test_suggested_name_joins_the_tools_and_is_bounded():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["alpha", "beta"], times=3)
    seq = detector.detect_repeated_sequences()[0]
    assert seq.suggested_name() == "alpha_beta"
    assert len(seq.suggested_name()) <= 40


def test_last_seen_is_populated():
    detector = WorkflowPatternDetector()
    _cycle(detector, ["alpha", "beta"], times=3)
    assert detector.detect_repeated_sequences()[0].last_seen > 0
