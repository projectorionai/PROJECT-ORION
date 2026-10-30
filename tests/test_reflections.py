"""Application observations stay concise, bounded and distinct from reasoning."""

from __future__ import annotations

import pytest

from orion_core.reflections import ReflectionHistory, concise_text, reflection_entry


def test_repeated_delivery_and_history_have_one_bounded_storage():
    history = ReflectionHistory(3)
    for number in range(10):
        assert history.add({"id": number, "text": f"Activity {number}"})
        assert not history.add({"id": number, "text": f"Activity {number}"})
    assert [item.event_id for item in history.entries] == ["9", "8", "7"]
    history.clear()
    assert not history.entries


@pytest.mark.parametrize("value", [None, "", "  ", {"text": ""}, {"text": ["unexpected"]}])
def test_empty_or_malformed_note_does_not_fill_a_panel(value):
    assert reflection_entry(value) is None


def test_source_context_and_evidence_are_explicit_and_bounded():
    entry = reflection_entry({
        "id": 9, "at": "09:02:01", "observed_at": "2026-09-30T08:02:01+00:00",
        "text": "A desktop action was requested.", "kind": "decision",
        "context": {"focus": "Writing", "active_project": "ORION", "recent_conversation": "private"},
        "evidence": ["Desktop · open Calculator"] * 100, "origin": "observation",
    })
    assert entry.heading == "09:02:01 · Activity note"
    assert entry.context == "Focus: Writing · Project: ORION"
    assert "private" not in entry.context
    assert len(entry.evidence) == 4
    assert entry.observed_at == "2026-09-30T08:02:01+00:00"


def test_generated_reflection_is_not_labelled_as_a_verified_action():
    entry = reflection_entry({"text": "Review the desktop result.", "kind": "decision"})
    assert entry.heading == "Action reflection"
    assert entry.origin == "generated_reflection"


def test_untrusted_fields_remain_plain_bounded_text():
    entry = reflection_entry({
        "text": "line\x00one\n" + "long phrase " * 10_000,
        "at": "<script>", "observed_at": "invalid", "id": {"bad": True},
        "context": ["wrong type"], "evidence": "not a list",
    })
    assert len(entry.text) <= 640
    assert entry.text.endswith("…")
    assert entry.at == entry.observed_at == entry.context == ""
    assert entry.evidence == ()
    assert "\x00" not in entry.text


@pytest.mark.parametrize("limit", [True, 0, 201, "50"])
def test_history_rejects_unbounded_or_ambiguous_limits(limit):
    with pytest.raises(ValueError):
        ReflectionHistory(limit)


def test_idless_redelivery_is_deduplicated():
    history = ReflectionHistory()
    note = {"at": "09:00:00", "text": "One observation."}
    assert history.add(note)
    assert not history.add(dict(note))


def test_malformed_evidence_cannot_inject_large_structures():
    entry = reflection_entry({"text": "A note.", "evidence": [None, {}, "valid"]})
    assert entry.evidence == ("valid",)
    assert concise_text({"secret": "not coerced"}) == ""
