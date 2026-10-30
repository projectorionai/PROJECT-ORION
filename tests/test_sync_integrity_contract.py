"""Remote journal validation must preserve immutable data and logical clocks."""

from dataclasses import replace

import pytest

from orion_core.sync.journal import JournalEntry, SyncJournal, value_hash


@pytest.fixture
def journal(tmp_path):
    store = SyncJournal(tmp_path / "journal.db", node_id="local")
    try:
        yield store
    finally:
        store.close()


def entry(payload="value", *, node="peer", clock=1):
    return JournalEntry(node, clock, "notes", "notes", "fact", value_hash(payload), payload)


def test_mismatched_hash_rolls_back_prior_entries_and_clock(journal):
    bad = replace(entry(), value_hash=value_hash("other value"))
    with pytest.raises(ValueError, match="hash"):
        journal.apply_remote([entry("pending", clock=90), bad])
    assert journal.all_entries() == []
    assert journal.record("notes", "notes", "next", "next").lamport_ts == 1


def test_conflicting_identity_rolls_back_the_entire_batch(journal):
    original = journal.record("notes", "notes", "fact", "original")
    conflicting = entry("changed", node=original.node_id, clock=original.lamport_ts)
    with pytest.raises(ValueError, match="conflict"):
        journal.apply_remote([entry("pending", clock=90), conflicting])
    assert [row.payload for row in journal.all_entries()] == ["original"]
    assert journal.record("notes", "notes", "next", "next").lamport_ts == 2


@pytest.mark.parametrize("clock", [-1, 0, True, 1.5, "2", (1 << 63) - 1, 1 << 80])
@pytest.mark.parametrize("as_dict", [False, True])
def test_invalid_clocks_are_not_coerced_into_accepted_entries(journal, clock, as_dict):
    candidate = entry(clock=clock)
    with pytest.raises(ValueError, match="clock"):
        journal.apply_remote([candidate.as_dict() if as_dict else candidate])
    assert journal.all_entries() == []


@pytest.mark.parametrize("field,value", [
    ("node_id", ""), ("node_id", "   "), ("node_id", None),
    ("payload", 123), ("payload", None), ("category", None),
    ("value_hash", None), ("value_hash", ""),
])
def test_malformed_fields_are_rejected(journal, field, value):
    candidate = entry().as_dict()
    candidate[field] = value
    with pytest.raises(ValueError):
        journal.apply_remote([candidate])
    assert journal.all_entries() == []


def test_hash_case_is_canonical_and_exact_replay_is_idempotent(journal):
    candidate = entry()
    assert journal.apply_remote([replace(candidate, value_hash=candidate.value_hash.upper())]) == 1
    assert journal.apply_remote([candidate]) == 0
    assert journal.all_entries()[0].value_hash == value_hash("value")


def test_legacy_segment_without_hash_still_works(journal):
    candidate = entry().as_dict()
    del candidate["value_hash"]
    assert journal.apply_remote([candidate]) == 1
    assert journal.record("notes", "notes", "next", "next").lamport_ts == 2


def test_hash_preserving_scope_change_is_still_an_identity_conflict(journal):
    candidate = entry()
    journal.apply_remote([candidate])
    with pytest.raises(ValueError, match="conflict"):
        journal.apply_remote([replace(candidate, key="different")])
    assert journal.all_entries()[0].key == "fact"


def test_remote_clock_boundary_allows_one_local_tick_and_fails_cleanly_after(journal):
    assert journal.apply_remote([entry(clock=(1 << 63) - 2)]) == 1
    assert journal.record("notes", "notes", "last", "last").lamport_ts == (1 << 63) - 1
    with pytest.raises(ValueError, match="exhausted"):
        journal.record("notes", "notes", "overflow", "overflow")
    assert len(journal.all_entries()) == 2
