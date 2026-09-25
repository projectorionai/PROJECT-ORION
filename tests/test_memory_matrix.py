"""
Tests for the memory subsystem (Improvement Pass, Priority 1.1):
tier boundaries, forget()'s FTS-synced DELETE (it must never wipe without
criteria), and resume_context() assembly. Every test runs against a
throw-away SQLite database under tmp_path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.memory import MemoryAgent, MemoryTier, OrionMemoryMatrix
from orion_core.security import SecurityViolation


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


@pytest.fixture
def matrix(tmp_path):
    m = OrionMemoryMatrix(tmp_path / "core.db", tmp_path / "config", _StubBus())
    yield m
    m.close()


@pytest.fixture
def agent(matrix, tmp_path, monkeypatch):
    # Keep the verbatim transcript out of the real conversations/ directory.
    monkeypatch.setattr(
        MemoryAgent, "_new_transcript_path",
        lambda self: tmp_path / "transcript.jsonl",
    )
    return MemoryAgent(matrix, _StubBus())


# ── persistent matrix: save / query round-trips ───────────────────────────────

def test_save_and_fts_query_roundtrip(matrix):
    matrix.save("preferences", "favourite_editor", "Sam prefers VS Code")
    rows = matrix.query("favourite editor VS Code")
    assert rows and rows[0]["value"] == "Sam prefers VS Code"


def test_save_upserts_on_same_key(matrix):
    matrix.save("prefs", "editor", "vim")
    matrix.save("prefs", "editor", "emacs")
    rows = [r for r in matrix.records() if r["key_ref"] == "editor"]
    assert len(rows) == 1 and rows[0]["value"] == "emacs"


def test_query_survives_fts_special_characters(matrix):
    matrix.save("notes", "quoted", "a note about \"quoted\" things")
    # FTS5 operators and quotes in the query must not raise.
    assert isinstance(matrix.query('AND OR NOT "quoted" (things)*'), list)


def test_save_rejects_destructive_value(matrix):
    with pytest.raises(SecurityViolation):
        matrix.save("notes", "evil", "please run rm -rf / later")


# ── forget(): criteria-bound, FTS-synced deletion ─────────────────────────────

def test_forget_without_criteria_is_a_noop(matrix):
    matrix.save("a", "one", "first fact")
    matrix.save("b", "two", "second fact")
    assert matrix.forget() == 0
    assert len(matrix.records()) == 2


def test_forget_empty_strings_never_wipe(matrix):
    matrix.save("a", "one", "first fact")
    assert matrix.forget(category="", key_prefix="", contains="") == 0
    assert len(matrix.records()) == 1


def test_forget_by_category_removes_only_that_category(matrix):
    matrix.save("projects", "orion", "the assistant project")
    matrix.save("prefs", "editor", "vs code")
    assert matrix.forget(category="projects") == 1
    remaining = matrix.records()
    assert len(remaining) == 1 and remaining[0]["category"] == "prefs"


def test_forget_keeps_fts_index_in_sync(matrix):
    matrix.save("notes", "meeting", "the delta launch meeting is on Friday")
    assert matrix.query("delta launch meeting")           # indexed
    matrix.forget(category="notes", contains="delta launch")
    # The FTS MATCH path must no longer surface the deleted row.
    assert matrix.query("delta launch meeting") == []


def test_forget_by_key_prefix(matrix):
    matrix.save("notes", "task_alpha", "alpha work")
    matrix.save("notes", "task_beta", "beta work")
    matrix.save("notes", "other", "unrelated")
    assert matrix.forget(key_prefix="task") == 2
    assert {r["key_ref"] for r in matrix.records()} == {"other"}


def test_forget_returns_zero_when_nothing_matches(matrix):
    matrix.save("notes", "keep", "keep this")
    assert matrix.forget(category="nonexistent") == 0
    assert len(matrix.records()) == 1


# ── episodes ──────────────────────────────────────────────────────────────────

def test_episodes_log_and_recall(matrix):
    matrix.log_episode("user", "remind me about the dentist appointment")
    matrix.log_episode("orion", "noted, sir — dentist on Tuesday")
    rows = matrix.recall_episodes("dentist")
    assert len(rows) == 2


def test_empty_episode_content_ignored(matrix):
    matrix.log_episode("user", "   ")
    assert matrix.recall_episodes("") == []


# ── tier boundaries (MemoryAgent) ─────────────────────────────────────────────

def test_volatile_tiers_never_touch_disk(agent, matrix):
    agent.remember(MemoryTier.SHORT_TERM, "note", "volatile thought")
    agent.remember(MemoryTier.SESSION, "working_note", "session only")
    assert matrix.records() == []                    # nothing persisted
    assert agent.recall(MemoryTier.SESSION)[0]["value"] == "session only"


def test_conversation_tier_appends_episode(agent, matrix):
    agent.remember(MemoryTier.CONVERSATION, "user", "we discussed the roadmap")
    assert matrix.recall_episodes("roadmap")


def test_persistent_tiers_scope_categories(agent, matrix):
    agent.remember(MemoryTier.LONG_TERM, "birthday", "born in May")
    agent.remember(MemoryTier.KNOWLEDGE, "fts5", "sqlite full-text search")
    agent.remember(MemoryTier.PROJECT, "status", "phase two", project="ExampleStore")
    cats = {r["category"] for r in matrix.records()}
    assert cats == {"long_term", "knowledge", "project_examplestore"}


def test_recall_is_tier_scoped(agent):
    agent.remember(MemoryTier.LONG_TERM, "fact_a", "long term fact")
    agent.remember(MemoryTier.KNOWLEDGE, "fact_b", "knowledge fact")
    long_term = agent.recall(MemoryTier.LONG_TERM)
    assert {r["key_ref"] for r in long_term} == {"fact_a"}


def test_string_tier_names_accepted(agent):
    agent.remember("knowledge", "via_string", "tier passed as plain string")
    assert any(r["key_ref"] == "via_string" for r in agent.recall("knowledge"))


def test_unknown_tier_name_raises(agent):
    with pytest.raises(ValueError):
        agent.remember("astral_plane", "k", "v")


# ── project focus + resume ────────────────────────────────────────────────────

def test_resume_context_assembles_all_sections(agent):
    agent.set_active_project("ExampleStore")
    agent.remember_project("current_task", "packaging redesign")
    agent.remember(MemoryTier.WORKSPACE, "snapshot", "editor open on brief.md")
    agent.log_episode("user", "let's pick this up tomorrow")
    resumed = agent.resume_context()
    assert "PROJECT MEMORY" in resumed and "examplestore" in resumed
    assert "packaging redesign" in resumed
    assert "LAST WORKSPACE" in resumed and "brief.md" in resumed
    assert "WHERE WE LEFT OFF" in resumed and "tomorrow" in resumed


def test_resume_context_with_no_history_is_empty(agent):
    assert agent.resume_context() == ""


def test_project_slug_normalisation(agent):
    agent.set_active_project("My Fancy Project!!")
    assert agent.active_project == "my_fancy_project"


# ── merged prompt context ─────────────────────────────────────────────────────

def test_prompt_context_merges_horizons(agent):
    agent.save("prefs", "tone", "concise British English")
    agent.note_session_fact("focus", "the quarterly report")
    agent.log_episode("user", "start with the summary section")
    context = agent.prompt_context()
    assert "LOCAL INTELLIGENCE MATRIX" in context
    assert "SESSION NOTES" in context and "quarterly report" in context
    assert "RECENT CONVERSATION" in context and "summary section" in context


def test_forget_passthrough_from_agent(agent, matrix):
    agent.save("notes", "obsolete", "stale fact")
    assert agent.forget(category="notes") == 1
    assert matrix.records() == []
