"""
Tests for context-driven personality tone (Mark XX architectural-audit pass,
Track H): before this, persona_text()/style_capsule() rendered identically
regardless of what the task actually was — one static blanket "adapt your
register" instruction that no code ever acted on, and a documented
style_capsule() that was defined but never called anywhere. This adds an
optional `context` that appends a specific tone directive on top of the
existing blanket rule (additive — a blank/unknown context changes nothing),
and threads a real signal (an agent's own name) into it via providers.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.identity as idm


class _Sig:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, n):
        s = _Sig()
        object.__setattr__(self, n, s)
        return s


def _manager(tmp_path, monkeypatch) -> idm.IdentityManager:
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    return idm.IdentityManager(_Bus())


# ── tone_for_context() ──────────────────────────────────────────────────────

def test_tone_for_context_returns_blank_for_unknown_context():
    assert idm.tone_for_context("") == ""
    assert idm.tone_for_context("something-unrecognised") == ""


def test_tone_for_context_resolves_a_plain_category():
    assert "technical" in idm.tone_for_context("coding")
    assert "analytical" in idm.tone_for_context("research")


def test_tone_for_context_is_case_insensitive():
    assert idm.tone_for_context("Coding") == idm.tone_for_context("coding")


def test_tone_for_context_resolves_a_dotted_task_label():
    # agents.py's real convention: task=f"reason.{self.name}"
    assert idm.tone_for_context("reason.coding") == idm.tone_for_context("coding")
    assert idm.tone_for_context("reason.research") != ""


def test_tone_for_context_dotted_unknown_segment_returns_blank():
    assert idm.tone_for_context("reason.unknown-agent") == ""


# ── persona_text(context=...) ───────────────────────────────────────────────

def test_persona_text_with_no_context_is_unchanged_from_before(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert manager.persona_text() == manager.persona_text(context="")


def test_persona_text_with_a_known_context_appends_the_specific_tone(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    plain = manager.persona_text()
    coding = manager.persona_text(context="coding")
    assert coding != plain
    assert "code-literal" in coding
    assert "code-literal" not in plain
    # additive — the tone is inserted alongside the existing blanket rule,
    # before the standing directives section, not a wholesale replacement.
    tone_pos = coding.index("code-literal")
    directives_pos = coding.index("Standing directives:")
    assert tone_pos < directives_pos


def test_persona_text_still_includes_the_blanket_register_rule(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    coding = manager.persona_text(context="coding")
    assert "Adapt your register to context" in coding


def test_persona_text_with_unknown_context_falls_back_to_unchanged(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert manager.persona_text(context="not-a-real-context") == manager.persona_text()


def test_persona_text_different_contexts_produce_different_personas(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    coding = manager.persona_text(context="coding")
    research = manager.persona_text(context="research")
    entertainment = manager.persona_text(context="entertainment")
    assert coding != research != entertainment


# ── style_capsule(context=...) ──────────────────────────────────────────────

def test_style_capsule_with_no_context_is_unchanged_from_before(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert manager.style_capsule() == manager.style_capsule(context="")


def test_style_capsule_with_a_known_context_appends_the_tone(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    plain = manager.style_capsule()
    fitness = manager.style_capsule(context="fitness")
    assert fitness != plain
    assert fitness.startswith(plain)
    assert "energetic" in fitness
