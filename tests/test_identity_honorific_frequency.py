"""
Tests for honorific_frequency (Mark XXI pass) — the persona previously told
the model to address the user with the honorific unconditionally
("unless instructed otherwise"), which is why ORION tacked "sir" onto
nearly every sentence. This makes frequency an explicit, configurable
instruction defaulting to "occasional" rather than "always".
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


# ── honorific_instruction() ─────────────────────────────────────────────────

def test_always_addresses_with_the_honorific():
    text = idm.honorific_instruction("sir", "always")
    assert "sir" in text
    assert "most sentences" in text


def test_occasional_restricts_to_specific_moments():
    text = idm.honorific_instruction("sir", "occasional")
    assert "sir" in text
    assert "greetings" in text
    assert "do not" in text.lower()


def test_never_drops_the_honorific_entirely():
    text = idm.honorific_instruction("sir", "never")
    assert "do not use" in text.lower()
    assert "casually" in text.lower()


def test_unknown_frequency_falls_back_to_occasional():
    assert idm.honorific_instruction("sir", "sometimes") == idm.honorific_instruction("sir", "occasional")


def test_blank_frequency_falls_back_to_occasional():
    assert idm.honorific_instruction("sir", "") == idm.honorific_instruction("sir", "occasional")


def test_frequency_is_case_insensitive():
    assert idm.honorific_instruction("sir", "ALWAYS") == idm.honorific_instruction("sir", "always")


def test_a_different_honorific_is_substituted():
    text = idm.honorific_instruction("boss", "always")
    assert "boss" in text
    assert "sir" not in text


# ── default preferences ─────────────────────────────────────────────────────

def test_default_frequency_is_occasional(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert manager.preferences["honorific_frequency"] == "occasional"


def test_old_config_without_the_new_key_gets_the_occasional_default(tmp_path, monkeypatch):
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "identity.json").write_text('{"honorific": "sir"}', encoding="utf-8")
    manager = idm.IdentityManager(_Bus())
    assert manager.preferences["honorific_frequency"] == "occasional"


# ── persona_text() / style_capsule() reflect the preference ────────────────

def test_persona_text_uses_the_occasional_instruction_by_default(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    persona = manager.persona_text()
    assert "greetings" in persona
    assert "unless instructed otherwise" not in persona


def test_persona_text_reflects_a_custom_frequency(tmp_path, monkeypatch):
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "identity.json").write_text(
        '{"honorific": "sir", "honorific_frequency": "never"}', encoding="utf-8")
    manager = idm.IdentityManager(_Bus())
    assert "do not use" in manager.persona_text().lower()


def test_style_capsule_uses_the_occasional_instruction_by_default(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert "occasionally" in manager.style_capsule()


def test_describe_reports_the_frequency(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert manager.describe()["honorific_frequency"] == "occasional"


# ── the owner's name from Preferences ───────────────────────────────────────

def test_the_name_given_in_preferences_is_kept_and_used(tmp_path, monkeypatch):
    """Preferences asked "You are called" and nothing ever read the answer."""
    manager = _manager(tmp_path, monkeypatch)
    assert "asked, in your Preferences" not in manager.persona_text()
    assert manager.set_user_name("  Alex   Morgan ") is True
    assert manager.set_user_name("Alex Morgan") is False      # unchanged
    reopened = _manager(tmp_path, monkeypatch)
    assert reopened.user_name() == "Alex Morgan"
    persona = reopened.persona_text()
    assert "'Alex Morgan'" in persona and "never in every reply" in persona
    assert "listener line says someone else" in persona
    assert reopened.set_user_name("") is True
    assert "user_name" not in _manager(tmp_path, monkeypatch).preferences


def test_the_header_greets_the_owner_by_first_name():
    from datetime import datetime
    from orion_core.gui.core_window import _greeting_words
    assert _greeting_words(datetime(2026, 10, 8, 9)) == "Good morning"
    assert _greeting_words(datetime(2026, 10, 8, 20), "Alex Morgan") == "Good evening, Alex"
