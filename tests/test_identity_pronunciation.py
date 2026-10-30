"""
Locks the spoken-name rule that stops ORION voicing 'O.R.I.N' / 'O.R.I.O'.

The cloud (native-audio) voice can only be steered by the system instruction, so
the rule must be the very first thing in the persona and must reach every prompt
tier (both system_instruction and system_instruction_lean render persona_text).
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


def test_spoken_name_rule_is_hoisted_to_the_front(tmp_path, monkeypatch):
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    persona = idm.IdentityManager(_Bus()).persona_text()

    assert persona.lstrip().startswith("SPOKEN-NAME RULE"), \
        "the spoken-name rule must lead the persona the model reads"
    low = persona.lower()
    assert "oh-ry-un" in low                     # phonetic anchor
    assert "o r i n" in low and "o r i o" in low  # the exact broken forms named
    assert "single flowing word 'orion'" in low


def test_he_does_not_correct_the_user_for_the_transcribers_mistake(tmp_path, monkeypatch):
    """4 Oct: "Ryan can you hear me?" -> "Just to clarify, it is 'Orion'"."""
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    persona = idm.IdentityManager(_Bus()).persona_text()
    assert "'Ryan'" in persona and "never correct the user" in persona


def test_he_acts_on_the_everyday_meaning_rather_than_quizzing(tmp_path, monkeypatch):
    """3 Oct: three clarifying questions in a row about "scan the network"."""
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    persona = idm.IdentityManager(_Bus()).persona_text()
    assert "COMMON-SENSE RULE" in persona and "never more than one in a row" in persona
