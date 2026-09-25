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
