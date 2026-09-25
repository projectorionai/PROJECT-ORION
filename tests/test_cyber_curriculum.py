"""
Tests for the cybersecurity curriculum subsystem (Section 11): structure/safety
validation, search, progress, the default-deny target-scope gate, and the honest
"no model retraining" note.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.cyber_curriculum import (
    CURRICULUM_MODULES,
    CERTIFICATION_TIMELINE,
    CyberCurriculum,
    SecurityActivityClass,
    gate_security_action,
    training_capability_note,
    validate_curriculum,
    validate_target_scope,
)


def test_all_twenty_modules_present_and_valid():
    assert len(CURRICULUM_MODULES) == 20
    assert validate_curriculum() == []


def test_offensive_modules_are_gated_and_carry_reminders():
    # Modules 9,10,11,13,14,17 are offensive/lab and MUST require isolation + a
    # legal reminder.
    for num in (9, 10, 11, 13, 14, 17):
        m = next(x for x in CURRICULUM_MODULES if x.number == num)
        assert m.isolation_required, f"module {num} must require isolation"
        assert m.legal_reminder, f"module {num} must carry a legal reminder"
        assert m.authorization_required


def test_malware_module_is_sandbox_only():
    m = next(x for x in CURRICULUM_MODULES if x.number == 13)
    lab = m.labs[0]
    assert "sandbox" in lab.name.lower()
    assert "no execution on the host" in lab.name.lower()


def test_certification_timeline_five_years():
    assert set(CERTIFICATION_TIMELINE) == {f"Year {i}" for i in range(1, 6)}
    assert "OSCP" in CERTIFICATION_TIMELINE["Year 4"]


def test_search_finds_relevant_modules():
    cur = CyberCurriculum()
    hits = cur.search("sql injection database")
    assert hits
    assert any(h["module"] == 5 for h in hits)  # Databases covers SQLi prevention
    assert cur.search("") == []


def test_outline_covers_everything():
    cur = CyberCurriculum()
    outline = cur.outline()
    assert len(outline) == 20
    assert all("safety" in row for row in outline)


def test_progress_tracking_roundtrip(tmp_path):
    cur = CyberCurriculum(progress_path=tmp_path / "progress.json")
    p0 = cur.progress("alice")
    assert p0["completed"] == [] and p0["next_module"] == 1
    cur.mark_complete("alice", 1)
    cur.mark_complete("alice", 2)
    p = cur.progress("alice")
    assert p["completed"] == [1, 2]
    assert p["next_module"] == 3
    assert p["percent"] == 10.0
    with pytest.raises(ValueError):
        cur.mark_complete("alice", 999)


# ── target-scope gate (default-deny) ─────────────────────────────────────────

@pytest.mark.parametrize("target,allowed", [
    ("localhost", True),
    ("127.0.0.1", True),
    ("http://localhost:8080", True),
    ("::1", True),
])
def test_localhost_allowed(target, allowed):
    ok, _ = validate_target_scope(target)
    assert ok is allowed


@pytest.mark.parametrize("target", [
    "example.com", "8.8.8.8", "192.168.1.10", "10.0.0.5", "", "scanme.nmap.org",
])
def test_external_and_private_denied_by_default(target):
    ok, reason = validate_target_scope(target)
    assert ok is False
    assert "deny" in reason.lower() or "denied" in reason.lower() or "unauthor" in reason.lower()


def test_explicit_authorization_allows_target():
    ok, _ = validate_target_scope("10.0.0.5", authorized_scope={"10.0.0.5"})
    assert ok is True


def test_action_gate_by_activity_class():
    # Knowledge/defensive/secure-coding never touch an external system.
    for cls in (SecurityActivityClass.EDUCATION, SecurityActivityClass.DEFENSIVE_ANALYSIS,
                SecurityActivityClass.SECURE_CODING):
        ok, _ = gate_security_action(cls)
        assert ok
    # Authorised lab requires an authorised/loopback target.
    ok_local, _ = gate_security_action(SecurityActivityClass.AUTHORIZED_LAB, "127.0.0.1")
    assert ok_local
    ok_ext, _ = gate_security_action(SecurityActivityClass.AUTHORIZED_LAB, "example.com")
    assert not ok_ext
    # Intrusive real-world actions are always denied.
    ok_intrusive, reason = gate_security_action(SecurityActivityClass.INTRUSIVE, "127.0.0.1")
    assert not ok_intrusive
    assert "not permitted" in reason.lower()


def test_training_note_is_honest():
    note = training_capability_note()
    assert "does not train" in note.lower() or "not train" in note.lower()
    assert "no model weights change" in note.lower()
