"""
Tests for SelfRepairAgent's approval gate (Improvement Pass, Priority 1.1).

The contract under guard:
  • no code path may write to source without an explicit confirm=True;
  • a repair that fails compile-validation is never offered / is auto-reverted;
  • only files inside the (patched) package directory are repairable;
  • revert_last restores the pre-repair backup.

PACKAGE_DIR, the journal and the backup directory are all redirected to
tmp_path, so the real orion_core source is untouchable from these tests.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.selfrepair as selfrepair_module
from orion_core.selfrepair import Incident, SelfRepairAgent

ORIGINAL_SOURCE = "VALUE = 1\n\n\ndef helper():\n    return VALUE\n"
FIXED_SOURCE = "VALUE = 2\n\n\ndef helper():\n    return VALUE\n"
BROKEN_SOURCE = "def broken(:\n    pass\n"


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


class _NoProviderRouter:
    def has_text_fallback(self):
        return False


class _ScriptedRouter:
    """Returns a scripted 'repaired file' as the model output."""

    def __init__(self, answer):
        self.answer = answer

    def has_text_fallback(self):
        return True

    async def generate_text(self, prompt, system_extra=""):
        profile = type("P", (), {"name": "scripted"})()
        return profile, self.answer


@pytest.fixture
def repair_env(tmp_path, monkeypatch):
    """An agent whose package dir, journal and backups all live in tmp_path."""
    pkg = (tmp_path / "pkg").resolve()
    pkg.mkdir()
    target = pkg / "fake_module.py"
    target.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    repair_dir = tmp_path / "self_repair"
    monkeypatch.setattr(selfrepair_module, "SELF_REPAIR_DIR", repair_dir)
    monkeypatch.setattr(selfrepair_module, "JOURNAL_PATH", repair_dir / "journal.jsonl")
    monkeypatch.setattr(selfrepair_module, "PACKAGE_DIR", pkg)
    monkeypatch.setattr(selfrepair_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(SelfRepairAgent, "BACKUP_DIR", repair_dir / "backups")
    (tmp_path / "tests").mkdir()                 # test root for the P3.3 run

    agent = SelfRepairAgent(_StubBus(), telemetry=None, router=_NoProviderRouter())
    return agent, target


# The patched module exposes VALUE=2; a test asserting that passes only when the
# fix is applied to the isolated copy. `pkg` is the copied package's dir name.
_PASSING_TEST = (
    "import sys\nfrom pathlib import Path\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
    "from pkg.fake_module import VALUE\n\n"
    "def test_value():\n    assert VALUE == 2\n"
)
_FAILING_TEST = (
    "import sys\nfrom pathlib import Path\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
    "from pkg.fake_module import VALUE\n\n"
    "def test_value():\n    assert VALUE == 999\n"
)


def _incident_for(agent, target, repaired=""):
    incident = Incident(
        id="inc-t", at="now", error_type="ValueError", message="boom",
        file=str(target), line=1, module=target.stem,
        traceback="Traceback...\nValueError: boom",
        repaired_content=repaired,
    )
    agent._incidents[incident.id] = incident
    return incident


# ── the approval gate ─────────────────────────────────────────────────────────

def test_unconfirmed_repair_never_touches_source(repair_env):
    agent, target = repair_env
    _incident_for(agent, target, repaired=FIXED_SOURCE)
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert result.ok
    assert "Nothing is applied" in result.text
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE


def test_default_confirm_is_false(repair_env):
    agent, target = repair_env
    _incident_for(agent, target, repaired=FIXED_SOURCE)
    asyncio.run(agent.repair_file("inc-t"))        # no confirm argument at all
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE


def test_confirmed_repair_applies_with_backup(repair_env):
    agent, target = repair_env
    incident = _incident_for(agent, target, repaired=FIXED_SOURCE)
    result = asyncio.run(agent.repair_file("inc-t", confirm=True))
    assert result.ok
    assert target.read_text(encoding="utf-8") == FIXED_SOURCE
    assert incident.applied is True
    backup = Path(incident.backup_path)
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == ORIGINAL_SOURCE


def test_failed_compile_on_apply_auto_reverts(repair_env):
    agent, target = repair_env
    incident = _incident_for(agent, target, repaired=BROKEN_SOURCE)
    result = asyncio.run(agent.repair_file("inc-t", confirm=True))
    assert not result.ok and "reverted" in result.text
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE
    assert incident.applied is False


def test_revert_last_restores_backup(repair_env):
    agent, target = repair_env
    incident = _incident_for(agent, target, repaired=FIXED_SOURCE)
    asyncio.run(agent.repair_file("inc-t", confirm=True))
    assert target.read_text(encoding="utf-8") == FIXED_SOURCE
    result = agent.revert_last()
    assert result.ok
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE
    assert incident.applied is False


def test_revert_with_no_applied_repair_is_benign(repair_env):
    agent, _target = repair_env
    result = agent.revert_last()
    assert "no applied repair" in result.text.lower()


# ── scope: only package files are repairable ──────────────────────────────────

def test_file_outside_package_refused_even_with_confirm(repair_env, tmp_path):
    agent, _target = repair_env
    outside = tmp_path / "elsewhere" / "victim.py"
    outside.parent.mkdir()
    outside.write_text("x = 1\n", encoding="utf-8")
    _incident_for(agent, outside, repaired="x = 2\n")
    result = asyncio.run(agent.repair_file("inc-t", confirm=True))
    assert not result.ok and "only repair files inside" in result.text
    assert outside.read_text(encoding="utf-8") == "x = 1\n"


def test_missing_incident_refused(repair_env):
    agent, _target = repair_env
    result = asyncio.run(agent.repair_file("no-such-id", confirm=True))
    assert not result.ok


# ── draft generation guards ───────────────────────────────────────────────────

def test_no_provider_means_no_draft_and_no_mutation(repair_env):
    agent, target = repair_env
    _incident_for(agent, target)                    # no repaired_content yet
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert not result.ok and "no language model" in result.text.lower()
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE


def test_non_compiling_model_output_never_offered(repair_env):
    agent, target = repair_env
    agent.router = _ScriptedRouter(BROKEN_SOURCE)
    incident = _incident_for(agent, target)
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert not result.ok and "does not compile" in result.text
    assert incident.repaired_content == ""          # never stored as a draft
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE


def test_fenced_model_output_is_unwrapped(repair_env):
    agent, target = repair_env
    agent.router = _ScriptedRouter(f"Here you go:\n```python\n{FIXED_SOURCE}```\n")
    incident = _incident_for(agent, target)
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert result.ok
    assert incident.repaired_content.strip() == FIXED_SOURCE.strip()
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE


def test_oversized_file_refuses_full_rewrite(repair_env):
    agent, target = repair_env
    target.write_text("x = 1  # padding\n" * 3000, encoding="utf-8")   # > 40 kB
    agent.router = _ScriptedRouter(FIXED_SOURCE)
    _incident_for(agent, target)
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert not result.ok and "diff" in result.text.lower()


# ── capture + journal ─────────────────────────────────────────────────────────

def test_capture_records_incident_without_applying(repair_env):
    agent, _target = repair_env
    try:
        raise ValueError("synthetic fault")
    except ValueError as exc:
        incident = agent.capture(type(exc), exc, exc.__traceback__)
    assert incident.error_type == "ValueError"
    assert incident.applied is False
    assert agent.latest() is incident


def test_journal_survives_apply_and_revert(repair_env):
    agent, target = repair_env
    _incident_for(agent, target, repaired=FIXED_SOURCE)
    asyncio.run(agent.repair_file("inc-t", confirm=True))
    agent.revert_last()
    events = [entry["event"] for entry in agent.journal_tail()]
    assert "repaired" in events and "reverted" in events


def test_propose_fix_writes_proposal_but_never_source(repair_env):
    agent, target = repair_env
    _incident_for(agent, target)
    result = asyncio.run(agent.propose_fix("inc-t"))
    assert result.ok
    assert "No change has been applied" in result.text
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE


# ── P3.3: isolated pytest run against the patched copy before approval ─────────

def test_preview_runs_module_tests_and_reports_pass(repair_env, tmp_path):
    agent, target = repair_env
    (tmp_path / "tests" / "test_fake_module.py").write_text(_PASSING_TEST, encoding="utf-8")
    incident = _incident_for(agent, target, repaired=FIXED_SOURCE)
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert result.ok
    assert incident.tests_passed is True
    assert "PASS" in result.text
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE   # still not applied


def test_preview_reports_test_failure_without_applying(repair_env, tmp_path):
    agent, target = repair_env
    (tmp_path / "tests" / "test_fake_module.py").write_text(_FAILING_TEST, encoding="utf-8")
    incident = _incident_for(agent, target, repaired=FIXED_SOURCE)
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    # The draft still "succeeds" as a proposal, but the verdict flags the failure
    # and nothing is applied — the human decides with full information.
    assert incident.tests_passed is False
    assert "FAIL" in result.text
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE


def test_preview_degrades_when_no_tests_match(repair_env):
    agent, target = repair_env                    # tests/ exists but is empty
    incident = _incident_for(agent, target, repaired=FIXED_SOURCE)
    result = asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert result.ok
    assert incident.tests_passed is None
    assert "no dedicated tests" in result.text.lower()


def test_relevant_tests_matches_by_stem(repair_env, tmp_path):
    agent, _target = repair_env
    (tmp_path / "tests" / "test_fake_module.py").write_text(_PASSING_TEST, encoding="utf-8")
    (tmp_path / "tests" / "test_other.py").write_text("def test_x(): pass\n", encoding="utf-8")
    matched = {p.name for p in agent._relevant_tests("fake_module")}
    assert matched == {"test_fake_module.py"}


def test_apply_still_gated_even_when_tests_pass(repair_env, tmp_path):
    # Passing tests inform the human but never trigger auto-apply.
    agent, target = repair_env
    (tmp_path / "tests" / "test_fake_module.py").write_text(_PASSING_TEST, encoding="utf-8")
    _incident_for(agent, target, repaired=FIXED_SOURCE)
    asyncio.run(agent.repair_file("inc-t", confirm=False))
    assert target.read_text(encoding="utf-8") == ORIGINAL_SOURCE   # unchanged until confirm
