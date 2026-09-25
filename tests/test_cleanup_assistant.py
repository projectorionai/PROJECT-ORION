"""
Tests for the safe file-cleanup assistant (Section 9).

Uses a temporary project tree and a fake recycle bin.  Verifies detection and
risk levels, that secrets/protected files are never marked for deletion, that
deletion requires an explicit plan+token, dry-run previews nothing, path
traversal and symlinks are blocked, files changed after approval are rejected,
permanent deletion needs a second confirmation, and an audit record is produced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import cleanup_assistant as cleanup_assistant_module
from orion_core.cleanup_assistant import (
    CleanupAssistant,
    CleanupError,
    ProposedAction,
    RiskLevel,
)


def _project(tmp_path: Path) -> Path:
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.cpython-313.pyc").write_bytes(b"x")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.bin").write_bytes(b"y")
    (tmp_path / "app.log").write_text("log line")
    (tmp_path / "notes.tmp").write_text("temp")
    (tmp_path / ".DS_Store").write_bytes(b"z")
    (tmp_path / "main.py").write_text("print('hi')")          # protected source
    (tmp_path / "data.db").write_bytes(b"db")                 # protected database
    (tmp_path / "api_keys.json").write_text('{"key":"SECRET"}')  # secret-bearing
    (tmp_path / "server.pem").write_text("KEY")              # secret-bearing
    return tmp_path


def _assistant(root: Path, trash_sink=None, git_tracked=None):
    return CleanupAssistant(
        [root],
        git_tracked=git_tracked or set(),
        trash=trash_sink,
        logger=lambda *_: None,
    )


def test_scan_detects_artifacts_and_assigns_risk(tmp_path):
    root = _project(tmp_path)
    report = _assistant(root).scan()
    by_type = {c.file_type for c in report.candidates}
    assert "cache directory" in by_type
    assert "build output" in by_type
    assert "regenerable artefact" in by_type
    assert "OS metadata" in by_type
    # LOW-risk artefacts are the deletable set.
    for c in report.deletable():
        assert c.risk is RiskLevel.LOW


def test_secrets_are_flagged_gitignore_not_delete_and_never_read(tmp_path):
    root = _project(tmp_path)
    report = _assistant(root).scan()
    secrets = [c for c in report.candidates if c.is_secret_bearing]
    names = {Path(c.path).name for c in secrets}
    assert {"api_keys.json", "server.pem"} <= names
    for c in secrets:
        assert c.proposed_action is ProposedAction.SUGGEST_GITIGNORE
        assert c.risk is RiskLevel.HIGH
        # The reason must not contain the secret value.
        assert "SECRET" not in c.reason and "KEY" not in c.reason.upper().replace("KEYS", "")
    assert ".env*" in report.gitignore_suggestions or "*.pem" in report.gitignore_suggestions


def test_source_and_database_never_suggested(tmp_path):
    root = _project(tmp_path)
    report = _assistant(root).scan()
    paths = {Path(c.path).name: c for c in report.candidates}
    assert "main.py" not in paths           # source never flagged
    assert "data.db" not in paths           # database never flagged


def test_deletion_requires_plan_and_only_approved_candidates(tmp_path):
    root = _project(tmp_path)
    a = _assistant(root, trash_sink=lambda p: os.remove(p))
    # A protected/secret file cannot be planned for deletion.
    with pytest.raises(CleanupError):
        a.plan_deletion([str(root / "api_keys.json")])
    with pytest.raises(CleanupError):
        a.plan_deletion([str(root / "main.py")])


def test_dry_run_previews_without_deleting(tmp_path):
    root = _project(tmp_path)
    deleted = []
    a = _assistant(root, trash_sink=lambda p: deleted.append(p))
    plan = a.plan_deletion([str(root / "app.log")])
    preview = a.dry_run(plan)
    assert preview and "would delete" in preview[0]["action"]
    assert deleted == []                    # nothing removed
    assert (root / "app.log").exists()


def test_confirm_and_delete_uses_recycle_bin_and_audits(tmp_path):
    root = _project(tmp_path)
    recycled = []
    a = _assistant(root, trash_sink=lambda p: recycled.append(p))
    plan = a.plan_deletion([str(root / "app.log"), str(root / "notes.tmp")])
    outcome = a.confirm_and_delete(plan.token)
    assert len(outcome.deleted) == 2
    assert outcome.recoverable is True
    assert len(recycled) == 2
    assert len(outcome.audit) == 2
    assert all(rec["action"] == "recycled" for rec in outcome.audit)


def test_path_traversal_is_blocked(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "app.log").write_text("x")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret elsewhere")
    a = _assistant(root, trash_sink=lambda p: None)
    with pytest.raises(CleanupError):
        a.plan_deletion([str(outside)])
    with pytest.raises(CleanupError):
        a.plan_deletion([str(root / ".." / "outside.txt")])


def test_symlink_is_refused(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "real.log").write_text("x")
    link = root / "link.log"
    try:
        link.symlink_to(root / "real.log")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")
    a = _assistant(root, trash_sink=lambda p: None)
    with pytest.raises(CleanupError):
        a.plan_deletion([str(link)])


def test_file_changed_after_approval_is_skipped(tmp_path):
    root = _project(tmp_path)
    recycled = []
    a = _assistant(root, trash_sink=lambda p: recycled.append(p))
    plan = a.plan_deletion([str(root / "app.log")])
    # Tamper after approval.
    import time
    time.sleep(0.01)
    (root / "app.log").write_text("changed content and size!!!")
    os.utime(root / "app.log", (time.time() + 5, time.time() + 5))
    outcome = a.confirm_and_delete(plan.token)
    assert outcome.deleted == []
    assert outcome.skipped and "changed after approval" in outcome.skipped[0]["reason"]
    assert recycled == []


def test_permanent_delete_requires_second_confirmation(tmp_path, monkeypatch):
    root = _project(tmp_path)
    # No trash available → deletion would be permanent.
    #
    # Passing trash_sink=None is NOT enough to express that: the constructor
    # reads None as "caller didn't specify, use the platform default", and
    # falls back to _default_trash(), which returns a real send2trash binding
    # whenever that package is installed. This test therefore passed only on
    # machines WITHOUT send2trash and failed on machines with it. Patch the
    # default itself so the "this system has no recycle bin" production path
    # is exercised deterministically either way.
    monkeypatch.setattr(cleanup_assistant_module, "_default_trash", lambda: None)
    a = _assistant(root, trash_sink=None)
    plan = a.plan_deletion([str(root / "app.log")])
    with pytest.raises(CleanupError):
        a.confirm_and_delete(plan.token)               # refuses without 2nd confirm
    # Re-plan (token consumed on the failed attempt? No — pop happens first).
    plan2 = a.plan_deletion([str(root / "app.log")])
    outcome = a.confirm_and_delete(plan2.token, second_confirm=True)
    assert outcome.deleted == [str((root / "app.log").resolve())]
    assert outcome.recoverable is False
    assert not (root / "app.log").exists()


def test_used_token_cannot_be_replayed(tmp_path):
    root = _project(tmp_path)
    a = _assistant(root, trash_sink=lambda p: os.remove(p))
    plan = a.plan_deletion([str(root / "notes.tmp")])
    a.confirm_and_delete(plan.token)
    with pytest.raises(CleanupError):
        a.confirm_and_delete(plan.token)
