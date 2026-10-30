"""Directory publication retries preserve owned staging and rollback payloads."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import skills as module


def package(path, version="1"):
    path.mkdir()
    (path / "skill.json").write_text(json.dumps({"name": "synthetic-skill", "version": version}))
    (path / "notes.txt").write_text("payload " + version)
    return path


def snapshot(path):
    return {file.relative_to(path).as_posix(): file.read_bytes()
            for file in path.rglob("*") if file.is_file()}


@pytest.fixture
def context(tmp_path, monkeypatch):
    from orion_core import recycle_bin
    # Keep recovery snapshots in the private test directory for verification.
    monkeypatch.setattr(recycle_bin, "refusal", lambda *_: "retained synthetic recovery")
    monkeypatch.setattr(module, "_WINDOWS", True, raising=False)
    sleeps = []
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=sleeps.append), raising=False)
    manager = module.SkillManager(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)),
                                  root=tmp_path / "skills")
    copies = []
    real_copy = module.shutil.copy2

    def copy(*args, **kwargs):
        copies.append(Path(args[0]).name)
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(module.shutil, "copy2", copy)
    return SimpleNamespace(root=tmp_path, manager=manager, sleeps=sleeps, copies=copies,
                           target=manager.root / "synthetic-skill")


def denied(code=5):
    error = PermissionError("synthetic Windows directory rename denial")
    error.winerror = code
    return error


def intercept_moves(monkeypatch, callback):
    calls = []
    original = {"replace": module.os.replace, "rename": module.os.rename}

    def wrap(operation):
        def move(source, target, *args, **kwargs):
            source, target = Path(source), Path(target)
            if source.is_dir():
                phase = ("publish" if source.name == "package" else
                         "restore" if source.name.startswith(".previous-") else "retire")
                calls.append((phase, source, target))
                callback(phase, source, target, calls, original[operation])
            return original[operation](source, target, *args, **kwargs)
        return move

    monkeypatch.setattr(module.os, "replace", wrap("replace"))
    monkeypatch.setattr(module.os, "rename", wrap("rename"))
    return calls


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_publication_denial_reuses_complete_snapshot_once(context, monkeypatch, code):
    source = package(context.root / "incoming")
    before = snapshot(source)

    def fault(phase, _source, _target, calls, _real):
        if phase == "publish" and len(calls) == 1:
            raise denied(code)

    calls = intercept_moves(monkeypatch, fault)
    result = context.manager.install(str(source))
    assert result.ok, result.text
    assert [phase for phase, *_ in calls] == ["publish", "publish"]
    assert len(context.sleeps) == 1 and sorted(context.copies) == ["notes.txt", "skill.json"]
    assert snapshot(source) == before and (context.target / "notes.txt").read_text() == "payload 1"
    assert context.manager.registry.get("synthetic-skill").version == "1"


@pytest.mark.parametrize("phase", ["retire", "publish", "restore"])
def test_upgrade_and_rollback_retry_only_the_denied_owned_move(context, monkeypatch, phase):
    assert context.manager.install(str(package(context.root / "old"))).ok
    before = snapshot(context.target)
    context.copies.clear()
    source = package(context.root / "incoming", "2")
    source_before = snapshot(source)
    failures = []

    def fault(current, _source, _target, _calls, _real):
        if phase == "restore" and current == "publish":
            raise OSError("synthetic permanent publication failure")
        if current == phase and not failures:
            failures.append(current)
            raise denied()

    calls = intercept_moves(monkeypatch, fault)
    result = context.manager.install(str(source))
    assert result.ok is (phase != "restore"), result.text
    assert len([call for call in calls if call[0] == phase]) == 2
    assert sorted(context.copies) == ["notes.txt", "skill.json"] and len(context.sleeps) == 1
    assert snapshot(source) == source_before
    if phase == "restore":
        assert snapshot(context.target) == before
        assert context.manager.registry.get("synthetic-skill").version == "1"
    else:
        assert (context.target / "notes.txt").read_text() == "payload 2"
        previous = [path for path in context.manager.installer.retained if path.name.startswith(".previous-")]
        assert len(previous) == 1 and snapshot(previous[0]) == before


def test_persistent_denial_is_bounded_and_restores_old_registry_and_bytes(context, monkeypatch):
    assert context.manager.install(str(package(context.root / "old"))).ok
    before = snapshot(context.target)
    context.copies.clear()
    source = package(context.root / "incoming", "2")

    def fault(phase, *_args):
        if phase == "publish":
            raise denied()

    calls = intercept_moves(monkeypatch, fault)
    result = context.manager.install(str(source))
    assert not result.ok
    assert len([call for call in calls if call[0] == "publish"]) == 5
    assert len(context.sleeps) == 4 and sum(context.sleeps) <= 0.326
    assert sorted(context.copies) == ["notes.txt", "skill.json"]
    assert snapshot(context.target) == before
    assert context.manager.registry.get("synthetic-skill").version == "1"
    assert (source / "notes.txt").read_text() == "payload 2"
    assert any((path / "package" / "notes.txt").read_text() == "payload 2"
               for path in context.manager.installer.retained if (path / "package").is_dir())


@pytest.mark.parametrize("windows,code", [(True, 2), (False, 5)])
def test_unrecognized_or_nonwindows_errors_are_not_retried(context, monkeypatch, windows, code):
    monkeypatch.setattr(module, "_WINDOWS", windows, raising=False)

    def fault(phase, *_args):
        if phase == "publish":
            raise denied(code)

    calls = intercept_moves(monkeypatch, fault)
    result = context.manager.install(str(package(context.root / "incoming")))
    assert not result.ok and len(calls) == 1 and context.sleeps == []
    assert not context.target.exists() and context.manager.registry.get("synthetic-skill") is None


@pytest.mark.parametrize("changed", ["payload", "destination", "directory_identity"])
def test_observed_changes_refuse_retry_without_overwriting_or_discarding_them(context, monkeypatch, changed):
    source = package(context.root / "incoming")
    before = snapshot(source)
    altered = []

    def fault(phase, staging, target, _calls, _real):
        if phase != "publish":
            return
        if changed == "destination":
            target.mkdir()
            victim = target / "foreign.txt"
        elif changed == "directory_identity":
            saved = staging.with_name("owned-original")
            assert saved.resolve().is_relative_to(context.root.resolve())
            _real(staging, saved)
            staging.mkdir()
            victim = staging / "foreign.txt"
        else:
            victim = staging / "notes.txt"
        victim.write_bytes(b"observed changed bytes")
        altered.append(victim)
        raise denied()

    calls = intercept_moves(monkeypatch, fault)
    result = context.manager.install(str(source))
    assert not result.ok and len([call for call in calls if call[0] == "publish"]) == 1
    assert context.sleeps == [] and snapshot(source) == before
    assert altered[0].read_bytes() == b"observed changed bytes"
    assert context.manager.registry.get("synthetic-skill") is None


def test_move_completed_before_error_is_verified_without_replaying(context, monkeypatch):
    def fault(phase, staging, target, _calls, real):
        if phase == "publish":
            real(staging, target)
            raise denied()

    calls = intercept_moves(monkeypatch, fault)
    result = context.manager.install(str(package(context.root / "incoming")))
    assert result.ok, result.text
    assert len(calls) == 1 and context.sleeps == []
    assert context.manager.registry.get("synthetic-skill").version == "1"


def test_failed_rollback_retains_both_recovery_payloads_and_old_registry(context, monkeypatch):
    assert context.manager.install(str(package(context.root / "old"))).ok
    before = snapshot(context.target)

    def fault(phase, *_args):
        if phase == "publish":
            raise OSError("synthetic permanent publication failure")
        if phase == "restore":
            raise denied()

    calls = intercept_moves(monkeypatch, fault)
    result = context.manager.install(str(package(context.root / "incoming", "2")))
    assert not result.ok and "Recovery copies retained" in result.text
    assert len([call for call in calls if call[0] == "restore"]) == 5
    assert context.manager.registry.get("synthetic-skill").version == "1"
    previous = [path for path in context.manager.installer.retained if path.name.startswith(".previous-")]
    assert len(previous) == 1 and snapshot(previous[0]) == before
    assert not context.target.exists()
