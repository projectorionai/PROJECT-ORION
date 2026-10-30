"""Invalid installation candidates must preserve existing working plugins."""
import json
import zipfile
import asyncio
import threading
from pathlib import Path

import pytest

from orion_core import plugin_registry as module

CONTRACT = ("def get_tool_schema():\n"
            "    return {'name':'demo','description':'demo','parameters':{}}\n"
            "def run(**kwargs):\n    return 'candidate'\n")


@pytest.fixture
def existing(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    registry = module.PluginRegistry(directory=tmp_path / "tools")
    assert registry.create("demo", "Original working plugin", "confirm").ok
    before = {path.name: path.read_bytes() for path in registry.directory.iterdir()}
    state = module.PLUGIN_STATE_PATH.read_bytes()
    return registry, before, state


def _preserved(existing):
    registry, before, state = existing
    assert {path.name: path.read_bytes() for path in registry.directory.iterdir()} == before
    assert module.PLUGIN_STATE_PATH.read_bytes() == state


def test_invalid_bundle_module_cannot_replace_working_plugin(existing, tmp_path):
    candidate = tmp_path / "invalid.orionplugin"
    with zipfile.ZipFile(candidate, "w") as bundle:
        bundle.writestr("demo_tool.py", "this is not valid Python")
    assert not existing[0].install_bundle(str(candidate)).ok
    _preserved(existing)


def test_manifest_only_failed_bundle_cannot_modify_existing_metadata(existing, tmp_path):
    candidate = tmp_path / "manifest-only.orionplugin"
    with zipfile.ZipFile(candidate, "w") as bundle:
        bundle.writestr("demo.plugin.json", json.dumps(
            {"name": "demo", "module": "demo_tool.py", "tier": "forbid"}))
    assert not existing[0].install_bundle(str(candidate)).ok
    _preserved(existing)


def test_invalid_loose_manifest_cannot_replace_working_plugin(existing, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "demo_tool.py"
    source.write_text(CONTRACT, encoding="utf-8")
    (incoming / "demo.plugin.json").write_text("broken JSON", encoding="utf-8")
    assert not existing[0].install(str(source)).ok
    _preserved(existing)


def test_bundle_with_a_broken_member_does_not_partially_install_good_members(existing, tmp_path):
    candidate = tmp_path / "partial.orionplugin"
    with zipfile.ZipFile(candidate, "w") as bundle:
        bundle.writestr("new_tool.py", CONTRACT)
        bundle.writestr("broken_tool.py", "def broken(")
    assert not existing[0].install_bundle(str(candidate)).ok
    _preserved(existing)


@pytest.mark.parametrize("kind", ["loose", "bundle"])
def test_oversized_candidate_keeps_working_files(existing, tmp_path, monkeypatch, kind):
    monkeypatch.setattr(module, "MAX_PLUGIN_FILE_BYTES", 32)
    candidate = tmp_path / ("demo_tool.py" if kind == "loose" else "oversized.orionplugin")
    if kind == "loose":
        candidate.write_text(CONTRACT, encoding="utf-8")
    else:
        with zipfile.ZipFile(candidate, "w") as bundle:
            bundle.writestr("demo_tool.py", CONTRACT)
    result = existing[0].install(str(candidate))
    assert not result.ok and "budget" in result.text.lower()
    _preserved(existing)


def test_portable_bundle_case_collisions_are_rejected_before_replacement(existing, tmp_path):
    candidate = tmp_path / "collision.orionplugin"
    with zipfile.ZipFile(candidate, "w") as bundle:
        bundle.writestr("demo_tool.py", CONTRACT)
        bundle.writestr("DEMO_tool.py", CONTRACT)
    assert not existing[0].install_bundle(str(candidate)).ok
    _preserved(existing)


@pytest.mark.parametrize("fail_rollback", [False, True], ids=["rollback_succeeds", "recovery_retained"])
def test_publication_failure_recovers_or_retains_previous_bytes(existing, tmp_path, monkeypatch, fail_rollback):
    registry, before, state = existing
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "demo_tool.py"
    source.write_text(CONTRACT, encoding="utf-8")
    original = module._replace
    def replace(source_path, target):
        path = Path(source_path)
        if Path(target).name == "demo.plugin.json" and path.parent.name != ".previous":
            raise PermissionError("synthetic manifest publication lock")
        if fail_rollback and path.parent.name == ".previous" and Path(target).name == "demo_tool.py":
            raise PermissionError("synthetic rollback lock")
        return original(source_path, target)
    monkeypatch.setattr(module, "_replace", replace)
    result = registry.install(str(source))
    assert not result.ok
    assert module.PLUGIN_STATE_PATH.read_bytes() == state
    if fail_rollback:
        assert "rollback could not finish" in result.text
        recovery = next(registry.directory.glob(".install-stage-*"))
        assert (recovery / ".previous" / "demo_tool.py").read_bytes() == before["demo_tool.py"]
    else:
        _preserved(existing)


async def test_install_metadata_save_cannot_overwrite_a_concurrent_disable(existing, monkeypatch):
    registry, _, _ = existing
    entered, release, disable_entered = threading.Event(), threading.Event(), threading.Event()
    original = module.os.replace
    writes = []
    def replace(source, target):
        if Path(target) == module.PLUGIN_STATE_PATH:
            writes.append(target)
            if len(writes) == 1:
                entered.set()
                assert release.wait(timeout=3)
        return original(source, target)
    monkeypatch.setattr(module.os, "replace", replace)
    installing = asyncio.create_task(asyncio.to_thread(registry._record_install, "demo", "synthetic update"))
    disabling = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        def disable():
            disable_entered.set()
            return registry.set_enabled("demo", False)
        disabling = asyncio.create_task(asyncio.to_thread(disable))
        assert await asyncio.to_thread(disable_entered.wait, 2)
        assert not disabling.done()
    finally:
        release.set()
        assert await installing
        if disabling is not None:
            assert (await disabling).ok
    restarted = module.PluginRegistry(directory=registry.directory)
    assert not restarted.is_enabled("demo")
