"""Isolated plugin discovery/path probes; no personal plugins are touched."""
import asyncio
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion_core.dynamic_loader import ReflectiveModuleLoader
from orion_core.plugin_manifest import ManifestError, PluginManifest, load_plugins, plan_load
from orion_core.data import ToolResult


@pytest.mark.parametrize("module", ["../outside.py", "..\\outside.py", "/outside.py",
                                    "C:\\outside.py", "C:outside.py", "nested/tool.py"])
def test_manifest_requires_a_module_filename_beside_it(module):
    with pytest.raises(ManifestError):
        PluginManifest.from_dict({"name": "synthetic", "module": module})


def test_unreadable_manifest_does_not_abort_other_plugins(tmp_path, monkeypatch):
    bad, good = tmp_path / "bad.plugin.json", tmp_path / "good.plugin.json"
    for path, name in ((bad, "bad"), (good, "good")):
        path.write_text(json.dumps({"name": name, "module": f"{name}_tool.py"}), encoding="utf-8")
        (tmp_path / f"{name}_tool.py").write_text("# synthetic", encoding="utf-8")
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == bad:
            raise PermissionError("synthetic sharing lock")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    plan = plan_load(tmp_path)
    assert [manifest.name for manifest in plan.loadable] == ["good"]
    assert "bad.plugin.json" in plan.skipped


def test_outside_source_is_neither_imported_nor_quarantined(tmp_path):
    directory = tmp_path / "plugins"
    directory.mkdir()
    outside = tmp_path / "unrelated.py"
    source = b"this is synthetic invalid Python ["
    outside.write_bytes(source)
    (directory / "synthetic.plugin.json").write_text(
        json.dumps({"name": "synthetic", "module": "../unrelated.py"}), encoding="utf-8")
    log = []
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *args: log.append(args)))
    loader = ReflectiveModuleLoader(bus)
    loader.custom_tools_dir = directory
    asyncio.run(load_plugins(loader, SimpleNamespace(), directory, bus))
    assert outside.is_file(), "plugin discovery renamed an unrelated outside source into quarantine"
    assert outside.read_bytes() == source
    assert not (tmp_path / "_quarantine").exists()


@pytest.mark.parametrize("stage", ["import", "activation"])
def test_failed_plugin_is_not_reported_as_active(tmp_path, stage):
    (tmp_path / "broken_tool.py").write_text("# synthetic", encoding="utf-8")
    (tmp_path / "broken.plugin.json").write_text(
        json.dumps({"name": "broken", "module": "broken_tool.py"}), encoding="utf-8")
    async def load(_path):
        return SimpleNamespace(succeeded=stage != "import", module_name="broken_tool",
                               error_log=["synthetic import failure"])
    async def activate(*_):
        return ToolResult("synthetic activation failure", ok=False)
    log = []
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *args: log.append(args)))
    loader = SimpleNamespace(load_and_register=load, activate_tool=activate)
    plan = asyncio.run(load_plugins(loader, SimpleNamespace(), tmp_path, bus))
    assert not plan.loadable
    assert "synthetic" in plan.skipped["broken"]
    assert any("0 active" in str(args) for args in log)


def test_plugin_discovery_io_does_not_occupy_the_event_loop(tmp_path, monkeypatch):
    import orion_core.plugin_manifest as manifests
    entered, release = threading.Event(), threading.Event()
    owner = threading.current_thread()
    observed = []
    def slow_plan(_):
        observed.append(threading.current_thread())
        entered.set()
        if threading.current_thread() is not owner:
            assert release.wait(timeout=3)
        return manifests.LoadPlan()
    monkeypatch.setattr(manifests, "plan_load", slow_plan)
    async def scenario():
        bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
        task = asyncio.create_task(load_plugins(None, None, tmp_path, bus))
        try:
            async def _wait_entered():
                while not entered.is_set():
                    if task.done():
                        await task
                        raise AssertionError("load_plugins finished before entering plan_load")
                    await asyncio.sleep(0.001)
            await asyncio.wait_for(_wait_entered(), timeout=5.0)
            assert observed[0] is not owner
            assert not task.done()
        finally:
            release.set()
            await task
    asyncio.run(scenario())
