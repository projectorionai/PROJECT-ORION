"""Portable mapped plugins retain one owner and cannot introduce aliases."""
import json
import zipfile

import pytest

from orion_core import plugin_registry as module


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    registry = module.PluginRegistry(directory=tmp_path / "tools")
    assert registry.create("demo", "Synthetic portable plugin", "confirm").ok
    return registry


@pytest.mark.parametrize("module_name", ["implementation_tool.py", "implementation.py"])
def test_mapped_export_import_has_exactly_one_manifest_owner(registry, tmp_path, module_name):
    (registry.directory / "demo_tool.py").rename(registry.directory / module_name)
    old = registry.directory / "demo.plugin.json"
    payload = json.loads(old.read_text(encoding="utf-8"))
    payload.update(module=module_name, events=["state"], interval_seconds=30, secrets=["SYNTHETIC_KEY"])
    old.unlink()
    (registry.directory / "metadata.plugin.json").write_text(json.dumps(payload), encoding="utf-8")
    archive = tmp_path / "portable.orionplugin"
    assert registry.export_bundle("demo", str(archive)).ok
    target = module.PluginRegistry(directory=tmp_path / "receiver")
    receipt = target.install_bundle(str(archive))
    assert receipt.ok
    records = target.scan()
    assert len(records) == 1 and records[0].name == "demo" and records[0].healthy
    assert sorted(path.name for path in target.directory.glob("*.plugin.json")) == ["metadata.plugin.json"]
    assert json.loads((target.directory / "metadata.plugin.json").read_text(encoding="utf-8")) == payload


def test_install_cannot_alias_an_existing_management_identity(registry, tmp_path):
    before = {path.name: path.read_bytes() for path in registry.directory.iterdir()}
    archive = tmp_path / "collision.orionplugin"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("different_tool.py", (registry.directory / "demo_tool.py").read_bytes())
        bundle.writestr("different.plugin.json", json.dumps({"name": "Demo", "module": "different_tool.py"}))
    assert not registry.install_bundle(str(archive)).ok
    assert {path.name: path.read_bytes() for path in registry.directory.iterdir()} == before


def test_scaffolding_does_not_alias_a_mapped_existing_identity(registry):
    (registry.directory / "demo_tool.py").rename(registry.directory / "implementation_tool.py")
    manifest = registry.directory / "demo.plugin.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(name="Demo", module="implementation_tool.py")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not registry.create("demo", "An alias", "confirm").ok
    assert not (registry.directory / "demo_tool.py").exists()
