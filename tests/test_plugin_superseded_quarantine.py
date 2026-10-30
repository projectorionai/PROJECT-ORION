"""Superseded and quarantined plugins are deliberate states, not red errors.

Owner report from the Plugins page (25 Oct 2026, Mark 37.2):
  * gesture_control_interface / _manager / _tool showed RED as "tier confirm ·
    error - held back: it duplicates ORION's built-in tool 'gesture_control'";
  * jv_telegram_remote showed RED as "module file 'jv_telegram_remote_tool.py'
    not found" although the loader had quarantined that module itself.
All on tmp plugin folders; nothing under the owner's config is touched.
"""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orion_core import dynamic_loader, plugin_registry
from orion_core.dynamic_loader import ModuleLoadOutcome, ReflectiveModuleLoader
from orion_core.plugin_manifest import load_plugins
from orion_core.plugin_registry import PluginRegistry

GOOD_MODULE = '''
def get_tool_schema():
    return {"name": "%s", "description": "Synthetic plugin.",
            "parameters": {"type": "object", "properties": {}, "required": []}}


def run(**kwargs):
    return "ok"
'''


def _bus():
    return SimpleNamespace(log=SimpleNamespace(emit=Mock()),
                           dashboard_event=SimpleNamespace(emit=Mock()))


def _manifest(folder, name, module=None):
    module = module or f"{name}_tool.py"
    (folder / f"{name}.plugin.json").write_text(json.dumps({
        "name": name, "description": f"{name} plugin", "module": module,
        "tier": "confirm", "version": "33.0.0"}), encoding="utf-8")
    return folder / module


@pytest.fixture()
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin_registry, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    tools = tmp_path / "custom_tools"
    tools.mkdir()
    return tools


def _quarantine(folder, module, source, reason):
    qdir = folder / "_quarantine"
    qdir.mkdir(exist_ok=True)
    (qdir / f"{module}.broken").write_text(source, encoding="utf-8")
    (qdir / "index.json").write_text(json.dumps({f"{module}.broken": {
        "tool": module[:-3], "reason": reason, "at": "2026-10-20T18:22:31Z"}}), encoding="utf-8")


def _logged(bus):
    return [str(call.args[0]) for call in bus.log.emit.call_args_list]


# ── the quarantined Telegram plugin ─────────────────────────────────────────

def test_a_quarantined_module_shows_as_quarantined_not_missing(folder):
    _manifest(folder, "jv_telegram_remote")
    _quarantine(folder, "jv_telegram_remote_tool.py", GOOD_MODULE % "jv_telegram_remote",
                "import did not complete within 8s")
    bus = _bus()
    registry = PluginRegistry(bus, folder)
    loader = SimpleNamespace(load_and_register=Mock())
    asyncio.run(load_plugins(loader, Mock(), folder, bus, registry=registry))

    record = registry.get("jv_telegram_remote")
    assert record.status() == "quarantined"
    assert record.quarantined == "import did not complete within 8s"
    assert record.last_error == ""
    assert any("is quarantined" in line for line in _logged(bus))
    assert not any("not found" in line for line in _logged(bus))
    doctor = registry.doctor()
    assert doctor.ok and "quarantined - import did not complete" in doctor.text
    assert registry.snapshot()["quarantined"] == 1


def test_restoring_a_quarantined_module_that_imports_orion_internals_is_refused(folder):
    _manifest(folder, "jv_telegram_remote")
    _quarantine(folder, "jv_telegram_remote_tool.py",
                "from orion_core import app\n" + GOOD_MODULE % "jv_telegram_remote", "timeout")
    registry = PluginRegistry(_bus(), folder)
    result = registry.restore_quarantined("jv_telegram_remote")
    assert not result.ok and "cannot be restored" in result.text
    assert (folder / "_quarantine" / "jv_telegram_remote_tool.py.broken").is_file()
    assert not (folder / "jv_telegram_remote_tool.py").exists()


def test_restoring_a_clean_quarantined_module_puts_it_back_and_keeps_a_copy(folder):
    _manifest(folder, "jv_telegram_remote")
    source = GOOD_MODULE % "jv_telegram_remote"
    _quarantine(folder, "jv_telegram_remote_tool.py", source, "timeout")
    registry = PluginRegistry(_bus(), folder)
    result = registry.restore_quarantined("jv_telegram_remote")
    assert result.ok, result.text
    assert (folder / "jv_telegram_remote_tool.py").read_text(encoding="utf-8") == source
    archived = list((folder / "_superseded").glob("jv_telegram_remote_tool.py.broken.*"))
    assert len(archived) == 1 and archived[0].read_text(encoding="utf-8") == source
    index = json.loads((folder / "_quarantine" / "index.json").read_text(encoding="utf-8"))
    assert index == {}
    assert registry.get("jv_telegram_remote").status() != "quarantined"


def test_removing_a_quarantined_plugin_archives_it_and_deletes_nothing(folder):
    _manifest(folder, "jv_telegram_remote")
    _quarantine(folder, "jv_telegram_remote_tool.py", GOOD_MODULE % "x", "timeout")
    before = {p.name: p.read_bytes() for p in [folder / "jv_telegram_remote.plugin.json",
                                                folder / "_quarantine" / "jv_telegram_remote_tool.py.broken"]}
    registry = PluginRegistry(_bus(), folder)
    result = registry.remove("jv_telegram_remote")
    assert result.ok, result.text
    assert registry.get("jv_telegram_remote") is None
    kept = {p.name.rsplit(".", 1)[0]: p.read_bytes() for p in (folder / "_superseded").iterdir()
            if p.name != "index.json"}
    assert kept == before                         # every byte kept, under _superseded
    assert not registry.is_enabled("jv_telegram_remote")   # tombstoned


# ── gesture plugins superseded by the built-in ──────────────────────────────

class _DuplicateLoader:
    """The loader's verdict for the owner's three gesture plugins."""

    def __init__(self, duplicates):
        self.duplicates = duplicates

    async def load_and_register(self, path):
        name = path.stem.removesuffix("_tool")
        if name in self.duplicates:
            return ModuleLoadOutcome(False, path.stem, module_path=path, error_log=[
                "held back: it duplicates ORION's built-in tool 'gesture_control'"],
                superseded_by="gesture_control")
        return ModuleLoadOutcome(False, path.stem, module_path=path, error_log=["boom"])


def test_duplicates_of_a_built_in_are_archived_as_superseded_not_errors(folder):
    names = ["gesture_control_interface", "gesture_control_manager", "gesture_control_tool"]
    originals = {}
    for name in names:
        module = _manifest(folder, name)
        module.write_text(GOOD_MODULE % name, encoding="utf-8")
        originals[module.name] = module.read_bytes()
        originals[f"{name}.plugin.json"] = (folder / f"{name}.plugin.json").read_bytes()
    _manifest(folder, "unrelated").write_text(GOOD_MODULE % "unrelated", encoding="utf-8")
    bus = _bus()
    registry = PluginRegistry(bus, folder)

    asyncio.run(load_plugins(_DuplicateLoader(set(names)), Mock(), folder, bus, registry=registry))

    assert [r.name for r in registry.scan()] == ["unrelated"]     # only the real failure left
    assert registry.get("unrelated").status() == "error"
    archive = folder / "_superseded"
    kept = {p.name.rsplit(".", 1)[0]: p.read_bytes() for p in archive.iterdir() if p.name != "index.json"}
    assert kept == originals                      # verified copies of every file
    receipts = registry.superseded_archive()
    assert {r["name"] for r in receipts} == set(names)
    assert all(r["superseded_by"] == "gesture_control" for r in receipts)
    assert {item["name"] for item in registry.snapshot()["archived"]} == set(names)
    lines = _logged(bus)
    assert not any("failed to load" in line and "gesture" in line for line in lines)
    assert sum("superseded by the built-in 'gesture_control'" in line for line in lines) == 3


def test_a_superseded_plugin_that_cannot_be_moved_is_still_neutral(folder, monkeypatch):
    _manifest(folder, "gesture_control_manager").write_text(GOOD_MODULE % "g", encoding="utf-8")
    registry = PluginRegistry(_bus(), folder)
    monkeypatch.setattr(registry, "_archive_files", Mock(side_effect=OSError("locked")))
    result = registry.supersede("gesture_control_manager", "gesture_control")
    assert not result.ok
    record = registry.get("gesture_control_manager")
    assert record.status() == "superseded" and record.last_error == ""
    assert (folder / "gesture_control_manager_tool.py").is_file()


def test_the_real_loader_names_the_built_in_it_was_held_back_for(tmp_path, monkeypatch):
    monkeypatch.setattr(dynamic_loader, "_bundled_plugins", lambda: frozenset())
    (tmp_path / ".shipped.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "gesture_control_interface_tool.py"
    path.write_text('raise AssertionError("never imported")\n', encoding="utf-8")
    loader = ReflectiveModuleLoader.__new__(ReflectiveModuleLoader)
    loader.native_names = lambda: {"gesture_control"}
    loader.bus = _bus()
    outcome = asyncio.run(loader.load_and_register(path))
    assert not outcome.succeeded and outcome.superseded_by == "gesture_control"


# ── the Plugins page ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def _texts(widget):
    from PyQt6.QtWidgets import QLabel, QPushButton
    return (" ".join(c.text() for c in widget.findChildren(QLabel)),
            {b.text(): b for b in widget.findChildren(QPushButton)})


def test_the_page_shows_quarantined_with_restore_and_remove(qapp):
    from orion_core.constants import C
    from orion_core.gui.plugin_deck import _PluginRow
    restored = []
    row = _PluginRow({"name": "jv_telegram_remote", "tier": "confirm", "version": "33.0.0",
                      "status": "quarantined", "quarantined": "import did not complete within 8s",
                      "enabled": True}, on_toggle=Mock(), on_restore=restored.append,
                     on_remove=Mock())
    text, buttons = _texts(row)
    assert "Quarantined: import did not complete" in text and "not found" not in text
    assert set(buttons) == {"Restore", "Remove"}
    assert buttons["Restore"].accessibleName() == "Restore plugin jv_telegram_remote"
    buttons["Restore"].click()
    assert restored == ["jv_telegram_remote"]
    assert C.BAD not in text


def test_the_page_shows_superseded_as_neutral(qapp):
    from orion_core.constants import C
    from orion_core.gui.plugin_deck import _PluginRow
    row = _PluginRow({"name": "gesture_control_manager", "tier": "confirm", "status": "superseded",
                      "superseded_by": "gesture_control", "enabled": True, "last_error": ""},
                     on_toggle=Mock())
    text, buttons = _texts(row)
    assert "Superseded by built-in gesture_control" in text
    assert "error" not in text.split("not an error")[0].lower()
    assert C.BAD not in text and C.PRI not in text
    assert not buttons["Superseded"].isEnabled()


# ── Mark XXXIX: obsolete plugins are retired, not "rejected" ─────────────────
# Owner, 27 Oct 2026: "ORION's got 5 plugins rejected so this must be fixed as
# they are now obsolete". Five switched-off plugins were counted at every boot
# as "24 active, 5 rejected".

def test_obsolete_plugins_are_archived_and_the_boot_line_rejects_nothing(folder, monkeypatch):
    from orion_core import startup_report
    from orion_core.plugin_manifest import OBSOLETE_PLUGINS

    said = []
    monkeypatch.setattr(startup_report, "say", lambda line, bus=None: said.append(line))
    for name in OBSOLETE_PLUGINS:
        _manifest(folder, name).write_text(GOOD_MODULE % name, encoding="utf-8")
    bus = _bus()
    registry = PluginRegistry(bus, folder)
    registry.set_enabled("json_validator", False)
    loader = SimpleNamespace(load_and_register=Mock())
    plan = asyncio.run(load_plugins(loader, Mock(), folder, bus, registry=registry))

    loader.load_and_register.assert_not_called()
    assert not list(folder.glob("*.plugin.json")) and not list(folder.glob("*_tool.py"))
    archived = {entry["name"]: entry for entry in registry.superseded_archive()}
    assert set(archived) == set(OBSOLETE_PLUGINS)
    assert archived["json_validator"]["superseded_by"] == "diagnostics"
    assert all(str(r).startswith("superseded by") for r in plan.skipped.values())
    assert said and "0 active, 0 rejected (5 retired), 5 total." in said[-1]


def test_a_switched_off_plugin_is_not_counted_as_rejected(folder, monkeypatch):
    from orion_core import startup_report

    said = []
    monkeypatch.setattr(startup_report, "say", lambda line, bus=None: said.append(line))
    _manifest(folder, "jv_quiz").write_text(GOOD_MODULE % "jv_quiz", encoding="utf-8")
    _manifest(folder, "broken_one", "missing_tool.py")
    bus = _bus()
    registry = PluginRegistry(bus, folder)
    registry.set_enabled("jv_quiz", False)
    loader = SimpleNamespace(load_and_register=Mock())
    asyncio.run(load_plugins(loader, Mock(), folder, bus, registry=registry))
    assert "0 active, 1 rejected (1 switched off), 2 total." in said[-1]
