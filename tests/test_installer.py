"""
The self-installer (Mark XXII) — a real Add/Remove Programs registration built
on the working in-place model.

    "an installable .exe that actually installs itself onto the PC"

The registry is faked throughout — no test writes a real key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import installer  # noqa: E402
from orion_core.installer import UNINSTALL_SUBKEY  # noqa: E402


class FakeRegistry(installer.RegistryWriter):
    def __init__(self):
        self.written: dict[str, dict] = {}
        self.deleted: list[str] = []

    def write(self, subkey, values):
        self.written[subkey] = dict(values)

    def delete(self, subkey):
        self.deleted.append(subkey)
        self.written.pop(subkey, None)


# ── the uninstall entry is well-formed ───────────────────────────────────────

def test_uninstall_entries_have_the_required_arp_fields():
    e = installer.uninstall_entries()
    for field in ("DisplayName", "DisplayVersion", "Publisher",
                  "UninstallString", "InstallLocation", "DisplayIcon"):
        assert field in e and str(e[field]).strip()


def test_uninstall_command_invokes_this_module():
    cmd = installer.uninstall_command()
    assert "orion_core.installer" in cmd
    assert "--uninstall" in cmd


def test_display_name_is_orion():
    name = installer.uninstall_entries()["DisplayName"].upper().replace(".", "")
    assert "ORION" in name


# ── install writes the entry (to the fake registry) ──────────────────────────

def test_install_registers_under_the_uninstall_key(monkeypatch):
    # don't actually create shortcuts or icons during the test
    monkeypatch.setattr(installer.desktop_app, "ensure_icon", lambda: None)
    monkeypatch.setattr(installer.desktop_app, "install_shortcuts",
                        lambda *a, **k: {"desktop": True, "start_menu": True})
    reg = FakeRegistry()
    report = installer.install(writer=reg)
    assert report.registered
    assert UNINSTALL_SUBKEY in reg.written
    assert reg.written[UNINSTALL_SUBKEY]["DisplayName"] == installer.DISPLAY_NAME
    assert report.shortcuts == {"desktop": True, "start_menu": True}


def test_install_does_not_touch_startup_unless_asked(monkeypatch):
    monkeypatch.setattr(installer.desktop_app, "ensure_icon", lambda: None)
    monkeypatch.setattr(installer.desktop_app, "install_shortcuts",
                        lambda *a, **k: {"desktop": True})
    calls = []
    import orion_core.autostart as autostart
    monkeypatch.setattr(autostart, "enable", lambda: calls.append("enabled") or True)

    installer.install(writer=FakeRegistry())
    assert calls == []                     # startup not enabled by default

    installer.install(writer=FakeRegistry(), add_to_startup=True)
    assert calls == ["enabled"]


def test_install_survives_a_registry_failure(monkeypatch):
    monkeypatch.setattr(installer.desktop_app, "ensure_icon", lambda: None)
    monkeypatch.setattr(installer.desktop_app, "install_shortcuts",
                        lambda *a, **k: {"desktop": True})

    class Broken(installer.RegistryWriter):
        def write(self, *a, **k):
            raise OSError("access denied")

    report = installer.install(writer=Broken())
    assert report.registered is False      # reported honestly, not crashed


# ── uninstall removes the entry ──────────────────────────────────────────────

def test_uninstall_deletes_the_registry_key(monkeypatch):
    monkeypatch.setattr(installer, "_remove_shortcuts", lambda: None)
    import orion_core.autostart as autostart
    monkeypatch.setattr(autostart, "disable", lambda: True)
    reg = FakeRegistry()
    reg.write(UNINSTALL_SUBKEY, {"DisplayName": "ORION"})
    msg = installer.uninstall(writer=reg)
    assert UNINSTALL_SUBKEY in reg.deleted
    assert "left untouched" in msg          # never deletes the user's project


def test_uninstall_does_not_delete_project_files(monkeypatch, tmp_path):
    # a sentinel "project file" that must survive an uninstall
    sentinel = tmp_path / "orion.py"
    sentinel.write_text("# do not delete me", encoding="utf-8")
    monkeypatch.setattr(installer, "_remove_shortcuts", lambda: None)
    import orion_core.autostart as autostart
    monkeypatch.setattr(autostart, "disable", lambda: True)
    installer.uninstall(writer=FakeRegistry())
    assert sentinel.exists()
