"""The docked Screen Face is NOT on screen when ORION starts (Mark XXXVII).

The owner's complaint: the dock opened by itself at every start. Now
windows_settings["dock_on_startup"] is off by default and the dock appears only
when it is asked for: the tray's Show Dock, the Show Dock at Startup switch in
Preferences (from the next start), the jump list's Show Dock task, or "show your
dock". Until then it is not even built - a hidden dock would still own a face
render thread.

Everything here uses fakes: no tray, no window, no real dock.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import app, windows_settings  # noqa: E402


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _Bus:
    def __init__(self):
        self.log, self.voice_mute_request, self.speak_request = _Signal(), _Signal(), _Signal()


class _Window:
    def __init__(self):
        self.calls = []
        self._overlay_active = False

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))
            return False
        return record


class _FakeDock:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.wanted = False
        self.shown = self.hidden = 0
        self.bus = None

    def connect_bus(self, bus):
        self.bus = bus

    def show_dock(self):
        self.wanted = True
        self.shown += 1

    def hide_dock(self):
        self.wanted = False
        self.hidden += 1

    def stop(self):
        pass


class _Presence:
    def __init__(self):
        self.pause_hearing_when_locked = True
        self.greet_on_unlock = True
        self.toasts = SimpleNamespace(set_tray=lambda tray: None, show=lambda *a, **k: None,
                                      handle_action=lambda *a: True)
        self.docks = []

    def quiet(self):
        return False

    def make_dock(self, **kwargs):
        dock = _FakeDock(**kwargs)
        self.docks.append(dock)
        return dock

    def stop(self):
        pass


@pytest.fixture
def start(monkeypatch, tmp_path):
    """start(**settings) -> the running integration with a fake presence."""
    from orion_core import background, windows_presence, windows_shell

    monkeypatch.setattr(windows_settings, "SETTINGS_PATH", tmp_path / "w.json")
    for key in windows_settings.DEFAULTS:
        monkeypatch.delenv(f"ORION_WIN_{key.upper()}", raising=False)
    captured = {}
    monkeypatch.setattr(windows_shell, "start_windows_shell",
                        lambda bus, on_command: captured.setdefault("on_command", on_command)
                        and SimpleNamespace(stop=lambda: None))
    presence = _Presence()
    monkeypatch.setattr(windows_presence, "start_windows_presence",
                        lambda bus, **kw: presence)
    monkeypatch.setattr(app, "_toast_approvals", lambda *a, **k: None)
    monkeypatch.setattr(background, "spawn", lambda coro, **k: coro.close())

    def _start(**settings):
        for key, value in settings.items():
            assert windows_settings.set_value(key, value)
        bus, window = _Bus(), _Window()
        worker = SimpleNamespace(microphone_enabled=True, submit_text=lambda t: None,
                                 set_microphone_enabled=lambda on: None)
        parts = app._start_windows_integration(bus, window, worker)
        return SimpleNamespace(bus=bus, window=window, presence=presence, parts=parts,
                               on_command=captured["on_command"])

    return _start


# ── the setting ──────────────────────────────────────────────────────────────

def test_the_dock_is_not_shown_at_startup_by_default():
    assert windows_settings.DEFAULTS["dock_on_startup"] is False
    assert windows_settings.DEFAULTS["dock_enabled"] is True      # allowed, just not automatic


def test_the_setting_round_trips_and_wrong_types_are_refused(start):
    assert windows_settings.set_value("dock_on_startup", True)
    assert windows_settings.get("dock_on_startup") is True
    assert not windows_settings.set_value("dock_on_startup", "yes")
    assert windows_settings.get("dock_on_startup") is True


def test_the_env_override_can_turn_it_on(monkeypatch, start):
    monkeypatch.setenv("ORION_WIN_DOCK_ON_STARTUP", "1")
    assert windows_settings.get("dock_on_startup") is True


# ── startup ──────────────────────────────────────────────────────────────────

def test_startup_builds_and_shows_no_dock_by_default(start):
    running = start()
    assert running.presence.docks == []            # not even built
    assert running.parts.dock is None
    assert callable(running.window._dock_control) and running.window._dock_visible() is False


def test_a_tray_sign_in_start_shows_only_the_tray_icon(start, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ORION.exe", "--tray"])
    assert app._start_in_tray()
    running = start()
    assert running.presence.docks == [] and running.parts.dock is None
    assert not any("dock" in str(line).lower() and "showing" in str(line).lower()
                   for line in running.bus.log.calls)


def test_startup_shows_the_dock_when_the_switch_is_on(start):
    running = start(dock_on_startup=True)
    (dock,) = running.presence.docks
    assert dock.wanted and dock.shown == 1 and dock.bus is running.bus
    assert running.window._dock_visible() is True


def test_dock_enabled_off_beats_the_startup_switch(start):
    running = start(dock_enabled=False, dock_on_startup=True)
    assert running.presence.docks == []


# ── asking for it ────────────────────────────────────────────────────────────

def test_asking_for_the_dock_builds_it_once_and_shows_it(start):
    running = start()
    assert running.window._dock_control(True) is True
    (dock,) = running.presence.docks
    assert dock.wanted and running.parts.dock is dock
    assert running.window._dock_control(True) is True
    assert len(running.presence.docks) == 1 and dock.shown == 2     # reused, shown again


def test_hiding_and_toggling(start):
    running = start()
    control = running.window._dock_control
    assert control(False) is False and running.presence.docks == []   # nothing to hide, nothing built
    assert control(None) is True                                      # toggle on
    assert control(None) is False                                     # toggle off
    (dock,) = running.presence.docks
    assert dock.hidden == 1 and not dock.wanted
    assert running.window._dock_visible() is False
    assert control(None) is True and dock.wanted


def test_asking_for_the_dock_switches_a_disabled_dock_back_on(start):
    running = start(dock_enabled=False)
    assert running.window._dock_control(True) is True
    assert windows_settings.get("dock_enabled") is True


def test_hide_dock_in_its_own_menu_means_not_at_the_next_start(start):
    running = start(dock_on_startup=True)
    (dock,) = running.presence.docks
    dock.kwargs["on_hide"]()
    assert windows_settings.get("dock_on_startup") is False
    assert windows_settings.get("dock_enabled") is True           # the dock stays available


def test_the_jump_list_task_asks_for_the_dock(start):
    running = start()
    running.on_command({"action": "dock", "state": "show", "origin": "cli"})
    (dock,) = running.presence.docks
    assert dock.wanted
    running.on_command({"action": "dock", "state": "hide", "origin": "cli"})
    assert not dock.wanted


def test_no_presence_means_a_logged_refusal_not_a_crash(start, monkeypatch):
    from orion_core import windows_presence

    def boom(bus, **kw):
        raise RuntimeError("no presence")

    monkeypatch.setattr(windows_presence, "start_windows_presence", boom)
    running = start()
    assert running.window._dock_control(True) is False
    assert any("dock" in str(c).lower() for c in running.bus.log.calls)


# ── the tray menu ────────────────────────────────────────────────────────────

def test_the_tray_entry_follows_the_dock(monkeypatch):
    from orion_core import system_tray as st

    assert (st.LABEL_SHOW_DOCK, st.LABEL_HIDE_DOCK) == ("Show Dock", "Hide Dock")
    assert st.dock_menu_label(False) == "Show Dock" and st.dock_menu_label(True) == "Hide Dock"
    monkeypatch.setattr(st, "tray_available", lambda: False)
    asked = []
    window = SimpleNamespace(_dock_control=lambda show=None: asked.append(show) or True,
                             _dock_visible=lambda: bool(asked))
    tray = st.OrionTray(app=object(), window=window, bus=SimpleNamespace())
    assert tray.dock_visible() is False
    tray.toggle_dock()
    assert asked == [None] and tray.dock_visible() is True


def test_the_tray_dock_entry_is_safe_without_a_dock_control(monkeypatch):
    from orion_core import system_tray as st

    monkeypatch.setattr(st, "tray_available", lambda: False)
    tray = st.OrionTray(app=object(), window=SimpleNamespace(), bus=SimpleNamespace())
    tray.toggle_dock()                               # no control installed: a no-op
    assert tray.dock_visible() is False


def test_the_real_tray_menu_has_show_dock_between_show_and_overlay():
    source = (ROOT / "orion_core" / "system_tray.py").read_text(encoding="utf-8")
    assert source.index("(LABEL_SHOW, self.show_window)") < source.index(
        "(LABEL_SHOW_DOCK, self.toggle_dock)") < source.index("(LABEL_OVERLAY, self.overlay)")
    assert "aboutToShow.connect(self._refresh_dock_entry)" in source


# ── Preferences ──────────────────────────────────────────────────────────────

def test_preferences_has_the_show_dock_at_startup_switch():
    from orion_core.gui.labels import title_case
    from orion_core.gui.preferences import WINDOWS_TOGGLES

    rows = {key: (label, hint) for key, label, hint in WINDOWS_TOGGLES}
    label, hint = rows["dock_on_startup"]
    assert label == "Show Dock at Startup" and title_case(label) == label
    assert "only appears when you ask" in hint
    assert "tray" in rows["start_in_tray_at_login"][1].lower()
    assert "dock showing" not in rows["start_in_tray_at_login"][1]


# ── voice ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,target", [
    ("show your dock", "show"), ("show the dock", "show"), ("bring up your dock", "show"),
    ("open the dock please", "show"), ("dock yourself", "show"), ("ORION, show your dock", "show"),
    ("can you show your docked face", "show"),
    ("hide your dock", "hide"), ("put away the dock", "hide"), ("undock yourself", "hide"),
    ("close your dock.", "hide"),
])
def test_voice_phrases_for_the_dock(phrase, target):
    from orion_core.reflex import match_reflex

    match = match_reflex(phrase)
    assert match is not None, phrase
    assert match.tool == "interface_control"
    assert match.args == {"action": "dock", "target": target}


@pytest.mark.parametrize("phrase", ["show docker containers", "open the dock yard report", "dockerize it"])
def test_the_dock_reflex_does_not_swallow_other_sentences(phrase):
    from orion_core.reflex import match_reflex

    match = match_reflex(phrase)
    assert match is None or match.args.get("action") != "dock"


@pytest.mark.parametrize("action,target,emitted", [
    ("dock", "show", "show_dock"), ("dock", "hide", "hide_dock"),
    ("show_dock", "", "show_dock"), ("hide_dock", "", "hide_dock"), ("dock", "", "show_dock"),
])
def test_the_tool_puts_the_dock_command_on_the_bus(action, target, emitted):
    from orion_core.dispatch_desktop import DesktopDispatchMixin

    sent = []
    tool = DesktopDispatchMixin()
    tool.bus = SimpleNamespace(gui_command=SimpleNamespace(emit=sent.append))
    result = tool.interface_control({"action": action, "target": target})
    assert result.ok and "dock" in result.text.lower()
    assert sent == [{"action": emitted, "target": ""}]


def test_the_core_window_hands_a_dock_command_to_the_control():
    from orion_core.gui.core_window import OrionCoreWindow

    asked = []
    stub = SimpleNamespace(bus=SimpleNamespace(log=_Signal()),
                           _dock_control=lambda show=None: asked.append(show))
    OrionCoreWindow._on_gui_command(stub, {"action": "show_dock", "target": ""})
    OrionCoreWindow._on_gui_command(stub, {"action": "hide_dock", "target": ""})
    OrionCoreWindow._on_gui_command(stub, {"action": "dock", "target": "hide"})
    assert asked == [True, False, False]


def test_a_bare_window_without_a_control_says_so():
    from orion_core.gui.core_window import OrionCoreWindow

    stub = SimpleNamespace(bus=SimpleNamespace(log=_Signal()))
    OrionCoreWindow._on_gui_command(stub, {"action": "show_dock", "target": ""})
    assert "not available" in stub.bus.log.calls[0][0]


def test_the_schema_tells_the_model_about_the_dock():
    source = (ROOT / "orion_core" / "dispatch_schema.py").read_text(encoding="utf-8")
    assert "dock (put your docked Screen Face on the screen edge" in source
