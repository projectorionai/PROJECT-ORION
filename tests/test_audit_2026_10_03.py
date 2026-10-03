"""
Regression tests for the smaller defects found by the 2026-10-03 audit.

Each was reproduced against a running ORION first — a full desktop boot under
Qt's offscreen platform, a headless node driven over HTTP, and a pass that
called every declared tool with each of its actions — so every test here
describes something that actually went wrong, not a hypothetical.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6.QtWidgets")


def _run(coro):
    return asyncio.run(coro)


# ── Ctrl+C and quit ─────────────────────────────────────────────────────────

class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def fire(self):
        for slot in list(self.slots):
            slot()


class _FakeApp:
    def __init__(self):
        self.aboutToQuit = _FakeSignal()


def test_a_quit_before_shutdown_is_wired_is_remembered_and_honoured():
    from orion_core.app import _QuitLatch

    app = _FakeApp()
    latch = _QuitLatch(app)
    app.aboutToQuit.fire()                # Ctrl+C mid-boot
    calls = []
    assert latch.honour(lambda: calls.append("shutdown")) is True
    assert calls == ["shutdown"]


def test_no_quit_means_no_shutdown():
    from orion_core.app import _QuitLatch

    latch = _QuitLatch(_FakeApp())
    calls = []
    assert latch.honour(lambda: calls.append("shutdown")) is False
    assert calls == []


def test_ctrl_c_leaves_the_loop_with_exit_not_quit():
    """Qt 6's quit() asks every window to close and gives up if one refuses;
    ORION's windows refuse by design (they hide to the tray). Measured: about
    half of all Ctrl+C presses were ignored."""
    from orion_core import app as app_module

    source = inspect.getsource(app_module.main)
    assert "signal.signal(signal.SIGINT, lambda *_: app.exit(0))" in source
    assert "lambda *_: app.quit()" not in source


def test_qt_quit_really_is_vetoed_by_a_window_that_refuses_to_close():
    """The behaviour the fix depends on, checked against the installed Qt.

    Every timer is a stoppable object, stopped before returning: a stray
    single-shot exit() would fire later inside another test's event loop."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])

    class Stubborn(QWidget):
        def closeEvent(self, event):  # noqa: N802
            event.ignore()

    def run(first) -> bool:
        """Run a loop that *first* should end; True if the watchdog had to."""
        fallback = {"needed": False}
        kick, watchdog = QTimer(), QTimer()
        kick.setSingleShot(True)
        watchdog.setSingleShot(True)
        kick.timeout.connect(first)
        watchdog.timeout.connect(lambda: (fallback.__setitem__("needed", True), app.exit(0)))
        kick.start(0)
        watchdog.start(300)
        try:
            app.exec()
        finally:
            kick.stop()
            watchdog.stop()
        return fallback["needed"]

    window = Stubborn()
    window.show()
    try:
        assert run(app.quit), "quit() was not vetoed; the exit(0) fix is moot"
        assert not run(lambda: app.exit(0)), "exit(0) must leave the loop regardless"
    finally:
        window.hide()
        window.deleteLater()


# ── patch notes describe the build that is running ─────────────────────────

def test_the_newest_release_is_the_running_version():
    from orion_core import __version__
    from orion_core.changelog import RELEASES

    newest = RELEASES[0].version.split(".")[0]
    assert newest == __version__.split(".")[0], (
        f"changelog stops at {RELEASES[0].version} while the package is "
        f"{__version__}: 'what's new?' would describe an old release")


@pytest.mark.parametrize("query, codename", [
    ("32", "Mark XXXII"),
    ("v31.0", "Mark XXXI"),
    ("XXXI", "Mark XXXI"),
    ("xxx", "Mark XXX —"),
    ("forge", "The Forge Awakens"),
])
def test_releases_are_found_by_version_or_codename_word(query, codename):
    from orion_core.changelog import Changelog

    release = Changelog().find(query)
    assert release is not None and release.codename.startswith(codename)


# ── tools that must not default to the project itself ──────────────────────

class _Bus:
    def __getattr__(self, name):
        class _S:
            def emit(self, *a):
                pass
        return _S()


@pytest.mark.parametrize("action", ["write_text", "append_text", "delete", "mkdir"])
def test_file_controller_mutations_need_an_explicit_path(action):
    from orion_core.dispatch_files import FilesDispatchMixin

    class _Host(FilesDispatchMixin):
        pass

    result = _Host().file_controller({"action": action, "text": "x"})
    assert not result.ok
    assert "needs a path" in result.text


def test_learning_a_folder_needs_a_folder(tmp_path):
    from orion_core.learning import LearningService

    service = LearningService.__new__(LearningService)
    service.bus = _Bus()
    result = _run(service.learn_folder(""))
    assert not result.ok and "Which folder" in result.text


def test_code_changes_runs_off_the_event_loop():
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    assert inspect.iscoroutinefunction(KnowledgeDispatchMixin.code_changes_tool)
    source = inspect.getsource(KnowledgeDispatchMixin.code_changes_tool)
    assert "asyncio.to_thread" in source


# ── the web controller ─────────────────────────────────────────────────────

def _web():
    from orion_core.web import WebController

    return WebController.__new__(WebController)


def test_about_blank_is_not_turned_into_an_https_url():
    web = _web()
    assert web._normalise("about:blank") == "about:blank"
    assert web._normalise("") == "about:blank"
    assert web._normalise("example.com") == "https://example.com"


class _Recorder:
    def __init__(self):
        self.typed = []
        self.keys = []

    def type_text(self, text):
        self.typed.append(text)

    def send_hotkeys(self, keys):
        self.keys.append(keys)


class _Vision:
    def __init__(self, text):
        self.text = text

    async def detect_dialogs(self):
        from orion_core.data import ToolResult
        return ToolResult(self.text)


def test_the_file_dialog_step_never_types_without_a_dialog():
    web = _web()
    web.control = _Recorder()
    web.vision = _Vision("No open dialogs or pop-ups detected.")
    result = _run(web.handle_file_dialog("C:/Users/me/report.pdf"))
    assert not result.ok
    assert web.control.typed == [] and web.control.keys == []


def test_the_file_dialog_step_needs_a_path():
    web = _web()
    web.control = _Recorder()
    web.vision = _Vision("Open dialogs / pop-ups:\n- 'Open' [Window]")
    result = _run(web.handle_file_dialog("   "))
    assert not result.ok and web.control.typed == []


def test_the_file_dialog_step_types_into_a_dialog_that_is_there():
    web = _web()
    web.control = _Recorder()
    web.vision = _Vision("Open dialogs / pop-ups:\n- 'File Upload' [Window] @ (0, 0, 600, 400)")
    result = _run(web.handle_file_dialog("C:/Users/me/report.pdf"))
    assert result.ok
    assert web.control.typed == ["C:/Users/me/report.pdf"]
    assert web.control.keys == ["enter"]


# ── the cursor halo off Windows ────────────────────────────────────────────

def test_cursor_position_off_windows_uses_qt_not_a_failing_import(monkeypatch):
    from orion_core.gui import cursor_overlay as co

    monkeypatch.setattr(co, "_GET_CURSOR_POS", None)
    monkeypatch.setattr(co, "_scaled_display", lambda: False)
    imported = []
    real_import = __import__

    def spy(name, *a, **k):
        if name == "pyautogui":
            imported.append(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", spy)
    position = co.CursorOverlay._cursor_pos(None)
    assert isinstance(position, tuple) and len(position) == 2
    assert imported == []


# ── installation and deployment ────────────────────────────────────────────

_WINDOWS_ONLY = re.compile(r"^(pywin32|uiautomation|winrt-[\w.]+|dxcam)\b", re.IGNORECASE)


def test_windows_only_requirements_carry_a_platform_marker():
    """Without markers ``pip install -r requirements.txt`` is unsatisfiable on
    Linux and macOS (pywin32 and winrt publish Windows wheels only)."""
    unmarked = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec and _WINDOWS_ONLY.match(spec) and 'sys_platform == "win32"' not in spec:
            unmarked.append(spec)
    assert not unmarked, unmarked


def test_the_pairing_qr_dependency_is_declared():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^qrcode\b", text, re.MULTILINE)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_sync_script_reads_the_config_wherever_it_was_moved(tmp_path):
    script = ROOT / "deploy" / "sync_state.sh"
    home = tmp_path / "home"
    data = tmp_path / "data"
    home.mkdir()
    data.mkdir()

    def where(**env):
        merged = {k: v for k, v in os.environ.items() if k != "ORION_CONFIG_DIR"}
        merged.update(env)
        return subprocess.run(["bash", str(script), "where"], env=merged,
                              capture_output=True, text=True, check=True).stdout.strip()

    assert where(ORION_LOCAL_HOME=str(home)) == f"{home}/config"
    (home / "config_location.txt").write_text(f"{data}\n", encoding="utf-8")
    assert where(ORION_LOCAL_HOME=str(home)) == str(data)
    assert where(ORION_LOCAL_HOME=str(home), ORION_CONFIG_DIR="/explicit") == "/explicit"


def test_sync_script_only_ships_databases_orion_writes():
    script = (ROOT / "deploy" / "sync_state.sh").read_text(encoding="utf-8")
    shared = re.search(r"SHARED=\(\n(.*?)\n\)", script, re.DOTALL).group(1).split()
    sources = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                        for p in (ROOT / "orion_core").rglob("*.py"))
    for name in shared:
        assert f'"{name}"' in sources, f"{name} is synced but nothing writes it"


def test_sync_script_refuses_to_push_under_a_running_node():
    script = (ROOT / "deploy" / "sync_state.sh").read_text(encoding="utf-8")
    assert "could not stop it — continuing\"\n" not in script
    assert 'ORION_FORCE' in script and "exit 1" in script


# ── shutdown noise and node durability ─────────────────────────────────────

def test_a_telemetry_step_after_the_loop_stopped_ends_quietly():
    """The last step of an exit used to raise twice and print a traceback
    after "ORION has shutdown cleanly"."""
    from orion_core.gui.core_window import OrionCoreWindow

    class _Window:
        def write_log(self, line):
            raise AssertionError("must not try to recover with no loop running")

    step = OrionCoreWindow.start_telemetry(_Window())
    with pytest.raises(StopIteration):      # returned, rather than raising
        step.send(None)


def test_the_brain_animator_tolerates_a_timer_already_destroyed():
    from PyQt6 import sip
    from PyQt6.QtWidgets import QApplication

    from orion_core.gui.brain_instruments import _Animator

    QApplication.instance() or QApplication([])

    class _Model:
        class _Changed:
            def connect(self, slot):
                pass

        changed = _Changed()

        def needs_frames(self):
            return True

    animator = _Animator(_Model(), [])
    sip.delete(animator.timer)               # what exit teardown does first
    animator.set_visible(False)              # used to raise RuntimeError
    animator.set_visible(True)


def test_the_headless_node_checkpoints_like_the_desktop():
    from orion_core import server

    source = inspect.getsource(server.run_headless)
    assert "housekeeper.run_checkpoints()" in source
    assert "housekeeper.checkpoint_all" in source
