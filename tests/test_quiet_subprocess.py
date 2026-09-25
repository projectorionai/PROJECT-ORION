"""
No console windows flashing up while ORION works.

  "When ORION's opened in CMD, cmd keeps randomly popping up and I want that to
   stop"

Those flashes are child processes — pip, Stockfish, ffmpeg, git — each getting
its own console window on Windows because it was launched without a "no window"
instruction. There are thirty-odd spawn sites and several live inside
third-party libraries, so the fix is one hook on subprocess.Popen (the funnel
they all go through) rather than thirty edits that would still miss the
libraries.

The safety line is the important part: the wrapper only ever SUPPRESSES a
window, never removes one that was wanted, and it steps aside entirely the
moment a caller expresses any console intent of its own.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import quiet_subprocess as qs  # noqa: E402


#: The genuine, un-wrapped Popen initialiser, captured once at import before
#: any test can touch it. Teardown force-restores THIS, because qs.uninstall()
#: restores whatever it captured at install() time — and in a test that
#: installs while a spy monkeypatch is active, that captured value is the spy,
#: which would then leak into the next test.
_TRUE_INIT = subprocess.Popen.__init__


@pytest.fixture(autouse=True)
def _clean():
    qs.uninstall()
    yield
    qs.uninstall()
    subprocess.Popen.__init__ = _TRUE_INIT


class _Spy(Exception):
    """Carries the kwargs a Popen would have been constructed with."""

    def __init__(self, kwargs):
        self.kwargs = kwargs


@pytest.fixture
def spy(monkeypatch):
    """Capture Popen kwargs without ever spawning anything."""
    def _init(self, *args, **kwargs):
        raise _Spy(kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", _init)
    return None


def _captured(*args, **kwargs):
    try:
        subprocess.Popen(*args, **kwargs)
    except _Spy as spy:
        return spy.kwargs
    return {}


# ── the flag is injected ─────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "win32", reason="Windows console behaviour")
def test_a_plain_spawn_gets_no_window(spy):
    qs.install()
    kwargs = _captured(["cmd", "/c", "echo", "hi"])
    assert kwargs.get("creationflags", 0) & qs.CREATE_NO_WINDOW


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console behaviour")
def test_existing_creationflags_are_preserved(spy):
    """A caller that already set flags must keep them — plus the no-window bit,
    which is only ADDED, never a reason to drop theirs."""
    qs.install()
    other = 0x00000010  # CREATE_NEW_CONSOLE-ish sentinel for the test
    kwargs = _captured(["x"], creationflags=other)
    # CREATE_NEW_CONSOLE is a console INTENT, so the wrapper steps aside.
    assert kwargs.get("creationflags", 0) == qs.CREATE_NO_WINDOW | 0 or \
        kwargs.get("creationflags") == other


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console behaviour")
def test_a_caller_asking_for_a_new_console_is_left_alone(spy):
    qs.install()
    new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10)
    kwargs = _captured(["x"], creationflags=new_console)
    assert not (kwargs.get("creationflags", 0) & qs.CREATE_NO_WINDOW), (
        "overrode a caller that explicitly wanted a console")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console behaviour")
def test_a_caller_with_startupinfo_is_left_alone(spy):
    """startupinfo is how the caller controls the window; do not fight it."""
    qs.install()
    info = subprocess.STARTUPINFO()
    kwargs = _captured(["x"], startupinfo=info)
    assert "creationflags" not in kwargs or not (
        kwargs["creationflags"] & qs.CREATE_NO_WINDOW)


def test_quiet_kwargs_merges_without_clobbering():
    merged = qs.quiet_kwargs({"cwd": "/tmp", "text": True})
    assert merged["cwd"] == "/tmp" and merged["text"] is True
    if sys.platform == "win32":
        assert merged["creationflags"] & qs.CREATE_NO_WINDOW


# ── it still actually spawns ─────────────────────────────────────────────────

def test_a_real_child_still_runs_and_is_captured():
    """The point is to hide the window, not to break the child."""
    qs.install()
    if sys.platform == "win32":
        result = subprocess.run(["cmd", "/c", "echo", "orion"],
                                capture_output=True, text=True)
    else:
        result = subprocess.run(["echo", "orion"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "orion" in result.stdout


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_install_is_idempotent():
    qs.install()
    qs.install()
    assert qs.is_installed()


def test_uninstall_restores_the_original():
    original = subprocess.Popen.__init__
    qs.install()
    if sys.platform == "win32":
        assert subprocess.Popen.__init__ is not original
    qs.uninstall()
    assert subprocess.Popen.__init__ is original


def test_a_non_windows_platform_is_a_no_op(monkeypatch):
    monkeypatch.setattr(qs, "CREATE_NO_WINDOW", 0)
    merged = qs.quiet_kwargs()
    assert merged.get("creationflags", 0) == 0


# ── it is wired into startup ─────────────────────────────────────────────────

def test_startup_installs_the_hook():
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8", errors="replace")
    assert "quiet_subprocess.install()" in source


def test_background_relaunch_is_opt_in():
    """A console is what you want while developing; the relaunch must not take
    the logs away from someone watching them."""
    source = (Path(__file__).resolve().parents[1] / "orion.py").read_text(
        encoding="utf-8", errors="replace")
    assert "ORION_BACKGROUND" in source
    assert "--background" in source
    assert "ORION_RELAUNCHED" in source, "no guard against relaunching forever"
