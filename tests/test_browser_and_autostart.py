"""
Which browser ORION opens, and starting with the machine.

  "ORION must use Microsoft Edge or msedge.exe as the main browser it goes to -
   the default browser must be used each time."
  "I want ORION to be able to turn on when my PC starts up."

The browser half is two requirements that only look like one: ORION must not
have a browser of his own choosing at all — he should land wherever the user's
own links land, with their profile and their sessions — and on this machine
that resolves to Edge.

The autostart half is mostly about the failure mode that produces NOTHING: an
entry pointing at the wrong interpreter or a moved project runs at sign-in,
fails on an import, and shows the user no indication whatsoever. Most of what
is tested here is that this cannot happen quietly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import autostart, browser  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_browser_cache():
    browser.reset_cache()
    yield
    browser.reset_cache()


# ── the browser ──────────────────────────────────────────────────────────────

def test_an_explicit_override_wins(monkeypatch, tmp_path):
    """A user who names a browser has settled the question."""
    fake = tmp_path / "somebrowser.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv(browser.ENV_OVERRIDE, str(fake))
    browser.reset_cache()
    assert browser.browser_path() == str(fake)


def test_the_os_default_outranks_edge(monkeypatch):
    """'The default browser must be used each time' — even when Edge exists."""
    monkeypatch.delenv(browser.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(browser, "default_browser_path", lambda: r"C:\firefox.exe")
    monkeypatch.setattr(browser, "edge_path", lambda: r"C:\msedge.exe")
    browser.browser_path.cache_clear()
    assert browser.browser_path() == r"C:\firefox.exe"


def test_edge_is_the_fallback_when_no_default_is_readable(monkeypatch):
    monkeypatch.delenv(browser.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(browser, "default_browser_path", lambda: None)
    monkeypatch.setattr(browser, "edge_path", lambda: r"C:\msedge.exe")
    browser.browser_path.cache_clear()
    assert browser.browser_path() == r"C:\msedge.exe"


@pytest.mark.parametrize("command,expected", [
    ('"C:\\Program Files\\Microsoft\\Edge\\msedge.exe" --single-argument %1',
     "C:\\Program Files\\Microsoft\\Edge\\msedge.exe"),
    ('"C:\\browser.exe"', "C:\\browser.exe"),
])
def test_a_quoted_registry_command_yields_the_executable(command, expected, monkeypatch):
    """Registry shell commands are not paths — they carry switches and %1."""
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert browser._executable_from_command(command) == expected


def test_an_unquoted_command_is_cut_at_the_switch_not_the_first_space(monkeypatch):
    """Windows paths are full of spaces; cutting at the first one is wrong."""
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert browser._executable_from_command(
        r"C:\Program Files\App\b.exe -- %1") == r"C:\Program Files\App\b.exe"


def test_a_command_pointing_nowhere_is_rejected():
    assert browser._executable_from_command(
        r'"C:\definitely\not\here\nope.exe" %1') is None
    assert browser._executable_from_command("") is None


def test_opening_nothing_does_nothing():
    assert browser.open_url("") is False
    assert browser.open_url("   ") is False


def test_the_url_goes_to_the_resolved_browser(monkeypatch):
    launched = {}

    class _Popen:
        def __init__(self, argv, **kwargs):
            launched["argv"] = argv

    monkeypatch.setattr(browser, "browser_path", lambda: r"C:\msedge.exe")
    monkeypatch.setattr(browser.subprocess, "Popen", _Popen)
    assert browser.open_url("https://example.com")
    assert launched["argv"] == [r"C:\msedge.exe", "https://example.com"]


def test_a_failed_launch_falls_back_rather_than_raising(monkeypatch):
    """A missing browser must not take down whatever asked for the link."""
    monkeypatch.setattr(browser, "browser_path", lambda: r"C:\gone.exe")
    monkeypatch.setattr(browser.subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: True)
    assert browser.open_url("https://example.com") is True


def test_the_name_is_speakable(monkeypatch):
    monkeypatch.setattr(browser, "browser_path", lambda: r"C:\x\msedge.exe")
    browser.browser_name.cache_clear()
    assert browser.browser_name() == "Microsoft Edge"


def test_every_link_opener_routes_through_the_resolver():
    """One place decides which browser opens, or the rule is not enforced.

    Call sites used to call webbrowser.open directly, which honours a BROWSER
    environment variable and the FILE association rather than the https
    PROTOCOL association — so ORION could open links somewhere the user's own
    clicks never go.
    """
    root = Path(__file__).resolve().parents[1] / "orion_core"
    offenders = []
    for path in list(root.glob("*.py")) + list(root.glob("gui/*.py")):
        if path.name == "browser.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "webbrowser.open(" in text:
            offenders.append(path.name)
    assert not offenders, f"still calling webbrowser.open directly: {offenders}"


def test_the_cdp_copilot_drives_the_same_browser():
    """The co-pilot used to hard-code Chrome ahead of everything, so ORION
    drove a browser with none of the user's logins while every other part of
    him opened Edge."""
    source = (Path(__file__).resolve().parents[1]
              / "orion_core" / "browser_copilot.py").read_text(
                  encoding="utf-8", errors="replace")
    find = source[source.index("def find_browser"):]
    find = find[:find.index("\n    # ")]
    assert "from .browser import browser_path" in find
    assert find.index("msedge.exe") < find.index("chrome.exe")


# ── starting with Windows ────────────────────────────────────────────────────

def test_the_entry_point_actually_exists():
    """Registering a command that runs nothing is the whole failure mode."""
    assert autostart.entry_script().is_file()


def test_the_command_quotes_its_paths():
    """This project lives under a path containing spaces (a OneDrive folder).
    An unquoted Run value silently launches nothing at all. The exe form quotes
    one path (2 quotes); the interpreter form quotes two (4)."""
    command = autostart.startup_command()
    assert command.startswith('"')
    expected = 2 if autostart.best_launch_exe() is not None else 4
    assert command.count('"') == expected


def test_best_launch_exe_prefers_the_standalone_over_the_launcher(monkeypatch, tmp_path):
    """The whole 'still opens as Python 3.13' fix: when the frozen standalone
    (its own process) exists, it must win over the launcher ORION.exe (which
    spawns pythonw and therefore shows as Python)."""
    monkeypatch.setattr(autostart, "entry_script", lambda: tmp_path / "orion.py")
    (tmp_path / "ORION.exe").write_text("launcher", encoding="utf-8")
    dist = tmp_path / "dist" / "ORION"
    dist.mkdir(parents=True)
    (dist / "ORION.exe").write_text("standalone", encoding="utf-8")
    assert autostart.best_launch_exe() == dist / "ORION.exe"


def test_best_launch_exe_falls_back_to_the_launcher(monkeypatch, tmp_path):
    monkeypatch.setattr(autostart, "entry_script", lambda: tmp_path / "orion.py")
    (tmp_path / "ORION.exe").write_text("launcher", encoding="utf-8")
    assert autostart.best_launch_exe() == tmp_path / "ORION.exe"


def test_best_launch_exe_is_none_when_no_executable_is_built(monkeypatch, tmp_path):
    monkeypatch.setattr(autostart, "entry_script", lambda: tmp_path / "orion.py")
    assert autostart.best_launch_exe() is None


def test_startup_command_registers_the_standalone_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(autostart, "entry_script", lambda: tmp_path / "orion.py")
    dist = tmp_path / "dist" / "ORION"
    dist.mkdir(parents=True)
    (dist / "ORION.exe").write_text("standalone", encoding="utf-8")
    command = autostart.startup_command()
    assert command.startswith('"') and command.count('"') == 2
    assert "dist" in command and "ORION.exe" in command


def test_the_interpreter_is_the_one_orion_is_running_under():
    """Not a guess about where Python lives: this project has a .venv AND a
    separate system Python, and they do not have the same packages."""
    interpreter = Path(autostart.launch_interpreter())
    assert interpreter.exists()
    assert interpreter.parent == Path(sys.executable).parent


def test_a_console_window_is_not_left_open_at_logon(monkeypatch):
    """python.exe at sign-in leaves a console behind the UI all session."""
    interpreter = Path(autostart.launch_interpreter())
    twin = interpreter.with_name(interpreter.name.replace("pythonw", "python"))
    if twin.exists() and twin != interpreter:
        assert "pythonw" in interpreter.name


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only facility")
def test_status_reports_a_real_registry_state():
    state = autostart.status()
    assert state.supported
    assert isinstance(state.enabled, bool)
    assert state.describe()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only facility")
def test_a_registered_entry_matches_what_would_be_registered_now(monkeypatch):
    """is_stale() is the difference between 'starts with Windows' and 'the
    user believes it starts with Windows'."""
    expected = '"C:\\example\\ORION.exe"'
    monkeypatch.setattr(autostart, "startup_command", lambda: expected)
    monkeypatch.setattr(autostart, "status", lambda: autostart.AutostartState(
        supported=True, enabled=True, command=expected))
    assert not autostart.is_stale()
    monkeypatch.setattr(autostart, "status", lambda: autostart.AutostartState(
        supported=True, enabled=True, command='"C:\\old-example\\ORION.exe"'))
    assert autostart.is_stale()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only facility")
def test_enable_refuses_an_interpreter_that_cannot_run_orion(monkeypatch, tmp_path):
    """Better to register nothing than to register something that fails
    silently at every sign-in.

    This guard is for the raw-interpreter path only — a built ORION.exe carries
    its own interpreter resolution, so force the no-exe path to exercise it.
    """
    monkeypatch.setattr(autostart, "entry_script", lambda: tmp_path / "orion.py")
    (tmp_path / "orion.py").write_text("", encoding="utf-8")   # exists, no exe beside it
    monkeypatch.setattr(autostart, "interpreter_can_run_orion", lambda *a, **k: False)
    state = autostart.enable()
    assert not state.enabled
    assert "PyQt6" in state.detail


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only facility")
def test_enable_refuses_when_the_entry_point_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(autostart, "entry_script", lambda: tmp_path / "gone.py")
    state = autostart.enable()
    assert not state.enabled
    assert "entry point" in state.detail


def test_a_non_windows_machine_says_so_rather_than_failing(monkeypatch):
    monkeypatch.setattr(autostart.sys, "platform", "linux")
    state = autostart.status()
    assert not state.supported
    assert "Windows" in state.describe()


def test_the_probe_uses_the_console_twin(monkeypatch):
    """pythonw.exe has no console and cannot report a result back."""
    seen = {}

    class _Done:
        returncode = 0

    def _run(argv, **kwargs):
        seen["exe"] = argv[0]
        return _Done()

    import subprocess

    monkeypatch.setattr(subprocess, "run", _run)
    interpreter = Path(autostart.launch_interpreter())
    autostart.interpreter_can_run_orion(str(interpreter))
    if "pythonw" in interpreter.name:
        assert "pythonw" not in Path(seen["exe"]).name
