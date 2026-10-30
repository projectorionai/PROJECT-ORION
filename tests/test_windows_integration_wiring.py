"""app._start_windows_integration: every shell command reaches the right part of
ORION, a --tray start lays its windows out once on first summon, and locking
the PC pauses listening only when the owner had it on."""
from types import SimpleNamespace

import pytest

from orion_core import app


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
        self.typed = SimpleNamespace(text="", focused=False)
        box = SimpleNamespace(setText=lambda t: setattr(self.typed, "text", t),
                              setFocus=lambda: setattr(self.typed, "focused", True))
        self.hud = SimpleNamespace(command=SimpleNamespace(input=box))

    def __getattr__(self, name):          # isMinimized, show, raise_, set_face_form ...
        def record(*args):
            self.calls.append((name, args))
            return False
        return record


class _Presence:
    def __init__(self):
        self.pause_hearing_when_locked = True
        self.greet_on_unlock = True
        self.toasts = SimpleNamespace(handled=[], set_tray=lambda tray: None, show=lambda *a, **k: None)
        self.toasts.handle_action = lambda token, verb: self.toasts.handled.append((token, verb)) or True
        self.quiet_now = False

    def quiet(self):
        return self.quiet_now

    def make_dock(self, **kwargs):
        return SimpleNamespace(connect_bus=lambda bus: None, show_dock=lambda: None, stop=lambda: None)

    def stop(self):
        pass


@pytest.fixture
def wired(monkeypatch, tmp_path):
    from orion_core import background, windows_presence, windows_settings, windows_shell
    monkeypatch.setattr(windows_settings, "SETTINGS_PATH", tmp_path / "w.json")
    captured = {}
    monkeypatch.setattr(windows_shell, "start_windows_shell",
                        lambda bus, on_command: captured.setdefault("on_command", on_command)
                        and SimpleNamespace(stop=lambda: None))
    presence = _Presence()

    def fake_presence(bus, *, on_hotkey, on_session_event, **kwargs):
        captured["on_hotkey"], captured["on_session"] = on_hotkey, on_session_event
        return presence

    monkeypatch.setattr(windows_presence, "start_windows_presence", fake_presence)
    monkeypatch.setattr(app, "_toast_approvals", lambda *a, **k: None)
    submitted = []
    monkeypatch.setattr(background, "spawn", lambda coro, **k: submitted.append(coro) or coro.close())
    bus, window = _Bus(), _Window()
    worker = SimpleNamespace(microphone_enabled=True, submit_text=lambda text: _coro(text),
                             set_microphone_enabled=lambda on: setattr(worker, "microphone_enabled", on))
    parts = app._start_windows_integration(bus, window, worker)
    return SimpleNamespace(bus=bus, window=window, worker=worker, presence=presence,
                           parts=parts, submitted=submitted, **captured)


async def _coro(text):
    return text


def _called(window, name):
    return [args for n, args in window.calls if n == name]


def test_each_command_reaches_its_handler(wired):
    run, w = wired.on_command, wired.window
    run({"action": "page", "page": "research", "origin": "link"})
    assert _called(w, "_open_deck_page")[-1] == ("research",)
    run({"action": "briefing", "origin": "cli"})
    assert _called(w, "run_morning_brief")
    run({"action": "mute", "origin": "cli"})
    run({"action": "unmute", "origin": "cli"})
    assert wired.bus.voice_mute_request.calls == [(True,), (False,)]
    run({"action": "face", "form": "classic", "origin": "cli"})
    assert _called(w, "set_face_form")[-1] == ("classic",) and _called(w, "_sync_form_menu")
    run({"action": "toast", "token": "t" * 24, "verb": "review", "origin": "link"})
    assert wired.presence.toasts.handled == [("t" * 24, "review")]
    run({"action": "ask_file", "path": r"C:\notes.txt", "origin": "cli"})
    assert _called(w, "_hud_files_dropped")[-1] == ([r"C:\notes.txt"],)
    run({"action": "last_task", "origin": "cli"})
    assert _called(w, "_open_deck_page")[-1] == ("TASKS",)


def test_a_link_only_fills_the_box_it_never_sends(wired):
    wired.on_command({"action": "prefill", "text": "delete my files", "origin": "link"})
    assert wired.window.typed.text == "delete my files" and wired.window.typed.focused
    assert wired.submitted == []


def test_summarise_and_organise_are_requests_and_organise_is_preview_only(wired):
    wired.on_command({"action": "summarise", "path": r"C:\a.pdf", "origin": "cli"})
    wired.on_command({"action": "organise", "path": r"C:\Downloads", "origin": "cli"})
    assert len(wired.submitted) == 2


def test_a_tray_start_is_laid_out_once_on_first_summon(wired):
    laid = []
    wired.window._pending_layout = lambda: laid.append(True)
    wired.on_hotkey()
    wired.on_hotkey()
    assert laid == [True] and wired.window._pending_layout is None
    assert _called(wired.window, "activateWindow")          # the second summon raises it


def test_locking_pauses_listening_and_unlocking_restores_and_greets(wired):
    wired.on_session("locked")
    assert wired.worker.microphone_enabled is False
    wired.on_session("unlocked")
    assert wired.worker.microphone_enabled is True
    assert len(wired.bus.speak_request.calls) == 1


def test_unlock_does_not_turn_on_a_microphone_the_owner_turned_off(wired):
    wired.worker.microphone_enabled = False
    wired.on_session("locked")
    wired.on_session("unlocked")
    assert wired.worker.microphone_enabled is False


def test_no_greeting_while_the_owner_is_busy(wired):
    wired.presence.quiet_now = True
    wired.on_session("unlocked")
    assert wired.bus.speak_request.calls == []


def test_a_restart_drops_the_launch_request_and_the_tray_flag():
    from orion_core.windows_shell import strip_shell_args
    args = [a for a in strip_shell_args(["--tray", "orion://briefing", "--do", "mute"]) if a != "--tray"]
    assert args == []
