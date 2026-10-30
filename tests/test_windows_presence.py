"""
Windows presence (Mark XXXVII.1) - the hotkey, the session watch, the busy
state and notifications with buttons, driven entirely by fakes.

Nothing here touches Windows. There is no real RegisterHotKey, no real message
window, no real SHQueryUserNotificationState and no real toast: ``user32``,
``kernel32``, ``shell32``, the message-window factory and the toast backend are
injected, and the Qt thread is replaced by a recording dispatcher. What is
held, and why:

  * a chord is parsed strictly - a bare letter as a GLOBAL hotkey would
    swallow that letter in every application;
  * the hotkey registers on its own thread with MOD_NOREPEAT, falls back to
    ctrl+alt+o when Windows says no, says which is live, and unregisters on
    the registering thread when stopped;
  * session and power messages map to exactly four events, and one wake is
    one event even though Windows reports it twice;
  * a toast button is only honoured for a token ORION issued, once, before it
    expired, for a verb the toast offered - because the URL it launches is one
    any program on the machine could also launch;
  * every piece of notification text is escaped.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication  # noqa: E402,F401  (GUI test: see conftest)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import windows_presence as wp  # noqa: E402
from orion_core import windows_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _own_settings(tmp_path, monkeypatch):
    """Every test reads and writes ITS settings file, never the owner's."""
    monkeypatch.setattr(windows_settings, "SETTINGS_PATH", tmp_path / "windows_integration.json")
    for key in windows_settings.DEFAULTS:
        monkeypatch.delenv(f"ORION_WIN_{key.upper()}", raising=False)


def sync(fn):
    """A dispatcher that runs at once, standing in for the Qt thread."""
    fn()


# ══════════════════════════════════════════════════════════════════════════════
# CHORDS
# ══════════════════════════════════════════════════════════════════════════════

def test_a_chord_is_parsed_into_flags_and_a_virtual_key():
    chord = wp.parse_chord("win+alt+o")
    assert chord.modifiers == wp.MOD_WIN | wp.MOD_ALT
    assert chord.vk == ord("O")
    assert chord.label == "Win+Alt+O" and chord.text == "win+alt+o"

    assert wp.parse_chord("Ctrl + Shift + F12").vk == 0x7B
    assert wp.parse_chord("control+5").vk == ord("5")
    assert wp.parse_chord("ctrl+space").vk == 0x20
    assert wp.parse_chord("alt+ctrl+z").label == "Ctrl+Alt+Z"      # one printed order
    assert wp.parse_chord("super+f24").vk == 0x70 + 23


@pytest.mark.parametrize("text", [
    "", "   ", "o", "f9",               # a global hotkey with no modifier steals the key everywhere
    "ctrl", "ctrl+alt",                 # modifiers only
    "ctrl+a+b",                         # two keys
    "ctrl+ctrl+a",                      # repeated modifier
    "ctrl+;", "ctrl+[",                 # punctuation moves with the keyboard layout
    "ctrl+f25", "ctrl+f0", "ctrl+nonsense",
])
def test_a_chord_that_cannot_be_a_global_hotkey_is_refused(text):
    with pytest.raises(wp.ChordError):
        wp.parse_chord(text)


# ══════════════════════════════════════════════════════════════════════════════
# THE GLOBAL HOTKEY
# ══════════════════════════════════════════════════════════════════════════════

class FakeUser32:
    """The slice of user32 the hotkey thread uses, with a real blocking queue
    behind GetMessageW so the thread behaves like a message loop."""

    def __init__(self, refuse=()):
        self.refuse = set(refuse)               # {(modifiers, vk)} that Windows rejects
        self.registered: dict[int, tuple[int, int]] = {}
        self.register_calls: list[tuple] = []
        self.unregister_calls: list[tuple[int, int]] = []
        self.unregister_thread: list[int] = []
        self.posted: list[tuple] = []
        self.loop_thread: int | None = None
        self._messages: queue.Queue = queue.Queue()

    def PeekMessageW(self, ptr, hwnd, low, high, flags):
        self.loop_thread = threading.get_ident()
        return 0

    def RegisterHotKey(self, hwnd, hotkey_id, modifiers, vk):
        self.register_calls.append((hwnd, hotkey_id, modifiers, vk))
        if (modifiers & ~wp.MOD_NOREPEAT, vk) in self.refuse:
            return 0
        self.registered[hotkey_id] = (modifiers, vk)
        return 1

    def UnregisterHotKey(self, hwnd, hotkey_id):
        self.unregister_calls.append((hwnd, hotkey_id))
        self.unregister_thread.append(threading.get_ident())
        self.registered.pop(hotkey_id, None)
        return 1

    def GetMessageW(self, ptr, hwnd, low, high):
        message, wparam = self._messages.get(timeout=5)
        if message == wp.WM_QUIT:
            return 0
        ptr.contents.message = message
        ptr.contents.wParam = wparam
        return 1

    def PostThreadMessageW(self, thread_id, message, wparam, lparam):
        self.posted.append((thread_id, message))
        self._messages.put((message, wparam))
        return 1

    def press(self, hotkey_id=0x4F52):
        self._messages.put((wp.WM_HOTKEY, hotkey_id))


class FakeKernel32:
    def GetCurrentThreadId(self):
        return 4242


def _hotkey(user32, callback=lambda: None, **kwargs):
    kwargs.setdefault("dispatch", sync)
    kwargs.setdefault("min_interval", 0.0)
    return wp.GlobalHotkey(callback, user32=user32, kernel32=FakeKernel32(), **kwargs)


def test_the_hotkey_registers_on_its_own_thread_with_norepeat_and_fires_the_callback():
    user32 = FakeUser32()
    fired = threading.Event()
    hotkey = _hotkey(user32, fired.set)
    assert hotkey.start() is True
    try:
        assert user32.register_calls == [
            (None, 0x4F52, wp.MOD_WIN | wp.MOD_ALT | wp.MOD_NOREPEAT, ord("O"))]
        assert user32.loop_thread != threading.get_ident()        # not the caller's thread
        assert hotkey.active_chord.label == "Win+Alt+O" and hotkey.used_fallback is False
        assert "Win+Alt+O" in hotkey.describe() and "any window" in hotkey.describe()
        assert hotkey.running

        user32.press()
        assert fired.wait(2.0)
    finally:
        hotkey.stop()


def test_with_the_default_dispatcher_the_callback_runs_on_the_gui_thread():
    """The real marshalling path: hotkey thread -> queued Qt signal -> GUI thread."""
    from PyQt6.QtTest import QTest

    globals().setdefault("_QT_APP", QApplication.instance() or QApplication([]))  # held, or it is collected
    user32 = FakeUser32()
    seen = []
    hotkey = wp.GlobalHotkey(lambda: seen.append(threading.get_ident()), user32=user32,
                             kernel32=FakeKernel32(), min_interval=0.0)       # default dispatch
    assert hotkey.start()
    try:
        user32.press()
        for _ in range(100):
            QTest.qWait(20)                       # the GUI thread's own event loop delivers it
            if seen:
                break
        assert seen == [threading.get_ident()]
        assert seen[0] != user32.loop_thread      # not the hotkey thread
    finally:
        hotkey.stop()


def test_a_message_for_a_different_hotkey_id_is_ignored():
    user32 = FakeUser32()
    fired = []
    hotkey = _hotkey(user32, lambda: fired.append(1))
    hotkey.start()
    try:
        user32.press(hotkey_id=999)
        user32.press()                       # the one that is ours, as a barrier
        for _ in range(100):
            if fired:
                break
            threading.Event().wait(0.02)
        assert fired == [1]
    finally:
        hotkey.stop()


def test_stop_posts_quit_and_unregisters_on_the_registering_thread():
    user32 = FakeUser32()
    hotkey = _hotkey(user32)
    hotkey.start()
    loop_thread = user32.loop_thread
    thread = hotkey._thread
    hotkey.stop()
    assert user32.posted == [(4242, wp.WM_QUIT)]
    assert user32.unregister_calls == [(None, 0x4F52)] and not user32.registered
    # RegisterHotKey belongs to the thread that made it, so that thread must undo it.
    assert user32.unregister_thread == [loop_thread]
    assert not thread.is_alive()
    assert hotkey.running is False and hotkey.active_chord is None
    hotkey.stop()                             # twice is harmless
    assert user32.unregister_calls == [(None, 0x4F52)]


def test_stop_before_start_is_harmless():
    wp.GlobalHotkey(lambda: None, user32=FakeUser32(), kernel32=FakeKernel32()).stop()


def test_when_windows_refuses_the_chord_the_fallback_is_tried_and_reported():
    user32 = FakeUser32(refuse={(wp.MOD_WIN | wp.MOD_ALT, ord("O"))})
    hotkey = _hotkey(user32)
    assert hotkey.start() is True
    try:
        assert [call[2] & ~wp.MOD_NOREPEAT for call in user32.register_calls] == [
            wp.MOD_WIN | wp.MOD_ALT, wp.MOD_CONTROL | wp.MOD_ALT]
        assert hotkey.used_fallback is True and hotkey.label == "Ctrl+Alt+O"
        text = hotkey.describe()
        assert "Ctrl+Alt+O" in text and "Win+Alt+O" in text and "taken" in text
    finally:
        hotkey.stop()


def test_when_both_chords_are_refused_there_is_no_hotkey_and_it_says_why(monkeypatch):
    monkeypatch.setattr(wp, "_last_error", lambda: wp.ERROR_HOTKEY_ALREADY_REGISTERED)
    user32 = FakeUser32(refuse={(wp.MOD_WIN | wp.MOD_ALT, ord("O")),
                                (wp.MOD_CONTROL | wp.MOD_ALT, ord("O"))})
    hotkey = _hotkey(user32)
    assert hotkey.start() is False
    assert hotkey.running is False and hotkey.active_chord is None
    assert "unavailable" in hotkey.describe() and "already taken" in hotkey.describe()
    assert not user32.registered
    hotkey.stop()


def test_an_unusable_chord_in_settings_falls_back_and_explains():
    user32 = FakeUser32()
    hotkey = _hotkey(user32, chord="o")                   # no modifier
    assert hotkey.start() is True
    try:
        assert hotkey.label == "Ctrl+Alt+O" and hotkey.used_fallback is True
        assert "not usable" in hotkey.describe()
    finally:
        hotkey.stop()
    assert _hotkey(FakeUser32(), chord="o", fallback=None).start() is False


def test_without_win32_there_is_no_hotkey(monkeypatch):
    monkeypatch.setattr(wp, "_load_lib", lambda name: None)
    hotkey = wp.GlobalHotkey(lambda: None)
    assert hotkey.start() is False
    assert "Windows feature" in hotkey.describe()


def test_a_held_chord_is_one_press_and_a_raising_callback_does_not_stop_the_watch():
    user32 = FakeUser32()
    calls = []

    def callback():
        calls.append(1)
        raise RuntimeError("consumer bug")

    hotkey = _hotkey(user32, callback, min_interval=60.0)
    hotkey.start()
    try:
        user32.press()
        user32.press()                         # inside the interval: ignored
        for _ in range(100):
            if calls:
                break
            threading.Event().wait(0.02)
        threading.Event().wait(0.1)
        assert calls == [1]
        assert hotkey.running                  # the watch survived the exception
    finally:
        hotkey.stop()


def test_the_callback_is_handed_to_the_dispatcher_not_run_on_the_hotkey_thread():
    user32 = FakeUser32()
    seen = []
    ran = threading.Event()

    def dispatch(fn):
        seen.append(fn)                        # queued, as the Qt thread would
        ran.set()

    called = []
    hotkey = _hotkey(user32, lambda: called.append(1), dispatch=dispatch)
    hotkey.start()
    try:
        user32.press()
        assert ran.wait(2.0)
        assert called == [] and len(seen) == 1     # nothing ran until the "Qt thread" did
        seen[0]()
        assert called == [1]
    finally:
        hotkey.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION AND POWER
# ══════════════════════════════════════════════════════════════════════════════

def test_session_and_power_messages_map_to_four_events():
    wm_session, wm_power = wp.WM_WTSSESSION_CHANGE, wp.WM_POWERBROADCAST
    assert wp.map_session_message(wm_session, wp.WTS_SESSION_LOCK) == "locked"
    assert wp.map_session_message(wm_session, wp.WTS_SESSION_UNLOCK) == "unlocked"
    assert wp.map_session_message(wm_power, wp.PBT_APMSUSPEND) == "suspending"
    assert wp.map_session_message(wm_power, wp.PBT_APMRESUMEAUTOMATIC) == "resumed"
    assert wp.map_session_message(wm_power, wp.PBT_APMRESUMESUSPEND) == "resumed"
    # Everything else Windows says about sessions and power is not ORION's business.
    assert wp.map_session_message(wm_session, 1) is None            # console connect
    assert wp.map_session_message(wm_power, 0x0A) is None           # power status change
    assert wp.map_session_message(0x0010, 0) is None                # WM_CLOSE
    assert set(wp.SESSION_EVENTS) == {"locked", "unlocked", "suspending", "resumed"}


def test_the_window_procedure_logic_emits_events_and_tracks_lock_state():
    events = []
    watcher = wp.SessionWatcher(events.append, dispatch=sync)
    assert watcher.locked is False
    assert watcher.handle_message(wp.WM_WTSSESSION_CHANGE, wp.WTS_SESSION_LOCK, 0) == 0
    assert watcher.locked is True
    assert watcher.handle_message(wp.WM_WTSSESSION_CHANGE, wp.WTS_SESSION_UNLOCK, 0) == 0
    assert watcher.locked is False
    assert watcher.handle_message(wp.WM_POWERBROADCAST, wp.PBT_APMSUSPEND, 0) == 1   # TRUE: accepted
    assert events == ["locked", "unlocked", "suspending"]
    assert watcher.last_event == "suspending"
    # Not a session message: left to Windows' default handling.
    assert watcher.handle_message(0x0001, 0, 0) is None
    assert watcher.handle_message(wp.WM_WTSSESSION_CHANGE, 1, 0) is None
    assert events == ["locked", "unlocked", "suspending"]


def test_one_wake_is_one_event_even_though_windows_reports_it_twice():
    now = [100.0]
    events = []
    watcher = wp.SessionWatcher(events.append, dispatch=sync, clock=lambda: now[0])
    watcher.handle_message(wp.WM_POWERBROADCAST, wp.PBT_APMRESUMEAUTOMATIC, 0)
    now[0] += 2.0
    watcher.handle_message(wp.WM_POWERBROADCAST, wp.PBT_APMRESUMESUSPEND, 0)
    assert events == ["resumed"]
    # A later wake - even with no suspend reported between (modern standby) - is a new one.
    now[0] += 3600.0
    watcher.handle_message(wp.WM_POWERBROADCAST, wp.PBT_APMRESUMEAUTOMATIC, 0)
    assert events == ["resumed", "resumed"]


def test_a_raising_session_callback_does_not_break_the_window_procedure():
    def boom(_event):
        raise RuntimeError("consumer bug")

    watcher = wp.SessionWatcher(boom, dispatch=sync)
    assert watcher.handle_message(wp.WM_WTSSESSION_CHANGE, wp.WTS_SESSION_LOCK, 0) == 0


class FakeWindow:
    """Stands in for the message-only window. Its run() blocks the way a
    message loop does, until post_quit()."""

    instances: list = []

    def __init__(self, handler, *, opens=True):
        self.handler = handler
        self.opens = opens
        self.calls: list[str] = []
        self.run_thread = None
        self._quit = threading.Event()
        FakeWindow.instances.append(self)

    def open(self):
        self.calls.append("open")
        return self.opens

    def run(self):
        self.calls.append("run")
        self.run_thread = threading.get_ident()
        self._quit.wait(5)

    def post_quit(self):
        self.calls.append("post_quit")
        self._quit.set()

    def close(self):
        self.calls.append("close")


def test_the_watcher_pumps_on_its_own_thread_and_closes_cleanly():
    FakeWindow.instances.clear()
    events = []
    watcher = wp.SessionWatcher(events.append, dispatch=sync, window_factory=FakeWindow)
    assert watcher.start() is True and watcher.running
    window = FakeWindow.instances[0]
    for _ in range(100):
        if window.run_thread:
            break
        threading.Event().wait(0.02)
    assert window.run_thread not in (None, threading.get_ident())
    window.handler(wp.WM_WTSSESSION_CHANGE, wp.WTS_SESSION_LOCK, 0, None)     # a message arrives
    assert events == ["locked"]
    assert "lock, unlock, sleep and wake" in watcher.describe()

    thread = watcher._thread
    watcher.stop()
    assert window.calls == ["open", "run", "post_quit", "close"]
    assert not thread.is_alive() and watcher.running is False
    watcher.stop()                                            # idempotent


def test_a_window_that_cannot_be_created_is_reported_not_raised():
    FakeWindow.instances.clear()
    watcher = wp.SessionWatcher(lambda e: None, dispatch=sync,
                                window_factory=lambda handler: FakeWindow(handler, opens=False))
    assert watcher.start() is False
    assert "could not be created" in watcher.describe()
    watcher.stop()

    def explode(handler):
        raise OSError("no user32")

    broken = wp.SessionWatcher(lambda e: None, dispatch=sync, window_factory=explode)
    assert broken.start() is False and "OSError" in broken.describe()


# ══════════════════════════════════════════════════════════════════════════════
# IS THE PERSON BUSY?
# ══════════════════════════════════════════════════════════════════════════════

class FakeShell32:
    def __init__(self, code=5, hresult=0, error=None):
        self.code, self.hresult, self.error = code, hresult, error

    def SHQueryUserNotificationState(self, ptr):
        if self.error:
            raise self.error
        ptr.contents.value = self.code
        return self.hresult


@pytest.mark.parametrize("code,expected", [
    (1, "busy"),                  # QUNS_NOT_PRESENT
    (2, "fullscreen_app"),        # QUNS_BUSY
    (3, "fullscreen_app"),        # QUNS_RUNNING_D3D_FULL_SCREEN
    (4, "presentation"),          # QUNS_PRESENTATION_MODE
    (5, "available"),             # QUNS_ACCEPTS_NOTIFICATIONS
    (6, "quiet_hours"),           # QUNS_QUIET_TIME
    (7, "fullscreen_app"),        # QUNS_APP
    (0, "unknown"), (8, "unknown"), (-1, "unknown"), (None, "unknown"), ("x", "unknown"),
])
def test_the_notification_state_maps_to_a_busy_state(code, expected):
    assert wp.busy_state_from_quns(code) == expected
    assert expected in wp.BUSY_STATES


def test_user_busy_state_asks_shell32_and_never_raises():
    assert wp.user_busy_state(FakeShell32(code=2)) == "fullscreen_app"
    assert wp.user_busy_state(FakeShell32(code=6)) == "quiet_hours"
    assert wp.user_busy_state(FakeShell32(code=5, hresult=0x80004005)) == "unknown"   # failed HRESULT
    assert wp.user_busy_state(FakeShell32(error=OSError("shell gone"))) == "unknown"


def test_user_busy_state_without_win32_is_unknown(monkeypatch):
    monkeypatch.setattr(wp, "_load_lib", lambda name: None)
    assert wp.user_busy_state() == "unknown"


def test_staying_quiet_needs_both_the_setting_and_a_busy_windows():
    for state in ("busy", "fullscreen_app", "presentation", "quiet_hours"):
        assert wp.should_stay_quiet(state) is True, state
    for state in ("available", "unknown"):
        assert wp.should_stay_quiet(state) is False, state
    assert wp.should_stay_quiet(shell32=FakeShell32(code=4)) is True          # asks Windows itself

    assert windows_settings.set_value("quiet_when_busy", False)
    assert wp.should_stay_quiet("presentation") is False                     # the owner opted out


# ══════════════════════════════════════════════════════════════════════════════
# NOTIFICATIONS WITH BUTTONS
# ══════════════════════════════════════════════════════════════════════════════

class FakeBackend:
    def __init__(self, available=True, reason="", error=None):
        self._available = (available, reason)
        self.error = error
        self.shown: list[dict] = []

    def available(self):
        return self._available

    def show(self, xml, *, tag, expires_s):
        if self.error:
            raise self.error
        self.shown.append({"xml": xml, "tag": tag, "expires_s": expires_s})


class FakeTray:
    def __init__(self):
        self.balloons: list[tuple[str, str]] = []

    def notify(self, title, message):
        self.balloons.append((title, message))
        return True


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _center(backend=None, tray=None, shortcut=True, clock=None, said=None, dispatch=sync):
    return wp.ToastCenter(
        dispatch=dispatch, backend=backend if backend is not None else FakeBackend(),
        shortcut_check=lambda: shortcut, tray=tray, run_async=lambda fn: fn(),
        say=(said.append if said is not None else None), clock=clock or Clock())


def _parse(xml):
    return ET.fromstring(xml)


def _token_of(receipt):
    assert receipt.token and len(receipt.token) >= 30
    return receipt.token


def test_toast_text_is_escaped_so_markup_in_it_stays_text():
    hostile_title = '<b>"ORION" & \'friends\'</b>'
    hostile_body = 'x</text><action content="evil" arguments="orion://toast?t=1&a=wipe"/>'
    hostile_label = '"/><action content="evil'
    xml = wp.build_toast_xml(hostile_title, hostile_body, [(hostile_label, "open")], "TOKEN123", "open")
    root = _parse(xml)                                      # well-formed, or this raises
    texts = [node.text for node in root.iter("text")]
    assert texts == [hostile_title, hostile_body]           # exactly the strings, as text
    actions = list(root.iter("action"))
    assert len(actions) == 1                                # nothing was injected
    assert actions[0].get("content") == hostile_label
    assert actions[0].get("arguments") == "orion://toast?t=TOKEN123&a=open"
    assert actions[0].get("activationType") == "protocol"
    assert root.get("launch") == "orion://toast?t=TOKEN123&a=open"
    assert root.get("activationType") == "protocol"
    assert root.find("visual/binding").get("template") == "ToastGeneric"
    assert "<b>" not in xml and "&amp;a=open" in xml        # raw markup never appears; & is escaped


def test_toast_text_loses_characters_xml_cannot_carry_and_is_bounded():
    xml = wp.build_toast_xml("Hi\x00\x08there\x1f", "line1\nline2\ttab", [], None)
    root = _parse(xml)
    assert [n.text for n in root.iter("text")] == ["Hithere", "line1\nline2\ttab"]
    assert root.find("actions") is None                     # no token, no buttons
    long = _parse(wp.build_toast_xml("T" * 500, "B" * 5000, [], None))
    titles = [n.text for n in long.iter("text")]
    assert len(titles[0]) <= 120 and len(titles[1]) <= 400 and titles[0].endswith("…")


def test_a_toast_with_buttons_issues_a_single_use_url_safe_token():
    backend, calls = FakeBackend(), []
    center = _center(backend)
    receipt = center.show("Update ready", "Restart?", [("Restart Now", "restart"), ("Later", "later")],
                          on_action=calls.append, expires_s=120)
    assert receipt.via == "toast" and receipt.delivered and receipt.buttons
    token = _token_of(receipt)
    assert all(c.isalnum() or c in "-_" for c in token)       # token_urlsafe(24)
    assert len(token) == 32
    assert backend.shown[0]["expires_s"] == 120

    root = _parse(backend.shown[0]["xml"])
    buttons = {a.get("content"): a.get("arguments") for a in root.iter("action")}
    assert buttons == {"Restart Now": f"orion://toast?t={token}&a=restart",
                       "Later": f"orion://toast?t={token}&a=later"}

    assert center.handle_action(token, "restart") is True
    assert calls == ["restart"]
    # Used: the same click, a different button on the same toast, a replay.
    assert center.handle_action(token, "restart") is False
    assert center.handle_action(token, "later") is False
    assert calls == ["restart"] and center.pending_count() == 0


def test_each_toast_gets_its_own_token():
    center = _center()
    first = center.show("A", "", [("Go", "go")], on_action=lambda v: None)
    second = center.show("B", "", [("Go", "go")], on_action=lambda v: None)
    assert first.token != second.token and center.pending_count() == 2


def test_an_unknown_token_or_a_verb_that_was_not_offered_does_nothing():
    calls = []
    center = _center()
    token = _token_of(center.show("Hi", "", [("Open", "open")], on_action=calls.append))
    assert center.handle_action("not-a-token", "open") is False
    assert center.handle_action(token, "wipe_disk") is False              # not offered
    assert center.handle_action(token, "open; rm") is False               # not even a valid verb
    assert center.handle_action(token, "") is False
    assert center.handle_action(None, "open") is False
    assert center.handle_action(token, None) is False
    assert center.handle_action("x" * 500, "open") is False
    assert calls == []
    # A wrong guess does not spend the real token.
    assert center.handle_action(token, "open") is True and calls == ["open"]


def test_a_token_expires():
    clock, calls = Clock(), []
    center = _center(clock=clock)
    token = _token_of(center.show("Hi", "", [("Open", "open")], on_action=calls.append, expires_s=60))
    clock.now += 59
    assert center.pending_count() == 1
    clock.now += 2                                                     # past the 60 s
    assert center.handle_action(token, "open") is False
    assert calls == [] and center.pending_count() == 0


def test_the_toast_body_can_carry_a_default_verb_which_is_then_an_offered_verb():
    calls, backend = [], FakeBackend()
    center = _center(backend)
    receipt = center.show("Hi", "", [("Later", "later")], on_action=calls.append, default_verb="open")
    root = _parse(backend.shown[0]["xml"])
    assert root.get("launch") == f"orion://toast?t={receipt.token}&a=open"
    assert center.handle_action(receipt.token, "open") is True and calls == ["open"]


def test_the_action_runs_on_the_qt_thread_through_the_dispatcher():
    queued, calls = [], []
    center = _center(dispatch=queued.append)
    token = _token_of(center.show("Hi", "", [("Open", "open")], on_action=calls.append))
    assert center.handle_action(token, "open") is True
    assert calls == [] and len(queued) == 1                            # handed over, not yet run
    queued[0]()
    assert calls == ["open"]


def test_a_raising_action_callback_is_contained():
    def boom(verb):
        raise RuntimeError("consumer bug")

    center = _center()
    token = _token_of(center.show("Hi", "", [("Open", "open")], on_action=boom))
    assert center.handle_action(token, "open") is True               # scheduled; the failure is logged


def test_invalid_actions_are_dropped_and_at_most_five_are_shown():
    backend = FakeBackend()
    center = _center(backend)
    actions = [("Good", "good"), ("", "nolabel"), ("Bad verb", "has space"), ("Dup", "good"),
               ("Inject", 'x"/><y'), ("Five", "five"), ("Six", "six"), ("Seven", "seven"),
               ("Eight", "eight"), ("Nine", "nine"), "garbage", None]
    center.show("Hi", "", actions, on_action=lambda v: None)
    root = _parse(backend.shown[0]["xml"])
    labels = [a.get("content") for a in root.iter("action")]
    assert labels == ["Good", "Five", "Six", "Seven", "Eight"]


def test_buttons_without_a_callback_are_not_shown():
    backend = FakeBackend()
    center = _center(backend)
    receipt = center.show("Hi", "plain", [("Open", "open")])
    assert receipt.token is None and receipt.buttons is False and center.pending_count() == 0
    assert _parse(backend.shown[0]["xml"]).find("actions") is None


def test_pending_tokens_are_bounded():
    center = _center()
    first = center.show("0", "", [("Go", "go")], on_action=lambda v: None).token
    for number in range(wp.MAX_PENDING + 5):
        center.show(str(number), "", [("Go", "go")], on_action=lambda v: None)
    assert center.pending_count() <= wp.MAX_PENDING
    assert center.handle_action(first, "go") is False                  # the oldest was forgotten


def test_without_winrt_it_falls_back_to_a_tray_balloon_and_says_so():
    tray, said = FakeTray(), []
    center = _center(FakeBackend(available=False, reason="the WinRT notification packages are missing"),
                     tray=tray, said=said)
    calls = []
    receipt = center.show("Update", "Restart?", [("Restart", "restart")], on_action=calls.append)
    assert receipt.via == "balloon" and receipt.buttons is False and receipt.delivered
    assert receipt.token is None and center.pending_count() == 0       # nothing to click, so no token
    assert tray.balloons == [("Update", "Restart?")]
    assert "WinRT" in receipt.reason and "without buttons" in receipt.summary()
    assert said and "tray balloon without buttons" in said[0]
    assert "tray balloons" in center.describe()


def test_without_the_start_menu_shortcut_it_falls_back_and_says_why():
    tray = FakeTray()
    center = _center(tray=tray, shortcut=False)
    receipt = center.show("Hi", "there", [("Open", "open")], on_action=lambda v: None)
    assert receipt.via == "balloon" and "Start-menu shortcut" in receipt.reason
    assert tray.balloons == [("Hi", "there")]


def test_a_windows_refusal_falls_back_after_the_token_is_revoked():
    tray, calls = FakeTray(), []
    backend = FakeBackend(error=OSError("RPC server unavailable"))
    center = _center(backend, tray=tray)
    receipt = center.show("Hi", "there", [("Open", "open")], on_action=calls.append)
    assert receipt.via == "balloon" and "OSError" in receipt.reason
    assert center.pending_count() == 0                                 # a button nobody can see is not live
    assert tray.balloons == [("Hi", "there")]


def test_an_identity_windows_does_not_know_falls_back_with_the_reason():
    tray = FakeTray()
    center = _center(FakeBackend(error=wp.ToastUnavailable(
        "Windows does not know ORION's notification identity yet")), tray=tray)
    calls = []
    receipt = center.show("Hi", "there", [("Open", "open")], on_action=calls.append)
    assert receipt.via == "balloon" and "identity" in receipt.reason
    assert center.pending_count() == 0 and tray.balloons == [("Hi", "there")]


def test_the_real_backend_turns_the_platforms_not_found_into_unavailable():
    """HRESULT 0x80070490 from the notifier means the AppUserModelID has no
    Start-menu shortcut carrying it: Windows would drop the toast silently."""
    backend = wp.WinrtToastBackend()

    class Unregistered:
        @property
        def setting(self):
            raise OSError(0, "Element not found", None, -2147023728)

    class Odd:
        @property
        def setting(self):
            raise OSError(0, "something else", None, -2147024809)

    class Fine:
        setting = 0

    with pytest.raises(wp.ToastUnavailable, match="notification identity"):
        backend._read_setting(Unregistered())
    assert backend._read_setting(Odd()) is None            # unreadable is not a refusal
    assert backend._read_setting(Fine()) == 0


def test_the_toast_identity_is_left_alone_when_it_already_matches():
    installed = []
    assert wp.ensure_toast_identity(read_aumid=lambda: wp.APP_USER_MODEL_ID,
                                    install=lambda: installed.append(1)) == "ok"
    assert installed == []                                 # nothing touched in the Start menu


def test_a_shortcut_with_another_identity_is_refreshed_once_asked():
    installed = []

    def install():
        installed.append(1)
        return {"start_menu": True}

    # What this checkout actually had: a shortcut stamped by an older build.
    assert wp.ensure_toast_identity(read_aumid=lambda: "Owner.ORION.Assistant",
                                    install=install) == "refreshed"
    assert wp.ensure_toast_identity(read_aumid=lambda: None, install=install) == "refreshed"
    assert installed == [1, 1]


def test_a_shortcut_that_cannot_be_written_is_reported_not_raised():
    assert wp.ensure_toast_identity(read_aumid=lambda: None,
                                    install=lambda: {"start_menu": False}).startswith("failed")

    def boom():
        raise OSError("read-only start menu")

    assert wp.ensure_toast_identity(read_aumid=lambda: None, install=boom) == "failed: OSError"


def test_notifications_switched_off_in_windows_are_respected():
    tray = FakeTray()
    center = _center(FakeBackend(error=wp.ToastDisabled("notifications for ORION are switched off")),
                     tray=tray)
    receipt = center.show("Hi", "there", [("Open", "open")], on_action=lambda v: None)
    assert receipt.via == "none" and tray.balloons == []               # no sneaking round the owner's choice
    assert center.pending_count() == 0


def test_with_toasts_off_in_preferences_only_plain_balloons_are_used():
    assert windows_settings.set_value("toasts", False)
    backend, tray = FakeBackend(), FakeTray()
    center = _center(backend, tray=tray)
    receipt = center.show("Hi", "there", [("Open", "open")], on_action=lambda v: None)
    assert backend.shown == [] and receipt.via == "balloon"
    assert "switched off in Preferences" in receipt.reason
    assert receipt.buttons is False and center.pending_count() == 0


def test_with_no_toast_and_no_tray_nothing_is_shown_and_nothing_raises():
    center = _center(FakeBackend(available=False, reason="no winrt"), tray=None)
    receipt = center.show("Hi", "there")
    assert receipt.via == "none" and receipt.delivered is False
    assert "no tray icon" in receipt.reason


def test_a_failing_backend_check_cannot_raise_out_of_show():
    class Broken(FakeBackend):
        def available(self):
            raise RuntimeError("probe blew up")

    receipt = _center(Broken()).show("Hi", "there")
    assert receipt.via == "none" and receipt.delivered is False


def test_the_real_backend_reports_when_winrt_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "winrt.windows.ui.notifications", None)   # import fails
    ok, reason = wp.WinrtToastBackend().available()
    assert ok is False and "missing" in reason


def test_the_real_backend_is_available_here_without_showing_anything():
    """Import-only: confirms the packages the build bundles are importable. No
    notifier is created and nothing is shown."""
    ok, reason = wp.WinrtToastBackend().available()
    if not ok and "missing" in reason:
        pytest.skip(reason)
    assert ok


# ══════════════════════════════════════════════════════════════════════════════
# THE HANDLE
# ══════════════════════════════════════════════════════════════════════════════

class FakeBus:
    def __init__(self):
        self.lines: list[str] = []
        self.log = SimpleNamespace(emit=self.lines.append)


class FakePart:
    """A hotkey or watcher stand-in that records what the handle did to it."""

    def __init__(self, callback, **kwargs):
        self.callback, self.kwargs = callback, kwargs
        self.started = self.stopped = 0

    def start(self):
        self.started += 1
        return True

    def stop(self):
        self.stopped += 1

    def describe(self):
        return f"fake part {self.kwargs.get('chord', '')}"


def _start(bus=None, **overrides):
    bus = bus or FakeBus()
    pressed, events, made = [], [], {}

    def hotkey_factory(callback, **kwargs):
        made["hotkey"] = FakePart(callback, **kwargs)
        return made["hotkey"]

    def watcher_factory(callback, **kwargs):
        made["watcher"] = FakePart(callback, **kwargs)
        return made["watcher"]

    handle = wp.start_windows_presence(
        bus, on_hotkey=lambda: pressed.append(1), on_session_event=events.append,
        hotkey_factory=hotkey_factory, watcher_factory=watcher_factory,
        toast_backend=FakeBackend(), dispatch=sync, **overrides)
    return handle, bus, pressed, events, made


def test_the_handle_starts_the_hotkey_and_watch_from_the_stored_settings():
    assert windows_settings.set_value("hotkey", "ctrl+shift+j")
    handle, bus, pressed, events, made = _start(palette=("#ff0000", "#ffffff"))
    try:
        assert made["hotkey"].kwargs["chord"] == "ctrl+shift+j"
        assert made["hotkey"].started == 1 and made["watcher"].started == 1
        assert handle.hotkey is made["hotkey"] and handle.watcher is made["watcher"]
        assert isinstance(handle.toasts, wp.ToastCenter)
        assert any("HOTKEY" in line for line in bus.lines)
        assert any("SESSION" in line for line in bus.lines)
        assert wp.active_hotkey_text() == "fake part ctrl+shift+j"
    finally:
        handle.stop()
    assert wp.active_hotkey_text() == ""


def test_the_handle_forwards_hotkey_presses_and_session_events():
    handle, bus, pressed, events, made = _start()
    try:
        made["hotkey"].callback()
        assert pressed == [1]
        made["watcher"].callback("locked")
        assert handle.locked is True and events == ["locked"]
        made["watcher"].callback("suspending")
        made["watcher"].callback("resumed")
        assert handle.locked is True                       # waking up does not unlock a locked session
        made["watcher"].callback("unlocked")
        assert handle.locked is False
        assert events == ["locked", "suspending", "resumed", "unlocked"]
    finally:
        handle.stop()


def test_a_second_hotkey_mutes_the_microphone_and_stops_with_the_rest():
    bus = FakeBus()
    made, toggled = [], []

    def hotkey_factory(callback, **kwargs):
        made.append(FakePart(callback, **kwargs))
        return made[-1]

    handle = wp.start_windows_presence(
        bus, on_hotkey=lambda: None, on_session_event=lambda _e: None,
        on_mic_hotkey=lambda: toggled.append(1), hotkey_factory=hotkey_factory,
        watcher_factory=lambda cb, **kw: FakePart(cb, **kw),
        toast_backend=FakeBackend(), dispatch=sync)
    try:
        mic = handle.mic_hotkey
        assert mic is made[1] and mic.kwargs["chord"] == wp.MIC_CHORD == "win+alt+m"
        assert mic.kwargs["hotkey_id"] != made[0].kwargs.get("hotkey_id", 0x4F52)
        mic.callback()
        assert toggled == [1]
    finally:
        handle.stop()
    assert mic.stopped == 1


def test_a_consumer_that_raises_cannot_break_the_forwarding():
    bus = FakeBus()
    made = {}

    def hotkey_factory(callback, **kwargs):
        made["hotkey"] = FakePart(callback, **kwargs)
        return made["hotkey"]

    def boom(*_args):
        raise RuntimeError("lead's bug")

    handle = wp.start_windows_presence(bus, on_hotkey=boom, on_session_event=boom,
                                       hotkey_factory=hotkey_factory,
                                       watcher_factory=lambda cb, **kw: FakePart(cb, **kw),
                                       toast_backend=FakeBackend(), dispatch=sync)
    try:
        made["hotkey"].callback()                          # does not raise
    finally:
        handle.stop()


def test_a_switched_off_hotkey_is_not_registered():
    assert windows_settings.set_value("hotkey_enabled", False)
    handle, bus, pressed, events, made = _start()
    try:
        assert "hotkey" not in made and handle.hotkey is None
        assert "switched off" in handle.describe()[0]
        assert "watcher" in made                           # lock and sleep awareness is independent
    finally:
        handle.stop()


def test_the_handle_survives_parts_that_cannot_start():
    def broken(*_args, **_kwargs):
        raise OSError("no user32")

    bus = FakeBus()
    handle = wp.start_windows_presence(bus, on_hotkey=lambda: None, on_session_event=lambda e: None,
                                       hotkey_factory=broken, watcher_factory=broken,
                                       toast_backend=FakeBackend(), dispatch=sync)
    assert handle.hotkey is None and handle.watcher is None
    assert any("could not start" in line for line in bus.lines)
    handle.stop()


def test_stop_stops_everything_once():
    handle, bus, pressed, events, made = _start()
    handle.stop()
    handle.stop()
    assert made["hotkey"].stopped == 1 and made["watcher"].stopped == 1


def test_the_handle_answers_the_leads_questions_from_settings_and_windows(monkeypatch):
    handle, *_ = _start()
    try:
        monkeypatch.setattr(wp, "user_busy_state", lambda *a, **k: "presentation")
        assert handle.busy_state() == "presentation" and handle.quiet() is True
        assert handle.greet_on_unlock is True and handle.pause_hearing_when_locked is True
        windows_settings.set_value("greet_on_unlock", False)
        windows_settings.set_value("quiet_when_busy", False)
        assert handle.greet_on_unlock is False and handle.quiet() is False
        assert len(handle.describe()) == 3
    finally:
        handle.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE-LEVEL GUARDS
# ══════════════════════════════════════════════════════════════════════════════

def test_the_module_never_blocks_the_qt_thread_with_a_wait_loop():
    """GetMessageW and the WinRT call each live on a daemon thread of their own."""
    source = (ROOT / "orion_core" / "windows_presence.py").read_text(encoding="utf-8")
    assert source.count("daemon=True") >= 3                  # hotkey, session, toast
    assert "name=\"orion-hotkey\"" in source and "name=\"orion-session\"" in source
    assert "time.sleep" not in source                        # nothing sleeps on any thread


def test_the_build_bundles_the_winrt_modules_the_toasts_import():
    import build_standalone

    for module in ("winrt.windows.ui.notifications", "winrt.windows.data.xml.dom"):
        assert module in build_standalone.HIDDEN_IMPORTS
    assert "winrt" in build_standalone.COLLECT_ALL


# ══════════════════════════════════════════════════════════════════════════════
# PREFERENCES: THE WINDOWS INTEGRATION GROUP
# ══════════════════════════════════════════════════════════════════════════════

_EXPECTED_LABELS = {
    "hotkey_enabled": "Global Hotkey",
    "dock_enabled": "Docked Screen Face",
    "dock_on_startup": "Show Dock at Startup",
    "explorer_menu": 'Explorer "Ask ORION" Menu',
    "protocol": "ORION Links",
    "jump_list": "Taskbar Jump List",
    "toasts": "Notifications with Buttons",
    "greet_on_unlock": "Greet Me When I Unlock",
    "quiet_when_busy": "Stay Quiet When I'm Busy",
    "pause_hearing_when_locked": "Pause Listening When Locked",
    "start_in_tray_at_login": "Start in Tray at Sign-In",
}


def _group():
    from orion_core.gui.preferences import _WindowsIntegration

    globals().setdefault("_QT_APP", QApplication.instance() or QApplication([]))  # held, or it is collected
    return _WindowsIntegration()


def test_the_windows_integration_group_has_a_toggle_for_every_switch():
    from orion_core.gui.labels import title_case
    from orion_core.gui.preferences import WINDOWS_TOGGLES

    group = _group()
    try:
        assert group.title() == "Windows Integration"
        assert {key for key, _l, _h in WINDOWS_TOGGLES} == set(_EXPECTED_LABELS)
        # Every on/off switch windows_settings has is reachable here - none forgotten.
        bool_keys = {k for k, v in windows_settings.DEFAULTS.items() if isinstance(v, bool)}
        assert set(group.checks) == bool_keys
        for key, box in group.checks.items():
            assert box.text() == _EXPECTED_LABELS[key]
            assert title_case(box.text()) == box.text()           # names are Title Case
            assert box.accessibleName() == box.text()
            assert box.toolTip()
            assert box.isChecked() is windows_settings.DEFAULTS[key]
        assert "after a restart" in group.hint.text()
        assert "Win+Alt+O" in group.hotkey_status.text()           # the chord, shown
    finally:
        group.deleteLater()


def test_toggling_a_box_saves_it_to_the_settings_file_and_emits_a_line(tmp_path):
    group = _group()
    seen = []
    group.changed.connect(lambda key, value: seen.append((key, value)))
    try:
        group.checks["start_in_tray_at_login"].setChecked(False)
        group.checks["hotkey_enabled"].setChecked(False)
        group.checks["toasts"].setChecked(False)
        group.checks["toasts"].setChecked(True)
        stored = json.loads((tmp_path / "windows_integration.json").read_text(encoding="utf-8"))
        assert stored["start_in_tray_at_login"] is False and stored["hotkey_enabled"] is False
        assert stored["toasts"] is True
        assert windows_settings.get("start_in_tray_at_login") is False
        assert seen == [("start_in_tray_at_login", False), ("hotkey_enabled", False),
                        ("toasts", False), ("toasts", True)]
    finally:
        group.deleteLater()


def test_the_group_reflects_what_is_already_stored():
    windows_settings.set_value("greet_on_unlock", False)
    windows_settings.set_value("start_in_tray_at_login", False)
    group = _group()
    try:
        assert group.checks["greet_on_unlock"].isChecked() is False
        assert group.checks["start_in_tray_at_login"].isChecked() is False
        assert group.checks["dock_enabled"].isChecked() is True
    finally:
        group.deleteLater()


def test_a_switch_that_could_not_be_saved_is_put_back(monkeypatch):
    group = _group()
    try:
        monkeypatch.setattr(windows_settings, "set_value", lambda key, value: False)
        group.checks["dock_enabled"].setChecked(False)
        assert group.checks["dock_enabled"].isChecked() is True    # never shows an unsaved choice
        assert "could not be saved" in group.status.text()
    finally:
        group.deleteLater()


_DIALOG_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.gui.preferences import PreferencesDialog
dialog = PreferencesDialog(assistant_name="Orion", user_name="")
group = dialog.windows_integration
print("@@" + json.dumps({{"title": group.title(), "keys": sorted(group.checks),
                          "has_signal": hasattr(dialog, "windows_setting_changed")}}), flush=True)
os._exit(0)
'''


def test_the_preferences_dialog_carries_the_group():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="orion-prefs-win-") as config_dir:
        env = dict(os.environ, ORION_CONFIG_DIR=config_dir)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        proc = subprocess.run([sys.executable, "-c", _DIALOG_PROBE.format(root=str(ROOT).replace("\\", "/"))],
                              cwd=ROOT, capture_output=True, text=True, timeout=240, env=env)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("@@")]
    assert lines, f"probe failed:\n{proc.stdout[-1000:]}\n{proc.stderr[-2000:]}"
    result = json.loads(lines[-1][2:])
    assert result["title"] == "Windows Integration"
    assert set(result["keys"]) == set(_EXPECTED_LABELS) and result["has_signal"]


# ── QUNS_BUSY is believed only for a window that really fills its monitor ────

class _FakeUser32:
    """GetForegroundWindow & co. for foreground_covers_monitor()."""

    def __init__(self, rect, *, cls="GameWindow", zoomed=False, pid=None,
                 monitor=(0, 0, 1920, 1080)):
        self.rect, self.cls, self.zoomed, self.monitor = rect, cls, zoomed, monitor
        import os as _os
        self.pid = pid if pid is not None else _os.getpid() + 1

    def GetForegroundWindow(self):
        return 1234

    def GetClassNameW(self, hwnd, buffer, size):
        buffer.value = self.cls
        return len(self.cls)

    def IsZoomed(self, hwnd):
        return 1 if self.zoomed else 0

    def GetWindowThreadProcessId(self, hwnd, pid_ref):
        pid_ref._obj.value = self.pid
        return 1

    def GetWindowRect(self, hwnd, rect_ref):
        r = rect_ref._obj
        r.left, r.top, r.right, r.bottom = self.rect
        return 1

    def MonitorFromWindow(self, hwnd, flags):
        return 99

    def GetMonitorInfoW(self, monitor, info_ref):
        info = info_ref._obj
        info.rcMonitor.left, info.rcMonitor.top, info.rcMonitor.right, info.rcMonitor.bottom = self.monitor
        return 1


def test_a_borderless_window_filling_the_monitor_is_full_screen():
    from orion_core import windows_presence as wp
    assert wp.foreground_covers_monitor(_FakeUser32((0, 0, 1920, 1080))) is True


def test_a_maximised_window_is_not_full_screen_even_without_a_taskbar():
    # 24 Oct: Windows said QUNS_BUSY with only ordinary maximised windows open.
    from orion_core import windows_presence as wp
    assert wp.foreground_covers_monitor(_FakeUser32((-8, -8, 1928, 1088), zoomed=True)) is False


def test_the_desktop_and_orions_own_windows_never_count():
    import os
    from orion_core import windows_presence as wp
    assert wp.foreground_covers_monitor(_FakeUser32((0, 0, 1920, 1080), cls="WorkerW")) is False
    assert wp.foreground_covers_monitor(_FakeUser32((0, 0, 1920, 1080), pid=os.getpid())) is False


def test_a_window_smaller_than_its_monitor_is_not_full_screen():
    from orion_core import windows_presence as wp
    assert wp.foreground_covers_monitor(_FakeUser32((0, 0, 1920, 1040))) is False



def test_the_microphone_hotkey_tries_every_fallback_and_names_each_refusal():
    """26-29 Oct 2026: Win+Alt+M and Ctrl+Alt+M were both taken on the owner's
    PC, the log named only the second, and the microphone had no hotkey."""
    user32 = FakeUser32(refuse={(wp.MOD_WIN | wp.MOD_ALT, ord("M")),
                                (wp.MOD_CONTROL | wp.MOD_ALT, ord("M"))})
    hotkey = _hotkey(user32, chord=wp.MIC_CHORD, fallback=wp.MIC_FALLBACK_CHORD, hotkey_id=0x4D55)
    assert hotkey.start() is True
    try:
        assert hotkey.active_chord.label == "Ctrl+Alt+Shift+M" and hotkey.used_fallback
    finally:
        hotkey.stop()
    everything = FakeUser32(refuse={(wp.MOD_WIN | wp.MOD_ALT, ord("M")),
                                    (wp.MOD_CONTROL | wp.MOD_ALT, ord("M")),
                                    (wp.MOD_CONTROL | wp.MOD_ALT | wp.MOD_SHIFT, ord("M"))})
    none = _hotkey(everything, chord=wp.MIC_CHORD, fallback=wp.MIC_FALLBACK_CHORD, hotkey_id=0x4D55)
    assert none.start() is False
    assert none.error.count(":") == 3 and "Win+Alt+M" in none.error and "Ctrl+Alt+M" in none.error
