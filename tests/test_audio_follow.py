"""ORION follows the audio device the user is actually using.

He resolved his microphone and speaker once, at startup, and held them for the
life of the process. Plug in headphones an hour later and Windows moves its
default while ORION carries on talking to speakers nobody is listening to, and
listening to a microphone nobody is speaking into. Nothing fails — he simply
stops being able to hear you, and you stop being able to hear him.

The second rule is the one worth testing: a device the USER pinned outranks a
default, because that was a decision. Unless it has been unplugged, at which
point it is not a decision any more, it is a dead end.

Offline: no audio hardware is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import audio_follow  # noqa: E402
from orion_core.audio_follow import AudioFollower, DefaultDevices  # noqa: E402

#: The autouse fixture below replaces this per test; one test needs the real one.
_REAL_ENDPOINT_IS_ACTIVE = audio_follow.endpoint_is_active


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, line: str) -> None:
        self.lines.append(line)


class _Signal:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    def emit(self, *args) -> None:
        self.sent.append(args)


class _Bus:
    def __init__(self) -> None:
        self.log = _Log()
        self.audio_device_request = _Signal()


@pytest.fixture(autouse=True)
def _hermetic_audio(monkeypatch):
    """Nothing here may read Core Audio, re-initialise PortAudio or leave a
    held default behind: a follower now asks for a PortAudio refresh before it
    moves, and that must never reach the real thing from a test."""
    monkeypatch.setattr(audio_follow.audio_devices, "request_reinit",
                        lambda *_a, **_k: False)
    monkeypatch.setattr(audio_follow.audio_devices, "_system_default_name",
                        lambda _kind: None)
    monkeypatch.setattr(audio_follow.audio_devices, "_held_default", {})
    monkeypatch.setattr(audio_follow, "endpoint_is_active", lambda _id: None)
    monkeypatch.setattr(audio_follow, "desktop_output_state", lambda: None)


def _follower(monkeypatch, before, after, pin="", pin_live=True):
    bus = _Bus()
    follower = AudioFollower(bus=bus)
    follower._last = before
    monkeypatch.setattr(audio_follow, "read_defaults", lambda refresh=True: after)
    monkeypatch.setattr(audio_follow, "pinned", lambda _k: pin)
    monkeypatch.setattr(audio_follow, "pin_is_live", lambda _k: pin_live)
    return follower, bus


def test_nothing_changing_moves_nothing(monkeypatch):
    same = DefaultDevices("Mic A", "Speakers A")
    follower, bus = _follower(monkeypatch, same, same)
    assert follower.check_once() == []
    assert bus.audio_device_request.sent == []


def test_headphones_going_in_moves_the_output(monkeypatch):
    follower, bus = _follower(
        monkeypatch,
        DefaultDevices("Mic A", "Speakers"),
        DefaultDevices("Mic A", "Headphones"))
    # The log names the device by the friendly name Core Audio gave, not by the
    # endpoint ID that DefaultDevices carries.
    monkeypatch.setattr(audio_follow, "desktop_output_state",
                        lambda: ("Headphones", False, 50))
    assert follower.check_once() == ["output"]
    assert bus.audio_device_request.sent == [("output", "default")]
    assert any("Headphones" in line for line in bus.log.lines)


def test_both_can_move_at_once(monkeypatch):
    """A headset changes the default input AND output in one step."""
    follower, bus = _follower(
        monkeypatch,
        DefaultDevices("Mic A", "Speakers"),
        DefaultDevices("Headset Mic", "Headset"))
    assert sorted(follower.check_once()) == ["input", "output"]
    assert len(bus.audio_device_request.sent) == 2


def test_a_pinned_device_is_not_overridden(monkeypatch):
    """That was a decision. Following would silently undo it the next time
    anything changed."""
    follower, bus = _follower(
        monkeypatch,
        DefaultDevices("Mic A", "Speakers"),
        DefaultDevices("Mic A", "Headphones"),
        pin="Xrocker", pin_live=True)
    assert follower.check_once() == []
    assert bus.audio_device_request.sent == []
    assert any("pinned" in line for line in bus.log.lines), (
        "it ignored the change without saying why")


def test_a_pinned_device_that_is_unplugged_is_abandoned(monkeypatch):
    """Not a preference being honoured — ORION talking to something that is
    not there."""
    follower, bus = _follower(
        monkeypatch,
        DefaultDevices("Mic A", "Speakers"),
        DefaultDevices("Mic A", "Headphones"),
        pin="Unplugged Thing", pin_live=False)
    assert follower.check_once() == ["output"]
    assert bus.audio_device_request.sent == [("output", "default")]


def test_it_asks_for_the_default_not_the_device_name(monkeypatch):
    """Passing a NAME would pin it, and ORION would then refuse to follow the
    next change because he would think a choice had been made."""
    follower, bus = _follower(
        monkeypatch,
        DefaultDevices("Mic A", "Speakers"),
        DefaultDevices("Mic A", "Headphones"))
    follower.check_once()
    kind, spec = bus.audio_device_request.sent[0]
    assert spec == "default", f"following pinned {spec!r}"


def test_devices_are_compared_by_name_not_index():
    """Indices are renumbered as devices come and go, which produces phantom
    changes and misses real ones."""
    assert DefaultDevices("A", "B").differs_from(DefaultDevices("A", "B")) == []
    assert DefaultDevices("A", "B").differs_from(DefaultDevices("A", "C")) == ["output"]


def test_a_broken_bus_does_not_stop_the_watch(monkeypatch):
    class _Broken:
        log = _Log()

        class audio_device_request:
            @staticmethod
            def emit(*_a):
                raise RuntimeError("gone")

    follower = AudioFollower(bus=_Broken())
    follower._last = DefaultDevices("A", "B")
    monkeypatch.setattr(audio_follow, "read_defaults",
                        lambda refresh=True: DefaultDevices("A", "C"))
    monkeypatch.setattr(audio_follow, "pinned", lambda _k: "")
    assert follower.check_once() == []          # reported, not raised


def test_the_poll_is_slow_on_purpose():
    """Noticing a change means re-initialising PortAudio, which is not free.
    A headset is replugged a few times a day."""
    assert audio_follow.POLL_SECONDS >= 2.0


def test_it_is_wired_into_startup():
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "AudioFollower(bus=bus)" in source
    assert "orion-audio-follow" in source
    assert "audio_task.cancel()" in source, "the watcher outlives shutdown"


# -- setting a device from the GUI must not need a restart -------------------

def test_preferences_applies_a_device_live_not_on_next_start():
    """audio_devices.set_device() only PERSISTS a choice; it does not reopen
    the streams. Picking a microphone in Preferences therefore did nothing
    until ORION was restarted — the opposite of what somebody opening that
    dialog needs, since they are usually there BECAUSE he cannot hear them.

    The bus handler persists it, reopens the live stream on the thread that
    owns it, and says which device it landed on.
    """
    import inspect

    from orion_core.gui.core_window import OrionCoreWindow

    body = inspect.getsource(OrionCoreWindow.open_preferences)
    assert "audio_device_request.emit(kind, wanted)" in body
    applied = body.index("audio_device_request.emit")
    assert "set_device(kind, wanted)" not in body[applied - 400:applied], (
        "the persist-only path is back")


def test_the_interrupt_button_actually_stops_him():
    """Clicking it must cut playback, not merely ask politely."""
    import inspect

    from orion_core.gui.core_window import OrionCoreWindow
    from orion_core.live_worker import GenAILiveWorker

    wiring = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert "hud.command.interrupted.connect(self._stop_turn)" in wiring

    stop = inspect.getsource(OrionCoreWindow._stop_turn)
    assert "cancel_turn()" in stop

    cancel = inspect.getsource(GenAILiveWorker.cancel_turn)
    assert "interrupt_all()" in cancel, "it does not cut the audio"


# -- the crash this cost, and must never cost again --------------------------

def test_it_never_tears_portaudio_down():
    """The first version called sd._terminate() / sd._initialize() to make
    PortAudio notice a newly plugged device. That destroys state the OPEN
    capture and playback streams are still using, and it killed ORION with
    STATUS_HEAP_CORRUPTION inside a minute of starting. The fault dump named
    _refresh_portaudio directly.

    Asking Windows for the default endpoint is read-only and touches nothing
    ORION is using.
    """
    import ast
    import inspect

    from orion_core import audio_follow as module

    tree = ast.parse(inspect.getsource(module))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any("sounddevice" in (n or "") for n in names), (
                "sounddevice is back in the watcher")
    for forbidden in ("_terminate", "_initialize"):
        assert forbidden not in called, (
            f"{forbidden}() re-entered the watcher — this is the crash")


def test_devices_are_identified_by_endpoint_id():
    """Indices are renumbered as hardware comes and goes, and two identical
    headsets share a friendly name. An endpoint ID is stable and unique."""
    import sys

    if sys.platform != "win32":
        return
    from orion_core.audio_follow import read_defaults

    current = read_defaults()
    # A Windows endpoint ID looks like {0.0.0.00000000}.{guid}
    for value in (current.input_name, current.output_name):
        if value:
            assert value.startswith("{") and "}." in value, value
    assert read_defaults() == current, "the identity is not stable"


def test_reading_the_default_costs_nothing_the_loop_would_notice():
    """It runs off the event loop, but a watcher that is expensive anyway is
    a watcher somebody will be tempted to remove."""
    import sys
    import time

    if sys.platform != "win32":
        return
    from orion_core.audio_follow import POLL_SECONDS, read_defaults

    read_defaults()
    started = time.perf_counter()
    for _ in range(20):
        read_defaults()
    per_call = (time.perf_counter() - started) / 20
    assert per_call < 0.100, f"{per_call*1000:.0f} ms per check is too slow"
    assert per_call / POLL_SECONDS < 0.01, "over 1% duty cycle"


def test_the_desktop_muting_or_moving_is_written_down():
    """10 Oct: "my desktop keeps randomly muting" - ORION never mutes or moves
    the default output unasked, and nothing recorded what it was doing."""
    from orion_core.audio_follow import describe_output_change
    monitor = ("XV252Q (NVIDIA High Definition Audio)", False, 100)
    assert describe_output_change(monitor, monitor) == ""
    assert describe_output_change(None, monitor) == ""
    assert "was muted" in describe_output_change(monitor, (monitor[0], True, 100))
    assert "moved from XV252Q" in describe_output_change(
        monitor, ("Speakers (4- fifine Microphone)", False, 100))
    assert "100% -> 0%" in describe_output_change(monitor, (monitor[0], False, 0))
    assert describe_output_change(monitor, (monitor[0], False, 96)) == ""   # small nudges


def test_the_watcher_logs_an_output_change(monkeypatch):
    import orion_core.audio_follow as af
    logged = []
    states = iter([("Monitor", False, 100), ("Monitor", True, 100)])
    monkeypatch.setattr(af, "desktop_output_state", lambda: next(states))
    monkeypatch.setattr(af, "read_defaults", lambda refresh=True: af.DefaultDevices())
    watcher = af.AudioFollower(bus=None)
    watcher._log = logged.append
    watcher.prime()
    watcher.check_once()
    assert logged == ["AUDIO: the desktop was muted (Monitor)."]


# -- 20 Oct: the headset switched off and ORION stayed mute ------------------
#
# Windows moved the default to the monitor; PortAudio's cached device list did
# not know, every open failed "device ID out of range", and nothing refreshed
# it. The follower now refreshes PortAudio (with every stream closed) BEFORE it
# asks ORION to reopen, names devices by friendly name, and declines to follow
# onto a monitor's speakers while the device it was on is still connected.

HEADSET_ID = "{0.0.0.00000000}.{headset-guid}"
MONITOR_ID = "{0.0.0.00000000}.{monitor-guid}"
HEADSET = "Speakers (HyperX Cloud III S Wireless)"
MONITOR = "XV252Q (NVIDIA High Definition Audio)"


class _Order:
    """The order in which the follower refreshed PortAudio and moved ORION."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def reinit(self, reason, force=False):
        self.events.append(("reinit", reason, force))
        return True

    def apply(self, *args):
        self.events.append(("apply", *args))


def _watch(monkeypatch, *, before, after, before_output, after_output, active=None):
    """A follower whose Windows state is scripted (no Core Audio, no PortAudio)."""
    order = _Order()
    bus = _Bus()
    bus.audio_device_request = type("S", (), {"emit": staticmethod(order.apply)})()
    follower = AudioFollower(bus=bus)
    follower._last = before
    follower._output = before_output
    scripted = {"defaults": after, "output": after_output, "active": dict(active or {})}
    monkeypatch.setattr(audio_follow, "read_defaults", lambda refresh=True: scripted["defaults"])
    monkeypatch.setattr(audio_follow, "desktop_output_state", lambda: scripted["output"])
    monkeypatch.setattr(audio_follow, "endpoint_is_active", lambda eid: scripted["active"].get(eid))
    monkeypatch.setattr(audio_follow, "pinned", lambda _k: "")
    monkeypatch.setattr(audio_follow.audio_devices, "request_reinit", order.reinit)
    return follower, bus, order, scripted


def test_the_follow_line_names_the_device_not_its_endpoint_id(monkeypatch):
    follower, bus, _order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic-old}", "{spk-old}"),
        after=DefaultDevices("{mic-old}", HEADSET_ID),
        before_output=("Speakers (Realtek)", False, 40),
        after_output=(HEADSET, False, 40))
    assert follower.check_once() == ["output"]
    line = next(l for l in bus.log.lines if "following the system default" in l)
    assert HEADSET in line and "{0.0.0" not in line and "guid" not in line


def test_the_input_is_named_by_its_core_audio_name_too(monkeypatch):
    follower, bus, _order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic-old}", HEADSET_ID),
        after=DefaultDevices("{mic-new}", HEADSET_ID),
        before_output=(HEADSET, False, 40), after_output=(HEADSET, False, 40))
    monkeypatch.setattr(audio_follow.audio_devices, "_system_default_name",
                        lambda kind: "Microphone (Webcam C920)" if kind == "input" else None)
    assert follower.check_once() == ["input"]
    assert any("now Microphone (Webcam C920)" in l for l in bus.log.lines)


def test_portaudio_is_refreshed_before_orion_is_asked_to_reopen(monkeypatch):
    follower, _bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: False})                 # the headset is switched off
    assert follower.check_once() == ["output"]
    kinds = [event[0] for event in order.events]
    assert kinds == ["reinit", "apply"], "it reopened on a list it had not refreshed"
    assert order.events[0][2] is True, "a real default change must force the refresh"
    assert MONITOR in order.events[0][1]
    assert order.events[1][1:] == ("output", "default")


def test_one_refresh_serves_input_and_output_moving_together(monkeypatch):
    follower, _bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic-a}", "{spk-a}"),
        after=DefaultDevices("{mic-b}", "{spk-b}"),
        before_output=("Speakers", False, 40), after_output=("Headset", False, 40))
    assert sorted(follower.check_once()) == ["input", "output"]
    assert [e[0] for e in order.events].count("reinit") == 1
    assert order.events[0][0] == "reinit"


def test_nothing_changing_refreshes_nothing(monkeypatch):
    same = DefaultDevices("{mic}", HEADSET_ID)
    follower, _bus, order, _s = _watch(
        monkeypatch, before=same, after=same,
        before_output=(HEADSET, False, 40), after_output=(HEADSET, False, 40))
    assert follower.check_once() == []
    assert order.events == []


def test_a_failed_refresh_still_moves_orion_as_before(monkeypatch):
    follower, _bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", "{spk-a}"), after=DefaultDevices("{mic}", "{spk-b}"),
        before_output=("A", False, 40), after_output=("B", False, 40))
    monkeypatch.setattr(audio_follow.audio_devices, "request_reinit",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert follower.check_once() == ["output"]
    assert order.events == [("apply", "output", "default")]


# -- a monitor's speakers are not where the user went ------------------------

def test_display_endpoints_are_recognised_by_name():
    for name in (MONITOR, "Digital Output (AMD High Definition Audio Device)",
                 "LG TV (HDMI)", "DELL U2720Q (DisplayPort)", "Display Port Out",
                 "Intel(R) Display Audio", "Intel Display Audio"):
        assert audio_follow.is_display_endpoint(name), name
    for name in (HEADSET, "Headphones (2- soundcore P31i)", "Speakers (Realtek(R) Audio)",
                 "Speakers (Xrocker)", "", None):
        assert not audio_follow.is_display_endpoint(name), name


def test_the_monitor_does_not_steal_the_voice_while_the_headset_is_on(monkeypatch):
    follower, bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: True})
    assert follower.check_once() == []
    assert order.events == [], "it moved (or refreshed) for a monitor it should ignore"
    why = [l for l in bus.log.lines if "staying on it" in l and "pinned" not in l]
    assert len(why) == 1 and HEADSET in why[0] and MONITOR in why[0]
    # Held, so a later reopen does not quietly wander onto the monitor either.
    assert audio_follow.audio_devices._held_default == {"output": HEADSET}


def test_declining_the_monitor_is_said_once_not_every_poll(monkeypatch):
    follower, bus, _order, scripted = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: True})
    for _ in range(4):
        assert follower.check_once() == []
    assert sum("staying on it" in l for l in bus.log.lines) == 1


def test_the_headset_switching_off_sends_the_voice_to_the_monitor(monkeypatch):
    """Windows moves the default as the headset goes: the monitor is then the
    only output, so it is followed - and the refresh comes first."""
    follower, _bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: False})
    assert follower.check_once() == ["output"]
    assert [e[0] for e in order.events] == ["reinit", "apply"]
    assert audio_follow.audio_devices._held_default == {}


def test_a_headset_that_goes_after_the_monitor_took_the_default_is_followed_then(monkeypatch):
    follower, bus, order, scripted = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: True})
    assert follower.check_once() == []                       # staying put
    scripted["active"][HEADSET_ID] = False                   # now it switches off
    assert follower.check_once() == ["output"]               # no default change needed
    assert [e[0] for e in order.events] == ["reinit", "apply"]
    assert any("no longer connected" in l and HEADSET in l for l in bus.log.lines)
    assert audio_follow.audio_devices._held_default == {}
    assert follower.check_once() == []                       # and it is not repeated


def test_the_headset_coming_back_is_followed_back(monkeypatch):
    follower, _bus, order, scripted = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: False})
    assert follower.check_once() == ["output"]               # headset off: monitor
    order.events.clear()
    scripted["defaults"] = DefaultDevices("{mic}", HEADSET_ID)
    scripted["output"] = (HEADSET, False, 40)
    scripted["active"][HEADSET_ID] = True
    assert follower.check_once() == ["output"]               # headset on: back to it
    assert [e[0] for e in order.events] == ["reinit", "apply"]


def test_the_headset_coming_back_ends_a_stay_cleanly(monkeypatch):
    follower, _bus, _order, scripted = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: True})
    assert follower.check_once() == []                       # staying on the headset
    scripted["defaults"] = DefaultDevices("{mic}", HEADSET_ID)
    scripted["output"] = (HEADSET, False, 40)
    assert follower.check_once() == ["output"]               # Windows agrees again
    assert follower._staying is None
    assert audio_follow.audio_devices._held_default == {}


def test_a_move_between_two_ordinary_outputs_is_still_followed(monkeypatch):
    follower, _bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", "{a}"), after=DefaultDevices("{mic}", "{b}"),
        before_output=("Speakers (Realtek)", False, 40),
        after_output=(HEADSET, False, 40), active={"{a}": True})
    assert follower.check_once() == ["output"]


def test_display_to_display_is_followed_there_is_nothing_better_to_stay_on(monkeypatch):
    follower, _bus, _order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", MONITOR_ID),
        after=DefaultDevices("{mic}", "{tv}"),
        before_output=(MONITOR, False, 40), after_output=("LG TV (HDMI)", False, 40),
        active={MONITOR_ID: True})
    assert follower.check_once() == ["output"]


def test_when_windows_cannot_be_asked_it_follows_as_it_always_did(monkeypatch):
    """endpoint_is_active() is None off Windows or without pycaw."""
    follower, _bus, _order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={})                                           # unknown
    assert follower.check_once() == ["output"]


def test_a_pinned_output_is_still_the_users_decision_over_all_of_this(monkeypatch):
    follower, bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: False})
    monkeypatch.setattr(audio_follow, "pinned", lambda _k: "Xrocker")
    monkeypatch.setattr(audio_follow, "pin_is_live", lambda _k: True)
    assert follower.check_once() == []
    assert order.events == []


def test_endpoint_is_active_says_unknown_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(audio_follow.sys, "platform", "linux")
    assert _REAL_ENDPOINT_IS_ACTIVE(HEADSET_ID) is None
    monkeypatch.setattr(audio_follow.sys, "platform", "win32")
    assert _REAL_ENDPOINT_IS_ACTIVE("") is None


# -- 25 Oct: "can you do a restart because I can't hear you" ------------------
#
# 11:50:36 Windows moved the default to the XV252Q monitor - where the owner
# listens (all of 24 Oct ran on it) - and ORION stayed on the HyperX headset
# because its wireless dongle still reported "connected" with the headset off.

def test_a_display_that_is_the_owners_usual_output_is_followed(monkeypatch, tmp_path):
    follower, bus, order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: True})                       # the dongle stays "connected"
    (tmp_path / "habit.json").write_text(
        '{"seconds": {"%s": 5400, "%s": 300}}' % (MONITOR, HEADSET), encoding="utf-8")
    follower.habit_path = tmp_path / "habit.json"
    follower._habit = follower._load_habit()
    assert follower.check_once() == ["output"]
    assert follower._staying is None
    assert any("usually listen" in line for line in bus.log.lines)


def test_a_waking_monitor_still_does_not_steal_a_headset_in_use(monkeypatch, tmp_path):
    follower, _bus, _order, _s = _watch(
        monkeypatch,
        before=DefaultDevices("{mic}", HEADSET_ID),
        after=DefaultDevices("{mic}", MONITOR_ID),
        before_output=(HEADSET, False, 40), after_output=(MONITOR, False, 40),
        active={HEADSET_ID: True})
    (tmp_path / "habit.json").write_text(
        '{"seconds": {"%s": 9000, "%s": 120}}' % (HEADSET, MONITOR), encoding="utf-8")
    follower.habit_path = tmp_path / "habit.json"
    follower._habit = follower._load_habit()
    assert follower.check_once() == []
    assert follower._staying is not None


def test_listening_time_is_recorded_only_where_a_habit_file_is_given(monkeypatch, tmp_path):
    same = DefaultDevices("{mic}", MONITOR_ID)
    follower, _bus, _order, _s = _watch(
        monkeypatch, before=same, after=same,
        before_output=(MONITOR, False, 40), after_output=(MONITOR, False, 40))
    follower.check_once()
    assert follower._habit == {}                          # no path: nothing kept or written
    follower.habit_path = tmp_path / "habit.json"
    follower.HABIT_SAVE_EVERY_S = 1.0
    follower.check_once()
    import json as _json
    saved = _json.loads((tmp_path / "habit.json").read_text(encoding="utf-8"))
    assert saved["seconds"][MONITOR] == follower.poll_seconds
