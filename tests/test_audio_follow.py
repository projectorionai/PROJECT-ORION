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
