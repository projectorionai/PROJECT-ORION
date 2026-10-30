"""PortAudio is re-initialised when its device list goes stale - and never
while a stream is open.

On 20 Oct the owner's Bluetooth headset switched off. Windows moved the default
output to the monitor, PortAudio's list (read once, at initialisation) no
longer matched the machine, and every open failed with "A device ID has been
used that is out of range for your system" - 799 times in fourteen minutes,
once a second, while ORION stayed mute until the process was killed.

The cure is to re-initialise PortAudio (sd._terminate() + sd._initialize()).
The hazard is that doing so with ANY stream open frees the state that stream
still points at: STATUS_HEAP_CORRUPTION, seen inside a minute on 2026-09-20.
So these tests are mostly about the second half: the coordinator closes the
renderer's and the microphone's streams first, waits, and gives up rather
than terminate under a stream that did not close.

Offline: a fake sounddevice module that counts its open streams. No hardware,
no microphone, no real PortAudio.
"""

from __future__ import annotations

import ast
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import audio as audio_module  # noqa: E402
from orion_core import audio_devices as ad  # noqa: E402
from orion_core.audio import AudioPlaybackThread, MicrophoneEngine  # noqa: E402

#: Verbatim from session.log, 2026-10-20 19:22:35.
STALE = ("Error opening RawOutputStream: Unanticipated host error [PaErrorCode -9999]: "
         "'A device ID has been used that is out of range for your system.' [MME error 2]")

HEADSET = "Speakers (HyperX Cloud III S Wireless)"
MONITOR = "XV252Q (NVIDIA High Definition Audio)"
HEADSET_MIC = "Headset Mic (HyperX Cloud III S Wireless)"
WEBCAM_MIC = "Microphone (Webcam C920)"


def _device(name, out=0, inp=0):
    return {"name": name, "max_output_channels": out, "max_input_channels": inp, "hostapi": 0}


DEVICES = [
    _device(HEADSET, out=2),          # 0
    _device(MONITOR, out=2),          # 1
    _device(HEADSET_MIC, inp=1),      # 2
    _device(WEBCAM_MIC, inp=1),       # 3
]
#: The same machine after the headset went: PortAudio renumbers on re-init, so
#: the monitor is no longer at index 1 and the webcam mic is no longer at 3.
DEVICES_AFTER = [
    _device(WEBCAM_MIC, inp=1),                  # 0
    _device("Speakers (Realtek Audio)", out=2),  # 1
    _device(MONITOR, out=2),                     # 2
]


class PaError(Exception):
    pass


# ── a sounddevice that counts what is open ────────────────────────────────────

class FakeStream:
    def __init__(self, sd, kind, device):
        self.sd, self.kind, self.device = sd, kind, device
        self.closed = False
        self.written: list[bytes] = []
        self.block_close = None            # a threading.Event the close waits on
        sd.open_streams += 1
        sd.opened.append((kind, device))
        sd.streams.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        if self.block_close is not None:
            self.block_close.wait(10)
        if not self.closed:
            self.closed = True
            self.sd.open_streams -= 1

    def write(self, piece):
        self.written.append(bytes(piece))
        return False


class FakeSd:
    """Records every terminate and, crucially, how many streams were open then."""

    def __init__(self, devices=DEVICES):
        self.devices = list(devices)
        self.open_streams = 0
        self.opened: list[tuple] = []
        self.streams: list[FakeStream] = []
        self.calls: list[str] = []
        self.violations: list[int] = []     # open-stream counts seen AT terminate
        self.fail_open: Exception | None = None
        self.attempts = 0
        self.after_initialize = None
        self.on_terminate = None
        self.initialize_errors: list[Exception] = []
        self.active_stream = None

    def _terminate(self):
        self.calls.append("terminate")
        if self.open_streams:
            self.violations.append(self.open_streams)
        if self.on_terminate:
            self.on_terminate()

    def _initialize(self):
        self.calls.append("initialize")
        if self.initialize_errors:
            raise self.initialize_errors.pop(0)
        if self.after_initialize:
            self.after_initialize()

    def query_devices(self, device=None, kind=None):
        if kind:
            key = "max_input_channels" if kind == "input" else "max_output_channels"
            return next(d for d in self.devices if d[key] > 0)
        return list(self.devices)

    def query_hostapis(self, index=None):
        apis = [{"name": "MME"}]
        return apis if index is None else apis[index]

    def get_stream(self):
        if self.active_stream is None:
            raise RuntimeError("play()/rec()/playrec() was not called yet")
        return self.active_stream

    def _make(self, kind, kwargs):
        self.attempts += 1
        if self.fail_open is not None:
            raise self.fail_open
        return FakeStream(self, kind, kwargs.get("device"))

    def RawOutputStream(self, **kwargs):
        return self._make("output", kwargs)

    def RawInputStream(self, **kwargs):
        return self._make("input", kwargs)


@pytest.fixture
def fake_audio(monkeypatch):
    """A machine with a headset, a monitor and two microphones, and no hardware."""
    sd = FakeSd()
    names = {"output": HEADSET, "input": HEADSET_MIC}
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(audio_module, "sd", sd)
    monkeypatch.setattr(audio_module, "_boost_thread_priority", lambda *_a, **_k: None)
    monkeypatch.setattr(ad, "_system_default_name", lambda kind: names.get(kind))
    monkeypatch.setattr(ad, "_load_config", lambda: {})
    monkeypatch.delenv("ORION_AUDIO_INPUT", raising=False)
    monkeypatch.delenv("ORION_AUDIO_OUTPUT", raising=False)
    monkeypatch.setattr(ad, "_held_default", {})
    monkeypatch.setattr(ad, "_live", {})
    monkeypatch.setattr(ad, "_transport_cache", {})
    return SimpleNamespace(sd=sd, names=names)


class Lines:
    def __init__(self):
        self.lines: list[str] = []

    def emit(self, line):
        self.lines.append(line)


class _Sig:
    def emit(self, *_a):
        pass


class _Bus:
    """Any signal the audio code reaches for exists and does nothing; log records."""

    def __init__(self):
        self.log = Lines()

    def __getattr__(self, name):
        signal = _Sig()
        setattr(self, name, signal)
        return signal


def make_bus():
    bus = _Bus()
    return bus, bus.log


class Clock:
    """Time that only moves when something sleeps."""

    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def coordinator(sd, clock=None, **kwargs):
    clock = clock or Clock()
    return ad.PortAudioReinit(clock=clock, sleep=clock.sleep, portaudio=lambda: sd, **kwargs), clock


def enrol(coord, owner):
    """Register *owner* and keep it alive: the coordinator holds owners weakly
    (the real ones are held by the app), so a temporary would vanish."""
    coord.register(owner)
    coord.__dict__.setdefault("_test_refs", []).append(owner)
    return owner


class Owner:
    """A stand-in for the renderer or the microphone: holds one fake stream."""

    def __init__(self, sd, label, *, closes_after=0, log=None):
        self.sd, self.reinit_label = sd, label
        self.stream = FakeStream(sd, "output", 0)
        self.closes_after = closes_after        # polls before it lets go; None = never
        self.polls = 0
        self.released = False
        self.restored = 0
        self.lines = log if log is not None else []

    def reinit_release(self):
        pass

    def reinit_released(self):
        self.polls += 1
        if self.closes_after is not None and self.polls > self.closes_after and not self.released:
            self.stream.close()
            self.released = True
        return self.released

    def reinit_restore(self):
        self.restored += 1
        self.released = False
        if self.stream.closed:                       # reopen on the fresh list
            self.stream = FakeStream(self.sd, "output", 0)

    def reinit_log(self, message):
        self.lines.append(message)


# ── the coordinator: close first, then terminate, then reopen ─────────────────

def test_every_stream_is_closed_before_portaudio_is_terminated(fake_audio):
    sd = fake_audio.sd
    renderer = Owner(sd, "the voice output", closes_after=2)
    mic = Owner(sd, "the microphone", closes_after=5)
    coord, _clock = coordinator(sd)
    coord.register(renderer)
    coord.register(mic)
    assert sd.open_streams == 2
    assert coord.request_reinit("test", force=True) is True
    assert sd.calls == ["terminate", "initialize"]
    assert sd.violations == [], "PortAudio was terminated with a stream open"
    assert renderer.restored == 1 and mic.restored == 1
    assert sd.open_streams == 2, "both owners reopened afterwards"


def test_one_clear_line_names_the_defaults_by_friendly_name(fake_audio):
    sd = fake_audio.sd
    owner = Owner(sd, "the voice output", closes_after=0)
    coord, _clock = coordinator(sd)
    coord.register(owner)
    assert coord.request_reinit("the default output changed to " + MONITOR, force=True)
    done = [line for line in owner.lines if "re-initialised" in line]
    assert len(done) == 1 and len(owner.lines) == 1, owner.lines
    assert MONITOR in done[0] and f"default output: {HEADSET}" in done[0]
    assert f"default input: {HEADSET_MIC}" in done[0]
    assert "{0.0.0" not in done[0]


def test_it_gives_up_rather_than_terminate_under_a_microphone_that_will_not_close(fake_audio):
    sd = fake_audio.sd
    renderer = Owner(sd, "the voice output", closes_after=0)
    mic = Owner(sd, "the microphone", closes_after=None)       # never lets go
    coord, clock = coordinator(sd)
    coord.register(renderer)
    coord.register(mic)
    started = clock()
    assert coord.request_reinit("test", force=True) is False
    assert sd.calls == [], "terminate was called with a microphone stream open"
    assert sd.violations == []
    assert clock() - started == pytest.approx(3.0, abs=0.05), "it did not wait the 3 s"
    assert renderer.restored == 1 and mic.restored == 1, "streams were not handed back"
    assert coord.abandoned == 1 and coord.completed == 0
    said = [line for line in renderer.lines + mic.lines if "NOT re-initialised" in line]
    assert len(said) == 1
    assert "the microphone" in said[0] and "the voice output" not in said[0]


def test_it_gives_up_when_the_renderer_does_not_close_either(fake_audio):
    sd = fake_audio.sd
    renderer = Owner(sd, "the voice output", closes_after=None)
    mic = Owner(sd, "the microphone", closes_after=0)
    coord, _clock = coordinator(sd)
    coord.register(renderer)
    coord.register(mic)
    assert coord.request_reinit("test", force=True) is False
    assert sd.calls == []
    assert any("the voice output" in line for line in renderer.lines + mic.lines)


def test_an_abandoned_attempt_is_logged_once_not_every_time(fake_audio):
    sd = fake_audio.sd
    owner = Owner(sd, "the microphone", closes_after=None)
    coord, clock = coordinator(sd)
    coord.register(owner)
    for _ in range(5):
        coord.request_reinit("again", force=True)
        clock.t += 5
    assert coord.abandoned == 5
    assert sum("NOT re-initialised" in line for line in owner.lines) == 1
    clock.t += 61                                      # a minute on, one summary
    coord.request_reinit("again", force=True)
    assert sum("NOT re-initialised" in line for line in owner.lines) == 2
    assert "repeated" in owner.lines[-1]
    assert sd.calls == []


def test_it_waits_for_a_probe_stream_to_close(fake_audio):
    """probe() and transport_works() open streams of their own for a moment."""
    sd = fake_audio.sd
    owner = Owner(sd, "the voice output", closes_after=0)
    coord, clock = coordinator(sd)
    coord.register(owner)
    guard = ad.stream_guard()
    guard.__enter__()
    released = []
    real_sleep = clock.sleep

    def sleeping(seconds):
        real_sleep(seconds)
        if clock.t > 1000.5 and not released:        # the probe finishes half a second in
            released.append(clock.t)
            assert sd.calls == [], "terminated while the probe stream was open"
            guard.__exit__(None, None, None)

    coord._sleep = sleeping
    assert coord.request_reinit("test", force=True) is True
    assert released and sd.calls == ["terminate", "initialize"]


def test_it_gives_up_if_a_probe_never_closes(fake_audio):
    sd = fake_audio.sd
    coord, _clock = coordinator(sd)
    enrol(coord, Owner(sd, "the voice output", closes_after=0))
    with ad.stream_guard():
        assert coord.request_reinit("test", force=True) is False
    assert sd.calls == []


def test_it_waits_for_a_play_or_rec_stream(fake_audio):
    sd = fake_audio.sd
    sd.active_stream = SimpleNamespace(active=True)
    coord, clock = coordinator(sd)
    real_sleep = clock.sleep

    def sleeping(seconds):
        real_sleep(seconds)
        sd.active_stream.active = False               # playback finishes

    coord._sleep = sleeping
    assert coord.request_reinit("test", force=True) is True
    assert sd.calls == ["terminate", "initialize"]


def test_a_closed_play_stream_does_not_block_it(fake_audio):
    sd = fake_audio.sd

    class Closed:
        @property
        def active(self):
            raise PaError("Error checking stream activity: Invalid stream pointer")

    sd.active_stream = Closed()
    coord, _clock = coordinator(sd)
    assert coord.request_reinit("test", force=True) is True


def test_probe_and_transport_check_hold_the_guard_while_their_stream_exists(fake_audio):
    sd = fake_audio.sd
    seen = []
    original = sd._make

    def spying(kind, kwargs):
        seen.append(ad._transient_streams)
        return original(kind, kwargs)

    sd._make = spying
    assert ad.probe("output", 0)[0]
    assert ad.transport_works("input", 2, use_cache=False)[0] in (True, False)
    assert seen and all(count == 1 for count in seen), seen
    assert ad._transient_streams == 0, "the guard leaked"


def test_a_probe_that_starts_mid_reinit_waits_for_it_to_finish(fake_audio):
    sd = fake_audio.sd
    entered: list[int] = []
    worker = {}

    class Owner2(Owner):
        def reinit_release(self):
            def probe_like():
                with ad.stream_guard():
                    entered.append(len(sd.calls))
            worker["t"] = threading.Thread(target=probe_like)
            worker["t"].start()
            time.sleep(0.05)                          # let it reach the guard

    owner = Owner2(sd, "the voice output", closes_after=0)
    coord = ad.PortAudioReinit(portaudio=lambda: sd)
    coord.register(owner)
    assert coord.request_reinit("test", force=True)
    worker["t"].join(5)
    assert entered == [2], f"the probe ran during the re-init: {entered}"


def test_device_queries_wait_while_portaudio_is_down(fake_audio):
    sd = fake_audio.sd
    seen = {}

    def query_during_terminate():
        def query():
            seen["devices"] = ad._devices()
            seen["calls_when_answered"] = list(sd.calls)
        seen["thread"] = threading.Thread(target=query)
        seen["thread"].start()
        time.sleep(0.1)                               # the query is now blocked

    sd.on_terminate = query_during_terminate
    coord = ad.PortAudioReinit(portaudio=lambda: sd)
    enrol(coord, Owner(sd, "the voice output", closes_after=0))
    assert coord.request_reinit("test", force=True)
    seen["thread"].join(5)
    assert seen["calls_when_answered"] == ["terminate", "initialize"], (
        "a query ran against PortAudio between terminate and initialize")


def test_a_second_request_while_one_runs_is_covered_by_it(fake_audio):
    sd = fake_audio.sd
    results = []

    class Reentrant(Owner):
        def reinit_release(self):
            results.append(coord.request_reinit("nested", force=True))

    coord, _clock = coordinator(sd)
    enrol(coord, Reentrant(sd, "the voice output", closes_after=0))
    assert coord.request_reinit("first", force=True)
    assert results == [False]
    assert sd.calls == ["terminate", "initialize"], "it terminated twice"


def test_requests_close_together_are_one_event_unless_forced(fake_audio):
    sd = fake_audio.sd
    coord, clock = coordinator(sd)
    enrol(coord, Owner(sd, "the voice output", closes_after=0))
    assert coord.request_reinit("first") is True
    clock.t += 1.0
    assert coord.request_reinit("a stale-list failure a moment later") is False
    assert sd.calls == ["terminate", "initialize"]
    assert coord.request_reinit("a real Windows default change", force=True) is True
    assert sd.calls == ["terminate", "initialize"] * 2
    clock.t += 10.0
    assert coord.request_reinit("much later") is True


def test_a_failed_initialize_is_retried_and_reported_and_owners_still_restored(fake_audio):
    sd = fake_audio.sd
    sd.initialize_errors = [PaError("Error initializing PortAudio: Unanticipated host error")]
    owner = Owner(sd, "the voice output", closes_after=0)
    coord, _clock = coordinator(sd)
    coord.register(owner)
    assert coord.request_reinit("test", force=True) is False
    assert sd.calls == ["terminate", "initialize", "initialize"], "PortAudio was left down"
    assert owner.restored == 1
    assert coord.failed == 1
    assert any("could not be re-initialised" in line for line in owner.lines)


def test_real_portaudio_is_never_reinitialised_under_pytest(monkeypatch):
    real_looking = SimpleNamespace(_lib=object(), _terminate=lambda: pytest.fail("terminated"))
    monkeypatch.setitem(sys.modules, "sounddevice", real_looking)
    assert ad._real_portaudio() is None
    coord = ad.PortAudioReinit()
    assert coord.request_reinit("test", force=True) is False
    fake = SimpleNamespace(_terminate=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    assert ad._real_portaudio() is fake


def test_the_async_request_runs_off_the_callers_thread(fake_audio):
    sd = fake_audio.sd
    coord = ad.PortAudioReinit(portaudio=lambda: sd)
    enrol(coord, Owner(sd, "the voice output", closes_after=0))
    caller = threading.get_ident()
    sd.on_terminate = lambda: sd.violations.append(0) if threading.get_ident() == caller else None
    assert coord.request_reinit_async("test", force=True) is True
    deadline = time.monotonic() + 5
    while coord.completed == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert coord.completed == 1 and sd.violations == []


def test_only_the_coordinator_ever_terminates_portaudio():
    """The crash was a terminate with streams open. Anything else calling it is a
    second, unguarded way to do the same thing."""
    callers = []
    for path in sorted((ROOT / "orion_core").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        class Walk(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr in {"_terminate", "_initialize"}
                        and isinstance(func.value, ast.Name) and func.value.id in {"sd", "sounddevice"}):
                    callers.append((path.name, ".".join(self.stack), func.attr))
                self.generic_visit(node)

        Walk().visit(tree)
    assert sorted(callers) == [("audio_devices.py", "PortAudioReinit._reinit", "_initialize"),
                               ("audio_devices.py", "PortAudioReinit._reinit", "_initialize"),
                               ("audio_devices.py", "PortAudioReinit._reinit", "_terminate")], callers


# ── log hygiene ───────────────────────────────────────────────────────────────

def test_a_repeating_failure_is_logged_once_then_counted_every_minute():
    clock, out = Clock(0.0), []
    log = ad.FailureLog(out.append, 60.0, clock)
    for _ in range(100):                               # once a second, as the renderer did
        log.note("AUDIO: output device open failed (None) - " + STALE)
        clock.t += 1.0
    assert out[0] == "AUDIO: output device open failed (None) - " + STALE
    assert len(out) == 2                               # the first, and one summary at 60 s
    assert "repeated 60 more times in the last 60 s" in out[1]
    log.reset()                                        # recovered: report what was held back
    assert len(out) == 3 and "repeated 39 more times before it recovered" in out[2]


def test_799_failures_in_fourteen_minutes_are_about_fifteen_lines():
    clock, out = Clock(0.0), []
    log = ad.FailureLog(out.append, 60.0, clock)
    for _ in range(799):
        log.note("AUDIO: output device open failed (None) - " + STALE)
        clock.t += 14 * 60 / 799
    log.reset()
    assert len(out) <= 17, len(out)
    held = sum(int(line.split("repeated ")[1].split(" ")[0]) for line in out[1:])
    assert held == 798, "the count must add up to what happened"


def test_different_failures_each_get_their_first_line():
    out = []
    log = ad.FailureLog(out.append, 60.0, Clock(0.0))
    log.note("open failed (None) - A")
    log.note("open failed (None) - B")
    log.note("open failed (None) - A")
    assert out == ["open failed (None) - A", "open failed (None) - B"]


def test_a_clean_recovery_reports_nothing_extra():
    out = []
    log = ad.FailureLog(out.append, 60.0, Clock(0.0))
    log.note("x")
    log.reset()
    assert out == ["x"]


@pytest.mark.parametrize("text", [
    STALE,
    "Error opening RawOutputStream: Invalid device [PaErrorCode -9996]",
    "Error opening RawOutputStream: Device unavailable [PaErrorCode -9985]",
    "A device ID has been used that is out of range for your system",
    "PaErrorCode -9999",
])
def test_stale_list_errors_are_recognised(text):
    assert ad.looks_like_stale_device_list(PaError(text))


@pytest.mark.parametrize("text", [
    "Invalid sample rate [PaErrorCode -9997]",
    "Invalid number of channels [PaErrorCode -9998]",
    "No output device matching 'xrocker'",
    "",
])
def test_other_failures_are_not_mistaken_for_a_stale_list(text):
    assert not ad.looks_like_stale_device_list(PaError(text))


# ── the renderer's self-heal ──────────────────────────────────────────────────

def _renderer(fake_audio, monkeypatch, *, error=STALE):
    bus, log = make_bus()
    fake_audio.sd.fail_open = PaError(error) if error else None
    requests: list[tuple[float, str]] = []
    now = [0.0]
    monkeypatch.setattr(ad, "request_reinit_async",
                        lambda reason, force=False: requests.append((now[0], reason)) or True)
    monkeypatch.setattr(ad, "system_default_index", lambda kind: 1)
    playback = AudioPlaybackThread(bus)
    playback.tries = 0
    original = playback._try_open

    def counting(device):                              # one _try_open = one self-heal attempt
        playback.tries += 1
        return original(device)

    playback._try_open = counting
    return playback, log, requests, now


def test_a_stale_list_asks_for_exactly_one_reinit_then_backs_off(fake_audio, monkeypatch):
    playback, _log, requests, now = _renderer(fake_audio, monkeypatch)
    attempts: list[float] = []
    for step in range(int(14 * 60 / 0.5)):             # the 14 minutes of 20 Oct
        now[0] = step * 0.5
        before = playback.tries
        playback._heal_output(now[0])
        if playback.tries > before:
            attempts.append(now[0])
    # The log showed ~800 attempts; backing off 1, 2, 4, 8, 16, 30, 30 ... s.
    assert attempts[:8] == [0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 61.0, 91.0]
    assert len(attempts) < 40, f"{len(attempts)} attempts: it is still hammering"
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    assert max(gaps) == 30.0 and gaps[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    # After the second stale failure (t=1) exactly one request - and no more
    # for two minutes, however many failures follow.
    assert requests[0][0] == 1.0 and "stale" in requests[0][1]
    assert [t for t, _ in requests if t < 121.0] == [1.0]
    assert 5 <= len(requests) <= 7, "later ones are for a device that appears later"


def test_one_stale_failure_alone_is_not_enough(fake_audio, monkeypatch):
    playback, _log, requests, now = _renderer(fake_audio, monkeypatch)
    playback._heal_output(0.0)
    assert requests == []
    playback._heal_output(1.0)
    assert len(requests) == 1


def test_failures_that_are_not_a_stale_list_back_off_but_never_reinitialise(fake_audio, monkeypatch):
    playback, _log, requests, now = _renderer(
        fake_audio, monkeypatch, error="Error opening RawOutputStream: Invalid sample rate [PaErrorCode -9997]")
    for step in range(int(300 / 0.5)):
        now[0] = step * 0.5
        playback._heal_output(now[0])
    assert requests == []
    assert playback.tries < 20, "it is still retrying every second"


def test_a_stale_failure_between_ordinary_ones_does_not_count_as_consecutive(fake_audio, monkeypatch):
    playback, _log, requests, now = _renderer(fake_audio, monkeypatch)
    sd = fake_audio.sd
    playback._heal_output(0.0)                         # stale
    sd.fail_open = PaError("Error opening RawOutputStream: Invalid sample rate [PaErrorCode -9997]")
    playback._heal_output(1.0)                         # not stale: the run is broken
    sd.fail_open = PaError(STALE)
    playback._heal_output(3.0)                         # stale again, but only the first
    assert requests == []


def test_success_resets_the_backoff_and_the_failure_log(fake_audio, monkeypatch):
    playback, log, requests, now = _renderer(fake_audio, monkeypatch)
    sd = fake_audio.sd
    for t in (0.0, 1.0, 3.0):
        playback._heal_output(t)
    assert playback._open_failures == 3 and playback._next_open_at == 7.0
    sd.fail_open = None
    assert playback._heal_output(7.0) is False         # reopened
    assert playback._stream is not None
    assert (playback._open_failures, playback._stale_failures, playback._next_open_at) == (0, 0, 0.0)
    assert any("recovered" in line for line in log.lines)
    playback._stream.close()
    playback._stream = None
    sd.fail_open = PaError(STALE)
    playback._heal_output(8.0)
    assert playback._next_open_at == 9.0, "the next outage must start again at 1 s"


def test_the_renderers_failure_log_is_first_line_then_a_count(fake_audio, monkeypatch):
    playback, log, _requests, _now = _renderer(fake_audio, monkeypatch)
    clock = Clock(0.0)
    playback._open_log = ad.FailureLog(log.emit, 60.0, clock)
    for _ in range(100):
        assert playback._try_open(None) is False
        clock.t += 1.0
    failed = [line for line in log.lines if "output device open failed" in line]
    assert len(failed) == 2 and failed[0].endswith(STALE) and "repeated 60 more" in failed[1]
    playback._note_open_ok()
    assert any("before it recovered" in line for line in log.lines)


def test_a_renderer_that_cannot_open_at_start_still_counts_that_failure(fake_audio, monkeypatch):
    playback, _log, requests, now = _renderer(fake_audio, monkeypatch)
    playback._try_open(None)
    playback._note_open_failed(0.0)                    # what run() does after its first try
    playback._heal_output(1.0)                         # the first self-heal pass
    assert len(requests) == 1


def test_a_reinit_in_the_middle_of_a_sentence_keeps_the_rest_of_it(fake_audio):
    bus, _log = make_bus()
    playback = AudioPlaybackThread(bus)
    playback._stream = FakeStream(fake_audio.sd, "output", 0)
    sentence = bytes(range(256)) * 64                  # 16 KB
    playback._reinit_hold.set()
    playback._write(sentence, playback._generation)
    assert playback._stream.written == [], "it wrote into a stream about to be closed"
    assert playback._held_tail == sentence
    playback._stream.close()


def test_a_queued_device_switch_survives_the_renumbering(fake_audio):
    sd = fake_audio.sd
    bus, _log = make_bus()
    playback = AudioPlaybackThread(bus)
    playback._pending_device = 1                       # the monitor, as numbered now
    playback._reinit_hold.set()
    parked = threading.Thread(target=playback._park_for_reinit)
    parked.start()
    deadline = time.monotonic() + 3
    while not playback._reinit_closed.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert playback._reinit_closed.is_set()
    sd.devices = list(DEVICES_AFTER)                   # PortAudio renumbered
    playback._reinit_hold.clear()
    parked.join(3)
    assert playback._pending_device == 2               # still the monitor, by name


def test_the_renderer_thread_parks_for_a_reinit_and_reopens_on_the_fresh_list(fake_audio):
    """The real renderer thread, the real coordinator, a fake PortAudio."""
    sd = fake_audio.sd
    bus, log = make_bus()
    playback = AudioPlaybackThread(bus)
    coord = ad.PortAudioReinit(portaudio=lambda: sd)
    coord.register(playback)

    def renumber():
        sd.devices = list(DEVICES_AFTER)
        fake_audio.names["output"] = MONITOR           # Windows' default moved
    sd.after_initialize = renumber

    playback.start()
    try:
        _wait_for(lambda: sd.open_streams == 1)
        assert sd.opened[-1] == ("output", 0)
        assert coord.request_reinit("the default output changed to " + MONITOR, force=True)
        assert sd.calls == ["terminate", "initialize"]
        assert sd.violations == [], "terminated while the renderer's stream was open"
        _wait_for(lambda: sd.open_streams == 1 and len(sd.opened) == 2)
        assert sd.opened[-1] == ("output", 2), "it did not reopen on the monitor's new index"
        assert any("recovered" in line and MONITOR in line for line in log.lines)
    finally:
        playback.stop()
        playback.join(3)


def test_the_renderer_thread_rides_out_a_stale_list_until_the_reinit_fixes_it(fake_audio, monkeypatch):
    """20 Oct end to end: the open fails stale, the renderer asks once, the list is
    refreshed with its stream (there is none) closed, and sound comes back."""
    sd = fake_audio.sd
    bus, log = make_bus()
    sd.fail_open = PaError(STALE)
    playback = AudioPlaybackThread(bus)
    playback.REINIT_AFTER_FAILURES = 2
    coord = ad.PortAudioReinit(portaudio=lambda: sd, min_interval=0.0)
    coord.register(playback)
    monkeypatch.setattr(ad, "request_reinit_async",
                        lambda reason, force=False: coord.request_reinit_async(reason, force=force))
    sd.after_initialize = lambda: setattr(sd, "fail_open", None)    # fresh list: opens work
    playback.start()
    try:
        _wait_for(lambda: sd.open_streams == 1, timeout=8)
        assert sd.calls == ["terminate", "initialize"], "exactly one re-init"
        assert sd.violations == []
        assert sum("output device open failed" in l for l in log.lines) <= 2
        assert any("recovered" in line for line in log.lines)
    finally:
        playback.stop()
        playback.join(3)


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the audio thread")


# ── the microphone ────────────────────────────────────────────────────────────

def _gate():
    """What stop()/restart touch on the gate thread, without starting one."""
    return SimpleNamespace(note_signal=lambda: None, stop=lambda: None,
                           join=lambda timeout=None: None)


def _mic(fake_audio, *, stream=True):
    bus, log = make_bus()
    mic = MicrophoneEngine(None, None, bus, lambda: True, None)
    mic._gatekeeper = _gate()
    if stream:
        mic._stream = FakeStream(fake_audio.sd, "input", 2)
    return mic, log


def _unblock(sd):
    for stream in sd.streams:
        if stream.block_close is not None:
            stream.block_close.set()


def test_the_microphone_closes_waits_and_reopens_on_the_fresh_list(fake_audio):
    sd = fake_audio.sd
    mic, log = _mic(fake_audio)
    coord = ad.PortAudioReinit(portaudio=lambda: sd)
    coord.register(mic)
    assert coord.request_reinit("test", force=True)
    assert sd.calls == ["terminate", "initialize"] and sd.violations == []
    assert mic._stream is not None and sd.open_streams == 1
    assert mic._reinit_parked is False
    mic._stream.close()


def test_a_picked_microphone_is_found_again_by_name_not_by_its_old_index(fake_audio):
    sd = fake_audio.sd
    mic, log = _mic(fake_audio)
    mic._forced_input = 3                              # the webcam mic, as numbered now
    coord = ad.PortAudioReinit(portaudio=lambda: sd)
    coord.register(mic)
    sd.after_initialize = lambda: setattr(sd, "devices", list(DEVICES_AFTER))
    assert coord.request_reinit("test", force=True)
    assert mic._forced_input == 0, "it kept a stale index and would open a stranger"
    assert sd.opened[-1] == ("input", 0)
    mic._stream.close()


def test_a_picked_microphone_that_has_gone_falls_back_to_the_default(fake_audio):
    sd = fake_audio.sd
    mic, log = _mic(fake_audio)
    mic._forced_input = 2                              # the headset mic - about to vanish
    coord = ad.PortAudioReinit(portaudio=lambda: sd)
    coord.register(mic)
    sd.after_initialize = lambda: setattr(sd, "devices", list(DEVICES_AFTER))
    assert coord.request_reinit("test", force=True)
    assert mic._forced_input is None
    assert any("no longer present" in line for line in log.lines)
    assert sd.opened[-1] == ("input", None) and mic._stream is not None
    mic._stream.close()


def test_the_microphone_is_not_released_while_a_background_close_is_still_running(fake_audio):
    """stop(wait=False) hands the close to a thread: until it ends a stream
    exists, so PortAudio must not be terminated."""
    sd = fake_audio.sd
    mic, _log = _mic(fake_audio)
    mic._stream.block_close = threading.Event()        # a close stuck on a dead device
    try:
        mic.stop(wait=False)
        mic._gatekeeper = _gate()                      # as if start() ran again
        mic.reinit_release()
        time.sleep(0.1)
        assert mic.reinit_released() is False, "reported closed while a close was in flight"
        coord = ad.PortAudioReinit(portaudio=lambda: sd, close_timeout=0.3)
        coord.register(mic)
        assert coord.request_reinit("test", force=True) is False
        assert sd.calls == [], "terminated under a stream that was still closing"
    finally:
        _unblock(sd)                                   # the stuck close finishes
    audio_module.finish_pending_closes(3)
    mic.reinit_release()
    _wait_for(mic.reinit_released)                     # and now it can be released
    mic.reinit_restore()
    if mic._stream is not None:
        mic._stream.close()


def test_a_microphone_that_will_not_close_in_time_abandons_the_whole_reinit(fake_audio):
    sd = fake_audio.sd
    mic, mic_log = _mic(fake_audio)
    mic._stream.block_close = threading.Event()
    bus, log = make_bus()
    playback = AudioPlaybackThread(bus)
    coord = ad.PortAudioReinit(portaudio=lambda: sd, close_timeout=0.3)
    coord.register(mic)
    coord.register(playback)
    try:
        assert coord.request_reinit("test", force=True) is False
        assert sd.calls == [] and sd.violations == []
        assert mic._reinit_parked is False and not playback._reinit_hold.is_set(), (
            "an abandoned attempt left something parked")
        assert any("NOT re-initialised" in line and "the microphone" in line
                   for line in log.lines + mic_log.lines)
    finally:
        _unblock(sd)
    time.sleep(0.2)
    for stream in sd.streams:
        stream.close()


def test_a_microphone_change_asked_for_mid_reinit_waits_and_then_applies(fake_audio):
    sd = fake_audio.sd
    mic, log = _mic(fake_audio)
    mic.reinit_release()
    _wait_for(mic.reinit_released)
    opened_before = len(sd.opened)
    mic._forced_input = 3
    mic._restart_stream(3)                             # the user picked a mic right now
    assert len(sd.opened) == opened_before, "opened a stream while parked"
    assert any("applies once the audio devices have been refreshed" in l for l in log.lines)
    sd.devices = list(DEVICES_AFTER)                   # renumbered
    mic.reinit_restore()
    assert sd.opened[-1] == ("input", 0)
    mic._stream.close()


def test_a_microphone_stopped_while_parked_is_not_reopened(fake_audio):
    sd = fake_audio.sd
    mic, _log = _mic(fake_audio)
    mic.reinit_release()
    _wait_for(mic.reinit_released)
    mic.stop()
    opened = len(sd.opened)
    mic.reinit_restore()
    assert len(sd.opened) == opened and mic._stream is None


def test_the_microphone_start_defers_to_a_reinit_in_progress(fake_audio, monkeypatch):
    """start() opening a stream mid-terminate is the crash."""
    sd = fake_audio.sd
    mic, log = _mic(fake_audio, stream=False)
    mic._gatekeeper = None
    mic._reinit_parked = True
    monkeypatch.setattr(audio_module, "AudioGateThread",
                        lambda *a, **k: SimpleNamespace(start=lambda: None, set_interruptible=lambda _v: None,
                                                        note_signal=lambda: None))
    monkeypatch.setattr("orion_core.utils.capture_disabled", lambda: False)
    mic.start()
    assert sd.opened == [] and any("once the audio devices have been refreshed" in l for l in log.lines)


def test_a_microphone_that_cannot_open_on_a_stale_list_asks_for_one_reinit(fake_audio, monkeypatch):
    sd = fake_audio.sd
    sd.fail_open = PaError(STALE)
    mic, _log = _mic(fake_audio, stream=False)
    asked = []
    monkeypatch.setattr(ad, "request_reinit_async",
                        lambda reason, force=False: asked.append(reason) or True)
    mic._restart_stream(2)
    assert len(asked) == 1 and "microphone" in asked[0] and "stale" in asked[0]


def test_a_microphone_that_cannot_open_for_another_reason_asks_for_nothing(fake_audio, monkeypatch):
    sd = fake_audio.sd
    sd.fail_open = PaError("Invalid sample rate [PaErrorCode -9997]")
    mic, _log = _mic(fake_audio, stream=False)
    asked = []
    monkeypatch.setattr(ad, "request_reinit_async",
                        lambda reason, force=False: asked.append(reason) or True)
    mic._restart_stream(2)
    assert asked == []


def test_a_stale_microphone_open_is_repaired_by_the_reinit_it_asked_for(fake_audio, monkeypatch):
    """The mic-side twin of 20 Oct: the open fails stale, one re-init, and it listens."""
    sd = fake_audio.sd
    mic, _log = _mic(fake_audio, stream=False)
    coord = ad.PortAudioReinit(portaudio=lambda: sd, min_interval=0.0)
    coord.register(mic)
    sd.fail_open = PaError(STALE)
    # Synchronous here so the test is deterministic; in the app it is a thread.
    monkeypatch.setattr(ad, "request_reinit_async",
                        lambda reason, force=False: coord.request_reinit(reason, force=force))
    sd.after_initialize = lambda: setattr(sd, "fail_open", None)
    mic._restart_stream(2)
    assert sd.calls == ["terminate", "initialize"] and sd.violations == []
    assert mic._stream is not None and sd.open_streams == 1
    mic._stream.close()


# ── the held default and the renumbering helpers ──────────────────────────────

def test_a_held_default_outranks_windows_default_until_released(fake_audio):
    fake_audio.names["output"] = MONITOR
    assert ad.system_default_index("output") == 1
    ad.hold_default("output", HEADSET)
    assert ad.system_default_index("output") == 0
    ad.hold_default("output", None)
    assert ad.system_default_index("output") == 1


def test_snapshots_follow_a_device_through_a_renumbering(fake_audio):
    sd = fake_audio.sd
    snap = ad.device_snapshot(3)
    assert snap == (WEBCAM_MIC, 0)
    sd.devices = list(DEVICES_AFTER)
    assert ad.device_from_snapshot("input", snap) == 0
    assert ad.device_from_snapshot("output", snap) is None, "wrong direction"
    assert ad.device_from_snapshot("input", (HEADSET_MIC, 0)) is None, "it has gone"
    assert ad.device_snapshot(None) is None and ad.device_snapshot(99) is None
