"""A goodbye from the MODEL arrives later than one from the local voice.

`speak_text` queues instantly, so a short fixed grace was enough to see
`is_busy()` go true before polling for it to go false. Routing the farewell
through the live session broke that assumption: a live turn takes a second or
three to come back, so the first poll finds silence, concludes he has already
finished, and fires the shutdown before he has said a word.

That is the clipped goodbye all over again, reintroduced by the fix for it —
which is why the wait now covers ORION STARTING as well as finishing.

Offline: a fake pipeline on a real clock, no audio device.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.live_worker import GenAILiveWorker  # noqa: E402


class _Pipeline:
    """Silent for `latency` seconds, then busy for `duration`."""

    def __init__(self, latency: float, duration: float) -> None:
        self.started_at = time.monotonic()
        self.latency = latency
        self.duration = duration

    def is_busy(self) -> bool:
        elapsed = time.monotonic() - self.started_at
        return self.latency <= elapsed < self.latency + self.duration


class _NeverStopping:
    @staticmethod
    def is_set() -> bool:
        return False


def _worker(pipeline) -> GenAILiveWorker:
    worker = object.__new__(GenAILiveWorker)
    worker.speech = pipeline
    worker.stop_event = _NeverStopping()
    return worker


@pytest.mark.parametrize("latency,duration", [
    (0.05, 0.4),     # local voice: effectively instant
    (0.60, 0.5),     # a quick live turn
    (1.20, 0.6),     # a slow one
])
def test_he_is_heard_in_full_however_late_the_words_arrive(latency, duration):
    pipeline = _Pipeline(latency, duration)
    worker = _worker(pipeline)
    started = time.monotonic()
    asyncio.run(GenAILiveWorker._await_speech(
        worker, start_limit=4.0, end_limit=6.0, tail=0.1))
    waited = time.monotonic() - started
    assert waited >= latency + duration, (
        f"returned after {waited:.2f}s but he was still speaking until "
        f"{latency + duration:.2f}s")


def test_silence_does_not_hold_shutdown_open():
    """If he never speaks at all, the wait must still end quickly — a bounded
    start window, not the full ceiling."""

    class _Silent:
        @staticmethod
        def is_busy() -> bool:
            return False

    started = time.monotonic()
    asyncio.run(GenAILiveWorker._await_speech(
        _worker(_Silent()), start_limit=0.5, end_limit=6.0, tail=0.05))
    assert time.monotonic() - started < 2.0


def test_a_broken_pipeline_never_stalls_the_shutdown():
    class _Broken:
        @staticmethod
        def is_busy() -> bool:
            raise RuntimeError("device gone")

    started = time.monotonic()
    asyncio.run(GenAILiveWorker._await_speech(
        _worker(_Broken()), start_limit=5.0, end_limit=5.0))
    assert time.monotonic() - started < 1.0


def test_a_wedged_device_cannot_hold_it_open_forever():
    """Both halves are bounded; a pipeline stuck busy must still let go."""

    class _Stuck:
        @staticmethod
        def is_busy() -> bool:
            return True

    started = time.monotonic()
    asyncio.run(GenAILiveWorker._await_speech(
        _worker(_Stuck()), start_limit=0.2, end_limit=0.5, tail=0.05))
    assert time.monotonic() - started < 2.0


def test_the_shutdown_path_uses_it():
    """A helper the shutdown does not call fixes nothing."""
    source = inspect.getsource(GenAILiveWorker._shutdown_after_farewell)
    assert "_await_speech" in source
    assert "asyncio.sleep(0.6)" not in source, "the old fixed grace is back"
