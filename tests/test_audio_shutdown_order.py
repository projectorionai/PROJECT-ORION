"""Silence the voice before closing the device it speaks through.

ORION's frozen build died intermittently with STATUS_HEAP_CORRUPTION
(0xC0000374). `dist/ORION/config/crash_reports/faulthandler.log` named the two
threads involved, on 2026-09-18 and again on 2026-09-20:

    Current thread:  sounddevice.py:1167 in close     (audio.py, renderer)
    Thread:          pyttsx3/drivers/sapi5.py:179 in startLoop

SpeechQueueManager.stop() ran its two threads down in START order — playback
first, then TTS — so the renderer closed its PortAudio stream while the speech
thread could still be inside runAndWait() driving it.

Both stop() methods only set a flag, so reversing the order alone would not
have been enough: teardown would still race them. Each is now joined, and the
join is bounded so a wedged audio device cannot hang shutdown.

Offline: pure source and a dummy thread, no audio device.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.audio import SpeechQueueManager, _join_bounded  # noqa: E402


def _body() -> str:
    source = inspect.getsource(SpeechQueueManager.stop)
    return source.split('"""')[-1]          # past the docstring


def test_the_voice_is_stopped_before_the_device_is_closed():
    body = _body()
    assert body.index("self.tts.stop()") < body.index("self.playback.stop()"), (
        "playback closes the PortAudio stream; stopping it first leaves SAPI5 "
        "speaking into a stream being torn down underneath it")


def test_each_thread_is_actually_waited_for():
    """stop() only sets a flag on both threads. Without a join, shutdown
    continues while they are still in native code — which is the race."""
    body = _body()
    assert body.count("_join_bounded") == 2
    assert body.index("_join_bounded(self.tts") < body.index("_join_bounded(self.playback")


def test_the_wait_is_bounded():
    """A wedged audio device must not be able to hang shutdown. app.py's
    watchdog is the last resort behind this, and it force-exits."""
    assert 0 < SpeechQueueManager.STOP_JOIN_SECONDS <= 10


def test_join_reports_whether_the_thread_really_stopped():
    slow = threading.Thread(target=lambda: time.sleep(1.5), daemon=True)
    slow.start()
    assert _join_bounded(slow, 0.05) is False, "a still-running thread is not stopped"
    assert _join_bounded(slow, 3.0) is True


def test_join_survives_the_awkward_cases():
    """Shutdown is the worst place for a new exception, and a thread joining
    itself would wait forever."""
    assert _join_bounded(None, 1.0) is False

    class _NotAThread:
        def is_alive(self):
            raise RuntimeError("no")

    assert _join_bounded(_NotAThread(), 1.0) is False

    seen: list = []

    def _self_join(holder):
        seen.append(_join_bounded(holder[0], 5.0))

    holder: list = []
    thread = threading.Thread(target=lambda: _self_join(holder), daemon=True)
    holder.append(thread)
    thread.start()
    thread.join(timeout=3.0)
    assert not thread.is_alive(), "a thread joining itself deadlocked"
    assert seen == [False]
