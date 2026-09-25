"""ORION has to actually finish saying goodbye.

`app._await_farewell` polls `speech.is_busy()` and waits for it to go idle, so
the goodbye is not clipped. `worker.speech` is a `SpeechQueueManager`, and that
class had no `is_busy` — so the first poll raised AttributeError, the
function's own `except Exception: return` swallowed it, and it returned
immediately, every time.

Nothing failed visibly. The goodbye was simply cut off, and the SAPI5 speech
thread was left inside the engine while teardown carried on around it — which
is the same thread pair as the STATUS_HEAP_CORRUPTION dumps, and the reason
shutdown sometimes hung until the 25-second watchdog force-exited.

The lesson is the shape of the bug rather than the missing method: a caller
reaching for an attribute inside a bare `except` cannot tell "idle" from "does
not exist".

Offline: no QApplication, no audio device.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import app as orion_app  # noqa: E402
from orion_core.audio import SpeechQueueManager  # noqa: E402


def _polled_attributes(func) -> set[str]:
    """Every ``speech.<name>`` the function touches."""
    source = inspect.getsource(func)
    return set(re.findall(r"\bspeech\.(\w+)", source))


def test_everything_the_farewell_polls_actually_exists():
    """The whole defect in one assertion."""
    missing = [name for name in _polled_attributes(orion_app._await_farewell)
               if not hasattr(SpeechQueueManager, name)]
    assert not missing, (
        f"_await_farewell polls {missing} on SpeechQueueManager, which does "
        f"not have it — the bare except turns that into an instant return")


def test_is_busy_covers_queued_speech_not_just_active_output():
    """A farewell still waiting its turn has not been said yet, and the state
    machine's flag goes false between two queued utterances."""
    source = inspect.getsource(SpeechQueueManager.is_busy)
    assert "machine.is_active()" in source
    assert "self.tts" in source and "self.playback" in source


def test_is_busy_survives_a_source_that_cannot_answer():
    """Shutdown is the worst place for a new exception."""

    class _Machine:
        @staticmethod
        def is_active() -> bool:
            return False

    class _Hostile:
        def is_busy(self):
            raise RuntimeError("device gone")

    class _Old:
        """No is_busy, but the older is_active — must still be understood."""

        def is_active(self):
            return True

    manager = object.__new__(SpeechQueueManager)
    manager.machine = _Machine()
    manager.tts = _Hostile()
    manager.playback = _Old()
    assert manager.is_busy() is True          # the old-style source answered

    manager.playback = _Hostile()
    assert manager.is_busy() is False         # nothing usable: do not block


def test_the_farewell_wait_is_still_bounded():
    """Fixing the poll must not turn a clipped goodbye into a hung shutdown."""
    signature = inspect.signature(orion_app._await_farewell)
    limit = signature.parameters["limit"].default
    assert 0 < limit <= 30
    assert "deadline" in inspect.getsource(orion_app._await_farewell)
