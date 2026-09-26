"""
Quiet mode: "go on standby" stops ORION VOLUNTEERING without deafening him.

Two failures matter more than anything else here and both are pinned below:

  * "stand by" must never be mistaken for "shut down" — the phrase is used to
    ask for silence, and resolving it as a power command would close ORION
    instead of quieting him.
  * "stop being quiet" contains "be quiet". If the ON pattern won that race
    the user would be trapped in silence by the very phrase meant to end it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.live_worker import GenAILiveWorker  # noqa: E402


class _Signal:
    def __init__(self): self.messages = []
    def emit(self, *a): self.messages.append(a[0] if len(a) == 1 else a)
    def connect(self, *a): pass


class _StubBus:
    def __init__(self):
        self.log = _Signal()

    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _worker():
    w = types.SimpleNamespace()
    w.bus = _StubBus()
    w.spoken = []
    w._say = w.spoken.append
    w.quiet_mode = False
    w.paused = False
    w.stop_event = types.SimpleNamespace(is_set=lambda: False)
    w._QUIET_ON_RE = GenAILiveWorker._QUIET_ON_RE
    w._QUIET_OFF_RE = GenAILiveWorker._QUIET_OFF_RE
    # Mute ("mute yourself") is resolved first, inside the same handler; the
    # real matcher is used so a quiet phrase it swallowed would fail here.
    w._MUTE_ON_RE = GenAILiveWorker._MUTE_ON_RE
    w._MUTE_OFF_RE = GenAILiveWorker._MUTE_OFF_RE
    w.muted = []
    w.set_voice_muted = lambda muted, announce=True: w.muted.append(muted)
    w._handle_mute_command = (
        lambda t, announce=True: GenAILiveWorker._handle_mute_command(w, t, announce))
    return w


def _quiet(w, text: str) -> bool:
    return GenAILiveWorker._handle_quiet_command(w, text)


# ── entering ─────────────────────────────────────────────────────────────────

def test_go_on_standby_enters_quiet_mode():
    w = _worker()
    assert _quiet(w, "orion go on standby") is True
    assert w.quiet_mode is True


def test_the_usual_phrasings_all_work():
    for phrase in ("go on standby", "standby mode", "stand by",
                   "be quiet", "stay quiet", "be silent",
                   "don't speak", "do not interrupt", "quiet mode",
                   "leave me alone", "give me a minute",
                   "give me some quiet", "silence yourself"):
        w = _worker()
        assert _quiet(w, phrase) is True, phrase
        assert w.quiet_mode is True, phrase


def test_entering_is_acknowledged_once():
    """Saying nothing at all would leave the user unsure he heard."""
    w = _worker()
    _quiet(w, "go on standby")
    assert len(w.spoken) == 1


# ── leaving ──────────────────────────────────────────────────────────────────

def test_stop_being_quiet_leaves_rather_than_re_entering():
    """It contains 'be quiet' — the trap this ordering exists to avoid."""
    w = _worker()
    w.quiet_mode = True
    assert _quiet(w, "stop being quiet") is True
    assert w.quiet_mode is False


def test_the_usual_release_phrasings_all_work():
    for phrase in ("wake up", "come back", "i'm back", "resume", "carry on",
                   "you can speak again", "stop being quiet",
                   "exit standby", "cancel quiet", "talk to me again"):
        w = _worker()
        w.quiet_mode = True
        assert _quiet(w, phrase) is True, phrase
        assert w.quiet_mode is False, phrase


def test_leaving_is_acknowledged():
    w = _worker()
    w.quiet_mode = True
    _quiet(w, "wake up")
    assert len(w.spoken) == 1


# ── it must not fire on ordinary conversation ────────────────────────────────

def test_ordinary_sentences_are_not_quiet_commands():
    for phrase in ("it's quiet in here", "the quiet part of town",
                   "what does standby power mean", "i went quiet for a while",
                   "tell me about silence", "read me the news"):
        w = _worker()
        assert _quiet(w, phrase) is False, phrase
        assert w.quiet_mode is False, phrase


# ── the critical interaction with power commands ─────────────────────────────

def test_stand_by_does_not_shut_orion_down():
    """The whole reason quiet mode is resolved before the power classifier."""
    import asyncio

    w = _worker()
    w.dispatcher = types.SimpleNamespace(system_guard=None)
    w._handle_quiet_command = lambda t: GenAILiveWorker._handle_quiet_command(w, t)
    w.request_shutdown = _Signal()
    handled = asyncio.run(GenAILiveWorker._handle_power_command(w, "stand by"))
    assert handled is True
    assert w.quiet_mode is True
    # nothing was asked to shut down
    assert not any("shut" in str(m).lower() for m in w.spoken)


# ── the gate on unprompted speech ────────────────────────────────────────────

def _announce(w, text: str) -> None:
    GenAILiveWorker.announce(w, text)


def _announcer():
    w = _worker()
    w.session = None
    w.connected = False
    w._normalise_phrase = lambda t: t.lower()
    w._last_announced_norm = ""
    w._last_announced_at = 0.0
    w.ANNOUNCE_DEDUP_SECONDS = 30
    return w


def test_unprompted_speech_is_suppressed_while_quiet():
    w = _announcer()
    w.quiet_mode = True
    _announce(w, "A reminder: stand up.")
    assert w.spoken == []
    assert any("held back" in str(m) for m in w.bus.log.messages)


def test_unprompted_speech_flows_again_once_released():
    w = _announcer()
    w.quiet_mode = True
    _announce(w, "A reminder: stand up.")
    w.quiet_mode = False
    _announce(w, "A reminder: stand up.")
    assert any("proactive" in str(m) for m in w.bus.log.messages)


def test_quiet_mode_does_not_suppress_while_off():
    w = _announcer()
    _announce(w, "Security note: something happened.")
    assert not any("held back" in str(m) for m in w.bus.log.messages)


def test_proactive_notice_is_spoken_directly_even_with_live_session():
    """A notice must never become a model turn that it can merely acknowledge."""
    w = _announcer()
    w.session = object()
    w.connected = True
    _announce(w, "The front door is open.")
    assert w.spoken == ["The front door is open."]


def test_pausing_still_silences_independently():
    """Quiet mode is additional to pause, not a replacement for it."""
    w = _announcer()
    w.paused = True
    _announce(w, "anything")
    assert w.spoken == []
