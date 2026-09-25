"""
The one channel no "be quiet" can close.

  "If there are things that are damaging, concerning to my PC, spilling of
   private data, anything that harms human / humans he must inform me directly
   with voice and this is predominantly important even if he is on standby and
   it overrides everything so he MUST communicate these things to me."

Before this there was no such channel. ``announce()`` — the only path for
unprompted speech — returns early on ``paused`` and on ``quiet_mode``, and
correctly so: being told to be quiet and then still being talked at is the
failure it exists to prevent. But that meant a genuine danger notice was
dropped by exactly the same code, silently, with the user believing ORION was
watching.

The fix is a separate route rather than a flag on the existing one, because
danger is not a louder reminder. ``alert()`` ignores standby, quiet and pause;
everything else still obeys them.

It is deliberately narrow. It does not wake ORION, does not resume the
conversation and does not reopen the microphone — it says the thing and leaves
the user in the state they asked for.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.live_worker import GenAILiveWorker  # noqa: E402


@pytest.fixture
def worker():
    w = GenAILiveWorker.__new__(GenAILiveWorker)
    w.spoken: list[str] = []
    w.logs: list[str] = []
    w._say = w.spoken.append
    w._normalise_phrase = lambda text: str(text).lower().strip()
    w.bus = types.SimpleNamespace(
        log=types.SimpleNamespace(emit=w.logs.append),
        dashboard_event=types.SimpleNamespace(emit=lambda *a: None),
    )
    w.standby_mode = False
    w.quiet_mode = False
    w.paused = False
    w.stop_event = types.SimpleNamespace(is_set=lambda: False)
    w._last_alert_norm = ""
    w._last_alert_at = 0.0
    return w


DANGER = "A process is reading your saved passwords."


# ── it overrides every silencing state ───────────────────────────────────────

@pytest.mark.parametrize("state", ["standby_mode", "quiet_mode", "paused"])
def test_an_alert_is_spoken_in_every_silenced_state(worker, state):
    setattr(worker, state, True)
    assert worker.alert(DANGER) is True
    assert worker.spoken == [DANGER], f"{state} swallowed a safety alert"


def test_an_alert_survives_all_three_at_once(worker):
    worker.standby_mode = worker.quiet_mode = worker.paused = True
    assert worker.alert(DANGER, reason="data exfiltration") is True
    assert worker.spoken == [DANGER]


def test_the_log_records_what_it_overrode(worker):
    """So the override is auditable — a channel that ignores the user's stated
    wishes has to leave a trace saying it did."""
    worker.standby_mode = worker.quiet_mode = True
    worker.alert(DANGER, reason="credential access")
    line = worker.logs[0]
    assert line.startswith("SAFETY")
    assert "standby" in line and "quiet" in line
    assert "credential access" in line


def test_an_ordinary_state_needs_no_override_note(worker):
    worker.alert(DANGER)
    assert "overriding" not in worker.logs[0]


# ── it does not become a siren ───────────────────────────────────────────────

def test_an_immediate_repeat_is_suppressed(worker):
    """A stuck sensor must not turn the override channel into an alarm that
    cannot be silenced — which would make the whole feature something the user
    disables."""
    assert worker.alert(DANGER) is True
    assert worker.alert(DANGER) is False
    assert worker.spoken == [DANGER]


def test_a_different_danger_is_always_spoken(worker):
    worker.alert(DANGER)
    assert worker.alert("Your disk is failing its SMART checks.") is True
    assert len(worker.spoken) == 2


def test_the_same_danger_is_spoken_again_much_later(worker):
    worker.alert(DANGER)
    worker._last_alert_at = time.monotonic() - (worker.ALERT_DEDUP_SECONDS + 5)
    assert worker.alert(DANGER) is True


def test_the_alert_window_is_far_longer_than_the_chatty_one(worker):
    """These are rare by construction; reminders are not."""
    assert worker.ALERT_DEDUP_SECONDS > GenAILiveWorker.ANNOUNCE_DEDUP_SECONDS * 5


def test_an_empty_alert_does_nothing(worker):
    assert worker.alert("") is False
    assert worker.alert("   ") is False
    assert worker.spoken == []


# ── it stays narrow ──────────────────────────────────────────────────────────

def test_alerting_does_not_wake_him(worker):
    """He warns you and leaves you in the state you asked for. Waking on every
    alert would make 'go on standby' meaningless."""
    worker.standby_mode = worker.quiet_mode = worker.paused = True
    worker.alert(DANGER)
    assert worker.standby_mode is True
    assert worker.quiet_mode is True
    assert worker.paused is True


def test_a_broken_dashboard_does_not_lose_the_warning(worker):
    """The voice is the requirement; the UI event is a nicety."""
    worker.bus.dashboard_event = types.SimpleNamespace(
        emit=lambda *a: (_ for _ in ()).throw(RuntimeError("bus deleted")))
    assert worker.alert(DANGER) is True
    assert worker.spoken == [DANGER]


# ── ordinary announcements are still correctly silenced ──────────────────────

def test_quiet_mode_still_silences_ordinary_announcements(worker):
    """The override must not blunt the thing it sits beside — otherwise 'be
    quiet' stops meaning anything."""
    worker.quiet_mode = True
    worker.announce("a routine reminder")
    assert worker.spoken == []


def test_pause_still_silences_ordinary_announcements(worker):
    worker.paused = True
    worker.announce("a routine reminder")
    assert worker.spoken == []


# ── the wire is actually connected ───────────────────────────────────────────
#
# A channel nothing calls is not a safety feature. The full chain was:
#   sentinel -> POLICY.should_speak(CRITICAL) -> "spoken regardless of state"
#            -> bus.speak_request -> announce() -> DROPPED on quiet_mode.
# The policy was right and the delivery layer silently disagreed with it.

def test_the_bus_carries_danger_on_its_own_signal():
    from orion_core.bus import OrionBus

    assert hasattr(OrionBus, "safety_alert")


def test_the_worker_listens_on_it():
    import inspect

    import orion_core.live_worker as module

    assert "self.bus.safety_alert.connect(self.alert)" in inspect.getsource(module)


def test_the_sentinel_sends_critical_findings_down_it():
    import inspect

    import orion_core.sentinel as module

    source = inspect.getsource(module)
    assert "safety_alert.emit" in source
    assert "Urgency.CRITICAL" in source


def test_non_critical_findings_still_use_the_ordinary_channel():
    """Otherwise every routine notice would override 'be quiet', and the user
    would turn the whole thing off."""
    import inspect

    import orion_core.sentinel as module

    source = inspect.getsource(module)
    assert "self.bus.speak_request.emit(spoken)" in source


def test_the_policy_still_rules_critical_speakable_in_standby():
    from orion_core.proactive_policy import Attention, ProactivePolicy, Urgency

    policy = ProactivePolicy()
    policy.set_attention(Attention.STANDBY, "test")
    decision = policy.should_speak("disk_failure", Urgency.CRITICAL)
    assert decision.speak is True
    assert decision.urgency >= Urgency.CRITICAL
