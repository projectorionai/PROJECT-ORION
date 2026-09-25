"""
Unprompted speech has to earn itself.

The complaint: "ORION speaks regarding processes which start up when I've said
in the past I don't want it announcing things which AREN'T dangerous or affect
me in any way." And: "he keeps pestering me when I'm there and I am busy."

These tests pin both — the classification, and the FOCUS state that is quiet
without being deaf.
"""

from __future__ import annotations

import pytest

from orion_core.proactive_policy import (
    URGENCY_BY_KIND, Attention, ProactivePolicy, Urgency,
)


@pytest.fixture
def policy():
    return ProactivePolicy()


# ── the actual complaint ──────────────────────────────────────────────────────

def test_a_process_starting_is_never_spoken(policy):
    """This is the exact narration the user asked to stop."""
    decision = policy.should_speak("process_started")
    assert decision.speak is False
    assert decision.urgency is Urgency.AMBIENT
    assert "does not affect you" in decision.reason


def test_a_process_starting_is_still_recorded(policy):
    policy.should_speak("process_started")
    policy.should_speak("process_started")
    assert policy.suppressed["process_started"] == 2, (
        "silent must not mean invisible — it still has to be inspectable")


def test_orions_own_model_server_would_not_be_narrated(policy):
    """llama-server.exe is what actually triggered it."""
    assert policy.should_speak("process_started").speak is False


@pytest.mark.parametrize("kind", [
    "process_started", "process_ended", "cpu_high", "ram_high",
    "feed_recovered", "provider_switched", "tool_forged", "self_improvement",
    "index_built", "presence_check", "focus_block", "network_mode",
])
def test_every_ambient_kind_stays_silent_in_every_state(policy, kind):
    for state in (Attention.OPEN, Attention.FOCUS, Attention.STANDBY):
        policy.attention = state
        assert policy.should_speak(kind).speak is False, (kind, state)


# ── things that DO earn speaking ──────────────────────────────────────────────

@pytest.mark.parametrize("kind", [
    "battery_critical", "disk_full", "security_alert", "breach",
    "emergency", "audio_failure", "crash",
])
def test_critical_things_are_spoken_even_when_busy(policy, kind):
    policy.enter_focus()
    assert policy.should_speak(kind).speak is True, kind


def test_critical_things_are_spoken_even_in_standby(policy):
    policy.set_attention(Attention.STANDBY)
    assert policy.should_speak("battery_critical").speak is True


def test_actionable_things_are_spoken_when_the_user_is_available(policy):
    assert policy.should_speak("reminder").speak is True
    assert policy.should_speak("protocol").speak is True


def test_actionable_things_are_held_while_busy(policy):
    policy.enter_focus()
    decision = policy.should_speak("reminder")
    assert decision.speak is False
    assert "busy" in decision.reason


def test_an_unlisted_kind_defaults_to_actionable_not_ambient(policy):
    """A new announcement must not become unspeakable by accident."""
    assert policy.classify("something_brand_new") is Urgency.ACTIONABLE
    assert policy.should_speak("something_brand_new").speak is True


# ── FOCUS: quiet, but not deaf ────────────────────────────────────────────────

def test_focus_is_a_different_state_from_standby(policy):
    policy.enter_focus()
    assert policy.attention is Attention.FOCUS
    assert policy.attention is not Attention.STANDBY


def test_speaking_to_orion_releases_an_auto_entered_focus(policy):
    for _ in range(policy.IGNORED_BEFORE_FOCUS):
        policy.note_ignored()
    assert policy.attention is Attention.FOCUS

    policy.user_spoke()
    assert policy.attention is Attention.OPEN, (
        "talking to him proves you are available")


def test_speaking_does_not_release_a_focus_the_user_asked_for(policy):
    """If they SAID they were busy, one word to him should not undo it."""
    policy.enter_focus("user asked")
    policy.user_spoke()
    assert policy.attention is Attention.FOCUS


def test_repeatedly_ignoring_him_makes_him_take_the_hint(policy):
    assert policy.attention is Attention.OPEN
    for _ in range(policy.IGNORED_BEFORE_FOCUS - 1):
        policy.note_ignored()
    assert policy.attention is Attention.OPEN, "not on the first snub"
    policy.note_ignored()
    assert policy.attention is Attention.FOCUS


def test_auto_focus_expires_so_he_is_not_mute_for_ever(policy, monkeypatch):
    import orion_core.proactive_policy as mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    for _ in range(policy.IGNORED_BEFORE_FOCUS):
        policy.note_ignored()
    assert policy.attention is Attention.FOCUS

    clock["t"] += policy.AUTO_FOCUS_SECONDS + 1
    policy.should_speak("reminder")          # any decision re-checks expiry
    assert policy.attention is Attention.OPEN


def test_ignoring_him_while_already_quiet_changes_nothing(policy):
    policy.enter_focus()
    for _ in range(10):
        policy.note_ignored()
    assert policy.attention is Attention.FOCUS


# ── reporting ─────────────────────────────────────────────────────────────────

def test_the_report_says_what_was_held_back(policy):
    policy.should_speak("process_started")
    policy.should_speak("cpu_high")
    text = policy.report()
    assert "process_started" in text
    assert "AMBIENT" in text


def test_the_report_is_honest_when_nothing_was_held(policy):
    assert "Nothing has been held back" in policy.report()


def test_describe_exposes_the_state(policy):
    policy.enter_focus()
    described = policy.describe()
    assert described["attention"] == "FOCUS"
    assert "suppressed" in described


# ── the sentinel actually uses it ─────────────────────────────────────────────

def test_the_sentinel_routes_alerts_through_the_policy():
    import inspect

    from orion_core import sentinel

    source = inspect.getsource(sentinel.SentinelAgent._alert)
    assert "POLICY.should_speak" in source, (
        "the sentinel must not reach for the voice channel directly")


def test_the_process_alert_is_classified_ambient():
    import inspect

    from orion_core import sentinel

    source = inspect.getsource(sentinel.SentinelAgent._check_new_processes)
    assert 'kind="process_started"' in source


def test_battery_critical_is_still_classified_critical():
    assert URGENCY_BY_KIND["battery_critical"] is Urgency.CRITICAL


# ── the voice commands ────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase", [
    "i'm busy", "i am busy right now", "i'm concentrating", "focus mode",
    "do not disturb", "don't bother me", "no interruptions",
    "i'm in the middle of something", "hold your thoughts",
])
def test_busy_phrases_enter_focus(phrase):
    from orion_core.live_worker import GenAILiveWorker

    assert GenAILiveWorker._FOCUS_ON_RE.search(phrase), phrase


@pytest.mark.parametrize("phrase", [
    "i'm done", "i'm free now", "exit focus mode", "you can talk again",
    "what did i miss",
])
def test_release_phrases_leave_focus(phrase):
    from orion_core.live_worker import GenAILiveWorker

    assert GenAILiveWorker._FOCUS_OFF_RE.search(phrase), phrase


def test_ordinary_speech_is_not_a_focus_command():
    from orion_core.live_worker import GenAILiveWorker

    for phrase in ("what's the weather", "open my email",
                   "how busy is the processor"):
        assert not GenAILiveWorker._FOCUS_ON_RE.search(phrase), phrase


def test_the_presence_check_is_gated():
    import inspect

    from orion_core.live_worker import GenAILiveWorker

    source = inspect.getsource(GenAILiveWorker._begin_presence_check)
    assert 'should_speak("presence_check")' in source, (
        "'are you still there?' is the pestering the user objected to")
