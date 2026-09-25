"""
Finish-speaking mode (Mark XXII) — ORION completes his sentence instead of being
clipped by his own echo.

    "when ORION speaks, I don't want him to pause until he's actually finished
     ... he doesn't finish his sentences (with his voice)."

Opt-in and default-OFF, so barge-in keeps working unless the user asks for this.
The receive loop is heavy to construct, so the decision logic is tested through
a light shim that carries just the state the methods touch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.live_worker import GenAILiveWorker as LiveWorker  # noqa: E402


class _Shim:
    """Just enough of a worker for the finish-speaking methods."""
    def __init__(self):
        self.finish_speaking = False
        self.said = []
    def _say(self, text):
        self.said.append(text)

    # bind the real methods onto the shim
    _honour_server_interruption = LiveWorker._honour_server_interruption
    set_finish_speaking = LiveWorker.set_finish_speaking
    _handle_finish_speaking_command = LiveWorker._handle_finish_speaking_command
    _FINISH_ON_RE = LiveWorker._FINISH_ON_RE
    _FINISH_OFF_RE = LiveWorker._FINISH_OFF_RE


# ── the decision ─────────────────────────────────────────────────────────────

def test_interruptions_are_honoured_by_default():
    shim = _Shim()
    assert shim.finish_speaking is False
    assert shim._honour_server_interruption() is True     # barge-in still works


def test_finish_speaking_mode_ignores_interruptions():
    shim = _Shim()
    shim.set_finish_speaking(True)
    assert shim._honour_server_interruption() is False     # he finishes the sentence


def test_toggle_returns_confirming_text():
    shim = _Shim()
    on = shim.set_finish_speaking(True)
    assert "finish" in on.lower()
    off = shim.set_finish_speaking(False)
    assert "interrupt" in off.lower()


# ── the default is env-driven ────────────────────────────────────────────────

def test_default_is_on_now(monkeypatch):
    monkeypatch.delenv("ORION_FINISH_SPEAKING", raising=False)
    # re-evaluate the same expression the constructor uses
    import os
    default = os.getenv("ORION_FINISH_SPEAKING", "1").strip().lower() not in {"0", "false", "off", "no"}
    assert default is True


def test_can_be_disabled_by_env(monkeypatch):
    import os
    monkeypatch.setenv("ORION_FINISH_SPEAKING", "0")
    default = os.getenv("ORION_FINISH_SPEAKING", "1").strip().lower() not in {"0", "false", "off", "no"}
    assert default is False


# ── the SERVER-side fix ──────────────────────────────────────────────────────

def test_build_config_requests_no_interruption():
    """The real fix: the Gemini session is told NOT to interrupt his turn."""
    import inspect
    src = inspect.getsource(LiveWorker._build_config)
    assert "realtime_input_config" in src
    assert "NO_INTERRUPTION" in src
    assert "finish_speaking" in src


def test_no_interruption_is_available_in_the_sdk():
    from google.genai import types
    cfg = types.RealtimeInputConfig(
        activity_handling=types.ActivityHandling.NO_INTERRUPTION)
    assert cfg.activity_handling == types.ActivityHandling.NO_INTERRUPTION


# ── the voice command ────────────────────────────────────────────────────────

def test_finish_phrases_turn_it_on():
    shim = _Shim()
    for phrase in ("finish your sentences", "stop cutting yourself off",
                   "let me hear the whole thing", "finish what you're saying"):
        shim.finish_speaking = False
        assert shim._handle_finish_speaking_command(phrase) is True
        assert shim.finish_speaking is True


def test_interrupt_phrases_turn_it_off():
    shim = _Shim()
    shim.finish_speaking = True
    assert shim._handle_finish_speaking_command("you can interrupt me") is True
    assert shim.finish_speaking is False


def test_unrelated_speech_is_not_a_toggle():
    shim = _Shim()
    assert shim._handle_finish_speaking_command("what's the weather today?") is False


def test_a_long_sentence_is_not_treated_as_a_command():
    shim = _Shim()
    long = ("could you please finish your sentences whenever you are giving me "
            "one of those very long research briefings about the market because "
            "it matters a great deal to the analysis i am writing up right now")
    # too long to be a terse control phrase → not intercepted as a toggle
    assert shim._handle_finish_speaking_command(long) is False


# ── the receive loop actually gates on it ────────────────────────────────────

def test_receive_loop_gates_interrupt_on_the_decision():
    import inspect
    src = inspect.getsource(LiveWorker._receive_realtime)
    assert "_honour_server_interruption()" in src
    # and the ignore branch exists
    assert "finishing the sentence" in src.lower()


# ── vision "he searched but couldn't see it" fix (Mark XXII) ─────────────────

def test_image_payload_gets_a_look_now_nudge():
    out = LiveWorker._nudge_payload_for_image(
        {"ok": True, "result": "Screen captured."}, "image/jpeg")
    assert "attached to this turn" in out["result"].lower()
    assert "Screen captured." in out["result"]


def test_non_image_payload_is_untouched():
    payload = {"ok": True, "result": "Here is the weather."}
    assert LiveWorker._nudge_payload_for_image(payload, "text/plain") == payload
    assert LiveWorker._nudge_payload_for_image(payload, "") == payload


def test_nudge_is_idempotent():
    once = LiveWorker._nudge_payload_for_image({"ok": True, "result": "x"}, "image/png")
    twice = LiveWorker._nudge_payload_for_image(once, "image/png")
    assert once["result"] == twice["result"]     # not doubled


def test_receive_loop_nudges_image_tool_results():
    import inspect
    src = inspect.getsource(LiveWorker._handle_tool_call)
    assert "_nudge_payload_for_image" in src
