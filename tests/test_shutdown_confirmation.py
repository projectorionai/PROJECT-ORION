"""
Confirming a shutdown, and saying goodbye properly before going.

  "When ORION shuts down I want him to confirm the shut down with me and see if
   I meant it and then if I confirm, before shutting down he should say 'okay I
   will see you soon sir' or something then proceed with the shutdown, let him
   say goodbye first and he should variate his goodbyes."

Four things were wrong, and only the first is the obvious one.

He acted on the FIRST mention of "shut down" — a phrase that turns up inside
ordinary sentences, and one an open microphone can catch mid-conversation. The
cost is asymmetric: a needless confirmation costs two seconds, a wrong shutdown
costs the session.

He then slept a flat 2.4 s and fired. Because that path sets _farewell_spoken,
app.py's own _await_farewell is skipped, so nothing downstream was waiting
either — the goodbye was cut off mid-sentence by the shutdown it was announcing.

The farewell pool held three lines, one of which was unreachable outside
late-night hours. A fixed sign-off is the most machine-like thing an assistant
can do, because it is the last thing you hear every single time.

And `asyncio.create_task(_do_shutdown())` discarded its handle — the same
ownership bug fixed elsewhere in this codebase, except here the task that could
be collected mid-flight is the shutdown itself.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.live_worker as live_worker_module  # noqa: E402
from orion_core.live_worker import GenAILiveWorker  # noqa: E402


class _Harness:
    def __init__(self, worker, said, fired):
        self.worker = worker
        self.said = said
        self.fired = fired

    @property
    def last(self) -> str:
        return self.said[-1] if self.said else ""


@pytest.fixture
def h():
    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    said: list[str] = []
    fired: list[str] = []
    worker._say = said.append
    worker.bus = types.SimpleNamespace(
        log=types.SimpleNamespace(emit=lambda m: None),
        request_shutdown=types.SimpleNamespace(emit=lambda: fired.append("SHUTDOWN")),
        request_restart=types.SimpleNamespace(emit=lambda: fired.append("RESTART")),
    )
    worker.dispatcher = types.SimpleNamespace(identity=None)
    worker.speech = types.SimpleNamespace(is_busy=lambda: False)
    worker._pending_shutdown = None
    worker._last_farewell = ""
    worker._farewell_spoken = False
    return _Harness(worker, said, fired)


# ── he asks first ────────────────────────────────────────────────────────────

def test_shutdown_is_not_acted_on_immediately(h):
    """The defect: the first mention of 'shut down' powered the machine off."""
    h.worker._ask_to_confirm_shutdown()
    assert h.fired == [], "shut down without asking"
    assert "?" in h.last, f"did not ask a question: {h.last!r}"


def test_the_question_names_what_will_happen(h):
    h.worker._ask_to_confirm_shutdown()
    assert "shut down" in h.last.lower()


def test_a_restart_asks_about_restarting_not_shutting_down(h):
    h.worker._ask_to_confirm_shutdown(is_restart=True)
    assert "restart" in h.last.lower()
    assert "shut down" not in h.last.lower()


# ── answering it ─────────────────────────────────────────────────────────────

def test_yes_proceeds_with_a_goodbye(h):
    h.worker._ask_to_confirm_shutdown()
    assert h.worker._resolve_shutdown_confirmation("yes please") is True
    assert h.worker._farewell_spoken is True
    assert h.last.startswith("Very good."), h.last


def test_no_stands_down_and_says_so(h):
    h.worker._ask_to_confirm_shutdown()
    assert h.worker._resolve_shutdown_confirmation("no, not now") is True
    assert h.fired == [], "shut down after being told no"
    assert h.worker._pending_shutdown is None
    assert h.last, "declining silently is indistinguishable from not hearing"


def test_an_unrelated_reply_is_not_swallowed(h):
    """A pending confirmation must not eat the next ordinary request — the
    user asking about the weather has not agreed to anything."""
    h.worker._ask_to_confirm_shutdown()
    assert h.worker._resolve_shutdown_confirmation("what is the weather") is False
    assert h.fired == []


def test_saying_it_twice_counts_as_meaning_it(h):
    """Repeating the command after being asked has answered the question;
    demanding the word 'yes' would be pedantic."""
    h.worker._ask_to_confirm_shutdown()
    h.worker._ask_to_confirm_shutdown()
    assert h.worker._farewell_spoken is True
    assert h.last.startswith("Very good.")


def test_a_stale_yes_cannot_power_the_machine_down(h):
    """A 'yes' ten minutes later, in an unrelated conversation, is not consent
    to a question the user has long forgotten asking."""
    h.worker._ask_to_confirm_shutdown()
    h.worker._pending_shutdown["at"] = time.monotonic() - 9999
    assert h.worker._resolve_shutdown_confirmation("yes") is False
    assert h.fired == []


def test_an_expired_window_lapses_silently(h):
    """Announcing 'I won't shut down then' for a question the user moved on
    from is noise."""
    h.worker._ask_to_confirm_shutdown()
    before = len(h.said)
    h.worker._pending_shutdown["at"] = time.monotonic() - 9999
    h.worker._shutdown_confirmation_open()
    assert len(h.said) == before


def test_a_yes_with_no_question_pending_does_nothing(h):
    assert h.worker._resolve_shutdown_confirmation("yes") is False
    assert h.fired == []


# ── the goodbye actually finishes ────────────────────────────────────────────

def test_shutdown_waits_for_the_goodbye_to_finish(h):
    """The 2.4 s guess cut the longer sign-offs off mid-sentence — and because
    this path sets _farewell_spoken, app.py's _await_farewell is skipped, so
    nothing downstream was waiting either."""
    calls = {"n": 0}

    def _busy():
        calls["n"] += 1
        return calls["n"] < 4          # busy for the first few polls

    h.worker.speech = types.SimpleNamespace(is_busy=_busy)
    asyncio.run(h.worker._shutdown_after_farewell(restart=False))
    assert calls["n"] >= 4, "did not wait for speech to finish"
    assert h.fired == ["SHUTDOWN"]


def test_a_wedged_audio_device_cannot_block_shutdown_forever(h):
    """Bounded wait: never speaking again must not mean never shutting down."""
    h.worker.speech = types.SimpleNamespace(is_busy=lambda: (_ for _ in ()).throw(
        RuntimeError("audio device gone")))
    asyncio.run(h.worker._shutdown_after_farewell(restart=False))
    assert h.fired == ["SHUTDOWN"]


def test_restart_emits_restart_not_shutdown(h):
    asyncio.run(h.worker._shutdown_after_farewell(restart=True))
    assert h.fired == ["RESTART"]


def test_the_shutdown_task_is_owned(h):
    """`asyncio.create_task(_do_shutdown())` discarded its handle. The loop
    keeps only a weak reference, so the task that could vanish mid-flight was
    the shutdown itself."""
    import inspect
    source = (inspect.getsource(GenAILiveWorker._begin_shutdown)
              + inspect.getsource(GenAILiveWorker._farewell_then_go))
    assert "background.spawn" in source
    assert "asyncio.create_task" not in source


def test_shutdown_still_happens_without_a_running_loop(h):
    """The goodbye moved INTO the spawned coroutine so it can be routed
    through the live session and awaited. background.spawn closes the
    coroutine and returns None when there is no loop — which would have lost
    the goodbye and the shutdown along with it."""
    h.worker._farewell_then_go("Goodbye.", restart=False,
                               task_name="test-shutdown")
    assert h.last == "Goodbye.", "nothing was said"
    assert h.fired == ["SHUTDOWN"], "the shutdown never fired"


# ── varied goodbyes ──────────────────────────────────────────────────────────

def test_the_requested_phrasing_is_in_the_pool():
    assert any("see you soon" in line.lower()
               for line in live_worker_module._FAREWELLS_DAY)


def test_there_are_several_goodbyes_to_choose_from():
    assert len(live_worker_module._FAREWELLS_DAY) >= 5
    assert len(live_worker_module._FAREWELLS_LATE) >= 3


def test_a_goodbye_never_immediately_repeats(h):
    """With a handful of options, plain random choice lands on the same one
    back to back often enough to notice — which defeats having several."""
    previous = ""
    for _ in range(40):
        current = h.worker.compose_farewell()
        assert current != previous, f"repeated {current!r} back to back"
        previous = current


def test_goodbyes_actually_vary(h):
    seen = {h.worker.compose_farewell() for _ in range(40)}
    assert len(seen) >= 3, f"only ever said {seen}"


def test_late_night_gets_a_night_farewell(h, monkeypatch):
    import orion_core.live_worker as module

    class _Late(module.datetime):
        @classmethod
        def now(cls, tz=None):
            return module.datetime(2026, 8, 9, 23, 30)

    monkeypatch.setattr(module, "datetime", _Late)
    from orion_core.time_service import TIME
    monkeypatch.setattr(TIME, "_clock", lambda: module.datetime(2026, 8, 9, 23, 30))
    for _ in range(8):
        # Once per iteration: compose_farewell() is stateful (it avoids
        # repeating the previous line), so calling it twice in one expression
        # compares two different farewells.
        farewell = h.worker.compose_farewell().lower()
        assert "night" in farewell or "sleep" in farewell, farewell


def test_the_honorific_is_used_when_configured(h):
    h.worker.dispatcher = types.SimpleNamespace(
        identity=types.SimpleNamespace(preferences={
            "honorific": "sir", "honorific_frequency": "always"}))
    assert "sir" in h.worker.compose_farewell()


def test_a_broken_identity_does_not_break_the_goodbye(h):
    class _Bad:
        @property
        def preferences(self):
            raise RuntimeError("identity unavailable")

    h.worker.dispatcher = types.SimpleNamespace(identity=_Bad())
    assert h.worker.compose_farewell(), "no goodbye at all"


# ── the trigger routes through the question ──────────────────────────────────

def test_the_power_command_asks_rather_than_acting():
    import inspect
    source = inspect.getsource(GenAILiveWorker._handle_power_command)
    assert "_ask_to_confirm_shutdown" in source
    assert "request_shutdown.emit" not in source, (
        "still shuts down directly from the power command")


def test_the_reply_is_checked_before_anything_else_claims_it():
    """A bare 'yes' is not a power command and would otherwise sail past into
    a normal model turn."""
    import inspect
    source = inspect.getsource(GenAILiveWorker)
    resolve = source.index("_resolve_shutdown_confirmation(text)")
    power = source.index("_handle_power_command(lowered)")
    assert resolve < power


# ── the Live model's own call asks too ───────────────────────────────────────

def _live_call(h, said, *names, own_line=""):
    """Run the Live tool handler for *names* after the user said *said*."""
    ran = []
    w = h.worker
    w._turn_said = said
    w._own_lines = live_worker_module.deque(maxlen=12)
    if own_line:
        w._remember_own_line(own_line)
    w.session = None
    w.mic = None
    w._trace = None
    w._turn_active = False
    w._emit_state = lambda *_a: None
    w._transport_ready = lambda: False
    w.observe_for_reflex = lambda *_a, **_k: None
    w.observe_resolver_shadow = lambda *_a: None
    w.connected = False

    async def run_tool(fc):
        ran.append(fc.name)
        return {"ok": True, "result": "done"}, None
    w._run_tool_call = run_tool
    w._farewell_spoken = False
    calls = [types.SimpleNamespace(name=n, id=str(i), args={}) for i, n in enumerate(names)]
    asyncio.run(w._handle_tool_call(types.SimpleNamespace(function_calls=calls)))
    # An accepted power call is answered in place and leaves after the
    # model's goodbye (_leave_after_model_goodbye), never dispatched directly.
    assert not any(n in w.POWER_TOOLS for n in ran), "a power tool ran directly"
    if w._farewell_spoken:
        ran.extend(n for n in names if n in w.POWER_TOOLS)
    return ran


def test_one_goodbye_at_shutdown_in_his_own_voice(h):
    """10 Oct: "when shutting down both of their voices activate at the same
    time saying goodbye" - the Live model said goodbye AND app.py's local
    farewell played, because the tool shut ORION down at once."""
    w = h.worker
    w._drop_live_output = False
    waited = []

    async def no_wait(*a, **k):
        waited.append(1)
    w._await_speech = no_wait

    async def model_speaks():
        started = asyncio.Event()
        w._farewell_audio_started = started
        go = asyncio.ensure_future(w._leave_after_model_goodbye(restart=False))
        await asyncio.sleep(0.05)
        started.set()                                  # the model's goodbye audio arrives
        await go
    asyncio.run(model_speaks())
    assert h.fired == ["SHUTDOWN"] and h.said == [], "a second goodbye was spoken"

    h.fired.clear()
    w.MODEL_GOODBYE_START_S = 0.05

    async def model_silent():
        w._farewell_audio_started = asyncio.Event()
        await w._leave_after_model_goodbye(restart=False)
    asyncio.run(model_silent())
    assert h.fired == ["SHUTDOWN"] and len(h.said) == 1, "silent model: exactly one local goodbye"
    assert waited == [1, 1]


def test_a_shutdown_nobody_asked_for_is_held_and_asked_about(h):
    """The resolver's record has shutdown_orion called on "Can you not see?"
    and "Oh my god, bro". Typed and locally heard "shut down" always asked;
    the Live model's own call went straight through."""
    assert _live_call(h, "Can you not see?", "shutdown_orion") == []
    assert h.worker._shutdown_confirmation_open()
    # Its yes, to the question it was told to ask, lets the next call through.
    assert _live_call(h, "Yes.", "shutdown_orion") == ["shutdown_orion"]


def test_a_held_power_call_does_not_hold_the_others(h):
    assert _live_call(h, "what's the weather", "weather", "shutdown_orion") == ["weather"]


def test_asked_for_shutdowns_go_straight_through(h):
    for said in ("Ryan, shut down.", "shut down to please.",
                 "Okay , all right , I'm going to bed good night."):
        h.worker._pending_shutdown = None
        assert _live_call(h, said, "shutdown_orion") == ["shutdown_orion"], said
    h.worker._pending_shutdown = None
    assert _live_call(h, "Yes please.", "shutdown_orion",
                      own_line="Shall I shut down for the night?") == ["shutdown_orion"]
    h.worker._pending_shutdown = None
    assert _live_call(h, "Ryan, restart", "restart_orion") == ["restart_orion"]


# ── overheard fragments while the live voice is down ──────────────────────────

def test_an_overheard_fragment_gets_no_stock_reply(h):
    """3 Oct: twelve "Understood. What would you like me to do?" in two
    minutes, each to a fragment of a conversation ORION was not in."""
    from orion_core.voice_variety import BANK
    w = h.worker
    w._overheard = "boundary brood"
    filler = BANK["local_idle"][0]
    assert w._overheard_filler("boundary brood", filler)
    assert not w._overheard_filler("boundary brood", "It is ten past three.")  # a real answer
    assert not w._overheard_filler("what time is it", filler)                  # typed / named
    w._overheard = ""                                                          # said his name
    assert not w._overheard_filler("boundary brood", filler)


def test_a_shutdown_asked_for_with_extra_words_goes_through(h):
    """8 Oct: "Okay, what I want you to do is shut down for now because I have
    a lecture." - a request with a reason attached, not a device command."""
    for said in ("Okay, what I want you to do is shut down for now because I have a lecture.",
                 "Yeah, just shut down for now please have a lecture."):
        h.worker._pending_shutdown = None
        assert _live_call(h, said, "shutdown_orion") == ["shutdown_orion"], said
    for said in ("Why did you shut down earlier?", "Don't shut down yet, I'm still working",
                 "Are you going to shut down tonight?"):
        h.worker._pending_shutdown = None
        assert _live_call(h, said, "shutdown_orion") == [], said


def test_shut_down_as_live_mishears_it_goes_straight_through(h):
    """12 Oct 11:16 the owner said "shut down for now"; Live wrote "Should I
    down for now, please, ..." and ORION asked for confirmation, so he had to
    say it twice. 10 Oct: "Should I done for an hour round." The model heard
    the audio and chose to shut down; only the transcript was garbled."""
    for said in ("Should I down for now, please, all right, I'm just going to work a new bit more.",
                 "Should I done for an hour round.",
                 "Shut done for now.",
                 "Shot down for today please.",
                 "Right, I'm done for now.",
                 "Okay, talk to you later."):
        h.worker._pending_shutdown = None
        assert _live_call(h, said, "shutdown_orion") == ["shutdown_orion"], said
    # Look-alikes that are not "shut down": still asked about.
    for said in ("Should I go down to the shop?", "I need to download that file.",
                 "Should I sit down for this?", "The website was down earlier."):
        h.worker._pending_shutdown = None
        assert _live_call(h, said, "shutdown_orion") == [], said
