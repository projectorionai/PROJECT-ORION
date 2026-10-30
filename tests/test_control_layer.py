"""
Headless tests for the AutonomousControlLayer (Improvement Pass, Priority 1.1):
coordinate clamping, the ORION_AUTONOMY kill-switch, and the pyautogui
FAILSAFE/PAUSE configuration. The real input backends are replaced with
recording fakes before construction so no test ever moves the actual cursor
or types into the host.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.control import AutonomousControlLayer


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubDisplay:
    """A 2-monitor virtual desktop spanning x 0..5119, y 0..1439."""

    def clamp_to_desktop(self, vx, vy):
        return (min(max(int(vx), 0), 5119), min(max(int(vy), 0), 1439))

    def cursor_position(self):
        return (50, 60)

    def to_virtual(self, monitor, x, y):
        return (int(x) + 2560 * int(monitor), int(y))


class _FakePG(types.ModuleType):
    """Stands in for pyautogui; records every call, moves nothing."""

    def __init__(self):
        super().__init__("pyautogui")
        self.PAUSE = None
        self.FAILSAFE = None
        self.calls = []

    def moveTo(self, x, y, duration=0.0):
        self.calls.append(("moveTo", x, y))

    def click(self, x, y, clicks=1, interval=0.0, button="left"):
        self.calls.append(("click", x, y, clicks, button))

    def dragTo(self, x, y, duration=0.0, button="left"):
        self.calls.append(("dragTo", x, y, button))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def typewrite(self, text, interval=0.0):
        self.calls.append(("typewrite", text))


class _FakeKB(types.ModuleType):
    def __init__(self):
        super().__init__("keyboard")
        self.written = []

    def write(self, text, delay=0.0):
        self.written.append(text)


@pytest.fixture
def layer(monkeypatch):
    """A control layer whose backends are recording fakes, in fast pointer
    mode (one jump, one click) - human motion has its own tests below."""
    fake_pg, fake_kb = _FakePG(), _FakeKB()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    monkeypatch.setitem(sys.modules, "keyboard", fake_kb)
    monkeypatch.delenv("ORION_AUTONOMY", raising=False)
    monkeypatch.setenv("ORION_SHOW_WORKING", "0")
    built = AutonomousControlLayer(_StubBus(), _StubDisplay())
    assert built._pg is fake_pg and built._kb is fake_kb
    return built


# ── backend configuration: FAILSAFE stays armed, PAUSE stays off ─────────────

def test_failsafe_and_pause_configuration(layer):
    assert layer._pg.FAILSAFE is True   # corner-abort must remain a hard stop
    assert layer._pg.PAUSE == 0.0       # pacing is deliberate, not global


def test_missing_pyautogui_degrades_gracefully(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyautogui", None)  # import → TypeError
    monkeypatch.setitem(sys.modules, "keyboard", None)
    built = AutonomousControlLayer(_StubBus(), _StubDisplay())
    assert built._pg is None
    result = built.move_cursor(10, 10)
    assert not result.ok and "offline" in result.text


# ── the owner's live turn (28 Oct 2026) ───────────────────────────────────────

def test_the_owners_turn_may_click_type_and_scroll_but_background_work_may_not(layer):
    """Mark XXXX.2 allowed move_cursor and keyboard_input on his turn, but each
    action was checked under its own name, so a click he asked for was still
    refused as "needs a foreground session"."""
    from orion_core import execution_policy as policy
    actions = (lambda: layer.click(5, 5), lambda: layer.scroll(3),
               lambda: layer.drag_cursor(0, 0, 5, 5), lambda: layer.type_text("hi"))
    with policy.conversation_scope("voice"):
        for action in actions:
            result = action()
            assert result.ok, result.text
    for origin in ("job", "plugin", "agent"):
        with policy.background_scope(origin):
            for action in actions + (lambda: layer.send_hotkeys("ctrl+l"),):
                result = action()
                assert not result.ok and result.receipt["status"] == "requires_foreground"


# ── ORION_AUTONOMY kill-switch ────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_kill_switch_env_disables_all_actions(monkeypatch, value):
    fake_pg = _FakePG()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    monkeypatch.setitem(sys.modules, "keyboard", _FakeKB())
    monkeypatch.setenv("ORION_AUTONOMY", value)
    built = AutonomousControlLayer(_StubBus(), _StubDisplay())
    assert built.enabled is False
    for action in (
        lambda: built.move_cursor(5, 5),
        lambda: built.click(5, 5),
        lambda: built.drag_cursor(0, 0, 5, 5),
        lambda: built.scroll(3),
        lambda: built.type_text("hello"),
    ):
        result = action()
        assert not result.ok and "disabled" in result.text.lower()
    assert fake_pg.calls == []          # the backend was never touched


@pytest.mark.parametrize("value", ["1", "true", "on", ""])
def test_kill_switch_defaults_to_enabled(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "pyautogui", _FakePG())
    monkeypatch.setitem(sys.modules, "keyboard", _FakeKB())
    if value:
        monkeypatch.setenv("ORION_AUTONOMY", value)
    else:
        monkeypatch.delenv("ORION_AUTONOMY", raising=False)
    assert AutonomousControlLayer(_StubBus(), _StubDisplay()).enabled is True


def test_set_enabled_runtime_toggle(layer):
    layer.set_enabled(False)
    assert not layer.move_cursor(5, 5).ok
    assert layer._pg.calls == []
    layer.set_enabled(True)
    assert layer.move_cursor(5, 5).ok
    assert ("moveTo", 5, 5) in layer._pg.calls


# ── coordinate clamping ───────────────────────────────────────────────────────

def test_move_clamps_to_visible_desktop(layer):
    layer.move_cursor(999999, -500)
    assert ("moveTo", 5119, 0) in layer._pg.calls


def test_click_clamps_and_records_region(layer):
    result = layer.click(-100, 999999)
    assert result.ok
    assert ("click", 0, 1439, 1, "left") in layer._pg.calls
    x, y, w, h = layer.last_action["region"]
    assert x >= 0 and y >= 0 and w >= 1 and h >= 1
    assert x + w <= 5120 and y + h <= 1440


def test_monitor_relative_coordinates_resolve_to_virtual(layer):
    layer.move_cursor(100, 200, monitor=1)
    assert ("moveTo", 2660, 200) in layer._pg.calls


def test_drag_clamps_both_endpoints(layer):
    layer.drag_cursor(-50, -50, 999999, 999999)
    assert ("moveTo", 0, 0) in layer._pg.calls
    assert ("dragTo", 5119, 1439, "left") in layer._pg.calls


def test_click_without_coordinates_uses_cursor_position(layer):
    layer.click()
    assert ("click", 50, 60, 1, "left") in layer._pg.calls


# ── typed text passes through the security firewall ──────────────────────────

def test_type_text_blocks_destructive_payloads(layer):
    result = layer.type_text("rm -rf /")
    assert not result.ok
    assert layer._kb.written == [] and layer._pg.calls == []


def test_type_text_blocks_core_mutation(layer):
    result = layer.type_text("del orion_core")
    assert not result.ok
    assert layer._kb.written == []


def test_type_text_benign_goes_to_keyboard_backend(layer):
    result = layer.type_text("hello world")
    assert result.ok
    assert layer._kb.written == ["hello world"]


def test_type_text_falls_back_to_pyautogui(layer):
    layer._kb = None
    layer.type_text("fallback text")
    assert ("typewrite", "fallback text") in layer._pg.calls


def test_empty_type_text_refused(layer):
    assert not layer.type_text("").ok


# ── action metadata for the verifier ─────────────────────────────────────────

def test_actions_record_last_action_metadata(layer):
    layer.click(100, 100)
    meta = layer.last_action
    assert meta["action"] == "click"
    assert meta["x"] == 100 and meta["y"] == 100
    assert meta["region"] is not None


# ── human pointer motion (human_pointer): the planner ────────────────────────

import math
import random

from orion_core import human_pointer as hp
from orion_core import human_typing as ht


def _progress(plan, start, end):
    (sx, sy), (ex, ey) = start, end
    length = math.hypot(ex - sx, ey - sy)
    ux, uy = (ex - sx) / length, (ey - sy) / length
    return [(s.x - sx) * ux + (s.y - sy) * uy for s in plan.steps]


def _random_moves(count=150):
    rng = random.Random(7)
    for seed in range(count):
        start = (rng.randint(0, 1919), rng.randint(0, 1079))
        end = (rng.randint(0, 1919), rng.randint(0, 1079))
        if start != end:
            yield seed, start, end


def test_human_path_ends_exactly_on_the_target():
    for seed, start, end in _random_moves():
        plan = hp.plan_pointer_path(start, end, seed=seed)
        assert plan.end == end, (seed, start, end)
        assert plan.mode == hp.HUMAN and len(plan.steps) >= 2


def test_human_path_progress_towards_the_target_never_goes_backwards():
    for seed, start, end in _random_moves():
        plan = hp.plan_pointer_path(start, end, seed=seed, overshoot=False)
        progress = _progress(plan, start, end)
        assert all(b >= a for a, b in zip(progress, progress[1:])), (seed, start, end)
        assert max(progress) <= math.hypot(end[0] - start[0], end[1] - start[1]) + 1e-9


def test_human_path_is_curved_not_a_ruler_line():
    plan = hp.plan_pointer_path((100, 500), (1500, 500), seed=3, overshoot=False)
    assert max(abs(s.y - 500) for s in plan.steps) >= 10


def test_human_move_duration_follows_fitts_law_within_the_range():
    for distance in (10, 40, 150, 600, 1500, 4000):
        for seed in range(20):
            plan = hp.plan_pointer_path((0, 0), (distance, 0), seed=seed)
            assert 0.25 - 1e-9 <= plan.duration <= 0.9 + 1e-9, (distance, plan.duration)
    times = [hp.fitts_duration(d) for d in (20, 100, 400, 1600)]
    assert times == sorted(times) and times[0] < times[-1]
    assert hp.fitts_duration(0) == 0.0


def test_long_moves_overshoot_a_little_and_correct():
    plans = [hp.plan_pointer_path((0, 500), (1500, 500), seed=s) for s in range(40)]
    overshooting = [p for p in plans if p.overshoot]
    assert overshooting and len(overshooting) < len(plans)
    for plan in overshooting:
        assert plan.end == (1500, 500)
        assert 0 < max(s.x for s in plan.steps) - 1500 <= 30
    short = [hp.plan_pointer_path((0, 0), (200, 0), seed=s) for s in range(40)]
    assert not any(p.overshoot for p in short)


def test_human_path_stays_on_the_screens():
    def inside(x, y):
        return 0 <= x < 1920 and 0 <= y < 1080
    for seed in range(60):
        plan = hp.plan_pointer_path((5, 3), (1915, 4), seed=seed, inside=inside)
        assert all(inside(s.x, s.y) for s in plan.steps), seed
        assert plan.end == (1915, 4)


def test_fast_mode_is_one_instant_jump():
    plan = hp.plan_pointer_path((10, 10), (1800, 900), mode="fast")
    assert [(s.x, s.y) for s in plan.steps] == [(1800, 900)]
    assert plan.duration == 0.0 and plan.mode == hp.FAST


def test_pointer_plans_replay_exactly_from_a_seed():
    first = hp.plan_pointer_path((40, 60), (1200, 700), seed=11)
    assert first == hp.plan_pointer_path((40, 60), (1200, 700), seed=11)


@pytest.mark.parametrize("said,speed", [("fast", "fast"), ("Instantly", "fast"), (True, "fast"),
                                        ("human", "human"), ("naturally", "human"),
                                        ("", None), ("sideways", None), (None, None)])
def test_speed_words_normalise(said, speed):
    assert hp.normalise_speed(said) == speed


def test_click_timing_holds_and_gaps_are_human():
    for seed in range(200):
        single = hp.plan_click(1, seed=seed)
        assert len(single.holds) == 1 and not single.gaps
        assert 0.06 <= single.holds[0] <= 0.14
        assert hp.SETTLE_MIN_S <= single.settle <= hp.SETTLE_MAX_S
        double = hp.plan_click(2, seed=seed, double_click_s=0.5)
        assert all(0.06 <= hold <= 0.14 for hold in double.holds)
        assert 0.03 <= double.gaps[0] <= 0.15
        assert double.holds[0] + double.gaps[0] <= 0.3     # well inside 0.5 s
    for seed in range(50):                                 # a quick double-click setting
        tight = hp.plan_click(2, seed=seed, double_click_s=0.2)
        assert tight.gaps[0] >= 0.03
        assert tight.holds[0] + tight.gaps[0] <= max(0.12, tight.holds[0] + 0.03) + 1e-9
    fast = hp.plan_click(2, mode="fast")
    assert fast.settle == 0.0 and set(fast.holds) == {0.0} and set(fast.gaps) == {0.0}


# ── human pointer motion: the control layer replays it ───────────────────────

class _HumanPG(_FakePG):
    """Records separate press and release (what a human click needs)."""

    def mouseDown(self, x=None, y=None, button="left"):
        self.calls.append(("mouseDown", x, y, button))

    def mouseUp(self, x=None, y=None, button="left"):
        self.calls.append(("mouseUp", x, y, button))


@pytest.fixture
def human(monkeypatch):
    """A layer in its default (human) mode; sleeps are recorded, not slept."""
    fake_pg, fake_kb = _HumanPG(), _FakeKB()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    monkeypatch.setitem(sys.modules, "keyboard", fake_kb)
    monkeypatch.delenv("ORION_AUTONOMY", raising=False)
    monkeypatch.delenv("ORION_SHOW_WORKING", raising=False)
    monkeypatch.setenv("ORION_KEEP_CURSOR", "1")
    slept = []
    monkeypatch.setattr("orion_core.control.time.sleep", slept.append)
    built = AutonomousControlLayer(_StubBus(), _StubDisplay())
    built._rng = random.Random(5)
    built.slept = slept
    return built


def _moves(calls):
    return [(c[1], c[2]) for c in calls if c[0] == "moveTo"]


def test_human_click_travels_a_path_then_presses_and_releases(human):
    assert human.pointer_mode() == hp.HUMAN
    assert human.click(900, 500).ok
    moves = _moves(human._pg.calls)
    assert len(moves) > 10 and moves[-1] == (900, 500)
    presses = [c for c in human._pg.calls if c[0] in {"mouseDown", "mouseUp", "click"}]
    assert presses == [("mouseDown", 900, 500, "left"), ("mouseUp", 900, 500, "left")]
    assert 0.06 <= human.slept[-1] <= 0.14                 # the button was held
    assert human.last_action["mode"] == hp.HUMAN


def test_human_double_click_has_a_gap_inside_the_double_click_time(human, monkeypatch):
    monkeypatch.setattr(AutonomousControlLayer, "_double_click_s", staticmethod(lambda: 0.5))
    assert human.double_click(400, 300).ok
    presses = [c[0] for c in human._pg.calls if c[0] in {"mouseDown", "mouseUp"}]
    assert presses == ["mouseDown", "mouseUp", "mouseDown", "mouseUp"]
    settle, hold1, gap, hold2 = human.slept[-4:]           # the click's own waits
    assert hp.SETTLE_MIN_S <= settle <= hp.SETTLE_MAX_S
    assert 0.06 <= hold1 <= 0.14 and 0.06 <= hold2 <= 0.14
    assert 0.03 <= gap <= 0.15 and hold1 + gap < 0.5


def test_fast_click_jumps_and_clicks_without_waiting(human):
    assert human.click(900, 500, mode="fast").ok
    assert human._pg.calls == [("moveTo", 900, 500), ("click", 900, 500, 1, "left")]
    assert human.slept == []


def test_the_requests_speed_reaches_the_layer(human):
    with hp.speed_scope("fast"):
        assert human.pointer_mode() == hp.FAST
        assert human.move_cursor(700, 400).ok
    assert human._pg.calls == [("moveTo", 700, 400)]
    assert human.pointer_mode() == hp.HUMAN                 # the scope ended
    assert human.pointer_mode("fast") == hp.FAST            # an explicit word wins


def test_human_move_cursor_lands_exactly(human):
    assert human.move_cursor(3000, 1200).ok
    moves = _moves(human._pg.calls)
    assert moves[-1] == (3000, 1200) and len(moves) > 10


def test_switching_autonomy_off_mid_path_stops_the_pointer_and_the_click(human, monkeypatch):
    def disable_soon(_seconds):
        if len(human._pg.calls) >= 5:
            human.set_enabled(False)
    monkeypatch.setattr("orion_core.control.time.sleep", disable_soon)
    result = human.click(1800, 900)
    assert not result.ok and "disabled" in result.text.lower()
    assert _moves(human._pg.calls)[-1] != (1800, 900)
    assert not any(c[0] in {"mouseDown", "click"} for c in human._pg.calls)


def test_human_drag_holds_the_button_along_the_path_and_lets_go_at_the_end(human):
    assert human.drag_cursor(100, 100, 600, 400).ok
    calls = human._pg.calls
    down = calls.index(("mouseDown", 100, 100, "left"))
    assert calls[-1] == ("mouseUp", 600, 400, "left")
    carried = _moves(calls[down:])
    assert len(carried) > 5 and carried[-1] == (600, 400)
    assert not any(c[0] == "dragTo" for c in calls)


def test_an_interrupted_drag_still_releases_the_button(human):
    def fail_midway(x, y):
        if any(c[0] == "mouseDown" for c in human._pg.calls):
            raise RuntimeError("input backend failed")
        human._pg.calls.append(("moveTo", x, y))
    human._pg.moveTo = fail_midway
    result = human.drag_cursor(100, 100, 600, 400)
    assert not result.ok and "not repeated" in result.text
    assert human._pg.calls[-1][0] == "mouseUp"


# ── human typing rhythm (human_typing) ────────────────────────────────────────

SENTENCE = "The other day, I typed this. Then 42 more words; quickly!\nDone."


def test_key_gaps_stay_in_bounds_and_keep_the_requested_pace():
    for seed in range(30):
        profile = ht.TypingProfile(wpm=90, seed=seed)
        delays = ht.keystroke_delays(SENTENCE, profile)
        assert len(delays) == len(SENTENCE)
        assert all(ht.MIN_KEY_S <= d <= ht.MAX_KEY_S for d in delays)
        expected = profile.base_delay() * len(SENTENCE)
        assert abs(sum(delays) - expected) / expected < 0.1


def test_rhythm_is_quicker_inside_common_bigrams_and_slower_after_punctuation():
    assert ht.key_rhythm("t", "h") < 1.0 < ht.key_rhythm(" ", "w")
    assert ht.key_rhythm(".", " ") > ht.key_rhythm(",", " ") > ht.key_rhythm(" ", "w")
    assert ht.key_rhythm("a", "B") > ht.key_rhythm("a", "b")
    delays = ht.keystroke_delays("the cat. the", ht.TypingProfile(wpm=80, jitter=0, pause_rate=0))
    after_t, after_c, after_stop = delays[0], delays[4], delays[7]
    assert after_t < after_c < after_stop


def test_thinking_pauses_fall_at_word_boundaries_and_cost_at_most_a_quarter():
    text = "one two three four five six seven"
    profile = ht.TypingProfile(wpm=60, jitter=0, pause_rate=0)
    plain = ht.keystroke_delays(text, profile)
    paused = ht.keystroke_delays(text, ht.TypingProfile(wpm=60, jitter=0, pause_rate=1.0, seed=2))
    budget = profile.base_delay() * len(text)
    assert abs(sum(paused) - budget) < 1e-6 and abs(sum(plain) - budget) < 1e-6
    # The keys between the pauses were sped up to pay for them...
    letters = [i for i, ch in enumerate(text) if ch != " "]
    assert all(paused[i] < plain[i] for i in letters)
    # ...and the pauses (at spaces only) took no more than a quarter.
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    scale = paused[letters[0]] / plain[letters[0]]
    assert sum(paused[i] - plain[i] * scale for i in spaces) <= 0.25 * budget + 1e-6


def test_typing_plans_replay_from_a_seed_and_fast_mode_waits_for_nothing():
    profile = ht.TypingProfile(wpm=120, typo_rate=0.2, seed=9)
    assert ht.plan_keystrokes(SENTENCE, profile) == ht.plan_keystrokes(SENTENCE, profile)
    fast = ht.TypingProfile(wpm=120, typo_rate=0.5, fast=True)
    assert ht.keystroke_delays(SENTENCE, fast) == [0.0] * len(SENTENCE)
    plan = ht.plan_keystrokes(SENTENCE, fast)
    assert "".join(k.char for k in plan) == SENTENCE
    assert all(k.kind == "char" and k.delay == 0.0 for k in plan)


def test_typos_are_corrected_and_still_spell_the_text():
    plan = ht.plan_keystrokes("hello world", ht.TypingProfile(typo_rate=0.6, seed=4))
    typed: list[str] = []
    for stroke in plan:
        if stroke.kind == "backspace":
            typed.pop()
        else:
            typed.append(stroke.char)
    assert "".join(typed) == "hello world"
    assert any(k.kind == "typo" for k in plan)
    assert all(ht.MIN_KEY_S <= k.delay <= ht.MAX_KEY_S for k in plan)


def test_human_typing_sends_one_key_at_a_time_with_the_planned_gaps(human):
    assert human.type_text("hi there", human=True, wpm=100, seed=1).ok
    assert human._kb.written == list("hi there")
    assert len(human.slept) == len("hi there")
    assert all(ht.MIN_KEY_S <= s <= ht.MAX_KEY_S for s in human.slept)


def test_fast_typing_writes_the_text_in_one_go_without_waiting(human):
    assert human.type_text("hi there").ok
    assert human._kb.written == ["hi there"] and human.slept == []
