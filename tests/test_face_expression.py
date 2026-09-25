"""
Tests for ORION's facial performance (Mark XXIII).

The face geometry is emitted once; everything the avatar DOES is these
scalars, fed to the shader as uniforms. So this is where "does the avatar
feel alive rather than robotic" is actually decided, and the tests are
written against the things that make it read as machinery when they go wrong:
metronomic blinking, symmetric motion, a jaw that tracks amplitude exactly,
and expressions that cut instead of easing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm.cognition_state import Mode  # noqa: E402
from orion_core.swarm.render.face_expression import (  # noqa: E402
    BLINK_MAX_GAP_S,
    BLINK_MIN_GAP_S,
    Expression,
    FaceAnimator,
)

FRAME = 1.0 / 60.0


def _run(animator: FaceAnimator, seconds: float, amplitude: float = 0.0):
    """Advance at 60 fps, returning every frame's expression."""
    return [animator.update(FRAME, amplitude)
            for _ in range(int(seconds * 60))]


def _blink_times(frames) -> list[float]:
    times, previous = [], 0.0
    for i, frame in enumerate(frames):
        if frame.blink > 0.5 >= previous:
            times.append(i * FRAME)
        previous = frame.blink
    return times


# ── blinking ─────────────────────────────────────────────────────────────────

def test_orion_blinks():
    assert len(_blink_times(_run(FaceAnimator(), 60.0))) > 4


def test_blink_intervals_are_irregular():
    """A fixed interval is instantly readable as machinery."""
    times = _blink_times(_run(FaceAnimator(), 180.0))
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert len(gaps) > 8
    assert len(set(round(g, 1) for g in gaps)) > len(gaps) // 2


def test_blink_gaps_stay_in_a_human_range():
    times = _blink_times(_run(FaceAnimator(), 180.0))
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) >= BLINK_MIN_GAP_S - 0.2
    assert max(gaps) <= BLINK_MAX_GAP_S + 0.4


def test_a_blink_is_brief():
    frames = _run(FaceAnimator(), 60.0)
    closed = sum(1 for f in frames if f.blink > 0.5)
    assert 0 < closed < len(frames) * 0.08     # eyes open the vast majority


def test_blink_is_fully_open_between_blinks():
    assert any(f.blink == 0.0 for f in _run(FaceAnimator(), 20.0))


# ── speech ───────────────────────────────────────────────────────────────────

def test_speaking_opens_the_mouth():
    frames = _run(FaceAnimator(), 2.0, amplitude=0.8)
    assert max(f.mouth_open for f in frames) > 0.4


def test_silence_closes_the_mouth():
    animator = FaceAnimator()
    _run(animator, 2.0, amplitude=0.9)
    frames = _run(animator, 3.0, amplitude=0.0)
    assert frames[-1].mouth_open < 0.05


def test_the_mouth_is_smoothed_not_a_copy_of_amplitude():
    """Following raw amplitude produces a chattering mouth."""
    animator = FaceAnimator()
    first = animator.update(FRAME, 1.0)
    assert first.mouth_open < 0.5          # cannot snap open in one frame


def test_the_jaw_trails_the_mouth():
    """Giving the jaw mass is most of what stops the face reading as a
    puppet."""
    animator = FaceAnimator()
    frames = _run(animator, 0.5, amplitude=1.0)
    assert frames[-1].jaw_drop < frames[-1].mouth_open


def test_speech_lifts_the_brows_slightly():
    quiet = _run(FaceAnimator(), 1.0, amplitude=0.0)[-1]
    loud = _run(FaceAnimator(), 1.0, amplitude=0.9)[-1]
    assert loud.brow_raise > quiet.brow_raise


def test_amplitude_is_clamped():
    frame = _run(FaceAnimator(), 1.0, amplitude=9.9)[-1]
    assert 0.0 <= frame.mouth_open <= 1.0


# ── expression by cognition mode ─────────────────────────────────────────────

def test_thinking_furrows_and_concentrates():
    animator = FaceAnimator()
    animator.set_mode(Mode.THINKING)
    frame = _run(animator, 3.0)[-1]
    assert frame.brow_raise < 0
    assert frame.focus > 0.5


def test_listening_lifts_the_brows():
    animator = FaceAnimator()
    animator.set_mode(Mode.LISTENING)
    assert _run(animator, 3.0)[-1].brow_raise > 0


def test_standby_dims_and_relaxes_the_face():
    """The visual half of being told to be quiet."""
    animator = FaceAnimator()
    animator.set_mode(Mode.STANDBY)
    frame = _run(animator, 4.0)[-1]
    assert frame.energy < 0.5
    assert frame.focus < 0.2


def test_executing_raises_energy():
    idle = _run(FaceAnimator(), 3.0)[-1]
    animator = FaceAnimator()
    animator.set_mode(Mode.EXECUTING)
    assert _run(animator, 3.0)[-1].energy > idle.energy


def test_expression_eases_rather_than_cutting():
    """A mode change must be a transition, not a jump cut."""
    animator = FaceAnimator()
    _run(animator, 2.0)
    animator.set_mode(Mode.THINKING)
    first = animator.update(FRAME, 0.0)
    settled = _run(animator, 3.0)[-1]
    assert abs(first.focus) < abs(settled.focus)


def test_an_unknown_mode_falls_back_safely():
    animator = FaceAnimator()
    animator.set_mode("not a mode")          # type: ignore[arg-type]
    assert _run(animator, 1.0)[-1].energy > 0


# ── idle life ────────────────────────────────────────────────────────────────

def test_the_face_breathes():
    frames = _run(FaceAnimator(), 8.0)
    values = [f.breath for f in frames]
    assert max(values) > 0 > min(values)


def test_idle_drift_never_repeats_a_simple_figure():
    """Two different rates, so the sway does not trace a loop."""
    frames = _run(FaceAnimator(), 12.0)
    assert any(abs(f.drift_x - f.drift_y) > 1e-4 for f in frames)


def test_the_eyes_move_on_their_own():
    frames = _run(FaceAnimator(), 20.0)
    assert len({round(f.gaze_x, 2) for f in frames}) > 3


def test_gaze_stays_within_a_believable_range():
    """Eyes that swing to the extremes look anxious, not thoughtful."""
    for frame in _run(FaceAnimator(), 30.0):
        assert abs(frame.gaze_x) <= 0.5
        assert abs(frame.gaze_y) <= 0.4


def test_a_deliberate_look_overrides_the_idle_wander():
    """Face tracking must actually point his eyes at the person."""
    animator = FaceAnimator()
    animator.look_at(0.9, -0.5)
    frame = _run(animator, 1.5)[-1]
    assert frame.gaze_x > 0.3
    assert frame.gaze_y < 0.0


# ── the shader contract ──────────────────────────────────────────────────────

def test_every_uniform_is_supplied():
    uniforms = _run(FaceAnimator(), 1.0)[-1].as_uniforms()
    for name in ("u_blink", "u_mouth", "u_jaw", "u_brow", "u_gaze_x",
                 "u_gaze_y", "u_breath", "u_drift_x", "u_drift_y",
                 "u_energy", "u_focus"):
        assert name in uniforms
        assert isinstance(uniforms[name], float)


def test_animation_is_frame_rate_independent():
    """The same elapsed time must give the same posture at any frame rate,
    or the face behaves differently on a slower machine."""
    slow = FaceAnimator()
    slow.set_mode(Mode.THINKING)
    for _ in range(30):
        slow.update(1.0 / 30.0, 0.0)

    fast = FaceAnimator()
    fast.set_mode(Mode.THINKING)
    for _ in range(120):
        fast.update(1.0 / 120.0, 0.0)

    assert abs(slow.update(0.0, 0.0).focus - fast.update(0.0, 0.0).focus) < 0.05


def test_a_huge_frame_gap_cannot_destabilise_the_face():
    """A stall must not teleport the expression."""
    animator = FaceAnimator()
    frame = animator.update(30.0, 0.5)
    assert 0.0 <= frame.mouth_open <= 1.0
    assert -1.0 <= frame.brow_raise <= 1.0


# ── spectral articulation (Mark XXII) — the mouth forms syllables ────────────

def test_no_spectrum_is_neutral_articulation():
    from orion_core.swarm.render.face_expression import FaceAnimator
    a = FaceAnimator()
    assert a._articulation() == 1.0          # falls back to pure amplitude


def test_vowels_open_the_mouth_wider_than_sibilants():
    from orion_core.swarm.render.face_expression import FaceAnimator
    vowel = FaceAnimator(); vowel.set_spectrum({"low": 0.1, "mid": 0.8, "high": 0.1})
    sib = FaceAnimator();   sib.set_spectrum({"low": 0.1, "mid": 0.1, "high": 0.8})
    assert vowel._articulation() > 1.0        # a vowel opens fuller
    assert sib._articulation() < 1.0          # a sibilant stays tighter
    # and it actually reaches the mouth, at equal amplitude
    ve = None; se = None
    for _ in range(20):
        ve = vowel.update(0.05, amplitude=0.9)
    for _ in range(20):
        se = sib.update(0.05, amplitude=0.9)
    assert ve.mouth_open > se.mouth_open


def test_set_spectrum_ignores_bad_input():
    from orion_core.swarm.render.face_expression import FaceAnimator
    a = FaceAnimator()
    a.set_spectrum(None); a.set_spectrum("nope"); a.set_spectrum(42)
    assert a._articulation() == 1.0           # unchanged


def test_articulation_is_bounded():
    from orion_core.swarm.render.face_expression import FaceAnimator
    a = FaceAnimator()
    for bands in ({"low": 9, "mid": 0, "high": 0}, {"low": 0, "mid": 9, "high": 0},
                  {"low": 0, "mid": 0, "high": 9}, {"low": 1, "mid": 1, "high": 1}):
        a.set_spectrum(bands)
        assert 0.4 <= a._articulation() <= 1.3


def test_spectrum_wired_from_bus_to_face():
    # the whole chain: swarm_view forwards voice_spectrum to the GL face
    import inspect
    from orion_core.gui import swarm_view
    src = inspect.getsource(swarm_view)
    assert "voice_spectrum.connect" in src
    assert "set_speech_spectrum" in src
