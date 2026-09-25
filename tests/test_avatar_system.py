"""
Avatar system tests — AvatarStateMachine, AvatarAnimationManager,
AvatarEngine and AvatarController, plus the FaceTracker degradation path.

All headless: the engine layer imports no Qt, the controller drives a fake
face widget, and no test needs a webcam.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.avatar import (AVATAR_STATES, AvatarAnimationManager,
                               AvatarController, AvatarEngine,
                               AvatarStateMachine, canonical_state)


# ── canonical state mapping ───────────────────────────────────────────────────

def test_canonical_state_maps_loose_strings():
    assert canonical_state("SPEAKING") == "speaking"
    assert canonical_state("Deep RESEARCH sweep") == "researching"
    assert canonical_state("tool execution") == "executing"
    assert canonical_state("ALERT: fault") == "warning"
    assert canonical_state("") == "idle"
    assert canonical_state("completely unknown") == "idle"
    for state in AVATAR_STATES:
        assert canonical_state(state) == state


# ── state machine ─────────────────────────────────────────────────────────────

def test_transitions_are_gradual_never_abrupt():
    machine = AvatarStateMachine()
    machine.request("listening")
    assert not machine.settled
    weights = machine.weights()
    assert weights.get("idle", 0) > 0.9          # still mostly the old state
    machine.tick(0.3)
    weights = machine.weights()
    assert 0.05 < weights["listening"] < 0.95    # mid-blend
    machine.tick(0.4)
    assert machine.settled
    assert machine.weights() == {"listening": 1.0}


def test_notification_is_transient_and_reverts():
    machine = AvatarStateMachine()
    machine.request("thinking")
    machine.tick(1.0)
    machine.request("notification")
    machine.tick(1.0)                            # finish the blend
    assert machine.state == "notification"
    machine.tick(10.0)                           # hold expires
    machine.tick(1.0)
    assert machine.state == "thinking"           # back where we were


def test_user_state_outranks_transient_overlay():
    machine = AvatarStateMachine()
    machine.request("notification")
    machine.tick(1.0)
    machine.request("speaking")                  # persistent request wins now
    machine.tick(1.0)
    assert machine.state == "speaking"
    machine.tick(30.0)                           # no decay hijack afterwards
    assert machine.state == "speaking"


def test_renotify_refreshes_hold_without_reblending():
    machine = AvatarStateMachine()
    machine.request("warning")
    machine.tick(1.0)
    assert machine.request("warning") is False   # no new transition
    assert machine.state == "warning"


# ── animation channels ────────────────────────────────────────────────────────

def test_blink_scheduler_fires_and_recovers():
    animation = AvatarAnimationManager(rng=random.Random(7))
    blinked = False
    for _ in range(400):                          # 20 simulated seconds
        channels = animation.tick(0.05, "idle")
        if channels["blink"] > 0:
            blinked = True
    assert blinked
    assert animation.tick(0.05, "idle")["breath"] >= 0.0


def test_lip_envelope_attack_and_decay():
    animation = AvatarAnimationManager(rng=random.Random(1))
    animation.set_amplitude(1.0)
    rise = animation.tick(0.05, "speaking")["mouth"]
    assert rise > 0.2                             # fast attack
    animation.set_amplitude(0.0)
    fall = animation.tick(0.05, "speaking")["mouth"]
    assert fall < rise                            # decaying
    for _ in range(40):
        fall = animation.tick(0.05, "speaking")["mouth"]
    assert fall < 0.05                            # settles closed


def test_spectrum_envelope_attack_and_decay():
    animation = AvatarAnimationManager(rng=random.Random(1))
    animation.set_spectrum(1.0, 0.5, 0.2)
    rise = animation.tick(0.05, "speaking")
    assert rise["spectrum_low"] > 0.2               # fast attack, same as amplitude
    animation.set_spectrum(0.0, 0.0, 0.0)
    fall = animation.tick(0.05, "speaking")
    assert fall["spectrum_low"] < rise["spectrum_low"]
    for _ in range(40):
        fall = animation.tick(0.05, "speaking")
    assert fall["spectrum_low"] < 0.05
    assert fall["spectrum_mid"] < 0.05
    assert fall["spectrum_high"] < 0.05


def test_spectrum_clamps_out_of_range_input():
    animation = AvatarAnimationManager(rng=random.Random(1))
    animation.set_spectrum(5.0, -3.0, 1.0)
    assert animation.spectrum_target == (1.0, 0.0, 1.0)


def test_face_tracking_presence_hysteresis_and_clamp():
    animation = AvatarAnimationManager(rng=random.Random(2))
    # One noisy detection must NOT engage tracking.
    animation.observe_face(1.0, 0.0, True)
    assert not animation.tracking
    for _ in range(3):
        animation.observe_face(1.0, 1.0, True)
    assert animation.tracking
    for _ in range(120):
        animation.tick(0.05, "idle")
    # Yaw settles clamped within the readable limit (plus tiny sway).
    assert abs(animation.yaw) <= AvatarAnimationManager.YAW_LIMIT + 0.05
    assert abs(animation.pitch) <= AvatarAnimationManager.PITCH_LIMIT + 0.05
    # Losing the face needs 8 consecutive misses.
    for _ in range(7):
        animation.observe_face(0, 0, False)
    assert animation.tracking
    animation.observe_face(0, 0, False)
    assert not animation.tracking


# ── engine frames ─────────────────────────────────────────────────────────────

def test_engine_frame_shape_and_state_mapping():
    engine = AvatarEngine(rng=random.Random(3))
    engine.request_state("researching")
    for _ in range(30):
        frame = engine.tick(0.05)
    assert frame["state"] == "researching"
    assert frame["js_state"] == "THINKING"        # rig-level mapping
    assert "RESEARCHING" in frame["label"]
    assert frame["settled"] is True
    for key in ("morph", "amplitude", "yaw", "pitch", "breath", "blink",
                "energy", "tracking", "spectrum_low", "spectrum_mid", "spectrum_high"):
        assert key in frame


def test_dormant_states_collapse_to_orb_smoothly():
    engine = AvatarEngine(rng=random.Random(4))
    engine.request_state("LISTENING")
    for _ in range(60):
        frame = engine.tick(0.05)
    assert frame["morph"] == 1.0                  # materialised
    engine.request_state("STANDBY")
    first = engine.tick(0.05)["morph"]
    assert 0.0 < first < 1.0                      # collapsing, not snapping
    for _ in range(80):
        frame = engine.tick(0.05)
    assert frame["morph"] == 0.0                  # fully the orb


def test_amplitude_implies_speaking():
    engine = AvatarEngine(rng=random.Random(5))
    engine.set_amplitude(0.8)
    for _ in range(30):
        frame = engine.tick(0.05)
    assert frame["state"] == "speaking"
    assert frame["amplitude"] > 0.3


# ── controller ────────────────────────────────────────────────────────────────

class FakeFace:
    def __init__(self):
        self.states, self.amps, self.poses, self.labels, self.morphs = \
            [], [], [], [], []
        self.spectra = []

    def set_state(self, state):
        self.states.append(state)

    def set_amplitude(self, value):
        self.amps.append(value)

    def set_spectrum(self, low, mid, high):
        self.spectra.append((low, mid, high))

    def set_pose(self, yaw, pitch):
        self.poses.append((yaw, pitch))

    def set_label(self, text):
        self.labels.append(text)

    def set_morph(self, value):
        self.morphs.append(value)


def test_controller_pushes_only_on_change_and_survives_faults():
    face = FakeFace()
    controller = AvatarController(face=face)
    controller.on_state("LISTENING")
    for _ in range(20):
        controller.tick(0.05)
    assert face.states.count("LISTENING") == 1    # state pushed once
    assert len(face.poses) == 20                  # pose every tick
    assert any("LISTENING" in label for label in face.labels)
    # A face that explodes must never break the behaviour loop.
    face.set_pose = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    frame = controller.tick(0.05)
    assert frame["state"] == "listening"
    status = controller.status()
    assert status["state"] == "listening"


def test_controller_pushes_spectrum_to_face():
    face = FakeFace()
    controller = AvatarController(face=face)
    controller.on_state("SPEAKING")
    controller.on_voice_spectrum({"low": 0.8, "mid": 0.1, "high": 0.05})
    for _ in range(10):
        controller.tick(0.05)
    assert len(face.spectra) == 10                   # pushed every tick, like amplitude
    assert face.spectra[-1][0] > 0.3                  # low band rose toward the target


def test_controller_voice_spectrum_ignores_non_dict_payload():
    controller = AvatarController(face=FakeFace())
    controller.on_voice_spectrum("not a dict")        # must not raise
    controller.on_voice_spectrum(None)
    assert controller.engine.animation.spectrum_target == (0.0, 0.0, 0.0)


def test_controller_face_sample_routes_to_engine():
    controller = AvatarController(face=FakeFace())
    for _ in range(3):
        controller.on_face_sample({"present": True, "x": 0.5, "y": -0.2})
    assert controller.engine.animation.tracking


# ── face tracker degradation ──────────────────────────────────────────────────

def test_face_tracker_degrades_without_camera_stack(monkeypatch):
    from orion_core import face_tracking
    samples = []
    tracker = face_tracking.FaceTracker(samples.append)
    monkeypatch.setattr(type(tracker), "available", False)
    result = tracker.start()
    assert not result.ok
    assert "OpenCV" in result.text
    assert tracker.status().text.startswith("Face tracking unavailable")
    stop = tracker.stop()                          # always safe
    assert stop.ok
    assert samples and samples[-1]["present"] is False
