"""
Tests for Track E — the perception loop.

ORION's eyes were a shutter, not a stream: every look was a paid cloud call, so
looking often was unaffordable, so he looked rarely, so he never noticed
anything. The fix is to separate watching (local, continuous, free) from
understanding (remote, occasional, expensive).

What must be true:

  * the local layer never needs a camera, numpy, or a cloud model to work —
    it is arithmetic on a small grid, and it is testable as such;
  * the FIRST frame is not an event storm — there is nothing to differ against;
  * an object that passes out of view keeps its IDENTITY for a grace period and
    resumes the same track when it returns, rather than being reported as a
    departure and then a fresh arrival;
  * the cloud is called on CHANGE, never on mere motion, never faster than the
    floor, and at least as often as the heartbeat;
  * every degradation costs capability, not stability: no frame source, no
    describer, a camera returning None forever, a malformed frame.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.concurrency import ToolClass, classify
from orion_core.perception import (
    CLOUD_HEARTBEAT_S,
    MIN_CLOUD_INTERVAL_S,
    Blob,
    Grid,
    MotionDetector,
    PerceptionEvent,
    PerceptionLoop,
    SamplingPolicy,
    SceneMemory,
    TrackState,
    average_hash,
    hamming,
    to_grid,
)


# ──────────────────────────────────────────────────────────────────────────────
# STUBS AND HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, *_a, **_k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Clock:
    """Time under test control — permanence and rate limits need it."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


def _blank(width=32, height=24, value=40):
    return [[value] * width for _ in range(height)]


def _with_square(x0, y0, size=6, value=220, width=32, height=24, background=40):
    """A frame with one bright square — a stand-in for an object."""
    rows = _blank(width, height, background)
    for y in range(y0, min(height, y0 + size)):
        for x in range(x0, min(width, x0 + size)):
            rows[y][x] = value
    return rows


def _grid(rows):
    return to_grid(rows)


def _blob_at(cx, cy, cells=36, width=32, height=24, span=6):
    x0 = int(cx * width) - span // 2
    y0 = int(cy * height) - span // 2
    return Blob(x0, y0, x0 + span - 1, y0 + span - 1, cells, width, height)


# ──────────────────────────────────────────────────────────────────────────────
# FRAME → GRID
# ──────────────────────────────────────────────────────────────────────────────

def test_a_nested_sequence_frame_becomes_a_grid():
    grid = to_grid(_blank(), width=8, height=6)
    assert isinstance(grid, Grid)
    assert grid.width == 8 and grid.height == 6 and grid.size == 48
    assert set(grid.cells) == {40}


def test_a_colour_frame_is_reduced_to_luminance():
    """Frames arrive BGR from OpenCV; pure blue must not read as bright."""
    blue = [[(255, 0, 0)] * 4 for _ in range(4)]
    green = [[(0, 255, 0)] * 4 for _ in range(4)]
    assert to_grid(blue, 4, 4).cells[0] < to_grid(green, 4, 4).cells[0]


def test_an_unreadable_frame_yields_none_rather_than_raising():
    for frame in (None, [], [[]], "not a frame"):
        assert to_grid(frame) is None


def test_a_grid_can_be_rescaled_from_another_grid():
    original = to_grid(_with_square(4, 4), 32, 24)
    assert to_grid(original, 8, 6) is not None


def test_the_scene_hash_is_stable_and_comparable():
    same, contrast = average_hash(_grid(_with_square(4, 4)))
    again, _ = average_hash(_grid(_with_square(4, 4)))
    moved, _ = average_hash(_grid(_with_square(20, 14)))
    assert same == again and hamming(same, again) == 0
    assert hamming(same, moved) > 0
    assert contrast > 0, "a structured frame must report real contrast"


def test_a_featureless_frame_reports_no_contrast():
    """The guard that stops a blank wall reporting a scene cut every second."""
    _bits, contrast = average_hash(_grid(_blank(value=40)))
    assert contrast == 0


# ──────────────────────────────────────────────────────────────────────────────
# MOTION AND BLOBS
# ──────────────────────────────────────────────────────────────────────────────

def test_the_first_frame_is_never_an_event():
    """Nothing to differ against — reporting total motion would storm at startup."""
    result = MotionDetector().update(_grid(_with_square(4, 4)))
    assert result.first_frame and not result.moving and not result.blobs


def test_a_still_scene_produces_no_motion():
    detector = MotionDetector()
    detector.update(_grid(_blank()))
    result = detector.update(_grid(_blank()))
    assert not result.moving and result.blobs == []


def test_a_moving_object_produces_one_blob_where_it_is():
    detector = MotionDetector()
    detector.update(_grid(_blank()))
    result = detector.update(_grid(_with_square(2, 2, size=6)))
    assert result.moving
    assert len(result.blobs) == 1
    assert result.blobs[0].where() == "top left"


def test_two_separate_objects_produce_two_blobs():
    rows = _blank()
    for y in range(2, 8):
        for x in range(2, 8):
            rows[y][x] = 220
    for y in range(16, 22):
        for x in range(24, 30):
            rows[y][x] = 220
    detector = MotionDetector()
    detector.update(_grid(_blank()))
    result = detector.update(_grid(rows))
    assert len(result.blobs) == 2


def test_sensor_speckle_is_not_an_object():
    rows = _blank()
    rows[10][10] = 220          # a single cell, below the minimum blob size
    detector = MotionDetector()
    detector.update(_grid(_blank()))
    assert detector.update(_grid(rows)).blobs == []


def test_a_whole_scene_change_is_flagged():
    detector = MotionDetector()
    detector.update(_grid(_blank(value=20)))
    detector.update(_grid(_blank(value=20)))
    result = detector.update(_grid(_with_square(0, 0, size=24, width=32, height=24)))
    assert result.scene_changed


def test_a_full_frame_change_does_not_overflow_the_stack():
    """Everything moving is ONE component the size of the grid — the flood fill
    is iterative precisely so a light being switched on cannot crash the loop."""
    detector = MotionDetector()
    detector.update(_grid(_blank(value=10)))
    result = detector.update(_grid(_blank(value=250)))
    assert result.blobs and result.blobs[0].cells == 32 * 24


def test_one_object_moving_is_one_object_not_two():
    """Frame differencing marks where a thing WAS and where it IS.

    Undilated, an object moving less than its own width registers as two
    separate arrivals — one person walking past becomes a small crowd.
    """
    detector = MotionDetector()
    detector.update(_grid(_with_square(6, 8, size=6)))
    result = detector.update(_grid(_with_square(9, 8, size=6)))
    assert len(result.blobs) == 1, "the departure and arrival must be one object"


def test_two_objects_stay_two_despite_dilation():
    detector = MotionDetector()
    detector.update(_grid(_blank()))
    rows = _blank()
    for y in range(2, 8):
        for x in range(2, 8):
            rows[y][x] = 220
    for y in range(16, 22):
        for x in range(24, 30):
            rows[y][x] = 220
    assert len(detector.update(_grid(rows)).blobs) == 2


def test_a_low_contrast_scene_never_reports_a_cut():
    """A still camera on a blank wall reported a new scene every second."""
    import random
    rng = random.Random(4)
    detector = MotionDetector()
    for _ in range(12):
        noisy = [[40 + rng.randint(0, 5) for _ in range(32)] for _ in range(24)]
        result = detector.update(_grid(noisy))
        assert not result.scene_changed


def test_a_real_cut_still_registers():
    detector = MotionDetector()
    detector.update(_grid(_with_square(0, 0, size=16)))
    detector.update(_grid(_with_square(0, 0, size=16)))
    result = detector.update(_grid(_with_square(16, 8, size=16)))
    assert result.scene_changed


def test_a_partial_change_is_not_a_scene_cut():
    """A cut means MOST of the picture changed, not a corner of it."""
    detector = MotionDetector()
    detector.update(_grid(_with_square(0, 0, size=16)))
    detector.update(_grid(_with_square(0, 0, size=16)))
    result = detector.update(_grid(_with_square(0, 0, size=17)))
    assert not result.scene_changed


def test_a_real_camera_frame_reduces_without_converting_every_pixel():
    """720p is 2.7M pixels; converting them all cost ~100ms per frame, which
    would make a nonsense of 'local watching is free'."""
    import time
    numpy = pytest.importorskip("numpy")
    frame = numpy.full((720, 1280, 3), 30, dtype=numpy.uint8)
    frame[250:600, 100:280] = 200
    started = time.perf_counter()
    grid = to_grid(frame)
    elapsed = time.perf_counter() - started
    assert grid is not None and grid.size == 32 * 24
    assert max(grid.cells) > min(grid.cells), "the bright region must survive"
    assert elapsed < 0.01, f"reduction took {elapsed * 1000:.0f}ms per frame"


def test_a_greyscale_camera_frame_is_handled():
    numpy = pytest.importorskip("numpy")
    frame = numpy.full((480, 640), 90, dtype=numpy.uint8)
    grid = to_grid(frame)
    assert grid is not None and set(grid.cells) == {90}


def test_a_frame_of_the_wrong_shape_is_rejected():
    numpy = pytest.importorskip("numpy")
    assert to_grid(numpy.zeros((5,), dtype=numpy.uint8)) is None


def test_blob_position_is_described_in_plain_language():
    assert _blob_at(0.5, 0.5).where() == "centre"
    assert _blob_at(0.1, 0.1).where() == "top left"
    assert _blob_at(0.9, 0.9).where() == "bottom right"


# ──────────────────────────────────────────────────────────────────────────────
# SCENE MEMORY AND OBJECT PERMANENCE
# ──────────────────────────────────────────────────────────────────────────────

def test_a_new_object_is_reported_as_an_arrival():
    clock = _Clock()
    scene = SceneMemory()
    events = scene.update([_blob_at(0.5, 0.5)], clock())
    assert [e.kind for e in events] == ["appeared"]
    assert len(scene.active()) == 1


def test_a_moving_object_keeps_its_identity():
    clock = _Clock()
    scene = SceneMemory()
    scene.update([_blob_at(0.4, 0.5)], clock())
    original = scene.active()[0].id
    events = scene.update([_blob_at(0.48, 0.5)], clock.advance(0.5))
    assert events == []
    assert scene.active()[0].id == original
    assert scene.active()[0].sightings == 2


def test_a_far_jump_is_a_different_object():
    clock = _Clock()
    scene = SceneMemory()
    scene.update([_blob_at(0.1, 0.1)], clock())
    events = scene.update([_blob_at(0.9, 0.9)], clock.advance(0.5))
    assert [e.kind for e in events] == ["appeared"]


def test_a_vanished_object_is_occluded_not_gone():
    """The heart of object permanence: absence is not departure."""
    clock = _Clock()
    scene = SceneMemory(permanence_s=8.0)
    scene.update([_blob_at(0.5, 0.5)], clock())
    events = scene.update([], clock.advance(0.5))
    assert events == [], "disappearing must not immediately announce a departure"
    assert scene.active() == []
    assert [t.state for t in scene.remembered()] == [TrackState.OCCLUDED]


def test_a_returning_object_resumes_its_own_track():
    clock = _Clock()
    scene = SceneMemory(permanence_s=8.0)
    scene.update([_blob_at(0.5, 0.5)], clock())
    original = scene.remembered()[0].id
    scene.remembered()[0].label = "a person"
    scene.update([], clock.advance(1.0))                    # steps behind a monitor
    events = scene.update([_blob_at(0.56, 0.5)], clock.advance(1.0))
    assert [e.kind for e in events] == ["returned"]
    assert scene.active()[0].id == original, "same object, not a new one"
    assert scene.active()[0].label == "a person", "its name survived the absence"


def test_permanence_expires_into_a_departure():
    clock = _Clock()
    scene = SceneMemory(permanence_s=5.0)
    scene.update([_blob_at(0.5, 0.5)], clock())
    scene.update([], clock.advance(1.0))
    assert scene.remembered()[0].state is TrackState.OCCLUDED
    events = scene.update([], clock.advance(6.0))
    assert [e.kind for e in events] == ["left"]
    assert scene.remembered() == []


def test_an_object_returning_after_expiry_is_genuinely_new():
    clock = _Clock()
    scene = SceneMemory(permanence_s=3.0)
    scene.update([_blob_at(0.5, 0.5)], clock())
    original = scene.active()[0].id
    scene.update([], clock.advance(1.0))
    scene.update([], clock.advance(5.0))                    # expires
    events = scene.update([_blob_at(0.5, 0.5)], clock.advance(1.0))
    assert [e.kind for e in events] == ["appeared"]
    assert scene.active()[0].id != original


def test_one_blob_cannot_claim_two_tracks():
    clock = _Clock()
    scene = SceneMemory()
    scene.update([_blob_at(0.40, 0.5), _blob_at(0.52, 0.5)], clock())
    assert len(scene.active()) == 2
    scene.update([_blob_at(0.46, 0.5)], clock.advance(0.5))
    states = sorted(t.state.value for t in scene.remembered())
    assert states == ["active", "occluded"], "the nearer track wins; the other waits"


def test_the_scene_describes_both_present_and_remembered():
    clock = _Clock()
    scene = SceneMemory()
    scene.update([_blob_at(0.2, 0.2), _blob_at(0.8, 0.8)], clock())
    scene.update([_blob_at(0.2, 0.2)], clock.advance(0.5))
    description = scene.describe()
    assert "In view:" in description and "Out of sight but remembered:" in description


def test_an_empty_scene_says_so():
    assert "Nothing is moving" in SceneMemory().describe()


def test_gone_tracks_are_pruned():
    clock = _Clock()
    scene = SceneMemory(permanence_s=0.1)
    for index in range(80):
        scene.update([_blob_at(0.05 + (index % 18) * 0.05, 0.5)], clock.advance(1.0))
    scene.prune(keep=60)
    assert len(scene.tracks) <= 60


# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE CLOUD SAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def _event(kind):
    return PerceptionEvent(kind, "now", kind)


def test_the_first_real_change_grounds_the_scene():
    policy = SamplingPolicy()
    wanted, reason = policy.should_sample([_event("appeared")], 1000.0)
    assert wanted and "first look" in reason


def test_nothing_happening_never_calls_the_cloud():
    policy = SamplingPolicy()
    assert policy.should_sample([], 1000.0)[0] is False


def test_mere_motion_is_not_worth_understanding():
    """A curtain in a draught moves every frame and is worth naming never."""
    policy = SamplingPolicy()
    policy.record(1000.0)
    wanted, _reason = policy.should_sample([_event("moved")], 1000.0 + 60)
    assert not wanted


def test_a_change_within_the_floor_is_suppressed():
    policy = SamplingPolicy(min_interval_s=10.0)
    policy.record(1000.0)
    wanted, reason = policy.should_sample([_event("appeared")], 1003.0)
    assert not wanted and "since the last look" in reason
    assert policy.suppressed == 1


def test_a_change_past_the_floor_is_taken():
    policy = SamplingPolicy(min_interval_s=10.0)
    policy.record(1000.0)
    wanted, reason = policy.should_sample([_event("appeared")], 1020.0)
    assert wanted and "appeared" in reason


def test_every_change_kind_triggers_a_look():
    for kind in ("appeared", "returned", "left", "scene_changed"):
        policy = SamplingPolicy(min_interval_s=1.0)
        policy.record(1000.0)
        assert policy.should_sample([_event(kind)], 1010.0)[0], kind


def test_a_static_scene_is_re_grounded_by_the_heartbeat():
    """A description from five minutes ago quietly becomes wrong."""
    policy = SamplingPolicy(heartbeat_s=300.0)
    policy.record(1000.0)
    assert not policy.should_sample([], 1200.0)[0]
    wanted, reason = policy.should_sample([], 1301.0)
    assert wanted and "re-grounding" in reason


def test_the_policy_counts_what_it_saved():
    policy = SamplingPolicy(min_interval_s=10.0)
    policy.record(1000.0)
    for _ in range(4):
        policy.should_sample([_event("appeared")], 1001.0)
    assert policy.stats() == {"cloud_calls": 1, "suppressed": 4,
                              "min_interval_s": 10.0, "heartbeat_s": CLOUD_HEARTBEAT_S}


# ──────────────────────────────────────────────────────────────────────────────
# THE LOOP
# ──────────────────────────────────────────────────────────────────────────────

def _loop(frames, describe=None, clock=None, **kwargs):
    """A loop fed a scripted list of frames, one per tick."""
    queue = list(frames)

    def _source():
        return queue.pop(0) if queue else None

    return PerceptionLoop(_Bus(), frame_source=_source, describe=describe,
                          clock=clock or _Clock(), **kwargs)


async def test_a_tick_with_no_frame_is_counted_not_fatal():
    loop = _loop([])
    assert await loop.tick() == []
    assert loop.blind_frames == 1 and loop.frames == 0


async def test_the_first_frame_produces_no_events():
    loop = _loop([_blank()])
    assert await loop.tick() == []
    assert loop.frames == 1


async def test_an_arrival_is_seen_locally_without_any_cloud_call():
    loop = _loop([_blank(), _with_square(4, 4)])
    await loop.tick()
    events = await loop.tick()
    assert [e.kind for e in events] == ["appeared"]
    assert loop.policy.samples == 0, "no describer wired — must not pretend to look"


async def test_a_change_spends_exactly_one_cloud_call():
    calls = []

    async def _describe(_frame):
        calls.append(1)
        return "A person at the desk."

    loop = _loop([_blank(), _with_square(4, 4)], describe=_describe)
    await loop.tick()
    await loop.tick()
    assert len(calls) == 1
    assert loop.last_description == "A person at the desk."


async def test_the_description_names_the_track_so_it_is_not_asked_twice():
    async def _describe(_frame):
        return "A person at the desk."

    loop = _loop([_blank(), _with_square(4, 4)], describe=_describe)
    await loop.tick()
    await loop.tick()
    assert loop.scene.active()[0].label == "A person at the desk."


async def test_a_still_scene_costs_nothing_at_all():
    calls = []

    async def _describe(_frame):
        calls.append(1)
        return "Nothing."

    loop = _loop([_blank()] * 20, describe=_describe)
    for _ in range(20):
        await loop.tick()
    assert loop.frames == 20 and not calls, "20 frames watched, zero cloud calls"


async def test_a_failing_describer_does_not_stop_the_watching():
    async def _describe(_frame):
        raise RuntimeError("vision provider down")

    loop = _loop([_blank(), _with_square(4, 4), _with_square(4, 4)],
                 describe=_describe)
    for _ in range(3):
        await loop.tick()
    assert loop.frames == 3 and loop.last_description == ""


async def test_a_malformed_frame_is_skipped_not_fatal():
    loop = _loop([_blank(), "not a frame", _with_square(4, 4)])
    for _ in range(3):
        await loop.tick()
    assert loop.frames == 2 and loop.blind_frames == 1


async def test_the_loop_refuses_to_start_without_a_frame_source():
    loop = PerceptionLoop(_Bus(), frame_source=None)
    assert "no camera" in loop.start()
    assert not loop.running


async def test_starting_and_stopping_is_idempotent_and_clean():
    loop = _loop([_blank()] * 200, fps=10.0)
    assert "Watching now" in loop.start()
    assert "already watching" in loop.start()
    await asyncio.sleep(0.05)
    assert "stopped watching" in await loop.stop()
    assert not loop.running
    assert "not watching" in await loop.stop()


async def test_the_report_states_what_local_watching_saved():
    async def _describe(_frame):
        return "A person."

    loop = _loop([_blank()] + [_with_square(4, 4)] * 9, describe=_describe)
    for _ in range(10):
        await loop.tick()
    report = loop.report()
    assert "frames understood for free" in report
    assert loop.policy.samples < loop.frames


async def test_events_are_listed_most_recent_last():
    loop = _loop([_blank(), _with_square(2, 2), _with_square(20, 16)])
    for _ in range(3):
        await loop.tick()
    assert "appeared" in loop.recent()


async def test_an_idle_loop_says_so():
    assert "not watching" in PerceptionLoop(_Bus()).report()
    assert "Nothing has happened" in PerceptionLoop(_Bus()).recent()


# ──────────────────────────────────────────────────────────────────────────────
# DISPATCHER WIRING
# ──────────────────────────────────────────────────────────────────────────────

def _dispatcher():
    from orion_core.bus import OrionBus
    from orion_core.dispatcher import OrionDispatcher
    return OrionDispatcher(
        bus=OrionBus(), memory=None, grabber=None, file_intel=None,
        desktop=None, vision=None, outlook=None, notion=None,
        agent_manager=None, briefing=None,
    )


def test_the_perception_tool_is_declared_and_routed():
    from orion_core.dispatcher import TOOL_DECLARATIONS
    assert "perception" in _dispatcher().handler_table()
    assert any(t["name"] == "perception" for t in TOOL_DECLARATIONS)


def test_reading_perception_is_parallel_but_starting_it_is_not():
    """The per-call classification fix, applied to a tool built after it."""
    assert classify("perception", {"action": "status"}) is ToolClass.PARALLEL
    assert classify("perception", {"action": "scene"}) is ToolClass.PARALLEL
    assert classify("perception", {"action": "start"}) is ToolClass.SERIAL
    assert classify("perception", {"action": "stop"}) is ToolClass.SERIAL


async def test_the_perception_tool_degrades_when_the_loop_is_absent():
    result = await _dispatcher().dispatch("perception", {"action": "status"})
    assert not result.ok and "not available" in result.text


async def test_the_perception_tool_reports_the_scene():
    dispatcher = _dispatcher()
    loop = _loop([_blank(), _with_square(4, 4)])
    dispatcher.perception = loop
    await loop.tick()
    await loop.tick()
    result = await dispatcher.dispatch("perception", {"action": "scene"})
    assert result.ok and "In view:" in result.text


async def test_an_unsupported_perception_action_is_refused():
    dispatcher = _dispatcher()
    dispatcher.perception = _loop([])
    result = await dispatcher.dispatch("perception", {"action": "hallucinate"})
    assert not result.ok and "Unsupported" in result.text


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTER VISION — the instruments, the naming layer, and vision rules
# ──────────────────────────────────────────────────────────────────────────────

def _np():
    return pytest.importorskip("numpy")


def _cv2():
    return pytest.importorskip("cv2")


def _texture(shift=0):
    np, cv2 = _np(), _cv2()
    rng = np.random.default_rng(7)
    base = cv2.resize((rng.random((60, 80)) * 255).astype(np.uint8), (640, 480),
                      interpolation=cv2.INTER_NEAREST)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    return np.roll(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), shift, axis=1)


def test_colours_are_named_and_located():
    np = _np()
    from orion_core.vision_lab import VisionLab
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :320] = (0, 0, 220)          # red on the left (BGR)
    frame[:, 320:] = (220, 60, 0)         # blue on the right
    reading = VisionLab().read(frame)
    names = {c.name: c.share for c in reading.colours}
    assert names["red"] == pytest.approx(0.5, abs=0.02)
    assert names["blue"] == pytest.approx(0.5, abs=0.02)
    assert reading.colour_share("red", "left") == pytest.approx(1.0)
    assert reading.colour_share("red", "right") == 0.0


def test_optical_flow_says_which_way_things_moved():
    from orion_core.vision_lab import VisionLab
    lab = VisionLab()
    lab.read(_texture())
    reading = lab.read(_texture(shift=12))
    assert reading.motion.is_moving and reading.motion.direction == "right"
    assert lab.read(_texture(shift=12)).motion.is_moving is False    # nothing moved since


def test_a_covered_lens_is_told_apart_from_a_dark_room():
    np = _np()
    from orion_core.vision_lab import VisionLab
    reading = VisionLab().read(np.zeros((480, 640, 3), np.uint8))
    assert reading.covered and reading.dark
    assert "covered" in reading.summary()


def test_the_view_as_a_grid_of_numbers():
    np = _np()
    from orion_core.vision_lab import VisionLab
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, 320:] = 255
    grid = VisionLab().read(frame).number_grid().splitlines()
    assert len(grid) == 12 and all(len(row) == 16 for row in grid)
    assert grid[0].startswith("0000") and grid[0].endswith("9999")


def test_every_camera_view_renders_and_a_fault_never_raises():
    np = _np()
    from orion_core.object_detection import Detection
    from orion_core.vision_lab import OVERLAY_MODES, VisionLab, render
    frame = _texture()
    reading = VisionLab().read(frame)
    box = [Detection("cup", 0.9, (0.1, 0.1, 0.4, 0.5))]
    for mode in OVERLAY_MODES:
        view = render(frame, mode, reading, box, (), width=220)
        assert view.shape == (165, 220, 3), mode
    assert render("not a frame", "edges") == "not a frame"
    assert render(frame, "edges", None).dtype == np.uint8


def test_zone_words_are_understood_and_typos_refused():
    from orion_core.vision_lab import normalise_zone
    assert normalise_zone("top left") == "top-left"
    assert normalise_zone("middle") == "centre"
    assert normalise_zone("anywhere") is None
    with pytest.raises(ValueError):
        normalise_zone("upstairs")


# ── object detection: decoding, without needing the model ──────────────────

def test_detector_output_layouts_are_recognised_from_shape():
    from orion_core.object_detection import output_layout
    assert output_layout((1, 84, 8400)) == "v8"
    assert output_layout((1, 8400, 85)) == "yolox"
    assert output_layout((1, 25200, 85)) == "v5"
    with pytest.raises(ValueError):
        output_layout((1, 7, 7))


def test_ultralytics_output_is_decoded_and_duplicates_suppressed():
    np = _np()
    from orion_core.object_detection import COCO_LABELS, decode
    out = np.zeros((1, 84, 10), np.float32)
    cup, person = COCO_LABELS.index("cup"), COCO_LABELS.index("person")
    out[0, :4, 3] = (320, 320, 100, 200)
    out[0, 4 + cup, 3] = 0.9
    out[0, :4, 4] = (325, 322, 100, 200)        # the same cup, found twice
    out[0, 4 + cup, 4] = 0.6
    out[0, :4, 5] = (100, 100, 40, 40)
    out[0, 4 + person, 5] = 0.5
    boxes, scores, classes = decode(out, "v8", 640, 0.4)
    assert sorted(COCO_LABELS[c] for c in classes) == ["cup", "person"]
    assert boxes[0].tolist() == pytest.approx([270, 220, 370, 420])


def test_yolox_grid_offsets_are_decoded():
    np = _np()
    from orion_core.object_detection import COCO_LABELS, decode
    out = np.zeros((1, 8400, 85), np.float32)
    row = 5 * 80 + 10                                 # stride-8 cell (x=10, y=5)
    out[0, row, :4] = (0.5, 0.5, np.log(4), np.log(4))
    out[0, row, 4] = 1.0
    out[0, row, 5 + COCO_LABELS.index("person")] = 0.9
    boxes, scores, classes = decode(out, "yolox", 640, 0.4)
    assert len(boxes) == 1 and COCO_LABELS[classes[0]] == "person"
    assert boxes[0].tolist() == pytest.approx([84 - 16, 44 - 16, 84 + 16, 44 + 16])


def test_detection_maps_boxes_back_to_the_camera_frame(tmp_path):
    np = _np()
    _cv2()
    from orion_core.object_detection import COCO_LABELS, ObjectDetector

    class _Session:
        def run(self, _names, feeds):
            out = np.zeros((1, 84, 1), np.float32)
            out[0, :4, 0] = (320, 320, 64, 64)       # centre of the 640 canvas
            out[0, 4 + COCO_LABELS.index("laptop"), 0] = 0.8
            return [out]

    detector = ObjectDetector(model_dir=tmp_path)
    detector._session, detector._layout, detector._input_name = _Session(), "v8", "images"
    detector._ensure_session = lambda: True
    found = detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert [d.label for d in found] == ["laptop"]
    assert found[0].zone == "centre"
    assert found[0].centre == pytest.approx((0.5, 0.5), abs=0.01)


def test_spoken_names_map_to_detector_classes():
    from orion_core.object_detection import canonical_label
    assert canonical_label("phone") == "cell phone"
    assert canonical_label("Cups") == "cup"
    assert canonical_label("someone") == "person"
    assert canonical_label("unicorn") is None


class _FakeResponse:
    """What urllib's opener hands back: a readable context manager."""

    def __init__(self, payload, headers=None):
        import io
        self._body = io.BytesIO(payload)
        self.headers = headers or {}

    def read(self, size=-1):
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_detector_is_only_ever_installed_verified(tmp_path, monkeypatch):
    import dataclasses
    import hashlib
    from orion_core import object_detection as od
    payload = b"onnx-bytes" * 100
    spec = dataclasses.replace(od.DEFAULT_MODEL, size=len(payload),
                               sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(od, "DEFAULT_MODEL", spec)
    detector = od.ObjectDetector(model_dir=tmp_path)
    tampered = detector.install_model(
        opener=lambda *_a, **_k: _FakeResponse(b"x" * len(payload)))
    assert "couldn't install" in tampered and not list(tmp_path.iterdir())
    installed = detector.install_model(opener=lambda *_a, **_k: _FakeResponse(payload))
    assert "checksum verified" in installed
    assert detector.model_path() == tmp_path / spec.filename


def test_an_interrupted_model_download_leaves_nothing_behind(tmp_path):
    from orion_core.model_store import ModelDownloadError, download_verified
    with pytest.raises(ModelDownloadError):
        download_verified("https://example.invalid/m.task", tmp_path / "m.task",
                          opener=lambda *_a, **_k: _FakeResponse(
                              b"x" * 400, {"Content-Length": "1000"}))
    assert not list(tmp_path.iterdir()), \
        "a truncated model was left where exists() would accept it"


# ── pose: gestures from landmarks ───────────────────────────────────────────

def _body(moves=None):
    """33 visible landmarks: standing, arms down; *moves* repositions some."""
    from orion_core import pose_tracking as pt
    body = [[0.5, 0.5, 1.0] for _ in range(33)]
    points = {pt.NOSE: (0.5, 0.25), pt.LEFT_SHOULDER: (0.58, 0.4),
              pt.RIGHT_SHOULDER: (0.42, 0.4), pt.LEFT_WRIST: (0.6, 0.7),
              pt.RIGHT_WRIST: (0.4, 0.7)}
    points.update(moves or {})
    for index, (x, y) in points.items():
        body[index] = [x, y, 1.0]
    return [tuple(p) for p in body]


def test_body_gestures_are_read_from_landmarks():
    from orion_core import pose_tracking as pt
    assert pt.interpret(_body()) == {"person"}
    left_up = pt.interpret(_body({pt.LEFT_WRIST: (0.6, 0.2)}))
    assert {"hand_raised", "left_hand_up"} <= left_up and "right_hand_up" not in left_up
    both = pt.interpret(_body({pt.LEFT_WRIST: (0.6, 0.1), pt.RIGHT_WRIST: (0.4, 0.1)}))
    assert "both_hands_up" in both
    t_pose = pt.interpret(_body({pt.LEFT_WRIST: (0.9, 0.4), pt.RIGHT_WRIST: (0.1, 0.4)}))
    assert "arms_out" in t_pose and "hand_raised" not in t_pose


def test_an_unseen_wrist_is_not_a_raised_hand():
    from orion_core import pose_tracking as pt
    body = [list(p) for p in _body({pt.LEFT_WRIST: (0.6, 0.1)})]
    body[pt.LEFT_WRIST][2] = 0.1                       # low visibility: a guess
    assert "hand_raised" not in pt.interpret([tuple(p) for p in body])


# ── vision rules ─────────────────────────────────────────────────────────────

def _det(label, x=0.2, y=0.5, confidence=0.9):
    from orion_core.object_detection import Detection
    return Detection(label, confidence, (x - 0.05, y - 0.05, x + 0.05, y + 0.05))


def _obs(now, detections=None, **kw):
    from orion_core.vision_rules import Observation
    return Observation(now=now, detections=detections, **kw)


def test_a_rule_fires_once_per_arrival_not_every_frame(tmp_path):
    from orion_core.vision_rules import VisionRules
    rules = VisionRules(tmp_path / "rules.json")
    rules.add(trigger="object", target="cup", hold_s=1.0, cooldown_s=5)
    cup = [_det("cup")]
    assert rules.evaluate(_obs(0.0, cup)) == []           # holding
    assert len(rules.evaluate(_obs(1.2, cup))) == 1       # held 1s: fires
    for t in (2, 10, 60):
        assert rules.evaluate(_obs(t, cup)) == []         # still there: no repeat
    assert rules.evaluate(_obs(61, [])) == []             # gone...
    assert rules.evaluate(_obs(64.5, [])) == []           # ...long enough to re-arm
    assert rules.evaluate(_obs(65, cup)) == []
    assert len(rules.evaluate(_obs(66.1, cup))) == 1      # a new arrival fires again


def test_a_frame_the_detector_skipped_changes_nothing(tmp_path):
    from orion_core.vision_rules import VisionRules
    rules = VisionRules(tmp_path / "rules.json")
    rules.add(trigger="object", target="person", hold_s=2.0)
    rules.evaluate(_obs(0.0, [_det("person")]))
    rules.evaluate(_obs(1.0, None))                      # not measured: not a reset
    assert len(rules.evaluate(_obs(2.1, [_det("person")]))) == 1


def test_something_leaving_is_noticed_only_after_it_was_seen(tmp_path):
    from orion_core.vision_rules import VisionRules
    rules = VisionRules(tmp_path / "rules.json")
    rules.add(trigger="object_gone", target="phone", hold_s=10)
    assert rules.evaluate(_obs(0, [])) == []              # never seen: nothing gone
    rules.evaluate(_obs(1, [_det("cell phone")]))
    rules.evaluate(_obs(2, []))
    assert rules.evaluate(_obs(11, [])) == []             # 9s: could be someone in the way
    fired = rules.evaluate(_obs(12.5, []))
    assert len(fired) == 1 and "gone" in fired[0].detail
    assert rules.evaluate(_obs(100, [])) == []            # once, until it is seen again


def test_zones_counts_colours_and_poses(tmp_path):
    from types import SimpleNamespace
    from orion_core.pose_tracking import PersonPose, PoseReading
    from orion_core.vision_rules import VisionRules
    rules = VisionRules(tmp_path / "rules.json")
    rules.add(trigger="object", target="cup", zone="right", hold_s=0)
    rules.add(trigger="count", target="person", count=2, hold_s=0)
    rules.add(trigger="pose", target="raise my hand", hold_s=0)
    assert rules.evaluate(_obs(0, [_det("cup", x=0.2)])) == []       # wrong zone
    two = [_det("person", x=0.2), _det("person", x=0.8)]
    assert [f.rule.trigger for f in rules.evaluate(_obs(1, two))] == ["count"]
    hand = PoseReading((PersonPose(((0.5, 0.5, 1.0),), frozenset({"person", "hand_raised"})),))
    assert [f.rule.trigger for f in rules.evaluate(_obs(2, None, pose=hand))] == ["pose"]
    red = SimpleNamespace(colour_share=lambda name, zone=None: 0.4 if name == "red" else 0.0,
                          dark=False, covered=False, motion=None)
    rules.add(trigger="colour", target="red", hold_s=0)
    assert [f.rule.trigger for f in rules.evaluate(_obs(3, None, reading=red))] == ["colour"]


def test_rules_are_validated_when_made_and_survive_a_restart(tmp_path):
    from orion_core.vision_rules import VisionRules
    path = tmp_path / "rules.json"
    rules = VisionRules(path)
    for bad in (dict(trigger="object", target="unicorn"),
                dict(trigger="teleport"),
                dict(trigger="object", target="cup", zone="upstairs"),
                dict(trigger="object", target="cup", action="workflow")):
        with pytest.raises(ValueError):
            rules.add(**bad)
    rule = rules.add(trigger="object", target="mug", action="workflow",
                     workflow="focus mode", cooldown_s=1)
    assert rule.target == "cup" and rule.cooldown_s >= 5          # floor enforced
    rules.set_watch_on_startup(True)
    again = VisionRules(path)
    assert again.watch_on_startup and again.rules[0].workflow == "focus mode"
    assert again.needs() == {"objects"}


# ── the loop, with the naming layer ─────────────────────────────────────────

class _FakeDetector:
    available = True

    def __init__(self, found):
        self.found, self.calls = list(found), 0

    def detect(self, _frame):
        self.calls += 1
        return list(self.found)

    def describe_state(self):
        return "fake detector ready"


async def test_an_arrival_named_locally_costs_no_cloud_call():
    from orion_core.object_detection import Detection
    asked = []

    async def _describe(_frame):
        asked.append(1)
        return "a person"

    whole = Detection("person", 0.9, (0.0, 0.0, 1.0, 1.0))
    loop = _loop([_blank(), _with_square(4, 4)], describe=_describe,
                 objects=_FakeDetector([whole]))
    loop.overlay_needs = {"objects"}
    await loop.tick()
    events = await loop.tick()
    assert any(e.kind == "named" and "person" in e.detail for e in events)
    assert asked == [], "the cloud was paid to name something already named locally"
    assert loop.named_locally == 1 and loop.scene.active()[0].label == "person"


async def test_without_a_detector_the_cloud_still_names_arrivals():
    asked = []

    async def _describe(_frame):
        asked.append(1)
        return "a person"

    loop = _loop([_blank(), _with_square(4, 4)], describe=_describe)
    await loop.tick()
    await loop.tick()
    assert asked == [1]


async def test_detection_runs_only_when_something_needs_it():
    detector = _FakeDetector([])
    loop = _loop([_blank()] * 4, objects=detector)
    for _ in range(4):
        await loop.tick()
    assert detector.calls == 0, "the detector ran with no rule or view asking for it"


async def test_a_vision_rule_speaks_when_its_object_appears(tmp_path):
    from orion_core.object_detection import Detection
    from orion_core.vision_rules import VisionRules
    rules = VisionRules(tmp_path / "rules.json")
    rules.add(trigger="object", target="person", hold_s=0, message="Hello there.")
    clock = _Clock()
    loop = _loop([_blank(), _with_square(4, 4)], clock=clock, rules=rules,
                 objects=_FakeDetector([Detection("person", 0.9, (0, 0, 1, 1))]))
    await loop.tick()
    clock.advance(1)
    await loop.tick()
    assert ("Hello there.",) in loop.bus.speak_request.emitted
    assert any(e.kind == "rule" for e in loop.events)


async def test_a_vision_rule_can_start_a_workflow(tmp_path):
    from orion_core.object_detection import Detection
    from orion_core.vision_rules import VisionRules
    started = []
    rules = VisionRules(tmp_path / "rules.json")
    rules.add(trigger="object", target="cup", hold_s=0, action="workflow", workflow="tea time")
    loop = _loop([_blank(), _blank()], rules=rules,
                 objects=_FakeDetector([Detection("cup", 0.9, (0, 0, 0.2, 0.2))]),
                 on_workflow=lambda name, text: started.append((name, text)))
    await loop.tick()
    await loop.tick()
    assert started and started[0][0] == "tea time" and "cup" in started[0][1]


async def test_watching_switches_the_camera_on_and_back_off():
    class _Camera:
        def __init__(self):
            self.on = False

        def is_capturing(self):
            return self.on

        def start(self):
            self.on = True
            return type("R", (), {"ok": True})()

        def stop(self):
            self.on = False

    camera = _Camera()
    loop = _loop([], camera=camera)
    assert "switched the camera on" in loop.start()
    assert camera.on
    await loop.stop()
    assert not camera.on


async def test_the_tool_measures_detects_and_sets_rules(tmp_path):
    from orion_core.object_detection import Detection
    from orion_core.vision_rules import VisionRules
    np = _np()
    _cv2()
    dispatcher = _dispatcher()
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :320] = (0, 0, 220)
    loop = PerceptionLoop(_Bus(), frame_source=lambda: frame, clock=_Clock(),
                          objects=_FakeDetector([Detection("cup", 0.9, (0.1, 0.4, 0.3, 0.6))]),
                          rules=VisionRules(tmp_path / "rules.json"))
    dispatcher.perception = loop
    measured = await dispatcher.dispatch("perception", {"action": "grid"})
    assert measured.ok and "red" in measured.text and "grid of numbers" in measured.text
    seen = await dispatcher.dispatch("perception", {"action": "detect"})
    assert seen.ok and "a cup (left)" in seen.text
    refused = await dispatcher.dispatch("perception", {"action": "add_rule", "when": "object",
                                                      "target": "unicorn"})
    assert not refused.ok and "can't set that rule up" in refused.text
    added = await dispatcher.dispatch("perception", {"action": "add_rule", "when": "object",
                                                    "target": "cup", "zone": "left"})
    assert added.ok and "Rule added" in added.text
    listed = await dispatcher.dispatch("perception", {"action": "rules"})
    assert "r1" in listed.text
    await loop.stop()


def test_changing_rules_or_the_camera_is_serial_but_listing_is_not():
    assert classify("perception", {"action": "rules"}) is ToolClass.PARALLEL
    for action in ("add_rule", "remove_rule", "install_detector", "detect", "overlay"):
        assert classify("perception", {"action": action}) is ToolClass.SERIAL, action
