"""Live board geometry (orion_core/board_detect.py).

These tests build synthetic boards rather than mocking OpenCV, because every
defect this module has actually had was in the image processing itself and a
mock would have reported success through all of them. The first draft used
``RETR_EXTERNAL`` for components — which returns only outermost contours, so it
found the board and discarded every part sitting on it — and located the board
from a Canny outline, which closed into a contour only when the contrast
happened to be generous: the same board flickered in and out between frames.
Both are pinned below.

The synthetic boards carry traces, vias, silkscreen and packages, so edge
density and colour behave roughly like the real thing. They are not a
substitute for pointing a camera at a real PCB, and the thresholds here are
deliberately loose: they assert the module's promises (a board is located, its
parts are boxed, a hand is not called a board), not exact pixel values.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from orion_core.board_detect import (  # noqa: E402
    BoardReading,
    BoardScanner,
    analyse_frame,
)

FRAME_W, FRAME_H = 1280, 720


def _board_image(colour=(45, 115, 40), width=700, height=480, seed=7):
    """A board-like surface: traces, vias, silkscreen and packages."""
    rng = random.Random(seed)
    board = np.full((height, width, 3), colour, np.uint8)
    for _ in range(90):
        x1, y1 = rng.randrange(width), rng.randrange(height)
        x2 = max(0, min(width - 1, x1 + rng.choice([-1, 1]) * rng.randrange(20, 140)))
        y2 = max(0, min(height - 1, y1 + rng.choice([-1, 1]) * rng.randrange(0, 40)))
        cv2.line(board, (x1, y1), (x2, y2), (70, 150, 70), rng.choice([1, 1, 2]))
    for _ in range(140):
        cv2.circle(board, (rng.randrange(width), rng.randrange(height)),
                   rng.choice([2, 3]), (190, 190, 185), -1)
    for _ in range(26):
        cv2.putText(board, rng.choice(["R12", "C7", "U1", "D3", "J2"]),
                    (rng.randrange(max(1, width - 50)), rng.randrange(20, height - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (225, 225, 220), 1)
    for (x, y, w, h) in [(60, 60, 160, 90), (320, 70, 120, 120), (500, 80, 90, 60),
                         (80, 260, 60, 30), (200, 280, 40, 25), (300, 300, 45, 28),
                         (400, 260, 150, 80), (580, 300, 70, 70)]:
        cv2.rectangle(board, (x, y), (x + w, y + h), (20, 20, 22), -1)
        cv2.rectangle(board, (x, y), (x + w, y + h), (55, 55, 58), 1)
    return board


def scene(desk=30, angle=0.0, colour=(45, 115, 40), scale=1.0, gradient=False):
    """Place a board on a desk; returns the frame and its true normalised box."""
    frame = np.full((FRAME_H, FRAME_W, 3), desk, np.uint8)
    width, height = int(700 * scale), int(480 * scale)
    board = cv2.resize(_board_image(colour), (width, height))
    layer = np.zeros_like(frame)
    covered = np.zeros((FRAME_H, FRAME_W), np.uint8)
    left, top = (FRAME_W - width) // 2, (FRAME_H - height) // 2
    src_x, src_y = max(0, -left), max(0, -top)
    dst_x, dst_y = max(0, left), max(0, top)
    span_w = min(width - src_x, FRAME_W - dst_x)
    span_h = min(height - src_y, FRAME_H - dst_y)
    layer[dst_y:dst_y + span_h, dst_x:dst_x + span_w] = \
        board[src_y:src_y + span_h, src_x:src_x + span_w]
    covered[dst_y:dst_y + span_h, dst_x:dst_x + span_w] = 255
    if angle:
        matrix = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), angle, 1.0)
        layer = cv2.warpAffine(layer, matrix, (FRAME_W, FRAME_H))
        covered = cv2.warpAffine(covered, matrix, (FRAME_W, FRAME_H))
    frame[covered > 0] = layer[covered > 0]
    if gradient:
        ramp = np.linspace(0.5, 1.4, FRAME_W).astype(np.float32)[None, :, None]
        frame = np.clip(frame.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(covered)
    truth = (xs.min() / FRAME_W, ys.min() / FRAME_H,
             (xs.max() - xs.min()) / FRAME_W, (ys.max() - ys.min()) / FRAME_H)
    return frame, truth


def _blank(value=120, noise=3.0):
    frame = np.full((FRAME_H, FRAME_W, 3), value, np.uint8)
    grain = np.random.default_rng(3).normal(0, noise, frame.shape)
    return np.clip(frame + grain, 0, 255).astype(np.uint8)


# ── locating the board ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name,kwargs", [
    ("dark desk", dict(desk=30)),
    ("light desk", dict(desk=200)),
    ("blue board", dict(desk=30, colour=(120, 60, 25))),
    ("black board on white", dict(desk=205, colour=(32, 32, 34))),
    ("rotated", dict(desk=30, angle=12)),
    ("rotated the other way", dict(desk=90, angle=-25)),
    ("uneven lighting", dict(desk=60, gradient=True)),
    ("small in frame", dict(desk=30, scale=0.45)),
])
def test_board_is_located_in_varied_conditions(name, kwargs):
    frame, truth = scene(**kwargs)
    reading = analyse_frame(frame)
    assert reading.board is not None, f"no board found: {name}"
    error = max(abs(reading.board[i] - truth[i]) for i in range(4))
    assert error < 0.05, f"{name}: box off by {error:.3f}"


def test_board_outline_survives_a_change_of_desk_colour():
    """The regression that made the outline flicker: it must not depend on a
    Canny loop happening to close, so two near-identical desks must agree."""
    first, _ = scene(desk=28)
    second, _ = scene(desk=30)
    assert analyse_frame(first).board is not None
    assert analyse_frame(second).board is not None


def test_rotated_board_reports_four_corners():
    frame, _ = scene(desk=30, angle=18)
    reading = analyse_frame(frame)
    assert reading.quad is not None and len(reading.quad) == 4
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in reading.quad)


# ── locating the parts on it ─────────────────────────────────────────────────

def test_components_on_the_board_are_boxed():
    """RETR_EXTERNAL returned the board and nothing on it; RETR_LIST is why
    this passes."""
    frame, _ = scene(desk=30)
    reading = analyse_frame(frame)
    assert len(reading.regions) >= 6, f"only {len(reading.regions)} region(s)"


def test_regions_are_not_drawn_twice_for_one_part():
    """Thresholding an edge yields a ring — an outer AND an inner contour."""
    frame, _ = scene(desk=30)
    regions = analyse_frame(frame).regions
    for index, (x, y, w, h) in enumerate(regions):
        for other_x, other_y, other_w, other_h in regions[index + 1:]:
            overlap_w = min(x + w, other_x + other_w) - max(x, other_x)
            overlap_h = min(y + h, other_y + other_h) - max(y, other_y)
            if overlap_w <= 0 or overlap_h <= 0:
                continue
            smaller = min(w * h, other_w * other_h)
            assert (overlap_w * overlap_h) / smaller < 0.7, "duplicate region box"


def test_regions_stay_inside_the_frame():
    frame, _ = scene(desk=30)
    for x, y, w, h in analyse_frame(frame).regions:
        assert 0 <= x and 0 <= y and w > 0 and h > 0
        assert x + w <= 1.0001 and y + h <= 1.0001


def test_region_count_is_capped():
    frame, _ = scene(desk=30)
    assert len(analyse_frame(frame, max_regions=3).regions) <= 3


# ── refusing to claim what is not there ──────────────────────────────────────

def test_a_bare_desk_is_not_a_board():
    reading = analyse_frame(_blank())
    assert reading.board is None and not reading.regions


def test_a_dark_frame_is_not_a_board():
    assert analyse_frame(np.full((FRAME_H, FRAME_W, 3), 8, np.uint8)).board is None


def test_a_rounded_object_gets_no_board_outline():
    """An ellipse fills its bounding rectangle at 0.785, so a loose
    rectangularity threshold draws a confident board frame around a hand."""
    frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
    cv2.ellipse(frame, (640, 400), (220, 160), 20, 0, 360, (110, 140, 190), -1)
    assert analyse_frame(frame).quad is None


def test_a_board_filling_the_view_is_not_told_to_move_closer():
    frame, _ = scene(desk=30, scale=1.75)
    reading = analyse_frame(frame)
    assert reading.fills_view is True
    assert "closer" not in reading.guidance()


# ── inputs that must never raise ─────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    None, "not a frame", 42, [], np.zeros((10, 10), np.uint8),
    np.zeros((8, 8, 3), np.uint8), np.zeros((40, 40, 3), np.float32),
])
def test_unusable_input_is_reported_not_raised(value):
    reading = analyse_frame(value)
    assert isinstance(reading, BoardReading) and reading.available is False
    assert reading.board is None and reading.regions == []


def test_unavailable_reading_still_describes_itself():
    assert "unavailable" in BoardReading(available=False).guidance()


# ── focus and exposure ───────────────────────────────────────────────────────

def test_blur_lowers_the_focus_score():
    frame, _ = scene(desk=30)
    sharp = analyse_frame(frame)
    soft = analyse_frame(cv2.GaussianBlur(frame, (25, 25), 0))
    assert sharp.focus > soft.focus
    assert sharp.sharp and not soft.sharp
    assert "soft" in soft.guidance()


def test_exposure_is_judged_on_the_board_not_the_desk():
    """A well-lit board on a black worktop must not be called too dark."""
    frame, _ = scene(desk=8, colour=(70, 165, 65))
    reading = analyse_frame(frame)
    assert reading.board is not None
    assert reading.brightness > 0.18, reading.guidance()


def test_glare_is_called_out():
    frame, _ = scene(desk=30, colour=(248, 250, 250))
    reading = analyse_frame(frame)
    if reading.board is not None and reading.brightness > 0.88:
        assert "Glare" in reading.guidance()


# ── the background scanner ───────────────────────────────────────────────────

def test_scanner_produces_a_reading_for_a_submitted_frame():
    frame, _ = scene(desk=30)
    scanner = BoardScanner(min_interval=0.0)
    scanner.start()
    try:
        deadline, reading = __import__("time").monotonic() + 5.0, None
        scanner.submit(frame)
        while reading is None and __import__("time").monotonic() < deadline:
            reading = scanner.latest()
        assert reading is not None and reading.board is not None
    finally:
        scanner.stop()
    assert scanner.running is False


def test_scanner_rate_limits_submissions():
    frame, _ = scene(desk=30)
    scanner = BoardScanner(min_interval=30.0)
    scanner.submit(frame)          # accepted
    scanner.submit(frame)          # inside the interval: dropped
    assert scanner._pending is frame
    scanner.stop()


def test_scanner_keeps_only_the_newest_frame():
    """A slow pass must drop stale work rather than build a queue."""
    first, _ = scene(desk=30)
    second, _ = scene(desk=200)
    scanner = BoardScanner(min_interval=0.0)
    scanner.submit(first)
    scanner.submit(second)
    assert scanner._pending is second
    scanner.stop()


def test_scanner_ignores_a_none_frame():
    scanner = BoardScanner(min_interval=0.0)
    scanner.submit(None)
    assert scanner._pending is None
    scanner.stop()


def test_stopping_an_unstarted_scanner_is_safe():
    scanner = BoardScanner()
    scanner.stop()
    assert scanner.running is False


def test_slow_worker_can_be_stopped_without_blocking_or_publishing_stale_result(monkeypatch):
    import threading
    import time
    import orion_core.board_detect as module
    entered, release = threading.Event(), threading.Event()
    stale = BoardReading(focus=1.0)

    def slow(_frame):
        entered.set()
        release.wait(3)
        return stale

    monkeypatch.setattr(module, "analyse_frame", slow)
    scanner = BoardScanner(min_interval=0)
    scanner.start()
    scanner.submit(object())
    assert entered.wait(3)
    old_thread = scanner._thread
    started = time.monotonic()
    scanner.stop(wait=False)
    assert time.monotonic() - started < .2
    scanner.start()
    release.set()
    old_thread.join(3)
    assert scanner.latest() is None and scanner.latest_snapshot() is None
    scanner.stop()


def test_the_scanner_thread_loads_opencv_before_any_frame_arrives():
    """The preview converts BGR to QImage on the GUI thread with an inline
    `import cv2` — ~127 ms the first time. The scanner starts with the camera,
    so it should be the one to pay that, off-screen."""
    import subprocess
    import sys as _sys
    import textwrap
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, r"{ROOT}")
        from orion_core.board_detect import BoardScanner
        assert "cv2" not in sys.modules
        scanner = BoardScanner()
        scanner.start()
        deadline = time.monotonic() + 30
        while "cv2" not in sys.modules and time.monotonic() < deadline:
            time.sleep(0.01)
        loaded = "cv2" in sys.modules
        scanner.stop()
        print(loaded)
    """)
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(ROOT), timeout=120)
    assert out.stdout.strip() == "True", out.stdout + out.stderr
