"""Live, local geometry for the electronics workbench camera.

Every number here comes from the pixels in front of the camera — edges,
contours, contrast — and nothing here identifies a component, reads a part
number or judges a circuit. That distinction is the whole point: this runs
continuously on the live preview so the user can *see* ORION looking at the
board, while component identification stays with the deliberate, model-backed
capture in :mod:`orion_core.electronics_inspection`.

So the vocabulary is deliberately geometric — a "region" is a rectangle with
board-like contrast, not a resistor. The workbench draws these in a different
colour and weight from model findings for the same reason.

Coordinates are normalised ``[x, y, width, height]`` against the frame handed
in, matching the inspection report's contract, so the canvas draws both with
one code path. OpenCV is imported inside the worker: importing this module
must stay free for a workbench that is never opened.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

#: Detection runs on a downscaled copy; small enough that a full pass costs a
#: few milliseconds, large enough to keep small parts as distinct blobs.
WORK_WIDTH = 384

#: A region must occupy at least this fraction of the working image to be
#: drawn. Below it, sensor noise and solder-mask speckle dominate and the
#: overlay turns into confetti.
MIN_REGION_AREA = 0.0009
MAX_REGION_AREA = 0.30

#: A board candidate has to be a substantial part of the view; anything less is
#: a bright object on the desk, not the thing being inspected.
MIN_BOARD_AREA = 0.06

#: Laplacian variance below this reads as soft focus on typical webcam frames.
FOCUS_SOFT = 55.0


@dataclass(frozen=True)
class BoardReading:
    """One frame's worth of local geometry. All boxes are normalised."""

    board: list[float] | None = None
    quad: list[list[float]] | None = None
    regions: list[list[float]] = field(default_factory=list)
    focus: float = 0.0
    brightness: float = 0.0
    width: int = 0
    height: int = 0
    available: bool = True
    #: The board is too close to have a visible outline — detail fills the view.
    fills_view: bool = False

    @property
    def sharp(self) -> bool:
        return self.focus >= FOCUS_SOFT

    def guidance(self) -> str:
        """One honest sentence about framing — never about the circuit."""
        if not self.available:
            return "Live geometry is unavailable; capture still works."
        if self.brightness < 0.18:
            return "Too dark to resolve markings — add diffuse light."
        if self.brightness > 0.88:
            return "Glare is washing out the surface — angle the light away."
        if self.board is None:
            return "No board-shaped edge yet — fill more of the frame."
        if self.fills_view:
            return ("Filling the frame" + ("" if self.sharp else ", edges still soft")
                    + f" - {len(self.regions)} region(s) of interest.")
        coverage = self.board[2] * self.board[3]
        if coverage < 0.18:
            return "Board found — move closer so it fills the guide."
        if not self.sharp:
            return "Board framed, edges still soft — hold steady for focus."
        return f"Board framed and sharp - {len(self.regions)} region(s) of interest."


_EMPTY = BoardReading(available=False)


def _normalise(x: float, y: float, w: float, h: float,
               width: float, height: float) -> list[float] | None:
    """Clamp a working-image rect into a normalised box, or drop it."""
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None
    nx, ny = max(0.0, x / width), max(0.0, y / height)
    nw, nh = min(w / width, 1.0 - nx), min(h / height, 1.0 - ny)
    if nw <= 0 or nh <= 0:
        return None
    return [round(nx, 5), round(ny, 5), round(nw, 5), round(nh, 5)]


#: A candidate this deeply inside an already-kept box is the same feature seen
#: again, not a second one.
_CONTAINED = 0.7


def _contained(box: list[float], kept: list[list[float]]) -> bool:
    """True when most of ``box`` already sits inside a box we are drawing.

    Thresholding an edge gives each package a *ring*, and a ring has both an
    outer and an inner boundary — two contours for one part. Suppressing the
    smaller, enclosed one is what stops every component being outlined twice.
    """
    x, y, w, h = box
    own = w * h
    if own <= 0:
        return False
    for other in kept:
        ox, oy, ow, oh = other
        overlap_w = min(x + w, ox + ow) - max(x, ox)
        overlap_h = min(y + h, oy + oh) - max(y, oy)
        if overlap_w > 0 and overlap_h > 0 and (overlap_w * overlap_h) / own >= _CONTAINED:
            return True
    return False


#: Mean Lab distance from the frame's border colour, below which the camera is
#: looking at one uniform surface — a desk, a wall, a lens cap — and there is
#: no board to outline. Measured: a bare desk sits near 1, a board in view
#: above 20.
_FLAT_SURFACE = 4.0

#: How completely a contour must fill its own minimum-area rectangle before
#: that rectangle is honest to draw as the board's outline. Above an ellipse's
#: 0.785, deliberately: at a looser threshold a hand in shot gets a confident
#: four-corner board frame drawn round it. A real board, at any rotation,
#: fills its minimum-area rectangle almost completely.
_RECTANGULAR = 0.82


#: The board pass only needs to place a large rectangle, so it runs on a
#: half-size copy: the Lab conversion and per-pixel distance dominate its cost
#: and both scale with area.
BOARD_WIDTH = 192


def _find_board(cv2: Any, np: Any, source: Any, width: int,
                height: int) -> tuple[Any, Any, Any]:
    """Locate the board as the large object that differs from its surroundings.

    Not by its edges: a board outline traced with Canny only closes into a
    contour when the contrast happens to be generous, so an edge-based board
    box appears and vanishes between frames on the same board — which reads as
    broken. What is dependable is that the desk owns the frame's border and the
    board does not look like the desk, so the board is found by colour distance
    from the border in Lab (perceptual, so it survives tinted lighting) rather
    than by hoping for an unbroken outline.
    """
    scale = min(1.0, BOARD_WIDTH / float(source.shape[1]))
    work = (cv2.resize(source, (max(8, int(source.shape[1] * scale)),
                                max(8, int(source.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
            if scale < 1.0 else source)
    rows, columns = work.shape[:2]
    area = float(rows * columns)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.int16)
    band = max(3, rows // 24)
    border = np.concatenate([
        lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
        lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3)])
    distance = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    if float(distance.mean()) < _FLAT_SURFACE:
        return None, None, None          # one flat surface fills the view
    _level, mask = cv2.threshold(
        np.clip(distance, 0, 255).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close first so scattered components merge into the board they sit on,
    # then open to shed isolated speckle the threshold picked up off-board.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest = max(contours, key=cv2.contourArea)
    filled = cv2.contourArea(largest)
    if filled / area < MIN_BOARD_AREA:
        return None, None, None
    x, y, w, h = cv2.boundingRect(largest)
    board = _normalise(x, y, w, h, columns, rows)
    if board is None:
        return None, None, None
    quad = None
    rotated = cv2.minAreaRect(largest)
    span = float(rotated[1][0]) * float(rotated[1][1])
    # Only claim a four-corner outline when the object really is rectangular;
    # a hand or a cable would otherwise get a confident board frame drawn on it.
    if span > 0 and filled / span >= _RECTANGULAR:
        quad = [[round(float(px) / columns, 5), round(float(py) / rows, 5)]
                for px, py in cv2.boxPoints(rotated)]
    # The caller filters regions against this rect, so hand it back in the
    # region pass's own pixel space rather than this smaller one.
    rect = (int(board[0] * width), int(board[1] * height),
            max(1, int(board[2] * width)), max(1, int(board[3] * height)))
    return board, quad, rect


def analyse_frame(frame: Any, *, max_regions: int = 20) -> BoardReading:
    """Measure board geometry in one BGR frame. Never raises."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return _EMPTY
    try:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4) or array.dtype != np.uint8:
            return _EMPTY
        source_h, source_w = int(array.shape[0]), int(array.shape[1])
        if source_w < 32 or source_h < 32:
            return _EMPTY
        scale = min(1.0, WORK_WIDTH / float(source_w))
        work = (cv2.resize(array[:, :, :3],
                           (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
                           interpolation=cv2.INTER_AREA)
                if scale < 1.0 else array[:, :, :3])
        grey = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape[:2]
        area = float(width * height)
        # Laplacian variance on the working copy: a relative focus signal for
        # THIS resolution, not a calibrated optical measurement.
        focus = float(cv2.Laplacian(grey, cv2.CV_64F).var())

        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        board, quad, board_rect = _find_board(cv2, np, work, width, height)
        # Exposure advice should describe the board, not the desk around it —
        # a dark worktop otherwise reads as "too dark" over a well-lit board.
        if board_rect is not None:
            bx, by, bw, bh = board_rect
            patch = grey[by:by + bh, bx:bx + bw]
            brightness = float(patch.mean()) / 255.0 if patch.size else float(grey.mean()) / 255.0
        else:
            brightness = float(grey.mean()) / 255.0

        # Regions of interest: bright/dark blobs with board-like contrast.
        # Adaptive thresholding copes with the uneven lighting of a hand-held
        # board far better than a global one.
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 6)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        # RETR_LIST, deliberately, not RETR_EXTERNAL: every package sits INSIDE
        # the board's own outline, so keeping only outermost contours returns
        # the board and discards every component on it.
        blobs, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, list[float]]] = []
        for blob in blobs:
            ratio = cv2.contourArea(blob) / area
            if ratio < MIN_REGION_AREA or ratio > MAX_REGION_AREA:
                continue
            x, y, w, h = cv2.boundingRect(blob)
            if w < 4 or h < 4:
                continue
            if max(w, h) / float(min(w, h)) > 12.0:
                continue          # a trace or a frame edge, not a part-shaped blob
            if board_rect is not None:
                bx, by, bw, bh = board_rect
                centre_x, centre_y = x + w / 2, y + h / 2
                if not (bx <= centre_x <= bx + bw and by <= centre_y <= by + bh):
                    continue      # off the board: desk clutter
            box = _normalise(x, y, w, h, width, height)
            if box is not None:
                candidates.append((ratio, box))
        # Largest first, so the box kept for a duplicated feature is the one
        # that encloses the whole part rather than its hollow interior.
        candidates.sort(key=lambda item: item[0], reverse=True)
        limit = max(0, int(max_regions))
        regions: list[list[float]] = []
        for _ratio, box in candidates:
            if len(regions) >= limit:
                break
            if not _contained(box, regions):
                regions.append(box)

        # A board held close enough to fill the view has no surrounding desk to
        # stand out against, so the contrast pass finds nothing — yet this is
        # the *best* framing, and telling the user to "fill more of the frame"
        # there would be precisely backwards. Detail covering the whole frame
        # is the evidence that the board is what fills it.
        fills_view = False
        if board is None and len(regions) >= 3:
            fills_view = True
            board = [0.0, 0.0, 1.0, 1.0]
        return BoardReading(
            board=board, quad=quad, regions=regions, focus=round(focus, 2),
            brightness=round(brightness, 4), width=source_w, height=source_h,
            fills_view=fills_view)
    except Exception:
        # A detection overlay must never be able to take the preview down.
        return _EMPTY


class BoardScanner:
    """Runs :func:`analyse_frame` off the caller's thread, newest frame wins.

    The GUI submits the frame it just drew and reads whatever reading is ready;
    it never waits. A backlog is impossible by construction — the inbox holds
    exactly one frame, so a slow pass drops stale work instead of queuing it.
    """

    def __init__(self, *, min_interval: float = 0.14) -> None:
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._pending: Any = None
        self._reading: BoardReading | None = None
        self._snapshot: tuple[Any, BoardReading, float] | None = None
        self._pending_time = 0.0
        self._generation = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run = 0.0

    def start(self) -> None:
        if self.running:
            return
        self._generation += 1
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._last_run = 0.0
        self._thread = threading.Thread(
            target=self._run, args=(self._stop, self._wake, self._generation),
            name="orion-board-scan", daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        """Invalidate in-flight work; GUI callers use wait=False to stay responsive."""
        self._stop.set()
        self._wake.set()
        self._generation += 1
        thread = self._thread
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        with self._lock:
            self._pending = None
            self._reading = None
            self._snapshot = None

    @property
    def running(self) -> bool:
        return not self._stop.is_set() and self._thread is not None and self._thread.is_alive()

    def submit(self, frame: Any) -> None:
        """Offer a frame. Rate-limited; the previous pending frame is discarded."""
        if frame is None or self._stop.is_set():
            return
        now = time.monotonic()
        if now - self._last_run < self.min_interval:
            return
        self._last_run = now
        with self._lock:
            self._pending = frame
            self._pending_time = now
        self._wake.set()

    def latest(self) -> BoardReading | None:
        with self._lock:
            return self._reading

    def latest_snapshot(self) -> tuple[Any, BoardReading, float] | None:
        """Return geometry together with the exact source frame and capture time."""
        with self._lock:
            return self._snapshot

    def _run(self, stop: threading.Event, wake: threading.Event, generation: int) -> None:
        # Load the detection dependency on the worker before frames arrive.
        # The GUI displays BGR directly with Qt and never waits for this import.
        try:
            import cv2  # noqa: F401
        except Exception:
            pass          # analyse_frame reports it as unavailable, per frame
        while not stop.is_set():
            wake.wait(0.25)
            wake.clear()
            if stop.is_set():
                return
            with self._lock:
                if generation != self._generation:
                    return
                frame, self._pending = self._pending, None
                captured_at = self._pending_time
            if frame is None:
                continue
            reading = analyse_frame(frame)
            with self._lock:
                if stop.is_set() or generation != self._generation:
                    return
                self._reading = reading
                self._snapshot = (frame, reading, captured_at)


__all__ = ["BoardReading", "BoardScanner", "analyse_frame",
           "FOCUS_SOFT", "MIN_BOARD_AREA", "WORK_WIDTH"]
