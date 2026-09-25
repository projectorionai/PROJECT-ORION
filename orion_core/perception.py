"""
The perception loop — continuous local vision, cloud only when it matters.

ORION's eyes were a shutter, not a stream. Every look at the world was a
discrete request: capture a frame, send it to a cloud vision model, read the
description back, forget it. Between requests he was blind, and because each
look cost a call, looking often was unaffordable — so he looked rarely, which
meant he never noticed anything, which meant he only ever saw what he was
explicitly asked to look at.

The fix is not a faster shutter. It is to separate *watching* from
*understanding*:

    Watching is local, continuous and free.  A downsampled luminance grid
    twice a second, differenced against the last one, is enough to know that
    something moved, roughly where, how big it was, and whether the scene as a
    whole has changed. That is arithmetic on a 32×24 grid — microseconds, no
    network, no model, no cost, running the whole time.

    Understanding is remote, occasional and expensive.  A cloud model is asked
    what something *is*, and it is asked only when the local layer has already
    established that something happened worth naming.

Between the two sits the part that makes it feel like sight rather than
sampling: **scene memory with object permanence**. A tracked object that leaves
the frame does not evaporate — it becomes OCCLUDED, keeps its identity for a
grace period, and if something similar reappears nearby it resumes the same
track rather than being reported as a new arrival. Without this, a person
walking behind a monitor is announced as a departure and then an arrival, twice
a minute, forever; the cloud gets called each time, and ORION sounds like he has
never seen anyone before.

Two constraints shaped the design:

* **It never opens a camera.** Frames are *borrowed* from whatever already
  holds the device, through a caller-supplied ``frame_source``. Two handles on
  one webcam is the documented Windows MSMF failure (-1072873821), and the
  face tracker is usually already holding it.
* **It degrades to nothing.** No frame source, no numpy, no cloud describer, a
  camera that returns None forever — each costs capability, not stability. The
  loop keeps running and reports honestly that it is seeing nothing.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, Sequence

from .bus import OrionBus
from .utils import first_line, utc_stamp


# ── the local watching layer ─────────────────────────────────────────────────

GRID_W, GRID_H = 32, 24          # luminance grid: small enough to be free
HASH_W, HASH_H = 8, 8            # aHash reduction: 64 bits, the standard size
PIXEL_DELTA = 18                 # per-cell luminance change that counts (0-255)
MIN_BLOB_CELLS = 4               # smaller than this is sensor noise, not an object
MOTION_RATIO = 0.02              # fraction of cells moving that counts as motion
SCENE_CUT_BITS = 12              # aHash hamming distance that means "different scene"
SCENE_CUT_MOTION = 0.25          # ...and a quarter of the frame must actually have moved
MIN_CONTRAST = 12                # below this spread, the hash is reading sensor noise

# ── the tracking layer ───────────────────────────────────────────────────────

GATE = 0.22                      # normalised distance within which a blob is "the same object"
REVIVE_GATE = 0.35               # wider gate for re-identifying a returning object
PERMANENCE_S = 8.0               # how long an unseen object keeps its identity

# ── the sampling layer ───────────────────────────────────────────────────────

MIN_CLOUD_INTERVAL_S = 6.0       # never call the cloud more often than this
CLOUD_HEARTBEAT_S = 300.0        # ...but re-ground at least this often
DEFAULT_FPS = 2.0

# ── the naming layer (object detection, pose) ────────────────────────────────
#
# Both cost tens of milliseconds of CPU, so neither runs every frame. They run
# only when something needs them (a vision rule, or the live overlay), then on
# motion, or on a slow heartbeat so a static object — or an absence — is still
# confirmed. The floor keeps a burst of motion from becoming a burst of models.

DETECT_MIN_INTERVAL_S = 0.5
DETECT_HEARTBEAT_S = 4.0
POSE_MIN_INTERVAL_S = 0.4
POSE_HEARTBEAT_S = 2.0


class TrackState(str, Enum):
    ACTIVE = "active"            # seen in the current frame
    OCCLUDED = "occluded"        # not seen, but still within its permanence window
    GONE = "gone"                # permanence expired; identity released


@dataclass(frozen=True)
class Grid:
    """A downsampled luminance field. The unit of local perception."""

    width: int
    height: int
    cells: tuple[int, ...]

    def at(self, x: int, y: int) -> int:
        return self.cells[y * self.width + x]

    @property
    def size(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Blob:
    """A connected region of change — an object-shaped thing, unnamed."""

    x0: int
    y0: int
    x1: int
    y1: int
    cells: int
    width: int
    height: int

    @property
    def centroid(self) -> tuple[float, float]:
        """Normalised 0..1 centre, so gates are resolution-independent."""
        return (((self.x0 + self.x1 + 1) / 2) / max(1, self.width),
                ((self.y0 + self.y1 + 1) / 2) / max(1, self.height))

    @property
    def area(self) -> float:
        return self.cells / max(1, self.width * self.height)

    def where(self) -> str:
        """Plain-language position, which is what a person actually wants."""
        cx, cy = self.centroid
        column = "left" if cx < 0.34 else ("right" if cx > 0.66 else "centre")
        row = "top" if cy < 0.34 else ("bottom" if cy > 0.66 else "middle")
        return "centre" if (column, row) == ("centre", "middle") else f"{row} {column}"


@dataclass
class MotionResult:
    """What the local layer saw in one frame."""

    ratio: float = 0.0
    blobs: list[Blob] = field(default_factory=list)
    scene_hash: int = 0
    scene_changed: bool = False
    first_frame: bool = False

    @property
    def moving(self) -> bool:
        return self.ratio >= MOTION_RATIO


@dataclass
class Track:
    """One object, followed across frames — and across absences."""

    id: int
    blob: Blob
    state: TrackState = TrackState.ACTIVE
    first_seen: float = 0.0
    last_seen: float = 0.0
    sightings: int = 1
    disappearances: int = 0
    label: str = ""              # filled in by the cloud, when it is worth asking

    @property
    def centroid(self) -> tuple[float, float]:
        return self.blob.centroid

    def describe(self) -> str:
        name = self.label or "something"
        if self.state is TrackState.OCCLUDED:
            return f"{name} (last seen {self.blob.where()}, out of sight)"
        return f"{name} ({self.blob.where()})"


@dataclass
class PerceptionEvent:
    """Something worth telling the rest of ORION about."""

    kind: str                    # appeared · left · returned · scene_changed · described
    at: str
    detail: str
    track_id: int | None = None

    def as_line(self) -> str:
        return f"[{self.at}] {self.kind}: {self.detail}"


# ──────────────────────────────────────────────────────────────────────────────
# FRAME → GRID
# ──────────────────────────────────────────────────────────────────────────────

def to_grid(frame: Any, width: int = GRID_W, height: int = GRID_H) -> Grid | None:
    """Reduce any frame to a small luminance grid, or None if unreadable.

    Accepts what the camera layer actually produces (a numpy BGR or greyscale
    array) and what tests find convenient (nested sequences). Everything after
    this point works on ``Grid`` alone, so the perception logic is testable
    without a camera, without numpy and without a single real image.
    """
    # Text is iterable and would otherwise reduce, character by character, to a
    # perfectly plausible all-black frame — the worst kind of wrong, because
    # nothing downstream could tell it from a dark room.
    if frame is None or isinstance(frame, (str, bytes, bytearray)):
        return None
    if hasattr(frame, "shape") and hasattr(frame, "__array__"):
        grid = _numpy_grid(frame, width, height)
        if grid is not None:
            return grid
    # The whole conversion is guarded, not just the row extraction: a camera
    # handing back a short, ragged or simply strange buffer must read as "no
    # frame", never as an exception on the perception path.
    try:
        rows = _as_rows(frame)
        if not rows or not rows[0] or isinstance(rows[0], (str, bytes, bytearray)):
            return None
        source_h, source_w = len(rows), len(rows[0])
        cells: list[int] = []
        for y in range(height):
            # Nearest-neighbour: box-filtering a frame we are about to threshold
            # anyway costs time and changes nothing about what survives.
            source_y = min(source_h - 1, y * source_h // height)
            row = rows[source_y]
            for x in range(width):
                source_x = min(source_w - 1, x * source_w // width)
                cells.append(_luma(row[source_x]))
    except Exception:
        return None
    return Grid(width, height, tuple(cells)) if len(cells) == width * height else None


def _numpy_grid(frame: Any, width: int, height: int) -> Grid | None:
    """Reduce a real camera frame by INDEXING, not by converting it.

    The obvious implementation — ``frame.tolist()`` then walk it — materialises
    every pixel as a Python object: 2.7 million of them for one 720p frame,
    measured at ~100ms each. At two frames a second that is a fifth of a core
    burned continuously, which would make a nonsense of the claim that local
    watching is free. Sampling 768 points with fancy indexing touches only the
    pixels that survive, and costs microseconds.
    """
    try:
        import numpy as np
    except Exception:
        return None
    try:
        array = np.asarray(frame)
        if array.ndim == 3:
            array = array[..., :3]
        elif array.ndim != 2:
            return None
        source_h, source_w = int(array.shape[0]), int(array.shape[1])
        if source_h < 1 or source_w < 1:
            return None
        ys = np.minimum(np.arange(height) * source_h // height, source_h - 1)
        xs = np.minimum(np.arange(width) * source_w // width, source_w - 1)
        sampled = array[np.ix_(ys, xs)].astype(np.float32)
        if sampled.ndim == 3:
            # BGR, as OpenCV delivers it.
            sampled = (sampled[..., 0] * 0.114 + sampled[..., 1] * 0.587
                       + sampled[..., 2] * 0.299)
        cells = np.clip(sampled, 0, 255).astype(np.int16).reshape(-1)
    except Exception:
        return None
    return Grid(width, height, tuple(int(v) for v in cells))


def _as_rows(frame: Any) -> Sequence[Sequence[Any]]:
    if hasattr(frame, "shape") and hasattr(frame, "tolist"):
        return frame.tolist()          # numpy array, any dtype/channel count
    if isinstance(frame, Grid):
        return [list(frame.cells[y * frame.width:(y + 1) * frame.width])
                for y in range(frame.height)]
    return frame                        # already a nested sequence


def _luma(pixel: Any) -> int:
    """Perceived brightness of one pixel, whatever shape it arrived in.

    Total by construction — anything unreadable is black. A string is iterable,
    so a text buffer mistaken for a frame reaches here one character at a time;
    raising on it would put a ValueError in the middle of the perception path
    for what is really just "this is not an image".
    """
    if isinstance(pixel, bool):
        return 0
    if isinstance(pixel, (int, float)):
        return max(0, min(255, int(pixel)))
    try:
        channels = [float(c) for c in list(pixel)[:3]]
    except (TypeError, ValueError):
        return 0
    if not channels:
        return 0
    if len(channels) < 3:
        return max(0, min(255, int(channels[0])))
    # Frames arrive BGR from OpenCV; the weights are the standard luma
    # coefficients, applied in that order.
    blue, green, red = channels
    return max(0, min(255, int(0.114 * blue + 0.587 * green + 0.299 * red)))


def average_hash(grid: Grid) -> tuple[int, int]:
    """64-bit perceptual hash of the scene, plus the contrast it was taken at.

    The contrast matters as much as the hash. aHash thresholds every cell
    against the mean, so on a nearly flat image — a blank wall, a dim room —
    the cells all sit within a couple of levels of that mean and ordinary
    sensor noise flips half the bits between consecutive identical frames.
    Compared naively, a still camera pointed at a wall reports a scene cut
    every second. The caller uses the spread to decide whether the hash is
    saying anything at all.
    """
    reduced: list[int] = []
    for y in range(HASH_H):
        for x in range(HASH_W):
            source_x = min(grid.width - 1, x * grid.width // HASH_W)
            source_y = min(grid.height - 1, y * grid.height // HASH_H)
            reduced.append(grid.at(source_x, source_y))
    mean = sum(reduced) / max(1, len(reduced))
    bits = 0
    for index, value in enumerate(reduced):
        if value > mean:
            bits |= 1 << index
    return bits, max(reduced) - min(reduced)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ──────────────────────────────────────────────────────────────────────────────
# MOTION AND BLOBS
# ──────────────────────────────────────────────────────────────────────────────

class MotionDetector:
    """Differences consecutive grids into motion, blobs and scene cuts."""

    def __init__(self, delta: int = PIXEL_DELTA,
                 min_cells: int = MIN_BLOB_CELLS) -> None:
        self.delta = int(delta)
        self.min_cells = int(min_cells)
        self._previous: Grid | None = None
        self._hash = 0
        self._contrast = 0

    def reset(self) -> None:
        self._previous = None
        self._hash = 0
        self._contrast = 0

    def update(self, grid: Grid) -> MotionResult:
        scene_hash, contrast = average_hash(grid)
        previous, self._previous = self._previous, grid
        if previous is None or previous.size != grid.size:
            # The first frame has nothing to differ against. Reporting it as
            # total motion would fire an event storm at every startup.
            self._hash, self._contrast = scene_hash, contrast
            return MotionResult(scene_hash=scene_hash, first_frame=True)
        changed = [
            index for index in range(grid.size)
            if abs(grid.cells[index] - previous.cells[index]) >= self.delta
        ]
        ratio = len(changed) / max(1, grid.size)
        # A scene *cut* means most of the picture is different, so it must show
        # up in the raw difference as well as in the hash. Requiring both is
        # what stops a low-contrast wall — where the hash flips on noise alone —
        # from reporting a new scene every second.
        # EITHER frame having real structure is enough for the comparison to
        # mean something — lights going off is a flat frame following a
        # structured one, and is exactly the cut worth reporting. The guard is
        # only there for the case where BOTH frames are featureless and the
        # hash is reading nothing but noise.
        readable = max(contrast, self._contrast) >= MIN_CONTRAST
        result = MotionResult(
            ratio=ratio,
            blobs=self._blobs(changed, grid),
            scene_hash=scene_hash,
            scene_changed=(readable
                           and hamming(scene_hash, self._hash) >= SCENE_CUT_BITS
                           and ratio >= SCENE_CUT_MOTION),
        )
        self._hash, self._contrast = scene_hash, contrast
        return result

    def _blobs(self, changed: Sequence[int], grid: Grid) -> list[Blob]:
        """Connected components over changed cells (4-connectivity, iterative).

        Iterative rather than recursive because a frame where everything moves
        is one component the size of the grid, and recursion would put a
        stack-overflow in the path of a light being switched on.

        The mask is dilated by one cell first. Frame differencing marks where a
        thing WAS and where it now IS, so an object that moves less than its
        own width — the ordinary case at two frames a second — otherwise
        registers as two separate arrivals, and one person walking past becomes
        a small crowd. One cell of dilation closes that gap without merging
        genuinely distinct objects, which are far further apart than that.

        Size is then measured against the ORIGINAL cells, not the dilated ones.
        Dilation exists to connect regions, never to inflate them: measuring
        the grown mask would turn a single speckle of sensor noise into a
        five-cell "object" and put the noise floor back where it started.
        """
        original = set(changed)
        remaining = _dilate(changed, grid)
        blobs: list[Blob] = []
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            component = [seed]
            while stack:
                index = stack.pop()
                x, y = index % grid.width, index // grid.width
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                        continue
                    neighbour = ny * grid.width + nx
                    if neighbour in remaining:
                        remaining.discard(neighbour)
                        stack.append(neighbour)
                        component.append(neighbour)
            core = [index for index in component if index in original]
            if len(core) < self.min_cells:
                continue
            xs = [i % grid.width for i in core]
            ys = [i // grid.width for i in core]
            blobs.append(Blob(min(xs), min(ys), max(xs), max(ys),
                              len(core), grid.width, grid.height))
        blobs = _merge_split_regions(blobs)
        blobs.sort(key=lambda b: -b.cells)
        return blobs


# ──────────────────────────────────────────────────────────────────────────────
# SCENE MEMORY WITH OBJECT PERMANENCE
# ──────────────────────────────────────────────────────────────────────────────

def _merge_split_regions(blobs: list[Blob]) -> list[Blob]:
    """Rejoin the two halves of one moving object.

    Differencing two frames marks where a thing WAS and where it now IS. When
    it moves further than its own width those regions are disjoint, and no
    amount of dilation will close the hole in the middle — so one person
    walking past is reported as two arrivals, and the scene fills with phantom
    objects that each cost a cloud call.

    The discriminator is scale: the two halves of one object are separated by
    roughly one object-width, whereas two genuinely different things in a room
    are much further apart than either of them is wide. So two regions merge
    when the gap between them is no larger than the smaller region — and never
    if the result would swallow most of the frame, which would glue a busy
    scene into a single meaningless blob.
    """
    merged = list(blobs)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                gap_x = max(0, max(a.x0, b.x0) - min(a.x1, b.x1) - 1)
                gap_y = max(0, max(a.y0, b.y0) - min(a.y1, b.y1) - 1)
                reach = max(2, min(min(a.x1 - a.x0 + 1, a.y1 - a.y0 + 1),
                                   min(b.x1 - b.x0 + 1, b.y1 - b.y0 + 1)))
                if gap_x > reach or gap_y > reach:
                    continue
                union = Blob(min(a.x0, b.x0), min(a.y0, b.y0),
                             max(a.x1, b.x1), max(a.y1, b.y1),
                             a.cells + b.cells, a.width, a.height)
                span = (union.x1 - union.x0 + 1) * (union.y1 - union.y0 + 1)
                if span > 0.6 * a.width * a.height:
                    continue
                merged[i] = union
                merged.pop(j)
                changed = True
                break
            if changed:
                break
    return merged


def _dilate(changed: Sequence[int], grid: Grid) -> set[int]:
    """Grow the change mask by one cell in each direction."""
    grown = set(changed)
    for index in changed:
        x, y = index % grid.width, index // grid.width
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                grown.add(ny * grid.width + nx)
    return grown


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class SceneMemory:
    """What is in the scene, including what has temporarily stopped being visible.

    The permanence rule is the point. An object that is not matched this frame
    is not gone; it is OCCLUDED, and it keeps its identity — and its label, so
    the cloud is not asked to name it again — for ``permanence_s``. A blob
    appearing near an occluded track's last position resumes that track. Only
    when the window expires is the identity released and a departure reported.

    Without it, anything that passes behind anything else produces a departure
    and an arrival every few seconds, each one a cloud call and a spoken
    interruption, and ORION appears to have no memory at all.
    """

    def __init__(self, gate: float = GATE, revive_gate: float = REVIVE_GATE,
                 permanence_s: float = PERMANENCE_S) -> None:
        self.gate = float(gate)
        self.revive_gate = float(revive_gate)
        self.permanence_s = float(permanence_s)
        self.tracks: dict[int, Track] = {}
        self._counter = 0

    def active(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state is TrackState.ACTIVE]

    def remembered(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state is not TrackState.GONE]

    def update(self, blobs: Sequence[Blob], now: float) -> list[PerceptionEvent]:
        """Fold this frame's blobs into the scene. Returns what changed."""
        events: list[PerceptionEvent] = []
        candidates = [t for t in self.tracks.values() if t.state is not TrackState.GONE]
        pairs = sorted(
            ((_distance(track.centroid, blob.centroid), track, blob)
             for track in candidates for blob in blobs),
            key=lambda triple: triple[0],
        )
        claimed_tracks: set[int] = set()
        claimed_blobs: set[int] = set()
        for distance, track, blob in pairs:
            # An OCCLUDED track is allowed a wider gate: it may have moved
            # while it was out of sight, which is exactly why it went missing.
            limit = self.gate if track.state is TrackState.ACTIVE else self.revive_gate
            if distance > limit:
                continue
            if track.id in claimed_tracks or id(blob) in claimed_blobs:
                continue
            claimed_tracks.add(track.id)
            claimed_blobs.add(id(blob))
            returning = track.state is TrackState.OCCLUDED
            track.blob = blob
            track.state = TrackState.ACTIVE
            track.last_seen = now
            track.sightings += 1
            if returning:
                events.append(PerceptionEvent(
                    "returned", utc_stamp(),
                    f"{track.label or 'something'} is back in view at {blob.where()}",
                    track.id))

        for blob in blobs:
            if id(blob) in claimed_blobs:
                continue
            self._counter += 1
            track = Track(id=self._counter, blob=blob, first_seen=now, last_seen=now)
            self.tracks[track.id] = track
            events.append(PerceptionEvent(
                "appeared", utc_stamp(),
                f"movement at {blob.where()}, about "
                f"{blob.area * 100:.0f}% of the view", track.id))

        for track in candidates:
            if track.id in claimed_tracks:
                continue
            if track.state is TrackState.ACTIVE:
                track.state = TrackState.OCCLUDED
                track.disappearances += 1
            elif now - track.last_seen >= self.permanence_s:
                track.state = TrackState.GONE
                events.append(PerceptionEvent(
                    "left", utc_stamp(),
                    f"{track.label or 'something'} last seen {track.blob.where()} "
                    f"has gone", track.id))
        return events

    def describe(self) -> str:
        present = self.active()
        occluded = [t for t in self.tracks.values() if t.state is TrackState.OCCLUDED]
        if not present and not occluded:
            return "Nothing is moving in view."
        lines = []
        if present:
            lines.append("In view: " + "; ".join(t.describe() for t in present))
        if occluded:
            lines.append("Out of sight but remembered: "
                         + "; ".join(t.describe() for t in occluded))
        return "\n".join(lines)

    def prune(self, keep: int = 60) -> None:
        gone = [t for t in self.tracks.values() if t.state is TrackState.GONE]
        if len(self.tracks) <= keep:
            return
        for track in sorted(gone, key=lambda t: t.last_seen)[:len(self.tracks) - keep]:
            self.tracks.pop(track.id, None)


# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE CLOUD SAMPLING
# ──────────────────────────────────────────────────────────────────────────────

class SamplingPolicy:
    """Decides when understanding is worth paying for.

    The local layer runs continuously and free; this is the valve in front of
    the expensive one. It opens on a *change* — something arrived, something
    left, the scene cut — never on mere motion, because a curtain moving in a
    draught is motion every frame and is worth naming exactly never.

    Two bounds keep it honest in both directions: a floor, so a burst of
    activity cannot become a burst of calls; and a heartbeat ceiling, so a
    scene that has been static for a long time is still re-grounded
    occasionally — drift is real, and a description from five minutes ago
    quietly becomes wrong.
    """

    TRIGGERS = frozenset({"appeared", "returned", "left", "scene_changed"})

    def __init__(self, min_interval_s: float = MIN_CLOUD_INTERVAL_S,
                 heartbeat_s: float = CLOUD_HEARTBEAT_S) -> None:
        self.min_interval_s = float(min_interval_s)
        self.heartbeat_s = float(heartbeat_s)
        self.last_sample: float | None = None
        self.samples = 0
        self.suppressed = 0

    def should_sample(self, events: Iterable[PerceptionEvent],
                      now: float) -> tuple[bool, str]:
        kinds = {event.kind for event in events}
        triggered = sorted(kinds & self.TRIGGERS)
        if self.last_sample is None:
            # Nothing has ever been described; the first real event grounds it.
            return (True, f"first look ({', '.join(triggered)})") if triggered \
                else (False, "waiting for something to happen")
        since = now - self.last_sample
        if triggered and since < self.min_interval_s:
            self.suppressed += 1
            return False, f"{', '.join(triggered)} but only {since:.0f}s since the last look"
        if triggered:
            return True, ", ".join(triggered)
        if since >= self.heartbeat_s:
            return True, f"nothing has changed for {since / 60:.0f} minutes — re-grounding"
        return False, "nothing worth a second look"

    def record(self, now: float) -> None:
        self.last_sample = now
        self.samples += 1

    def stats(self) -> dict[str, Any]:
        return {"cloud_calls": self.samples, "suppressed": self.suppressed,
                "min_interval_s": self.min_interval_s,
                "heartbeat_s": self.heartbeat_s}


# ──────────────────────────────────────────────────────────────────────────────
# THE LOOP
# ──────────────────────────────────────────────────────────────────────────────

class PerceptionLoop:
    """Continuous local watching, with occasional paid-for understanding."""

    def __init__(
        self,
        bus: OrionBus,
        frame_source: Callable[[], Any] | None = None,
        describe: Callable[[Any], Awaitable[str]] | None = None,
        *,
        fps: float = DEFAULT_FPS,
        policy: SamplingPolicy | None = None,
        scene: SceneMemory | None = None,
        telemetry: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        lab: Any | None = None,
        objects: Any | None = None,
        pose: Any | None = None,
        rules: Any | None = None,
        camera: Any | None = None,
        on_workflow: Callable[[str, str], Any] | None = None,
    ) -> None:
        from .vision_lab import VisionLab
        self.bus = bus
        self.frame_source = frame_source
        self.describe = describe
        self.fps = max(0.2, min(10.0, float(fps)))
        self.policy = policy or SamplingPolicy()
        self.scene = scene or SceneMemory()
        self.telemetry = telemetry
        self.clock = clock
        self.detector = MotionDetector()
        # The instruments (vision_lab), the naming layer (object detection,
        # pose) and what to do about it (vision rules). All optional: without
        # them this is exactly the motion-and-permanence loop it always was.
        self.lab = lab if lab is not None else VisionLab(fps_hint=self.fps)
        self.objects = objects
        self.pose_tracker = pose
        self.rules = rules
        self.camera = camera
        self.on_workflow = on_workflow
        self.overlay_needs: set[str] = set()     # set by the live overlay
        self.last_reading: Any = None
        self.last_detections: list[Any] = []
        self.last_detections_at: float | None = None
        self.last_pose: Any = None
        self.last_pose_at: float | None = None
        self.named_locally = 0
        self._locally_named: set[int] = set()
        self._owns_camera = False
        self.events: deque[PerceptionEvent] = deque(maxlen=120)
        self.frames = 0
        self.blind_frames = 0
        self.last_description = ""
        self.started_at = 0.0
        self._task: Any = None
        self._stop = asyncio.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> str:
        if self.running:
            return "I am already watching."
        if self.frame_source is None:
            return ("I have no camera to watch through — nothing is holding a "
                    "frame source for me to borrow.")
        # get_running_loop, not get_event_loop: with no loop running the old
        # call returned an idle one, create_task scheduled onto it, the task
        # never ran — and this still replied "Watching now". (It is also the
        # call Python 3.14 turns into an error.)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return "I can't start watching: there is no event loop running to watch on."
        # The frames are borrowed from the camera's owner. If nothing has the
        # camera open, "watch" used to mean staring at a camera that was off —
        # every frame blind — so the camera is switched on here, and switched
        # back off by stop() if this is what turned it on.
        camera_note = self._switch_camera_on()
        self._stop = asyncio.Event()
        self.started_at = self.clock()
        self.detector.reset()
        self.lab.reset()
        self._task = loop.create_task(self._run(), name="orion-perception")
        self.bus.log.emit(f"PERCEPTION: watching at {self.fps:g} fps.")
        return f"Watching now — {self.fps:g} frames a second, locally.{camera_note}"

    def _switch_camera_on(self) -> str:
        camera = self.camera
        if camera is None:
            return ""
        try:
            if camera.is_capturing():
                return ""
            result = camera.start()
        except Exception as exc:
            return f" (I couldn't switch the camera on: {first_line(exc, 100)})"
        if getattr(result, "ok", True):
            self._owns_camera = True
            return " I've switched the camera on to do it."
        return f" ({getattr(result, 'text', 'the camera would not start')})"

    async def stop(self) -> str:
        if not self.running:
            return "I am not watching at the moment."
        self._stop.set()
        task, self._task = self._task, None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
        except Exception:
            pass
        if self.pose_tracker is not None:
            await asyncio.to_thread(self.pose_tracker.close)
        if self._owns_camera and self.camera is not None:
            self._owns_camera = False
            try:
                await asyncio.to_thread(self.camera.stop)
            except Exception:
                pass
        self.bus.log.emit("PERCEPTION: stopped watching.")
        return "I've stopped watching."

    async def _run(self) -> None:
        interval = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                try:
                    await self.tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # One bad frame must never end the loop; eyes that close
                    # permanently on a malformed buffer are worse than no eyes.
                    self.bus.log.emit(f"PERCEPTION: frame fault - {first_line(exc, 120)}")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    # ── one frame ─────────────────────────────────────────────────────────────

    async def tick(self) -> list[PerceptionEvent]:
        """Watch one frame. The whole loop, minus the waiting."""
        frame = self.frame_source() if self.frame_source is not None else None
        grid = to_grid(frame)
        if grid is None:
            self.blind_frames += 1
            return []
        self.frames += 1
        now = self.clock()
        motion = self.detector.update(grid)
        # The instruments read every frame (a few ms, off the loop); optical
        # flow needs the previous frame, so this happens before any return.
        reading = await asyncio.to_thread(self.lab.read, frame)
        if reading is not None:
            self.last_reading = reading
        events: list[PerceptionEvent] = []
        if motion.first_frame:
            return []
        if motion.scene_changed:
            events.append(PerceptionEvent(
                "scene_changed", utc_stamp(), "the whole scene changed"))
        events.extend(self.scene.update(motion.blobs, now))
        self.scene.prune()

        detections = None
        if self._due("objects", now, motion.moving):
            detections = await asyncio.to_thread(self.objects.detect, frame)
            self.last_detections, self.last_detections_at = detections, now
            events.extend(self._name_tracks(detections))
        pose = None
        if self._due("pose", now, motion.moving):
            pose = await asyncio.to_thread(self.pose_tracker.read, frame)
            if pose is not None:
                self.last_pose, self.last_pose_at = pose, now

        for event in events:
            self.events.append(event)
            self.bus.log.emit(f"PERCEPTION: {event.detail}")
        if events:
            try:
                self.bus.dashboard_event.emit("perception", self.status())
            except Exception:
                pass

        if self.rules is not None:
            self._apply_rules(now, reading, detections, pose, events)

        # An arrival the detector has already named needs no paid look.
        for_cloud = [e for e in events if e.track_id is None or e.kind == "scene_changed"
                     or e.track_id not in self._locally_named]
        wanted, reason = self.policy.should_sample(for_cloud, now)
        if wanted:
            await self._understand(frame, reason, now)
        if self.telemetry is not None and events:
            try:
                self.telemetry.metrics.incr("perception.events", len(events))
            except Exception:
                pass
        return events

    # ── the naming layer ──────────────────────────────────────────────────────

    def _due(self, kind: str, now: float, moving: bool) -> bool:
        """Whether object detection / pose should run on this frame."""
        if kind == "objects":
            instrument, last = self.objects, self.last_detections_at
            floor, heartbeat = DETECT_MIN_INTERVAL_S, DETECT_HEARTBEAT_S
        else:
            instrument, last = self.pose_tracker, self.last_pose_at
            floor, heartbeat = POSE_MIN_INTERVAL_S, POSE_HEARTBEAT_S
        if instrument is None or not getattr(instrument, "available", False):
            return False
        wanted = kind in self.overlay_needs or (
            self.rules is not None and kind in self.rules.needs())
        if not wanted:
            return False
        if last is None:
            return True
        since = now - last
        return since >= floor and (moving or since >= heartbeat)

    def _name_tracks(self, detections: Sequence[Any]) -> list[PerceptionEvent]:
        """Give each unnamed moving thing the name of the object it sits in.

        A track named here is also one the cloud never needs to be asked
        about — see the for_cloud filter in tick()."""
        named = []
        for track in self.scene.active():
            if track.label:
                continue
            cx, cy = track.centroid
            holders = [d for d in detections if d.contains(cx, cy)]
            if not holders:
                continue
            best = max(holders, key=lambda d: d.confidence)
            track.label = best.label
            self._locally_named.add(track.id)
            self.named_locally += 1
            named.append(PerceptionEvent(
                "named", utc_stamp(),
                f"that's {'an' if best.label[0] in 'aeiou' else 'a'} {best.label} "
                f"({track.blob.where()}) — named locally", track.id))
        return named

    def _apply_rules(self, now: float, reading: Any, detections: Any, pose: Any,
                     events: Sequence[PerceptionEvent]) -> None:
        from .vision_rules import Observation, act
        try:
            firings = self.rules.evaluate(Observation(now, reading, detections, pose, events))
        except Exception as exc:
            self.bus.log.emit(f"PERCEPTION: vision rules fault - {first_line(exc, 120)}")
            return
        for firing in firings:
            try:
                done = act(firing, say=self.bus.speak_request.emit, log=self.bus.log.emit,
                           run_workflow=self.on_workflow)
            except Exception as exc:
                done = f"failed: {first_line(exc, 100)}"
            self.events.append(PerceptionEvent(
                "rule", utc_stamp(), f"{firing.rule.name}: {firing.detail} — {done}"))

    # ── looking on demand (the tool's analyse / detect / pose) ────────────────

    def _frame(self) -> Any:
        try:
            return self.frame_source() if self.frame_source is not None else None
        except Exception:
            return None

    async def grab(self, count: int = 1, gap_s: float = 0.0, wait_s: float = 4.0) -> list[Any]:
        """Up to *count* frames, *gap_s* apart, in ONE camera session.

        Switches the camera on for the look if nothing has it open, and back
        off afterwards unless watching is what needs it."""
        frames: list[Any] = []
        started_here = False
        try:
            if self._frame() is None and self.camera is not None \
                    and not self.camera.is_capturing():
                result = await asyncio.to_thread(self.camera.start)
                started_here = bool(getattr(result, "ok", True))
            for index in range(max(1, count)):
                deadline = self.clock() + wait_s
                frame = self._frame()
                while frame is None and self.clock() < deadline:
                    await asyncio.sleep(0.15)
                    frame = self._frame()
                if frame is None:
                    break
                frames.append(frame)
                if index < count - 1:
                    await asyncio.sleep(gap_s)
        finally:
            if started_here and not self.running:
                await asyncio.to_thread(self.camera.stop)
            elif started_here:
                self._owns_camera = True
        return frames

    async def look(self, wait_s: float = 4.0) -> Any:
        frames = await self.grab(1, wait_s=wait_s)
        return frames[0] if frames else None

    async def read_now(self) -> Any:
        """A fresh instrument reading, including motion (two frames apart)."""
        if self.running and self.last_reading is not None:
            return self.last_reading
        from .vision_lab import VisionLab
        frames = await self.grab(2, gap_s=0.4)
        if not frames:
            return None
        lab = VisionLab(fps_hint=1 / 0.4)
        reading = None
        for frame in frames:
            reading = await asyncio.to_thread(lab.read, frame)
        return reading

    async def detect_now(self) -> tuple[list[Any] | None, str]:
        if self.objects is None:
            return None, "Object detection is not part of this build."
        if not self.objects.available:
            return None, self.objects.describe_state()
        frame = await self.look()
        if frame is None:
            return None, "I can't get a frame from the camera."
        detections = await asyncio.to_thread(self.objects.detect, frame)
        self.last_detections, self.last_detections_at = detections, self.clock()
        return detections, ""

    async def pose_now(self) -> tuple[Any, str]:
        if self.pose_tracker is None or not self.pose_tracker.available:
            state = self.pose_tracker.describe_state() if self.pose_tracker is not None \
                else "pose tracking is not part of this build"
            return None, state[0].upper() + state[1:] + "."
        frame = await self.look()
        if frame is None:
            return None, "I can't get a frame from the camera."
        pose = await asyncio.to_thread(self.pose_tracker.read, frame)
        if pose is None:
            return None, f"Pose tracking failed: {self.pose_tracker.error or 'no result'}."
        self.last_pose, self.last_pose_at = pose, self.clock()
        return pose, ""

    def instruments(self) -> str:
        parts = ["local instruments: edges, colours, motion flow, light (always on)"]
        if self.objects is not None:
            parts.append(self.objects.describe_state())
        if self.pose_tracker is not None:
            parts.append(self.pose_tracker.describe_state())
        if self.rules is not None:
            enabled = len(self.rules.enabled_rules())
            parts.append(f"{enabled} vision rule(s) armed" if enabled else "no vision rules")
        return "; ".join(parts) + "."

    async def _understand(self, frame: Any, reason: str, now: float) -> None:
        """Spend a cloud call: ask what the thing actually is."""
        if self.describe is None:
            return
        self.policy.record(now)
        self.bus.log.emit(f"PERCEPTION: looking properly — {reason}.")
        try:
            description = str(await self.describe(frame)).strip()
        except Exception as exc:
            self.bus.log.emit(f"PERCEPTION: could not look - {first_line(exc, 120)}")
            return
        if not description:
            return
        self.last_description = description
        # Name the tracks that have no name yet, so permanence also preserves
        # the identity across an absence and the cloud is not asked twice.
        for track in self.scene.active():
            if not track.label:
                track.label = first_line(description, 60)
        event = PerceptionEvent("described", utc_stamp(), description)
        self.events.append(event)
        try:
            self.bus.dashboard_event.emit("perception", self.status())
        except Exception:
            pass

    # ── reporting ─────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "watching": self.running,
            "fps": self.fps,
            "frames": self.frames,
            "blind_frames": self.blind_frames,
            "uptime_s": round(self.clock() - self.started_at, 1) if self.started_at else 0.0,
            "in_view": len(self.scene.active()),
            "remembered": len(self.scene.remembered()),
            "scene": self.scene.describe(),
            "last_description": self.last_description,
            "named_locally": self.named_locally,
            "objects_in_view": [d.label for d in self.last_detections],
            **self.policy.stats(),
        }

    def report(self) -> str:
        status = self.status()
        if not status["watching"] and not status["frames"]:
            return "I am not watching anything at the moment."
        efficiency = ""
        if status["frames"]:
            saved = status["frames"] - status["cloud_calls"]
            efficiency = (f" I have looked at {status['frames']:,} frames locally "
                          f"and needed the cloud {status['cloud_calls']} time(s) — "
                          f"{saved:,} frames understood for free.")
        head = ("Watching" if status["watching"] else "Not watching") + \
            f" — {status['in_view']} thing(s) in view, " \
            f"{status['remembered']} remembered.{efficiency}"
        parts = [head, status["scene"]]
        if status["named_locally"]:
            parts.append(f"{status['named_locally']} arrival(s) named by the local object "
                         "detector, without asking the cloud.")
        if status["last_description"]:
            parts.append(f"Last proper look: {status['last_description']}")
        parts.append("Instruments — " + self.instruments())
        if status["blind_frames"]:
            parts.append(f"({status['blind_frames']} frame(s) came back empty — "
                         "the camera may be in use elsewhere.)")
        return "\n\n".join(parts)

    def recent(self, limit: int = 12) -> str:
        events = list(self.events)[-max(1, limit):]
        return "\n".join(e.as_line() for e in events) or "Nothing has happened yet."


__all__ = [
    "Blob",
    "CLOUD_HEARTBEAT_S",
    "GATE",
    "GRID_H",
    "GRID_W",
    "Grid",
    "MIN_CLOUD_INTERVAL_S",
    "MOTION_RATIO",
    "MotionDetector",
    "MotionResult",
    "PERMANENCE_S",
    "PerceptionEvent",
    "PerceptionLoop",
    "SamplingPolicy",
    "SceneMemory",
    "Track",
    "TrackState",
    "average_hash",
    "hamming",
    "to_grid",
]
