"""
The camera as numbers — local computer-vision instruments.

A camera frame is a grid of numbers: every pixel three bytes of blue, green and
red. The perception loop already reduces that grid to 32×24 brightness values
to notice that *something* moved. This module reads the same frame properly,
the way an OpenCV pipeline does, and says what it measured:

    edges      Canny on a 160×120 greyscale copy. Edge density says how much
               structure is in view (a blank wall is ~0, a cluttered desk is
               high) and, per zone, where that structure is.
    colours    every pixel of a 64×48 copy named by hue/saturation/value
               (red, orange, … white, grey, black), counted, and ranked — so
               "a lot of red has appeared on the left" is a measurement, not a
               guess.
    motion     dense optical flow (Farneback) between this frame and the last.
               The existing motion grid says THAT something moved; flow says
               which way and how fast.
    light      mean brightness, contrast and sharpness (variance of the
               Laplacian) — enough to tell "the lights went off" and "the lens
               is covered" apart from "nothing is happening".
    digits     the 16×12 brightness grid as the digits 0-9: literally the
               camera turned into a grid of numbers, readable in a reply.

Everything here is arithmetic on small arrays: a full reading is a few
milliseconds, runs off the event loop (the caller uses asyncio.to_thread), and
costs nothing to run all day. Naming what the objects ARE is the job of
object_detection.py; bodies and hands are pose_tracking.py.

``render()`` draws any of these views over a frame for the live camera window.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Sequence

WORK_W, WORK_H = 160, 120        # edges, flow and light are measured at this size
COLOUR_W, COLOUR_H = 64, 48      # colour naming needs even less
DIGITS_W, DIGITS_H = 16, 12      # the "grid of numbers" a person can read
CANNY_LO, CANNY_HI = 60, 150
FLOW_MOVING_PX = 0.8             # per-pixel flow (at WORK_W) that counts as moving
MOTION_FRACTION = 0.01           # share of pixels that must move for "moving"
DARK_BRIGHTNESS = 40             # mean luminance below which the view is dark
COVERED_BRIGHTNESS = 18          # ...and this dark AND this flat means covered
COVERED_CONTRAST = 6

#: 3×3 zones, row-major, in the words a person would use.
ZONES = ("top-left", "top", "top-right",
         "left", "centre", "right",
         "bottom-left", "bottom", "bottom-right")

#: Named colours and the BGR used to paint them in the colours view.
COLOUR_NAMES = ("black", "grey", "white", "red", "orange", "yellow", "green",
                "cyan", "blue", "purple", "pink", "brown")
_PAINT = {
    "black": (20, 20, 20), "grey": (128, 128, 128), "white": (240, 240, 240),
    "red": (40, 40, 220), "orange": (30, 140, 250), "yellow": (40, 220, 240),
    "green": (60, 190, 60), "cyan": (220, 210, 40), "blue": (220, 90, 30),
    "purple": (190, 60, 150), "pink": (180, 120, 250), "brown": (40, 80, 140),
}

#: Words people use for zones, mapped onto the nine.
ZONE_ALIASES = {
    "any": None, "anywhere": None, "": None, "whole": None, "all": None,
    "center": "centre", "middle": "centre", "top-centre": "top", "top centre": "top",
    "bottom-centre": "bottom", "bottom centre": "bottom", "middle-left": "left",
    "middle-right": "right", "upper": "top", "lower": "bottom",
}


def normalise_zone(zone: str | None) -> str | None:
    """A zone name from free text, or None for the whole frame.

    Raises ValueError for something that is not a zone, so a typo in a rule
    is refused when it is made rather than silently never matching."""
    text = str(zone or "").strip().lower().replace("_", "-")
    text = text.replace("top left", "top-left").replace("top right", "top-right")
    text = text.replace("bottom left", "bottom-left").replace("bottom right", "bottom-right")
    if text in ZONE_ALIASES:
        return ZONE_ALIASES[text]
    if text in ZONES:
        return text
    raise ValueError(f"'{zone}' is not a zone — use one of: any, {', '.join(ZONES)}")


def zone_of(x: float, y: float) -> str:
    """The zone a normalised (0..1) point falls in."""
    col = 0 if x < 1 / 3 else (2 if x > 2 / 3 else 1)
    row = 0 if y < 1 / 3 else (2 if y > 2 / 3 else 1)
    return ZONES[row * 3 + col]


def _zone_slices(height: int, width: int) -> list[tuple[slice, slice]]:
    ys = (0, height // 3, 2 * height // 3, height)
    xs = (0, width // 3, 2 * width // 3, width)
    return [(slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1]))
            for r in range(3) for c in range(3)]


@dataclass(frozen=True)
class ColourShare:
    name: str
    share: float


@dataclass(frozen=True)
class MotionReading:
    moving: float                 # share of the frame in motion (0..1)
    speed: float                  # mean flow of the moving pixels, frame-widths per second
    direction: str                # left · right · up · down · several ways · still
    zones: tuple[float, ...]      # share moving in each of the nine zones

    @property
    def is_moving(self) -> bool:
        return self.moving >= MOTION_FRACTION

    def describe(self) -> str:
        if not self.is_moving:
            return "nothing is moving"
        busiest = ZONES[max(range(9), key=lambda i: self.zones[i])]
        way = ("in several directions" if self.direction == "several ways"
               else f"towards the {self.direction} of the view")
        return (f"{self.moving:.0%} of the view is moving, mostly {way} "
                f"({self.speed:.2f} frame-widths a second), busiest {busiest}")


@dataclass
class FrameReading:
    """Everything the instruments measured in one frame."""

    brightness: float
    contrast: float
    sharpness: float
    edge_density: float
    edge_zones: tuple[float, ...]
    colours: tuple[ColourShare, ...]
    motion: MotionReading | None
    digits: tuple[str, ...]
    dark: bool
    covered: bool
    # Kept for zone queries and the overlay; not part of the summary.
    colour_map: Any = field(default=None, repr=False)     # COLOUR_H×COLOUR_W name indices
    flow_magnitude: Any = field(default=None, repr=False)  # WORK_H×WORK_W, px per frame

    def colour_share(self, name: str, zone: str | None = None) -> float:
        """Share (0..1) of the frame — or of one zone — that is colour *name*."""
        if self.colour_map is None or name not in COLOUR_NAMES:
            return 0.0
        index = COLOUR_NAMES.index(name)
        area = self.colour_map
        if zone is not None:
            ys, xs = _zone_slices(*area.shape)[ZONES.index(zone)]
            area = area[ys, xs]
        return float((area == index).mean()) if area.size else 0.0

    def number_grid(self) -> str:
        return "\n".join(self.digits)

    def summary(self) -> str:
        light = ("the lens looks covered (black and featureless)" if self.covered
                 else "it is dark" if self.dark
                 else f"brightness {self.brightness:.0f}/255, contrast {self.contrast:.0f}")
        focus = "sharp" if self.sharpness >= 100 else ("soft" if self.sharpness >= 30 else "blurred")
        busiest = ZONES[max(range(9), key=lambda i: self.edge_zones[i])]
        structure = (f"edges cover {self.edge_density:.1%} of the view "
                     f"(most detail {busiest})")
        palette = ", ".join(f"{c.name} {c.share:.0%}" for c in self.colours[:4]) or "no colour read"
        motion = self.motion.describe() if self.motion is not None else "motion needs a second frame"
        return (f"Light: {light}; the image is {focus} (sharpness {self.sharpness:.0f}).\n"
                f"Structure: {structure}.\n"
                f"Colours: {palette}.\n"
                f"Motion: {motion}.")


# ── colour naming, vectorised ────────────────────────────────────────────────

def name_colours(hsv: Any) -> Any:
    """Name every pixel of an OpenCV HSV image (H 0-179). Returns indices
    into COLOUR_NAMES, same shape as the image minus the channel axis."""
    import numpy as np
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    out = np.full(h.shape, COLOUR_NAMES.index("grey"), dtype=np.uint8)
    chromatic = (s >= 50) & (v >= 50)
    hue_bins = (
        ("red", (h < 8) | (h >= 170)),
        ("orange", (h >= 8) & (h < 20)),
        ("yellow", (h >= 20) & (h < 34)),
        ("green", (h >= 34) & (h < 80)),
        ("cyan", (h >= 80) & (h < 100)),
        ("blue", (h >= 100) & (h < 130)),
        ("purple", (h >= 130) & (h < 150)),
        ("pink", (h >= 150) & (h < 170)),
    )
    for name, mask in hue_bins:
        out[chromatic & mask] = COLOUR_NAMES.index(name)
    # Brown is dark orange/red, which is how a wooden desk actually reads.
    brown = chromatic & (v < 150) & ((h < 20) | (h >= 170)) & (s >= 80)
    out[brown] = COLOUR_NAMES.index("brown")
    out[~chromatic & (v >= 200)] = COLOUR_NAMES.index("white")
    out[v < 50] = COLOUR_NAMES.index("black")
    return out


def _flow_direction(dx: float, dy: float, magnitude: float) -> str:
    if magnitude <= 1e-6:
        return "still"
    if (dx * dx + dy * dy) ** 0.5 < 0.35 * magnitude:
        return "several ways"      # the motion cancels out: people moving apart, shaking
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


class VisionLab:
    """Stateful only in the previous frame, which optical flow needs."""

    def __init__(self, fps_hint: float = 2.0) -> None:
        self.fps_hint = max(0.1, float(fps_hint))
        self._previous: Any = None

    def reset(self) -> None:
        self._previous = None

    def read(self, frame: Any) -> FrameReading | None:
        """Measure one BGR frame. None if it is not an image."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
            return None
        if frame.shape[0] < 8 or frame.shape[1] < 8:
            return None
        small = cv2.resize(frame, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        brightness = float(grey.mean())
        contrast = float(grey.std())
        sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        edges = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), CANNY_LO, CANNY_HI)
        edge_mask = edges > 0
        edge_zones = tuple(float(edge_mask[ys, xs].mean())
                           for ys, xs in _zone_slices(WORK_H, WORK_W))

        tiny = cv2.resize(small, (COLOUR_W, COLOUR_H), interpolation=cv2.INTER_AREA)
        colour_map = name_colours(cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV))
        counts = np.bincount(colour_map.ravel(), minlength=len(COLOUR_NAMES))
        total = float(colour_map.size)
        colours = tuple(ColourShare(COLOUR_NAMES[i], float(counts[i] / total))
                        for i in np.argsort(counts)[::-1] if counts[i] / total >= 0.02)

        motion = None
        magnitude = None
        if self._previous is not None and self._previous.shape == grey.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self._previous, grey, None, 0.5, 3, 15, 2, 5, 1.1, 0)
            magnitude = np.hypot(flow[..., 0], flow[..., 1])
            moving_mask = magnitude > FLOW_MOVING_PX
            share = float(moving_mask.mean())
            if moving_mask.any():
                dx = float(flow[..., 0][moving_mask].mean())
                dy = float(flow[..., 1][moving_mask].mean())
                mean_mag = float(magnitude[moving_mask].mean())
            else:
                dx = dy = mean_mag = 0.0
            zones = tuple(float(moving_mask[ys, xs].mean())
                          for ys, xs in _zone_slices(WORK_H, WORK_W))
            motion = MotionReading(
                moving=share,
                speed=mean_mag / WORK_W * self.fps_hint,
                direction=_flow_direction(dx, dy, mean_mag) if share >= MOTION_FRACTION else "still",
                zones=zones,
            )
        self._previous = grey

        levels = cv2.resize(grey, (DIGITS_W, DIGITS_H), interpolation=cv2.INTER_AREA)
        digits = tuple("".join(str(min(9, int(value) * 10 // 256)) for value in row)
                       for row in levels)
        return FrameReading(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            edge_density=float(edge_mask.mean()),
            edge_zones=edge_zones,
            colours=colours,
            motion=motion,
            digits=digits,
            dark=brightness < DARK_BRIGHTNESS,
            covered=brightness < COVERED_BRIGHTNESS and contrast < COVERED_CONTRAST,
            colour_map=colour_map,
            flow_magnitude=magnitude,
        )


# ── drawing the views ────────────────────────────────────────────────────────

OVERLAY_MODES = ("normal", "edges", "motion", "colours", "numbers", "detect")

#: MediaPipe pose connections worth drawing (shoulders, arms, torso, legs).
POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
              (23, 24), (23, 25), (25, 27), (24, 26), (26, 28))

_ACCENT = (230, 200, 60)          # cyan-ish BGR, ORION's HUD colour
_BOX = (80, 220, 255)


@functools.lru_cache(maxsize=8)
def _digit_glyphs(cell_w: int, cell_h: int) -> tuple[Any, ...]:
    """Boolean masks of the digits 0-9, sized to one grid cell."""
    import cv2
    import numpy as np
    scale = max(0.3, cell_h / 34.0)
    masks = []
    for digit in range(10):
        canvas = np.zeros((cell_h, cell_w), np.uint8)
        cv2.putText(canvas, str(digit), (int(cell_w * 0.28), int(cell_h * 0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
        masks.append(canvas > 96)
    return tuple(masks)


def render(frame: Any, mode: str = "normal", reading: FrameReading | None = None,
           detections: Sequence[Any] = (), poses: Sequence[Any] = (),
           width: int = 440) -> Any:
    """One camera view, drawn for a person to look at. Never raises: on any
    fault the plain (resized) frame comes back."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return frame
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return frame
    try:
        height = max(1, int(frame.shape[0] * width / frame.shape[1]))
        view = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if mode == "edges":
            grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), CANNY_LO, CANNY_HI)
            view = (view * 0.35).astype(np.uint8)
            view[edges > 0] = _ACCENT
        elif mode == "motion" and reading is not None and reading.flow_magnitude is not None:
            heat = np.clip(reading.flow_magnitude * 40.0, 0, 255).astype(np.uint8)
            heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
            coloured = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
            view = cv2.addWeighted(view, 0.45, coloured, 0.55, 0)
        elif mode == "colours" and reading is not None and reading.colour_map is not None:
            palette = np.array([_PAINT[name] for name in COLOUR_NAMES], dtype=np.uint8)
            painted = cv2.resize(palette[reading.colour_map], (width, height),
                                 interpolation=cv2.INTER_NEAREST)
            view = cv2.addWeighted(view, 0.15, painted, 0.85, 0)
        elif mode == "numbers" and reading is not None:
            # 192 putText calls cost 29 ms a frame — too much for the GUI
            # thread at 10 fps. Ten glyphs are drawn once per cell size and
            # blitted instead.
            view = (view * 0.3).astype(np.uint8)
            cell_w, cell_h = width // DIGITS_W, height // DIGITS_H
            glyphs = _digit_glyphs(cell_w, cell_h)
            for row, line in enumerate(reading.digits):
                for col, digit in enumerate(line):
                    shade = 90 + int(digit) * 18
                    cell = view[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
                    cell[glyphs[int(digit)]] = (shade, shade, 90)
        for person in poses:
            points = [(int(x * width), int(y * height), vis)
                      for x, y, vis in getattr(person, "landmarks", ())]
            for a, b in POSE_EDGES:
                if a < len(points) and b < len(points) and points[a][2] > 0.5 and points[b][2] > 0.5:
                    cv2.line(view, points[a][:2], points[b][:2], _ACCENT, 2, cv2.LINE_AA)
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            p1 = (int(x1 * width), int(y1 * height))
            p2 = (int(x2 * width), int(y2 * height))
            cv2.rectangle(view, p1, p2, _BOX, 2, cv2.LINE_AA)
            label = f"{detection.label} {detection.confidence:.0%}"
            cv2.putText(view, label, (p1[0] + 3, max(12, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _BOX, 1, cv2.LINE_AA)
        return view
    except Exception:
        return frame


__all__ = [
    "COLOUR_NAMES", "ColourShare", "FrameReading", "MotionReading", "OVERLAY_MODES",
    "VisionLab", "ZONES", "name_colours", "normalise_zone", "render", "zone_of",
]
