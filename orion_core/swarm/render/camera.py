"""
camera.py — the orbit camera (Mark XXII, Phase 1).

Spherical orbit around a target, with every controllable quantity split into
a CURRENT and a GOAL value. Input moves the goal; `update(dt)` eases current
toward it. That split is what produces the "camera easing" and "smooth
interpolation" the brief asks for, and it does so frame-rate independently:
the smoothing uses an exponential decay evaluated against real elapsed time,
not a fixed per-frame fraction, so the motion feels identical at 30 fps and
144 fps instead of becoming sluggish when the frame rate drops.

Pure numpy. No Qt, no GL — the widget only feeds it mouse deltas and asks for
matrices, so all the behaviour here is unit-testable without a context.
"""

from __future__ import annotations

import math

import numpy as np

Vec3 = tuple[float, float, float]

# Elevation is clamped just short of the poles. At exactly ±90° the view
# direction becomes parallel to the world up vector and the lookAt basis is
# degenerate — the classic gimbal flip where the scene snaps upside down.
_MAX_ELEVATION = math.radians(89.0)

MIN_DISTANCE = 4.0
MAX_DISTANCE = 260.0

# Framing: with cognition rings the outermost (Background) orbits at radius
# 74 and its member shells add ~9 more, so the field spans roughly 166 units
# — twice the old sphere. Derived from the ring table rather than hard-coded,
# so adding or resizing a ring reframes the camera automatically instead of
# silently pushing the outer ring off screen.
def _default_distance() -> float:
    from .rings import RINGS
    outermost = max(r.radius + r.wobble for r in RINGS)
    # 1.35x measured against real 1600x900 renders: 1.95 framed the whole
    # sphere but left the field a small island in a large empty frame, and
    # the clusters never actually occupy the full ring radius at once.
    return (outermost + 9.0) * 1.35


_DEFAULT_DISTANCE = _default_distance()

# Cinematic drift. ~0.006 rad/frame at 60fps is a full revolution in roughly
# three minutes — present as atmosphere, never as motion you have to fight.
DRIFT_RATE = 0.00018
BREATH_RATE = 0.16          # radians/sec of the zoom oscillation
BREATH_DEPTH = 0.035        # ±3.5% of the base distance
_DEFAULT_AZIMUTH = math.radians(35.0)
_DEFAULT_ELEVATION = math.radians(22.0)

# The bearing ORION's face actually points along. See face_on().
FACE_AZIMUTH = math.pi / 2.0


def _ease(current: float, goal: float, dt: float, smoothing: float) -> float:
    """Frame-rate independent exponential approach.

    `smoothing` is the fraction of the remaining distance covered per second.
    Using 1-(1-s)^dt rather than a raw lerp is what keeps the feel constant
    when dt varies."""
    if dt <= 0.0:
        return current
    t = 1.0 - (1.0 - smoothing) ** dt
    return current + (goal - current) * min(1.0, max(0.0, t))


def _normalise(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / length


class OrbitCamera:
    """Orbit/zoom/pan camera with eased motion."""

    __slots__ = ("azimuth", "elevation", "distance", "target",
                 "_goal_azimuth", "_goal_elevation", "_goal_distance",
                 "_goal_target", "rotate_smoothing", "zoom_smoothing",
                 "pan_smoothing", "fov_degrees", "near", "far",
                 "_base_distance", "drift_enabled")

    def __init__(self) -> None:
        self.azimuth = _DEFAULT_AZIMUTH
        self.elevation = _DEFAULT_ELEVATION
        self.distance = _DEFAULT_DISTANCE
        self.target = np.zeros(3, dtype=np.float64)

        self._goal_azimuth = self.azimuth
        self._goal_elevation = self.elevation
        self._goal_distance = self.distance
        self._goal_target = self.target.copy()
        # The distance the zoom "breathing" oscillates around. Tracked
        # separately from _goal_distance so the breath never compounds with
        # itself into a slow drift toward one of the zoom limits.
        self._base_distance = self.distance

        # Atmosphere for the whole-field view; suppressed when the camera is
        # deliberately held somewhere, such as on ORION's face in the orb.
        # A neural map needs a dependable frame of reference. Atmospheric
        # drift is available for special presentations, not normal operation.
        self.drift_enabled = False

        self.rotate_smoothing = 0.999
        self.zoom_smoothing = 0.995
        self.pan_smoothing = 0.995

        self.fov_degrees = 52.0
        self.near = 0.1
        self.far = 1000.0

    # ── input ────────────────────────────────────────────────────────────────

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        """Drag to orbit. Radians, applied to the goal so motion stays eased."""
        self._goal_azimuth += d_azimuth
        self._goal_elevation = max(-_MAX_ELEVATION,
                                   min(_MAX_ELEVATION,
                                       self._goal_elevation + d_elevation))

    def zoom(self, factor: float) -> None:
        """Multiplicative zoom — a wheel notch should feel the same whether
        you are 5 units out or 200, which additive stepping does not deliver."""
        if factor <= 0.0:
            return
        self._goal_distance = max(MIN_DISTANCE,
                                  min(MAX_DISTANCE, self._goal_distance * factor))
        # Deliberate zoom re-bases the breath, so the drift oscillates around
        # where the user actually left the camera.
        self._base_distance = self._goal_distance

    def zoom_by_steps(self, steps: float, per_step: float = 0.88) -> None:
        """Wheel helper: positive steps move closer."""
        if steps:
            self.zoom(per_step ** steps)

    def pan(self, dx: float, dy: float) -> None:
        """Slide the target across the view plane.

        Scaled by distance so a given mouse movement drags the same amount of
        *screen* regardless of zoom — panning while zoomed in would otherwise
        fling the target across the scene."""
        right, up = self.basis()
        scale = self.distance * 0.0016
        self._goal_target = self._goal_target + right * (-dx * scale) + up * (dy * scale)

    def focus_on(self, position: Vec3, distance: float | None = None) -> None:
        """Fly to a node — the eased path is the point, so selection reads as
        movement through the graph rather than a teleport."""
        self._goal_target = np.array(position, dtype=np.float64)
        if distance is not None:
            self._goal_distance = max(MIN_DISTANCE, min(MAX_DISTANCE, distance))
            self._base_distance = self._goal_distance

    def face_on(self, distance: float | None = None) -> None:
        """Frame ORION head-on.

        His geometry faces +Z and the eye orbits (cos az, ·, sin az), so his
        front is azimuth 90° — NOT azimuth 0, which photographs his right ear.
        Getting this wrong is invisible in any headless test and produces a
        portrait of the side of his head, so the bearing lives here as one
        named thing rather than being re-derived at each call site.

        A touch of elevation, because a dead-level camera on a symmetrical
        head reads as a mugshot.
        """
        self._goal_azimuth = FACE_AZIMUTH
        self._goal_elevation = math.radians(5.0)
        self._goal_target = np.zeros(3, dtype=np.float64)
        if distance is not None:
            self._goal_distance = max(MIN_DISTANCE, min(MAX_DISTANCE, distance))
            self._base_distance = self._goal_distance

    def reset(self) -> None:
        self._goal_azimuth = _DEFAULT_AZIMUTH
        self._goal_elevation = _DEFAULT_ELEVATION
        self._goal_distance = _DEFAULT_DISTANCE
        self._base_distance = _DEFAULT_DISTANCE
        self._goal_target = np.zeros(3, dtype=np.float64)

    def snap_to_goal(self) -> None:
        """Skip the easing. Used on first show, where animating in from a
        default pose would look like a glitch rather than a transition."""
        self.azimuth = self._goal_azimuth
        self.elevation = self._goal_elevation
        self.distance = self._goal_distance
        self.target = self._goal_target.copy()

    # ── simulation ───────────────────────────────────────────────────────────

    def cinematic_drift(self, elapsed: float) -> None:
        """A slow orbital drift and gentle zoom 'breathing'.

        Applied to the GOAL, not the current pose, so it composes with the
        easing rather than fighting it — and any deliberate user input simply
        moves the goal again and takes over immediately. Deliberately tiny:
        the brief asks for cinematic, not aggressive, and a camera that
        visibly wanders while you are trying to read a node is an irritation
        rather than atmosphere.

        Skipped entirely when drift is off: the compact overlay orb holds
        ORION head-on, and a drift that slowly rotated him away from the
        viewer would turn his portrait into the back of his head."""
        if not self.drift_enabled:
            return
        self._goal_azimuth += DRIFT_RATE
        breath = math.sin(elapsed * BREATH_RATE) * BREATH_DEPTH
        self._goal_distance = max(
            MIN_DISTANCE, min(MAX_DISTANCE, self._base_distance * (1.0 + breath)))

    def update(self, dt: float) -> bool:
        """Advance easing by *dt* seconds. Returns True if anything moved —
        the render loop uses this to skip redrawing a settled camera."""
        before = (self.azimuth, self.elevation, self.distance, tuple(self.target))
        self.azimuth = _ease(self.azimuth, self._goal_azimuth, dt, self.rotate_smoothing)
        self.elevation = _ease(self.elevation, self._goal_elevation, dt,
                               self.rotate_smoothing)
        self.distance = _ease(self.distance, self._goal_distance, dt, self.zoom_smoothing)
        self.target = np.array([
            _ease(self.target[i], self._goal_target[i], dt, self.pan_smoothing)
            for i in range(3)], dtype=np.float64)
        after = (self.azimuth, self.elevation, self.distance, tuple(self.target))
        return not self._close_enough(before, after)

    @staticmethod
    def _close_enough(a: tuple, b: tuple, epsilon: float = 1e-5) -> bool:
        if abs(a[0] - b[0]) > epsilon or abs(a[1] - b[1]) > epsilon:
            return False
        if abs(a[2] - b[2]) > epsilon:
            return False
        return all(abs(a[3][i] - b[3][i]) <= epsilon for i in range(3))

    @property
    def is_settled(self) -> bool:
        """True when current has effectively reached goal."""
        return self._close_enough(
            (self.azimuth, self.elevation, self.distance, tuple(self.target)),
            (self._goal_azimuth, self._goal_elevation, self._goal_distance,
             tuple(self._goal_target)),
            epsilon=1e-4)

    # ── geometry ─────────────────────────────────────────────────────────────

    def eye(self) -> np.ndarray:
        """World-space camera position from the spherical parameters."""
        cos_e = math.cos(self.elevation)
        return self.target + np.array([
            math.cos(self.azimuth) * cos_e * self.distance,
            math.sin(self.elevation) * self.distance,
            math.sin(self.azimuth) * cos_e * self.distance,
        ], dtype=np.float64)

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """The camera's right and up vectors — needed by pan, and by any
        billboarding the renderer does."""
        forward = _normalise(self.target - self.eye())
        world_up = np.array([0.0, 1.0, 0.0])
        right = _normalise(np.cross(forward, world_up))
        up = _normalise(np.cross(right, forward))
        return right, up

    def view_matrix(self) -> np.ndarray:
        """Right-handed lookAt, row-major (textbook convention).

        The GL layer is responsible for transposing on upload; keeping the
        maths in the conventional orientation is what makes it checkable
        against a reference by eye."""
        eye = self.eye()
        forward = _normalise(self.target - eye)
        world_up = np.array([0.0, 1.0, 0.0])
        right = _normalise(np.cross(forward, world_up))
        up = np.cross(right, forward)
        m = np.identity(4, dtype=np.float64)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -forward
        m[0, 3] = -float(np.dot(right, eye))
        m[1, 3] = -float(np.dot(up, eye))
        m[2, 3] = float(np.dot(forward, eye))
        return m

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """Standard perspective projection, row-major."""
        aspect = max(1e-4, float(aspect))
        f = 1.0 / math.tan(math.radians(self.fov_degrees) * 0.5)
        m = np.zeros((4, 4), dtype=np.float64)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (self.far + self.near) / (self.near - self.far)
        m[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        m[3, 2] = -1.0
        return m

    def view_projection(self, aspect: float) -> np.ndarray:
        return self.projection_matrix(aspect) @ self.view_matrix()

    def gl_view_projection(self, aspect: float) -> np.ndarray:
        """The combined matrix as a flat, column-major float32 array — the
        exact layout glUniformMatrix4fv expects with transpose=False."""
        return np.ascontiguousarray(
            self.view_projection(aspect).T, dtype=np.float32).ravel()
