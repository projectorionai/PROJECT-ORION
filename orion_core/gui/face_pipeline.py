"""
Getting the face off the event loop, without changing the face.

Under qasync the Qt thread **is** the asyncio event loop. Every millisecond
QPainter spends rasterising ORION's head is a millisecond the audio callback
and the live socket do not get, and the head measures ~16 ms a frame at
30 Hz — about 480 ms of every second. That is the same shape of problem that
made ORION's voice stutter before, and it does not get better by drawing less
of him.

So the rasterisation moves to a worker thread and the GUI thread only blits
the result. Nothing about the mesh, the rig, the polygon count or the shading
changes: the identical ``HoloHead`` draws the identical picture with the
identical code. It just does it somewhere else.

Why a thread and not a subprocess
---------------------------------
The directive this implements describes extracting an OpenGL pipeline into a
subprocess over shared memory. ORION's face is not OpenGL — it is QPainter
software rasterisation — and that changes the right answer:

  * QPainter **on a QImage** is explicitly safe off the GUI thread. (QPixmap
    and QWidget are not, which is why the image is the handoff format.) So the
    work can leave the loop without leaving the process.
  * A subprocess would have to ship every finished frame back as pixels. At
    1500x800 that is 4.8 MB a frame, 144 MB/s at 30 Hz, plus a full frame of
    latency before anything appears. The copy would cost more than the paint.

A thread gets the whole benefit — the loop stops rasterising — with a pointer
swap instead of a memory copy.

Who owns what
-------------
The worker owns the ``HoloHead`` **entirely**: it both steps the animation and
paints it. That is deliberate. Stepping on the GUI thread while painting on
the worker would have two threads touching the same vertex arrays, which is a
data race that shows up as a torn face once an hour and cannot be reproduced.

So every GUI-thread call — ``set_amplitude``, ``set_viseme``, ``glance``,
``apply_emotion`` — is recorded as a command and applied by the worker at the
top of its next frame. Commands are coalesced by name: if the GUI sets the
amplitude forty times between two frames, the worker applies the last one,
because the other thirty-nine describe moments that have already passed.

Dropped frames are correct
--------------------------
If the machine is busy the worker renders fewer frames and the GUI blits the
last finished one again. That is the entire point: a late face is invisible,
whereas late audio is a click. Nothing here can block the loop — the GUI never
waits for a frame, it takes whatever is ready.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from PyQt6.QtGui import QImage, QPainter

#: How long the worker sleeps when it has nothing to do. Short enough to pick
#: up a resize promptly, long enough not to spin a core.
IDLE_SLEEP_S = 0.02

#: A frame older than this is stale enough that the panel says so rather than
#: showing it indefinitely — it means the worker has died or wedged.
STALE_AFTER_S = 2.0


class FrameBuffer:
    """Two images and a lock: one being painted, one being shown.

    The lock is held only for the pointer swap and the read, never across a
    paint or a blit, so the two threads never wait on each other for longer
    than a few instructions.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._front: QImage | None = None      # finished, safe for the GUI
        self._back: QImage | None = None       # the worker is painting this
        self._at: float = 0.0
        self._generation = 0

    def back_image(self, width: int, height: int,
                   fill: Any = None) -> QImage:
        """The image the worker should paint into, reallocated only on resize.

        Reallocating a megapixel image thirty times a second would hand back
        everything this saves.
        """
        image = self._back
        if image is None or image.width() != width or image.height() != height:
            image = QImage(width, height, QImage.Format.Format_RGB32)
            self._back = image
        if fill is not None:
            image.fill(fill)
        return image

    def publish(self) -> None:
        """Swap the finished image to the front. Called by the worker."""
        with self._lock:
            self._front, self._back = self._back, self._front
            self._at = time.monotonic()
            self._generation += 1

    def take(self) -> tuple[QImage | None, int]:
        """The newest finished frame, and its generation. Called by the GUI."""
        with self._lock:
            return self._front, self._generation

    @property
    def age(self) -> float:
        with self._lock:
            return 0.0 if self._at == 0.0 else time.monotonic() - self._at

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation


class FaceRenderPipeline:
    """A worker thread that steps and paints a head into a frame buffer.

    ``head`` is whatever object the caller hands over — this module never
    constructs, wraps or restyles one. It calls ``step`` and ``paint`` on it
    and nothing else, so the face this draws is the face that was passed in.
    """

    def __init__(self, head: Any, *, fps: float = 30.0,
                 on_frame: Callable[[], None] | None = None) -> None:
        self.head = head
        self.frames = FrameBuffer()
        self.fps = max(1.0, float(fps))
        self._on_frame = on_frame

        self._commands: dict[str, tuple[tuple, dict]] = {}
        self._command_lock = threading.Lock()
        self._geometry: tuple[int, int, float, float, float] | None = None
        self._palette: tuple[Any, Any, Any] | None = None
        #: What `step` is called with besides dt — loudness, whether ORION is
        #: speaking, and which state he is in. A plain dict assigned whole
        #: rather than mutated, so the worker always reads a consistent set
        #: rather than a half-updated one.
        self._step_kwargs: dict[str, Any] = {}

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

        #: Frames the worker finished, and frames the loop never had to paint.
        self.rendered = 0
        self.paint_ms = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        thread = threading.Thread(target=self._run, name="orion-face-render",
                                  daemon=True)
        self._thread = thread
        thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        """Ask the worker to finish and wait briefly for it.

        Joined rather than abandoned: a daemon thread holding a QImage while
        Qt tears down its graphics stack is how "destroyed but pending"
        warnings turn into a crash on exit.
        """
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ── the GUI thread's side ────────────────────────────────────────────────

    def set_geometry(self, width: int, height: int, cx: float, cy: float,
                     radius: float) -> None:
        """Where and how big to draw. Recorded, not acted on immediately."""
        self._geometry = (max(8, int(width)), max(8, int(height)),
                          float(cx), float(cy), float(radius))
        self._wake.set()

    def set_palette(self, primary: Any, accent: Any, background: Any) -> None:
        self._palette = (primary, accent, background)
        self._wake.set()

    def set_step_kwargs(self, **kwargs: Any) -> None:
        """What `step` gets besides dt: amplitude, speaking, state.

        Assigned as a whole dict rather than mutated in place so the worker
        can never read a half-updated set — an amplitude from this frame with
        a speaking flag from the last one would make the mouth lie.
        """
        self._step_kwargs = dict(kwargs)
        self._wake.set()

    def call(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Queue a method call on the head, to run on the worker thread.

        Coalesced by name. If the GUI sets the amplitude forty times between
        two frames the worker applies the last one — the other thirty-nine
        describe moments that have already gone.
        """
        with self._command_lock:
            self._commands[name] = (args, kwargs)
        self._wake.set()

    # ── the worker's side ────────────────────────────────────────────────────

    def _drain(self) -> None:
        with self._command_lock:
            pending, self._commands = self._commands, {}
        for name, (args, kwargs) in pending.items():
            method = getattr(self.head, name, None)
            if method is None:
                continue
            try:
                method(*args, **kwargs)
            except Exception:
                # A bad command must not take the renderer down; the face
                # going still is worse than one ignored viseme.
                pass

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            frame_started = time.monotonic()
            dt = max(0.0, min(0.25, frame_started - last))
            last = frame_started

            self._drain()
            geometry = self._geometry
            palette = self._palette
            if geometry is None or palette is None:
                # Nothing has told us how big the face is yet.
                self._wake.wait(IDLE_SLEEP_S)
                self._wake.clear()
                continue

            try:
                self._render_one(dt, geometry, palette)
            except Exception:
                # Same reasoning as above, one level up: a single bad frame
                # must not end the pipeline.
                pass

            self.paint_ms = (time.monotonic() - frame_started) * 1000.0
            budget = 1.0 / self.fps
            remaining = budget - (time.monotonic() - frame_started)
            if remaining > 0:
                self._wake.wait(remaining)
                self._wake.clear()

    def _render_one(self, dt: float, geometry: tuple, palette: tuple) -> None:
        width, height, cx, cy, radius = geometry
        primary, accent, background = palette

        step = getattr(self.head, "step", None)
        if step is not None:
            step(dt, **self._step_kwargs)

        image = self.frames.back_image(width, height, background)
        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.head.paint(painter, cx, cy, radius, primary, accent,
                            background)
        finally:
            painter.end()
        self.frames.publish()
        self.rendered += 1
        if self._on_frame is not None:
            try:
                self._on_frame()
            except Exception:
                pass


__all__ = ["IDLE_SLEEP_S", "STALE_AFTER_S", "FaceRenderPipeline", "FrameBuffer"]
