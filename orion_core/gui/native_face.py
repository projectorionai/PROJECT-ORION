"""
native_face.py — ORION's face panel, drawn by the native renderer.

The Core Window's job on the primary monitor is to BE ORION: his face, large,
and nothing else. The Command Deck is a separate window on the second monitor.
That separation is deliberate — merging them put a swarm, a deck and a face in
one frame and the result read as clutter rather than as a presence.

What this class is: an ADAPTER, not an avatar. ORION is drawn once, by the
scene renderer, in portrait framing. This wraps that renderer in the small
interface the rest of the app already speaks to a face — `set_state`,
`set_amplitude`, `set_speaking`, `set_label`, `apply_emotion`, and a `timer`
attribute — so `AvatarController` and the bus connections keep working
untouched. There is no second face to keep in sync, because there is no
second face.

The renderer is attached AFTER construction. The window is built long before
app.py has any backends, and creating a GL context during window construction
would put it on the startup path — the same reason the swarm page builds its
view lazily.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

# Bus/engine state names mapped onto the renderer's cognition modes. Anything
# unrecognised is left alone rather than snapped to idle — the state string
# has grown over many passes and a stray value must not blank his expression.
_STATE_MODES = {
    "LISTENING": "listening",
    "THINKING": "thinking",
    "PROCESSING": "thinking",
    "SPEAKING": "speaking",
    "EXECUTING": "executing",
    "RESEARCHING": "researching",
    "STANDBY": "standby",
    "IDLE": "idle",
    "ERROR": "error",
}


class NativeFacePanel(QWidget):
    """ORION's face for the Core Window: the scene renderer, framed on him."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._renderer: Any = None
        self.state_name = "STANDBY"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._placeholder = QLabel("◉  ORION is waking.")
        self._placeholder.setObjectName("mutedLabel")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder, 1)

        self.label = QLabel("")
        self.label.setObjectName("mutedLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self._layout = layout
        # Interface parity with the 2-D rig, which owns a repaint timer that
        # overlay mode retunes. The renderer drives its own frame loop, so
        # this one is deliberately inert — but code that pokes `face.timer`
        # must not crash.
        self.timer = QTimer(self)

    # ── construction seam ────────────────────────────────────────────────────

    def attach_renderer(self, renderer: Any) -> None:
        """Hand over the live scene view. Idempotent."""
        if renderer is None or renderer is self._renderer:
            return
        self._renderer = renderer
        self._layout.removeWidget(self._placeholder)
        self._placeholder.hide()
        self._placeholder.deleteLater()
        self._layout.insertWidget(0, renderer, 1)
        # Portrait: the camera held on his face, chrome off. The very same
        # framing the compact overlay orb uses.
        if hasattr(renderer, "set_portrait_mode"):
            renderer.set_portrait_mode(True)
        # Re-apply whatever state arrived while he was still waking.
        self.set_state(self.state_name)

    @property
    def renderer(self) -> Any:
        return self._renderer

    def _gl(self) -> Any:
        return getattr(self._renderer, "gl_view", None)

    # ── the face-rig interface ───────────────────────────────────────────────

    def set_state(self, state: Any) -> None:
        self.state_name = str(state or "STANDBY").upper()
        gl = self._gl()
        if gl is None:
            return
        mode = _STATE_MODES.get(self.state_name)
        if mode is None:
            return
        try:
            gl.set_face_mode(mode)
            scene = getattr(gl, "scene", None)
            cognition = getattr(scene, "cognition", None)
            if cognition is not None:
                cognition.observe_bus_state(self.state_name)
        except Exception:
            pass

    def set_amplitude(self, value: Any) -> None:
        gl = self._gl()
        if gl is None:
            return
        try:
            gl.set_speech_amplitude(float(value))
        except (TypeError, ValueError):
            pass

    def set_speaking(self, active: Any) -> None:
        self.set_state("SPEAKING" if active else "LISTENING")

    def set_pose(self, yaw: Any, pitch: Any) -> None:
        """Head follow from face tracking — becomes his gaze."""
        gl = self._gl()
        if gl is None:
            return
        try:
            gl.look_at(float(yaw), float(pitch))
        except (TypeError, ValueError):
            pass

    def set_label(self, text: Any) -> None:
        self.label.setText(str(text or ""))

    def apply_emotion(self, name: Any, params: Any = None) -> None:
        """Emotion parameter sets.

        The native rig expresses emotion through its cognition mode and speech
        envelope rather than through the 2-D rig's voxel-density parameters,
        so only the parts that have a real counterpart are honoured. Silently
        accepting the rest and rendering nothing would be worse: the caller
        would have no way to know the request went nowhere."""
        del params
        text = str(name or "").strip().upper()
        if text in _STATE_MODES:
            self.set_state(text)
