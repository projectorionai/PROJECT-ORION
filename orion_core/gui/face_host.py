"""ORION's face in its own process.

Why
---
Qt 6's WebEngine view presents every frame through the GUI thread of the
process it lives in, and in ORION that thread is also the asyncio loop running
his Python. So any Python stall - a tool, a memory query, a garbage collection
- held the face's frames for exactly as long. MEASURED (10 Oct 2026, real
windows, a 200 ms block every 500 ms):

    page in ORION's process      144 frames in 4 s, 7 gaps > 50 ms, longest 200 ms
    page in its own process      375 frames in 6.25 s (60 fps), 0 gaps, longest 17 ms

Nothing in ORION can promise the GUI thread never stalls, so the face is moved
off it: ``ORION.exe --face-host`` runs the real QuantumFace3D in a separate
process with its own GUI thread and interpreter, and :class:`RemoteFace`
embeds that window in the Core Window and forwards the same calls.

Rules this follows (each one is a failure seen while building it)
-----------------------------------------------------------------
* ORION never waits on the host. Commands go through a bounded queue drained
  by a writer thread; if the host stops reading, frames are dropped rather than
  the GUI thread blocking on a full pipe.
* The host never waits on ORION. An embedded window whose parent stopped
  pumping messages hung the first probe; the host only reads its stdin on a
  worker thread and quits when it closes (ORION gone).
* A host that dies is restarted and its state replayed; one that keeps dying
  (or never shows a window) turns the feature off for this run and reports
  itself dead, so the Core Window's existing probe falls back to the
  in-process face.

ORION_FACE_PROCESS=0 keeps the face in ORION's own process.

Mark XXXV W1.2 generalised the process machinery into gui/page_host.py (the
globe and the Brain instruments use it too); this module keeps the face's
own surface: its API, its commands and ``--face-host``.
"""

from __future__ import annotations

import subprocess
from typing import Any, Callable

from . import page_host
from .page_host import HOST_CHROMIUM_FLAGS, PROTOCOL_PREFIX, RemoteSurface, encode  # noqa: F401

FLAG = "--face-host"

#: The face calls forwarded to the host (QuantumFace3D's public surface).
METHODS = frozenset({
    "set_amplitude", "set_spectrum", "set_speaking", "set_state", "apply_emotion",
    "set_viseme", "set_pose", "set_morph", "set_avatar_frame", "set_presentation",
    "set_label", "set_form", "set_orb_camera_enabled", "set_orb_hands",
})
#: Calls answered asynchronously through a callback.
QUERIES = frozenset({"check_state", "check_alive", "report"})
#: Re-sent to a restarted host so it comes back as it was.
STICKY = ("set_form", "set_state", "set_label", "set_orb_camera_enabled", "set_presentation")


def enabled() -> bool:
    return page_host.enabled("face")


def host_command() -> list[str]:
    """How to start the host: the same executable ORION is running as."""
    return page_host.host_command("face", flag=FLAG)


# ── the host process ─────────────────────────────────────────────────────────

def apply(face: Any, message: dict[str, Any], reply: Callable[[dict[str, Any]], None],
          quit_app: Callable[[], None]) -> None:
    """Run one command from ORION against the real face. Never raises."""
    method = str(message.get("m") or "")
    args = message.get("a") or []
    if method == "__quit__":
        quit_app()
        return
    if method in QUERIES:
        rid = message.get("id")
        answer = lambda value: reply({"t": "reply", "id": rid, "value": value})  # noqa: E731
        try:
            if method == "report":
                _report(face, answer)
            else:
                getattr(face, method)(answer)
        except Exception as exc:
            answer(f"error: {exc}")
        return
    if method not in METHODS:
        return
    try:
        getattr(face, method)(*args)
    except Exception as exc:
        reply({"t": "log", "line": f"FACE: host could not {method} - {exc}"})


def _report(face: Any, answer: Callable[[Any], None]) -> None:
    """The page's own frame rate and health counters, for diagnostics."""
    view = getattr(face, "view", None)
    if view is None:
        answer(None)
        return
    view.page().runJavaScript(
        "JSON.stringify({fps:(window.__dbg&&window.__dbg.fps)||0,"
        "health:window.orionHealth||null})",
        lambda raw: answer(raw))


def run_host() -> int:
    """Entry point for ``ORION.exe --face-host`` (= ``--page-host face``)."""
    return page_host.run_host("face")


# ── ORION's side ─────────────────────────────────────────────────────────────

class RemoteFace(RemoteSurface):
    """The Core Window's face, rendered by the face host and embedded here.

    Same methods as QuantumFace3D, so the window drives it identically. The
    process handling (bounded writer, restart and replay, give-up) is
    page_host.RemoteSurface's."""

    available = True
    owns_materialisation = True
    supports_orb_hands = True

    STICKY = STICKY
    LOG_TAG = "FACE"
    HOST_LOG_PREFIX = "[FACE3D]"

    def __init__(self, parent: Any = None, *, command: list[str] | None = None,
                 popen: Callable[..., Any] = subprocess.Popen) -> None:
        super().__init__(parent, page="face", kind="face",
                         command=command or host_command(), popen=popen,
                         background="#020207")
        self._form = "face"
        self.state_name = "IDLE"
        from .hud import CentralHud
        self._startup_orb = CentralHud(self)
        self._startup_orb.set_state(self.state_name)
        self._show_startup_orb()

    def _show_startup_orb(self) -> None:
        orb = getattr(self, "_startup_orb", None)
        if orb is None:
            return
        if self._layout.indexOf(orb) < 0:
            self._layout.insertWidget(0, orb, 1)
        orb.show()
        if self.isVisible() and not self._closing:
            orb.timer.start()
        else:
            orb.timer.stop()

    def _on_embedded(self, part: str) -> None:
        if part == "main":
            orb = getattr(self, "_startup_orb", None)
            if orb is not None:
                orb.timer.stop()
                orb.hide()
                self._layout.removeWidget(orb)

    def _on_restarting(self) -> None:
        self._show_startup_orb()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        orb = getattr(self, "_startup_orb", None)
        if orb is not None and not orb.isHidden() and not self._closing:
            orb.timer.start()

    def hideEvent(self, event: Any) -> None:
        orb = getattr(self, "_startup_orb", None)
        if orb is not None:
            orb.timer.stop()
        super().hideEvent(event)

    def shutdown(self) -> None:
        orb = getattr(self, "_startup_orb", None)
        if orb is not None:
            orb.timer.stop()
            if not self.failed:
                orb.hide()
        super().shutdown()

    def _dead_answer(self) -> Any:
        return False

    # ── the face API (QuantumFace3D's) ──────────────────────────────────────

    def check_state(self, callback: Callable[[Any], None]) -> None:
        self._query("check_state", callback, dead="failed")

    def check_alive(self, callback: Callable[[Any], None]) -> None:
        self._query("check_alive", callback, dead=False)

    def report(self, callback: Callable[[Any], None]) -> None:
        self._query("report", callback, dead=None)

    @property
    def form(self) -> str:
        return self._form

    def set_form(self, form: str) -> str:
        self._form = str(form or "face")
        self._send("set_form", self._form)
        return self._form

    def set_avatar_frame(self, frame: dict[str, Any]) -> None:
        self.state_name = str(frame.get("js_state") or "IDLE").upper()
        self._startup_orb.set_state(self.state_name)
        self._startup_orb.set_amplitude(frame.get("amplitude", 0.0))
        self._send("set_avatar_frame", frame)

    def set_amplitude(self, value: float) -> None:
        self._startup_orb.set_amplitude(value)
        self._send("set_amplitude", value)

    def set_spectrum(self, low: float, mid: float, high: float) -> None:
        self._send("set_spectrum", low, mid, high)

    def set_speaking(self, active: bool) -> None:
        self._startup_orb.set_state("SPEAKING" if active else "LISTENING")
        self._send("set_speaking", active)

    def set_state(self, state: str) -> None:
        self.state_name = str(state or "STANDBY").upper()
        self._startup_orb.set_state(self.state_name)
        self._send("set_state", state)

    def apply_emotion(self, name: str, params: Any = None) -> None:
        self._send("apply_emotion", name, params)

    def set_viseme(self, *args: Any) -> None:
        self._send("set_viseme", *args)

    def set_pose(self, yaw: float, pitch: float) -> None:
        self._send("set_pose", yaw, pitch)

    def set_morph(self, value: float) -> None:
        self._send("set_morph", value)

    def set_presentation(self, payload: dict) -> None:
        self._send("set_presentation", payload)

    def set_label(self, text: str) -> None:
        self._send("set_label", text)

    def set_orb_camera_enabled(self, enabled: bool) -> None:
        self._send("set_orb_camera_enabled", enabled)

    def set_orb_hands(self, sample: Any) -> None:
        self._send("set_orb_hands", sample)


__all__ = ["FLAG", "METHODS", "RemoteFace", "enabled", "host_command", "run_host", "apply", "encode"]
