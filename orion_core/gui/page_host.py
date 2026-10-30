"""Heavy pages in their own process (Mark XXXV W1.2).

Why
---
Qt draws every window of a process on that process's GUI thread, and ORION's
GUI thread is also his asyncio loop. Any Python stall therefore froze whatever
was animating for exactly as long. The face was the first page moved out
(gui/face_host.py, 10 Oct 2026: 10 stutters >= 50 ms in-process against 0 in
its own process, with ORION's thread blocked 40% of the time). The globe
(Cesium) and the Brain page's instruments stuttered for the same reason.

``ORION.exe --page-host <kind>`` runs one page in a separate process with its
own GUI thread; ORION embeds its window(s) (``QWindow.fromWinId``) and talks
to it over JSON lines on stdin/stdout. Kinds:

* ``face``  - QuantumFace3D (``--face-host`` is the same thing).
* ``web``   - any WebEngine page: ORION sends the HTML, runs JavaScript and
  receives the page's QWebChannel bridge calls (the globe).
* ``brain`` - the Brain page's Iris and Signal Lanes, fed with the same
  BrainModel events ORION's own model receives.

Rules (each one learned while building the face host)
------------------------------------------------------
* ORION never waits on a host. Commands go through a bounded queue drained by
  a writer thread; a host that stops reading loses frames, never ORION's time.
* A host never waits on ORION. It reads stdin on a worker thread and quits
  when that closes (ORION gone).
* A host that dies is restarted and its state replayed; one that keeps dying,
  or never shows a window, is given up for this run and the caller falls back
  to drawing in-process.

ORION_PAGE_PROCESS=0 keeps every page except the face in ORION's process;
ORION_GLOBE_PROCESS=0 / ORION_BRAIN_PROCESS=0 do it for one page, and
ORION_FACE_PROCESS=0 for the face, as before.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import weakref
from collections import deque
from pathlib import Path
from typing import Any, Callable

FLAG = "--page-host"
PROTOCOL_PREFIX = "@@"

#: Chromium flags ORION's own process uses for its pages (app._cap_web_heaps,
#: app._steady_web_surfaces); a host runs pages too, so it needs them as well.
HOST_CHROMIUM_FLAGS = "--js-flags=--max-old-space-size=160 --disable-features=CalculateNativeWinOcclusion"

#: Which environment switch turns each page's process off.
_SWITCHES = {"face": "ORION_FACE_PROCESS", "globe": "ORION_GLOBE_PROCESS",
             "brain": "ORION_BRAIN_PROCESS"}
_OFF = {"0", "false", "no", "off"}


def enabled(page: str = "face") -> bool:
    """Whether *page* is drawn in its own process (default yes)."""
    page = str(page or "").lower()
    if page != "face" and os.getenv("ORION_PAGE_PROCESS", "1").strip().lower() in _OFF:
        return False
    switch = _SWITCHES.get(page, f"ORION_{page.upper()}_PROCESS")
    return os.getenv(switch, "1").strip().lower() not in _OFF


def disable(page: str) -> None:
    """Off for the rest of this run: the next one built draws in-process."""
    os.environ[_SWITCHES.get(page, f"ORION_{str(page).upper()}_PROCESS")] = "0"


def host_command(kind: str, flag: str = FLAG) -> list[str]:
    """How to start a host: the same executable ORION is running as."""
    tail = [flag] if flag != FLAG else [flag, kind]
    if getattr(sys, "frozen", False):
        return [sys.executable, *tail]
    entry = Path(__file__).resolve().parents[2] / "orion.py"
    return [sys.executable, str(entry), *tail]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


def encode(method: str, args: tuple = (), rid: int | None = None) -> str:
    message: dict[str, Any] = {"m": method, "a": _jsonable(list(args))}
    if rid is not None:
        message["id"] = rid
    return json.dumps(message, separators=(",", ":")) + "\n"


# ── frame cadence, measured where the frames are drawn ───────────────────────

class FrameMeter:
    """Gaps between drawn frames: what a stutter is, measured, not guessed."""

    HITCH_MS = 50.0

    def __init__(self, keep: int = 600) -> None:
        self.stamps: deque[float] = deque(maxlen=keep)

    def tick(self, now: float | None = None) -> None:
        self.stamps.append(time.perf_counter() if now is None else now)

    def reset(self) -> None:
        self.stamps.clear()

    def summary(self) -> dict[str, Any]:
        stamps = list(self.stamps)
        gaps = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
        span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
        return {"frames": len(stamps),
                "fps": round((len(stamps) - 1) / span, 1) if span > 0 else 0.0,
                "hitches": sum(1 for g in gaps if g >= self.HITCH_MS),
                "longest_ms": round(max(gaps), 1) if gaps else 0.0}


# ── the host process ─────────────────────────────────────────────────────────

class HostPage:
    """What a kind builds in the host: its windows and how it takes commands."""

    focusable = False

    def __init__(self, windows: dict[str, Any],
                 apply: Callable[[dict[str, Any], Callable, Callable], None]) -> None:
        self.windows = windows
        self.apply = apply


def _make_sender(out: Any) -> Callable[[dict[str, Any]], None]:
    lock = threading.Lock()

    def send(message: dict[str, Any]) -> None:
        line = PROTOCOL_PREFIX + json.dumps(_jsonable(message), separators=(",", ":")) + "\n"
        with lock:
            try:
                out.write(line)
                out.flush()
            except (OSError, ValueError):
                pass
    return send


def _raise_priority() -> None:
    """Above normal: the face is what the user is looking at."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00008000)
    except Exception:
        pass


# An embedded host window is a CHILD of ORION's window, and Windows hands a
# child's WM_SETCURSOR to the parent first: DefWindowProc sends it across and
# WAITS for ORION's thread. The pointer resting on the face sends one on every
# move, so whenever ORION's thread was busy the face's thread sat waiting with
# it and the face stuttered. MEASURED 13 Oct 2026, real windows, ORION's thread
# blocked 200 ms of every 500 ms, WM_SETCURSOR at 60 Hz: 37 stutters >= 50 ms
# in 21 s, answers taking up to 202 ms; same blocks with the pointer still: 0.
# The host answers it itself. Nothing changes on screen: Qt sets the page's
# cursor on every mouse move, and its window class has none for Windows to set.
WM_SETCURSOR = 0x0020
HTCLIENT = 1


def _cursor_filter() -> Any:
    """A native event filter answering WM_SETCURSOR without asking ORION."""
    from ctypes import wintypes

    from PyQt6 import sip
    from PyQt6.QtCore import QAbstractNativeEventFilter
    from PyQt6.QtWidgets import QApplication

    # Both halves of the answer must be a voidptr: PyQt 6.11 cannot convert
    # (False, None) or a bare False and Python dies with a fatal thread-state
    # error inside the window procedure (measured 13 Oct, a real window).
    handled, passed_on = sip.voidptr(1), sip.voidptr(0)

    class OwnCursor(QAbstractNativeEventFilter):
        def nativeEventFilter(self, event_type: Any, message: Any) -> tuple:  # noqa: N802
            msg = wintypes.MSG.from_address(int(message))
            if (msg.message == WM_SETCURSOR and msg.lParam & 0xFFFF == HTCLIENT
                    and QApplication.overrideCursor() is None):
                return True, handled
            return False, passed_on

    return OwnCursor()


# -- face ---------------------------------------------------------------------

def _prepare_face() -> Any:
    from .face3d import QuantumFace3D      # WebEngine: before the QApplication
    return QuantumFace3D


def _build_face(prepared: Any, send: Callable) -> HostPage | None:
    from . import face_host
    if not prepared.available:
        send({"t": "failed", "reason": "WebEngine is not available"})
        return None
    face = prepared()
    return HostPage({"main": face},
                    lambda message, reply, quit_app: face_host.apply(face, message, reply, quit_app))


# -- web ----------------------------------------------------------------------

_PY_TYPES = {"str": str, "bool": bool, "float": float, "int": int}
_QT_TYPES = {"QString": "str", "bool": "bool", "double": "float", "float": "float",
             "int": "int", "qlonglong": "int", "uint": "int"}


def bridge_spec(obj: Any) -> dict[str, list[str]]:
    """The slots a QWebChannel object offers its page, read from Qt's own
    meta-object: {"setWorldLayer": ["str", "bool"], ...}. Slots whose
    argument types cannot cross the pipe as JSON are left out."""
    spec: dict[str, list[str]] = {}
    try:
        from PyQt6.QtCore import QMetaMethod
        meta = obj.metaObject()
        for index in range(meta.methodOffset(), meta.methodCount()):
            method = meta.method(index)
            if method.methodType() != QMetaMethod.MethodType.Slot:
                continue
            types = [_QT_TYPES.get(bytes(t).decode()) for t in method.parameterTypes()]
            if None in types:
                continue
            spec[bytes(method.name()).decode()] = types      # type: ignore[assignment]
    except Exception:
        return {}
    return spec


def _bridge_object(name: str, spec: dict[str, list[str]], emit: Callable[[str, list], None]) -> Any:
    """A QObject whose slots match *spec* and forward each call to ORION."""
    from PyQt6.QtCore import QObject, pyqtSlot
    attrs: dict[str, Any] = {}
    for method, types in spec.items():
        if not str(method).isidentifier():
            continue
        try:
            py_types = [_PY_TYPES[t] for t in types]
        except KeyError:
            continue

        def slot(self: Any, *args: Any, _method: str = method) -> None:
            emit(_method, list(args))
        attrs[method] = pyqtSlot(*py_types, name=method)(slot)
    return type(f"RemoteBridge_{name}", (QObject,), attrs)()


class _WebHost:
    """One WebEngine page, driven by ORION over the pipe."""

    PENDING_LIMIT = 64

    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        from PyQt6.QtWebEngineCore import QWebEnginePage
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        host = self
        self.send = send
        self.allow: set[str] = set()
        self.loaded = False
        self.pending: list[tuple[str, Any]] = []
        self.channel: Any = None
        self.bridges: list[Any] = []

        class Page(QWebEnginePage):
            def javaScriptConsoleMessage(self, level: Any, message: str, line: int, source: str) -> None:  # noqa: N802
                text = str(message or "")
                name = getattr(level, "name", str(level))
                if "Error" in name or "WEBGL" in text or "JS_ERROR" in text:
                    host.event("console", f"{text} @ {source}:{line}")

            def acceptNavigationRequest(self, url: Any, nav_type: Any, is_main_frame: bool) -> bool:  # noqa: N802
                target = url.toString()
                if target in host.allow or target.startswith("data:"):
                    return super().acceptNavigationRequest(url, nav_type, is_main_frame)
                if is_main_frame and target.startswith(("http://", "https://")):
                    host.event("open_url", target)        # ORION decides; the page stays
                    return False
                return super().acceptNavigationRequest(url, nav_type, is_main_frame)

        self._Page = Page
        self.view = QWebEngineView()
        self.page = Page(self.view)
        self.view.setPage(self.page)
        self.view.loadFinished.connect(self._loaded)
        self.view.renderProcessTerminated.connect(
            lambda status, code: self.event("crashed", str(getattr(status, "name", status)), int(code)))

    def event(self, name: str, *args: Any) -> None:
        self.send({"t": "event", "name": name, "args": list(args)})

    def _loaded(self, ok: bool) -> None:
        self.loaded = bool(ok)
        self.event("loaded", bool(ok))
        if ok:
            pending, self.pending = self.pending, []
            for code, rid in pending:
                self._run(code, rid)

    #: The most a page's disk cache may hold.
    DISK_CACHE_BYTES = 256 * 1024 * 1024

    def _use_disk_cache(self, folder: str) -> bool:
        """Put the page on a named profile with an on-disk HTTP cache.

        Qt's default profile is off the record, so its cache lived only as
        long as the process: every time the globe's process started again
        (each open after it was released off screen, each crash restart) it
        downloaded Cesium and every map tile it had already shown (28 Oct
        2026). No cookies are kept."""
        from PyQt6.QtWebEngineCore import QWebEngineProfile
        from PyQt6.QtWidgets import QApplication
        try:
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)
            # Owned by the application, so it outlives every page using it.
            profile = QWebEngineProfile(path.name, QApplication.instance())
            profile.setCachePath(str(path))
            profile.setPersistentStoragePath(str(path))
            profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.DiskHttpCache)
            profile.setHttpCacheMaximumSize(self.DISK_CACHE_BYTES)
            profile.setPersistentCookiesPolicy(
                QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies)
        except Exception as exc:
            self.event("console", f"disk cache unavailable - {exc}")
            return False
        old, self.page = self.page, self._Page(profile, self.view)
        self.view.setPage(self.page)
        old.deleteLater()
        return True

    def _setup(self, options: Any) -> None:
        from PyQt6.QtWebChannel import QWebChannel
        options = options if isinstance(options, dict) else {}
        if options.get("cache") and not self.loaded:
            self._use_disk_cache(str(options["cache"]))
        self.allow = {str(u) for u in options.get("allow") or []}
        bridges = options.get("bridges") or {}
        if not isinstance(bridges, dict) or not bridges:
            return
        self.channel = QWebChannel(self.page)
        for name, spec in bridges.items():
            if not isinstance(spec, dict):
                continue
            emit = (lambda method, args, _name=str(name):
                    self.event("bridge", _name, method, args))
            obj = _bridge_object(str(name), spec, emit)
            self.bridges.append(obj)
            self.channel.registerObject(str(name), obj)
        self.page.setWebChannel(self.channel)

    def _run(self, code: str, rid: Any) -> None:
        if rid is None:
            self.page.runJavaScript(code)
        else:
            self.page.runJavaScript(
                code, lambda value: self.send({"t": "reply", "id": rid, "value": _jsonable(value)}))

    def apply(self, message: dict[str, Any], reply: Callable, quit_app: Callable) -> None:
        method = str(message.get("m") or "")
        args = message.get("a") or []
        rid = message.get("id")
        if method == "setup" and args:
            self._setup(args[0])
        elif method == "set_html" and args:
            from PyQt6.QtCore import QUrl
            self.loaded = False
            base = str(args[1]) if len(args) > 1 and args[1] else ""
            self.view.setHtml(str(args[0]), QUrl(base) if base else QUrl())
        elif method == "run_js" and args:
            # Held until the page has loaded: runJavaScript on a page that has
            # only just been given its HTML can kill the process natively.
            if self.loaded:
                self._run(str(args[0]), rid)
            else:
                self.pending.append((str(args[0]), rid))
                del self.pending[:-self.PENDING_LIMIT]


def _prepare_web() -> Any:
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6 import QtWebEngineWidgets  # noqa: F401 - before the QApplication
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return None


def _build_web(_prepared: Any, send: Callable) -> HostPage:
    web = _WebHost(send)
    page = HostPage({"main": web.view}, web.apply)
    page.focusable = True            # a page may have inputs (the globe's route form)
    page.host = web                  # type: ignore[attr-defined]
    return page


# -- brain ---------------------------------------------------------------------

#: BrainModel's events, forwarded as they happen in ORION.
BRAIN_EVENTS = frozenset({
    "heard", "heard_utterance", "set_state", "set_speaking", "tool_started",
    "tool_finished", "set_telemetry", "set_link", "load_snapshot",
    "work_started", "work_finished",
})


class _BrainHost:
    """The Iris and the Signal Lanes, animated in the host at 60 fps."""

    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        from PyQt6.QtCore import QEvent, QObject, QTimer
        from .brain_instruments import (BrainModel, IrisGauge, SignalLanes, _Animator,
                                        display_state)
        self.send = send
        self.model = BrainModel()
        self.iris = IrisGauge(self.model)
        self.iris.state_label = display_state
        self.lanes = SignalLanes(self.model)
        self.stalled = 0
        self.lanes.stalled_count = lambda: self.stalled
        self.animator = _Animator(self.model, [self.iris, self.lanes], attended_ms=16)
        self.settle = QTimer()
        self.settle.setInterval(500)
        self.settle.timeout.connect(self.model.settle)
        self.meter = FrameMeter()
        meter = self.meter

        class Paints(QObject):
            def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
                if event.type() == QEvent.Type.Paint:
                    meter.tick()
                return False

        self._paints = Paints()
        self.iris.installEventFilter(self._paints)

    def set_background(self, colour: str) -> None:
        from PyQt6.QtGui import QColor, QPalette
        for widget in (self.iris, self.lanes):
            palette = widget.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(str(colour)))
            widget.setPalette(palette)
            widget.setAutoFillBackground(True)
            widget.update()

    def apply(self, message: dict[str, Any], reply: Callable, quit_app: Callable) -> None:
        method = str(message.get("m") or "")
        args = message.get("a") or []
        try:
            if method in BRAIN_EVENTS:
                getattr(self.model, method)(*args)
            elif method == "set_visible":
                visible = bool(args and args[0])
                self.animator.set_visible(visible)
                (self.settle.start if visible else self.settle.stop)()
            elif method == "set_attended":
                self.animator.set_attended(bool(args and args[0]))
            elif method == "set_stalled":
                self.stalled = int(args[0]) if args else 0
            elif method == "set_background" and args:
                self.set_background(str(args[0]))
            elif method == "report":
                reply({"t": "reply", "id": message.get("id"), "value": self.meter.summary()})
            elif method == "reset_meter":
                self.meter.reset()
        except Exception as exc:
            reply({"t": "log", "line": f"BRAIN: host could not {method} - {exc}"})


def _build_brain(_prepared: Any, send: Callable) -> HostPage:
    brain = _BrainHost(send)
    page = HostPage({"iris": brain.iris, "lanes": brain.lanes}, brain.apply)
    page.host = brain                # type: ignore[attr-defined]
    return page


#: kind -> (imports that must precede the QApplication, builder, raise priority?)
_KINDS: dict[str, tuple[Callable[[], Any], Callable[[Any, Callable], HostPage | None], bool]] = {
    "face": (_prepare_face, _build_face, True),
    "web": (_prepare_web, _build_web, False),
    "brain": (lambda: None, _build_brain, False),
}
KINDS = tuple(_KINDS)


def run_host(kind: str) -> int:
    """Entry point for ``ORION.exe --page-host <kind>`` (and ``--face-host``)."""
    from ..mcp_bundled import _ensure_stdio
    _ensure_stdio()
    send = _make_sender(sys.stdout)
    entry = _KINDS.get(str(kind or ""))
    if entry is None:
        send({"t": "failed", "reason": f"no page kind {kind!r}"})
        return 2
    prepare, build, priority = entry
    if kind in {"face", "web"}:
        flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{flags} {HOST_CHROMIUM_FLAGS}".strip()
    prepared = prepare()
    from PyQt6.QtCore import QObject, Qt, pyqtSignal
    from PyQt6.QtWidgets import QApplication

    app = QApplication([sys.argv[0]])
    if priority:
        _raise_priority()
    if sys.platform == "win32":
        own_cursor = _cursor_filter()            # lives as long as app.exec()
        app.installNativeEventFilter(own_cursor)
    try:
        page = build(prepared, send)
    except Exception as exc:
        send({"t": "failed", "reason": f"{type(exc).__name__}: {exc}"})
        return 3
    if page is None:
        return 3
    flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
    if not page.focusable:
        flags |= Qt.WindowType.WindowDoesNotAcceptFocus
    for widget in page.windows.values():
        widget.setWindowFlags(flags)
        widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        widget.resize(640, 640)
        widget.move(-20000, -20000)    # never seen before it is embedded
        widget.show()

    class Inbox(QObject):
        got = pyqtSignal(object)

    inbox = Inbox()

    def quit_quietly() -> None:
        # Hidden FIRST. When ORION's window goes, Qt un-embeds this one, and a
        # window that still says "visible" then flashed up on its own while
        # WebEngine wound down (owner, 12 Oct: "his face screen pops up for a
        # split second right after shutting down").
        for widget in page.windows.values():
            try:
                widget.hide()
            except RuntimeError:
                pass
        app.quit()

    def dispatch(message: Any) -> None:
        if not isinstance(message, dict):
            return
        if message.get("m") == "__quit__":
            quit_quietly()
            return
        page.apply(message, send, quit_quietly)

    inbox.got.connect(dispatch)

    def read_commands() -> None:
        for line in sys.stdin:
            try:
                inbox.got.emit(json.loads(line))
            except (ValueError, RuntimeError):
                continue
        try:
            inbox.got.emit({"m": "__quit__"})      # ORION is gone
        except RuntimeError:
            pass

    threading.Thread(target=read_commands, name=f"page-host-{kind}-stdin", daemon=True).start()

    # A window ORION has embedded can be REPLACED: when WebEngine's GPU
    # surface comes up, Qt destroys the top-level native window and creates
    # another of the right surface type. The handle ORION embedded is then
    # dead and the page draws, unseen, in a new top-level window at -20000.
    # (Found 11 Oct 2026: the face and globe hosts measured 60 fps that way,
    # off screen.) Every new handle is announced, and ORION embeds it again.
    from PyQt6.QtCore import QEvent

    announced: dict[str, int] = {}

    def announce(part: str, widget: Any) -> None:
        wid = int(widget.winId())
        if announced.get(part) != wid:
            if part in announced:
                # A REPLACED window is a fresh top-level wherever Qt last
                # placed it: park it off screen until ORION embeds it again,
                # or it shows for a moment as a window of its own.
                widget.move(-20000, -20000)
            announced[part] = wid
            send({"t": "wid", "wid": wid, "pid": os.getpid(), "part": part})

    class WinIdWatch(QObject):
        def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.WinIdChange:
                for part, widget in page.windows.items():
                    if widget is obj:
                        # After the change has settled (the new handle exists).
                        QTimer.singleShot(0, lambda p=part, w=widget: announce(p, w))
            return False

    from PyQt6.QtCore import QTimer
    watch = WinIdWatch()
    for part, widget in page.windows.items():
        widget.installEventFilter(watch)
        announce(part, widget)
    return app.exec()


# ── ORION's side ─────────────────────────────────────────────────────────────

try:
    from PyQt6.QtCore import QTimer, pyqtSignal
    from PyQt6.QtGui import QWindow
    from PyQt6.QtWidgets import QVBoxLayout, QWidget
except Exception:                      # pragma: no cover - Qt missing
    QWidget = object                   # type: ignore[assignment,misc]


#: Every page host ORION has started and not yet let go of.
_LIVE_SURFACES: "weakref.WeakSet[Any]" = weakref.WeakSet()


def shutdown_all() -> int:
    """Hide and stop every page host - the first thing ORION's shutdown does,
    before its own windows go, so no host window can outlive them on screen.
    Returns how many were asked."""
    count = 0
    for surface in list(_LIVE_SURFACES):
        try:
            surface.shutdown()
            count += 1
        except Exception:
            pass
    return count


class RemoteSurface(QWidget):  # type: ignore[misc,valid-type]
    """A page drawn by a host process and embedded here.

    Single-window pages embed into this widget itself. A page with several
    windows ("parts") embeds each into a holder from :meth:`part`, which the
    owner places in its own layout.
    """

    #: Restarts allowed before the page is given up for this run.
    MAX_RESTARTS = 3
    #: How long a host may take to show its windows before it counts as failed.
    START_TIMEOUT_MS = 15000
    #: Queued commands beyond this are dropped (frames are superseded anyway).
    OUTBOX_LIMIT = 512
    #: Commands re-sent to a restarted host so it comes back as it was.
    STICKY: tuple[str, ...] = ()
    #: Prefix of this page's log lines.
    LOG_TAG = "PAGE"
    #: Host stdout lines (not protocol) with this prefix are logged, re-tagged.
    HOST_LOG_PREFIX = ""
    #: Called once if the page is given up for this run (the owner falls back).
    on_failed: Callable[[], Any] | None = None

    _inbox = pyqtSignal(object)

    def __init__(self, parent: Any = None, *, page: str = "page", kind: str = "web",
                 parts: tuple[str, ...] = ("main",), command: list[str] | None = None,
                 popen: Callable[..., Any] = subprocess.Popen, focusable: bool = False,
                 background: str = "#020207") -> None:
        super().__init__(parent)
        self.page_name = page
        self.setStyleSheet(f"background:{background};")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._command = command or host_command(kind)
        self._popen = popen
        self._focusable = focusable
        self._parts = tuple(parts) or ("main",)
        self._holders: dict[str, Any] = {}
        for name in self._parts:
            if name == "main":
                self._holders[name] = self
            else:
                holder = QWidget(self)
                holder.setStyleSheet(f"background:{background};")
                box = QVBoxLayout(holder)
                box.setContentsMargins(0, 0, 0, 0)
                holder._box = box           # type: ignore[attr-defined]
                self._holders[name] = holder
        self._proc: Any = None
        self._containers: dict[str, Any] = {}
        self._outbox: "queue.Queue[str | None]" = queue.Queue(maxsize=self.OUTBOX_LIMIT)
        self._sticky: dict[str, tuple] = {}
        self._callbacks: dict[int, Callable[[Any], None]] = {}
        self._callback_dead: dict[int, Any] = {}
        self._next_id = 1
        self._restarts = 0
        self._closing = False
        self.failed = False
        self.host_pid: int | None = None
        #: The native window handle embedded for each part, as last announced.
        self.host_wids: dict[str, int] = {}
        self._inbox.connect(self._on_message)
        _LIVE_SURFACES.add(self)
        self._start()

    # ── parts ───────────────────────────────────────────────────────────────

    def part(self, name: str) -> Any:
        """The widget a part's window is embedded into (place it yourself)."""
        return self._holders[name]

    @property
    def _container(self) -> Any:
        """The main part's container (the face's tests read this name)."""
        return self._containers.get("main")

    @_container.setter
    def _container(self, value: Any) -> None:
        if value is None:
            self._containers.pop("main", None)
        else:
            self._containers["main"] = value

    # ── process ─────────────────────────────────────────────────────────────

    def _inactive(self) -> bool:
        if self._closing or self.failed:
            return True
        try:
            from PyQt6 import sip
            return sip.isdeleted(self)
        except (ImportError, TypeError):
            return False

    def _start(self) -> None:
        # A restart timer can outlive shutdown, fallback or the C++ widget.
        if self._inactive():
            return
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = self._popen(self._command, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=flags, bufsize=1, text=True,
                                     encoding="utf-8", errors="replace")
        except Exception as exc:
            self._log(f"{self.LOG_TAG}: could not start the {self.page_name} process - {exc}")
            self._give_up()
            return
        proc = self._proc
        tag = self.page_name.replace(" ", "-")
        threading.Thread(target=self._read, args=(proc,), name=f"{tag}-host-read",
                         daemon=True).start()
        threading.Thread(target=self._write, args=(proc, self._outbox), name=f"{tag}-host-write",
                         daemon=True).start()
        self._replay()
        started = proc
        QTimer.singleShot(self.START_TIMEOUT_MS, lambda: self._check_started(started))

    def _replay(self) -> None:
        for method, args in self._sticky.items():
            self._send(method, *args)

    def _check_started(self, proc: Any) -> None:
        if proc is self._proc and not self._containers and not self._closing:
            self._log(f"{self.LOG_TAG}: the {self.page_name} process never showed a window.")
            self._give_up()

    def _read(self, proc: Any) -> None:
        prefix = self.HOST_LOG_PREFIX

        def deliver(message: Any) -> None:
            try:
                # Keep a strong sender reference until queued delivery. Old
                # window handles, replies and events cannot affect a restart.
                self._inbox.emit((proc, message))
            except RuntimeError:
                pass                       # the C++ widget is already gone

        try:
            for line in proc.stdout:
                if line.startswith(PROTOCOL_PREFIX):
                    try:
                        deliver(json.loads(line[len(PROTOCOL_PREFIX):]))
                    except ValueError:
                        continue
                elif prefix and line.startswith(prefix):
                    deliver({"t": "log", "line": f"{self.LOG_TAG}: "
                             + line[len(prefix):].strip()})
        except (OSError, ValueError):
            pass
        deliver({"t": "exit", "proc": id(proc)})

    def _write(self, proc: Any, outbox: "queue.Queue[str | None]") -> None:
        while True:
            line = outbox.get()
            if line is None or proc is not self._proc:
                break
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (OSError, ValueError):
                break
            if self._closing:
                break

    def _send(self, method: str, *args: Any, rid: int | None = None) -> None:
        if self._closing or self.failed:
            return
        if method in self.STICKY:
            self._sticky[method] = args
        try:
            self._outbox.put_nowait(encode(method, args, rid))
        except queue.Full:
            pass                       # a stalled host loses frames, never ORION's time

    def _on_message(self, message: Any) -> None:
        if self._inactive():
            return
        if isinstance(message, tuple) and len(message) == 2:
            sender, message = message
            if sender is not self._proc:
                return
        if not isinstance(message, dict):
            return
        kind = message.get("t")
        if kind == "wid":
            self.host_pid = message.get("pid")
            self._embed(int(message.get("wid") or 0), str(message.get("part") or "main"))
        elif kind == "reply":
            rid = int(message.get("id") or 0)
            callback = self._callbacks.pop(rid, None)
            self._callback_dead.pop(rid, None)
            if callback is not None:
                try:
                    callback(message.get("value"))
                except Exception:
                    pass
        elif kind == "event":
            args = message.get("args")
            try:
                self._on_event(str(message.get("name") or ""), args if isinstance(args, list) else [])
            except Exception as exc:
                self._log(f"{self.LOG_TAG}: {self.page_name} event failed - {exc}")
        elif kind == "log":
            self._log(str(message.get("line") or ""))
        elif kind == "failed":
            self._log(f"{self.LOG_TAG}: the {self.page_name} process cannot draw - "
                      f"{message.get('reason')}")
            self._give_up()
        elif kind == "exit" and message.get("proc") == id(self._proc):
            self._on_exit()

    def _on_event(self, name: str, args: list[Any]) -> None:
        """A page event (subclasses)."""

    def _embed(self, wid: int, part: str = "main") -> None:
        """Embed (or re-embed: a host announces a replaced window) one part."""
        if not wid or self._closing or part not in self._holders:
            return
        if self.host_wids.get(part) == wid and part in self._containers:
            return
        self.host_wids[part] = wid
        from PyQt6.QtCore import Qt
        holder = self._holders[part]
        window = QWindow.fromWinId(wid)
        container = QWidget.createWindowContainer(window, holder)
        container.setFocusPolicy(Qt.FocusPolicy.ClickFocus if self._focusable
                                 else Qt.FocusPolicy.NoFocus)
        old = self._containers.get(part)
        box = self._layout if holder is self else holder._box
        if old is not None:
            box.removeWidget(old)
            old.deleteLater()
        self._containers[part] = container
        box.addWidget(container)
        if old is not None:
            self._log(f"{self.LOG_TAG}: the {self.page_name} window was replaced by its "
                      f"process - embedded again ({part}).")
        self._on_embedded(part)

    def _on_embedded(self, part: str) -> None:
        """A part's window is on screen (subclasses)."""

    def _drop_containers(self) -> None:
        """Take dead hosts' windows out of the layout. (The face host only
        forgot them, so a restarted face shared its space with an empty one.)"""
        for part, old in list(self._containers.items()):
            holder = self._holders.get(part, self)
            box = self._layout if holder is self else holder._box
            try:
                box.removeWidget(old)
                old.deleteLater()
            except RuntimeError:
                pass
        self._containers.clear()
        self.host_wids.clear()

    def _on_exit(self) -> None:
        if self._inactive():
            return
        proc = self._proc
        self._proc = None
        self._retire_outbox()
        self._outbox = queue.Queue(maxsize=self.OUTBOX_LIMIT)
        self._reap_process(proc)
        self._drop_containers()
        self._finish_queries()
        if self._inactive():
            return
        self._restarts += 1
        if self._restarts > self.MAX_RESTARTS:
            self._log(f"{self.LOG_TAG}: the {self.page_name} process keeps stopping - "
                      "drawing it in ORION's own process.")
            self._give_up()
            return
        self._log(f"{self.LOG_TAG}: the {self.page_name} process stopped - "
                  f"restarting it ({self._restarts}).")
        self._on_restarting()
        QTimer.singleShot(1000, self._start)

    def _on_restarting(self) -> None:
        """The host died and is about to be restarted (subclasses)."""

    def _give_up(self) -> None:
        """Off for this run: the next one built draws in-process."""
        if self.failed:
            return
        self.failed = True
        disable(self.page_name)
        # A failed host may still be alive but silent. Retire it even when
        # its owner has no fallback handler attached yet.
        self.shutdown()
        if self.on_failed is not None:
            try:
                self.on_failed()
            except Exception as exc:
                self._log(f"{self.LOG_TAG}: falling back failed - {exc}")

    def _dead_answer(self) -> Any:
        return None

    def _finish_queries(self) -> None:
        callbacks, self._callbacks = self._callbacks, {}
        dead, self._callback_dead = self._callback_dead, {}
        for rid, callback in callbacks.items():
            try:
                callback(dead.get(rid, self._dead_answer()))
            except Exception:
                pass

    def _retire_outbox(self, *, quit_host: bool = False) -> None:
        # Superseded frames must not keep the writer asleep on a full queue,
        # or prevent the final quit/sentinel from being admitted.
        while True:
            try:
                self._outbox.get_nowait()
            except queue.Empty:
                break
        try:
            if quit_host:
                self._outbox.put_nowait(encode("__quit__"))
            self._outbox.put_nowait(None)
        except queue.Full:
            pass

    def _log(self, line: str) -> None:
        if not line:
            return
        try:
            from .. import session_log
            session_log.write(line)
        except Exception:
            pass

    def _hide_host_windows(self) -> None:
        """Hide the host's windows from THIS side, at once. When ORION's own
        window goes, Qt un-embeds them, and a host window still marked visible
        flashed up on its own until the host had finished quitting (owner,
        12 Oct). ShowWindowAsync never waits on the other process."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            for wid in list(self.host_wids.values()):
                if wid and user32.IsWindow(wid):
                    user32.ShowWindowAsync(wid, 0)          # SW_HIDE
        except Exception:
            pass

    def shutdown(self) -> None:
        """Hide the host's windows, then ask it to quit; it also quits on its
        own when ORION exits."""
        self._hide_host_windows()
        if self._closing:
            return
        self._closing = True
        proc = self._proc
        self._retire_outbox(quit_host=True)
        self._finish_queries()
        self._reap_process(proc)

    def _reap_process(self, proc: Any) -> None:
        # EOF can precede actual process exit. Keep ownership of a retired
        # process until it exits or the same bounded shutdown grace kills it.
        if proc is None:
            return
        def _reap() -> None:
            try:
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        threading.Thread(target=_reap, name=f"{self.page_name}-host-reap", daemon=True).start()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self.shutdown()
        super().closeEvent(event)

    def _query(self, method: str, callback: Callable[[Any], None], *, dead: Any,
               args: tuple = ()) -> None:
        if self._inactive() or self._proc is None:
            try:
                callback(dead)
            except Exception:
                pass
            return
        rid = self._next_id
        self._next_id += 1
        self._callbacks[rid] = callback
        self._callback_dead[rid] = dead
        self._send(method, *args, rid=rid)


# ── the web page (the globe) ─────────────────────────────────────────────────

def _url_text(url: Any) -> str:
    if url is None:
        return ""
    to_string = getattr(url, "toString", None)
    return str(to_string() if callable(to_string) else url)


class RemoteWebView(RemoteSurface):
    """A WebEngine page in a host, with the part of QWebEngineView's API the
    globe uses: ``setHtml``, ``page().runJavaScript`` (with or without a
    callback), ``loadFinished`` and ``renderProcessTerminated``.

    *bridges* are the QWebChannel objects the page talks to, keyed by the
    name the page looks them up by. Their slots are read from Qt's own
    meta-object, mirrored in the host, and a call from the page runs the real
    slot here - only a slot that exists on the object, never another name.
    """

    STICKY = ("setup", "set_html")
    LOG_TAG = "PAGE"

    loadFinished = pyqtSignal(bool)
    renderProcessTerminated = pyqtSignal(object, int)

    def __init__(self, parent: Any = None, *, page: str = "web",
                 bridges: dict[str, Any] | None = None, allow: tuple[str, ...] = (),
                 on_js_error: Callable[[str], Any] | None = None,
                 on_open_url: Callable[[str], Any] | None = None,
                 log_tag: str = "PAGE", cache: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("kind", "web")
        kwargs.setdefault("focusable", True)
        super().__init__(parent, page=page, **kwargs)
        # Messages only arrive through the queued inbox, so nothing reads these
        # before the constructor has finished.
        self.LOG_TAG = log_tag
        self._bridges = dict(bridges or {})
        self._specs = {name: bridge_spec(obj) for name, obj in self._bridges.items()}
        self._allow = tuple(allow)
        self._on_js_error = on_js_error
        self._on_open_url = on_open_url
        setup = {"bridges": self._specs, "allow": list(self._allow)}
        if cache:
            setup["cache"] = str(cache)            # _WebHost._use_disk_cache
        self._send("setup", setup)

    def page(self) -> "RemoteWebView":
        return self

    def setHtml(self, html: str, base_url: Any = None) -> None:  # noqa: N802 - Qt's name
        self._send("set_html", str(html), _url_text(base_url))

    def runJavaScript(self, code: str, callback: Callable[[Any], None] | None = None) -> None:  # noqa: N802
        if callback is None:
            self._send("run_js", str(code))
        else:
            self._query("run_js", callback, dead=None, args=(str(code),))

    def _on_restarting(self) -> None:
        self.loadFinished.emit(False)      # the page is gone until the new one loads

    def _on_event(self, name: str, args: list[Any]) -> None:
        if name == "loaded":
            self.loadFinished.emit(bool(args and args[0]))
        elif name == "crashed":
            status = args[0] if args else "unknown"
            code = int(args[1]) if len(args) > 1 else -1
            self.renderProcessTerminated.emit(status, code)
        elif name == "console":
            if self._on_js_error is not None and args:
                self._on_js_error(str(args[0])[:400])
        elif name == "open_url":
            url = str(args[0]) if args else ""
            if url.startswith(("http://", "https://")):
                if self._on_open_url is not None:
                    self._on_open_url(url)
                else:
                    from ..browser import open_url
                    open_url(url)
        elif name == "bridge" and len(args) >= 3:
            self._call_bridge(str(args[0]), str(args[1]), args[2])

    def _call_bridge(self, name: str, method: str, params: Any) -> None:
        obj = self._bridges.get(name)
        types = self._specs.get(name, {}).get(method)
        if obj is None or types is None or not isinstance(params, list) or len(params) != len(types):
            self._log(f"{self.LOG_TAG}: ignored a bridge call that is not on the page's bridge "
                      f"({name}.{method})")
            return
        try:
            values = [_PY_TYPES[t](v) for t, v in zip(types, params)]
        except (TypeError, ValueError):
            return
        getattr(obj, method)(*values)


# ── the Brain page's instruments ─────────────────────────────────────────────

class RemoteBrain(RemoteSurface):
    """The Iris and the Signal Lanes, drawn by a host at 60 fps.

    Fed with the BrainModel events ORION's own model receives; the host keeps
    its own model and animates it, so a stall in ORION delays an event but
    never freezes the instruments."""

    STICKY = ("set_background", "set_visible", "set_attended")
    LOG_TAG = "BRAIN"
    #: The model's picture now (BrainModel.snapshot); a restarted host is
    #: given it, so it comes back showing what ORION's own model shows.
    snapshot_provider: Callable[[], dict[str, Any]] | None = None

    def __init__(self, parent: Any = None, *, background: str = "#0b0d12",
                 **kwargs: Any) -> None:
        kwargs.setdefault("kind", "brain")
        super().__init__(parent, page="brain", parts=("iris", "lanes"),
                         background=background, **kwargs)
        self._send("set_background", background)

    def _replay(self) -> None:
        super()._replay()
        provider = self.snapshot_provider
        if provider is not None:
            try:
                self._send("load_snapshot", provider())
            except Exception:
                pass

    def forward(self, method: str, *args: Any) -> None:
        if method in BRAIN_EVENTS:
            self._send(method, *args)

    def set_visible(self, visible: bool) -> None:
        self._send("set_visible", bool(visible))

    def set_attended(self, attended: bool) -> None:
        self._send("set_attended", bool(attended))

    def set_stalled(self, count: int) -> None:
        self._send("set_stalled", int(count))

    def report(self, callback: Callable[[Any], None]) -> None:
        self._query("report", callback, dead=None)


__all__ = [
    "BRAIN_EVENTS", "FLAG", "FrameMeter", "HOST_CHROMIUM_FLAGS", "KINDS", "PROTOCOL_PREFIX",
    "RemoteBrain", "RemoteSurface", "RemoteWebView", "bridge_spec", "disable", "enabled",
    "encode", "host_command", "run_host",
]
