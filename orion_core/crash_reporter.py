"""
crash_reporter.py — ORION's black box.

The self-repair agent (selfrepair.py) captures *Python* exceptions into an
in-memory incident list + a JSONL journal.  That misses the failures that
actually took the GUI down in practice:

  • Qt-level faults — ``qFatal`` / ``qCritical`` from Chromium (QtWebEngine) and
    OpenGL, which abort the process without ever reaching a Python ``except``;
  • hard C-level crashes — a GPU-driver or WebGL fault segfaulting the embedded
    Chromium render process, which leaves no Python traceback at all;
  • the globe's / face's render process terminating (blank or frozen view).

This module is the black box that records all of them to plain-text files under
``config/crash_reports/`` so a launch that "crashed a few times" is diagnosable
after the fact — and so ORION can read its own crash reports back.

Design rules: it must NEVER raise (a crash reporter that crashes is worse than
none), it chains rather than replaces ``sys.excepthook`` (so selfrepair still
runs), and every writer is wrapped so a full disk or a locked file is silent.
"""

from __future__ import annotations

import faulthandler
import io
import platform
import re
import sys
import traceback as tb_module
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

CRASH_DIR = CONFIG_DIR / "crash_reports"
_RETENTION = 40                     # keep the most recent N reports
_MAX_REPORTS_PER_KIND_BURST = 6     # stop a crash loop from flooding the disk

# The plain-language guide dropped into the folder so it explains itself.
_README = """\
O.R.I.O.N. — CRASH REPORTS
==========================

This folder is ORION's black box.  Whenever something goes wrong that could
take a component — or the whole app — down, ORION writes a plain-text report
here saying WHAT happened, WHY, and what you can do about it.

Each *.txt file is one incident, named  <date>_<time>_<kind>.txt :

  python       - an internal Python error (full traceback included)
  qt-fatal     - a fatal Qt / interface fault
  qt-gpu       - a graphics / GPU driver fault
  globe-render - the 3-D globe's render process crashed (usually a GPU reset)
  globe-js     - a script error inside the globe page
  face / other - component-specific faults

Every report opens with a "WHAT THIS MEANS" section in plain English, so you
never need to decode a stack trace to understand it.  The newest files are the
most relevant; ORION keeps the most recent 40 and prunes older ones.
faulthandler.log captures hard native crashes (segfaults) that leave no Python
traceback at all.

You do not have to open these by hand — ask ORION "why did you crash?" or open
the Diagnostics Centre, which lists recent crashes with their plain-English
cause.
"""


def interpret(kind: str, summary: str) -> dict[str, Any]:
    """Translate a crash *kind* + *summary* into a plain-English explanation the
    user can act on.  Shared by the report body, the Diagnostics Centre and the
    on-screen banner so a crash is never a silent mystery.

    Returns keys: headline (one line), explanation, advice, spoken (a JARVIS
    line, or ''), severe (bool)."""
    k = (kind or "").lower()
    blob = f"{k} {(summary or '').lower()}"
    # Native faults first: "access violation" is in the GPU vocabulary below,
    # but most of ORION's recorded access violations were not graphics at all
    # (30 Sep 2026: an import of win32ui on a worker thread).
    if k.startswith("native"):
        graphics = any(t in blob for t in (
            "opengl", "paintgl", "webengine", "webgl", "globe", "d3d", "gl_", "vulkan"))
        advice = ("Updating your graphics driver is the usual remedy if it recurs."
                  if graphics else
                  "If it recurs, the report names the threads involved — ask ORION "
                  "to run a diagnostic, or share the report.")
        if k == "native-exit":
            return {
                "headline": "A native fault occurred while ORION was shutting down",
                "explanation": ("A compiled library faulted during shutdown, after the "
                                "session had ended — not while ORION was in use. "
                                f"{summary}"),
                "advice": advice,
                "spoken": "",
                "severe": False,
            }
        return {
            "headline": "ORION's previous session ended in a native crash",
            "explanation": ("A compiled library (not ORION's Python code) faulted and "
                            "Windows closed the app without a Python traceback. "
                            f"{summary}"),
            "advice": advice,
            "spoken": "",
            "severe": True,
        }
    gpu = any(t in blob for t in (
        "gpu", "webgl", "opengl", "gl_", "d3d", "vulkan", "angle",
        "render process", "access violation", "context lost", "context_lost",
        "device removed", "graphics", "tdr"))
    if "globe-render" in k or (gpu and "globe" in k) or ("render" in k and gpu):
        return {
            "headline": "The 3-D globe's graphics driver reset under load",
            "explanation": (
                "The globe asked the graphics card to draw faster than its "
                "driver could keep up with, so the driver reset the display "
                "context. ORION rebuilds the globe automatically when this "
                "happens."),
            "advice": (
                "If it keeps recurring, update your graphics driver. The globe "
                "already drops to a lighter render mode while you move it, so "
                "it should stay stable even under fast zooming."),
            "spoken": ("The globe's graphics driver reset under load. "
                       "I've rebuilt it and logged the details to my crash reports."),
            "severe": True,
        }
    if gpu:
        return {
            "headline": "A graphics / GPU fault occurred",
            "explanation": ("A component that draws with the GPU — the globe or "
                            "the face — hit a graphics-driver fault."),
            "advice": "Updating your graphics driver is the usual remedy if it recurs.",
            "spoken": ("I caught a graphics fault and recovered. "
                       "The details are in my crash reports."),
            "severe": True,
        }
    if "globe-js" in k:
        return {
            "headline": "A script error inside the globe page",
            "explanation": ("Something in the globe's web page raised an error. "
                            "The globe keeps running; this is recorded for diagnosis."),
            "advice": "No action needed unless the globe visibly misbehaves.",
            "spoken": "",
            "severe": False,
        }
    if "qt-fatal" in k:
        return {
            "headline": "A fatal interface (Qt) error aborted a component",
            "explanation": "A low-level Qt / interface fault aborted a component.",
            "advice": "If ORION closed unexpectedly, restart him; the report has the detail.",
            "spoken": ("A fatal interface fault occurred. "
                       "I've written a full report to my crash reports."),
            "severe": True,
        }
    if "python" in k:
        return {
            "headline": "An unexpected internal (Python) error",
            "explanation": ("An unhandled Python exception was raised; the "
                            "self-repair agent has the traceback."),
            "advice": "The traceback below points to the module at fault.",
            "spoken": "",
            "severe": False,
        }
    return {
        "headline": f"A {kind or 'system'} fault was recorded",
        "explanation": "ORION captured a fault for later diagnosis.",
        "advice": "See the detail below.",
        "spoken": "",
        "severe": False,
    }


def _field(text: str, name: str) -> str:
    """Pull a  '<name> : <value>'  header line back out of a written report."""
    match = re.search(rf"^{re.escape(name)}\s*:\s*(.*)$", text, re.M)
    return match.group(1).strip() if match else ""


# ── reading faulthandler.log back ─────────────────────────────────────────────
#
# faulthandler can only WRITE: a hard crash leaves a stack dump in
# faulthandler.log and nothing else, so the Diagnostics Centre, "why did you
# crash?" and the crash banner never heard about it. The 30 Sep 2026 access
# violation sat there unread for three days. At the next start-up the previous
# session's slice of the log is read back and turned into a normal report.

_ARMED = re.compile(r"^===== faulthandler armed (\d{8}_\d{6})(?:_\d+)? =====[ \t]*$", re.M)
_FATAL = re.compile(r"^(?:Windows fatal exception|Fatal Python error): *(.+)$", re.M)
_THREAD = re.compile(r"^(Current thread|Thread) 0x[0-9a-fA-F]+ \(most recent call first\):",
                     re.M)
_FRAME = re.compile(r'^File "(?P<file>[^"]+)", line (?P<line>\d+) in (?P<func>.+)$')
#: COM/RPC HRESULTs (FACILITY_RPC 0x8001xxxx, RPC_S_* 0x800706xx) - first-chance
#: exceptions COM raises and handles itself; faulthandler logs them all.
_FIRST_CHANCE_COM = re.compile(r"\bcode 0x800(?:1[0-9a-f]{4}|706[0-9a-f]{2})\b", re.I)
#: Once the log passes this size only its newest part is kept.
_FAULT_LOG_LIMIT = 512 * 1024
_FAULT_LOG_KEEP = 128 * 1024


def _thread_stacks(block: str) -> list[tuple[bool, list[str]]]:
    """(is the faulting thread, stripped stack lines) for each thread dumped."""
    heads = list(_THREAD.finditer(block))
    stacks: list[tuple[bool, list[str]]] = []
    for index, head in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(block)
        body = block[head.end():end]
        stop = _ARMED.search(body)
        if stop is not None:
            body = body[:stop.start()]
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        stacks.append((head.group(1) == "Current thread", lines))
    return stacks


def _short_path(path: str) -> str:
    """'C:\\...\\ORION Main\\orion_core\\audio.py' -> 'orion_core/audio.py'."""
    parts = path.replace("\\", "/").split("/")
    for anchor in ("orion_core", "site-packages"):
        if anchor in parts:
            index = parts.index(anchor)
            parts = parts[index:] if anchor == "orion_core" else parts[index + 1:]
            break
    return "/".join(parts)


def _frame(line: str) -> tuple[str, str, str] | None:
    match = _FRAME.match(line)
    if match is None:
        return None
    return _short_path(match.group("file")), match.group("line"), match.group("func").strip()


def _own(frame: tuple[str, str, str] | None) -> bool:
    return frame is not None and frame[0].startswith("orion_core/")


def _describe(frame: tuple[str, str, str]) -> str:
    return f"{frame[0]}:{frame[1]} {frame[2]}"


def analyse_fault_block(block: str) -> dict[str, Any] | None:
    """Summarise one session's slice of faulthandler.log.

    None when the slice holds no fatal error. Otherwise: ``error`` (the
    exception name), ``where`` (the faulting thread's position), ``during``
    (what other threads were in the middle of — a native module being
    imported is the usual suspect), ``shutting_down`` (the shutdown watchdog
    thread was alive) and ``summary`` (one line combining them).
    """
    fatal = _FATAL.search(block or "")
    if fatal is None:
        return None
    stacks = _thread_stacks(block)
    shutting_down = any(
        (frame := _frame(line)) is not None and frame[2] == "_force_exit"
        for _current, lines in stacks for line in lines)
    current = next((lines for is_current, lines in stacks if is_current), [])
    frames = [frame for frame in map(_frame, current) if frame is not None]
    where = ""
    if frames:
        where = f"in {_describe(frames[0])}"
        own = next((frame for frame in frames if _own(frame)), None)
        if own is not None and own != frames[0]:
            where += f" (called from {_describe(own)})"
    elif current:
        where = "on a thread with no Python frame"
    if current and current[0].lower().startswith("garbage-collecting"):
        where = f"during garbage collection {where}".strip()
    during: list[str] = []
    for is_current, lines in stacks:
        if is_current or not any("importlib._bootstrap" in line for line in lines):
            continue
        parsed = [frame for frame in map(_frame, lines) if frame is not None]
        target = next((frame for frame in parsed
                       if frame[2] == "<module>" and not frame[0].startswith("<frozen")),
                      None)
        if target is None:
            continue
        module = target[0].removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        caller = next((frame for frame in parsed if _own(frame)), None)
        text = f"another thread was importing {module}"
        if caller is not None:
            text += f" (from {_describe(caller)})"
        during.append(text)
    summary = f"{fatal.group(1).strip()} {where}".strip()
    if during:
        summary += "; at that moment " + "; ".join(during[:2])
    return {"error": fatal.group(1).strip(), "where": where, "during": during,
            "shutting_down": shutting_down, "summary": summary + "."}


def last_real_fault(session: str) -> dict[str, Any] | None:
    """The last fatal dump in one session's slice that is not a first-chance
    COM/RPC exception, analysed; None when there is none.

    One session can log forty handled COM dumps and then really crash (Mark
    XL.4): reading only the first dump would call it COM noise and hide the
    access violation behind it. Each dump runs from its fatal line to the
    next one."""
    starts = [m.start() for m in _FATAL.finditer(session or "")]
    for index in range(len(starts) - 1, -1, -1):
        end = starts[index + 1] if index + 1 < len(starts) else len(session)
        found = analyse_fault_block(session[starts[index]:end])
        if found is not None and not _FIRST_CHANCE_COM.search(str(found.get("error") or "")):
            return found
    return None


def _clean_shutdown_at() -> datetime | None:
    """When the last clean shutdown report was written (local time)."""
    try:
        text = (CONFIG_DIR / "diagnostics" / "last_shutdown.txt").read_text(
            encoding="utf-8", errors="replace")
        return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def previous_native_fault(log_path: Path) -> dict[str, Any] | None:
    """The fault that ended the PREVIOUS session, read before this session
    arms faulthandler (so the last slice of the log is that session's).

    Adds ``kind``: ``native-crash`` (died while in use) or ``native-exit``
    (faulted while shutting down). Returns None for a session that logged a
    Windows exception but carried on — faulthandler's handler sees first-chance
    exceptions that something then handled (0x8001010d from COM, logged
    23 Sep 2026, was one), and calling those crashes would cry wolf.
    Never raises.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        faulted_at = datetime.fromtimestamp(log_path.stat().st_mtime)
    except Exception:
        return None
    marks = list(_ARMED.finditer(text))
    if not marks:
        return None
    found = last_real_fault(text[marks[-1].end():])
    if found is None:
        # A COM/RPC HRESULT (0x8001010e RPC_E_WRONG_THREAD from a WinRT media
        # call, 27 Oct 2026 02:36, logged while ORION shut down cleanly) is
        # raised and caught inside COM: never what ended a process. Filed as
        # crashes, three of them each sent a Reliability Agent to investigate.
        return None
    try:
        armed = datetime.strptime(marks[-1].group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    shutdown = _clean_shutdown_at()
    if shutdown is not None and shutdown >= armed:
        # The session reached a clean shutdown. A dump written after that is
        # an exit fault; one written before it was survived.
        if faulted_at < shutdown:
            return None
        found["kind"] = "native-exit"
    else:
        found["kind"] = "native-exit" if found["shutting_down"] else "native-crash"
    found["armed"] = armed
    found["faulted_at"] = faulted_at
    return found


def _trim_fault_log(log_path: Path) -> None:
    """Keep faulthandler.log bounded: past the limit, keep its newest part,
    starting at a session marker so no dump is cut in half. Never raises."""
    try:
        if log_path.stat().st_size <= _FAULT_LOG_LIMIT:
            return
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = text[-_FAULT_LOG_KEEP:]
        mark = _ARMED.search(tail)
        if mark is not None:
            tail = tail[mark.start():]
        from .atomic_io import atomic_write_text
        atomic_write_text(log_path, tail)
    except Exception:
        pass


class CrashReporter:
    """Writes crash reports and installs Qt / faulthandler / excepthook hooks."""

    def __init__(self, bus: Any, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._prev_excepthook: Any = None
        self._fault_log: io.TextIOBase | None = None
        self._burst: dict[str, int] = {}
        self._announced: set[str] = set()   # spoke about this kind already this session
        try:
            CRASH_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ── installation ──────────────────────────────────────────────────────────

    def install(self) -> None:
        """Arm all capture paths.  Call AFTER selfrepair.install so this wraps
        (and still calls) its excepthook."""
        self._write_readme()
        self._install_excepthook()
        self._install_qt_handler()
        self._install_faulthandler()
        try:
            self.bus.log.emit("CRASH: black-box recorder armed → config/crash_reports/.")
        except Exception:
            pass

    def _write_readme(self) -> None:
        """Drop a plain-language README so the folder explains itself."""
        try:
            CRASH_DIR.mkdir(parents=True, exist_ok=True)
            readme = CRASH_DIR / "README.txt"
            if not readme.exists():
                readme.write_text(_README, encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _install_excepthook(self) -> None:
        try:
            self._prev_excepthook = sys.excepthook
            sys.excepthook = self._excepthook
        except Exception:
            pass

    def _excepthook(self, exc_type: Any, exc: Any, tb: Any) -> None:
        detail = "".join(tb_module.format_exception(exc_type, exc, tb))
        self.report("python", f"{getattr(exc_type, '__name__', exc_type)}: {exc}", detail)
        # Chain to whatever was installed before us (selfrepair, then default).
        if self._prev_excepthook is not None and self._prev_excepthook is not self._excepthook:
            try:
                self._prev_excepthook(exc_type, exc, tb)
            except Exception:
                pass

    def _install_qt_handler(self) -> None:
        """Capture Qt's own log stream — this is the ONLY place QtWebEngine
        (Chromium) and OpenGL fatals surface before the process dies."""
        try:
            from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
        except Exception:
            return

        def handler(mode: Any, context: Any, message: str) -> None:
            try:
                text = str(message or "")
                if mode == QtMsgType.QtFatalMsg:
                    self.report("qt-fatal", text, self._stack_now())
                elif mode == QtMsgType.QtCriticalMsg:
                    # WebEngine render-process / GPU faults arrive as Critical.
                    low = text.lower()
                    if any(k in low for k in ("webengine", "gpu", "render process",
                                              "opengl", "gl_", "webgl", "sandbox",
                                              "d3d", "vulkan")):
                        self.report("qt-gpu", text, self._stack_now())
            except Exception:
                pass

        try:
            qInstallMessageHandler(handler)
        except Exception:
            pass

    def _install_faulthandler(self) -> None:
        """A segfault in the C layer (GPU driver, Chromium) leaves no Python
        trace; faulthandler dumps the native stack to a file so even a hard
        crash is not a total mystery."""
        log_path = CRASH_DIR / "faulthandler.log"
        # Read the previous session's slice BEFORE this session's marker is
        # appended — after it, the last slice would be this (empty) session.
        previous = previous_native_fault(log_path)
        _trim_fault_log(log_path)
        try:
            self._fault_log = log_path.open(
                "a", encoding="utf-8", errors="replace"
            )
            self._fault_log.write(f"\n===== faulthandler armed {_stamp()} =====\n")
            self._fault_log.flush()
            faulthandler.enable(file=self._fault_log, all_threads=True)
        except Exception:
            pass
        if previous is not None:
            self._report_native(previous)

    def _report_native(self, fault: dict[str, Any]) -> str:
        """Turn the previous session's native fault into an ordinary report."""
        try:
            faulted = fault["faulted_at"].strftime("%Y-%m-%d %H:%M:%S")
            armed = fault["armed"].strftime("%Y-%m-%d %H:%M:%S")
            detail = (
                f"Session started {armed}; the fault was written to "
                f"faulthandler.log at {faulted}.\n"
                "Reconstructed at the next start-up — the activity log above is "
                "from that later session, not from the crash.\n\n"
                "Faulting position: " + (fault["where"] or "(unknown)") + "\n"
                + "".join(f"Concurrently: {item}\n" for item in fault["during"])
                + "\nThe full multi-thread dump is in faulthandler.log under the "
                f"marker 'faulthandler armed {fault['armed'].strftime('%Y%m%d_%H%M%S')}'.")
            return self.report(fault["kind"], fault["summary"], detail, occurred=faulted)
        except Exception:
            return ""

    # ── the public report API ─────────────────────────────────────────────────

    def report(self, kind: str, summary: str, detail: str = "",
               occurred: str = "") -> str:
        """Write one crash report and return its path (or '' on failure).
        ``kind`` is a short slug: python | qt-fatal | qt-gpu | globe | face."""
        try:
            # Throttle a tight crash loop so it cannot fill the disk.
            n = self._burst.get(kind, 0) + 1
            self._burst[kind] = n
            if n > _MAX_REPORTS_PER_KIND_BURST:
                return ""
            stamp = _stamp()
            path = CRASH_DIR / f"{stamp}_{_safe(kind)}.txt"
            body = self._compose(kind, summary, detail, occurred)
            path.write_text(body, encoding="utf-8", errors="replace")
            self._prune()
            try:
                self.bus.log.emit(f"CRASH: recorded [{kind}] → {path.name} — {summary[:100]}")
                self.bus.dashboard_event.emit("crash", {"kind": kind, "summary": summary[:160],
                                                         "file": str(path)})
            except Exception:
                pass
            if self.telemetry is not None:
                try:
                    self.telemetry.metrics.incr("crash.reports")
                    self.telemetry.health.beat("crash_reporter", "WARN", f"{kind}: {summary[:80]}")
                except Exception:
                    pass
            # Make it VISIBLE, not just logged: a HUD banner every time, and a
            # one-per-kind spoken notice for the serious faults — so the user
            # can see (and hear) why ORION just stumbled and where the full
            # explanation lives.
            info = interpret(kind, summary)
            try:
                self.bus.banner.emit(
                    f"⚠ {info['headline']} — details in config/crash_reports", 9000)
            except Exception:
                pass
            if info.get("severe") and info.get("spoken") and kind not in self._announced:
                self._announced.add(kind)
                try:
                    self.bus.speak_request.emit(info["spoken"])
                except Exception:
                    pass
            return str(path)
        except Exception:
            return ""

    def recent(self, limit: int = 10) -> list[dict[str, str]]:
        """The most recent crash reports (for the diagnostics view / a tool),
        each carrying a plain-English ``why`` so the caller can show a cause
        without parsing the file."""
        try:
            files = sorted(
                (p for p in CRASH_DIR.glob("*.txt") if p.name != "README.txt"),
                key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return []
        out: list[dict[str, str]] = []
        for p in files[:limit]:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            kind = _field(text, "kind")
            summary = _field(text, "summary")
            why = interpret(kind, summary)["headline"] if (kind or summary) else ""
            occurred = _field(text, "time").split(" (recorded")[0].strip()
            out.append({"file": p.name, "path": str(p), "kind": kind,
                        "summary": summary, "why": why, "time": occurred,
                        "head": text[:400]})
        return out

    # ── helpers ────────────────────────────────────────────────────────────────

    def _compose(self, kind: str, summary: str, detail: str, occurred: str = "") -> str:
        info = interpret(kind, summary)
        lines = [
            "O.R.I.O.N. CRASH REPORT",
            "=" * 60,
            (f"time     : {occurred} (recorded {_stamp(human=True)})" if occurred
             else f"time     : {_stamp(human=True)}"),
            f"kind     : {kind}",
            f"summary  : {summary}",
            "",
            "── WHAT THIS MEANS ─────────────────────────────────────────",
            info["headline"] + ".",
            info["explanation"],
            f"→ {info['advice']}",
            "",
            "── environment ─────────────────────────────────────────────",
            f"platform : {platform.platform()}",
            f"python   : {platform.python_version()}",
            f"qt       : {_qt_version()}",
            f"memory   : {_memory_line()}",
            f"gpu      : {_gpu_line()}",
            "",
            "── recent activity log ─────────────────────────────────────",
            self._recent_logs(),
            "",
            "── detail / traceback ──────────────────────────────────────",
            detail.strip() or "(none captured)",
            "",
        ]
        return "\n".join(lines)

    def _recent_logs(self) -> str:
        if self.telemetry is None:
            return "(telemetry unavailable)"
        try:
            rows = self.telemetry.log.recent(limit=30)
            return "\n".join(str(r) for r in rows) or "(empty)"
        except Exception:
            return "(unavailable)"

    @staticmethod
    def _stack_now() -> str:
        try:
            return "".join(tb_module.format_stack()[:-1])
        except Exception:
            return ""

    def _prune(self) -> None:
        try:
            files = sorted(
                (p for p in CRASH_DIR.glob("*.txt") if p.name != "README.txt"),
                key=lambda p: p.stat().st_mtime)
            for old in files[:-_RETENTION]:
                try:
                    old.unlink()
                except Exception:
                    pass
        except Exception:
            pass


# ── module-level conveniences ──────────────────────────────────────────────────

_ACTIVE: CrashReporter | None = None


def install(bus: Any, telemetry: Any | None = None) -> CrashReporter:
    """Create, arm and remember the process-wide crash reporter."""
    global _ACTIVE
    reporter = CrashReporter(bus, telemetry)
    reporter.install()
    _ACTIVE = reporter
    return reporter


def report(kind: str, summary: str, detail: str = "") -> str:
    """Report a crash from anywhere (e.g. the globe render process handler)."""
    if _ACTIVE is not None:
        return _ACTIVE.report(kind, summary, detail)
    return ""


def recent(limit: int = 10) -> list[dict[str, str]]:
    """Recent crash reports from anywhere (e.g. the Diagnostics Centre panel),
    each with a plain-English ``why``.  Empty if the recorder isn't armed."""
    if _ACTIVE is not None:
        return _ACTIVE.recent(limit)
    return []


def _stamp(human: bool = False) -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") if human else now.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(text))[:40]


def _qt_version() -> str:
    try:
        from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        return f"Qt {QT_VERSION_STR} / PyQt {PYQT_VERSION_STR}"
    except Exception:
        return "unknown"


def _memory_line() -> str:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"{vm.percent:.0f}% used of {vm.total / 1e9:.1f} GB"
    except Exception:
        return "unavailable"


def _gpu_line() -> str:
    try:
        from . import gpu_stats
        s = gpu_stats.sample()
        if s:
            return str(s)
    except Exception:
        pass
    return "unavailable"
