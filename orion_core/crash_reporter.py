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
        try:
            self._fault_log = (CRASH_DIR / "faulthandler.log").open(
                "a", encoding="utf-8", errors="replace"
            )
            self._fault_log.write(f"\n===== faulthandler armed {_stamp()} =====\n")
            self._fault_log.flush()
            faulthandler.enable(file=self._fault_log, all_threads=True)
        except Exception:
            pass

    # ── the public report API ─────────────────────────────────────────────────

    def report(self, kind: str, summary: str, detail: str = "") -> str:
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
            body = self._compose(kind, summary, detail)
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
            out.append({"file": p.name, "path": str(p), "kind": kind,
                        "summary": summary, "why": why, "head": text[:400]})
        return out

    # ── helpers ────────────────────────────────────────────────────────────────

    def _compose(self, kind: str, summary: str, detail: str) -> str:
        info = interpret(kind, summary)
        lines = [
            "O.R.I.O.N. CRASH REPORT",
            "=" * 60,
            f"time     : {_stamp(human=True)}",
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
