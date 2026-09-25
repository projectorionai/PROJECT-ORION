"""
A clean console, without hiding anything that matters.

  "When ORION shuts down ... I want the shutdown to be nice and smooth in CMD
   just saying 'ORION has shutdown cleanly' - Any errors (that must be
   legitimate and affecting performance and usage) must be in a compiled
   reports folder."

Two rules, and the second is what keeps the first honest.

**Nothing is discarded.** Every line filtered off the console is written to
``config/reports/console/`` first. "Quiet" here means routed, never deleted —
a suppressed message that turns out to matter must still be findable, and a
system that silently drops diagnostics is worse than a noisy one.

**Only known-benign third-party chatter is filtered.** Each pattern below is
listed with why it is not a fault. Anything unrecognised goes to the console
exactly as before, because the failure mode of an over-eager filter is the one
that costs you a day: a real error that never printed.

What was actually on the console
--------------------------------
Three sources, none of them a defect in ORION:

``google.genai`` logs, once per process, that a Live response contained 'text'
and 'thought' parts alongside audio. That is a description of ORION working
normally — he requests transcriptions and the model thinks — not a warning
about anything.

``chess.engine`` logs an error parsing Stockfish's ``multipv`` info lines. The
move is still returned correctly; python-chess simply cannot parse one variety
of ``info`` string. It is per-analysis, so it scrolls.

And ORION's own shutdown printed over a hundred "Task was destroyed but it is
pending!" lines, which is a real defect and is fixed at source (see
``SelfRepairAgent.drain``) rather than filtered here. Suppressing that would
have hidden the bug instead of solving it — which is precisely the line this
module has to hold.
"""

from __future__ import annotations

import logging
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

REPORTS_DIR = CONFIG_DIR / "reports" / "console"

#: (logger prefix, message pattern, why this is not a fault).
#: The reason is required — a filter entry nobody can justify is a filter entry
#: that will one day hide something real.
BENIGN: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "google.genai",
        re.compile(r"non-data parts in the response", re.I),
        "ORION asks the Live channel for transcriptions and the model emits "
        "thoughts, so audio responses legitimately carry 'text' and 'thought' "
        "parts. This describes him working, not failing.",
    ),
    (
        "chess.engine",
        re.compile(r"exception parsing (pv|score) from info", re.I),
        "python-chess cannot parse one shape of Stockfish multipv info line. "
        "The move is still returned and played correctly.",
    ),
    (
        "chess.engine",
        re.compile(r"stderr", re.I),
        "Stockfish writes banner and diagnostic text to stderr on startup.",
    ),
)


class _Router(logging.Filter):
    """Sends known-benign third-party records to a file instead of the console."""

    def __init__(self) -> None:
        super().__init__()
        self.routed = 0
        self._sink: Any = None

    def _reason(self, record: logging.LogRecord) -> str:
        name = record.name or ""
        try:
            message = record.getMessage()
        except Exception:
            return ""
        for prefix, pattern, why in BENIGN:
            if name.startswith(prefix) and pattern.search(message):
                return why
        return ""

    def filter(self, record: logging.LogRecord) -> bool:
        why = self._reason(record)
        if not why:
            return True                    # unrecognised: print it, always
        self.routed += 1
        self._write(record, why)
        return False

    def _write(self, record: logging.LogRecord, why: str) -> None:
        try:
            if self._sink is None:
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                path = REPORTS_DIR / f"{datetime.now():%Y-%m-%d}-console.log"
                self._sink = path.open("a", encoding="utf-8")
            stamp = datetime.now().strftime("%H:%M:%S")
            self._sink.write(
                f"[{stamp}] {record.name} {record.levelname}: "
                f"{record.getMessage()}\n         kept quiet because: {why}\n")
            self._sink.flush()
        except Exception:
            pass                            # logging must never raise

    def close(self) -> None:
        if self._sink is not None:
            try:
                self._sink.close()
            except Exception:
                pass
            self._sink = None


_ROUTER = _Router()
_INSTALLED = False


def install() -> None:
    """Attach the router to the root logger's console handlers. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    root = logging.getLogger()
    if not root.handlers:
        # Nothing configured logging yet, so records reach lastResort and print
        # unfiltered. Give the root a handler we can actually attach to.
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        root.addHandler(handler)
    for handler in root.handlers:
        handler.addFilter(_ROUTER)
    # Some libraries log through their own handlers rather than propagating.
    for name in {prefix for prefix, _, _ in BENIGN}:
        logger = logging.getLogger(name)
        logger.addFilter(_ROUTER)
        for handler in logger.handlers:
            handler.addFilter(_ROUTER)


def routed_count() -> int:
    """How many lines were kept off the console this session."""
    return _ROUTER.routed


def report_fault(exc: BaseException, context: str = "") -> Path | None:
    """Write a genuine fault to the reports folder and return where it went.

    Used for the faults that DO matter, so "the console stayed quiet" never
    means "the error went nowhere".
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}-fault.log"
        body = "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
        path.write_text(
            f"ORION fault report\n{datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"context: {context or 'unspecified'}\n\n{body}",
            encoding="utf-8")
        return path
    except Exception:
        return None


def close() -> None:
    _ROUTER.close()


# ── is this ORION closing, or ORION breaking? ────────────────────────────────

#: What qasync raises when the Qt loop is stopped while a coroutine is still
#: awaiting — i.e. every ordinary shutdown and every self-restart.
QASYNC_STOPPED = "event loop stopped before future completed"


def is_clean_shutdown(exc: BaseException, *, requested: bool = False) -> bool:
    """Whether *exc* is ORION being closed rather than ORION breaking.

    ``requested`` is the authoritative signal (Qt's aboutToQuit fired, so a
    human or ORION himself asked to close). The message check is the fallback
    for the paths that stop the loop without going through aboutToQuit, such
    as a self-restart. Anything else stays a fault.

    Lives here, and not in app.py, for a practical reason: this is a pure
    predicate that tests want to exercise directly, and importing app.py pulls
    in QtWebEngine — which hard-crashes any Qt test that runs afterwards in the
    same process (no traceback, just a dead interpreter). A predicate with no
    dependencies should not sit inside the module with the heaviest one.
    """
    if not isinstance(exc, RuntimeError):
        return False
    if requested:
        return True
    return QASYNC_STOPPED in str(exc).strip().lower().rstrip(".")


__all__ = ["BENIGN", "QASYNC_STOPPED", "REPORTS_DIR", "close", "install",
           "is_clean_shutdown", "report_fault", "routed_count"]
