"""
SelfRepairAgent (Phase 6) — self-healing runtime with a human approval gate.

When something throws, ORION:

    1. captures the stack trace and the recent structured logs;
    2. identifies the failing module and line from the traceback;
    3. reads the surrounding source;
    4. asks the provider layer for a fix (unified diff + rationale);
    5. compile-validates the patched file, then runs the module's own test
       subset against an *isolated copy* (Priority 3.3) so the approval prompt
       carries pass/fail, not just "it compiles";
    6. presents the patch — with that verdict — for approval.

Proposals are saved under ``config/self_repair/`` for review. The repair-file
path can apply approved source changes with backups. Bounded auto-repair is
also enabled by default: eligible small changes require passing isolated tests,
obey a protected-file denylist and have a per-session cap. Set
``ORION_AUTOREPAIR=0`` to disable that automatic path.

Wiring: ``install()`` registers a ``sys.excepthook`` wrapper and an asyncio
exception handler so faults are captured wherever they occur, without any
call site needing to know about self-repair.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import threading
from collections import OrderedDict
import traceback as tb_module
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR, CONFIG_DIR, PACKAGE_DIR
from .data import ToolResult
from .utils import first_line, utc_stamp
from .journal_io import tail_lines

SELF_REPAIR_DIR = CONFIG_DIR / "self_repair"
# The repair journal records distinct faults and every applied/reverted fix.
# Identical captures are sampled to keep fault storms from filling the disk.
JOURNAL_PATH = SELF_REPAIR_DIR / "journal.jsonl"
_JOURNAL_LOCK = threading.RLock()
JOURNAL_REPEAT_WINDOW_S = 60.0


@dataclass
class Incident:
    id: str
    at: str
    error_type: str
    message: str
    file: str = ""
    line: int = 0
    module: str = ""
    traceback: str = ""
    logs: list[dict[str, Any]] = field(default_factory=list)
    proposal_path: str = ""
    repaired_content: str = ""   # full corrected file the model produced
    backup_path: str = ""        # where the pre-repair original was saved
    applied: bool = False
    # Result of running the module's tests against the patched copy before
    # approval (Priority 3.3): True/False when tests ran, None when none applied.
    tests_passed: Optional[bool] = None
    tests_detail: str = ""
    # Transient live-channel faults (WebSocket 1011, deadline expired, go-away)
    # are external and self-recovering: they are recorded for visibility but must
    # never pin the self-repair subsystem to DEGRADED, and they clear on the next
    # successful connection.  resolved marks any incident no longer open.
    transient: bool = False
    resolved: bool = False
    # False for informational reports (failed forge session, subsystem notice):
    # visible in diagnostics but never a reason to mark self-repair DEGRADED.
    actionable: bool = True
    # True when the user REPORTED a defect rather than the runtime catching one.
    # There is no traceback to narrow the search, so the source window shown to
    # the model is deliberately wider.
    reported: bool = False
    # The model's own statement of why it broke, kept with the drafted fix.
    root_cause: str = ""

    def summary(self) -> str:
        loc = f"{Path(self.file).name}:{self.line}" if self.file else "unknown"
        return f"[{self.id}] {self.error_type}: {self.message[:120]} ({loc})"


class SelfRepairAgent:
    # ── bounded autonomy (improvement #27) ────────────────────────────────────
    # A repair may auto-apply ONLY when every gate passes: the incident is
    # self-contained (missing third-party dependency, or a small fix whose
    # dedicated tests PASS on an isolated copy), the target is not
    # safety-critical, and the session cap has not been reached.  Everything
    # else stays approval-gated exactly as before.  ORION_AUTOREPAIR=0 turns
    # autonomy off entirely.
    AUTO_DENYLIST = frozenset({
        "security.py", "selfrepair.py", "constants.py", "identity.py",
        "remote.py", "providers.py",
        # The gates themselves. An unattended patch to any of these could
        # loosen the very checks that make unattended patching safe: the
        # confirmation tokens, the spend gate, the forge/plugin sandbox and
        # loader, the write/parallel classification, the phone's tiers.
        "system_guard.py", "dispatch_desktop.py", "dispatch_web.py",
        "forge.py", "forge_contract.py", "forge_autofix.py", "dynamic_loader.py", "sandbox.py",
        "plugin_registry.py", "plugin_runner.py", "concurrency.py",
        "remote_capability.py", "spillage.py",
    })
    #: Incidents kept in memory; the oldest resolved ones go first.
    MAX_INCIDENTS = 200
    MAX_AUTO_REPAIRS_PER_SESSION = 2
    MAX_AUTO_DIFF_LINES = 12

    #: Concurrent auto-repairs. Repairs call models and run test suites, so
    #: this is a workload ceiling, not just tidiness.
    MAX_CONCURRENT_REPAIRS = 4

    def _trim_incidents(self) -> None:
        """Keep the incident table bounded. It only ever grew: a long session
        with a recurring fault held every incident it had ever seen."""
        excess = len(self._incidents) - self.MAX_INCIDENTS
        if excess <= 0:
            return
        resolved = [k for k, inc in self._incidents.items()
                    if getattr(inc, "resolved", False)]
        for key in (resolved + list(self._incidents))[:excess]:
            self._incidents.pop(key, None)

    # ── transient live-channel faults ─────────────────────────────────────────
    # The Gemini Live socket dropping (1011, deadline expired, go-away, keepalive
    # timeout) is an EXTERNAL, self-recovering event, not a fault in ORION's own
    # code.  The reconnect logic handles it, so it must not leave the self-repair
    # subsystem pinned to DEGRADED — that was the "self repair still degraded"
    # report.  These are recorded, then cleared on the next successful connect.
    _TRANSIENT_TYPES = frozenset({
        "ConnectionClosedError", "ConnectionClosedOK", "ConnectionClosed",
        "TimeoutError", "ConnectionResetError", "ServerError",
        "WebSocketException", "IncompleteReadError", "ConnectionError",
    })
    _TRANSIENT_MARKERS = (
        "1011", "1006", "1013", "deadline expired", "connection closed",
        "websocket error", "go away", "goaway", "keepalive ping",
        "temporarily unavailable", "operation could complete",
        "received 1011", "live channel", "sent 1011",
    )

    @classmethod
    def _is_transient(cls, error_type: str, message: str) -> bool:
        """True when an incident describes the live cloud channel dropping rather
        than a defect in ORION's own source."""
        if str(error_type or "") in cls._TRANSIENT_TYPES:
            return True
        msg = str(message or "").lower()
        return any(marker in msg for marker in cls._TRANSIENT_MARKERS)

    # ── faults produced BY the repair machinery ───────────────────────────────
    # A self-repair agent that files incidents about its own scheduling will
    # spiral, and it did: a repair task dispatched re-entrantly raises
    #
    #   RuntimeError: Cannot enter into task <attempt_auto_repair()> while
    #   another task <run_application()> is being executed
    #
    # which reached _loop_exception_handler, became inc-N, and scheduled
    # another repair — which raised the same thing. Five in a row in the user's
    # log, each one manufactured by the handling of the one before it.
    #
    # The scheduling-ordering defect that produced them is fixed at source (see
    # app.py's teardown), but the feedback path is a hazard in its own right and
    # is cut here as well: these are conditions OF the repairer, so the
    # repairer refuses to treat them as work. Matched on the message rather
    # than the type because RuntimeError is far too common to blanket-ignore.
    _SELF_INFLICTED_MARKERS = (
        "cannot enter into task",
        "is being executed",
        "task was destroyed but it is pending",
        "attempt_auto_repair",
        "event loop is closed",
        "no running event loop",
    )

    @classmethod
    def _is_self_inflicted(cls, exc: Any) -> bool:
        """True when the fault IS the repair/scheduling machinery misfiring.

        Recording these is worse than useless: it pins self_repair DEGRADED on
        a condition that describes nothing about ORION's code, and it feeds a
        loop that generates the next one.
        """
        text = f"{exc}".lower()
        if any(marker in text for marker in cls._SELF_INFLICTED_MARKERS):
            return True
        # A traceback that runs through this module is by definition the
        # repairer failing, not something for the repairer to fix.
        tb = getattr(exc, "__traceback__", None)
        here = str(Path(__file__).resolve())
        while tb is not None:
            try:
                if Path(tb.tb_frame.f_code.co_filename).resolve() == Path(here):
                    return True
            except Exception:
                pass
            tb = tb.tb_next
        return False

    def __init__(self, bus: OrionBus, telemetry: Any, router: Any) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.router = router
        self._incidents: dict[str, Incident] = {}
        self._counter = 0
        self._prev_excepthook: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._auto_repairs = 0
        self._auto_attempted: set[tuple[str, str, int]] = set()
        # Live repair tasks, so shutdown can cancel them instead of leaving
        # asyncio to destroy them pending and print a traceback for each.
        self._tasks: set[Any] = set()
        self._shutting_down = False
        self._reported_self_inflicted = False
        SELF_REPAIR_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _auto_enabled() -> bool:
        return os.getenv("ORION_AUTOREPAIR", "1").strip().lower() not in {
            "0", "false", "no", "off"}

    # ── installation ──────────────────────────────────────────────────────────

    def install(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._prev_excepthook = sys.excepthook
        sys.excepthook = self._excepthook
        self._loop = loop
        if loop is not None:
            loop.set_exception_handler(self._loop_exception_handler)
        # Recover transient live-channel incidents the moment the connection
        # returns, so a WebSocket 1011 blip never leaves self-repair DEGRADED.
        try:
            self.bus.connection_state.connect(self._on_connection_state)
        except Exception:
            pass
        mode = ("low-risk incidents auto-repair; the rest need approval"
                if self._auto_enabled() else "capture-only; approval required to apply")
        self.bus.log.emit(f"REPAIR: self-healing runtime armed ({mode}).")

    def _excepthook(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if not self._is_self_inflicted(exc):
                self.capture(exc_type, exc, tb)
        except Exception:
            pass
        if self._prev_excepthook is not None:
            self._prev_excepthook(exc_type, exc, tb)

    def _loop_exception_handler(self, loop: Any, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if exc is not None:
            if self._is_self_inflicted(exc):
                # Say it once, plainly, and do not open an incident — see
                # _SELF_INFLICTED_MARKERS. Silence here would hide a real
                # scheduling regression; an incident would start a spiral.
                if not self._reported_self_inflicted:
                    self._reported_self_inflicted = True
                    try:
                        self.bus.log.emit(
                            f"REPAIR: ignoring a scheduling fault in the repair "
                            f"machinery itself ({type(exc).__name__}: "
                            f"{first_line(str(exc), 80)}) - not a defect in ORION's code.")
                    except RuntimeError:
                        pass
                return
            self.capture(type(exc), exc, exc.__traceback__)
            return
        message = str(context.get("message", "unknown"))
        # During shutdown the Qt bus is destroyed before the loop finishes
        # draining, so emitting on it raises "wrapped C/C++ object of type
        # OrionBus has been deleted" — FROM INSIDE the exception handler,
        # which asyncio then reports as "Unhandled error in custom exception
        # handler", once per pending task. A dozen of those tracebacks were
        # the last thing the user saw on every clean exit, and none of them
        # described a real fault.
        try:
            self.bus.log.emit(f"REPAIR: loop error - {message}")
        except RuntimeError:
            print(f"REPAIR: loop error - {message}")

    def shutdown(self) -> None:
        """Stop repairing and let go of the loop.

        Called before the bus is torn down. Without it the handler outlives
        the objects it reports through, and every task still pending at exit
        produces a traceback about the bus rather than about itself.

        Cancels but deliberately does NOT clear: see :meth:`drain`.
        """
        self._shutting_down = True
        loop = self._loop
        if loop is not None:
            try:
                loop.set_exception_handler(None)
            except Exception:
                pass
        for task in list(self._tasks):
            if not task.done():
                task.cancel()

    async def drain(self, timeout: float = 3.0) -> int:
        """Let cancelled repairs actually finish cancelling. Returns how many.

        ``task.cancel()`` only SCHEDULES cancellation — it throws CancelledError
        into the coroutine at its next suspension point. The old shutdown
        cancelled and then immediately did ``self._tasks.clear()``, dropping the
        last reference while every task was still in the 'cancelling' state and
        giving the loop no further turn to deliver anything. Python then printed,
        for each one:

            Task was destroyed but it is pending!
            task: <Task cancelling name='Task-330'
                  coro=<SelfRepairAgent.attempt_auto_repair() ...>>

        — over a hundred of them on a single exit, plus a "coroutine ... was
        never awaited" warning pointing at the ``_tasks.clear()`` line itself,
        which is a confusing place to be sent since nothing is wrong there.

        Awaiting the cancelled tasks consumes their CancelledError properly and
        the warnings stop, because there is now nothing pending to destroy.
        Bounded, because a repair wedged in a blocking call must not hold
        shutdown open.
        """
        pending = [task for task in self._tasks if not task.done()]
        if not pending:
            self._tasks.clear()
            return 0
        try:
            await asyncio.wait(pending, timeout=max(0.1, timeout))
        except Exception:
            pass
        # Retrieve every outcome so none is left unobserved — an unretrieved
        # exception is reported by the collector at an arbitrary later moment.
        for task in pending:
            if task.done() and not task.cancelled():
                try:
                    task.exception()
                except Exception:
                    pass
        self._tasks.clear()
        return len(pending)

    # ── what actually went wrong ──────────────────────────────────────────────

    # Fault classes ORION can name in plain terms, with what each one means
    # for whether he can fix it himself. Keyed on the exception type because
    # that is the one thing every incident reliably carries.
    _DIAGNOSES: dict[str, tuple[str, str]] = {
        "ModuleNotFoundError": ("a package he needs is not installed",
                                "he can install it himself"),
        "ImportError":         ("a package he needs is not installed",
                                "he can install it himself"),
        "AttributeError":      ("he called something that does not exist on that object",
                                "usually a rename that missed a call site"),
        "TypeError":           ("he passed the wrong shape of argument",
                                "usually a signature that changed under a caller"),
        "KeyError":            ("he read a key that was not there",
                                "usually data arriving in a shape he did not expect"),
        "FileNotFoundError":   ("a file he expected is missing",
                                "either a path that moved or something never written"),
        "PermissionError":     ("the operating system refused him access",
                                "not a code fault — he cannot repair this one"),
        "RuntimeError":        ("something failed at runtime",
                                "the traceback below is the only reliable guide"),
        "ConnectionError":     ("a network call could not be completed",
                                "external, and usually self-recovering"),
        "ValueError":          ("a value was not usable where it was used",
                                "usually validation missing at the boundary"),
        "ZeroDivisionError":   ("he divided by zero",
                                "a guard is missing on an empty measurement"),
    }

    def diagnose(self, incident: Incident) -> str:
        """One line naming the fault, where it is, and whether he can fix it.

        Composed from the incident he already has rather than from a second
        model call, so it is available the instant the fault is captured — and
        it can never be wrong about the location, because it is reading the
        traceback rather than describing it.
        """
        what, prospect = self._DIAGNOSES.get(
            incident.error_type,
            ("an unrecognised fault", "he will read the source to find out"))
        where = (f"in {Path(incident.file).name} line {incident.line}"
                 if incident.file else "somewhere without a traceback")
        if incident.module:
            where += f" ({incident.module})"
        detail = first_line(incident.message, 160) or "no message"
        if not incident.actionable:
            prospect = "informational — nothing to repair"
        elif incident.transient:
            prospect = "external and self-recovering"
        return f"{what} {where} — {detail}. Prospect: {prospect}."

    # ── capture ───────────────────────────────────────────────────────────────

    def capture(self, exc_type: Any, exc: Any, tb: Any) -> Incident:
        self._counter += 1
        incident_id = f"inc-{self._counter}"
        frames = tb_module.extract_tb(tb) if tb is not None else []
        # Deepest frame inside ORION's own package is the most actionable.
        own_frame = None
        for frame in reversed(frames):
            try:
                if PACKAGE_DIR in Path(frame.filename).resolve().parents:
                    own_frame = frame
                    break
            except Exception:
                continue
        target = own_frame or (frames[-1] if frames else None)
        incident = Incident(
            id=incident_id,
            at=utc_stamp(),
            error_type=getattr(exc_type, "__name__", str(exc_type)),
            message=str(exc),
            file=target.filename if target else "",
            line=int(target.lineno) if target and target.lineno else 0,
            module=Path(target.filename).stem if target else "",
            traceback="".join(tb_module.format_exception(exc_type, exc, tb))[-4000:],
            logs=self.telemetry.log.recent(limit=25) if self.telemetry else [],
        )
        incident.transient = self._is_transient(incident.error_type, incident.message)
        self._incidents[incident_id] = incident
        self._trim_incidents()
        if self.telemetry is not None:
            self.telemetry.metrics.incr("repair.incidents")
        self._refresh_health()
        if incident.transient:
            self.bus.log.emit(
                f"REPAIR: transient live-channel event noted {incident.summary()} "
                "— the reconnect logic recovers this; it will clear on reconnect.")
        else:
            # Say what actually went wrong, not just that something did.
            # "captured [inc-3] RuntimeError" tells the user a fault happened
            # and nothing about it — they asked to know what ORION is aware of
            # when he repairs himself, and the diagnosis is already in hand at
            # this point.
            self.bus.log.emit(f"REPAIR: captured {incident.summary()}")
            self.bus.log.emit(f"REPAIR: {self.diagnose(incident)}")
        self.bus.dashboard_event.emit("incident", incident.summary())
        self._journal("captured", incident)
        # External channel drops are not code faults — never spend a repair
        # attempt on them; the transport layer already recovers them.
        if self._auto_enabled() and not incident.transient:
            self._schedule_auto_repair(incident)
        return incident

    # ── health accounting ─────────────────────────────────────────────────────

    def _refresh_health(self) -> None:
        """The self-repair subsystem is DEGRADED only while an *actionable*
        (code-level, unresolved) incident is open.  Transient live-channel drops
        and resolved incidents never pin it degraded — they are surfaced through
        the connection/provider health instead."""
        if self.telemetry is None:
            return
        actionable = [
            inc for inc in self._incidents.values()
            if not inc.resolved and not inc.transient and inc.actionable
        ]
        try:
            if actionable:
                self.telemetry.health.beat(
                    "self_repair", "DEGRADED", actionable[-1].summary())
            else:
                pending = sum(
                    1 for inc in self._incidents.values()
                    if inc.transient and not inc.resolved)
                detail = (f"healthy — {pending} transient channel event(s) awaiting reconnect"
                          if pending else "healthy — no open incidents")
                self.telemetry.health.beat("self_repair", "OK", detail)
        except Exception:
            pass

    def resolve_transient(self, reason: str = "live channel reconnected") -> int:
        """Clear open transient (live-channel) incidents — called when the
        connection recovers.  Returns how many were cleared."""
        cleared = 0
        for inc in self._incidents.values():
            if inc.transient and not inc.resolved:
                inc.resolved = True
                cleared += 1
                self._journal("resolved", inc, note=reason)
        if cleared:
            self.bus.log.emit(
                f"REPAIR: {cleared} transient live-channel event(s) cleared ({reason}).")
        self._refresh_health()
        return cleared

    def _on_connection_state(self, payload: Any) -> None:
        """React to connection-state changes on the bus: a recovered channel
        resolves the transient faults captured while it was down."""
        try:
            state = payload.get("state") if isinstance(payload, dict) else str(payload or "")
            if state in {"connected", "local_fallback"}:
                self.resolve_transient(f"channel {state}")
        except Exception:
            pass

    def record_incident(self, error_type: str, message: str, *,
                        module: str = "", logs: list[dict[str, Any]] | None = None) -> Incident:
        """Capture a NON-exception fault (a failed forge session, a subsystem
        that reported failure without raising) so diagnostics and the
        self_repair tool can see it.  Capture-only: no auto-repair is
        scheduled — these incidents describe generated artefacts or external
        conditions, not a broken source file to patch."""
        self._counter += 1
        incident = Incident(
            id=f"inc-{self._counter}",
            at=utc_stamp(),
            error_type=str(error_type or "Fault"),
            message=str(message or "")[:2000],
            module=module,
            actionable=False,
            logs=logs if logs is not None else (
                self.telemetry.log.recent(limit=25) if self.telemetry else []),
        )
        self._incidents[incident.id] = incident
        self._trim_incidents()
        if self.telemetry is not None:
            self.telemetry.metrics.incr("repair.incidents")
        self.bus.log.emit(f"REPAIR: recorded {incident.summary()}")
        self.bus.dashboard_event.emit("incident", incident.summary())
        self._journal("captured", incident, note=f"reported by {module or 'subsystem'}")
        self._refresh_health()
        return incident

    # ── proactive repair: the user reports a defect ──────────────────────────
    #
    # Everything above is REACTIVE: an exception reaches sys.excepthook or the
    # asyncio handler, and only then does ORION have something to repair. That
    # leaves the most common case unserved — behaviour that is wrong but does
    # not raise. "You keep mishearing shutdown as a power command" produces no
    # traceback, so there was previously no way to ask ORION to fix it.
    #
    # A reported defect has no traceback to point at a file, so the location is
    # derived by searching the package. That search is deterministic and
    # explainable, and the user sees which file was chosen before approving
    # anything — the existing approval gate is unchanged.

    _STOPWORDS = frozenset({
        "the", "and", "for", "with", "that", "this", "から", "you", "your", "its",
        "orion", "please", "when", "then", "than", "have", "has", "was", "were",
        "keeps", "keep", "does", "doesn", "don", "not", "but", "should", "would",
        "could", "into", "from", "about", "always", "never", "sometimes", "still",
        "fix", "broken", "wrong", "issue", "problem", "bug", "error", "failing",
    })

    def _terms(self, text: str) -> list[str]:
        import re as _re
        words = _re.findall(r"[a-z_][a-z0-9_]{2,}", str(text or "").lower())
        return [w for w in words if w not in self._STOPWORDS]

    _DESCRIPTIVE_MODULES = {
        "dispatch_schema.py": 0.15, "tool_vocabulary.py": 0.15,
        "changelog.py": 0.1, "code_changelog.py": 0.1, "self_knowledge.py": 0.3,
        "capability_health.py": 0.5,
    }

    def locate_candidates(self, description: str, limit: int = 3) -> list[tuple[str, int, float]]:
        """Ranked (file, line, score) candidates for a described defect.

        Searches the WHOLE package, sub-packages included. The previous
        version globbed ``orion_core/*.py`` only, so nothing under gui/ could
        ever be found — a report about the "Compact Overlay" button (defined
        in gui/core_window.py) was pinned on app.py line 50.

        A PHRASE from the report that appears verbatim in a file is worth far
        more than scattered words: users quote what they see on screen, and a
        button label or log message leads straight to its code.
        """
        import re as _re

        terms = self._terms(description)
        lowered_desc = str(description or "").lower()
        words = _re.findall(r"[a-z0-9&']+", lowered_desc)
        phrases = {" ".join(words[i:i + n]) for n in (2, 3) for i in range(len(words) - n + 1)}
        phrases = {p for p in phrases if len(p) >= 8
                   and not all(w in self._STOPWORDS for w in p.split())}
        quoted = {q.lower() for q in _re.findall(r"[\"“']([^\"”']{4,60})[\"”']", str(description or ""))}
        scored: list[tuple[str, int, float]] = []
        for path in sorted(PACKAGE_DIR.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lowered = content.lower()
            stem = path.stem.lower()
            score = 0.0
            for term in terms:
                if term in stem or stem in term:
                    score += 10.0
                score += min(4.0, lowered.count(term) * 0.25)
            anchor_terms = list(quoted) + sorted(phrases, key=len, reverse=True)
            for phrase in quoted:
                if phrase in lowered:
                    score += 40.0
            for phrase in phrases:
                if phrase in lowered:
                    score += 6.0
            # Modules that only DESCRIBE features in prose — the tool schema,
            # the retrieval vocabulary, changelogs — match every report and
            # contain none of the behaviour.
            score *= self._DESCRIPTIVE_MODULES.get(path.name, 1.0)
            if score <= 0:
                continue
            line_no, best_hits = 1, 0.0
            for number, line in enumerate(lowered.splitlines(), 1):
                hits = sum(3.0 for p in anchor_terms if p in line) + \
                    sum(1.0 for term in terms if term in line)
                if hits > best_hits:
                    line_no, best_hits = number, hits
            scored.append((str(path), line_no, score))
        scored.sort(key=lambda row: -row[2])
        return scored[:limit]

    def locate_source(self, description: str) -> tuple[str, int, str]:
        """Best-guess (file, line, rationale) for a described defect."""
        terms = self._terms(description)
        if not terms:
            return "", 0, "no searchable terms in the description"
        candidates = self.locate_candidates(description)
        if not candidates:
            return "", 0, "no module matched the description"
        best_path, best_line, _score = candidates[0]
        others = ", ".join(Path(p).name for p, _l, _s in candidates[1:])
        return (best_path, best_line,
                f"matched {Path(best_path).relative_to(PACKAGE_DIR.parent)} "
                f"(line {best_line})" + (f"; also considered {others}" if others else ""))

    def request_repair(self, description: str, file: str = "",
                       module: str = "") -> Incident:
        """Record a defect the USER reported, located well enough to repair.

        The incident is actionable and behaves exactly like a captured fault
        from here on: ``propose`` drafts a fix, ``repair`` applies one with
        confirmation. Nothing auto-applies — a reported defect is a description,
        not evidence, so it never bypasses the approval gate.
        """
        description = str(description or "").strip()
        located_line, rationale = 0, ""
        target = str(file or "").strip()
        if target:
            rationale = "file supplied by the user"
            located_line = 1
        else:
            target, located_line, rationale = self.locate_source(description)

        self._counter += 1
        incident = Incident(
            id=f"inc-{self._counter}",
            at=utc_stamp(),
            error_type="ReportedDefect",
            message=description[:2000],
            file=target,
            line=located_line,
            module=module or (Path(target).stem if target else ""),
            reported=True,
            actionable=bool(target),
            logs=self.telemetry.log.recent(limit=25) if self.telemetry else [],
        )
        self._incidents[incident.id] = incident
        self._trim_incidents()
        if self.telemetry is not None:
            self.telemetry.metrics.incr("repair.reported")
        self.bus.log.emit(
            f"REPAIR: user reported '{first_line(description, 90)}' — {rationale}")
        self._journal("reported", incident, note=rationale)
        return incident

    # ── bounded autonomy: schedule + gates (improvement #27) ─────────────────

    def _schedule_auto_repair(self, incident: Incident) -> None:
        """Queue an auto-repair attempt on the event loop.  capture() can fire
        from any thread (sys.excepthook), so the hop is thread-safe; a missing
        or closed loop simply leaves the incident approval-gated."""
        loop = self._loop
        if loop is None or loop.is_closed() or self._shutting_down:
            return

        def _spawn(deferrals: int = 0) -> None:
            # Re-checked inside the hop: shutdown can begin between scheduling
            # and running, and a coroutine created then is never awaited.
            if self._shutting_down:
                return
            # A callback should never see a current task. If it does, Qt is
            # being pumped from inside a coroutine and we are running
            # RE-ENTRANTLY — the classic qasync hazard, since qasync dispatches
            # every asyncio callback as a Qt timer event. Creating a task now
            # means its first step can land in the same nested pump and asyncio
            # rejects it with "Cannot enter into task". Wait for a genuinely
            # idle turn instead. Bounded, because a program that never leaves
            # its pump should drop the repair rather than retry forever.
            try:
                if asyncio.current_task() is not None:
                    if deferrals < 5:
                        loop.call_later(0.05, _spawn, deferrals + 1)
                    return
            except RuntimeError:
                pass
            # A hard ceiling on concurrent repairs. The user's shutdown log
            # showed Task-291 through Task-403 — over a hundred repair tasks
            # alive at once, which is not a workload, it is a symptom. Repairs
            # call models and run test suites; a hundred in parallel would
            # starve the machine even if they all succeeded. The spiral that
            # produced them is fixed (see _is_self_inflicted), and this is the
            # backstop that keeps any future one from being unbounded.
            live = sum(1 for existing in self._tasks if not existing.done())
            if live >= self.MAX_CONCURRENT_REPAIRS:
                return
            task = loop.create_task(self.attempt_auto_repair(incident))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        try:
            loop.call_soon_threadsafe(_spawn)
        except Exception:
            pass

    async def attempt_auto_repair(self, incident: Incident) -> None:
        """Auto-repair a LOW-RISK, SELF-CONTAINED incident; leave everything
        else for approval.  Every outcome is journalled; never raises."""
        try:
            if self._auto_repairs >= self.MAX_AUTO_REPAIRS_PER_SESSION:
                return
            signature = (incident.error_type, incident.file, incident.line)
            if signature in self._auto_attempted:
                return                       # a crash loop gets ONE attempt
            self._auto_attempted.add(signature)

            # Gate A — missing third-party dependency: install it, no source
            # change at all (the lowest-risk incident class there is).
            if incident.error_type in {"ModuleNotFoundError", "ImportError"}:
                from .forge import parse_missing_module
                missing = parse_missing_module(incident.message)
                if missing:
                    from .dependencies import DynamicPackageResolver
                    outcome = await DynamicPackageResolver(self.bus).resolve_and_install([missing])
                    if outcome.succeeded:
                        self._auto_repairs += 1
                        self._journal("auto_dependency", incident, note=missing)
                        self.bus.log.emit(
                            f"REPAIR: auto-installed missing dependency '{missing}' — "
                            "the fault should clear on the next attempt.")
                    return

            # Gate B — a small source fix, validated end-to-end before it may
            # touch the live file.
            target = Path(incident.file) if incident.file else None
            if target is None or not target.is_file():
                return
            if PACKAGE_DIR not in target.resolve().parents:
                return
            if target.name in self.AUTO_DENYLIST:
                return
            if not incident.repaired_content:
                generated = await self._generate_repaired_file(incident, target)
                if not generated.ok or not incident.repaired_content:
                    return
            original = target.read_text(encoding="utf-8", errors="replace")
            changed = self._diff_lines(original, incident.repaired_content)
            if changed > self.MAX_AUTO_DIFF_LINES:
                self._journal("auto_declined", incident,
                              note=f"diff too large ({changed} lines) — approval required")
                return
            passed, detail = await asyncio.to_thread(
                self._run_patched_tests, target, incident.repaired_content)
            incident.tests_passed = passed
            incident.tests_detail = detail
            if passed is not True:            # failing OR absent tests → human
                self._journal("auto_declined", incident,
                              note=f"tests not green ({detail}) — approval required")
                return
            result = await asyncio.to_thread(self._apply_repaired, incident, target)
            if result.ok:
                self._auto_repairs += 1
                self._journal("auto_applied", incident, note=detail)
                self.bus.log.emit(
                    f"REPAIR: auto-applied a validated {changed}-line fix to "
                    f"{target.name} (its tests pass; backup kept). Restart to load it — "
                    "say 'revert the last repair' to undo.")
        except Exception:
            pass                               # autonomy must never crash capture

    @staticmethod
    def _diff_lines(original: str, patched: str) -> int:
        """Changed-line count between original and patched (headers excluded)."""
        import difflib
        changed = 0
        for line in difflib.unified_diff(
                original.splitlines(), patched.splitlines(), lineterm=""):
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                changed += 1
        return changed

    # ── repair journal (persistent learning across restarts) ─────────────────

    def _journal(self, event: str, incident: Incident, note: str = "") -> None:
        """Append one line of repair experience; never raises."""
        try:
            entry = {
                "at": utc_stamp(),
                "event": event,          # captured | repaired | reverted
                "error": f"{incident.error_type}: {incident.message[:200]}",
                "file": Path(incident.file).name if incident.file else "",
                "line": incident.line,
                "note": note,
            }
            with _JOURNAL_LOCK:
                # Fault storms must not append the same capture thousands of
                # times a second. Preserve distinct faults and every repair /
                # resolution event; report skipped repeats on the next sample.
                repeats = getattr(self, "_journal_repeats", None)
                if repeats is None:
                    repeats = self._journal_repeats = OrderedDict()
                key = (entry["error"], entry["file"], entry["line"], note)
                now = time.monotonic()
                if event == "captured":
                    previous = repeats.get(key)
                    if previous and now - previous[0] < JOURNAL_REPEAT_WINDOW_S:
                        repeats[key] = (previous[0], previous[1] + 1)
                        repeats.move_to_end(key)
                        return
                    if previous and previous[1]:
                        entry["suppressed_repeats"] = previous[1]
                JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                with JOURNAL_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
                if event == "captured":
                    repeats[key] = (now, 0)
                    repeats.move_to_end(key)
                    while len(repeats) > 128:
                        repeats.popitem(last=False)
        except Exception:
            pass

    def journal_tail(self, limit: int = 8) -> list[dict[str, Any]]:
        """The most recent journal entries — past faults inform new repairs."""
        out: list[dict[str, Any]] = []
        for line in tail_lines(JOURNAL_PATH, limit):
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    out.append(entry)
            except ValueError:
                continue
        return out

    # ── inspection ────────────────────────────────────────────────────────────

    def incidents(self) -> list[Incident]:
        return list(self._incidents.values())

    def latest(self) -> Optional[Incident]:
        return list(self._incidents.values())[-1] if self._incidents else None

    def _read_source_window(self, file: str, line: int, radius: int = 30) -> str:
        try:
            lines = Path(file).read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        start = max(0, line - radius)
        end = min(len(lines), line + radius)
        return "\n".join(f"{n+1:>5}  {lines[n]}" for n in range(start, end))

    # ── fix proposal (never applied automatically) ────────────────────────────

    async def propose_fix(self, incident_id: str = "") -> ToolResult:
        incident = self._incidents.get(incident_id) or self.latest()
        if incident is None:
            return ToolResult("No captured incidents to repair.", ok=False)
        # A reported defect has no traceback narrowing the location, so give the
        # model far more of the file to reason over than a stack frame needs.
        radius = 120 if incident.reported else 30
        source = (self._read_source_window(incident.file, incident.line, radius)
                  if incident.file else "")
        log_tail = "\n".join(
            f"{r['level']} {r['component']}: {r['message']}" for r in incident.logs[-12:]
        )
        persona = (
            "SPECIALIST MODE — Site Reliability & Repair Engineer. Given a Python "
            "traceback, the failing source window and recent logs, diagnose the root "
            "cause and produce a MINIMAL fix as a unified diff (--- / +++ / @@). Do "
            "not rewrite unrelated code. Explain the root cause in two sentences, then "
            "give the diff. If the cause is external (missing dependency, bad config), "
            "say so and give the exact remediation command instead of a code diff."
        )
        history = "\n".join(
            f"- {e.get('event', '?')}: {e.get('error', '')} "
            f"({e.get('file', '')}:{e.get('line', 0)}) {e.get('note', '')}"
            for e in self.journal_tail()
        )
        if incident.reported:
            # No traceback exists: the user described behaviour that is wrong
            # but does not raise. Say so plainly rather than letting the model
            # hallucinate a stack trace it was never given.
            prompt = (
                f"Reported defect (no exception was raised — the behaviour is "
                f"simply wrong):\n{incident.message}\n\n"
                f"Most likely source ({incident.file}, centred on line "
                f"{incident.line}) — this location was found by searching the "
                f"package for the terms in the report, so verify it is really "
                f"the right place before proposing a change:\n{source}\n\n"
                f"Recent logs:\n{log_tail}"
                + (f"\n\nRepair history (past faults and fixes — avoid regressions):"
                   f"\n{history}" if history else "")
                + "\n\nIf this file is NOT where the behaviour lives, say which "
                  "module you would look in instead and why, rather than "
                  "inventing a change here."
            )
        else:
            prompt = (
                f"Error: {incident.error_type}: {incident.message}\n\n"
                f"Traceback:\n{incident.traceback}\n\n"
                f"Failing source ({incident.file}, around line {incident.line}):\n{source}\n\n"
                f"Recent logs:\n{log_tail}"
                + (f"\n\nRepair history (past faults and fixes — avoid regressions):\n{history}"
                   if history else "")
            )
        if not self.router.has_text_fallback():
            proposal = (
                f"# Self-repair context for {incident.id}\n\n"
                "No text provider is configured, so ORION cannot draft the patch itself.\n"
                "The captured context below is what the live model needs to propose a fix.\n\n"
                f"## Error\n{incident.error_type}: {incident.message}\n\n"
                f"## Traceback\n```\n{incident.traceback}\n```\n\n"
                f"## Source window\n```\n{source}\n```\n"
            )
        else:
            # A PATCH, validated — not advice. The old proposal was prose
            # ("I would recommend reviewing app.py…") written by whatever
            # small model was left; nothing in it could be applied.
            drafted = await self._draft_validated_fix(incident)
            if drafted is not None:
                proposal = drafted
            else:
                try:
                    profile, answer = await self.router.generate_text(
                        prompt, system_extra=persona, task="repair", max_tokens=4096)
                except TypeError:
                    profile, answer = await self.router.generate_text(prompt, system_extra=persona)
                except Exception as exc:
                    return ToolResult(f"Repair proposal failed: {first_line(exc)}", ok=False)
                proposal = (
                    f"# Self-repair proposal for {incident.id}  (via {profile.name})\n\n"
                    f"## Incident\n{incident.summary()}\n\n"
                    "No validated patch could be produced automatically; analysis "
                    f"follows.\n\n{answer}\n")
        path = SELF_REPAIR_DIR / f"proposal_{incident.id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        # The save used to be swallowed and then announced regardless — the log
        # said "written to" and the reply gave a path to a file that did not
        # exist. The proposal is still worth returning inline when it cannot be
        # saved; it must not be described as saved.
        try:
            path.write_text(proposal, encoding="utf-8")
            incident.proposal_path = str(path)
            saved = f"Saved for review at: {path}\n"
            self.bus.log.emit(
                f"REPAIR: proposal for {incident.id} written to {path.name} (review required).")
        except OSError as exc:
            saved = (f"I couldn't save it to disk ({first_line(exc, 60)}) — "
                     "the full proposal is below.\n")
            self.bus.log.emit(
                f"REPAIR: proposal for {incident.id} could not be saved - {first_line(exc, 80)}")
        return ToolResult(
            f"Repair proposal for {incident.summary()} prepared.\n"
            + saved
            + "No change has been applied — approve and copy the fix across manually.\n\n"
            + proposal[:3000]
        )

    async def _draft_validated_fix(self, incident: Incident) -> str | None:
        """Draft, compile and test a fix; return the proposal document or None.

        For a reported defect each located candidate is tried in turn, because
        the first guess is a search result, not a stack frame: when the model
        says "NOT HERE", the next candidate gets its chance.
        """
        import difflib

        targets: list[tuple[str, int]] = []
        if incident.file:
            targets.append((incident.file, incident.line))
        if incident.reported:
            for path, line, _score in self.locate_candidates(incident.message):
                if all(Path(path) != Path(t) for t, _l in targets):
                    targets.append((path, line))
        for file, line in targets[:3]:
            target = Path(file)
            if not target.is_file() or PACKAGE_DIR not in target.resolve().parents:
                continue
            incident.file, incident.line = str(target), line
            incident.repaired_content = ""
            generated = await self._generate_repaired_file(incident, target)
            if not generated.ok or not incident.repaired_content:
                continue
            original = target.read_text(encoding="utf-8", errors="replace")
            passed, detail = await asyncio.to_thread(
                self._run_patched_tests, target, incident.repaired_content)
            incident.tests_passed, incident.tests_detail = passed, detail
            diff = "".join(difflib.unified_diff(
                original.splitlines(keepends=True),
                incident.repaired_content.splitlines(keepends=True),
                fromfile=f"a/{target.relative_to(PACKAGE_DIR.parent).as_posix()}",
                tofile=f"b/{target.relative_to(PACKAGE_DIR.parent).as_posix()}"))
            verdict = ("its tests PASS" if passed is True else
                       "its tests FAIL — review before approving" if passed is False
                       else "no dedicated tests exist for this module")
            return (f"# Self-repair proposal for {incident.id}\n\n"
                    f"## Incident\n{incident.summary()}\n\n"
                    f"## Root cause\n{incident.root_cause or '(not stated)'}\n\n"
                    f"## Verification\nCompiles; {verdict} ({detail}).\n\n"
                    f"## Patch ({self._diff_lines(original, incident.repaired_content)} "
                    f"changed lines)\n```diff\n{diff[:20000]}\n```\n\n"
                    "Nothing has been applied. Say \"apply the repair\" (self_repair "
                    "repair with confirm=true) to back up the original and write it; "
                    "a restart then loads it, and \"revert the last repair\" undoes it.\n")
        return None

    # ── test / verify suite ───────────────────────────────────────────────────

    async def run_tests(self, path: str = "") -> ToolResult:
        """
        Run the verification suite: byte-compile the package (fast smoke) and,
        if a tests/ directory exists, pytest.  Executed off the event loop.
        """
        root = Path(path).expanduser() if path.strip() else BASE_DIR

        def _run() -> tuple[str, bool]:
            import subprocess
            out: list[str] = []
            compile_proc = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", str(PACKAGE_DIR)],
                capture_output=True, text=True, timeout=120,
            )
            out.append(f"compileall: exit {compile_proc.returncode}")
            ok = compile_proc.returncode == 0
            if compile_proc.stdout.strip():
                out.append(compile_proc.stdout.strip()[-1500:])
            if compile_proc.stderr.strip():
                out.append(compile_proc.stderr.strip()[-1500:])
            if (root / "tests").is_dir():
                try:
                    pytest_proc = subprocess.run(
                        [sys.executable, "-m", "pytest", "-q", str(root / "tests")],
                        cwd=str(root), capture_output=True, text=True, timeout=300,
                    )
                    out.append(f"pytest: exit {pytest_proc.returncode}\n"
                               + (pytest_proc.stdout or "")[-2500:])
                    if pytest_proc.stderr.strip():
                        out.append(pytest_proc.stderr.strip()[-1500:])
                    ok = ok and pytest_proc.returncode == 0
                except FileNotFoundError:
                    out.append("Could not start the test interpreter.")
                    ok = False
                except subprocess.TimeoutExpired:
                    out.append("pytest timed out.")
                    ok = False
            else:
                out.append("No tests/ directory; compile smoke only.")
            return "\n".join(out), ok

        try:
            output, ok = await asyncio.to_thread(_run)
        except Exception as exc:
            return ToolResult(f"Test run failed: {first_line(exc)}", ok=False)
        if self.telemetry is not None:
            self.telemetry.health.beat("self_repair", "OK" if ok else "DEGRADED", "tests run")
        return ToolResult(output, ok=ok)

    # ── approval-gated CODE self-repair (fixes its own source) ────────────────

    BACKUP_DIR = SELF_REPAIR_DIR / "backups"

    async def repair_file(self, incident_id: str = "", confirm: bool = False) -> ToolResult:
        """
        Actually fix ORION's own code — but only with explicit approval.

        First call (confirm=False): the model rewrites the *entire* failing file
        with a minimal correction; the corrected file is validated for compile
        and previewed, but nothing is written to the live source.

        Second call (confirm=True): the original is backed up, the corrected
        file is written, re-validated (reverted automatically if it fails to
        compile), and ORION reports that a restart is needed to load it.

        Only files inside the orion_core package can be repaired.
        """
        incident = self._incidents.get(incident_id) or self.latest()
        if incident is None:
            return ToolResult("No captured incident to repair.", ok=False)
        target = Path(incident.file)
        if not target.is_file() or PACKAGE_DIR not in target.resolve().parents:
            return ToolResult(
                f"I can only repair files inside the orion_core package "
                f"(the fault is in {incident.file or 'an unknown file'}).", ok=False)

        # Generate the corrected file if we don't have one yet.
        if not incident.repaired_content:
            gen = await self._generate_repaired_file(incident, target)
            if not gen.ok:
                return gen

        if not confirm:
            # Priority 3.3: exercise the module's own tests against the patched
            # copy so the human approves with pass/fail in hand, not just a
            # compile check. Isolated (a temp copy) — the live source is never
            # touched, and this still never auto-applies.
            passed, detail = await asyncio.to_thread(
                self._run_patched_tests, target, incident.repaired_content)
            incident.tests_passed = passed
            incident.tests_detail = detail
            if passed is True:
                verdict = f"✅ its tests PASS ({detail})."
            elif passed is False:
                verdict = (f"⚠️ its tests FAIL ({detail}). Review carefully before "
                           "approving — the fix may be incomplete or wrong.")
            else:
                verdict = f"(no dedicated tests to run: {detail})."
            preview = incident.repaired_content[:1400]
            return ToolResult(
                f"I've drafted a corrected '{target.name}'; it compiles and "
                f"{verdict} Nothing is applied yet — approve with confirm=true to "
                "back up the original and write the fix (a restart then loads it).\n\n"
                f"Preview (first lines):\n{preview}", ok=True)

        return await asyncio.to_thread(self._apply_repaired, incident, target)

    # ── isolated test run against the patched copy (Priority 3.3) ─────────────

    def _relevant_tests(self, stem: str) -> list[Path]:
        """Test files that exercise the module being patched, by naming
        convention ``tests/test_*<stem>*.py`` (e.g. memory → test_memory_matrix)."""
        tests_dir = BASE_DIR / "tests"
        if not tests_dir.is_dir():
            return []
        matches = sorted(set(tests_dir.glob(f"test_{stem}*.py"))
                         | set(tests_dir.glob(f"test_*{stem}*.py")))
        return matches

    def _run_patched_tests(self, target: Path, content: str) -> tuple[Optional[bool], str]:
        """Copy the package to a temp dir, swap in the patched file, and run the
        module's tests there. Returns (passed|None, one-line detail). Never
        mutates the live source and never raises."""
        import shutil
        import subprocess
        import tempfile

        stem = target.stem
        matches = self._relevant_tests(stem)
        if not matches:
            return None, f"no tests matching '{stem}'"
        tmp = Path(tempfile.mkdtemp(prefix="orion_repair_tests_"))
        try:
            pkg_copy = tmp / PACKAGE_DIR.name
            shutil.copytree(
                PACKAGE_DIR, pkg_copy,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.db",
                                              "*.db-wal", "*.db-shm"))
            # Relative to the package, so a patch to gui/core_window.py lands
            # in gui/ — writing it to the package root tested the ORIGINAL.
            try:
                relative = target.resolve().relative_to(PACKAGE_DIR.resolve())
            except ValueError:
                relative = Path(target.name)
            (pkg_copy / relative).write_text(content, encoding="utf-8")
            dst_tests = tmp / "tests"
            dst_tests.mkdir()
            conftest = BASE_DIR / "tests" / "conftest.py"
            if conftest.exists():
                shutil.copy2(conftest, dst_tests / "conftest.py")
            for test_file in matches:
                shutil.copy2(test_file, dst_tests / test_file.name)
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q",
                 *[str(dst_tests / t.name) for t in matches]],
                cwd=str(tmp), capture_output=True, text=True, timeout=300,
            )
            summary = ""
            for line in reversed((proc.stdout or "").strip().splitlines()):
                if line.strip():
                    summary = line.strip()
                    break
            names = ", ".join(t.name for t in matches)
            return (proc.returncode == 0), f"{names}: {summary}" if summary else names
        except FileNotFoundError:
            return None, "pytest not installed"
        except subprocess.TimeoutExpired:
            return None, "tests timed out"
        except Exception as exc:
            return None, f"could not run tests ({first_line(exc, 80)})"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    #: How a fix is asked for. EDITS, not a rewritten file: the old request
    #: was "return the COMPLETE corrected file", which (a) could never fit a
    #: real module inside the reply budget — core_window.py is 120k chars —
    #: so the file came back cut off, failed to compile and was refused, and
    #: (b) invited the model to "tidy" code it was not asked to touch.
    _EDIT_INSTRUCTION = (
        "You are ORION's repair engineer. Fix the defect with the SMALLEST "
        "correct change to the file you are given. Reply in exactly this "
        "format and nothing else:\n\n"
        "ROOT CAUSE: <one or two sentences>\n\n"
        "<<<<<<< SEARCH\n<lines copied EXACTLY from the file, enough to be "
        "unique>\n=======\n<the replacement lines>\n>>>>>>> REPLACE\n\n"
        "Use one SEARCH/REPLACE block per change (several are allowed). The "
        "SEARCH text must match the file character for character, including "
        "indentation. Never invent code outside the blocks. If the defect is "
        "not in this file, reply only: NOT HERE: <which module and why>.")

    _BLOCK_RE = __import__("re").compile(
        r"<{5,9} ?SEARCH[^\n]*\n(.*?)\n?={5,9}[^\n]*\n(.*?)\n?>{5,9} ?REPLACE", __import__("re").S)

    @classmethod
    def parse_edits(cls, answer: str) -> list[tuple[str, str]]:
        """(search, replace) pairs from a SEARCH/REPLACE reply."""
        return [(s, r) for s, r in cls._BLOCK_RE.findall(str(answer or "")) if s.strip()]

    @staticmethod
    def apply_edits(original: str, edits: list[tuple[str, str]]) -> str:
        """Apply *edits* to *original*, exactly — or with line-ending and
        trailing-space tolerance. Raises ValueError naming the block that did
        not match, so the model can be told precisely what to fix."""
        text = original
        crlf = "\r\n" in original
        for index, (search, replace) in enumerate(edits, 1):
            candidates = [search]
            if crlf:
                candidates.append(search.replace("\r\n", "\n").replace("\n", "\r\n"))
            done = False
            for needle in candidates:
                if needle and text.count(needle) == 1:
                    body = replace.replace("\r\n", "\n").replace("\n", "\r\n") if crlf else replace
                    text = text.replace(needle, body, 1)
                    done = True
                    break
            if done:
                continue
            # Tolerate trailing whitespace differences, line by line.
            lines = text.splitlines(keepends=True)
            want = [l.rstrip() for l in search.replace("\r\n", "\n").split("\n")]
            n = len(want)
            hits = [i for i in range(len(lines) - n + 1)
                    if [l.rstrip() for l in lines[i:i + n]] == want]
            if len(hits) != 1:
                raise ValueError(
                    f"SEARCH block {index} {'matched nothing' if not hits else 'is ambiguous'}"
                    f" — it must be copied exactly from the file and be unique: "
                    f"{first_line(search, 80)!r}")
            start = hits[0]
            eol = "\r\n" if crlf else "\n"
            body = replace.replace("\r\n", "\n")
            new_lines = [l + eol for l in body.split("\n")] if body else []
            lines[start:start + n] = new_lines
            text = "".join(lines)
        return text

    async def _generate_repaired_file(self, incident: Incident, target: Path) -> ToolResult:
        """Draft a validated fix for *target*: edits applied, and it compiles.

        Works on any file size (edits are applied here, not retyped by the
        model). One self-correcting retry when a SEARCH block does not match
        or the result will not compile — the precise reason is fed back.
        """
        try:
            original = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(f"Could not read {target.name}: {first_line(exc)}", ok=False)
        if not self.router.has_text_fallback():
            return ToolResult(
                "No language model is reachable to draft the code fix. When one is "
                "available, ask me to repair the file again.", ok=False)
        if incident.reported:
            radius = 160
            window = self._read_source_window(str(target), incident.line or 1, radius)
            problem = (f"REPORTED DEFECT (no exception — the behaviour is wrong):\n"
                       f"{incident.message}")
        else:
            window = self._read_source_window(str(target), incident.line or 1, 60)
            problem = (f"Error: {incident.error_type}: {incident.message}\n\n"
                       f"Traceback:\n{incident.traceback}")
        # The whole file when it fits comfortably; a generous window otherwise.
        body = (original if len(original) <= 60000
                else f"(showing lines around {incident.line}; line numbers are for "
                     f"orientation only and must NOT be copied into SEARCH text)\n{window}")
        try:
            shown = target.resolve().relative_to(PACKAGE_DIR.parent.resolve()).as_posix()
        except ValueError:
            shown = target.name
        prompt = f"{problem}\n\nFILE: {shown}\n```python\n{body}\n```"
        import py_compile
        import tempfile

        feedback = ""
        last_problem = ""
        for attempt in range(2):
            try:
                _profile, answer = await self.router.generate_text(
                    prompt + feedback, instruction=self._EDIT_INSTRUCTION,
                    task="repair", max_tokens=8192)
            except TypeError:        # an older router / test double
                _profile, answer = await self.router.generate_text(
                    prompt + feedback, system_extra=self._EDIT_INSTRUCTION)
            except Exception as exc:
                return ToolResult(f"Repair generation failed: {first_line(exc)}", ok=False)
            answer = str(answer or "")
            if answer.strip().upper().startswith("NOT HERE"):
                incident.root_cause = answer.strip()[:600]
                return ToolResult(f"The model judged the defect is not in {target.name}: "
                                  f"{first_line(answer, 200)}", ok=False)
            edits = self.parse_edits(answer)
            if not edits:
                # A model that ignored the format and sent the whole file.
                # Acceptable for a small module (and still compile-checked);
                # for a large one it cannot have retyped it faithfully.
                whole = self._strip_fences(answer)
                full_lines = original.count("\n") + 1
                if len(original) <= 40000 and whole.strip() and (
                        full_lines <= 60 or whole.count("\n") + 1 >= full_lines // 2):
                    corrected = whole
                else:
                    last_problem = ("the reply was not a set of SEARCH/REPLACE edits (a diff) "
                                    "and the file is too large to accept a rewrite")
                    feedback = ("\n\nYOUR REPLY CONTAINED NO SEARCH/REPLACE BLOCKS. "
                                "Use the exact format.")
                    continue
            else:
                try:
                    corrected = self.apply_edits(original, edits)
                except ValueError as exc:
                    last_problem = f"an edit did not apply ({exc})"
                    feedback = f"\n\nYOUR EDIT COULD NOT BE APPLIED: {exc}. Try again."
                    continue
            if corrected == original:
                last_problem = "the edits changed nothing"
                feedback = "\n\nYOUR EDITS CHANGED NOTHING. Make the fix."
                continue
            tmp = Path(tempfile.gettempdir()) / f"orion_repair_{incident.id}.py"
            try:
                tmp.write_text(corrected, encoding="utf-8")
                py_compile.compile(str(tmp), doraise=True)
            except Exception as exc:
                last_problem = f"does not compile ({first_line(exc, 100)})"
                feedback = (f"\n\nTHE EDITED FILE DOES NOT COMPILE: {first_line(exc, 160)}. "
                            "Fix your edit.")
                continue
            finally:
                tmp.unlink(missing_ok=True)
            cause = answer.split("<<<<<<<", 1)[0].strip()
            incident.root_cause = cause[:800]
            incident.repaired_content = corrected
            return ToolResult("draft ready")
        if last_problem.startswith("does not compile"):
            return ToolResult(f"The drafted fix for {target.name} {last_problem}; "
                              "I won't offer to apply it.", ok=False)
        return ToolResult(
            f"I could not produce a clean, applicable diff for {target.name} "
            f"({last_problem or 'no usable edit'}).", ok=False)

    def _apply_repaired(self, incident: Incident, target: Path) -> ToolResult:
        """Replace *target* with the drafted fix — only ever with a proven backup.

        This rewrites ORION's own source, so the ordering is the whole design.
        An earlier version backed up, wrote, compiled and — on ANY failure —
        "restored" by copying the backup file back over the target. That was
        backwards on the failure that matters: when writing the backup itself
        failed part-way (disk full, a OneDrive lock), a truncated backup
        existed, so the "rollback" overwrote an intact original with it and
        then reported "I reverted it". A failed restore was swallowed with the
        same claim.

        Now: the original is held in memory as BYTES (the old text round-trip
        with errors="replace" was not even byte-faithful); nothing touches the
        target until the backup on disk reads back identical; the target is
        written atomically, so a failed write leaves it untouched and needs no
        restore; and a restore that fails is reported as exactly that, with
        the path to recover from.
        """
        import py_compile
        from .atomic_io import atomic_write_bytes, atomic_write_text

        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        backup = self.BACKUP_DIR / f"{target.name}.{stamp}.bak"
        try:
            original = target.read_bytes()
        except OSError as exc:
            return ToolResult(
                f"I couldn't read {target.name} to back it up "
                f"({first_line(exc, 60)}), so I changed nothing.", ok=False)
        try:
            self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(backup, original)
            if backup.read_bytes() != original:
                raise OSError("the backup did not read back identical to the original")
        except OSError as exc:
            return ToolResult(
                f"I couldn't make a verified backup of {target.name} "
                f"({first_line(exc, 60)}), so I changed nothing.", ok=False)
        try:
            atomic_write_text(target, incident.repaired_content, encoding="utf-8")
        except OSError as exc:
            # Atomic: a failed write never reached the target.
            return ToolResult(
                f"I couldn't write the fix to {target.name} "
                f"({first_line(exc, 60)}); the original is untouched.", ok=False)
        try:
            py_compile.compile(str(target), doraise=True)
        except Exception as exc:
            try:
                atomic_write_bytes(target, original)
            except OSError as restore_exc:
                self.bus.log.emit(
                    f"REPAIR: {target.name} failed to compile AND could not be "
                    f"restored ({first_line(restore_exc, 80)}) — backup at {backup}.")
                return ToolResult(
                    f"The fix for {target.name} did not compile "
                    f"({first_line(exc, 60)}), and restoring the original ALSO "
                    f"failed ({first_line(restore_exc, 60)}). {target.name} is "
                    f"currently broken. The original is saved at {backup} — copy "
                    "it back over the file before restarting me.", ok=False)
            return ToolResult(
                f"The fix for {target.name} did not compile "
                f"({first_line(exc, 60)}), so I reverted it to the original.", ok=False)
        incident.applied = True
        incident.backup_path = str(backup)
        self.bus.log.emit(f"REPAIR: applied fix to {target.name} (backup {backup.name}).")
        self.bus.banner.emit(f"SELF-REPAIR APPLIED: {target.name} — restart to load", 3)
        self._journal("repaired", incident, note=f"backup {backup.name}")
        return ToolResult(
            f"Done — I've corrected {target.name} and it compiles. The original is "
            f"backed up at {backup.name}. I can restart myself now to load the fix "
            "(the restart_orion tool) — shall I? Say 'revert the last repair' if "
            "anything seems off afterwards.")

    def revert_last(self) -> ToolResult:
        applied = [i for i in self._incidents.values() if i.applied and i.backup_path]
        if not applied:
            return ToolResult("There's no applied repair to revert.")
        incident = applied[-1]
        target = Path(incident.file)
        backup = Path(incident.backup_path)
        try:
            from .atomic_io import atomic_write_bytes
            # Bytes, atomically: the backup is a byte copy of the original, and
            # a revert that fails part-way must not leave the source truncated.
            atomic_write_bytes(target, backup.read_bytes())
            incident.applied = False
            self._journal("reverted", incident)
            return ToolResult(
                f"Reverted {target.name} to the pre-repair backup. "
                "I can restart myself to load it — just say the word.")
        except Exception as exc:
            return ToolResult(f"Revert failed: {first_line(exc)}", ok=False)

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        # If the model wrapped the file in a ```python fenced block (often with
        # prose around it), extract the block's contents.
        import re as _re
        m = _re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text, _re.DOTALL)
        if m:
            return m.group(1).strip("\n")
        # Otherwise strip stray leading/trailing fence lines.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text
