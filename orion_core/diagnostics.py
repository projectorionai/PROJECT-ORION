"""
DiagnosticsEngine — ORION runs complex self-tests on itself.

When ORION hits a multitude of errors (or you simply ask "run a full
diagnostic"), this engine executes a battery of checks across the whole system
and returns a ranked PASS / WARN / FAIL report:

    • compile       — byte-compile every module in the package;
    • imports       — import every submodule (catches broken wiring);
    • dependencies  — required vs optional third-party packages present;
    • database      — SQLite integrity_check on the memory store;
    • config        — every JSON config file parses;
    • tools         — declared tools all resolve to a handler;
    • health        — telemetry health of every registered component;
    • resources     — CPU/RAM/disk headroom;
    • permissions   — the config directory is writable.

It pairs with the SelfRepairAgent: a FAIL here is exactly the context ORION
feeds into a repair proposal.  Everything heavy runs via ``asyncio.to_thread``.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import CONFIG_DIR, CORE_DB_PATH, PACKAGE_DIR
from .data import ToolResult
from .module_presence import module_present
from .utils import first_line, utc_stamp

# (module, required?) — required failures are FAIL, optional are WARN.
DEPENDENCIES: tuple[tuple[str, bool], ...] = (
    ("PyQt6", True), ("qasync", True), ("aiohttp", True), ("sounddevice", True),
    ("mss", True), ("psutil", True), ("google.genai", True), ("PIL", True),
    ("numpy", True), ("cv2", True), ("pyautogui", True), ("pywinauto", True),
    ("pygetwindow", True), ("keyboard", True), ("screeninfo", True),
    ("rapidocr_onnxruntime", False), ("pytesseract", False), ("docx", False),
    ("matplotlib", False), ("pypdf", False), ("pyttsx3", False), ("vosk", False),
    ("win32com", False),
)


def _module_present(module: str) -> bool:
    """Is ``module`` installed — WITHOUT executing it.

    This runs on a worker thread while the Qt loop paints.  Importing a
    package here ran its native initialisers off the main thread: on
    30 Sep 2026 the periodic diagnostic's ``import pywinauto`` pulled in
    ``win32ui`` (MFC) and the app died with a Windows access violation
    (crash_reports/faulthandler.log).  A presence check only needs the
    import system's lookup, never the module body.
    """
    return module_present(module)


#: Packages that belong in a separate training/worker process, never in the
#: running app.  The import check used to load every module, so 5 minutes
#: after start the periodic diagnostic imported grandmaster/nnue.py and left
#: PyTorch resident (+144 MB, measured 3 Oct 2026) for the rest of the session.
WORKER_ONLY_PACKAGES = frozenset({
    "torch", "tensorflow", "jax", "transformers", "sentence_transformers",
})


def _import_time_names(source: bytes, module: str, is_package: bool) -> set[str]:
    """Absolute names a module imports while its body executes.

    Function bodies are skipped (those imports are lazy); class bodies,
    ``try`` and ``if`` blocks at module level all run at import time.
    Relative imports are resolved against ``module``.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    package = module if is_package else module.rpartition(".")[0]
    names: set[str] = set()
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                base = ".".join(parts)
                base = f"{base}.{node.module}" if node.module else base
            else:
                base = node.module or ""
            names.add(base)
            # ``from .pkg import sub`` imports the submodule ``pkg.sub`` too.
            names.update(f"{base}.{alias.name}" for alias in node.names
                         if alias.name != "*")
        stack.extend(ast.iter_child_nodes(node))
    return names


def _worker_only_modules(modules: dict[str, Path]) -> dict[str, set[str]]:
    """Map each package module that would load a worker-only package at
    import time (directly or through another package module) to the
    worker-only packages it needs."""
    direct: dict[str, set[str]] = {}
    internal: dict[str, set[str]] = {}
    for mod, path in modules.items():
        try:
            source = path.read_bytes()
        except OSError:
            continue
        names = _import_time_names(source, mod, path.name == "__init__.py")
        roots = {name.split(".")[0] for name in names}
        direct[mod] = roots & WORKER_ONLY_PACKAGES
        internal[mod] = {name for name in names if name in modules and name != mod}
    needs = {mod: set(found) for mod, found in direct.items()}
    changed = True
    while changed:  # propagate through package-internal imports
        changed = False
        for mod, deps in internal.items():
            for dep in deps:
                extra = needs.get(dep, set()) - needs[mod]
                if extra:
                    needs[mod] |= extra
                    changed = True
    return {mod: found for mod, found in needs.items() if found}


@dataclass
class Check:
    name: str
    status: str          # PASS | WARN | FAIL
    detail: str

    def line(self) -> str:
        icon = {"PASS": "✓", "WARN": "▲", "FAIL": "✗"}.get(self.status, "?")
        return f"  {icon} {self.name}: {self.detail}"


class DiagnosticsEngine:
    # Periodic autonomous health check (run()): cheap relative to
    # ImprovementHeartbeat's own 25-minute cadence (forge.py), which does a
    # full LLM review and can forge a whole new tool — byte-compiling and
    # re-importing the package is comparatively light, and imports are
    # no-ops for anything already in sys.modules.
    INTERVAL_S = 20 * 60
    FIRST_TICK_DELAY_S = 5 * 60

    def __init__(self, bus: OrionBus, memory: Any, telemetry: Any | None = None,
                 dispatcher: Any | None = None, forge: Any | None = None,
                 repair: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        self.dispatcher = dispatcher
        # Optional runtime references: with these attached the diagnostic can
        # report what the FORGE and self-repair layers have actually been
        # doing this session — a diagnostic that says "all healthy" seconds
        # after a forge failure is worse than no diagnostic at all.
        self.forge = forge
        self.repair = repair

    async def run(self) -> None:
        """Periodic autonomous self-diagnostic — so the Diagnostics Centre
        panel has fresh data without the user needing to ask for it.
        run_full() already emits bus.dashboard_event("diagnostics", ...);
        this just calls it on a schedule instead of on request only."""
        await asyncio.sleep(self.FIRST_TICK_DELAY_S)
        while True:
            try:
                await self.run_full()
            except Exception as exc:
                self.bus.log.emit(f"DIAG: periodic self-diagnostic failed - {exc}")
            await asyncio.sleep(self.INTERVAL_S)

    async def run_full(self) -> ToolResult:
        self.bus.log.emit("DIAG: running full self-diagnostic…")
        from . import background, diagnostic_imports
        stop = threading.Event()
        with diagnostic_imports.cancellation_scope(stop):
            work = background.spawn(asyncio.to_thread(self._run_all), name="orion-diagnostics")
        if work is None:
            return ToolResult("Diagnostic did not start: no active event loop.", ok=False)
        try:
            checks = await asyncio.shield(work)
        except asyncio.CancelledError:
            stop.set()  # the bounded verifier observes this and retires its child
            raise
        fails = [c for c in checks if c.status == "FAIL"]
        warns = [c for c in checks if c.status == "WARN"]
        verdict = ("all systems healthy" if not fails and not warns else
                   f"{len(fails)} fault(s), {len(warns)} warning(s)")
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("diag.fails", float(len(fails)))
            self.telemetry.metrics.incr("diag.run")
        self.bus.dashboard_event.emit("diagnostics", {
            "at": utc_stamp(), "fails": len(fails), "warns": len(warns),
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in checks],
        })
        header = f"Diagnostic complete — {verdict}."
        body = "\n".join(c.line() for c in checks)
        return ToolResult(f"{header}\n{body}", ok=not fails)

    # ── individual checks (sync; run in a worker thread) ──────────────────────

    def _run_all(self) -> list[Check]:
        from .diagnostic_imports import cancelled
        checks = []
        for check in (self._check_compile, self._check_imports, self._check_dependencies,
                      self._check_database, self._check_config, self._check_tools,
                      self._check_health, self._check_resources, self._check_permissions,
                      self._check_forge, self._check_incidents, self._check_crashes,
                      self._check_capabilities):
            if cancelled():
                break
            checks.append(check())
        return checks

    def _check_capabilities(self) -> Check:
        """Roll the capability-health matrix into one line, so 'run a diagnostic'
        surfaces a capability that is present but not actually usable (the OCR
        silent-fault class)."""
        try:
            from .capability_health import capability_matrix
            rows = capability_matrix(deep=False)
        except Exception as exc:
            return Check("capabilities", "WARN", first_line(exc, 70))
        missing = [r.name for r in rows if r.status() == "MISSING"]
        degraded = [r.name for r in rows if r.status() == "DEGRADED"]
        if missing or degraded:
            parts = []
            if missing:
                parts.append(f"missing: {', '.join(missing)}")
            if degraded:
                parts.append(f"degraded: {', '.join(degraded)}")
            parts.append("ask 'capability health' for the full matrix")
            return Check("capabilities", "WARN", "; ".join(parts))
        return Check("capabilities", "PASS",
                     f"{len(rows)} dependency/registration checks passed; runtime readiness not established")

    async def capability_report(self, deep: bool = True) -> ToolResult:
        """The full capability-health matrix — 'why can't ORION do X?'. deep=True
        runs functional probes (e.g. an actual OCR read)."""
        from .capability_health import capability_matrix, render_matrix
        rows = await asyncio.to_thread(capability_matrix, deep)
        ready = all(r.status() == "OK" for r in rows)
        receipt = {"capability_health": [row.as_dict() for row in rows],
                   "scope": "Probe evidence only; dependency checks do not establish runtime readiness"}
        recent = getattr(getattr(self, "dispatcher", None), "recent_executions", None)
        if callable(recent):
            receipt["recent_tool_executions"] = recent(limit=10)
        return ToolResult(render_matrix(rows), ok=ready, receipt=receipt)

    def _check_forge(self) -> Check:
        if self.forge is None:
            return Check("forge", "PASS", "forge not attached this session")
        # Quarantine and contract failures are PERSISTED state — tools held
        # back from a previous run — not just this session's activity. Only
        # looking at `sessions` (this run's forge attempts) made a quarantined
        # tool from an earlier session invisible to self-diagnostic entirely:
        # it reported "no forge sessions this run" even while a broken tool
        # sat inert on disk. Surface both here so "run self-diagnostic" is
        # enough to learn a tool is held back and how to fix it.
        held_back: list[str] = []
        try:
            quarantine = list(getattr(self.forge, "_quarantined", lambda: [])())
        except Exception:
            quarantine = []
        if quarantine:
            names = ", ".join(row.get("tool", "?") for row in quarantine[:3])
            held_back.append(f"{len(quarantine)} quarantined ({names}) — re-forge to replace")
        try:
            contract_failures = dict(getattr(self.forge.loader, "contract_failures", {}) or {})
        except Exception:
            contract_failures = {}
        if contract_failures:
            names = ", ".join(list(contract_failures)[:3])
            held_back.append(f"{len(contract_failures)} break the tool contract ({names}) — re-forge to replace")
        try:
            sessions = list(getattr(self.forge, "sessions", {}).values())
        except Exception as exc:
            return Check("forge", "WARN", first_line(exc, 60))
        failed = [s for s in sessions if not s.activation_ok]
        if failed:
            latest = failed[-1]
            held_back.append(
                f"{len(failed)}/{len(sessions)} session(s) failed this run — latest "
                f"'{latest.tool_name}': "
                + first_line(str(latest.errors[-1:] or ['unknown']), 90))
        if held_back:
            return Check("forge", "WARN", "; ".join(held_back))
        if not sessions:
            return Check("forge", "PASS", "no forge sessions this run; nothing held back")
        return Check("forge", "PASS", f"{len(sessions)} forge session(s), all activated")

    def _check_incidents(self) -> Check:
        if self.repair is None:
            return Check("incidents", "PASS", "self-repair not attached")
        try:
            incidents = self.repair.incidents()
        except Exception as exc:
            return Check("incidents", "WARN", first_line(exc, 60))
        open_incidents = [i for i in incidents if not i.applied]
        if open_incidents:
            latest = open_incidents[-1]
            return Check(
                "incidents", "WARN",
                f"{len(open_incidents)} captured incident(s) — latest "
                f"[{latest.id}] {latest.error_type}: {first_line(latest.message, 80)}")
        return Check("incidents", "PASS", "no captured incidents this session")

    #: How far back a crash still counts as "recent" for the diagnostic.
    CRASH_WINDOW_S = 7 * 24 * 3600

    def _check_crashes(self) -> Check:
        """Serious crash reports from the last week — including a native crash
        that ended a previous session, which no in-process check can see."""
        try:
            from . import crash_reporter
            reports = crash_reporter.recent(10)
        except Exception as exc:
            return Check("crashes", "WARN", first_line(exc, 60))
        cutoff = time.time() - self.CRASH_WINDOW_S
        serious = []
        for row in reports:
            try:
                if Path(row["path"]).stat().st_mtime < cutoff:
                    continue
            except (OSError, KeyError):
                continue
            if crash_reporter.interpret(row.get("kind", ""), row.get("summary", ""))["severe"]:
                serious.append(row)
        if serious:
            latest = serious[0]
            return Check("crashes", "WARN",
                         f"{len(serious)} serious crash report(s) this week — latest "
                         f"{latest['file']}: {first_line(latest.get('summary') or latest.get('why'), 110)}")
        return Check("crashes", "PASS", "no serious crash reports this week")

    def _check_compile(self) -> Check:
        failures: list[str] = []
        count = 0
        for path in PACKAGE_DIR.rglob("*.py"):
            count += 1
            try:
                # A diagnostic must also work in a read-only installation.  A
                # bytecode cache is not needed to establish that source parses.
                compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            except (OSError, SyntaxError, ValueError) as exc:
                failures.append(f"{path.name}: {first_line(exc, 60)}")
        if failures:
            return Check("compile", "FAIL", f"{len(failures)} file(s) fail to compile: "
                         + "; ".join(failures[:4]))
        if getattr(sys, "frozen", False):
            from .diagnostic_imports import inventory
            known = inventory(PACKAGE_DIR)
            coverage = (f"{len(known['modules'])} PYZ modules contain bundled bytecode"
                        if "modules" in known else known["error"])
            return Check("compile", "WARN", f"{count} loose source file(s) compile; "
                         f"{coverage}; full source compilation is unavailable in this frozen build")
        if not count:
            return Check("compile", "WARN", "0 Python source files found; compilation coverage is unavailable")
        return Check("compile", "PASS", f"{count} Python source file(s) compile without writing bytecode")

    def _check_imports(self) -> Check:
        from .diagnostic_imports import check
        try:
            result = check(PACKAGE_DIR)
        except Exception as exc:
            return Check("imports", "FAIL", "Isolated import diagnostic failed: " + first_line(exc, 160))
        return Check("imports", result["status"], result["detail"])

    def _check_dependencies(self) -> Check:
        missing_req: list[str] = []
        missing_opt: list[str] = []
        for module, required in DEPENDENCIES:
            if not _module_present(module):
                (missing_req if required else missing_opt).append(module)
        if missing_req:
            return Check("dependencies", "FAIL", "missing required: " + ", ".join(missing_req))
        if missing_opt:
            return Check("dependencies", "WARN", "optional not installed: " + ", ".join(missing_opt))
        return Check("dependencies", "PASS", "all required and optional packages present")

    def _check_database(self) -> Check:
        if not CORE_DB_PATH.exists():
            return Check("database", "WARN", "memory DB not created yet")
        try:
            conn = sqlite3.connect(str(CORE_DB_PATH))
            result = conn.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            if result and result[0] == "ok":
                return Check("database", "PASS", "SQLite integrity_check ok")
            return Check("database", "FAIL", f"integrity: {result}")
        except Exception as exc:
            return Check("database", "FAIL", first_line(exc, 60))

    # Directories under CONFIG_DIR that hold JSON belonging to something other
    # than ORION's own config — an embedded Chromium profile (browser_copilot)
    # ships files like FirstPartySetsPreloaded/sets.json, which are JSON LINES
    # (one object per line), not a single JSON document. Validating them as
    # ORION config produced a permanent, unfixable-looking "invalid JSON"
    # failure that had nothing to do with ORION's own settings.
    _CONFIG_SCAN_EXCLUDE_DIRS = frozenset({"browser_profile"})

    def _check_config(self) -> Check:
        bad: list[str] = []
        for path in CONFIG_DIR.rglob("*.json"):
            if self._CONFIG_SCAN_EXCLUDE_DIRS.intersection(path.relative_to(CONFIG_DIR).parts[:-1]):
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                bad.append(f"{path.name}: {first_line(exc, 40)}")
        if bad:
            return Check("config", "FAIL", "invalid JSON: " + "; ".join(bad[:3]))
        return Check("config", "PASS", "all config JSON valid")

    def _check_tools(self) -> Check:
        try:
            from .dispatcher import TOOL_DECLARATIONS
        except Exception as exc:
            return Check("tools", "FAIL", f"declarations unimportable: {first_line(exc, 50)}")
        declarations = (getattr(self.dispatcher, "TOOL_DECLARATIONS", TOOL_DECLARATIONS)
                        if self.dispatcher is not None else TOOL_DECLARATIONS)
        names = [d.get("name") for d in declarations]
        if any(not isinstance(name, str) or not name for name in names):
            return Check("tools", "FAIL", "tool declaration has an invalid name")
        dupes = sorted(name for name, count in Counter(names).items() if count > 1)
        if dupes:
            return Check("tools", "FAIL", f"duplicate tool names: {', '.join(dupes[:5])}")
        if self.dispatcher is not None:
            try:
                native = self.dispatcher.handler_table()
                forged = getattr(self.dispatcher, "_tool_handlers", {}) or {}
                mcp = (getattr(self.dispatcher, "_mcp_routes", {}) or {}
                       if getattr(self.dispatcher, "mcp_host", None) is not None else {})
                available = {name for routes in (native, forged) for name, handler in routes.items()
                             if callable(handler)} | set(mcp)
                missing = sorted(set(names) - available)
            except Exception as exc:
                return Check("tools", "FAIL", f"routing inspection failed: {first_line(exc, 60)}")
            if missing:
                return Check("tools", "FAIL", f"declared tools without a live route: {', '.join(missing[:5])}")
        return Check("tools", "PASS", f"{len(names)} tool declarations route cleanly")

    def _check_health(self) -> Check:
        if self.telemetry is None:
            return Check("health", "WARN", "telemetry not attached")
        try:
            rows = self.telemetry.health.snapshot()
        except Exception as exc:
            return Check("health", "WARN", first_line(exc, 50))
        down = [r["name"] for r in rows if r.get("status") in {"DOWN", "DEGRADED"}]
        if down:
            return Check("health", "WARN", "degraded components: " + ", ".join(down))
        return Check("health", "PASS", f"{len(rows)} component(s) healthy")

    def _check_resources(self) -> Check:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage(os.path.abspath(os.sep)).percent
        except Exception as exc:
            return Check("resources", "WARN", first_line(exc, 50))
        if disk >= 95 or ram >= 95:
            return Check("resources", "WARN", f"CPU {cpu:.0f}% RAM {ram:.0f}% disk {disk:.0f}% — tight")
        return Check("resources", "PASS", f"CPU {cpu:.0f}% RAM {ram:.0f}% disk {disk:.0f}%")

    def _check_permissions(self) -> Check:
        probe: str | None = None
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            # A fixed filename can be overwritten or deleted by another
            # diagnostic process, or collide with a file left after a crash.
            handle, probe = tempfile.mkstemp(prefix=".diag_write_test.", dir=CONFIG_DIR)
            with os.fdopen(handle, "wb") as stream:
                stream.write(b"ok")
            return Check("permissions", "PASS", "config directory is writable")
        except Exception as exc:
            return Check("permissions", "FAIL", f"config not writable: {first_line(exc, 50)}")
        finally:
            if probe is not None:
                try:
                    os.unlink(probe)
                except OSError:
                    pass
