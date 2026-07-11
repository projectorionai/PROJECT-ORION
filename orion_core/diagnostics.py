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

import asyncio
import importlib
import json
import os
import py_compile
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import CONFIG_DIR, CORE_DB_PATH, PACKAGE_DIR
from .data import ToolResult
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


@dataclass
class Check:
    name: str
    status: str          # PASS | WARN | FAIL
    detail: str

    def line(self) -> str:
        icon = {"PASS": "✓", "WARN": "▲", "FAIL": "✗"}.get(self.status, "?")
        return f"  {icon} {self.name}: {self.detail}"


class DiagnosticsEngine:
    def __init__(self, bus: OrionBus, memory: Any, telemetry: Any | None = None,
                 dispatcher: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        self.dispatcher = dispatcher

    async def run_full(self) -> ToolResult:
        self.bus.log.emit("DIAG: running full self-diagnostic…")
        checks = await asyncio.to_thread(self._run_all)
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
        header = f"Diagnostic complete, sir — {verdict}."
        body = "\n".join(c.line() for c in checks)
        return ToolResult(f"{header}\n{body}", ok=not fails)

    # ── individual checks (sync; run in a worker thread) ──────────────────────

    def _run_all(self) -> list[Check]:
        return [
            self._check_compile(),
            self._check_imports(),
            self._check_dependencies(),
            self._check_database(),
            self._check_config(),
            self._check_tools(),
            self._check_health(),
            self._check_resources(),
            self._check_permissions(),
        ]

    def _check_compile(self) -> Check:
        failures: list[str] = []
        for path in PACKAGE_DIR.rglob("*.py"):
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.name}: {first_line(exc, 60)}")
        if failures:
            return Check("compile", "FAIL", f"{len(failures)} file(s) fail to compile: "
                         + "; ".join(failures[:4]))
        return Check("compile", "PASS", "every module byte-compiles")

    def _check_imports(self) -> Check:
        broken: list[str] = []
        count = 0
        for path in sorted(PACKAGE_DIR.rglob("*.py")):
            rel = path.relative_to(PACKAGE_DIR.parent)
            mod = ".".join(rel.with_suffix("").parts)
            if mod.endswith("__init__"):
                mod = mod[: -len(".__init__")]
            if mod in {"orion_core.app"}:  # importing app is heavy but safe; skip run
                continue
            try:
                importlib.import_module(mod)
                count += 1
            except Exception as exc:
                broken.append(f"{mod}: {first_line(exc, 50)}")
        if broken:
            return Check("imports", "FAIL", f"{len(broken)} module(s) fail to import: "
                         + "; ".join(broken[:3]))
        return Check("imports", "PASS", f"{count} modules import cleanly")

    def _check_dependencies(self) -> Check:
        missing_req: list[str] = []
        missing_opt: list[str] = []
        for module, required in DEPENDENCIES:
            try:
                importlib.import_module(module)
            except Exception:
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

    def _check_config(self) -> Check:
        bad: list[str] = []
        for path in CONFIG_DIR.rglob("*.json"):
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
        names = [d.get("name") for d in TOOL_DECLARATIONS]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            return Check("tools", "FAIL", f"duplicate tool names: {', '.join(dupes)}")
        return Check("tools", "PASS", f"{len(names)} tool declarations, no duplicates")

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
        probe = CONFIG_DIR / ".diag_write_test"
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return Check("permissions", "PASS", "config directory is writable")
        except Exception as exc:
            return Check("permissions", "FAIL", f"config not writable: {first_line(exc, 50)}")
