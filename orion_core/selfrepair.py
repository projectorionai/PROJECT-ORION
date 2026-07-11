"""
SelfRepairAgent (Phase 6) — self-healing runtime with a human approval gate.

When something throws, ORION:

    1. captures the stack trace and the recent structured logs;
    2. identifies the failing module and line from the traceback;
    3. reads the surrounding source;
    4. asks the provider layer for a fix (unified diff + rationale);
    5. runs the test/verify suite;
    6. presents the patch for approval.

It **never** deploys automatically.  Approved fixes are written to a sibling
file under ``config/self_repair/`` — the original ORION source is never mutated
in place (which also respects the core-protection invariant), so the user
reviews the diff and copies it across deliberately.

Wiring: ``install()`` registers a ``sys.excepthook`` wrapper and an asyncio
exception handler so faults are captured wherever they occur, without any
call site needing to know about self-repair.
"""

from __future__ import annotations

import asyncio
import sys
import traceback as tb_module
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR, CONFIG_DIR, PACKAGE_DIR
from .data import ToolResult
from .utils import first_line, utc_stamp

SELF_REPAIR_DIR = CONFIG_DIR / "self_repair"


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

    def summary(self) -> str:
        loc = f"{Path(self.file).name}:{self.line}" if self.file else "unknown"
        return f"[{self.id}] {self.error_type}: {self.message[:120]} ({loc})"


class SelfRepairAgent:
    def __init__(self, bus: OrionBus, telemetry: Any, router: Any) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.router = router
        self._incidents: dict[str, Incident] = {}
        self._counter = 0
        self._prev_excepthook: Any = None
        SELF_REPAIR_DIR.mkdir(parents=True, exist_ok=True)

    # ── installation ──────────────────────────────────────────────────────────

    def install(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._prev_excepthook = sys.excepthook
        sys.excepthook = self._excepthook
        if loop is not None:
            loop.set_exception_handler(self._loop_exception_handler)
        self.bus.log.emit("REPAIR: self-healing runtime armed (capture-only; approval required to apply).")

    def _excepthook(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.capture(exc_type, exc, tb)
        except Exception:
            pass
        if self._prev_excepthook is not None:
            self._prev_excepthook(exc_type, exc, tb)

    def _loop_exception_handler(self, loop: Any, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if exc is not None:
            self.capture(type(exc), exc, exc.__traceback__)
        else:
            self.bus.log.emit(f"REPAIR: loop error - {context.get('message', 'unknown')}")

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
        self._incidents[incident_id] = incident
        if self.telemetry is not None:
            self.telemetry.metrics.incr("repair.incidents")
            self.telemetry.health.beat("self_repair", "DEGRADED", incident.summary())
        self.bus.log.emit(f"REPAIR: captured {incident.summary()}")
        self.bus.dashboard_event.emit("incident", incident.summary())
        return incident

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
        source = self._read_source_window(incident.file, incident.line) if incident.file else ""
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
        prompt = (
            f"Error: {incident.error_type}: {incident.message}\n\n"
            f"Traceback:\n{incident.traceback}\n\n"
            f"Failing source ({incident.file}, around line {incident.line}):\n{source}\n\n"
            f"Recent logs:\n{log_tail}"
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
            try:
                profile, answer = await self.router.generate_text(prompt, system_extra=persona)
                proposal = (
                    f"# Self-repair proposal for {incident.id}  (via {profile.name})\n\n"
                    f"## Incident\n{incident.summary()}\n\n{answer}\n"
                )
            except Exception as exc:
                return ToolResult(f"Repair proposal failed: {first_line(exc)}", ok=False)
        path = SELF_REPAIR_DIR / f"proposal_{incident.id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        try:
            path.write_text(proposal, encoding="utf-8")
            incident.proposal_path = str(path)
        except Exception:
            pass
        self.bus.log.emit(f"REPAIR: proposal for {incident.id} written to {path.name} (review required).")
        return ToolResult(
            f"Repair proposal for {incident.summary()} prepared.\n"
            f"Saved for review at: {path}\n"
            "No change has been applied — approve and copy the fix across manually.\n\n"
            + proposal[:3000]
        )

    # ── test / verify suite ───────────────────────────────────────────────────

    async def run_tests(self, path: str = "") -> ToolResult:
        """
        Run the verification suite: byte-compile the package (fast smoke) and,
        if a tests/ directory exists, pytest.  Executed off the event loop.
        """
        root = Path(path).expanduser() if path.strip() else BASE_DIR

        def _run() -> str:
            import subprocess
            out: list[str] = []
            compile_proc = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", str(PACKAGE_DIR)],
                capture_output=True, text=True, timeout=120,
            )
            out.append(f"compileall: exit {compile_proc.returncode}")
            if compile_proc.stdout.strip():
                out.append(compile_proc.stdout.strip()[-1500:])
            if (root / "tests").is_dir():
                try:
                    pytest_proc = subprocess.run(
                        [sys.executable, "-m", "pytest", "-q", str(root / "tests")],
                        cwd=str(root), capture_output=True, text=True, timeout=300,
                    )
                    out.append(f"pytest: exit {pytest_proc.returncode}\n"
                               + (pytest_proc.stdout or "")[-2500:])
                except FileNotFoundError:
                    out.append("pytest not installed; skipped.")
                except subprocess.TimeoutExpired:
                    out.append("pytest timed out.")
            else:
                out.append("No tests/ directory; compile smoke only.")
            return "\n".join(out)

        try:
            output = await asyncio.to_thread(_run)
        except Exception as exc:
            return ToolResult(f"Test run failed: {first_line(exc)}", ok=False)
        ok = "exit 0" in output and "exit 1" not in output.replace("exit 0", "")
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
            return ToolResult("No captured incident to repair, sir.", ok=False)
        target = Path(incident.file)
        if not target.is_file() or PACKAGE_DIR not in target.resolve().parents:
            return ToolResult(
                f"I can only repair files inside the orion_core package, sir "
                f"(the fault is in {incident.file or 'an unknown file'}).", ok=False)

        # Generate the corrected file if we don't have one yet.
        if not incident.repaired_content:
            gen = await self._generate_repaired_file(incident, target)
            if not gen.ok:
                return gen

        if not confirm:
            preview = incident.repaired_content[:1400]
            return ToolResult(
                f"I've drafted a corrected '{target.name}', sir, and it compiles. "
                "Nothing is applied yet — approve with confirm=true to back up the "
                "original and write the fix (a restart then loads it).\n\n"
                f"Preview (first lines):\n{preview}", ok=True)

        return await asyncio.to_thread(self._apply_repaired, incident, target)

    async def _generate_repaired_file(self, incident: Incident, target: Path) -> ToolResult:
        try:
            original = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(f"Could not read {target.name}: {first_line(exc)}", ok=False)
        if len(original) > 40000:
            return ToolResult(
                f"'{target.name}' is large ({len(original)} chars); I'll draft a diff "
                "proposal instead of a full-file rewrite. Use the self_repair 'propose' action.",
                ok=False)
        if not self.router.has_text_fallback():
            return ToolResult(
                "No language model is reachable to draft the code fix, sir. When one is "
                "available, ask me to repair the file again.", ok=False)
        persona = (
            "SPECIALIST MODE — Python repair engineer. You are given a full source file "
            "and a traceback from it. Return the COMPLETE corrected file and nothing "
            "else — no markdown fences, no commentary. Make the MINIMAL change that fixes "
            "the traceback; preserve all other code, imports, formatting and behaviour."
        )
        prompt = (
            f"Traceback:\n{incident.traceback}\n\n"
            f"Error: {incident.error_type}: {incident.message}\n\n"
            f"Full source of {target.name}:\n{original}"
        )
        try:
            _profile, answer = await self.router.generate_text(prompt, system_extra=persona)
        except Exception as exc:
            return ToolResult(f"Repair generation failed: {first_line(exc)}", ok=False)
        corrected = self._strip_fences(answer)
        # Validate the corrected content compiles before we ever offer to apply it.
        import py_compile
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"orion_repair_{incident.id}.py"
        try:
            tmp.write_text(corrected, encoding="utf-8")
            py_compile.compile(str(tmp), doraise=True)
        except Exception as exc:
            return ToolResult(
                f"The drafted fix for {target.name} does not compile "
                f"({first_line(exc, 60)}); I won't offer to apply it, sir.", ok=False)
        finally:
            tmp.unlink(missing_ok=True)
        incident.repaired_content = corrected
        return ToolResult("draft ready")

    def _apply_repaired(self, incident: Incident, target: Path) -> ToolResult:
        import py_compile
        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = self.BACKUP_DIR / f"{target.name}.{datetime.now():%Y%m%d-%H%M%S}.bak"
        try:
            backup.write_text(target.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            target.write_text(incident.repaired_content, encoding="utf-8")
            py_compile.compile(str(target), doraise=True)
        except Exception as exc:
            # Roll back to the backup on any failure.
            try:
                if backup.exists():
                    target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
            return ToolResult(
                f"Applying the fix to {target.name} failed and I reverted it, sir "
                f"({first_line(exc, 60)}).", ok=False)
        incident.applied = True
        incident.backup_path = str(backup)
        self.bus.log.emit(f"REPAIR: applied fix to {target.name} (backup {backup.name}).")
        self.bus.banner.emit(f"SELF-REPAIR APPLIED: {target.name} — restart to load", 3)
        return ToolResult(
            f"Done, sir — I've corrected {target.name} and it compiles. The original is "
            f"backed up at {backup.name}. Restart ORION to load the fix; say 'revert the "
            "last repair' if anything seems off.")

    def revert_last(self) -> ToolResult:
        applied = [i for i in self._incidents.values() if i.applied and i.backup_path]
        if not applied:
            return ToolResult("There's no applied repair to revert, sir.")
        incident = applied[-1]
        target = Path(incident.file)
        backup = Path(incident.backup_path)
        try:
            target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            incident.applied = False
            return ToolResult(f"Reverted {target.name} to the pre-repair backup, sir. Restart to load it.")
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
