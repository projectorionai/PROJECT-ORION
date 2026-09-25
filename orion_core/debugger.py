"""
debugger.py — a real interactive debugger, driven by stdlib pdb.

Mark XX design-spec asked for a debugger inside the Development workspace.
A full step-through IDE debugger (breakpoints painted in a gutter, a
variable-inspector tree, a call-stack view) is a multi-week undertaking on
its own; what's genuinely achievable here — and genuinely real, not a
placeholder — is driving Python's own `pdb` as a subprocess: it already
implements breakpoints, step/next/continue, frame navigation and
expression printing, needs no extra dependency, and is exactly what a
terminal-based "debug this script" session already looks like. This module
automates that session: it launches `python -m pdb <script>`, and lets a
caller send it commands and read back pdb's own responses.

DebugSession owns one subprocess and a background reader thread (pdb's
prompt has no trailing newline, and Windows pipes don't support `select`,
so a polling reader thread is the portable way to detect "is more output
coming or are we at the next prompt"). DebuggerService wraps at most one
active session, the same "single active thing" shape used elsewhere in
this codebase (one workflow run selected, one chess game).
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .bus import OrionBus
from .data import ToolResult

_PROMPT = "(Pdb) "
_READ_TIMEOUT = 10.0


class DebugSession:
    """One live `python -m pdb <script>` subprocess."""

    def __init__(self, script_path: str, args: list[str] | None = None) -> None:
        self.script_path = script_path
        self.args = list(args or [])
        self._proc: subprocess.Popen | None = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._reader: threading.Thread | None = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> ToolResult:
        if self.is_running():
            return ToolResult("A debug session is already running. Stop it first.", ok=False)
        path = Path(self.script_path)
        if not path.is_file():
            return ToolResult(f"No such file: {self.script_path}", ok=False)
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "pdb", str(path), *self.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(path.parent),
            )
        except OSError as exc:
            return ToolResult(f"Could not launch debugger: {exc}", ok=False)
        self._queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        return self._wait_for_prompt()

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for chunk in iter(lambda: proc.stdout.read(1), ""):
                self._queue.put(chunk)
        except (ValueError, OSError):
            pass

    def _wait_for_prompt(self, timeout: float = _READ_TIMEOUT) -> ToolResult:
        deadline = time.monotonic() + timeout
        buf = ""
        while time.monotonic() < deadline:
            try:
                buf += self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    break
                continue
            if buf.endswith(_PROMPT):
                return ToolResult(buf[: -len(_PROMPT)].strip())
        if self._proc is not None and self._proc.poll() is not None:
            text = buf.strip() or "Debug session ended."
            return ToolResult(f"{text}\n[process exited]")
        return ToolResult(
            f"{buf.strip() or '(no output)'}\n[timed out waiting for the debugger]", ok=False
        )

    def send(self, command: str) -> ToolResult:
        if not self.is_running():
            return ToolResult("No debug session is running.", ok=False)
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        try:
            proc.stdin.write(command.strip() + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            return ToolResult(f"Could not send command: {exc}", ok=False)
        return self._wait_for_prompt()

    def stop(self) -> ToolResult:
        if not self.is_running():
            self._proc = None
            return ToolResult("No debug session was running.")
        proc = self._proc
        assert proc is not None
        try:
            if proc.stdin is not None:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._proc = None
        return ToolResult("Debug session stopped.")


class DebuggerService:
    """The single active debug session a GUI panel or dispatcher tool drives."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._session: DebugSession | None = None

    def is_running(self) -> bool:
        return self._session is not None and self._session.is_running()

    def start(self, script_path: str, args: list[str] | None = None) -> ToolResult:
        if self.is_running():
            return ToolResult("A debug session is already running. Stop it first.", ok=False)
        session = DebugSession(script_path, args)
        result = session.start()
        if result.ok:
            self._session = session
            self.bus.log.emit(f"DEBUGGER: session started for {script_path}")
        else:
            self.bus.log.emit(f"DEBUGGER: failed to start - {result.text[:200]}")
        return result

    def command(self, text: str) -> ToolResult:
        if self._session is None:
            return ToolResult("No debug session. Start one first.", ok=False)
        return self._session.send(text)

    def stop(self) -> ToolResult:
        if self._session is None:
            return ToolResult("No debug session was running.")
        result = self._session.stop()
        self._session = None
        self.bus.log.emit("DEBUGGER: session stopped")
        return result
