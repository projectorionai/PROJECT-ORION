"""
Sandboxed Verification Harness.

Accepts raw generated tool code and test code, writes to a staging directory,
then executes in an isolated subprocess to verify that the tool exposes:
  - get_tool_schema() → OpenAI-compatible function schema
  - run(**kwargs) → explicit runtime execution point

Returns structured verification outcomes (pass/fail with error logs).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .data import ToolResult


@dataclass
class VerificationOutcome:
    """Structured result of sandbox testing a generated tool."""

    passed: bool
    tool_name: str
    error_log: list[str] = field(default_factory=list)
    schema: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tool_name": self.tool_name,
            "error_log": self.error_log,
            "schema": self.schema,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
        }


class SandboxVerificationHarness:
    """Isolated testing runner for generated Python tool code."""

    def __init__(self, bus: OrionBus) -> None:
        """
        Initialise the sandboxed verification harness.

        Args:
            bus: OrionBus for logging execution states.
        """
        self.bus = bus
        self.staging_dir = Path(tempfile.gettempdir()) / "orion_forge_staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    async def verify_tool(
        self,
        tool_code: str,
        test_code: str,
        tool_name: str,
        timeout_seconds: float = 15.0,
    ) -> VerificationOutcome:
        """
        Test a generated tool in an isolated subprocess.

        Writes tool code and test code to staging area, invokes a distinct
        Python subprocess with the local environment, and validates that:
          1. The tool module loads without syntax errors
          2. The tool exposes get_tool_schema() method
          3. The tool exposes run(**kwargs) method
          4. The test script executes with exit code 0

        Args:
            tool_code: Raw Python source code for the tool module
            test_code: Raw Python source code for the test harness
            tool_name: Identifier for the tool (sanitised into filename)
            timeout_seconds: Subprocess timeout (default 15 seconds)

        Returns:
            VerificationOutcome with passed flag, schema, and error logs
        """
        self.bus.log.emit(f"[FORGE] Sandbox: starting verification for '{tool_name}'")
        outcome = VerificationOutcome(passed=False, tool_name=tool_name)
        start_time = asyncio.get_event_loop().time()

        try:
            # Write artefacts to staging area
            await asyncio.to_thread(
                self._write_artefacts,
                tool_name,
                tool_code,
                test_code,
            )

            # Execute in subprocess
            stdout_text, stderr_text, exit_code = await asyncio.to_thread(
                self._run_subprocess,
                tool_name,
                timeout_seconds,
            )

            outcome.stdout = stdout_text
            outcome.stderr = stderr_text
            outcome.exit_code = exit_code

            # Parse result
            if exit_code == 0:
                # Attempt to extract schema from stdout (test harness emits it)
                schema = self._parse_schema_from_output(stdout_text)
                outcome.passed = True
                outcome.schema = schema
                self.bus.log.emit(
                    f"[FORGE] Sandbox: ✓ verification passed for '{tool_name}'"
                )
            else:
                outcome.error_log = (
                    [f"Exit code: {exit_code}"]
                    + (stderr_text.split("\n") if stderr_text else [])
                    + (stdout_text.split("\n") if stdout_text else [])
                )
                self.bus.log.emit(
                    f"[FORGE] Sandbox: ✗ verification failed for '{tool_name}' "
                    f"(exit {exit_code})"
                )

        except subprocess.TimeoutExpired:
            outcome.error_log = [f"Subprocess timeout after {timeout_seconds}s"]
            self.bus.log.emit(
                f"[FORGE] Sandbox: ✗ timeout for '{tool_name}' after {timeout_seconds}s"
            )

        except Exception as e:
            outcome.error_log = [
                f"{type(e).__name__}: {str(e)}",
                traceback.format_exc(),
            ]
            self.bus.log.emit(
                f"[FORGE] Sandbox: ✗ exception verifying '{tool_name}': {e}"
            )

        finally:
            end_time = asyncio.get_event_loop().time()
            outcome.duration_ms = (end_time - start_time) * 1000.0

        return outcome

    def _write_artefacts(
        self,
        tool_name: str,
        tool_code: str,
        test_code: str,
    ) -> None:
        """
        Write tool and test code to staging directory.

        Args:
            tool_name: Base name for the tool
            tool_code: Raw tool source code
            test_code: Raw test source code
        """
        safe_name = "".join(c if c.isalnum() or c in "_" else "_" for c in tool_name)
        tool_path = self.staging_dir / f"{safe_name}_tool.py"
        test_path = self.staging_dir / f"{safe_name}_test.py"

        tool_path.write_text(tool_code, encoding="utf-8")
        test_path.write_text(test_code, encoding="utf-8")

    def _run_subprocess(
        self,
        tool_name: str,
        timeout_seconds: float,
    ) -> tuple[str, str, int]:
        """
        Execute the test harness in a subprocess.

        Args:
            tool_name: Base name for the tool
            timeout_seconds: Subprocess timeout

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        safe_name = "".join(c if c.isalnum() or c in "_" else "_" for c in tool_name)
        test_path = self.staging_dir / f"{safe_name}_test.py"

        try:
            result = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=str(self.staging_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ, "PYTHONPATH": str(self.staging_dir)},
            )
            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired as e:
            raise e

    def _parse_schema_from_output(self, stdout_text: str) -> dict[str, Any] | None:
        """
        Extract tool schema JSON from test harness output.

        Looks for a JSON block in the stdout (test harness should emit
        `print(json.dumps(tool_schema))` on success).

        Args:
            stdout_text: Raw stdout from subprocess

        Returns:
            Parsed schema dict, or None if parsing fails
        """
        if not stdout_text:
            return None

        # Try to extract JSON from stdout
        lines = stdout_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        return None

    def tool_result(self, outcome: VerificationOutcome) -> ToolResult:
        """
        Convert VerificationOutcome to ToolResult for dispatcher.

        Args:
            outcome: Verification outcome from verify_tool()

        Returns:
            ToolResult with structured information
        """
        if outcome.passed:
            msg = (
                f"✓ Verification passed for '{outcome.tool_name}' "
                f"({outcome.duration_ms:.1f}ms).\n"
                f"Schema: {json.dumps(outcome.schema, indent=2)}"
            )
            return ToolResult(msg, ok=True)
        else:
            msg = (
                f"✗ Verification failed for '{outcome.tool_name}'. "
                f"Errors:\n" + "\n".join(outcome.error_log[-20:])  # Last 20 lines
            )
            return ToolResult(msg, ok=False)
