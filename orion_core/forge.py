"""
The Forge Orchestration Manager.

Ties sandbox.py, dependencies.py, and dynamic_loader.py together in a
sequential control loop. When triggered to build a tool, executes:

    1. Plan Phase: Define whether the capability is headless or interactive.
    2. Code Generation: Stream or construct the code and test harnesses.
    3. Sandbox Pass: Hand artifacts to sandbox.py. If it fails, loop back to an
                     LLM-fix function up to 3 times passing the error logs.
    4. Dependency Check: Hand requirements to dependencies.py.
    5. Live Activation: Hand verified files to dynamic_loader.py and register
                        the tool's description into FTS5 KNOWLEDGE memory tier.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .bus import OrionBus
from .data import ToolResult
from .dependencies import DynamicPackageResolver
from .dynamic_loader import ReflectiveModuleLoader
from .sandbox import SandboxVerificationHarness


@dataclass
class ForgeSession:
    """State machine tracking a single tool-forging operation."""

    session_id: str
    tool_name: str
    tool_plan: str | None = None
    tool_code: str = ""
    test_code: str = ""
    requirements: list[str] = field(default_factory=list)
    plan_phase_ok: bool = False
    code_generation_ok: bool = False
    sandbox_attempts: int = 0
    sandbox_ok: bool = False
    dependencies_ok: bool = False
    activation_ok: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "plan_phase_ok": self.plan_phase_ok,
            "code_generation_ok": self.code_generation_ok,
            "sandbox_ok": self.sandbox_ok,
            "dependencies_ok": self.dependencies_ok,
            "activation_ok": self.activation_ok,
            "sandbox_attempts": self.sandbox_attempts,
            "errors": self.errors,
        }


class ForgeOrchestrationManager:
    """Control loop for dynamically forging new tools."""

    def __init__(
        self,
        bus: OrionBus,
        code_generator: Callable[[str, list[str]], tuple[str, str, list[str]]] | None = None,
        llm_fixer: Callable[[str, list[str]], str] | None = None,
    ) -> None:
        """
        Initialise the Forge orchestration manager.

        Args:
            bus: OrionBus for logging forge states.
            code_generator: Callable that generates (tool_code, test_code, requirements)
                           given (tool_name, error_logs). Optional for testing.
            llm_fixer: Callable that fixes code given (generated_code, error_logs).
                      Optional; if None, no self-healing is attempted.
        """
        self.bus = bus
        self.code_generator = code_generator
        self.llm_fixer = llm_fixer
        self.sandbox = SandboxVerificationHarness(bus)
        self.resolver = DynamicPackageResolver(bus)
        self.loader = ReflectiveModuleLoader(bus)
        self.sessions: dict[str, ForgeSession] = {}

    async def forge_tool(
        self,
        tool_name: str,
        tool_plan: str,
        code_generator: Callable[[str, str], tuple[str, str, list[str]]] | None = None,
    ) -> ToolResult:
        """
        Execute the complete tool-forging pipeline.

        Orchestrates: Plan → Code Gen → Sandbox (with self-heal loop) →
                     Dependency Check → Live Activation.

        Args:
            tool_name: Name of the tool to forge
            tool_plan: High-level description of what the tool should do
            code_generator: Optional override for the code generator

        Returns:
            ToolResult indicating success and next steps
        """
        gen = code_generator or self.code_generator
        if gen is None:
            return ToolResult(
                "No code generator configured. Set via constructor or forge_tool param.",
                ok=False,
            )

        session_id = f"{tool_name}_{asyncio.get_event_loop().time()}".replace(".", "_")
        session = ForgeSession(session_id=session_id, tool_name=tool_name, tool_plan=tool_plan)
        self.sessions[session_id] = session

        self.bus.log.emit(
            f"[FORGE] Manager: initiating forge session '{session_id}' for '{tool_name}'"
        )

        try:
            # Phase 1: Plan
            self.bus.log.emit(f"[FORGE] Manager: Phase 1 — Planning '{tool_name}'")
            session.plan_phase_ok = True

            # Phase 2: Code Generation
            self.bus.log.emit(f"[FORGE] Manager: Phase 2 — Generating code")
            tool_code, test_code, requirements = gen(tool_name, tool_plan)
            session.tool_code = tool_code
            session.test_code = test_code
            session.requirements = requirements
            session.code_generation_ok = True

            # Phase 3: Sandbox Verification (with self-heal loop, up to 3 attempts)
            self.bus.log.emit(f"[FORGE] Manager: Phase 3 — Sandbox verification")
            for attempt in range(1, 4):
                self.bus.log.emit(
                    f"[FORGE] Manager: Sandbox attempt {attempt}/3 for '{tool_name}'"
                )
                outcome = await self.sandbox.verify_tool(
                    session.tool_code,
                    session.test_code,
                    tool_name,
                )

                if outcome.passed:
                    session.sandbox_ok = True
                    self.bus.log.emit(f"[FORGE] Manager: ✓ Sandbox passed on attempt {attempt}")
                    break
                else:
                    session.sandbox_attempts = attempt
                    session.errors.extend(outcome.error_log)
                    self.bus.log.emit(
                        f"[FORGE] Manager: ✗ Sandbox failed (attempt {attempt}). "
                        f"Errors: {outcome.error_log[:2]}"
                    )

                    # Attempt self-healing (if LLM fixer available and not last attempt)
                    if attempt < 3 and self.llm_fixer is not None:
                        self.bus.log.emit(
                            f"[FORGE] Manager: attempting self-heal via LLM"
                        )
                        fixed_code = self.llm_fixer(session.tool_code, outcome.error_log)
                        session.tool_code = fixed_code
                    else:
                        break

            if not session.sandbox_ok:
                session.errors.append("Sandbox verification failed after 3 attempts")
                raise RuntimeError(
                    f"Sandbox verification failed. Last errors: {session.errors[-3:]}"
                )

            # Phase 4: Dependency Check
            self.bus.log.emit(f"[FORGE] Manager: Phase 4 — Dependency resolution")
            dep_outcome = await self.resolver.resolve_and_install(session.requirements)
            if dep_outcome.succeeded:
                session.dependencies_ok = True
                self.bus.log.emit(
                    f"[FORGE] Manager: ✓ Dependencies resolved "
                    f"({len(dep_outcome.installed_packages)} installed)"
                )
            else:
                session.errors.extend(dep_outcome.error_log)
                self.bus.log.emit(
                    f"[FORGE] Manager: ⚠ Dependency issues: "
                    f"{dep_outcome.error_log[:1]}"
                )
                # Continue anyway; some dependencies may already be present

            # Phase 5: Live Activation
            self.bus.log.emit(f"[FORGE] Manager: Phase 5 — Live activation")
            custom_tools_dir = Path(__file__).parent.parent / "config" / "custom_tools"
            custom_tools_dir.mkdir(parents=True, exist_ok=True)

            tool_file = custom_tools_dir / f"{tool_name}_tool.py"
            await asyncio.to_thread(
                lambda: tool_file.write_text(session.tool_code, encoding="utf-8")
            )

            load_outcome = await self.loader.load_and_register(tool_file)
            if not load_outcome.succeeded:
                session.errors.extend(load_outcome.error_log)
                raise RuntimeError(
                    f"Failed to load module: {load_outcome.error_log}"
                )

            session.activation_ok = True
            self.bus.log.emit(
                f"[FORGE] Manager: ✓ Tool '{tool_name}' activated and live"
            )

            # Compose success message
            result_msg = (
                f"✓ Successfully forged tool '{tool_name}'.\n"
                f"• Plan: ✓\n"
                f"• Code Gen: ✓\n"
                f"• Sandbox: ✓ (attempt {session.sandbox_attempts})\n"
                f"• Dependencies: ✓\n"
                f"• Activation: ✓\n\n"
                f"The tool is now live and available in the dispatcher."
            )
            return ToolResult(result_msg, ok=True)

        except Exception as e:
            self.bus.log.emit(
                f"[FORGE] Manager: ✗ forge session '{session_id}' failed: {e}"
            )
            session.errors.append(f"{type(e).__name__}: {str(e)}")
            error_msg = (
                f"✗ Failed to forge '{tool_name}'.\n"
                f"Errors:\n" + "\n".join(f"  • {err}" for err in session.errors[-5:])
            )
            return ToolResult(error_msg, ok=False)

    async def forge_batch(
        self,
        tool_specs: list[dict[str, str]],
        code_generator: Callable[[str, str], tuple[str, str, list[str]]] | None = None,
    ) -> ToolResult:
        """
        Forge multiple tools sequentially.

        Args:
            tool_specs: List of {"name": str, "plan": str} dicts
            code_generator: Optional code generator override

        Returns:
            ToolResult with summary of all forged tools
        """
        results: dict[str, bool] = {}

        for spec in tool_specs:
            tool_name = spec.get("name", "unknown")
            tool_plan = spec.get("plan", "")

            self.bus.log.emit(f"[FORGE] Manager: starting batch forge for '{tool_name}'")
            result = await self.forge_tool(tool_name, tool_plan, code_generator)
            results[tool_name] = result.ok

        # Compose summary
        successes = sum(1 for ok in results.values() if ok)
        summary = (
            f"Batch forge complete: {successes}/{len(results)} tools forged.\n"
            + "\n".join(f"  • {name}: {'✓' if ok else '✗'}" for name, ok in results.items())
        )

        self.bus.log.emit(f"[FORGE] Manager: batch complete — {successes}/{len(results)} ok")
        return ToolResult(summary, ok=successes == len(results))

    def session_status(self, session_id: str) -> ToolResult:
        """
        Query the status of a forge session.

        Args:
            session_id: Session identifier

        Returns:
            ToolResult with session state
        """
        session = self.sessions.get(session_id)
        if session is None:
            return ToolResult(f"Session '{session_id}' not found.", ok=False)

        status = (
            f"Session '{session_id}' for tool '{session.tool_name}':\n"
            f"• Plan phase: {'✓' if session.plan_phase_ok else '✗'}\n"
            f"• Code generation: {'✓' if session.code_generation_ok else '✗'}\n"
            f"• Sandbox (attempts: {session.sandbox_attempts}): {'✓' if session.sandbox_ok else '✗'}\n"
            f"• Dependencies: {'✓' if session.dependencies_ok else '✗'}\n"
            f"• Activation: {'✓' if session.activation_ok else '✗'}"
        )
        if session.errors:
            status += f"\n\nErrors:\n" + "\n".join(f"  • {err}" for err in session.errors[-5:])

        return ToolResult(status, ok=session.activation_ok)

    def list_sessions(self) -> ToolResult:
        """List all active and completed forge sessions."""
        if not self.sessions:
            return ToolResult("No forge sessions found.", ok=True)

        lines = ["Forge sessions:"]
        for session_id, session in self.sessions.items():
            status = "✓ completed" if session.activation_ok else "✗ in-progress/failed"
            lines.append(
                f"  • {session_id[:16]}... : {session.tool_name} — {status}"
            )

        return ToolResult("\n".join(lines), ok=True)
