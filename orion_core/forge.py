"""
The Forge Orchestration Manager (Mark II).

Ties sandbox.py, dependencies.py, and dynamic_loader.py together in a
control loop. When triggered to build a tool, executes:

    1. Plan Phase: Define whether the capability is headless or interactive.
    2. Code Generation: Construct the module and its test harness, informed by
                        what previous forge failures taught (forge_lessons).
    3. Verify Pass: Hand artefacts to sandbox.py, which checks the tool
                    CONTRACT before running the model's own test. A failure is
                    classified by forge_diagnosis into a failure class and a
                    repair target, and the repair rewrites whichever artefact
                    is actually at fault — module, test, or both.
    4. Dependency Check: Hand requirements to dependencies.py.
    5. Live Activation: Hand verified files to dynamic_loader.py and register
                        the tool's description into FTS5 KNOWLEDGE memory tier.

The Mark II changes exist because the Mark I loop failed for structural
reasons rather than unlucky ones:

  • It only ever repaired the MODULE. When the generated *test* was wrong —
    calling run() with undeclared arguments, asserting an exact string, needing
    the network — the model rewrote sound code until the attempts ran out and
    the session died. Repairs are now aimed by diagnosis.
  • Every failure got the same generic "fix this" prompt, so the model had to
    re-derive the root cause from a wall of stderr each time. Each failure class
    now carries its own corrective directive.
  • Nothing was remembered between sessions, so identical mistakes recurred
    forever. Failures now accumulate in a lesson corpus that is injected into
    the next generation prompt.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .journal_io import tail_lines
from .dependencies import DynamicPackageResolver
from .dynamic_loader import ReflectiveModuleLoader, shadowed_builtin
from .forge_artefacts import ForgeArtefacts, extract_artefacts, synthesise_test
from .forge_diagnosis import (
    Diagnosis,
    FailureClass,
    RepairTarget,
    diagnose,
    failing_artefact,
    parse_missing_module,
)
from .forge_autofix import (
    add_argument_coercion,
    add_missing_imports,
    autofix_contract,
    check_test_policy,
    reconcile_schema,
)
from .forge_fixtures import FIXTURE_GUIDE, requirement_identity, validate_plan
from .forge_lessons import ForgeLessonStore
from .providers import NoTextProviderError, ProviderError
from .sandbox import SandboxVerificationHarness
from .atomic_io import atomic_write_bytes
from .utils import canonical_module_id, first_line, utc_stamp

# How many CODE repair attempts a session may spend. Dependency installs and
# other environmental repairs are free — they do not consume one of these.
# Three was too few once repairs became targeted: a session that correctly
# identifies a test defect on attempt two should still have room to fix a
# genuine module bug behind it.
DEFAULT_MAX_ATTEMPTS = 5
MAX_DEPENDENCY_REPAIRS = 4


def _max_attempts() -> int:
    try:
        value = int(os.getenv("ORION_FORGE_ATTEMPTS", "").strip() or DEFAULT_MAX_ATTEMPTS)
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS
    return max(1, min(10, value))


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
    # Mark II: the classified story of what went wrong and what was repaired,
    # so a failed session explains itself instead of dumping raw stderr.
    diagnoses: list[str] = field(default_factory=list)
    # How the model's artefacts were recovered ("blocks", "json", "bare") —
    # a drift back to the fragile JSON path is visible rather than silent.
    transport: str = ""
    review_candidate: str = ""
    plan_contract: dict[str, Any] | None = None
    examples_verified: bool = False

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
            "diagnoses": self.diagnoses,
            "transport": self.transport,
            "plan": copy.deepcopy(self.plan_contract),
            "examples_verified": self.examples_verified,
            "review_candidate": self.review_candidate,
        }


def forge_tool_name(raw: str) -> str:
    """A forged tool's name, safe as a file name ("<name>_tool.py"): lower
    case, [a-z0-9_] only, at most 60 characters, and without a trailing
    "_tool" - the file name adds that, and a name that already ended in it
    became resource_optimizer_tool_tool.py (owner's plugin list, 26 Oct 2026)."""
    name = re.sub(r"[^a-z0-9_]+", "_", str(raw or "").strip().lower()).strip("_")[:60]
    while name.endswith("_tool") and len(name) > len("_tool"):
        name = name[: -len("_tool")].rstrip("_")
    return "" if name == "tool" else name


def _custom_tools_dir() -> Path:
    """Where forged tools live: the same folder the plugin registry reads.

    The module's CONFIG_DIR (not Path(__file__)/../config): identical from a
    source checkout, but in the frozen app __file__ is inside the bundle, so
    forged tools were written into a folder every .exe rebuild replaces."""
    return CONFIG_DIR / "custom_tools"


class ForgeOrchestrationManager:
    """Control loop for dynamically forging new tools."""

    def __init__(
        self,
        bus: OrionBus,
        code_generator: Callable[[str, list[str]], tuple[str, str, list[str]]] | None = None,
        llm_fixer: Callable[[str, list[str]], str] | None = None,
        llm_repairer: Callable[..., Any] | None = None,
        lessons: ForgeLessonStore | None = None,
        history: Any = None,
    ) -> None:
        """
        Initialise the Forge orchestration manager.

        Args:
            bus: OrionBus for logging forge states.
            code_generator: Callable that generates (tool_code, test_code, requirements)
                           given (tool_name, tool_plan). Optional for testing.
            llm_fixer: Legacy module-only fixer, called as (tool_code, error_logs)
                      and returning corrected module source. Retained for
                      compatibility; ``llm_repairer`` supersedes it.
            llm_repairer: Targeted repairer called as
                      (tool_code, test_code, diagnosis, error_log) and returning
                      ``ForgeArtefacts`` (or a (tool_code, test_code) pair). This
                      is the one that can fix a broken TEST, which the legacy
                      fixer structurally could not.
            lessons: Optional persistent lesson store. When supplied, failures
                     are recorded and fed back into later generations; when None
                     the Forge simply does not learn (used by tests so a run
                     never writes into the user's config directory).
        """
        self.bus = bus
        self.code_generator = code_generator
        self.llm_fixer = llm_fixer
        self.llm_repairer = llm_repairer
        self.lessons = lessons
        self.history = history
        self.max_attempts = _max_attempts()
        self.sandbox = SandboxVerificationHarness(bus)
        self.resolver = DynamicPackageResolver(bus)
        self.loader = ReflectiveModuleLoader(bus)
        # The loader refuses, before import, a plugin that duplicates one of
        # these built-in tools (read from the dispatcher once it is attached).
        self.loader.native_names = self._native_tool_names
        self.sessions: dict[str, ForgeSession] = {}
        # Set at the composition root so a verified tool is registered into
        # the LIVE dispatcher immediately (no restart needed).
        self.dispatcher: Any = None
        # Optional SelfRepairAgent: a failed forge session is recorded as a
        # capture-only incident so diagnostics and the self_repair tool can
        # actually SEE it (previously "run a diagnostic" straight after a
        # forge failure reported nothing wrong).
        self.repair: Any = None
        # Sessions that failed ONLY because no text provider was reachable
        # (every remaining cloud provider rate-limited/out of credit, no
        # local model running) — not a code-generation defect, so the plan
        # deserves another go once a provider is plausibly back, rather than
        # staying dead until something happens to re-propose the identical
        # tool from scratch.  Drained by ImprovementHeartbeat.tick().
        self.pending_retries: list[tuple[str, str]] = []

    @staticmethod
    async def _call(fn: Callable[..., Any], *args: Any) -> Any:
        """Invoke a generator/fixer that may be sync or async."""
        result = fn(*args) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn, *args)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _improvement_engine(self):
        """Reuse measured history without changing ordinary Forge policy."""
        from .forge_improvement import ForgeImprovementEngine
        engine = self.__dict__.get("_measured_improvement")
        if engine is None:
            engine = ForgeImprovementEngine(
                self, CONFIG_DIR / "self_improvement" / "benchmarks", _custom_tools_dir())
            self._measured_improvement = engine
        return engine

    async def improve_tool(self, tool_name: str, benchmark: str, *,
                           generations: int = 3, budget_seconds: int = 300,
                           tool_plan: str = "") -> ToolResult:
        """Compare bounded generations with independent, fixed benchmarks."""
        return await self._improvement_engine().run(
            tool_name, benchmark, generations=generations,
            budget_seconds=budget_seconds, plan=tool_plan)

    async def improvement_status(self, run_id: str = "") -> ToolResult:
        return await self._improvement_engine().status(run_id)

    def improvement_benchmarks(self) -> ToolResult:
        return self._improvement_engine().catalogue()

    def _native_tool_names(self) -> set[str]:
        """The dispatcher's built-in tool names (empty before it is attached)."""
        table = getattr(self.dispatcher, "handler_table", None)
        if not callable(table):
            return set()
        try:
            return set(table())
        except Exception:
            return set()

    def _triage(self, tool_name: str, tool_plan: str) -> ToolResult | None:
        """A refusal when an existing tool already covers the request
        (forge_triage: known routes, built-in tools, installed plugins), else
        None. Pure text analysis over the live catalogue: no model call."""
        from .forge_triage import catalogue_from_dispatcher, triage
        try:
            verdict = triage(tool_name, tool_plan, catalogue_from_dispatcher(self.dispatcher))
        except Exception as exc:
            self.bus.log.emit(f"[FORGE] Manager: triage skipped - {first_line(exc, 100)}")
            return None
        if not verdict.covered:
            return None
        self.bus.log.emit(f"[FORGE] Manager: not forging '{tool_name}' - {verdict.call()} "
                          f"already covers it ({verdict.basis}).")
        return ToolResult(verdict.message(tool_name), ok=False, receipt={
            "operation": "forge", "state": "covered", "tool": tool_name,
            "covered_by": verdict.tool, "triage": verdict.to_dict()})

    @staticmethod
    def _mechanical_fixes(session: ForgeSession,
                          error_log: list[str] | None = None) -> list[str]:
        """Free, deterministic repairs (forge_autofix), before the first
        verification and after a failure: standard imports the module or the
        test uses but never made (and any a NameError in *error_log* names),
        get_tool_schema() reconciled with run(), and typed arguments
        converted on entry. A pass whose result would not parse leaves its
        input alone. Analysis trials keep their own fail-closed source rules
        and are not touched. Returns what was fixed."""
        if session.tool_name.startswith("trial_"):
            return []
        target = (failing_artefact("\n".join(str(line) for line in error_log), session.tool_name)
                  if error_log else None)
        module_log = error_log if target in (None, RepairTarget.MODULE) else None
        test_log = error_log if target in (None, RepairTarget.TEST) else None
        notes: list[str] = []
        code = session.tool_code
        for repair in (lambda text: add_missing_imports(text, module_log),
                       lambda text: reconcile_schema(text, session.tool_name),
                       add_argument_coercion):
            try:
                result = repair(code)
            except Exception:
                continue
            if result.changed:
                code = result.code
                notes.extend(result.fixed)
        test = session.test_code
        if str(test or "").strip():
            try:
                fixed = add_missing_imports(test, test_log, test_of=session.tool_name)
            except Exception:
                fixed = None
            if fixed is not None and fixed.changed:
                test = fixed.code
                notes.extend(f"test: {note}" for note in fixed.fixed)
        session.tool_code, session.test_code = code, test
        return notes

    @staticmethod
    def _test_policy_problems(session: ForgeSession) -> list[str]:
        """Why the generated test may not run in the verifier (it would touch
        the real disk, home folder or network, sleep, or wait on input)."""
        if session.tool_name.startswith("trial_"):
            return []
        try:
            return check_test_policy(session.test_code)
        except Exception:
            return []

    async def _repair_refused_test(self, session: ForgeSession, attempt: int,
                                   problems: list[str]) -> bool:
        """Record why the test was refused before it ran, then send only the
        TEST back for repair, told how to meet the policy with the staged
        fixtures. Consumes the attempt; True when a repair was made."""
        session.sandbox_attempts = attempt
        log = [f"TEST POLICY: {problem}" for problem in problems]
        session.errors.extend(log)
        text = " ".join(problems)
        diagnosis = Diagnosis(
            FailureClass.NETWORK if "network" in text else
            FailureClass.TIMEOUT if ("sleeps" in text or "input()" in text) else
            FailureClass.FILESYSTEM,
            RepairTarget.TEST,
            detail=f"the test was refused before it ran: {problems[0]}",
            directive=("Rewrite only the TEST so that it meets the verification policy; "
                       "change nothing in the module.\n" + FIXTURE_GUIDE))
        session.diagnoses.append(f"attempt {attempt}: test policy — {problems[0]}")
        self.bus.log.emit(f"[FORGE] Manager: ✗ attempt {attempt} — the test was refused "
                          f"before it ran: {problems[0]}")
        if self.lessons is not None:
            self.lessons.record_failure(session.tool_name, diagnosis, log)
        if attempt >= self.max_attempts:
            return False
        return await self._apply_repair(session, diagnosis, log)

    async def _apply_repair(
        self,
        session: ForgeSession,
        diagnosis: Diagnosis,
        error_log: list[str],
    ) -> bool:
        """Rewrite whichever artefact the diagnosis blames.

        Returns False when no repairer is configured at all, which tells the
        caller to stop rather than spin through the remaining attempts running
        the same failing code.

        The diagnosis and its directive are appended to the error log handed to
        the repairer, so even the legacy module-only ``llm_fixer`` receives the
        root-cause analysis instead of raw stderr alone.
        """
        context = list(error_log) + [
            f"DIAGNOSIS: {diagnosis.failure_class.value} — {diagnosis.detail}",
            f"REPAIR TARGET: {diagnosis.target.value}",
            f"DIRECTIVE: {diagnosis.directive}",
        ]
        if session.tool_name.startswith("trial_"):
            from .forge_trial_policy import generation_guidance
            context.append(generation_guidance(session.tool_name))
        elif diagnosis.repairs_test and FIXTURE_GUIDE not in diagnosis.directive:
            # A rewritten test can lean on the staged fixtures instead of the
            # real disk, network and clock.
            context.append(FIXTURE_GUIDE)

        if self.llm_repairer is not None:
            self.bus.log.emit(
                f"[FORGE] Manager: repairing {diagnosis.target.value} "
                f"({diagnosis.failure_class.value})")
            result = await self._call(
                self.llm_repairer, session.tool_code, session.test_code,
                diagnosis, context)
            self._absorb_repair(session, result, diagnosis)
            self._guard_session_source(session)
            return True

        if self.llm_fixer is not None:
            # Legacy path: module-only. It cannot fix a broken test, which is
            # precisely the limitation Mark II exists to remove — but it must
            # keep working for any caller still wired the old way.
            self.bus.log.emit("[FORGE] Manager: attempting self-heal via LLM (legacy fixer)")
            fixed_code = await self._call(self.llm_fixer, session.tool_code, context)
            if str(fixed_code or "").strip():
                session.tool_code = str(fixed_code)
            self._guard_session_source(session)
            return True

        return False

    @staticmethod
    def _check_planned_dependencies(session: ForgeSession) -> None:
        """A generator or repair cannot expand a frozen dependency contract."""
        if session.plan_contract is None:
            return
        allowed = {requirement_identity(item) for item in session.plan_contract["dependencies"]}
        for item in session.requirements:
            if requirement_identity(item) not in allowed:
                raise ValueError(f"Unplanned Forge dependency {item!r}; revise the plan before installing packages")

    @staticmethod
    def _planned_missing_requirement(session: ForgeSession, missing: str) -> str:
        """Keep a planned pin when a missing import triggers free resolution."""
        if session.plan_contract is None:
            return missing
        ForgeOrchestrationManager._check_planned_dependencies(session)
        name, _constraint = requirement_identity(missing)
        for requirement in session.plan_contract["dependencies"]:
            if requirement_identity(requirement)[0] == name:
                return requirement
        raise ValueError(f"Unplanned Forge dependency {missing!r}; no package was installed")

    @staticmethod
    def _guard_session_source(session: ForgeSession) -> None:
        """Apply the same admission gate at generation, repair and publication."""
        from .security import SecuritySanitiser, SecurityViolation
        ForgeOrchestrationManager._check_planned_dependencies(session)
        trial = session.tool_name if session.tool_name.startswith("trial_") else None
        if trial and session.requirements:
            raise SecurityViolation("Analysis trials cannot declare or install dependencies")
        SecuritySanitiser.guard_forged_source(
            session.tool_code, f"forge.{session.tool_name}.module", trial_name=trial)
        SecuritySanitiser.guard_forged_source(
            session.test_code, f"forge.{session.tool_name}.test", trial_name=trial, trial_test=True)

    @staticmethod
    def _absorb_repair(
        session: ForgeSession,
        result: Any,
        diagnosis: Diagnosis,
    ) -> None:
        """Fold a repairer's output back into the session.

        Accepts ``ForgeArtefacts`` or a plain ``(tool_code, test_code)`` pair.
        Only the artefacts the diagnosis actually blamed are replaced: a repair
        aimed at the test must never quietly swap out a module that verified
        its contract moments earlier, and an empty field means "unchanged"
        rather than "delete this".
        """
        tool_code, test_code, requirements = "", "", []
        if isinstance(result, ForgeArtefacts):
            tool_code, test_code = result.tool_code, result.test_code
            requirements = result.requirements
        elif isinstance(result, (tuple, list)) and result:
            tool_code = str(result[0] or "")
            test_code = str(result[1] or "") if len(result) > 1 else ""
        elif isinstance(result, str):
            tool_code = result

        if diagnosis.repairs_module and tool_code.strip():
            session.tool_code = tool_code
        if diagnosis.repairs_test and test_code.strip():
            session.test_code = test_code
        for requirement in requirements:
            if requirement not in session.requirements:
                session.requirements.append(requirement)

    async def forge_tool(
        self,
        tool_name: str,
        tool_plan: str,
        code_generator: Callable[[str, str], tuple[str, str, list[str]]] | None = None,
        *,
        autonomous: bool = False,
        force: bool = False,
    ) -> ToolResult:
        """
        Execute the complete tool-forging pipeline.

        ``autonomous`` marks a forge ORION started on his own (the
        self-improvement heartbeat). Only those announce a failure aloud: a
        forge the user asked for already reports back through its tool result,
        which the voice model reads out, so announcing it too said it twice.

        Before any model call the request is triaged (forge_triage): a need an
        existing tool already meets is answered with that tool, not a second
        copy of it. ``force`` skips that triage for a caller that has shown
        the existing tool falls short; a name that duplicates a built-in tool
        is refused either way, because the loader would never activate it.

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
        # The name becomes a FILE name (<name>_tool.py). Unsanitised, a name
        # like "../../x" wrote outside custom_tools; the heartbeat path already
        # cleaned its names, the forge tool did not.
        raw_name = str(tool_name or "")
        tool_name = forge_tool_name(raw_name)
        if not tool_name:
            return ToolResult(f"'{raw_name}' is not a usable tool name.", ok=False)
        native = set()
        table = getattr(self.dispatcher, "handler_table", None)
        if callable(table):
            try:
                native = set(table())
            except Exception:
                native = set()
        if tool_name in native:
            # A forged file with a built-in's name was reported live while the
            # dispatcher kept routing to the built-in — it could never run.
            return ToolResult(
                f"'{tool_name}' is already one of ORION's built-in tools; "
                "choose a distinct name for the new capability.", ok=False)
        twin = shadowed_builtin(tool_name, native)
        if twin:
            return ToolResult(
                f"'{tool_name}' would duplicate ORION's built-in tool '{twin}', and a "
                f"duplicate is never activated. Use {twin}, or name the new tool for "
                "what it adds.", ok=False,
                receipt={"operation": "forge", "state": "covered", "tool": tool_name,
                         "covered_by": twin})
        if not force:
            covered = self._triage(tool_name, tool_plan)
            if covered is not None:
                return covered
        locks = self.__dict__.setdefault("_name_locks", {})
        lock = locks.setdefault(tool_name, asyncio.Lock())
        if lock.locked():
            return ToolResult(f"'{tool_name}' is already being forged.", ok=False)
        async with lock:
            try:
                return await self._forge_tool_locked(tool_name, tool_plan, gen,
                                                     autonomous=autonomous)
            finally:
                locks.pop(tool_name, None)
                # Source/test text in old sessions can otherwise grow forever.
                finished = [key for key, value in self.sessions.items()
                            if value.tool_name not in locks]
                for key in finished[:-100]:
                    self.sessions.pop(key, None)

    async def _forge_tool_locked(self, tool_name: str, tool_plan: str,
                                 gen: Callable[[str, str], tuple[str, str, list[str]]],
                                 *, autonomous: bool = False,
                                 ) -> ToolResult:
        """The forge pipeline for one (sanitised, locked) tool name."""
        session_id = f"{tool_name}_{asyncio.get_running_loop().time()}".replace(".", "_")
        session = ForgeSession(session_id=session_id, tool_name=tool_name, tool_plan=tool_plan)
        self.sessions[session_id] = session

        self.bus.log.emit(
            f"[FORGE] Manager: initiating forge session '{session_id}' for '{tool_name}'"
        )

        try:
            from .forge_isolation import customer_verification_mode, save_review_candidate
            customer_mode = customer_verification_mode()
            if customer_mode and os.getenv("ORION_FORGE_SANDBOX", "process").strip().lower() != "container":
                raise ValueError("Customer Forge requires ORION_FORGE_SANDBOX=container; no source will run on the host")
            # Phase 1: Plan
            self.bus.log.emit(f"[FORGE] Manager: Phase 1 — Planning '{tool_name}'")
            generation_brief = tool_plan
            brain = getattr(gen, "__self__", None)
            # The real brain plans before it writes code. Injected legacy
            # generators retain their two-argument contract; they can opt into
            # fixed example acceptance through ForgeArtefacts.plan.
            if (not tool_name.startswith("trial_") and isinstance(brain, LlmForgeBrain)
                    and getattr(gen, "__name__", "") == "generate"):
                session.plan_contract = validate_plan(await brain.plan(tool_name, tool_plan))
                generation_brief = (tool_plan + "\n\nFIXED PLAN CONTRACT (implement exactly; "
                                    "do not change its examples or declared inputs):\n"
                                    + json.dumps(session.plan_contract, ensure_ascii=False))
            session.plan_phase_ok = True

            # Phase 2: Code Generation
            self.bus.log.emit(f"[FORGE] Manager: Phase 2 — Generating code")
            generated = await self._call(gen, tool_name, generation_brief)
            if isinstance(generated, ForgeArtefacts):
                session.tool_code = generated.tool_code
                session.test_code = generated.test_code
                session.requirements = list(generated.requirements)
                session.transport = generated.transport
                if generated.plan is not None:
                    supplied = validate_plan(generated.plan)
                    if session.plan_contract is not None and supplied != session.plan_contract:
                        raise ValueError("Generation changed the fixed Forge plan")
                    session.plan_contract = supplied
            else:
                tool_code, test_code, requirements = generated
                session.tool_code = tool_code
                session.test_code = test_code
                session.requirements = list(requirements or [])
            # Ordinary tools retain the fallback harness. Analysis trials need
            # explicit acceptance tests, not an import-only smoke test.
            if not str(session.test_code or "").strip():
                if not tool_name.startswith("trial_"):
                    session.test_code = synthesise_test(tool_name)
            session.code_generation_ok = True

            # Security guard (Mark X.13): forged code must never reach into
            # orion_core itself — previously only claimed below in a
            # comment, never actually enforced (SecuritySanitiser was never
            # imported here). guard_forged_source() makes that claim true;
            # a violation raises and is caught by the except block below
            # like any other forge failure (recorded as an incident, fed to
            # diagnostics).
            self._guard_session_source(session)
            if customer_mode and session.requirements:
                raise ValueError("Customer Forge cannot install generated dependencies; use the reviewed verifier image")
            # What generated code most often gets wrong mechanically (a
            # missing import, run() and the schema disagreeing, numbers that
            # arrive as floats) is fixed before the first verification, free.
            prepared = self._mechanical_fixes(session)
            if prepared:
                self._guard_session_source(session)
                session.diagnoses.append("before verification: autofix — " + "; ".join(prepared))
                self.bus.log.emit("[FORGE] Manager: ✓ prepared locally (no model call) — "
                                  + "; ".join(prepared))

            # Phase 3: Verification with diagnosis-driven repair.
            #
            # Each failure is classified before anything is rewritten, so the
            # repair lands on the artefact actually at fault.  A missing package
            # is installed rather than sent to the model, and does not consume a
            # code attempt (ADA parity).
            self.bus.log.emit(f"[FORGE] Manager: Phase 3 — Verification")
            dependency_fixes: set[str] = set()
            attempt = 0
            verification_runs = 0
            # Free mechanical repairs must still have a finite total budget.
            max_verification_runs = self.max_attempts + MAX_DEPENDENCY_REPAIRS + 5
            last_error_log: list[str] = []
            # (module, test) pairs already run: a free repair never re-runs one.
            verified: set[tuple[str, str]] = set()
            while attempt < self.max_attempts and verification_runs < max_verification_runs:
                attempt += 1
                verification_runs += 1
                self._guard_session_source(session)
                policy = self._test_policy_problems(session)
                if policy:
                    # A test that would touch the owner's real disk, home folder
                    # or network, sleep, or wait on input never runs.
                    if not await self._repair_refused_test(session, attempt, policy):
                        break
                    continue
                self.bus.log.emit(
                    f"[FORGE] Manager: verification attempt {attempt}/{self.max_attempts} "
                    f"for '{tool_name}'"
                )
                verified.add((session.tool_code, session.test_code))
                outcome = await self.sandbox.verify_tool(
                    session.tool_code,
                    session.test_code,
                    tool_name,
                    **({"plan": copy.deepcopy(session.plan_contract)}
                       if session.plan_contract is not None else {}),
                )
                if (outcome.passed and session.plan_contract is not None
                        and (getattr(outcome, "examples_verified", False) is not True
                             or getattr(outcome, "plan", None) != session.plan_contract)):
                    raise ValueError("Verification did not execute the three planned examples; candidate not accepted")
                if self.history is not None:
                    await asyncio.to_thread(
                        self.history.record, module=f"custom_tools/{tool_name}_tool.py",
                        before=None, after=session.tool_code, reasoning=tool_plan,
                        status="verified" if outcome.passed else "failed",
                        verification={"attempt": verification_runs, **outcome.to_dict()})

                if outcome.passed:
                    session.examples_verified = getattr(outcome, "examples_verified", False) is True
                    session.sandbox_ok = True
                    session.sandbox_attempts = attempt
                    self.bus.log.emit(
                        f"[FORGE] Manager: ✓ verification passed on attempt {attempt}")
                    # A failure we recovered from is worth far less as a warning
                    # than one that keeps killing sessions — damp its weight.
                    if last_error_log and self.lessons is not None:
                        self.lessons.record_resolution(
                            last_error_log,
                            note=f"repaired during '{tool_name}' forge")
                    break

                session.sandbox_attempts = attempt
                error_log = [str(e) for e in (getattr(outcome, "error_log", None) or [])]
                last_error_log = error_log
                session.errors.extend(error_log)

                diagnosis = diagnose(
                    error_log,
                    tool_name,
                    timed_out=bool(getattr(outcome, "timed_out", False)),
                    already_installed=dependency_fixes,
                )
                if (getattr(outcome, "stage", "") == "examples"
                        and diagnosis.target not in {RepairTarget.DEPENDENCY, RepairTarget.NONE}):
                    # The planner's answers are an independent acceptance gate,
                    # so changing the model's own test cannot resolve a failure.
                    diagnosis.target = RepairTarget.MODULE
                    diagnosis.directive += ("\nRepair the module against this fixed plan; "
                                            "do not weaken or rewrite the examples:\n"
                                            + json.dumps(session.plan_contract, ensure_ascii=False))
                session.diagnoses.append(f"attempt {attempt}: {diagnosis.summary()}")
                self.bus.log.emit(
                    f"[FORGE] Manager: ✗ attempt {attempt} — {diagnosis.summary()}")
                # The specific contract problems, not just "does not satisfy the
                # tool contract". Without these the log said a violation had
                # happened and never once said WHAT, so neither the user nor
                # anyone reading the log afterwards could tell.
                for line in error_log:
                    if "CONTRACT:" in str(line):
                        self.bus.log.emit(f"[FORGE]   → {str(line).strip()}")
                if self.lessons is not None:
                    self.lessons.record_failure(tool_name, diagnosis, error_log)

                # ── free, deterministic repair first ─────────────────────────
                # A contract violation is nearly always mechanical (run() needs
                # **kwargs, a positional-only parameter, a required argument the
                # schema never declares). Those have exactly one correct fix,
                # derivable from the code — so they are fixed here rather than
                # spent on a model call. In the reported session the forge burnt
                # a 6000-token-per-minute budget in four seconds and every
                # provider went into a five-minute cooldown; repairs that never
                # needed asking are the cheapest possible way to stop that.
                if diagnosis.failure_class is FailureClass.CONTRACT_VIOLATION:
                    autofix = autofix_contract(
                        session.tool_code, getattr(outcome, "schema", None),
                        error_log)
                    if autofix.changed:
                        session.tool_code = autofix.code
                        self._guard_session_source(session)
                        session.diagnoses.append(
                            f"attempt {attempt}: autofix — {autofix.summary()}")
                        self.bus.log.emit(
                            f"[FORGE] Manager: ✓ repaired the contract locally "
                            f"(no model call) — {autofix.summary()}")
                        if autofix.complete:
                            # Nothing left that needs judgement: re-verify
                            # immediately without consuming a code attempt.
                            attempt -= 1
                            continue

                # The other free repairs: an import the traceback or the source
                # shows missing, run() and the schema reconciled, typed
                # arguments converted. Never for a package to install or a
                # verifier that could not run, and never a pair already run.
                if diagnosis.target not in (RepairTarget.DEPENDENCY, RepairTarget.NONE):
                    fixed = self._mechanical_fixes(session, error_log)
                    if fixed and (session.tool_code, session.test_code) not in verified:
                        self._guard_session_source(session)
                        session.diagnoses.append(
                            f"attempt {attempt}: autofix — " + "; ".join(fixed))
                        self.bus.log.emit("[FORGE] Manager: ✓ repaired locally (no model "
                                          "call) — " + "; ".join(fixed))
                        attempt -= 1
                        continue

                # Environmental repair: install the package and re-verify for
                # free.  The tool's OWN name is never routed here — the
                # diagnosis layer recognises a self-import and sends it to the
                # repairer instead, because installing it from PyPI at best
                # fails and at worst pulls in an unrelated same-named package
                # that shadows the generated code (the 'dependency_resolver'
                # incident, 2026-07-17).
                if (diagnosis.target is RepairTarget.DEPENDENCY
                        and diagnosis.missing_module
                        and len(dependency_fixes) < MAX_DEPENDENCY_REPAIRS):
                    missing = diagnosis.missing_module
                    if tool_name.startswith("trial_"):
                        from .security import SecurityViolation
                        raise SecurityViolation(
                            "Analysis trials cannot install a missing dependency")
                    if customer_mode or customer_verification_mode():
                        raise ValueError("Customer Forge cannot install generated dependencies on the host")
                    requirement = self._planned_missing_requirement(session, missing)
                    dependency_fixes.add(missing)
                    self.bus.log.emit(
                        f"[FORGE] Manager: missing module '{missing}' — installing "
                        "and re-verifying (no code attempt consumed)"
                    )
                    dep = await self.resolver.resolve_and_install([requirement])
                    if dep.succeeded:
                        if requirement not in session.requirements:
                            session.requirements.append(requirement)
                        attempt -= 1          # dependency repair, not a code repair
                        continue

                if diagnosis.target is RepairTarget.NONE:
                    # Nothing a code change can fix (the verifier itself could
                    # not run). Stop now and say why, instead of paying for
                    # repairs of code that was never executed.
                    self.bus.log.emit(
                        f"[FORGE] Manager: ✗ cannot verify on this machine - "
                        f"{diagnosis.detail}. Stopping without repairs.")
                    break
                if attempt >= self.max_attempts:
                    break
                if not await self._apply_repair(session, diagnosis, error_log):
                    self.bus.log.emit(
                        "[FORGE] Manager: no repairer configured — stopping.")
                    break

            if not session.sandbox_ok:
                trail = "; ".join(session.diagnoses[-3:]) or "no diagnosis captured"
                session.errors.append(
                    f"Verification failed after {session.sandbox_attempts} attempt(s) — {trail}")
                raise RuntimeError(f"Verification failed. {trail}")

            # Verification inside a container is not permission to import that
            # source into ORION. Customer mode produces an inert review receipt,
            # and never reaches dependency installation or live publication.
            if customer_mode or customer_verification_mode():
                if (getattr(outcome, "isolation", "") != "container"
                        or getattr(outcome, "typecheck", "") != "passed"):
                    raise ValueError("Customer candidate needs container verification and a passed type check")
                candidate = await asyncio.to_thread(save_review_candidate,
                    _custom_tools_dir().parent / "forge_candidates", tool_name,
                    session.tool_code, session.test_code, outcome.to_dict())
                session.review_candidate = str(candidate)
                return ToolResult(
                    f"'{tool_name}' passed container verification and is saved for review at {candidate}. "
                    "No dependencies were installed on the host and the tool is not live.",
                    evidence=[{"candidate": str(candidate), "verified": True,
                               "activated": False, "isolation": "container"}])

            # Phase 4: Dependency Check
            self.bus.log.emit(f"[FORGE] Manager: Phase 4 — Dependency resolution")
            self._guard_session_source(session)
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
                raise RuntimeError("Declared dependencies could not be resolved")

            # Phase 5: Live Activation
            self.bus.log.emit(f"[FORGE] Manager: Phase 5 — Live activation")
            custom_tools_dir = _custom_tools_dir()
            custom_tools_dir.mkdir(parents=True, exist_ok=True)

            tool_file = custom_tools_dir / f"{tool_name}_tool.py"
            # Never persist a placeholder or unparseable body — a stub 'code'
            # once reached disk and crashed the loader on every boot.
            self._assert_valid_tool_source(session.tool_code, tool_name)
            await self._activate_verified(session, tool_file)

            session.activation_ok = True
            self.bus.log.emit(
                f"[FORGE] Manager: ✓ Tool '{tool_name}' activated and live"
            )

            # Compose success message
            attempts = max(1, session.sandbox_attempts or 1)
            result_msg = (
                f"✓ Successfully forged tool '{tool_name}'.\n"
                f"• Plan: ✓\n"
                f"• Code Gen: ✓\n"
                f"• Contract + {'3 examples + ' if session.examples_verified else ''}test: ✓ "
                f"(attempt {attempts}/{self.max_attempts})\n"
                f"• Dependencies: ✓\n"
                f"• Activation: ✓\n"
            )
            if session.diagnoses:
                result_msg += (
                    "\nRepaired along the way:\n"
                    + "\n".join(f"  • {d}" for d in session.diagnoses[-3:]) + "\n"
                )
            result_msg += "\nThe tool is now live and available in the dispatcher."
            return ToolResult(result_msg, ok=True)

        except Exception as e:
            self.bus.log.emit(
                f"[FORGE] Manager: ✗ forge session '{session_id}' failed: {e}"
            )
            session.errors.append(f"{type(e).__name__}: {str(e)}")
            if self.history is not None:
                try:
                    await asyncio.to_thread(
                        self.history.record, module=f"custom_tools/{tool_name}_tool.py",
                        before=None, after=session.tool_code or None, reasoning=tool_plan,
                        status="failed", verification={"stage": "orchestration",
                            "error": f"{type(e).__name__}: {str(e)[:2000]}"})
                except Exception as history_error:
                    self.bus.log.emit(f"[FORGE] Change evidence could not be persisted: {type(history_error).__name__}")
            retry_scheduled = False
            if isinstance(e, ProviderError):
                # This session never really got a fair shot — every remaining
                # provider was unavailable, not that the generated code was
                # wrong. Queue it for the heartbeat to retry automatically
                # rather than let it silently stay dead.
                self.pending_retries.append((tool_name, tool_plan))
                retry_scheduled = True
                self.bus.log.emit(
                    f"[FORGE] Manager: '{tool_name}' queued for automatic retry "
                    "at the next self-improvement cycle (no provider was reachable)."
                )
            # Make the failure a first-class incident: diagnostics and the
            # self_repair tool must be able to see and discuss it afterwards.
            # Incident.summary() truncates message to 120 chars, which is
            # exactly what ate the retry/status context before (the real
            # incident this fix responds to trailed off mid-word) — so the
            # retry verdict goes FIRST, ahead of the session id and full
            # error chain, to guarantee it survives the truncation and the
            # model has accurate ground truth instead of inventing "still
            # diagnosing" on top of a session that already ended.
            if self.repair is not None:
                try:
                    retry_note = ("retry scheduled at next self-improvement cycle"
                                  if retry_scheduled else "no retry scheduled")
                    self.repair.record_incident(
                        "ForgeSessionFailure",
                        f"forge '{tool_name}' [{retry_note}]: {first_line(e, 80)}",
                        module="forge",
                    )
                except Exception:
                    pass
            # A failed autonomous repair must be visible to the person it was
            # trying to help.  Keep this on the proactive speech channel so it
            # appears in the first window and the activity log; provider-outage
            # retries are excluded because those are not verification failures.
            if (autonomous and session.sandbox_attempts and not session.sandbox_ok
                    and not retry_scheduled):
                try:
                    self.bus.speak_request.emit(
                        f"I tried to build {tool_name.replace('_', ' ')} on my own, "
                        f"but it failed verification after {session.sandbox_attempts} "
                        "attempts. I've left it switched off and noted why."
                    )
                except Exception:
                    pass
            # Lead with the CLASSIFIED story rather than raw stderr: "the test
            # asserted an exact string the tool never promised" is actionable,
            # five trailing traceback lines are not.
            lines = [f"✗ Failed to forge '{tool_name}'."]
            if session.diagnoses:
                lines.append("What went wrong:")
                lines.extend(f"  • {d}" for d in session.diagnoses[-4:])
            lines.append("Last errors:")
            lines.extend(
                f"  • {first_line(err, 160)}"
                for err in session.errors[-4:] if str(err).strip()
            )
            lines.append(
                "Retry: scheduled automatically at the next self-improvement cycle."
                if retry_scheduled else
                "Retry: none scheduled — no further attempt is currently planned."
            )
            return ToolResult("\n".join(lines), ok=False)

    async def _activate_verified(self, session: ForgeSession, tool_file: Path) -> None:
        """Restore the stable file and registry if publication fails or is cancelled.

        This is a tool-generation swap, not importlib.reload of live core code.
        Existing calls keep their old callable and complete normally.
        """
        from .forge_isolation import customer_verification_mode
        if customer_verification_mode():
            raise ValueError("Customer Forge does not import or activate generated source on the host")
        self._guard_session_source(session)
        import sys
        cancelled = asyncio.Event()
        candidate = session.tool_code.encode("utf-8")

        async def transaction() -> None:
            before = await asyncio.to_thread(
                lambda: tool_file.read_bytes() if tool_file.exists() else None)
            module_name = tool_file.stem
            old_module = sys.modules.get(module_name)
            modules = getattr(self.loader, "loaded_modules", {})
            old_loaded = modules.get(module_name)
            dispatcher = self.dispatcher
            missing = object()
            previous_handler = missing
            previous_schemas: list[Any] = []
            previous_values: list[Any] = []
            attempted_handler = missing
            attempted_module = missing
            attempted_loaded = missing
            published_schemas: list[Any] | None = None
            published_values: list[Any] | None = None
            tool_name = session.tool_name
            if dispatcher is not None:
                previous_handler = getattr(dispatcher, "_tool_handlers", {}).get(tool_name, missing)
                previous_schemas = [d for d in getattr(dispatcher, "TOOL_DECLARATIONS", [])
                                    if d.get("name") == tool_name]
                previous_values = copy.deepcopy(previous_schemas)
            touched_registry = False
            published = False
            receipt: Path | None = None

            def owns_generation() -> bool:
                # A matching callable alone is insufficient: another publisher
                # may have changed its schema or re-imported the same bytes.
                if dispatcher is not None:
                    current = getattr(dispatcher, "_tool_handlers", {}).get(tool_name, missing)
                    schemas = [d for d in getattr(dispatcher, "TOOL_DECLARATIONS", [])
                               if d.get("name") == tool_name]
                    if touched_registry and current is attempted_handler and published_schemas is not None:
                        expected, values = published_schemas, published_values
                    elif current is previous_handler:
                        expected, values = previous_schemas, previous_values
                    else:
                        return False
                    if (len(schemas) != len(expected) or schemas != values
                            or any(current is not original for current, original in zip(schemas, expected))):
                        return False
                expected_module = old_module if attempted_module is missing else attempted_module
                expected_loaded = old_loaded if attempted_loaded is missing else attempted_loaded
                return (sys.modules.get(module_name) is expected_module
                        and modules.get(module_name) is expected_loaded)

            async def restore(reason: str) -> bool:
                # Preserve the whole later generation when its runtime or
                # source has moved, including an identical-byte re-import.
                def restore_source() -> bool:
                    if not owns_generation() or tool_file.is_symlink():
                        return False
                    current = tool_file.read_bytes() if tool_file.exists() else None
                    if current == before:
                        return True
                    if current != candidate:
                        return False
                    if before is None:
                        tool_file.unlink(missing_ok=True)
                    else:
                        atomic_write_bytes(tool_file, before)
                    return True

                restored = owns_generation() and await asyncio.to_thread(restore_source)
                restored = restored and owns_generation()
                if restored:
                    if dispatcher is not None and touched_registry:
                        handlers = getattr(dispatcher, "_tool_handlers", {})
                        if previous_handler is missing:
                            handlers.pop(tool_name, None)
                        else:
                            handlers[tool_name] = previous_handler
                        declarations = getattr(dispatcher, "TOOL_DECLARATIONS", None)
                        if declarations is not None:
                            declarations[:] = [d for d in declarations if d.get("name") != tool_name]
                            declarations.extend(previous_schemas)
                    if old_module is None:
                        sys.modules.pop(module_name, None)
                    else:
                        sys.modules[module_name] = old_module
                    if old_loaded is None:
                        modules.pop(module_name, None)
                    else:
                        modules[module_name] = old_loaded
                if self.history is not None:
                    await asyncio.to_thread(
                        self.history.record, module=f"custom_tools/{tool_file.name}",
                        before=candidate.decode("utf-8"),
                        after=before.decode("utf-8") if before is not None and restored else None,
                        reasoning=reason if restored else "A later source or runtime generation was preserved. " + reason,
                        status="rolled_back" if restored else "preserved_newer",
                        verification={"restored": bool(restored)})
                return bool(restored)

            try:
                if customer_verification_mode():
                    raise ValueError("Customer Forge activation was disabled before publication")
                from .forge_deployment import prepare
                receipt = await asyncio.to_thread(prepare, tool_file, before, candidate)
                await asyncio.to_thread(atomic_write_bytes, tool_file, candidate)
                if customer_verification_mode():
                    raise ValueError("Customer Forge activation was disabled before host import")
                load_outcome = await self.loader.load_and_register(tool_file)
                attempted_handler = getattr(load_outcome, "handler_fn", missing)
                imported_module = getattr(load_outcome, "module_obj", None)
                current_module = sys.modules.get(module_name)
                current_loaded = modules.get(module_name)
                if imported_module is not None:
                    attempted_module = imported_module
                elif (attempted_handler is not missing and attempted_handler is not None
                        and getattr(current_module, "run", None) is attempted_handler):
                    attempted_module = current_module
                if (isinstance(current_loaded, dict)
                        and current_loaded.get("handler") is attempted_handler):
                    attempted_loaded = current_loaded
                if not load_outcome.succeeded:
                    raise RuntimeError(f"Failed to load module: {load_outcome.error_log}")
                if not owns_generation():
                    raise RuntimeError("Tool generation changed while loading; the newer state was preserved")
                if cancelled.is_set():
                    raise asyncio.CancelledError
                if dispatcher is not None:
                    declared_name = (getattr(load_outcome, "tool_schema", None) or {}).get("name", tool_name)
                    # A model must not use its schema to replace a DIFFERENT
                    # built-in or custom tool under a harmless filename.
                    if declared_name != tool_name:
                        raise ValueError("Verified schema name differs from the requested tool name")
                    touched_registry = True
                    activation = await self.loader.activate_tool(load_outcome, dispatcher)
                    if not activation.ok:
                        raise RuntimeError(f"Dispatcher activation failed: {activation.text}")
                    published_schemas = [d for d in getattr(dispatcher, "TOOL_DECLARATIONS", [])
                                         if d.get("name") == tool_name]
                    published_values = copy.deepcopy(published_schemas)
                if cancelled.is_set():
                    raise asyncio.CancelledError
                if self.history is not None:
                    await asyncio.to_thread(
                        self.history.record, module=f"custom_tools/{tool_file.name}",
                        before=before.decode("utf-8") if before is not None else None,
                        after=candidate.decode("utf-8"),
                        reasoning=session.tool_plan, status="applied",
                        verification={"syntax_contract_tests": True,
                                      "attempts": session.sandbox_attempts})
                if cancelled.is_set():
                    raise asyncio.CancelledError
                if dispatcher is not None and hasattr(dispatcher, "__dict__"):
                    # A fault in a later call can revert this generation. A
                    # stale request must never roll back a newer generation.
                    async def rollback(handler: Any, reason: str) -> bool:
                        if handler is not attempted_handler:
                            return False
                        if getattr(dispatcher, "_tool_handlers", {}).get(tool_name) is not handler:
                            return False
                        locks = self.__dict__.setdefault("_name_locks", {})
                        lock = locks.setdefault(session.tool_name, asyncio.Lock())
                        if lock.locked():
                            return False
                        try:
                            async with lock:
                                # A manual edit after deployment belongs to its author.
                                current = await asyncio.to_thread(tool_file.read_bytes)
                                if (current != candidate or not owns_generation() or
                                        getattr(dispatcher, "_tool_handlers", {}).get(tool_name) is not handler):
                                    return False
                                return await restore(reason)
                        finally:
                            locks.pop(session.tool_name, None)
                    callbacks = dispatcher.__dict__.setdefault("_forge_rollbacks", {})
                    callbacks[tool_name] = rollback
                await asyncio.to_thread(receipt.unlink, missing_ok=True)
                published = True
            finally:
                if not published and receipt is not None:
                    await restore("Activation failed or was cancelled; recovery was requested.")
                    await asyncio.to_thread(receipt.unlink, missing_ok=True)

        task = asyncio.create_task(transaction())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            # Retrieve failures even when caller cancellation wins the race.
            if not task.cancelled():
                task.exception()
            raise

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

    @staticmethod
    def _assert_valid_tool_source(code: str, tool_name: str) -> None:
        """Refuse to write a forged tool whose body is a placeholder or does not
        parse as Python.  This is the guard that prevents the historic failure
        where a bare 'code' stub was persisted and the loader then raised
        'name code is not defined' on every subsequent boot."""
        import ast
        body = str(code or "").strip()
        if len(body) < 40 or body in {"code", "pass", "...", "None"}:
            raise RuntimeError(
                f"refusing to write '{tool_name}_tool.py': generated body is "
                "empty or a placeholder — regenerating instead of persisting a stub")
        try:
            ast.parse(body)
        except SyntaxError as exc:
            raise RuntimeError(
                f"refusing to write '{tool_name}_tool.py': generated body is not "
                f"valid Python ({exc})") from exc

    async def reload_persisted_tools(self) -> int:
        """Re-activate every previously forged tool at startup (ADA parity:
        her skills survive restarts; ORION's forged tools previously existed
        on disk but were never re-registered).  A tool that fails to load is
        skipped with a log line — one broken artefact never blocks the rest.
        Returns the number of tools brought back to life.

        Every tool now passes the same contract gate a freshly forged one does,
        so a tool created before that gate existed — schema and run() quietly
        disagreeing — is caught here at boot instead of raising TypeError in the
        middle of a live conversation.  Boot also reports what is being held
        back, because a silently skipped tool is a tool nobody re-forges.
        """
        from .forge_isolation import customer_verification_mode
        if customer_verification_mode():
            self.bus.log.emit("[FORGE] Customer verification-only mode: persisted generated tools are not loaded or rehabilitated.")
            return 0
        if self.dispatcher is None:
            return 0
        custom_tools_dir = _custom_tools_dir()
        from .forge_deployment import recover_pending
        recoveries = await asyncio.to_thread(recover_pending, custom_tools_dir, self.history)
        conflicted = {item["file"] for item in recoveries if not item["restored"]}
        for item in recoveries:
            self.bus.log.emit(f"[FORGE] Interrupted deployment {item['file']}: "
                              + ("previous generation restored" if item["restored"] else item["error"]))
        # Before loading, give quarantined tools a second chance: any that were
        # condemned for a missing package now installed (the sklearn/jsonschema
        # case) are restored here and picked up by the glob below in this same
        # pass, so a resolved blocker no longer means a permanently lost tool.
        try:
            revived = await self.loader.rehabilitate_quarantined()
            if revived:
                self.bus.log.emit(
                    f"[FORGE] Manager: brought {len(revived)} tool(s) back from "
                    f"quarantine — {', '.join(revived)}.")
        except Exception as exc:
            self.bus.log.emit(
                f"[FORGE] Manager: quarantine rehabilitation skipped - {first_line(exc, 100)}")
        reloaded = 0
        rejected: list[str] = []
        registry = getattr(self.dispatcher, "plugins", None)
        from .plugin_manifest import discover_manifests
        manifests, _manifest_errors = await asyncio.to_thread(discover_manifests, custom_tools_dir)
        owned_modules = {manifest.module for manifest in manifests}
        for tool_file in sorted(custom_tools_dir.glob("*_tool.py")):
            if tool_file.name in conflicted:
                continue
            manifest = tool_file.with_name(tool_file.stem[: -len("_tool")] + ".plugin.json")
            if tool_file.name in owned_modules or (tool_file.stem.endswith("_tool") and manifest.exists()):
                # A plugin: load_plugins owns it and honours its enabled state.
                # Loading it here too imported it twice and brought DISABLED
                # plugins back at every boot.
                continue
            try:
                if registry is not None and not registry.is_enabled(tool_file.stem[:-len("_tool")]):
                    state_error = getattr(registry, "state_error", "")
                    reason = (f"saved plugin preferences unavailable — {state_error}" if state_error
                              else "disabled by the user")
                    self.bus.log.emit(f"[FORGE] Persisted tool '{tool_file.stem}' held — {reason}.")
                    continue
                outcome = await self.loader.load_and_register(tool_file)
                if not outcome.succeeded:
                    detail = first_line(str((outcome.error_log or [""])[-1]), 120)
                    rejected.append(f"{tool_file.stem}: {detail}")
                    self.bus.log.emit(
                        f"[FORGE] Manager: persisted tool '{tool_file.stem}' failed to "
                        f"load — {detail}")
                    continue
                activation = await self.loader.activate_tool(outcome, self.dispatcher)
                if activation.ok:
                    reloaded += 1
                else:
                    rejected.append(f"{tool_file.stem}: {first_line(activation.text, 120)}")
                    self.bus.log.emit(
                        f"[FORGE] Manager: persisted tool '{tool_file.stem}' failed to "
                        f"activate — {first_line(activation.text, 120)}")
            except Exception as exc:
                rejected.append(f"{tool_file.stem}: {first_line(exc, 120)}")
                self.bus.log.emit(
                    f"[FORGE] Manager: persisted tool '{tool_file.stem}' skipped — "
                    f"{first_line(exc, 120)}")
        if reloaded:
            self.bus.log.emit(
                f"[FORGE] Manager: {reloaded} previously forged tool(s) reloaded and live.")
        if rejected:
            self.bus.log.emit(
                f"[FORGE] Manager: {len(rejected)} forged tool(s) held back and NOT live — "
                "run forge(action='health') for the detail and re-forge them.")
        quarantine = self._quarantined()
        if quarantine:
            self.bus.log.emit(
                f"[FORGE] Manager: {len(quarantine)} tool(s) sit in quarantine "
                "(broken on load); forge(action='health') lists them.")
        return reloaded

    def _quarantined(self) -> list[dict[str, Any]]:
        lister = getattr(self.loader, "quarantined", None)
        if not callable(lister):
            return []
        try:
            return list(lister())
        except Exception:
            return []

    def health(self) -> ToolResult:
        """What the Forge is holding back, and why.

        Three populations were previously invisible: tools quarantined for
        failing to load, tools that load but break the contract, and the lesson
        corpus.  All of them are actionable — a held-back tool is a capability
        the user thinks ORION has and it does not.
        """
        lines: list[str] = ["Forge health:"]
        from .forge_isolation import customer_verification_mode
        try:
            customer_mode = customer_verification_mode()
        except ValueError as exc:
            return ToolResult(f"Forge health: invalid customer policy — {exc}", ok=False)
        if customer_mode:
            lines.append("  • Customer verification-only mode: generated code is held for review and is not activated.")

        contract_failures = dict(getattr(self.loader, "contract_failures", {}) or {})
        quarantine = self._quarantined()
        live = len(getattr(self.loader, "loaded_modules", {}) or {})
        lines.append(f"  • {live} forged tool(s) loaded and live.")

        if contract_failures:
            lines.append(f"  • {len(contract_failures)} tool(s) break the tool contract "
                         "and are NOT callable:")
            for name, problems in list(contract_failures.items())[:6]:
                lines.append(f"      - {name}: {problems[0]}")
                for extra in problems[1:3]:
                    lines.append(f"        {extra}")
            lines.append("    Re-forge these; their schema and run() disagree.")

        if quarantine:
            lines.append(f"  • {len(quarantine)} tool(s) quarantined (would not load):")
            for row in quarantine[:6]:
                stamp = f" [{row['at']}]" if row.get("at") else ""
                lines.append(f"      - {row['tool']}{stamp}: {first_line(row['reason'], 110)}")
            lines.append("    They are inert on disk; re-forge to replace them.")

        if self.lessons is not None:
            try:
                described = self.lessons.describe()
                lines.append(
                    f"  • Lesson corpus: {described['lessons']} recorded, "
                    f"{described['unresolved']} still unresolved.")
            except Exception:
                pass

        sessions = list(self.sessions.values())
        if sessions:
            ok = sum(1 for s in sessions if s.activation_ok)
            lines.append(f"  • This run: {ok}/{len(sessions)} forge session(s) succeeded.")

        if not customer_mode and not contract_failures and not quarantine:
            lines.append("  • Nothing is held back — every forged tool on disk is live.")
        return ToolResult("\n".join(lines), ok=not (contract_failures or quarantine))

    def snapshot(self) -> dict[str, Any]:
        """Structured view of health() — for the Diagnostics Centre panel,
        which needs fields to render rather than pre-formatted text. Reads
        the exact same underlying state as health(); the two must never
        drift, so this is the only place either one queries the loader
        directly."""
        contract_failures = dict(getattr(self.loader, "contract_failures", {}) or {})
        quarantine = self._quarantined()
        live = len(getattr(self.loader, "loaded_modules", {}) or {})
        sessions = list(self.sessions.values())
        ok = sum(1 for s in sessions if s.activation_ok)
        lessons = self.lessons.describe() if self.lessons is not None else {}
        return {
            "live": live,
            "contract_failures": [
                {"name": name, "problems": list(problems)}
                for name, problems in contract_failures.items()
            ],
            "quarantine": quarantine,
            "session_ok": ok,
            "session_total": len(sessions),
            "lessons": lessons,
        }

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


# ──────────────────────────────────────────────────────────────────────────────
# THE FORGE BRAIN — LLM code generation + self-heal (ported from Ada-SI).
#
# The orchestration loop above existed since Mark X.7+ but was constructed
# WITHOUT a code generator ("ForgeOrchestrationManager(bus)" at the
# composition root), so every forge attempt died with "No code generator
# configured" — the Forge had hands but no brain.  Ada-SI's pipeline
# (chat/tool_creator.py → tool_verify.py → revise-with-error-logs) supplies
# the missing pattern: a disciplined generate → verify → revise loop driven
# through the model, with strict-JSON artefacts.  This class is that brain,
# implemented over ORION's own ProviderRouter so it works with whichever
# cloud or local model is available.
# ──────────────────────────────────────────────────────────────────────────────

_FORGE_CONTRACT = """MODULE CONTRACT (the loader and sandbox enforce this exactly):
• Export get_tool_schema() -> dict: an OpenAI-style function schema:
  {"name": "<tool_name>", "description": "<one sentence>",
   "parameters": {"type": "object", "properties": {...}, "required": [...]}}
• Export run(**kwargs) -> str: the tool entry point. It must accept the
  schema's parameters as keyword arguments, handle its own errors, and
  always return a readable string.
• THE SCHEMA AND run() ARE CHECKED AGAINST EACH OTHER BEFORE ANY TEST RUNS.
  Every key in the schema's "properties" must be a keyword argument run()
  accepts (or run() must take **kwargs); every parameter of run() without a
  default must appear in "properties"; every name in "required" must appear in
  "properties"; no positional-only parameters. A mismatch fails verification
  outright with a CONTRACT error.
• Pure Python. Prefer the standard library; list any third-party import in the
  REQUIREMENTS section. No GUI, no input(), no infinite loops, no threads.
• Never read or write outside the current working directory; never touch
  O.R.I.O.N.'s own source files.
• The test script runs in a subprocess in that same directory. It must import
  the module as '<tool_name>_tool' or '<tool_name>' (both are staged beside
  it), exercise run() with realistic arguments USING THE SCHEMA'S OWN
  PARAMETER NAMES, print what it checks, and raise on failure.
• The test must be OFFLINE and DETERMINISTIC: no network calls, no clock or
  randomness dependence, no sleeps or polling. Assert on properties that are
  actually guaranteed (type, substring, non-empty) rather than an exact string
  the tool never promised — an over-specific assertion is a TEST defect and
  will be treated as one."""


# The artefact layout the Forge asks for.  Source travels in fenced code blocks
# rather than inside JSON string values: models emit fences reliably, whereas
# escaping a 200-line module into a JSON string fails on a single stray
# newline and kills the whole session.
_FORGE_RESPONSE_FORMAT = """RESPOND IN EXACTLY THIS LAYOUT — three sections, \
fenced code blocks, no other prose:

=== TOOL ===
```python
<the complete module source>
```
=== TEST ===
```python
<the complete test script source>
```
=== REQUIREMENTS ===
```
<one pip package per line, or leave the block empty for none>
```"""


_FORGE_INSTRUCTION = (
    "You are FORGE, O.R.I.O.N.'s capability foundry — a precise Python "
    "engineer. You return complete, runnable files inside fenced code blocks "
    "under the section headers you are given, and nothing else: no commentary "
    "before or after, no explanation, no diffs, no ellipses or '# ...' "
    "placeholders standing in for code you did not write."
)


_FORGE_JSON_INSTRUCTION = (
    "You are FORGE, O.R.I.O.N.'s capability foundry — a precise Python "
    "engineer. You output STRICT JSON artefacts and nothing else: no markdown "
    "fences, no prose, no commentary. Inside JSON string values, newlines "
    "MUST be escaped as \\n and double quotes as \\\"."
)


#: Reply budget for a forge turn: a module AND its tests. Left unset, a turn
#: was capped at the conversational 2,048 tokens (and at 256 while OpenRouter
#: was out of credit), so generated code arrived cut off mid-function, failed
#: to parse or verify, and the forge reported yet another failure. This — not
#: the ideas — is why the forge "barely worked".
FORGE_REPLY_TOKENS = 8192


async def strict_json_turn(router: Any, prompt: str) -> dict[str, Any]:
    """One strict-JSON model turn with a single self-correcting retry.

    Malformed JSON (raw newlines inside code strings, a missed comma) used to
    kill an entire forge session outright ("Expecting ',' delimiter").  Now the
    parse error is fed back to the model once, and mechanical repair handles
    the common escaping mistakes before we ever give up.  Falls back to the
    legacy ``system_extra`` call form for routers that predate ``instruction``.
    """
    async def _turn(text: str) -> str:
        try:
            _, raw = await router.generate_text(
                text, instruction=_FORGE_JSON_INSTRUCTION, task="forge",
                max_tokens=FORGE_REPLY_TOKENS)
        except TypeError:   # older router / test double without these keywords
            try:
                _, raw = await router.generate_text(text, instruction=_FORGE_JSON_INSTRUCTION)
            except TypeError:
                _, raw = await router.generate_text(
                    text, system_extra="You output strict JSON artefacts, nothing else.")
        return raw

    raw = await _turn(prompt)
    try:
        return LlmForgeBrain._parse_json(raw)
    except (RuntimeError, json.JSONDecodeError, ValueError) as exc:
        retry = (
            f"{prompt}\n\nYOUR PREVIOUS RESPONSE WAS NOT VALID JSON "
            f"({first_line(exc, 120)}). Respond again with ONE strict JSON object "
            "only — escape every newline in string values as \\n and every "
            "double quote as \\\" — no fences, no prose."
        )
        raw = await _turn(retry)
        return LlmForgeBrain._parse_json(raw)


async def blocks_turn(
    router: Any,
    prompt: str,
    instruction: str = "",
    require_module: bool = True,
) -> ForgeArtefacts:
    """One artefact-producing model turn, with a single self-correcting retry.

    Unlike the JSON path this cannot fail on escaping — a fenced block is
    delimited, not encoded — so the only recoverable failure left is a model
    that ignored the layout entirely, which the retry addresses directly.

    ``require_module`` says what "usable" means for this turn. Generation needs
    a module; a repair aimed only at the TEST legitimately returns no module at
    all, and demanding one there would burn a retry on every such repair and
    then fail the session for a response that was perfectly correct.
    """
    system = instruction or _FORGE_INSTRUCTION

    async def _turn(text: str) -> str:
        try:
            _profile, raw = await router.generate_text(
                text, instruction=system, task="forge", max_tokens=FORGE_REPLY_TOKENS)
        except TypeError:   # older router / test double without these keywords
            try:
                _profile, raw = await router.generate_text(text, instruction=system)
            except TypeError:
                _profile, raw = await router.generate_text(text, system_extra=system)
        return raw

    def _usable(candidate: ForgeArtefacts) -> bool:
        if require_module:
            return candidate.complete
        return bool(candidate.tool_code.strip() or candidate.test_code.strip())

    artefacts = extract_artefacts(await _turn(prompt))
    if _usable(artefacts):
        return artefacts
    wanted = "A USABLE MODULE" if require_module else "ANY USABLE SOURCE"
    retry = (
        f"{prompt}\n\nYOUR PREVIOUS RESPONSE DID NOT CONTAIN {wanted}. "
        "Respond again using the exact section layout, with the complete file "
        "inside a ```python fenced block under each header. Do not explain, do "
        "not abbreviate, do not use placeholders."
    )
    return extract_artefacts(await _turn(retry))


class LlmForgeBrain:
    """Code generator + targeted repairer for the Forge, via ProviderRouter."""

    #: Errors that mean every provider was busy, not that the prompt failed.
    _PROVIDERS_BUSY = re.compile(r"(?i)all text providers failed|no text provider")

    async def _turn(self, turn, prompt: str, **kwargs):
        """One model turn, waiting out a provider cooldown once before failing.

        A forge is several turns; a free-tier rate limit hit in the middle of
        one used to throw the whole session away, even though the provider
        was back in under half a minute.
        """
        try:
            return await turn(self.router, prompt, **kwargs)
        except Exception as exc:
            waiter = getattr(self.router, "wait_for_text", None)
            if not (self._PROVIDERS_BUSY.search(str(exc)) and callable(waiter)):
                raise
            if not await waiter(90.0):
                raise
            return await turn(self.router, prompt, **kwargs)

    def __init__(self, bus: OrionBus, router: Any,
                 lessons: ForgeLessonStore | None = None, history: Any = None) -> None:
        self.bus = bus
        self.router = router
        # Optional: past failures become guidance in the generation prompt, so
        # the Forge stops rediscovering the same mistakes every session.
        self.lessons = lessons
        self.history = history

    # ── generation ────────────────────────────────────────────────────────────

    async def plan(self, tool_name: str, tool_plan: str) -> dict[str, Any]:
        """One bounded planning turn, before code or its own tests exist."""
        prompt = (
            "Plan one small headless ORION tool before writing any code. "
            "Return only a JSON object with inputs (parameter name -> JSON schema "
            "with lowercase type), required (input names), output (the useful "
            "non-empty string run() returns), dependencies (package requirements; "
            "prefer the stdlib and bundled requests, psutil, numpy), and examples "
            "(EXACTLY THREE objects). Each example needs arguments and expect "
            "(an expected answer substring that proves its result); optionally "
            "expect_error: false, files: {relative_path: text}, http: {URL: canned_body}. "
            "Use private fixture files and {workdir} in arguments for absolute "
            "fixture paths. No real user paths, network, installs, devices, "
            "accounts or services in examples. Include a successful example. "
            "Keep the whole plan under 32 KiB, with at most 24 inputs and 12 "
            "dependencies. Do not produce module or test source.\n\n"
            f"TOOL NAME: {tool_name}\nREQUEST: {str(tool_plan)[:12000]}"
        )

        async def turn(router, text):
            instruction = "Return one bounded JSON tool plan, without code or prose."
            try:
                _profile, raw = await router.generate_text(
                    text, instruction=instruction, task="forge.plan", max_tokens=2400)
            except TypeError:
                _profile, raw = await router.generate_text(text, system_extra=instruction)
            if len(str(raw).encode("utf-8")) > 32768:
                raise ValueError("Forge planner response exceeds the 32 KiB limit")
            return self._parse_json(raw)

        try:
            planned = await asyncio.wait_for(self._turn(turn, prompt), timeout=45.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Forge planning did not finish within 45 seconds; no code was generated") from None
        return validate_plan(planned)

    async def generate(self, tool_name: str, tool_plan: str) -> ForgeArtefacts:
        guidance = ""
        if self.lessons is not None:
            try:
                guidance = await asyncio.to_thread(self.lessons.guidance_block, tool_plan)
            except Exception:
                guidance = ""     # memory must never block a forge
        if self.history is not None:
            try:
                guidance += "\n" + await asyncio.to_thread(self.history.guidance_block, tool_plan)
            except Exception:
                pass
        prompt = (
            "Build a new tool for O.R.I.O.N.\n\n"
            f"TOOL NAME: {tool_name}\n"
            f"CAPABILITY PLAN: {tool_plan}\n\n"
            f"{_FORGE_CONTRACT}\n\n"
            # The fixtures staged beside every verification (forge_fixtures);
            # analysis trials follow their own test rules instead.
            + ("" if tool_name.startswith("trial_") else f"{FIXTURE_GUIDE}\n\n")
            + (f"{guidance}\n\n" if guidance else "")
            + _FORGE_RESPONSE_FORMAT
        )
        if tool_name.startswith("trial_"):
            from .forge_trial_policy import generation_guidance
            prompt += "\n\n" + generation_guidance(tool_name)
        artefacts = await self._turn(blocks_turn, prompt)
        if not artefacts.tool_code.strip():
            raise RuntimeError("the model returned no tool module")
        if not artefacts.test_code.strip() and not tool_name.startswith("trial_"):
            # A missing test harness must never abort the forge — synthesise
            # a minimal loader-level test instead.
            artefacts.test_code = synthesise_test(tool_name)
        if artefacts.transport != "blocks":
            self.bus.log.emit(
                f"[FORGE] Brain: artefacts recovered via the '{artefacts.transport}' "
                "fallback — the model did not follow the block layout.")
        return artefacts

    # ── targeted repair ───────────────────────────────────────────────────────

    async def repair(
        self,
        tool_code: str,
        test_code: str,
        diagnosis: Diagnosis,
        error_log: list[str],
    ) -> ForgeArtefacts:
        """Repair whichever artefact the diagnosis blames.

        The model is shown BOTH files — it cannot judge whether a failing
        assertion is a test defect or a module defect while looking at only one
        of them — but is asked to return only the sections it needs to change.
        """
        errors = "\n".join(str(e) for e in (error_log or [])[-14:])
        wanted: list[str] = []
        if diagnosis.repairs_module:
            wanted.append("=== TOOL ===\n```python\n<the complete corrected module>\n```")
        if diagnosis.repairs_test:
            wanted.append("=== TEST ===\n```python\n<the complete corrected test>\n```")
        if not wanted:      # defensive: always give the model something to do
            wanted.append("=== TOOL ===\n```python\n<the complete corrected module>\n```")

        prompt = (
            "A forged tool failed verification. Repair it.\n\n"
            f"CURRENT MODULE:\n```python\n{tool_code}\n```\n\n"
            f"CURRENT TEST:\n```python\n{test_code}\n```\n\n"
            f"VERIFICATION OUTPUT:\n{errors}\n\n"
            f"ROOT CAUSE: {diagnosis.failure_class.value} — {diagnosis.detail}\n"
            f"WHAT TO CHANGE: {diagnosis.target.value}\n"
            f"HOW: {diagnosis.directive}\n\n"
            f"{_FORGE_CONTRACT}\n\n"
            "Return ONLY the section(s) below, each holding the COMPLETE "
            "corrected file — never a diff, never a fragment:\n"
            + "\n".join(wanted)
        )
        artefacts = await self._turn(
            blocks_turn, prompt, require_module=diagnosis.repairs_module)
        # Preserve whatever the repair deliberately left alone.
        if not artefacts.tool_code.strip():
            artefacts.tool_code = tool_code
        if not artefacts.test_code.strip():
            artefacts.test_code = test_code
        return artefacts

    async def fix(self, tool_code: str, error_log: list[str]) -> str:
        """Legacy module-only fixer, kept for callers wired the old way."""
        errors = "\n".join(str(e) for e in (error_log or [])[-8:])
        prompt = (
            "A forged tool module failed verification. Correct it.\n\n"
            f"CURRENT MODULE SOURCE:\n```python\n{tool_code}\n```\n\n"
            f"VERIFICATION OUTPUT:\n{errors}\n\n"
            f"{_FORGE_CONTRACT}\n\n"
            "Return ONLY:\n=== TOOL ===\n```python\n"
            "<the complete corrected module>\n```"
        )
        artefacts = await self._turn(blocks_turn, prompt)
        return artefacts.tool_code if artefacts.tool_code.strip() else tool_code

    # ── strict-JSON parsing (models love fences; strip them ruthlessly) ──────

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        candidates = [text]
        # The outermost brace span (prose-wrapped JSON).
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])
        last_error: Exception | None = None
        for candidate in candidates:
            for variant in (candidate, LlmForgeBrain._repair_json(candidate)):
                try:
                    data = json.loads(variant)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError as exc:
                    last_error = exc
        raise RuntimeError(
            f"model response was not JSON ({first_line(last_error, 80)}): "
            f"{first_line(text, 120)}")

    @staticmethod
    def _repair_json(text: str) -> str:
        """Mechanically repair the JSON mistakes models actually make when
        embedding source code: raw newlines/tabs inside string literals and
        trailing commas.  Conservative — a string-state walk, no guessing."""
        out: list[str] = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                    out.append(ch)
                elif ch == "\\":
                    escaped = True
                    out.append(ch)
                elif ch == '"':
                    in_string = False
                    out.append(ch)
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\t":
                    out.append("\\t")
                elif ch == "\r":
                    pass
                else:
                    out.append(ch)
            else:
                if ch == '"':
                    in_string = True
                out.append(ch)
        repaired = "".join(out)
        return re.sub(r",\s*([}\]])", r"\1", repaired)


# ──────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT HEARTBEAT — autonomous self-improvement (ported from Ada-SI).
#
# Ada runs a periodic "heartbeat" maintenance pass: a model turn reviews her
# recent logs and consolidates what it learns, without the user asking.  This
# is ORION's equivalent, pointed at self-improvement:
#
#     every tick →  review recent log lines, faults and forge history
#                →  record an improvement analysis to the journal
#                →  if the analysis names a MISSING CAPABILITY that a new,
#                   additive tool would provide, forge it autonomously
#                   (sandbox-verified, loaded into config/custom_tools).
#
# Boundaries, deliberately kept: forged tools are ADDITIVE — they never touch
# orion_core (SecuritySanitiser.guard_forged_source rejects any generated
# module/test that imports orion_core or mutates orion.py/orion_core, called
# right after code generation in forge_tool() — see security.py), and
# patches to ORION's own source remain approval-gated through the
# SelfRepairAgent. Disable with ORION_SELF_IMPROVE=0.
# ──────────────────────────────────────────────────────────────────────────────

SELF_IMPROVE_DIR = CONFIG_DIR / "self_improvement"
IMPROVE_JOURNAL = SELF_IMPROVE_DIR / "journal.jsonl"


class ImprovementHeartbeat:
    """Periodic autonomous review-and-forge pulse."""

    # These were deliberately conservative because the Forge used to fail more
    # often than it succeeded, so every autonomous attempt was mostly a cost.
    # Mark II changed that calculation: repairs are aimed, the contract is
    # checked before activation, and failures teach the next attempt — so the
    # pulse can afford to be meaningfully more ambitious.  Both remain
    # env-overridable for anyone who wants the old caution back.
    INTERVAL_S = 25 * 60          # a tick every 25 minutes (was 45)
    FIRST_TICK_DELAY_S = 8 * 60   # let the session warm up first
    LOG_BUFFER_LINES = 220
    MAX_AUTO_FORGES_PER_SESSION = 5   # was 2
    # A tool idea gets one retry after its first failure (a fluke provider
    # hiccup or a repairable defect happens), but not unlimited attempts —
    # without this, a tool concept the model can never quite specify a
    # coherent contract for (observed in the wild: 'synthesis_enhancer',
    # 'emotion_recognizer' — both abstract enough that schema/run()/test
    # kept disagreeing) gets re-proposed and re-burns forge attempts EVERY
    # tick forever, since each tick reasons fresh from the same evidence
    # and nothing previously stopped it from reaching the same conclusion.
    # This cap is checked against the FULL persisted journal, not just this
    # session, so it survives a restart instead of resetting to zero.
    MAX_ATTEMPTS_PER_TOOL_NAME = 2
    # Review-only existing calculation contracts. A model cannot nominate
    # arbitrary core/device tools or replace the benchmark oracle.
    MEASURED_TARGETS = {"orion_latency_report": "latency_report_v1"}

    def __init__(self, bus: OrionBus, router: Any,
                 forge: ForgeOrchestrationManager,
                 telemetry: Any | None = None,
                 memory: Any | None = None) -> None:
        self.bus = bus
        self.router = router
        self.forge = forge
        self.telemetry = telemetry
        # MemoryAgent (optional): 'note' actions consolidate into the durable
        # knowledge tier, ADA-heartbeat style, so reviews compound over time.
        self.memory = memory
        self.enabled = os.getenv("ORION_SELF_IMPROVE", "1").strip().lower() not in {
            "0", "false", "no", "off"}
        self._stop = asyncio.Event()
        self._log_ring: list[str] = []
        self._forged_this_session = 0
        # True while ticks are paused because no text provider is available, so
        # the pause is announced once, not every 45-minute cycle.
        self._provider_paused = False
        # Passively cached latest snapshots from subsystems that already
        # broadcast their own findings on bus.dashboard_event — diagnostics'
        # FAIL/WARN checks and the cognitive loop's awareness digest used to
        # go nowhere near self-improvement, so every tick reasoned from a log
        # tail alone even though a structured health report already existed.
        # Only the latest matters (current state, not history), so each new
        # event on a channel simply overwrites the last one.
        self._last_diagnostics: dict[str, Any] | None = None
        self._last_awareness: dict[str, Any] | None = None
        SELF_IMPROVE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            bus.log.connect(self._on_log)      # passive tap on the live log
            bus.dashboard_event.connect(self._on_dashboard_event)
        except Exception:
            pass

    def _on_log(self, line: str) -> None:
        self._log_ring.append(str(line))
        if len(self._log_ring) > self.LOG_BUFFER_LINES:
            del self._log_ring[: len(self._log_ring) - self.LOG_BUFFER_LINES]

    def _on_dashboard_event(self, channel: str, payload: Any) -> None:
        if channel == "diagnostics" and isinstance(payload, dict):
            self._last_diagnostics = payload
        elif channel == "awareness" and isinstance(payload, dict):
            self._last_awareness = payload

    def stop(self) -> None:
        self._stop.set()

    # ── the loop ──────────────────────────────────────────────────────────────

    async def run(self) -> None:
        if not self.enabled:
            self.bus.log.emit("IMPROVE: self-improvement heartbeat disabled (ORION_SELF_IMPROVE=0).")
            return
        self.bus.log.emit(
            f"IMPROVE: heartbeat online — a self-review every {self.INTERVAL_S // 60} minutes; "
            "new capabilities are forged additively, core patches stay approval-gated.")
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=self.FIRST_TICK_DELAY_S)
            return                                   # stopped during warm-up
        except asyncio.TimeoutError:
            pass
        while not self._stop.is_set():
            # The improvement pulse REQUIRES text generation.  When no provider
            # is available, pause this optional work quietly instead of faulting
            # every cycle — deterministic tools and recovery keep running, and
            # the tick resumes automatically once a provider returns.
            if not self._text_generation_available():
                if not self._provider_paused:
                    self._provider_paused = True
                    self.bus.log.emit(
                        "IMPROVE: paused — no text provider available; "
                        "will resume automatically when one returns.")
            else:
                if self._provider_paused:
                    self._provider_paused = False
                    self.bus.log.emit("IMPROVE: text provider available again — resuming self-review.")
                from . import resource_governor
                if not resource_governor.optional_work_allowed():
                    # Memory is pressured: a review means model calls, test
                    # runs and maybe a forge — optional work that waits.
                    self.bus.log.emit("IMPROVE: self-review deferred — memory is under pressure.")
                    await asyncio.sleep(0)
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=self.INTERVAL_S)
                    except asyncio.TimeoutError:
                        pass
                    continue
                try:
                    await self.tick()
                except NoTextProviderError:
                    # Raced into degraded mode between the check and the call —
                    # treat as a graceful pause, not a fault.
                    self._provider_paused = True
                    self.bus.log.emit("IMPROVE: paused — text provider became unavailable mid-tick.")
                except Exception as exc:
                    self.bus.log.emit(f"IMPROVE: tick fault - {first_line(exc)}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.INTERVAL_S)
                return
            except asyncio.TimeoutError:
                continue

    async def _providers_ready(self) -> bool:
        """A text provider is usable now, or will be within a minute."""
        waiter = getattr(self.router, "wait_for_text", None)
        if not callable(waiter):
            return self._text_generation_available()
        try:
            return bool(await waiter(60.0))
        except Exception:
            return True

    def _text_generation_available(self) -> bool:
        """Best-effort check that the router has a usable text provider."""
        checker = getattr(self.router, "text_available", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return True   # never block ticks on a probe error
        return True

    def _failed_attempt_counts(self) -> dict[str, int]:
        """How many times each tool name has already failed forge
        verification, tallied from the FULL persisted journal (every tick
        ever recorded, not just this session) — so a repeatedly-doomed idea
        stays blocked across restarts instead of getting a clean slate every
        time the app relaunches. Never raises: a journal read failure just
        means an empty history, not a broken tick."""
        counts: dict[str, int] = {}
        try:
            if IMPROVE_JOURNAL.exists():
                for line in IMPROVE_JOURNAL.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    for item in entry.get("forged") or []:
                        if not isinstance(item, dict) or item.get("ok") is not False:
                            continue
                        if item.get("outage"):
                            # No provider was reachable: the idea never got a
                            # fair try, so it must not count toward abandoning
                            # it (two outages used to kill a good idea).
                            continue
                        name = str(item.get("tool") or "").strip()
                        if name:
                            counts[name] = counts.get(name, 0) + 1
        except Exception:
            return {}
        return counts

    def _failed_for_outage(self, name: str, result: Any) -> bool:
        """Whether a failed forge failed for want of a provider (and so was
        queued for retry) rather than because the generated code was wrong."""
        if getattr(result, "ok", False):
            return False
        return any(queued == name for queued, _plan in self.forge.pending_retries)

    @staticmethod
    def _covered_by(result: Any) -> str:
        """The existing tool named when the Forge declined to build a copy of it."""
        receipt = getattr(result, "receipt", None)
        if isinstance(receipt, dict) and receipt.get("state") == "covered":
            return str(receipt.get("covered_by") or "an existing tool")
        return ""

    async def tick(self) -> dict[str, Any]:
        """One self-review pass.  Returns the parsed analysis (for tests)."""
        entry: dict[str, Any] = {"at": utc_stamp(), "analysis": "", "actions": [], "forged": []}
        failed_counts = self._failed_attempt_counts()
        abandoned = {name for name, n in failed_counts.items()
                    if n >= self.MAX_ATTEMPTS_PER_TOOL_NAME}
        # Drain any sessions queued for retry by a prior tick's provider
        # outage BEFORE proposing anything new — these failed for lack of a
        # reachable provider, not a code defect, so they deserve another try
        # now rather than staying dead until something happens to re-propose
        # the identical tool from scratch.
        retries = list(self.forge.pending_retries)
        self.forge.pending_retries.clear()
        for name, plan in retries:
            if name in abandoned:
                entry["forged"].append({
                    "tool": name, "skipped": "repeated failures",
                    "previous_failures": failed_counts.get(name, 0)})
                continue
            if self._forged_this_session >= self.MAX_AUTO_FORGES_PER_SESSION:
                entry["forged"].append({"skipped": "session forge cap reached", "tool": name})
                continue
            if not await self._providers_ready():
                self.forge.pending_retries.append((name, plan))
                entry["forged"].append({"tool": name, "skipped": "providers cooling"})
                continue
            self.bus.log.emit(
                f"IMPROVE: retrying '{name}' — the previous attempt failed for "
                "lack of an available provider.")
            result = await self.forge.forge_tool(name, plan, autonomous=True)
            covered = self._covered_by(result)
            if covered:
                entry["forged"].append({"tool": name, "skipped": f"already covered by {covered}"})
                continue
            self._forged_this_session += 1
            entry["forged"].append({"tool": name, "ok": result.ok,
                                    "detail": first_line(result.text, 160), "retry": True,
                                    "outage": self._failed_for_outage(name, result)})
            if result.ok:
                await asyncio.to_thread(self._announce_new_capability, name, plan)

        if not self._has_actionable_evidence():
            # Nothing is failing: say so and stop. A model asked "what would
            # improve you?" with no evidence invents a need — six weeks of
            # this journal were the same two invented tools, proposed,
            # skipped and re-proposed every cycle.
            entry["analysis"] = "No actionable evidence this cycle — nothing forged."
            self._journal(entry)
            return entry
        context = self._compose_context(abandoned)
        measured_targets = json.dumps(self.MEASURED_TARGETS, sort_keys=True)
        prompt = (
            "You are O.R.I.O.N. running your autonomous self-improvement "
            "heartbeat (no user is in this turn). Review your own recent "
            "operational evidence and decide what would genuinely improve you. "
            "Examine it through EVERY lens the evidence below actually supports "
            "(skip any lens with nothing to show, but do not stop at the first "
            "one that does): UNMET REQUESTS — things the user asked for more "
            "than once that you could not do, the strongest reason to forge; "
            "diagnostics FAILs/WARNs pointing at something "
            "broken; telemetry tool-failure counts pointing at an unreliable or "
            "missing capability; recurring error patterns in the log; the "
            "awareness snapshot's focus/deadlines suggesting a capability gap; "
            "and previous self-reviews that noted something never acted on.\n\n"
            f"{context}\n\n"
            "Respond with STRICT JSON only:\n"
            '{"analysis": "<3-6 sentences: what is working, what is failing or '
            'missing, judged ONLY from the evidence above>",\n'
            ' "actions": [\n'
            '   {"type": "note", "text": "<an observation worth remembering>"},\n'
            '   {"type": "forge_tool", "tool_name": "<snake_case>", '
            '"plan": "<precise capability plan for a NEW additive tool>", '
            '"reason": "<the evidence that justifies it>"}\n'
            '   ,{"type": "improve_tool", "tool_name": "<registered target>", '
            '"benchmark": "<its registered benchmark>", "plan": "<specific measured goal>", '
            '"reason": "<operational evidence>"}\n'
            " ]}\n"
            "Rules: propose forge_tool ONLY when the evidence shows a concrete, "
            "recurring need a new tool would close; otherwise return notes (or "
            "an empty actions list). Never propose editing existing source. "
            "An improve_tool action only measures contained candidates and retains review evidence; "
            "it does not activate code or train a hosted model. Use it only for a concrete "
            f"observed defect in these exact target/benchmark pairs: {measured_targets}. "
            "Do not run benchmarks just because a slow AI provider or microphone is mentioned; "
            "optimising the timing-analysis calculation will not fix provider or audio latency."
        )
        data = await strict_json_turn(self.router, prompt)
        analysis = str(data.get("analysis") or "").strip()
        actions = [a for a in (data.get("actions") or []) if isinstance(a, dict)]
        entry["analysis"] = analysis
        entry["actions"] = actions
        # The review has now seen these unmet requests; one that does not
        # recur by its third showing stops steering the forge.
        gap_keys = list(getattr(self, "_gap_keys", []) or [])
        if gap_keys:
            entry["unmet_requests"] = len(gap_keys)
            try:
                from .capability_gaps import LEDGER
                await asyncio.to_thread(LEDGER.mark_presented, gap_keys)
            except Exception:
                pass
        for action in actions:
            if str(action.get("type")) == "note":
                self._consolidate_note(str(action.get("text") or ""))
                continue
            if str(action.get("type")) == "improve_tool":
                name = str(action.get("tool_name") or "")
                benchmark = str(action.get("benchmark") or "")
                seen = self.__dict__.setdefault("_measured_this_session", set())
                if self.MEASURED_TARGETS.get(name) != benchmark:
                    entry["forged"].append({"tool": name, "skipped": "unsupported measured contract"})
                    continue
                if name in seen or name in abandoned:
                    entry["forged"].append({"tool": name, "skipped": "measured attempt already made or abandoned"})
                    continue
                if self._forged_this_session >= self.MAX_AUTO_FORGES_PER_SESSION:
                    entry["forged"].append({"tool": name, "skipped": "session forge cap reached"})
                    break
                if not await self._providers_ready():
                    entry["forged"].append({"tool": name, "skipped": "providers cooling"})
                    break
                seen.add(name)
                result = await self.forge.improve_tool(
                    name, benchmark, generations=2, budget_seconds=240,
                    tool_plan=str(action.get("plan") or "")[:4096])
                self._forged_this_session += 1
                entry["forged"].append({"tool": name, "ok": result.ok, "measured": True,
                                        "activated": False, "detail": first_line(result.text, 160)})
                continue
            if str(action.get("type")) != "forge_tool":
                continue
            if self._forged_this_session >= self.MAX_AUTO_FORGES_PER_SESSION:
                entry["forged"].append({"skipped": "session forge cap reached"})
                break
            name = re.sub(r"[^a-z0-9_]+", "_", str(action.get("tool_name") or "").lower()).strip("_")
            plan = str(action.get("plan") or "").strip()
            if not name or len(plan) < 20:
                continue
            relative = self._abandoned_relative(name, abandoned)
            if name in abandoned or relative:
                self.bus.log.emit(
                    f"IMPROVE: skipping '{name}' — "
                    + (f"it has already failed forge verification "
                       f"{failed_counts.get(name, 0)} time(s)" if name in abandoned
                       else f"it is the same idea as '{relative}', which has "
                            f"already failed repeatedly")
                    + "; not repeating it.")
                entry["forged"].append({
                    "tool": name, "skipped": "repeated failures",
                    "previous_failures": failed_counts.get(relative or name, 0)})
                continue
            if not getattr(self, "_strong_evidence", True):
                # Log text alone: keep the idea as a note, do not build it.
                entry["forged"].append({"tool": name,
                                        "skipped": "weak evidence (log text only)"})
                self._consolidate_note(
                    f"Proposed tool idea (not built - only log text supported it): "
                    f"{action.get('tool_name')}: {first_line(action.get('plan', ''), 160)}")
                continue
            if not await self._providers_ready():
                # Every provider is rate limited: queue it for the next cycle
                # instead of starting a forge that can only fail and then
                # spending the next idea the same way (seen 2026-09-26).
                self.forge.pending_retries.append((name, plan))
                entry["forged"].append({"tool": name, "skipped": "providers cooling"})
                self.bus.log.emit(
                    f"IMPROVE: '{name}' waits for the next cycle — every text "
                    "provider is rate limited right now.")
                break
            self.bus.log.emit(
                f"IMPROVE: forging '{name}' autonomously — {first_line(action.get('reason', ''), 100)}")
            result = await self.forge.forge_tool(name, plan, autonomous=True)
            covered = self._covered_by(result)
            if covered:
                # Triage found an existing tool for it: nothing was built, so
                # it is neither a failure nor a use of the session's forges.
                entry["forged"].append({"tool": name, "skipped": f"already covered by {covered}"})
                continue
            self._forged_this_session += 1
            entry["forged"].append({"tool": name, "ok": result.ok,
                                    "detail": first_line(result.text, 160),
                                    "outage": self._failed_for_outage(name, result)})
            if result.ok:
                await asyncio.to_thread(self._announce_new_capability, name, plan)
        self._journal(entry)
        if analysis:
            signal = getattr(self.bus, "thought", None)
            if signal is not None:
                signal.emit({"id": f"review:{entry['at']}", "kind": "reflection",
                             "origin": "generated_reflection", "observed_at": entry["at"],
                             "text": first_line(analysis, 640),
                             "evidence": [f"Recorded Forge operations: {len(entry['forged'])}"]})
        return entry

    def _announce_new_capability(self, name: str, plan: str) -> None:
        """Say so when an autonomous forge succeeds — a failure was always
        spoken, a success only logged, so the user never learned they could
        now ask. When the tool answers a recorded unmet request, quote it and
        mark that need addressed (it returns if the user keeps asking)."""
        spoken = name.replace("_", " ")
        example = ""
        try:
            from .capability_gaps import LEDGER
            key = LEDGER.match_tool(name, plan)
            if key:
                cluster = LEDGER.mark_addressed(key, name) or {}
                example = (cluster.get("examples") or [""])[-1]
        except Exception:
            example = ""
        if example:
            line = (f"Earlier you asked me: '{example}'. I couldn't then, but I've built "
                    f"a tool for it - {spoken} - so just ask again.")
        else:
            line = f"I've built myself a new ability: {spoken}. {first_line(plan, 160)}"
        try:
            self.bus.log.emit(f"IMPROVE: new capability '{name}' is live.")
            self.bus.speak_request.emit(line)
        except Exception:
            pass

    #: Words that name a KIND of tool rather than what it is about.
    _GENERIC_NAME_WORDS = frozenset({
        "manager", "tracker", "generator", "monitor", "tool", "helper", "agent",
        "engine", "system", "service", "assistant", "analyzer", "analyser",
        "handler", "builder", "creator", "enhancer", "personal", "smart", "auto",
        "insight", "insights", "recognizer", "recogniser", "detector"})

    @classmethod
    def _concepts(cls, name: str) -> set[str]:
        return {word[:5] for word in str(name).lower().split("_")
                if len(word) >= 4 and word not in cls._GENERIC_NAME_WORDS}

    @classmethod
    def _abandoned_relative(cls, name: str, abandoned: set[str]) -> str:
        """An abandoned idea *name* is really a rename of, or ""."""
        concepts = cls._concepts(name)
        if not concepts:
            return ""
        for other in sorted(abandoned):
            if other != name and concepts & cls._concepts(other):
                return other
        return ""

    #: The improvement loop's and the forge's OWN log lines. Matched without
    #: regard to case or brackets: the forge logs "[FORGE] ...", which the old
    #: prefix list ("FORGE", "[Forge") missed — so every failed forge was read
    #: as fresh trouble and triggered another review and another forge.
    _OWN_CHATTER_RE = re.compile(
        r"(?i)^\s*\[?\s*(improve|forge|self[- _]?repair|repair|sandbox)\b")

    #: A log line that is evidence of something going wrong.
    _TROUBLE_RE = re.compile(
        r"(?i)\b(error|failed|failure|exception|traceback|crash|timed out|"
        r"could not|unavailable|refused)\b")

    def _has_actionable_evidence(self) -> bool:
        """Whether anything is actually wrong enough to reason about.

        Only a clean OBSERVATION counts as "nothing wrong": with no log seen
        and no diagnostics yet there is no basis for skipping, so the review
        runs as it always did.
        """
        # Whether what follows may justify FORGING, or only notes. A log line
        # containing "error" is in almost every session, and on its own it
        # led reviews to invent tools nobody asked for (the journal to 30 Sep
        # 2026: 40 failed forges, nine of them variations on "emotion").
        self._strong_evidence = False
        # Something the user asked for, and did not get, more than once — the
        # most direct evidence there is. Counted once per new occurrence.
        try:
            from .capability_gaps import LEDGER
            if LEDGER.recurring() and LEDGER.version != self.__dict__.get("_gaps_version_seen"):
                self._gaps_version_seen = LEDGER.version
                self._strong_evidence = True
                return True
        except Exception:
            pass
        if not self._log_ring and self._last_diagnostics is None:
            # Nothing observed yet: no basis for holding anything back.
            self._strong_evidence = True
            return True
        diag = self._last_diagnostics or {}
        if int(diag.get("fails", 0) or 0):
            self._strong_evidence = True
            return True
        if int(diag.get("warns", 0) or 0):
            return True
        if self.telemetry is not None:
            try:
                counters = self.telemetry.metrics.snapshot().get("counters", {})
                failures = {k: v for k, v in counters.items()
                            if k.startswith("tool.") and k.endswith(".failures")}
                seen = self.__dict__.setdefault("_failures_seen", {})
                # NEW failures only. "Any tool has ever failed" stayed true for
                # the rest of the session after the first failure, so every
                # tick was actionable and the review ran forever.
                fresh = any(v > seen.get(k, 0) for k, v in failures.items())
                seen.update(failures)
                if fresh:
                    self._strong_evidence = True
                    return True
            except Exception:
                pass
        for line in self._log_ring[-160:]:
            text = str(line)
            if self._OWN_CHATTER_RE.match(text):
                continue       # its own chatter is not evidence about ORION
            if self._TROUBLE_RE.search(text):
                return True
        return False

    def _consolidate_note(self, text: str) -> None:
        """Persist a self-review observation into the durable knowledge tier
        (deduplicated by content key) so lessons survive restarts."""
        text = text.strip()
        if not text or self.memory is None:
            return
        try:
            import hashlib
            key = f"self_review_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"
            self.memory.remember_knowledge(key, text[:500])
        except Exception:
            pass   # consolidation must never fault a tick

    # ── evidence + journal ────────────────────────────────────────────────────

    def _compose_context(self, abandoned: set[str] | None = None) -> str:
        sections = []
        # What the user wanted and did not get leads: everything below is
        # ORION's view of himself; this is the user's view of ORION.
        self._gap_keys: list[str] = []
        try:
            from .capability_gaps import LEDGER
            unmet, self._gap_keys = LEDGER.context_for_review()
            if unmet:
                sections.append(unmet)
        except Exception:
            pass
        if abandoned:
            sections.append(
                "=== ABANDONED TOOL IDEAS (do not re-propose these) ===\n"
                + "\n".join(f"- {name}" for name in sorted(abandoned))
                + "\nEach of these has already failed forge verification "
                f"{self.MAX_ATTEMPTS_PER_TOOL_NAME}+ times. They are almost "
                "certainly too abstract to give a coherent, testable contract "
                "(vague inputs/outputs) — propose something else, or nothing, "
                "instead of retrying them.")
        if self._last_diagnostics:
            fails = self._last_diagnostics.get("fails", 0)
            warns = self._last_diagnostics.get("warns", 0)
            problem_checks = [
                c for c in (self._last_diagnostics.get("checks") or [])
                if c.get("status") in {"FAIL", "WARN"}
            ]
            lines = [f"{fails} fail(s), {warns} warning(s) at {self._last_diagnostics.get('at', '?')}"]
            lines += [f"  {c.get('status')}: {c.get('name')} — {c.get('detail')}"
                     for c in problem_checks]
            sections.append("=== LATEST DIAGNOSTICS ===\n" + "\n".join(lines))
        if self._last_awareness:
            aw = self._last_awareness
            lines = [f"Focus: {aw.get('focus') or '(none)'}",
                    f"Active project: {aw.get('active_project') or '(none)'}"]
            deadlines = aw.get("deadlines") or []
            if deadlines:
                lines.append("Deadlines in view: " + "; ".join(str(d) for d in deadlines))
            sections.append("=== AWARENESS SNAPSHOT ===\n" + "\n".join(lines))
        if self.telemetry is not None:
            try:
                snapshot = self.telemetry.metrics.snapshot()
                counters = snapshot.get("counters", {})
                failures = sorted(
                    ((k, v) for k, v in counters.items()
                     if k.startswith("tool.") and k.endswith(".failures") and v > 0),
                    key=lambda kv: -kv[1],
                )[:10]
                total_calls = counters.get("tool.calls", 0)
                total_failures = counters.get("tool.failures", 0)
                lines = [f"Total tool calls: {total_calls:g}, total failures: {total_failures:g}"]
                if failures:
                    lines.append("Highest per-tool failure counts:")
                    lines += [f"  {name.removeprefix('tool.').removesuffix('.failures')}: "
                             f"{count:g}" for name, count in failures]
                gauges = snapshot.get("gauges", {})
                if "diag.fails" in gauges:
                    lines.append(f"Latest diagnostics gauge: diag.fails={gauges['diag.fails']:g}")
                sections.append("=== TELEMETRY (counters/gauges) ===\n" + "\n".join(lines))
            except Exception:
                pass
        if self._log_ring:
            sections.append("=== RECENT LOG (newest last) ===\n"
                            + "\n".join(self._log_ring[-120:]))
        repair_journal = CONFIG_DIR / "self_repair" / "journal.jsonl"
        try:
            if repair_journal.exists():
                # Faults in ORION — not the forge's own past failures, which
                # only taught the next review to propose the same tools again.
                tail = [line for line in tail_lines(repair_journal, 30)
                        if "ForgeSessionFailure" not in line][-12:]
                if tail:
                    sections.append("=== SELF-REPAIR JOURNAL (tail) ===\n" + "\n".join(tail))
        except Exception:
            pass
        # PREVIOUS SELF-REVIEWS are deliberately NOT shown any more. Each
        # review read the last six, found "repeated failures synthesising
        # tools", and concluded it should synthesise those tools again: the
        # evidence was its own output.
        forge_lines = [
            f"{s.tool_name}: {'ok' if s.activation_ok else 'failed'} ({len(s.errors)} error(s))"
            for s in self.forge.sessions.values()
        ]
        if forge_lines:
            sections.append("=== FORGE SESSIONS THIS RUN ===\n" + "\n".join(forge_lines))
        return "\n\n".join(sections) or "=== NO EVIDENCE CAPTURED YET THIS SESSION ==="

    #: The journal is re-read in full every tick (_failed_attempt_counts) and
    #: was append-only. Past this size the oldest reviews are folded into one
    #: summary line that keeps every recorded failure, so the abandon list —
    #: what stops a doomed idea being retried forever — is exactly preserved.
    JOURNAL_LIMIT_BYTES = 512 * 1024
    JOURNAL_KEEP_ENTRIES = 150

    def _journal(self, entry: dict[str, Any]) -> None:
        try:
            with IMPROVE_JOURNAL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            return
        self._compact_journal()

    def _compact_journal(self) -> None:
        try:
            if IMPROVE_JOURNAL.stat().st_size <= self.JOURNAL_LIMIT_BYTES:
                return
            lines = [line for line in IMPROVE_JOURNAL.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            older, kept = lines[:-self.JOURNAL_KEEP_ENTRIES], lines[-self.JOURNAL_KEEP_ENTRIES:]
            if not older:
                return
            failures: list[dict[str, Any]] = []
            for line in older:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                for item in entry.get("forged") or []:
                    if (isinstance(item, dict) and item.get("ok") is False
                            and not item.get("outage") and str(item.get("tool") or "").strip()):
                        failures.append({"tool": str(item["tool"]).strip(), "ok": False,
                                         "compacted": True})
            summary = {"at": utc_stamp(), "compacted": len(older),
                       "analysis": f"{len(older)} older review(s) compacted; "
                                   "their forge failures are kept below.",
                       "actions": [], "forged": failures}
            from .atomic_io import atomic_write_text
            atomic_write_text(IMPROVE_JOURNAL, "\n".join(
                [json.dumps(summary, ensure_ascii=False)] + kept) + "\n")
        except Exception:
            pass

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_minutes": self.INTERVAL_S // 60,
            "forged_this_session": self._forged_this_session,
            "journal": str(IMPROVE_JOURNAL),
        }
