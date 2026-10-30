"""
Sandboxed Verification Harness (Mark II).

Accepts raw generated tool code and test code, writes them to a staging
directory, then verifies the artefacts in isolated subprocesses.

Verification runs in ordered stages:

    1. CONFORMANCE — a deterministic probe we control.  It imports the module
       (so import-time side effects and syntax errors surface immediately),
       then checks the tool contract by inspection: the schema's shape, and
       whether ``run()``'s signature can actually accept the parameters the
       schema advertises.  Previously a tool could pass verification with exit
       code 0 and still explode the first time the dispatcher called it,
       because nothing ever compared the schema against the function.  Problems
       are reported as ``CONTRACT: …`` lines, which the diagnosis layer
       recognises as a contract violation rather than a mystery.

    2. EXAMPLES — when supplied, the independently planned shape and three
       expected answers, exercised with private files and fake HTTP.
    3. TEST — the model's own harness, run only once the earlier checks pass. Its
       failures are therefore about behaviour, not about the contract, which
       makes them far easier to classify and repair.

Two environment defects that masqueraded as generated-code bugs are also fixed
here: the time budget now adapts to what the module imports (a tool that pulls
in ``requests`` is no longer killed mid-import and reported as a code fault),
and the child process is forced to UTF-8 I/O (on Windows the inherited cp1252
stdout made any test printing '✓' die with ``UnicodeEncodeError``, which the
old pipeline dutifully sent to the model as a code error to fix).
"""

from __future__ import annotations

import asyncio
import ast
import copy
import importlib.resources as importlib_resources
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .data import ToolResult
from .utils import module_name_variants

# ── memory limit ─────────────────────────────────────────────────────────────
# The time budget above bounds hang time but never bounded memory: a forged
# tool with a runaway allocation (an accidental infinite list-append, a huge
# comprehension) could consume all available RAM before the timeout even
# fires. A polling watchdog (psutil — already a dependency) kills the
# subprocess if its RSS crosses this cap, same spirit as the timeout: a
# best-effort backstop, not a hardened OS-level sandbox.
def _memory_limit_mb() -> int:
    try:
        return max(128, min(int(os.getenv("ORION_FORGE_SANDBOX_MEMORY_MB", "512")), 2048))
    except ValueError:
        return 512


SANDBOX_MEMORY_LIMIT_MB = _memory_limit_mb()
_MEMORY_POLL_INTERVAL_S = 0.25
MAX_OUTPUT_BYTES = 4 * 1024 * 1024

# ── time budget ───────────────────────────────────────────────────────────────
# A flat 15 seconds killed legitimate tools mid-import: `import pandas` alone
# can outlast it on a cold filesystem, and the failure looked exactly like an
# infinite loop in generated code.  The budget now reflects what the module
# actually imports.

BASE_TIMEOUT_S = 15.0
MAX_TIMEOUT_S = 90.0
CONFORMANCE_TIMEOUT_CAP_S = 30.0

_IMPORT_COST_S: tuple[tuple[str, float], ...] = (
    ("torch", 45.0), ("transformers", 45.0), ("tensorflow", 45.0),
    ("selenium", 40.0), ("playwright", 40.0),
    ("matplotlib", 25.0), ("scipy", 20.0), ("sklearn", 30.0),
    ("requests", 25.0), ("httpx", 25.0), ("aiohttp", 25.0), ("urllib", 20.0),
    ("socket", 20.0), ("smtplib", 20.0), ("ftplib", 20.0),
    ("pandas", 20.0), ("openpyxl", 15.0), ("numpy", 12.0),
    ("PIL", 12.0), ("cv2", 25.0), ("bs4", 12.0), ("lxml", 15.0),
)

_IMPORT_RE = re.compile(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)")


def budget_for(*sources: str) -> float:
    """Time budget in seconds for verifying *sources*.

    Costs are taken as a maximum rather than a sum: importing both ``requests``
    and ``numpy`` costs roughly what the slower one costs, not the total.
    """
    imported: set[str] = set()
    for source in sources:
        imported.update(_IMPORT_RE.findall(str(source or "")))
    surcharge = max(
        (cost for name, cost in _IMPORT_COST_S if name in imported),
        default=0.0,
    )
    return min(MAX_TIMEOUT_S, BASE_TIMEOUT_S + surcharge)


# ── the conformance probe ─────────────────────────────────────────────────────
# Runs inside the sandbox with the tool name as argv[1].  It never CALLS run() —
# invoking a tool with fabricated arguments could have real side effects — it
# only inspects the signature.  Exits 0 and prints the schema as JSON on
# success; exits 1 having printed 'CONTRACT: …' lines on failure.
#
# The rules themselves live in forge_contract.py, which the loader also uses
# in-process.  The probe cannot IMPORT that module (the staging directory is
# deliberately isolated from orion_core, so generated code can never reach
# ORION's own package), so its source is shipped into the sandbox as text and
# this main block is appended.  One implementation, two callers, no drift.

_PROBE_MAIN = '''

# ── probe entry point (appended by sandbox.probe_source) ─────────────────────
if __name__ == "__main__":
    import importlib.util, json, pathlib, sys
    from orion_forge_fixtures import block_network

    tool_name = sys.argv[1]
    candidates = [pathlib.Path(tool_name + "_tool.py"), pathlib.Path(tool_name + ".py")]
    module_path = next((p for p in candidates if p.exists()), None)
    if module_path is None:
        print("CONTRACT: the staged module file could not be found", file=sys.stderr)
        raise SystemExit(1)

    spec = importlib.util.spec_from_file_location(tool_name + "_conformance", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        with block_network():
            spec.loader.exec_module(module)
    except BaseException as exc:
        print("CONTRACT: import raised %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)

    found = []
    tool_schema = None
    if not hasattr(module, "get_tool_schema"):
        found.append("the module must export get_tool_schema()")
    else:
        try:
            with block_network():
                tool_schema = module.get_tool_schema()
        except BaseException as exc:
            found.append("get_tool_schema() raised %s: %s" % (type(exc).__name__, exc))

    if tool_schema is not None:
        found.extend(contract_problems(tool_schema, getattr(module, "run", None)))
    elif not found:
        found.append("get_tool_schema() returned nothing")

    if found:
        for problem in found:
            print("CONTRACT: " + problem, file=sys.stderr)
        raise SystemExit(1)

    try:
        print(json.dumps(tool_schema))
    except (TypeError, ValueError):
        print("CONTRACT: the schema is not JSON-serialisable", file=sys.stderr)
        raise SystemExit(1)
    print("conformance OK")
'''

CONFORMANCE_SCRIPT_NAME = "_orion_conformance.py"
EXAMPLES_SCRIPT_NAME = "_orion_examples.py"
_EXAMPLES_MAIN = '''
import importlib.util, json, pathlib, sys
from orion_forge_fixtures import block_network, run_examples, validate_plan, plan_fingerprint

plan = validate_plan(json.loads(pathlib.Path("_orion_examples.json").read_text(encoding="utf-8")))
path = pathlib.Path(sys.argv[1] + "_tool.py")
spec = importlib.util.spec_from_file_location(sys.argv[1] + "_examples", path)
module = importlib.util.module_from_spec(spec)
try:
    with block_network():
        spec.loader.exec_module(module)
        run_examples(module, plan["examples"], plan=plan)
except BaseException as exc:
    print("EXAMPLES: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
    raise SystemExit(1)
print("ORION_EXAMPLES_COMPLETE:" + json.dumps({"count": len(plan["examples"]),
                                               "plan": plan_fingerprint(plan)}, sort_keys=True))
'''


def probe_source() -> str:
    """The conformance probe script: forge_contract.py plus a main block."""
    from . import forge_contract
    # In a PyInstaller build the module lives in the PYZ archive and
    # ``forge_contract.__file__`` points at a source path that was never copied
    # into ``_internal``.  Prefer a real adjacent/resource file when present,
    # then use inspect for source checkouts and test doubles.  The standalone
    # build explicitly ships the source as data (see build_standalone.py), so
    # the first two paths also keep the probe and the loader on one contract.
    candidates: list[Path] = []
    module_file = getattr(forge_contract, "__file__", "")
    if module_file:
        candidates.append(Path(module_file))
    try:
        resource = importlib_resources.files("orion_core").joinpath("forge_contract.py")
        candidates.append(Path(str(resource)))
    except Exception:
        pass
    rules = ""
    for candidate in candidates:
        try:
            rules = candidate.read_text(encoding="utf-8")
            if "def contract_problems" in rules:
                break
        except (OSError, UnicodeError):
            continue
    if "def contract_problems" not in rules:
        try:
            rules = inspect.getsource(forge_contract)
        except (OSError, TypeError, IOError) as exc:
            raise RuntimeError(
                "Forge contract source is unavailable in this build; "
                "rebuild ORION with forge_contract.py included as data."
            ) from exc
    return rules + _PROBE_MAIN


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
    # Which stage produced this outcome: "conformance", "test", or "" when the
    # harness itself failed before either ran.
    stage: str = ""
    # True when the subprocess was killed by its time budget — the diagnosis
    # layer treats that very differently from a code fault.
    timed_out: bool = False
    # True when the subprocess was killed for exceeding SANDBOX_MEMORY_LIMIT_MB
    # — distinct from timed_out: a real diagnosis, not "needs a longer budget".
    memory_exceeded: bool = False
    # The budget actually granted, for logs and for explaining a timeout.
    timeout_s: float = BASE_TIMEOUT_S
    typecheck: str = "not_run"
    isolation: str = "process"  # process isolation is NOT an OS security boundary
    examples_verified: bool = False
    plan: dict[str, Any] | None = None

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
            "stage": self.stage,
            "timed_out": self.timed_out,
            "memory_exceeded": self.memory_exceeded,
            "timeout_s": self.timeout_s,
            "typecheck": self.typecheck,
            "isolation": self.isolation,
            "examples_verified": self.examples_verified,
            "plan": copy.deepcopy(self.plan),
        }


class SandboxVerificationHarness:
    """Isolated two-stage verification runner for generated Python tool code."""

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
        timeout_seconds: float | None = None,
        *,
        require_container: bool = False,
        container_image: str | None = None,
        plan: dict[str, Any] | None = None,
    ) -> VerificationOutcome:
        """Give each attempt private files and reap its child on cancellation.

        A shallow worker copy retains injected runners used by embedders/tests,
        without mutating this harness while another tool is being verified.
        Trusted improvement benchmarks require containment and type checking
        independently of the owner's ordinary Forge settings. The explicit
        image can be an immutable local Docker image identifier; it is never
        pulled implicitly.
        """
        if type(require_container) is not bool:
            raise ValueError("require_container must be a boolean")
        if container_image is not None and not require_container:
            raise ValueError("an explicit verifier image requires container verification")
        worker = copy.copy(self)
        worker._cancelled = threading.Event()
        # One attempt cannot switch execution back onto the host between its
        # checks if settings are changed while a worker is awaiting a process.
        worker._isolation_mode = "container" if require_container else os.getenv("ORION_FORGE_SANDBOX", "process").strip().lower()
        worker._typecheck_policy = "required" if require_container else os.getenv("ORION_FORGE_TYPECHECK", "auto").strip().lower()
        worker._container_image = container_image or os.getenv("ORION_FORGE_CONTAINER_IMAGE", "orion-forge-verifier:local")
        worker._schema_authoritative = worker._isolation_mode == "container"
        if plan is not None:
            from .forge_fixtures import validate_plan
            worker._plan_contract = validate_plan(plan)
        with tempfile.TemporaryDirectory(prefix="attempt-", dir=self.staging_dir) as directory:
            worker.staging_dir = Path(directory)
            task = asyncio.create_task(worker._verify_staged(
                tool_code, test_code, tool_name, timeout_seconds))
            try:
                return await asyncio.shield(task)
            finally:
                worker._cancelled.set()
                # to_thread cancellation does not stop a subprocess. Wait for
                # the polling runner to kill/reap it before removing its files.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue

    async def _verify_staged(
        self,
        tool_code: str,
        test_code: str,
        tool_name: str,
        timeout_seconds: float | None = None,
    ) -> VerificationOutcome:
        """
        Verify a generated tool in isolated subprocesses.

        Stage 1 runs the deterministic conformance probe (imports the module,
        validates the schema, checks that ``run()`` can accept the parameters
        the schema declares).  Stage 2 runs the model's own test harness.  The
        first stage to fail is the outcome, so the caller always learns about
        a broken contract before it learns about a failed assertion.

        Args:
            tool_code: Raw Python source code for the tool module
            test_code: Raw Python source code for the test harness
            tool_name: Identifier for the tool (sanitised into filename)
            timeout_seconds: Explicit budget; when None it adapts to the imports
                found in the sources.

        Returns:
            VerificationOutcome with passed flag, schema, stage and error logs
        """
        self.bus.log.emit(f"[FORGE] Sandbox: starting verification for '{tool_name}'")
        budget = budget_for(tool_code, test_code)
        outcome = VerificationOutcome(passed=False, tool_name=tool_name, timeout_s=budget)
        outcome.isolation = getattr(self, "_isolation_mode", os.getenv("ORION_FORGE_SANDBOX", "process").strip().lower())
        start_time = asyncio.get_running_loop().time()

        try:
            if timeout_seconds is not None:
                if isinstance(timeout_seconds, bool):
                    raise ValueError("verification timeout must be a finite number from 0 to 90 seconds")
                requested = float(timeout_seconds)
                if not math.isfinite(requested) or not 0 < requested <= MAX_TIMEOUT_S:
                    raise ValueError("verification timeout must be greater than 0 and at most 90 seconds")
                budget = requested
                outcome.timeout_s = budget
            if outcome.isolation not in {"process", "container"}:
                raise ValueError("ORION_FORGE_SANDBOX must be process or container")
            from .forge_isolation import customer_verification_mode
            customer_mode = customer_verification_mode()
            if customer_mode and outcome.isolation != "container":
                raise ValueError("Customer Forge verification requires ORION_FORGE_SANDBOX=container")
            outcome.stage = "syntax"
            if max(len(tool_code), len(test_code)) > 1_048_576:
                raise ValueError("Forge source exceeds the 1 MiB verification limit")
            ast.parse(tool_code, filename=f"{tool_name}_tool.py")
            test_tree = ast.parse(test_code, filename=f"{tool_name}_test.py")
            plan = getattr(self, "_plan_contract", None)
            if plan is not None:
                from .forge_fixtures import validate_plan
                plan = validate_plan(plan)
                outcome.plan = plan
            await asyncio.to_thread(
                self._write_artefacts,
                tool_name,
                tool_code,
                test_code,
                **({"plan": plan} if plan is not None else {}),
            )

            # Never inherit mypy configuration/plugins from the working tree.
            # 'required' fails closed if the optional verifier is unavailable;
            # 'auto' keeps older installations usable and reports the omission.
            policy = getattr(self, "_typecheck_policy", os.getenv("ORION_FORGE_TYPECHECK", "auto").strip().lower())
            if policy not in {"auto", "required", "off"}:
                raise ValueError("ORION_FORGE_TYPECHECK must be auto, required or off")
            if customer_mode:
                policy = "required"
            import importlib.util
            has_mypy = outcome.isolation == "container" or importlib.util.find_spec("mypy") is not None
            outcome.typecheck = "disabled" if policy == "off" else "unavailable"
            if policy == "required" and not has_mypy:
                outcome.stage = "typecheck"
                outcome.error_log = ["mypy is required but unavailable in the verifier interpreter"]
                return outcome
            if policy != "off" and has_mypy:
                outcome.stage = "typecheck"
                values = await asyncio.to_thread(
                    self._run_script, "_orion_typecheck.py", budget,
                    f"{self._safe_name(tool_name)}_tool.py")
                out, err, code, timed, memory = values
                outcome.typecheck = "passed" if code == 0 and not timed and not memory else "failed"
                if outcome.typecheck == "failed":
                    outcome.stdout, outcome.stderr, outcome.exit_code = out, err, code
                    outcome.timed_out, outcome.memory_exceeded = timed, memory
                    outcome.error_log = self._error_lines(code, out, err, timed, budget, memory)
                    return outcome

            # ── Stage 1: contract conformance ────────────────────────────────
            conformance_budget = min(budget, CONFORMANCE_TIMEOUT_CAP_S)
            stdout_text, stderr_text, exit_code, timed_out, mem_exceeded = await asyncio.to_thread(
                self._run_script,
                CONFORMANCE_SCRIPT_NAME,
                conformance_budget,
                self._safe_name(tool_name),
            )
            if exit_code != 0 or timed_out or mem_exceeded:
                outcome.stage = "conformance"
                outcome.stdout, outcome.stderr = stdout_text, stderr_text
                outcome.exit_code, outcome.timed_out = exit_code, timed_out
                outcome.memory_exceeded = mem_exceeded
                outcome.error_log = self._error_lines(
                    exit_code, stdout_text, stderr_text, timed_out, conformance_budget,
                    mem_exceeded)
                self.bus.log.emit(
                    f"[FORGE] Sandbox: ✗ contract check failed for '{tool_name}' "
                    f"({'memory limit' if mem_exceeded else 'timeout' if timed_out else f'exit {exit_code}'})"
                )
                return outcome

            outcome.schema = self._parse_schema_from_output(stdout_text)
            from .forge_contract import schema_problems
            if ("conformance OK" not in stdout_text.splitlines()
                    or schema_problems(outcome.schema)[0]):
                outcome.stage = "conformance"
                outcome.error_log = ["CONTRACT: verifier exited without a completed, valid schema check"]
                return outcome

            # The planner's fixed expectations are checked before the model's
            # own test. This uses the same bounded child/container runner;
            # generated code is never imported into the orchestration process.
            if plan is not None:
                outcome.stage = "examples"
                out, err, code, timed, memory = await asyncio.to_thread(
                    self._run_script, EXAMPLES_SCRIPT_NAME, budget, self._safe_name(tool_name))
                if code != 0 or timed or memory:
                    outcome.stdout, outcome.stderr, outcome.exit_code = out, err, code
                    outcome.timed_out, outcome.memory_exceeded = timed, memory
                    outcome.error_log = self._error_lines(code, out, err, timed, budget, memory)
                    return outcome
                from .forge_fixtures import plan_fingerprint
                expected_receipt = {"count": 3, "plan": plan_fingerprint(plan)}
                receipts = []
                for line in out.splitlines():
                    if line.startswith("ORION_EXAMPLES_COMPLETE:"):
                        try:
                            receipts.append(json.loads(line.split(":", 1)[1]))
                        except (ValueError, TypeError):
                            pass
                if receipts != [expected_receipt]:
                    outcome.error_log = ["EXAMPLES: verifier exited without completing all three fixed-plan examples"]
                    outcome.stdout, outcome.stderr, outcome.exit_code = out, err, code
                    return outcome
                outcome.examples_verified = True

            # ── Final stage: the model's own test harness ────────────────────
            test_script = f"{self._safe_name(tool_name)}_test.py"
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name.startswith("test_") for n in ast.walk(test_tree)):
                test_args = ("_orion_pytest.py", budget, test_script)
            else:
                test_args = (test_script, budget)
            stdout_text, stderr_text, exit_code, timed_out, mem_exceeded = await asyncio.to_thread(
                self._run_script,
                *test_args,
            )
            outcome.stage = "test"
            outcome.stdout = stdout_text
            outcome.stderr = stderr_text
            outcome.exit_code = exit_code
            outcome.timed_out = timed_out
            outcome.memory_exceeded = mem_exceeded

            if exit_code == 0 and not timed_out and not mem_exceeded:
                outcome.passed = True
                # Container evaluation uses the conformance probe's actual
                # schema. A generated test (or a benchmark JSON receipt) must
                # not replace it with a different printed dictionary.
                self.bus.log.emit(
                    f"[FORGE] Sandbox: ✓ verification passed for '{tool_name}' "
                    f"(contract + {'examples + ' if plan is not None else ''}test, budget {budget:.0f}s)"
                )
            else:
                outcome.error_log = self._error_lines(
                    exit_code, stdout_text, stderr_text, timed_out, budget, mem_exceeded)
                self.bus.log.emit(
                    f"[FORGE] Sandbox: ✗ test failed for '{tool_name}' "
                    f"({'memory limit' if mem_exceeded else 'timeout' if timed_out else f'exit {exit_code}'})"
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
            end_time = asyncio.get_running_loop().time()
            outcome.duration_ms = (end_time - start_time) * 1000.0

        return outcome

    # ── staging ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_name(tool_name: str) -> str:
        return "".join(c if c.isalnum() or c in "_" else "_" for c in str(tool_name or ""))

    def _write_artefacts(
        self,
        tool_name: str,
        tool_code: str,
        test_code: str,
        *,
        plan: dict[str, Any] | None = None,
    ) -> None:
        """
        Write tool, test and conformance-probe code to the staging directory.

        Args:
            tool_name: Base name for the tool
            tool_code: Raw tool source code
            test_code: Raw test source code
        """
        safe_name = self._safe_name(tool_name)
        test_path = self.staging_dir / f"{safe_name}_test.py"
        test_path.write_text(test_code, encoding="utf-8")
        (self.staging_dir / CONFORMANCE_SCRIPT_NAME).write_text(
            probe_source(), encoding="utf-8")
        (self.staging_dir / "_orion_pytest.py").write_text(
            "import pytest, sys\nraise SystemExit(pytest.main([sys.argv[1], "
            "'-q', '--noconftest', '-p', 'no:cacheprovider', '--confcutdir=.']))\n",
            encoding="utf-8")
        (self.staging_dir / "_orion_typecheck.py").write_text(
            "import mypy.api, sys, os\n"
            "out, err, code = mypy.api.run(['--config-file', '_orion_mypy.ini', "
            "'--no-incremental', '--cache-dir', os.devnull, '--follow-imports=skip', '--ignore-missing-imports', "
            "'--check-untyped-defs', sys.argv[1]])\n"
            "print(out); print(err, file=sys.stderr); raise SystemExit(code)\n",
            encoding="utf-8")
        (self.staging_dir / "_orion_mypy.ini").write_text("[mypy]\n", encoding="utf-8")
        if plan is not None:
            from .forge_fixtures import validate_plan
            (self.staging_dir / "_orion_examples.json").write_text(
                json.dumps(validate_plan(plan), ensure_ascii=False, allow_nan=False), encoding="utf-8")
            (self.staging_dir / EXAMPLES_SCRIPT_NAME).write_text(_EXAMPLES_MAIN, encoding="utf-8")
        # The verification fixtures a generated test may import (a private
        # folder, fake HTTP, a frozen clock, a fake bus), and the module the
        # deterministic import repair names. Written before the module
        # aliases, so a tool that is itself named like it keeps its own source.
        from .forge_fixtures import FIXTURE_MODULE_NAME, fixture_source
        (self.staging_dir / f"{FIXTURE_MODULE_NAME}.py").write_text(
            fixture_source(), encoding="utf-8")
        # Stage the module source under EVERY name the generated test might
        # import it by — its bare name, a `_tool` suffix, AND the snake_case
        # form of a CamelCase tool name.  Generated tests routinely
        # `import enhanced_research_module` for a tool the Forge named
        # 'EnhancedResearchModule'; without staging that alias the import
        # fails, gets misread as a missing pip dependency, and — worst case —
        # pip installs an unrelated PyPI package of the same name that then
        # shadows the generated code (the 'dependency_resolver' incident,
        # 2026-07-17; the 'enhanced_research_module' failure, same day).
        # Staging sits first on sys.path, so these aliases always win over
        # site-packages.
        names = set(module_name_variants(tool_name)) or {f"{safe_name}_tool"}
        names.add(f"{safe_name}_tool")
        for name in names:
            if name:
                (self.staging_dir / f"{name}.py").write_text(
                    tool_code, encoding="utf-8")
        if getattr(self, "_isolation_mode", os.getenv("ORION_FORGE_SANDBOX", "process").strip().lower()) == "container" and os.name != "nt":
            # The verifier runs as nobody. A 0700 TemporaryDirectory would
            # otherwise make every read-only bind mount inaccessible to it.
            self.staging_dir.chmod(0o755)
            for path in self.staging_dir.iterdir():
                path.chmod(0o644)

    # ── execution ────────────────────────────────────────────────────────────

    def _run_script(
        self,
        script_name: str,
        timeout_seconds: float,
        *args: str,
    ) -> tuple[str, str, int, bool, bool]:
        """
        Execute a staged script in a subprocess, watched by a memory-limit
        poll alongside the existing wall-clock timeout.

        stdout/stderr are redirected to temp files rather than PIPE — a
        polling wait loop reading a bounded OS pipe risks deadlock if the
        child fills the buffer before being polled; files have no such limit
        and are trivial to read back once the process exits.

        Args:
            script_name: File name inside the staging directory
            timeout_seconds: Subprocess wall-clock timeout
            args: Extra command-line arguments for the script

        Returns:
            Tuple of (stdout, stderr, exit_code, timed_out, memory_exceeded)
        """
        isolation = getattr(self, "_isolation_mode", os.getenv("ORION_FORGE_SANDBOX", "process").strip().lower())
        if isolation != "container":
            from .execution_policy import require_foreground
            blocked = require_foreground("unrestricted_forge_verifier_process")
            if blocked is not None:
                return "", blocked.text, -1, False, False
        script_path = self.staging_dir / script_name
        stem = self._safe_name(script_path.stem)
        stdout_path = self.staging_dir / f"{stem}.stdout.tmp"
        stderr_path = self.staging_dir / f"{stem}.stderr.tmp"
        memory_limit_bytes = SANDBOX_MEMORY_LIMIT_MB * 1024 * 1024

        proc: subprocess.Popen | None = None
        timed_out = False
        memory_exceeded = False
        container_name = ""
        result = ("", "verification did not complete", -1, False, False)
        try:
            cancelled = getattr(self, "_cancelled", None)
            if cancelled is not None and cancelled.is_set():
                return "", "verification cancelled", -1, False, False
            # From source: this interpreter. In the installed app: ORION.exe in
            # its explicit --forge-verify mode (forge_runtime), which used to be
            # refused outright - every Forge session there failed (8-10 Oct).
            from .forge_runtime import verifier_command
            command = verifier_command(str(script_path), args)
            isolation = getattr(self, "_isolation_mode", os.getenv("ORION_FORGE_SANDBOX", "process").strip().lower())
            if isolation == "container":
                from .forge_isolation import container_command
                container_name = "orion-forge-" + uuid.uuid4().hex
                command = container_command(
                    self.staging_dir, script_name, args, name=container_name,
                    image=getattr(self, "_container_image", os.getenv("ORION_FORGE_CONTAINER_IMAGE", "orion-forge-verifier:local")),
                    memory_mb=SANDBOX_MEMORY_LIMIT_MB)
            with open(stdout_path, "w", encoding="utf-8") as out_f, \
                 open(stderr_path, "w", encoding="utf-8") as err_f:
                if isolation != "container":
                    blocked = require_foreground("unrestricted_forge_verifier_process")
                    if blocked is not None:
                        return "", blocked.text, -1, False, False
                proc = subprocess.Popen(
                    command,
                    cwd=str(self.staging_dir),
                    stdout=out_f,
                    stderr=err_f,
                    env=self._child_env(),
                    creationflags=0x08000000 if os.name == "nt" else 0,
                )
                deadline = time.monotonic() + float(timeout_seconds)
                while True:
                    try:
                        proc.wait(timeout=_MEMORY_POLL_INTERVAL_S)
                        break
                    except subprocess.TimeoutExpired:
                        if cancelled is not None and cancelled.is_set():
                            self._kill(proc)
                            break
                        if stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_OUTPUT_BYTES:
                            self._kill(proc)
                            raise ValueError("verification output limit exceeded")
                        if time.monotonic() >= deadline:
                            timed_out = True
                            self._kill(proc)
                            break
                        if self._rss_bytes(proc.pid) > memory_limit_bytes:
                            memory_exceeded = True
                            self._kill(proc)
                            break
            # A child may print too much and exit between polls. It must not
            # pass merely because bounded reads would hide the excess output.
            if stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_OUTPUT_BYTES:
                raise ValueError("verification output limit exceeded")
            exit_code = proc.returncode if proc.returncode is not None else -1
            stdout_text = self._read_and_clear(stdout_path)
            stderr_text = self._read_and_clear(stderr_path)
            result = stdout_text, stderr_text, exit_code, timed_out, memory_exceeded
        except Exception as exc:
            # The harness itself failing (e.g. can't spawn the interpreter)
            # must not crash the forge pipeline — surface it as a verification
            # failure like any other, via the caller's own except block.
            if proc is not None:
                self._kill(proc)
            stdout_text = self._read_and_clear(stdout_path)
            stderr_text = self._read_and_clear(stderr_path)
            result = stdout_text, f"Verifier failed: {type(exc).__name__}: {exc}\n{stderr_text}", -1, timed_out, memory_exceeded
        finally:
            if container_name:
                # Killing docker.exe alone leaves its container running.
                try:
                    cleanup = subprocess.run(["docker", "rm", "--force", container_name],
                                   capture_output=True, text=True, timeout=5,
                                   env=self._child_env(), creationflags=0x08000000 if os.name == "nt" else 0)
                    # --rm may already have removed a finished container. Any
                    # other error leaves its retirement unconfirmed.
                    if cleanup.returncode and "No such container" not in cleanup.stderr:
                        raise RuntimeError("Docker could not confirm verifier container removal")
                except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                    result = (result[0], result[1] + "\nContainer cleanup failed: " +
                              f"{type(exc).__name__}: {exc}", -1, result[3], result[4])
        return result

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        """Kill the process and any children it spawned (best-effort)."""
        try:
            import psutil
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    @staticmethod
    def _rss_bytes(pid: int) -> int:
        """Current resident-set size of *pid*, or 0 if it can't be sampled
        (process already gone, psutil unavailable, permission denied) — a
        sampling failure must never be mistaken for exceeding the limit."""
        try:
            import psutil
            return int(psutil.Process(pid).memory_info().rss)
        except Exception:
            return 0

    @staticmethod
    def _read_and_clear(path: Path) -> str:
        try:
            with path.open("rb") as stream:
                text = stream.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")
        except Exception:
            text = ""
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return text

    def _child_env(self) -> dict[str, str]:
        """Environment for the verification subprocess.

        ``PYTHONIOENCODING`` is the important one: on Windows the child would
        otherwise inherit a cp1252 stdout, so any generated test printing a tick
        or an arrow died with ``UnicodeEncodeError`` — a harness defect that the
        old pipeline reported to the model as a bug in its code, and which no
        amount of code repair could ever fix.
        """
        # Generated code runs here BEFORE anyone has read it, so it gets the
        # environment it needs to run — not ORION's secrets. Anything that
        # names a key, token, password or credential is withheld.
        secretish = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
                     "AUTH", "COOKIE", "PRIVATE", "_SID")
        inherited = {k: v for k, v in os.environ.items()
                     if not any(mark in k.upper() for mark in secretish)}
        return {
            **inherited,
            "PYTHONPATH": str(self.staging_dir),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }

    @staticmethod
    def _error_lines(
        exit_code: int,
        stdout_text: str,
        stderr_text: str,
        timed_out: bool,
        budget: float,
        memory_exceeded: bool = False,
    ) -> list[str]:
        """Compose the error log a failure hands to the diagnosis layer."""
        if memory_exceeded:
            header = f"Subprocess killed: exceeded {SANDBOX_MEMORY_LIMIT_MB}MB memory limit"
        elif timed_out:
            header = f"Subprocess timeout after {budget:.0f}s"
        else:
            header = f"Exit code: {exit_code}"
        return (
            [header]
            + (stderr_text.split("\n") if stderr_text else [])
            + (stdout_text.split("\n") if stdout_text else [])
        )

    def _parse_schema_from_output(self, stdout_text: str) -> dict[str, Any] | None:
        """
        Extract tool schema JSON from subprocess output.

        The conformance probe prints the schema as a single JSON line; a
        generated test may also emit one via `print(json.dumps(schema))`.

        Args:
            stdout_text: Raw stdout from subprocess

        Returns:
            Parsed schema dict, or None if parsing fails
        """
        if not stdout_text:
            return None

        for line in stdout_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        return parsed
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
                f"({outcome.duration_ms:.1f}ms, contract + test).\n"
                f"Schema: {json.dumps(outcome.schema, indent=2)}"
            )
            return ToolResult(msg, ok=True)
        stage = f" at the {outcome.stage} stage" if outcome.stage else ""
        msg = (
            f"✗ Verification failed for '{outcome.tool_name}'{stage}. "
            f"Errors:\n" + "\n".join(outcome.error_log[-20:])  # Last 20 lines
        )
        return ToolResult(msg, ok=False)
