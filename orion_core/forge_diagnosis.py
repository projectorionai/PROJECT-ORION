"""
Forge failure diagnosis (Mark II).

The original self-heal loop treated every sandbox failure identically: whatever
went wrong, it sent the module source and the raw error log back to the model
with "diagnose the failure and correct the module".  That is wrong in two
expensive ways.

First, it always blames the **module**.  A large share of forge failures are
caused by the generated *test* — it calls ``run()`` with arguments the schema
never declared, asserts on an exact string the tool was never asked to return,
or needs the network.  Sending only the module means the model rewrites
perfectly good code over and over and can never win; the session burns every
attempt and dies.

Second, a syntax error, a missing package, a timeout and a failed assertion
need completely different corrective instructions.  One generic prompt for all
four wastes the model's attention on the wrong hypothesis.

This module turns a raw failure into a :class:`Diagnosis`: what kind of failure
it is, **which artefact to repair**, and a specific directive for that class.
It is pure text analysis — no model call, no I/O — so it is cheap, deterministic
and fully testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .utils import canonical_module_id


class FailureClass(str, Enum):
    """The kinds of failure the Forge can tell apart."""

    MISSING_DEPENDENCY = "missing_dependency"
    SELF_IMPORT = "self_import"
    SYNTAX_ERROR = "syntax_error"
    CONTRACT_VIOLATION = "contract_violation"
    ASSERTION_FAILED = "assertion_failed"
    TIMEOUT = "timeout"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


class RepairTarget(str, Enum):
    """Which artefact the next repair attempt should rewrite."""

    MODULE = "module"
    TEST = "test"
    BOTH = "both"
    DEPENDENCY = "dependency"          # handled by pip, not by the model
    NONE = "none"                      # nothing a repair can do


@dataclass
class Diagnosis:
    """A classified sandbox failure plus the instruction to act on it."""

    failure_class: FailureClass
    target: RepairTarget
    detail: str = ""
    directive: str = ""
    missing_module: str = ""
    # False when the failure was environmental (a missing package we can just
    # install), so the orchestrator does not burn a code-repair attempt on it.
    consumes_attempt: bool = True

    @property
    def repairs_module(self) -> bool:
        return self.target in (RepairTarget.MODULE, RepairTarget.BOTH)

    @property
    def repairs_test(self) -> bool:
        return self.target in (RepairTarget.TEST, RepairTarget.BOTH)

    def summary(self) -> str:
        return f"{self.failure_class.value} → repair {self.target.value}: {self.detail}"[:240]


# ── raw-text signals ──────────────────────────────────────────────────────────

_MISSING_MODULE_PATTERNS = (
    r"No module named '([^']+)'",
    r'No module named "([^"]+)"',
    r"ModuleNotFoundError: No module named ([^\s]+)",
)

_SYNTAX_RE = re.compile(r"(?m)^\s*(SyntaxError|IndentationError|TabError)\b")
_ASSERTION_RE = re.compile(r"(?m)^\s*AssertionError\b|^\s*assert\s")
_TIMEOUT_RE = re.compile(r"(?i)subprocess timeout|timed out after|TimeoutExpired")

# A test calling run() wrongly, or a module not honouring the contract.
_SIGNATURE_RE = re.compile(
    r"(?i)unexpected keyword argument|missing \d+ required positional|"
    r"takes \d+ positional argument|got multiple values for argument"
)
_MISSING_EXPORT_RE = re.compile(
    r"(?i)(?:module|object) .*has no attribute '(get_tool_schema|run)'|"
    r"name '(get_tool_schema|run)' is not defined|"
    r"must export get_tool_schema"
)
_CONFORMANCE_RE = re.compile(r"(?i)^CONTRACT:", re.MULTILINE)

_NETWORK_RE = re.compile(
    r"(?i)ConnectionError|ConnectionRefused|ConnectionReset|NewConnectionError|"
    r"MaxRetryError|gaierror|Name or service not known|getaddrinfo failed|"
    r"URLError|SSLError|CertificateError|socket\.timeout|ReadTimeout|"
    r"Temporary failure in name resolution|HTTPError|Failed to establish a new connection"
)
_FILESYSTEM_RE = re.compile(
    r"(?i)PermissionError|FileNotFoundError|IsADirectoryError|NotADirectoryError|"
    r"OSError: \[Errno 13\]|Access is denied"
)

# The last traceback frame's filename — tells us whether the module or the test
# actually blew up.  This is the highest-value signal in the whole classifier.
_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


def _joined(error_log: Sequence[str] | str | None) -> str:
    if error_log is None:
        return ""
    if isinstance(error_log, str):
        return error_log
    return "\n".join(str(line) for line in error_log)


def parse_missing_module(text: str) -> str | None:
    """Top-level package name from an import error, or None."""
    if not text:
        return None
    for pattern in _MISSING_MODULE_PATTERNS:
        match = re.search(pattern, str(text))
        if match:
            raw = match.group(1).strip().strip("'\"")
            if raw:
                return raw.split(".")[0]
    return None


def failing_artefact(text: str, tool_name: str) -> RepairTarget | None:
    """Which file the deepest traceback frame points at.

    The sandbox stages the module under the tool's name and the harness under
    ``<tool>_test.py``, so the filename in the last frame is a direct answer to
    "whose fault is this?".  Returns None when no frame is attributable.
    """
    frames = _FRAME_RE.findall(str(text or ""))
    if not frames:
        return None
    canonical = canonical_module_id(tool_name)
    for filename, _line in reversed(frames):
        stem = re.sub(r"\.py$", "", str(filename).replace("\\", "/").split("/")[-1])
        folded = canonical_module_id(stem)
        if not folded:
            continue
        if folded.endswith("test") and canonical and canonical in folded:
            return RepairTarget.TEST
        if folded == f"{canonical}test":
            return RepairTarget.TEST
        if canonical and canonical in folded:
            return RepairTarget.MODULE
    return None


def _self_import(missing: str, tool_name: str) -> bool:
    """True when the 'missing module' is the forged tool itself.

    ``EnhancedResearchModule`` whose test does ``import
    enhanced_research_module`` is the same artefact; installing that name from
    PyPI at best fails and at worst shadows the generated code with an
    unrelated package.
    """
    if not missing:
        return False
    own = {
        canonical_module_id(tool_name),
        canonical_module_id(f"{tool_name}_tool"),
        canonical_module_id(f"{tool_name}_probe"),
    }
    return canonical_module_id(missing) in own


# ── directives ────────────────────────────────────────────────────────────────
# One targeted instruction per failure class.  These are appended to the repair
# prompt so the model spends its attention on the right hypothesis instead of
# re-deriving what went wrong from a wall of stderr.

_DIRECTIVES: dict[FailureClass, str] = {
    FailureClass.SYNTAX_ERROR: (
        "The source does not parse. Fix the syntax error at the reported line — "
        "check unbalanced brackets, an unterminated string, a stray fence marker "
        "or inconsistent indentation. Return the COMPLETE corrected file, never a "
        "fragment or a diff."
    ),
    FailureClass.CONTRACT_VIOLATION: (
        "The module and its schema disagree, or the test calls run() with "
        "arguments the schema never declared. Make them consistent: every key in "
        "the schema's properties must be an accepted keyword argument of run(), "
        "every entry in 'required' must appear in properties, and the test must "
        "call run() using exactly those keyword names. Prefer adapting the TEST "
        "to the declared schema rather than widening the module."
    ),
    FailureClass.ASSERTION_FAILED: (
        "An assertion failed. Decide honestly WHICH side is wrong before you "
        "rewrite anything: an over-specific test (asserting an exact string, a "
        "float equality, an ordering the tool never promised, or live data) is a "
        "test defect, and you must relax the test rather than distort the module "
        "to satisfy it. Only change the module when the tool genuinely computes "
        "the wrong answer. State which you concluded in one comment line."
    ),
    FailureClass.TIMEOUT: (
        "Verification exceeded its time budget. Remove anything unbounded: "
        "sleeps, retry loops, polling, large downloads, or waiting on input. The "
        "test must exercise run() with small, local, deterministic arguments and "
        "finish in well under a second."
    ),
    FailureClass.NETWORK: (
        "Verification failed on a network call. The sandbox has no dependable "
        "network access, so the TEST must not make one: exercise run() with "
        "local arguments, or stub the network boundary. Keep the module's real "
        "network capability intact — it is only the test that must run offline."
    ),
    FailureClass.FILESYSTEM: (
        "The tool or its test touched a path it may not use. Confine all file "
        "access to the current working directory, create any file the test needs "
        "inside the test itself, and clean up afterwards. Never read or write "
        "outside the working directory."
    ),
    FailureClass.RUNTIME_ERROR: (
        "The tool raised at runtime. Anchor the fix to the actual exception and "
        "the line it names — do not guess. run() must handle its own errors and "
        "always return a readable string rather than propagating an exception."
    ),
    FailureClass.SELF_IMPORT: (
        "The test tried to import the tool under a name the sandbox does not "
        "stage. The module is available as both '<tool_name>.py' and "
        "'<tool_name>_tool.py' in the working directory — import one of those "
        "exactly, or load it by path with importlib. This is a TEST defect: do "
        "not change the module."
    ),
    FailureClass.UNKNOWN: (
        "The failure is not self-explanatory. Re-read the error output, state "
        "the most likely root cause in one comment line, and make the smallest "
        "change that addresses it."
    ),
}


def diagnose(
    error_log: Sequence[str] | str | None,
    tool_name: str,
    *,
    timed_out: bool = False,
    already_installed: Iterable[str] = (),
) -> Diagnosis:
    """Classify a sandbox failure and say what to repair.

    Args:
        error_log: the sandbox's error lines (or one joined blob).
        tool_name: the tool being forged — needed to recognise self-imports and
            to attribute traceback frames.
        timed_out: True when the subprocess was killed by its time budget.
        already_installed: packages this session has already tried to install,
            so a second identical ModuleNotFoundError is not diagnosed as an
            installable dependency again (that was an install loop).

    Returns:
        A :class:`Diagnosis` carrying the failure class, repair target and a
        class-specific directive.
    """
    text = _joined(error_log)
    installed = {str(p).lower() for p in already_installed}

    def _built(cls: FailureClass, target: RepairTarget, detail: str,
               *, missing: str = "", consumes: bool = True) -> Diagnosis:
        return Diagnosis(
            failure_class=cls,
            target=target,
            detail=detail[:240],
            directive=_DIRECTIVES.get(cls, _DIRECTIVES[FailureClass.UNKNOWN]),
            missing_module=missing,
            consumes_attempt=consumes,
        )

    # 1. Timeout — the process never got to report anything useful.
    if timed_out or _TIMEOUT_RE.search(text):
        return _built(FailureClass.TIMEOUT, RepairTarget.BOTH,
                      "verification exceeded its time budget")

    # 2. A missing import: either the tool importing itself (a test defect) or a
    #    genuine third-party package (an environment fix, not a code repair).
    missing = parse_missing_module(text)
    if missing:
        if _self_import(missing, tool_name):
            return _built(FailureClass.SELF_IMPORT, RepairTarget.TEST,
                          f"the test imports '{missing}', which is the tool itself")
        if missing.lower() not in installed:
            return _built(FailureClass.MISSING_DEPENDENCY, RepairTarget.DEPENDENCY,
                          f"missing package '{missing}'", missing=missing,
                          consumes=False)
        # Already tried to install it and it is still missing — the name is
        # wrong, not the environment.  Make the model pick a real package.
        return _built(
            FailureClass.MISSING_DEPENDENCY, RepairTarget.BOTH,
            f"'{missing}' could not be installed and is still missing",
            missing=missing,
        )

    # 3. Syntax — attribute it to whichever file the parser named.
    if _SYNTAX_RE.search(text):
        target = failing_artefact(text, tool_name) or RepairTarget.MODULE
        return _built(FailureClass.SYNTAX_ERROR, target, "the source does not parse")

    # 4. Explicit contract problems: our conformance probe, a missing export, or
    #    a call-signature mismatch between the schema, run() and the test.
    if _CONFORMANCE_RE.search(text) or _MISSING_EXPORT_RE.search(text):
        return _built(FailureClass.CONTRACT_VIOLATION, RepairTarget.MODULE,
                      "the module does not satisfy the tool contract")
    if _SIGNATURE_RE.search(text):
        return _built(FailureClass.CONTRACT_VIOLATION, RepairTarget.BOTH,
                      "run() was called with arguments it does not accept")

    # 5. Environmental failures that look like code bugs if you squint.
    if _NETWORK_RE.search(text):
        return _built(FailureClass.NETWORK, RepairTarget.TEST,
                      "verification attempted a network call")
    if _FILESYSTEM_RE.search(text):
        target = failing_artefact(text, tool_name) or RepairTarget.BOTH
        return _built(FailureClass.FILESYSTEM, target,
                      "verification touched a path it may not use")

    # 6. A failed assertion is the class where the TEST is most often at fault,
    #    so both artefacts go to the model with an instruction to judge which.
    if _ASSERTION_RE.search(text):
        return _built(FailureClass.ASSERTION_FAILED, RepairTarget.BOTH,
                      "an assertion in the verification harness failed")

    # 7. A named exception we could not place — repair whichever file raised it.
    if re.search(r"(?m)^\s*[A-Z][A-Za-z]*(?:Error|Exception)\b", text):
        target = failing_artefact(text, tool_name) or RepairTarget.MODULE
        return _built(FailureClass.RUNTIME_ERROR, target,
                      "the tool raised at runtime")

    if not text.strip():
        return _built(FailureClass.UNKNOWN, RepairTarget.BOTH,
                      "verification failed without producing any output")
    return _built(FailureClass.UNKNOWN, RepairTarget.BOTH, "unclassified failure")


def error_signature(error_log: Sequence[str] | str | None, limit: int = 160) -> str:
    """A stable, comparable fingerprint for a failure.

    Paths, line numbers, hex addresses, temp-directory names and quoted literals
    all vary between otherwise identical failures, so they are normalised away.
    Two runs that failed for the same reason produce the same signature, which
    is what lets the lesson store deduplicate and count recurrences.
    """
    text = _joined(error_log)
    # Prefer the last exception line — the actual cause, not the frames above it.
    exception_lines = [
        line.strip() for line in text.splitlines()
        if re.match(r"^\s*[A-Za-z_.]*(?:Error|Exception|Warning)\b", line.strip())
    ]
    if exception_lines:
        core = exception_lines[-1]
    else:
        tail = text.strip().splitlines()
        core = tail[-1] if tail else ""
    core = re.sub(r'File "[^"]+"', 'File "<path>"', core)
    core = re.sub(r"0x[0-9a-fA-F]+", "<addr>", core)
    core = re.sub(r"\bline \d+", "line <n>", core)
    core = re.sub(r"[/\\][^\s'\"]+", "<path>", core)
    core = re.sub(r"\b\d+\b", "<n>", core)
    core = re.sub(r"\s+", " ", core).strip()
    return core[:limit]


__all__ = [
    "Diagnosis",
    "FailureClass",
    "RepairTarget",
    "diagnose",
    "error_signature",
    "failing_artefact",
    "parse_missing_module",
]
