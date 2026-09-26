"""
Tool concurrency classification and batch planning.

ORION was single-threaded by construction. The Gemini Live channel routinely
returns SEVERAL function calls in one turn — "search the web, check my
resources and recall what we discussed" is one turn with three calls — and the
worker executed them in a plain ``for`` loop, awaiting each in sequence. Three
independent two-second lookups took six seconds instead of two, every time.

The reason it was never parallelised is that most of ORION's tools are *not*
safe to run concurrently: they drive one physical machine. Two ``open_app``
calls racing, or a window focus interleaved with a click, produce nonsense.
Correct parallelism therefore needs a per-tool judgement, and that judgement is
what lives here.

Three classes:

    PARALLEL   Read-only or otherwise independent — a query, a lookup, a
               capture, an analysis. Any number may run at once.
    SERIAL     Touches shared state, the desktop, the filesystem or an external
               account. Runs alone, in the order the model asked for.
    EXCLUSIVE  Lifecycle and self-modification — restart, self-repair, forging
               a tool, running a whole plan. Nothing else runs alongside it.

**SERIAL is the default.** A tool nobody has classified — including every tool
ORION forges for itself at runtime, whose side effects are unknowable — is
treated as unsafe to parallelise. Speed is never worth a race on the user's
actual machine, so the fast path is opt-in and the safe path is automatic.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class ToolClass(str, Enum):
    """How a tool may be scheduled relative to other tools."""

    PARALLEL = "parallel"
    SERIAL = "serial"
    EXCLUSIVE = "exclusive"


# Read-only: a query, retrieval, capture or analysis. Running several at once
# cannot corrupt anything because none of them writes.
#
# `vision_analyse` / `capture_screen` / `vision_verify` are here deliberately:
# VolatileScreenGrabber keeps its mss instance in threading.local precisely
# because mss objects are not thread-safe, so concurrent captures from separate
# worker threads are safe by construction.
PARALLEL_TOOLS: frozenset[str] = frozenset({
    # retrieval and recall
    "web_search", "find_files", "recall_conversation", "conversation_recall",
    "query_intelligence", "transcript", "second_brain", "awareness",
    # introspection and status
    "capabilities", "display_info", "resource_status", "token_usage",
    "patch_notes", "self_changes", "code_changes", "diagnostics", "proactive_check",
    # perception (thread-local capture handles)
    "vision_analyse", "capture_screen", "vision_verify",
    # domain knowledge
    "cyber_knowledge", "programming_knowledge", "neuro_knowledge",
    "founder_knowledge",
    # analysis and intelligence
    "research", "product_research", "tiktok_intel", "instagram_intel",
    "competitor_intel", "creator_intel", "business_advisor", "brand_growth",
    "geo", "breach_check", "executive", "agent_dispatch",
    # Perception reporting (status/scene/events). Its start/stop branches are
    # lifecycle and are demoted by WRITE_ACTIONS below.
    "perception",
})

# Deliberately absent from the set above, despite being read-only: `reason` and
# `strategy`. A single reasoning pass already fans out to a panel of
# specialists, each of which may take several instrument readings
# simultaneously, and `strategy` can end by calling `reason`. Batching two of
# those would multiply that fan-out straight into provider rate limits, so both
# fall through to the SERIAL default on purpose rather than by oversight.

# Lifecycle and self-modification. These change what ORION *is* — the process,
# its source, or its tool table — so anything running alongside them could be
# observing a half-changed system.
EXCLUSIVE_TOOLS: frozenset[str] = frozenset({
    "shutdown_orion", "restart_orion", "self_repair", "forge",
    "backup", "cleanup_review", "execute_plan", "autoplan", "protocol",
})

# Hard ceiling on how many tools may run at once. The limit is not CPU — most
# of these are network- or IO-bound — but blast radius: a batch that fails
# should be small enough to reason about, and provider rate limits are real.
DEFAULT_MAX_PARALLEL = 6

# ── the multi-action problem ─────────────────────────────────────────────────
#
# The tables above classify TOOLS, but concurrency safety is a property of the
# CALL. Several of ORION's tools are read-only in most of their modes and carry
# one branch that writes: `second_brain` mostly recalls but can ingest,
# `awareness` mostly reports but can add a task, `find_files` only searches
# unless you ask it to open what it found. Classifying those as PARALLEL — as
# this module originally did — is right for the branch you thought about and
# wrong for the branch you didn't.
#
# `awareness` is the sharp one: add_priority/add_task/complete_task are
# read-modify-write over one cognitive-state file, so two batched together lose
# an update outright. That is exactly the corruption this module exists to
# prevent, waved through because the classification was one level too coarse.
#
# So a call whose arguments select a write branch is demoted to SERIAL. The
# tables stay deliberately explicit rather than pattern-matched: guessing which
# action names write from their spelling is how the branch you didn't think
# about gets waved through a second time.
WRITE_ACTIONS: dict[str, frozenset[str]] = {
    "second_brain":  frozenset({"ingest"}),
    "awareness":     frozenset({"add_priority", "priority", "add_task", "task",
                                "complete_task", "done", "remove_task", "delete_task",
                                "dismiss_task", "drop_task", "clear_overdue",
                                "remove_overdue", "clear_overdue_tasks"}),
    "research":      frozenset({"start", "queue", "stop", "reviewed", "paper"}),
    "creator_intel": frozenset({"add_creator", "signup"}),
    "executive":     frozenset({"schedule", "book", "track", "track_project"}),
    # Opening or revealing launches a program; searching does not.
    "find_files":    frozenset({"open", "reveal", "show", "show_in_folder"}),
    # Reading the screen is free; anything that takes the camera is not —
    # two at once fight over one device handle.
    "vision_analyse": frozenset({"scan", "inspect", "identify", "identify_object",
                                 "scan_object", "what_is_this", "count", "count_items",
                                 "read_label", "scan_text", "pcb", "electronics",
                                 "inspect_pcb", "scan_pcb", "circuit_board",
                                 "inspect_electronics", "camera", "webcam"}),
    # Reading what perception has seen is free; starting or stopping the loop
    # owns the camera and a background task, and must not race itself.
    # Rules, the detector download and the overlay are writes; so are the
    # one-off looks, because each may switch the camera on.
    "perception":    frozenset({"start", "watch", "on", "stop", "off", "add_rule", "rule",
                               "when", "remove_rule", "delete_rule", "enable_rule",
                               "disable_rule", "autostart", "watch_on_startup",
                               "install_detector", "install_objects", "install_model",
                               "overlay", "camera_view", "show", "analyse", "analyze",
                               "numbers", "read", "measure", "grid", "detect",
                               "objects", "identify", "pose", "body", "gestures"}),
}

# Flags that turn an observation into an action. `find_files` with open=true
# calls os.startfile: a tool that reads the disk suddenly launches an
# application, which is not something to do twice at once by accident.
WRITE_FLAGS: dict[str, tuple[str, ...]] = {
    "find_files": ("open", "open_first", "reveal"),
}


def max_parallel() -> int:
    """Concurrency ceiling, overridable with ORION_MAX_PARALLEL_TOOLS."""
    try:
        value = int(os.getenv("ORION_MAX_PARALLEL_TOOLS", "").strip()
                    or DEFAULT_MAX_PARALLEL)
    except ValueError:
        return DEFAULT_MAX_PARALLEL
    return max(1, min(16, value))


def parallelism_enabled() -> bool:
    """False when ORION_PARALLEL_TOOLS is switched off.

    A single switch back to the old strictly-sequential behaviour, so a
    suspected concurrency problem can be ruled in or out in one restart rather
    than by bisecting the classification table.
    """
    return os.getenv("ORION_PARALLEL_TOOLS", "1").strip().lower() not in {
        "0", "false", "no", "off"}


def selects_write_branch(name: str, args: Mapping[str, Any] | None) -> bool:
    """True when *args* steer a read-only tool onto a writing branch."""
    if not args:
        return False
    actions = WRITE_ACTIONS.get(name)
    if actions:
        action = str(args.get("action") or args.get("mode") or "").strip().lower()
        if action in actions:
            return True
    return any(args.get(flag) for flag in WRITE_FLAGS.get(name, ()))


def classify(name: str, args: Mapping[str, Any] | None = None) -> ToolClass:
    """How this call may be scheduled. Unknown tools are SERIAL, never PARALLEL.

    Pass *args* whenever they are available. Without them a multi-action tool
    is judged on its most common mode, which is the coarse behaviour this
    module shipped with; with them, a call that selects a write branch is
    demoted to SERIAL even though the tool is nominally read-only.
    """
    tool = str(name or "").strip()
    if tool in EXCLUSIVE_TOOLS:
        return ToolClass.EXCLUSIVE
    if tool in PARALLEL_TOOLS:
        return ToolClass.SERIAL if selects_write_branch(tool, args) else ToolClass.PARALLEL
    return ToolClass.SERIAL


def plan_batches(
    names: Sequence[str],
    limit: int | None = None,
    args: Sequence[Mapping[str, Any] | None] | None = None,
) -> list[list[int]]:
    """Group calls into execution batches, as lists of indices into *names*.

    Consecutive PARALLEL calls collapse into one batch that runs concurrently;
    every SERIAL or EXCLUSIVE call gets a batch of its own. Because only
    *adjacent* parallel calls merge, the relative order of side-effecting work
    is exactly what the model asked for — a batch never reorders anything, it
    only removes waiting between calls that cannot interfere.

    Example::

        ["web_search", "resource_status", "open_app", "find_files"]
        → [[0, 1], [2], [3]]

    With parallelism disabled, or a limit of 1, every call gets its own batch —
    identical to the original sequential loop.

    *args* is an optional sequence of argument dicts positionally matching
    *names*. Supply it when the caller has them: it is what lets a call like
    ``awareness{action: add_task}`` be recognised as a write and kept out of a
    batch, rather than being judged on its tool name alone.
    """
    ceiling = max(1, int(limit if limit is not None else max_parallel()))
    if not parallelism_enabled():
        ceiling = 1

    batches: list[list[int]] = []
    current: list[int] = []
    for index, name in enumerate(names):
        call_args = args[index] if args is not None and index < len(args) else None
        if classify(name, call_args) is ToolClass.PARALLEL and ceiling > 1:
            current.append(index)
            if len(current) >= ceiling:
                batches.append(current)
                current = []
            continue
        if current:
            batches.append(current)
            current = []
        batches.append([index])
    if current:
        batches.append(current)
    return batches


def describe_plan(names: Sequence[str], batches: Iterable[Sequence[int]]) -> str:
    """One-line human summary of a batch plan, for the log."""
    parts: list[str] = []
    for batch in batches:
        members = [str(names[i]) for i in batch]
        parts.append(f"[{' ‖ '.join(members)}]" if len(members) > 1 else members[0])
    return " → ".join(parts)


__all__ = [
    "DEFAULT_MAX_PARALLEL",
    "EXCLUSIVE_TOOLS",
    "PARALLEL_TOOLS",
    "WRITE_ACTIONS",
    "WRITE_FLAGS",
    "ToolClass",
    "classify",
    "describe_plan",
    "max_parallel",
    "parallelism_enabled",
    "plan_batches",
    "selects_write_branch",
]
