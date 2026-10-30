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


# Scheduling safety is not a guarantee that a tool leaves the user's desktop
# alone. Background execution has its own execution_policy scope; web_search
# uses a retrieval transport there, while an interactive call may open a tab.
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
    "query_intelligence", "transcript", "second_brain", "awareness", "process_file",
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
    # Personal goals: reading them back is free; every other action writes
    # goals.db and is held SERIAL by READ_ONLY_ACTIONS below.
    "goals",
    "project_rehearsal", "context_capsule", "experiment_notebook",
    # Perception reporting (status/scene/events). Its start/stop branches are
    # lifecycle and are demoted by WRITE_ACTIONS below.
    "perception",
    # Mark XXXX.4: the security checks only read. The Cybersecurity Agent was
    # refused them ("not read-only") and answered from generic advice while
    # ORION said an audit was running. Writes stay SERIAL (READ_ONLY_ACTIONS).
    "security_watch", "antivirus",
})

# Deliberately absent from the set above, despite being read-only: `reason` and
# `strategy`. A single reasoning pass already fans out to a panel of
# specialists, each of which may take several instrument readings
# simultaneously, and `strategy` can end by calling `reason`. Batching two of
# those would multiply that fan-out straight into provider rate limits, so both
# fall through to the SERIAL default on purpose rather than by oversight.
#
# `sing` is SERIAL on purpose too: it plays a song through ORION's one voice
# `conversation_recording` stays SERIAL: session start/stop/forget share one
# microphone tap and must not overlap, including calls through Live's gateway.
# output, and two songs (or a song and another sound) must never overlap.

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
    # remember/record are ingest's aliases at the service (audit 11 Oct 2026).
    "second_brain":  frozenset({"ingest", "remember", "record"}),
    "awareness":     frozenset({"add_priority", "priority", "add_task", "task",
                                "complete_task", "done", "complete_tasks", "mark_done",
                                "finish_tasks", "completed", "list_tasks", "tasks",
                                "remove_task", "delete_task",
                                "dismiss_task", "drop_task", "clear_overdue",
                                "remove_overdue", "clear_overdue_tasks"}),
    "research":      frozenset({"start", "queue", "stop", "reviewed", "paper"}),
    "creator_intel": frozenset({"add_creator", "signup"}),
    # A meeting summary writes memory and the knowledge graph; plan/workflow
    # rewrite the shared cognitive-state file (awareness's lost-update risk).
    "executive":     frozenset({"schedule", "book", "track", "track_project",
                                "meeting_summary", "summarise_meeting", "summarize_meeting",
                                "minutes", "plan", "workflow", "plan_workflow"}),
    # Each of these writes a setting or a stored record (audit 11 Oct 2026):
    # a provider budget, a custom monitor name, long-term memory, a new
    # baseline of ORION's own source snapshot.
    "token_usage":   frozenset({"set_budget", "budget_set"}),
    "display_info":  frozenset({"set_name", "rename", "name"}),
    "conversation_recall": frozenset({"compress"}),
    "code_changes":  frozenset({"snapshot", "baseline", "remember"}),
    # Locate/fly changes the scene; directions opens a browser; route changes
    # the Globe. Administrative records, weather and distance only observe.
    "geo":           frozenset({"locate", "find", "fly", "directions", "navigate",
                                "route", "show_route", "route_on_globe"}),
    # Opening or revealing launches a program; searching does not.
    "find_files":    frozenset({"open", "reveal", "show", "show_in_folder"}),
    # Reading the screen is free; anything that takes the camera is not —
    # two at once fight over one device handle.
    "vision_analyse": frozenset({"scan", "inspect", "identify", "identify_object",
                                 "forget_camera", "forget_scene", "forget_scenes",
                                 "watch_camera_past", "camera_past", "recent_camera",
                                 "scan_object", "what_is_this", "count", "count_items",
                                 "read_label", "scan_text", "pcb", "electronics",
                                 "inspect_pcb", "scan_pcb", "circuit_board",
                                 "inspect_electronics", "camera", "webcam",
                                 "watch_camera", "watch_me", "watch_webcam",
                                 # The camera tools (dispatch_vision._CAMERA_TOOL_ACTIONS):
                                 # they take the camera AND save or record. Left out when
                                 # they were added, so they ran PARALLEL (audit 6 Oct 2026,
                                 # A07); a test now holds every alias there to this table.
                                 "photo", "take_photo", "picture", "take_picture",
                                 "snapshot", "selfie", "save_photo", "document",
                                 "scan_document", "scan_page", "scan_paper",
                                 "scan_letter", "read_code", "qr", "qr_code", "barcode",
                                 "scan_code", "read_qr", "read_barcode", "codes",
                                 "record", "record_video", "record_clip", "video_clip",
                                 "film"}),
    # Reading what perception has seen is free; starting or stopping the loop
    # owns the camera and a background task, and must not race itself.
    # Rules, the detector download and the overlay are writes; so are the
    # one-off looks, because each may switch the camera on.
    "perception":    frozenset({"start", "watch", "on", "stop", "off", "add_rule", "rule",
                               "forget_memory", "forget_camera", "forget_scenes", "forget",
                               "history_on", "history_off",
                               "when", "remove_rule", "delete_rule", "enable_rule",
                               "disable_rule", "autostart", "watch_on_startup",
                               "cloud_on", "cloud_off",
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
    # Each reflexes/shortcuts/learned alias reaches these same write branches.
    # Gate the arguments even on another action: an observational caller must
    # never carry a shortcut mutation through a future alias.
    "diagnostics": ("promote", "forget"),
}

# Named operator specialists can take their playbook's bounded actions. Keep
# this lightweight table aligned with PLAYBOOKS.acting (covered by a test),
# rather than importing the provider/agent runtime into this classifier.
# AgentManager refuses acting auto-routes, so auto remains observational.
ACTING_SPECIALISTS: frozenset[str] = frozenset({"files"})
WRITE_ARG_VALUES: dict[str, dict[str, frozenset[str]]] = {
    "agent_dispatch": {"agent": ACTING_SPECIALISTS},
}

# An omitted geo action is locate (dispatch_web.geo_tool), which moves the
# Globe. Its mode argument selects a directions transport, not the action.
_DEFAULT_ACTIONS: dict[str, str] = {"geo": "locate"}

# Tools whose DEFAULT is a write and whose write aliases keep growing are
# listed the other way round: only these actions are read-only, and anything
# else - including no action at all - is a write. `research` with no action
# STARTS a background run, and it gained a dozen write aliases (conduct, deep,
# dossier, investigate, cancel, add, ...) that a denylist never caught up with;
# `transcript` with no action EXPORTS a file (audit 11 Oct 2026). Reflex
# learning and the Live batch scheduler both trust this module's answer.
READ_ONLY_ACTIONS: dict[str, frozenset[str]] = {
    "context_capsule": frozenset({"list", "show"}),
    "experiment_notebook": frozenset({"list", "show", "analyse"}),
    "process_file": frozenset({"recall", "history", "memory"}),
    "research":   frozenset({"status", "progress", "agenda", "programme", "list",
                             "findings", "evidence", "claims", "validate",
                             "validate_sources", "sources", "opportunities",
                             "opportunity", "gaps", "neuroscience_pilot"}),
    "transcript": frozenset({"path", "where", "recall", "search"}),
    "security_watch": frozenset({"audit", "score", "how_secure", "hardening", "health",
                                 "status", "report", "check", "network", "dashboard",
                                 "netsec", "posture", "phone_activity", "phone_log",
                                 "phone_steps"}),
    "antivirus": frozenset({"status", "threats", "describe", "file", "downloads"}),
    # goals.ACTIONS has a dozen write aliases (log, record, track, done ...)
    # and gains more; only these read. "plan" writes milestones, so it is not here.
    "goals":      frozenset({"list", "overview", "status", "all", "goals", "show", "details",
                             "describe", "get", "detail", "review", "weekly_review", "week",
                             "attention", "at_risk", "today", "nudges"}),
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
    args = args or {}
    if any(str(args.get(argument) or "").strip().lower() in values
           for argument, values in WRITE_ARG_VALUES.get(name, {}).items()):
        return True
    readable = READ_ONLY_ACTIONS.get(name)
    if readable is not None:
        action = str(args.get("action") or "").strip().lower()
        if action.replace(" ", "_").replace("-", "_") not in readable:
            return True
    actions = WRITE_ACTIONS.get(name)
    if actions:
        action = str(args.get("action") or _DEFAULT_ACTIONS.get(name)
                     or args.get("mode") or "").strip().lower()
        # Normalised as the dispatchers do ("take photo" is take_photo), and
        # through the shared alias table the handlers apply: "clear_completed"
        # reaches awareness's complete_tasks branch, so it is a write here too.
        if action in actions or action.replace(" ", "_").replace("-", "_") in actions:
            return True
        from .action_aliases import normalise_action
        if normalise_action(name, action) in actions:
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
    "ACTING_SPECIALISTS",
    "DEFAULT_MAX_PARALLEL",
    "EXCLUSIVE_TOOLS",
    "PARALLEL_TOOLS",
    "WRITE_ACTIONS",
    "WRITE_ARG_VALUES",
    "WRITE_FLAGS",
    "ToolClass",
    "classify",
    "describe_plan",
    "max_parallel",
    "parallelism_enabled",
    "plan_batches",
    "selects_write_branch",
]
