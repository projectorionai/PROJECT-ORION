"""
The other words a model uses for an action a tool already has.

From the owner's sessions (Oct 2026): "Unsupported window action:
close_window" (window_control only knew "close"), and "I've done all those
tasks" sent awareness an action called clear_completed, which no branch
handled, so nothing was marked done. Each handler had grown its own alias
sets; this is the one shared table, applied where those tools read their
action, so a near-miss name reaches the branch it obviously meant.

Only unambiguous synonyms belong here. A word that could mean two different
operations ("kill": close politely or terminate?) stays unrecognised and is
refused with the list of real actions, rather than guessed.

The concurrency classifier reads the RAW action before any handler runs, so
it resolves aliases through this same table (concurrency.selects_write_branch):
otherwise "clear_completed", unknown to its write list, would be judged
read-only - batched, retried and backgrounded as if it changed nothing.
"""

from __future__ import annotations

from typing import Any

#: tool -> {alias: canonical action}
ALIASES: dict[str, dict[str, str]] = {
    "window_control": {
        "close_window": "close", "closewindow": "close", "close_the_window": "close",
        "exit": "close", "exit_window": "close", "quit": "close", "shut": "close",
        "minimize": "minimise", "minimize_window": "minimise", "minimise_window": "minimise",
        "maximize": "maximise", "maximize_window": "maximise", "maximise_window": "maximise",
        "focus_window": "focus", "bring_to_front": "focus", "front": "focus",
        "activate_window": "focus", "switch_to_window": "focus", "show_window": "focus",
        "list_windows": "list", "windows": "list", "move_window": "move",
    },
    "awareness": {
        "clear_completed": "complete_tasks", "clear_completed_tasks": "complete_tasks",
        "clear_done": "complete_tasks", "remove_completed": "complete_tasks",
        "remove_done": "complete_tasks", "mark_all_done": "complete_tasks",
        "complete_all": "complete_tasks", "done_all": "complete_tasks",
        "all_done": "complete_tasks", "finish_all": "complete_tasks",
        "mark_complete": "complete_tasks", "mark_completed": "complete_tasks",
        # 25 Oct 2026: "can you clear the several due tasks" - the overdue
        # ones the briefing had just read out.
        "clear_due": "clear_overdue", "clear_due_tasks": "clear_overdue",
        "remove_due_tasks": "clear_overdue", "dismiss_overdue": "clear_overdue",
        "clear_late_tasks": "clear_overdue", "clear_past_due": "clear_overdue",
        "show_tasks": "list_tasks", "open_tasks": "list_tasks",
    },
    # 30 Oct 2026: setting up a sleep routine sent notion_workspace action
    # "create" - refused, and told to the owner as "that did not work" in the
    # middle of a research run that was working.
    "notion_workspace": {
        "create": "create_task", "add": "create_task", "new": "create_task",
        "add_todo": "create_task", "create_todo": "create_task",
        "list": "list_tasks", "show_tasks": "list_tasks", "get_tasks": "list_tasks",
        "complete": "complete_task", "mark_done": "complete_task", "finish": "complete_task",
        "update": "update_task", "edit": "update_task", "archive": "archive_task",
        "events": "upcoming_events", "add_event": "create_event", "new_event": "create_event",
    },
}

#: Aliases that also say WHICH items: "clear_completed" with no titles means
#: the tasks just read out, exactly as all_listed=true does.
_IMPLIED: dict[str, dict[str, dict[str, Any]]] = {
    "awareness": {alias: {"all_listed": True} for alias in (
        "clear_completed", "clear_completed_tasks", "clear_done", "remove_completed",
        "remove_done", "mark_all_done", "complete_all", "done_all", "all_done", "finish_all")},
}


def _key(action: Any) -> str:
    return str(action or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalise_action(tool: str, action: Any) -> str:
    """*action* as *tool*'s handler spells it (lower-case; aliases resolved)."""
    key = _key(action)
    return ALIASES.get(str(tool), {}).get(key, key)


def normalise_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """*args* with the action resolved and anything the alias implies added
    (never overriding what the caller supplied)."""
    if not isinstance(args, dict) or "action" not in args:
        return args
    key = _key(args.get("action"))
    canonical = normalise_action(tool, key)
    if canonical == args.get("action") and key not in _IMPLIED.get(str(tool), {}):
        return args
    fixed = dict(args)
    fixed["action"] = canonical
    for name, value in _IMPLIED.get(str(tool), {}).get(key, {}).items():
        if name == "all_listed" and (fixed.get("titles") or fixed.get("title")
                                     or fixed.get("text")):
            continue                    # the caller named the tasks itself
        fixed.setdefault(name, value)
    return fixed


def aliases_for(tool: str) -> frozenset[str]:
    return frozenset(ALIASES.get(str(tool), {}))


__all__ = ["ALIASES", "aliases_for", "normalise_action", "normalise_args"]
