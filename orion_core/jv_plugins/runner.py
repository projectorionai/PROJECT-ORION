"""Dispatch the reviewed adapters with bounded work and honest failures."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from ..data import ToolResult
from . import runtime

_locks: dict[str, asyncio.Lock] = {}
_game_worker_lock = threading.Lock()
_game_workers: set[asyncio.Task] = set()
GAME_TIMEOUT_S = 110.0


def _run_game_action(arguments: dict[str, Any]) -> ToolResult:
    """Run only after the owner loop has reserved physical admission."""
    from ..execution_policy import require_foreground
    deferred = require_foreground("game launcher desktop/process control")
    if deferred is not None:
        return deferred
    from . import legacy_game_updater as game
    text = game.game_updater(arguments, player=runtime.Player())
    return ToolResult(text, ok=not any(word in text.casefold() for word in (
        "failed", "not installed", "nothing to do", "do not know", "tell me", "not in your", "i found neither")))


async def _execute_game_action(arguments: dict[str, Any]) -> ToolResult:
    from ..execution_policy import require_foreground
    deferred = require_foreground("game launcher desktop/process control")
    if deferred is not None:
        return deferred
    # Reserve before joining the executor queue. Busy callers must neither
    # enqueue another launcher action nor wait for their own timeout first.
    physical = _game_worker_lock
    if not physical.acquire(blocking=False):
        return ToolResult("A launcher action is still running; check its result before another action.", ok=False,
                          receipt={"kind": "plugin", "status": "pending", "attempted": False})
    start_guard = threading.Lock()
    state = {"started": False, "abandoned": False, "released": False}

    def release_physical() -> None:
        with start_guard:
            if not state["released"]:
                state["released"] = True
                physical.release()

    def abandon() -> bool:
        with start_guard:
            state["abandoned"] = True
            return state["started"]

    def release_unclaimed() -> None:
        with start_guard:
            if state["started"]:
                return
            state["abandoned"] = True
        release_physical()

    def physical_call() -> ToolResult | None:
        try:
            # Atomic against timeout/cancellation: a wrapper which has not
            # claimed execution must never start the handler afterwards.
            with start_guard:
                if state["abandoned"]:
                    return None
                state["started"] = True
            return _run_game_action(arguments)
        finally:
            # A stale rejected wrapper cannot release a later owner's lock.
            release_physical()

    async def work() -> ToolResult | None:
        try:
            return await asyncio.to_thread(physical_call)
        except BaseException:
            # Submission failure or cancellation of a queued executor future
            # may prevent physical_call/finally from running at all.
            release_unclaimed()
            raise

    def retire(task: asyncio.Task) -> None:
        _game_workers.discard(task)
        if not task.cancelled():
            task.exception()  # Observe faults after the public caller left.
        release_unclaimed()  # Also handles cancellation before work starts.

    coroutine = work()
    try:
        task = asyncio.create_task(coroutine, name="orion-jv-game-worker")
    except BaseException:
        coroutine.close()
        release_unclaimed()
        raise
    _game_workers.add(task)
    task.add_done_callback(retire)
    try:
        # Keep the admitted worker owned through public timeout/cancellation.
        # Only the physical wrapper may release an already-started handler.
        done, _ = await asyncio.wait({task}, timeout=GAME_TIMEOUT_S)
        if done:
            return task.result()
        attempted = abandon()
        return ToolResult("This action exceeded its time limit; completion is unverified. A launcher operation already accepted by the OS may continue.", ok=False,
                          receipt={"kind": "plugin", "status": "uncertain", "attempted": attempted})
    except asyncio.CancelledError:
        abandon()
        raise


async def execute(name: str, arguments: dict[str, Any]) -> ToolResult:
    from ..gemini_schema import decode_json_args
    from .catalogue import schema
    from .forge_trials import forge_trial
    from .integrations import calorie_counter, chat_takeover, cloud_agent
    from .learning import document_review, quiz
    from .spreadsheets import excel_writer
    from .telegram import telegram_remote
    from .wellbeing import pomodoro, water_reminder

    handlers = {"jv_document_review": document_review, "jv_quiz": quiz, "jv_pomodoro": pomodoro,
        "jv_water_reminder": water_reminder, "jv_excel_writer": excel_writer,
        "jv_calorie_counter": calorie_counter, "jv_chat_takeover": chat_takeover,
        "jv_telegram_remote": telegram_remote, "jv_cloud_agent": cloud_agent,
        "jv_forge_trial": forge_trial}
    try:
        arguments = decode_json_args(arguments, schema(name)["parameters"])
        if not isinstance(arguments, dict):
            return ToolResult("Plugin arguments must be an object.", ok=False)
        if name == "jv_game_updater":
            action = str(arguments.get("action") or "list")
            if action not in {"list", "download_status", "install", "update", "schedule", "cancel_schedule", "schedule_status"}:
                return ToolResult("Choose a supported game updater action.", ok=False)
            if action == "install" and str(arguments.get("platform") or "both") == "epic":
                return ToolResult("The supplied integration can update installed Epic games; installing an Epic title needs its launcher.", ok=False)
            clean = dict(arguments, action=action, shutdown_when_done=False)
            return await _execute_game_action(clean)
        async with _locks.setdefault(name, asyncio.Lock()):
            if name == "jv_reminder":
                return await runtime.native("reminder", arguments)
            handler = handlers.get(name)
            if handler is None:
                return ToolResult("Unknown reviewed plugin adapter.", ok=False)
            return await asyncio.wait_for(handler(arguments), timeout=320 if name == "jv_forge_trial" else 110)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        return ToolResult("This action exceeded its time limit; completion is unverified. A launcher operation already accepted by the OS may continue.", ok=False,
                          receipt={"kind": "plugin", "status": "uncertain", "attempted": True})
    except (ValueError, TypeError) as exc:
        return ToolResult(f"Invalid plugin input: {exc}", ok=False)
    except ImportError as exc:
        missing = str(getattr(exc, "name", "") or "the required dependency")
        return ToolResult(f"Install the plugin dependency {missing} before using this action.", ok=False)
    except Exception as exc:  # noqa: BLE001 -- external service faults become an honest status
        # Provider and transport exceptions can contain credential-bearing URLs.
        return ToolResult(f"The plugin action failed ({type(exc).__name__}). Check its configuration and service status.", ok=False)
