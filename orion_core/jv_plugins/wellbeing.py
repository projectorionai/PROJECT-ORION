"""Explicit, cancellable Pomodoro and hydration sessions."""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone

from ..data import ToolResult
from . import runtime

_pomodoro_task: asyncio.Task | None = None
_water_task: asyncio.Task | None = None
_focus: dict = {}
_water: dict = {}


def number(args: dict, key: str, default: float, low: float, high: float) -> float:
    value = float(args.get(key, default))
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}")
    return value


async def _cancel(task: asyncio.Task | None) -> None:
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _focus_loop() -> None:
    try:
        while True:
            await asyncio.sleep(max(0, _focus["deadline"] - time.monotonic()))
            if _focus["phase"] == "work":
                stats = await asyncio.to_thread(runtime.load, "pomodoro_stats")
                day = datetime.now(timezone.utc).astimezone().date().isoformat()
                stats[day] = int(stats.get(day, 0)) + 1
                await asyncio.to_thread(runtime.save, "pomodoro_stats", stats)
                _focus["completed"] += 1
                _focus["phase"] = "break"
                minutes = _focus["long_break"] if _focus["completed"] % 4 == 0 else _focus["break"]
                runtime.say(f"Focus block complete. Take a {minutes:g}-minute break.")
            else:
                _focus["phase"] = "work"
                minutes = _focus["work"]
                runtime.say(f"Your break is complete. Resume { _focus['label'] }.")
            _focus["deadline"] = time.monotonic() + minutes * 60
            runtime.emit("jv_pomodoro", _focus_status())
    except asyncio.CancelledError:
        raise
    except Exception:
        _focus["phase"] = "stopped"
        runtime.emit("jv_pomodoro", {"phase": "error", "message": "Focus timer stopped after a persistence error."})
        raise


def _focus_status() -> dict:
    return {k: v for k, v in _focus.items() if k != "deadline"} | {
        "remaining_seconds": max(0, round(_focus.get("deadline", 0) - time.monotonic()))}


async def pomodoro(args: dict) -> ToolResult:
    global _pomodoro_task
    action = str(args.get("action") or "status").lower()
    if action == "start":
        # Validate before cancelling an existing useful session.
        work = number(args, "work_minutes", 25, 1, 180)
        brk = number(args, "break_minutes", 5, 1, 60)
        long = number(args, "long_break_minutes", 15, 1, 90)
        label = str(args.get("task") or "your focus task").strip()[:200]
        await _cancel(_pomodoro_task)
        _focus.clear()
        _focus.update(phase="work", label=label, work=work, **{"break": brk},
                      long_break=long, completed=0, deadline=time.monotonic() + work * 60)
        _pomodoro_task = runtime.spawn(_focus_loop(), "pomodoro")
        runtime.emit("jv_pomodoro", _focus_status())
        return ToolResult(f"Started a {work:g}-minute focus block on {label}, followed by a {brk:g}-minute break.")
    if action == "stop":
        await _cancel(_pomodoro_task)
        _pomodoro_task = None
        _focus.clear()
        runtime.emit("jv_pomodoro", {"phase": "stopped"})
        return ToolResult("Pomodoro stopped.")
    if action == "stats":
        stats = await asyncio.to_thread(runtime.load, "pomodoro_stats")
        return ToolResult(f"Completed focus blocks: {sum(int(x) for x in stats.values())}; "
                          f"today: {stats.get(datetime.now(timezone.utc).astimezone().date().isoformat(), 0)}.")
    if action != "status":
        return ToolResult("Pomodoro actions: start, stop, status or stats.", ok=False)
    if _pomodoro_task is None or _pomodoro_task.done():
        return ToolResult("No Pomodoro session is running.")
    status = _focus_status()
    return ToolResult(f"{status['label']}: {status['phase']}, {status['remaining_seconds']} seconds remaining; "
                      f"{status['completed']} blocks completed this session.")


async def _water_loop(interval: float) -> None:
    while _water["consumed_ml"] < _water["goal_ml"]:
        await asyncio.sleep(interval * 60)
        if _water["consumed_ml"] >= _water["goal_ml"]:
            break
        runtime.say("Time for a water break.")
        runtime.emit("jv_water", dict(_water))


async def water_reminder(args: dict) -> ToolResult:
    global _water_task
    action = str(args.get("action") or "status").lower()
    today = datetime.now(timezone.utc).astimezone().date().isoformat()
    stored = await asyncio.to_thread(runtime.load, "hydration")
    if stored.get("day") != today:
        stored = {"day": today, "consumed_ml": 0, "goal_ml": 2000}
    _water.clear()
    _water.update(stored)
    if action == "start":
        interval = number(args, "interval_minutes", 60, 5, 240)
        goal = number(args, "goal_ml", _water.get("goal_ml", 2000), 250, 5000)
        await _cancel(_water_task)
        _water["goal_ml"] = int(goal)
        _water_task = runtime.spawn(_water_loop(interval), "water")
        await asyncio.to_thread(runtime.save, "hydration", _water)
        return ToolResult(f"Water reminders every {interval:g} minutes. Your chosen goal is {goal:g} ml today.")
    if action == "stop":
        await _cancel(_water_task)
        _water_task = None
        return ToolResult("Water reminders stopped.")
    if action == "log":
        amount = number(args, "amount_ml", 250, 1, 2000)
        _water["consumed_ml"] += int(amount)
        await asyncio.to_thread(runtime.save, "hydration", _water)
        if _water["consumed_ml"] >= _water["goal_ml"]:
            await _cancel(_water_task)
            _water_task = None
    elif action == "reset":
        _water["consumed_ml"] = 0
        await asyncio.to_thread(runtime.save, "hydration", _water)
    elif action != "status":
        return ToolResult("Water actions: start, stop, log, reset or status.", ok=False)
    runtime.emit("jv_water", dict(_water))
    running = _water_task is not None and not _water_task.done()
    return ToolResult(f"Water logged today: {_water['consumed_ml']}/{_water['goal_ml']} ml. "
                      f"Reminders {'running' if running else 'stopped'}.")
