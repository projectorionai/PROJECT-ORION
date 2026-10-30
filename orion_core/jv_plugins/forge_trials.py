"""A reviewable experiment ledger around Forge's existing verifier and rollback."""

from __future__ import annotations

import asyncio
import re
import time
import uuid

from ..data import ToolResult
from . import runtime


async def forge_trial(args: dict) -> ToolResult:
    action = str(args.get("action") or "list").lower()
    state = await asyncio.to_thread(runtime.load, "forge_trials")
    trials = state.setdefault("trials", {})
    if action == "propose":
        plan = str(args.get("plan") or "").strip()
        label = re.sub(r"[^a-z0-9_]+", "_", str(args.get("name") or "").lower()).strip("_")[:40]
        if not plan or not label:
            return ToolResult("Supply a trial name, its purpose and measurable acceptance checks in plan.", ok=False)
        identifier = uuid.uuid4().hex[:12]
        trials[identifier] = {"id": identifier, "name": "trial_" + label + "_" + identifier[:6],
            "plan": plan[:12000], "status": "proposed", "created_at": time.time(), "attempts": 0}
        if len(trials) > 100:
            oldest = next((key for key, row in trials.items() if row["status"] != "running"), None)
            if oldest:
                del trials[oldest]
        await asyncio.to_thread(runtime.save, "forge_trials", state)
        return ToolResult(f"Trial {identifier} proposed as {trials[identifier]['name']}. Review its acceptance checks, "
                          "then explicitly run it. The existing live tools have not changed.")
    if action == "list":
        rows = [f"{r['id']}: {r['name']} — {r['status']}" for r in trials.values()]
        return ToolResult("\n".join(rows) or "No Forge trials recorded.")
    identifier = str(args.get("id") or "")
    trial = trials.get(identifier)
    if trial is None:
        return ToolResult("Choose a recorded Forge trial id.", ok=False)
    if action == "status":
        return ToolResult(f"{trial['name']}: {trial['status']}\n{trial['plan']}\n{trial.get('result', '')}")
    dispatcher = runtime.context().dispatcher
    forge = getattr(dispatcher, "forge", None)
    if forge is None:
        return ToolResult("This action needs the running desktop Forge service.", ok=False)
    if action == "run":
        if trial["status"] != "proposed" or trial["attempts"] >= 1:
            return ToolResult("This trial has already run. Propose a revised experiment to try again.", ok=False)
        trial.update(status="running", attempts=1)
        await asyncio.to_thread(runtime.save, "forge_trials", state)
        plan = (trial["plan"] + "\n\nTRIAL CONTRACT: create an isolated analysis-only tool. "
                "Never modify ORION's core, existing tools, secrets, permissions or settings. "
                "No network, subprocess, synthetic input or device access. Include meaningful tests "
                "for the acceptance checks and invalid inputs. State uncertainties in results. "
                "The reserved trial_ name enforces a pure-function source subset at generation, "
                "every repair and loading; do not declare third-party requirements. "
                "This tool is a trial capability; passing tests is not evidence of general intelligence.")
        try:
            outcome = await asyncio.wait_for(forge.forge_tool(trial["name"], plan), timeout=300)
            trial.update(status="verified" if outcome.ok else "failed", result=outcome.text[:10000])
        except asyncio.CancelledError:
            trial.update(status="interrupted", result="Trial cancelled; Forge restores any incomplete activation.")
            await asyncio.to_thread(runtime.save, "forge_trials", state)
            raise
        except Exception as exc:  # noqa: BLE001 -- external service faults become an honest status
            trial.update(status="failed", result=f"Trial stopped: {type(exc).__name__}.")
            outcome = ToolResult(trial["result"], ok=False)
        await asyncio.to_thread(runtime.save, "forge_trials", state)
        runtime.emit("jv_forge_trial", dict(trial))
        return outcome
    if action == "rollback":
        handler = getattr(dispatcher, "_tool_handlers", {}).get(trial["name"])
        rollback = getattr(dispatcher, "_forge_rollbacks", {}).get(trial["name"])
        if handler is None or rollback is None:
            return ToolResult("This process has no rollback receipt for that trial. Disable the trial plugin; "
                              "the Forge deployment ledger retains its recovery evidence.", ok=False)
        restored = await rollback(handler, "User requested trial rollback")
        if not restored:
            return ToolResult("The trial changed or is busy; rollback was refused to preserve the newer work.", ok=False)
        trial.update(status="rolled_back")
        await asyncio.to_thread(runtime.save, "forge_trials", state)
        return ToolResult("Trial generation rolled back.")
    return ToolResult("Forge trial actions: propose, list, status, run or rollback.", ok=False)
