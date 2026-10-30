"""Exercise the actual reviewed PC control path in a new, disposable document."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Signal:
    def emit(self, *_args):
        pass


async def verify() -> dict:
    from types import SimpleNamespace
    from orion_core.control import AutonomousControlLayer
    from orion_core.data import ToolResult
    from orion_core.display import DisplayTopologyManager
    from orion_core.remote_capability import RemoteConfirmationRegistry, RemoteToolGate
    document = ROOT / ".test-runs" / f"ORION-control-check-{uuid.uuid4().hex[:8]}.txt"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("ORION desktop command test.\n", encoding="utf-8")
    expected = "ORION has written this through the reviewed desktop control path."
    bus = SimpleNamespace(log=Signal(), control_activity=Signal(), control_caption=Signal())
    control = AutonomousControlLayer(bus, DisplayTopologyManager())
    control.show_working = False
    class Desktop:
        TOOL_DECLARATIONS = [{"name": "open_app"}, {"name": "desktop_control"}]
        async def dispatch(self, name, args):
            if name == "open_app":
                await asyncio.to_thread(subprocess.Popen, ["notepad.exe", str(document)])
                for _ in range(40):
                    if await asyncio.to_thread(control._find_window, document.stem):
                        return ToolResult("Opened the disposable test document.")
                    await asyncio.sleep(0.25)
                return ToolResult("The test document did not open.", ok=False)
            return await asyncio.to_thread(control.edit_text, expected,
                                            title=document.stem, replace=False)
    registry = RemoteConfirmationRegistry()
    gate = RemoteToolGate(Desktop(), registry)
    steps = [{"tool": "open_app", "args": {"app_name": "notepad"}},
             {"tool": "desktop_control", "args": {"action": "edit_text", "text": expected,
                                                     "title": document.stem}}]
    with gate.for_device("local-diagnostic"):
        proposed = await gate.dispatch_plan(steps)
    if not proposed.ok:
        return {"ok": False, "result": proposed.text}
    pending = registry.pending_for("local-diagnostic")[0]
    approved = registry.resolve(pending.id, pending.token, True, device_id="local-diagnostic")
    result = await gate.run_approved(approved)
    readback = await asyncio.to_thread(control._read_editable_text, document.stem)
    return {"ok": bool(result.ok and readback and expected in readback),
            "result": result.text, "document": str(document),
            "actual_text_verified": bool(readback and expected in readback)}


if __name__ == "__main__":
    outcome = asyncio.run(verify())
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["ok"] else 1)
