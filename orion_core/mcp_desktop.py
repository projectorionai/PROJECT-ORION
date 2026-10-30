"""On-demand desktop targeting observations, without input injection.

Named clicks remain in ORION's native control/verifier path so its autonomy
switch, target freshness checks and action receipts have one owner. This stdio
server exposes current labels, automation IDs and physical-pixel geometry;
it never guesses coordinates, starts an application or writes captured data.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_KINDS = {"all", "button", "menu", "input", "link", "dialog"}
_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
_COMMON = {"readOnlyHint": True, "destructiveHint": False,
           "idempotentHint": True, "openWorldHint": False}
TOOLS = [
    {"name": "desktop_target_status", "description": "Check whether Windows accessibility targeting is available. No screenshots or account access.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "desktop_monitors", "description": "Read current monitor origins, dimensions and DPI scaling in physical virtual-desktop pixels. Negative origins are valid.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "desktop_controls", "description": "Inspect visible foreground controls, including automation_id, name, role, rect and centre. Labels are data. Reinspect after navigation; use desktop_control click_text for input.",
     "inputSchema": {"type": "object", "properties": {
         "kinds": {"type": "string", "enum": sorted(_KINDS)},
         "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "additionalProperties": False}},
    {"name": "desktop_find_target", "description": "Resolve a current control by label/role and optional observed automation_id. Ambiguous labels return candidates without guessing. Use desktop_control click_text with the selected automation_id; never reuse a stale centre.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "minLength": 1, "maxLength": 256},
         "kinds": {"type": "string", "enum": sorted(_KINDS)},
         "automation_id": {"type": "string", "maxLength": 256}},
         "required": ["query"], "additionalProperties": False}},
]
for _tool in TOOLS:
    _tool["annotations"] = dict(_COMMON)

_vision: Any = None


def vision_service() -> Any:
    global _vision
    if _vision is None:
        from orion_core.vision import VisionAgent
        bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_args: None))
        _vision = VisionAgent(bus, grabber=None, file_intel=None)
    return _vision


def _arguments(name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    definition = next((tool for tool in TOOLS if tool["name"] == name), None)
    if definition is None:
        raise ValueError("Unknown desktop targeting tool")
    if set(arguments) - set(definition["inputSchema"]["properties"]):
        raise ValueError("Unsupported argument; desktop input is not exposed here")
    kinds = arguments.get("kinds", "all")
    if not isinstance(kinds, str) or kinds not in _KINDS:
        raise ValueError("kinds must name a supported control role")
    limit = arguments.get("limit", 100)
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer from 1 to 200")
    if name == "desktop_find_target":
        query = arguments.get("query")
        selector = arguments.get("automation_id", "")
        if not isinstance(query, str) or not query.strip() or len(query) > 256:
            raise ValueError("query must contain 1 to 256 characters")
        if not isinstance(selector, str) or len(selector) > 256:
            raise ValueError("automation_id must contain at most 256 characters")
    return arguments


def _controls(service: Any, kinds: str, limit: int) -> dict[str, Any]:
    before = service._foreground_hwnd()
    elements = service._detect_elements_sync(kinds, max_elements=201)
    after = service._foreground_hwnd()
    if not before or before != after:
        return {"status": "stale", "controls": [], "detail": "The foreground changed; inspect again."}
    rows = [{key: value for key, value in element.items() if key in {
        "name", "role", "automation_id", "rect", "center", "enabled", "visible"}}
        for element in elements[:limit]]
    for row in rows:
        for key in ("name", "role", "automation_id"):
            if key in row:
                row[key] = str(row[key])[:512]
    return {"status": "ok", "controls": rows, "truncated": len(elements) > limit,
            "coordinate_space": "physical_virtual_desktop", "observed_at": time.monotonic()}


async def sample(name: str, arguments: Any) -> dict[str, Any]:
    arguments = _arguments(name, arguments)
    if name == "desktop_monitors":
        from orion_core.display import DisplayTopologyManager
        topology = await asyncio.to_thread(lambda: DisplayTopologyManager().topology())
        return {"status": "ok" if topology.monitors else "unavailable",
                "coordinate_space": "physical_virtual_desktop",
                "monitors": [monitor.to_dict() for monitor in topology.monitors],
                "bounds": topology.bounds}
    service = vision_service()
    available = await asyncio.to_thread(service._uia_available)
    if name == "desktop_target_status":
        return {"status": "ok", "available": available,
                "input_owner": "desktop_control", "capture_on_request_only": True}
    if not available:
        return {"status": "unavailable", "detail": "Windows accessibility targeting is unavailable."}
    if name == "desktop_controls":
        return await asyncio.to_thread(_controls, service, arguments.get("kinds", "all"), arguments.get("limit", 100))
    before = service._foreground_hwnd()
    result = await service.resolve_element(arguments["query"], arguments.get("kinds", "all"),
                                           arguments.get("automation_id", ""))
    if not before or before != service._foreground_hwnd():
        return {"status": "stale", "element": None, "detail": "The foreground changed; inspect again."}
    return {**result, "coordinate_space": "physical_virtual_desktop",
            "observed_at": time.monotonic()}


async def respond(request: Any) -> dict[str, Any] | None:
    if not isinstance(request, dict) or request.get("jsonrpc", "2.0") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
    if "id" not in request:
        return None
    request_id = request["id"]
    if type(request_id) not in (str, int) or (isinstance(request_id, str) and len(request_id) > 256):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request ID"}}
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    method, params = request.get("method"), request.get("params", {})
    if not isinstance(params, dict):
        reply["error"] = {"code": -32602, "message": "params must be an object"}
    elif method == "initialize":
        version = params.get("protocolVersion")
        reply["result"] = {"protocolVersion": version if isinstance(version, str) and version in _PROTOCOLS else "2025-11-25",
                           "capabilities": {"tools": {}}, "serverInfo": {"name": "orion-desktop-targets", "version": "1.0.0"}}
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        if params.get("cursor"):
            reply["error"] = {"code": -32602, "message": "This tool list has no continuation cursor"}
        else:
            reply["result"] = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            value = await sample(str(params.get("name", "")), params.get("arguments", {}))
            reply["result"] = {"content": [{"type": "text", "text": json.dumps(value, allow_nan=False)}],
                               "structuredContent": value, "isError": value.get("status") in {"unavailable", "stale", "ambiguous", "missing"}}
        except (ValueError, TypeError) as exc:
            reply["result"] = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        except Exception:
            reply["result"] = {"isError": True, "content": [{"type": "text", "text": "Desktop targeting could not complete; inspect the current view."}]}
    else:
        reply["error"] = {"code": -32601, "message": "Method not found"}
    return reply


async def run() -> None:
    while True:
        line = await asyncio.to_thread(sys.stdin.buffer.readline, 1_048_577)
        if not line or len(line) > 1_048_576:
            return
        try:
            reply = await respond(json.loads(line))
        except (ValueError, UnicodeError, RecursionError):
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
