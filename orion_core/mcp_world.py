"""ORION MCP tools for attributed official world data; no arbitrary URLs."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# The MCP host also launches bundled modules as file paths.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orion_core.world_feeds import LAYER_NAMES, get_world_feeds, shutdown_world_feeds

_DESCRIPTION = {
    "satellites": "Space station SGP4 predictions from CelesTrak orbital elements, with epoch and source; not observations.",
    "earthquakes": "USGS earthquakes from the past day, with magnitude, depth, time and source.",
    "tsunami": "NOAA NTWC and PTWC tsunami bulletins, including information statements; not all global active alerts.",
    "weather": "Met Office UK severe weather warning summaries and direct official links.",
    "events": "NASA EONET open natural events from the past 30 days - wildfires, severe storms "
              "(with their tracks), volcanoes, floods and ice - each with kind, latest position, "
              "magnitude and reporting agency. Curated, not exhaustive.",
}
TOOLS = [
    {"name": f"world_{layer}", "description": description,
     "inputSchema": {"type": "object", "properties": {
         "limit": {"type": "integer", "minimum": 1, "maximum": 200,
                   "description": "Maximum records returned, in feed order; default 50."}},
         "additionalProperties": False},
     "annotations": {"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": True}}
    for layer, description in _DESCRIPTION.items()
] + [{"name": "world_feed_status",
      "description": "Cached source health, last successful download and next request time; never treats failure as no warnings.",
      "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
      "annotations": {"readOnlyHint": True, "destructiveHint": False,
                      "idempotentHint": True, "openWorldHint": False}}]


async def sample(name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be an object")
    if name == "world_feed_status":
        if arguments:
            raise ValueError("This diagnostic accepts no arguments")
        snapshot = await get_world_feeds().snapshot(refresh=False)
        for layer in snapshot["layers"].values():
            layer["record_count"] = len(layer.pop("items"))
        return snapshot
    layer = name.removeprefix("world_")
    if name != f"world_{layer}" or layer not in LAYER_NAMES:
        raise ValueError("Unknown world feed tool")
    if set(arguments) - {"limit"}:
        raise ValueError("Only limit is supported; caller URLs are not accepted")
    limit = arguments.get("limit", 50)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer from 1 to 200")
    snapshot = await get_world_feeds().snapshot((layer,))
    data = snapshot["layers"][layer]
    data["record_count"] = len(data["items"])
    data["truncated"] = data["record_count"] > limit
    data["items"] = data["items"][:limit]
    return {"generated_at": snapshot["generated_at"], "layer": layer, **data}


async def respond(request: Any) -> dict[str, Any] | None:
    if not isinstance(request, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "Invalid request"}}
    if "id" not in request:
        return None
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request["id"]}
    method, params = request.get("method"), request.get("params", {})
    if not isinstance(params, dict):
        reply["error"] = {"code": -32602, "message": "params must be an object"}
    elif method == "initialize":
        requested = params.get("protocolVersion")
        reply["result"] = {
            "protocolVersion": requested if requested in {
                "2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"} else "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "orion-world", "version": "1.0.0"},
        }
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        reply["result"] = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            value = await sample(str(params.get("name", "")), params.get("arguments", {}))
            reply["result"] = {
                "content": [{"type": "text", "text": json.dumps(value, allow_nan=False)}],
                "structuredContent": value,
                "isError": value.get("status") == "unavailable",
            }
        except (ValueError, TypeError, RuntimeError) as exc:
            reply["result"] = {"isError": True,
                               "content": [{"type": "text", "text": str(exc)}]}
    else:
        reply["error"] = {"code": -32601, "message": "Method not found"}
    return reply


async def run() -> None:
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline, 1_048_577)
            if not line or len(line) > 1_048_576:
                return
            try:
                reply = await respond(json.loads(line))
            except (ValueError, UnicodeError, RecursionError):
                reply = {"jsonrpc": "2.0", "id": None,
                         "error": {"code": -32700, "message": "Parse error"}}
            if reply is not None:
                print(json.dumps(reply, allow_nan=False), flush=True)
    finally:
        await shutdown_world_feeds()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
