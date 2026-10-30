"""Small read-only stdio MCP server; uses ORION's existing psutil dependency.

Run as a script with a Python interpreter. No shell, file writes, model loads,
network listener, process arguments or environment disclosure are exposed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

TOOLS = [
    {"name": "system_snapshot", "description": "Read current CPU and RAM utilisation.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
     "annotations": {"readOnlyHint": True, "destructiveHint": False}},
    {"name": "process_memory", "description": "Read the largest processes by resident RAM (names only).",
     "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                     "additionalProperties": False},
     "annotations": {"readOnlyHint": True, "destructiveHint": False}},
]


def sample(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import psutil
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    if name == "system_snapshot":
        if arguments:
            raise ValueError("system_snapshot accepts no arguments")
        memory = psutil.virtual_memory()
        return {"at": datetime.now(timezone.utc).isoformat(),
                "cpu_percent_since_previous_sample": psutil.cpu_percent(interval=None),
                "logical_cpus": psutil.cpu_count(), "ram_total_bytes": memory.total,
                "ram_available_bytes": memory.available, "ram_percent": memory.percent}
    if name == "process_memory":
        limit = arguments.get("limit", 5)
        if set(arguments) - {"limit"} or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer from 1 to 20")
        processes = []
        for proc in psutil.process_iter(["pid", "name", "memory_info"], ad_value=None):
            info = proc.info
            if info.get("memory_info") is not None:
                processes.append({"pid": info["pid"], "name": info["name"],
                                  "rss_bytes": info["memory_info"].rss})
        return {"processes": sorted(processes, key=lambda p: p["rss_bytes"], reverse=True)[:limit]}
    raise ValueError("unknown telemetry tool")


def respond(request: Any) -> dict[str, Any] | None:
    if not isinstance(request, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
    if "id" not in request:
        return None
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request["id"]}
    method, params = request.get("method"), request.get("params") or {}
    if not isinstance(params, dict):
        reply["error"] = {"code": -32602, "message": "params must be an object"}
    elif method == "initialize":
        requested = params.get("protocolVersion")
        reply["result"] = {"protocolVersion": requested if requested in {
            "2024-11-05", "2025-03-26", "2025-06-18"} else "2024-11-05",
            "capabilities": {"tools": {}}, "serverInfo": {"name": "orion-telemetry", "version": "1.0.0"}}
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        reply["result"] = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            value = sample(str(params.get("name", "")), params.get("arguments", {}))
            reply["result"] = {"content": [{"type": "text", "text": json.dumps(value)}]}
        except (ValueError, ImportError) as exc:
            reply["result"] = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        except Exception as exc:
            reply["result"] = {"isError": True, "content": [{"type": "text", "text": type(exc).__name__}]}
    else:
        reply["error"] = {"code": -32601, "message": "Method not found"}
    return reply


def main() -> None:
    try:
        import psutil
        psutil.cpu_percent(interval=None)  # prime the non-blocking counter
    except ImportError:
        pass
    while True:
        line = sys.stdin.buffer.readline(1_048_577)
        if not line:
            return
        if len(line) > 1_048_576:
            return  # bound input and refuse a desynchronised transport
        try:
            reply = respond(json.loads(line))
        except (ValueError, UnicodeError):
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
