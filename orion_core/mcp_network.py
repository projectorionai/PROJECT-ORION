"""Read-only local MCP diagnostics for ORION's phone connection.

The server exposes only this PC's tailnet identity and whether its gateway
accepts a local TCP connection. It never accepts a target URL or runs a shell.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from typing import Any


TOOLS = [
    {"name": "tailnet_status", "description": "Check this PC's Tailscale state and MagicDNS name.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
     "annotations": {"readOnlyHint": True, "destructiveHint": False}},
    {"name": "phone_gateway_status", "description": "Check whether ORION's phone gateway accepts TCP connections on this PC.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
     "annotations": {"readOnlyHint": True, "destructiveHint": False}},
]


def _tailscale_status() -> dict[str, Any]:
    executable = shutil.which("tailscale")
    if executable is None and sys.platform == "win32":
        candidate = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                 "Tailscale", "tailscale.exe")
        executable = candidate if os.path.isfile(candidate) else None
    if not executable:
        return {}
    try:
        result = subprocess.run([executable, "status", "--json"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=5, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        parsed = json.loads(result.stdout) if result.returncode == 0 else {}
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def sample(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or arguments:
        raise ValueError("This diagnostic accepts no arguments")
    if name == "tailnet_status":
        status = _tailscale_status()
        self_node = status.get("Self") or {}
        if not isinstance(self_node, dict):
            self_node = {}
        addresses = self_node.get("TailscaleIPs") or []
        return {
            "state": str(status.get("BackendState") or "unavailable"),
            "magicdns": str(self_node.get("DNSName") or "").rstrip("."),
            "ipv4": next((str(ip) for ip in addresses if isinstance(ip, str)
                          and ip.startswith("100.")), ""),
        }
    if name == "phone_gateway_status":
        host = os.getenv("ORION_REMOTE_HOST", "127.0.0.1").strip() or "127.0.0.1"
        try:
            port = int(os.getenv("ORION_REMOTE_PORT", "8000"))
        except ValueError:
            port = 8000
        if not 1 <= port <= 65535:
            port = 8000
        # Only ORION's own configured listener is probed; there is no
        # caller-supplied host or general network scanner.
        try:
            with socket.create_connection((host, port), timeout=0.75):
                listening = True
        except (OSError, ValueError):
            listening = False
        return {"mode": os.getenv("ORION_REMOTE_EXPOSURE", "serve"),
                "host": host, "port": port, "tcp_listening": listening,
                "tls_requested": os.getenv("ORION_REMOTE_TLS", "0") == "1"}
    raise ValueError("Unknown network diagnostic")


def respond(request: Any) -> dict[str, Any] | None:
    if not isinstance(request, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "Invalid request"}}
    if "id" not in request:
        return None
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request["id"]}
    method, params = request.get("method"), request.get("params") or {}
    if not isinstance(params, dict):
        reply["error"] = {"code": -32602, "message": "params must be an object"}
    elif method == "initialize":
        requested = params.get("protocolVersion")
        reply["result"] = {
            "protocolVersion": requested if requested in {
                "2024-11-05", "2025-03-26", "2025-06-18"} else "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "orion-network", "version": "1.0.0"},
        }
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        reply["result"] = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            value = sample(str(params.get("name", "")), params.get("arguments", {}))
            reply["result"] = {"content": [{"type": "text", "text": json.dumps(value)}]}
        except ValueError as exc:
            reply["result"] = {"isError": True,
                               "content": [{"type": "text", "text": str(exc)}]}
    else:
        reply["error"] = {"code": -32601, "message": "Method not found"}
    return reply


def main() -> None:
    while True:
        line = sys.stdin.buffer.readline(1_048_577)
        if not line or len(line) > 1_048_576:
            return
        try:
            reply = respond(json.loads(line))
        except (ValueError, UnicodeError):
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
