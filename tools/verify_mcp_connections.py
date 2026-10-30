"""Discover enabled MCP tools without calling account or mutation tools."""
from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orion_core.mcp_host import MCPHost, MCPServerConn, MCPProtocolError, load_config


class DiagnosticConnection(MCPServerConn):
    def __init__(self, *args):
        super().__init__(*args)
        self.trace = []
    async def _request(self, method, params=None):
        started = time.monotonic()
        outcome = "ok"
        try:
            return await super()._request(method, params)
        except MCPProtocolError as error:
            outcome = self._redact(str(error))[:200]
            raise
        except asyncio.CancelledError:
            outcome = "cancelled or timed out"
            raise
        finally:
            self.trace.append({"method": method, "seconds": round(time.monotonic() - started, 2),
                               "outcome": outcome})


async def verify(only: list[str] | None = None) -> dict:
    silent = SimpleNamespace(log=SimpleNamespace(emit=lambda *_args: None))
    host = MCPHost(silent)
    semaphore = asyncio.Semaphore(3)
    results = {}
    async def one(name, spec):
        if not spec.get("enabled"):
            results[name] = {"state": "disabled"}
            return
        ready = host._prepare(name, spec)
        if ready is None:
            results[name] = {"state": host.status[name]}
            return
        async with semaphore:
            connection = DiagnosticConnection(name, ready, silent)
            try:
                ok = await asyncio.wait_for(connection.start(), timeout=90)
                results[name] = {"state": "connected" if ok else "unavailable",
                    "tools": len(connection.tools)}
                if not ok:
                    results[name]["detail"] = connection.last_error[:200]
                    results[name]["trace"] = connection.trace
                if ok and name in {"orion_telemetry", "orion_network"}:
                    tool = "system_snapshot" if name == "orion_telemetry" else "phone_gateway_status"
                    reply = await connection.call_tool(tool, {})
                    results[name]["read_probe"] = "ok" if not reply.is_error else "failed"
            except asyncio.TimeoutError:
                results[name] = {"state": "connection timed out"}
            finally:
                await connection.stop()
    await asyncio.gather(*(one(name, spec) for name, spec in load_config().get("servers", {}).items()
                           if isinstance(spec, dict) and (not only or name in only)))
    return {"servers": dict(sorted(results.items())), "scope": "Stdio handshake and tool discovery. No account tools executed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+")
    arguments = parser.parse_args()
    report = asyncio.run(verify(arguments.only))
    path = ROOT / "docs" / "MCP_CONNECTION_AUDIT_2026-09-29.json"
    if arguments.only and path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous["servers"].update(report["servers"])
        report["servers"] = dict(sorted(previous["servers"].items()))
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in report["servers"].items():
        print(f"{name}: {result['state']}, {result.get('tools', 0)} tools")
    print(path)
