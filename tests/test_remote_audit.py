"""
RemoteAuditLog — Cloud Roadmap C4's durable "who/when/what tool/what result".

Before 2026-10-03 the only record of what a phone made ORION do was fifty
in-memory entries covering /v1/agent/tasks alone, lost at every restart.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult  # noqa: E402
from orion_core.remote import RemoteAgentQueue, RemoteAuthManager, RemoteGateway  # noqa: E402
from orion_core.remote_audit import RemoteAuditLog  # noqa: E402
from orion_core.remote_capability import (  # noqa: E402
    RemoteConfirmationRegistry,
    RemoteToolGate,
)


class _Signal:
    def emit(self, *payload):
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


class _Dispatcher:
    async def dispatch(self, name, args):
        return ToolResult(f"secret result for {args}")


def _run(coro):
    return asyncio.run(coro)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_events_are_appended_as_json_lines(tmp_path):
    log = RemoteAuditLog(tmp_path / "audit.jsonl")
    log.record("tool", device="abcdef0123456789", tool="open_app", status="ran",
               args={"app_name": "notepad"})
    rows = _lines(tmp_path / "audit.jsonl")
    assert rows[0]["event"] == "tool" and rows[0]["tool"] == "open_app"
    assert rows[0]["device"] == "abcdef01"          # a short id, never the full one
    assert rows[0]["args"] == ["app_name"]


def test_argument_values_and_results_are_never_stored(tmp_path):
    path = tmp_path / "audit.jsonl"
    queue = RemoteAgentQueue(dispatcher=_Dispatcher(), audit=RemoteAuditLog(path))
    _run(queue.run("query_intelligence", {"query": "my bank PIN"}, device_id="dev12345"))
    text = path.read_text(encoding="utf-8")
    assert "bank PIN" not in text and "secret result" not in text
    assert "query_intelligence" in text and "done" in text


def test_denied_agent_tasks_are_recorded(tmp_path):
    path = tmp_path / "audit.jsonl"
    queue = RemoteAgentQueue(dispatcher=_Dispatcher(), audit=RemoteAuditLog(path))
    _run(queue.run("desktop_control", {"action": "click"}, device_id="dev12345"))
    _run(queue.run("not_a_tool", {}, device_id="dev12345"))
    statuses = [(r["tool"], r["status"]) for r in _lines(path)]
    assert statuses == [("desktop_control", "denied"), ("not_a_tool", "denied")]


def test_the_gate_records_refusals_parks_and_runs(tmp_path):
    path = tmp_path / "audit.jsonl"
    gate = RemoteToolGate(_Dispatcher(), RemoteConfirmationRegistry(),
                          audit=RemoteAuditLog(path))
    _run(gate.dispatch("self_repair", {"action": "apply"}, device_id="dev"))
    _run(gate.dispatch("outlook_mail", {"action": "send", "to": "x@y.z"}, device_id="dev"))
    _run(gate.dispatch("query_intelligence", {"query": "q"}, device_id="dev"))
    statuses = [r["status"] for r in _lines(path)]
    assert statuses == ["refused", "parked for approval", "ran"]


def test_pairing_tokens_and_revocation_are_recorded(tmp_path):
    path = tmp_path / "audit.jsonl"
    auth = RemoteAuthManager(config_dir=tmp_path)
    auth.audit = RemoteAuditLog(path)
    auth.complete_pairing("0000-0000")                       # wrong code
    code = auth.begin_pairing()
    device_id, _refresh = auth.complete_pairing(code, "phone")
    auth.issue_access(device_id, "not the refresh token")
    auth.revoke_device(device_id)
    events = [(r["event"], r["status"]) for r in _lines(path)]
    assert events == [("pair_failed", "refused"), ("paired", "ok"),
                      ("token_refused", "bad token"), ("revoked", "ok")]


def test_the_log_is_capped_and_rotated_once(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = RemoteAuditLog(path, max_bytes=4096)
    for i in range(400):
        log.record("tool", device="dev", tool=f"tool_{i}", status="ran")
    assert path.stat().st_size <= 4096
    assert path.with_name("audit.jsonl.1").exists()
    newest = log.recent(1)[0]
    assert newest["tool"] == "tool_399"


def test_recent_reads_newest_first_and_filters_by_device(tmp_path):
    log = RemoteAuditLog(tmp_path / "audit.jsonl")
    log.record("tool", device="aaaa1111", tool="one", status="ran")
    log.record("tool", device="bbbb2222", tool="two", status="ran")
    log.record("tool", device="aaaa1111", tool="three", status="failed")
    assert [r["tool"] for r in log.recent(10)] == ["three", "two", "one"]
    assert [r["tool"] for r in log.recent(10, device="aaaa1111ffff")] == ["three", "one"]
    assert "three: failed" in log.render()


def test_a_broken_disk_never_breaks_the_request(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    log = RemoteAuditLog(blocker / "audit.jsonl")
    log.record("tool", tool="x", status="ran")          # must not raise
    assert log.recent() == []


def test_the_gateway_keeps_its_audit_beside_its_other_state(tmp_path):
    gateway = RemoteGateway(None, None, _Bus(), dispatcher=_Dispatcher(),
                            config_dir=tmp_path)
    assert gateway.audit.path == tmp_path / "diagnostics" / "remote_audit.jsonl"
    assert gateway.auth.audit is gateway.audit
    assert gateway.agent_queue.audit is gateway.audit
