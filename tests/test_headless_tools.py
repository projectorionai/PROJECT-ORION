"""
Headless node: remote tools and offline answers (audit 2026-10-03).

Booting ``orion.py --headless`` and calling it showed that every
``/v1/agent/tasks`` request — even read-only, whitelisted ones — answered 503
"no dispatcher wired on this node", and with no model reachable every chat turn
answered 502 while the node held 88 freshly seeded knowledge entries.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")

from orion_core.data import ToolResult  # noqa: E402
from orion_core.headless_tools import HEADLESS_TOOLS, HeadlessToolHost  # noqa: E402
from orion_core.local_brain import LocalBrain  # noqa: E402
from orion_core.remote import REMOTE_TOOL_WHITELIST, RemoteGateway  # noqa: E402
from orion_core.remote_capability import (  # noqa: E402
    RemoteConfirmationRegistry,
    RemoteToolGate,
)


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _Bus:
    def __getattr__(self, name):
        signal = _Signal()
        object.__setattr__(self, name, signal)
        return signal


_KNOWLEDGE = [
    {"category": "knowledge", "key_ref": "prog_deadlock",
     "value": "Deadlock: Four Coffman conditions: mutual exclusion, hold-and-wait, "
              "no pre-emption, circular wait.", "updated_at": "2026-10-03"},
    {"category": "knowledge", "key_ref": "cyber_zero_trust",
     "value": "Zero trust: Never trust based on network location.",
     "updated_at": "2026-10-03"},
    {"category": "projects", "key_ref": "garden",
     "value": "Garden plan: tomatoes along the south fence.", "updated_at": "2026-10-03"},
]


class _Memory:
    """Just enough of MemoryAgent: word-overlap search over a few rows."""

    def __init__(self, rows=None):
        self.rows = list(rows if rows is not None else _KNOWLEDGE)
        self.episodes = []

    def query(self, query, limit=8):
        words = {w for w in query.lower().split() if len(w) > 2}
        hits = [r for r in self.rows if words & set(r["value"].lower().replace(":", " ").split())]
        return hits[:limit]

    def records(self, query="", limit=8):
        return self.rows[:limit]

    def recall_episodes(self, query, limit=10):
        return [{"created_at": "2026-10-03T09:00:00Z", "role": "user",
                 "content": "we talked about deadlock"}]

    def log_episode(self, role, content):
        self.episodes.append((role, content))

    def prompt_context(self, limit=18):
        return ""

    def remember(self, *a, **k):
        pass


class _NoModelRouter:
    def has_text_fallback(self):
        return False

    def text_profiles(self):
        return []


def _host(memory=None) -> HeadlessToolHost:
    return HeadlessToolHost(_Bus(), memory or _Memory())


def _run(coro):
    return asyncio.run(coro)


# ── the host ────────────────────────────────────────────────────────────────

def test_the_host_serves_only_read_only_remote_tools():
    assert HEADLESS_TOOLS <= REMOTE_TOOL_WHITELIST
    assert set(_host().handler_table()) == set(HEADLESS_TOOLS)


def test_whitelisted_memory_tools_run_on_the_node():
    host = _host()
    found = _run(host.dispatch("query_intelligence", {"query": "deadlock"}))
    assert found.ok and "prog_deadlock" in found.text
    recalled = _run(host.dispatch("recall_conversation", {"query": "deadlock"}))
    assert recalled.ok and "deadlock" in recalled.text


def test_patch_notes_and_node_health_answer():
    from orion_core.changelog import Changelog
    host = HeadlessToolHost(_Bus(), _Memory(), changelog=Changelog())
    assert _run(host.dispatch("patch_notes", {})).ok
    health = _run(host.dispatch("diagnostics", {}))
    assert health.ok and "Headless node" in health.text


def test_desktop_tools_are_refused_plainly_not_as_unknown():
    host = _host()
    assert not host.can_run("open_app")
    result = _run(host.dispatch("open_app", {"app_name": "notepad"}))
    assert not result.ok
    assert "needs the desktop" in result.text


def test_a_handler_fault_is_a_result_not_an_exception():
    class _Broken(_Memory):
        def query(self, *a, **k):
            raise RuntimeError("database is locked")

    result = _run(_host(_Broken()).dispatch("query_intelligence", {"query": "x"}))
    assert not result.ok and "database is locked" in result.text


# ── the gate does not ask the phone to approve the impossible ───────────────

def test_the_gate_does_not_park_a_confirmation_the_node_cannot_honour():
    registry = RemoteConfirmationRegistry()
    gate = RemoteToolGate(_host(), registry)
    # outlook_mail send is CONFIRM-tier on the desktop.
    result = _run(gate.dispatch("outlook_mail", {"action": "send", "to": "a@b.c"},
                                device_id="dev"))
    assert not result.ok and "needs the desktop" in result.text
    assert not registry.pending_for("dev")


def test_the_gate_still_parks_confirmations_for_a_dispatcher_without_can_run():
    class _Desktop:
        async def dispatch(self, name, args):
            return ToolResult("ran")

    registry = RemoteConfirmationRegistry()
    gate = RemoteToolGate(_Desktop(), registry)
    result = _run(gate.dispatch("outlook_mail", {"action": "send", "to": "a@b.c"},
                                device_id="dev"))
    assert result.ok and result.media and "confirm" in result.media


# ── offline answers from resident knowledge ─────────────────────────────────

def _brain(memory=None) -> LocalBrain:
    return LocalBrain(_Bus(), memory or _Memory(), _host(memory), None)


@pytest.mark.parametrize("question, expected", [
    ("what is a deadlock?", "Coffman"),
    ("explain zero trust", "network location"),
    ("tell me about deadlock", "Coffman"),
])
def test_general_questions_are_answered_from_seeded_knowledge(question, expected):
    assert expected in _run(_brain().respond(question))


def test_personal_notes_are_not_recited_as_general_knowledge():
    reply = _run(_brain().respond("what is a garden plan?"))
    assert "tomatoes" not in reply


def test_a_loose_match_falls_through_rather_than_being_recited():
    # "lock" shares no title word with "Deadlock"; nothing should be recited.
    reply = _run(_brain().respond("what is the capital of peru?"))
    assert "Coffman" not in reply and "network location" not in reply


def test_a_knowledge_answer_can_be_corrected():
    class _Learning:
        def __init__(self):
            self.calls = []

        async def correct(self, topic, detail):
            self.calls.append((topic, detail))
            return ToolResult("Corrected — I'll remember that.")

    learning = _Learning()
    host = HeadlessToolHost(_Bus(), _Memory(), learning=learning)
    brain = LocalBrain(_Bus(), _Memory(), host, None)
    _run(brain.respond("what is a deadlock?"))
    reply = _run(brain.respond("no, that's wrong — actually it needs five conditions"))
    assert learning.calls and "Corrected" in reply


# ── the node end to end ─────────────────────────────────────────────────────

def test_a_headless_gateway_answers_tasks_and_offline_chat(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    memory = _Memory()
    host = HeadlessToolHost(_Bus(), memory)

    class _Knowledge:
        def is_neuro_query(self, text):
            return False

        def answer(self, text):
            return None

    gateway = RemoteGateway(_NoModelRouter(), memory, _Bus(), dispatcher=host,
                            knowledge=_Knowledge(), config_dir=tmp_path)

    async def flow():
        code = gateway.begin_pairing()
        client = TestClient(TestServer(gateway._build_app()))
        await client.start_server()
        try:
            creds = await (await client.post(
                "/v1/auth/pair", json={"code": code, "device_name": "t"})).json()
            token = (await (await client.post("/v1/auth/token", json={
                "device_id": creds["device_id"],
                "refresh_token": creds["refresh_token"]})).json())["access_token"]
            auth = {"Authorization": f"Bearer {token}"}

            task = await client.post("/v1/agent/tasks", headers=auth, json={
                "tool": "query_intelligence", "args": {"query": "deadlock"}})
            assert task.status == 200
            assert "prog_deadlock" in (await task.json())["result"]

            chat = await client.post("/api/chat", headers=auth,
                                     json={"message": "what is a deadlock?"})
            assert chat.status == 200
            body = await chat.json()
            assert "Coffman" in body["reply"] and body["provider"] == "local-brain"
        finally:
            await client.close()

    _run(flow())
