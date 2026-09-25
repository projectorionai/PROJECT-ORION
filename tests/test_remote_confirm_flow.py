"""
HTTP end-to-end test for the full-parity remote confirmation loop (#12).

A CONFIRM-tier action is parked as a pending confirmation; the phone approves it
over /api/confirm with the single-use token; only then does the tool reach the
dispatcher. Wrong tokens and unauthenticated calls are refused.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.remote import RemoteGateway
from orion_core.remote_capability import describe_action


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Memory:
    def log_episode(self, *a):
        pass

    def prompt_context(self, limit=18):
        return ""


class _RecordingDispatcher:
    def __init__(self):
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, dict(args or {})))
        return ToolResult(f"ran {name}")


def _gateway(tmp_path, dispatcher):
    return RemoteGateway(object(), _Memory(), _Bus(),
                         dispatcher=dispatcher, config_dir=tmp_path)


async def _client(gateway):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(gateway._build_app()))
    await client.start_server()
    return client


async def _pair_and_token(client, gateway):
    code = gateway.begin_pairing()
    creds = await (await client.post("/v1/auth/pair", json={"code": code})).json()
    tok = (await (await client.post("/v1/auth/token", json={
        "device_id": creds["device_id"],
        "refresh_token": creds["refresh_token"]})).json())["access_token"]
    return creds["device_id"], tok


def test_confirm_approval_runs_the_tool(tmp_path):
    async def flow():
        dispatcher = _RecordingDispatcher()
        gateway = _gateway(tmp_path, dispatcher)
        client = await _client(gateway)
        try:
            device_id, token = await _pair_and_token(client, gateway)
            hdr = {"Authorization": f"Bearer {token}"}

            # A CONFIRM action parks (as the gate would do on a remote turn).
            conf = gateway.confirmations.create(
                "messaging", {"action": "send", "contact": "+44"},
                device_id, describe_action("messaging", {"action": "send"}))
            assert dispatcher.calls == []          # nothing has run yet

            # Wrong token is refused; the tool still hasn't run.
            bad = await client.post("/api/confirm", json={
                "id": conf.id, "token": "wrong", "decision": "approve"}, headers=hdr)
            assert bad.status == 409
            assert dispatcher.calls == []

            # Correct token approves → the tool finally runs, exactly once.
            ok = await client.post("/api/confirm", json={
                "id": conf.id, "token": conf.token, "decision": "approve"}, headers=hdr)
            body = await ok.json()
            assert ok.status == 200 and body["ok"] and body["status"] == "done"
            assert dispatcher.calls == [("messaging", {"action": "send", "contact": "+44"})]

            # Replay of the same approval is refused (single-use).
            replay = await client.post("/api/confirm", json={
                "id": conf.id, "token": conf.token, "decision": "approve"}, headers=hdr)
            assert replay.status == 409
            assert len(dispatcher.calls) == 1
        finally:
            await client.close()
    asyncio.run(flow())


def test_confirm_denial_does_not_run_the_tool(tmp_path):
    async def flow():
        dispatcher = _RecordingDispatcher()
        gateway = _gateway(tmp_path, dispatcher)
        client = await _client(gateway)
        try:
            device_id, token = await _pair_and_token(client, gateway)
            conf = gateway.confirmations.create(
                "peripherals", {"action": "shutdown"}, device_id, "shut down")
            resp = await client.post("/api/confirm", json={
                "id": conf.id, "token": conf.token, "decision": "deny"},
                headers={"Authorization": f"Bearer {token}"})
            body = await resp.json()
            assert resp.status == 200 and body["status"] == "denied"
            assert dispatcher.calls == []
        finally:
            await client.close()
    asyncio.run(flow())


def test_confirm_requires_authentication(tmp_path):
    async def flow():
        dispatcher = _RecordingDispatcher()
        gateway = _gateway(tmp_path, dispatcher)
        client = await _client(gateway)
        try:
            conf = gateway.confirmations.create("messaging", {"action": "send"}, "d", "send")
            resp = await client.post("/api/confirm", json={
                "id": conf.id, "token": conf.token, "decision": "approve"})
            assert resp.status == 401
            assert dispatcher.calls == []
        finally:
            await client.close()
    asyncio.run(flow())


def test_other_paired_device_cannot_approve_or_receive_confirmation(tmp_path):
    async def flow():
        dispatcher = _RecordingDispatcher()
        gateway = _gateway(tmp_path, dispatcher)
        client = await _client(gateway)
        try:
            owner_id, owner_token = await _pair_and_token(client, gateway)
            other_id, other_token = await _pair_and_token(client, gateway)
            owner_events = asyncio.Queue()
            other_events = asyncio.Queue()
            gateway._event_devices[owner_events] = owner_id
            gateway._event_devices[other_events] = other_id
            conf = gateway.confirmations.create(
                "messaging", {"action": "send"}, owner_id, "send")
            await gateway._push_confirm(owner_id, conf)
            assert (await owner_events.get())["token"] == conf.token
            assert other_events.empty()

            denied = await client.post("/api/confirm", json={
                "id": conf.id, "token": conf.token, "decision": "approve"},
                headers={"Authorization": f"Bearer {other_token}"})
            assert denied.status == 409
            assert dispatcher.calls == []

            approved = await client.post("/api/confirm", json={
                "id": conf.id, "token": conf.token, "decision": "approve"},
                headers={"Authorization": f"Bearer {owner_token}"})
            assert approved.status == 200
            assert len(dispatcher.calls) == 1
        finally:
            await client.close()
    asyncio.run(flow())
