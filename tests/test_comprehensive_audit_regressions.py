"""Behavioural regressions from the September 2026 repository audit."""

from __future__ import annotations

import asyncio
import io
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from orion_core import background
from orion_core.data import ToolResult
from orion_core.neural import MLP, SparseMLP
from orion_core.remote import RemoteAuthManager, RemoteGateway
from orion_core.remote_capability import RemoteConfirmationRegistry, RemoteToolGate


def test_reloaded_dense_network_restarts_adam_consistently():
    trained = MLP([2, 3, 1], seed=4)
    x, y = [[0.2, -0.3]], [[0.8]]
    for _ in range(20):
        trained.train_batch(x, y)
    loaded, _ = MLP.from_bytes(trained.to_bytes(include_optimizer=False))
    fresh = MLP(trained.sizes, seed=4)
    for target, source in zip(fresh.W + fresh.b, trained.W + trained.b):
        target[:] = source
    loaded.train_batch(x, y)
    fresh.train_batch(x, y)
    for actual, expected in zip(loaded.W + loaded.b, fresh.W + fresh.b):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert loaded.stats.steps == trained.stats.steps + 1


@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("damage", ["shape", "nan", "truncated"])
def test_corrupt_neural_checkpoint_is_rejected(tmp_path, kind, damage):
    net = MLP([2, 3, 1]) if kind == "dense" else SparseMLP(8, 3, 2)
    data = net.to_bytes()
    if damage == "truncated":
        data = data[:-20]
    else:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        key = "W0" if kind == "dense" else "table"
        if damage == "shape":
            arrays[key] = arrays[key][:-1]
        else:
            arrays[key].flat[0] = np.nan
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        data = output.getvalue()
    checkpoint = tmp_path / "network.npz"
    checkpoint.write_bytes(data)
    assert type(net).load(checkpoint) is None


def test_background_future_is_named_drained_and_failure_reported(monkeypatch):
    messages = []

    class Bus:
        class log:
            emit = staticmethod(messages.append)

    monkeypatch.setattr(background, "_RUNNING", set())
    monkeypatch.setattr(background, "_BUS", Bus())

    async def scenario():
        future = asyncio.get_running_loop().create_future()
        task = background.spawn(future, name="future-work")
        try:
            assert background.names() == ["future-work"]
            assert list(await background.drain(timeout=0)) == ["future-work"]
            future.set_exception(RuntimeError("worker failed"))
            with pytest.raises(RuntimeError, match="worker failed"):
                await task
            await asyncio.sleep(0)
            assert any("future-work" in line and "worker failed" in line for line in messages)
            assert background.in_flight() == 0
        finally:
            background.cancel_all()
            if not future.done():
                future.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_cancelling_unstarted_background_future_cancels_owned_work(monkeypatch):
    monkeypatch.setattr(background, "_RUNNING", set())

    async def scenario():
        future = asyncio.get_running_loop().create_future()
        task = background.spawn(future, name="queued-future")
        background.cancel_all()
        await asyncio.gather(task, return_exceptions=True)
        assert future.cancelled()

    asyncio.run(scenario())


def test_background_adopts_existing_task_without_replacing_it(monkeypatch):
    monkeypatch.setattr(background, "_RUNNING", set())

    async def scenario():
        task = asyncio.create_task(asyncio.sleep(0))
        assert background.spawn(task, name="existing-task") is task
        await task

    asyncio.run(scenario())


def test_access_token_with_unicode_signature_is_rejected(tmp_path):
    auth = RemoteAuthManager(config_dir=tmp_path)
    device, refresh = auth.complete_pairing(auth.begin_pairing())
    token, _ = auth.issue_access(device, refresh)
    encoded, _, _ = token.rpartition(".")
    assert auth.verify_access(encoded + ".caf\u00e9") is None
    assert auth.verify_access(token) == device


def test_unicode_credentials_fail_cleanly_through_http(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    async def scenario():
        dispatcher = RecordingDispatcher()
        gateway = RemoteGateway(
            None, object(), SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)),
            dispatcher=dispatcher, config_dir=tmp_path,
        )
        device, refresh = gateway.auth.complete_pairing(gateway.auth.begin_pairing())
        token, _ = gateway.auth.issue_access(device, refresh)
        encoded, _, _ = token.rpartition(".")
        async with TestClient(TestServer(gateway._build_app())) as client:
            response = await client.get(
                "/api/confirmations", params={"token": encoded + ".caf\u00e9"})
            assert response.status == 401
            conf = gateway.confirmations.create("messaging", {}, device, "send")
            response = await client.post(
                "/api/confirm",
                json={"id": conf.id, "token": "caf\u00e9", "decision": "approve"},
                headers={"Authorization": "Bearer " + token},
            )
            assert response.status == 409
            assert dispatcher.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["token", "device_id"])
def test_unicode_confirmation_credentials_are_rejected(field):
    registry = RemoteConfirmationRegistry()
    conf = registry.create("messaging", {"action": "send"}, "phone", "send")
    values = {"token": conf.token, "device_id": "phone"}
    values[field] = "caf\u00e9"
    assert registry.resolve(conf.id, approve=True, **values) is None
    assert conf.status == "pending"


def test_pending_confirmation_owns_its_nested_parameters():
    registry = RemoteConfirmationRegistry()
    args = {"action": "send", "payload": {"recipients": ["reviewed"]}}
    conf = registry.create("messaging", args, "phone", "send")
    args["payload"]["recipients"].append("unreviewed")
    assert conf.args["payload"]["recipients"] == ["reviewed"]


class RecordingDispatcher:
    def __init__(self):
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        await asyncio.sleep(0)
        return ToolResult("completed")


@pytest.mark.parametrize("state", ["pending", "denied", "expired", "forged"])
def test_unapproved_or_unregistered_confirmation_cannot_execute(state):
    registry = RemoteConfirmationRegistry()
    dispatcher = RecordingDispatcher()
    gate = RemoteToolGate(dispatcher, registry)
    conf = registry.create("messaging", {"action": "send"}, "phone", "send")
    if state == "forged":
        conf = replace(conf, status="approved")
    else:
        conf.status = state
    result = asyncio.run(gate.run_approved(conf))
    assert not result.ok
    assert dispatcher.calls == []


def test_concurrent_approved_executions_consume_authorisation_once():
    registry = RemoteConfirmationRegistry()
    dispatcher = RecordingDispatcher()
    gate = RemoteToolGate(dispatcher, registry)
    conf = registry.create("messaging", {"action": "send"}, "phone", "send")
    assert registry.resolve(conf.id, conf.token, True, device_id="phone") is conf

    async def scenario():
        results = await asyncio.gather(gate.run_approved(conf), gate.run_approved(conf))
        assert sum(result.ok for result in results) == 1

    asyncio.run(scenario())
    assert len(dispatcher.calls) == 1


def test_approved_action_is_checked_against_current_policy(monkeypatch):
    from orion_core import remote_capability

    monkeypatch.setattr(remote_capability, "_PLUGIN_TIERS", {})
    registry = RemoteConfirmationRegistry()
    dispatcher = RecordingDispatcher()
    gate = RemoteToolGate(dispatcher, registry)
    conf = registry.create("audit_plugin", {}, "phone", "use plugin")
    registry.resolve(conf.id, conf.token, True, device_id="phone")
    remote_capability.register_plugin_tier("audit_plugin", "forbid")
    result = asyncio.run(gate.run_approved(conf))
    assert not result.ok
    assert dispatcher.calls == []
