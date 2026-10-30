"""Remote PC power tests inject all OS actions; the test machine stays on."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from orion_core.remote_power import PowerError, RemotePowerController


def _id():
    return str(uuid.uuid4())


class Runner:
    def __init__(self, result=0):
        self.calls = 0
        self.result = result

    async def __call__(self):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_shutdown_waits_for_confirmation_and_countdown(tmp_path):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0.02)
        try:
            assert (await control.status("phone"))["pending"] is None
            challenge = await control.prepare("phone")
            assert runner.calls == 0
            response = await control.shutdown("phone", _id(), challenge["confirmation"], lambda: True)
            assert response["state"] == "scheduled" and runner.calls == 0
            await control._timer
            assert runner.calls == 1
            assert (await control.status("phone"))["pending"]["state"] == "initiated"
        finally:
            await control.close()
    asyncio.run(flow())


def test_duplicate_request_never_runs_twice_even_after_restart(tmp_path):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0)
        key = _id()
        challenge = await control.prepare("phone")
        await control.shutdown("phone", key, challenge["confirmation"], lambda: True)
        await control._timer
        assert (await control.shutdown("phone", key, "consumed", lambda: True))["state"] == "initiated"
        await control.close()
        restarted = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0)
        assert (await restarted.shutdown("phone", key, "consumed", lambda: True))["state"] == "initiated"
        assert runner.calls == 1
        await restarted.close()
    asyncio.run(flow())


@pytest.mark.parametrize("bad", [None, True, "", "x", str(uuid.uuid1()), "A" * 36])
def test_malformed_request_ids_are_refused(tmp_path, bad):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True)
        with pytest.raises(PowerError) as caught:
            await control.shutdown("phone", bad, "anything", lambda: True)
        assert caught.value.status == 400 and runner.calls == 0
        await control.close()
    asyncio.run(flow())


def test_confirmation_is_expiring_and_device_bound(tmp_path):
    async def flow():
        now = [1000.0]
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, clock=lambda: now[0])
        challenge = await control.prepare("alice")
        with pytest.raises(PowerError) as caught:
            await control.shutdown("bob", _id(), challenge["confirmation"], lambda: True)
        assert caught.value.status == 403
        now[0] += 91
        with pytest.raises(PowerError) as caught:
            await control.shutdown("alice", _id(), challenge["confirmation"], lambda: True)
        assert caught.value.status == 403 and runner.calls == 0
        await control.close()
    asyncio.run(flow())


def test_cancel_is_owner_only_and_prevents_execution(tmp_path):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0.05)
        challenge = await control.prepare("alice")
        key = _id()
        await control.shutdown("alice", key, challenge["confirmation"], lambda: True)
        with pytest.raises(PowerError) as caught:
            await control.cancel("bob", _id(), key)
        assert caught.value.status == 404
        cancel_id = _id()
        assert (await control.cancel("alice", cancel_id, key))["state"] == "cancelled"
        assert (await control.cancel("alice", cancel_id, key))["state"] == "cancelled"
        with pytest.raises(PowerError) as mismatched:
            await control.cancel("alice", cancel_id, _id())
        assert mismatched.value.status == 409
        await asyncio.sleep(0.06)
        assert runner.calls == 0
        await control.close()
    asyncio.run(flow())


def test_parallel_shutdowns_admit_one_os_request(tmp_path):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0.02)
        first = await control.prepare("alice")
        second = await control.prepare("bob")
        results = await asyncio.gather(
            control.shutdown("alice", _id(), first["confirmation"], lambda: True),
            control.shutdown("bob", _id(), second["confirmation"], lambda: True),
            return_exceptions=True)
        assert sum(isinstance(result, PowerError) and result.status == 409 for result in results) == 1
        await control._timer
        assert runner.calls == 1
        await control.close()
    asyncio.run(flow())


def test_revocation_during_countdown_cancels_power_action(tmp_path):
    async def flow():
        runner, authorised = Runner(), [True]
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0.02)
        challenge = await control.prepare("phone")
        await control.shutdown("phone", _id(), challenge["confirmation"], lambda: authorised[0])
        authorised[0] = False
        await control._timer
        assert runner.calls == 0
        assert (await control.status("phone"))["pending"]["state"] == "cancelled"
        await control.close()
    asyncio.run(flow())


def test_close_and_restart_never_resurrect_a_countdown(tmp_path):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0.02)
        challenge = await control.prepare("phone")
        key = _id()
        await control.shutdown("phone", key, challenge["confirmation"], lambda: True)
        await control.close()
        restarted = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0)
        state = await restarted.status("phone")
        assert state["pending"]["state"] == "cancelled"
        await asyncio.sleep(0.03)
        assert runner.calls == 0
        await restarted.close()
    asyncio.run(flow())


def test_unclean_restart_cancels_persisted_schedule(tmp_path):
    async def flow():
        runner = Runner()
        key = _id()
        ledger = {key: {"device_id": "phone", "action": "shutdown", "state": "scheduled",
                        "created_at": 1000.0, "shutdown_at": 2000.0, "message": "Scheduled"}}
        (tmp_path / "remote_power_requests.json").write_text(json.dumps(ledger), encoding="utf-8")
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0)
        assert (await control.status("phone"))["pending"]["state"] == "cancelled"
        await asyncio.sleep(0)
        assert runner.calls == 0
        await control.close()
    asyncio.run(flow())


@pytest.mark.parametrize("result,state", [(5, "blocked"), (OSError("refused"), "uncertain"),
                                         (asyncio.TimeoutError(), "uncertain")])
def test_os_failure_is_honest_and_never_retried(tmp_path, result, state):
    async def flow():
        runner = Runner(result)
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0)
        challenge = await control.prepare("phone")
        key = _id()
        await control.shutdown("phone", key, challenge["confirmation"], lambda: True)
        await control._timer
        assert (await control.status("phone"))["pending"]["state"] == state
        await control.shutdown("phone", key, "used", lambda: True)
        assert runner.calls == 1
        await control.close()
    asyncio.run(flow())


def test_disk_failure_does_not_schedule_os_action(tmp_path, monkeypatch):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0)
        challenge = await control.prepare("phone")
        from orion_core import remote_power
        monkeypatch.setattr(remote_power, "atomic_write_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("locked")))
        with pytest.raises(PowerError) as caught:
            await control.shutdown("phone", _id(), challenge["confirmation"], lambda: True)
        assert caught.value.status == 503 and runner.calls == 0 and control._timer is None
        await control.close()
    asyncio.run(flow())


def test_countdown_state_save_failure_is_terminal_without_os_admission(tmp_path, monkeypatch):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0.02)
        challenge = await control.prepare("phone")
        await control.shutdown("phone", _id(), challenge["confirmation"], lambda: True)
        from orion_core import remote_power
        original = remote_power.atomic_write_text
        failures = [True]
        def saving(*args, **kwargs):
            if failures[0]:
                failures[0] = False
                raise OSError("locked during countdown")
            return original(*args, **kwargs)
        monkeypatch.setattr(remote_power, "atomic_write_text", saving)
        await control._timer
        assert runner.calls == 0
        status = await control.status("phone")
        assert status["pending"]["state"] == "blocked"
        assert "not requested" in status["pending"]["message"]
        assert (await control.prepare("phone"))["confirmation"]
        await control.close()
    asyncio.run(flow())


def test_timeout_and_process_reap_are_bounded(monkeypatch):
    async def flow():
        from orion_core import remote_power
        limits = []
        class Process:
            returncode = None
            killed = False
            def kill(self):
                self.killed = True
            async def wait(self):
                return 0
        process = Process()
        async def create(*args, **kwargs):
            return process
        async def wait_for(awaitable, timeout):
            limits.append(timeout)
            if timeout == 3:
                return await awaitable
            awaitable.close()
            raise asyncio.TimeoutError()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        monkeypatch.setattr(asyncio, "wait_for", wait_for)
        with pytest.raises(asyncio.TimeoutError):
            await remote_power._windows_shutdown()
        assert process.killed and limits == [3.0, 8.0, 1.0]
    asyncio.run(flow())


def test_closing_during_admitted_command_reports_uncertainty(tmp_path):
    async def flow():
        started = asyncio.Event()
        async def runner():
            started.set()
            await asyncio.sleep(100)
            return 0
        control = RemotePowerController(tmp_path, runner=runner, windows=True, delay_s=0)
        challenge = await control.prepare("phone")
        await control.shutdown("phone", _id(), challenge["confirmation"], lambda: True)
        await started.wait()
        await asyncio.wait_for(control.close(), timeout=0.5)
        assert (await control.status("phone"))["pending"]["state"] == "uncertain"
        await control.close()
    asyncio.run(flow())


def test_daily_ledger_limit_prunes_only_old_terminal_records(tmp_path):
    async def flow():
        from orion_core.remote_power import MAX_RECORDS
        now = [200000.0]
        control = RemotePowerController(tmp_path, runner=Runner(), windows=True, clock=lambda: now[0])
        await control.status("phone")
        control._records = {_id(): {"device_id": "phone", "action": "cancel", "state": "cancelled",
                                   "created_at": now[0], "message": "Cancelled"}
                            for _ in range(MAX_RECORDS)}
        with pytest.raises(PowerError) as caught:
            await control.prepare("phone")
        assert caught.value.status == 429
        now[0] += 86401
        assert (await control.prepare("phone"))["confirmation"]
        assert len(control._records) == 0
        await control.close()
    asyncio.run(flow())


@pytest.mark.parametrize("content", ['{', '[]', '{"x": {}}'])
def test_corrupt_ledger_fails_closed(tmp_path, content):
    async def flow():
        (tmp_path / "remote_power_requests.json").write_text(content, encoding="utf-8")
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=True)
        with pytest.raises(PowerError) as caught:
            await control.prepare("phone")
        assert caught.value.status == 503 and runner.calls == 0
        await control.close()
    asyncio.run(flow())


def test_non_windows_host_never_offers_shutdown(tmp_path):
    async def flow():
        runner = Runner()
        control = RemotePowerController(tmp_path, runner=runner, windows=False)
        assert not (await control.status("phone"))["available"]
        with pytest.raises(PowerError):
            await control.prepare("phone")
        assert runner.calls == 0
        await control.close()
    asyncio.run(flow())


def test_fixed_windows_command_has_no_force_or_remote_target(monkeypatch):
    async def flow():
        from orion_core import remote_power
        captured = []
        class Process:
            returncode = 0
            async def wait(self):
                return 0
        async def create(*args, **kwargs):
            captured.extend(args)
            assert "shell" not in kwargs
            return Process()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        assert await remote_power._windows_shutdown() == 0
        assert captured[1:5] == ["/s", "/t", "0", "/d"]
        assert "/f" not in captured and "/m" not in captured
    asyncio.run(flow())


class _Signal:
    def emit(self, *args):
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


def test_routes_enforce_pairing_strict_body_and_owner_control(tmp_path):
    async def flow():
        from aiohttp.test_utils import TestClient, TestServer
        from orion_core.remote import RemoteGateway
        gateway = RemoteGateway(object(), object(), _Bus(), config_dir=tmp_path)
        client = TestClient(TestServer(gateway._build_app()))
        control = gateway.power_controller
        runner = Runner()
        control._runner, control._windows, control._delay_s = runner, True, 1
        await client.start_server()
        async def pair():
            credentials = gateway.auth.complete_pairing(gateway.begin_pairing())
            token, _ = gateway.auth.issue_access(*credentials)
            return {"Authorization": "Bearer " + token}, credentials[0]
        try:
            assert (await client.get("/api/pc/power")).status == 401
            assert (await client.post("/api/pc/power/shutdown", json={})).status == 401
            headers, owner = await pair()
            other, _ = await pair()
            assert (await client.post("/api/pc/power/prepare", headers=headers, json={"extra": 1})).status == 400
            assert (await client.post("/api/pc/power/prepare", headers=headers, data=b"x" * 2049)).status == 413
            challenge = await (await client.post("/api/pc/power/prepare", headers=headers, json={})).json()
            assert runner.calls == 0
            key = _id()
            response = await client.post("/api/pc/power/shutdown", headers=headers,
                                         json={"request_id": key, "confirmation": challenge["confirmation"]})
            assert response.status == 200 and (await response.json())["state"] == "scheduled"
            assert runner.calls == 0
            assert (await client.post("/api/pc/power/cancel", headers=other,
                                      json={"request_id": _id(), "shutdown_id": key})).status == 404
            assert (await client.get("/api/pc/power", headers=other)).status == 200
            assert (await (await client.get("/api/pc/power", headers=other)).json())["pending"] is None
            assert (await client.post("/api/pc/power/cancel", headers=headers,
                                      json={"request_id": _id(), "shutdown_id": key})).status == 200
            assert runner.calls == 0
            # An access token cannot be substituted into a query string.
            assert (await client.get("/api/pc/power", params={"token": headers["Authorization"][7:]})).status == 401
            gateway.auth.revoke_device(owner)
            assert (await client.get("/api/pc/power", headers=headers)).status == 401
        finally:
            await client.close()
            assert control._closed
    asyncio.run(flow())


def test_gateway_stop_cancels_countdown_even_before_runner_cleanup(tmp_path):
    async def flow():
        from orion_core.remote import RemoteGateway
        gateway = RemoteGateway(object(), object(), _Bus(), config_dir=tmp_path)
        gateway._build_app()
        control = gateway.power_controller
        runner = Runner()
        control._runner, control._windows, control._delay_s = runner, True, 0.01
        challenge = await control.prepare("phone")
        await control.shutdown("phone", _id(), challenge["confirmation"], lambda: True)
        await gateway.stop()
        await asyncio.sleep(0.02)
        assert runner.calls == 0 and control._closed
    asyncio.run(flow())
