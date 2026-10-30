"""Remote administrative failures cannot strand the live chess engine."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from orion_core.chess_lichess import LichessError
from orion_core.grandmaster import bridge as bridge_module
from orion_core.grandmaster.bridge import GrandmasterBridge


def make_bridge(tmp_path, client):
    signal = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    result = GrandmasterBridge(
        SimpleNamespace(log=signal, dashboard_event=signal), None,
        settings_path=tmp_path / "lichess.json", grandmaster=None,
    )
    result.client, result.running = client, True
    return result


def test_expired_outgoing_challenge_releases_the_remote_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge_module, "time", SimpleNamespace(monotonic=lambda: 100.0))
    calls = []

    class Client:
        async def challenge(self, *arguments):
            calls.append(arguments)
            return {"challenge": {"id": "fresh001"}}

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        await bridge.authorise("friend", "Friend")
        bridge._pending["expired1"] = 99.0
        accepted, message = await bridge.challenge("friend")
        assert accepted, message
        assert "expired1" not in bridge._pending
        assert bridge._pending == {"fresh001": 130.0}

    asyncio.run(flow())
    assert len(calls) == 1


def test_unexpired_outgoing_challenge_keeps_the_remote_slot_reserved(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge_module, "time", SimpleNamespace(monotonic=lambda: 100.0))

    class Client:
        async def challenge(self, *_args):
            pytest.fail("A second outstanding challenge was sent")

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        await bridge.authorise("friend", "Friend")
        bridge._pending["pending1"] = 101.0
        accepted, message = await bridge.challenge("friend")
        assert not accepted and "occupied" in message
        assert bridge._pending == {"pending1": 101.0}

    asyncio.run(flow())


def test_slow_outgoing_write_cannot_expire_or_admit_a_second_challenge(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(bridge_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def flow():
        entered, release = asyncio.Event(), asyncio.Event()
        calls, declined = [], []

        class Client:
            async def challenge(self, *arguments):
                calls.append(arguments)
                entered.set()
                await release.wait()
                return {"challenge": {"id": "fresh001"}}

            async def decline_challenge(self, identifier, reason):
                declined.append((identifier, reason))

        bridge = make_bridge(tmp_path, Client())
        await bridge.authorise("friend", "Friend")
        first = asyncio.create_task(bridge.challenge("friend"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            clock[0] = 131.0
            second = asyncio.create_task(bridge.challenge("friend"))
            try:
                done, _pending = await asyncio.wait({second}, timeout=0.5)
                assert second in done, "A second challenge reached the blocked network request"
                accepted, message = second.result()
                assert not accepted and "occupied" in message
                await bridge._on_challenge({
                    "id": "incoming", "challenger": {"id": "friend"},
                    "variant": {"key": "standard"},
                    "timeControl": {"type": "clock", "limit": 300},
                })
                assert declined == [("incoming", "later")]
            finally:
                second.cancel()
                await asyncio.gather(second, return_exceptions=True)
        finally:
            release.set()
            first_result = await asyncio.wait_for(first, 1)
        assert first_result[0]
        assert len(calls) == 1
        assert bridge._pending == {"fresh001": 161.0}

    asyncio.run(flow())


def test_slow_incoming_accept_keeps_its_slot_and_ignores_duplicate_delivery(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(bridge_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def flow():
        entered, release = asyncio.Event(), asyncio.Event()
        accepted, declined = [], []

        class Client:
            async def accept_challenge(self, identifier):
                accepted.append(identifier)
                entered.set()
                await release.wait()

            async def decline_challenge(self, identifier, reason):
                declined.append((identifier, reason))

            async def challenge(self, *_arguments):
                pytest.fail("An outgoing challenge bypassed the incoming acceptance owner")

        def challenge(identifier):
            return {"id": identifier, "challenger": {"id": "friend"},
                    "variant": {"key": "standard"},
                    "timeControl": {"type": "clock", "limit": 300}}

        bridge = make_bridge(tmp_path, Client())
        await bridge.authorise("friend", "Friend")
        first = asyncio.create_task(bridge._on_challenge(challenge("incoming")))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            clock[0] = 131.0
            result, message = await bridge.challenge("friend")
            assert not result and "occupied" in message
            await bridge._on_challenge(challenge("incoming"))
            await bridge._on_challenge(challenge("other001"))
            assert declined == [("other001", "later")]
        finally:
            release.set()
            await asyncio.wait_for(first, 1)
        assert accepted == ["incoming"]
        assert bridge._pending == {"incoming": 161.0}

    asyncio.run(flow())


def test_cancelling_challenge_holds_ownership_until_transport_cancellation_settles(tmp_path):
    async def flow():
        entered, cancelled, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        class Client:
            async def challenge(self, *arguments):
                calls.append(arguments)
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await settled.wait()
                    raise

        bridge = make_bridge(tmp_path, Client())
        await bridge.authorise("friend", "Friend")
        first = asyncio.create_task(bridge.challenge("friend"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            first.cancel()
            await asyncio.wait_for(cancelled.wait(), 1)
            accepted, message = await bridge.challenge("friend")
            assert not accepted and "occupied" in message
            assert len(calls) == 1
            assert bridge._challenge_writes
        finally:
            settled.set()
            await asyncio.gather(first, return_exceptions=True)
        assert not bridge._challenge_writes
        assert not bridge._pending

    asyncio.run(flow())


@pytest.mark.parametrize("fault", [OSError("offline"), asyncio.CancelledError()])
def test_failed_or_cancelled_abort_does_not_permanently_stop_move_generation(tmp_path, fault):
    class Client:
        async def abort(self, _game_id):
            raise fault

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        with pytest.raises(type(fault)):
            await bridge.abort_game("active01")
        assert "active01" not in bridge._stopping

    asyncio.run(flow())


def test_failed_abort_preserves_an_authoritative_terminal_marker(tmp_path):
    class Client:
        async def abort(self, _game_id):
            raise OSError("offline")

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        bridge._stopping.add("ended001")
        with pytest.raises(OSError):
            await bridge.abort_game("ended001")
        assert "ended001" in bridge._stopping

    asyncio.run(flow())


def test_failed_abort_preserves_a_terminal_event_received_during_the_request(tmp_path):
    async def flow():
        class Client:
            async def abort(self, game_id):
                bridge._stopping.add(game_id)
                raise OSError("offline")

        bridge = make_bridge(tmp_path, Client())
        with pytest.raises(OSError):
            await bridge.abort_game("ended001")
        assert "ended001" in bridge._stopping
        assert not bridge._stop_requests

    asyncio.run(flow())


def test_failed_abort_wakes_a_quiet_game_stream_to_resume_from_server_authority(tmp_path):
    async def flow():
        entered = asyncio.Event()

        class Client:
            async def abort(self, _game_id):
                raise OSError("offline")

        async def quiet_stream():
            entered.set()
            await asyncio.Event().wait()
            yield {}

        bridge = make_bridge(tmp_path, Client())
        events = quiet_stream()
        watched = bridge._watched(events, "active01")
        waiting = asyncio.create_task(anext(watched))
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(OSError):
            await bridge.abort_game("active01")
        with pytest.raises(LichessError, match="fresh game state"):
            await asyncio.wait_for(waiting, 1)
        await watched.aclose()
        await events.aclose()
        assert not bridge._stop_requests

    asyncio.run(flow())


def test_refused_abort_falls_back_to_resign_and_keeps_successful_stop_marker(tmp_path):
    calls = []

    class Client:
        async def abort(self, game_id):
            calls.append(("abort", game_id))
            raise LichessError(400, "The game has started")

        async def resign(self, game_id):
            calls.append(("resign", game_id))

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        await bridge.abort_game("active01")
        assert "active01" in bridge._stopping

    asyncio.run(flow())
    assert calls == [("abort", "active01"), ("resign", "active01")]


def test_failed_resign_fallback_restores_live_move_generation(tmp_path):
    class Client:
        async def abort(self, _game_id):
            raise LichessError(400, "The game has started")

        async def resign(self, _game_id):
            raise LichessError(503, "Unavailable")

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        with pytest.raises(LichessError):
            await bridge.abort_game("active01")
        assert "active01" not in bridge._stopping

    asyncio.run(flow())


def test_remote_resignation_cannot_report_success_when_both_requests_failed(tmp_path):
    from orion_core.chess_engine import ChessService
    from orion_core.chess_lichess import LichessGame

    class Client:
        async def resign(self, _game_id):
            raise OSError("Unavailable")

        async def abort(self, _game_id):
            raise OSError("Unavailable")

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        bridge.games["active01"] = LichessGame("active01")
        result = await ChessService.lichess_resign(SimpleNamespace(lichess=bridge))
        assert not result.ok
        assert "could not confirm" in result.text.lower()
        assert "Resigned my" not in result.text

    asyncio.run(flow())


def test_remote_resignation_accepts_an_abort_receipt_without_claiming_the_game_finished(tmp_path):
    from orion_core.chess_engine import ChessService
    from orion_core.chess_lichess import LichessGame

    calls = []

    class Client:
        async def resign(self, game_id):
            calls.append(("resign", game_id))
            raise LichessError(400, "Not started")

        async def abort(self, game_id):
            calls.append(("abort", game_id))

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        bridge.games["active01"] = LichessGame("active01")
        result = await ChessService.lichess_resign(SimpleNamespace(lichess=bridge))
        assert result.ok and "1 stop request(s)" in result.text
        assert "Waiting for the final game state" in result.text
        assert bridge.games["active01"].status == "started"

    asyncio.run(flow())
    assert calls == [("resign", "active01"), ("abort", "active01")]


def test_remote_resignation_reports_partial_failure_without_exposing_transport_secrets(tmp_path):
    from orion_core.chess_engine import ChessService
    from orion_core.chess_lichess import LichessGame

    class Client:
        async def resign(self, game_id):
            if game_id == "failed01":
                raise RuntimeError("private-token-in-error")

        async def abort(self, _game_id):
            raise RuntimeError("private-token-in-error")

    async def flow():
        bridge = make_bridge(tmp_path, Client())
        bridge.games = {name: LichessGame(name) for name in ("active01", "failed01")}
        result = await ChessService.lichess_resign(SimpleNamespace(lichess=bridge))
        assert not result.ok
        assert "1 game(s)" in result.text and "1 stop request(s) confirmed" in result.text
        assert "private-token" not in result.text

    asyncio.run(flow())
