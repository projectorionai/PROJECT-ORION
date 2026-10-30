"""Offline regression tests: no bot token, network, real data, or CUDA required."""
from __future__ import annotations

import asyncio
import random
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import chess
import pytest

from orion_core.grandmaster.identity import Identity, IdentityStore
from orion_core.grandmaster.search import (
    Position,
    SearchProcess,
    make_brain,
    run_search,
)
from orion_core.grandmaster.strength import (
    Limits,
    handicap,
    human_delay,
    select_move,
    time_budget,
)


def test_sqlite_identity_migration_revoke_and_mapping(tmp_path):
    store = IdentityStore(tmp_path / "identities.sqlite3")
    store.migrate(["Jordan_Chess"])
    store.put(Identity("JORDAN_CHESS", "Jordan, long-term friend", 1500, True))
    assert store.get("jordan_chess").entity == "Jordan, long-term friend"
    assert store.get("unknown") is None
    assert store.get("../bad") is None
    store.remove("Jordan_Chess")
    store.migrate(["Jordan_Chess"])
    assert store.get("jordan_chess") is None


def test_fen_ipc_preserves_repetition_and_rejects_tampering():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(uci)
    position = Position.from_board(board)
    assert position.board().can_claim_threefold_repetition()
    with pytest.raises(ValueError, match="mismatch"):
        replace(position, fen=chess.STARTING_FEN).board()


def test_handicap_nodes_and_root_distribution(tmp_path):
    weak, strong = handicap(800, 0.1), handicap(2400, 0.1)
    assert weak.depth < strong.depth and weak.nodes < strong.nodes
    brain = make_brain(str(tmp_path / "brain.json"))
    result = run_search(brain, Position.from_board(chess.Board()), Limits(0.5, 20, 100, 800, True))
    assert result.nodes <= 100
    assert chess.Move.from_uci(result.uci) in chess.Board().legal_moves
    rng = random.Random(31)
    limits = Limits(elo=2604, multipv=True, exploration=0.75)
    choices = {select_move([("a2a3", 0), ("h2h3", -100)], limits, rng)[0] for _ in range(100)}
    assert choices == {"a2a3", "h2h3"}


def test_time_reserve_includes_delay():
    for clock in [1, 10, 1000, 10000, 180000]:
        search, total = time_budget(clock, 0)
        delay = human_delay(total, search, 500, True, random.Random(1))
        assert delay >= 0 and total <= clock / 1000
        assert search + delay <= max(0.005, total)


def test_real_spawn_search_loop_responsive_and_cancel_drains(tmp_path):
    async def run():
        worker = SearchProcess(tmp_path / "brain.json", tmp_path)
        position = Position.from_board(chess.Board())
        ticks = []
        async def heartbeat():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)
        pulse = asyncio.create_task(heartbeat())
        try:
            result = await worker.search(position, Limits(seconds=0.1))
            assert chess.Move.from_uci(result.uci) in chess.Board().legal_moves
            task = asyncio.create_task(worker.search(position, Limits(seconds=3)))
            await asyncio.sleep(0.1)
            start = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert time.monotonic() - start < 1.0
            again = await worker.search(position, Limits(seconds=0.05))
            assert again.uci
            assert len(ticks) >= 10
            assert max(b - a for a, b in pairwise(ticks)) < 0.3
        finally:
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
            await worker.close()
    asyncio.run(run())


def test_qsearch_target_and_terminal_wdl_perspective(tmp_path):
    from orion_core.grandmaster.dataset import Dataset
    from orion_core.grandmaster.selfplay import quiescent_target
    brain = make_brain(str(tmp_path / "brain.json"))
    board = chess.Board()
    cp = quiescent_target(brain, board, 0.1)
    assert isinstance(cp, int)
    positions = [(board.fen(), cp)]
    board.push_uci("e2e4")
    positions.append((board.fen(), -cp))
    dataset = Dataset(tmp_path / "data.sqlite3")
    try:
        for _ in range(20):
            dataset.append(positions, "1-0", "classical")
        assert dataset.count() == 40
        rows = dataset.sample(100, random.Random(4))
        assert len(rows) == 100
        for fen, _cp, wdl in rows:
            assert wdl == (1 if fen.split()[1] == "w" else -1)
        with pytest.raises(ValueError, match="Unfinished"):
            dataset.append(positions, "*", "classical")
    finally:
        dataset.close()


def test_multipv_uses_complete_iterations(tmp_path):
    brain = make_brain(str(tmp_path / "brain.json"))
    result = run_search(brain, Position.from_board(chess.Board()), Limits(0.3, 2, 50000, 1600, True))
    assert len(result.candidates) == 20
    assert result.cp == dict(result.candidates)[result.uci]


def test_nnue_torch_matches_incremental_mmap_and_gradients(tmp_path):
    torch = pytest.importorskip("torch")
    from orion_core.grandmaster.nnue import HalfKP, batch
    from orion_core.grandmaster.weights import Inference, export_weights
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = HalfKP()
    path = tmp_path / "model.nnue"
    export_weights(model.state_dict(), path)
    inference = Inference.load(path)
    boards = [chess.Board(), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
              chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
              chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")]
    rng = random.Random(6)
    for board in boards:
        for _ in range(6):
            ours, theirs, target = batch([(board.fen(), 35, 1)])
            value = model(ours, theirs)
            assert abs(inference.evaluate(board) - value.item() * 300) < 0.51
            loss = torch.nn.functional.binary_cross_entropy_with_logits(value, target)
            loss.backward()
            assert torch.isfinite(model.feature.weight.grad).all()
            model.zero_grad(set_to_none=True)
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
            if rng.random() < 0.3:
                board.pop()                    # verify the unmake accumulator too


def test_vram_gate_fails_closed_and_caps_8gb():
    from orion_core.grandmaster.nnue import cuda_budget
    gib = 1024**3
    assert cuda_budget(0, 8*gib) == 0
    assert cuda_budget(2*gib, 8*gib) == 0
    assert cuda_budget(8*gib, 8*gib) <= 1280 * 1024**2
    assert cuda_budget(32*gib, 32*gib) <= 1280 * 1024**2


def test_champion_promotion_gate_and_stale_manifest(tmp_path):
    pytest.importorskip("torch")
    from orion_core.grandmaster.arena import lower_bound, promote
    from orion_core.grandmaster.nnue import HalfKP
    from orion_core.grandmaster.weights import Inference, active_model, export_weights
    candidate = tmp_path / "candidate.nnue"
    export_weights(HalfKP().state_dict(), candidate)
    assert not promote(tmp_path, candidate, None, [0.5] * 80)
    assert not promote(tmp_path, candidate, None, [1.0] * 10)
    assert lower_bound([1] * 80) > 0.5
    assert promote(tmp_path, candidate, None, [1.0] * 80)
    champion = active_model(tmp_path)
    mapped = Inference.load(champion)
    assert isinstance(mapped.evaluate(chess.Board()), int)
    assert not promote(tmp_path, candidate, None, [1.0] * 80)
    assert active_model(tmp_path) == champion


def test_polyglot_compilation_castling_and_deduplication(tmp_path):
    import chess.polyglot

    from orion_core.grandmaster.book import compile_book, raw_move
    pgn = tmp_path / "masters.pgn"
    pgn.write_text('[WhiteElo "2800"]\n[BlackElo "2750"]\n[Result "1-0"]\n\n'
                   '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0\n', encoding="utf-8")
    assert compile_book([pgn], tmp_path) == 10
    before = (tmp_path / "book.bin").read_bytes()
    compile_book([pgn], tmp_path)
    assert (tmp_path / "book.bin").read_bytes() == before
    with chess.polyglot.open_reader(tmp_path / "book.bin") as reader:
        board = chess.Board()
        assert reader.find(board).move.uci() == "e2e4"
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]:
            board.push_uci(uci)
        assert reader.find(board).move.uci() == "e1g1"
        assert raw_move(board, chess.Move.from_uci("e1g1")) & 63 == chess.H1


def test_missing_tablebases_fall_back(tmp_path):
    from orion_core.grandmaster.book import tablebase_move
    board = chess.Board("8/8/8/8/8/3k4/8/3KR3 w - - 0 1")
    assert tablebase_move(board, tmp_path) is None


class Signal:
    def emit(self, *_args, **_kwargs):
        pass


class FakeSearch:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def search(self, *_args):
        self.started.set()
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_terminal_event_cancels_search_and_sends_no_stale_move(tmp_path):
    from orion_core.chess_lichess import LichessGame
    from orion_core.grandmaster.bridge import GrandmasterBridge
    async def run():
        search = FakeSearch()
        moves = []
        snapshots = []
        class PositionSignal:
            def emit(self, _topic, payload):
                if payload.get("phase") == "lichess_position":
                    snapshots.append(payload["fen"])
        class Client:
            async def stream_game(self, game_id):
                yield {"type": "gameFull", "white": {"id": "orionbot"}, "black": {"id": "friend"},
                       "state": {"moves": "", "status": "started", "wtime": 180000, "btime": 180000}}
                await search.started.wait()
                yield {"type": "gameState", "moves": "f2f3 e7e5 g2g4 d8h4", "status": "mate", "winner": "black"}
            async def make_move(self, game_id, uci):
                moves.append(uci)
        bridge = GrandmasterBridge(SimpleNamespace(log=Signal(), dashboard_event=PositionSignal()), None,
            settings_path=tmp_path / "settings.json", grandmaster=SimpleNamespace(search=search))
        bridge.client, bridge.username = Client(), "orionbot"
        await bridge.authorise("friend", "Known friend")
        game = LichessGame("abcdefgh")
        result = await asyncio.wait_for(bridge._play_stream(game, {}), 2)
        assert result and search.cancelled and not moves
        assert chess.Board(snapshots[-1]).is_checkmate()
    asyncio.run(run())


def test_challenge_reservation_blocks_second_friend_and_unknown(tmp_path):
    from orion_core.grandmaster.bridge import GrandmasterBridge
    async def run():
        accepted, declined = [], []
        class Client:
            async def accept_challenge(self, challenge_id):
                accepted.append(challenge_id)
            async def decline_challenge(self, challenge_id, reason):
                declined.append((challenge_id, reason))
        bridge = GrandmasterBridge(SimpleNamespace(log=Signal(), dashboard_event=Signal()), None,
            settings_path=tmp_path / "settings.json", grandmaster=None)
        bridge.client = Client()
        await bridge.authorise("friend", "Known friend")
        def challenge(game_id, handle):
            return {"id": game_id, "challenger": {"id": handle}, "variant": {"key": "standard"},
                    "timeControl": {"type": "clock", "limit": 300}}
        await bridge._on_challenge(challenge("unknown1", "stranger"))
        await bridge._on_challenge(challenge("friend01", "friend"))
        await bridge._on_challenge(challenge("friend02", "friend"))
        assert accepted == ["friend01"]
        assert len(declined) == 2
        assert not bridge.settings.accept_anyone
    asyncio.run(run())


def test_outgoing_challenge_failure_preserves_public_result_contract(tmp_path):
    from orion_core.grandmaster.bridge import GrandmasterBridge
    async def run():
        class Client:
            async def challenge(self, *_args):
                raise OSError("Unavailable")
        bridge = GrandmasterBridge(SimpleNamespace(log=Signal(), dashboard_event=Signal()), None,
            settings_path=tmp_path / "settings.json", grandmaster=None)
        bridge.client, bridge.running = Client(), True
        await bridge.authorise("friend", "Known friend")
        ok, message = await bridge.challenge("friend")
        assert not ok and "could not confirm" in message
        assert not bridge._pending
    asyncio.run(run())


def _gpu_test_child(stop, pipe):
    while not stop.wait(0.01):
        pass
    pipe.send({"status": "stopped"})
    pipe.close()


def test_gpu_preemption_reaps_process_before_lease_release(tmp_path, monkeypatch):
    from orion_core import admission
    from orion_core.grandmaster.supervisor import Grandmaster
    policy = admission.Admission(quiet_s=0)
    monkeypatch.setattr(admission, "ADMISSION", policy)
    async def run():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")
        order = []
        async def training():
            async with policy.lease("gpu", "training") as ticket:
                result = await swarm._run_child(_gpu_test_child, lambda stop, pipe: (stop, pipe), gpu_ticket=ticket)
                assert result["status"] == "deferred"
                assert swarm._child is None
                order.append("reaped")
        task = asyncio.create_task(training())
        while swarm._child is None:
            await asyncio.sleep(0.01)
        async with policy.lease("gpu", "interactive"):
            order.append("llm")
        await task
        await swarm.close()
        assert order == ["reaped", "llm"]
    asyncio.run(run())


def test_repeated_cancellation_cannot_release_gpu_before_reaping(tmp_path, monkeypatch):
    from orion_core import admission
    from orion_core.grandmaster.supervisor import Grandmaster
    policy = admission.Admission(quiet_s=0)
    monkeypatch.setattr(admission, "ADMISSION", policy)
    async def run():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")
        async def training():
            async with policy.lease("gpu", "training"):
                await swarm._run_child(_gpu_test_child, lambda stop, pipe: (stop, pipe))
        task = asyncio.create_task(training())
        while swarm._child is None or swarm._child.pid is None:
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert swarm._child is None
        async with policy.lease("gpu", "interactive"):
            pass
        await swarm.close()
    asyncio.run(run())


def test_cpu_training_worker_uses_terminal_dataset(tmp_path):
    import multiprocessing as mp
    pytest.importorskip("torch")
    from orion_core.grandmaster.dataset import Dataset
    from orion_core.grandmaster.nnue import train_pass
    dataset = Dataset(tmp_path / "positions.sqlite3")
    try:
        rng = random.Random(91)
        for _ in range(64):
            board, rows = chess.Board(), []
            for _ply in range(24):
                if board.is_game_over():
                    break
                board.push(rng.choice(list(board.legal_moves)))
                rows.append((board.fen(), 100))
            dataset.append(rows, "1-0", "test")
    finally:
        dataset.close()
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    stop = ctx.Event()
    process = ctx.Process(target=train_pass,
                          args=(str(tmp_path), None, stop, sender, 4, 32, "cpu"))
    try:
        process.start()
        sender.close()
        process.join(30)
        assert not process.is_alive()
        assert receiver.poll()
        result = receiver.recv()
        assert result["status"] in ("candidate", "rejected"), result
        assert result["before"] > 0 and result["after"] > 0
        if result["status"] == "candidate":
            from orion_core.grandmaster.weights import Inference
            assert isinstance(Inference.load(result["path"]).evaluate(chess.Board()), int)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        receiver.close()
        sender.close()
        process.close()


def test_berserk_chat_is_off_loop_with_timeout_and_no_redirect(tmp_path, monkeypatch):
    pytest.importorskip("berserk", reason="optional Grandmaster chat dependency; requirements/chess-grandmaster.txt")
    import requests

    from orion_core.grandmaster.transport import SecureClient
    main_thread = threading.get_ident()
    calls = []
    def request(session, method, url, **kwargs):
        calls.append((threading.get_ident(), kwargs))
        assert url == "https://lichess.org/api/bot/game/abcdefgh/chat"
        time.sleep(0.05)
        response = requests.Response()
        response.status_code = 200
        response._content = b'{}'
        return response
    monkeypatch.setattr(requests.Session, "request", request)
    async def run():
        client = SecureClient("test-only-token")
        try:
            await client.chat("abcdefgh", "Well played")
        finally:
            await client.close()
    asyncio.run(run())
    assert calls[0][0] != main_thread
    assert calls[0][1]["timeout"] == (3, 5)
    assert calls[0][1]["allow_redirects"] is False


def test_missing_optional_chat_dependency_is_actionable_and_closes_cleanly(monkeypatch):
    from orion_core.chess_lichess import LichessError
    from orion_core.grandmaster.transport import SecureClient
    monkeypatch.setitem(sys.modules, "berserk", None)

    async def run():
        client = SecureClient("test-only-token")
        try:
            with pytest.raises(LichessError, match="requirements/chess-grandmaster.txt"):
                await client.chat("abcdefgh", "Well played")
        finally:
            await client.close()

    asyncio.run(run())


def test_zip_member_names_cannot_escape_target(tmp_path):
    from orion_core.grandmaster.book import unpack_pgn
    source = tmp_path / "masters.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../../escaped.pgn", "1. e4 e5 *")
    target = tmp_path / "games.pgn"
    unpack_pgn(source, target)
    assert target.read_text() == "1. e4 e5 *"
    assert not (tmp_path.parent / "escaped.pgn").exists()


def test_spawn_entry_does_not_import_ui_or_relaunch():
    code = ("import runpy,sys; runpy.run_path('orion.py',run_name='__mp_main__'); "
            "assert 'orion_core.app' not in sys.modules; assert 'PyQt6' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr


def test_corrupt_active_model_preserves_classical_engine(tmp_path):
    from orion_core.grandmaster.search import refresh_weights
    (tmp_path / "active.json").write_text('{"sha256":"../escape"}', encoding="utf-8")
    brain = make_brain(str(tmp_path / "brain.json"))
    assert refresh_weights(brain, tmp_path, "classical") == "classical"
    assert brain.nnue is None


def test_quiet_game_stream_wakes_on_move_failure(tmp_path):
    from orion_core.grandmaster.bridge import GrandmasterBridge
    async def run():
        bridge = GrandmasterBridge(SimpleNamespace(log=Signal(), dashboard_event=Signal()), None,
            settings_path=tmp_path / "settings.json", grandmaster=None)
        async def events():
            yield {"type": "gameFull"}
            await asyncio.sleep(100)
        async def fail():
            await asyncio.sleep(0.02)
            raise OSError("Ambiguous POST")
        stream = bridge._watched(events(), "abcdefgh")
        try:
            await anext(stream)
            bridge._responses["abcdefgh"] = asyncio.create_task(fail())
            with pytest.raises(OSError, match="Ambiguous"):
                await asyncio.wait_for(anext(stream), 1)
        finally:
            await stream.aclose()
    asyncio.run(run())


def test_swarm_lifecycle_serialises_start_stop_and_restarts(tmp_path, monkeypatch):
    from orion_core import admission
    from orion_core.grandmaster.supervisor import Grandmaster
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(quiet_s=0))
    async def run():
        swarm = Grandmaster(tmp_path, tmp_path / "brain.json")
        try:
            await swarm.start(pgn_sources=())
            assert swarm._selfplay.is_alive()
            await asyncio.gather(swarm.stop_training(), swarm.stop_training())
            assert swarm._selfplay is None and not swarm._started
            await swarm.start(pgn_sources=())
            assert swarm._selfplay.is_alive()
        finally:
            await swarm.close()
        assert not swarm._tasks and not swarm._owners
        assert swarm._selfplay is None
    asyncio.run(run())


def test_chess_service_async_shutdown_drains_search_workers(tmp_path):
    from orion_core.chess_engine import ChessService
    from orion_core.grandmaster.supervisor import Grandmaster
    async def run():
        service = ChessService(SimpleNamespace(log=Signal(), dashboard_event=Signal()))
        service._grandmaster = Grandmaster(tmp_path, tmp_path / "brain.json")
        move, _cp = await service._lichess_think(chess.Board(), 0.05)
        assert move in chess.Board().legal_moves
        await service.aclose()
        assert service.grandmaster.search._pool is None
    asyncio.run(run())
