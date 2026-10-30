"""
Tests for ChessService — hermetic, no real Stockfish binary or network
required (the UCI engine is mocked the same way test_security_recon.py
mocks the nmap module: a fake object standing in for the real dependency).
"""

from __future__ import annotations

import asyncio
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import pytest

from orion_core import chess_engine as chess_engine_module
from orion_core.chess_engine import ChessService


@pytest.fixture(autouse=True)
def _close_process_backed_chess_services(monkeypatch):
    """The service now owns persistent spawned pools, just like a native engine."""
    services = []
    original = ChessService.__init__

    def initialise(service, *args, **kwargs):
        original(service, *args, **kwargs)
        services.append(service)

    monkeypatch.setattr(ChessService, "__init__", initialise)
    yield
    for service in services:
        asyncio.run(service.aclose())


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _FakeScore:
    def __init__(self, cp: int) -> None:
        self._cp = cp

    def score(self, mate_score: int = 100_000) -> int:
        return self._cp


class _FakePovScore:
    """A score fixed from White's absolute perspective, like the real one."""

    def __init__(self, white_cp: int) -> None:
        self._white_cp = white_cp

    def white(self) -> _FakeScore:
        return _FakeScore(self._white_cp)

    def black(self) -> _FakeScore:
        return _FakeScore(-self._white_cp)


class _FakePlayResult:
    def __init__(self, move) -> None:
        self.move = move


class _FakeEngine:
    """Always plays the first legal move; analyse() reports a fixed
    White-perspective score unless a per-call override is supplied."""

    def __init__(self, white_cp: int = 20) -> None:
        self.configured: list[dict] = []
        self.quit_called = False
        self.white_cp = white_cp

    def configure(self, options):
        self.configured.append(options)

    def play(self, board, limit, **kw):
        move = next(iter(board.legal_moves), None)
        return _FakePlayResult(move)

    def analyse(self, board, limit, **kw):
        move = next(iter(board.legal_moves), None)
        return {"pv": [move] if move is not None else [], "score": _FakePovScore(self.white_cp)}

    def quit(self):
        self.quit_called = True


def _service(monkeypatch, tmp_path) -> ChessService:
    engine_path = tmp_path / "stockfish.exe"
    engine_path.write_bytes(b"fake")   # so _ensure_engine sees it as installed
    monkeypatch.setattr(chess_engine_module, "ENGINE_PATH", engine_path)
    service = ChessService(_StubBus())

    fake_engine = _FakeEngine()

    async def _fake_ensure_engine():
        if service._engine is None:
            service._engine = fake_engine
            await service._apply_elo(service.elo)
        return service._engine

    service._ensure_engine = _fake_ensure_engine
    return service


# ── ELO → UCI option mapping ────────────────────────────────────────────────

def test_elo_below_1320_uses_skill_level():
    options = ChessService._elo_options(800)
    assert options["UCI_LimitStrength"] is False
    assert options["Skill Level"] == 0


def test_elo_at_or_above_1320_uses_uci_elo():
    options = ChessService._elo_options(1500)
    assert options["UCI_LimitStrength"] is True
    assert options["UCI_Elo"] == 1500


def test_elo_just_below_1320_is_near_top_skill_level():
    options = ChessService._elo_options(1319)
    assert options["Skill Level"] == 20


# ── new_game / resign / set_elo ─────────────────────────────────────────────

def test_new_game_as_white_does_not_trigger_an_engine_move(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    result = asyncio.run(service.new_game(elo=1600, player_colour="white"))
    assert result.ok
    assert service.board.fullmove_number == 1
    assert service.board.turn == chess.WHITE
    assert service.elo == 1600


def test_new_game_as_black_gets_an_immediate_reply(monkeypatch, tmp_path):
    """Playing black means the opponent moves first, without being asked.

    That opponent is now ORION rather than Stockfish — the user asked for his
    own engine and every ordinary "play chess with me" used to route to the UCI
    process instead. The behaviour under test (a reply arrives immediately) is
    unchanged; only who produces it.
    """
    service = _service(monkeypatch, tmp_path)
    result = asyncio.run(service.new_game(player_colour="black"))
    assert result.ok
    assert "Orion plays" in result.text or "orion plays" in result.text.lower()
    assert len(service.history) == 1


def test_new_game_as_black_against_stockfish_when_asked(monkeypatch, tmp_path):
    """Stockfish is still reachable — it is opt-in, not removed."""
    service = _service(monkeypatch, tmp_path)
    service.set_opponent("stockfish")
    result = asyncio.run(service.new_game(player_colour="black"))
    assert result.ok
    assert service.sides["white"] == "engine"
    assert len(service.history) == 1


def test_resign_without_a_game_is_refused(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    result = asyncio.run(service.resign())
    assert not result.ok


def test_resign_ends_an_active_game(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    result = asyncio.run(service.resign())
    assert result.ok
    assert service.board_state()["active"] is False


def test_set_elo_clamps_to_the_supported_range(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.set_elo(50))
    assert service.elo == ChessService.MIN_ELO
    asyncio.run(service.set_elo(99999))
    assert service.elo == ChessService.MAX_ELO


# ── make_move ────────────────────────────────────────────────────────────────

def test_make_move_without_a_game_is_refused(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    result = asyncio.run(service.make_move("e4"))
    assert not result.ok


def test_make_move_rejects_an_illegal_move(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    result = asyncio.run(service.make_move("e2e5"))   # pawns don't jump two ranks from e2 to e5
    assert not result.ok
    assert "legal" in result.text.lower()


def test_make_move_accepts_san_and_replies(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    result = asyncio.run(service.make_move("e4"))
    assert result.ok
    assert "You played e4" in result.text
    assert len(service.history) == 2   # the user's move + the engine's reply


def test_make_move_accepts_uci_notation(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    result = asyncio.run(service.make_move("e2e4"))
    assert result.ok


def test_checkmate_ends_the_game_without_an_engine_reply(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    # Fool's mate: fastest checkmate in chess, delivered by the user as Black
    # (a move is refused when it is not the user's turn, so the user holds
    # Black here).
    service.sides = {"white": "engine", "black": "user"}
    service.board = chess.Board()
    for san in ("f3", "e5", "g4"):
        service.board.push_san(san)
    result = asyncio.run(service.make_move("Qh4#"))
    assert result.ok
    assert "Checkmate" in result.text
    assert service._active is False


# ── board_state ──────────────────────────────────────────────────────────────

def test_board_state_shape(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game(elo=1200, player_colour="white"))
    state = service.board_state()
    assert state["available"] is True
    assert state["turn"] == "white"
    assert state["elo"] == 1200
    assert state["player_colour"] == "white"
    assert state["active"] is True


def test_board_state_unavailable_when_chess_package_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(chess_engine_module, "CHESS_AVAILABLE", False)
    service = ChessService(_StubBus())
    assert service.board_state() == {"available": False}


# ── engine unavailable degrades cleanly ─────────────────────────────────────

def test_engine_unavailable_falls_back_to_heuristic_play(monkeypatch, tmp_path):
    """When Stockfish can't be found OR auto-installed, chess still works —
    deliberately, not a regression: ORION falls back to the built-in
    heuristic evaluator rather than refusing to play (see the module
    docstring's "detect, don't assume" note). The engine backend name
    reflects the degradation so it's visible in status(), not silent."""
    service = ChessService(_StubBus())
    # Stockfish only starts when it is asked for: as the analyst here. ORION
    # grading his own games never goes near it (see ChessService.analyst).
    service.analyst = "stockfish"
    monkeypatch.setattr(chess_engine_module, "ENGINE_PATH", tmp_path / "missing.exe")

    async def _fail_install(bus=None):
        return None

    monkeypatch.setattr(chess_engine_module, "install_stockfish", _fail_install)
    result = asyncio.run(service.new_game(player_colour="black"))
    assert result.ok
    assert service.backend == "heuristic"


# ── analysis ─────────────────────────────────────────────────────────────────

def test_analyse_move_reports_a_verdict(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    asyncio.run(service.make_move("e4"))
    result = asyncio.run(service.analyse_move())
    assert result.ok
    assert "1. e4" in result.text
    assert "centipawn" not in result.text.lower()


def test_analyse_move_with_no_moves_played_is_refused(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    result = asyncio.run(service.analyse_move())
    assert not result.ok


def test_analyse_move_out_of_range_is_refused(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    asyncio.run(service.make_move("e4"))
    result = asyncio.run(service.analyse_move(move_number=99))
    assert not result.ok


def test_analyse_game_reports_a_summary(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    asyncio.run(service.new_game())
    asyncio.run(service.make_move("e4"))
    asyncio.run(service.make_move("Nf3"))
    result = asyncio.run(service.analyse_game())
    assert result.ok
    # In words, not numbers: "it talks about centipawn loss" was the complaint.
    assert "summary" in result.text.lower()
    assert "centipawn" not in result.text.lower()


def test_grades_follow_winning_chances_not_raw_numbers():
    """One scale for every grade (the old verdict and label disagreed), and a
    pawn given away in a level position matters more than in a won one."""
    from orion_core.chess_theory import grade
    assert grade(30, 30, is_best=True) == "best"
    assert grade(30, 20) == "excellent"
    assert grade(0, -60) == "inaccuracy"
    assert grade(0, -300) == "blunder"
    assert grade(900, 800) in ("excellent", "good")    # already winning
    assert grade(0, -100) != grade(900, 800)


# ── eval bar ────────────────────────────────────────────────────────────────

def test_win_probability_is_neutral_with_no_score():
    assert ChessService._win_probability(None, None) == 0.5


def test_win_probability_favours_white_for_positive_cp():
    assert ChessService._win_probability(400, None) > 0.5


def test_win_probability_favours_black_for_negative_cp():
    assert ChessService._win_probability(-400, None) < 0.5


def test_win_probability_saturates_near_the_edges_for_mate():
    assert ChessService._win_probability(None, 3) == 0.98
    assert ChessService._win_probability(None, -2) == 0.02


def test_evaluate_current_returns_the_engines_white_perspective_score(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    service.analyst = "stockfish"      # the engine's score, so the engine grades
    asyncio.run(service.new_game())
    info = asyncio.run(service.evaluate_current())
    assert info["cp"] == 20
    assert info["mate"] is None
    assert 0.0 < info["white_fraction"] < 1.0


def test_evaluation_does_not_cache_an_old_score_for_a_new_position(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path)
    initial_key = service._key(service.board)

    async def flow():
        started = asyncio.Event()
        release = asyncio.Event()

        async def evaluate(board):
            assert service._key(board) == initial_key
            started.set()
            await release.wait()
            return 42, None

        service._evaluate = evaluate
        pending = asyncio.create_task(service.evaluate_current())
        await started.wait()
        service.board.push_uci("e2e4")
        new_key = service._key(service.board)
        release.set()
        result = await pending
        assert result["cp"] == 42
        assert service._evals[initial_key] == (42, None)
        assert new_key not in service._evals
        assert getattr(service, "_last_shown_eval", None) is None

    asyncio.run(flow())


def test_evaluate_current_is_neutral_when_chess_package_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(chess_engine_module, "CHESS_AVAILABLE", False)
    service = ChessService(_StubBus())
    info = asyncio.run(service.evaluate_current())
    assert info == {"cp": None, "mate": None, "white_fraction": 0.5}


# ── Stockfish binary extraction (real zip, no network) ──────────────────────

def test_extract_binary_finds_the_exe_inside_a_zip(monkeypatch, tmp_path):
    engine_dir = tmp_path / "engines"
    engine_path = engine_dir / "stockfish.exe"
    monkeypatch.setattr(chess_engine_module, "ENGINE_DIR", engine_dir)
    monkeypatch.setattr(chess_engine_module, "ENGINE_PATH", engine_path)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("stockfish/stockfish-windows-x86-64.exe", b"pretend-binary-contents")
        archive.writestr("stockfish/README.txt", b"not the binary")
    zip_bytes = buffer.getvalue()

    result_path = chess_engine_module._extract_binary(zip_bytes)
    assert result_path == engine_path
    assert engine_path.read_bytes() == b"pretend-binary-contents"


def test_extract_binary_raises_when_no_exe_present(tmp_path, monkeypatch):
    monkeypatch.setattr(chess_engine_module, "ENGINE_DIR", tmp_path)
    monkeypatch.setattr(chess_engine_module, "ENGINE_PATH", tmp_path / "stockfish.exe")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", b"nothing useful")
    with pytest.raises(RuntimeError):
        chess_engine_module._extract_binary(buffer.getvalue())


def test_install_stockfish_is_a_noop_when_already_present(tmp_path, monkeypatch):
    engine_path = tmp_path / "stockfish.exe"
    engine_path.write_bytes(b"already here")
    monkeypatch.setattr(chess_engine_module, "ENGINE_PATH", engine_path)
    result = asyncio.run(chess_engine_module.install_stockfish(_StubBus()))
    assert result == engine_path


def test_install_stockfish_reports_non_windows_platforms_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(chess_engine_module, "ENGINE_PATH", tmp_path / "stockfish.exe")
    monkeypatch.setattr(chess_engine_module.sys, "platform", "linux", raising=False)
    result = asyncio.run(chess_engine_module.install_stockfish(_StubBus()))
    assert result is None


# ── Lichess: friends play ORION remotely (chess_lichess.py) ──────────────────
#
# Everything below runs against a scripted fake of the Lichess Bot API — no
# network, no token. The bridge is outbound-only by construction: it only
# ever calls the client's methods, and the real client only makes requests.

from orion_core import chess_lichess as lichess_module  # noqa: E402
from orion_core.chess_lichess import (  # noqa: E402
    LichessAuthError, LichessBridge, LichessSettings, challenge_decision,
    iter_ndjson, read_token, think_time,
)

_SECRET = "lip_TEST_TOKEN_must_never_be_logged"


class _RecordingSignal:
    def __init__(self, sink):
        self._sink = sink

    def emit(self, *args, **kwargs):
        self._sink.append(args)

    def connect(self, *a, **k):
        pass


class _RecordingBus:
    def __init__(self):
        self.events: list = []

    def __getattr__(self, name):
        return _RecordingSignal(self.events)


def _challenge(cid, who, limit=300, increment=3, variant="standard", rated=False, kind="clock"):
    control = {"type": kind, "limit": limit, "increment": increment,
               "show": f"{limit // 60}+{increment}"} if kind == "clock" else {"type": kind}
    return {"type": "challenge", "challenge": {
        "id": cid, "challenger": {"id": who.lower(), "name": who},
        "variant": {"key": variant}, "rated": rated, "timeControl": control,
        "direction": "in"}}


class _FakeLichess:
    """Scripted Lichess: an event stream, then one game in which the friend
    (Black) replies 1...e5 and then resigns."""

    def __init__(self, events, account=None, fail_first_stream=False):
        self.calls: list = []
        self._events = events
        self._account = account or {"id": "orionbot", "username": "OrionBot", "title": "BOT"}
        self._fail_first = fail_first_stream
        self.streams_opened = 0
        self.closed = False

    async def account(self):
        return self._account

    async def stream_events(self, on_open=None):
        self.streams_opened += 1
        if self._fail_first and self.streams_opened == 1:
            raise ConnectionError("dropped")
        if on_open:
            on_open()
        for event in self._events:
            yield event
        await asyncio.sleep(3600)            # a real stream stays open

    async def stream_game(self, game_id):
        yield {"type": "gameFull", "id": game_id, "rated": False,
               "clock": {"initial": 300000, "increment": 3000},
               "white": {"id": "orionbot", "name": "OrionBot"},
               "black": {"id": "sam", "name": "Sam"}, "initialFen": "startpos",
               "state": {"type": "gameState", "moves": "", "wtime": 300000,
                         "btime": 300000, "winc": 3000, "binc": 3000, "status": "started"}}
        while not any(c[0] == "move" for c in self.calls):
            await asyncio.sleep(0.01)
        first = next(c[2] for c in self.calls if c[0] == "move")
        yield {"type": "chatLine", "username": "Sam", "text": "gl", "room": "player"}
        yield {"type": "gameState", "moves": f"{first} e7e5", "wtime": 299000,
               "btime": 298000, "winc": 3000, "binc": 3000, "status": "started"}
        while sum(1 for c in self.calls if c[0] == "move") < 2:
            await asyncio.sleep(0.01)
        second = [c[2] for c in self.calls if c[0] == "move"][1]
        yield {"type": "gameState", "moves": f"{first} e7e5 {second}", "wtime": 297000,
               "btime": 290000, "winc": 3000, "binc": 3000, "status": "resign",
               "winner": "white"}

    async def accept_challenge(self, cid):
        self.calls.append(("accept", cid))

    async def decline_challenge(self, cid, reason="generic"):
        self.calls.append(("decline", cid, reason))

    async def make_move(self, game_id, uci):
        self.calls.append(("move", game_id, uci))

    async def chat(self, game_id, text, room="player"):
        self.calls.append(("chat", game_id, text))

    async def accept_draw(self, game_id):
        self.calls.append(("draw", game_id))

    async def abort(self, game_id):
        self.calls.append(("abort", game_id))

    async def resign(self, game_id):
        self.calls.append(("resign", game_id))

    async def challenge(self, username, limit, increment, rated, colour):
        self.calls.append(("challenge", username, limit, increment, rated, colour))
        return {"id": "c9"}

    async def close(self):
        self.closed = True


async def _first_legal(board, seconds):
    return next(iter(board.legal_moves)), 0


def _bridge(tmp_path, fake, think=_first_legal, token=_SECRET, finished=None):
    settings = LichessSettings(allow=["sam"])
    settings.save(tmp_path / "lichess.json")
    bridge = LichessBridge(_RecordingBus(), think, settings_path=tmp_path / "lichess.json",
                           client_factory=lambda t: fake, token_reader=lambda: token,
                           on_finish=finished)
    bridge.BACKOFF_START = 0.01
    bridge._backoff = 0.01
    return bridge


def test_lichess_only_accepts_friends_at_a_clock_orion_can_keep_up_with():
    s = LichessSettings(allow=["sam"])
    ok = lambda ch: challenge_decision(ch["challenge"], s, 0)  # noqa: E731
    assert ok(_challenge("a", "Sam")) == (True, "")
    assert ok(_challenge("b", "Stranger")) == (False, "generic")
    assert ok(_challenge("c", "Sam", limit=60)) == (False, "tooFast")
    assert ok(_challenge("d", "Sam", kind="correspondence")) == (False, "timeControl")
    assert ok(_challenge("e", "Sam", variant="chess960")) == (False, "standard")
    assert challenge_decision(_challenge("f", "Sam")["challenge"], s, 1) == (False, "later")
    s.accept_anyone = True
    assert ok(_challenge("g", "Stranger")) == (True, "")


def test_lichess_plays_a_whole_game_against_a_friend(tmp_path):
    finished = []
    fake = _FakeLichess([_challenge("c1", "Sam"), _challenge("c2", "Stranger"),
                         {"type": "gameStart", "game": {"gameId": "g1", "fullId": "g1abcd"}}])
    bridge = _bridge(tmp_path, fake, finished=lambda moves, result, game:
                     finished.append((moves, result, game.opponent)))

    async def run():
        ok, text = await bridge.start()
        assert ok, text
        for _ in range(300):
            if finished:
                break
            await asyncio.sleep(0.01)
        await bridge.stop()
    asyncio.run(run())
    assert ("accept", "c1") in fake.calls
    assert ("decline", "c2", "generic") in fake.calls
    moves = [c for c in fake.calls if c[0] == "move"]
    assert len(moves) == 2                          # answered each position exactly once
    board = chess.Board()
    for m in (moves[0][2], "e7e5", moves[1][2]):
        board.push_uci(m)                           # every move posted was legal
    assert any(c[0] == "chat" for c in fake.calls)  # greeted, once
    assert finished and finished[0][1] == "1-0" and finished[0][2] == "Sam"
    assert "beat Sam" in bridge.status_text()
    assert fake.closed


def test_lichess_never_logs_the_token(tmp_path):
    fake = _FakeLichess([_challenge("c1", "Sam")])
    bridge = _bridge(tmp_path, fake)

    async def run():
        await bridge.start()
        await asyncio.sleep(0.05)
        await bridge.stop()
    asyncio.run(run())
    everything = repr(bridge.bus.events) + bridge.status_text()
    assert _SECRET not in everything


def test_lichess_reconnects_after_a_dropped_stream(tmp_path):
    fake = _FakeLichess([_challenge("c1", "Sam")], fail_first_stream=True)
    bridge = _bridge(tmp_path, fake)

    async def run():
        await bridge.start()
        for _ in range(200):
            if ("accept", "c1") in fake.calls:
                break
            await asyncio.sleep(0.01)
        await bridge.stop()
    asyncio.run(run())
    assert fake.streams_opened >= 2
    assert ("accept", "c1") in fake.calls


def test_lichess_refuses_without_a_token_or_a_bot_account(tmp_path):
    no_token = _bridge(tmp_path, _FakeLichess([]), token="")
    ok, text = asyncio.run(no_token.start())
    assert not ok and "token" in text and not no_token.running

    human = _bridge(tmp_path, _FakeLichess([], account={"username": "Human", "title": None}))
    ok, text = asyncio.run(human.start())
    assert not ok and "BOT" in text


def test_lichess_stops_on_a_rejected_token(tmp_path):
    class _Rejected(_FakeLichess):
        async def account(self):
            raise LichessAuthError(401, "No such token")
    bridge = _bridge(tmp_path, _Rejected([]))
    ok, text = asyncio.run(bridge.start())
    assert not ok and "rejected" in text


def test_lichess_challenge_validates_the_name_and_sends_whole_minutes(tmp_path):
    fake = _FakeLichess([])
    bridge = _bridge(tmp_path, fake)

    async def run():
        bad = await bridge.challenge("sam/../../api/account", 5, 3)
        good = await bridge.challenge("Alex_99", 10, 5, rated=True, colour="white")
        await bridge.stop()
        return bad, good
    bad, good = asyncio.run(run())
    assert not bad[0]
    assert good[0]
    assert ("challenge", "Alex_99", 600, 5, True, "white") in fake.calls
    assert "alex_99" in LichessSettings.load(tmp_path / "lichess.json").allow   # can rematch


def test_lichess_allow_list_persists(tmp_path):
    bridge = _bridge(tmp_path, _FakeLichess([]))
    assert "can challenge me" in bridge.allow("@Jordan")
    assert "valid" in bridge.allow("no spaces allowed")
    assert "jordan" in LichessSettings.load(tmp_path / "lichess.json").allow
    bridge.disallow("jordan")
    assert "jordan" not in LichessSettings.load(tmp_path / "lichess.json").allow


def test_lichess_think_time_follows_the_clock():
    state = {"wtime": 300_000, "btime": 300_000, "winc": 3000, "binc": 3000}
    assert think_time(state, True, 1.5) == 1.5                  # plenty of time: his usual
    assert think_time({"wtime": 9_000, "winc": 0}, True, 1.5) <= 0.2   # short on time
    assert think_time({"wtime": 60_000, "winc": 0}, True, 1.5) == 1.5
    assert 0.05 <= think_time({"wtime": 1_000, "winc": 0}, True, 1.5) <= 0.2


def test_lichess_ndjson_skips_keepalives_and_junk():
    async def lines():
        for raw in (b"\n", b'{"type": "gameStart", "game": {"gameId": "x"}}\n', b"  \n",
                    b"not json\n", b'{"type": "challenge"}\n'):
            yield raw

    async def collect():
        return [obj async for obj in iter_ndjson(lines())]
    assert [o["type"] for o in asyncio.run(collect())] == ["gameStart", "challenge"]


def test_lichess_token_comes_from_the_key_store(tmp_path, monkeypatch):
    import json as _json
    store = tmp_path / "api_keys.json"
    store.write_text(_json.dumps({"integrations": {"lichess": {"bot_token": "lip_abc"}}}))
    import orion_core.constants as constants
    monkeypatch.setattr(constants, "API_CONFIG_PATH", store)
    monkeypatch.delenv(lichess_module.TOKEN_ENV, raising=False)
    assert read_token() == "lip_abc"
    monkeypatch.setenv(lichess_module.TOKEN_ENV, "lip_env")
    assert read_token() == "lip_env"


def test_the_service_routes_lichess_through_orions_own_engine(tmp_path, monkeypatch):
    """ChessService builds the bridge with his own brain as the player."""
    monkeypatch.setattr(lichess_module, "SETTINGS_PATH", tmp_path / "lichess.json")
    service = ChessService(_StubBus())
    bridge = service.lichess
    assert bridge.think == service._lichess_think
    move, _score = asyncio.run(service._lichess_think(chess.Board(), 0.1))
    assert move in chess.Board().legal_moves
    assert "can challenge me" in service.lichess_allow("sam").text
    service.shutdown()


def test_lichess_game_stream_reconnects_mid_game(tmp_path):
    """A dropped GAME stream is reopened (Lichess re-sends the full game);
    abandoning it would lose on time."""
    class _Flaky(_FakeLichess):
        game_streams = 0

        async def stream_game(self, game_id):
            type(self).game_streams += 1
            if type(self).game_streams == 1:
                raise ConnectionError("reset by peer")
            async for event in super().stream_game(game_id):
                yield event

    finished = []
    fake = _Flaky([{"type": "gameStart", "game": {"gameId": "g1"}}])
    bridge = _bridge(tmp_path, fake, finished=lambda m, r, g: finished.append(r))

    async def run():
        await bridge.start()
        for _ in range(400):
            if finished:
                break
            await asyncio.sleep(0.01)
        await bridge.stop()
    asyncio.run(run())
    assert _Flaky.game_streams >= 2
    assert finished == ["1-0"]


def test_lichess_retries_a_move_that_failed_to_send(tmp_path):
    from orion_core.chess_lichess import LichessError, LichessGame

    class _Glitchy(_FakeLichess):
        failures = 0

        async def make_move(self, game_id, uci):
            if type(self).failures < 1:
                type(self).failures += 1
                raise LichessError(502, "Bad gateway")
            await super().make_move(game_id, uci)

    fake = _Glitchy([])
    bridge = _bridge(tmp_path, fake)
    bridge.client = fake
    sent = asyncio.run(bridge._send_move(LichessGame("g1"), chess.Move.from_uci("e2e4")))
    assert sent and ("move", "g1", "e2e4") in fake.calls

    class _Illegal(_FakeLichess):
        async def make_move(self, game_id, uci):
            self.calls.append(("tried", uci))
            raise LichessError(400, "Not your turn, or game already over")

    fake = _Illegal([])
    bridge.client = fake
    assert not asyncio.run(bridge._send_move(LichessGame("g1"), chess.Move.from_uci("e2e4")))
    assert fake.calls == [("tried", "e2e4")]               # 400 is not retried
