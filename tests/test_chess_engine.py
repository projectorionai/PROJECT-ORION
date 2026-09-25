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
