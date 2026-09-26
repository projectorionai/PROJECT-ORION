"""
ORION's own chess brain.

"When ORION plays chess, it must NOT use the stockfish engine as I want him to
be good at chess by himself and he must be able to dynamically learn from
games."

So: does he actually play, is he actually any good, and does he actually learn?
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

chess = pytest.importorskip("chess")

from orion_core.chess_brain import (  # noqa: E402
    BASE_TABLES, MATE_SCORE, BookEntry, ChessBrain,
)


@pytest.fixture
def brain(tmp_path):
    return ChessBrain(path=tmp_path / "brain.json", seed=11)


# ── it plays legal chess ─────────────────────────────────────────────────────

def test_it_returns_a_legal_move(brain):
    board = chess.Board()
    move = brain.choose_move(board, seconds=0.3)
    assert move in board.legal_moves


def test_it_leaves_the_board_exactly_as_it_found_it(brain):
    """The search pushes and pops constantly — a leak corrupts the game."""
    board = chess.Board()
    before = board.fen()
    brain.choose_move(board, seconds=0.3)
    assert board.fen() == before
    assert len(board.move_stack) == 0


def test_an_interrupted_search_still_restores_the_board(brain):
    """A timeout unwinds mid-recursion; the pops must still happen.

    This is a real defect that was caught by playing: on timeout the board was
    left several plies deep and the returned move was rejected as illegal.
    """
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push(chess.Move.from_uci(uci))
    depth_before = len(board.move_stack)
    move = brain.choose_move(board, seconds=0.02, max_depth=6)  # forces a timeout
    assert len(board.move_stack) == depth_before
    assert move in board.legal_moves


def test_it_plays_from_any_position(brain):
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1")
    assert brain.choose_move(board, seconds=0.3) in board.legal_moves


def test_no_legal_moves_returns_none(brain):
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert brain.choose_move(mated, seconds=0.2) is None


# ── it is actually any good ──────────────────────────────────────────────────

def test_it_takes_a_free_queen(brain):
    """The most basic test of an engine: see material and take it."""
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    move = brain.choose_move(board, seconds=0.6)
    assert move.uci() == "e4d5", f"played {move} instead of winning the queen"


def test_it_finds_mate_in_one(brain):
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K2R w KQ - 0 1")
    move = brain.choose_move(board, seconds=0.8)
    board.push(move)
    assert board.is_checkmate(), f"{move} was not mate"


def test_it_does_not_hang_a_queen(brain):
    """Quiescence exists so it does not stop the search mid-trade."""
    board = chess.Board("4k3/8/8/8/8/4n3/8/3QK3 w - - 0 1")
    move = brain.choose_move(board, seconds=0.6)
    board.push(move)
    # After its move the queen must not simply be capturable for free.
    if board.piece_at(chess.parse_square("d1")):
        assert not any(m.to_square == chess.parse_square("d1")
                       for m in board.legal_moves)


def test_it_beats_a_random_player(brain):
    """The real bar: it must be meaningfully better than chance."""
    rng = random.Random(5)
    wins = 0
    for game in range(4):
        board = chess.Board()
        brain_white = game % 2 == 0
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 60:
            ours = (board.turn == chess.WHITE) == brain_white
            move = (brain.choose_move(board, seconds=0.2, max_depth=3) if ours
                    else rng.choice(list(board.legal_moves)))
            if move is None:
                break
            board.push(move)
        result = board.result(claim_draw=True)
        if (result == "1-0") == brain_white and result != "1/2-1/2":
            wins += 1
    assert wins >= 3, f"only won {wins}/4 against random play"


def test_it_explains_its_move(brain):
    brain.choose_move(chess.Board(), seconds=0.3)
    assert brain.last_reason
    assert "depth" in brain.last_reason or "book" in brain.last_reason


# ── evaluation ───────────────────────────────────────────────────────────────

def test_material_dominates_the_evaluation(brain):
    even = brain.evaluate(chess.Board())
    a_queen_up = brain.evaluate(chess.Board(
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    assert a_queen_up > even + 700


def test_checkmate_is_scored_as_terminal(brain):
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert brain.evaluate(board) == -MATE_SCORE


def test_stalemate_is_a_draw(brain):
    assert brain.evaluate(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0


def test_a_passed_pawn_is_worth_something(brain):
    passed = brain.evaluate(chess.Board("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1"))
    blocked = brain.evaluate(chess.Board("4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1"))
    assert passed > blocked - 100


# ── it learns ────────────────────────────────────────────────────────────────

WON_GAME = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]


def test_learning_records_the_opening(brain):
    summary = brain.learn_from_game(WON_GAME, "1-0")
    assert "book position" in summary
    assert brain.book, "nothing was recorded"


def test_a_repeatedly_won_move_gets_played(brain):
    for _ in range(3):
        brain.learn_from_game(WON_GAME, "1-0")
    move = brain.choose_move(chess.Board(), seconds=0.3)
    assert move.uci() == "e2e4"
    assert "book" in brain.last_reason


def test_a_repeatedly_lost_move_is_abandoned(brain):
    """Experience must cut both ways or it is not learning."""
    for _ in range(4):
        brain.learn_from_game(WON_GAME, "0-1")     # White lost every time
    board = chess.Board()
    brain.choose_move(board, seconds=0.3)
    assert "book" not in brain.last_reason, (
        "a move that has lost four times must not be repeated from the book")


def test_one_game_is_not_enough_evidence(brain):
    brain.learn_from_game(WON_GAME, "1-0")
    brain.choose_move(chess.Board(), seconds=0.3)
    assert "book" not in brain.last_reason, "a single game is not a book entry"


def test_learning_persists_across_restarts(brain, tmp_path):
    for _ in range(3):
        brain.learn_from_game(WON_GAME, "1-0")
    fresh = ChessBrain(path=brain.path, seed=11)
    assert fresh.games_learned == 3
    assert fresh.book
    assert fresh.choose_move(chess.Board(), seconds=0.3).uci() == "e2e4"


def test_evaluation_tables_drift_from_wins(brain):
    before = list(brain.tables[1])
    for _ in range(3):
        brain.learn_from_game(WON_GAME, "1-0")
    assert brain.tables[1] != before, "the evaluation never changed"


def test_a_draw_does_not_move_the_tables(brain):
    """A draw says nothing about placement; learning from it flattens them."""
    before = list(brain.tables[2])
    brain.learn_from_game(WON_GAME, "1/2-1/2")
    assert brain.tables[2] == before


def test_the_tables_can_never_drift_far(brain):
    for _ in range(200):
        brain.learn_from_game(WON_GAME, "1-0")
    for piece, table in brain.tables.items():
        base = BASE_TABLES[piece]
        for index, value in enumerate(table):
            assert abs(value - base[index]) <= brain.MAX_DRIFT + 0.01, (
                f"piece {piece} square {index} drifted too far")


def test_an_empty_game_is_handled(brain):
    assert "Nothing to learn" in brain.learn_from_game([], "1-0")


def test_illegal_moves_in_a_game_stop_learning_safely(brain):
    assert brain.learn_from_game(["e2e4", "e2e4", "z9z9"], "1-0")


def test_describe_reports_real_numbers(brain):
    brain.learn_from_game(WON_GAME, "1-0")
    text = brain.describe()
    assert "no external engine" in text
    assert "1 game" in text


# ── book bookkeeping ─────────────────────────────────────────────────────────

def test_book_entry_scoring():
    entry = BookEntry(wins=3, draws=2, losses=1)
    assert entry.games == 6
    assert entry.score == pytest.approx((3 + 1.0) / 6)
    assert entry.confidence() == 1.0


def test_an_unplayed_entry_is_neutral():
    assert BookEntry().score == 0.5


# ── the service uses it, not Stockfish ───────────────────────────────────────

def test_the_service_plays_orions_moves_from_orions_brain():
    import inspect

    from orion_core.chess_engine import ChessService

    source = inspect.getsource(ChessService)
    assert "_orion_move" in source
    assert "chess_brain" in source or "ChessBrain" in source


def test_orions_move_path_never_calls_the_uci_engine():
    import inspect

    from orion_core.chess_engine import ChessService

    source = inspect.getsource(ChessService._orion_move)
    assert "_ensure_engine" not in source, "ORION must not use Stockfish to play"
    assert "choose_move" in source


def test_a_finished_game_is_learned_from():
    """Every move passes through one place, and a game that ends there is
    studied — played for real rather than checked by reading the source."""
    import asyncio

    svc = _service()
    asyncio.run(svc.new_game(white="user", black="user"))
    before = svc.brain.games_learned
    for san in ("f3", "e5", "g4", "Qh4#"):
        assert asyncio.run(svc.make_move(san)).ok
    assert svc.board.is_checkmate()
    assert svc.brain.games_learned == before + 1


# ── who the user actually ends up playing ────────────────────────────────────
#
# "ORION still seems to be using the engine not his own chess engine for
# himself ... when I asked him to use his own chess engine he says he doesn't
# understand and he still uses the stockfish engine."
#
# Both halves of that were real and they were separate defects:
#
#   * every ordinary "play chess with me" went through new_game(player_colour=…),
#     which hard-coded the other side to "engine" — the raw UCI process. The
#     brain was wired in correctly and simply never got asked for a move.
#   * there was no action anywhere in the tool surface for changing opponent,
#     so "use your own chess engine" had nothing to resolve to. That is what
#     "he doesn't understand" actually was: a missing verb, not a missing
#     capability.

import asyncio  # noqa: E402


class _Sig:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


_SERVICES: list = []


def _service():
    from orion_core.chess_engine import ChessService
    svc = ChessService(_Bus())
    _SERVICES.append(svc)
    return svc


@pytest.fixture(autouse=True)
def _shut_services_down():
    """A game where ORION opens (the user plays black) evaluates the position
    with the real Stockfish, and python-chess drives it from a NON-daemon
    thread. Left running, that thread kept the whole suite's interpreter
    alive after the summary line — every full run hung at exit until killed."""
    yield
    while _SERVICES:
        _SERVICES.pop().shutdown()


def test_playing_chess_with_orion_means_playing_orion():
    """The defect: the default opponent was Stockfish."""
    svc = _service()
    asyncio.run(svc.new_game(player_colour="white"))
    assert svc.sides["black"] == "orion", (
        f"the user was put against {svc.sides['black']}, not ORION")


def test_playing_black_still_faces_orion():
    svc = _service()
    asyncio.run(svc.new_game(player_colour="black"))
    assert svc.sides["white"] == "orion"


def test_use_your_own_chess_engine_is_understood():
    """The phrase the user actually said, and its obvious neighbours."""
    svc = _service()
    svc.set_opponent("stockfish")
    for phrase in ("your own engine", "yourself", "orion", "you", "own", "brain"):
        svc.set_opponent("stockfish")
        result = svc.set_opponent(phrase)
        assert result.ok, f"'{phrase}' was not understood"
        assert svc.sides["black"] == "orion", f"'{phrase}' did not switch"


def test_stockfish_is_still_available_on_request():
    svc = _service()
    assert svc.set_opponent("stockfish").ok
    assert svc.sides["black"] == "engine"


def test_switching_applies_to_the_game_in_progress():
    """Said mid-game, 'use your own engine' plainly means now."""
    svc = _service()
    asyncio.run(svc.new_game(player_colour="white"))
    svc.set_opponent("stockfish")
    assert svc.sides["black"] == "engine"
    result = svc.set_opponent("yourself")
    assert svc.sides["black"] == "orion"
    assert "in progress" in result.text


def test_an_unknown_opponent_is_refused_not_guessed():
    svc = _service()
    result = svc.set_opponent("a badger")
    assert not result.ok
    assert svc.sides["black"] == "orion", "a bad request changed the game"


def test_the_tool_schema_offers_the_switch_and_does_not_say_stockfish_first():
    """The model can only call what the schema describes."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    chess_schema = next(s for s in TOOL_DECLARATIONS if s["name"] == "chess")
    description = chess_schema["description"]
    assert "set_opponent" in description
    assert "use your own chess engine" in description.lower()
    assert "opponent" in chess_schema["parameters"]["properties"]
    assert "against Stockfish" not in description, (
        "the description still tells the model the opponent is Stockfish")


def test_the_dispatcher_routes_the_switch():
    import inspect

    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    source = inspect.getsource(ProductivityDispatchMixin.chess_tool)
    assert "set_opponent" in source


# ── Sep 2026: the engine rebuilt, measured, and made to explain itself ───────
#
# "Improve ORION's own chess engine comprehensively" and "it must talk in
# general chess terms ... never centipawns". Each test below pins a defect
# that was real, or a piece of knowledge that was measured to matter.

import re  # noqa: E402

from orion_core.chess_brain import MATE_BOUND, _see, mate_distance  # noqa: E402


def _after(*sans, fen=None):
    board = chess.Board(fen) if fen else chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def test_static_exchange_sees_the_recapture():
    """_see once XORed the capturing piece back onto its own square, so it
    'recaptured' from where it started and every capture looked like it won
    its victim for nothing — which also blunted the search's pruning."""
    elephant = _after("d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Nbd7", "cxd5", "exd5")
    assert _see(elephant, elephant.parse_san("Nxd5")) == -220      # Nf6 takes back
    trade = chess.Board("4k3/8/5n2/3p4/4P3/8/8/4K3 w - - 0 1")
    assert _see(trade, trade.parse_san("exd5")) == 0
    free = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    assert _see(free, free.parse_san("exd5")) == 100
    battery = chess.Board("3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1")
    assert _see(battery, battery.parse_san("Rxd5")) == 100          # doubled rooks


def test_the_transposition_table_is_bounded(brain):
    """It was keyed by (position, depth) and never cleared: it grew for as
    long as ORION ran."""
    brain.TT_MAX_ENTRIES = 400
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8")
    for _ in range(3):
        brain.choose_move(board, seconds=0.3, use_book=False)
    assert 0 < len(brain._tt) <= 400


def test_it_searches_deeper_than_it_used_to(brain):
    """Measured: the old engine reached depth 2 in a middlegame at ORION's
    1.5 s; it generated legal moves four times per evaluation and called
    can_claim_threefold_repetition (~470 µs) at every node."""
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8")
    brain.choose_move(board, seconds=0.5, use_book=False)
    assert brain._last_depth >= 4, brain.last_reason


def test_a_forced_mate_reports_its_distance(brain):
    """The eval bar said 'M1' for every mate: mate was detected at ±100000
    while ORION's own mate score is 30000, and the distance was hard-coded."""
    board = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")          # mate in 2 (Stockfish)
    score = brain.analyse(board, seconds=1.0)
    assert score > MATE_BOUND
    assert mate_distance(score) == 2


def test_the_service_eval_bar_shows_mate_distance():
    from orion_core.chess_engine import _mate_moves, _white_view
    assert _white_view(MATE_SCORE - 3, True)[1] == 2                # White mates in 2
    assert _white_view(MATE_SCORE - 3, False)[1] == -2              # Black to move mates
    assert _white_view(150, False) == (-150, None)
    assert _mate_moves(-(MATE_SCORE - 4)) == -2


def test_quiescence_does_not_stand_pat_in_check(brain):
    """In check there is no 'I could just not capture' — every evasion must
    be searched, and having none is mate."""
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    brain._deadline = float("inf")
    brain.nodes = 0
    assert brain._qsearch(mated, -40_000, 40_000, 0) == -MATE_SCORE


def test_the_rule_of_the_square(brain):
    """A pawn the lone king cannot catch queens: the evaluation knows it."""
    runs = chess.Board("8/8/8/1P6/8/8/6k1/K7 w - - 0 1")           # black king too far
    caught = chess.Board("8/8/3k4/1P6/8/8/8/K7 w - - 0 1")         # black king in the square
    assert brain.evaluate(runs) > brain.evaluate(caught) + 300


def test_mop_up_drives_the_bare_king_to_the_edge(brain):
    centre = chess.Board("8/8/8/4k3/8/8/8/Q3K3 w - - 0 1")
    edge = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert brain.evaluate(edge) > brain.evaluate(centre)


def test_material_that_cannot_win_is_scored_as_drawish(brain):
    """A knight up with no pawns cannot force mate."""
    assert abs(brain.evaluate(chess.Board("8/8/4k3/8/8/2N5/8/4K3 w - - 0 1"))) < 60


def test_it_converts_king_and_queen_against_king(brain):
    """Knowledge you can watch: a won ending actually gets won."""
    board = chess.Board("8/8/8/4k3/8/8/8/Q3K3 w - - 0 1")
    rng = random.Random(3)
    for _ in range(40):
        if board.is_game_over():
            break
        board.push(brain.choose_move(board, seconds=0.15, use_book=False))
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    assert board.is_checkmate(), board.fen()


def test_a_rook_on_an_open_file_is_worth_more(brain):
    open_file = chess.Board("4k3/pp4pp/8/8/8/8/PP4PP/3RK3 w - - 0 1")
    closed = chess.Board("4k3/pp4pp/8/8/8/8/PP4PP/R3K3 w - - 0 1")
    assert brain.evaluate(open_file) > brain.evaluate(closed)


# ── coaching in words ────────────────────────────────────────────────────────

def _explain(opening, played, seconds=0.6):
    from orion_core import chess_theory
    board = _after(*opening.split())
    move = board.parse_san(played)
    review = ChessBrain(path=Path(tempfile.mkdtemp()) / "b.json").review_move(
        board, move, seconds=seconds)
    return chess_theory.explain_move(board, move, review, mover="you", opponent="I")


def test_a_blunder_is_explained_with_the_refutation_and_the_better_move():
    verdict = _explain("e4 e5 Qh5 Nc6 Bc4", "Nf6")
    assert verdict["quality"] == "blunder"
    assert "Qxf7#" in verdict["text"] and "checkmate" in verdict["text"]
    assert "Better was 3... g6" in verdict["text"]
    assert "stops the threat" in verdict["text"]


def test_the_explanation_never_speaks_in_numbers():
    for opening, played in (("e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4", "Qe7"),
                            ("e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5", "Nxd5"),
                            ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3", "e5")):
        text = _explain(opening, played)["text"]
        assert "centipawn" not in text.lower()
        assert not re.search(r"[+-]\d", text), text    # no "+0.35" evaluations


def test_a_best_move_says_what_it_does():
    verdict = _explain("e4 e5 Qh5 Nc6 Bc4", "g6")
    assert verdict["quality"] == "best"
    assert "stops the threat of Qxf7#" in verdict["text"]


def test_the_game_summary_names_the_turning_points():
    from orion_core.chess_theory import summarise_game
    moments = [
        {"mover": "you", "quality": "best", "drop": 0.0, "text": "1. e4 — the best move.", "ply": 1},
        {"mover": "you", "quality": "blunder", "drop": 0.4,
         "text": "3... Nf6 — a blunder. It allows 4. Qxf7#.", "ply": 6},
    ]
    text = summarise_game(moments, "White wins (checkmate)", {})
    assert "moments that decided it" in text
    assert "Qxf7#" in text and "1 blunder" in text
    assert "centipawn" not in text.lower()


# ── the service: turns, thinking, navigation ─────────────────────────────────

def test_a_move_out_of_turn_is_refused():
    svc = _service()
    asyncio.run(svc.new_game(player_colour="white"))
    svc.board.push_san("e4")            # as if the user had moved: Black (ORION) to move
    result = asyncio.run(svc.make_move("e5"))
    assert not result.ok and "isn't your move" in result.text


def test_no_move_while_orion_is_thinking():
    svc = _service()
    asyncio.run(svc.new_game(player_colour="white"))
    svc._thinking = True
    result = asyncio.run(svc.make_move("e4"))
    assert not result.ok and "thinking" in result.text
    assert svc.board.move_stack == []


def test_orion_searches_a_snapshot_never_the_live_board():
    """The worker thread used to push and pop moves on the live board while
    the user could still play on it."""
    svc = _service()
    asyncio.run(svc.new_game(player_colour="white"))
    seen = []
    real = svc.brain.choose_move

    def spy(board, *a, **k):
        seen.append(board)
        return real(board, *a, **k)

    svc.brain.choose_move = spy
    asyncio.run(svc.make_move("a3"))
    assert seen and all(b is not svc.board for b in seen)


def test_autonomous_play_gives_orion_his_real_time_and_depth():
    """It gave him 0.3 s a move and ignored the depth it was handed."""
    import inspect
    from orion_core.chess_engine import ORION_SECONDS, ChessService
    params = inspect.signature(ChessService.start_autonomous).parameters
    assert params["orion_movetime_ms"].default == int(ORION_SECONDS * 1000)

    svc = _service()
    calls = []

    async def fake_orion(board, seconds=1.5, max_depth=None):
        calls.append((seconds, max_depth))
        return None

    svc._orion_move = fake_orion

    async def run():
        await svc.new_game(orion_side="both")
        svc.start_autonomous(delay_seconds=0.2, orion_depth=7)
        await asyncio.sleep(0.3)
    asyncio.run(run())
    # (the first call is new_game's own opening reply; then the loop's)
    assert (ORION_SECONDS, 7) in calls


def test_voice_navigation_moves_the_view_not_the_game():
    svc = _service()
    asyncio.run(svc.new_game(white="user", black="user"))
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        assert asyncio.run(svc.make_move(san)).ok
    live = svc.board.fen()
    assert svc.step(-2).ok and svc.view_ply == 3
    assert svc.goto_move(2, "black").ok and svc.view_ply == 4
    assert svc.goto_ply(0).ok and svc.view_ply == 0
    assert not asyncio.run(svc.make_move("a6")).ok       # browsing: no moves
    assert svc.resume_live().ok and svc.is_live
    assert svc.board.fen() == live


def test_the_board_turns_round_when_you_play_black():
    svc = _service()
    asyncio.run(svc.new_game(player_colour="black"))
    assert svc.board_flipped() is True
    svc.flip()
    assert svc.board_flipped() is False


def test_a_new_game_from_the_panel_defaults_to_orion_not_stockfish():
    svc = _service()
    asyncio.run(svc.new_game())
    assert svc.sides == {"white": "user", "black": "orion"}


def test_a_played_move_is_reviewed_in_words_in_the_background():
    svc = _service()

    async def run():
        await svc.new_game(white="user", black="user")
        for san in ("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"):
            assert (await svc.make_move(san)).ok
        assert await svc.wait_for_reviews(timeout=60)
        return await svc.analyse_move(3, "black")
    result = asyncio.run(run())
    assert result.ok
    assert "blunder" in result.text and "Qxf7#" in result.text
    assert "centipawn" not in result.text.lower()


def test_old_theory_is_topped_up_not_double_counted(brain):
    from orion_core import chess_theory
    brain.book[chess_theory.THEORY_MARKER] = {"seeded": BookEntry()}   # a v1 brain
    start = " ".join(chess.Board().fen().split(" ")[:4])
    brain.book[start] = {"e2e4": BookEntry(wins=9, draws=2, losses=0)}
    added = chess_theory.seed_into(brain)
    assert added > 0
    assert brain.book[start]["e2e4"].wins == 9          # not summed a second time
    assert "d2d4" in brain.book[start]                  # new theory arrived
    assert chess_theory.seed_into(brain) == 0


def test_orions_move_never_waits_behind_a_review(brain):
    """A background review holds the brain; ORION interrupts it so his reply
    comes first, and the review is re-run later."""
    import threading
    import time
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8")
    out = {}
    worker = threading.Thread(target=lambda: out.update(
        r=brain.review_move(board, board.parse_san("a3"), seconds=5.0)))
    started = time.perf_counter()
    worker.start()
    time.sleep(0.2)
    brain.interrupt()
    worker.join(timeout=5)
    assert out["r"] == {"interrupted": True}
    assert time.perf_counter() - started < 2.0
    assert brain.choose_move(board, seconds=0.3, use_book=False) in board.legal_moves
    assert "depth" in brain.last_reason
