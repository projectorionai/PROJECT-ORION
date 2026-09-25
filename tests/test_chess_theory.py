"""
ORION knowing his openings, not rediscovering them.

  "can you train him to be extremely good at chess ... Teach him all the
   openings, theory, midgames, endgames."

He has his own engine (verified elsewhere — it never calls Stockfish). What it
lacked was a head start: the opening book began empty, so he played the first
dozen moves from first principles for hundreds of games before it improved.
This seeds his book with real main-line theory as both colours, so he plays
principled, tested openings immediately.

The seeding must be additive and idempotent: it tops ORION up without ever
erasing the experience he earns from his own games.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

chess = pytest.importorskip("chess")

from orion_core import chess_theory  # noqa: E402
from orion_core.chess_brain import BookEntry, ChessBrain  # noqa: E402


# ── the repertoire is real, legal theory ─────────────────────────────────────

def test_every_line_is_legal_chess():
    """A typo in a line would seed an illegal or wrong position. Play each one
    out in full."""
    for name, line in chess_theory.REPERTOIRE.items():
        board = chess.Board()
        for san in line:
            try:
                move = board.parse_san(san)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"{name}: '{san}' is not legal here — {exc}")
            board.push(move)


def test_the_repertoire_is_broad():
    """'all the openings' — cover the major first moves and defences."""
    assert len(chess_theory.REPERTOIRE) >= 20
    firsts = {line[0] for line in chess_theory.REPERTOIRE.values()}
    assert {"e4", "d4", "c4", "Nf3"} <= firsts


def test_both_colours_are_covered():
    """He must play theory whether he has White or Black."""
    names = " ".join(chess_theory.REPERTOIRE).lower()
    assert "defence" in names or "defense" in names   # Black systems
    assert "lopez" in names or "italian" in names     # White systems


def test_the_book_builds_without_error():
    book = chess_theory.build_theory_book()
    assert len(book) > 100
    # Every entry is a winning-record BookEntry so it clears the trust bar.
    sample = next(iter(book.values()))
    entry = next(iter(sample.values()))
    assert entry.score > 0.45
    assert entry.confidence() >= 1.0


# ── seeding into a brain ─────────────────────────────────────────────────────

@pytest.fixture
def brain(tmp_path):
    return ChessBrain(path=tmp_path / "brain.json", seed=3)


def test_seeding_populates_the_book(brain):
    added = chess_theory.seed_into(brain)
    assert added > 100
    assert brain.book


def test_he_plays_theory_as_white(brain):
    chess_theory.seed_into(brain)
    move = brain.choose_move(chess.Board(), seconds=0.3)
    assert "book" in brain.last_reason, "did not play from the seeded book"
    # A sound first move, not something random.
    assert move.uci() in {"e2e4", "d2d4", "c2c4", "g1f3"}


def test_he_plays_theory_as_black(brain):
    chess_theory.seed_into(brain)
    board = chess.Board()
    board.push_san("e4")
    brain.choose_move(board, seconds=0.3)
    assert "book" in brain.last_reason


def test_transpositions_share_entries(brain):
    """Because the book is keyed on the position, reaching the same position by
    a different move order finds the same theory."""
    chess_theory.seed_into(brain)
    # 1.d4 Nf6 2.c4 and 1.c4 Nf6 2.d4 reach the same position.
    a = chess.Board()
    for san in ("d4", "Nf6", "c4"):
        a.push_san(san)
    b = chess.Board()
    for san in ("c4", "Nf6", "d4"):
        b.push_san(san)
    key_a = " ".join(a.fen().split(" ")[:4])
    key_b = " ".join(b.fen().split(" ")[:4])
    assert key_a == key_b
    assert key_a in brain.book


# ── it must not trample real experience ──────────────────────────────────────

def test_seeding_is_idempotent(brain):
    first = chess_theory.seed_into(brain)
    second = chess_theory.seed_into(brain)
    assert first > 0
    assert second == 0, "theory was seeded twice"


def test_seeding_survives_a_reload(brain, tmp_path):
    chess_theory.seed_into(brain)
    brain.save()
    reloaded = ChessBrain(path=brain.path, seed=3)
    assert chess_theory.THEORY_MARKER in reloaded.book
    assert chess_theory.seed_into(reloaded) == 0, "re-seeded after reload"


def test_it_adds_to_rather_than_replaces_a_learned_move(brain):
    """ORION's own hard-won record and the book's are the same currency and
    simply sum — his experience is never erased."""
    # Teach him a real result first.
    board = chess.Board()
    key = " ".join(board.fen().split(" ")[:4])
    brain.book[key] = {"e2e4": BookEntry(wins=3, draws=0, losses=1)}
    chess_theory.seed_into(brain)
    entry = brain.book[key]["e2e4"]
    assert entry.wins >= 3 + chess_theory.SEED_WINS  # summed, not overwritten
    assert entry.losses == 1                          # his loss is still recorded


def test_the_marker_key_is_never_a_playable_position(brain):
    """The 'seeded' sentinel must not be selectable as a move for any real
    board."""
    chess_theory.seed_into(brain)
    board = chess.Board()
    real_key = " ".join(board.fen().split(" ")[:4])
    assert chess_theory.THEORY_MARKER != real_key
    # It has a zero-games entry, which the book selector rejects anyway.
    sentinel = brain.book[chess_theory.THEORY_MARKER]["seeded"]
    assert sentinel.games == 0


# ── it is wired into the runtime brain, not the test brain ───────────────────

def test_the_service_seeds_theory():
    import inspect

    from orion_core.chess_engine import ChessService

    source = inspect.getsource(ChessService)
    assert "chess_theory" in source
    assert "seed_into" in source


def test_coverage_reports_real_numbers():
    cov = chess_theory.coverage()
    assert cov["openings"] == len(chess_theory.REPERTOIRE)
    assert cov["book_positions"] > 100


# ── Mark XXII: the deepened repertoire is all legal, and it grew ─────────────

def test_every_repertoire_line_is_fully_legal_san():
    """A typo would silently truncate a line (build_theory_book breaks on it),
    quietly shrinking ORION's book. Assert every SAN in every line is legal."""
    import chess
    from orion_core.chess_theory import REPERTOIRE
    for name, line in REPERTOIRE.items():
        board = chess.Board()
        for i, san in enumerate(line):
            try:
                board.push_san(san)
            except Exception as exc:  # pragma: no cover - failure detail
                raise AssertionError(f"{name}: illegal SAN at move {i+1} "
                                     f"{san!r} — {exc}")


def test_repertoire_is_substantial():
    from orion_core.chess_theory import REPERTOIRE, coverage
    assert len(REPERTOIRE) >= 50
    cov = coverage()
    assert cov["book_positions"] >= 340
