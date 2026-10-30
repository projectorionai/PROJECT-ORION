"""Quiet training views preserve complete-game archives and split membership."""
from __future__ import annotations

import random

import chess
import pytest

from orion_core.grandmaster.dataset import Dataset, is_quiet_position


CAPTURE = "4k3/8/8/4p3/4R3/8/8/4K3 w - - 0 1"
CHECK = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
WHITE_PROMOTION = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
BLACK_PROMOTION = "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"
EN_PASSANT = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"


@pytest.mark.parametrize("fen", [CAPTURE, CHECK, WHITE_PROMOTION, BLACK_PROMOTION, EN_PASSANT])
def test_quiet_rule_excludes_legal_tactical_continuations(fen):
    board = chess.Board(fen)
    assert board.is_valid()
    assert not is_quiet_position(board)


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    # Moving the pinned bishop to capture d3 would expose its king.
    "k3r3/8/8/8/8/3p4/4B3/4K3 w - - 0 1",
    # A pawn on the seventh rank cannot promote through its own rook.
    "R7/P6k/8/8/8/8/8/4K3 w - - 0 1",
])
def test_quiet_rule_uses_legal_moves_and_retains_blocked_promotions(fen):
    board = chess.Board(fen)
    assert board.is_valid()
    assert is_quiet_position(board)


def test_historical_quiet_views_preserve_archives_and_both_game_splits(tmp_path):
    dataset = Dataset(tmp_path / "positions.sqlite3")
    board = chess.Board()
    board.push_uci("g1f3")
    quiet_fens = {chess.STARTING_FEN, board.fen()}
    positions = [(fen, index * 10) for index, fen in enumerate(
        [chess.STARTING_FEN, board.fen(), CAPTURE, CHECK, WHITE_PROMOTION,
         BLACK_PROMOTION, EN_PASSANT])]
    try:
        for _ in range(20):
            dataset.append(positions, "1-0", "historical-classical")
        before = dataset.db.execute("SELECT * FROM games ORDER BY id").fetchall()
        count = dataset.count()

        training = dataset.sample(128, random.Random(22), quiet_only=True)
        validation = dataset.validation_rows(128, random.Random(22), quiet_only=True)
        sampled_validation = dataset.sample(128, random.Random(22), validation=True, quiet_only=True)
        assert len(training) == len(sampled_validation) == 128
        assert len(validation) == len(quiet_fens)
        for rows in (training, validation, sampled_validation):
            assert {row[0] for row in rows} == quiet_fens
            assert all(is_quiet_position(chess.Board(row[0])) for row in rows)
            assert all(row[2] == (1 if chess.Board(row[0]).turn else -1) for row in rows)
        # The ordinary view continues exposing the archived tactical rows.
        assert len(dataset._game(10)) == len(positions)
        assert len(dataset._game(10, quiet_only=True)) == len(quiet_fens)
        assert len(dataset.validation_rows(128, random.Random(22))) == len(positions)
        assert dataset.count() == count == len(positions) * 20
        assert dataset.db.execute("SELECT * FROM games ORDER BY id").fetchall() == before
    finally:
        dataset.close()


def test_quiet_sample_cannot_borrow_rows_from_opposite_split(tmp_path):
    dataset = Dataset(tmp_path / "positions.sqlite3")
    try:
        for game_id in range(1, 11):
            dataset.append([(chess.STARTING_FEN, game_id)], "1-0", "archive")
        training = dataset.sample(128, random.Random(1), quiet_only=True)
        validation = dataset.sample(128, random.Random(1), validation=True, quiet_only=True)
        assert {row[1] for row in training} == set(range(1, 10))
        assert {row[1] for row in validation} == {10}
        assert {row[1] for row in dataset.validation_rows(128, random.Random(1), quiet_only=True)} == {10}
    finally:
        dataset.close()


def test_empty_quiet_view_and_cache_are_bounded_without_deleting_games(tmp_path):
    dataset = Dataset(tmp_path / "positions.sqlite3")
    try:
        for game_id in range(1, 141):
            dataset.append([(CAPTURE, game_id)], "1-0", "archive")
            assert dataset._game(game_id, quiet_only=True) == []
        assert len(dataset._decoded) <= 128
        assert len(dataset._quiet_decoded) <= 128
        assert dataset.sample(16, random.Random(2), quiet_only=True) == []
        assert dataset.validation_rows(16, random.Random(2), quiet_only=True) == []
        assert dataset.count() == 140
        assert dataset.db.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 140
    finally:
        dataset.close()


def test_sparse_quiet_rows_fill_both_splits_without_whole_game_filtering(tmp_path):
    dataset = Dataset(tmp_path / "positions.sqlite3")
    try:
        for game_id in range(1, 21):
            dataset.append([(CAPTURE, 0)] * 59 + [(chess.STARTING_FEN, game_id)],
                           "1-0", "historical")
        count = dataset.count()
        training = dataset.sample(128, random.Random(3), quiet_only=True)
        validation = dataset.sample(128, random.Random(3), validation=True, quiet_only=True)
        assert len(training) == len(validation) == 128
        assert all(row[0] == chess.STARTING_FEN for row in training + validation)
        assert all(row[1] % 10 != 0 for row in training)
        assert {row[1] for row in validation} == {10, 20}
        # Sampling inspects individual rows rather than constructing 20 full
        # filtered views; distinct validation_rows still has that cached view.
        assert not dataset._quiet_decoded
        assert len(dataset.validation_rows(128, random.Random(3), quiet_only=True)) == 1
        assert dataset._quiet_decoded
        assert dataset.count() == count == 1200
    finally:
        dataset.close()


def test_repeated_quiet_sampling_reuses_bounded_position_checks(tmp_path, monkeypatch):
    from orion_core.grandmaster import dataset as data
    dataset = Dataset(tmp_path / "positions.sqlite3")
    original = data.is_quiet_position
    checked = []

    def observe(board):
        checked.append(board.fen())
        return original(board)

    monkeypatch.setattr(data, "is_quiet_position", observe)
    try:
        for _ in range(20):
            dataset.append([(CAPTURE, 0), (chess.STARTING_FEN, 0)], "1/2-1/2", "fixture")
        for _ in range(4):
            assert len(dataset.sample(128, random.Random(3), quiet_only=True)) == 128
        assert len(checked) == 2
        # Bound the cache even when a long-running trainer sees many distinct
        # positions; cache keys do not alter or rewrite stored training rows.
        for clock in range(8200):
            fen = chess.STARTING_FEN.rsplit(" ", 2)[0] + f" {clock} 1"
            dataset._is_quiet(fen)
        assert len(dataset._quiet_positions) == 8192
        assert dataset.count() == 40
    finally:
        dataset.close()


def test_quiet_collection_skips_tactical_labels_and_clears_previous_game_state(monkeypatch):
    import threading
    from types import SimpleNamespace
    from orion_core.grandmaster import selfplay
    from orion_core.grandmaster.curriculum import TrainingStart
    calls = []

    def target(_brain, board):
        calls.append(board.fen())
        return 17

    monkeypatch.setattr(selfplay, "quiescent_target", target)
    monkeypatch.setattr(selfplay, "run_search", lambda _brain, position, *_args, **_kwargs:
                        SimpleNamespace(uci=next(iter(position.board().legal_moves)).uci()))
    brain = SimpleNamespace(_tt={"previous game": 1}, _history=[99] * 4096)
    start = TrainingStart("fixture", CAPTURE)
    selfplay.play_episode(brain, threading.Event(), threading.Event(), random.Random(0),
                          max_plies=1, start=start, quiet_only=True)
    assert calls == []
    assert brain._tt == {} and brain._history == [0] * 4096
