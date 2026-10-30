"""Cooperative daemon self-play: one CPU, bounded episodes and compressed tuples."""
from __future__ import annotations

import random
import time
from pathlib import Path

from .curriculum import OPENINGS, training_start
from .dataset import Dataset, is_quiet_position
from .search import Position, make_brain, refresh_weights, run_search, worker_priority
from .strength import Limits


def quiescent_target(brain, board, seconds=0.02) -> int | None:
    from ..chess_brain import MAX_PLY, _SearchTimeout
    brain.nodes = 0
    brain.node_limit = 100_000
    brain._deadline = time.monotonic() + seconds
    brain._killers = [[None, None] for _ in range(MAX_PLY + 2)]
    brain._path = brain._game_path(board)
    try:
        return brain._qsearch(board.copy(), -32000, 32000, 0)
    except _SearchTimeout:
        return None              # never label a truncated qsearch as a quiet value


def play_episode(brain, stop, pause, rng, *, seconds=0.06, max_plies=400, version="classical",
                 start=None, quiet_only=False):
    import chess
    if start is None:
        board = chess.Board()
        for uci in rng.choice(OPENINGS):
            board.push_uci(uci)
    else:
        board = start.board(rng)
    # Independent games cannot inherit prior-game repetition/search heuristics.
    brain._tt.clear()
    brain._history = [0] * 4096
    positions = []
    for ply in range(max_plies):
        if stop.is_set():
            return None
        while pause.is_set():
            if stop.wait(0.05):
                return None
        if board.is_game_over(claim_draw=True):
            return positions, board.result(claim_draw=True)
        if not board.is_check() and (not quiet_only or is_quiet_position(board)):
            cp = quiescent_target(brain, board)
            if cp is not None and abs(cp) < 29000:
                positions.append((board.fen(), cp))
        limits = Limits(seconds, 8, 20000, 2604, True, 0.25 if ply < 24 else 0.05)
        answer = run_search(brain, Position.from_board(board), limits, seed=rng.getrandbits(64), version=version)
        if answer.uci is None:
            return None
        board.push_uci(answer.uci)
    # Capped games are discarded; treating them as draws poisons the WDL target.
    return (positions, board.result(claim_draw=True)) if board.is_game_over(claim_draw=True) else None


def collect_episode(dataset, brain, stop, pause, rng, *, seconds=0.06, version="classical",
                    quiet_only=False):
    """Rotate durable lessons; interrupted games never become labelled draws."""
    start = training_start(dataset.curriculum_progress()["attempts"])
    episode = play_episode(brain, stop, pause, rng, seconds=seconds, version=version,
                           start=start, quiet_only=quiet_only)
    if episode is not None and episode[0]:
        dataset.append(*episode, version, pathway=start.key)
        return episode
    if not stop.is_set():
        dataset.record_censored(start.key)
    return None


def selfplay_main(brain_path: str, directory: str, stop, pause):
    worker_priority()
    directory = Path(directory)
    dataset = Dataset(directory / "positions.sqlite3")
    brain = make_brain(brain_path, stop)
    version = "classical"
    rng = random.Random()
    try:
        while not stop.is_set():
            version = refresh_weights(brain, directory, version)
            collect_episode(dataset, brain, stop, pause, rng, version=version)
    finally:
        dataset.close()


def practice_main(brain_path, directory, games, stop, pause, result_pipe):
    worker_priority()
    directory = Path(directory)
    dataset = Dataset(directory / "positions.sqlite3")
    brain = make_brain(brain_path, stop)
    version, completed = "classical", 0
    try:
        for _ in range(max(1, min(200, games))):
            if stop.is_set():
                break
            version = refresh_weights(brain, directory, version)
            episode = collect_episode(dataset, brain, stop, pause, random.Random(), version=version)
            if episode is not None:
                completed += 1
        result_pipe.send({"status": "practice complete", "games": completed,
                          "curriculum": dataset.curriculum_progress()})
    finally:
        dataset.close()
        result_pipe.close()
