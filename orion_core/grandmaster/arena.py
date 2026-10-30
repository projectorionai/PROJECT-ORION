"""Paired colour-reversed validation; immutable models and atomic manifest swap."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path

from .search import Position, make_brain, run_search, worker_priority
from .selfplay import OPENINGS
from .strength import Limits
from .weights import Inference, active_model, file_hash


def lower_bound(pair_scores: list[float], alpha=0.01) -> float:
    """One-sided Hoeffding bound over independent colour-pair scores in [0,1]."""
    if not pair_scores or not 0 < alpha < 1 or any(not 0 <= s <= 1 for s in pair_scores):
        return 0.0
    return sum(pair_scores) / len(pair_scores) - math.sqrt(math.log(1 / alpha) / (2 * len(pair_scores)))


def promote(directory: Path, candidate: Path, expected: str | None, pair_scores: list[float],
            *, minimum_pairs=40, alpha=0.01) -> bool:
    if len(pair_scores) < minimum_pairs or lower_bound(pair_scores, alpha) <= 0.5:
        return False
    current = active_model(directory)
    if (current.name if current else None) != expected:
        return False                          # stale arena result cannot replace a newer champion
    verified = Inference.load(candidate)
    del verified
    digest = file_hash(candidate)
    destination = directory / (digest + ".nnue")
    if not destination.exists():
        # Immutable file first; manifest last. Readers can keep old mmaps alive.
        shutil.copyfile(candidate, destination)
        with destination.open("r+b") as handle:
            os.fsync(handle.fileno())
    if file_hash(destination) != digest:
        raise ValueError("Champion digest mismatch")
    manifest = {"sha256": digest, "pairs": len(pair_scores),
                "score": sum(pair_scores) / len(pair_scores),
                "lower_bound": lower_bound(pair_scores, alpha), "alpha": alpha,
                "previous": expected}
    temporary = directory / ("active-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory / "active.json")
    finally:
        temporary.unlink(missing_ok=True)
    return True


def arena_game(white, black, opening, stop, pause, *, seconds=0.08, max_plies=400):
    import chess
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    for brain in (white, black):
        brain._tt.clear()
        brain._history = [0] * 4096
        brain.book = {}
    for _ in range(max_plies):
        if stop.is_set():
            return None
        while pause.is_set():
            if stop.wait(0.05):
                return None
        if board.is_game_over(claim_draw=True):
            return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[board.result(claim_draw=True)]
        brain = white if board.turn else black
        result = run_search(brain, Position.from_board(board), Limits(seconds=seconds))
        if result.uci is None:
            return None
        board.push_uci(result.uci)
    return None                              # censored games are not fabricated draws


def arena_main(brain_path, directory, candidate, stop, pause, result_pipe, pairs=80):
    worker_priority()
    directory, candidate = Path(directory), Path(candidate)
    try:
        incumbent_path = active_model(directory)
        expected = incumbent_path.name if incumbent_path else None
        candidate_brain = make_brain(brain_path, stop)
        candidate_brain.nnue = Inference.load(candidate)
        incumbent = make_brain(brain_path, stop)
        if incumbent_path:
            incumbent.nnue = Inference.load(incumbent_path)
        rng, scores, seen = random.Random(), [], set()
        import chess
        for _ in range(pairs * 4):
            if stop.is_set() or len(scores) >= pairs:
                break
            opening = list(rng.choice(OPENINGS))
            board = chess.Board()
            for uci in opening:
                board.push_uci(uci)
            # Fresh suite for each candidate; reverse colours from identical FEN.
            for _ply in range(2):
                move = rng.choice(list(board.legal_moves))
                opening.append(move.uci())
                board.push(move)
            if board.fen() in seen:
                continue
            seen.add(board.fen())
            a = arena_game(candidate_brain, incumbent, opening, stop, pause)
            b = arena_game(incumbent, candidate_brain, opening, stop, pause)
            if a is not None and b is not None:
                scores.append((a + 1 - b) / 2)
        # A fixed horizon is essential: do not repeatedly test and stop on luck.
        promoted = (not stop.is_set() and len(scores) == pairs and
                    promote(directory, candidate, expected, scores))
        result_pipe.send({"status": "promoted" if promoted else "rejected", "pairs": len(scores),
                          "score": sum(scores) / len(scores) if scores else 0,
                          "lower_bound": lower_bound(scores)})
    except Exception as exc:  # noqa: BLE001 - report worker failures through bounded IPC
        result_pipe.send({"status": "error", "reason": type(exc).__name__ + ": " + str(exc)[:160]})
    finally:
        # Release Windows mappings before candidate cleanup by the parent.
        result_pipe.close()
