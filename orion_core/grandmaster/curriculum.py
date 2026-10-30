"""Offline self-play starts; completed games remain the only WDL evidence."""
from __future__ import annotations

import random
from dataclasses import dataclass


VERSION = "mixed-v1"

# Keep the arena's opening suite independent of the training-only endgames.
OPENINGS = (
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("d2d4", "g8f6", "c2c4", "g7g6"),
    ("g1f3", "d7d5", "c2c4", "e7e6"),
    ("e2e4", "e7e6", "d2d4", "d7d5"),
)


@dataclass(frozen=True)
class TrainingStart:
    key: str
    fen: str | None = None

    def board(self, rng):
        import chess
        board = chess.Board(self.fen) if self.fen else chess.Board()
        if self.fen:
            # Swap colours and ranks together: both sides learn conversion
            # and defence, preserving legal pawn directions and side to move.
            if rng.getrandbits(1):
                board = board.mirror()
        else:
            for uci in rng.choice(OPENINGS):
                board.push_uci(uci)
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            raise ValueError("Invalid self-play curriculum start")
        return board


OPENING = TrainingStart("opening")
ENDGAMES = (
    TrainingStart("pawn_endgame", "8/5pk1/6p1/3p4/3P4/6P1/5PK1/8 w - - 0 1"),
    TrainingStart("rook_endgame", "6kr/5p1p/6p1/8/8/6P1/5P1P/R5K1 w - - 0 1"),
    TrainingStart("bishop_endgame", "8/5pk1/4b1p1/3p4/3P4/4B1P1/5PK1/8 w - - 0 1"),
    TrainingStart("knight_endgame", "8/5pk1/4n1p1/3p4/3P4/4N1P1/5PK1/8 b - - 0 1"),
    TrainingStart("queen_conversion", "8/k7/8/8/8/8/3Q2K1/8 w - - 0 1"),
    TrainingStart("rook_conversion", "8/k7/8/8/8/8/3R2K1/8 b - - 0 1"),
)
PATHWAYS = (OPENING.key,) + tuple(start.key for start in ENDGAMES)


def training_start(attempts: int) -> TrainingStart:
    """Balanced shuffled blocks; resume exactly from durable attempt counts.

    Capped games advance the rotation, so a difficult lesson cannot starve
    every later pathway. Interruptions do not consume a lesson. Shuffle each
    block independently so strict alternation cannot align endgames with the
    dataset's every-tenth-game validation split.
    """
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
        raise ValueError("Invalid curriculum progress")
    block, offset = divmod(attempts, 2 * len(ENDGAMES))
    starts = [OPENING] * len(ENDGAMES) + list(ENDGAMES)
    random.Random(20261019 + block).shuffle(starts)
    return starts[offset]
