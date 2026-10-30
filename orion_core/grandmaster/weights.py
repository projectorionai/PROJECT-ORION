"""HalfKP features and versioned ORION .nnue files (not Stockfish format).

No pickle or Torch in inference. Immutable, validated, memory-mapped float32
weights are replaced by reference only between searches. Accumulators update
by adding/removing changed feature rows, including king changes and unmake.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import uuid
from pathlib import Path

FEATURES = 1 + 64 * 10 * 64
WIDTH = 128
SCALE_CP = 300.0
SHAPES = {
    "feature.weight": (FEATURES, WIDTH), "bias": (WIDTH,),
    "hidden1.weight": (32, 2 * WIDTH), "hidden1.bias": (32,),
    "hidden2.weight": (32, 32), "hidden2.bias": (32,),
    "output.weight": (1, 32), "output.bias": (1,),
}
MAGIC = b"ORNNUE1\n"
HEADER = json.dumps({"features": "HalfKP-v1", "width": WIDTH, "scale_cp": SCALE_CP},
                    sort_keys=True).encode("ascii")
OFFSET = len(MAGIC) + 4 + len(HEADER)
FILE_SIZE = OFFSET + sum(math.prod(shape) * 4 for shape in SHAPES.values())


def features(board, perspective: bool) -> tuple[int, ...]:
    import chess
    king = board.king(perspective)
    if king is None:
        raise ValueError("HalfKP requires both kings")
    flip = 0 if perspective == chess.WHITE else 63
    king ^= flip
    result = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        piece_index = piece.piece_type - 1 + (0 if piece.color == perspective else 5)
        result.append(1 + (king * 10 + piece_index) * 64 + (square ^ flip))
    return tuple(sorted(result))


def feature_sets(board) -> tuple[set[int], set[int]]:
    """Build both HalfKP perspectives in one pass for search inference."""
    import chess

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("HalfKP requires both kings")
    white_base = 1 + white_king * 640
    black_base = 1 + (black_king ^ 63) * 640
    white, black = set(), set()
    for piece_type in range(chess.PAWN, chess.QUEEN + 1):
        for colour in (chess.WHITE, chess.BLACK):
            white_offset = white_base + (piece_type - 1 + (0 if colour else 5)) * 64
            black_offset = black_base + (piece_type - 1 + (0 if not colour else 5)) * 64
            for square in chess.scan_forward(board.pieces_mask(piece_type, colour)):
                white.add(white_offset + square)
                black.add(black_offset + (square ^ 63))
    return white, black


def export_weights(state, path: Path) -> str:
    """Atomic file; state contains NumPy arrays or CPU torch tensors."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(MAGIC + struct.pack("<I", len(HEADER)) + HEADER)
            for name, shape in SHAPES.items():
                value = state[name]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                array = np.asarray(value, dtype="<f4")
                if name == "feature.weight" and "shared.weight" in state:
                    shared = state["shared.weight"]
                    if hasattr(shared, "detach"):
                        shared = shared.detach().cpu().numpy()
                    shared = np.asarray(shared, dtype="<f4")
                    if shared.shape != (641, WIDTH) or not np.isfinite(shared).all() or np.any(shared[0]):
                        raise ValueError("Invalid shared piece-square factor")
                    if array.shape != shape:
                        raise ValueError("Invalid HalfKP table")
                    array = array.copy()
                    array[1:] += np.tile(shared[1:], (64, 1))
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"Invalid NNUE tensor: {name}")
                if name == "feature.weight" and np.any(array[0]):
                    raise ValueError("Padding feature must be zero")
                output.write(array.tobytes(order="C"))
            output.flush()
            os.fsync(output.fileno())
        digest = file_hash(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def save_training_checkpoint(candidate: Path, checkpoint: Path):
    """Retain learning progress independently of the playing champion."""
    temporary = checkpoint.with_suffix(".tmp")
    try:
        with candidate.open("rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)


def read_weights(path: Path):
    import numpy as np
    path = Path(path)
    if path.stat().st_size != FILE_SIZE:
        raise ValueError("Wrong NNUE size")
    with path.open("rb") as source:
        if source.read(OFFSET) != MAGIC + struct.pack("<I", len(HEADER)) + HEADER:
            raise ValueError("Unsupported NNUE architecture")
    tensors, offset = {}, OFFSET
    for name, shape in SHAPES.items():
        array = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=shape)
        if not np.isfinite(array).all():
            raise ValueError("Non-finite NNUE weights")
        tensors[name] = array
        offset += array.nbytes
    if np.any(tensors["feature.weight"][0]):
        raise ValueError("Invalid padding feature")
    return tensors


class Inference:
    def __init__(self, tensors):
        import numpy as np

        # ndarray views retain their memmap bases without the per-operation
        # memmap subclass overhead in every search leaf.
        self.tensors = ({name: np.asarray(value) for name, value in tensors.items()}
                        if any(isinstance(value, np.memmap) for value in tensors.values())
                        else tensors)
        self._sets = [None, None]
        self._accumulators = [None, None]
        self._updates = 0

    @classmethod
    def load(cls, path):
        return cls(read_weights(path))

    def _accumulate(self, current, colour):
        index = int(colour)
        previous = self._sets[index]
        table = self.tensors["feature.weight"]
        if previous is None:
            acc = self.tensors["bias"].copy()
            if current:
                acc += table[list(current)].sum(axis=0)
        else:
            acc = self._accumulators[index]
            added, removed = current - previous, previous - current
            if added:
                acc += table[list(added)].sum(axis=0)
            if removed:
                acc -= table[list(removed)].sum(axis=0)
        self._sets[index], self._accumulators[index] = current, acc
        return acc.clip(0, 1)

    def evaluate(self, board) -> int:
        import numpy as np
        self._updates += 1
        if self._updates % 256 == 0:
            # Bound floating-point drift after long push/pop search sequences.
            self._sets = [None, None]
        white, black = feature_sets(board)
        ours, theirs = (white, black) if board.turn else (black, white)
        x = np.concatenate((self._accumulate(ours, board.turn),
                            self._accumulate(theirs, not board.turn)))
        for name in ("hidden1", "hidden2"):
            x = (self.tensors[name + ".weight"] @ x + self.tensors[name + ".bias"]).clip(0, 1)
        logit = float((self.tensors["output.weight"] @ x + self.tensors["output.bias"])[0])
        return round(max(-20000, min(20000, logit * SCALE_CP)))


def active_model(directory: Path) -> Path | None:
    manifest = Path(directory) / "active.json"
    if not manifest.exists():
        return None
    if manifest.stat().st_size > 4096:
        raise ValueError("Oversized model manifest")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    digest = data.get("sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid model digest")
    return Path(directory) / (digest + ".nnue")
