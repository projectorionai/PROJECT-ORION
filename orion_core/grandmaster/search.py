"""Spawn-safe search workers. ProcessPoolExecutor uses bounded, pickled queue IPC.

FEN alone loses repetition history. Requests carry the initial FEN, UCI history
and expected final FEN, which the worker verifies before searching.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lifetime import finish
from .strength import Limits, select_move


@dataclass(frozen=True)
class Position:
    initial_fen: str
    moves: tuple[str, ...]
    fen: str

    @classmethod
    def from_board(cls, board):
        return cls(board.root().fen(), tuple(m.uci() for m in board.move_stack), board.fen())

    def board(self):
        import chess
        if len(self.moves) > 1200 or len(self.fen) > 128 or len(self.initial_fen) > 128:
            raise ValueError("Position exceeds IPC bounds")
        board = chess.Board(self.initial_fen)
        if not board.is_valid():
            raise ValueError("Invalid initial position")
        for move in self.moves:
            board.push_uci(move)
        if board.fen() != self.fen or not board.is_valid():
            raise ValueError("Position history/FEN mismatch")
        return board


@dataclass(frozen=True)
class SearchResult:
    uci: str | None
    cp: int                         # root side to move, never White implicitly
    depth: int
    nodes: int
    elapsed: float
    pv: tuple[str, ...] = ()
    in_book: bool = False
    candidates: tuple[tuple[str, int], ...] = ()
    version: str = "classical"


def worker_priority():
    # Must happen before NumPy/Torch import in every spawned process.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        import psutil
        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except (ImportError, OSError):
        pass


def make_brain(path: str, cancelled=None):
    """Subclass rather than change the existing tested alpha-beta implementation."""
    from ..chess_brain import _INF, ChessBrain, _SearchTimeout

    class IsolatedBrain(ChessBrain):
        def __init__(self):
            super().__init__(Path(path))
            self.cancelled = cancelled
            self.node_limit = 2_000_000
            self.ranked = False
            self.candidates = []
            self.nnue = None
            self._managed_nnue = False       # supervisor/arena owns this worker's model

        def _checkpoint(self):
            if self.nodes >= self.node_limit or (self.cancelled is not None and self.cancelled.is_set()):
                raise _SearchTimeout

        def _search(self, *args, **kwargs):
            self._checkpoint()
            return super()._search(*args, **kwargs)

        def _qsearch(self, *args, **kwargs):
            self._checkpoint()
            return super()._qsearch(*args, **kwargs)

        def _eval_stm(self, board):
            if self.nnue is not None:
                return self.nnue.evaluate(board)
            return super()._eval_stm(board)

        def _search_root(self, board, depth, alpha, beta, root_moves):
            if not self.ranked:
                return super()._search_root(board, depth, alpha, beta, root_moves)
            # Full windows produce comparable multi-PV scores. Only publish a
            # complete iteration; partial alpha-beta bounds are not probabilities.
            scored = []
            self._path.append(hash(board._transposition_key()))
            try:
                for move in root_moves:
                    board.push(move)
                    try:
                        score = -self._search(board, depth - 1, -_INF, _INF, 1)
                    finally:
                        board.pop()
                    scored.append((move.uci(), score))
            finally:
                self._path.pop()
            self.candidates = sorted(scored, key=lambda s: s[1], reverse=True)
            import chess
            best, cp = self.candidates[0]
            return cp, chess.Move.from_uci(best)

    return IsolatedBrain()


_brain: Any = None
_cancel: Any = None
_directory: Path | None = None
_version = "classical"
_brain_stamp = None


def initialise(brain_path: str, directory: str, cancelled):
    global _brain, _cancel, _directory, _brain_stamp
    worker_priority()
    _cancel, _directory = cancelled, Path(directory)
    _brain = make_brain(brain_path, cancelled)
    try:
        _brain_stamp = _brain.path.stat().st_mtime_ns
    except OSError:
        _brain_stamp = None


def refresh_weights(brain, directory: Path, version: str) -> str:
    """Read immutable weights at a search boundary; existing search owns its model."""
    from .weights import Inference, active_model, file_hash
    try:
        selected = active_model(directory)
        if selected is not None and selected.name != version:
            # Build/validate completely BEFORE changing the reference or clearing TT.
            if file_hash(selected) != selected.stem:
                return version
            replacement = Inference.load(selected)
            brain.nnue = replacement
            brain._tt.clear()
            return selected.name
    except (OSError, ValueError, TypeError, KeyError):
        # A corrupt/incomplete candidate never takes a live game down.
        return version
    return version


def run_search(brain, position: Position, limits: Limits, *, seed=None, book: Path | None = None,
               tablebase: Path | None = None, version="classical") -> SearchResult:
    board = position.board()
    start = time.monotonic()
    if board.is_game_over():
        return SearchResult(None, -30000 if board.is_checkmate() else 0, 0, 0, 0, version=version)
    brain.node_limit = limits.nodes
    brain.ranked = limits.multipv or limits.exploration > 0
    brain.candidates = []
    if tablebase:
        from .book import tablebase_move
        exact = tablebase_move(board, tablebase)
        if exact is not None:
            uci, cp = exact
            return SearchResult(uci, cp, 0, 0, time.monotonic() - start, (uci,), version=version)
    if book and not brain.ranked:
        from .book import book_move
        uci = book_move(board, book, random.Random(seed))
        if uci:
            return SearchResult(uci, 0, 0, 0, time.monotonic() - start, (uci,), True, version=version)
    move = brain.choose_move(board, limits.seconds, limits.depth, use_book=not brain.ranked)
    cp = brain._last_score
    if brain.candidates:
        import chess
        uci, cp = select_move(brain.candidates, limits, random.Random(seed))
        move = chess.Move.from_uci(uci)
    in_book = "book" in brain.last_reason
    pv = tuple(m.uci() for m in brain.last_pv)
    if move and (not pv or pv[0] != move.uci()):
        pv = (move.uci(),)
    return SearchResult(move.uci() if move else None, cp, brain._last_depth,
                        0 if in_book else brain.nodes, time.monotonic() - start, pv, in_book,
                        tuple(brain.candidates), version)


def execute(position: Position, limits: Limits, operation: str, played: str | None):
    global _version, _brain_stamp
    try:
        stamp = _brain.path.stat().st_mtime_ns
    except OSError:
        stamp = None
    if stamp is not None and stamp != _brain_stamp:
        _brain.load()
        _brain._load_value()
        _brain._tt.clear()
        _brain_stamp = stamp
    _version = refresh_weights(_brain, _directory, _version)
    if operation == "review":
        import chess
        _brain.node_limit = limits.nodes
        _brain.ranked = False
        return _brain.review_move(position.board(), chess.Move.from_uci(played), limits.seconds)
    if operation == "analyse":
        board = position.board()
        _brain.node_limit = limits.nodes
        _brain.ranked = False
        return _brain.analyse(board, limits.seconds, limits.depth)
    return run_search(_brain, position, limits, book=_directory / "book.bin",
                      tablebase=_directory / "syzygy", version=_version)


class SearchProcess:
    """One worker and one outstanding request. Cancellation retains the slot
    until the worker acknowledges it, preventing unbounded executor backlogs.
    """
    def __init__(self, brain_path: Path, directory: Path):
        self.brain_path, self.directory = str(brain_path), str(directory)
        self._context = mp.get_context("spawn")
        self._cancel = self._context.Event()
        self._pool = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._waiters = 0

    def _submit(self, position, limits, operation, played):
        # Process creation and the first import can block: submit from an I/O thread.
        if self._pool is None:
            self._pool = ProcessPoolExecutor(1, mp_context=self._context, initializer=initialise,
                                             initargs=(self.brain_path, self.directory, self._cancel))
        return self._pool.submit(execute, position, limits, operation, played)

    async def search(self, position: Position, limits=None, *, operation="search", played=None):
        limits = limits or Limits()
        if self._closed:
            raise RuntimeError("Search worker closed")
        if self._waiters >= 2:
            raise RuntimeError("Search worker busy")
        self._waiters += 1
        try:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("Search worker closed")
                self._cancel.clear()
                submission = asyncio.create_task(asyncio.to_thread(self._submit, position, limits, operation, played))
                future = None
                try:
                    concurrent = await asyncio.shield(submission)
                    future = asyncio.wrap_future(concurrent)
                    return await asyncio.shield(future)
                finally:
                    self._cancel.set()
                    interrupted = False
                    if future is None:
                        concurrent, interrupted = await finish(submission)
                        future = asyncio.wrap_future(concurrent)
                    # Search checkpoints and deadlines bound this drain. Never
                    # cancel the asyncio wrapper: that does not stop a process.
                    if not future.done():
                        _result, again = await finish(future)
                        interrupted |= again
                    if interrupted:
                        raise asyncio.CancelledError
        finally:
            self._waiters -= 1

    def shutdown(self):
        self._closed = True
        self._cancel.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    async def close(self):
        self._closed = True
        self._cancel.set()
        async with self._lock:
            if self._pool is not None:
                await asyncio.to_thread(self._pool.shutdown, wait=True, cancel_futures=True)
                self._pool = None
