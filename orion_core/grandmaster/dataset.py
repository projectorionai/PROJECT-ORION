"""WAL database of zlib-compressed games; terminal labels use side-to-move POV."""
from __future__ import annotations

import json
import math
import random
import sqlite3
import time
import zlib
from collections import OrderedDict
from pathlib import Path

from ..db import apply_pragmas


def is_quiet_position(board) -> bool:
    """A stand-pat position with no check, legal capture or legal promotion."""
    import chess
    if board.is_check() or next(board.generate_legal_captures(), None) is not None:
        return False
    promotion_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
    promoting = board.pawns & board.occupied_co[board.turn] & promotion_rank
    return not promoting or not any(
        move.promotion for move in board.generate_legal_moves(from_mask=promoting))


class Dataset:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10)
        apply_pragmas(self.db)
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("PRAGMA wal_autocheckpoint=256")
        self.db.execute("PRAGMA journal_size_limit=8388608")
        # Hard 2 GiB main-database ceiling. Freed pages are reused on eviction.
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        self.db.execute(f"PRAGMA max_page_count={2 * 1024**3 // page_size}")
        self.db.execute("""CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
            model TEXT NOT NULL, result TEXT NOT NULL, positions INTEGER NOT NULL,
            payload BLOB NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS curriculum_progress (
            pathway TEXT PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0,
            completed INTEGER NOT NULL DEFAULT 0, positions INTEGER NOT NULL DEFAULT 0)""")
        self.db.commit()
        self._decoded = OrderedDict()
        self._quiet_decoded = OrderedDict()
        self._quiet_positions = OrderedDict()

    def close(self):
        self.db.close()

    def append(self, positions: list[tuple[str, int]], result: str, version: str, *, pathway=None):
        if pathway is not None:
            self._validate_pathway(pathway)
        if result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError("Unfinished games have no WDL target")
        if not positions or len(positions) > 600:
            raise ValueError("Invalid game length")
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
        rows = []
        import chess
        for fen, cp in positions:
            if (not isinstance(fen, str) or len(fen) > 128
                    or not math.isfinite(cp) or abs(cp) > 32000):
                raise ValueError("Invalid training tuple")
            if not chess.Board(fen).is_valid():
                raise ValueError("Invalid training position")
            score = white_score if fen.split()[1] == "w" else 1 - white_score
            # WDL is explicit (-1,0,1), evaluation is always side to move.
            rows.append((fen, int(cp), int(score * 2 - 1)))
        payload = zlib.compress(json.dumps(rows, separators=(",", ":")).encode("ascii"), level=6)
        with self.db:
            self.db.execute("INSERT INTO games(created, model, result, positions, payload) VALUES(?,?,?,?,?)",
                            (time.time(), version, result, len(rows), payload))
            self.db.execute("DELETE FROM games WHERE id <= (SELECT MAX(id)-50000 FROM games)")
            if pathway is not None:
                # The game and its progress commit together, including across
                # crashes; no separately saved JSON cursor can skip experience.
                self._record_pathway(pathway, completed=1, positions=len(rows))

    @staticmethod
    def _validate_pathway(pathway):
        from .curriculum import PATHWAYS
        if pathway not in PATHWAYS:
            raise ValueError("Unknown chess learning pathway")

    def _record_pathway(self, pathway, *, completed=0, positions=0):
        self.db.execute("""INSERT INTO curriculum_progress(pathway, attempts, completed, positions)
            VALUES(?, 1, ?, ?) ON CONFLICT(pathway) DO UPDATE SET
            attempts=attempts+1, completed=completed+excluded.completed,
            positions=positions+excluded.positions""", (pathway, completed, positions))

    def record_censored(self, pathway):
        """Advance a capped lesson without inventing a result or training row."""
        self._validate_pathway(pathway)
        with self.db:
            self._record_pathway(pathway)

    def curriculum_progress(self):
        from .curriculum import VERSION
        pathways = {key: {"attempts": attempts, "completed": completed,
                          "censored": attempts - completed, "positions": positions}
                    for key, attempts, completed, positions in self.db.execute(
                        "SELECT pathway, attempts, completed, positions FROM curriculum_progress ORDER BY pathway")}
        return {"version": VERSION, "pathways": pathways,
                **{key: sum(row[key] for row in pathways.values())
                   for key in ("attempts", "completed", "censored", "positions")}}

    def _is_quiet(self, fen):
        if fen in self._quiet_positions:
            self._quiet_positions.move_to_end(fen)
            return self._quiet_positions[fen]
        import chess
        quiet = is_quiet_position(chess.Board(fen))
        self._quiet_positions[fen] = quiet
        while len(self._quiet_positions) > 8192:
            self._quiet_positions.popitem(last=False)
        return quiet

    def count(self) -> int:
        return int(self.db.execute("SELECT COALESCE(SUM(positions),0) FROM games").fetchone()[0])

    def _game(self, game_id, *, quiet_only=False):
        if quiet_only:
            if game_id in self._quiet_decoded:
                self._quiet_decoded.move_to_end(game_id)
                return self._quiet_decoded[game_id]
            game = [row for row in self._game(game_id)
                    if self._is_quiet(row[0])]
            self._quiet_decoded[game_id] = game
            while len(self._quiet_decoded) > 128:
                self._quiet_decoded.popitem(last=False)
            return game
        if game_id in self._decoded:
            self._decoded.move_to_end(game_id)
            return self._decoded[game_id]
        record = self.db.execute("SELECT payload FROM games WHERE id=?", (game_id,)).fetchone()
        if record is None:
            return []
        decoder = zlib.decompressobj()
        raw = decoder.decompress(record[0], 200_001)
        if len(raw) > 200_000 or not decoder.eof:
            raise ValueError("Corrupt compressed training game")
        game = json.loads(raw)
        self._decoded[game_id] = game
        while len(self._decoded) > 128:
            self._decoded.popitem(last=False)
        return game

    def validation_rows(self, size: int, rng: random.Random, *, quiet_only=False):
        """Fixed, distinct positions from complete held-out trajectories.

        Sampling with replacement could previously make a single position look
        like a validation set of 128 examples. Never inflate that evidence.
        """
        if not 1 <= size <= 2048:
            raise ValueError("Validation size exceeds bound")
        ids = [row[0] for row in self.db.execute("SELECT id FROM games WHERE id % 10 = 0 ORDER BY id")]
        rng.shuffle(ids)
        rows, seen = [], set()
        for game_id in ids:
            game = list(self._game(game_id, quiet_only=quiet_only))
            rng.shuffle(game)
            for row in game:
                key = " ".join(row[0].split()[:4])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if len(rows) == size:
                    return rows
        return rows

    def sample(self, size: int, rng: random.Random, *, validation=False, quiet_only=False):
        if not 1 <= size <= 2048:
            raise ValueError("Batch size exceeds bound")
        if quiet_only:
            ids = [row[0] for row in self.db.execute(
                "SELECT id FROM games WHERE (id % 10 = 0) = ?", (int(validation),))]
            if not ids:
                return []
            rows = []
            # Check only the selected archived row, avoiding a full game's FEN
            # parsing per draw. Quiet-rich games contribute more accepted rows.
            # Preselect the split so scarce held-out rows get the same budget.
            for _ in range(size * 100):
                game = self._game(rng.choice(ids))
                if game:
                    row = rng.choice(game)
                    if self._is_quiet(row[0]):
                        rows.append(row)
                if len(rows) == size:
                    break
            return rows
        low, high = self.db.execute("SELECT MIN(id), MAX(id) FROM games").fetchone()
        if low is None:
            return []
        rows = []
        # Split by complete game; positions from one trajectory never leak into
        # both train and validation. Rejection sampling avoids ORDER BY RANDOM.
        for _ in range(size * 30):
            game_id = rng.randint(low, high)
            if (game_id % 10 == 0) != validation:
                continue
            game = self._game(game_id, quiet_only=quiet_only)
            if game:
                rows.append(rng.choice(game))
            if len(rows) == size:
                break
        return rows
