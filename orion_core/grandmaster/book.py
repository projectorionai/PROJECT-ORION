"""Streaming PGN -> SQLite aggregates -> sorted Polyglot; mmap readers only."""
from __future__ import annotations

import hashlib
import math
import os
import random
import sqlite3
import struct
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from ..db import apply_pragmas

DEFAULT_SOURCES = ("https://www.pgnmentor.com/players/Carlsen.zip",)


def raw_move(board, move):
    import chess
    target = move.to_square
    if board.is_castling(move):
        target = chess.square(7 if chess.square_file(target) > chess.square_file(move.from_square) else 0,
                              chess.square_rank(move.from_square))
    promotion = move.promotion - 1 if move.promotion else 0
    return target | (move.from_square << 6) | (promotion << 12)


def compile_book(pgn_paths, directory: Path, stop=None, min_elo=2500, max_ply=24):
    import chess
    import chess.pgn
    import chess.polyglot

    from ..atomic_io import _replace

    class Mainline(chess.pgn.GameBuilder):
        def begin_variation(self):
            return chess.pgn.SKIP

        def visit_comment(self, comment):
            pass

        def handle_error(self, error):
            self.game.errors.append(error)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "book.sqlite3", timeout=10)
    temporary = directory / ("book-" + uuid.uuid4().hex + ".tmp")
    try:
        apply_pragmas(db)
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA cache_size=-4096")
        db.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT, move INTEGER, n INTEGER, score REAL, PRIMARY KEY(key,move))")
        db.execute("CREATE TABLE IF NOT EXISTS sources (digest TEXT PRIMARY KEY)")
        db.execute("CREATE TABLE IF NOT EXISTS seen_games (digest TEXT PRIMARY KEY)")
        for path in pgn_paths:
            if stop is not None and stop.is_set():
                return 0
            path = Path(path)
            if path.stat().st_size > 32 * 1024**2:
                raise ValueError("PGN exceeds 32 MiB")
            from .weights import file_hash
            digest = file_hash(path)
            if db.execute("SELECT 1 FROM sources WHERE digest=?", (digest,)).fetchone():
                continue
            with db, path.open("r", encoding="utf-8-sig", errors="replace") as source:
                while True:
                    if stop is not None and stop.is_set():
                        raise InterruptedError("Book compilation cancelled")
                    game = chess.pgn.read_game(source, Visitor=Mainline)
                    if game is None:
                        break
                    try:
                        minimum = min(int(game.headers.get("WhiteElo", "0")), int(game.headers.get("BlackElo", "0")))
                    except ValueError:
                        continue
                    result = game.headers.get("Result")
                    if minimum < min_elo or game.errors or result not in ("1-0", "0-1", "1/2-1/2"):
                        continue
                    if game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
                        continue
                    board = game.board()
                    if not board.is_valid():
                        continue
                    mainline = tuple(game.mainline_moves())
                    if len(mainline) > 600:
                        continue
                    signature = hashlib.sha256((str(dict(game.headers)) + board.fen() +
                                                " ".join(m.uci() for m in mainline)).encode()).hexdigest()
                    if db.execute("SELECT 1 FROM seen_games WHERE digest=?", (signature,)).fetchone():
                        continue
                    white_score = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}[result]
                    for ply, move in enumerate(mainline):
                        if ply >= max_ply:
                            break
                        if move not in board.legal_moves:
                            break
                        score = white_score if board.turn else 1 - white_score
                        key = f"{chess.polyglot.zobrist_hash(board):016x}"
                        db.execute("""INSERT INTO entries VALUES(?,?,1,?) ON CONFLICT(key,move)
                                      DO UPDATE SET n=n+1, score=score+excluded.score""",
                                   (key, raw_move(board, move), score))
                        board.push(move)
                    db.execute("INSERT INTO seen_games VALUES(?)", (signature,))
                db.execute("INSERT INTO sources VALUES(?)", (digest,))
                # Bound persistent aggregates; prioritise supported positions.
                db.execute("DELETE FROM entries WHERE rowid IN (SELECT rowid FROM entries ORDER BY n DESC, rowid LIMIT -1 OFFSET 200000)")
        count = 0
        with temporary.open("wb") as output:
            for key, move, n, score in db.execute("SELECT key,move,n,score FROM entries ORDER BY key,move"):
                # Beta(1,1) posterior, weighted by evidence. All entries nonzero.
                weight = max(1, min(65535, round(100 * math.sqrt(n) * (score + 1) / (n + 2))))
                output.write(struct.pack(">QHHI", int(key, 16), move, weight, 0))
                count += 1
            output.flush()
            os.fsync(output.fileno())
        if count:
            _replace(str(temporary), directory / "book.bin")
        return count
    finally:
        db.close()
        temporary.unlink(missing_ok=True)


def book_move(board, path: Path, rng: random.Random):
    import chess.polyglot
    if not Path(path).is_file() or board.fullmove_number > 16:
        return None
    try:
        with chess.polyglot.open_reader(path) as reader:
            return reader.weighted_choice(board, random=rng).move.uci()
    except (OSError, ValueError, IndexError):
        return None


def tablebase_move(board, directory: Path):
    import chess
    import chess.syzygy
    if len(board.piece_map()) > 6 or board.castling_rights or not Path(directory).is_dir():
        return None
    # python-chess opens each table with mmap.ACCESS_READ; max_fds bounds the
    # number of mappings, not just descriptors. No read_bytes/table copies.
    try:
        with chess.syzygy.open_tablebase(str(directory), max_fds=4) as tables:
            candidates = []
            for move in board.legal_moves:
                child = board.copy()
                zeroing = board.is_zeroing(move)
                child.push(move)
                if child.is_checkmate():
                    return move.uci(), 29999
                if child.is_game_over(claim_draw=True):
                    wdl, dtz = 0, 0
                else:
                    wdl, dtz = -tables.probe_wdl(child), -tables.probe_dtz(child)
                    distance = 1 if zeroing else abs(dtz) + 1
                    # Respect the fifty-move rule. Cursed wins/blessed losses
                    # are treated as draws, never as a forced conversion.
                    if abs(wdl) == 1 or (abs(wdl) == 2 and child.halfmove_clock + abs(dtz) >= 100):
                        wdl = 0
                    dtz = distance
                order = -abs(dtz) if wdl > 0 else abs(dtz) if wdl < 0 else 0
                candidates.append((wdl, order, move.uci()))
            if not candidates:
                return None
            wdl, _distance, uci = max(candidates)
            return uci, wdl * 10000
    except (KeyError, OSError, ValueError):
        # Incomplete table sets are common. Never mix partial root probes.
        return None


def unpack_pgn(archive: Path, target: Path):
    """Copy one bounded member; never extract archive paths onto the filesystem."""
    temporary = target.with_suffix(".unpacked")
    try:
        with zipfile.ZipFile(archive) as package:
            members = package.infolist()
            if len(members) > 4:
                raise ValueError("Unexpected PGN archive")
            games = [m for m in members if not m.is_dir() and m.filename.lower().endswith(".pgn")]
            if len(games) != 1 or games[0].file_size > 32 * 1024**2:
                raise ValueError("Archive must contain one PGN of at most 32 MiB")
            size = 0
            with package.open(games[0]) as source, temporary.open("wb") as output:
                for chunk in iter(lambda: source.read(65536), b""):
                    size += len(chunk)
                    if size > 32 * 1024**2:
                        raise ValueError("Inflated PGN exceeds limit")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


async def download_pgn(url: str, directory: Path) -> Path:
    """Approved public PGN providers; bounded ZIP inflation, no redirects."""
    import asyncio

    import aiohttp
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in {"www.pgnmentor.com", "theweekinchess.com"}
            or parts.port not in (None, 443) or parts.username or parts.password
            or not parts.path.endswith((".pgn", ".zip")) or parts.query or parts.fragment):
        raise ValueError("Use an HTTPS PGN or ZIP from an approved public provider")
    directory = Path(directory)
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(url.encode()).hexdigest() + ".pgn")
    temporary = target.with_suffix(".download")
    size = 0
    try:
        async with aiohttp.ClientSession(trust_env=False, auto_decompress=False) as session:  # noqa: SIM117
            async with session.get(url, allow_redirects=False, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status != 200:
                    raise RuntimeError(f"PGN download returned HTTP {response.status}")
                handle = await asyncio.to_thread(temporary.open, "wb")
                try:
                    async for chunk in response.content.iter_chunked(65536):
                        size += len(chunk)
                        if size > 32 * 1024**2:
                            raise ValueError("PGN download exceeds 32 MiB")
                        await asyncio.to_thread(handle.write, chunk)
                finally:
                    await asyncio.to_thread(handle.close)
        if parts.path.endswith(".zip"):
            await asyncio.to_thread(unpack_pgn, temporary, target)
        else:
            await asyncio.to_thread(os.replace, temporary, target)
        return target
    finally:
        await asyncio.to_thread(temporary.unlink, missing_ok=True)
