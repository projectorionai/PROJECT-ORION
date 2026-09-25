"""
ChessService — chess against ORION's own engine (chess_brain.py) or, on
request, Stockfish (auto-installed if missing, with a built-in heuristic
fallback if it truly cannot be obtained); coaching in plain chess language;
autonomous ORION/engine games; and move-history review.

Four capabilities, one shared session state:

    play         — ``new_game``/``make_move``/``resign``/``set_elo``. A game
                   has exactly one human side (``player_colour``) unless
                   started via the newer ``orion_side``/``white``/``black``
                   convenience, which allows ORION or the engine to hold
                   either or both sides (needed for autonomous/training
                   games). A move is refused when it is not the user's turn
                   or while ORION is still thinking — his search runs on a
                   snapshot of the position on a worker thread, and the live
                   board is only ever changed on the event loop.

    coaching     — every move the user plays is REVIEWED in the background
                   once ORION has replied, so the reply is never held up by
                   it: ORION's own search finds the best move in the position
                   and scores the played move at the same depth
                   (``ChessBrain.review_move``), and ``chess_theory`` turns
                   that into words — the grade (best/excellent/good/
                   inaccuracy/mistake/blunder, from how much of the player's
                   winning chances the move gave away), what the move allowed
                   (the refutation, the fork, the piece left hanging, the
                   weakened king), and which move was better and why. No
                   centipawn figure is ever said to the player. With
                   ``analyst='stockfish'`` Stockfish supplies the numbers
                   instead; the words come from the board either way.

    autonomous   — ``start_autonomous`` launches an ``asyncio.create_task``
                   loop that alternates moves for whichever sides are *not*
                   human-controlled — "play against the engine" (orion vs
                   engine) or "train against yourself" (orion vs orion) —
                   until the game ends or the user cancels. Never blocks the
                   Qt event loop: all engine I/O runs via ``asyncio.to_thread``.

    review       — every ply is recorded with its FEN, so the GUI (arrow
                   keys, the move list) or the dispatcher (a spoken "go back
                   three moves", "show move 12", "flip the board") can browse
                   history via ``goto_ply``/``step``/``goto_move`` without
                   ever mutating the live ``chess.Board``. ``view_ply`` is a
                   read-only cursor into ``history``; ``is_live`` reports
                   whether it sits at the current position, and moves are
                   refused unless it does.

Works with or without a real UCI engine binary: python-chess itself is
required (move generation/validation) but Stockfish is not.
"""

from __future__ import annotations

import asyncio
import io
import math
import random
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import chess
    import chess.engine
    CHESS_AVAILABLE = True
except Exception:  # pragma: no cover - python-chess is a hard requirement in practice
    CHESS_AVAILABLE = False
    chess = None  # type: ignore

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line

# Standard centipawn piece values for the fallback evaluator (king excluded —
# checkmate/stalemate are handled as special cases, not material).
_PIECE_VALUES: dict[int, int] = {}
if CHESS_AVAILABLE:
    _PIECE_VALUES = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
    }

# Actors that can control a side. "user" never moves on its own — the
# autonomous loop refuses to start unless both sides are drawn from
# _AUTO_ACTORS, and 'move' only auto-replies when the side now to move is one
# of them (a single ply — continuous play is the autonomous loop's job).
_ACTORS = {"user", "orion", "engine"}
_AUTO_ACTORS = {"orion", "engine"}

MATE_CP = 100_000  # large-but-finite stand-in for a forced mate score

#: ORION's thinking time for a move, interactive or autonomous. The autonomous
#: loop used to give him 0.3 s (and ignore the depth it was handed), so the
#: games he "trained" in were played by a much weaker version of himself.
ORION_SECONDS = 1.5
#: Time for each of the two searches a coaching review makes. Runs after
#: ORION has replied, so it costs the player nothing.
REVIEW_SECONDS = 0.8
#: A review waits this long before starting: the GUI repaints first, and a
#: short-lived caller (a test's asyncio.run) is gone before a thread starts.
REVIEW_SETTLE_SECONDS = 0.3

# ── Stockfish binary: locate, or auto-install on Windows ───────────────────
#
# Chess should work the first time a user asks for it. Rather than requiring
# a manual "install Stockfish and put it on PATH" step, ORION downloads the
# official Windows build once and caches it under BASE_DIR/engines — the same
# "detect, don't assume, then fetch if we can" approach DynamicPackageResolver
# takes with pip packages. If the binary can't be found OR installed (wrong
# platform, no network, download blocked), the heuristic evaluator keeps
# chess playable, just with shallower judgement.
ENGINE_DIR = BASE_DIR / "engines"
ENGINE_PATH = ENGINE_DIR / ("stockfish.exe" if sys.platform == "win32" else "stockfish")
_STOCKFISH_ZIP_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/latest/download/"
    "stockfish-windows-x86-64-avx2.zip"
)


def _extract_binary(zip_bytes: bytes) -> Path:
    """Pull the first ``*.exe`` out of a downloaded Stockfish release zip and
    place it at ``ENGINE_PATH``. Raises if the archive has no executable —
    a corrupt/unexpected download must never be silently accepted."""
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        exe_name = next((n for n in archive.namelist() if n.lower().endswith(".exe")), None)
        if exe_name is None:
            raise RuntimeError("Stockfish archive did not contain an .exe")
        ENGINE_PATH.write_bytes(archive.read(exe_name))
    return ENGINE_PATH


async def install_stockfish(bus: OrionBus | None = None) -> Optional[Path]:
    """Fetch and cache the Stockfish binary if it isn't already present.
    Windows-only (the pinned release asset is a Windows build); other
    platforms are expected to have Stockfish available via their package
    manager, so this reports cleanly rather than guessing at their layout."""
    if ENGINE_PATH.is_file():
        return ENGINE_PATH
    if sys.platform != "win32":
        if bus is not None:
            bus.log.emit(
                "CHESS: no bundled Stockfish for this platform — install it via your "
                "package manager (or set ORION_STOCKFISH_PATH) for full engine strength."
            )
        return None
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(_STOCKFISH_ZIP_URL, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp.raise_for_status()
                zip_bytes = await resp.read()
        path = await asyncio.to_thread(_extract_binary, zip_bytes)
        if bus is not None:
            bus.log.emit(f"CHESS: Stockfish installed to {path}.")
        return path
    except Exception as exc:
        if bus is not None:
            bus.log.emit(
                f"CHESS: could not auto-install Stockfish ({first_line(exc, 100)}); "
                "falling back to the built-in heuristic evaluator."
            )
        return None


def _mate_moves(score: int) -> int:
    """Moves to mate for a mate score (either side's view), signed like it."""
    from .chess_brain import MATE_SCORE
    plies = MATE_SCORE - abs(int(score))
    moves = (plies + 1) // 2
    return moves if score > 0 else -moves


def _white_view(score: int, white_to_move: bool) -> tuple[int, Optional[int]]:
    """A search score from the side to move's view, as (centipawns, mate in N)
    from White's — the shape the evaluation bar wants. A forced mate is
    reported as a mate with its distance, never as a very large number."""
    from .chess_brain import MATE_BOUND
    white = int(score) if white_to_move else -int(score)
    if abs(white) > MATE_BOUND:
        mate = _mate_moves(white)
        return (MATE_CP if white > 0 else -MATE_CP), (mate or None)
    return white, None


@dataclass
class ChessMoveRecord:
    """One played ply: enough to render history, review any position, and
    explain the move in words without re-running the engine."""

    ply: int
    san: str
    uci: str
    mover: str            # "white" or "black"
    actor: str            # "user" | "orion" | "engine"
    fen_before: str
    fen_after: str
    number: int = 1       # the move number printed before it ("14." / "14...")
    quality: Optional[str] = None     # best/excellent/good/inaccuracy/mistake/blunder
    comment: str = ""                 # plain-language coaching or ORION's reason
    better: Optional[str] = None      # the stronger move, in SAN, when there was one
    drop: float = 0.0                 # winning chance given away (0..1) — internal
    reviewed: bool = False

    @property
    def label(self) -> str:
        """'14. Nf3' or '14... Nf6'."""
        return f"{self.number}{'.' if self.mover == 'white' else '...'} {self.san}"

    def line(self) -> str:
        tag = f" [{self.quality}]" if self.quality else ""
        return f"{self.label} ({self.actor}){tag}"


@dataclass
class AutonomousRun:
    """Bookkeeping for a background orion/engine training game."""

    white: str
    black: str
    delay_seconds: float
    task: "asyncio.Task[None]"
    started: float = field(default_factory=time.monotonic)
    moves_played: int = 0


class ChessService:
    """Owns the live game, the lazily-launched engine, move history and any
    running autonomous task. One instance is shared by the dispatcher tool
    and the GUI board view, exactly as other Mark X services (research,
    protocols, reminders) are shared singletons hung off ``OrionDispatcher``."""

    MIN_ELO = 400
    MAX_ELO = 3000
    DEFAULT_ELO = 1500

    def __init__(self, bus: OrionBus, router: Any | None = None,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.router = router
        self.telemetry = telemetry
        self.board: "chess.Board" = chess.Board() if CHESS_AVAILABLE else None  # type: ignore
        self.start_fen = self.board.fen() if self.board is not None else ""
        self.history: list[ChessMoveRecord] = []
        self.view_ply = 0
        #: None = orient the board automatically (the user's colour at the
        #: bottom); True/False once the user has flipped it themselves.
        self.view_flipped: Optional[bool] = None
        self.sides: dict[str, str] = {"white": "user", "black": "orion"}
        #: Who the user plays when they just say "play chess with me". ORION
        #: himself, using chess_brain.py — Stockfish is opt-in, and is kept for
        #: analysis (grading moves after the fact) and as a sparring partner.
        self.opponent = "orion"
        #: Who grades moves and drives the evaluation bar. ORION himself by
        #: default: Stockfish used to do this on every move of every game,
        #: which is why it was launched (and auto-installed) even when the
        #: user never asked for it. Now only "analyse with stockfish" does.
        self.analyst = "orion"
        self._practice: Any = None
        self._practice_stop: Any = None
        self.elo = self.DEFAULT_ELO
        self._active = False
        #: True while ORION (or Stockfish) is choosing a move. Moves are
        #: refused meanwhile: the old path let the board be played on while a
        #: worker thread was pushing and popping moves on that same board.
        self._thinking = False
        self._busy = False
        #: Evaluations of positions already seen, White's view: board key ->
        #: (cp, mate). Filled for free by ORION's own searches and by reviews.
        self._evals: dict[str, tuple[int, Optional[int]]] = {}
        self._review_queue: list[ChessMoveRecord] = []
        self._review_task: Any = None
        self._autonomous: Optional[AutonomousRun] = None
        # The raw UCI engine handle — lazily created by _ensure_engine so a
        # game can start instantly (no install dialogue up front) and the
        # heuristic evaluator quietly stands in until it's ready or if it
        # can never be installed.
        self._engine: Optional["chess.engine.SimpleEngine"] = None
        self.backend = "not started"

    # ── availability ─────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return self.view_ply == len(self.history)

    @property
    def thinking(self) -> bool:
        return self._thinking

    def _unavailable(self) -> Optional[ToolResult]:
        if not CHESS_AVAILABLE:
            return ToolResult(
                "The chess subsystem needs the 'chess' package (python-chess), which "
                "is not installed. Run: pip install chess", ok=False,
            )
        return None

    def _emit(self, payload: dict[str, Any]) -> None:
        try:
            self.bus.dashboard_event.emit("chess", payload)
        except Exception:
            pass

    # ── engine lifecycle (lazy, auto-installing) ────────────────────────────

    async def _ensure_engine(self) -> Optional["chess.engine.SimpleEngine"]:
        """Return the live UCI engine, launching (and installing, if needed)
        it on first use. Returns None — never raises — when no engine could
        be obtained, so callers fall back to the heuristic evaluator."""
        if self._engine is not None:
            return self._engine
        path = ENGINE_PATH if ENGINE_PATH.is_file() else await install_stockfish(self.bus)
        if not path or not Path(path).is_file():
            self.backend = "heuristic"
            return None
        try:
            self._engine = await asyncio.to_thread(chess.engine.SimpleEngine.popen_uci, str(path))
            self.backend = "stockfish"
            self.bus.log.emit(f"CHESS: UCI engine loaded ({Path(path).name}).")
            await self._apply_elo(self.elo)
        except Exception as exc:
            self.backend = "heuristic"
            self.bus.log.emit(
                f"CHESS: found an engine binary but it would not launch "
                f"({first_line(exc, 120)}); using the built-in heuristic evaluator."
            )
            self._engine = None
        return self._engine

    @staticmethod
    def _elo_options(elo: int) -> dict[str, Any]:
        """Map a target ELO onto Stockfish UCI options. Below 1320, Stockfish's
        own strength-limiting floor, use Skill Level (0-20); at or above it,
        UCI_LimitStrength + UCI_Elo gives a much finer-grained target."""
        if elo < 1320:
            skill = max(0, min(20, round((elo - 1000) / 16)))
            return {"UCI_LimitStrength": False, "Skill Level": skill}
        return {"UCI_LimitStrength": True, "UCI_Elo": int(elo)}

    async def _apply_elo(self, elo: int) -> None:
        if self._engine is None:
            return
        try:
            await asyncio.to_thread(self._engine.configure, self._elo_options(elo))
        except Exception as exc:
            self.bus.log.emit(f"CHESS: could not apply ELO {elo} ({first_line(exc, 100)}).")

    async def set_elo(self, elo: int) -> ToolResult:
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        self.elo = max(self.MIN_ELO, min(self.MAX_ELO, int(elo)))
        if self.opponent != "engine":
            # His own number is a rating, not a dial: it only moves by playing.
            return ToolResult(
                f"My rating is earned, not set — I'm {self.brain.rating_text()}. "
                f"I've noted {self.elo} as Stockfish's strength for when you want "
                "to play it instead.")
        await self._ensure_engine()
        await self._apply_elo(self.elo)
        return ToolResult(f"Stockfish's strength set to ≈{self.elo} ELO.")

    # ── evaluation (the eval bar) ────────────────────────────────────────────

    @staticmethod
    def _key(board: "chess.Board") -> str:
        return " ".join(board.fen().split(" ")[:4])

    def _remember_eval(self, board: "chess.Board", cp: int, mate: Optional[int]) -> None:
        if len(self._evals) > 512:
            self._evals.clear()
        self._evals[self._key(board)] = (int(cp), mate)

    async def _evaluate(self, board: "chess.Board") -> tuple[int, Optional[int]]:
        """(centipawns, mate in N) from White's point of view. ORION's own
        search does this unless the user chose Stockfish as the analyst —
        Stockfish is never started just to fill the bar."""
        if board.is_checkmate():
            return (-MATE_CP if board.turn == chess.WHITE else MATE_CP), None
        if board.is_game_over(claim_draw=True):
            return 0, None
        if self.analyst != "stockfish":
            try:
                from .chess_brain import MATE_BOUND
                score = await asyncio.to_thread(self.brain.analyse, board.copy(), 0.2)
                if abs(score) > MATE_BOUND:
                    return (MATE_CP if score > 0 else -MATE_CP), (_mate_moves(score) or None)
                return int(score), None
            except Exception as exc:
                self.bus.log.emit(f"CHESS: my own analysis failed ({first_line(exc, 100)}); "
                                  "using the quick heuristic for this position.")
                return self._heuristic_eval(board), None
        engine = await self._ensure_engine()
        if engine is not None:
            try:
                def _run() -> tuple[int, Optional[int]]:
                    info = engine.analyse(board, chess.engine.Limit(depth=12, time=0.4))
                    white = info["score"].white()
                    try:
                        mate = white.mate()
                    except Exception:
                        mate = None
                    return int(white.score(mate_score=MATE_CP)), (mate or None)
                return await asyncio.to_thread(_run)
            except Exception as exc:
                self.bus.log.emit(
                    f"CHESS: engine analysis failed ({first_line(exc, 100)}); "
                    "using the heuristic evaluator for this position."
                )
        return self._heuristic_eval(board), None

    async def _evaluate_cp(self, board: "chess.Board") -> int:
        """Centipawns only, White's view (mates saturate at ±MATE_CP)."""
        cp, _mate = await self._evaluate(board)
        return cp

    @staticmethod
    def _win_probability(cp: Optional[int], mate: Optional[int]) -> float:
        """White's win fraction (0..1) for a display eval bar. A forced mate
        saturates near the edge rather than going all the way to 0/1, so the
        bar still reads as "decisive" without looking broken."""
        if mate is not None:
            return 0.98 if mate > 0 else 0.02
        if cp is None:
            return 0.5
        return 1.0 / (1.0 + math.exp(-cp / 400.0))

    async def evaluate_current(self) -> dict[str, Any]:
        """The live position's evaluation from White's perspective, for a
        GUI eval bar. Neutral when the chess package is unavailable.

        While ORION is thinking nothing new is computed — his search is about
        to produce this very number for free, and a second search would only
        slow his down."""
        if not CHESS_AVAILABLE:
            return {"cp": None, "mate": None, "white_fraction": 0.5}
        cached = self._evals.get(self._key(self.board))
        if cached is None and self._thinking:
            cached = getattr(self, "_last_shown_eval", None)
        if cached is None:
            cached = await self._evaluate(self.board)
            self._remember_eval(self.board, *cached)
        self._last_shown_eval = cached
        cp, mate = cached
        return {"cp": cp, "mate": mate, "white_fraction": self._win_probability(cp, mate)}

    # ── ORION's own engine ───────────────────────────────────────────────────

    @property
    def brain(self) -> Any:
        """ORION's OWN chess brain, built on first use.

        Deliberately lazy: it loads a learned book and evaluation tables from
        disk, and nothing should pay for that unless a game is actually played.
        """
        existing = getattr(self, "_brain", None)
        if existing is None:
            from .chess_brain import ChessBrain
            existing = ChessBrain()
            # Give him centuries of opening theory as a head start, so he plays
            # principled book lines from move one instead of relearning the Ruy
            # Lopez over hundreds of games. Additive and idempotent — his own
            # results still count and are never overwritten.
            try:
                from . import chess_theory
                added = chess_theory.seed_into(existing)
                if added:
                    existing.save()
                    self.bus.log.emit(
                        f"CHESS: seeded {added} book positions of opening theory "
                        f"({len(chess_theory.REPERTOIRE)} openings) into ORION's brain.")
            except Exception as exc:
                self.bus.log.emit(f"CHESS: opening theory not seeded - {first_line(exc, 80)}")
            self._brain = existing
        return existing

    async def _orion_move(self, board: "chess.Board", seconds: float = ORION_SECONDS,
                          max_depth: Optional[int] = None) -> Optional["chess.Move"]:
        """ORION's move, from ORION's own engine — never Stockfish.

        "I want him to be good at chess by himself and he must be able to
        dynamically learn from games." So his side of the board is played by
        chess_brain.py: his own search, his own evaluation, and an opening book
        he writes from his own results. Stockfish remains available ONLY as an
        opponent to practise against and as the analysis engine for grading
        moves after the fact — it never chooses a move for him.

        He searches a SNAPSHOT of the position on a worker thread. The old
        path handed the live board itself to the thread, which pushed and
        popped moves on it while the GUI and make_move could still use it.
        """
        from .chess_brain import MAX_PLY
        brain = self.brain
        snapshot = board.copy()
        depth = int(max_depth) if max_depth else MAX_PLY

        def _search() -> tuple[Optional["chess.Move"], dict[str, Any]]:
            # Hold the brain's lock across the search AND the read-back, so a
            # review cannot slip in between and overwrite the numbers.
            with brain._search_lock:
                move = brain.choose_move(snapshot, seconds, depth)
                return move, {"score": brain._last_score, "pv": list(brain.last_pv),
                              "reason": brain.last_reason}

        move, info = await asyncio.to_thread(_search)
        if move is not None:
            self._last_orion_info = info
            self.bus.log.emit(f"CHESS: ORION plays {move.uci()} — {info['reason']}")
            if "book" not in info["reason"]:
                after = board.copy(stack=False)
                after.push(move)
                cp, mate = _white_view(info["score"], board.turn == chess.WHITE)
                if mate is not None:
                    # Mate in N counted from before his move; after it, N-1.
                    mate = mate - 1 if mate > 0 else mate + 1
                    mate = mate or None
                self._remember_eval(after, cp, mate)
        return move

    def set_opponent(self, who: str) -> ToolResult:
        """Choose who the user plays: ORION himself, or Stockfish.

        This exists because the user asked ORION to use his own engine and was
        told he did not understand — there was no action for it anywhere in the
        tool surface, so the request had nowhere to land. It switches the
        current game too, not just the next one, since "use your own engine"
        said mid-game plainly means now.
        """
        choice = str(who or "").strip().lower()
        aliases = {
            "orion": "orion", "you": "orion", "yourself": "orion",
            "your own": "orion", "your own engine": "orion", "own": "orion",
            "brain": "orion", "self": "orion", "orion's": "orion",
            "engine": "engine", "stockfish": "engine", "uci": "engine",
            "computer": "engine",
        }
        actor = aliases.get(choice)
        if actor is None:
            return ToolResult(
                f"I can play you myself, or set Stockfish against you — "
                f"'{who}' is neither.", ok=False)
        self.opponent = actor
        # Swap whoever is currently the non-human side in the live game.
        swapped = False
        for colour, holder in list(self.sides.items()):
            if holder in _AUTO_ACTORS and holder != actor:
                self.sides[colour] = actor
                swapped = True
        if actor == "orion":
            text = ("You're playing me now — my own engine, my own evaluation, "
                    "and an opening book I write from my own results. "
                    f"{self.brain.describe()}")
        else:
            text = (f"Stockfish has the board now, about {self.elo} ELO. "
                    "Say 'play me yourself' to switch back.")
        if swapped:
            text += " Switched for the game in progress as well."
            self._emit({"phase": "sides", "white": self.sides["white"],
                        "black": self.sides["black"]})
        self.bus.log.emit(f"CHESS: opponent set to {actor}.")
        return ToolResult(text)

    def learn_from_finished_game(self) -> str:
        """Feed the completed game back into ORION's brain."""
        try:
            result = self.board.result(claim_draw=True)
            moves = [record.uci for record in self.history]
        except Exception:
            return "No finished game to learn from."
        if result == "*" or not moves:
            return "The game is not over yet."
        summary = self.brain.learn_from_game(moves, result)
        self.bus.log.emit(f"CHESS: {summary}")
        return summary

    async def _best_move(self, board: "chess.Board", depth: int | None = None,
                         movetime_ms: int | None = None) -> Optional["chess.Move"]:
        """Stockfish's move for the side to move (heuristic if unavailable)."""
        engine = await self._ensure_engine()
        if engine is not None:
            try:
                snapshot = board.copy()

                def _run() -> Optional["chess.Move"]:
                    limit = chess.engine.Limit(
                        depth=depth or 12, time=(movetime_ms or 400) / 1000.0)
                    result = engine.play(snapshot, limit)
                    return result.move
                move = await asyncio.to_thread(_run)
                if move is not None:
                    return move
            except Exception as exc:
                self.bus.log.emit(
                    f"CHESS: engine search failed ({first_line(exc, 100)}); "
                    "using the heuristic move choice for this ply."
                )
        return self._heuristic_best_move(board)

    def _heuristic_eval(self, board: "chess.Board") -> int:
        if board.is_checkmate():
            return -MATE_CP if board.turn == chess.WHITE else MATE_CP
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        score = 0
        for piece_type, value in _PIECE_VALUES.items():
            score += value * (
                len(board.pieces(piece_type, chess.WHITE))
                - len(board.pieces(piece_type, chess.BLACK))
            )
        mobility = board.legal_moves.count()
        score += (2 * mobility) if board.turn == chess.WHITE else (-2 * mobility)
        return score

    def _heuristic_best_move(self, board: "chess.Board") -> Optional["chess.Move"]:
        legal = list(board.legal_moves)
        if not legal:
            return None
        maximise = board.turn == chess.WHITE
        scored: list[tuple[int, "chess.Move"]] = []
        for move in legal:
            trial = board.copy(stack=False)
            trial.push(move)
            scored.append((self._heuristic_eval(trial), move))
        scored.sort(key=lambda pair: pair[0], reverse=maximise)
        best_score = scored[0][0]
        top = [mv for cp, mv in scored if abs(cp - best_score) <= 15]
        return random.choice(top)

    # ── game lifecycle ───────────────────────────────────────────────────────

    async def new_game(self, white: str | None = None, black: str | None = None,
                       orion_side: str | None = None, starting_fen: str | None = None,
                       elo: int | None = None, player_colour: str | None = None) -> ToolResult:
        """Start a fresh game.

        Two ways to pick who plays which side:
          * ``player_colour`` ('white'/'black') — the simple case: exactly
            one human side, the current opponent (ORION unless the user chose
            Stockfish) plays the other.
          * ``orion_side`` ('white'/'black'/'both'/'none') or explicit
            ``white``/``black`` actor names — for autonomous/training games
            where ORION and/or the raw engine hold either side.
        """
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        if orion_side:
            side = orion_side.strip().lower()
            mapping = {
                "white": ("orion", "engine"), "black": ("engine", "orion"),
                "both": ("orion", "orion"), "none": ("user", "engine"),
            }
            if side not in mapping:
                return ToolResult(
                    f"orion_side must be white, black, both, or none (got '{orion_side}').",
                    ok=False,
                )
            white, black = mapping[side]
        elif player_colour is not None and white is None and black is None:
            pc = player_colour.strip().lower()
            # THE OPPONENT IS ORION, not Stockfish. This mapping is why the
            # user kept reporting "he still uses the stockfish engine": every
            # ordinary "play chess with me" arrives on this path, and it named
            # "engine" — the raw UCI process — as the other side. The whole
            # point was "I want him to be good at chess by himself", so playing
            # ORION is the default and Stockfish is opt-in (opponent='engine').
            opponent = (self.opponent or "orion")
            if pc == "white":
                white, black = "user", opponent
            elif pc == "black":
                white, black = opponent, "user"
            else:
                return ToolResult(
                    f"player_colour must be 'white' or 'black' (got '{player_colour}').",
                    ok=False,
                )
        white = (white or "user").strip().lower()
        # The same default as player_colour: the user's opponent, not the UCI
        # engine. A bare new_game() (the GUI's button) used to start a
        # Stockfish game whatever the user had chosen.
        black = (black or self.opponent or "orion").strip().lower()
        if white not in _ACTORS or black not in _ACTORS:
            return ToolResult(f"Sides must each be one of {sorted(_ACTORS)}.", ok=False)
        self._cancel_autonomous_if_running()
        try:
            self.board = chess.Board(starting_fen) if starting_fen else chess.Board()
        except ValueError as exc:
            return ToolResult(f"Invalid starting position: {exc}", ok=False)
        self.start_fen = self.board.fen()
        self.history = []
        self.view_ply = 0
        self.view_flipped = None
        self.sides = {"white": white, "black": black}
        self._evals.clear()
        self._review_queue.clear()
        self._learned_signature = None
        self._active = True
        if elo is not None:
            self.elo = max(self.MIN_ELO, min(self.MAX_ELO, int(elo)))
            await self._apply_elo(self.elo)
        if self.analyst == "stockfish" or "engine" in (white, black):
            # Start it now, not on the first move: the backend is then known
            # (and reported) from the outset, and the first reply is not
            # held up by a process launch.
            await self._ensure_engine()
        self.bus.log.emit(f"CHESS: new game — white={white}, black={black}.")
        self._emit({"phase": "new_game", "white": white, "black": black,
                    "fen": self.start_fen})
        text = f"New game. White is {white}, Black is {black}."
        # White may be non-human (engine reply straight away, or ORION's own
        # opening move) — a single reactive ply, mirroring make_move's rule.
        replies = await self._maybe_auto_reply()
        if replies:
            opener = "Stockfish" if white == "engine" else white.capitalize()
            text += f"\n{opener} plays: " + "\n".join(r.comment for r in replies if r.comment)
        elif white == "user":
            text += " Your move."
        return ToolResult(text)

    async def resign(self) -> ToolResult:
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        if not self._active:
            return ToolResult("There's no active game to resign.", ok=False)
        self._active = False
        self._cancel_autonomous_if_running()
        self.bus.log.emit("CHESS: game resigned.")
        # A resignation is a result like any other: the user resigns, so the
        # user's colour loses.
        try:
            resigner = self._player_colour()
            self._rate_game("black" if resigner == "white" else "white")
        except Exception:
            pass
        self._emit({"phase": "resign"})
        return ToolResult("Resigned. GG.")

    # ── moves ─────────────────────────────────────────────────────────────────

    async def make_move(self, move_text: str, actor: str = "user") -> ToolResult:
        """Validate and play one of the user's moves; if the side to move next
        is ORION's or Stockfish's, it replies immediately (a single ply —
        continuous, unattended play is ``start_autonomous``'s job, so the two
        never race). The coaching review of the move follows in the
        background and never delays the reply."""
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        if not self._active:
            return ToolResult("There's no active game — start one first.", ok=False)
        if not self.is_live:
            return ToolResult(
                "You're browsing an earlier position — say 'live' (or jump to the "
                "latest move) before playing a new one.", ok=False,
            )
        if self.board.is_game_over():
            return ToolResult(f"The game is already over: {self._result_text()}.", ok=False)
        if self._thinking or self._busy:
            return ToolResult("One moment — I'm still thinking about my move.", ok=False)
        to_move = self._colour_name(self.board.turn)
        if actor == "user" and self.sides.get(to_move) != "user":
            return ToolResult(f"It isn't your move — {self._to_move_text()}", ok=False)
        move_text = SecuritySanitiser.guard_text(str(move_text or "").strip(), "chess.move")
        if not move_text:
            return ToolResult("What move would you like to play?", ok=False)
        try:
            move = self._parse_move(self.board, move_text)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        self._busy = True
        try:
            record = self._push_record(move, actor)
            replies = await self._maybe_auto_reply()
        finally:
            self._busy = False
        if record.actor == "user":
            self._queue_review(record)
        lines = [f"You played {record.san}."] if record.actor == "user" else []
        for reply in replies:
            who = "I play" if reply.actor == "orion" else "Stockfish plays"
            lines.append(f"{who} {reply.comment or reply.label}")
        text = "\n".join(lines)
        if self.board.is_game_over():
            text += f"\n\n{self._game_over_line()}"
        return ToolResult(text or "Move played.", ok=True)

    def _game_over_line(self) -> str:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None and outcome.termination is chess.Termination.CHECKMATE:
            return f"Checkmate! {self._result_text()}."
        return f"Game over: {self._result_text()}."

    async def _maybe_auto_reply(self) -> list[ChessMoveRecord]:
        """A single reactive engine/ORION reply when the side now on the
        move isn't human — never a loop (that's start_autonomous's job), so
        this can never chain through a "user" side or race the autonomous
        task."""
        if self.board.is_game_over():
            return []
        auto_actor = self.sides.get(self._colour_name(self.board.turn))
        if auto_actor not in _AUTO_ACTORS:
            return []
        record = await self._auto_move(auto_actor)
        return [record] if record is not None else []

    async def _auto_move(self, actor: str, seconds: float = ORION_SECONDS,
                         max_depth: Optional[int] = None) -> Optional[ChessMoveRecord]:
        """Choose and play one move for ORION or Stockfish, marked as thinking
        throughout so the user cannot play into the middle of it."""
        before = self.board.copy(stack=False)
        self._thinking = True
        if actor == "orion" and self._brain_loaded():
            # A background review may hold his brain; his move comes first.
            # The review notices, gives up, and is re-run afterwards.
            self.brain.interrupt()
        self._emit({"phase": "thinking", "actor": actor})
        try:
            if actor == "orion":
                move = await self._orion_move(self.board, seconds, max_depth)
            else:
                move = await self._best_move(self.board)
        finally:
            self._thinking = False
        if move is None or move not in self.board.legal_moves:
            self._emit({"phase": "idle"})
            return None
        comment = before.san(move)
        if actor == "orion":
            info = getattr(self, "_last_orion_info", None) or {}
            if "book" in str(info.get("reason", "")):
                comment = f"{comment} — straight from my opening book."
            else:
                try:
                    from . import chess_theory
                    comment = chess_theory.describe_intent(
                        before, move, list(info.get("pv") or [move]),
                        int(info.get("score") or 0), mover="I",
                        opponent=self._opponent_of(self._colour_name(before.turn)))
                except Exception:
                    comment = f"{comment}."
        else:
            comment = f"{comment}."
        return self._push_record(move, actor, comment)

    def _maybe_learn(self) -> None:
        """If that move ended the game, study it — once."""
        try:
            if not self.board.is_game_over(claim_draw=True):
                return
            signature = (len(self.history), self.board.result(claim_draw=True))
            if getattr(self, "_learned_signature", None) == signature:
                return
            self._learned_signature = signature
            self.learn_from_finished_game()
            result = self.board.result(claim_draw=True)
            winner = {"1-0": "white", "0-1": "black"}.get(result)
            self._rate_game(winner)
        except Exception:
            pass

    def _rate_game(self, winner: str | None) -> None:
        """Move ORION's Elo after a game he played against a real opponent.
        *winner* is "white", "black" or None for a draw. Self-play (both
        sides his) and games he did not play in are not rated."""
        mine = [colour for colour, holder in self.sides.items() if holder == "orion"]
        if len(mine) != 1:
            return
        other = self.sides["black" if mine[0] == "white" else "white"]
        if other not in ("user", "engine"):
            return
        score = 0.5 if winner is None else (1.0 if winner == mine[0] else 0.0)
        opponent = "user" if other == "user" else "stockfish"
        line = self.brain.record_result(score, opponent,
                                        None if other == "user" else float(self.elo))
        self.bus.log.emit(f"CHESS: {line}")

    def _push_record(self, move: "chess.Move", actor: str,
                     comment: str = "") -> ChessMoveRecord:
        """Play *move* on the live board and record it. Instant: nothing here
        searches — grading happens later, in the background."""
        board = self.board
        record = ChessMoveRecord(
            ply=len(self.history) + 1, san=board.san(move), uci=move.uci(),
            mover=self._colour_name(board.turn), actor=actor,
            fen_before=board.fen(), fen_after="", number=board.fullmove_number,
            comment=comment,
        )
        board.push(move)
        record.fen_after = board.fen()
        self.history.append(record)
        self.view_ply = len(self.history)
        if board.is_game_over():
            self._active = False
        self.bus.log.emit(f"CHESS: {record.line()}")
        self._emit({"phase": "move", "ply": record.ply, "san": record.san,
                    "actor": record.actor, "fen": record.fen_after,
                    "comment": record.comment})
        # A finished game is the only training signal ORION gets, so it is taken
        # here — the one place every move passes through, whoever played it.
        self._maybe_learn()
        return record

    async def _push_and_grade(self, move: "chess.Move", actor: str) -> ChessMoveRecord:
        """Play *move* and queue its coaching review (kept for callers that
        push a move directly; the review is never awaited here)."""
        record = self._push_record(move, actor)
        self._queue_review(record)
        return record

    def _parse_move(self, board: "chess.Board", move_text: str) -> "chess.Move":
        try:
            return board.parse_san(move_text)
        except ValueError:
            pass
        try:
            move = chess.Move.from_uci(move_text.lower().replace("-", ""))
        except ValueError:
            raise ValueError(f"'{move_text}' is not a move I recognise.")
        if move not in board.legal_moves:
            raise ValueError(f"'{move_text}' is not legal in this position.")
        return move

    # ── coaching: reviews in plain chess language ───────────────────────────

    def _speaker(self, actor: str, colour: str | None = None) -> str:
        """How a side is named in commentary: 'you', 'I', or 'Stockfish'.
        When ORION holds both sides, by colour instead."""
        if actor == "user":
            both_user = self.sides.get("white") == self.sides.get("black") == "user"
            return colour.capitalize() if (both_user and colour) else "you"
        if actor == "orion":
            if self.sides.get("white") == self.sides.get("black") == "orion" and colour:
                return colour.capitalize()
            return "I"
        return "Stockfish"

    def _opponent_of(self, colour: str) -> str:
        """How the side facing *colour* is named, in the same words."""
        other = "black" if colour == "white" else "white"
        return self._speaker(self.sides.get(other, "user"), other)

    def _queue_review(self, record: ChessMoveRecord) -> None:
        if record.reviewed or record in self._review_queue:
            return
        self._review_queue.append(record)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return                       # no loop: reviewed when next asked for
        task = self._review_task
        if task is None or task.done() or task.get_loop() is not loop:
            self._review_task = loop.create_task(self._review_worker(),
                                                 name="orion-chess-review")

    async def _review_worker(self) -> None:
        """Review queued moves one at a time, never while ORION is thinking —
        a review is a search too, and two at once would halve his."""
        try:
            await asyncio.sleep(REVIEW_SETTLE_SECONDS)
            while self._review_queue:
                while self._thinking:
                    await asyncio.sleep(0.05)
                record = self._review_queue[0]
                if not record.reviewed:
                    try:
                        await self._review(record)
                    except Exception as exc:
                        self.bus.log.emit(f"CHESS: could not review {record.label} "
                                          f"({first_line(exc, 100)}).")
                        record.reviewed = True
                if record.reviewed:
                    # By identity: analyse_move may have put another record at
                    # the front meanwhile.
                    self._review_queue = [r for r in self._review_queue if r is not record]
        except asyncio.CancelledError:
            raise

    async def _review(self, record: ChessMoveRecord) -> None:
        """Grade one move and explain it in words (see chess_theory)."""
        from . import chess_theory
        board = chess.Board(record.fen_before)
        move = chess.Move.from_uci(record.uci)
        review = None
        if self.analyst == "stockfish":
            review = await self._stockfish_review(board, move)
        if review is None:
            review = await asyncio.to_thread(self.brain.review_move, board, move,
                                             REVIEW_SECONDS)
        if review and review.get("interrupted"):
            return          # ORION needed the board: stays queued, tried again after
        if not review:
            record.reviewed = True
            return
        mover = self._speaker(record.actor, record.mover)
        opponent = self._opponent_of(record.mover)
        verdict = chess_theory.explain_move(board, move, review, mover=mover, opponent=opponent)
        record.quality = verdict["quality"]
        record.comment = verdict["text"]
        record.better = verdict["better"]
        record.drop = float(verdict["drop"])
        record.reviewed = True
        after = chess.Board(record.fen_after)
        cp, mate = _white_view(review["played_score"], board.turn == chess.WHITE)
        if mate is not None and not after.is_game_over():
            mate = mate - 1 if mate > 0 else mate + 1
            mate = mate or None
        self._remember_eval(after, cp, mate)
        self._emit({"phase": "review", "ply": record.ply, "quality": record.quality,
                    "comment": record.comment, "better": record.better})

    async def _stockfish_review(self, board: "chess.Board", move: "chess.Move"
                                ) -> Optional[dict[str, Any]]:
        """The same review with Stockfish supplying the numbers."""
        engine = await self._ensure_engine()
        if engine is None:
            return None
        from .chess_brain import MATE_SCORE

        def _stm(info: dict, turn: bool) -> int:
            white = info["score"].white()
            try:
                mate = white.mate()
            except Exception:
                mate = None
            if mate:
                winner_white = mate > 0
                plies = 2 * abs(mate) - (1 if winner_white == turn else 0)
                score_white = (MATE_SCORE - plies) if winner_white else -(MATE_SCORE - plies)
            else:
                score_white = int(white.score(mate_score=MATE_CP))
            return score_white if turn else -score_white

        def _run() -> dict[str, Any]:
            limit = chess.engine.Limit(depth=14, time=0.5)
            best_info = engine.analyse(board, limit)
            played_info = engine.analyse(board, limit, root_moves=[move])
            best_pv = list(best_info.get("pv") or [])
            played_pv = list(played_info.get("pv") or [])
            return {"best": best_pv[0] if best_pv else move,
                    "best_score": _stm(best_info, board.turn),
                    "played": move, "played_score": _stm(played_info, board.turn),
                    "best_line": best_pv, "reply_line": played_pv[1:], "depth": 14}
        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            self.bus.log.emit(f"CHESS: Stockfish review failed ({first_line(exc, 100)}); "
                              "reviewing with my own search instead.")
            return None

    async def wait_for_reviews(self, timeout: float = 20.0) -> bool:
        """Wait until every queued review is done. False on timeout."""
        deadline = time.monotonic() + timeout
        while self._review_queue and time.monotonic() < deadline:
            task = self._review_task
            if task is None or task.done():
                self._review_task = asyncio.get_running_loop().create_task(
                    self._review_worker())
            await asyncio.sleep(0.05)
        return not self._review_queue

    # ── analysis on request ───────────────────────────────────────────────────

    def _find_record(self, move_number: int | None, colour: str | None
                     ) -> Optional[ChessMoveRecord]:
        """The record for "move 12" (White's unless *colour* says otherwise, or
        the user's own when they played only one side), or — with no number —
        the user's most recent move, else the last move played."""
        if not self.history:
            return None
        user_colours = [c for c, h in self.sides.items() if h == "user"]
        if move_number is None:
            if len(user_colours) == 1:
                for record in reversed(self.history):
                    if record.mover == user_colours[0]:
                        return record
            return self.history[-1]
        wanted = (colour or "").strip().lower() or None
        if wanted is None and len(user_colours) == 1:
            wanted = user_colours[0]
        candidates = [r for r in self.history if r.number == int(move_number)]
        if wanted is not None:
            exact = [r for r in candidates if r.mover == wanted]
            if exact:
                return exact[0]
        return candidates[0] if candidates else None

    async def analyse_move(self, move_number: int | None = None,
                           colour: str | None = None) -> ToolResult:
        """Explain one move in words: its grade, what it did or allowed, and
        what was better and why. *move_number* is the chess move number."""
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        if not self.history:
            return ToolResult("No moves have been played yet.", ok=False)
        record = self._find_record(move_number, colour)
        if record is None:
            last = self.history[-1].number
            return ToolResult(f"There's no move {move_number} — the game has reached "
                              f"move {last}.", ok=False)
        if not record.reviewed:
            if record not in self._review_queue:
                self._review_queue.insert(0, record)
            await self.wait_for_reviews(timeout=15.0)
        if not record.reviewed:
            return ToolResult(f"{record.label}: still thinking about that one — ask me "
                              "again in a moment.", ok=False)
        return ToolResult(record.comment or f"{record.label}.")

    async def analyse_game(self) -> ToolResult:
        """The game in words: how each side played and the moves that decided it."""
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        if not self.history:
            return ToolResult("No moves have been played yet.", ok=False)
        from . import chess_theory
        # Every move the user played gets a verdict; in a game the user did
        # not play (ORION against Stockfish, or himself), every move does.
        humans = any(h == "user" for h in self.sides.values())
        for record in self.history:
            if not record.reviewed and (record.actor == "user" or not humans):
                self._queue_review(record)
        await self.wait_for_reviews(timeout=30.0)
        moments = [{"mover": self._speaker(r.actor, r.mover), "quality": r.quality,
                    "drop": r.drop, "text": r.comment, "ply": r.ply}
                   for r in self.history if r.reviewed and r.quality]
        result = self._result_text() if self.board.is_game_over() else "in progress"
        summary = chess_theory.summarise_game(moments, result, {})
        if not moments:
            summary = "I haven't finished looking over the moves yet — ask me again in a moment."
        return ToolResult(summary)

    # ── autonomous play ──────────────────────────────────────────────────────

    def start_autonomous(self, delay_seconds: float = 2.0,
                         orion_depth: Optional[int] = None,
                         orion_movetime_ms: int = int(ORION_SECONDS * 1000)) -> ToolResult:
        """Kick off a background loop that keeps playing both non-human
        sides (orion vs engine, or orion vs orion for 'train against
        yourself') until the game ends or ``stop_autonomous`` cancels it.
        Refuses to start while a human side is on the board — that side
        would just never move, and 'move' already handles single-ply
        reactive engine replies for the interactive game."""
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        if self.sides.get("white") not in _AUTO_ACTORS or self.sides.get("black") not in _AUTO_ACTORS:
            return ToolResult(
                "Autonomous play needs both sides set to 'orion' or 'engine' — start a "
                "new game with orion_side='white'/'black'/'both' first.", ok=False,
            )
        if self._autonomous is not None and not self._autonomous.task.done():
            return ToolResult("Autonomous play is already running.", ok=False)
        if self.board.is_game_over():
            return ToolResult(f"The game is already over: {self._result_text()}. Start a new game first.", ok=False)
        delay_seconds = max(0.2, min(30.0, float(delay_seconds)))
        task = asyncio.create_task(
            self._autonomous_run(delay_seconds, orion_depth, orion_movetime_ms)
        )
        self._autonomous = AutonomousRun(
            white=self.sides["white"], black=self.sides["black"],
            delay_seconds=delay_seconds, task=task,
        )
        return ToolResult(
            f"Beginning autonomous play — {self.sides['white']} vs {self.sides['black']}. "
            "Say 'stop' to end it early."
        )

    def stop_autonomous(self) -> ToolResult:
        if self._autonomous is None or self._autonomous.task.done():
            return ToolResult("No autonomous game is currently running.")
        self._autonomous.task.cancel()
        return ToolResult("Stopping autonomous play.")

    def autonomous_status(self) -> ToolResult:
        run = self._autonomous
        if run is None:
            return ToolResult("No autonomous game has been run this session.")
        if not run.task.done():
            elapsed = time.monotonic() - run.started
            return ToolResult(
                f"Autonomous {run.white} vs {run.black} — {run.moves_played} ply played, "
                f"{elapsed:.0f}s elapsed. Move {len(self.history)}: {self._to_move_text()}."
            )
        return ToolResult(f"Autonomous game finished — {self._result_text()}.")

    async def _autonomous_run(self, delay_seconds: float, orion_depth: Optional[int],
                              orion_movetime_ms: int) -> None:
        self.bus.log.emit(
            f"CHESS: autonomous play started — {self.sides['white']} vs {self.sides['black']}."
        )
        self._emit({"phase": "autonomous_start", "white": self.sides["white"],
                    "black": self.sides["black"]})
        seconds = max(0.1, (orion_movetime_ms or ORION_SECONDS * 1000) / 1000.0)
        try:
            while not self.board.is_game_over(claim_draw=True):
                actor = self.sides[self._colour_name(self.board.turn)]
                # ORION's own engine — never Stockfish (chess_brain.py) — at
                # his real time control and the depth he was asked for.
                record = await self._auto_move(actor, seconds, orion_depth)
                if record is None:
                    break
                self._queue_review(record)
                if self._autonomous is not None:
                    self._autonomous.moves_played += 1
                await asyncio.sleep(delay_seconds)
            self.bus.log.emit(f"CHESS: autonomous play finished — {self._result_text()}.")
            self.bus.banner.emit(f"CHESS: game over — {self._result_text()}", 3)
            self._emit({"phase": "autonomous_end", "result": self._result_text()})
        except asyncio.CancelledError:
            self.bus.log.emit("CHESS: autonomous play cancelled.")
            raise
        except Exception as exc:
            self.bus.log.emit(f"CHESS: autonomous play fault - {first_line(exc, 160)}")

    def _cancel_autonomous_if_running(self) -> None:
        if self._autonomous is not None and not self._autonomous.task.done():
            self._autonomous.task.cancel()

    # ── review / history navigation ──────────────────────────────────────────

    def position_fen(self, ply: int) -> Optional[str]:
        if ply == 0:
            return self.start_fen
        if 1 <= ply <= len(self.history):
            return self.history[ply - 1].fen_after
        return None

    def _view_label(self) -> str:
        if self.is_live:
            return "LIVE position"
        if self.view_ply == 0:
            return f"the starting position (of {len(self.history)} ply played)"
        record = self.history[self.view_ply - 1]
        return f"reviewing after {record.label} (ply {self.view_ply} of {len(self.history)})"

    def goto_ply(self, ply: int) -> ToolResult:
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        ply = max(0, min(len(self.history), int(ply)))
        self.view_ply = ply
        self._emit({"phase": "view", "ply": ply, "live": self.is_live})
        fen = self.position_fen(ply)
        text = f"{_cap_first(self._view_label())}.\n{self._render_board(fen)}"
        if ply and self.history[ply - 1].comment:
            text += f"\n{self.history[ply - 1].comment}"
        return ToolResult(text)

    def step(self, delta: int) -> ToolResult:
        return self.goto_ply(self.view_ply + int(delta))

    def resume_live(self) -> ToolResult:
        return self.goto_ply(len(self.history))

    def goto_move(self, move_number: int, colour: str | None = None) -> ToolResult:
        """Show the position after move *move_number* (White's unless
        *colour* is 'black')."""
        wanted = (colour or "white").strip().lower()
        for record in self.history:
            if record.number == int(move_number) and record.mover == wanted:
                return self.goto_ply(record.ply)
        for record in self.history:
            if record.number == int(move_number):
                return self.goto_ply(record.ply)
        if not self.history:
            return ToolResult("No moves have been played yet.", ok=False)
        return ToolResult(f"There's no move {move_number} yet — the game has reached "
                          f"move {self.history[-1].number}.", ok=False)

    def board_flipped(self) -> bool:
        """Is Black at the bottom? Automatically so when the user plays Black
        (and only Black), unless the user has flipped the board themselves."""
        if self.view_flipped is not None:
            return self.view_flipped
        return self.sides.get("black") == "user" and self.sides.get("white") != "user"

    def flip(self) -> ToolResult:
        self.view_flipped = not self.board_flipped()
        self._emit({"phase": "flip", "flipped": self.view_flipped})
        return ToolResult("Board flipped — Black is at the bottom now." if self.view_flipped
                          else "Board flipped — White is at the bottom now.")

    def history_text(self) -> ToolResult:
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        if not self.history:
            return ToolResult("No moves have been played yet.")
        return ToolResult("Move history:\n" + "\n".join(r.line() for r in self.history))

    def history_list(self) -> list[dict[str, Any]]:
        """Structured history for the GUI's move-list widget."""
        return [
            {"ply": r.ply, "san": r.san, "label": r.label, "actor": r.actor,
             "quality": r.quality, "fen": r.fen_after, "comment": r.comment,
             "number": r.number, "mover": r.mover}
            for r in self.history
        ]

    # ── status / rendering ────────────────────────────────────────────────────

    def _player_colour(self) -> str:
        """Which side (if any) the human is on — the simple single-human-side
        shape ``board_state()`` reports for voice/dispatcher use. Defaults to
        'white' when there's no human side at all (an autonomous game)."""
        if self.sides.get("black") == "user" and self.sides.get("white") != "user":
            return "black"
        return "white"

    def users_turn(self) -> bool:
        """Can the user play a move right now?"""
        return (self._active and self.is_live and not self._thinking and not self._busy
                and not self.board.is_game_over()
                and self.sides.get(self._colour_name(self.board.turn)) == "user")

    def board_state(self) -> dict[str, Any]:
        """Plain-dict snapshot of the live game for the dispatcher/voice
        layer (contrast with ``board_result``/``status``, which return a
        rendered ``ToolResult`` for the GUI/spoken path)."""
        if not CHESS_AVAILABLE or self.board is None:
            return {"available": False}
        return {
            "available": True,
            "fen": self.board.fen(),
            "turn": self._colour_name(self.board.turn),
            "elo": self.elo,
            "orion_rating": round(self.brain.rating["orion"]) if self._brain_loaded() else None,
            "your_rating": round(self.brain.rating["you"]) if self._brain_loaded() else None,
            "analyst": self.analyst,
            "player_colour": self._player_colour(),
            "active": self._active,
            "thinking": self._thinking,
            "viewing_ply": None if self.is_live else self.view_ply,
            "check": self.board.is_check(),
            "history_san": [r.san for r in self.history],
            "status": self._result_text() if self.board.is_game_over()
                      else ("Resigned." if not self._active else "in progress"),
        }

    def board_result(self) -> ToolResult:
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        fen = self.position_fen(self.view_ply)
        return ToolResult(f"{_cap_first(self._view_label())}.\n{self._render_board(fen)}")

    def status(self) -> ToolResult:
        unavailable = self._unavailable()
        if unavailable is not None:
            return unavailable
        running = self._autonomous is not None and not self._autonomous.task.done()
        return ToolResult(
            f"Engine backend: {self.backend}. Sides: white={self.sides['white']}, "
            f"black={self.sides['black']}. Ply {len(self.history)} played. "
            f"{'Viewing history at ply ' + str(self.view_ply) + '.' if not self.is_live else 'At the live position.'} "
            f"{'Autonomous play is running.' if running else ''} "
            f"{self._to_move_text() if not self.board.is_game_over() else self._result_text()}"
        )

    def _render_board(self, fen: Optional[str]) -> str:
        if not fen:
            return "(no position)"
        try:
            board = chess.Board(fen)
        except ValueError:
            return "(invalid position)"
        return board.unicode(borders=False, empty_square=".")

    def _to_move_text(self) -> str:
        colour = self._colour_name(self.board.turn)
        return f"{colour.capitalize()} ({self.sides.get(colour, 'user')}) to move."

    def _result_text(self) -> str:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "in progress"
        if outcome.winner is None:
            return f"draw ({outcome.termination.name.lower().replace('_', ' ')})"
        winner = "White" if outcome.winner else "Black"
        return f"{winner} wins ({outcome.termination.name.lower().replace('_', ' ')})"

    @staticmethod
    def _colour_name(colour: bool) -> str:
        return "white" if colour else "black"

    # ── his rating, his practice, his analysis ─────────────────────────────────

    def _brain_loaded(self) -> bool:
        return getattr(self, "_brain", None) is not None

    def rating(self) -> ToolResult:
        from .chess_brain import MEASURED_STRENGTH
        brain = self.brain
        you = brain.rating
        text = (f"I'm rated {brain.rating_text()}; you're {round(you['you'])} over "
                f"{you['you_games']} game(s) against me.")
        recent = (you.get("log") or [])[-5:]
        if recent:
            marks = {1.0: "W", 0.5: "D", 0.0: "L"}
            text += " My last results: " + " ".join(
                marks.get(float(r["score"]), "?") + ("" if r["vs"] == "user" else "(SF)")
                for r in recent) + "."
        if MEASURED_STRENGTH:
            text += f" {MEASURED_STRENGTH}"
        return ToolResult(text + " " + brain.intuition_text())

    def set_analyst(self, who: str) -> ToolResult:
        choice = str(who or "").strip().lower()
        if choice in {"stockfish", "engine", "sf"}:
            self.analyst = "stockfish"
            self._evals.clear()
            return ToolResult("Stockfish will grade the moves and drive the evaluation "
                              "bar. Say 'analyse with your own engine' to switch back.")
        self.analyst = "orion"
        self._evals.clear()
        return ToolResult("I'll grade the moves myself, with my own search.")

    def start_practice(self, games: int = 10) -> ToolResult:
        """Self-play in the background: he plays himself and his intuition
        network learns from every position. Never blocks a live game — the
        practice runs on its own copy of the search."""
        import threading as _threading
        running = self._practice is not None and not self._practice.done()
        if running:
            return ToolResult("I'm already practising — say 'stop practising' to end it.")
        games = max(1, min(200, int(games)))
        self._practice_stop = _threading.Event()
        brain = self.brain

        def _progress(done: int, result: str) -> None:
            if done % 5 == 0 or done == games:
                self.bus.log.emit(f"CHESS: practice game {done}/{games} finished ({result}).")

        async def _run() -> None:
            summary = await asyncio.to_thread(brain.practice, games, 0.05,
                                              self._practice_stop, _progress)
            self.bus.log.emit(f"CHESS: {summary}")

        self._practice = asyncio.get_running_loop().create_task(
            _run(), name="orion-chess-practice")
        return ToolResult(f"Practising {games} game(s) against myself in the background "
                          "— my intuition network learns from every position. I'll say "
                          "how it went in the log.")

    def stop_practice(self) -> ToolResult:
        if self._practice is None or self._practice.done():
            return ToolResult("I'm not practising at the moment.")
        self._practice_stop.set()
        return ToolResult("Stopping after the current practice game.")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        stop = getattr(self, "_practice_stop", None)
        if stop is not None:
            stop.set()
        self._cancel_autonomous_if_running()
        task = self._review_task
        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:
                pass
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None


def _cap_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
