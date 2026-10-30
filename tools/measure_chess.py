"""Resumable, colour-paired chess measurements; never trains or promotes a model.

Use ``python -m tools.measure_chess --help`` for output and machine-label options.
Create ``STOP`` in the output directory to checkpoint and stop. Remove that
marker deliberately before resuming with the same arguments and source files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import threading
import time


# A published fixed suite. The historical 16-position suite was not retained;
# these results must not be pooled with that earlier measurement.
OPENINGS = tuple(line.split() for line in (
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4",
    "e2e4 e7e5 g1f3 g8f6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 c2c4 e7e6",
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6",
))
LEVELS = (2400, 2600, 2800)
SOURCE_FILES = (
    "tools/measure_chess.py", "orion_core/chess_brain.py",
    "orion_core/chess_theory.py", "orion_core/neural.py",
    "orion_core/grandmaster/search.py", "orion_core/grandmaster/weights.py",
    "orion_core/grandmaster/strength.py",
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def schedule(pairs, head_to_head=False, levels=LEVELS):
    """Both colours share an opening and opponent level, before the next pair."""
    if not levels and not head_to_head:
        raise ValueError("At least one opponent level is required")
    return [{"opening": (i if head_to_head else i // len(levels)) % len(OPENINGS),
             "elo": None if head_to_head else levels[i % len(levels)]}
            for i in range(pairs)]


def performance(observations):
    """Logistic maximum-likelihood performance; None for unbounded extremes."""
    if not observations:
        return None
    total = sum(score for _, score in observations)
    if total == 0 or total == len(observations):
        return None
    low, high = -10000.0, 10000.0
    for _ in range(70):
        middle = (low + high) / 2
        expected = sum(1 / (1 + 10 ** ((elo - middle) / 400)) for elo, _ in observations)
        if expected < total:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def summarise(pairs, *, head_to_head=False, samples=2000):
    """Bootstrap whole colour pairs within opponent strata, never single games."""
    groups = {}
    scores = []
    for pair in pairs:
        if len(pair["games"]) != 2 or any(g["score"] not in (0, 0.5, 1) for g in pair["games"]):
            raise ValueError("Only completed colour pairs may enter the rating")
        groups.setdefault(pair["elo"] or 0, []).append(pair)
        scores.extend(g["score"] for g in pair["games"])
    observations = [(elo, g["score"]) for elo, group in groups.items()
                    for p in group for g in p["games"]]
    estimate = performance(observations)
    result = {"pairs": len(pairs), "games": len(scores), "wins": scores.count(1),
              "draws": scores.count(0.5), "losses": scores.count(0),
              "elo": round(estimate, 1) if estimate is not None else None,
              "scope": "equal-time difference" if head_to_head else "Stockfish UCI_Elo scale",
              "bootstrap_samples": samples, "ci95": None,
              "interval_method": "percentile bootstrap of colour pairs, stratified by opponent"}
    if not pairs or estimate is None or any(len(group) < 2 for group in groups.values()):
        return result
    rng, estimates = random.Random(20260927), []
    for _ in range(samples):
        sample = [(elo, g["score"]) for elo, group in groups.items()
                  for p in rng.choices(group, k=len(group)) for g in p["games"]]
        rating = performance(sample)
        if rating is None:
            rating = -math.inf if sum(s for _, s in sample) == 0 else math.inf
        estimates.append(rating)
    estimates.sort()
    bounds = [estimates[int((samples - 1) * q)] for q in (0.025, 0.975)]
    result["ci95"] = [round(b, 1) if math.isfinite(b) else None for b in bounds]
    return result


@contextmanager
def exclusive_run(directory):
    """A crashed worker releases its OS lock without trusting a saved PID."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "measurement.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Watchdog:
    def __init__(self, output, seconds, minimum_mb, available=None):
        if available is None:
            import psutil
            available = lambda: psutil.virtual_memory().available / 1048576
        self.available = available
        self.output, self.minimum_mb = output, minimum_mb
        self.deadline = time.monotonic() + seconds
        self.stop, self.done = threading.Event(), threading.Event()
        self.reason = ""
        self.lowest_mb = math.inf
        self.thread = threading.Thread(target=self._run, name="chess-memory-watchdog", daemon=True)

    def check(self):
        try:
            free = self.available()
            self.lowest_mb = min(self.lowest_mb, free)
            if free < self.minimum_mb:
                self.reason = "low memory"
            elif (self.output / "STOP").exists():
                self.reason = "stop marker"
            elif time.monotonic() >= self.deadline:
                self.reason = "time budget"
        except Exception:
            self.reason = "memory probe failed"
        if self.reason:
            self.stop.set()
        return not self.stop.is_set()

    def _run(self):
        while not self.done.wait(0.2) and self.check():
            pass

    def __enter__(self):
        self.check()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join(timeout=2)


def play_game(white, black, opening, stop, *, max_plies=600):
    """Callbacks own their clocks. Incomplete/illegal games never become draws."""
    import chess
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    started = time.monotonic()
    moves = []
    for _ in range(max_plies + 1):
        if stop.is_set():
            return None
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return {"result": outcome.result(), "moves": moves,
                    "termination": outcome.termination.name,
                    "seconds": round(time.monotonic() - started, 3)}
        if len(moves) == max_plies:
            return None
        move = (white if board.turn else black)(board)
        if stop.is_set():
            return None
        if move not in board.legal_moves:
            raise ValueError("Engine returned an illegal move")
        moves.append(move.uci())
        board.push(move)
    return None


def child_environment():
    """The external opponent needs OS plumbing, no application credentials."""
    allowed = {"systemroot", "windir", "temp", "tmp", "path"}
    return {key: value for key, value in os.environ.items() if key.lower() in allowed}


def close_opponent(engine):
    """Bound shutdown, then wait for the owned process's exit even after failure."""
    try:
        engine.quit()
    except Exception:
        engine.close()
    finally:
        engine.close()
        engine.returncode.result(timeout=10)


def load_state(path, contract):
    if not path.exists():
        return {"contract": contract, "pairs": [], "status": "ready", "runs": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["contract"] != contract:
        raise ValueError("Measurement inputs changed; use a new output directory")
    if len(state["pairs"]) > len(contract["schedule"]):
        raise ValueError("More saved pairs than scheduled")
    for index, pair in enumerate(state["pairs"]):
        expected = contract["schedule"][index]
        if pair["elo"] != expected["elo"] or pair["opening"] != expected["opening"]:
            raise ValueError("Saved pair differs from the fixed schedule")
        if [g["colour"] for g in pair["games"]] != ["white", "black"]:
            raise ValueError("Saved pair must contain both colours")
        import chess
        for game in pair["games"]:
            board = chess.Board()
            for move in contract["openings"][pair["opening"]] + game["moves"]:
                board.push_uci(move)
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.result() != game["result"]:
                raise ValueError("Saved game has no matching completed outcome")
            white_score = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[game["result"]]
            if game["score"] != (white_score if game["colour"] == "white" else 1 - white_score):
                raise ValueError("Saved score differs from the game outcome")
    return state


def run(args):
    # Set this before importing any project path constants. No real config is read.
    output = args.output.resolve()
    os.environ["ORION_CONFIG_DIR"] = str(output / "isolated_config")
    os.environ["ORION_NO_CAPTURE"] = "1"
    from orion_core.grandmaster.search import make_brain, worker_priority
    worker_priority()
    import chess
    import chess.engine
    import psutil
    from orion_core.atomic_io import atomic_write_text
    from orion_core.chess_theory import seed_into
    from orion_core.grandmaster.weights import Inference

    root = Path(__file__).resolve().parents[1]
    contract = {"version": 1, "seconds": args.move_seconds, "opponent_seconds": args.opponent_seconds,
                "head_to_head": args.head_to_head, "schedule": schedule(args.pairs, args.head_to_head, getattr(args, "levels", LEVELS)),
                "openings": OPENINGS, "seed": args.seed,
                "source": {name: digest(root / name) for name in SOURCE_FILES},
                "candidate": digest(args.candidate) if args.candidate else None,
                "stockfish": None if args.head_to_head else digest(args.stockfish),
                "python": platform.python_version(), "chess": chess.__version__,
                "platform": platform.platform(), "processor": platform.processor(),
                "cpu": args.machine, "threads": 1, "stockfish_hash_mb": 32,
                "brain": "shipped classical tables and seeded theory; no personal learning",
                "max_plies": 600, "minimum_free_mb": args.minimum_free_mb}
    # JSON-normalise tuples before comparing across restarts.
    contract = json.loads(json.dumps(contract))
    with exclusive_run(output):
        if any((output / "isolated_config" / name).exists() for name in ("brain.json", "chess_value.npz")):
            raise ValueError("Measurement config must not contain learned chess state")
        state_path = output / "results.json"
        state = load_state(state_path, contract)
        if len(state["pairs"]) == args.pairs:
            print(json.dumps(state.get("summary", {})), flush=True)
            return 0

        def save():
            atomic_write_text(state_path, json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")

        with Watchdog(output, args.seconds, args.minimum_free_mb) as guard:
            engine = None
            state["status"] = "running"
            state["pid"] = os.getpid()
            run_info = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "available_mb": round(psutil.virtual_memory().available / 1048576)}
            state["runs"].append(run_info)
            save()
            old_signal = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda *_: guard.stop.set())
            try:
                if not guard.check():
                    return 2
                candidate = output / "candidate.nnue"
                if args.candidate:
                    if not candidate.exists():
                        shutil.copyfile(args.candidate, candidate)
                    if digest(candidate) != contract["candidate"]:
                        raise ValueError("Candidate snapshot differs from the recorded input")
                if not args.head_to_head:
                    engine = chess.engine.SimpleEngine.popen_uci(
                        str(args.stockfish.resolve()), timeout=10, env=child_environment())
                    engine.configure({"Threads": 1, "Hash": 32, "UCI_LimitStrength": True})
                    option = engine.options["UCI_Elo"]
                    if any(not option.min <= p["elo"] <= option.max for p in contract["schedule"]):
                        raise ValueError(f"Opponent levels must be between {option.min} and {option.max}")
                    state["stockfish_id"] = engine.id
                for index in range(len(state["pairs"]), args.pairs):
                    if not guard.check():
                        break
                    spec = contract["schedule"][index]
                    if engine:
                        engine.configure({"UCI_Elo": spec["elo"]})
                    pair = {**spec, "games": []}
                    for colour in ("white", "black"):
                        # Fresh search caches for every game; seeded book retained.
                        brain = make_brain(str(output / "isolated_config" / "brain.json"), guard.stop)
                        brain._rng.seed(args.seed + index)
                        seed_into(brain)
                        if args.candidate:
                            brain.nnue = Inference.load(candidate)
                        ours = partial(brain.choose_move, seconds=args.move_seconds)
                        other = None
                        if engine:
                            game_id = object()
                            opponent = lambda board: engine.play(
                                board, chess.engine.Limit(time=args.opponent_seconds), game=game_id).move
                        else:
                            other = make_brain(str(output / "isolated_config" / "brain.json"), guard.stop)
                            other._rng.seed(args.seed + index)
                            seed_into(other)
                            opponent = partial(other.choose_move, seconds=args.move_seconds)
                        white, black = (ours, opponent) if colour == "white" else (opponent, ours)
                        result = play_game(white, black, OPENINGS[spec["opening"]], guard.stop)
                        # Drop NNUE mappings and transposition caches before the next game.
                        del white, black, ours, opponent, brain, other
                        if result is None:
                            if not guard.stop.is_set():
                                guard.reason = "censored game; fixed suite incomplete"
                                guard.stop.set()
                            break
                        white_score = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[result["result"]]
                        pair["games"].append({**result, "colour": colour,
                                              "score": white_score if colour == "white" else 1 - white_score})
                    if len(pair["games"]) != 2:
                        state["interrupted_pair"] = pair
                        break
                    state.pop("interrupted_pair", None)
                    state["pairs"].append(pair)
                    save()
                    print(json.dumps({"completed_games": 2 * len(state["pairs"]),
                                      "target_games": 2 * args.pairs,
                                      "available_mb": round(guard.available())}), flush=True)
            except Exception as exc:
                guard.reason = f"error: {type(exc).__name__}: {str(exc)[:200]}"
                guard.stop.set()
                raise
            finally:
                try:
                    if engine is not None:
                        close_opponent(engine)
                        run_info["opponent_closed"] = True
                finally:
                    signal.signal(signal.SIGINT, old_signal)
                    state["pid"] = None
                    state["status"] = "complete" if len(state["pairs"]) == args.pairs else "checkpointed"
                    state["reason"] = guard.reason or ("finished" if state["status"] == "complete" else "interrupted")
                    run_info["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    run_info["minimum_available_mb"] = round(guard.lowest_mb) if math.isfinite(guard.lowest_mb) else None
                    state["summary"] = summarise(state["pairs"], head_to_head=args.head_to_head)
                    state["summary"]["provisional"] = state["status"] != "complete"
                    save()
                    print(json.dumps({"status": state["status"], "reason": state["reason"],
                                      "summary": state["summary"]}), flush=True)
    return 0 if state["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, default=Path("engines/stockfish.exe"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--head-to-head", action="store_true", help="Candidate versus shipped classical engine")
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--levels", type=int, nargs="+", default=LEVELS,
                        help="Fixed opponent bracket on Stockfish's supported UCI_Elo scale")
    parser.add_argument("--move-seconds", type=float, default=1.5)
    parser.add_argument("--opponent-seconds", type=float, default=0.3)
    parser.add_argument("--seconds", type=float, default=14400)
    parser.add_argument("--minimum-free-mb", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--machine", required=True, help="Verified processor model for the report")
    args = parser.parse_args()
    if args.head_to_head and not args.candidate:
        parser.error("--head-to-head requires --candidate")
    if not 1 <= args.pairs <= 1000:
        parser.error("--pairs must be between 1 and 1000")
    if not args.levels or len(set(args.levels)) != len(args.levels) or any(n <= 0 for n in args.levels):
        parser.error("--levels must contain distinct positive ratings")
    for name in ("move_seconds", "opponent_seconds", "seconds", "minimum_free_mb"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.minimum_free_mb < 600 or max(args.move_seconds, args.opponent_seconds) > 30:
        parser.error("Keep at least 600 MB free and move budgets at or below 30 seconds")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
