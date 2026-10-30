"""Resumable local training and an 80-pair champion arena. No online games.

Run at most one worker: python -m tools.improve_chess --seconds 3300
Create config/grandmaster/improvement.stop to stop without losing completed pairs.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import platform
import random
import shutil
import threading
import time
from contextlib import contextmanager
from importlib.metadata import version as package_version
from pathlib import Path

PAIRS = 80
MOVE_SECONDS = 0.08
TRAINING_POLICY = "quiet-v2"
TRAINING_CHECKPOINT = "training-quiet-v2.nnue"
EVALUATION_FILES = (
    "tools/improve_chess.py", "orion_core/chess_brain.py", "orion_core/neural.py",
    "orion_core/grandmaster/arena.py", "orion_core/grandmaster/search.py",
    "orion_core/grandmaster/weights.py", "orion_core/grandmaster/strength.py",
    "orion_core/grandmaster/curriculum.py",
)


def evaluation_sources():
    """Identify the evaluation implementation used by one worker process."""
    root = Path(__file__).resolve().parents[1]
    return {"files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in EVALUATION_FILES},
            "python": platform.python_version(), "chess": package_version("chess"),
            "numpy": package_version("numpy")}


# Capture before imports, then eagerly load the lazy evaluator modules. If
# files change during startup, the first checkpoint detects the discrepancy
# instead of tagging already-imported code with a newer on-disk fingerprint.
IMPORTED_EVALUATION_SOURCE = evaluation_sources()

from orion_core.atomic_io import atomic_write_text
from orion_core.grandmaster.search import make_brain, refresh_weights, worker_priority

if __name__ == "__main__":
    worker_priority()

from orion_core import chess_brain as _chess_brain, neural as _neural
from orion_core.grandmaster.arena import arena_game, lower_bound, promote
from orion_core.grandmaster.dataset import Dataset
from orion_core.grandmaster.selfplay import OPENINGS, collect_episode
from orion_core.grandmaster.training_owner import TrainingLease
from orion_core.grandmaster.weights import (
    Inference,
    active_model,
    file_hash,
    save_training_checkpoint,
)
from tools.train_chess import fit


@contextmanager
def exclusive_worker(directory):
    """OS lock is released even after a crash; an old PID cannot block resuming."""
    owner = TrainingLease.acquire(directory)
    if owner is None:
        raise BlockingIOError(errno.EAGAIN, "Another trainer already owns this checkpoint directory")
    try:
        yield
    finally:
        owner.close()


def relative_strength(scores):
    """Descriptive Elo difference, not an absolute or cumulative rating."""
    if not scores:
        return {"pairs": 0, "score": None, "relative_elo": None}
    score = sum(scores) / len(scores)
    bound = math.sqrt(math.log(100) / (2 * len(scores)))

    def elo(value):
        return round(400 * math.log10(value / (1 - value)), 1) if 0 < value < 1 else None

    return {"pairs": len(scores), "score": score, "relative_elo": elo(score),
            "relative_elo_lower": elo(max(0, score - bound)),
            "relative_elo_upper": elo(min(1, score + bound)),
            "score_lower_bound": lower_bound(scores),
            "rating_scope": "This equal-time match only; not FIDE/Lichess Elo",
            "provisional": len(scores) != PAIRS}


class Improvement:
    def __init__(self, directory, brain_path, stop):
        self.directory, self.brain_path, self.stop = Path(directory), Path(brain_path), stop
        self.evaluation_source = IMPORTED_EVALUATION_SOURCE
        self.state_path = self.directory / "improvement.json"
        self.scratch = self.directory / "evaluation"
        self.scratch.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("version") != 1 or self.state.get("phase") not in ("collect", "train", "arena"):
                raise ValueError("Unsupported improvement state; preserve it for inspection")
        else:
            self.state = {"version": 1, "phase": "collect", "cycle": 0,
                          "seed": random.SystemRandom().randrange(2**31),
                          "games": 0, "positions": 0, "collected": 0, "history": []}
            checkpoint = self.directory / TRAINING_CHECKPOINT
            if not checkpoint.is_file():
                checkpoint = self.directory / "training.nnue"
            if checkpoint.is_file():
                self.prepare_arena(checkpoint)

    def save(self, **updates):
        updates.setdefault("pid", os.getpid())
        self.state.update(updates, updated_at=time.time())
        atomic_write_text(self.state_path, json.dumps(self.state, indent=2), encoding="utf-8")

    def prepare_arena(self, candidate):
        # Snapshot the whole opponent, including classical tables and the small
        # value net. A user's live-game lessons cannot change a match halfway.
        Inference.load(candidate)
        save_training_checkpoint(Path(candidate), self.scratch / "candidate.nnue")
        for source, target in ((self.brain_path, self.scratch / "brain.json"),
                               (self.brain_path.with_name("chess_value.npz"), self.scratch / "chess_value.npz")):
            if source.is_file():
                shutil.copyfile(source, target)
            else:
                target.unlink(missing_ok=True)
        incumbent = active_model(self.directory)
        expected = incumbent.name if incumbent else None
        rng = random.Random(self.state["seed"] + self.state["cycle"])
        import chess
        openings, seen = [], set()
        while len(openings) < PAIRS * 4:
            opening = list(rng.choice(OPENINGS))
            board = chess.Board()
            for uci in opening:
                board.push_uci(uci)
            for _ in range(2):
                move = rng.choice(list(board.legal_moves))
                opening.append(move.uci())
                board.push(move)
            if board.fen() not in seen:
                openings.append(opening)
                seen.add(board.fen())
        self.save(phase="arena", status="evaluating", arena={
            "candidate_sha256": file_hash(self.scratch / "candidate.nnue"),
            "incumbent": expected, "openings": openings, "attempt": 0,
            "scores": [], "results": [], "seconds_per_move": MOVE_SECONDS,
            "source": self.evaluation_source})

    def evaluation_changed(self):
        if evaluation_sources() == self.evaluation_source:
            return False
        # A running Python process still owns its imported implementation.
        # Checkpoint and let the next process load the edited source files.
        self.save(status="checkpointed", restart_reason="Evaluation source changed")
        self.stop.set()
        return True

    def arena(self):
        if self.evaluation_changed():
            return
        arena = self.state["arena"]
        candidate = self.scratch / "candidate.nnue"
        if file_hash(candidate) != arena["candidate_sha256"]:
            raise ValueError("Evaluation candidate changed; refuse mixed results")
        incumbent = active_model(self.directory)
        if (incumbent.name if incumbent else None) != arena["incumbent"]:
            self.finish_arena("stale", False)
            return
        if arena.get("source") != self.evaluation_source:
            # Legacy or changed-code results cannot be pooled with this worker.
            # Preserve them and re-evaluate the same immutable candidate.
            self.finish_arena("stale", False, reason="Evaluation source changed or unrecorded")
            self.prepare_arena(candidate)
            return
        scores = arena["scores"]
        if len(scores) > PAIRS or any(s not in (0, 0.25, 0.5, 0.75, 1) for s in scores):
            raise ValueError("Invalid arena scores")
        challenger = make_brain(str(self.scratch / "brain.json"), self.stop)
        challenger.nnue = Inference.load(candidate)
        champion = make_brain(str(self.scratch / "brain.json"), self.stop)
        if incumbent:
            if file_hash(incumbent) != incumbent.stem:
                raise ValueError("Incumbent checksum mismatch")
            champion.nnue = Inference.load(incumbent)
        pause = threading.Event()
        while not self.stop.is_set() and len(scores) < PAIRS and arena["attempt"] < len(arena["openings"]):
            if self.evaluation_changed():
                return
            opening = arena["openings"][arena["attempt"]]
            a = arena_game(challenger, champion, opening, self.stop, pause, seconds=MOVE_SECONDS)
            b = None if self.stop.is_set() else arena_game(
                champion, challenger, opening, self.stop, pause, seconds=MOVE_SECONDS)
            if self.stop.is_set():
                # Resume the same pair. Never count a time-budget interruption
                # as a loss/draw or silently select a different opening.
                self.save(status="checkpointed", strength=relative_strength(scores))
                return
            arena["attempt"] += 1
            if a is not None and b is not None:
                scores.append((a + 1 - b) / 2)
                arena["results"].extend([a, 1 - b])
            self.save(status="evaluating", strength=relative_strength(scores))
        if self.stop.is_set() or self.evaluation_changed():
            return
        promoted = len(scores) == PAIRS and promote(
            self.directory, candidate, arena["incumbent"], scores, minimum_pairs=PAIRS)
        status = "promoted" if promoted else ("rejected" if len(scores) == PAIRS else "incomplete")
        self.finish_arena(status, promoted)

    def finish_arena(self, status, promoted, *, reason=None):
        arena = self.state["arena"]
        results = arena["results"]
        verdict = {"at": time.time(), "status": status, "promoted": promoted,
                   "candidate": arena["candidate_sha256"], "incumbent": arena["incumbent"],
                   "source": arena.get("source"),
                   "wins": results.count(1), "draws": results.count(0.5), "losses": results.count(0),
                   **relative_strength(arena["scores"])}
        if reason:
            verdict["reason"] = reason
        history = self.scratch / "history"
        history.mkdir(parents=True, exist_ok=True)
        archive = history / f"arena-{self.state['cycle']:06d}-{time.time_ns()}.json"
        verdict["archive"] = archive.relative_to(self.directory).as_posix()
        atomic_write_text(archive, json.dumps({"arena": arena, "verdict": verdict}, indent=2),
                          encoding="utf-8")
        self.state["history"] = (self.state["history"] + [verdict])[-200:]
        self.save(phase="collect", status=status, last_verdict=verdict,
                  cycle=self.state["cycle"] + 1, collected=0)

    def collect(self):
        dataset = Dataset(self.directory / "positions.sqlite3")
        brain = make_brain(str(self.brain_path), self.stop)
        version = refresh_weights(brain, self.directory, "classical")
        rng = random.Random(self.state["seed"] + self.state["games"] + self.state["cycle"] * 10000)
        try:
            while not self.stop.is_set() and self.state["collected"] < 80:
                episode = collect_episode(dataset, brain, self.stop, threading.Event(), rng,
                                          seconds=0.04, version=version, quiet_only=True)
                if episode and episode[0]:
                    self.save(status="collecting", games=self.state["games"] + 1,
                              positions=self.state["positions"] + len(episode[0]),
                              collected=self.state["collected"] + 1,
                              curriculum=dataset.curriculum_progress())
                elif not self.stop.is_set():
                    self.save(status="collecting", curriculum=dataset.curriculum_progress())
            if not self.stop.is_set():
                self.save(phase="train", status="ready to train")
        finally:
            dataset.close()

    def train(self, remaining):
        # The static evaluator needs quiet targets. Keep the earlier objective's
        # checkpoint intact and start this lane from the material prior once.
        checkpoint = self.directory / TRAINING_CHECKPOINT
        initial = checkpoint if checkpoint.is_file() else None
        self.save(status="training", training_policy=TRAINING_POLICY,
                  training_checkpoint=TRAINING_CHECKPOINT)
        result = fit(self.directory, initial, steps=1024, device="cpu", quiet_only=True,
                     seed=self.state["seed"] + self.state["cycle"], timeout=min(300, remaining), cancel=self.stop)
        self.save(last_training=result)
        if result.get("status") == "candidate":
            if result.get("label_policy") != TRAINING_POLICY:
                raise ValueError("Training candidate has the wrong label policy")
            candidate = Path(result["path"])
            save_training_checkpoint(candidate, checkpoint)
            self.prepare_arena(candidate)
            candidate.unlink(missing_ok=True)  # unique export owned by this finished child
        elif result.get("status") == "rejected":
            self.save(phase="collect", status="needs more experience", collected=0,
                      cycle=self.state["cycle"] + 1)
        elif result.get("status") == "error":
            raise RuntimeError(result.get("reason", "Training failed"))
        else:
            self.save(status="deferred")
            self.stop.set()


def main(argv=None):
    from orion_core.constants import CONFIG_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=3300)
    parser.add_argument("--directory", type=Path, default=CONFIG_DIR / "grandmaster")
    parser.add_argument("--brain", type=Path, default=CONFIG_DIR / "chess_brain.json")
    args = parser.parse_args(argv)
    if not 10 <= args.seconds <= 3600:
        parser.error("A run must be 10–3600 seconds")
    worker_priority()
    stop = threading.Event()
    deadline = time.monotonic() + args.seconds
    sentinel = args.directory / "improvement.stop"

    def watch():
        while not stop.wait(0.25):
            if time.monotonic() >= deadline or sentinel.exists():
                stop.set()

    with exclusive_worker(args.directory):
        if sentinel.exists():
            print("Training is stopped by improvement.stop", flush=True)
            return 0
        runner = Improvement(args.directory, args.brain, stop)
        threading.Thread(target=watch, daemon=True, name="chess-training-stop").start()
        try:
            while not stop.is_set():
                if runner.evaluation_changed():
                    break
                remaining = deadline - time.monotonic()
                if runner.state["phase"] == "arena":
                    runner.arena()
                elif runner.state["phase"] == "collect":
                    runner.collect()
                elif remaining >= 30:
                    runner.train(remaining)
                else:
                    break
            runner.save(status="stopped" if sentinel.exists() else "checkpointed", pid=None)
        except Exception as exc:
            runner.save(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
            raise
        finally:
            stop.set()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
