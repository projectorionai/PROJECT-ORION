"""Bounded offline self-play and NNUE training, using ORION's own search only.

    python -m tools.train_chess --games 64 --steps 512 --device auto

Candidates are retained for the existing Grandmaster arena; this command never
changes active.json. No network, accounts, external engine or CUDA is required.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import threading
import time
from pathlib import Path


def training_worker(directory, initial, stop, pipe, steps, device, seed, factorized, material_init,
                    quiet_only=False):
    from orion_core.grandmaster.search import worker_priority
    worker_priority()
    from orion_core.grandmaster.nnue import train_pass
    train_pass(directory, initial, stop, pipe, steps=steps, device=device,
               seed=seed, factorized=factorized, material_init=material_init, quiet_only=quiet_only)


def fit(directory, initial, *, steps, device, seed, timeout, factorized=True, material_init=True,
        quiet_only=False, cancel=None):
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    stop = ctx.Event()
    process = ctx.Process(target=training_worker, args=(
        str(directory), str(initial) if initial else None, stop, sender,
        steps, device, seed, factorized, material_init, quiet_only))
    try:
        process.start()
        sender.close()
        deadline = time.monotonic() + timeout
        while (process.is_alive() and time.monotonic() < deadline
               and (cancel is None or not cancel.is_set())):
            process.join(0.25)
        if process.is_alive():
            return {"status": "deferred", "reason": "Training stopped or time budget reached"}
        return receiver.recv() if receiver.poll() else {"status": "error", "reason": "Training worker exited"}
    finally:
        stop.set()
        if process.pid is not None:
            process.join(0.5)
            if process.is_alive():
                process.terminate()
                process.join()
        receiver.close()
        sender.close()
        process.close()


def main(argv=None):
    from orion_core.constants import CONFIG_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=CONFIG_DIR / "grandmaster")
    parser.add_argument("--brain", type=Path, default=CONFIG_DIR / "chess_brain.json")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--move-seconds", type=float, default=0.025)
    parser.add_argument("--collection-seconds", type=float, default=300)
    parser.add_argument("--training-seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Train a fresh candidate while preserving playing weights")
    parser.add_argument("--quiet-only", action="store_true",
                        help="Use quiet-v2 static-evaluation targets and its separate training checkpoint")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if (not 0 <= args.games <= 1000 or not 1 <= args.steps <= 4096
            or not 0.005 <= args.move_seconds <= 2
            or not 1 <= args.collection_seconds <= 3600
            or not 1 <= args.training_seconds <= 3600):
        parser.error("Training budgets outside supported bounds")

    from orion_core.grandmaster.training_owner import TrainingLease
    owner = TrainingLease.acquire(args.directory)
    if owner is None:
        print(json.dumps({"status": "busy", "reason": "Another trainer owns this checkpoint directory"}), flush=True)
        return 2
    try:
        return _run(args)
    finally:
        owner.close()


def _run(args):

    from orion_core.atomic_io import atomic_write_text
    from orion_core.grandmaster.dataset import Dataset
    from orion_core.grandmaster.search import (
        make_brain,
        refresh_weights,
        worker_priority,
    )
    from orion_core.grandmaster.selfplay import collect_episode
    from orion_core.grandmaster.weights import active_model, save_training_checkpoint
    worker_priority()
    args.directory.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.directory / "last_training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "games_requested": args.games, "games_completed": 0,
              "games_censored": 0, "positions_added": 0, "status": "collecting",
              "label_policy": "quiet-v2" if args.quiet_only else "all-positions-v1",
              "promotion": "Not promoted; requires independent playing arena"}

    def publish():
        atomic_write_text(report_path, json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)

    stop, pause = threading.Event(), threading.Event()
    timer = threading.Timer(args.collection_seconds, stop.set)
    timer.daemon = True
    dataset = Dataset(args.directory / "positions.sqlite3")
    started = time.monotonic()
    timer.start()
    try:
        brain = make_brain(str(args.brain), stop)
        version = refresh_weights(brain, args.directory, "classical")
        rng = random.Random(args.seed)
        publish()
        for _ in range(args.games):
            if stop.is_set():
                break
            episode = collect_episode(dataset, brain, stop, pause, rng,
                                      seconds=args.move_seconds, version=version,
                                      quiet_only=args.quiet_only)
            if episode is not None and episode[0]:
                report["games_completed"] += 1
                report["positions_added"] += len(episode[0])
            else:
                report["games_censored"] += 1
            publish()
        report["curriculum"] = dataset.curriculum_progress()
        report["dataset_positions"] = dataset.count()
        report["validation_positions"] = len(dataset.validation_rows(
            1024, random.Random(args.seed), quiet_only=args.quiet_only))
    finally:
        timer.cancel()
        stop.set()
        dataset.close()
    report["collection_seconds"] = round(time.monotonic() - started, 2)
    report["status"] = "training"
    publish()
    checkpoint = args.directory / ("training-quiet-v2.nnue" if args.quiet_only else "training.nnue")
    initial = (checkpoint if checkpoint.is_file() else
               None if args.quiet_only else active_model(args.directory))
    if args.fresh:
        initial = None
    if args.compare_baseline:
        report["baseline"] = fit(args.directory, initial, steps=args.steps, device=args.device,
                                 seed=args.seed, timeout=args.training_seconds, factorized=False,
                                 material_init=False, quiet_only=args.quiet_only)
        publish()
    report["training"] = fit(args.directory, initial, steps=args.steps, device=args.device,
                              seed=args.seed, timeout=args.training_seconds, quiet_only=args.quiet_only)
    if report["training"].get("status") == "candidate":
        save_training_checkpoint(Path(report["training"]["path"]), checkpoint)
        report["checkpoint"] = str(checkpoint.resolve())
    report["status"] = report["training"]["status"]
    publish()
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
