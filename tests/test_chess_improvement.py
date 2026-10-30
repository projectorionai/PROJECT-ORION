"""Resuming tournaments must never fabricate gains or weaken promotion gates."""
import json
import threading
from types import SimpleNamespace

import pytest

from tools import improve_chess as training


def test_relative_strength_is_honest_about_small_and_extreme_samples():
    assert training.relative_strength([])["relative_elo"] is None
    assert training.relative_strength([0.5] * 80)["relative_elo"] == 0
    assert training.relative_strength([0.75] * 80)["relative_elo"] == pytest.approx(190.8)
    assert training.relative_strength([1] * 80)["relative_elo"] is None
    assert training.relative_strength([0.75])["provisional"]
    assert training.relative_strength([0.51] * 80)["score_lower_bound"] < 0.5


def test_exclusive_lock_releases_and_blocks_duplicates(tmp_path):
    with training.exclusive_worker(tmp_path), pytest.raises(OSError), training.exclusive_worker(tmp_path):
        pytest.fail("Duplicate trainer acquired lock")
    with training.exclusive_worker(tmp_path):
        pass


def ready_arena(tmp_path, monkeypatch):
    stop = threading.Event()
    runner = training.Improvement(tmp_path, tmp_path / "brain.json", stop)
    candidate = tmp_path / "evaluation/candidate.nnue"
    candidate.write_bytes(b"test")
    monkeypatch.setattr(training, "file_hash", lambda path: "fixture")
    monkeypatch.setattr(training, "Inference", SimpleNamespace(load=lambda path: object()))
    monkeypatch.setattr(training, "make_brain", lambda *args: SimpleNamespace(nnue=None))
    runner.save(phase="arena", arena={"candidate_sha256": "fixture", "incumbent": None,
        "scores": [], "results": [], "openings": [["e2e4"]] * 320, "attempt": 0,
        "source": runner.evaluation_source})
    return runner, stop


def test_interrupted_pair_is_retried_without_fabricating_a_result(tmp_path, monkeypatch):
    runner, stop = ready_arena(tmp_path, monkeypatch)
    calls = []

    def game(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            stop.set()
            return None
        return 1.0

    monkeypatch.setattr(training, "arena_game", game)
    monkeypatch.setattr(training, "promote", lambda *a, **kw: pytest.fail("Partial arena promoted"))
    runner.arena()
    restored = training.Improvement(tmp_path, tmp_path / "brain.json", threading.Event())
    assert restored.state["arena"]["attempt"] == 0
    assert restored.state["arena"]["scores"] == []
    assert restored.state["phase"] == "arena"


def test_resumed_arena_promotes_only_at_full_horizon(tmp_path, monkeypatch):
    runner, _stop = ready_arena(tmp_path, monkeypatch)
    runner.state["arena"].update(scores=[1.0] * 79, results=[1.0] * 158, attempt=79)
    outcomes = iter([1.0, 0.0])
    monkeypatch.setattr(training, "arena_game", lambda *a, **kw: next(outcomes))
    promotions = []

    def promote(*args, **kwargs):
        promotions.append((len(args[3]), kwargs["minimum_pairs"]))
        return True

    monkeypatch.setattr(training, "promote", promote)
    runner.arena()
    assert promotions == [(80, 80)]
    saved = json.loads(runner.state_path.read_text())
    assert saved["phase"] == "collect"
    assert saved["last_verdict"]["wins"] == 160
    assert saved["last_verdict"]["status"] == "promoted"


def test_exhausted_censored_arena_cannot_promote(tmp_path, monkeypatch):
    runner, _stop = ready_arena(tmp_path, monkeypatch)
    runner.state["arena"]["attempt"] = 319
    monkeypatch.setattr(training, "arena_game", lambda *a, **kw: None)
    monkeypatch.setattr(training, "promote", lambda *a, **kw: pytest.fail("Censored games promoted"))
    runner.arena()
    assert runner.state["last_verdict"]["status"] == "incomplete"
    assert runner.state["last_verdict"]["pairs"] == 0


def test_changed_champion_invalidates_resumed_arena(tmp_path, monkeypatch):
    runner, _stop = ready_arena(tmp_path, monkeypatch)
    monkeypatch.setattr(training, "active_model", lambda directory: tmp_path / "new.nnue")
    monkeypatch.setattr(training, "arena_game", lambda *a, **kw: pytest.fail("Stale arena ran"))
    runner.arena()
    assert runner.state["last_verdict"]["status"] == "stale"


@pytest.mark.parametrize("recorded_source", [None, {"files": {"old": "implementation"}}])
def test_changed_or_unrecorded_source_restarts_match_and_archives_results(
        tmp_path, monkeypatch, recorded_source):
    runner, _stop = ready_arena(tmp_path, monkeypatch)
    runner.state["arena"].update(source=recorded_source, scores=[1.0] * 79,
                                 results=[1.0, 0.0] * 79, attempt=79)
    monkeypatch.setattr(training, "arena_game", lambda *a, **kw: pytest.fail("Mixed-source match ran"))
    monkeypatch.setattr(training, "promote", lambda *a, **kw: pytest.fail("Mixed-source match promoted"))
    runner.arena()
    verdict = runner.state["last_verdict"]
    archive = json.loads((tmp_path / verdict["archive"]).read_text())
    assert verdict["status"] == "stale"
    assert verdict["pairs"] == 79
    assert archive["arena"]["scores"] == [1.0] * 79
    assert archive["arena"]["source"] == recorded_source
    assert runner.state["phase"] == "arena"
    assert runner.state["arena"]["scores"] == []
    assert runner.state["arena"]["source"] == runner.evaluation_source


def test_source_edited_during_final_pair_cannot_promote(tmp_path, monkeypatch):
    runner, stop = ready_arena(tmp_path, monkeypatch)
    runner.state["arena"].update(scores=[1.0] * 79, results=[1.0] * 158, attempt=79)
    outcomes = iter([1.0, 0.0])

    def game(*args, **kwargs):
        result = next(outcomes)
        if result == 0:
            monkeypatch.setattr(training, "evaluation_sources", lambda: {"files": {"edited": "source"}})
        return result

    monkeypatch.setattr(training, "arena_game", game)
    monkeypatch.setattr(training, "promote", lambda *a, **kw: pytest.fail("Edited-source worker promoted"))
    runner.arena()
    assert stop.is_set()
    assert runner.state["status"] == "checkpointed"
    assert len(runner.state["arena"]["scores"]) == 80
    assert runner.state["arena"]["source"] == runner.evaluation_source
    assert "last_verdict" not in runner.state


def test_checkpoint_status_clears_live_pid(tmp_path):
    runner = training.Improvement(tmp_path, tmp_path / "brain.json", threading.Event())
    runner.save(status="checkpointed", pid=None)
    assert json.loads(runner.state_path.read_text())["pid"] is None


@pytest.mark.parametrize("resume", [False, True])
def test_continuous_training_keeps_quiet_checkpoint_separate(tmp_path, monkeypatch, resume):
    runner = training.Improvement(tmp_path, tmp_path / "brain.json", threading.Event())
    original = tmp_path / "training.nnue"
    original.write_bytes(b"original objective")
    checkpoint = tmp_path / training.TRAINING_CHECKPOINT
    if resume:
        checkpoint.write_bytes(b"previous quiet candidate")
    candidate = tmp_path / "candidate-fixture.nnue"
    candidate.write_bytes(b"new quiet candidate")
    calls = []

    def fit(directory, initial, **kwargs):
        calls.append((directory, initial, kwargs))
        return {"status": "candidate", "path": str(candidate), "label_policy": "quiet-v2"}

    monkeypatch.setattr(training, "fit", fit)
    monkeypatch.setattr(training, "active_model", lambda *a: pytest.fail("Old objective resumed"))
    monkeypatch.setattr(runner, "prepare_arena", lambda path: runner.save(phase="arena"))
    runner.train(500)
    directory, initial, options = calls[0]
    assert directory == tmp_path
    assert initial == (checkpoint if resume else None)
    assert options["quiet_only"] is True
    assert options["device"] == "cpu" and options["steps"] == 1024
    assert options["timeout"] == 300 and options["cancel"] is runner.stop
    assert original.read_bytes() == b"original objective"
    assert checkpoint.read_bytes() == b"new quiet candidate"
    assert not candidate.exists()
    assert runner.state["training_policy"] == "quiet-v2"
    assert runner.state["training_checkpoint"] == training.TRAINING_CHECKPOINT
    assert runner.state["phase"] == "arena"


def test_wrong_training_policy_cannot_replace_quiet_checkpoint(tmp_path, monkeypatch):
    runner = training.Improvement(tmp_path, tmp_path / "brain.json", threading.Event())
    checkpoint = tmp_path / training.TRAINING_CHECKPOINT
    checkpoint.write_bytes(b"quiet progress")
    candidate = tmp_path / "candidate-fixture.nnue"
    candidate.write_bytes(b"wrong objective")
    monkeypatch.setattr(training, "fit", lambda *a, **kw: {
        "status": "candidate", "path": str(candidate), "label_policy": "all-positions-v1"})
    monkeypatch.setattr(runner, "prepare_arena", lambda *a: pytest.fail("Wrong policy evaluated"))
    with pytest.raises(ValueError, match="wrong label policy"):
        runner.train(500)
    assert checkpoint.read_bytes() == b"quiet progress"
    assert candidate.read_bytes() == b"wrong objective"


def test_curriculum_preserves_openings_and_covers_legal_endgames_for_both_colours():
    import chess
    from orion_core.grandmaster.curriculum import ENDGAMES, OPENING, PATHWAYS, training_start

    class FixedRandom:
        def __init__(self, mirror):
            self.mirror = mirror

        def getrandbits(self, _bits):
            return self.mirror

        def choice(self, values):
            return values[0]

    rotation = [training_start(index) for index in range(120)]
    assert rotation == [training_start(index) for index in range(120)]
    assert rotation[:12] != rotation[12:24]
    for offset in range(0, len(rotation), 12):
        block = rotation[offset:offset + 12]
        assert block.count(OPENING) == 6
        assert set(start.key for start in block) == set(PATHWAYS)
        assert [block.count(start) for start in ENDGAMES] == [1] * 6
    # Game IDs are 1-based; curriculum attempt counts are 0-based. A strict
    # alternating schedule put every opening into training, never validation.
    for validation in (False, True):
        categories = {"opening" if start is OPENING else "endgame"
                      for game_id, start in enumerate(rotation, 1)
                      if (game_id % 10 == 0) == validation}
        assert categories == {"opening", "endgame"}
    # Restarting at an arbitrary cursor reconstructs the same pending lesson.
    assert [training_start(index) for index in range(37, 120)] == rotation[37:]
    assert OPENING.board(FixedRandom(0)).move_stack == [
        chess.Move.from_uci(uci) for uci in training.OPENINGS[0]]
    for start in ENDGAMES:
        white = start.board(FixedRandom(0))
        mirrored = start.board(FixedRandom(1))
        assert white.is_valid() and mirrored.is_valid()
        assert not white.is_game_over(claim_draw=True)
        assert not mirrored.is_game_over(claim_draw=True)
        assert mirrored.fen() == white.mirror().fen()
        assert mirrored.turn != white.turn
        assert len(white.piece_map()) < 32


@pytest.mark.parametrize("attempts", [-1, 1.5, True, "1"])
def test_curriculum_rejects_invalid_progress(attempts):
    from orion_core.grandmaster.curriculum import training_start
    with pytest.raises(ValueError):
        training_start(attempts)


def test_curriculum_progress_resumes_without_changing_historical_games(tmp_path):
    import chess
    from orion_core.grandmaster.curriculum import training_start
    from orion_core.grandmaster.dataset import Dataset
    path = tmp_path / "positions.sqlite3"
    dataset = Dataset(path)
    try:
        dataset.append([(chess.STARTING_FEN, 0)], "1/2-1/2", "historical")
        archived = dataset.db.execute("SELECT * FROM games").fetchall()
        assert dataset.curriculum_progress()["attempts"] == 0
        dataset.record_censored("opening")
        dataset.append([(chess.STARTING_FEN, 23)], "1-0", "candidate", pathway="pawn_endgame")
        assert dataset.db.execute("SELECT * FROM games WHERE id=1").fetchall() == archived
        assert dataset.curriculum_progress()["attempts"] == 2
    finally:
        dataset.close()
    reopened = Dataset(path)
    try:
        progress = reopened.curriculum_progress()
        assert progress["attempts"] == 2 and progress["completed"] == 1
        assert progress["censored"] == 1 and progress["positions"] == 1
        assert progress["pathways"]["opening"]["censored"] == 1
        assert progress["pathways"]["pawn_endgame"]["completed"] == 1
        assert training_start(progress["attempts"]) == training_start(2)
    finally:
        reopened.close()


def test_curriculum_game_and_progress_rollback_together(tmp_path, monkeypatch):
    import chess
    from orion_core.grandmaster.dataset import Dataset
    dataset = Dataset(tmp_path / "positions.sqlite3")
    try:
        def failed_record(*_args, **_kwargs):
            raise OSError("Simulated progress write failure")

        monkeypatch.setattr(dataset, "_record_pathway", failed_record)
        with pytest.raises(OSError, match="progress write"):
            dataset.append([(chess.STARTING_FEN, 0)], "1/2-1/2", "fixture", pathway="opening")
        assert dataset.count() == 0
        assert dataset.curriculum_progress()["attempts"] == 0
        with pytest.raises(ValueError, match="pathway"):
            dataset.record_censored("unknown")
    finally:
        dataset.close()


def test_collection_rotates_after_caps_but_preserves_interrupted_lesson(tmp_path, monkeypatch):
    import chess
    import random
    from orion_core.grandmaster import selfplay
    from orion_core.grandmaster.curriculum import training_start
    from orion_core.grandmaster.dataset import Dataset
    dataset = Dataset(tmp_path / "positions.sqlite3")
    stop, pause, starts = threading.Event(), threading.Event(), []

    def episode(_brain, _stop, _pause, _rng, **options):
        starts.append(options["start"].key)
        assert options["quiet_only"] is True
        if len(starts) == 1:
            return None  # Capped game, no terminal result.
        if len(starts) == 2:
            stop.set()  # Timer or foreground interruption, same lesson resumes.
            return None
        return [(chess.STARTING_FEN, 0)], "1/2-1/2"

    monkeypatch.setattr(selfplay, "play_episode", episode)
    try:
        for _ in range(3):
            stop.clear()
            selfplay.collect_episode(dataset, object(), stop, pause, random.Random(1), quiet_only=True)
        assert starts == [training_start(0).key, training_start(1).key, training_start(1).key]
        progress = dataset.curriculum_progress()
        assert progress["attempts"] == 2 and progress["censored"] == 1
        assert progress["completed"] == 1 and dataset.count() == 1
    finally:
        dataset.close()


def test_continuous_collection_records_durable_curriculum(tmp_path, monkeypatch):
    import chess
    from orion_core.grandmaster.dataset import Dataset
    runner = training.Improvement(tmp_path, tmp_path / "brain.json", threading.Event())
    runner.state["collected"] = 79
    monkeypatch.setattr(training, "make_brain", lambda *_args: object())
    monkeypatch.setattr(training, "refresh_weights", lambda *_args: "fixture")

    def collect(dataset, _brain, _stop, _pause, _rng, **options):
        assert options["quiet_only"] is True and options["seconds"] == 0.04
        positions = [(chess.STARTING_FEN, 0)]
        dataset.append(positions, "1/2-1/2", "fixture", pathway="opening")
        return positions, "1/2-1/2"

    monkeypatch.setattr(training, "collect_episode", collect)
    runner.collect()
    assert runner.state["phase"] == "train"
    assert runner.state["curriculum"]["completed"] == 1
    dataset = Dataset(tmp_path / "positions.sqlite3")
    try:
        assert dataset.count() == 1
    finally:
        dataset.close()


@pytest.mark.parametrize("entry", ["background", "practice", "offline"])
def test_each_collection_entrypoint_shares_the_durable_rotation(tmp_path, monkeypatch, entry):
    import chess
    from orion_core.grandmaster import search, selfplay
    from orion_core.grandmaster.curriculum import training_start
    from orion_core.grandmaster.dataset import Dataset
    from tools import train_chess
    stop, pause, starts, replies = threading.Event(), threading.Event(), [], []
    monkeypatch.setattr(selfplay, "worker_priority", lambda: None)
    monkeypatch.setattr(selfplay, "make_brain", lambda *_args: object())
    monkeypatch.setattr(selfplay, "refresh_weights", lambda *_args: "fixture")

    def episode(_brain, worker_stop, _pause, _rng, **options):
        starts.append(options["start"].key)
        if entry == "background" and len(starts) == 3:
            worker_stop.set()
        return [(chess.STARTING_FEN, 0)], "1/2-1/2"

    monkeypatch.setattr(selfplay, "play_episode", episode)
    if entry == "background":
        selfplay.selfplay_main(str(tmp_path / "brain.json"), str(tmp_path), stop, pause)
    elif entry == "practice":
        pipe = SimpleNamespace(send=replies.append, close=lambda: None)
        selfplay.practice_main(str(tmp_path / "brain.json"), str(tmp_path), 3, stop, pause, pipe)
        assert replies[0]["games"] == 3
        assert replies[0]["curriculum"]["completed"] == 3
    else:
        monkeypatch.setattr(search, "worker_priority", lambda: None)
        monkeypatch.setattr(search, "make_brain", lambda *_args: object())
        monkeypatch.setattr(search, "refresh_weights", lambda *_args: "fixture")
        monkeypatch.setattr(train_chess, "fit", lambda *_args, **_kwargs: {"status": "rejected"})
        assert train_chess.main(["--directory", str(tmp_path), "--brain", str(tmp_path / "brain.json"),
                                 "--games", "3", "--steps", "1", "--quiet-only"]) == 0
        report = json.loads((tmp_path / "last_training.json").read_text())
        assert report["games_completed"] == 3
        assert report["curriculum"]["completed"] == 3
    assert starts == [training_start(index).key for index in range(3)]
    dataset = Dataset(tmp_path / "positions.sqlite3")
    try:
        assert dataset.curriculum_progress()["completed"] == 3
        assert dataset.count() == 3
    finally:
        dataset.close()
