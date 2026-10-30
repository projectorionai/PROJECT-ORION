"""Measurement integrity: real outcomes, paired evidence, bounded resource use."""
import json
import threading
from types import SimpleNamespace
from concurrent.futures import Future

import pytest

chess = pytest.importorskip("chess")
from tools import measure_chess as measurement  # noqa: E402


def test_suite_is_legal_balanced_and_covers_every_level():
    schedule = measurement.schedule(48)
    assert len({(p["opening"], p["elo"]) for p in schedule}) == 48
    for opening in measurement.OPENINGS:
        board = chess.Board()
        for move in opening:
            board.push_uci(move)
        assert board.is_valid() and not board.is_game_over()
    assert [p["elo"] for p in schedule[:3]] == [2400, 2600, 2800]


def test_rating_uses_opponent_levels_and_leaves_extremes_unbounded():
    assert measurement.performance([(2600, 1), (2600, 0)]) == pytest.approx(2600)
    assert measurement.performance([(2400, 1), (2800, 0)]) == pytest.approx(2600)
    assert measurement.performance([(0, 1)] * 3 + [(0, 0)]) == pytest.approx(190.8485)
    assert measurement.performance([(2600, 1)]) is None
    assert measurement.performance([(2600, 0)]) is None
    assert measurement.performance([]) is None


def test_lower_rating_bracket_remains_balanced():
    plan = measurement.schedule(9, levels=(1320, 1600, 1900))
    assert [p["elo"] for p in plan[:3]] == [1320, 1600, 1900]
    assert {p["opening"] for p in plan} == {0, 1, 2}
    with pytest.raises(ValueError, match="level"):
        measurement.schedule(2, levels=())


def test_bootstrap_samples_whole_colour_pairs():
    pairs = [{"elo": 2600, "games": [{"score": 1}, {"score": 0}]} for _ in range(8)]
    summary = measurement.summarise(pairs, samples=100)
    assert summary["ci95"] == [2600, 2600]
    assert (summary["wins"], summary["draws"], summary["losses"]) == (8, 0, 8)
    assert measurement.summarise(pairs[:1])["ci95"] is None
    with pytest.raises(ValueError, match="completed colour pairs"):
        measurement.summarise([{"elo": 2600, "games": [{"score": 1}]}])


def test_completed_game_result_and_moves_can_be_replayed():
    moves = iter(["f2f3", "e7e5", "g2g4", "d8h4"])
    player = lambda board: chess.Move.from_uci(next(moves))
    result = measurement.play_game(player, player, [], threading.Event(), max_plies=4)
    assert result["result"] == "0-1"
    assert result["termination"] == "CHECKMATE"
    assert len(result["moves"]) == 4


def test_capped_or_interrupted_games_are_never_draws():
    stop = threading.Event()
    player = lambda board: next(iter(board.legal_moves))
    assert measurement.play_game(player, player, [], stop, max_plies=1) is None
    def interrupt(board):
        stop.set()
        return player(board)
    assert measurement.play_game(interrupt, player, [], stop) is None
    with pytest.raises(ValueError, match="illegal"):
        measurement.play_game(lambda _: chess.Move.null(), player, [], threading.Event())


@pytest.mark.parametrize("trigger, reason", [("memory", "low memory"), ("file", "stop marker"),
                                           ("time", "time budget"), ("probe", "memory probe failed")])
def test_watchdog_fails_closed(tmp_path, trigger, reason):
    def available():
        if trigger == "probe":
            raise OSError("unavailable")
        return 599 if trigger == "memory" else 2000
    if trigger == "file":
        (tmp_path / "STOP").touch()
    guard = measurement.Watchdog(tmp_path, 100, 600, available)
    if trigger == "time":
        guard.deadline = 0
    with guard:
        assert not guard.check()
        assert guard.stop.is_set() and guard.reason == reason
    assert not guard.thread.is_alive()


def test_opponent_cleanup_waits_for_exit_even_when_quit_fails():
    calls = []
    exited = Future()
    def quit_engine():
        raise TimeoutError()
    def close():
        calls.append("close")
        if not exited.done():
            exited.set_result(-1)
    measurement.close_opponent(SimpleNamespace(quit=quit_engine, close=close, returncode=exited))
    assert calls and exited.done()


def test_opponent_environment_does_not_inherit_credentials(monkeypatch):
    monkeypatch.setenv("PROVIDER_API_KEY", "isolated-test-value")
    monkeypatch.setenv("ORION_CONFIG_DIR", "private-config")
    monkeypatch.setenv("SYSTEMROOT", "test-system-root")
    env = measurement.child_environment()
    assert "PROVIDER_API_KEY" not in env and "ORION_CONFIG_DIR" not in env
    assert env["SYSTEMROOT"] == "test-system-root"


def test_resume_rejects_changed_inputs_or_fabricated_scores(tmp_path):
    path = tmp_path / "results.json"
    contract = {"schedule": [{"opening": 0, "elo": 2600}], "openings": [[]]}
    state = measurement.load_state(path, contract)
    path.write_text(json.dumps(state))
    assert measurement.load_state(path, contract)["pairs"] == []
    with pytest.raises(ValueError, match="inputs changed"):
        measurement.load_state(path, {**contract, "candidate": "different"})
    game = {"moves": ["f2f3", "e7e5", "g2g4", "d8h4"], "result": "0-1"}
    state["pairs"] = [{"opening": 0, "elo": 2600, "games": [
        {**game, "colour": "white", "score": 0}, {**game, "colour": "black", "score": 1}]}]
    path.write_text(json.dumps(state))
    assert len(measurement.load_state(path, contract)["pairs"]) == 1
    state["pairs"][0]["games"][0]["score"] = 1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="score differs"):
        measurement.load_state(path, contract)
