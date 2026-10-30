"""Learning continuity, data integrity, and factorized chess evaluation."""
import io
import random

import numpy as np
import pytest

from orion_core.neural import MLP, SparseMLP


@pytest.mark.parametrize("sparse", [False, True])
def test_training_resumes_identically_after_checkpoint(sparse):
    net = SparseMLP(16, 6, 3, seed=7) if sparse else MLP([2, 6, 1], seed=7)
    x, y = ([[1, 3], [2, 7]], [0, 2]) if sparse else ([[0.2, -0.3], [0.9, 0.1]], [[0.8], [-0.4]])
    for _ in range(12):
        net.train_batch(x, y)
    loaded, _ = type(net).from_bytes(net.to_bytes())
    for _ in range(5):
        net.train_batch(x, y)
        loaded.train_batch(x, y)
    np.testing.assert_array_equal(net.forward(x), loaded.forward(x))
    assert net._t == loaded._t == 17


def test_new_sparse_feature_gets_its_own_adam_clock():
    net = SparseMLP(16, 4, 2, seed=3, l2=0)
    net.b1[:] = 1  # every feature has a nonzero gradient
    for _ in range(100):
        net.train_batch([[1]], [0])
    before = net.table[15].copy()
    net.train_batch([[15]], [1])
    change = np.abs(net.table[15] - before)
    assert net._row_steps[15] == 1
    assert net._row_steps[1] == 100
    assert np.max(change) <= net.lr * 1.001


@pytest.mark.parametrize("sparse", [False, True])
def test_invalid_weights_do_not_modify_network(sparse):
    net = SparseMLP(16, 4, 2) if sparse else MLP([2, 4, 1])
    x, y = ([[1]], [0]) if sparse else ([[0.2, 0.3]], [[0.8]])
    before = net.forward(x).copy()
    for weight in ([-1], [float("nan")], [0]):
        with pytest.raises(ValueError):
            net.train_batch(x, y, weights=weight)
    np.testing.assert_array_equal(before, net.forward(x))
    assert net.stats.steps == 0


def test_corrupt_optimizer_is_rejected(tmp_path):
    net = SparseMLP(16, 4, 2)
    with np.load(io.BytesIO(net.to_bytes()), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["tv"][0, 0] = -1
    path = tmp_path / "corrupt.npz"
    np.savez_compressed(path, **arrays)
    assert SparseMLP.load(path) is None


def test_legacy_sparse_checkpoint_still_loads():
    net = SparseMLP(16, 4, 2)
    net.train_batch([[1, 2]], [1])
    loaded, _ = SparseMLP.from_bytes(net.to_bytes(include_optimizer=False))
    np.testing.assert_array_equal(net.predict([1, 2]), loaded.predict([1, 2]))
    assert loaded._t == 0


def test_factorized_training_folds_exactly_into_fast_inference(tmp_path):
    chess = pytest.importorskip("chess")
    torch = pytest.importorskip("torch")
    from orion_core.grandmaster.nnue import HalfKP, batch
    from orion_core.grandmaster.weights import Inference, export_weights
    torch.manual_seed(17)
    model = HalfKP()
    with torch.no_grad():
        model.shared.weight[1:].normal_(0, 0.025)
    path = tmp_path / "factorized.nnue"
    export_weights(model.state_dict(), path)
    inference = Inference.load(path)
    restored = HalfKP()
    restored.load_nnue(path)
    board, rng = chess.Board(), random.Random(12)
    for _ in range(60):
        ours, theirs, _target = batch([(board.fen(), 0, 0)])
        with torch.no_grad():
            result = model(ours, theirs)
            assert abs(result.item() * 300 - inference.evaluate(board)) < 0.51
            torch.testing.assert_close(result, restored(ours, theirs))
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))


def test_search_features_match_training_features_across_positions():
    chess = pytest.importorskip("chess")
    from orion_core.grandmaster.weights import feature_sets, features

    board, rng = chess.Board(), random.Random(19)
    for _ in range(150):
        white, black = feature_sets(board)
        assert white == set(features(board, chess.WHITE))
        assert black == set(features(board, chess.BLACK))
        if board.is_game_over(claim_draw=True):
            board = chess.Board()
        else:
            board.push(rng.choice(list(board.legal_moves)))
    # A king move changes every feature row in that perspective; promotions
    # and en passant alter piece type or remove a pawn off the destination.
    for fen in (
        "8/P7/8/8/8/8/7p/4K2k w - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ):
        board = chess.Board(fen)
        white, black = feature_sets(board)
        assert white == set(features(board, chess.WHITE))
        assert black == set(features(board, chess.BLACK))


def test_shared_features_learn_across_king_contexts():
    torch = pytest.importorskip("torch")
    from orion_core.grandmaster.nnue import HalfKP
    model = HalfKP()
    # Same piece-square, different king. The unseen king row stays untouched,
    # but learns through the shared factor.
    first, other = torch.tensor([[1]]), torch.tensor([[1 + 640]])
    with torch.no_grad():
        model.bias.fill_(0.5)
    before = model.accumulate(other).detach().clone()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    model.accumulate(first).sum().backward()
    optimiser.step()
    with torch.no_grad():
        # Cancel the common bias step to isolate the shared piece-square lesson.
        model.bias.add_(0.01)
        after = model.accumulate(other)
    assert torch.all(after < before)


def test_new_chess_network_already_understands_material():
    chess = pytest.importorskip("chess")
    torch = pytest.importorskip("torch")
    from orion_core.grandmaster.nnue import HalfKP, batch
    model = HalfKP()
    board = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
    for turn, expected in ((chess.WHITE, 900), (chess.BLACK, -900)):
        board.turn = turn
        # e-file queen checks Black, which is legal only with Black to move;
        # place it on a2 for both perspectives instead.
        board.remove_piece_at(chess.E2)
        board.set_piece_at(chess.A2, chess.Piece(chess.QUEEN, chess.WHITE))
        ours, theirs, _ = batch([(board.fen(), expected, 0)])
        with torch.no_grad():
            assert model(ours, theirs).item() * 300 == pytest.approx(expected, abs=0.001)


def test_positional_training_preserves_material_path_on_resume(tmp_path):
    chess = pytest.importorskip("chess")
    torch = pytest.importorskip("torch")
    from orion_core.grandmaster.nnue import HalfKP, batch
    from orion_core.grandmaster.weights import export_weights
    model = HalfKP()
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
    ours, theirs, target = batch([(chess.STARTING_FEN, 100, 1)])
    before = model.output.bias.detach().clone()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(ours, theirs), target)
    loss.backward()
    model.constrain_material(gradients=True)
    optimiser.step()
    model.constrain_material()
    assert not torch.equal(before, model.output.bias)
    for name, index, anchor in model._material_anchors:
        assert torch.equal(model.get_parameter(name)[index], anchor)
    path = tmp_path / "trained.nnue"
    export_weights(model.state_dict(), path)
    restored = HalfKP()
    restored.load_nnue(path)
    assert restored._material_anchors
    with torch.no_grad():
        torch.testing.assert_close(model(ours, theirs), restored(ours, theirs))


def test_heldout_evidence_counts_distinct_positions(tmp_path):
    chess = pytest.importorskip("chess")
    from orion_core.grandmaster.dataset import Dataset
    dataset = Dataset(tmp_path / "data.sqlite3")
    try:
        for _ in range(20):
            dataset.append([(chess.STARTING_FEN, 20)] * 60, "1/2-1/2", "test")
        assert len(dataset.validation_rows(128, random.Random(1))) == 1
        with pytest.raises(ValueError):
            dataset.append([("8/8/8/8/8/8/8/8 w - - 0 1", 0)], "1-0", "test")
        with pytest.raises(ValueError):
            dataset.append([(chess.STARTING_FEN, float("nan"))], "1-0", "test")
        assert dataset.count() == 1200
    finally:
        dataset.close()


def test_capped_practice_does_not_invent_draw_training(tmp_path, monkeypatch):
    pytest.importorskip("chess")
    from orion_core import chess_brain
    brain = chess_brain.ChessBrain(tmp_path / "brain.json")
    monkeypatch.setattr(chess_brain, "PRACTICE_MAX_PLIES", 1)
    brain.practice(1, seconds_per_move=0.001)
    assert brain.practice_games == 0
    assert brain.value_positions == 0


def test_illegal_trajectory_is_not_learned(tmp_path):
    pytest.importorskip("chess")
    from orion_core.chess_brain import ChessBrain
    brain = ChessBrain(tmp_path / "brain.json")
    assert brain.train_intuition(["e2e4", "e7e5", "e4e7"], 1.0) == 0
    assert brain.value_positions == 0


def test_chess_replay_survives_restart(tmp_path):
    pytest.importorskip("chess")
    from orion_core.chess_brain import ChessBrain
    brain = ChessBrain(tmp_path / "brain.json")
    assert brain.train_intuition(["f2f3", "e7e5", "g2g4", "d8h4"], 0.0) == 4
    restored = ChessBrain(tmp_path / "brain.json")
    assert list(restored._value_replay.items) == list(brain._value_replay.items)
    assert restored.value_net._t == brain.value_net._t


def test_local_search_adopts_only_promoted_model(tmp_path):
    chess = pytest.importorskip("chess")
    pytest.importorskip("torch")
    import json

    from orion_core.chess_brain import ChessBrain
    from orion_core.grandmaster.nnue import HalfKP
    from orion_core.grandmaster.search import make_brain
    from orion_core.grandmaster.weights import export_weights
    directory = tmp_path / "grandmaster"
    candidate = directory / "candidate.nnue"
    digest = export_weights(HalfKP().state_dict(), candidate)
    local = ChessBrain(tmp_path / "brain.json")
    local.choose_move(chess.Board(), 0.005, max_depth=1)
    assert local.nnue is None
    # Model promotion itself has separate statistical gate tests. Here the
    # fixture supplies its resulting manifest to exercise local adoption.
    candidate.rename(directory / (digest + ".nnue"))
    (directory / "active.json").write_text(json.dumps({"sha256": digest}))
    local.choose_move(chess.Board(), 0.005, max_depth=1)
    assert local._nnue_version == digest + ".nnue"
    assert local._eval_stm(chess.Board()) == local.nnue.evaluate(chess.Board())
    clone = local._clone_for_search()
    assert clone.nnue is not local.nnue
    assert clone.nnue.tensors is local.nnue.tensors
    isolated = make_brain(str(tmp_path / "brain.json"))
    isolated.choose_move(chess.Board(), 0.005, max_depth=1)
    assert isolated.nnue is None  # arena and remote supervisor own their model


@pytest.mark.parametrize("quiet, existing", [(False, True), (True, True), (True, False)])
def test_offline_training_cli_preserves_objective_checkpoint(tmp_path, monkeypatch, quiet, existing):
    """A quiet pass must never resume or replace an all-position checkpoint."""
    import json
    from orion_core.grandmaster import search, weights
    from tools import train_chess

    monkeypatch.setattr(search, "worker_priority", lambda: None)
    legacy = tmp_path / "training.nnue"
    static = tmp_path / "training-quiet-v2.nnue"
    legacy.write_bytes(b"legacy-objective")
    if existing:
        static.write_bytes(b"quiet-objective")
    monkeypatch.setattr(weights, "active_model", lambda _directory: legacy)
    calls = []

    def rejected_fit(directory, initial, **kwargs):
        calls.append((directory, initial, kwargs))
        return {"status": "rejected"}

    monkeypatch.setattr(train_chess, "fit", rejected_fit)
    arguments = ["--directory", str(tmp_path), "--brain", str(tmp_path / "brain.json"),
                 "--games", "0", "--steps", "1", "--compare-baseline"]
    if quiet:
        arguments.append("--quiet-only")
    assert train_chess.main(arguments) == 0
    assert len(calls) == 2
    for directory, initial, settings in calls:
        assert directory == tmp_path
        assert initial == (static if quiet and existing else None if quiet else legacy)
        assert settings["quiet_only"] is quiet
    report = json.loads((tmp_path / "last_training.json").read_text())
    assert report["label_policy"] == ("quiet-v2" if quiet else "all-positions-v1")
    assert legacy.read_bytes() == b"legacy-objective"
    if existing:
        assert static.read_bytes() == b"quiet-objective"
    else:
        assert not static.exists()
