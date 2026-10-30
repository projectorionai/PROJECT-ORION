"""New capabilities must not erase or block the existing learned routes."""
import numpy as np
import pytest

from orion_core.intent_brain import CAPACITY, FEATURES, HIDDEN, STATE_FILE, IntentBrain
from orion_core.neural import SparseMLP


def test_full_old_action_head_expands_without_losing_history_or_optimizer(tmp_path):
    old = SparseMLP(FEATURES, HIDDEN, 256, seed=5)
    old.train_batch([[1, 2], [3, 4]], [0, 255])
    labels = [f"action_{i}" for i in range(256)]
    assert old.save(tmp_path / STATE_FILE, {"labels": labels, "seeded": True,
        "observed": 8, "user_counts": {"action_0": 8}, "recent": [["user phrase", "action_0"]]})
    loaded = IntentBrain(tmp_path)
    assert loaded.net.sizes == [FEATURES, HIDDEN, CAPACITY]
    assert loaded.labels == labels and loaded.observed == 8
    assert loaded.user_counts == {"action_0": 8}
    np.testing.assert_array_equal(loaded.net.W2[:, :256], old.W2)
    np.testing.assert_array_equal(loaded.net._m["W2"][:, :256], old._m["W2"])
    np.testing.assert_array_equal(loaded.net._row_steps, old._row_steps)
    assert loaded.net._t == old._t
    assert loaded._label("new_plugin") == 256
    assert loaded.save()
    restored = IntentBrain(tmp_path)
    assert restored.labels[-1] == "new_plugin" and restored.observed == 8


def test_spare_outputs_do_not_change_existing_probability_ratios(tmp_path):
    old = SparseMLP(FEATURES, HIDDEN, 256, seed=2)
    labels = [f"action_{i}" for i in range(256)]
    assert old.save(tmp_path / STATE_FILE, {"labels": labels})
    loaded = IntentBrain(tmp_path)
    from orion_core.intent_brain import features
    expected = old.predict(features("open calculator"))
    observed = loaded.probabilities("open calculator")
    np.testing.assert_allclose(list(observed.values()), expected, rtol=2e-6, atol=1e-8)


def test_public_trainer_never_reads_personal_state_or_discovers_plugins(tmp_path, monkeypatch):
    import json
    import sys

    from orion_core import constants, dispatch_schema, intent_brain, plugin_manifest
    from tools import train_intent

    private = tmp_path / "personal"
    active = private / "brain" / STATE_FILE
    active.parent.mkdir(parents=True)
    original = b"synthetic private checkpoint must not be read or copied"
    active.write_bytes(original)
    output = tmp_path / "public-audit"
    monkeypatch.setattr(constants, "CONFIG_DIR", private)
    monkeypatch.setattr(dispatch_schema, "TOOL_DECLARATIONS", [
        {"name": "calculate", "description": "Calculate an exact arithmetic expression."}])
    monkeypatch.setattr(intent_brain, "_tool_texts", lambda: ({"calculate": "arithmetic sum"}, {}))
    monkeypatch.setattr(plugin_manifest, "discover_manifests", lambda _path: pytest.fail(
        "public training inspected installed plugins"))
    path_type = type(active)
    real_exists = path_type.exists

    def exists(path):
        if path == active:
            pytest.fail("public training inspected the personal checkpoint")
        return real_exists(path)

    monkeypatch.setattr(path_type, "exists", exists)
    monkeypatch.setattr(sys, "argv", ["train_intent", "--public-only", "--epochs", "1", "--seed", "5",
                                    "--output", str(output)])
    train_intent.main()
    report = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert report["public_only"] and not report["promoted"]
    assert report["seed"] == 5
    assert "user history" not in report["training_source"]
    assert active.read_bytes() == original
    candidate = IntentBrain(output / "candidate")
    assert candidate.observed == 0 and not candidate.user_counts and not candidate._recent
    assert candidate.labels == ["calculate"]


def test_public_trainer_cannot_promote_to_personal_state(tmp_path, monkeypatch, capsys):
    import sys

    from tools import train_intent

    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["train_intent", "--public-only", "--apply",
                                    "--output", str(output)])
    with pytest.raises(SystemExit) as rejected:
        train_intent.main()
    assert rejected.value.code == 2
    assert "--public-only cannot be combined with --apply" in capsys.readouterr().err
    assert not output.exists()


def test_default_trainer_preserves_installed_state_and_plugin_input(tmp_path, monkeypatch):
    import sys

    from orion_core import constants, plugin_manifest
    from tools import train_intent

    private = tmp_path / "personal"
    active = private / "brain" / STATE_FILE
    active.parent.mkdir(parents=True)
    original = b"synthetic active checkpoint"
    active.write_bytes(original)
    output = tmp_path / "personal-audit"
    discovered = []
    monkeypatch.setattr(constants, "CONFIG_DIR", private)

    def invalid_plugins(path):
        discovered.append(path)
        return [], ["synthetic invalid manifest"]

    monkeypatch.setattr(plugin_manifest, "discover_manifests", invalid_plugins)
    monkeypatch.setattr(sys, "argv", ["train_intent", "--output", str(output)])
    with pytest.raises(RuntimeError, match="synthetic invalid manifest"):
        train_intent.main()
    assert discovered == [private / "custom_tools"]
    for name in ("baseline", "candidate"):
        assert (output / name / STATE_FILE).read_bytes() == original
    assert active.read_bytes() == original


@pytest.mark.parametrize("regression", ["fused", "ood", "none"])
def test_trainer_rejects_standalone_gain_that_regresses_routing_or_ood(
        tmp_path, monkeypatch, regression):
    import importlib.util
    import json
    import sys

    from orion_core import dispatch_schema, intent_brain, tool_resolver
    from tools import train_intent

    cases = tmp_path / "synthetic_holdout.py"
    cases.write_text("def holdout_cases():\n    return [('first', 'alpha', ()), "
                     "('second', 'alpha', ())]\n", encoding="utf-8")
    original_spec = importlib.util.spec_from_file_location
    monkeypatch.setattr(train_intent.importlib.util, "spec_from_file_location",
                        lambda name, _path: original_spec(name, cases))
    declarations = [{"name": name, "description": name}
                    for name in ("alpha", "beta", "gamma")]
    monkeypatch.setattr(dispatch_schema, "TOOL_DECLARATIONS", declarations)
    monkeypatch.setattr(intent_brain, "_tool_texts", lambda: ({}, {}))
    monkeypatch.setattr(tool_resolver, "lexical_scores", lambda text, _decl:
                        {"alpha": 0.0, "beta": 0.4 if text == "first" else 0.0,
                         "gamma": 0.0})

    class SyntheticBrain:
        def __init__(self, directory, seed=0):
            self.path = directory / STATE_FILE
            self.seeded = self.path.exists()
            self.labels = [d["name"] for d in declarations]
            self.user_counts = {}
            self.trained = False

        def seed(self, *_args, **_kwargs):
            self.seeded = True
            self.trained = self.path.parent.name == "candidate"
            return 2

        def save(self):
            self.path.write_bytes(b"candidate" if self.trained else b"baseline")
            return True

        def probabilities(self, text):
            if text == "first":
                values = ([0.45, 0.44, 0.11] if self.trained and regression == "fused"
                          else [0.6, 0.3, 0.1])
            elif text == "second":
                values = [0.6, 0.3, 0.1] if self.trained else [0.3, 0.6, 0.1]
            else:
                values = ([0.9, 0.05, 0.05] if self.trained and regression == "ood"
                          else [0.34, 0.33, 0.33])
            return dict(zip(self.labels, values))

        def top(self, text, k=10):
            return sorted(self.probabilities(text).items(), key=lambda row: -row[1])[:k]

    monkeypatch.setattr(intent_brain, "IntentBrain", SyntheticBrain)
    output = tmp_path / regression
    monkeypatch.setattr(sys, "argv", ["train_intent", "--public-only", "--epochs", "1",
                                    "--output", str(output)])
    train_intent.main()
    report = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert report["after"]["hits"]["1"] > report["before"]["hits"]["1"]
    assert report["top1_regressions"] == 0
    assert report["eligible"] == (regression == "none"), (
        "Standalone accuracy concealed a routing/OOD regression")
    assert not report["promoted"]
    if regression == "none":
        assert not report["promotion_gate_reasons"]
    else:
        assert any(regression in reason for reason in report["promotion_gate_reasons"])


def test_cold_training_gate_preserves_trusted_user_boost_without_learning(monkeypatch):
    from orion_core import intent_brain, spelling, tool_resolver
    from tools.train_intent import cold_routing_scores

    declarations = [{"name": name} for name in ("alpha", "beta")]
    lexical = lambda _text, _declarations: {"alpha": 10.0, "beta": 0.0}
    monkeypatch.setattr(tool_resolver, "lexical_scores", lexical)
    monkeypatch.setattr(tool_resolver, "hybrid_scores", lexical)
    monkeypatch.setattr(spelling, "correct", lambda text: text)

    class ExperiencedBrain:
        def __init__(self):
            self.user_counts = {"beta": intent_brain.TRUSTED_EXAMPLES}

        def probabilities(self, _text):
            return {"alpha": 0.1, "beta": 0.9}

        def resembles_user_phrasing(self, _text, label):
            return label == "beta"

        def ensure_known(self, _declarations):
            pass

    model = ExperiencedBrain()
    expected = intent_brain.combined_scores("synthetic request", declarations, model)
    assert expected["beta"] == pytest.approx(9.9)
    assert cold_routing_scores(model, "synthetic request", declarations) == expected
