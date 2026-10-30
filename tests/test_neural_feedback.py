"""
Planner feedback: consent, provenance, what counts as a label, and exports.

The store is the only source of training data for the optional planner, so
these check the rules that keep bad labels out: nothing without consent,
nothing secret, no correction without the prompt it corrects, and — the easy
one to get wrong — a tool that returned ok is NOT a positive label.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import neural_feedback as nf  # noqa: E402

PLAN = {"summary": "Check memory", "assumptions": [],
        "steps": [{"id": "a", "tool": "resource_status", "args": {}, "depends_on": [],
                   "success": "memory figures shown"}],
        "success_criteria": ["the user sees memory use"]}
BETTER = {"summary": "Check memory and tokens", "assumptions": [],
          "steps": [{"id": "a", "tool": "resource_status", "args": {}, "depends_on": [],
                     "success": "memory figures shown"},
                    {"id": "b", "tool": "token_usage", "args": {}, "depends_on": [],
                     "success": "token totals shown"}],
          "success_criteria": ["both are shown"]}


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv(nf.ENV_KILL, raising=False)
    made = nf.FeedbackStore(tmp_path / "feedback.db", clock=_Clock())
    yield made
    made.close()


def _prompt(goal: str = "how much memory am I using") -> dict[str, str]:
    return {"system": "You are the planning component.", "user": f"Request: {goal}"}


def _plan(store, task_id="plan-1", goal="how much memory am I using", plan=PLAN):
    return store.record_plan(task_id=task_id, prompt=_prompt(goal), plan=plan,
                             base_model="qwen3:4b", adapter_version="",
                             planner_version="planner-1")


# ── consent ──────────────────────────────────────────────────────────────────

def test_nothing_is_recorded_without_consent(store):
    assert store.consent()["stored"] is False
    assert _plan(store) is None
    assert "consent" in store.last_refusal
    assert store.events() == []


def test_the_environment_can_withdraw_consent_but_never_grant_it(store, monkeypatch):
    monkeypatch.setenv(nf.ENV_KILL, "1")
    assert store.consent_enabled() is False          # "1" is not a way to switch it on
    store.set_consent(True, changed_by="user:test")
    monkeypatch.setenv(nf.ENV_KILL, "0")
    assert store.consent_enabled() is False
    assert _plan(store) is None
    monkeypatch.delenv(nf.ENV_KILL)
    assert _plan(store) is not None


def test_withdrawing_consent_with_purge_deletes_everything(store):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_acceptance(task_id="plan-1", accepted=True, source="user_explicit")
    assert store.set_consent(False, changed_by="user:test", purge=True) == 2
    assert store.events() == []
    assert _plan(store, task_id="plan-2") is None


def test_consent_changes_record_who_made_them(store):
    with pytest.raises(ValueError):
        store.set_consent(True, changed_by="")
    store.set_consent(True, changed_by="user:cli")
    assert store.consent()["changed_by"] == "user:cli"


# ── provenance and immutability ──────────────────────────────────────────────

def test_a_plan_keeps_the_exact_prompt_and_its_provenance(store):
    store.set_consent(True, changed_by="user:test")
    event_id = _plan(store)
    [event] = store.events()
    assert event["id"] == event_id
    assert event["kind"] == "plan" and event["source"] == "planner"
    assert json.loads(event["prompt_json"]) == _prompt()
    assert json.loads(event["plan_json"])["steps"][0]["tool"] == "resource_status"
    assert event["base_model"] == "qwen3:4b"
    assert event["policy_version"] == nf.POLICY_VERSION
    assert store.verify(event)


def _key_shaped_fixture() -> str:
    """A provider-key-shaped string built at run time. Written out literally it
    would (rightly) stop the public-source export, which scans test files too."""
    import random
    import string
    tail = "".join(random.Random(20260928).choice(string.ascii_letters + string.digits)
                   for _ in range(40))
    return "-".join(("sk", "proj", tail))


def test_secrets_are_never_recorded(store):
    store.set_consent(True, changed_by="user:test")
    leaked = store.record_plan(
        task_id="plan-x", prompt=_prompt(f"use {_key_shaped_fixture()}"), plan=PLAN)
    assert leaked is None and "secret" in store.last_refusal
    assert store.events() == []


def test_a_correction_needs_the_recorded_prompt_and_a_user_source(store):
    store.set_consent(True, changed_by="user:test")
    assert store.record_correction(task_id="never-planned", corrected_plan=BETTER,
                                   source="user_explicit") is None
    _plan(store)
    assert store.record_correction(task_id="plan-1", corrected_plan=BETTER,
                                   source="executor") is None
    assert store.record_correction(task_id="plan-1", corrected_plan={"steps": []},
                                   source="user_explicit") is None
    assert store.record_correction(task_id="plan-1", corrected_plan=BETTER,
                                   source="user_explicit") is not None


def test_an_edited_event_no_longer_counts(store, tmp_path):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_acceptance(task_id="plan-1", accepted=True, source="user_explicit")
    assert len(store.examples()) == 1
    store.close()
    with sqlite3.connect(tmp_path / "feedback.db") as db:
        db.execute("UPDATE events SET plan_json = ? WHERE kind = 'plan'",
                   (nf.plan_text(BETTER),))
    assert store.examples() == []


# ── what is a label ──────────────────────────────────────────────────────────

def test_a_tool_returning_ok_is_not_a_label(store):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_outcome(task_id="plan-1", tool_returned_ok=True, postcondition_verified=None,
                         receipts=[{"step_id": "a", "tool": "resource_status",
                                    "tool_returned_ok": True}])
    assert store.examples() == []


def test_a_verified_outcome_or_an_acceptance_is(store):
    store.set_consent(True, changed_by="user:test")
    _plan(store, task_id="verified")
    store.record_outcome(task_id="verified", tool_returned_ok=True, postcondition_verified=True,
                         receipts=[])
    _plan(store, task_id="accepted", goal="another request")
    store.record_acceptance(task_id="accepted", accepted=True, source="user_explicit")
    examples = {e["task_id"]: e for e in store.examples()}
    assert set(examples) == {"verified", "accepted"}
    assert examples["verified"]["label_source"] == "executor"
    assert examples["accepted"]["messages"][-1] == {"role": "assistant",
                                                    "content": nf.plan_text(PLAN)}


def test_relayed_labels_need_an_approving_review(store):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    event = store.record_acceptance(task_id="plan-1", accepted=True, source="user_relayed")
    assert store.examples() == []
    store.review(event, "approve", reviewer="user:cli")
    assert len(store.examples()) == 1
    assert store.examples(explicit_only=True) == []
    store.review(event, "reject", reviewer="user:cli")
    assert store.examples() == []


def _labels(store, **kwargs):
    return sorted((e["type"], e["label_event"]) for e in store.examples(**kwargs))


def test_the_latest_verdict_decides(store):
    """accept->reject, reject->accept, correction->reject and a rejection after
    a verified outcome each end with the user's LAST word."""
    store.set_consent(True, changed_by="user:test")
    _plan(store, task_id="accept-reject", goal="one")
    store.record_acceptance(task_id="accept-reject", accepted=True, source="user_explicit")
    store.record_acceptance(task_id="accept-reject", accepted=False, source="user_explicit")

    _plan(store, task_id="reject-accept", goal="two")
    store.record_acceptance(task_id="reject-accept", accepted=False, source="user_explicit")
    accepted = store.record_acceptance(task_id="reject-accept", accepted=True,
                                       source="user_explicit")

    _plan(store, task_id="correct-reject", goal="three")
    store.record_correction(task_id="correct-reject", corrected_plan=BETTER,
                            source="user_explicit")
    store.record_acceptance(task_id="correct-reject", accepted=False, source="user_explicit")

    _plan(store, task_id="verified-reject", goal="four")
    store.record_outcome(task_id="verified-reject", receipts=[], tool_returned_ok=True,
                         postcondition_verified=True)
    store.record_acceptance(task_id="verified-reject", accepted=False, source="user_explicit")

    assert _labels(store) == [("sft", accepted)]
    first = store.export(Path(store.path).parent / "first")
    second = store.export(Path(store.path).parent / "second")
    assert first["data_hash"] == second["data_hash"]          # repeatable
    assert first["selection_policy"] == nf.SELECTION_POLICY
    assert first["label_sources"] == {"user_explicit": 1}


def test_a_relayed_rejection_still_withdraws_a_label(store):
    """Withdrawing is the safe direction, so a rejection counts even before a
    review; a positive relayed verdict does not."""
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_acceptance(task_id="plan-1", accepted=True, source="user_explicit")
    store.record_acceptance(task_id="plan-1", accepted=False, source="user_relayed")
    assert store.examples() == []


def test_a_correction_must_meet_the_planners_own_contract(store):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    denied = {**BETTER, "steps": [{"id": "a", "tool": "shutdown_orion", "args": {},
                                   "depends_on": [], "success": "gone"}]}
    approval = {**BETTER, "steps": [{"id": "a", "tool": "web_search",
                                     "args": {"query": "x", "confirm": True},
                                     "depends_on": [], "success": "results"}]}
    cycle = {**BETTER, "steps": [{**BETTER["steps"][0], "depends_on": ["b"]},
                                 {**BETTER["steps"][1], "depends_on": ["a"]}]}
    no_criteria = {**BETTER, "success_criteria": []}
    for corrected in (denied, approval, cycle, no_criteria, {"steps": []}, "not a plan"):
        assert store.record_correction(task_id="plan-1", corrected_plan=corrected,
                                       source="user_explicit") is None
        assert "not a valid plan" in store.last_refusal
    assert [e["kind"] for e in store.events()] == ["plan"]


def test_exporting_needs_consent_and_is_remembered(store, tmp_path, capsys, monkeypatch):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_acceptance(task_id="plan-1", accepted=True, source="user_explicit")
    store.export(tmp_path / "export")
    [written] = store.exports()
    assert written["directory"] == str((tmp_path / "export").resolve())
    store.set_consent(False, changed_by="user:test")
    with pytest.raises(PermissionError):
        store.export(tmp_path / "again")
    assert not (tmp_path / "again").exists()
    store.close()
    assert nf.main(["--db", str(store.path), "consent", "off"]) == 0
    assert str((tmp_path / "export").resolve()) in capsys.readouterr().out


def test_receipts_keep_the_kind_of_evidence(store):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_outcome(task_id="plan-1", tool_returned_ok=True, postcondition_verified=None,
                         receipts=[{"step_id": "a", "tool": "resource_status",
                                    "tool_returned_ok": True, "evidence": "screen_changed"}])
    [outcome] = [e for e in store.events() if e["kind"] == "outcome"]
    assert json.loads(outcome["receipts_json"])[0]["evidence"] == "screen_changed"
    assert store.examples() == []


def test_a_correction_gives_an_sft_example_and_a_preference_pair(store):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_correction(task_id="plan-1", corrected_plan=BETTER, source="user_explicit")
    by_type = {e["type"]: e for e in store.examples()}
    assert by_type["sft"]["messages"][-1]["content"] == nf.plan_text(BETTER)
    pair = by_type["dpo"]
    assert pair["chosen"][0]["content"] == nf.plan_text(BETTER)
    assert pair["rejected"][0]["content"] == nf.plan_text(PLAN)
    assert pair["prompt"][-1]["content"] == _prompt()["user"]


# ── exports ──────────────────────────────────────────────────────────────────

def test_export_splits_by_task_chronologically(store, tmp_path):
    store.set_consent(True, changed_by="user:test")
    for index in range(10):
        _plan(store, task_id=f"t{index}", goal=f"request number {index}")
        store.record_acceptance(task_id=f"t{index}", accepted=True, source="user_explicit")
    # Word for word the same example as t3: kept once, not twice.
    _plan(store, task_id="t3-again", goal="request number 3")
    store.record_acceptance(task_id="t3-again", accepted=True, source="user_explicit")

    manifest = store.export(tmp_path / "export", holdout_fraction=0.2, replay_fraction=0.25)
    tasks = manifest["tasks"]
    assert tasks["heldout"] == ["t8", "t9"]                  # the newest
    assert tasks["replay"] == ["t0", "t1"]                   # the oldest of the rest
    assert tasks["train"] == ["t2", "t3", "t4", "t5", "t6", "t7"]
    assert manifest["counts"]["sft_train.jsonl"] == 6
    assert manifest["counts"]["dpo_train.jsonl"] == 0
    assert nf.verify_export(tmp_path / "export")["data_hash"] == manifest["data_hash"]


def test_a_prompt_in_the_held_out_split_is_removed_from_training(store, tmp_path):
    store.set_consent(True, changed_by="user:test")
    _plan(store, task_id="old", goal="same words")
    store.record_acceptance(task_id="old", accepted=True, source="user_explicit")
    _plan(store, task_id="middle", goal="different words")
    store.record_acceptance(task_id="middle", accepted=True, source="user_explicit")
    # A different plan for the same prompt, so it is not a duplicate example.
    _plan(store, task_id="new", goal="same words", plan=BETTER)
    store.record_acceptance(task_id="new", accepted=True, source="user_explicit")
    manifest = store.export(tmp_path / "export", holdout_fraction=0.33, replay_fraction=0.0)
    assert manifest["tasks"]["heldout"] == ["new"]
    assert manifest["removed_for_prompt_leakage"] == 1
    assert manifest["counts"]["sft_train.jsonl"] == 1


def test_an_export_is_a_new_immutable_snapshot(store, tmp_path):
    store.set_consent(True, changed_by="user:test")
    _plan(store)
    store.record_correction(task_id="plan-1", corrected_plan=BETTER, source="user_explicit")
    store.export(tmp_path / "export")
    with pytest.raises(FileExistsError):
        store.export(tmp_path / "export")
    (tmp_path / "export" / "dpo_train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        nf.verify_export(tmp_path / "export")


def test_the_trainer_accepts_a_real_export_and_notices_tampering(store, tmp_path):
    from tools import train_planner_adapter as trainer
    store.set_consent(True, changed_by="user:test")
    for index in range(4):
        _plan(store, task_id=f"t{index}", goal=f"request {index}")
        store.record_correction(task_id=f"t{index}", corrected_plan=BETTER,
                                source="user_explicit")
    store.export(tmp_path / "export", holdout_fraction=0.25, replay_fraction=0.0)
    for mode in ("sft", "dpo"):
        _manifest, splits = trainer.load_export(tmp_path / "export", mode)
        assert len(splits["train"]) == 3 and len(splits["heldout"]) == 1
        trainer.validate_rows(splits, mode, min_examples=1)
    with pytest.raises(trainer.Refused):
        trainer.validate_rows(trainer.load_export(tmp_path / "export", "sft")[1], "sft",
                              min_examples=20)
    path = tmp_path / "export" / "sft_train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(trainer.Refused):
        trainer.load_export(tmp_path / "export", "sft")


def test_the_command_line_is_the_explicit_channel(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(nf.ENV_KILL, raising=False)
    db = tmp_path / "cli.db"
    assert nf.main(["--db", str(db), "consent", "on"]) == 0
    seeded = nf.FeedbackStore(db)
    seeded.record_plan(task_id="plan-cli", prompt=_prompt(), plan=PLAN)
    seeded.close()
    corrected = tmp_path / "corrected.json"
    corrected.write_text(json.dumps(BETTER), encoding="utf-8")
    assert nf.main(["--db", str(db), "correct", "plan-cli", str(corrected)]) == 0
    check = nf.FeedbackStore(db)
    [correction] = [e for e in check.events() if e["kind"] == "correction"]
    assert correction["source"] == "user_explicit"
    check.close()
    assert nf.main(["--db", str(db), "export", str(tmp_path / "out"), "--explicit-only"]) == 0
    assert "data_hash" in capsys.readouterr().out
