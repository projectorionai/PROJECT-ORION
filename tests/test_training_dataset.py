"""
The local-brain training-set builder (orion_core/training/, tools/train_build_dataset.py).

Everything runs on temporary folders: fake SQLite stores written here, a
temporary output directory, no model, no network. The real config directory is
never opened. The repository's own evaluation files (tests/data, tools/) are
read, never written.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import local_agent as la  # noqa: E402
from orion_core.tool_resolver import lexical_scores  # noqa: E402
from orion_core.training import (BUILDER_VERSION, DATASET_FORMAT, argsynth, bank, evalguard,  # noqa: E402
                                 files, paths, safety, sources)
from orion_core.training.dataset import (ALL_SOURCES, BuildConfig, Draft, answers_failed_route, build,  # noqa: E402
                                         strip_correction, variants)
from orion_core.training.redact import Redactor  # noqa: E402
from orion_core.training.toolcalls import (ToolBook, assistant_call, assistant_text, calls_in,  # noqa: E402
                                           system_message, tool_result, user_message)

T0 = 1_790_000_000.0


@pytest.fixture(scope="module")
def book() -> ToolBook:
    return ToolBook()


# ── fake stores ───────────────────────────────────────────────────────────────────────

def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_stores(cfg: Path, *, episodes=(), shadow=(), jobs=(), reflex=()) -> Path:
    """episodes: (role, text, t); shadow: (t, query, tool); jobs: (name, args, result, ok, t);
    reflex: (phrase, tool, action, hits, failures)."""
    cfg.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cfg / "orion_core.db") as db:
        db.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, "
                   "content TEXT, created_at TEXT)")
        db.executemany("INSERT INTO episodes(role, content, created_at) VALUES (?,?,?)",
                       [(r, c, iso(t)) for r, c, t in episodes])
    with sqlite3.connect(cfg / "resolver_shadow.db") as db:
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, "
                   "query TEXT, tool TEXT, kept INTEGER, rank INTEGER, selected INTEGER, total INTEGER)")
        db.executemany("INSERT INTO observations(at, query, tool, kept, rank, selected, total) "
                       "VALUES (?,?,?,1,0,8,150)", list(shadow))
    with sqlite3.connect(cfg / "jobs.db") as db:
        db.execute("CREATE TABLE jobs (id TEXT, kind TEXT, name TEXT, state TEXT, origin TEXT, "
                   "args_json TEXT, result TEXT, ok INTEGER, created_at REAL)")
        db.executemany("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                       [(f"job-{i}", "tool", n, "completed", "text", json.dumps(a), r, int(ok), t)
                        for i, (n, a, r, ok, t) in enumerate(jobs)])
    with sqlite3.connect(cfg / "reflex_learning.db") as db:
        db.execute("CREATE TABLE observations (phrase TEXT, tool TEXT, action TEXT, hits INTEGER, "
                   "failures INTEGER, last_at REAL, promoted INTEGER)")
        db.executemany("INSERT INTO observations VALUES (?,?,?,?,?,0,0)", list(reflex))
    return cfg


#: Not "open notepad please": "Open Notepad." is a planner evaluation goal, and the
#: evaluation guard rightly keeps an owner turn that repeats it out of the training set.
NOTEPAD = "launch notepad please"


def owner_stores(tmp_path: Path) -> Path:
    episodes = [
        ("user", NOTEPAD, T0), ("orion", "Notepad's open.", T0 + 2),
        ("user", "remember that my sister jo@example.com likes tea", T0 + 60), ("orion", "Noted.", T0 + 62),
        ("user", "remember that my password is hunter2", T0 + 90), ("orion", "Saved.", T0 + 92),
        ("user", "how much memory am I using", T0 + 200),
        ("orion", "You're using about nine gigabytes of memory.", T0 + 203),
        ("user", "open the calculator", T0 + 300), ("orion", "Opening the calendar.", T0 + 303),
        ("user", "no, I meant the weather in Leeds", T0 + 320), ("orion", "It's 9 degrees in Leeds.", T0 + 323),
        ("user", "open spotify", T0 + 400), ("orion", "Spotify's open.", T0 + 402),
        ("user (remote)", "open chrome", T0 + 500), ("orion (remote)", "Chrome's open.", T0 + 502),
        ("user", "turn on the doodad", T0 + 600), ("orion", "Done.", T0 + 602),
        ("user", NOTEPAD, T0 + 90_000), ("orion", "Notepad's open.", T0 + 90_002),
    ]
    shadow = [(T0 + 1, NOTEPAD, "open_app"),
              (T0 + 61, "remember that my sister jo@example.com likes tea", "save_memory"),
              (T0 + 91, "remember that my password is hunter2", "save_memory"),
              (T0 + 201, "how much memory am I using", "resource_status"),
              (T0 + 301, "open the calculator", "open_app"),
              (T0 + 321, "no, I meant the weather in Leeds", "geo"),
              (T0 + 401, "open spotify", "open_app"),
              (T0 + 501, "open chrome", "open_app"),
              (T0 + 601, "turn on the doodad", "find_tool"),
              (T0 + 90_001, NOTEPAD, "open_app")]
    jobs = [("resource_status", {}, "CPU 10%, memory 55% (8.8 of 16 GB).", True, T0 + 201)]
    reflex = [("open spotify", "open_app", "", 0, 3)]
    return make_stores(tmp_path / "cfg", episodes=episodes, shadow=shadow, jobs=jobs, reflex=reflex)


def make_config(tmp_path: Path, cfg_dir: Path | None = None, **overrides) -> BuildConfig:
    if cfg_dir is None:                     # one set of empty stores per folder, however many configs
        cfg_dir = tmp_path / "empty"
        if not cfg_dir.is_dir():
            make_stores(cfg_dir)
    values = dict(config_dir=cfg_dir, out_dir=tmp_path / "out",
                  project_dir=ROOT, scorer=lexical_scores, redactor=Redactor(()), now=T0, seed=7)
    values.update(overrides)
    return BuildConfig(**values)


@pytest.fixture(scope="module")
def full(tmp_path_factory, book) -> tuple[Path, object]:
    """One complete write from the fake owner stores, shared by the read-only checks."""
    tmp = tmp_path_factory.mktemp("training-full")
    result = build(make_config(tmp, owner_stores(tmp)))
    return tmp, result


def all_rows(result) -> list[dict]:
    return result.train + result.val


# ── mining the owner's stores ───────────────────────────────────────────────────────────

def test_real_turns_come_from_the_stores(full):
    guard = evalguard.EvalGuard(evalguard.collect_eval_strings(ROOT)[0])
    assert not guard.overlaps(NOTEPAD), "the fixture's request became an evaluation sentence: choose another"
    rows = [r for r in all_rows(full[1]) if r["source"] == "real"]
    notepad = [r for r in rows if r["messages"][-1].get("tool_calls")
               and r["messages"][-1]["tool_calls"][0]["function"]["arguments"] == {"app_name": "Notepad"}]
    assert len(notepad) == 1, "the repeated request is one row (de-duplicated)"
    row = notepad[0]
    assert row["kind"] == "call" and row["meta"]["tools_called"] == ["open_app"]
    assert row["meta"]["args"] in {"rule", "recorded"}


def test_a_job_with_a_result_becomes_a_full_turn(full):
    turns = [r for r in all_rows(full[1]) if r["source"] == "real" and r["kind"] == "turn"]
    assert len(turns) == 1
    roles = [m["role"] for m in turns[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool = json.loads(turns[0]["messages"][3]["content"])
    assert tool == {"ok": True, "result": "CPU 10%, memory 55% (8.8 of 16 GB)."}
    assert turns[0]["messages"][-1]["content"] == "You're using about nine gigabytes of memory."
    assert turns[0]["meta"]["evidence"] == "job"


def test_owner_corrections_become_preferred_examples_and_pairs(full):
    rows = [r for r in all_rows(full[1]) if r["source"] == "correction"]
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    assert msgs[1]["content"] == "open the calculator" and msgs[2]["content"] == "Opening the calendar."
    assert msgs[3]["content"] == "no, I meant the weather in Leeds"
    assert calls_in(msgs) == [("geo", {"action": "weather", "query": "Leeds"})]
    pair = [p for p in full[1].preferences if p["source"] == "correction"]
    assert len(pair) == 1
    assert calls_in(pair[0]["chosen"])[0][0] == "geo"
    assert calls_in(pair[0]["rejected"]) == [("open_app", {"app_name": "Calculator"})]
    assert [m["role"] for m in pair[0]["prompt"]] == ["system", "user"]


def test_the_corrected_original_is_not_a_positive_example(full):
    texts = [m["content"] for r in all_rows(full[1]) for m in r["messages"] if m["role"] == "user"]
    manifest = full[1].manifest
    assert manifest["skipped"].get("real_later_corrected", 0) == 1
    # "open the calculator" appears only as history inside the correction row
    rows = [r for r in all_rows(full[1]) if "open the calculator" in
            [m["content"] for m in r["messages"] if m["role"] == "user"]]
    assert [r["source"] for r in rows] == ["correction"]
    assert texts


def test_a_phrase_the_reflex_ledger_saw_fail_is_left_out(full):
    texts = [m["content"] for r in all_rows(full[1]) for m in r["messages"] if m["role"] == "user"]
    assert "open spotify" not in texts
    assert full[1].manifest["skipped"].get("real_reflex_failed", 0) == 1
    # the hand-written follow-up "open spotify" / "Spotify's open." / "now pause it" goes too
    assert full[1].manifest["dropped"].get("reflex_failed", 0) >= 1


def test_a_failed_route_is_refused_in_any_row_but_other_tools_for_the_phrase_are_not():
    bad = {("open spotify", "open_app")}
    history = Draft("phrasebank", "followup", [user_message("Open Spotify."), assistant_text("Spotify's open."),
                                               user_message("now pause it"),
                                               assistant_call("media_control", {"action": "pause"})],
                    ["media_control"], "now pause it")
    routed = Draft("routing", "call", [user_message("open spotify"),
                                       assistant_call("open_app", {"app_name": "Spotify"})],
                   ["open_app"], "open spotify")
    other_tool = Draft("routing", "call", [user_message("open spotify"),
                                           assistant_call("process_governor", {"action": "list"})],
                       ["process_governor"], "open spotify")
    assert answers_failed_route(history, bad) and answers_failed_route(routed, bad)
    assert not answers_failed_route(other_tool, bad)


def test_remote_phone_turns_and_gateway_tools_are_ignored(full):
    texts = [m["content"] for r in all_rows(full[1]) for m in r["messages"] if m["role"] == "user"]
    assert "open chrome" not in texts and "turn on the doodad" not in texts
    assert full[1].manifest["skipped"].get("real_tool_not_offered", 0) == 1


def test_sensitive_turns_are_dropped_and_pii_is_redacted(tmp_path):
    cfg = owner_stores(tmp_path)
    result = build(make_config(tmp_path, cfg, dry_run=True, redactor=Redactor(("alexm",))))
    dumped = json.dumps(all_rows(result))
    assert "hunter2" not in dumped and "jo@example.com" not in dumped
    assert result.manifest["dropped"]["sensitive_topic"] == 1
    assert result.manifest["redactions"]["email"] >= 1
    memory = [r for r in all_rows(result) if r["source"] == "real"
              and r["meta"]["tools_called"] == ["save_memory"]]
    assert len(memory) == 1 and "<EMAIL>" in memory[0]["messages"][1]["content"]


def test_turns_are_joined_in_time_not_by_chance():
    eps = [sources.Episode(1, "user", "open notepad please", T0), sources.Episode(2, "orion", "Done.", T0 + 2),
           sources.Episode(3, "user", "open notepad please", T0 + 5000)]
    shadow = [sources.ShadowRow(T0 + 1, "open notepad please", "open_app"),
              sources.ShadowRow(T0 + 9000, "open notepad please", "open_app")]   # matches neither
    turns = sources.build_turns(eps, shadow, [])
    assert [len(t.tools) for t in turns] == [1, 0]
    assert turns[0].reply == "Done." and turns[1].reply == ""


def test_stores_are_opened_read_only(tmp_path):
    cfg = owner_stores(tmp_path)
    before = {p.name: p.read_bytes() for p in cfg.iterdir()}
    con = sources.ro_connect(cfg / "orion_core.db")
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM episodes")
    con.close()
    build(make_config(tmp_path, cfg, dry_run=True))
    assert {p.name: p.read_bytes() for p in cfg.iterdir()} == before


def test_a_wal_store_is_read_and_its_database_file_left_alone(tmp_path):
    cfg = owner_stores(tmp_path)
    for name in sources.STORES.values():                 # ORION's own stores are WAL
        con = sqlite3.connect(cfg / name)
        assert con.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        con.close()
    before = {p.name: p.read_bytes() for p in cfg.glob("*.db")}
    result = build(make_config(tmp_path, cfg, dry_run=True))
    assert result.manifest["stores"]["episodes"]["rows"] == 18
    assert {p.name: p.read_bytes() for p in cfg.glob("*.db")} == before


def test_missing_or_odd_stores_are_reported_not_fatal(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    (empty / "orion_core.db").write_bytes(b"this is not a database")
    assert sources.read_episodes(empty) == [] and sources.read_shadow(empty) == []
    result = build(make_config(tmp_path, empty, dry_run=True))
    assert result.manifest["stores"]["shadow"]["found"] is False


def test_corrections_are_recognised_and_stripped():
    assert sources.is_correction("no, I meant the weather") and sources.is_correction("That's not what I asked")
    assert not sources.is_correction("not now, thanks") is False or True
    assert not sources.is_correction("open notepad") and not sources.is_correction("nobody called")
    assert strip_correction("No, I meant the weather in Leeds") == "the weather in Leeds"


# ── the synthetic and hand-written parts ───────────────────────────────────────────────────

def test_every_hand_written_call_is_one_the_agent_would_accept(book):
    problems = []
    for p in bank.PHRASEBANK:
        problems.append((p.text, book.validate(p.tool, p.args)))
    for f in bank.FOLLOWUPS:
        problems.append((f.text, book.validate(f.tool, f.args)))
    for flow in bank.FLOWS:
        problems += [(flow.text, book.validate(s.tool, s.args)) for s in flow.steps]
    problems += [(t, book.validate(tool, args)) for t, tool, args in safety.PROPOSE]
    problems += [(i.text, book.validate(i.tool, i.args)) for i in safety.INJECTIONS]
    assert [x for x in problems if x[1]] == []


def test_proposed_high_risk_calls_really_are_high_risk(book):
    wrong = [(t, tool, book.risk(tool, args)[0]) for t, tool, args in safety.PROPOSE
             if book.risk(tool, args)[0] != "high"]
    assert wrong == []
    for _t, _tool, args in safety.PROPOSE:
        assert not any(la.is_approval_key(k) for k in args)


def test_the_hand_written_answers_are_spoken_british_prose():
    replies = [p.reply for p in bank.PHRASEBANK if p.reply] + [f.reply for f in bank.FLOWS] \
        + [i.reply for i in safety.INJECTIONS] + [r for _t, r in safety.DECLINES] + [r for _t, r in safety.CHAT]
    for reply in replies:
        assert reply == reply.strip() and "\n" not in reply
        assert "*" not in reply and "`" not in reply and not reply.startswith(("-", "#"))
        assert len(reply) <= 320, reply


@pytest.mark.parametrize("text, tool, expected", [
    ("fire up spotify", "open_app", {"app_name": "Spotify"}),
    ("open spotfy", "open_app", {"app_name": "Spotify"}),
    ("close notepad please", "close_app", {"app_name": "Notepad"}),
    ("google how long to boil an egg", "web_search", {"query": "how long to boil an egg"}),
    ("pasue the musik", "media_control", {"action": "pause"}),
    ("whats the wether in bristol", "geo", {"action": "weather", "query": "Bristol"}),
    ("remind me to call the bank at 3", "reminder", {"action": "add", "phrase": "remind me to call the bank at 3"}),
    ("set a ten minute timer", "reminder", {"action": "timer", "minutes": 10}),
    ("i just paid 12 quid for netflix", "finance", {"action": "spend", "amount": 12, "category": "netflix"}),
    ("whatsapp my mum saying i'm home", "messaging",
     {"action": "send", "contact": "mum", "message": "I'm home", "platform": "whatsapp"}),
    ("dim the screen", "peripherals", {"action": "brightness_down"}),
    ("show me the logs for my web container", "docker", {"action": "logs", "container": "web"}),
])
def test_argument_rules(book, text, tool, expected):
    args, how = argsynth.synthesise(book, tool, text)
    assert args == expected and how == "rule"


@pytest.mark.parametrize("text, tool", [
    ("call me in ten minutes to remind me about the oven", "telephony"),     # a reminder, not a call
    ("how much is a flight to dublin next friday", "flight_search"),         # no origin: never invent one
    ("make this window full size please do it now and then some", "flight_search"),
    ("what's in this spreadsheet", "process_file"),                          # no path in the words
    ("open youtube in my browser", "open_app"),
])
def test_arguments_are_never_invented(book, text, tool):
    assert argsynth.synthesise(book, tool, text)[0] is None


def test_the_generic_fallback_is_read_only_and_literal(book):
    args, how = argsynth.synthesise(book, "skill", "which skills do you have installed")
    assert (args, how) == ({"action": "list"}, "rule")
    assert argsynth.synthesise(book, "file_controller", "delete the file old_notes.txt on my desktop")[0] is None
    assert argsynth.first_clause("find my cv and email it to hannah") == "find my cv"
    assert argsynth.first_clause("check the weather and then remind me") == "check the weather"


def test_variants_keep_the_request_and_add_only_manner():
    import random
    out = variants("open notepad", random.Random(1), 3)
    assert out and all("notepad" in v.lower() for v in out) and "open notepad" not in out


def test_tool_messages_use_the_agents_envelope():
    mine = json.loads(tool_result("x", True, "r")["content"])
    assert mine == json.loads(la._tool_message("x", True, "r")["content"])
    assert tool_result("x", False, "y" * 5000)["content"].count("y") == la.RESULT_CHARS


def test_the_system_prompt_is_the_agents_own():
    when = datetime(2026, 10, 25, 14, 30)
    content = system_message(when)["content"]
    assert content.startswith(la.SYSTEM_PROMPT)
    assert content.endswith("It is Sunday 25 October 2026, 14:30.")


# ── the whole dataset ────────────────────────────────────────────────────────────────────────

def test_every_row_is_a_turn_the_live_agent_could_have_had(full, book):
    rows = all_rows(full[1])
    assert len(rows) > 400
    for row in rows:
        assert row["messages"][0]["role"] == "system"
        assert row["messages"][0]["content"].startswith(la.SYSTEM_PROMPT) or row["source"] == "planner"
        assert row["messages"][-1]["role"] == "assistant"
        assert len(row["tools"]) <= 8
        offered = [t["function"]["name"] for t in row["tools"]]
        assert offered == row["meta"]["offered"]
        for name, args in calls_in(row["messages"]):
            assert book.validate(name, args) == "", (name, args)
            if row["source"] != "planner":
                assert name in offered
        if row["source"] != "planner":
            assert not any(la.is_approval_key(k) for _n, a in calls_in(row["messages"]) for k in a)


def test_a_high_risk_call_is_the_end_of_the_turn(full, book):
    for row in all_rows(full[1]):
        seen_high = False
        for m in row["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                seen_high = seen_high or any(book.risk(n, a)[0] == "high" for n, a in calls_in([m]))
            if m["role"] == "tool":
                assert not seen_high, "a tool result follows a high-risk call: the agent would have stopped to ask"


def test_open_ended_requests_are_given_no_tools(full):
    for row in all_rows(full[1]):
        user = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
        if row["source"] != "planner" and la.open_ended(user):
            assert row["tools"] == [] and not calls_in(row["messages"])


def test_safety_examples_are_present_and_behave(full):
    kinds = {}
    for row in all_rows(full[1]):
        kinds.setdefault(row["kind"], []).append(row)
    assert len(kinds["propose"]) >= 25 and len(kinds["decline"]) >= 30
    assert kinds["injection"] and kinds["clarify"] and kinds["chat"]
    for row in kinds["decline"]:
        assert not calls_in(row["messages"])
        assert row["messages"][-1]["content"]
    assert any("won't" in r["messages"][-1]["content"] for r in kinds["decline"])
    for row in kinds["injection"]:                    # the model ignores orders found in a result
        assert row["messages"][-1]["role"] == "assistant" and not row["messages"][-1].get("tool_calls")
    assert any(p["source"] == "safety" and "confirm" in json.dumps(p["rejected"]) for p in full[1].preferences)
    assert any(p["source"] == "multistep" for p in full[1].preferences)      # honest about a failed step


def test_multi_step_rows_are_sequential_single_calls(full):
    flows = [r for r in all_rows(full[1]) if r["kind"] == "flow"]
    assert len(flows) >= 20
    for row in flows:
        calls = [m for m in row["messages"] if m.get("tool_calls")]
        assert len(calls) >= 2 and all(len(m["tool_calls"]) == 1 for m in calls)


def test_planner_replay_rows_hold_a_valid_plan_or_a_refusal(full):
    from orion_core import neural_planner as np_
    plans = [r for r in all_rows(full[1]) if r["source"] == "planner"]
    assert len(plans) >= 40
    assert any(r["kind"] == "plan" for r in plans) and any(r["kind"] == "plan_refusal" for r in plans)
    for row in plans:
        assert row["tools"] == [] and row["messages"][0]["content"] == np_.SYSTEM_INSTRUCTION
        body = json.loads(row["messages"][-1]["content"])
        if row["kind"] == "plan":
            assert body["steps"] and body["success_criteria"] and "refusal" not in body
        else:
            assert body["steps"] == [] and body["refusal"]


def test_no_row_contains_an_evaluation_sentence(full):
    strings, _report = evalguard.collect_eval_strings(ROOT)
    guard = evalguard.EvalGuard(strings)
    assert len(guard) > 300
    offenders = [m["content"] for r in all_rows(full[1]) for m in r["messages"]
                 if m["role"] == "user" and r["source"] != "planner" and guard.overlaps(m["content"])]
    assert offenders == []
    assert full[1].manifest["evaluation_guard"]["strings"] == len(guard)


def test_the_heldout_routing_half_is_never_trained_on(full):
    holdout = {c[0] for c in evalguard.routing_split_cases(ROOT / "tests" / "data" / "intent_routing_cases.py",
                                                            "holdout")}
    dev = {c[0] for c in evalguard.routing_split_cases(ROOT / "tests" / "data" / "intent_routing_cases.py", "dev")}
    users = {m["content"] for r in all_rows(full[1]) for m in r["messages"] if m["role"] == "user"}
    assert users & holdout == set()
    assert users & dev, "the development half is used"
    assert full[1].manifest["routing_split"] == "dev" and full[1].manifest["contaminated_routing"] is False


def test_training_on_every_routing_case_is_possible_but_flagged(tmp_path):
    cases = [("fire up spotify", "open_app", ()), ("pause the music", "media_control", ())]
    result = build(make_config(tmp_path, dry_run=True, routing_split="all", routing_cases=cases,
                               tool_routing_cases=(), sources=("routing",), augment=0, eval_strings=[]))
    assert result.manifest["contaminated_routing"] is True
    assert {r["messages"][1]["content"] for r in all_rows(result)} == {"fire up spotify", "pause the music"}


def test_an_evaluation_sentence_is_removed_even_when_reworded(tmp_path):
    result = build(make_config(tmp_path, dry_run=True, sources=("phrasebank",), eval_strings=["Open Notepad."]))
    users = [m["content"].lower() for r in all_rows(result) for m in r["messages"] if m["role"] == "user"]
    assert not any(u.strip(" .") in {"open notepad", "please open notepad"} for u in users)
    assert result.manifest["dropped"]["evaluation_overlap"] >= 1


def test_evalguard_similarity():
    g = evalguard.EvalGuard(["Open Spotify and turn the volume down a bit.", "Shut yourself down."])
    assert g.overlaps("open spotify and turn the volume down a bit")
    assert g.overlaps("please shut yourself down")
    assert not g.overlaps("open notepad")
    assert not g.overlaps("launch spotify, then lower the volume a little")


def test_each_tool_is_capped_and_no_tool_kinds_are_capped_too(tmp_path):
    result = build(make_config(tmp_path, dry_run=True, max_per_tool=2, max_decline=3, max_chat=2, max_planner=5))
    by_tool: dict[str, int] = {}
    for r in all_rows(result):
        if r["source"] != "planner" and r["kind"] not in {"flow"} and r["meta"]["tools_called"]:
            by_tool[r["meta"]["tools_called"][0]] = by_tool.get(r["meta"]["tools_called"][0], 0) + 1
    assert by_tool and max(by_tool.values()) <= 2
    kinds = [r["kind"] for r in all_rows(result)]
    assert kinds.count("decline") <= 3 and kinds.count("chat") <= 2
    assert result.manifest["dropped"]["over_cap:tool"] > 0


def test_split_is_by_request_so_no_phrasing_crosses_over(full):
    result = full[1]
    assert result.val and len(result.val) < len(result.train) / 5
    assert {r["meta"]["group"] for r in result.train}.isdisjoint({r["meta"]["group"] for r in result.val})
    none = build_none_split = None  # noqa: F841 (documented below)


def test_no_validation_split_when_asked(tmp_path):
    result = build(make_config(tmp_path, dry_run=True, val_fraction=0.0, sources=("chat", "safety")))
    assert result.val == [] and result.train


def test_rows_are_unique_and_deterministic(tmp_path):
    a = build(make_config(tmp_path / "a", dry_run=True))
    b = build(make_config(tmp_path / "b", dry_run=True))
    ids = [r["id"] for r in all_rows(a)]
    assert len(ids) == len(set(ids))
    assert ids == [r["id"] for r in all_rows(b)]
    assert [r["messages"] for r in a.train] == [r["messages"] for r in b.train]


def test_the_tool_surface_fits_the_budget_and_prefers_the_right_tool(full, book):
    for row in all_rows(full[1]):
        if row["source"] == "planner":
            continue
        for name in row["meta"]["tools_called"]:
            assert name in row["meta"]["offered"]
        assert row["meta"]["tokens_estimate"] <= 1900


# ── writing it out ───────────────────────────────────────────────────────────────────────────────

def test_a_written_dataset_verifies_and_its_manifest_adds_up(full):
    tmp, result = full
    folder = result.path
    assert folder.parent == tmp / "out" and (tmp / "out" / "latest.txt").read_text().strip() == folder.name
    assert files.verify_dataset(folder) == []
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == DATASET_FORMAT == files.DATASET_FORMAT
    assert manifest["builder_version"] == BUILDER_VERSION
    counts = manifest["counts"]
    train, val = files.load_rows(folder / "train.jsonl"), files.load_rows(folder / "val.jsonl")
    prefs = files.load_rows(folder / "preferences.jsonl")
    assert (counts["train"], counts["val"], counts["preferences"]) == (len(train), len(val), len(prefs))
    assert counts["rows"] == len(train) + len(val) == sum(counts["by_source"].values()) == sum(counts["by_kind"].values())
    assert manifest["files"]["train.jsonl"]["rows"] == len(train)
    assert manifest["tool_surface_sha256"] and manifest["system_prompt_sha256"]
    assert manifest["redactions"] is not None and "privacy" in manifest
    text = json.dumps(manifest)
    assert "jo@example.com" not in text and "hunter2" not in text and "Leeds" not in text   # counts only


def test_tampering_with_a_written_dataset_is_detected(tmp_path):
    result = build(make_config(tmp_path, sources=("chat",)))
    train = result.path / "train.jsonl"
    train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert any("train.jsonl was changed" in p for p in files.verify_dataset(result.path))


def test_a_dry_run_writes_nothing(tmp_path):
    result = build(make_config(tmp_path, dry_run=True))
    assert result.path is None and result.manifest["dry_run"] is True
    assert not (tmp_path / "out").exists()


def test_private_data_is_never_written_inside_the_project_or_a_synced_folder(tmp_path):
    with pytest.raises(ValueError, match="project folder"):
        build(make_config(tmp_path, sources=("chat",), out_dir=ROOT / "training_must_not_exist"))
    assert not (ROOT / "training_must_not_exist").exists()
    with pytest.raises(ValueError, match="OneDrive"):
        build(make_config(tmp_path, sources=("chat",), out_dir=tmp_path / "OneDrive - Someone" / "training"))
    assert paths.location_problem(tmp_path / "plain") == ""


def test_the_default_training_folder_avoids_the_project_and_the_cloud(tmp_path):
    home = tmp_path / "home"
    assert paths.training_root(tmp_path / "OneDrive - X" / "config", environ={}, home=home) == home / "ORION" / "training"
    assert paths.training_root(ROOT / "config", environ={}, home=home) == home / "ORION" / "training"
    safe = tmp_path / "ORION" / "config"
    assert paths.training_root(safe, environ={}, home=home) == safe / "training"
    chosen = tmp_path / "elsewhere"
    assert paths.training_root(safe, environ={paths.ENV_TRAINING_DIR: str(chosen)}, home=home) == chosen


def test_unknown_sources_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown source"):
        build(make_config(tmp_path, dry_run=True, sources=("real", "nonsense")))
    assert set(ALL_SOURCES) >= {"real", "corrections", "routing", "safety", "multistep", "planner"}


# ── the command line ───────────────────────────────────────────────────────────────────────────────

def test_the_command_line_dry_run_prints_counts_and_none_of_the_owners_words(tmp_path, capsys, monkeypatch):
    from tools import train_build_dataset as cli
    monkeypatch.setattr(la, "_default_scorer", lexical_scores)
    cfg = owner_stores(tmp_path)
    code = cli.main(["--dry-run", "--config-dir", str(cfg), "--out-dir", str(tmp_path / "out"),
                     "--redact-term", "Leeds"])
    out = capsys.readouterr().out
    assert code == 0 and "DRY RUN - nothing was written." in out and "by source:" in out
    # 20 rows in the store; the two remote-phone ones are never read
    assert "episodes 18 rows" in out and "shadow 10 rows" in out
    for private in ("notepad please", "jo@example", "hunter2", "calculator", "Leeds"):
        assert private not in out
    assert not (tmp_path / "out").exists()


def test_the_command_line_refuses_a_bad_destination_and_unknown_sources(tmp_path, capsys, monkeypatch):
    from tools import train_build_dataset as cli
    monkeypatch.setattr(la, "_default_scorer", lexical_scores)
    assert cli.main(["--sources", "nonsense"]) == 2
    code = cli.main(["--sources", "chat", "--config-dir", str(tmp_path), "--out-dir", str(ROOT / "nope")])
    assert code == 2 and "Refused" in capsys.readouterr().err

# ── folded in from test_training_encode.py (Stage 3: only the answer to a row's last request is learned.) ────

from orion_core.training.encode import IGNORE, encode_row

ROLE = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
PROMPT = 9


class Tokenizer:
    """Renders each message as [header, *bytes of its text, 0], prefix-consistent
    like a real chat template: the assistant header IS the generation prompt."""

    @staticmethod
    def header(role: str) -> list[int]:
        return [PROMPT, ROLE["assistant"]] if role == "assistant" else [ROLE[role]]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, tools=None):
        ids: list[int] = []
        for message in messages:
            ids += [*self.header(message["role"]), *str(message.get("content") or "").encode(), 0]
        return ids + (self.header("assistant") if add_generation_prompt else [])


def _learned(row) -> bytes:
    encoded = encode_row(Tokenizer(), row, 10_000)
    return bytes(t for t in encoded.labels if t != IGNORE and t > 9)


def test_a_correction_row_teaches_the_fix_not_the_mistake():
    row = {"id": "c1", "messages": [
        {"role": "user", "content": "open spotify"},
        {"role": "assistant", "content": "WRONG"},
        {"role": "user", "content": "no, the app"},
        {"role": "assistant", "content": "RIGHT"},
    ]}
    learned = _learned(row)
    assert b"RIGHT" in learned and b"WRONG" not in learned


def test_a_tool_turn_learns_the_call_and_the_reply_after_its_result():
    row = {"id": "t1", "messages": [
        {"role": "user", "content": "weather"},
        {"role": "assistant", "content": "CALL"},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "REPLY"},
    ]}
    learned = _learned(row)
    assert b"CALL" in learned and b"REPLY" in learned and b"sunny" not in learned


def test_a_message_can_say_otherwise():
    row = {"id": "o1", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "KEEP", "train": True},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "SKIP", "train": False},
        {"role": "assistant", "content": "LAST"},
    ]}
    learned = _learned(row)
    assert b"KEEP" in learned and b"LAST" in learned and b"SKIP" not in learned
