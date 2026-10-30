"""Forget removes an item or topic from every store, keeps no copy, touches
nothing else, and only runs after the owner approves.

The owner's Companion panel (25 Oct 2026) said "Nothing tracked yet" above
three starred items - "120kg Bench Press Max", "Personal Record Deadlift",
"Personal Record Bench Press" - with no way to remove them. All tmp stores.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orion_core.cognition import CognitiveStateManager
from orion_core.companion import CompanionEngine
from orion_core.forget import ForgetEngine, redact, topic_pattern, topic_terms
from orion_core.knowledge_graph import KnowledgeGraphEngine
from orion_core.memory import MemoryAgent, OrionMemoryMatrix


class _Bus:
    def __getattr__(self, name):
        return SimpleNamespace(emit=Mock(), connect=Mock())


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    bus = _Bus()
    monkeypatch.setattr(MemoryAgent, "_new_transcript_path", lambda self: tmp_path / "talk" / "s.jsonl")
    (tmp_path / "talk").mkdir()
    matrix = OrionMemoryMatrix(config / "orion_core.db", config, bus)
    memory = MemoryAgent(matrix, bus)
    cognition = CognitiveStateManager(bus, Mock(), path=config / "cognitive_state.json")
    companion = CompanionEngine(bus, memory=None, cognition=cognition, db_path=config / "companion.db")
    graph = KnowledgeGraphEngine(bus, memory, db_path=config / "knowledge_graph.db")

    companion.record_achievement("Personal Record Bench Press", "Benched 170 kg")
    companion.record_achievement("Personal Record Deadlift", "Deadlifted 170 kg")
    companion.record_achievement("120kg Bench Press Max")
    companion.record_achievement("Finished the marathon")
    memory.remember("knowledge", "fitness_bench_press_pr", "The owner's bench press max is 120 kg.")
    memory.remember("knowledge", "fitness_review",
                    "The owner trains four days a week. His bench press stalled at 120 kg. "
                    "He wants to run more in the spring.")
    memory.remember("knowledge", "cooking", "The owner likes Thai green curry.")
    matrix.log_episode("user", "I benched 120 kg today, a new bench press max.")
    matrix.log_episode("orion", "Splendid, sir. Your bench press is improving. Shall I set a reminder "
                                "for your dentist appointment next Tuesday at nine?")
    matrix.log_episode("user", "What's the weather like in Colchester?")
    graph.ingest_record("conversation", "bench press PR", "I hit a bench press PR of 120 kg.")
    graph.ingest_record("conversation", "weather", "Rain in Colchester this afternoon.")
    cognition.add_task("Bench press 130kg by Christmas")
    cognition.dismiss_task("180kg deadlift")
    cognition.add_task("Book the dentist")
    with sqlite3.connect(config / "goals.db") as conn:
        conn.executescript("""
            CREATE TABLE goals (id INTEGER PRIMARY KEY, title TEXT, outcome TEXT DEFAULT '',
                                why TEXT DEFAULT '', metric TEXT DEFAULT '');
            CREATE TABLE milestones (id INTEGER PRIMARY KEY, goal_id INTEGER, title TEXT);
            CREATE TABLE key_results (id INTEGER PRIMARY KEY, goal_id INTEGER, title TEXT);
            CREATE TABLE entries (id INTEGER PRIMARY KEY, goal_id INTEGER, note TEXT DEFAULT '');
            CREATE TABLE links (id INTEGER PRIMARY KEY, goal_id INTEGER, ref TEXT);
            CREATE TABLE nudges (key TEXT PRIMARY KEY, day TEXT, at TEXT);
            INSERT INTO goals(id, title) VALUES (1, 'Bench press 130 kg'), (2, 'Read 12 books');
            INSERT INTO milestones(goal_id, title) VALUES (1, 'Bench 125'), (2, 'Book 6');
            INSERT INTO entries(goal_id, note) VALUES (1, 'bench session'), (2, 'finished Dune');
            INSERT INTO nudges VALUES ('goal:1:behind', '2026-10-01', 'x'), ('goal:2:behind', '2026-10-01', 'x');
        """)
    with sqlite3.connect(config / "standing_questions.db") as conn:
        conn.executescript("""
            CREATE TABLE questions (id INTEGER PRIMARY KEY, text TEXT);
            CREATE TABLE known_points (id INTEGER PRIMARY KEY, question_id INTEGER, point TEXT);
            CREATE TABLE checks (id INTEGER PRIMARY KEY, question_id INTEGER, summary TEXT);
            INSERT INTO questions VALUES (1, 'Best bench press programme?'), (2, 'Any news on Mars?');
            INSERT INTO known_points VALUES (1, 1, 'Use 5x5'), (2, 2, 'Rover landed');
        """)
    (config / "thought_journal.jsonl").write_text(
        json.dumps({"text": "He seems proud of his bench press."}) + "\n"
        + json.dumps({"text": "Current focus: WhatsApp."}) + "\n", encoding="utf-8")
    (tmp_path / "talk" / "s.jsonl").write_text(
        json.dumps({"role": "user", "content": "New bench press max!"}) + "\n"
        + json.dumps({"role": "user", "content": "Open Spotify"}) + "\n", encoding="utf-8")
    engine = ForgetEngine(config, memory=memory, companion=companion, graph=graph,
                          cognition=cognition, transcripts_dir=tmp_path / "talk")
    yield SimpleNamespace(config=config, matrix=matrix, memory=memory, companion=companion,
                          graph=graph, cognition=cognition, engine=engine, tmp=tmp_path)
    matrix.close()


def _count(path, sql):
    with sqlite3.connect(path) as conn:
        return conn.execute(sql).fetchone()[0]


def test_topic_words_drop_the_framing_and_match_inflections():
    assert topic_terms("my bench press records") == ["bench", "press"]
    pattern = topic_pattern("bench press")
    assert pattern.search("activity_bench_press") and pattern.search("Bench-pressed 120")
    assert not pattern.search("benchmark pressure")
    assert topic_pattern("a") is None
    text, hit = redact("I ran 5k. My bench press is 120. I slept well.", [pattern])
    assert hit and text == "I ran 5k. I slept well."


def test_survey_changes_nothing(stores):
    before = {p.name: p.read_bytes() for p in stores.config.iterdir() if p.suffix == ".json" or p.suffix == ".jsonl"}
    plan = stores.engine.survey("my bench press records")
    assert plan.matches and "Companion ledger" in plan.counts()
    assert {p.name: p.read_bytes() for p in stores.config.iterdir() if p.suffix == ".json" or p.suffix == ".jsonl"} == before
    assert len(stores.companion.achievements()) == 4


def test_forgetting_a_topic_removes_it_from_every_store_and_nothing_else(stores):
    engine, config = stores.engine, stores.config
    receipt = engine.execute(engine.survey("bench press"))
    assert receipt["problems"] == []

    titles = [a["title"] for a in stores.companion.achievements()]
    assert set(titles) == {"Personal Record Deadlift", "Finished the marathon"}  # another topic

    values = [r[0] for r in stores.matrix.conn.execute("SELECT value FROM intelligence")]
    assert not any("bench press" in v.lower() for v in values)
    assert "The owner likes Thai green curry." in values
    assert any(v.startswith("The owner trains four days a week.") for v in values)   # redacted, kept
    fts = stores.matrix.conn.execute(
        "SELECT count(*) FROM intelligence_fts WHERE intelligence_fts MATCH 'bench'").fetchone()[0]
    assert fts == 0

    episodes = [r[0] for r in stores.matrix.conn.execute("SELECT content FROM episodes")]
    assert "What's the weather like in Colchester?" in episodes
    assert not any("bench" in e.lower() for e in episodes)
    assert any("dentist appointment" in e for e in episodes)                 # other sentences kept
    assert stores.matrix.conn.execute(
        "SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH 'bench'").fetchone()[0] == 0
    assert stores.matrix.conn.execute(
        "SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH 'weather'").fetchone()[0] == 1

    graph = config / "knowledge_graph.db"
    assert _count(graph, "SELECT count(*) FROM events WHERE lower(text) LIKE '%bench%'") == 0
    assert _count(graph, "SELECT count(*) FROM events WHERE lower(text) LIKE '%colchester%'") == 1

    goals = config / "goals.db"
    assert _count(goals, "SELECT count(*) FROM goals") == 1
    assert _count(goals, "SELECT count(*) FROM milestones") == 1
    assert _count(goals, "SELECT count(*) FROM entries") == 1
    assert _count(goals, "SELECT count(*) FROM nudges") == 1
    standing = config / "standing_questions.db"
    assert _count(standing, "SELECT count(*) FROM questions") == 1
    assert _count(standing, "SELECT count(*) FROM known_points") == 1

    tasks = [t["title"] for t in stores.cognition.list_tasks()]
    assert tasks == ["Book the dentist"]
    assert stores.cognition.is_dismissed("180kg deadlift")                   # other topic untouched

    journal = (config / "thought_journal.jsonl").read_text(encoding="utf-8")
    assert "bench" not in journal and "WhatsApp" in journal
    talk = (stores.tmp / "talk" / "s.jsonl").read_text(encoding="utf-8")
    assert "bench" not in talk and "Spotify" in talk
    assert not list(config.glob(".forget-*"))                                # no copy left behind


def test_forgetting_one_item_removes_its_copies_only(stores):
    item = next({"store": "companion", "table": "achievements", **a}
                for a in stores.companion.achievements() if a["title"] == "Personal Record Deadlift")
    receipt = stores.engine.execute(stores.engine.survey(item=item))
    assert receipt["removed"]["Companion ledger"] == 1
    assert {a["title"] for a in stores.companion.achievements()} == {
        "120kg Bench Press Max", "Personal Record Bench Press", "Finished the marathon"}
    assert stores.cognition.is_dismissed("180kg deadlift")    # not a copy of this item


def test_an_approved_survey_never_removes_rows_added_after_it(stores):
    engine = stores.engine
    approved = engine.survey("bench press").identities()
    stores.companion.record_achievement("Bench press 125 kg")    # arrives after the prompt
    engine.execute(engine.survey("bench press"), approved)
    assert "Bench press 125 kg" in [a["title"] for a in stores.companion.achievements()]


# ── the confirmation gate (voice/tool path) ─────────────────────────────────

def _dispatcher(stores):
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    from orion_core.system_guard import SystemActionGuard

    host = KnowledgeDispatchMixin.__new__(KnowledgeDispatchMixin)
    host.bus = SimpleNamespace(log=SimpleNamespace(emit=Mock()), confirm_action=SimpleNamespace(emit=Mock()),
                               dashboard_event=SimpleNamespace(emit=Mock()))
    host.memory, host.companion, host.graph = stores.memory, stores.companion, stores.graph
    host.cognition, host.goal_tracker = stores.cognition, None
    host.system_guard = SystemActionGuard()
    host.forget_engine = lambda: stores.engine
    return host


def test_the_tool_only_raises_a_confirmation_and_removes_nothing(stores):
    host = _dispatcher(stores)
    result = asyncio.run(host.companion_tool({"action": "forget", "topic": "my bench press records"}))
    assert not result.ok and result.receipt["pending_confirmation"]
    assert "nothing has been removed" in result.text
    assert len(stores.companion.achievements()) == 4
    payload = host.bus.confirm_action.emit.call_args.args[0]
    assert payload["intent"] == "memory_forget" and payload["token"]
    assert payload["token"] not in result.text                     # the model never sees it


def test_only_the_owners_approval_releases_it_and_only_once(stores):
    from orion_core.system_guard import DecisionKind
    host = _dispatcher(stores)
    asyncio.run(host.companion_tool({"action": "forget", "topic": "bench press"}))
    token = host.bus.confirm_action.emit.call_args.args[0]["token"]
    decision = host.system_guard.confirm(token)
    assert decision.kind is DecisionKind.EXECUTE
    result = host.run_confirmed_forget(decision.payload)
    assert result.ok, result.text
    assert {a["title"] for a in stores.companion.achievements()} == {
        "Personal Record Deadlift", "Finished the marathon"}
    assert host.system_guard.confirm(token).kind is DecisionKind.REJECT       # single use


def test_a_cancelled_prompt_forgets_nothing(stores):
    host = _dispatcher(stores)
    asyncio.run(host.companion_tool({"action": "forget", "topic": "bench press"}))
    token = host.bus.confirm_action.emit.call_args.args[0]["token"]
    host.system_guard.cancel(token)
    assert host.system_guard.confirm(token).kind.name == "REJECT"
    assert len(stores.companion.achievements()) == 4


# ── the Companion panel ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def test_the_panel_never_says_nothing_tracked_above_listed_items(qapp, stores):
    from orion_core.gui.ops_deck import CompanionPanel
    panel = CompanionPanel(_Bus(), stores.companion)
    assert panel.lists.count() == 3
    assert "Nothing tracked yet" not in panel.brief.text()


def test_the_panel_forget_control_asks_first_and_then_removes(qapp, stores, monkeypatch):
    from PyQt6.QtWidgets import QPushButton
    from orion_core.gui.ops_deck import CompanionPanel
    panel = CompanionPanel(_Bus(), stores.companion)
    monkeypatch.setattr(panel, "_forget_engine", lambda: stores.engine)
    buttons = [b for b in panel.lists.findChildren(QPushButton) if b.text() == "Forget"]
    assert buttons and all(b.accessibleName().startswith("Forget ") for b in buttons)
    target = next(b for b in buttons if b.accessibleName() == "Forget 120kg Bench Press Max")

    monkeypatch.setattr(panel, "_confirm", lambda title, summary: False)
    target.click()
    assert len(stores.companion.achievements()) == 4               # declined: nothing gone

    monkeypatch.setattr(panel, "_confirm", lambda title, summary: True)
    target.click()
    titles = [a["title"] for a in stores.companion.achievements()]
    assert "120kg Bench Press Max" not in titles and len(titles) == 3
    assert "Forgot '120kg Bench Press Max'" in panel.status.text()


def test_the_forget_question_never_runs_a_nested_event_loop(monkeypatch):
    """28 Oct 2026: box.exec() inside the forget task starved every other task
    ("Cannot enter into task"), the microphone stream to Live among them."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget
    from orion_core.gui.ops_deck import CompanionPanel
    app = QApplication.instance() or QApplication([])

    def no_exec(self):
        raise AssertionError("exec() would nest an event loop inside a task")

    def answer(choice):
        def fake_open(box):
            next(b for b in box.buttons() if b.text() == choice).click()
        return fake_open

    monkeypatch.setattr(QMessageBox, "exec", no_exec)
    Host = type("Host", (QWidget,), {"_confirm_box": CompanionPanel._confirm_box,
                                     "_ask": CompanionPanel._ask})
    host = Host()
    monkeypatch.setattr(QMessageBox, "open", answer("Forget"))
    assert asyncio.run(host._ask("'x'", "1 record")) is True
    monkeypatch.setattr(QMessageBox, "open", answer("Cancel"))
    assert asyncio.run(host._ask("'x'", "1 record")) is False
    app.processEvents()


# ── erase everything (the right to erasure, 28 Oct 2026) ──────────────────────

def _config_with_data(tmp_path):
    config = tmp_path / "config"
    (config / "diagnostics").mkdir(parents=True)
    (config / "models").mkdir()
    (config / "diagnostics" / "session.log").write_text("YOU: hello", encoding="utf-8")
    (config / "models" / "encoder.onnx").write_bytes(b"model")
    for name, body in {"orion_core.db": "memories", "identity.json": "{}", "temporal.json": "{}",
                       "api_keys.json": "{}", "appearance.json": "{}"}.items():
        (config / name).write_text(body, encoding="utf-8")
    return config


def test_erase_personal_takes_what_records_the_owner_and_keeps_what_runs_orion(tmp_path):
    from orion_core import erase_all
    config = _config_with_data(tmp_path)
    plan = erase_all.survey("personal", config)
    assert {p.name for p in plan.paths} == {"diagnostics", "orion_core.db", "identity.json",
                                            "temporal.json"}
    assert plan.total_bytes > 0 and "Recycle Bin" in plan.summary()
    everything = erase_all.survey("everything", config)
    assert {p.name for p in everything.paths} >= {"api_keys.json", "models", "appearance.json"}
    with pytest.raises(ValueError):
        erase_all.survey("some", config)


def test_an_approved_erase_runs_at_the_next_start_into_the_recycle_bin(tmp_path):
    from orion_core import erase_all
    config = _config_with_data(tmp_path)
    assert erase_all.run_pending(config) == {"ran": False}            # nothing approved
    erase_all.schedule("personal", config)
    assert erase_all.pending(config) == "personal"
    binned, lines = [], []

    def recycle(path):
        binned.append(path.name)
        path.rename(tmp_path / f"bin-{path.name}")
        return True

    result = erase_all.run_pending(config, log=lines.append, recycle=recycle, refusal=lambda _p: None)
    assert result["moved"] == 4 and not result["kept"]
    assert set(binned) == {"diagnostics", "orion_core.db", "identity.json", "temporal.json",
                           erase_all.MARKER}
    assert (config / "api_keys.json").exists() and (config / "models").exists()
    assert erase_all.pending(config) == "" and "PRIVACY: erased 4" in lines[0]


def test_an_erase_that_cannot_move_everything_tries_again_next_start(tmp_path):
    from orion_core import erase_all
    config = _config_with_data(tmp_path)
    erase_all.schedule("personal", config)
    result = erase_all.run_pending(config, log=lambda _l: None, recycle=lambda _p: True,
                                   refusal=lambda p: "locked" if p.name == "orion_core.db" else None)
    assert result["kept"] and erase_all.pending(config) == "personal"


def test_erasing_needs_the_owners_turn_and_the_on_screen_confirmation(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace

    from orion_core import erase_all, execution_policy as policy
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    monkeypatch.setattr(erase_all, "CONFIG_DIR", _config_with_data(tmp_path))
    raised, logged = [], []

    class Guard:
        def request_confirmation(self, intent, origin, payload):
            raised.append((intent.value, payload))
            return SimpleNamespace(token="tok-1")

    host = KnowledgeDispatchMixin()
    host.system_guard = Guard()
    host.bus = SimpleNamespace(confirm_action=SimpleNamespace(emit=raised.append),
                               log=SimpleNamespace(emit=logged.append),
                               request_restart=SimpleNamespace(emit=lambda: None))
    survey = asyncio.run(host.erase_my_data_tool({"action": "survey"}))
    assert survey.ok and "4 item(s)" in survey.text
    with policy.background_scope("job"):
        refused = asyncio.run(host.erase_my_data_tool({"action": "erase"}))
    assert not refused.ok and refused.receipt["status"] == "requires_foreground" and not raised
    with policy.conversation_scope("voice"):
        asked = asyncio.run(host.erase_my_data_tool({"action": "erase"}))
    assert not asked.ok and asked.receipt["pending_confirmation"]
    assert raised[0] == ("data_erase", {"scope": "personal"}) and erase_all.pending() == ""
    done = host.run_confirmed_erase({"scope": "personal"})
    assert done.ok and erase_all.pending() == "personal"
    assert not host.run_confirmed_erase({"scope": "all of it"}).ok
