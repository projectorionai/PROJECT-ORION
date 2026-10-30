"""
Forget something everywhere ORION keeps it, and keep no copy.

25 Oct 2026, the owner: the Companion panel listed three old starred items -
"120kg Bench Press Max", "Personal Record Deadlift", "Personal Record Bench
Press" - and there was no way to remove them, let alone the copies of them
that live elsewhere (memory facts and their search index, the meaning
vectors, the knowledge graph, goals, standing questions, conversation
history, the thought journal, the task tombstones in the cognitive state).

Two steps, so nothing is removed on a model's say-so:

* ``survey`` is read-only. It finds every row in every store that mentions the
  topic (or the copies of one item), and says what it would do with each:
  delete a record that exists only to hold it, or redact just the sentences
  that mention it from a longer text that is about other things too.
* ``execute`` removes exactly the rows a survey listed - and only once the
  owner has approved that survey (the Companion panel's own dialog, or the
  on-screen confirmation the voice path raises through the system guard).
  Nothing is backed up: the owner asked for no trace. Rows that do not match
  are never touched.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .constants import CONFIG_DIR

#: Words that frame a request rather than name the thing ("forget my bench
#: press records" is about bench press).
_LEADING = {"my", "the", "all", "any", "every", "everything", "about", "of", "those",
            "these", "that", "this", "old", "a", "an"}
_TRAILING = {"records", "notes", "memories", "data", "stuff", "history", "entries",
             "details", "info", "information", "things", "items"}

#: A text keeps its redacted remainder only when this many words survive.
_MIN_REMAINDER_WORDS = 4


def topic_terms(topic: str) -> list[str]:
    """The words that name *topic*, framing words removed."""
    words = re.findall(r"[a-z0-9]+", str(topic or "").lower())
    while words and words[0] in _LEADING:
        words.pop(0)
    while len(words) > 1 and words[-1] in _TRAILING:
        words.pop()
    return words


def topic_pattern(topic: str) -> re.Pattern[str] | None:
    """A case-insensitive pattern for *topic*: the words in order, separated by
    any spaces/underscores/hyphens, the last one inflected ("bench pressed",
    "bench_press", "deadlifts"), never inside another word."""
    words = topic_terms(topic)
    if not words or sum(len(w) for w in words) < 3:
        return None
    body = r"[\s_\-]*".join(re.escape(w) for w in words)
    return re.compile(r"(?<![a-z0-9])" + body + r"[a-z]*(?![0-9])", re.IGNORECASE)


def phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    """An exact phrase (an item's title), case-insensitive, word-bounded."""
    words = re.findall(r"[a-z0-9]+", str(phrase or "").lower())
    if not words or sum(len(w) for w in words) < 3:
        return None
    body = r"[\W_]+".join(re.escape(w) for w in words)
    return re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", re.IGNORECASE)


def redact(text: str, patterns: Iterable[re.Pattern[str]]) -> tuple[str, bool]:
    """*text* without its sentences that match; (remainder, matched)."""
    patterns = list(patterns)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    kept = [s for s in sentences if s.strip() and not any(p.search(s) for p in patterns)]
    matched = len(kept) != len([s for s in sentences if s.strip()])
    return " ".join(kept).strip(), matched


@dataclass
class Match:
    store: str          # "companion", "memory", "graph", "goals", ...
    table: str
    key: Any            # row id / key within the table
    action: str         # "delete" or "redact"
    preview: str
    new_text: str = ""

    def ident(self) -> tuple[str, str, str]:
        return (self.store, self.table, str(self.key))


@dataclass
class ForgetPlan:
    label: str
    patterns: list[re.Pattern[str]] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[_STORE_NAMES.get(m.store, m.store)] = out.get(_STORE_NAMES.get(m.store, m.store), 0) + 1
        return out

    def summary(self) -> str:
        if not self.matches:
            return f"I hold nothing about {self.label}."
        parts = [f"{name} ({n})" for name, n in self.counts().items()]
        redacted = sum(1 for m in self.matches if m.action == "redact")
        text = (f"{len(self.matches)} record(s) about {self.label}: " + ", ".join(parts) + ".")
        if redacted:
            text += (f" {redacted} longer text(s) keep everything except the sentences "
                     "that mention it.")
        return text + " Nothing is kept once it is forgotten."

    def identities(self) -> list[list[str]]:
        return [list(m.ident()) for m in self.matches]


_STORE_NAMES = {
    "companion": "Companion ledger", "memory": "memory facts", "episodes": "conversation history",
    "graph": "knowledge graph", "goals": "goals", "standing": "standing questions",
    "cognition": "tasks and priorities", "journal": "thought journal",
    "transcripts": "conversation transcripts",
}


class ForgetEngine:
    """Survey and remove a topic or item across every ORION store."""

    def __init__(self, config_dir: Path | None = None, *, memory: Any = None,
                 companion: Any = None, graph: Any = None, cognition: Any = None,
                 goals: Any = None, transcripts_dir: Path | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.dir = Path(config_dir or CONFIG_DIR)
        self.matrix = getattr(memory, "matrix", memory)
        self.companion = companion
        self.graph = graph
        self.cognition = cognition
        self.goals = goals
        if transcripts_dir is None and memory is not None:
            path = getattr(memory, "_transcript_path", None)
            transcripts_dir = Path(path).parent if path else None
        self.transcripts_dir = transcripts_dir
        self._log = log or (lambda _m: None)

    # ── connections: the live engine's own where there is one ───────────────

    def _db(self, store: str) -> tuple[sqlite3.Connection | None, Any, bool]:
        """(connection, lock-or-None, owned). Owned connections are closed after."""
        live = {"memory": self.matrix, "companion": self.companion, "graph": self.graph}.get(store)
        conn = getattr(live, "conn", None)
        if isinstance(conn, sqlite3.Connection):
            return conn, getattr(live, "_lock", None), False
        name = {"memory": "orion_core.db", "companion": "companion.db",
                "graph": "knowledge_graph.db", "goals": "goals.db",
                "standing": "standing_questions.db"}[store]
        path = self.dir / name
        if not path.is_file():
            return None, None, False
        conn = sqlite3.connect(path, timeout=5.0)
        from .db import apply_pragmas
        apply_pragmas(conn)          # the same WAL/busy settings as the live stores
        return conn, None, True

    @staticmethod
    def _tables(conn: sqlite3.Connection) -> set[str]:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    # ── the survey (read-only) ───────────────────────────────────────────────

    def survey(self, topic: str = "", *, item: dict[str, Any] | None = None) -> ForgetPlan:
        patterns: list[re.Pattern[str]] = []
        label = str(topic or "").strip()
        if item:
            for text in (item.get("title"), item.get("detail")):
                if text and len(re.findall(r"[a-z0-9]+", str(text).lower())) >= 2:
                    pattern = phrase_pattern(str(text))
                    if pattern is not None:
                        patterns.append(pattern)
            label = f"'{item.get('title') or label}'"
        else:
            pattern = topic_pattern(topic)
            if pattern is not None:
                patterns.append(pattern)
            label = f"'{' '.join(topic_terms(topic)) or topic}'"
        plan = ForgetPlan(label=label, patterns=patterns)
        if item and item.get("store") == "companion" and item.get("table") and item.get("id") is not None:
            plan.matches.append(Match("companion", str(item["table"]), item["id"], "delete",
                                      str(item.get("title") or "")))
        if not patterns:
            return plan
        hit = lambda text: any(p.search(str(text or "")) for p in patterns)   # noqa: E731
        for surveyor in (self._survey_companion, self._survey_memory, self._survey_graph,
                         self._survey_goals, self._survey_standing, self._survey_cognition,
                         self._survey_journal, self._survey_transcripts):
            try:
                for match in surveyor(hit, patterns):
                    if match.ident() not in {m.ident() for m in plan.matches}:
                        plan.matches.append(match)
            except Exception as exc:          # one unreadable store never hides the rest
                self._log(f"FORGET: could not survey {surveyor.__name__[8:]} - {exc}")
        return plan

    def _rows(self, store: str, sql: str, args: tuple = ()) -> list[tuple]:
        conn, lock, owned = self._db(store)
        if conn is None:
            return []
        try:
            if lock is not None:
                with lock:
                    return list(conn.execute(sql, args))
            return list(conn.execute(sql, args))
        except sqlite3.OperationalError:
            return []                           # table absent in this store
        finally:
            if owned:
                conn.close()

    def _text_match(self, store: str, table: str, key: Any, text: str,
                    patterns: list[re.Pattern[str]], *, redactable: bool) -> Match:
        remainder, _ = redact(text, patterns)
        if redactable and len(remainder.split()) >= _MIN_REMAINDER_WORDS:
            return Match(store, table, key, "redact", text[:120], remainder)
        return Match(store, table, key, "delete", text[:120])

    def _survey_companion(self, hit, patterns) -> list[Match]:
        out = []
        for row_id, title, detail in self._rows("companion", "SELECT id, title, detail FROM achievements"):
            if hit(title) or hit(detail):
                out.append(Match("companion", "achievements", row_id, "delete", str(title)))
        for row_id, subject, detail in self._rows("companion", "SELECT id, subject, detail FROM activities"):
            if hit(subject) or hit(detail):
                out.append(Match("companion", "activities", row_id, "delete", str(subject)))
        for slug, title, target in self._rows("companion", "SELECT slug, title, target FROM goals"):
            if hit(title) or hit(target):
                out.append(Match("companion", "goals", slug, "delete", str(title)))
        for (name,) in self._rows("companion", "SELECT DISTINCT name FROM habits"):
            if hit(name):
                out.append(Match("companion", "habits", name, "delete", str(name)))
        for row_id, summary in self._rows("companion", "SELECT id, summary FROM sessions WHERE summary != ''"):
            if hit(summary):
                out.append(self._text_match("companion", "sessions", row_id, str(summary),
                                            patterns, redactable=True))
        return out

    def _survey_memory(self, hit, patterns) -> list[Match]:
        out = []
        for row_id, key_ref, value in self._rows("memory", "SELECT id, key_ref, value FROM intelligence"):
            if hit(key_ref) and not hit(value):
                out.append(Match("memory", "intelligence", row_id, "delete", str(value)[:120]))
            elif hit(value):
                match = self._text_match("memory", "intelligence", row_id, str(value), patterns,
                                         redactable=not hit(key_ref))
                out.append(match)
        for row_id, content in self._rows("memory", "SELECT id, content FROM episodes"):
            if hit(content):
                out.append(self._text_match("episodes", "episodes", row_id, str(content),
                                            patterns, redactable=True))
        for row_id, summary, searchable in self._rows(
                "memory", "SELECT id, summary, searchable FROM scene_memories"):
            if hit(summary) or hit(searchable):
                out.append(Match("memory", "scene_memories", row_id, "delete", str(summary)[:120]))
        return out

    def _survey_graph(self, hit, patterns) -> list[Match]:
        out = []
        for event_id, title, text in self._rows("graph", "SELECT id, title, text FROM events"):
            if hit(title) or hit(text):
                out.append(Match("graph", "events", event_id, "delete", str(title)[:120]))
        for entity_id, name, aliases in self._rows("graph", "SELECT id, name, aliases_json FROM entities"):
            if hit(name) or hit(aliases):
                out.append(Match("graph", "entities", entity_id, "delete", str(name)))
        for rel_id, evidence in self._rows("graph", "SELECT id, evidence FROM relationships"):
            if hit(evidence):
                out.append(Match("graph", "relationships", rel_id, "delete", str(evidence)[:120]))
        return out

    def _survey_goals(self, hit, patterns) -> list[Match]:
        out = []
        for gid, title, outcome, why, metric in self._rows(
                "goals", "SELECT id, title, outcome, why, metric FROM goals"):
            if any(hit(v) for v in (title, outcome, why, metric)):
                out.append(Match("goals", "goals", gid, "delete", str(title)))
        for table, column in (("milestones", "title"), ("key_results", "title"), ("entries", "note")):
            for row_id, text in self._rows("goals", f"SELECT id, {column} FROM {table}"):
                if hit(text):
                    out.append(Match("goals", table, row_id, "delete", str(text)[:120]))
        return out

    def _survey_standing(self, hit, patterns) -> list[Match]:
        out = []
        for qid, text in self._rows("standing", "SELECT id, text FROM questions"):
            if hit(text):
                out.append(Match("standing", "questions", qid, "delete", str(text)[:120]))
        for table, column in (("known_points", "point"), ("checks", "summary")):
            for row_id, text in self._rows("standing", f"SELECT id, {column} FROM {table}"):
                if hit(text):
                    out.append(Match("standing", table, row_id, "delete", str(text)[:120]))
        return out

    def _cognitive_state(self) -> dict[str, Any]:
        if self.cognition is not None and callable(getattr(self.cognition, "snapshot", None)):
            return dict(self.cognition.snapshot() or {})
        try:
            return json.loads((self.dir / "cognitive_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _survey_cognition(self, hit, patterns) -> list[Match]:
        state = self._cognitive_state()
        out = []
        for section in ("pending_tasks", "dismissed_tasks"):
            for key, record in (state.get(section) or {}).items():
                if isinstance(record, dict) and (hit(record.get("title")) or hit(key)):
                    out.append(Match("cognition", section, key, "delete", str(record.get("title") or key)))
        for key, record in (state.get("goals") or {}).items():
            if isinstance(record, dict) and (hit(record.get("title")) or hit(record.get("description"))):
                out.append(Match("cognition", "goals", key, "delete", str(record.get("title") or key)))
        for key, record in (state.get("research_sessions") or {}).items():
            if isinstance(record, dict) and hit(record.get("topic")):
                out.append(Match("cognition", "research_sessions", key, "delete", str(record.get("topic"))))
        for index, text in enumerate(state.get("user_priorities") or []):
            if hit(text):
                out.append(Match("cognition", "user_priorities", str(text), "delete", str(text)[:120]))
        mentioned = (state.get("last_mentioned_tasks") or {}).get("titles") or []
        if any(hit(t) for t in mentioned):
            out.append(Match("cognition", "last_mentioned_tasks", "titles", "redact", "tasks just read out"))
        return out

    def _jsonl_matches(self, store: str, path: Path, hit, fields: tuple[str, ...]) -> list[Match]:
        out = []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return out
        for number, line in enumerate(lines):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and any(hit(record.get(f)) for f in fields):
                out.append(Match(store, str(path), number, "delete",
                                 str(next((record.get(f) for f in fields if record.get(f)), ""))[:120]))
        return out

    def _survey_journal(self, hit, patterns) -> list[Match]:
        path = self.dir / "thought_journal.jsonl"
        return self._jsonl_matches("journal", path, hit, ("text",)) if path.is_file() else []

    def _survey_transcripts(self, hit, patterns) -> list[Match]:
        folder = self.transcripts_dir
        if folder is None or not Path(folder).is_dir():
            return []
        out = []
        for path in sorted(Path(folder).glob("*.jsonl")):
            out.extend(self._jsonl_matches("transcripts", path, hit, ("content", "text")))
        return out

    # ── removal (only what an approved survey listed) ───────────────────────

    def execute(self, plan: ForgetPlan, approved: Iterable[Iterable[str]] | None = None) -> dict[str, Any]:
        """Remove the rows of *plan* (restricted to *approved* identities when
        given - the survey the owner saw). Returns a receipt per store."""
        allowed = {tuple(str(x) for x in ident) for ident in approved} if approved is not None else None
        chosen = [m for m in plan.matches if allowed is None or m.ident() in allowed]
        receipt: dict[str, Any] = {"removed": {}, "redacted": {}, "problems": []}
        by_store: dict[str, list[Match]] = {}
        for match in chosen:
            by_store.setdefault(match.store, []).append(match)
        handlers = {"companion": self._forget_companion, "memory": self._forget_memory,
                    "episodes": self._forget_memory, "graph": self._forget_graph,
                    "goals": self._forget_goals, "standing": self._forget_standing,
                    "cognition": self._forget_cognition, "journal": self._forget_jsonl,
                    "transcripts": self._forget_jsonl}
        memory_rows = by_store.pop("memory", []) + by_store.pop("episodes", [])
        if memory_rows:
            by_store["memory"] = memory_rows
        for store, matches in by_store.items():
            try:
                handlers[store](matches, plan.patterns)
                for m in matches:
                    bucket = receipt["redacted" if m.action == "redact" else "removed"]
                    name = _STORE_NAMES.get(m.store, m.store)
                    bucket[name] = bucket.get(name, 0) + 1
            except Exception as exc:
                receipt["problems"].append(f"{_STORE_NAMES.get(store, store)}: {exc}")
        removed = sum(receipt["removed"].values())
        redacted = sum(receipt["redacted"].values())
        # The log names counts and stores only - never the forgotten text.
        self._log(f"FORGET: removed {removed} record(s), redacted {redacted}"
                  + (f"; {len(receipt['problems'])} store(s) failed" if receipt["problems"] else "") + ".")
        return receipt

    @staticmethod
    def _run(conn: sqlite3.Connection, lock: Any, work: Callable[[sqlite3.Connection], None]) -> None:
        def _go() -> None:
            try:
                if conn.isolation_level is None and not conn.in_transaction:
                    conn.execute("BEGIN")       # autocommit connection: one transaction
                elif conn.in_transaction:
                    conn.commit()               # never fold in someone else's pending write
                work(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        if lock is not None:
            with lock:
                _go()
        else:
            _go()

    def _with_db(self, store: str, work: Callable[[sqlite3.Connection], None]) -> None:
        conn, lock, owned = self._db(store)
        if conn is None:
            raise OSError(f"{store} store not found")
        try:
            self._run(conn, lock, work)
        finally:
            if owned:
                conn.close()

    def _forget_companion(self, matches: list[Match], patterns) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for m in matches:
                if m.table == "achievements":
                    conn.execute("DELETE FROM achievements WHERE id=?", (m.key,))
                elif m.table == "activities":
                    conn.execute("DELETE FROM activities WHERE id=?", (m.key,))
                elif m.table == "goals":
                    conn.execute("DELETE FROM goals WHERE slug=?", (m.key,))
                elif m.table == "habits":
                    conn.execute("DELETE FROM habits WHERE name=?", (m.key,))
                elif m.table == "sessions":
                    conn.execute("UPDATE sessions SET summary=? WHERE id=?",
                                 (m.new_text if m.action == "redact" else "", m.key))
        self._with_db("companion", work)

    def _forget_memory(self, matches: list[Match], patterns) -> None:
        dropped: dict[str, list[int]] = {"intelligence": [], "episodes": []}

        def work(conn: sqlite3.Connection) -> None:
            tables = self._tables(conn)
            for m in matches:
                if m.table == "intelligence":
                    if m.action == "redact":
                        conn.execute("UPDATE intelligence SET value=? WHERE id=?", (m.new_text, m.key))
                    else:
                        conn.execute("DELETE FROM intelligence WHERE id=?", (m.key,))
                    dropped["intelligence"].append(int(m.key))
                elif m.table == "episodes":
                    row = conn.execute("SELECT role, content, created_at FROM episodes WHERE id=?",
                                       (m.key,)).fetchone()
                    if row is None:
                        continue
                    if "episodes_fts" in tables:
                        # External-content index with no delete trigger: retire
                        # the old text by hand or it stays searchable.
                        conn.execute("INSERT INTO episodes_fts(episodes_fts, rowid, role, content, "
                                     "created_at) VALUES('delete', ?, ?, ?, ?)", (m.key, *row))
                    if m.action == "redact":
                        conn.execute("UPDATE episodes SET content=? WHERE id=?", (m.new_text, m.key))
                        if "episodes_fts" in tables:
                            conn.execute("INSERT INTO episodes_fts(rowid, role, content, created_at) "
                                         "VALUES (?, ?, ?, ?)", (m.key, row[0], m.new_text, row[2]))
                    else:
                        conn.execute("DELETE FROM episodes WHERE id=?", (m.key,))
                    dropped["episodes"].append(int(m.key))
                elif m.table == "scene_memories":
                    conn.execute("DELETE FROM scene_memories WHERE id=?", (m.key,))
            if "semantic_vectors" in tables:
                for source, ids in dropped.items():
                    conn.executemany("DELETE FROM semantic_vectors WHERE source=? AND row_id=?",
                                     [(source, i) for i in ids])

        self._with_db("memory", work)
        matrix = self.matrix
        for source, ids in dropped.items():
            if not ids or matrix is None:
                continue
            try:      # the in-RAM meaning index must drop them too
                lock = getattr(matrix, "_lock", None)
                if lock is not None and callable(getattr(matrix, "_bump_vectors", None)):
                    with lock:
                        matrix._bump_vectors(source, deletes=ids)
                if source in getattr(matrix, "_snapshots", {}):
                    matrix._refresh_in_background(source)
            except Exception:
                pass

    def _forget_graph(self, matches: list[Match], patterns) -> None:
        def work(conn: sqlite3.Connection) -> None:
            tables = self._tables(conn)
            for m in matches:
                if m.table == "events":
                    conn.execute("DELETE FROM events WHERE id=?", (m.key,))
                    conn.execute("DELETE FROM event_entities WHERE event_id=?", (m.key,))
                    if "event_vectors" in tables:
                        conn.execute("DELETE FROM event_vectors WHERE event_id=?", (m.key,))
                elif m.table == "entities":
                    conn.execute("DELETE FROM entities WHERE id=?", (m.key,))
                    conn.execute("DELETE FROM relationships WHERE source_id=? OR target_id=?",
                                 (m.key, m.key))
                    conn.execute("DELETE FROM event_entities WHERE entity_id=?", (m.key,))
                elif m.table == "relationships":
                    conn.execute("DELETE FROM relationships WHERE id=?", (m.key,))
        self._with_db("graph", work)
        if self.graph is not None:
            try:
                self.graph._semantic_version = getattr(self.graph, "_semantic_version", 0) + 1
            except Exception:
                pass

    def _forget_goals(self, matches: list[Match], patterns) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for m in matches:
                if m.table == "goals":
                    for child in ("milestones", "key_results", "entries", "links"):
                        conn.execute(f"DELETE FROM {child} WHERE goal_id=?", (m.key,))
                    conn.execute("DELETE FROM nudges WHERE key LIKE ?", (f"goal:{m.key}:%",))
                    conn.execute("DELETE FROM goals WHERE id=?", (m.key,))
                elif m.table in {"milestones", "key_results", "entries"}:
                    conn.execute(f"DELETE FROM {m.table} WHERE id=?", (m.key,))
                    if m.table == "milestones":
                        conn.execute("DELETE FROM nudges WHERE key=?", (f"milestone:{m.key}",))
        self._with_db("goals", work)
        invalidate = getattr(self.goals, "_invalidate", None)
        if callable(invalidate):
            invalidate()

    def _forget_standing(self, matches: list[Match], patterns) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for m in matches:
                if m.table == "questions":
                    conn.execute("DELETE FROM known_points WHERE question_id=?", (m.key,))
                    conn.execute("DELETE FROM checks WHERE question_id=?", (m.key,))
                    conn.execute("DELETE FROM questions WHERE id=?", (m.key,))
                else:
                    conn.execute(f"DELETE FROM {m.table} WHERE id=?", (m.key,))
        self._with_db("standing", work)

    def _forget_cognition(self, matches: list[Match], patterns) -> None:
        hit = lambda text: any(p.search(str(text or "")) for p in patterns)   # noqa: E731

        def scrub(state: dict[str, Any]) -> None:
            for m in matches:
                if m.table in {"pending_tasks", "dismissed_tasks", "goals", "research_sessions"}:
                    (state.get(m.table) or {}).pop(m.key, None)
                    if m.table == "goals":
                        state["active_goals"] = [g for g in state.get("active_goals") or [] if g != m.key]
                elif m.table == "user_priorities":
                    state["user_priorities"] = [p for p in state.get("user_priorities") or [] if p != m.key]
                elif m.table == "last_mentioned_tasks":
                    record = state.get("last_mentioned_tasks") or {}
                    record["titles"] = [t for t in record.get("titles") or [] if not hit(t)]

        mutate = getattr(self.cognition, "_mutate", None)
        if callable(mutate):
            mutate(scrub)
            return
        path = self.dir / "cognitive_state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        scrub(state)
        _atomic_write(path, json.dumps(state, indent=2, ensure_ascii=False))

    def _forget_jsonl(self, matches: list[Match], patterns) -> None:
        by_file: dict[str, set[int]] = {}
        for m in matches:
            by_file.setdefault(m.table, set()).add(int(m.key))
        for filename, numbers in by_file.items():
            path = Path(filename)
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            kept = [line for index, line in enumerate(lines) if index not in numbers]
            _atomic_write(path, "\n".join(kept) + ("\n" if kept else ""))


def _atomic_write(path: Path, text: str) -> None:
    """Replace *path* whole; the temporary copy never outlives the write."""
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".forget-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


__all__ = ["ForgetEngine", "ForgetPlan", "Match", "phrase_pattern", "redact", "topic_pattern",
           "topic_terms"]
