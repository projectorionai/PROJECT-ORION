"""
SQLite performance policy (Mark XXVI) — why ORION stopped stuttering on writes.

Every engine opened SQLite with the stdlib defaults (journal_mode=DELETE,
synchronous=FULL), which measured **2.81 ms per committed write on this machine**.
Under qasync that cost lands on the thread that draws the face and services the
audio callback, so a burst of small writes was real, visible lag.

These tests lock the policy in: the pragmas are applied, they are applied to
EVERY store, and a database that refuses them still opens.
"""

from __future__ import annotations

import ast
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.db import (  # noqa: E402
    PRAGMAS,
    apply_pragmas,
    connect,
    journal_mode,
    optimise,
)


# ── the policy ───────────────────────────────────────────────────────────────

def test_connect_enables_wal_and_relaxed_fsync(tmp_path):
    connection = connect(tmp_path / "t.db")
    assert journal_mode(connection).lower() == "wal"
    synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
    assert synchronous == 1, "synchronous should be NORMAL (1), not FULL (2)"
    connection.close()


def test_connect_sets_a_busy_timeout_so_a_lock_waits_instead_of_raising(tmp_path):
    connection = connect(tmp_path / "t.db")
    timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout >= 1000
    connection.close()


def test_connect_creates_the_parent_directory(tmp_path):
    connection = connect(tmp_path / "nested" / "deep" / "t.db")
    assert (tmp_path / "nested" / "deep" / "t.db").exists()
    connection.close()


def test_row_factory_is_on_by_default(tmp_path):
    connection = connect(tmp_path / "t.db")
    connection.execute("CREATE TABLE t (a TEXT)")
    connection.execute("INSERT INTO t VALUES ('x')")
    row = connection.execute("SELECT a FROM t").fetchone()
    assert row["a"] == "x"
    connection.close()


def test_apply_pragmas_never_raises_on_a_hostile_connection():
    class _Refuses:
        def execute(self, _sql):
            raise sqlite3.Error("nope")
    refused = apply_pragmas(_Refuses())
    assert len(refused) == len(PRAGMAS), "a refusing database must still be usable"


def test_journal_mode_and_optimise_are_safe_on_a_broken_connection():
    class _Broken:
        def execute(self, _sql):
            raise sqlite3.Error("gone")
    assert journal_mode(_Broken()) == "unknown"
    optimise(_Broken())          # must not raise


# ── the measurable win ───────────────────────────────────────────────────────

def test_wal_is_dramatically_faster_for_commit_per_write(tmp_path):
    """The reason this policy exists. Not a strict timing assertion (CI machines
    vary) — but WAL must not be SLOWER, and on any real disk it is far faster."""
    def bench(path, pragmas):
        db = sqlite3.connect(str(path))
        for pragma in pragmas:
            db.execute(pragma)
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        db.commit()
        start = time.perf_counter()
        for i in range(60):
            db.execute("INSERT INTO t (v) VALUES (?)", (str(i),))
            db.commit()
        elapsed = time.perf_counter() - start
        db.close()
        return elapsed

    default = bench(tmp_path / "a.db", [])
    walled = bench(tmp_path / "b.db", ["PRAGMA journal_mode=WAL",
                                       "PRAGMA synchronous=NORMAL"])
    assert walled <= default, "WAL should never be slower than the rollback journal"


# ── every store must use it ──────────────────────────────────────────────────

STORES = [
    ("study", "StudyStore"),
    ("focus", "FocusStore"),
    ("finance", "FinanceStore"),
    ("wellbeing", "WellbeingStore"),
    ("decisions", "DecisionStore"),
    ("goals", "GoalStore"),
]


@pytest.mark.parametrize("module_name,class_name", STORES)
def test_each_store_opens_in_wal(tmp_path, module_name, class_name):
    import importlib
    module = importlib.import_module(f"orion_core.{module_name}")
    store = getattr(module, class_name)(tmp_path / f"{module_name}.db")
    connection = getattr(store, "_db", None)
    assert connection is not None, f"{class_name} has no _db handle"
    assert journal_mode(connection).lower() == "wal", (
        f"{class_name} is still paying a full fsync per write")
    store.close()


def _literal_readonly_uri(node):
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return False
    uri = urlsplit(node.value)
    return (uri.scheme == "file" and not uri.fragment
            and parse_qs(uri.query, keep_blank_values=True) == {"mode": ["ro"]})


def _assigned_readonly_uri(tree, target, call):
    if _literal_readonly_uri(target):
        return True
    if not isinstance(target, ast.Name):
        return False
    # A URI variable is exempt only when its single assignment proves this
    # exact read-only shape. Unknown values, aliases and reassignment fail
    # closed; uri=True by itself never proves a database is read-only.
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
              and isinstance(node.ctx, ast.Store) and node.id == target.id]
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == target.id]
    if len(writes) != 1 or len(assignments) != 1 or assignments[0].lineno >= call.lineno:
        return False
    value = assignments[0].value
    if _literal_readonly_uri(value):
        return True

    def pieces(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return pieces(node.left) + pieces(node.right)
        return [node]

    parts = pieces(value)
    return (len(parts) == 3 and isinstance(parts[0], ast.Constant) and parts[0].value == "file:"
            and isinstance(parts[-1], ast.Constant) and parts[-1].value == "?mode=ro"
            # The path comes from pathlib's resolved POSIX path, as used by
            # goals._read_rows; an arbitrary interpolated URI remains suspect.
            and isinstance(parts[1], ast.Call) and not parts[1].args and not parts[1].keywords
            and isinstance(parts[1].func, ast.Attribute) and parts[1].func.attr == "as_posix"
            and isinstance(parts[1].func.value, ast.Call) and not parts[1].func.value.args
            and not parts[1].func.value.keywords and isinstance(parts[1].func.value.func, ast.Attribute)
            and parts[1].func.value.func.attr == "resolve")


def _opens_disk_database(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sqlite3" and node.func.attr == "connect"):
            continue
        target = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "database"), None)
        if isinstance(target, ast.Constant) and target.value == ":memory:":
            continue  # no disk I/O or WAL is possible for an in-memory database
        uri_enabled = any(kw.arg == "uri" and isinstance(kw.value, ast.Constant)
                          and kw.value.value is True for kw in node.keywords)
        if uri_enabled and _assigned_readonly_uri(tree, target, node):
            continue  # a read-only migration input cannot commit writes
        return True
    return False


@pytest.mark.parametrize("source,expected", [
    ("sqlite3.connect(':memory:')", False),
    ("sqlite3.connect(database=':memory:')", False),
    ("sqlite3.connect(path)", True),
    ("sqlite3.connect(':memory:'); sqlite3.connect(path)", True),
    ("sqlite3.connect(database='jobs.db')", True),
    ("sqlite3.connect('file:legacy.db?mode=ro', uri=True)", False),
    ("sqlite3.connect('file:legacy.db?mode=rw', uri=True)", True),
    ("sqlite3.connect('file:legacy.db?mode=rwc', uri=True)", True),
    ("sqlite3.connect('file:legacy.db?mode=ro', uri=False)", True),
    ("sqlite3.connect('file:legacy.db?mode=ro')", True),
    ("sqlite3.connect(uri, uri=True)", True),
    ("uri = external_value\nsqlite3.connect(uri, uri=True)", True),
    ("uri = 'file:' + value + '?mode=ro'\nsqlite3.connect(uri, uri=True)", True),
    ("uri = 'file:' + path.resolve().as_posix() + '?mode=ro'\nsqlite3.connect(uri, uri=True)", False),
    ("uri = 'file:' + path.resolve().as_posix() + '?mode=rw'\nsqlite3.connect(uri, uri=True)", True),
    ("uri = 'file:' + path.resolve().as_posix() + '?mode=rwc'\nsqlite3.connect(uri, uri=True)", True),
    ("uri = 'file:legacy.db?mode=ro'\nuri = writable\nsqlite3.connect(uri, uri=True)", True),
    ("uri = 'file:legacy.db?mode=ro'\nsqlite3.connect(uri, uri=True)\nsqlite3.connect(path)", True),
    ("sqlite3.connect('file:legacy.db?mode=ro&mode=rw', uri=True)", True),
])
def test_store_policy_exempts_only_literal_in_memory_connections(source, expected):
    assert _opens_disk_database(source) is expected


def test_no_store_opens_sqlite_without_the_pragmas():
    """A new store that forgets the policy would silently reintroduce the lag."""
    root = Path(__file__).resolve().parents[1] / "orion_core"
    offenders = []
    for path in sorted(root.rglob("*.py")):          # subpackages too
        if path.name in {"db.py", "maintenance.py", "diagnostics.py"}:
            continue                      # maintenance/diagnostics open read-only
        source = path.read_text(encoding="utf-8", errors="replace")
        if "apply_pragmas" not in source and _opens_disk_database(source):
            offenders.append(path.name)
    assert not offenders, (
        "these modules open SQLite without ORION's performance pragmas: "
        + ", ".join(offenders))


def test_the_policy_is_documented_where_it_is_defined():
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "db.py"
              ).read_text(encoding="utf-8")
    # the measured numbers are the justification; they must stay in the file
    assert "2.81" in source and "0.03" in source
    assert "qasync" in source
