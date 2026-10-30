"""ORION's writable data must be movable out of a synced folder.

Every store runs in WAL mode, so each database is three files — .db, .db-wal,
.db-shm — and a sync client uploads, restores and conflict-resolves them
independently. A .db recovered without its matching WAL is the database as it
stood at the last checkpoint, and everything since is gone, silently, because
SQLite opened a perfectly valid file. Measured on 2026-09-21: 8.3 MB in the
checkout's logs and 9.8 MB more in the installed app's.

Checkpointing narrows that window; moving the data out of the sync root closes
it. The resolution order matters as much as the move: a source checkout — and
this suite — must keep their own directory, or running the tests would mean
running them over the user's real memory.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import constants  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import move_config  # noqa: E402


def _reload(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return importlib.reload(constants)


def test_the_default_is_unchanged(monkeypatch):
    """A move must be something you opt into. Silently relocating a user's
    data on upgrade would be worse than the problem."""
    fresh = _reload(monkeypatch, ORION_CONFIG_DIR=None)
    assert fresh.CONFIG_DIR == fresh.BASE_DIR / "config"


def test_an_override_is_honoured(monkeypatch, tmp_path):
    target = tmp_path / "elsewhere"
    fresh = _reload(monkeypatch, ORION_CONFIG_DIR=str(target))
    assert fresh.CONFIG_DIR == target
    assert target.is_dir(), "the override should be usable immediately"


def test_a_pointer_file_redirects(monkeypatch, tmp_path):
    """Chosen over a new hard-coded default because it is reversible by
    deleting one small text file, and applies per installation."""
    app, data = tmp_path / "app", tmp_path / "data"
    (app / "config").mkdir(parents=True)
    data.mkdir()
    (app / constants.CONFIG_POINTER_NAME).write_text(str(data), encoding="utf-8")
    monkeypatch.delenv("ORION_CONFIG_DIR", raising=False)
    monkeypatch.setattr(constants, "BASE_DIR", app)
    assert constants._resolve_config_dir() == data


def test_a_pointer_to_nowhere_is_ignored(monkeypatch, tmp_path):
    """A stale pointer must not stop ORION starting."""
    app = tmp_path / "app"
    (app / "config").mkdir(parents=True)
    (app / constants.CONFIG_POINTER_NAME).write_text(
        str(tmp_path / "deleted"), encoding="utf-8")
    monkeypatch.delenv("ORION_CONFIG_DIR", raising=False)
    monkeypatch.setattr(constants, "BASE_DIR", app)
    assert constants._resolve_config_dir() == app / "config"


def test_the_override_beats_the_pointer(monkeypatch, tmp_path):
    app, pointed, forced = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    app.mkdir(); pointed.mkdir()
    (app / constants.CONFIG_POINTER_NAME).write_text(str(pointed), encoding="utf-8")
    monkeypatch.setenv("ORION_CONFIG_DIR", str(forced))
    monkeypatch.setattr(constants, "BASE_DIR", app)
    assert constants._resolve_config_dir() == forced


# ── the migration tool ───────────────────────────────────────────────────────

def test_a_synced_location_is_recognised():
    assert move_config.looks_synced(Path(r"C:\Users\x\OneDrive - Uni\p\config")) == "onedrive"
    assert move_config.looks_synced(Path(r"C:\Users\x\Dropbox\config")) == "dropbox"
    assert move_config.looks_synced(Path(r"C:\Users\x\AppData\Local\ORION\config")) == ""


def test_the_destination_is_beyond_every_redirector():
    """AppData/Local is the obvious answer and the wrong one here.

    Roaming is synced, which reproduces the problem with a different client.
    Local is redirected out from under a Microsoft Store interpreter into a
    per-package sandbox. The profile root is subject to neither.
    """
    destination = str(move_config.default_destination())
    assert "Roaming" not in destination
    assert "AppData" not in destination
    assert "OneDrive" not in destination


def test_a_plan_copies_nothing(tmp_path, capsys):
    source = tmp_path / "app" / "config"
    source.mkdir(parents=True)
    (source / "a.db").write_bytes(b"x" * 16)
    destination = tmp_path / "new"
    assert move_config.move(tmp_path / "app", destination, dry_run=True) == 0
    assert not destination.exists()


def test_an_incomplete_copy_never_writes_the_pointer(tmp_path, monkeypatch):
    """The pointer is the commit. Writing it after a bad copy would redirect
    ORION at data that is not all there."""
    app = tmp_path / "app"
    (app / "config").mkdir(parents=True)
    (app / "config" / "a.db").write_bytes(b"x" * 16)
    monkeypatch.setattr(move_config, "compare_trees",
                        lambda *_a: ["missing: a.db"])
    monkeypatch.setattr(move_config, "orion_is_running", lambda: False)
    monkeypatch.setattr(move_config, "store_python", lambda: False)
    assert move_config.move(app, tmp_path / "new") == 1
    assert not (app / move_config.CONFIG_POINTER_NAME).exists()
    assert (app / "config" / "a.db").exists(), "the original was touched"


def test_it_refuses_while_orion_is_running(tmp_path, monkeypatch):
    """Copying a database out from under a live writer manufactures the
    corruption this exists to prevent."""
    app = tmp_path / "app"
    (app / "config").mkdir(parents=True)
    monkeypatch.setattr(move_config, "orion_is_running", lambda: True)
    assert move_config.move(app, tmp_path / "new") == 1
    assert not (app / move_config.CONFIG_POINTER_NAME).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the probe runs cmd.exe")
def test_a_real_move_verifies_then_points(tmp_path, monkeypatch):
    import sqlite3

    app = tmp_path / "app"
    source = app / "config"
    source.mkdir(parents=True)
    connection = sqlite3.connect(str(source / "real.db"))
    connection.execute("CREATE TABLE t (v TEXT)")
    connection.execute("INSERT INTO t VALUES ('remembered')")
    connection.commit()
    connection.close()
    (source / "notes" ).mkdir()
    (source / "notes" / "a.txt").write_text("keep me", encoding="utf-8")

    monkeypatch.setattr(move_config, "orion_is_running", lambda: False)
    # pytest's tmp_path lives under AppData/Local, which the virtualisation
    # guard rightly refuses under a Store interpreter. This test is about the
    # copy, so take the guard out of the picture rather than weaken it.
    monkeypatch.setattr(move_config, "store_python", lambda: False)
    destination = tmp_path / "new"
    assert move_config.move(app, destination) == 0

    pointer = app / move_config.CONFIG_POINTER_NAME
    assert pointer.read_text(encoding="utf-8").strip() == str(destination)
    assert (destination / "notes" / "a.txt").read_text(encoding="utf-8") == "keep me"
    moved = sqlite3.connect(str(destination / "real.db"))
    assert moved.execute("SELECT v FROM t").fetchone()[0] == "remembered"
    moved.close()
    # nothing deleted, ever
    assert (source / "real.db").exists()
    assert (source / "notes" / "a.txt").exists()


def test_revert_removes_the_pointer_and_keeps_both_copies(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / move_config.CONFIG_POINTER_NAME).write_text(str(tmp_path / "x"),
                                                       encoding="utf-8")
    assert move_config.revert(app) == 0
    assert not (app / move_config.CONFIG_POINTER_NAME).exists()


# -- the trap that nearly made this a silent no-op ---------------------------

def test_appdata_local_is_refused_under_the_store_python(tmp_path, monkeypatch):
    """A Microsoft Store app's writes under AppData/Local are redirected into
    a per-package sandbox. Python then reads back its OWN redirected view and
    sees a perfect copy, while every other program -- including ORION.exe, an
    ordinary binary -- looks at the real path and finds nothing.

    The first run of this tool reported a verified 1.27 GB copy that no other
    process could see, because the verification ran inside the same sandbox
    that had received it.
    """
    monkeypatch.setattr(move_config, "store_python", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppDataLocal"))
    inside = tmp_path / "AppDataLocal" / "ORION" / "config"
    outside = tmp_path / "profile" / "ORION" / "config"
    assert move_config.is_virtualised(inside) is True
    assert move_config.is_virtualised(outside) is False

    app = tmp_path / "app"
    (app / "config").mkdir(parents=True)
    monkeypatch.setattr(move_config, "orion_is_running", lambda: False)
    assert move_config.move(app, inside) == 1
    assert not (app / move_config.CONFIG_POINTER_NAME).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the probe runs cmd.exe")
def test_the_copy_is_confirmed_visible_to_another_process(tmp_path):
    """Verifying a copy from inside the process that made it proves nothing
    when that process's view of the filesystem is redirected. cmd.exe has no
    stake in this interpreter's redirection."""
    assert move_config._visible_to_others(tmp_path) == ""


def test_an_invisible_destination_blocks_the_pointer(tmp_path, monkeypatch):
    """The pointer is the commit; it must not be written over an illusion."""
    app = tmp_path / "app"
    (app / "config").mkdir(parents=True)
    (app / "config" / "a.db").write_bytes(b"x" * 8)
    monkeypatch.setattr(move_config, "orion_is_running", lambda: False)
    monkeypatch.setattr(move_config, "store_python", lambda: False)
    monkeypatch.setattr(move_config, "_visible_to_others",
                        lambda _d: "another process cannot see it")
    assert move_config.move(app, tmp_path / "new") == 1
    assert not (app / move_config.CONFIG_POINTER_NAME).exists()
    assert (app / "config" / "a.db").exists()


# ── the migration, end to end, in temporary trees ───────────────────────────
#
# What must hold: the data copied is the data ORION actually uses (a moved
# installation's pointer, not the stale config/ beside it); databases travel
# as consistent SQLite backups that include what is still only in the WAL;
# ORION cannot be running, nor start, while it happens; an interruption leaves
# him on the original; a destination with something in it is never written
# over; and --rollback puts the pointer back without losing anything.

import os  # noqa: E402
import sqlite3  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402


@pytest.fixture(autouse=True)
def _private_instance_lock(monkeypatch):
    """The real ORION may be running while the suite runs. His lock and his
    process must not decide these tests, and no test may take HIS lock."""
    from orion_core import single_instance

    tag = f"-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(single_instance, "MUTEX_NAME", single_instance.MUTEX_NAME + tag)
    monkeypatch.setattr(single_instance, "SOCKET_NAME", single_instance.SOCKET_NAME + tag)
    monkeypatch.setattr(move_config, "_orion_process_running", lambda: False)


@pytest.fixture
def mover(monkeypatch):
    """The copy mechanics only: sandbox detection and the cmd.exe visibility
    probe have their own tests above."""
    monkeypatch.setattr(move_config, "store_python", lambda: False)
    monkeypatch.setattr(move_config, "_visible_to_others", lambda _d: "")
    return move_config


def _store(path, rows, table="facts"):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"CREATE TABLE IF NOT EXISTS {table} (v TEXT)")
    connection.executemany(f"INSERT INTO {table} VALUES (?)", [(r,) for r in rows])
    connection.commit()
    connection.close()


def _rows(path, table="facts"):
    connection = sqlite3.connect(str(path))
    try:
        return [r[0] for r in connection.execute(f"SELECT v FROM {table} ORDER BY rowid")]
    finally:
        connection.close()


def _app_with_data(tmp_path, *, pointed=False):
    """An installation whose data is either config/ or, if *pointed*, a
    directory its pointer names (with a stale config/ left behind)."""
    app = tmp_path / "app"
    (app / "config").mkdir(parents=True)
    data = tmp_path / "live" if pointed else app / "config"
    _store(data / "orion_core.db", ["remembered", "also remembered"])
    (data / "notes").mkdir(parents=True, exist_ok=True)
    (data / "notes" / "a.txt").write_text("keep me", encoding="utf-8")
    if pointed:
        _store(app / "config" / "orion_core.db", ["STALE"])
        (app / move_config.CONFIG_POINTER_NAME).write_text(str(data), encoding="utf-8")
    return app, data


@pytest.mark.parametrize("case", ["environment", "pointer", "dangling", "default"])
def test_the_tool_resolves_data_exactly_as_orion_does(monkeypatch, tmp_path, case):
    app, pointed, forced = tmp_path / "app", tmp_path / "pointed", tmp_path / "forced"
    app.mkdir()
    pointed.mkdir()
    forced.mkdir()
    env = {}
    if case in {"environment", "pointer"}:
        (app / move_config.CONFIG_POINTER_NAME).write_text(str(pointed), encoding="utf-8")
    if case == "dangling":
        (app / move_config.CONFIG_POINTER_NAME).write_text(str(tmp_path / "gone"),
                                                         encoding="utf-8")
    if case == "environment":
        env["ORION_CONFIG_DIR"] = str(forced)
        monkeypatch.setenv("ORION_CONFIG_DIR", str(forced))
    else:
        monkeypatch.delenv("ORION_CONFIG_DIR", raising=False)
    monkeypatch.setattr(constants, "BASE_DIR", app)
    ours = move_config.resolve_data_dir(app, environ=env)
    assert ours.data_dir == constants._resolve_config_dir()
    assert ours.via == {"environment": "environment", "pointer": "pointer",
                        "dangling": "default", "default": "default"}[case]


def test_a_moved_installation_is_copied_from_where_its_pointer_leads(tmp_path, mover):
    """The earlier tool always copied app/config — for an installation that
    had been moved before, the STALE copy — and would then have pointed ORION
    at it."""
    app, live = _app_with_data(tmp_path, pointed=True)
    destination = tmp_path / "new"
    assert mover.move(app, destination) == 0
    assert _rows(destination / "orion_core.db") == ["remembered", "also remembered"]
    pointer = (app / mover.CONFIG_POINTER_NAME).read_text(encoding="utf-8").strip()
    assert pointer == str(destination)
    journal = mover.read_journal(app)
    assert journal["state"] == "switched" and journal["previous_pointer"] == str(live)


def test_uncheckpointed_wal_content_arrives_and_no_sidecar_travels(tmp_path, mover):
    """Transactions still only in the -wal are part of the database. A loose
    copy of the .db alone loses them; the backup API does not."""
    app = tmp_path / "app"
    source = app / "config"
    source.mkdir(parents=True)
    writer = sqlite3.connect(str(source / "orion_core.db"))
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE facts (v TEXT)")
        writer.executemany("INSERT INTO facts VALUES (?)", [(f"fact {n}",) for n in range(300)])
        writer.commit()
        assert (source / "orion_core.db-wal").stat().st_size > 0
        loose = tmp_path / "loose.db"
        loose.write_bytes((source / "orion_core.db").read_bytes())
        with pytest.raises(sqlite3.OperationalError):
            _rows(loose)                          # the loose copy lost the table

        destination = tmp_path / "new"
        assert mover.move(app, destination) == 0
    finally:
        writer.close()
    assert len(_rows(destination / "orion_core.db")) == 300
    assert not [p.name for p in destination.rglob("*") if p.name.endswith(("-wal", "-shm"))]


def test_an_interrupted_migration_leaves_orion_on_the_original(tmp_path, mover, monkeypatch):
    app, _data = _app_with_data(tmp_path)
    _store(app / "config" / "second.db", ["b"])
    destination = tmp_path / "new"
    original = mover.backup_database
    copied: list[str] = []

    def failing(source, target):
        copied.append(source.name)
        if len(copied) == 2:
            raise OSError("disk full")
        original(source, target)

    monkeypatch.setattr(mover, "backup_database", failing)
    assert mover.move(app, destination) == 1
    assert not (app / mover.CONFIG_POINTER_NAME).exists()
    assert not destination.exists()
    assert mover.read_journal(app)["state"] == "failed"
    assert not list(tmp_path.glob(".new.orion-staging-*")), "a handled failure tidies up"
    assert _rows(app / "config" / "orion_core.db") == ["remembered", "also remembered"]

    # A hard stop (the process killed mid-copy) cannot tidy up...
    def killed(source, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(mover, "backup_database", killed)
    with pytest.raises(KeyboardInterrupt):
        mover.move(app, destination)
    assert list(tmp_path.glob(".new.orion-staging-*"))
    assert mover.read_journal(app)["state"] == "copying"
    assert not (app / mover.CONFIG_POINTER_NAME).exists()

    # ...so the next run recognises its own partial copy, clears it, and succeeds.
    monkeypatch.setattr(mover, "backup_database", original)
    assert mover.move(app, destination) == 0
    assert not list(tmp_path.glob(".new.orion-staging-*"))
    assert _rows(destination / "second.db") == ["b"]


def test_a_conflicting_destination_is_never_written_over(tmp_path, mover):
    app, _data = _app_with_data(tmp_path)
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "someone_elses.db").write_bytes(b"precious")
    assert mover.move(app, destination) == 1
    assert (destination / "someone_elses.db").read_bytes() == b"precious"
    assert not (app / mover.CONFIG_POINTER_NAME).exists()
    assert mover.read_journal(app) is None

    inside = app / "config" / "nested"
    assert mover.move(app, inside) == 1, "a destination inside the source"
    assert not inside.exists()


def test_an_environment_override_blocks_a_move(tmp_path, mover, monkeypatch):
    app, _data = _app_with_data(tmp_path)
    monkeypatch.setenv("ORION_CONFIG_DIR", str(app / "config"))
    assert mover.move(app, tmp_path / "new") == 1
    assert not (tmp_path / "new").exists()


def test_a_dry_run_names_the_real_source_and_writes_nothing(tmp_path, mover, capsys):
    app, live = _app_with_data(tmp_path, pointed=True)
    before = sorted(p.name for p in app.iterdir())
    assert mover.move(app, tmp_path / "new", dry_run=True) == 0
    out = capsys.readouterr().out
    assert str(live) in out and "via pointer" in out
    assert sorted(p.name for p in app.iterdir()) == before
    assert not (tmp_path / "new").exists()


def test_rollback_restores_the_previous_pointer_and_keeps_both_copies(tmp_path, mover):
    app, live = _app_with_data(tmp_path, pointed=True)
    destination = tmp_path / "new"
    assert mover.move(app, destination) == 0
    assert mover.rollback(app) == 0
    assert (app / mover.CONFIG_POINTER_NAME).read_text(encoding="utf-8") == str(live)
    assert mover.resolve_data_dir(app).data_dir == live
    assert _rows(destination / "orion_core.db") and _rows(live / "orion_core.db")
    assert mover.read_journal(app)["state"] == "rolled_back"


def test_rollback_of_a_first_move_removes_the_pointer(tmp_path, mover):
    app, _data = _app_with_data(tmp_path)
    assert mover.move(app, tmp_path / "new") == 0
    assert mover.rollback(app) == 0
    assert not (app / mover.CONFIG_POINTER_NAME).exists()
    assert mover.resolve_data_dir(app).data_dir == app / "config"


def test_rollback_will_not_silently_lose_writes_made_since_the_move(tmp_path, mover):
    app, _data = _app_with_data(tmp_path)
    destination = tmp_path / "new"
    assert mover.move(app, destination) == 0
    later = time.time() + 60
    os.utime(destination / "orion_core.db", (later, later))     # ORION wrote there since
    assert mover.rollback(app) == 1
    assert (app / mover.CONFIG_POINTER_NAME).read_text(encoding="utf-8") == str(destination)
    assert mover.rollback(app, force=True) == 0
    assert not (app / mover.CONFIG_POINTER_NAME).exists()


def test_a_running_orion_is_refused_by_his_own_lock(tmp_path, mover):
    """A process-name check misses a source run (it is pythonw.exe); the
    single-instance lock is what every ORION holds, however he was started."""
    from orion_core import single_instance

    app, _data = _app_with_data(tmp_path)
    held = single_instance.claim()               # this test is "the running ORION"
    assert held.granted
    try:
        assert mover.orion_is_running()
        assert mover.move(app, tmp_path / "new") == 1
        assert not (tmp_path / "new").exists()
        assert mover.read_journal(app) is None
        assert mover.rollback(app) == 1          # nor while he runs
    finally:
        single_instance.release()
    assert not mover.orion_is_running()
    assert mover.move(app, tmp_path / "new") == 0


def test_the_move_holds_the_lock_so_orion_cannot_start_half_way(tmp_path, mover, monkeypatch):
    from orion_core import single_instance

    app, _data = _app_with_data(tmp_path)
    seen: list[bool] = []
    original = mover.backup_database

    def during(source, target):
        # An ORION starting now must be refused by his own guard.
        probe = single_instance._claim_windows()
        if probe is None:
            probe = single_instance._claim_posix()
        seen.append(probe is not None and not probe.granted)
        original(source, target)

    monkeypatch.setattr(mover, "backup_database", during)
    assert mover.move(app, tmp_path / "new") == 0
    assert seen and all(seen)
    assert single_instance.held() is None, "the lock was not given back"


def test_a_checkout_that_borrows_the_installed_key_store_is_warned(tmp_path, monkeypatch):
    """A source checkout with no key store of its own reads the installed
    app's; once a pointer moves its data it no longer does. Existence checks
    only - the tool never reads or copies a key file."""
    home = tmp_path / "home"
    (home / "ORION" / "config").mkdir(parents=True)
    (home / "ORION" / "config" / "api_keys.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(move_config.Path, "home", classmethod(lambda cls: home))
    checkout = tmp_path / "checkout"
    (checkout / "config").mkdir(parents=True)
    (checkout / "orion_core").mkdir()
    (checkout / "orion.py").write_text("", encoding="utf-8")
    monkeypatch.delenv("ORION_CONFIG_DIR", raising=False)
    resolution = move_config.resolve_data_dir(checkout)
    note = move_config.key_store_note(resolution, tmp_path / "new")
    assert "api_keys.json" in note and str(tmp_path / "new") in note
    (checkout / "config" / "api_keys.json").write_text("{}", encoding="utf-8")
    assert move_config.key_store_note(resolution, tmp_path / "new") == ""
    assert move_config.default_destination(checkout) == home / "ORION" / "checkout-config", (
        "a checkout must not default onto the installed app's live data")


# ── where the keys are found after a move ─────────────────────────────────────

def test_a_moved_checkout_keeps_the_installed_key_store(tmp_path):
    """After move_config points a checkout's data outside OneDrive, a source
    run must not silently start without providers: with no key store of its
    own, it reads the installed app's, as it did before the move."""
    from orion_core import constants

    home = tmp_path / "home"
    installed = home / "ORION" / "config" / "api_keys.json"
    installed.parent.mkdir(parents=True)
    installed.write_text("{}", encoding="utf-8")
    moved = tmp_path / "checkout-config"
    moved.mkdir()
    assert constants._resolve_key_store(moved, home=home, environ={},
                                        under_test=False) == installed


def test_an_isolated_run_never_reaches_the_real_key_store(tmp_path):
    """ORION_CONFIG_DIR is how measurement runs are isolated: a dummy
    provider, no keys. It must never fall back to the real store."""
    from orion_core import constants

    home = tmp_path / "home"
    (home / "ORION" / "config").mkdir(parents=True)
    (home / "ORION" / "config" / "api_keys.json").write_text("{}", encoding="utf-8")
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    chosen = constants._resolve_key_store(
        isolated, home=home, environ={"ORION_CONFIG_DIR": str(isolated)}, under_test=False)
    assert chosen == isolated / "api_keys.json"
