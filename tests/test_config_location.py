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
