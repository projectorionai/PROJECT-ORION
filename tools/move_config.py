"""
Move ORION's writable data out of a synced folder.

Why
---
ORION keeps everything he writes in one data directory — nineteen SQLite
stores, conversation episodes, the knowledge graph, his voice profile. For a
source checkout that is ``config/`` inside the project, which on this machine
is inside OneDrive, and live SQLite inside a sync client is a genuinely bad
combination.

Every store runs in WAL mode, so each database is three files: ``.db``,
``.db-wal`` and ``.db-shm``. A sync client treats those as three unrelated
files and uploads, restores and conflict-resolves them independently. A ``.db``
recovered without its matching WAL is the database as it stood at the last
checkpoint, and everything since is gone — silently, because SQLite opened a
perfectly valid file. Measured on 2026-09-21: 8.3 MB in the checkout's logs and
a further 9.8 MB in the installed app's, none of it yet in any database.

Frequent checkpointing narrows that window. Moving the data out of the synced
folder closes it.

What this does
--------------
1. Resolves where THIS installation's data actually is, exactly as ORION does
   (``ORION_CONFIG_DIR``, then the pointer file beside the app, then
   ``config/``). An installation that was moved before is copied from where
   its pointer leads, never from the stale ``config/`` left behind.
2. Refuses while ORION is running, and holds ORION's own single-instance lock
   for the whole move so he cannot start half-way through.
3. Copies into a staging directory beside the destination: every SQLite
   database through SQLite's online backup API — a consistent snapshot that
   includes whatever is still only in the ``-wal`` file — and every other file
   byte for byte. The live ``-wal``/``-shm`` files are never copied loose.
4. Verifies the copy: ``PRAGMA integrity_check`` and the row count of every
   table against the source, a SHA-256 of every other file, and that a
   process outside this interpreter can see it.
5. Renames the staging directory into place and switches the pointer file
   atomically (write a temporary file, then replace). The pointer is the
   commit: until it changes, ORION uses the original.
6. Records each step in ``config_migration.json`` beside the app, so an
   interrupted run is recognised and ``--rollback`` knows what to restore.

The original is never deleted, and its databases are only ever opened
read-only (SQLite may create its shared-memory ``-shm`` file beside one; no
data is written). A staging directory this tool left behind after an
interruption is its own incomplete copy and is cleared on the next run; a
completed copy is never deleted by this tool.

Usage
-----
    python tools/move_config.py --dry-run                 # the plan; touches nothing
    python tools/move_config.py --to D:\\ORION-data        # move this checkout's data
    python tools/move_config.py --app dist/ORION --dry-run
    python tools/move_config.py --rollback                # restore the previous pointer

ORION must not be running: copying a database out from under a live writer is
how you manufacture the corruption this exists to prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion_core.atomic_io import _replace

CONFIG_POINTER_NAME = "config_location.txt"
JOURNAL_NAME = "config_migration.json"
STAGING_MARKER = ".orion-migration-staging"
_SQLITE_HEADER = b"SQLite format 3\0"
_SIDECARS = ("-wal", "-shm", "-journal")


def store_python() -> bool:
    r"""Whether this interpreter is the Microsoft Store build of Python.

    It matters here for one reason: a Store app's writes under
    ``AppData\Local`` are silently REDIRECTED into a per-package sandbox at
    ``AppData\Local\Packages\<package>\LocalCache\Local\``. Python then
    reads back its own redirected view and sees a perfect copy, while every
    other program — including ORION.exe, which is an ordinary binary — looks at
    the real path and finds nothing there.

    That is not a hypothetical. The first run of this tool reported a verified
    1.27 GB copy that no other process could see, because the verification ran
    inside the same sandbox that received it.
    """
    return "WindowsApps" in sys.base_prefix


def is_virtualised(destination: Path) -> bool:
    """Whether *destination* would be redirected out from under us."""
    if not store_python():
        return False
    local = (os.getenv("LOCALAPPDATA") or "").lower()
    return bool(local) and str(destination).lower().startswith(local)


def is_source_checkout(app_dir: Path) -> bool:
    return (app_dir / "orion.py").is_file() and (app_dir / "orion_core").is_dir()


def default_destination(app_dir: Path | None = None) -> Path:
    r"""Somewhere outside the sync folder that nothing will redirect.

    Not ``AppData\Local``, which is the obvious answer and the wrong one
    here: see store_python(). The user profile root is plain, visible, needs
    no administrator, and is subject to neither OneDrive nor Store
    virtualisation.

    ``~/ORION/config`` is where the installed app's data already lives, so a
    source checkout gets its own directory beside it: pointing the checkout
    at the installed app's live stores would be two ORIONs sharing one memory
    by accident, and copying over them would destroy it.
    """
    if app_dir is not None and is_source_checkout(app_dir):
        return Path.home() / "ORION" / "checkout-config"
    return Path.home() / "ORION" / "config"


def looks_synced(path: Path) -> str:
    """A reason *path* looks like it is inside a sync client's folder, or ""."""
    text = str(path).lower()
    for marker in ("onedrive", "dropbox", "google drive", "icloud", "nextcloud"):
        if marker in text:
            return marker
    return ""


# ── where the data actually is ───────────────────────────────────────────────

@dataclass
class Resolution:
    """Where one installation's data lives, and how ORION decided that."""

    app_dir: Path
    data_dir: Path
    via: str                  # "environment", "pointer" or "default"
    pointer: Path
    pointer_text: str | None  # what the pointer file says, if there is one

    @property
    def dangling_pointer(self) -> bool:
        return self.pointer_text is not None and self.via == "default"


def resolve_data_dir(app_dir: Path, environ: Any = None) -> Resolution:
    """Where ORION started from *app_dir* keeps its data.

    Mirrors ``orion_core.constants._resolve_config_dir`` (a test holds the two
    together) without importing it, because that resolves for THIS process's
    installation at import time, and the installation being moved may be a
    different one — the frozen app under dist/, say.
    """
    environ = os.environ if environ is None else environ
    pointer = app_dir / CONFIG_POINTER_NAME
    text: str | None = None
    try:
        if pointer.is_file():
            text = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        text = None
    override = str(environ.get("ORION_CONFIG_DIR", "") or "").strip()
    if override:
        return Resolution(app_dir, Path(override).expanduser(), "environment", pointer, text)
    if text:
        target = Path(text).expanduser()
        if target.is_dir():
            return Resolution(app_dir, target, "pointer", pointer, text)
    return Resolution(app_dir, app_dir / "config", "default", pointer, text)


def known_installations() -> list[Path]:
    """This checkout, and the standalone build beside it if there is one."""
    found = [ROOT]
    frozen = ROOT / "dist" / "ORION"
    if (frozen / "ORION.exe").is_file():
        found.append(frozen)
    return found


# ── is ORION running? ────────────────────────────────────────────────────────

def _instance_lock_exists() -> bool:
    """Whether ORION's single-instance lock is held, WITHOUT taking it.

    Opening (not creating) the named mutex answers the question and cannot
    stop an ORION starting a moment later, which matters for a --dry-run.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        from orion_core.single_instance import MUTEX_NAME
    except Exception:
        return False
    SYNCHRONIZE = 0x00100000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, MUTEX_NAME)
    if not handle:
        return False
    # Close at once: a handle kept open would prop the mutex up after ORION
    # exits and stop him ever starting again (see single_instance.py).
    kernel32.CloseHandle(handle)
    return True


def _orion_process_running() -> bool:
    """An ORION.exe by name - for builds older than the single-instance lock."""
    try:
        import psutil
    except Exception:
        return False
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() == "orion.exe":
                return True
        except Exception:
            continue
    return False


def orion_is_running() -> bool:
    return _instance_lock_exists() or _orion_process_running()


@contextmanager
def hold_instance_lock(directory: Path) -> Iterator[Any]:
    """ORION's own single-instance lock, held for the duration, or None.

    Holding it is what "stop the writers" means: a running ORION is refused
    here, and an ORION launched mid-move is refused by his own guard and
    brings up nothing instead of opening the databases being copied.
    """
    from orion_core import single_instance

    claim = single_instance.claim(directory=directory)
    if not claim.granted:
        yield None
        return
    try:
        yield claim
    finally:
        single_instance.release()


# ── copying ──────────────────────────────────────────────────────────────────

def is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(16) == _SQLITE_HEADER
    except OSError:
        return False


def _inventory(source: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """(databases, other files, live sidecars), relative to *source*."""
    files = sorted(p for p in source.rglob("*") if p.is_file())
    databases = {p for p in files if is_sqlite(p)}
    sidecars = {Path(str(db) + suffix) for db in databases for suffix in _SIDECARS}
    ordinary = [p for p in files if p not in databases and p not in sidecars]
    return ([p.relative_to(source) for p in sorted(databases)],
            [p.relative_to(source) for p in ordinary],
            [p.relative_to(source) for p in sorted(sidecars) if p.is_file()])


def _open_source(path: Path) -> sqlite3.Connection:
    """The source database, read-only, so the original is never written.

    Read-only still reads the ``-wal``: uncheckpointed transactions are part
    of the database, and exactly what a loose copy of the ``.db`` loses.
    """
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10.0)


def backup_database(source: Path, destination: Path) -> None:
    """A consistent copy of one database through the SQLite backup API."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(_open_source(source)) as reader, closing(sqlite3.connect(destination)) as writer:
        reader.backup(writer)
        writer.commit()


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Rows in every real table (virtual tables are counted by their shadows)."""
    names = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND sql NOT LIKE 'CREATE VIRTUAL TABLE%' ORDER BY name")]
    counts: dict[str, int] = {}
    for name in names:
        quoted = '"' + name.replace('"', '""') + '"'
        counts[name] = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
    return counts


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── verification ─────────────────────────────────────────────────────────────

def compare_trees(source: Path, destination: Path,
                  databases: Any = ()) -> list[str]:
    """Every file under *source* that did not arrive intact. Empty is good.

    Databases are only checked for presence here — a backup-API copy is not
    byte-identical to its source (the WAL is folded in), so their size would
    prove nothing; verify_databases() checks what matters. Live sidecars are
    not expected at all.
    """
    databases = {Path(p) for p in databases}
    sidecars = {Path(str(db) + suffix) for db in databases for suffix in _SIDECARS}
    problems: list[str] = []
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        if relative in sidecars:
            continue
        mirrored = destination / relative
        if not mirrored.is_file():
            problems.append(f"missing: {relative}")
            continue
        if relative in databases:
            continue
        try:
            if item.stat().st_size != mirrored.stat().st_size:
                problems.append(f"size differs: {relative}")
        except OSError as exc:
            problems.append(f"unreadable: {relative} ({exc})")
    return problems


def _open_copy(path: Path) -> sqlite3.Connection:
    """A copy, opened to read. Not mode=ro: a read-only connection cannot tidy
    the -shm/-wal it creates, and they would travel into place with the copy."""
    connection = sqlite3.connect(str(path), timeout=10.0)
    connection.execute("PRAGMA query_only=1")
    return connection


def verify_databases(source: Path, destination: Path, databases: Any) -> list[str]:
    """integrity_check on each copy, and every table's row count against the source."""
    problems: list[str] = []
    for relative in databases:
        try:
            with closing(_open_source(source / relative)) as original:
                expected = table_counts(original)
            with closing(_open_copy(destination / relative)) as copy:
                verdict = copy.execute("PRAGMA integrity_check").fetchone()[0]
                if verdict != "ok":
                    problems.append(f"{relative}: integrity_check said {verdict!r}")
                    continue
                got = table_counts(copy)
            if got != expected:
                wrong = sorted(k for k in set(got) | set(expected) if got.get(k) != expected.get(k))
                problems.append(f"{relative}: row counts differ in {', '.join(wrong[:5])}")
        except Exception as exc:
            problems.append(f"{relative}: {exc}")
    return problems


def verify_files(source: Path, destination: Path, files: Any) -> list[str]:
    problems: list[str] = []
    for relative in files:
        try:
            if file_digest(source / relative) != file_digest(destination / relative):
                problems.append(f"content differs: {relative}")
        except OSError as exc:
            problems.append(f"unreadable: {relative} ({exc})")
    return problems


def databases_open(directory: Path) -> list[str]:
    """Every database under *directory* that will not open and pass integrity_check."""
    broken: list[str] = []
    for store in sorted(p for p in directory.rglob("*") if p.is_file() and is_sqlite(p)):
        try:
            with closing(_open_copy(store)) as c:
                if c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    broken.append(f"{store.name}: integrity_check failed")
        except Exception as exc:
            broken.append(f"{store.name}: {exc}")
    return broken


def _visible_to_others(destination: Path) -> str:
    """"" if another process can see *destination*, else why not.

    Asks ``cmd.exe`` — a program with no stake in this interpreter's
    redirection — to count the files. If Python can see a copy that cmd
    cannot, the copy is in a sandbox and the whole move is an illusion.
    """
    probe = destination / ".orion-visibility-probe"
    try:
        probe.write_text("probe", encoding="utf-8")
    except Exception as exc:
        return f"could not write a probe file ({exc})"
    try:
        result = subprocess.run(
            ["cmd", "/c", "if", "exist", str(probe), "echo", "SEEN"],
            capture_output=True, text=True, timeout=30)
        if "SEEN" not in (result.stdout or ""):
            return (f"another process cannot see {probe} - this interpreter's "
                    "writes are being redirected")
    except Exception as exc:
        return f"could not run the visibility probe ({exc})"
    finally:
        try:
            probe.unlink()
        except Exception:
            pass
    return ""


# ── the journal and the pointer ──────────────────────────────────────────────

def _write_atomically(path: Path, text: str) -> None:
    """Write *text* to *path* so a reader sees the old file or the new, never half."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_journal(app_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((app_dir / JOURNAL_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_journal(app_dir: Path, journal: dict[str, Any]) -> None:
    _write_atomically(app_dir / JOURNAL_NAME, json.dumps(journal, indent=2))


def _stale_staging(destination: Path) -> list[Path]:
    """Incomplete copies an interrupted run of this tool left beside *destination*."""
    parent = destination.parent
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.glob(f".{destination.name}.orion-staging-*")
                  if p.is_dir() and (p / STAGING_MARKER).is_file())


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _within(inner: Path, outer: Path) -> bool:
    try:
        inner.resolve().relative_to(outer.resolve())
        return True
    except ValueError:
        return False


# ── the move ─────────────────────────────────────────────────────────────────

def key_store_note(resolution: Resolution, destination: Path) -> str:
    """Why a source checkout's API keys would stop being found, or "".

    orion_core.constants._resolve_key_store lets a checkout with no
    api_keys.json of its own borrow the installed app's store, but only while
    its data dir is the in-project config/. Once a pointer moves it, "an
    explicit config dir owns its keys". Checked by existence only: this tool
    never reads or copies a key file on its own initiative.
    """
    installed = Path.home() / "ORION" / "config" / "api_keys.json"
    if (resolution.via != "default" or not is_source_checkout(resolution.app_dir)
            or (resolution.data_dir / "api_keys.json").exists() or not installed.exists()):
        return ""
    return (f"this checkout has no api_keys.json of its own and currently uses {installed}. "
            f"After the move its source runs look for {destination / 'api_keys.json'} "
            "instead; put a key store there (or none, to run without cloud providers).")


def _print_plan(resolution: Resolution, destination: Path, inventory: Any,
                total: int) -> None:
    databases, files, sidecars = inventory
    print(f"  installation : {resolution.app_dir}"
          f"{' (source checkout)' if is_source_checkout(resolution.app_dir) else ''}")
    print(f"  data now     : {resolution.data_dir}  (via {resolution.via})")
    if resolution.dangling_pointer:
        print(f"  note         : {resolution.pointer.name} names {resolution.pointer_text!r}, "
              "which does not exist, so ORION ignores it")
    print(f"  sync folder  : {looks_synced(resolution.data_dir) or 'no'}")
    print(f"  destination  : {destination}")
    print(f"  pointer      : {resolution.pointer}  ->  {destination}")
    print(f"  contents     : {len(databases)} database(s) by SQLite backup, "
          f"{len(files)} other file(s) by copy, {total / 1024 / 1024:,.1f} MB")
    if sidecars:
        print(f"                 {len(sidecars)} live -wal/-shm file(s) are NOT copied "
              "loose; their contents go in with each backup")
    installations = known_installations()
    if resolution.app_dir.resolve() not in {p.resolve() for p in installations}:
        return
    for other in installations:
        if other.resolve() == resolution.app_dir.resolve():
            continue
        elsewhere = resolve_data_dir(other)
        print(f"  also here    : {other} keeps its data at {elsewhere.data_dir} "
              f"(via {elsewhere.via}); this run does not change it")


def move(app_dir: Path, destination: Path, dry_run: bool = False) -> int:
    app_dir = Path(app_dir)
    destination = Path(destination)
    resolution = resolve_data_dir(app_dir)
    source = resolution.data_dir

    if not source.is_dir():
        print(f"No data directory at {source}")
        return 1
    if resolution.via == "environment":
        print(f"  ORION_CONFIG_DIR is set ({source}). It outranks the pointer file, so "
              "moving the data and switching the pointer would change nothing while it "
              "is set. Unset it first, or move that directory by hand.")
        return 1
    if destination.resolve() == source.resolve() or _within(destination, source) \
            or _within(source, destination):
        print(f"  {destination} overlaps the data it would receive ({source}). "
              "Choose a destination outside it.")
        return 1

    # Names and sizes only for the plan: no file is opened until the copy, so
    # a dry run reads nothing (databases are confirmed by header when copied).
    everything = [p for p in source.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in everything)
    sidecars = [p for p in everything if p.name.endswith(_SIDECARS)]
    databases = [p for p in everything
                 if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"} and p not in sidecars]
    others = [p for p in everything if p not in sidecars and p not in databases]
    _print_plan(resolution, destination, (databases, others, sidecars), total)
    keys = key_store_note(resolution, destination)
    if keys:
        print(f"  keys         : {keys}")
    stale = _stale_staging(destination)
    journal = read_journal(app_dir)
    if journal and journal.get("state") in {"copying", "failed"}:
        print(f"  earlier run  : a migration started {journal.get('started', '?')} did not "
              "finish; ORION kept using the original. Its partial copy will be cleared.")
    occupied = destination.exists() and (not destination.is_dir() or any(destination.iterdir()))

    if dry_run:
        running = orion_is_running()
        print(f"  ORION        : {'RUNNING - stop him before the real run' if running else 'not running'}")
        if occupied:
            print("  destination  : NOT EMPTY - the real run would refuse")
        print("\n  --dry-run only; nothing was copied or written.")
        return 0

    if orion_is_running():
        print("\n  ORION is running. Copying a database out from under a live "
              "writer is how you manufacture the corruption this prevents.")
        return 1
    if is_virtualised(destination):
        print(f"\n  {destination} is under AppData\\Local and this is the "
              "Microsoft Store build of Python, whose writes there are "
              "redirected into a private sandbox. The copy would appear to "
              "succeed and ORION.exe would never see it. Choose a destination "
              "outside AppData (for example "
              f"{Path.home() / 'ORION' / 'config'}).")
        return 1
    if occupied:
        described = ""
        if journal and journal.get("destination") == str(destination) \
                and journal.get("state") == "copied":
            described = (" It holds a verified copy from a migration that stopped before "
                         "the pointer switched; ORION never used it.")
        print(f"\n  {destination} already exists and is not empty.{described} Refusing to "
              "overwrite - move it aside or choose another destination.")
        return 1
    needed = int(total * 1.1) + 50 * 1024 * 1024
    if _free_bytes(destination) < needed:
        print(f"\n  Not enough free space at {destination} ({needed / 2**20:,.0f} MB needed).")
        return 1

    with hold_instance_lock(source) as held:
        if held is None:
            print("\n  ORION's single-instance lock is held - ORION is running. "
                  "Stop him first.")
            return 1
        return _copy_verify_switch(app_dir, resolution, destination, stale)


def _copy_verify_switch(app_dir: Path, resolution: Resolution, destination: Path,
                        stale: list[Path]) -> int:
    source = resolution.data_dir
    for leftover in stale:
        shutil.rmtree(leftover, ignore_errors=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    staging = destination.parent / f".{destination.name}.orion-staging-{stamp}"
    journal: dict[str, Any] = {
        "version": 1, "state": "copying", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "app_dir": str(app_dir), "source": str(source), "source_via": resolution.via,
        "destination": str(destination), "staging": str(staging),
        "previous_pointer": resolution.pointer_text,
    }
    _save_journal(app_dir, journal)
    staging.mkdir(parents=True)
    (staging / STAGING_MARKER).write_text(stamp, encoding="utf-8")

    def fail(reason: str, problems: list[str]) -> int:
        print(f"  {reason} ({len(problems)}): " + "; ".join(problems[:5]))
        print("  The pointer was NOT written. The original is untouched.")
        shutil.rmtree(staging, ignore_errors=True)
        journal.update(state="failed", reason=reason, problems=problems[:20])
        _save_journal(app_dir, journal)
        return 1

    try:
        databases, files, sidecars = _inventory(source)
        print(f"\n  copying {len(databases)} database(s) through the SQLite backup API...")
        for relative in databases:
            backup_database(source / relative, staging / relative)
        print(f"  copying {len(files)} other file(s)...")
        for relative in files:
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, staging / relative)
    except Exception as exc:
        return fail("COPY FAILED", [str(exc)])

    print("  verifying: every file present...")
    problems = compare_trees(source, staging, databases)
    if problems:
        return fail("COPY INCOMPLETE", problems)
    print("  verifying: integrity_check and row counts of every database...")
    problems = verify_databases(source, staging, databases)
    if problems:
        return fail("DATABASES DO NOT MATCH", problems)
    print("  verifying: SHA-256 of every other file...")
    problems = verify_files(source, staging, files)
    if problems:
        return fail("FILES DO NOT MATCH", problems)
    print("  confirming the copy is visible outside this process...")
    unseen = _visible_to_others(staging)
    if unseen:
        return fail("THE COPY IS NOT WHERE IT APPEARS TO BE", [unseen])

    try:
        (staging / STAGING_MARKER).unlink()
        if destination.is_dir():
            destination.rmdir()          # checked empty above
        _replace(str(staging), destination)
    except OSError as exc:
        return fail("COULD NOT PUT THE COPY IN PLACE", [str(exc)])
    journal.update(state="copied", databases=len(databases), files=len(files),
                   live_sidecars_folded=len(sidecars))
    _save_journal(app_dir, journal)

    try:
        _write_atomically(resolution.pointer, str(destination))
    except OSError as exc:
        print(f"  THE POINTER COULD NOT BE WRITTEN ({exc}). ORION still uses {source}; "
              f"the verified copy is at {destination}.")
        return 1
    journal.update(state="switched", finished=time.time(),
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _save_journal(app_dir, journal)
    print(f"\n  done. {resolution.pointer.name} now points at {destination}.")
    print(f"  The original is still at {source}, untouched - delete it yourself once "
          "you are satisfied.")
    print(f"  To undo: python tools/move_config.py --app \"{app_dir}\" --rollback")
    keys = key_store_note(resolution, destination)
    if keys:
        print(f"  KEYS: {keys}")
    return 0


# ── rollback ─────────────────────────────────────────────────────────────────

def _changed_since(directory: Path, moment: float) -> list[Path]:
    """Files under *directory* written after *moment* (two seconds' grace)."""
    changed: list[Path] = []
    for item in directory.rglob("*"):
        try:
            if item.is_file() and item.stat().st_mtime > moment + 2.0:
                changed.append(item)
        except OSError:
            continue
    return changed


def rollback(app_dir: Path, force: bool = False) -> int:
    """Put the pointer back as it was before the last migration.

    Both copies stay where they are. If ORION has written to the new location
    since the move, going back would silently lose those writes, so that
    needs --force.
    """
    app_dir = Path(app_dir)
    pointer = app_dir / CONFIG_POINTER_NAME
    journal = read_journal(app_dir)
    if orion_is_running():
        print("  ORION is running. Stop him before switching his data back.")
        return 1
    if not journal or journal.get("state") == "rolled_back":
        # No record of a migration (or it was already undone): the behaviour
        # of the tool before it kept one — drop the pointer, keep both copies.
        if not pointer.is_file():
            print("  No pointer file; ORION is already using the original.")
            return 0
        target = pointer.read_text(encoding="utf-8").strip()
        pointer.unlink()
        print(f"  Pointer removed. ORION goes back to {app_dir / 'config'}.")
        print(f"  The moved copy is still at {target} - delete it yourself.")
        return 0

    destination = Path(journal.get("destination", ""))
    state = journal.get("state")
    with hold_instance_lock(Path(journal.get("source") or app_dir)) as held:
        if held is None:
            print("  ORION's single-instance lock is held - ORION is running. Stop him first.")
            return 1
        if state == "switched":
            changed = _changed_since(destination, float(journal.get("finished") or 0)) \
                if destination.is_dir() else []
            if changed and not force:
                print(f"  ORION has written to {destination} since the move "
                      f"({len(changed)} file(s), e.g. {changed[0].name}). The original "
                      f"at {journal.get('source')} does not have those changes, so going "
                      "back would lose them. Re-run with --force to roll back anyway.")
                return 1
            previous = journal.get("previous_pointer")
            if previous is None:
                if pointer.is_file():
                    pointer.unlink()
            else:
                _write_atomically(pointer, previous)
        else:
            # The pointer never switched; clear any partial copy of our own.
            staging = Path(journal.get("staging") or "")
            if staging.is_dir() and (staging / STAGING_MARKER).is_file():
                shutil.rmtree(staging, ignore_errors=True)
        journal.update(state="rolled_back", rolled_back_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        _save_journal(app_dir, journal)
    now = resolve_data_dir(app_dir)
    print(f"  Rolled back. ORION now uses {now.data_dir} (via {now.via}).")
    if destination.is_dir():
        print(f"  The copy at {destination} is left where it is - delete it yourself.")
    return 0


def revert(app_dir: Path, force: bool = False) -> int:
    """The earlier name for rollback()."""
    return rollback(app_dir, force=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", default=str(ROOT),
                        help="the installation: this checkout (default) or e.g. dist/ORION")
    parser.add_argument("--to", default=None, help="destination directory")
    parser.add_argument("--dry-run", "--plan", dest="dry_run", action="store_true",
                        help="print the plan; read, copy and write nothing")
    parser.add_argument("--rollback", "--revert", dest="rollback", action="store_true",
                        help="restore the pointer as it was before the last migration")
    parser.add_argument("--force", action="store_true",
                        help="with --rollback: go back even if the new location has changed")
    args = parser.parse_args(argv)

    app_dir = Path(args.app).resolve()
    if args.rollback:
        return rollback(app_dir, force=args.force)
    destination = Path(args.to).resolve() if args.to else default_destination(app_dir)
    return move(app_dir, destination, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
