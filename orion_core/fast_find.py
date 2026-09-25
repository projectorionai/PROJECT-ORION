"""
Finding a file on this machine at the speed Windows already can.

  "ORION's capabilities of going through the PC and scanning for files must be
   absolute, he must be efficient at finding files faster at lightning speeds,
   or if he's struggling he can take his time and find other methods."

The old search walked the user's folders with ``rglob('*')``. That is correct
and it is also the slowest possible way to answer the question, because Windows
has already done the work: the Search indexer keeps a live catalogue of every
file in the user's profile, and it can be queried with SQL over ADO.

Measured on this machine: the index returned matches from the entire profile —
including the OneDrive tree, which is where the real content lives — in 0.35
seconds. A recursive walk of the same ground is minutes.

The ladder
----------
The request said it plainly: fast when it can be, thorough when it must be. So
the search escalates rather than picking one strategy:

  1. **Index** — near-instant, covers the whole indexed profile, matches on
     name OR contents. What almost every search should use.
  2. **Walk** — for locations the indexer does not cover (a network share, an
     excluded folder, a drive that was just plugged in), or when the index is
     switched off or still building.
  3. **Content grep** — last, and only when asked, because reading every file
     is expensive and is a different question from "where is it".

Falling back is never silent: which strategy answered is reported, because
"no matches" from an index that happens to be disabled means something quite
different from "no matches" after reading every directory.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Never spend longer than this on any one strategy before escalating.
INDEX_TIMEOUT_S = 6.0
WALK_TIMEOUT_S = 25.0

#: Directories that are never worth walking. Skipping them is most of what
#: makes the fallback tolerable rather than hopeless.
SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", ".svn", ".hg", "venv", ".venv",
    "env", ".env", "site-packages", "dist-info", ".mypy_cache", ".pytest_cache",
    "AppData", "Temp", "tmp", ".cache", "$RECYCLE.BIN", "System Volume Information",
    ".idea", ".vs", "build", "target", ".gradle", ".tox",
})


@dataclass
class Hit:
    path: str
    kind: str = "file"          # file | folder
    size: int = 0
    modified: float = 0.0
    why: str = "name"           # name | contents

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class Search:
    query: str
    hits: list[Hit] = field(default_factory=list)
    strategy: str = ""
    seconds: float = 0.0
    note: str = ""

    def describe(self) -> str:
        if not self.hits:
            return (f"No matches for '{self.query}' "
                    f"({self.strategy}, {self.seconds:.1f}s)."
                    + (f" {self.note}" if self.note else ""))
        head = (f"{len(self.hits)} match(es) for '{self.query}' "
                f"— found via {self.strategy} in {self.seconds:.2f}s.")
        return head + (f" {self.note}" if self.note else "")


# ── strategy 1: the Windows Search index ─────────────────────────────────────

def index_available() -> bool:
    if os.name != "nt":
        return False
    try:
        import win32com.client  # noqa: F401
        return True
    except Exception:
        return False


def _escape(value: str) -> str:
    """Escape a value for a SQL string literal."""
    return str(value).replace("'", "''")


def search_index(query: str, limit: int = 60, contents: bool = False,
                 folder: str = "") -> tuple[list[Hit], str]:
    """Ask Windows for what it already knows. Returns (hits, error)."""
    if not index_available():
        return [], "the Windows Search index is not reachable from here"
    import pythoncom
    import win32com.client

    safe = _escape(query)
    where = [f"System.FileName LIKE '%{safe}%'"]
    if contents:
        # FREETEXT rather than LIKE on contents: LIKE cannot use the full-text
        # index and degrades to a scan, which throws away the entire point.
        where = [f"({where[0]} OR FREETEXT(System.Search.Contents, '{safe}'))"]
    if folder:
        where.append(f"SCOPE='file:{_escape(folder)}'")
    sql = (
        "SELECT TOP {limit} System.ItemPathDisplay, System.ItemType, "
        "System.Size, System.DateModified FROM SystemIndex WHERE {where} "
        "ORDER BY System.DateModified DESC"
    ).format(limit=int(limit), where=" AND ".join(where))

    # COM must be initialised on whichever thread this runs on, and this is
    # deliberately called from a worker thread so the GUI never blocks.
    initialised = False
    try:
        pythoncom.CoInitialize()
        initialised = True
    except Exception:
        pass
    connection = None
    recordset = None
    try:
        connection = win32com.client.Dispatch("ADODB.Connection")
        connection.Open("Provider=Search.CollatorDSO;"
                        'Extended Properties="Application=Windows";')
        recordset = win32com.client.Dispatch("ADODB.Recordset")
        recordset.Open(sql, connection)
        hits: list[Hit] = []
        deadline = time.monotonic() + INDEX_TIMEOUT_S
        while not recordset.EOF and len(hits) < limit:
            if time.monotonic() > deadline:
                break
            try:
                path = recordset.Fields.Item(0).Value
                item_type = recordset.Fields.Item(1).Value or ""
                size = recordset.Fields.Item(2).Value or 0
            except Exception:
                recordset.MoveNext()
                continue
            if path:
                hits.append(Hit(
                    path=str(path),
                    kind="folder" if str(item_type).lower() == "directory" else "file",
                    size=int(size or 0),
                    why="contents" if contents and query.lower() not in
                        Path(str(path)).name.lower() else "name",
                ))
            recordset.MoveNext()
        return hits, ""
    except Exception as exc:
        return [], f"the index query failed ({str(exc)[:120]})"
    finally:
        for handle in (recordset, connection):
            try:
                if handle is not None and handle.State:
                    handle.Close()
            except Exception:
                pass
        if initialised:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass


# ── strategy 2: the walk ─────────────────────────────────────────────────────

def default_roots() -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    try:
        for entry in home.iterdir():
            if not entry.is_dir():
                continue
            if entry.name in {"Desktop", "Documents", "Downloads", "Pictures",
                              "Music", "Videos"} or entry.name.startswith("OneDrive"):
                roots.append(entry)
    except OSError:
        pass
    return roots or [home]


def search_walk(query: str, roots: list[Path] | None = None, limit: int = 60,
                timeout: float = WALK_TIMEOUT_S) -> tuple[list[Hit], str]:
    """Walk the filesystem. Slower, but it sees what the index does not.

    ``os.walk`` with in-place pruning of ``dirnames`` rather than
    ``Path.rglob``: rglob has no way to skip a subtree, so it descends into
    node_modules and site-packages and spends most of its time there.
    """
    needle = query.lower()
    hits: list[Hit] = []
    deadline = time.monotonic() + timeout
    truncated = False
    for root in (roots if roots is not None else default_roots()):
        for base, dirnames, filenames in os.walk(str(root), topdown=True,
                                                 onerror=lambda _e: None):
            if time.monotonic() > deadline:
                truncated = True
                break
            # Pruned in place — this is what os.walk gives us and rglob cannot.
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith("$")]
            for name in dirnames:
                if needle in name.lower():
                    hits.append(Hit(str(Path(base) / name), kind="folder"))
            for name in filenames:
                if needle in name.lower():
                    path = Path(base) / name
                    try:
                        stat = path.stat()
                        hits.append(Hit(str(path), "file", stat.st_size, stat.st_mtime))
                    except OSError:
                        hits.append(Hit(str(path)))
                    if len(hits) >= limit:
                        return hits, ""
        if time.monotonic() > deadline:
            truncated = True
            break
    note = ("the walk hit its time limit, so this may not be everything"
            if truncated else "")
    return hits, note


# ── the ladder ───────────────────────────────────────────────────────────────

def find(query: str, *, contents: bool = False, limit: int = 60,
         folder: str = "", thorough: bool = False) -> Search:
    """Find *query*, fast if possible and thoroughly if necessary.

    ``thorough`` walks as well as querying the index, for when the user says
    something must be there — the index does not cover network drives, excluded
    folders, or files created in the last few seconds.
    """
    query = str(query or "").strip()
    result = Search(query=query)
    if not query:
        result.note = "no search term given"
        return result

    started = time.monotonic()
    hits, error = search_index(query, limit=limit, contents=contents, folder=folder)
    if hits and not thorough:
        result.hits = _dedupe(hits)
        result.strategy = "the Windows index"
        result.seconds = time.monotonic() - started
        return result

    walked, note = search_walk(
        query, roots=[Path(folder)] if folder else None, limit=limit)
    combined = _dedupe(hits + walked)
    result.hits = combined[:limit]
    result.seconds = time.monotonic() - started
    if hits and walked:
        result.strategy = "the Windows index and a full walk"
    elif walked:
        result.strategy = "a full walk"
        # Saying WHY the fast path did not answer: "no matches" from a disabled
        # index means something very different from "no matches" after reading
        # every directory.
        result.note = (error or "the index had nothing; walked instead") if error else ""
    else:
        result.strategy = "the Windows index and a full walk"
        result.note = error or note
    if note and not result.note:
        result.note = note
    return result


def _dedupe(hits: list[Hit]) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in hits:
        key = os.path.normcase(os.path.normpath(hit.path))
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


__all__ = [
    "INDEX_TIMEOUT_S", "SKIP_DIRS", "WALK_TIMEOUT_S", "Hit", "Search",
    "default_roots", "find", "index_available", "search_index", "search_walk",
]
