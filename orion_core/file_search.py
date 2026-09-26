"""
Find anything on this PC — by name, type, size and date.

  "I want ORION to be able to research specific files or folders in the
   entirety of the PC - this can include from name, size, date made and etc"

``fast_find`` answered one question — "where is a file called X" — and only
inside the user's profile folders. This answers the questions people actually
ask a file system: *the PDFs I downloaded last week*, *everything over 2 GB*,
*the spreadsheet I made in March*, *which folder is eating the disk*, across
every drive.

Three engines, fastest first, and the answer says which one spoke:

    Everything   voidtools' index of every NTFS volume. Milliseconds for the
                 whole PC, sizes and dates included. Used when es.exe and the
                 Everything service are present.
    Windows      The Windows Search index, through the same ADO provider
                 fast_find uses. Profile folders and anything else indexed.
    Walk         os.walk over every fixed drive with the heavy system trees
                 pruned, inside a time budget. Slow, but it sees everything.

Whatever engine answered, every filter is applied again here on the real
file's size and timestamps — so a result can never contradict the question
because an engine interpreted "last week" differently.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

#: Extensions per kind. Deliberately plain words — these arrive by voice.
KINDS: dict[str, tuple[str, ...]] = {
    "document": ("doc", "docx", "odt", "rtf", "txt", "md", "pdf", "pages", "tex"),
    "pdf": ("pdf",),
    "word": ("doc", "docx", "odt", "rtf"),
    "spreadsheet": ("xls", "xlsx", "xlsm", "csv", "ods", "numbers"),
    "presentation": ("ppt", "pptx", "odp", "key"),
    "image": ("jpg", "jpeg", "png", "gif", "bmp", "webp", "heic", "tif", "tiff",
              "svg", "raw", "cr2", "nef", "ico"),
    "photo": ("jpg", "jpeg", "png", "heic", "raw", "cr2", "nef", "webp"),
    "video": ("mp4", "mkv", "mov", "avi", "wmv", "webm", "m4v", "flv"),
    "audio": ("mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus"),
    "archive": ("zip", "rar", "7z", "tar", "gz", "bz2", "xz", "iso"),
    "code": ("py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "hpp",
             "cs", "go", "rs", "rb", "php", "html", "css", "json", "yaml",
             "yml", "toml", "sql", "sh", "ps1", "ipynb"),
    "executable": ("exe", "msi", "bat", "cmd", "ps1", "lnk", "appx", "msix"),
    "email": ("eml", "msg", "pst", "ost"),
    "font": ("ttf", "otf", "woff", "woff2"),
    "3d": ("stl", "obj", "fbx", "blend", "glb", "gltf", "3mf", "step", "stp"),
}
_KIND_ALIASES = {
    "documents": "document", "docs": "document", "pdfs": "pdf",
    "spreadsheets": "spreadsheet", "excel": "spreadsheet", "sheets": "spreadsheet",
    "presentations": "presentation", "powerpoint": "presentation", "slides": "presentation",
    "images": "image", "pictures": "image", "picture": "image", "photos": "photo",
    "videos": "video", "movies": "video", "films": "video", "music": "audio",
    "songs": "audio", "sound": "audio", "archives": "archive", "zips": "archive",
    "programs": "executable", "apps": "executable", "installers": "executable",
    "emails": "email", "fonts": "font", "models": "3d",
}

#: Trees a whole-PC walk skips unless asked for system files. They are huge,
#: rarely what anybody means, and several deny access anyway.
SYSTEM_SKIP = frozenset({
    "windows", "$recycle.bin", "system volume information", "programdata",
    "$windows.~bt", "$windows.~ws", "recovery", "windowsapps", "winsxs",
})
#: Skipped everywhere, always: build and package caches that dwarf real files.
NOISE_SKIP = frozenset({
    "node_modules", "__pycache__", ".git", ".venv", "venv", "site-packages",
    ".cache", ".gradle", ".m2", ".nuget", "cache", "caches",
})

#: Seconds a full-drive walk may take before it reports what it has.
WALK_BUDGET_S = 25.0


# ── the question ─────────────────────────────────────────────────────────────

@dataclass
class FileQuery:
    """What is being looked for. Every field optional; empty means "anything"."""

    name: str = ""
    extensions: tuple[str, ...] = ()
    min_size: int | None = None
    max_size: int | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    folder: str = ""
    kind: str = ""                   # "file", "folder" or "" for both
    include_system: bool = False
    sort: str = "relevance"          # relevance newest oldest largest smallest name
    limit: int = 40

    def is_empty(self) -> bool:
        return not any((self.name, self.extensions, self.min_size, self.max_size,
                        self.modified_after, self.modified_before,
                        self.created_after, self.created_before, self.folder))

    def describe(self) -> str:
        bits = []
        if self.name:
            bits.append(f"named like '{self.name}'")
        if self.extensions:
            bits.append("of type " + ", ".join("." + e for e in self.extensions[:8]))
        if self.min_size:
            bits.append(f"over {human_size(self.min_size)}")
        if self.max_size:
            bits.append(f"under {human_size(self.max_size)}")
        if self.modified_after or self.modified_before:
            bits.append("modified " + _range_text(self.modified_after, self.modified_before))
        if self.created_after or self.created_before:
            bits.append("created " + _range_text(self.created_after, self.created_before))
        if self.folder:
            bits.append(f"in {self.folder}")
        return (("folders " if self.kind == "folder" else "files " if self.kind == "file"
                 else "items ") + " ".join(bits)).strip()


@dataclass
class FileHit:
    path: str
    is_folder: bool = False
    size: int = 0
    modified: float = 0.0
    created: float = 0.0

    @property
    def name(self) -> str:
        return Path(self.path).name

    def line(self) -> str:
        when = datetime.fromtimestamp(self.modified).strftime("%d %b %Y %H:%M") \
            if self.modified else "?"
        size = "folder" if self.is_folder else human_size(self.size)
        return f"{self.path}  [{size}, modified {when}]"


@dataclass
class FileResults:
    query: FileQuery
    hits: list[FileHit] = field(default_factory=list)
    engine: str = ""
    seconds: float = 0.0
    note: str = ""
    truncated: bool = False

    def summary(self) -> str:
        what = self.query.describe()
        if not self.hits:
            return (f"No {what} found (searched with {self.engine or 'nothing'}, "
                    f"{self.seconds:.1f}s).{(' ' + self.note) if self.note else ''}")
        total = sum(h.size for h in self.hits if not h.is_folder)
        return (f"{len(self.hits)} {what} — found with {self.engine} in "
                f"{self.seconds:.2f}s"
                + (f", {human_size(total)} in total" if total else "")
                + ("; there may be more than shown" if self.truncated else "")
                + (f". {self.note}" if self.note else "."))


# ── parsing what people say ──────────────────────────────────────────────────

_UNITS = {"b": 1, "byte": 1, "bytes": 1, "k": 1024, "kb": 1024, "kib": 1024,
          "m": 1024 ** 2, "mb": 1024 ** 2, "mib": 1024 ** 2, "meg": 1024 ** 2,
          "megs": 1024 ** 2, "g": 1024 ** 3, "gb": 1024 ** 3, "gib": 1024 ** 3,
          "gig": 1024 ** 3, "gigs": 1024 ** 3, "t": 1024 ** 4, "tb": 1024 ** 4}
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]*)", re.I)


def human_size(value: int | float) -> str:
    value = float(value or 0)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def parse_size(text: Any) -> int | None:
    """'100MB', '1.5 gb', '500k', 2048 -> bytes. None if it is not a size."""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return int(text)
    match = _SIZE.search(str(text).strip().lower().replace(",", ""))
    if not match:
        return None
    unit = match.group(2) or "b"
    if unit not in _UNITS:
        return None
    return int(float(match.group(1)) * _UNITS[unit])


def parse_size_bounds(text: Any) -> tuple[int | None, int | None]:
    """'over 1GB', '>100mb', 'under 10 MB', 'between 1mb and 5mb'."""
    raw = str(text or "").strip().lower()
    if not raw:
        return None, None
    between = re.search(r"between\s+(.+?)\s+and\s+(.+)", raw)
    if between:
        return parse_size(between.group(1)), parse_size(between.group(2))
    size = parse_size(raw)
    if size is None:
        return None, None
    if re.search(r"<|under|less|smaller|below|at most|max", raw):
        return None, size
    return size, None


_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), 1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})


def _start_of_day(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return start, end


def parse_when(text: Any, now: datetime | None = None) -> tuple[datetime | None, datetime | None]:
    """A spoken time range -> (after, before). Either end may be open.

    today, yesterday, this/last week|month|year, last 3 days, in 2025,
    in March, March 2025, since 1 June, before 2026-01-01, after 5 May 2025.
    """
    raw = str(text or "").strip().lower()
    if not raw:
        return None, None
    now = now or datetime.now()
    today = _start_of_day(now)
    if raw in {"today", "this day"}:
        return today, None
    if raw == "yesterday":
        return today - timedelta(days=1), today
    if raw in {"this week", "thisweek"}:
        return today - timedelta(days=today.weekday()), None
    if raw == "last week":
        start = today - timedelta(days=today.weekday() + 7)
        return start, start + timedelta(days=7)
    if raw in {"this month", "thismonth"}:
        return today.replace(day=1), None
    if raw == "last month":
        first = today.replace(day=1)
        previous = (first - timedelta(days=1)).replace(day=1)
        return previous, first
    if raw in {"this year", "thisyear"}:
        return today.replace(month=1, day=1), None
    if raw == "last year":
        return datetime(now.year - 1, 1, 1), datetime(now.year, 1, 1)
    recent = re.match(r"(?:in the |within the )?(?:last|past)\s+(\d+)\s*(day|week|month|year|hour)s?", raw)
    if recent:
        count, unit = int(recent.group(1)), recent.group(2)
        days = {"hour": count / 24, "day": count, "week": 7 * count,
                "month": 30 * count, "year": 365 * count}[unit]
        return now - timedelta(days=days), None
    for word, direction in (("before", "before"), ("until", "before"),
                            ("after", "after"), ("since", "after")):
        if raw.startswith(word + " "):
            point = _parse_point(raw[len(word) + 1:], now)
            if point is None:
                return None, None
            return (point, None) if direction == "after" else (None, point)
    raw = re.sub(r"^(?:in|during|from)\s+", "", raw)
    year_only = re.fullmatch(r"(19|20)\d\d", raw)
    if year_only:
        year = int(raw)
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)
    month_year = re.fullmatch(r"([a-z]+)\s*((?:19|20)\d\d)?", raw)
    if month_year and month_year.group(1) in _MONTHS:
        month = _MONTHS[month_year.group(1)]
        year = int(month_year.group(2)) if month_year.group(2) else (
            now.year if month <= now.month else now.year - 1)
        return _month_bounds(year, month)
    point = _parse_point(raw, now)
    if point is not None:
        return point, point + timedelta(days=1)
    return None, None


def _parse_point(text: str, now: datetime) -> datetime | None:
    text = text.strip().lower().replace(",", "")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y",
                "%B %d %Y", "%b %d %Y", "%d %B", "%d %b", "%B %Y", "%b %Y"):
        try:
            value = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:
            value = value.replace(year=now.year)
        return value
    if re.fullmatch(r"(19|20)\d\d", text):
        return datetime(int(text), 1, 1)
    return None


def _range_text(after: datetime | None, before: datetime | None) -> str:
    fmt = "%d %b %Y"
    if after and before:
        return f"between {after.strftime(fmt)} and {before.strftime(fmt)}"
    if after:
        return f"since {after.strftime(fmt)}"
    return f"before {before.strftime(fmt)}" if before else ""


def extensions_for(kind: Any, extension: Any = None) -> tuple[str, ...]:
    """Extensions for a kind word ('pdfs', 'spreadsheets') and/or explicit ones."""
    out: list[str] = []
    for word in re.split(r"[,;/\s]+", str(kind or "").lower()):
        word = _KIND_ALIASES.get(word, word)
        out.extend(KINDS.get(word, ()))
    for ext in re.split(r"[,;\s]+", str(extension or "").lower()):
        ext = ext.strip().lstrip("*").lstrip(".")
        if ext:
            out.append(ext)
    seen: list[str] = []
    for ext in out:
        if ext not in seen:
            seen.append(ext)
    return tuple(seen)


def build_query(args: dict[str, Any]) -> FileQuery:
    """A FileQuery from tool arguments (which arrive from speech)."""
    kind_word = str(args.get("type") or args.get("kind") or "").strip().lower()
    item_kind = ""
    if kind_word in {"folder", "folders", "directory", "directories"}:
        item_kind, kind_word = "folder", ""
    elif kind_word in {"file", "files"}:
        item_kind, kind_word = "file", ""
    min_size = parse_size(args.get("min_size"))
    max_size = parse_size(args.get("max_size"))
    if args.get("size"):
        low, high = parse_size_bounds(args.get("size"))
        min_size = min_size if min_size is not None else low
        max_size = max_size if max_size is not None else high
    m_after, m_before = parse_when(args.get("modified") or args.get("date") or "")
    c_after, c_before = parse_when(args.get("created") or "")
    try:
        limit = max(1, min(500, int(args.get("limit") or 40)))
    except (TypeError, ValueError):
        limit = 40
    return FileQuery(
        name=str(args.get("query") or args.get("name") or "").strip(),
        extensions=extensions_for(kind_word, args.get("extension") or args.get("ext")),
        min_size=min_size, max_size=max_size,
        modified_after=m_after, modified_before=m_before,
        created_after=c_after, created_before=c_before,
        folder=os.path.expandvars(os.path.expanduser(str(args.get("folder") or args.get("in") or ""))),
        kind=item_kind or ("file" if (min_size or max_size) and not kind_word else ""),
        include_system=bool(args.get("system") or args.get("include_system")),
        sort=str(args.get("sort") or "relevance").strip().lower(),
        limit=limit,
    )


# ── matching, applied whatever engine answered ───────────────────────────────

def _name_matches(name: str, pattern: str) -> bool:
    if not pattern:
        return True
    low = name.lower()
    pattern = pattern.lower()
    if any(ch in pattern for ch in "*?"):
        import fnmatch
        return fnmatch.fnmatch(low, pattern)
    words = [w for w in re.split(r"\s+", pattern) if w]
    return all(w in low for w in words)


def matches(hit: FileHit, q: FileQuery) -> bool:
    if q.kind == "folder" and not hit.is_folder:
        return False
    if q.kind == "file" and hit.is_folder:
        return False
    if not _name_matches(hit.name, q.name):
        return False
    if q.extensions:
        if hit.is_folder or Path(hit.path).suffix.lower().lstrip(".") not in q.extensions:
            return False
    if not hit.is_folder:
        if q.min_size is not None and hit.size < q.min_size:
            return False
        if q.max_size is not None and hit.size > q.max_size:
            return False
    for stamp, after, before in ((hit.modified, q.modified_after, q.modified_before),
                                 (hit.created, q.created_after, q.created_before)):
        if (after or before) and not stamp:
            return False
        if after and stamp < after.timestamp():
            return False
        if before and stamp >= before.timestamp():
            return False
    return True


def _order(hits: list[FileHit], q: FileQuery) -> list[FileHit]:
    key = q.sort
    if key in {"newest", "recent", "latest", "date"}:
        return sorted(hits, key=lambda h: -h.modified)
    if key in {"oldest"}:
        return sorted(hits, key=lambda h: h.modified or float("inf"))
    if key in {"largest", "biggest", "size"}:
        return sorted(hits, key=lambda h: -h.size)
    if key in {"smallest"}:
        return sorted(hits, key=lambda h: h.size)
    if key in {"name", "alphabetical"}:
        return sorted(hits, key=lambda h: h.name.lower())
    if q.name:
        low = q.name.lower()
        # Exact name, then name-starts-with, then shortest names, newest first.
        return sorted(hits, key=lambda h: (h.name.lower() != low,
                                           not h.name.lower().startswith(low),
                                           len(h.name), -h.modified))
    return sorted(hits, key=lambda h: -h.modified)


def _stat_hit(path: str, is_folder: bool | None = None) -> FileHit | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    folder = os.path.isdir(path) if is_folder is None else is_folder
    created = getattr(stat, "st_birthtime", 0.0) or stat.st_ctime
    return FileHit(path, folder, 0 if folder else stat.st_size, stat.st_mtime, created)


# ── engine 1: Everything ─────────────────────────────────────────────────────

def everything_cli() -> str:
    """Path to es.exe, or "" when Everything's command line is not installed."""
    found = shutil.which("es.exe") or shutil.which("es")
    if found:
        return found
    for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", ""),
                 os.environ.get("ProgramFiles(x86)", "")):
        if not base:
            continue
        for candidate in (Path(base) / "Everything" / "es.exe",
                          Path(base) / "Microsoft" / "WinGet" / "Links" / "es.exe",
                          Path(base) / "voidtools" / "Everything" / "es.exe"):
            if candidate.exists():
                return str(candidate)
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if links.exists():
        for candidate in links.glob("voidtools.Everything.Cli*/es.exe"):
            return str(candidate)
    return ""


def _everything_terms(q: FileQuery) -> list[str]:
    """The search as separate es.exe arguments.

    Separate, and never quoted here. The old single string ended with
    exclusions like ``!"C:\\Windows\\"`` — a quoted path ending in a backslash,
    which reaches es.exe as an escaped quote and silently turned every search
    into one that matched nothing. The folder now goes through ``-path`` and
    the system trees are filtered afterwards (``_is_noise``).
    """
    terms: list[str] = []
    if q.name:
        terms += [q.name] if any(c in q.name for c in "*?") else q.name.split()
    if q.extensions:
        terms.append("ext:" + ";".join(q.extensions))
    if q.min_size is not None:
        terms.append(f"size:>={q.min_size}")
    if q.max_size is not None:
        terms.append(f"size:<={q.max_size}")
    for tag, after, before in (("dm", q.modified_after, q.modified_before),
                               ("dc", q.created_after, q.created_before)):
        if after:
            terms.append(f"{tag}:>={after.strftime('%Y-%m-%d')}")
        if before:
            terms.append(f"{tag}:<{before.strftime('%Y-%m-%d')}")
    return terms


#: Per-app stores under AppData that Everything sees and nobody means by "my
#: files" — a search for "proposal" found an app's cached skill folder first.
_APP_NOISE = ("\\appdata\\local\\temp\\", "\\appdata\\local\\packages\\")
#: Windows' own files at a drive root. "My biggest files" led with the 28 GB
#: pagefile, which is neither the user's nor something they can delete.
_ROOT_SYSTEM_FILES = frozenset({"pagefile.sys", "hiberfil.sys", "swapfile.sys",
                                "dumpstack.log", "dumpstack.log.tmp"})


def _is_noise(path: str, q: FileQuery) -> bool:
    """True for a hit inside a tree the walk would have pruned.

    Everything indexes the whole volume, so its answers need the pruning the
    walk applies as it goes. Only the part of the path below the folder that
    was asked about is judged: asking inside a cache searches the cache.
    """
    low = os.path.normcase(path)
    base = os.path.normcase(q.folder.rstrip("\\/")) if q.folder else ""
    if base and low.startswith(base):
        low = low[len(base):]
    folders = [part for part in low.replace("/", "\\").split("\\")[:-1] if part]
    if any(part in NOISE_SKIP for part in folders):
        return True
    if q.include_system:
        return False
    if any(part in SYSTEM_SKIP or part.startswith("$") for part in folders):
        return True
    if os.path.basename(low) in _ROOT_SYSTEM_FILES:
        return True
    return any(fragment in low for fragment in _APP_NOISE)


def search_everything(q: FileQuery, timeout: float = 15.0) -> tuple[list[FileHit], str]:
    exe = everything_cli()
    if not exe:
        return [], "Everything is not installed"
    sort = {"newest": "date-modified-descending", "recent": "date-modified-descending",
            "latest": "date-modified-descending", "oldest": "date-modified-ascending",
            "largest": "size-descending", "biggest": "size-descending",
            "smallest": "size-ascending", "name": "name-ascending"}.get(q.sort, "")
    # Newest first when no order was asked for, so the cap keeps the recent
    # matches ("bring up that proposal" means the latest one); _order still
    # puts an exact name ahead of them. The cap is generous because noise is
    # filtered after it.
    command = [exe, "-csv", "-full-path-and-name", "-size", "-date-modified",
               "-date-created", "-attributes", "-size-format", "1",
               "-date-format", "1", "-n", str(max(q.limit * 10, 500)),
               "-sort", sort or "date-modified-descending"]
    if q.folder:
        command += ["-path", q.folder]
    if q.kind == "folder":
        command.append("/ad")
    elif q.kind == "file" or q.extensions:
        command.append("/a-d")
    command += _everything_terms(q)
    try:
        from .quiet_subprocess import CREATE_NO_WINDOW
    except Exception:
        CREATE_NO_WINDOW = 0x08000000
    try:
        done = subprocess.run(command, capture_output=True, timeout=timeout,
                              creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"Everything did not answer ({type(exc).__name__})"
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or b"").decode("utf-8", "replace").strip()
        return [], f"Everything is not running ({detail[:80] or done.returncode})"
    text = done.stdout.decode("utf-8-sig", "replace")
    hits: list[FileHit] = []
    for row in csv.DictReader(io.StringIO(text)):
        path = str(row.get("Filename") or row.get("Full Path & Name") or "")
        if len(path) > 3:
            path = path.rstrip("\\")         # folders arrive as "...\name\"
        if not path or _is_noise(path, q):
            continue
        # Attributes arrive as the Win32 flag number ("16" is a folder), not
        # as letters: read as letters, every folder was taken for a file.
        attributes = str(row.get("Attributes") or "").strip()
        try:
            folder = bool(int(attributes) & 0x10)
        except ValueError:
            folder = "D" in attributes.upper()

        def _stamp(value: str) -> float:
            try:
                return datetime.fromisoformat(str(value).strip().replace("Z", "")).timestamp()
            except Exception:
                return 0.0

        try:
            size = int(row.get("Size") or 0)
        except ValueError:
            size = 0
        hits.append(FileHit(path, folder, size, _stamp(row.get("Date Modified", "")),
                            _stamp(row.get("Date Created", ""))))
    return hits, ""


# ── engine 2: the Windows Search index ───────────────────────────────────────

def _index_stamp(value: Any) -> float:
    """A Windows Search date (a pywintypes datetime through ADO) as a timestamp."""
    try:
        return float(value.timestamp()) if value is not None else 0.0
    except Exception:
        return 0.0


def search_windows_index(q: FileQuery) -> tuple[list[FileHit], str]:
    from .fast_find import _escape, index_available

    if not index_available():
        return [], "the Windows Search index is not reachable"
    import pythoncom
    import win32com.client

    where: list[str] = []
    if q.name and not any(c in q.name for c in "*?"):
        for word in q.name.split():
            where.append(f"System.FileName LIKE '%{_escape(word)}%'")
    if q.extensions:
        where.append("(" + " OR ".join(
            f"System.FileExtension = '.{_escape(e)}'" for e in q.extensions) + ")")
    if q.min_size is not None:
        where.append(f"System.Size >= {int(q.min_size)}")
    if q.max_size is not None:
        where.append(f"System.Size <= {int(q.max_size)}")
    for column, after, before in (("System.DateModified", q.modified_after, q.modified_before),
                                  ("System.DateCreated", q.created_after, q.created_before)):
        if after:
            where.append(f"{column} >= '{after.strftime('%Y-%m-%d %H:%M:%S')}'")
        if before:
            where.append(f"{column} < '{before.strftime('%Y-%m-%d %H:%M:%S')}'")
    # Files only. Unscoped, the index also answers with mail items and
    # attachments ("Assignment : LO...") that have no path on disk, so no
    # dates — and every date-filtered search came back empty.
    where.append(f"SCOPE='file:{_escape(q.folder)}'" if q.folder else "SCOPE='file:'")
    if q.kind == "folder":
        where.append("System.ItemType = 'Directory'")
    elif q.kind == "file":
        where.append("System.ItemType <> 'Directory'")
    sql = ("SELECT TOP {n} System.ItemPathDisplay, System.ItemType, System.Size, "
           "System.DateModified, System.DateCreated FROM SystemIndex"
           + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY System.DateModified DESC").format(n=max(q.limit * 4, 200))
    initialised = False
    try:
        pythoncom.CoInitialize()
        initialised = True
    except Exception:
        pass
    connection = recordset = None
    try:
        connection = win32com.client.Dispatch("ADODB.Connection")
        connection.Open("Provider=Search.CollatorDSO;Extended Properties=\"Application=Windows\";")
        recordset = win32com.client.Dispatch("ADODB.Recordset")
        recordset.Open(sql, connection)
        hits: list[FileHit] = []
        deadline = time.monotonic() + 12.0
        while not recordset.EOF and time.monotonic() < deadline:
            try:
                path = str(recordset.Fields.Item(0).Value or "")
                item_type = str(recordset.Fields.Item(1).Value or "")
                size = int(recordset.Fields.Item(2).Value or 0)
            except Exception:
                recordset.MoveNext()
                continue
            if path:
                is_folder = item_type.lower() == "directory"
                hit = _stat_hit(path, is_folder)
                if hit is None:
                    # Unreadable on disk (online-only, access denied): keep the
                    # index's own dates rather than none at all.
                    hit = FileHit(path, is_folder, size,
                                  _index_stamp(recordset.Fields.Item(3).Value),
                                  _index_stamp(recordset.Fields.Item(4).Value))
                hits.append(hit)
            recordset.MoveNext()
        return hits, ""
    except Exception as exc:
        return [], f"the Windows index query failed ({str(exc)[:100]})"
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


# ── engine 3: the walk ───────────────────────────────────────────────────────

def fixed_drives() -> list[str]:
    """Every fixed local drive's root (C:\\, D:\\ …)."""
    try:
        import psutil

        roots = [p.mountpoint for p in psutil.disk_partitions(all=False)
                 if "cdrom" not in p.opts.lower() and p.fstype]
        return roots or [os.path.abspath(os.sep)]
    except Exception:
        return [os.path.abspath(os.sep)]


def search_walk(q: FileQuery, budget: float = WALK_BUDGET_S) -> tuple[list[FileHit], str, bool]:
    roots = [q.folder] if q.folder else fixed_drives()
    deadline = time.monotonic() + budget
    found: list[FileHit] = []
    ceiling = max(q.limit * 6, 300)
    for root in roots:
        for base, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _e: None):
            if time.monotonic() > deadline:
                return found, "the full-drive walk hit its time limit", True
            keep = []
            for d in dirnames:
                low = d.lower()
                if low in NOISE_SKIP or low.startswith("$"):
                    continue
                if not q.include_system and low in SYSTEM_SKIP:
                    continue
                keep.append(d)
            dirnames[:] = keep
            if q.kind != "file":
                for d in dirnames:
                    if _name_matches(d, q.name):
                        hit = _stat_hit(os.path.join(base, d), True)
                        if hit and matches(hit, q):
                            found.append(hit)
            if q.kind != "folder":
                for name in filenames:
                    if q.name and not _name_matches(name, q.name):
                        continue
                    if q.extensions and name.rsplit(".", 1)[-1].lower() not in q.extensions:
                        continue
                    hit = _stat_hit(os.path.join(base, name), False)
                    if hit and matches(hit, q):
                        found.append(hit)
            if len(found) >= ceiling:
                return found, "", True
    return found, "", False


# ── the ladder ───────────────────────────────────────────────────────────────

def search(q: FileQuery, *, engines: Iterable[str] = ("everything", "index", "walk")) -> FileResults:
    """Answer *q* with the fastest engine that can, filtered and ordered here."""
    started = time.monotonic()
    result = FileResults(query=q)
    notes: list[str] = []
    for engine in engines:
        truncated = False
        if engine == "everything":
            raw, note = search_everything(q)
            label = "Everything (whole-PC index)"
        elif engine == "index":
            raw, note = search_windows_index(q)
            label = "the Windows Search index"
        else:
            raw, note, truncated = search_walk(q)
            label = "a walk of " + (q.folder or ", ".join(fixed_drives()))
        if note:
            notes.append(note)
        hits = [h for h in raw if matches(h, q)]
        if hits or engine == "walk":
            ordered = _order(_dedupe(hits), q)
            result.hits = ordered[:q.limit]
            result.truncated = truncated or len(ordered) > q.limit
            result.engine = label
            # An engine that could not run is worth a word only when nothing
            # better answered: "Everything is not installed" beside a good
            # answer from the index is noise.
            if engine != "everything" and notes:
                result.note = "; ".join(n for n in notes if n)[:200]
            break
    result.seconds = time.monotonic() - started
    return result


def _dedupe(hits: list[FileHit]) -> list[FileHit]:
    seen: set[str] = set()
    out: list[FileHit] = []
    for hit in hits:
        key = os.path.normcase(os.path.normpath(hit.path))
        if key not in seen:
            seen.add(key)
            out.append(hit)
    return out


# ── ORION's own documents ────────────────────────────────────────────────────

def orion_output_dirs() -> list[Path]:
    """Where ORION writes things he may be asked to bring up again."""
    from .constants import BASE_DIR, CONFIG_DIR

    candidates = [CONFIG_DIR / "self_repair", CONFIG_DIR / "research", CONFIG_DIR / "exports",
                  CONFIG_DIR / "reports", BASE_DIR / "research", BASE_DIR / "exports",
                  BASE_DIR / "reports", BASE_DIR / "projects",
                  Path.home() / "Documents" / "ORION"]
    return [p for p in candidates if p.exists()]


def find_own_documents(name: str, limit: int = 10) -> list[FileHit]:
    """Documents ORION himself wrote whose name matches — newest first.

    "Bring up that proposal" means the proposal he just saved, which lives in
    his own folders; a whole-PC search would rank some other file first.
    """
    q = FileQuery(name=name, kind="file", limit=limit, sort="newest")
    found: list[FileHit] = []
    for root in orion_output_dirs():
        hits, _note, _cut = search_walk(replace(q, folder=str(root)), budget=3.0)
        found.extend(hits)
    return _order(_dedupe(found), q)[:limit]


# ── acting on what was found ─────────────────────────────────────────────────

#: Opening one of these RUNS it. "Open the setup file" must never become
#: "execute whatever matched the word setup" — a document is safe to open for
#: the user, a program is a decision they make.
PROGRAM_EXTENSIONS = frozenset({
    ".exe", ".com", ".scr", ".pif", ".msi", ".msp", ".msix", ".appx",
    ".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf",
    ".wsh", ".hta", ".cpl", ".jar", ".reg", ".lnk", ".url", ".application",
    ".gadget", ".inf", ".sct", ".py", ".pyw",
})


def is_program(path: str) -> bool:
    """True when opening *path* would run code rather than show a document."""
    return os.path.splitext(str(path))[1].lower() in PROGRAM_EXTENSIONS


def open_path(path: str, *, allow_programs: bool = False) -> str:
    """Open *path* with its default app; Notepad when nothing is associated.

    Returns "" on success or a short reason. A Markdown proposal has no
    default application on a stock Windows install, and "the file exists but
    I cannot show it" is not an answer an assistant should give.

    A program or script is NOT run (see PROGRAM_EXTENSIONS): it is selected in
    File Explorer instead and the reason says so, unless *allow_programs*.
    """
    target = os.path.normpath(path)
    if not os.path.exists(target):
        return "that path does not exist"
    if os.name != "nt":
        return "opening files is only wired for Windows"
    if not allow_programs and os.path.isfile(target) and is_program(target):
        reveal_path(target)
        return ("it is a program or script, not a document, so I have not run "
                "it — it is selected in File Explorer for you to decide")
    try:
        os.startfile(target)  # type: ignore[attr-defined]
        return ""
    except OSError as exc:
        if os.path.isfile(target):
            try:
                subprocess.Popen(["notepad.exe", target],
                                 creationflags=0x08000000)
                return ""
            except OSError:
                pass
        return str(exc)[:160]


_FILLER_WORDS = frozenset({"my", "the", "a", "an", "our", "your", "his", "her",
                           "file", "document", "doc", "please", "up", "for", "me",
                           "latest", "last", "recent", "newest", "that", "this"})


#: What "open my X" may open: things a person makes and reads, not logs,
#: caches or program files that happen to share a word with the request.
OPENABLE_KINDS = ("document", "spreadsheet", "presentation", "image", "video", "audio")
_NOT_USER_CONTENT = re.compile(
    r"[\\/](appdata|\.[\w-]+|node_modules|site-packages|__pycache__|build|dist|"
    r"intermediates|windows|program files( \(x86\))?|programdata|\$recycle\.bin|temp|"
    r"browser_profile|extensions|user data|cache|icons?)[\\/]",
    re.IGNORECASE)


def _user_folders() -> list[str]:
    home = Path.home()
    roots = [home / "Desktop", home / "Documents", home / "Downloads",
             home / "Pictures", home / "Videos", home / "Music"]
    try:
        roots += [p for p in home.iterdir() if p.is_dir() and p.name.startswith("OneDrive")]
    except OSError:
        pass
    return [str(r) for r in roots if r.exists()]


def find_document(spoken: str, budget: float = 3.0) -> str:
    """The best document for a spoken name ("my CV", "the essay"), or "".

    Only user content qualifies (OPENABLE_KINDS, never a program, never under
    AppData/build/system folders), and every meaningful word must be a whole
    part of the file name — "cv" matches "CV 2026.docx", not "cv_debug.log".
    ORION's own documents first, then the fast index, then a short walk of
    the user's own folders; newest first. "" rather than a guess.
    """
    words = [w.strip(".") for w in re.findall(r"[\w'.-]+", str(spoken or "").lower())
             if w not in _FILLER_WORDS]
    words = [w for w in words if w]
    if not words:
        return ""
    name = " ".join(words)
    # Documents first; a picture or recording only when no document matches
    # ("my CV" is a document even though an icon somewhere is called cv.svg).
    for kinds in (("document", "spreadsheet", "presentation"), ("image", "video", "audio")):
        exts = tuple(dict.fromkeys(e for kind in kinds for e in KINDS.get(kind, ())))
        found = _find_by_kind(name, words, exts, deadline=time.monotonic() + budget / 2)
        if found:
            return found
    return ""


def _find_by_kind(name: str, words: list[str], extensions: tuple[str, ...],
                  deadline: float) -> str:
    def acceptable(hit: FileHit) -> bool:
        if hit.is_folder or is_program(hit.path) or _NOT_USER_CONTENT.search(hit.path):
            return False
        if Path(hit.path).suffix.lower().lstrip(".") not in extensions:
            return False
        parts = set(re.split(r"[^a-z0-9]+", Path(hit.path).stem.lower()))
        return all(any(part.startswith(w) for part in parts) for w in words)

    q = FileQuery(name=name, extensions=extensions, kind="file", limit=30, sort="newest")
    for hit in find_own_documents(name, limit=5):
        if acceptable(hit):
            return hit.path
    try:
        fast = search(q, engines=("everything", "index")).hits
    except Exception:
        fast = []
    for hit in _order(list(fast), q):
        if acceptable(hit):
            return hit.path
    found: list[FileHit] = []
    for root in _user_folders():
        left = deadline - time.monotonic()
        if left <= 0:
            break
        hits, _note, _cut = search_walk(replace(q, folder=root), budget=left)
        found.extend(h for h in hits if acceptable(h))
    ordered = _order(_dedupe(found), q)
    return ordered[0].path if ordered else ""


def reveal_path(path: str) -> str:
    """Show *path* selected in File Explorer."""
    target = os.path.normpath(path)
    if not os.path.exists(target):
        return "that path does not exist"
    try:
        subprocess.Popen(["explorer.exe", "/select,", target])
        return ""
    except OSError as exc:
        return str(exc)[:160]


def folder_usage(folder: str, top: int = 10, budget: float = 20.0) -> tuple[list[tuple[str, int]], bool]:
    """The largest immediate sub-folders of *folder* — what is eating the disk."""
    root = Path(os.path.expandvars(os.path.expanduser(folder or "C:\\")))
    deadline = time.monotonic() + budget
    totals: list[tuple[str, int]] = []
    cut = False
    try:
        children = [c for c in root.iterdir() if c.is_dir()]
    except OSError:
        return [], False
    for child in children:
        total = 0
        for base, dirnames, filenames in os.walk(child, onerror=lambda _e: None):
            if time.monotonic() > deadline:
                cut = True
                break
            for name in filenames:
                try:
                    total += os.stat(os.path.join(base, name)).st_size
                except OSError:
                    pass
        totals.append((str(child), total))
        if cut:
            break
    totals.sort(key=lambda row: -row[1])
    return totals[:top], cut


__all__ = [
    "FileHit", "FileQuery", "FileResults", "KINDS", "build_query",
    "everything_cli", "extensions_for", "find_own_documents", "fixed_drives",
    "folder_usage", "human_size", "matches", "open_path", "orion_output_dirs",
    "parse_size", "parse_size_bounds", "parse_when", "reveal_path", "search",
    "search_everything", "search_walk", "search_windows_index",
]
