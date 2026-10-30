"""
Contextual file management & autonomous organisation (improvement #18).

``FileOrganiser`` tidies cluttered folders (Downloads, Desktop, …) by sorting
files into typed sub-folders — Images, Documents, Audio, Video, Archives,
Installers, Code, Data, Other — optionally bucketed by month.

Safety first, because moving a user's files is not reversible-by-accident:
    • ``plan`` (the default) is a DRY RUN — it reports exactly what *would*
      move, and changes nothing.
    • ``apply`` performs the moves and writes an **undo log** so the exact set
      of moves can be reversed with ``undo``.
    • ORION's own workspace and protected paths are never touched, name
      collisions are resolved (never overwrites), and only regular files in the
      top level of the target folder are considered (no recursion).

Everything heavy (directory walks, moves) runs via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import shutil
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .bus import OrionBus
from .constants import BASE_DIR, CONFIG_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line, utc_stamp
from .atomic_io import atomic_write_text

CATEGORY_MAP: dict[str, set[str]] = {
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg", ".heic"},
    "Documents": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages", ".epub"},
    "Spreadsheets": {".xls", ".xlsx", ".csv", ".tsv", ".ods"},
    "Presentations": {".ppt", ".pptx", ".key", ".odp"},
    "Audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".aiff"},
    "Video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "Installers": {".exe", ".msi", ".dmg", ".pkg", ".deb", ".appimage"},
    "Code": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".rs", ".go", ".html", ".css", ".json", ".xml", ".yml", ".yaml", ".sh"},
    "Data": {".db", ".sqlite", ".sql", ".parquet", ".log"},
}

UNDO_DIR = CONFIG_DIR / "file_organiser"
LEGACY_UNDO_DIR = BASE_DIR / "config" / "file_organiser"
_MOVE_LOCK = threading.RLock()


@dataclass
class Move:
    src: str
    dest: str
    identity: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"src": self.src, "dest": self.dest}
        if self.identity is not None:
            result["identity"] = list(self.identity)
        return result


def _identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _matches_identity(path: Path, recorded: Any, *, require_unchanged: bool = True) -> bool:
    """Match an original file, allowing edits only after a committed move.

    An interrupted intent also needs unchanged contents metadata because no
    successful move was recorded. A completed move may be edited normally;
    its device and inode still distinguish it from a replacement file.
    """
    try:
        if (not isinstance(recorded, (list, tuple)) or len(recorded) != 4
                or recorded[1] == 0):
            return False
        info = path.stat(follow_symlinks=False)
        actual = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        return (stat.S_ISREG(info.st_mode)
                and (actual == tuple(recorded) if require_unchanged
                     else actual[:2] == tuple(recorded[:2])))
    except OSError:
        return False


def _append_journal_event(path: Path, event: dict[str, Any]) -> None:
    """Durable per-move receipts without rewriting all prior moves."""
    with path.with_suffix(".events").open("ab") as stream:
        stream.write((json.dumps(event) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _read_journal(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("moves", []), list):
        raise ValueError("invalid undo record")
    events = path.with_suffix(".events")
    if not data.get("complete", False) and events.exists():
        lines = events.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                if index == len(lines) - 1 and not line.endswith(b"\n"):
                    break  # An interrupted final append keeps the prior intent.
                raise
            if not isinstance(event, dict):
                raise ValueError("invalid journal event")
            if "intent" in event:
                data["pending"] = event["intent"]
            elif "commit" in event:
                committed = event["commit"]
                intent = committed
                if isinstance(committed, dict) and "source_identity" in committed:
                    # A copied move changes its inode. Prove which original
                    # intent this destination receipt completed; every path
                    # and all source metadata must still match exactly.
                    intent = dict(committed)
                    intent["identity"] = intent.pop("source_identity")
                if data.get("pending") != intent:
                    raise ValueError("commit does not match the pending move")
                data.setdefault("moves", []).append(committed)
                data["pending"] = None
            elif event.get("abort") is True:
                data["pending"] = None
            else:
                raise ValueError("unknown journal event")
    _validate_undo_paths(data)
    return data


def _append_restore_event(path: Path, event: dict[str, Any]) -> None:
    progress = path.with_suffix(".restore.events")
    if progress.is_symlink():
        raise OSError("The restore receipt must not be a symbolic link")
    with progress.open("ab") as stream:
        stream.write((json.dumps(event) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


class _RestoreReceiptError(OSError):
    """Further appends must stop until an incomplete receipt is recovered."""


def _restore_identity(value: Any) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(type(part) is int for part in value)
            and value[1] != 0 and value[2] >= 0)


def _read_restore_progress(path: Path, data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate all restore transitions before undo moves any file.

    Original receipt addresses are indexed once; replay and progress writes
    stay linear in the number of files rather than rewriting all prior moves.
    """
    records = list(data.get("moves", []))
    if data.get("pending") is not None:
        records.append(data["pending"])
    originals: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["src"], record["dest"])
        if key in originals:
            raise ValueError("duplicate original move in the restore journal")
        originals[key] = record
    root = Path(data["target"]).resolve()

    def validate_intent(intent: Any) -> tuple[str, str]:
        if not isinstance(intent, dict) or set(intent) != {"move", "path", "identity"}:
            raise ValueError("invalid restore intent")
        move = intent["move"]
        if not isinstance(move, dict) or not all(type(move.get(name)) is str for name in ("src", "dest")):
            raise ValueError("invalid original move in restore intent")
        key = (move["src"], move["dest"])
        original = originals.get(key)
        if (original is None or json.dumps(original, sort_keys=True) != json.dumps(move, sort_keys=True)):
            raise ValueError("restore intent does not match an original move")
        candidate = intent["path"]
        if not isinstance(candidate, str) or not candidate or not Path(candidate).is_absolute():
            raise ValueError("the restore path must be absolute")
        actual = Path(candidate).resolve()
        if (actual.parent != root or actual == Path(move["dest"]).resolve()
                or is_protected_path(actual)):
            raise ValueError("the restore path leaves the selected folder or touches protected code")
        identity = intent["identity"]
        if (not _restore_identity(identity)
                or (move.get("identity") is not None
                    and list(identity[:2]) != list(move["identity"][:2]))):
            raise ValueError("the restore intent has an invalid original file identity")
        return key

    progress: dict[tuple[str, str], dict[str, Any]] = {}
    events = path.with_suffix(".restore.events")
    if not events.exists():
        return progress
    if events.is_symlink():
        raise ValueError("the restore receipt must not be a symbolic link")
    raw = events.read_bytes()
    lines = raw.splitlines(keepends=True)
    valid_bytes = 0
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            break  # The final newline commits an append, even if JSON parses.
        event = json.loads(line)
        if not isinstance(event, dict) or len(event) != 1:
            raise ValueError("invalid restore event")
        if "restore_intent" in event:
            intent = event["restore_intent"]
            key = validate_intent(intent)
            previous = progress.get(key)
            if previous is not None and ("identity" in previous
                    or previous["intent"]["identity"] != intent["identity"]):
                raise ValueError("restore intent overwrites completed or changed progress")
            progress[key] = {"intent": intent}
        elif "restore_commit" in event:
            commit = event["restore_commit"]
            if not isinstance(commit, dict) or set(commit) != {"intent", "identity"}:
                raise ValueError("invalid restore commit")
            key = validate_intent(commit["intent"])
            previous = progress.get(key)
            if (previous is None or "identity" in previous or previous["intent"] != commit["intent"]
                    or not _restore_identity(commit["identity"])):
                raise ValueError("restore commit does not match its pending intent")
            progress[key] = commit
        else:
            raise ValueError("unknown restore event")
        valid_bytes += len(line)
    if valid_bytes < len(raw):
        # Appending after a torn last line would join two JSON records. Trim
        # only that uncommitted tail after validating every complete event.
        with events.open("r+b") as stream:
            if stream.read() != raw:
                raise OSError("The restore receipt changed while it was being read")
            stream.seek(valid_bytes)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
    return progress


def _validate_undo_paths(data: dict[str, Any]) -> None:
    """Check the entire receipt before undo can move even its first file."""
    selection = data.get("target")
    if not isinstance(selection, str) or not selection or not Path(selection).is_absolute():
        raise ValueError("the undo record has no absolute selected folder")
    root = Path(selection).resolve()
    if root == BASE_DIR.resolve() or is_protected_path(root):
        raise ValueError("the undo record selects protected ORION code")
    records = list(data.get("moves", []))
    if data.get("pending") is not None:
        records.append(data["pending"])
    for record in records:
        if (not isinstance(record, dict)
                or not all(isinstance(record.get(key), str) and record[key]
                           and Path(record[key]).is_absolute() for key in ("src", "dest"))):
            raise ValueError("the undo record contains invalid file paths")
        original, moved = Path(record["src"]).resolve(), Path(record["dest"]).resolve()
        if (original.parent != root or not moved.is_relative_to(root) or original == moved
                or is_protected_path(original) or is_protected_path(moved)):
            raise ValueError("an undo path leaves the selected folder or touches protected code")
        transition = "source_identity" in record
        for field in ("identity", "source_identity"):
            identity = record.get(field)
            if ((transition and identity is None)
                    or (identity is not None and (
                        not isinstance(identity, (list, tuple)) or len(identity) != 4
                        or not all(type(value) is int for value in identity)))):
                raise ValueError("the undo record contains an invalid file identity")


#: How many '(2)', '(3)'… suffixes to try before giving up and using
#: a random one. A thousand files of the same name is already absurd;
#: an infinite search is a hung move.
MAX_NAME_ATTEMPTS = 1000


class FileOrganiser:
    KNOWN_TARGETS = ("Downloads", "Desktop", "Documents")

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._metrics_warning = False
        UNDO_DIR.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    async def organise(self, folder: str = "Downloads", apply: bool = False,
                       by_month: bool = False) -> ToolResult:
        target = await asyncio.to_thread(self._resolve_target, folder)
        if target is None:
            return ToolResult(
                f"I couldn't find a safe folder for '{folder}'. I organise the "
                f"user's {', '.join(self.KNOWN_TARGETS)} or an explicit path.", ok=False)
        try:
            moves = await asyncio.to_thread(self._plan_moves, target, by_month)
        except OSError as exc:
            return ToolResult(f"Could not read '{target}': {exc}", ok=False)
        if not moves:
            return ToolResult(f"'{target.name}' is already tidy — nothing to organise.")
        if not apply:
            return self._dry_run_report(target, moves)
        try:
            applied = await asyncio.to_thread(self._apply_moves, target, moves)
        except OSError as exc:
            return ToolResult(f"File organisation stopped: {exc}", ok=False)
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("organiser.applied", float(len(applied)))
            except Exception as exc:
                if not self._metrics_warning:
                    self._metrics_warning = True
                    logging.getLogger(__name__).warning(
                        "Optional organiser metrics failed (%s)", type(exc).__name__)
        evidence = [{"kind": "file_move", **move.to_dict()} for move in applied]
        if len(applied) < len(moves):
            recovery = " Say 'undo file organisation' to reverse the completed moves." if applied else ""
            return ToolResult(
                f"Moved {len(applied)} of {len(moves)} file(s) in '{target.name}'; "
                f"{len(moves) - len(applied)} could not be moved.{recovery}",
                ok=False, evidence=evidence)
        return ToolResult(
            f"Organised {len(applied)} file(s) in '{target.name}' into typed "
            f"folders. Say 'undo file organisation' to reverse it.", evidence=evidence)

    async def undo(self) -> ToolResult:
        return await asyncio.to_thread(self._undo_last)

    async def preview(self, folder: str = "Downloads", by_month: bool = False) -> ToolResult:
        return await self.organise(folder, apply=False, by_month=by_month)

    # ── planning ──────────────────────────────────────────────────────────────

    def _resolve_target(self, folder: str) -> Optional[Path]:
        try:
            SecuritySanitiser.guard_text(folder, "organiser.folder")
        except SecurityViolation:
            return None
        name = folder.strip()
        candidate: Path
        if name.lower() in {t.lower() for t in self.KNOWN_TARGETS}:
            candidate = Path.home() / next(t for t in self.KNOWN_TARGETS if t.lower() == name.lower())
        else:
            candidate = Path(name).expanduser()
        try:
            candidate = candidate.resolve()
        except Exception:
            return None
        # Never organise ORION's own workspace or protected code.
        if candidate == BASE_DIR or BASE_DIR in candidate.parents or is_protected_path(candidate):
            return None
        return candidate if candidate.is_dir() else None

    def _category(self, suffix: str) -> str:
        s = suffix.lower()
        for category, suffixes in CATEGORY_MAP.items():
            if s in suffixes:
                return category
        return "Other"

    def _plan_moves(self, target: Path, by_month: bool) -> list[Move]:
        moves: list[Move] = []
        entries = list(target.iterdir())
        for entry in entries:
            if not entry.is_file() or entry.name.startswith("."):
                continue
            # Only this folder's top level is scanned. Its own name may be
            # Documents or Images; that does not mean its contents are sorted.
            category = self._category(entry.suffix)
            dest_dir = target / category
            if by_month:
                try:
                    month = datetime.fromtimestamp(entry.stat().st_mtime).strftime("%Y-%m")
                    dest_dir = dest_dir / month
                except OSError:
                    pass
            dest = self._unique_dest(dest_dir / entry.name)
            moves.append(Move(src=str(entry), dest=str(dest)))
        return moves

    def _unique_dest(self, dest: Path) -> Path:
        if not dest.exists() and not dest.is_symlink():
            return dest
        stem, suffix, i = dest.stem, dest.suffix, 1
        # Bounded. Unbounded, a directory that somehow answers "exists" to
        # every name — a permissions quirk, a network share misbehaving, a
        # filesystem returning stale entries — spins a core for ever inside a
        # file move, with no log line and no way to tell from a hang.
        for _ in range(MAX_NAME_ATTEMPTS):
            candidate = dest.with_name(f"{stem} ({i}){suffix}")
            if not candidate.exists() and not candidate.is_symlink():
                return candidate
            i += 1
        # Past that, a name nothing can collide with.
        import uuid

        return dest.with_name(f"{stem} ({uuid.uuid4().hex[:8]}){suffix}")

    def _dry_run_report(self, target: Path, moves: list[Move]) -> ToolResult:
        by_cat: dict[str, int] = {}
        for m in moves:
            cat = Path(m.dest).relative_to(target).parts[0]
            by_cat[cat] = by_cat.get(cat, 0) + 1
        summary = ", ".join(f"{n} → {cat}" for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]))
        sample = "\n".join(f"  {Path(m.src).name}  →  {Path(m.dest).parent.relative_to(target).as_posix()}/"
                           for m in moves[:12])
        return ToolResult(
            f"Dry run for '{target.name}' — {len(moves)} file(s) would move: {summary}.\n"
            f"{sample}\n"
            "Nothing has been moved. Say 'organise <folder> for real' (apply) to proceed."
        )

    # ── applying + undo ───────────────────────────────────────────────────────

    def _move_unique(self, source: Path, requested: Path,
                     before_move: Callable[[Path], None] | None = None) -> Path:
        """Choose a free name, then claim it without replacing another file.

        Windows rename refuses existing destinations. Elsewhere use a hard
        link, or an exclusive copy across filesystems; source removal follows
        successful creation. A collision between checking and moving retries.
        """
        if source.is_symlink() or not source.is_file():
            raise OSError("Only existing regular files can be organised or restored")
        for _ in range(MAX_NAME_ATTEMPTS):
            destination = self._unique_dest(requested)
            created = False
            try:
                if before_move is not None:
                    before_move(destination)
                if os.name == "nt":
                    try:
                        os.rename(source, destination)
                        return destination
                    except OSError as exc:
                        if exc.errno != errno.EXDEV:
                            raise
                else:
                    try:
                        os.link(source, destination, follow_symlinks=False)
                    except OSError as exc:
                        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS}:
                            raise
                    else:
                        created = True
                if not created:
                    with source.open("rb") as reader, destination.open("xb") as writer:
                        created = True
                        shutil.copyfileobj(reader, writer)
                        writer.flush()
                        os.fsync(writer.fileno())
                    shutil.copystat(source, destination)
                source.unlink()
                return destination
            except FileExistsError:
                if created:
                    destination.unlink(missing_ok=True)
                    raise
                continue
            except BaseException:
                if created:
                    destination.unlink(missing_ok=True)
                raise
        raise FileExistsError("No unoccupied destination could be claimed")

    def _apply_moves(self, target: Path, moves: list[Move]) -> list[Move]:
        with _MOVE_LOCK:
            return self._apply_moves_locked(target, moves)

    def _apply_moves_locked(self, target: Path, moves: list[Move]) -> list[Move]:
        applied: list[Move] = []
        if not moves:
            return applied
        root = target.resolve()
        undo_file = UNDO_DIR / f"undo_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}.json"
        journal: dict[str, Any] = {"target": str(root), "at": utc_stamp(), "moves": [],
                                   "pending": None, "complete": False}

        def save() -> None:
            atomic_write_text(undo_file, json.dumps(journal, indent=2), encoding="utf-8")

        # A writable recovery record is required before the first move.
        save()
        for m in moves:
            try:
                source = Path(m.src)
                dest = Path(m.dest)
                if source.parent.resolve() != root or not dest.resolve().is_relative_to(root):
                    raise OSError("The move would leave the selected folder")
                dest.parent.mkdir(parents=True, exist_ok=True)
                identity = _identity(source)

                def record_intent(candidate: Path) -> None:
                    if not candidate.resolve().is_relative_to(root):
                        raise OSError("The destination leaves the selected folder")
                    journal["pending"] = Move(m.src, str(candidate), identity).to_dict()
                    _append_journal_event(undo_file, {"intent": journal["pending"]})

                actual = self._move_unique(source, dest, before_move=record_intent)
            except Exception as exc:
                # No successful move was returned. If an implementation fails
                # after moving, retain the intent rather than hiding uncertainty.
                if not Path(m.src).exists() and journal["pending"] is not None:
                    raise OSError(f"An interrupted move is recorded in {undo_file}; "
                                  "use undo file organisation to recover it") from exc
                journal["pending"] = None
                _append_journal_event(undo_file, {"abort": True})
                self.bus.log.emit(f"ORGANISER: skipped {Path(m.src).name} - {exc}")
                continue
            # A cross-filesystem copy creates a different inode. The durable
            # intent keeps the pre-move identity; a committed receipt names
            # the actual destination file that a later undo may restore.
            applied.append(Move(src=m.src, dest=str(actual), identity=_identity(actual)))
            receipt = applied[-1].to_dict()
            if receipt["identity"] != list(identity):
                receipt["source_identity"] = list(identity)
            journal["moves"].append(receipt)
            journal["pending"] = None
            try:
                _append_journal_event(undo_file, {"commit": receipt})
            except OSError as exc:
                # The previous durable intent identifies this move even when
                # its completion receipt could not be written.
                raise OSError(f"Moved {actual}, but its receipt could not be saved. "
                              f"Recovery is recorded in {undo_file}; use undo file organisation") from exc
        if applied:
            journal["complete"] = True
            save()
            undo_file.with_suffix(".events").unlink(missing_ok=True)
            self.bus.log.emit(f"ORGANISER: moved {len(applied)} file(s); undo log written.")
        else:
            undo_file.with_suffix(".events").unlink(missing_ok=True)
            undo_file.unlink(missing_ok=True)
        return applied

    def _undo_last(self) -> ToolResult:
        with _MOVE_LOCK:
            try:
                return self._undo_last_locked()
            except OSError as exc:
                return ToolResult(f"The recovery record could not be updated: {exc}. "
                                  "It has been kept; you can retry undo safely.", ok=False)

    def _undo_last_locked(self) -> ToolResult:
        by_name: dict[str, Path] = {}
        # Existing undo records remain reachable after configuration relocation.
        # Explicit/test overrides do not inspect the user's legacy directory.
        if UNDO_DIR == CONFIG_DIR / "file_organiser" and LEGACY_UNDO_DIR != UNDO_DIR:
            by_name.update((path.name, path) for path in LEGACY_UNDO_DIR.glob("undo_*.json"))
        by_name.update((path.name, path) for path in UNDO_DIR.glob("undo_*.json"))
        logs = [by_name[name] for name in sorted(by_name)]
        if not logs:
            return ToolResult("There's no file-organisation to undo.")
        latest = logs[-1]
        try:
            data = _read_journal(latest)
            progress = _read_restore_progress(latest, data)
        except Exception as exc:
            return ToolResult(f"The most recent undo log is unreadable or unsafe: {first_line(exc)}. "
                              "It has been kept for review.", ok=False)
        intent = data.get("pending")
        if intent is not None:
            try:
                original, moved = Path(intent["src"]), Path(intent["dest"])
                if not original.exists() and _matches_identity(moved, intent.get("identity")):
                    data.setdefault("moves", []).append(intent)
                elif original.exists() and not moved.exists():
                    pass  # Interrupted before moving, or already put back.
                else:
                    return ToolResult(f"An interrupted move in {latest.name} needs review: "
                                      f"{original} → {moved}. Its identity could not be verified; "
                                      "the recovery record and files have been kept.", ok=False)
                data["pending"] = None
                data["complete"] = True
                atomic_write_text(latest, json.dumps(data, indent=2), encoding="utf-8")
            except (OSError, KeyError, TypeError, ValueError):
                return ToolResult("The interrupted move could not be verified or its recovery record saved.", ok=False)
        restored = 0
        pending = []

        def append_restore(event: dict[str, Any]) -> None:
            try:
                _append_restore_event(latest, event)
            except OSError as exc:
                raise _RestoreReceiptError(str(exc)) from exc

        for m in reversed(data.get("moves", [])):
            try:
                src = Path(m["dest"])   # where it now is
                dest = Path(m["src"])   # where it came from
                key = (m["src"], m["dest"])
                state = progress.get(key)
                if state is not None:
                    intent = state["intent"]
                    actual = Path(intent["path"])
                    if "identity" in state:
                        if (_matches_identity(src, intent["identity"], require_unchanged=False)
                                or not _matches_identity(actual, state["identity"], require_unchanged=False)):
                            raise OSError("The restored file is missing or replaced; its receipt is kept for review")
                        restored += 1
                        continue
                    if not src.exists() and _matches_identity(actual, intent["identity"]):
                        commit = {"intent": intent, "identity": list(_identity(actual))}
                        append_restore({"restore_commit": commit})
                        progress[key] = commit
                        restored += 1
                        continue
                    if (not _matches_identity(src, intent["identity"])
                            or actual.exists() or actual.is_symlink()):
                        raise OSError("The interrupted restore cannot be verified; its receipt is kept for review")
                    dest = actual
                elif not src.exists() and _matches_identity(
                        dest, m.get("identity"), require_unchanged=False):
                    restored += 1  # A prior undo completed before its receipt saved.
                    continue
                if m.get("identity") is not None and not _matches_identity(
                        src, m["identity"], require_unchanged=False):
                    raise OSError("The organised file was replaced; its undo record is kept for review")
                dest.parent.mkdir(parents=True, exist_ok=True)
                identity = list(_identity(src))
                if not _restore_identity(identity):
                    raise OSError("The organised file has no verifiable restore identity")

                def record_restore(candidate: Path) -> None:
                    actual = candidate.resolve()
                    if (not candidate.is_absolute()
                            or actual.parent != Path(data["target"]).resolve() or actual == src.resolve()
                            or is_protected_path(actual)):
                        raise OSError("The restore destination leaves the selected folder")
                    if not _matches_identity(src, identity):
                        raise OSError("The organised file changed before its restore intent could be recorded")
                    intent = {"move": m, "path": str(candidate), "identity": identity}
                    append_restore({"restore_intent": intent})
                    progress[key] = {"intent": intent}

                actual = self._move_unique(src, dest, before_move=record_restore)
                commit = {"intent": progress[key]["intent"], "identity": list(_identity(actual))}
                append_restore({"restore_commit": commit})
                progress[key] = commit
                restored += 1
            except _RestoreReceiptError as exc:
                return ToolResult(f"Restore progress could not be saved: {first_line(exc)}. "
                                  "The receipt has been kept; retry undo before moving more files.", ok=False)
            except Exception:
                pending.append(m)
        if pending:
            return ToolResult(f"Reversed {restored} move(s); {len(pending)} could not be restored. "
                              "Their undo record has been kept so you can retry.", ok=False)
        # Keep all progress if the base receipt is locked. Sidecars become
        # disposable only once no selectable undo journal remains.
        latest.unlink(missing_ok=True)
        for suffix in (".events", ".restore.events"):
            try:
                latest.with_suffix(suffix).unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Completed undo sidecar cleanup was unavailable")
        return ToolResult(f"Reversed {restored} move(s) — the files are back where they were.")
