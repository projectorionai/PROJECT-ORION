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
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import utc_stamp

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

UNDO_DIR = BASE_DIR / "config" / "file_organiser"


@dataclass
class Move:
    src: str
    dest: str

    def to_dict(self) -> dict[str, str]:
        return {"src": self.src, "dest": self.dest}


class FileOrganiser:
    KNOWN_TARGETS = ("Downloads", "Desktop", "Documents")

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        UNDO_DIR.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    async def organise(self, folder: str = "Downloads", apply: bool = False,
                       by_month: bool = False) -> ToolResult:
        target = await asyncio.to_thread(self._resolve_target, folder)
        if target is None:
            return ToolResult(
                f"I couldn't find a safe folder for '{folder}', sir. I organise the "
                f"user's {', '.join(self.KNOWN_TARGETS)} or an explicit path.", ok=False)
        moves = await asyncio.to_thread(self._plan_moves, target, by_month)
        if not moves:
            return ToolResult(f"'{target.name}' is already tidy, sir — nothing to organise.")
        if not apply:
            return self._dry_run_report(target, moves)
        applied = await asyncio.to_thread(self._apply_moves, target, moves)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("organiser.applied", float(len(applied)))
        return ToolResult(
            f"Organised {len(applied)} file(s) in '{target.name}', sir, into typed "
            f"folders. Say 'undo file organisation' to reverse it."
        )

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
        category_names = set(CATEGORY_MAP) | {"Other"}
        try:
            entries = list(target.iterdir())
        except OSError:
            return moves
        for entry in entries:
            if not entry.is_file() or entry.name.startswith("."):
                continue
            # Don't re-sort a file that already sits in one of our folders.
            if entry.parent.name in category_names:
                continue
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
        if not dest.exists():
            return dest
        stem, suffix, i = dest.stem, dest.suffix, 1
        while True:
            candidate = dest.with_name(f"{stem} ({i}){suffix}")
            if not candidate.exists():
                return candidate
            i += 1

    def _dry_run_report(self, target: Path, moves: list[Move]) -> ToolResult:
        by_cat: dict[str, int] = {}
        for m in moves:
            cat = Path(m.dest).parent.name
            by_cat[cat] = by_cat.get(cat, 0) + 1
        summary = ", ".join(f"{n} → {cat}" for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]))
        sample = "\n".join(f"  {Path(m.src).name}  →  {Path(m.dest).parent.name}/"
                           for m in moves[:12])
        return ToolResult(
            f"Dry run for '{target.name}', sir — {len(moves)} file(s) would move: {summary}.\n"
            f"{sample}\n"
            "Nothing has been moved. Say 'organise <folder> for real' (apply) to proceed."
        )

    # ── applying + undo ───────────────────────────────────────────────────────

    def _apply_moves(self, target: Path, moves: list[Move]) -> list[Move]:
        applied: list[Move] = []
        for m in moves:
            try:
                dest = Path(m.dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(m.src, str(dest))
                applied.append(m)
            except Exception as exc:
                self.bus.log.emit(f"ORGANISER: skipped {Path(m.src).name} - {exc}")
        if applied:
            undo_file = UNDO_DIR / f"undo_{datetime.now():%Y%m%d_%H%M%S}.json"
            undo_file.write_text(
                json.dumps({"target": str(target), "at": utc_stamp(),
                            "moves": [m.to_dict() for m in applied]}, indent=2),
                encoding="utf-8",
            )
            self.bus.log.emit(f"ORGANISER: moved {len(applied)} file(s); undo log written.")
        return applied

    def _undo_last(self) -> ToolResult:
        logs = sorted(UNDO_DIR.glob("undo_*.json"))
        if not logs:
            return ToolResult("There's no file-organisation to undo, sir.")
        latest = logs[-1]
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return ToolResult("The most recent undo log is unreadable, sir.", ok=False)
        restored = 0
        for m in reversed(data.get("moves", [])):
            try:
                src = Path(m["dest"])   # where it now is
                dest = Path(m["src"])   # where it came from
                if src.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(self._unique_dest(dest)))
                    restored += 1
            except Exception:
                continue
        latest.unlink(missing_ok=True)
        return ToolResult(f"Reversed {restored} move(s), sir — the files are back where they were.")
