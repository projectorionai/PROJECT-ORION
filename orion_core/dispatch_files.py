"""
Dispatch domain — Files, dev workbench, codebase, self-repair, diagnostics, cleanup, backup and forge.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import stat
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .undo import MAX_CAPTURE_BYTES as UNDO_MAX_BYTES
from .undo import push_undo
from .utils import first_line


def _undo_file_info(path: Path) -> os.stat_result:
    """An undo may only act on the same ordinary file, never through a link."""
    for candidate in reversed((path, *path.parents)):
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise OSError("The file path now contains a link or reparse point; it was kept")
        if candidate != path and not stat.S_ISDIR(info.st_mode):
            raise OSError("The file's parent changed; it was kept")
    if not stat.S_ISREG(info.st_mode):
        raise OSError("The target is no longer an ordinary file; it was kept")
    return info


def _undo_file_version(info: os.stat_result) -> tuple[int, ...]:
    # Windows path-stat and descriptor-stat expose different ctime semantics.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _undo_file_snapshot(path: Path) -> tuple[tuple[int, int], bytes]:
    before = _undo_file_info(path)
    if before.st_size > UNDO_MAX_BYTES:
        raise OSError("The file exceeds the undo byte limit")
    with path.open("rb") as handle:
        if _undo_file_version(os.fstat(handle.fileno())) != _undo_file_version(before):
            raise OSError("The file changed while opening it")
        content = handle.read(UNDO_MAX_BYTES + 1)
        if _undo_file_version(os.fstat(handle.fileno())) != _undo_file_version(before):
            raise OSError("The file changed while reading it")
    if (len(content) != before.st_size or len(content) > UNDO_MAX_BYTES
            or _undo_file_version(_undo_file_info(path)) != _undo_file_version(before)):
        raise OSError("The file changed while recording its undo state")
    return (before.st_dev, before.st_ino), content


def _undo_written_file(path: Path, before: bytes | None,
                       expected: tuple[tuple[int, int], int, bytes]) -> str:
    identity, size, digest = expected
    info = _undo_file_info(path)
    with path.open("r+b") as handle:
        opened = os.fstat(handle.fileno())
        if ((opened.st_dev, opened.st_ino) != identity or opened.st_size != size
                or _undo_file_version(opened) != _undo_file_version(info)):
            raise OSError("The file was replaced or changed after ORION wrote it; it was kept")
        content = handle.read(UNDO_MAX_BYTES + 1)
        if (len(content) != size or hashlib.sha256(content).digest() != digest
                or _undo_file_version(os.fstat(handle.fileno())) != _undo_file_version(opened)
                or _undo_file_version(_undo_file_info(path)) != _undo_file_version(opened)):
            raise OSError("The file changed after ORION wrote it; its newer contents were kept")
        if before is not None:
            handle.seek(0)
            handle.write(before)
            handle.truncate()
            return "restored the previous contents"
    # The pathname is checked again after closing the handle. This is an
    # observed ownership guard, not an atomic cross-process filesystem CAS.
    if _undo_file_version(_undo_file_info(path)) != _undo_file_version(opened):
        raise OSError("The file changed before removal; it was kept")
    path.unlink()
    return "removed the file it created"


def _scan_one_file(path: str) -> ToolResult:
    """Defender's verdict on one file the owner named (threat_guard.py)."""
    from .threat_guard import scan_file, signature_state
    target = Path(os.path.expandvars(os.path.expanduser(path.strip().strip('"')))) if path else None
    if target is None or not target.is_absolute() or not target.is_file():
        return ToolResult("Give me the full path of a file to scan.", ok=False)
    verdict = scan_file(target)
    signed = {"signed": "It is properly signed.", "unsigned": "It is not signed.",
              "invalid": "Its signature FAILS - it may have been tampered with."}.get(
                  signature_state(target), "")
    if verdict.state == "threat":
        return ToolResult(f"Defender identifies {target.name} as {verdict.threat}. Don't open "
                          f"it; let Windows Security remove it. {signed}".strip())
    if verdict.state == "clean":
        return ToolResult(f"Defender scanned {target.name} and found nothing. {signed}".strip())
    return ToolResult("Windows Defender could not scan it here - another antivirus may own "
                      "protection.", ok=False)


#: A live turn waits 45 s for a tool; the scan stops short of that and says
#: how far it got rather than being abandoned mid-folder (Mark XXXX.4).
DOWNLOADS_SCAN_DEADLINE_S = 35.0


def _scan_downloads(limit: int = 40, deadline_s: float = DOWNLOADS_SCAN_DEADLINE_S,
                    scan: Any = None) -> ToolResult:
    """Defender's verdict on the programs, scripts and archives in Downloads,
    newest first, for as long as a live turn can wait."""
    from .threat_guard import RISKY_EXTENSIONS, defender_cli, scan_file
    if defender_cli() is None:
        return ToolResult("Windows Defender isn't reachable here, so I can't scan Downloads.",
                          ok=False)
    folder = Path.home() / "Downloads"
    try:
        files = sorted((p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in RISKY_EXTENSIONS),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    except OSError:
        return ToolResult("I couldn't read your Downloads folder.", ok=False)
    if not files:
        return ToolResult("There are no programs, scripts or archives in Downloads to scan.")
    scan = scan or scan_file
    threats, unknown, scanned = [], 0, 0
    stop_at = time.monotonic() + deadline_s
    for item in files:
        if scanned and time.monotonic() >= stop_at:
            break
        verdict = scan(item)
        scanned += 1
        if verdict.state == "threat":
            threats.append(f"{item.name} ({verdict.threat})")
        elif verdict.state != "clean":
            unknown += 1
    head = f"Scanned the {scanned} newest programs, scripts and archives in Downloads"
    if scanned < len(files):
        head += f" ({len(files) - scanned} older ones not reached in time - ask again to continue)"
    if threats:
        return ToolResult(f"{head}. Defender flags {len(threats)}: {'; '.join(threats)}. Don't "
                          "open them; let Windows Security remove them.")
    tail = f" ({unknown} could not be read)" if unknown else ""
    return ToolResult(f"{head}: all clean{tail}.")


class FilesDispatchMixin:
    """Files, dev workbench, codebase, self-repair, diagnostics, cleanup, backup and forge."""

    def _file_references(self) -> Any:
        from .session_context import FileReferences
        refs = getattr(self, "_recent_file_references", None)
        if refs is None:
            refs = self._recent_file_references = FileReferences()
        return refs

    @staticmethod
    def _unclear_file_reference() -> ToolResult:
        return ToolResult("That file reference is ambiguous or has expired. Choose a numbered "
                          "match from the latest search, or give its name or full path.", ok=False)

    def _file_matches_result(self, text: str, paths: list[str], *,
                             selected: str | None = None, ok: bool = True) -> ToolResult:
        self._file_references().remember(paths, selected=selected)
        return ToolResult(text, ok=ok, receipt={"file_paths": paths, "selected_path": selected})

    @staticmethod
    def _filter_file_paths(paths: list[str], args: dict[str, Any]) -> list[str]:
        """Apply explicit constraints to content hits and spoken references
        too; neither is permission to bypass a folder, type, size or date."""
        from . import file_search
        filters = file_search.build_query(dict(args, query="", name=""))
        if not any((filters.folder, filters.kind, filters.extensions, filters.min_size is not None,
                    filters.max_size is not None, filters.modified_after, filters.modified_before,
                    filters.created_after, filters.created_before)):
            return paths
        return [path for path in paths if (hit := file_search._stat_hit(path)) is not None
                and file_search.matches(hit, filters)]

    async def find_files(self, args: dict[str, Any]) -> ToolResult:
        """Find anything on the PC by name, type, size and date — and act on it.

        Whole-PC search (Everything, then the Windows index, then a walk of
        every fixed drive) through file_search, with every filter re-applied
        on the real file. Actions: open (the best match — or the exact path),
        reveal (select it in File Explorer), usage (which folders are biggest).

        "Bring up that proposal" is an OPEN of a document ORION wrote himself,
        so his own output folders are searched first when opening: a
        whole-PC ranking would put somebody else's proposal ahead of the one
        he saved a minute ago.
        """
        from . import file_search

        action = str(args.get("action") or "").strip().lower()
        path_arg = str(args.get("path") or "").strip()
        open_first = bool(args.get("open") or args.get("open_first")) or action == "open"
        reveal_first = bool(args.get("reveal")) or action in {"reveal", "show", "show_in_folder"}
        if open_first or reveal_first:
            from .execution_policy import require_foreground
            blocked = require_foreground("shell_open_file" if open_first else "shell_reveal_file")
            if blocked is not None:
                return blocked
        if path_arg and (open_first or reveal_first):
            is_ref, referenced = self._file_references().resolve(path_arg)
            if is_ref:
                if referenced is None:
                    return self._unclear_file_reference()
                if not await asyncio.to_thread(self._filter_file_paths, [referenced], args):
                    return ToolResult("That numbered path does not match the supplied file filters "
                                      "or is no longer available.", ok=False)
                path_arg = referenced
        if action in {"usage", "space", "disk_usage", "biggest_folders", "sizes"}:
            folder = str(args.get("folder") or path_arg or "C:\\")
            rows, cut = await asyncio.to_thread(file_search.folder_usage, folder)
            if not rows:
                return ToolResult(f"Could not read the folders inside {folder}.", ok=False)
            lines = [f"{file_search.human_size(size):>10}  {name}" for name, size in rows]
            return ToolResult(f"Largest folders in {folder}"
                              + (" (stopped at the time limit — partial)" if cut else "")
                              + ":\n" + "\n".join(lines))
        if path_arg and (open_first or reveal_first):
            # On a worker, never on the loop: os.startfile is ShellExecute,
            # which pumps the message queue while it waits, and a pump inside
            # this task re-entered the others ("Cannot enter into task",
            # 8 Oct 23:32:09, the moment the research folder was opened).
            problem = await asyncio.to_thread(
                file_search.open_path if open_first else file_search.reveal_path,
                path_arg)
            if problem:
                return ToolResult(f"Could not {action} {path_arg}: {problem}", ok=False)
            self._file_references().select(path_arg)
            return ToolResult(f"{'Opened' if open_first else 'Showing'} {path_arg}.",
                              receipt={"file_paths": [path_arg], "selected_path": path_arg})

        raw_query = SecuritySanitiser.guard_text(
            str(args.get("query") or args.get("name") or ""), "find_files.query").strip()
        args = dict(args, query=raw_query)
        is_ref, referenced = (False, None) if args.get("contents") or args.get("inside") else (
            self._file_references().resolve(raw_query))
        if is_ref:
            if referenced is None:
                return self._unclear_file_reference()
            if not await asyncio.to_thread(self._filter_file_paths, [referenced], args):
                return ToolResult("That numbered path does not match the supplied file filters "
                                  "or is no longer available.", ok=False)
            if open_first or reveal_first:
                return await self.find_files({"path": referenced,
                                              "action": "open" if open_first else "reveal"})
            self._file_references().select(referenced)
            return ToolResult(f"Selected path: {referenced}",
                              receipt={"file_paths": [referenced], "selected_path": referenced})

        if args.get("contents") or args.get("inside"):
            # Searching INSIDE files is the Windows index's full-text job.
            from . import fast_find
            search = await asyncio.to_thread(
                fast_find.find, raw_query.lower(), contents=True,
                folder=str(args.get("folder") or args.get("in") or ""),
                thorough=bool(args.get("thorough")))
            unique = {os.path.normcase(os.path.normpath(hit.path)): hit.path for hit in search.hits}
            matches = await asyncio.to_thread(self._filter_file_paths, list(unique.values()), args)
            if not matches:
                self._file_references().remember([])
                return ToolResult(f"Nothing mentions '{raw_query}' ({search.strategy}).")
            if (open_first or reveal_first) and len(matches) > 1:
                shown = matches[:8]
                return self._file_matches_result(
                    "I found several matching paths. Which one should I "
                    + ("open" if open_first else "show") + "?\n"
                    + "\n".join(f"{i}. {path}" for i, path in enumerate(shown, 1)), shown)
            if open_first or reveal_first:
                problem = await asyncio.to_thread(
                    file_search.open_path if open_first else file_search.reveal_path, matches[0])
                if problem:
                    return self._file_matches_result(
                        f"Found {matches[0]} but could not "
                        f"{'open' if open_first else 'show'} it: {problem}", [matches[0]], ok=False)
                return self._file_matches_result(
                    f"{'Opened' if open_first else 'Showing'} {matches[0]}.",
                    [matches[0]], selected=matches[0])
            shown = matches[:25]
            return self._file_matches_result(f"{search.describe()}\n" + "\n".join(
                f"{i}. {path}" for i, path in enumerate(shown, 1)), shown)

        query = file_search.build_query(args)
        if query.is_empty():
            return ToolResult("Tell me what to look for — a name, a type (pdf, "
                              "video…), a size or a date.", ok=False)
        own: list[Any] = []
        if ((open_first or reveal_first) and query.name and query.kind != "folder"
                and not query.direct_path):
            own = await asyncio.to_thread(file_search.find_own_documents, query.name)
            own = [hit for hit in own if file_search.matches(hit, query)]
        results = await asyncio.to_thread(file_search.search, query)
        hits = file_search._dedupe(own + results.hits)
        if not hits and query.name and not query.direct_path:
            # Nothing has every word in its name: try the closest names
            # ("my biology essay" -> "Bio Essay Final.docx").
            closest = await asyncio.to_thread(file_search.relaxed_search, query)
            if closest is not None and closest.hits:
                return await asyncio.to_thread(       # may open a file (see above)
                    self._closest_files, closest, open_first, reveal_first)
        if not hits:
            self._file_references().remember([])
            return ToolResult(results.summary())
        if (open_first or reveal_first) and len(hits) > 1:
            shown = hits[:8]
            listing = "\n".join(f"{i}. {hit.line()}" for i, hit in enumerate(shown, 1))
            return self._file_matches_result("I found several matching paths. Which one should I "
                                             + ("open" if open_first else "show") + "?\n" + listing,
                                             [hit.path for hit in shown])
        if open_first or reveal_first:
            best = hits[0].path
            problem = await asyncio.to_thread(
                file_search.open_path if open_first else file_search.reveal_path, best)
            if problem:
                return self._file_matches_result(f"Found {best} but could not "
                                                 f"{'open' if open_first else 'show'} it: {problem}",
                                                 [best], ok=False)
            others = "\n".join(hit.line() for hit in hits[1:8])
            return self._file_matches_result(
                f"{'Opened' if open_first else 'Showing in File Explorer'}: {best}"
                + (f"\n{results.note}" if getattr(results, "note", "") else "")
                + (f"\nOther matches:\n{others}" if others else ""), [best], selected=best)
        from .session_context import FILE_REFERENCE_LIMIT
        shown = hits[:min(query.limit, FILE_REFERENCE_LIMIT)]
        note = f"\nShowing the first {len(shown)} matches." if len(hits) > len(shown) else ""
        return self._file_matches_result(results.summary() + note + "\n" + "\n".join(
            f"{i}. {hit.line()}" for i, hit in enumerate(shown, 1)), [hit.path for hit in shown])

    def _closest_files(self, closest: Any, open_first: bool, reveal_first: bool) -> ToolResult:
        """Report the closest names — and act on one only when it is the
        single name answering every word; otherwise say which they are."""
        from . import file_search
        terms = file_search._content_terms(closest.query.name)
        full = [hit for hit in closest.hits
                if file_search.relaxed_score(hit.name, terms) >= 1.0]
        shown = closest.hits[:8]
        listing = "\n".join(f"{i}. {hit.line()}" for i, hit in enumerate(shown, 1))
        if (open_first or reveal_first) and len(full) == 1:
            best = full[0].path
            problem = (file_search.open_path(best) if open_first
                       else file_search.reveal_path(best))
            if not problem:
                return self._file_matches_result(
                    f"No file has exactly '{closest.query.name}' in its name; the one "
                    f"that matches every word is {best} - "
                    f"{'opened' if open_first else 'shown in File Explorer'}.", [best], selected=best)
        ask = (" Which one did you mean?" if (open_first or reveal_first) else "")
        return self._file_matches_result(f"{closest.summary()}{ask}\n{listing}",
                                         [hit.path for hit in shown])

    def _scan_user_files(self, query: str) -> list[str]:
        """Time-boxed name search across the user's common folders (runs off-thread)."""
        home = Path.home()
        roots: list[Path] = []
        try:
            for entry in home.iterdir():
                if entry.is_dir() and (
                    entry.name in {"Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"}
                    or entry.name.startswith("OneDrive")
                ):
                    roots.append(entry)
        except Exception:
            pass
        roots.append(BASE_DIR)
        ignored = {"appdata", "node_modules", "__pycache__", ".git", ".venv", "venv"}
        matches: list[str] = []
        seen: set[str] = set()
        deadline = time.monotonic() + 8.0
        for root in roots:
            if time.monotonic() > deadline or len(matches) >= 40:
                break
            try:
                for path in root.rglob("*"):
                    if time.monotonic() > deadline or len(matches) >= 40:
                        break
                    if any(part.lower() in ignored or part.startswith(".") for part in path.parts):
                        continue
                    if query in path.name.lower():
                        resolved = str(path)
                        if resolved not in seen:
                            seen.add(resolved)
                            matches.append(resolved)
            except Exception:
                continue
        # Folders first, then the tightest name matches.
        matches.sort(key=lambda m: (0 if Path(m).is_dir() else 1, len(Path(m).name)))
        return matches

    def undo_tool(self, args: dict[str, Any]) -> ToolResult:
        """Reverse ORION's own last reversible action.

        Deliberately a tool rather than a confirmation prompt. Asking "are you
        sure?" before every write costs a round trip each time and trains the
        user to stop talking to the assistant; acting at once and offering a way
        back costs nothing until it is needed. Confirmation stays reserved for
        what genuinely cannot be reversed — a decision about reversibility, not
        about how alarming the word sounds.
        """
        from . import undo as undo_stack

        action = str(args.get("action") or "last").lower().strip()
        if action in {"list", "history", "show"}:
            entries = undo_stack.history()
            if not entries:
                return ToolResult("There is nothing I can undo at the moment.")
            lines = "\n".join(f"{i}. {label}" for i, label in enumerate(entries, 1))
            return ToolResult(f"I can undo these, most recent first:\n{lines}")
        ok, message = undo_stack.undo_last()
        return ToolResult(message, ok=ok)

    @staticmethod
    def _file_identity(path: Path) -> tuple[Any, ...]:
        stat = path.stat()
        identity = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if not path.is_dir():
            return identity
        import hashlib
        digest = hashlib.sha256()
        deadline = time.monotonic() + 2.0
        def failed(error):
            raise error
        for base, folders, files in os.walk(path, followlinks=False, onerror=failed):
            folders.sort()
            for name in sorted(folders + files):
                entry = Path(base) / name
                if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)():
                    raise OSError("The folder contains a linked path; its delete preview cannot be verified")
                item = entry.stat()
                digest.update(repr((str(entry.relative_to(path)), item.st_dev, item.st_ino,
                                    item.st_size, item.st_mtime_ns)).encode('utf-8'))
                if time.monotonic() > deadline:
                    raise OSError("The folder is too large to verify within the approval budget")
        return (*identity, digest.hexdigest())

    def _file_controller_open(self, spoken: str) -> ToolResult:
        """Open a file on screen. There was no such action: 30 Oct 2026, asked
        to "bring up the MD file", ORION read it with read_text and said "I've
        opened" it - nothing appeared."""
        from .execution_policy import require_foreground
        blocked = require_foreground("application_open")
        if blocked is not None:
            return blocked
        is_ref, resolved = self._file_references().resolve(spoken)
        if is_ref and not resolved:
            return self._unclear_file_reference()
        chosen = str(resolved if is_ref else spoken).strip().strip("\"'`")
        if not chosen:
            return ToolResult("Which file should I open? Give me its path.", ok=False)
        target = Path(chosen).expanduser()
        if not target.is_absolute():
            target = self._resolve_user_path(str(target))
        if not target.exists():
            return ToolResult(f"{target} does not exist, so nothing was opened.", ok=False)
        from . import file_search
        problem = file_search.open_path(str(target))
        if problem:
            return ToolResult(f"Could not open {target.name}: {problem}", ok=False)
        self._file_references().remember([str(target)])
        return ToolResult(f"Opened {target.name} on screen.",
                          receipt={"file_paths": [str(target)], "selected_path": str(target),
                                   "opened": True})

    #: Showing a file on screen, in the program Windows opens it with.
    _OPEN_ACTIONS = frozenset({"open", "show", "view", "bring_up", "display", "launch",
                               "open_file", "start"})

    def file_controller(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip().replace(" ", "_")
        if action in FilesDispatchMixin._OPEN_ACTIONS:
            return self._file_controller_open(str(args.get("path") or args.get("file") or ""))
        if action not in {"write_text", "append_text", "delete", "remove", "mkdir", "create_dir", "build_structure"}:
            return self._file_controller_execute(args)
        is_ref, _ = self._file_references().resolve(str(args.get("path") or args.get("directory") or ""))
        if is_ref:
            return ToolResult("Use the full target path when changing or deleting a file; "
                              "a spoken reference does not authorise a change.", ok=False)
        from .path_policy import Zone, measure
        from .system_guard import ActionIntent
        path = self._resolve_user_path(str(args.get("path") or args.get("directory") or BASE_DIR))
        verdict = self._ensure_write_safe(path)
        if verdict is not None:
            path = verdict.path
        chosen = dict(args, path=str(path))
        # Every delete is the owner's on-screen decision, wherever the file is
        # (owner, 30 Oct 2026: "delete files ... with my strict permission").
        # Files in ORION's own output areas used to go without asking.
        needs_approval = path.exists() and (
            action in {"delete", "remove"}
            or (getattr(verdict, "zone", None) is Zone.USER and action in {"write_text", "append_text"}))
        if needs_approval:
            from .execution_policy import require_foreground
            blocked = require_foreground("file_confirmation_dialog")
            if blocked is not None:
                return blocked
            guard = getattr(self, "system_guard", None)
            confirm = getattr(getattr(self, "bus", None), "confirm_action", None)
            if (not callable(getattr(guard, "request_confirmation", None))
                    or not callable(getattr(confirm, "emit", None))):
                return ToolResult(f"Kept {path}: the required on-screen approval is unavailable. "
                                  "Nothing was changed.", ok=False)
            deleting = action in {"delete", "remove"}
            intent = ActionIntent.FILE_DELETE if deleting else ActionIntent.FILE_WRITE
            payload = {"args": chosen, "identity": self._file_identity(path)}
            preview = measure(path, budget_s=2.0).describe() if deleting else "existing contents will change"
            decision = self.system_guard.request_confirmation(intent, origin="file_controller", payload=payload)
            self.bus.confirm_action.emit({"token": decision.token, "intent": intent.value,
                                          "message": (f"Move {path} to the Recycle Bin ({preview})?"
                                                      if deleting else f"Change {path}? {preview}.")})
            return ToolResult(f"Awaiting your on-screen approval for {path}. Nothing has changed.", ok=False)
        return self._file_controller_execute(chosen)

    def _run_confirmed_file_action(self, intent: Any, payload: Any) -> ToolResult:
        from .system_guard import ActionIntent
        if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
            return ToolResult("The file approval has no bound target.", ok=False)
        args = dict(payload["args"])
        action = str(args.get("action") or "").lower().strip()
        wanted = ActionIntent.FILE_DELETE if action in {"delete", "remove"} else ActionIntent.FILE_WRITE
        if intent is not wanted or action not in {"delete", "remove", "write_text", "append_text"}:
            return ToolResult("The approval does not match this file action.", ok=False)
        path = self._resolve_user_path(str(args.get("path") or ""))
        try:
            self._ensure_write_safe(path)
            if self._file_identity(path) != tuple(payload.get("identity") or ()):
                return ToolResult("The file or folder changed after the approval was requested. Please request it again.", ok=False)
            return self._file_controller_execute(args)
        except (OSError, SecurityViolation) as exc:
            return ToolResult(f"Nothing changed: {exc}", ok=False)

    def _file_controller_execute(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        raw_path = str(args.get("path") or args.get("directory") or BASE_DIR)
        if action in {"read", "read_text", "list", "dir", "inventory"}:
            is_ref, referenced = self._file_references().resolve(raw_path)
            if is_ref:
                if referenced is None:
                    return self._unclear_file_reference()
                raw_path = referenced
        path = self._resolve_user_path(raw_path)
        if action in {"search_codebase", "search", "grep"}:
            query = SecuritySanitiser.guard_text(
                str(args.get("query") or args.get("text") or ""), "file_controller.query"
            )
            if not query:
                return ToolResult("No codebase search query supplied.", ok=False)
            if not path.exists() or not path.is_dir():
                return ToolResult(f"Search root not found: {path}", ok=False)
            matches = self._search_codebase(path, query)
            if not matches:
                return ToolResult(f"No codebase matches for: {query}.")
            text  = "\n".join(matches[:80])
            chain: list[tuple[str, dict[str, Any]]] = []
            if args.get("diagnose") or args.get("inspect"):
                # The drive colon in C:\\... is part of the path. Only the
                # :line-number: separator belongs to the search formatter.
                first_path = re.split(r":\d+:\s?", matches[0], maxsplit=1)[0]
                chain.append(("file_controller", {"action": "read_text", "path": first_path}))
                chain.append((
                    "save_memory",
                    {"category": "projects", "key": "last_codebase_search",
                     "value": f"{query}: {matches[0][:240]}"}
                ))
            return ToolResult(text, chain=chain)
        if action in {"list", "dir", "inventory"}:
            if not path.exists():
                return ToolResult(f"Path not found: {path}", ok=False)
            if path.is_file():
                return ToolResult(f"File: {path} ({path.stat().st_size} bytes)")
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return ToolResult("\n".join(entries[:200]) or "Directory is empty.")
        if action in {"mkdir", "create_dir", "structure_build", "build_structure"}:
            self._ensure_write_safe(path)
            existed = path.exists()
            path.mkdir(parents=True, exist_ok=True)
            if not existed:
                # The reverse of "create a folder" is removing it ONLY while it
                # is still empty. If something has been put inside since, the
                # folder stays — undoing must never destroy the user's files to
                # tidy up after ORION.
                def _remove_if_empty(target=path) -> str:
                    if not target.exists():
                        return "it was already gone"
                    if any(target.iterdir()):
                        return "kept it — it is no longer empty"
                    target.rmdir()
                    return ""

                push_undo(f"created folder {path.name}", _remove_if_empty)
            return ToolResult(f"Directory structure ready: {path}")
        if action in {"write_text", "append_text"}:
            self._ensure_write_safe(path)
            text = SecuritySanitiser.guard_text(
                str(args.get("text") or args.get("content") or ""), "file_controller.text"
            )
            # Read the previous contents BEFORE writing. Undo must restore what
            # was actually there, never a reconstruction: an undo that restores
            # a guess is worse than none, because the user believes the original
            # is back. Oversized files are excluded and say so rather than ORION
            # quietly holding megabytes alive for the session.
            # Bytes, not text: a decoded-with-replacement copy of a cp1252 or
            # binary file is not "what was actually there".
            previous: bytes | None = None
            too_large = False
            if path.exists() and path.is_file():
                try:
                    _identity, previous = _undo_file_snapshot(path)
                except OSError:
                    too_large = True
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if action == "append_text" else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(text)

            if not too_large:
                try:
                    identity, written = _undo_file_snapshot(path)
                    own_bytes = text.replace("\n", os.linesep).encode("utf-8")
                    if mode == "a":
                        own_bytes = (previous or b"") + own_bytes
                    if written != own_bytes:
                        raise OSError("The file changed before its undo state could be recorded")
                    expected = (identity, len(written), hashlib.sha256(written).digest())
                except OSError:
                    too_large = True
            if not too_large:
                def _restore(target=path, before=previous, observed=expected) -> str:
                    return _undo_written_file(target, before, observed)
                push_undo(f"wrote {path.name}", _restore)
            note = "" if not too_large else (
                f"  (undo state unavailable or over {UNDO_MAX_BYTES // 1024} KB — this write cannot be undone)")
            return ToolResult(
                f"File {'appended' if mode == 'a' else 'written'}: {path}{note}")
        if action in {"read", "read_text"}:
            if not path.exists() or not path.is_file():
                return ToolResult(f"File not found: {path}", ok=False)
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(6000)
            self._file_references().select(str(path))
            return ToolResult(text, receipt={"file_paths": [str(path)], "selected_path": str(path)})
        if action in {"delete", "remove"}:
            self._ensure_write_safe(path)
            if not path.exists():
                return ToolResult(f"Path already absent: {path}")
            if path.is_dir():
                from . import recycle_bin
                why = recycle_bin.refusal(path)
                if why is not None:
                    return ToolResult(f"Kept {path}: it cannot go to the Recycle Bin ({why}). Nothing was deleted.", ok=False)
                if not recycle_bin.send_to_recycle_bin(path):
                    return ToolResult(f"Could not verify that {path} reached the Recycle Bin.", ok=False)
                def restore_directory(target=path):
                    if target.exists():
                        raise FileExistsError("The original folder path is occupied; nothing was overwritten")
                    if not recycle_bin.restore_from_recycle_bin(target):
                        raise FileNotFoundError("The folder could not be restored from the Recycle Bin")
                    return "restored the folder from the Recycle Bin"
                push_undo(f"deleted folder {path.name}", restore_directory)
                return ToolResult(f"Moved to the Recycle Bin: {path} (say \"undo\" to restore it)")

            # A file goes to the Recycle Bin and nowhere else. The old path
            # deleted outright and held a TEXT copy for undo (a photo or .docx
            # "restored" from that was corrupt); later, a file the bin could not
            # take still fell through to path.unlink() - permanent, on model
            # text alone, and it defeated recycle_bin.refusal(), whose whole
            # job is to stop the shell destroying what it cannot recycle. Now
            # such a file stays where it is and the reason is reported; deleting
            # it for good is the user's call (or the human-confirmed cleanup's).
            # The exact bytes of a small file are still held as a second way
            # back, for when the bin has since been emptied.
            from . import recycle_bin
            why = recycle_bin.refusal(path)
            if why is not None:
                return ToolResult(
                    f"Kept {path}: it can't go to the Recycle Bin ({why}), and I only delete "
                    "what can be put back. Nothing was deleted.", ok=False)
            captured: bytes | None = None
            try:
                if path.stat().st_size <= UNDO_MAX_BYTES:
                    captured = path.read_bytes()
            except OSError:
                captured = None

            def _restore(target=path, before=captured) -> str:
                if target.exists() or target.is_symlink():
                    if before is not None and target.is_file() and not target.is_symlink():
                        with target.open("rb") as existing:
                            if existing.read(len(before) + 1) == before:
                                return "the original contents were already restored"
                    raise FileExistsError("A file already occupies the original path; "
                                          "it was kept and the deleted file was not restored over it")
                if recycle_bin.restore_from_recycle_bin(target):
                    return "restored the file from the Recycle Bin"
                if before is None:
                    raise FileNotFoundError("The file is no longer in the Recycle Bin "
                                            "and no captured byte copy is available")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(before)
                return "restored the file"

            if recycle_bin.send_to_recycle_bin(path):
                push_undo(f"deleted {path.name}", _restore)
                return ToolResult(f"Moved to the Recycle Bin: {path}  (say \"undo\" to restore it)")
            if path.exists():
                return ToolResult(f"Kept {path}: moving it to the Recycle Bin failed. "
                                  "Nothing was deleted.", ok=False)
            # Gone from its place but not found in the bin: the outcome is
            # unverified, and is reported as exactly that.
            if captured is not None:
                push_undo(f"deleted {path.name}", _restore)
            return ToolResult(
                f"{path} is no longer there, but I could not confirm it reached the Recycle Bin "
                "- it may be gone for good."
                + ("  A copy is held: say \"undo\" to put it back." if captured is not None else ""),
                ok=False)
        return ToolResult(f"Unsupported file action: {action}", ok=False)

    async def process_file(self, args: dict[str, Any]) -> ToolResult:
        from . import file_review
        action = str(args.get("action") or "review").lower().strip()
        if action in {"recall", "history", "memory"}:
            return ToolResult(await asyncio.to_thread(file_review.recall, self.memory,
                             str(args.get("query") or args.get("prompt") or args.get("path") or "")))
        if action not in {"review", "scan", "analyse", "analyze", "read", ""}:
            return ToolResult(f"Unsupported file review action: {action}", ok=False)
        raw_path = str(args.get("path") or args.get("file_path") or "")
        if not raw_path:
            return ToolResult("No file path supplied.", ok=False)
        is_ref, referenced = self._file_references().resolve(raw_path)
        if is_ref:
            if referenced is None:
                return self._unclear_file_reference()
            raw_path = referenced
        prompt = str(args.get("prompt") or args.get("question") or args.get("instruction") or "")
        path   = Path(os.path.expandvars(os.path.expanduser(raw_path)))
        if not path.is_absolute():
            path = BASE_DIR / path
        # 30 Oct 2026: the dispatcher has no .router - the router is on .ai
        # (AIModeInfo), where dispatch_vision reads it. Asking self.router gave
        # file_review None, so no dropped or named picture was ever shown to a
        # vision model ("no vision model could look") and no file interpreted.
        router = getattr(getattr(self, "ai", None), "router", None) or getattr(self, "router", None)
        # inspect_async offloads all synchronous file I/O via asyncio.to_thread()
        result = await file_review.review(
            path, prompt=prompt, memory=getattr(self, "memory", None),
            router=router, file_intel=getattr(self, "file_intel", None),
            vision=getattr(self, "vision", None),
            ocr=getattr(getattr(self, "vision", None), "ocr_engine", None),
            detector=getattr(getattr(self, "perception", None), "objects", None),
            transcriber=getattr(getattr(self, "ai", None), "offline_stt", None))
        if result.ok:
            self._file_references().select(str(path))
            result.receipt = dict(result.receipt or {}, file_paths=[str(path)], selected_path=str(path))
            self.bus.log.emit(f"FILE: scanned {path.name}")
        return result

    DEV_COMMANDS = {
        "python", "py", "pytest", "pip", "node", "npm", "npx", "tsc",
        "cargo", "go", "dotnet", "git", "rustc",
    }

    CODE_SUFFIX_LANGS = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
        ".jsx": "JavaScript", ".cs": "C#", ".rs": "Rust", ".go": "Go",
        ".html": "HTML", ".css": "CSS", ".json": "JSON", ".md": "Markdown",
        ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML", ".sql": "SQL",
    }

    async def dev_workbench(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").lower().strip()
        if action in {"analyse", "analyse_repo", "analyze", "analyze_repo"}:
            root = self._resolve_user_path(str(args.get("path") or BASE_DIR))
            return await asyncio.to_thread(self._analyse_repo, root)
        if action in {"read", "read_file", "read_code"}:
            path  = self._resolve_user_path(str(args.get("path") or ""))
            start = max(1, int(args.get("start_line") or 1))
            count = max(1, min(400, int(args.get("line_count") or 200)))
            return await asyncio.to_thread(self._read_code, path, start, count)
        if action in {"run", "run_command", "test", "run_tests"}:
            return await self._run_dev_command(args)
        if action in {"create_python_project", "new_project", "scaffold"}:
            return self._create_python_project(args)
        return ToolResult(
            f"Unsupported workbench action: {action}. "
            "Use analyse_repo, read_file, run_command, or create_python_project.",
            ok=False,
        )

    def _analyse_repo(self, root: Path) -> ToolResult:
        if not root.is_dir():
            return ToolResult(f"Repository root not found: {root}", ok=False)
        ignored = {
            ".git", "__pycache__", ".venv", "venv", "node_modules", "target",
            "bin", "obj", ".mypy_cache", ".pytest_cache", "dist", "build",
        }
        key_names = {
            "readme.md", "pyproject.toml", "package.json", "cargo.toml",
            "go.mod", "requirements.txt", "setup.py", "tsconfig.json",
        }
        languages: dict[str, int] = {}
        line_totals: dict[str, int] = {}
        key_files: list[str] = []
        todo_count = 0
        scanned = 0
        for path in root.rglob("*"):
            if scanned >= 4000:
                break
            if any(part in ignored for part in path.parts):
                continue
            if not path.is_file():
                continue
            scanned += 1
            if path.name.lower() in key_names or path.suffix.lower() == ".csproj":
                key_files.append(str(path.relative_to(root)))
            language = self.CODE_SUFFIX_LANGS.get(path.suffix.lower())
            if language is None:
                continue
            languages[language] = languages.get(language, 0) + 1
            try:
                if path.stat().st_size < 600_000:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    line_totals[language] = line_totals.get(language, 0) + text.count("\n") + 1
                    todo_count += len(re.findall(r"(?i)\b(?:todo|fixme|hack)\b", text))
            except Exception:
                continue
        if not languages:
            return ToolResult(f"No recognised source files under {root}.")
        summary = ", ".join(
            f"{lang}: {count} file(s), ~{line_totals.get(lang, 0)} lines"
            for lang, count in sorted(languages.items(), key=lambda kv: kv[1], reverse=True)
        )
        return ToolResult(
            f"Repository analysis: {root}\n"
            f"Languages: {summary}.\n"
            f"Key files: {', '.join(key_files[:12]) or 'none detected'}.\n"
            f"Open TODO/FIXME markers: {todo_count}.  Files scanned: {scanned}."
        )

    def _read_code(self, path: Path, start: int, count: int) -> ToolResult:
        if not path.is_file():
            return ToolResult(f"File not found: {path}", ok=False)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(len(lines), start - 1 + count)
        if start > len(lines):
            return ToolResult(f"{path} has only {len(lines)} lines.", ok=False)
        numbered = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
        return ToolResult(f"{path} lines {start}-{end} of {len(lines)}:\n{numbered[:8000]}")

    async def _run_dev_command(self, args: dict[str, Any]) -> ToolResult:
        from .execution_policy import require_foreground
        blocked = require_foreground("unrestricted_development_process")
        if blocked is not None:
            return blocked
        command = SecuritySanitiser.guard_text(
            str(args.get("command") or ""), "dev.command"
        ).strip()
        if not command:
            return ToolResult("No development command supplied.", ok=False)
        parts = command.split()
        binary = Path(parts[0]).name.lower().removesuffix(".exe")
        if binary not in self.DEV_COMMANDS:
            return ToolResult(
                f"Command '{parts[0]}' is not on the development allowlist "
                f"({', '.join(sorted(self.DEV_COMMANDS))}).",
                ok=False,
            )
        cwd = self._resolve_user_path(str(args.get("path") or BASE_DIR))
        if cwd.is_file():
            cwd = cwd.parent
        if not cwd.is_dir():
            return ToolResult(f"Working directory not found: {cwd}", ok=False)

        def _execute() -> str:
            blocked = require_foreground("unrestricted_development_process")
            if blocked is not None:
                raise RuntimeError(blocked.text)
            completed = subprocess.run(
                parts, cwd=str(cwd), capture_output=True, text=True, errors="replace",
                timeout=120, shell=False,
            )
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            output = f"exit code {completed.returncode}\n{stdout[-5000:]}"
            if stderr:
                output += f"\nSTDERR:\n{stderr[-2000:]}"
            return output

        try:
            output = await asyncio.to_thread(_execute)
        except subprocess.TimeoutExpired:
            return ToolResult("Development command timed out after 120 seconds.", ok=False)
        except FileNotFoundError:
            return ToolResult(f"Command not found on this host: {parts[0]}", ok=False)
        return ToolResult(f"$ {command}  (cwd: {cwd})\n{output}")

    def _create_python_project(self, args: dict[str, Any]) -> ToolResult:
        raw_name = str(args.get("name") or args.get("project") or "new_project")
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip().lower()).strip("_") or "new_project"
        root = self._resolve_user_path(str(args.get("path") or (BASE_DIR / "projects" / name)))
        verdict = self._ensure_write_safe(root)
        root = Path(getattr(verdict, "path", root)).resolve()
        from .path_policy import Zone
        user_project = getattr(verdict, "zone", Zone.WORKSPACE) is Zone.USER
        if user_project and (root.exists() or root.is_symlink()):
            return ToolResult(
                f"Kept {root}: a project scaffold would overwrite files in an existing user folder. "
                "Choose a new project folder; nothing was changed.", ok=False)
        package = root / "src" / name
        tests   = root / "tests"
        planned = (package, tests, package / "__init__.py", package / "main.py",
                   tests / f"test_{name}.py", root / "README.md", root / "pyproject.toml",
                   root / ".gitignore")
        # A pre-existing src/tests junction may lead out of the requested
        # project. Check every destination before the first directory or file.
        for destination in planned:
            self._ensure_write_safe(destination)
            if not destination.resolve().is_relative_to(root):
                raise SecurityViolation(
                    f"Kept {root}: {destination} resolves outside the requested project. Nothing was changed.")
        if user_project:
            try:
                root.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                return ToolResult(f"Kept {root}: that user folder appeared while preparing the project. "
                                  "Choose a new folder; nothing was changed.", ok=False)
        package.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(f'"""{name} package."""\n', encoding="utf-8")
        (package / "main.py").write_text(
            "def main() -> None:\n"
            f'    print("Hello from {name}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n",
            encoding="utf-8",
        )
        (tests / f"test_{name}.py").write_text(
            f"from src.{name}.main import main\n\n\n"
            "def test_main_runs():\n"
            "    main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            f"# {raw_name.strip() or name}\n\nScaffolded by O.R.I.O.N.\n", encoding="utf-8"
        )
        (root / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{name.replace("_", "-")}"\n'
            'version = "0.1.0"\n'
            'requires-python = ">=3.10"\n',
            encoding="utf-8",
        )
        (root / ".gitignore").write_text("__pycache__/\n.venv/\n*.pyc\n", encoding="utf-8")
        return ToolResult(
            f"Python project scaffolded at {root} (src/{name}, tests, pyproject.toml, README)."
        )

    async def docker_tool(self, args: dict[str, Any]) -> ToolResult:
        """Container lifecycle over the real `docker` CLI (docker_control.py)
        — list/start/stop/restart/remove containers, list images, tail logs.
        Never raises: degrades to an actionable message when Docker isn't
        installed or the daemon isn't running."""
        from . import docker_control as dc
        action = str(args.get("action") or "list").lower().strip()
        target = str(args.get("container") or args.get("name") or args.get("id") or "")
        if action in {"list", "ps", "containers"}:
            containers = await asyncio.to_thread(dc.list_containers, bool(args.get("all", True)))
            if not containers:
                return ToolResult(dc.describe())
            lines = [
                f"{c.get('Names', '?')}  [{c.get('Image', '?')}]  {c.get('Status', '?')}"
                for c in containers
            ]
            return ToolResult("Containers:\n" + "\n".join(lines))
        if action in {"images"}:
            images = await asyncio.to_thread(dc.list_images)
            if not images:
                return ToolResult(dc.describe())
            lines = [
                f"{i.get('Repository', '?')}:{i.get('Tag', '?')}  {i.get('Size', '?')}"
                for i in images
            ]
            return ToolResult("Images:\n" + "\n".join(lines))
        if action in {"start", "stop", "restart", "remove", "rm"}:
            fn = {"start": dc.start_container, "stop": dc.stop_container,
                  "restart": dc.restart_container, "remove": dc.remove_container,
                  "rm": dc.remove_container}[action]
            return await asyncio.to_thread(fn, target)
        if action in {"logs", "log"}:
            tail = int(args.get("tail") or 100)
            return await asyncio.to_thread(dc.container_logs, target, tail)
        if action in {"status", "describe"}:
            return ToolResult(await asyncio.to_thread(dc.describe))
        return ToolResult(
            f"Unsupported docker action: {action}. "
            "Use list, images, start, stop, restart, remove, logs, or status.",
            ok=False,
        )

    async def debugger_tool(self, args: dict[str, Any]) -> ToolResult:
        """Real pdb-backed debug session (debugger.py) — start a script under
        the debugger, send commands (next/step/continue/p <expr>/break
        <file>:<line>), stop it. One active session at a time."""
        service = getattr(self, "debugger", None)
        if service is None:
            return ToolResult("Debugger service is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        if action in {"start", "run", "launch"}:
            path = str(args.get("path") or args.get("script") or "")
            if not path:
                return ToolResult("No script path supplied.", ok=False)
            extra = args.get("args") or []
            script_args = [str(a) for a in extra] if isinstance(extra, list) else []
            return await asyncio.to_thread(service.start, path, script_args)
        if action in {"command", "send", "next", "step", "continue", "print", "break"}:
            command = str(args.get("command") or args.get("text") or action)
            if action == "print" and args.get("expression"):
                command = f"p {args['expression']}"
            if action == "break" and args.get("location"):
                command = f"break {args['location']}"
            return await asyncio.to_thread(service.command, command)
        if action in {"stop", "quit", "end"}:
            return await asyncio.to_thread(service.stop)
        if action in {"status"}:
            return ToolResult("Debug session running." if service.is_running()
                               else "No debug session running.")
        return ToolResult(
            f"Unsupported debugger action: {action}. "
            "Use start, command, stop, or status.",
            ok=False,
        )

    def _search_codebase(self, root: Path, query: str) -> list[str]:
        query_lower  = query.lower()
        allowed      = {".py", ".txt", ".md", ".json", ".html", ".css", ".js", ".ts",
                        ".yml", ".yaml", ".toml"}
        ignored_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                        ".mypy_cache", ".pytest_cache"}
        matches: list[str] = []
        for file_path in root.rglob("*"):
            if len(matches) >= 160:
                break
            if any(part in ignored_dirs for part in file_path.parts):
                continue
            if not file_path.is_file() or file_path.suffix.lower() not in allowed:
                continue
            if is_protected_path(file_path):
                continue
            try:
                if file_path.stat().st_size > 1_200_000:
                    continue
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if query_lower in line.lower():
                            matches.append(f"{file_path}:{line_number}: {line.strip()[:220]}")
                            break
            except Exception:
                continue
        return matches

    async def codebase_copilot(self, args: dict[str, Any]) -> ToolResult:
        if self.copilot is None:
            return ToolResult("Developer Copilot is not available.", ok=False)
        action = str(args.get("action") or "analyse").lower().strip()
        path = str(args.get("path") or "")
        if action in {"analyse", "analyze", "index", "overview"}:
            return await self.copilot.analyse_repository(path)
        if action in {"find_symbol", "symbol"}:
            return await self.copilot.find_symbol(str(args.get("name") or args.get("query") or ""), path)
        if action in {"dependencies", "deps", "impact"}:
            return await self.copilot.dependency_report(str(args.get("module") or ""), path)
        if action in {"task", "refactor", "review", "bughunt", "tests", "docs"}:
            return await self.copilot.engineering_task(
                str(args.get("task") or action), path, str(args.get("focus") or "")
            )
        return ToolResult(
            f"Unsupported copilot action: {action}. Use analyse, find_symbol, "
            "dependencies, or task.",
            ok=False,
        )

    async def self_repair(self, args: dict[str, Any]) -> ToolResult:
        if self.selfrepair is None:
            return ToolResult("Self-repair agent is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "incidents", "list"}:
            incidents = self.selfrepair.incidents()
            if not incidents:
                return ToolResult("No incidents captured; the runtime is healthy.")
            return ToolResult("Captured incidents:\n" + "\n".join(i.summary() for i in incidents[-10:]))
        if action in {"report", "request", "report_defect", "reported"}:
            # Proactive repair: the user describes behaviour that is wrong but
            # never raises, so there is no captured incident to work from.
            description = str(args.get("description") or args.get("defect")
                              or args.get("message") or "").strip()
            if not description:
                return ToolResult(
                    "Tell me what is behaving incorrectly — e.g. "
                    "self_repair(action='report', description='you mishear "
                    "\"shut down\" as a power command').", ok=False)
            incident = self.selfrepair.request_repair(
                description, file=str(args.get("path") or ""))
            if not incident.file:
                return ToolResult(
                    f"Recorded as {incident.id}, but I could not work out which "
                    "module is responsible from that description. Give me a file "
                    "path, or more specific wording (a log line, a tool name, or "
                    "the exact phrase I got wrong).", ok=False)
            drafted = await self.selfrepair.propose_fix(incident.id)
            return ToolResult(
                f"Recorded as {incident.id}; I believe this lives in "
                f"{Path(incident.file).name}.\n\n{drafted.text}", ok=drafted.ok)
        if action in {"propose", "fix", "propose_fix"}:
            return await self.selfrepair.propose_fix(str(args.get("incident_id") or ""))
        if action in {"test", "run_tests", "verify"}:
            return await self.selfrepair.run_tests(str(args.get("path") or ""))
        if action in {"repair", "repair_file", "apply"}:
            # Approval-gated code fix: draft with confirm=false, apply with confirm=true.
            return await self.selfrepair.repair_file(
                incident_id=str(args.get("incident_id") or ""),
                confirm=bool(args.get("confirm")),
            )
        if action in {"revert", "undo"}:
            return self.selfrepair.revert_last()
        if action in {"diagnose", "diagnostics", "diagnostic", "full_check"} and self.diagnostics is not None:
            return await self.diagnostics.run_full()
        return ToolResult(
            f"Unsupported self_repair action: {action}. Use status, report "
            "(describe wrong behaviour that never raises), propose, run_tests, "
            "repair (confirm=true to apply), revert, or diagnose.",
            ok=False,
        )

    async def _self_benchmark(self, args: dict[str, Any]) -> ToolResult:
        """Mark XXXX.2: ORION benchmarks himself (self_benchmark.py) on his own
        text brain and his own tool resolver, and compares with his last run."""
        from . import reward_learning, self_benchmark
        from .execution_policy import attended_scope
        # Words + meaning + his learned intent network once it is built (the
        # plain resolver until then): the routing he actually does.
        from .intent_brain import routing_scores
        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        if router is None:
            return ToolResult("My text brain isn't reachable just now, so I can't benchmark it.",
                              ok=False)
        used: list[str] = []

        async def ask(prompt: str) -> str:
            profile, text = await router.generate_text(
                prompt, task="self_benchmark", budget="background", deadline_s=120.0)
            used.append(str(getattr(profile, "name", "") or ""))
            return str(text or "")

        menu = [d for d in getattr(self, "TOOL_DECLARATIONS", ()) or ()
                if d.get("name") not in {"find_tool", "use_tool"}]

        def route(request: str) -> list[str]:
            scores = routing_scores(request, menu)
            return [n for n in sorted(scores, key=lambda n: (-scores[n], n)) if scores[n] > 0][:3]

        wanted = [c.strip().lower() for c in re.split(r"[,\s]+", str(args.get("category") or ""))
                  if c.strip()] or list(self_benchmark.CATEGORIES)
        previous = (await asyncio.to_thread(self_benchmark.history, None, 1) or [None])[-1]
        # Minutes of work: it keeps the owner's turn (attended) once Live has
        # answered and left it to finish, so the coding checks can still run.
        with attended_scope("self-benchmark"):
            run = await self_benchmark.run_benchmark(
                ask, categories=wanted, route=route if menu else None,
                say=lambda m: self.bus.log.emit(f"BENCH: {m}"))
        run.provider = max(set(used), key=used.count) if used else ""
        # As an AI that learns: rewards, mistakes, reflexes, real-turn routing.
        run.learning.update(await asyncio.to_thread(reward_learning.learning_stats))
        await asyncio.to_thread(self_benchmark.save, run)
        return ToolResult(self_benchmark.report(run, previous))

    async def diagnostics_tool(self, args: dict[str, Any]) -> ToolResult:
        # "Why did you stop talking at 23:41?" - answered from ORION's own
        # records (session.log*, the Live journal, the shutdown report, crash
        # reports) with the exact lines quoted; see self_diagnosis.py. It needs
        # no diagnostics engine, so it is checked first. A bare 'why' stays
        # the capability matrix; 'why' WITH a time or question is an incident.
        requested = str(args.get("action") or "").lower().strip().replace(" ", "_").replace("-", "_")
        if requested in {"incident", "incidents", "what_happened", "whathappened", "happened",
                         "log", "logs", "events", "timeline", "session_log"} or (
                requested == "why" and (args.get("when") or args.get("question"))):
            from . import self_diagnosis
            question = str(args.get("question") or "").strip()
            when = str(args.get("when") or args.get("time") or "").strip()
            try:
                # File reading, off the event loop (bounded tail reads).
                text = await asyncio.to_thread(self_diagnosis.explain, question, when)
            except Exception as exc:
                return ToolResult(f"I could not read my own logs: {first_line(exc)}", ok=False)
            return ToolResult(text)
        if requested in {"benchmark", "self_benchmark", "test_yourself", "benchmark_yourself"}:
            return await self._self_benchmark(args)
        if requested in {"benchmark_history", "benchmarks", "scores"}:
            from . import self_benchmark
            runs = await asyncio.to_thread(self_benchmark.history)
            if not runs:
                return ToolResult("I haven't benchmarked myself yet - ask me to.")
            lines = [f"{time.strftime('%d %b %H:%M', time.localtime(r['at']))}: "
                     f"{r['overall']:.0f}% ({r.get('provider') or 'unknown'})" for r in runs[-8:]]
            return ToolResult("My last benchmark runs: " + "; ".join(lines) + ".")
        if requested in {"config", "config_check", "settings", "json", "validate_json",
                         "check_json"}:
            # Mark XXXIX: the json_validator plugin, rebuilt (json_check.py) -
            # needs no engine and no extra package.
            from . import json_check
            data = str(args.get("json") or args.get("json_data") or "")
            try:
                if data.strip():
                    text, ok = json_check.check_text(data, str(args.get("schema") or ""))
                else:
                    text, ok = await asyncio.to_thread(json_check.check_config)
            except Exception as exc:
                return ToolResult(f"I could not check that: {first_line(exc)}", ok=False)
            return ToolResult(text, ok=ok)
        if self.diagnostics is None:
            return ToolResult("The diagnostics engine is not available.", ok=False)
        action = str(args.get("action") or "full").lower().strip()
        if action in {"capabilities", "capability", "health", "matrix", "why", "can_do"}:
            deep = bool(args.get("deep", True))
            return await self.diagnostics.capability_report(deep=deep)
        if action in {"reflexes", "shortcuts", "learned"}:
            # What ORION has taught himself to answer instantly, what is queued
            # to be, and the two deliberate acts that change it. Promotion is
            # never automatic: a learned reflex bypasses the model, so a person
            # says yes to it (reward_learning only decides when to OFFER one).
            # Only an already-eligible candidate can be promoted, so this
            # cannot invent a shortcut out of nothing.
            try:
                from .reflex_learning import ReflexLearner
                learner = ReflexLearner()
            except Exception as exc:
                return ToolResult(f"Reflex learning unavailable: {exc}", ok=False)
            promote = str(args.get("promote") or "").strip()
            if promote:
                if learner.promote(promote):
                    return ToolResult(
                        f"Done - {promote!r} is now answered instantly, with no "
                        "model turn. Say 'forget that shortcut' to undo it.")
                return ToolResult(
                    f"{promote!r} is not an eligible candidate yet - it needs to "
                    "reach the same read-only tool a few more times first.",
                    ok=False)
            forget = str(args.get("forget") or "").strip()
            if forget:
                if learner.demote(forget) or learner.forget(forget):
                    return ToolResult(f"Forgotten - {forget!r} goes back to the model.")
                return ToolResult(f"No learned shortcut for {forget!r}.", ok=False)
            try:
                from . import reward_learning
                rewards = await asyncio.to_thread(reward_learning.summary)
            except Exception:
                rewards = ""
            return ToolResult("\n\n".join(p for p in (learner.report(), rewards) if p), ok=True)
        if action in {"routing", "resolver", "tool_routing", "shadow"}:
            # What the tool pre-filter WOULD do, measured on real turns while it
            # is switched off. Recall is the number that decides whether it can
            # safely be enabled; everything else is the benefit.
            try:
                from .resolver_shadow import ShadowEvaluator
                return ToolResult(ShadowEvaluator().report(), ok=True)
            except Exception as exc:
                return ToolResult(f"Shadow evaluation unavailable: {exc}", ok=False)
        if action in {"gaps", "unmet", "cant", "missing", "limitations"}:
            # What the user asked for that ORION could not do — the same
            # ledger the self-improvement review forges from.
            from .capability_gaps import LEDGER
            return ToolResult(LEDGER.report())
        if action in {"crashes", "crash", "crashed", "crash_reports"}:
            # "Why did you crash?" — the black box read back, including a
            # native crash that ended an earlier session (faulthandler.log).
            from . import crash_reporter
            try:
                limit = max(1, min(10, int(args.get("limit") or 5)))
            except (TypeError, ValueError):
                limit = 5
            rows = crash_reporter.recent(limit)
            if not rows:
                return ToolResult("No crash reports on record - nothing has crashed "
                                  "since the reports were last pruned.")
            lines = [f"The {len(rows)} most recent crash report(s), newest first:"]
            for row in rows:
                lines.append(f"- {row.get('time') or row['file'][:15]}: "
                             f"{row.get('why') or row.get('kind')} - "
                             f"{(row.get('summary') or '')[:240]} ({row['file']})")
            lines.append("Full reports are in config/crash_reports.")
            return ToolResult("\n".join(lines))
        if action in {"latency", "lag", "performance", "responsiveness", "slow"}:
            # Answers "why were you laggy?" with measurements rather than a
            # guess: every event-loop stall the watchdog caught, with the
            # function that caused it, plus p50/p95 for each timed operation.
            from .latency import report as latency_report
            return ToolResult(latency_report(), ok=True)
        return await self.diagnostics.run_full()

    def cleanup_review_tool(self, args: dict[str, Any]) -> ToolResult:
        """Analyse the project for unnecessary files (GitHub prep) and preview a
        deletion — but NEVER delete here. 'scan' lists candidates and .gitignore
        suggestions; 'plan' validates explicitly-selected files and raises an
        on-screen confirmation the user must approve (deletion runs only through
        that human-gated confirmation, never on model text)."""
        from .cleanup_assistant import CleanupError
        action = str(args.get("action") or "scan").lower().strip()
        assistant = self._cleanup()
        if action in {"scan", "review", "analyse", "analyze"}:
            report = assistant.scan()
            summary = report.as_dict()["summary"]
            lines = [
                f"Cleanup review of the project: {summary['total']} candidate(s), "
                f"{summary['deletable']} safe to remove, {summary['secret_bearing']} "
                "secret-bearing (excluded from Git, never deleted or shown)."
            ]
            for c in report.candidates[:40]:
                size_label = str(c.size_bytes) if c.size_complete else f"at least {c.size_bytes}"
                lines.append(
                    f"- [{c.risk.value}] {c.relpath} — {c.file_type}, "
                    f"{size_label} bytes, {'tracked' if c.tracked_by_git else 'untracked'}, "
                    f"action: {c.proposed_action.value}")
            if report.gitignore_suggestions:
                lines.append("Suggested .gitignore entries: " + ", ".join(report.gitignore_suggestions))
            if not report.complete:
                lines.append(f"Partial review: {report.stop_reason}. Only inspected entries are shown.")
            lines.append("Nothing has been deleted. Say which files (or the safe group) to remove "
                         "and confirm the on-screen prompt.")
            return ToolResult("\n".join(lines))
        if action in {"plan", "prepare", "select"}:
            from .execution_policy import require_foreground
            blocked = require_foreground("cleanup_confirmation_dialog")
            if blocked is not None:
                return blocked
            selected = args.get("paths") or args.get("files") or []
            if isinstance(selected, str):
                selected = [selected]
            if str(args.get("group") or "").lower() in {"safe", "deletable", "artefacts", "artifacts"}:
                selected = [c.path for c in assistant.scan().deletable()]
            try:
                plan = assistant.plan_deletion([str(p) for p in selected])
            except CleanupError as exc:
                return ToolResult(str(exc), ok=False)
            preview = assistant.dry_run(plan)
            # Surface the token to the confirmation UI ONLY — not to the model.
            self.bus.confirm_action.emit({
                "token": plan.token, "intent": "file_cleanup",
                "message": f"Delete {len(preview)} reviewed file(s)?",
                "items": [p["path"] for p in preview],
            })
            body = "\n".join(f"- {p['path']} ({p['action']})" for p in preview)
            return ToolResult(
                f"Prepared a deletion of {len(preview)} file(s). Dry-run preview:\n{body}\n"
                "Approve the on-screen confirmation to proceed; nothing is deleted yet.",
                ok=False)
        return ToolResult("cleanup_review actions: 'scan' or 'plan' (with 'paths' or group='safe').", ok=False)

    def _git_tracked_paths(self, root: Path) -> set[str]:
        """Resolved paths Git currently tracks under *root* (best effort)."""
        import subprocess
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z"], cwd=str(root),
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False,
            )
            tracked: set[str] = set()
            for rel in out.stdout.split("\0"):
                rel = rel.strip()
                if rel:
                    tracked.add(str((root / rel).resolve()))
            return tracked
        except Exception:
            return set()

    def _cleanup(self):
        """Lazily build the CleanupAssistant scoped to the project root."""
        assistant = getattr(self, "_cleanup_assistant", None)
        if assistant is None:
            from .cleanup_assistant import CleanupAssistant
            assistant = CleanupAssistant(
                [BASE_DIR],
                git_tracked=self._git_tracked_paths(BASE_DIR),
                logger=self.bus.log.emit,
            )
            self._cleanup_assistant = assistant
        return assistant

    def confirm_cleanup(self, token: str, second_confirm: bool = False) -> ToolResult:
        """Deterministic, human-gated release point for a planned file deletion
        (called by the GUI confirmation, not the model)."""
        from .cleanup_assistant import CleanupError
        assistant = self._cleanup()
        try:
            outcome = assistant.confirm_and_delete(str(token or ""), second_confirm=second_confirm)
        except CleanupError as exc:
            return ToolResult(str(exc), ok=False)
        return ToolResult(
            f"Removed {len(outcome.deleted)} file(s) "
            f"({'to the recycle bin' if outcome.recoverable else 'permanently'}); "
            f"skipped {len(outcome.skipped)}.")

    def cancel_cleanup(self, token: str) -> ToolResult:
        self._cleanup().cancel_plan(str(token or ""))
        return ToolResult("Cleanup cancelled — nothing was deleted.")

    async def organise_files_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.file_organiser is None:
            return ToolResult("The file organiser is not available.", ok=False)
        action = str(args.get("action") or "preview").lower().strip()
        folder = str(args.get("folder") or "Downloads")
        by_month = bool(args.get("by_month"))
        if action in {"preview", "plan", "dry_run"}:
            return await self.file_organiser.preview(folder, by_month=by_month)
        if action in {"apply", "organise", "organize", "run"}:
            return await self.file_organiser.organise(folder, apply=True, by_month=by_month)
        if action in {"undo", "reverse"}:
            return await self.file_organiser.undo()
        return ToolResult("Unsupported organise_files action. Use preview, apply, or undo.", ok=False)

    async def backup_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.backup is None:
            return ToolResult("The backup manager is not available.", ok=False)
        action = str(args.get("action") or "backup").lower().strip()
        if action in {"backup", "create", "now"}:
            return await self.backup.backup(str(args.get("note") or ""))
        if action in {"list", "show"}:
            return self.backup.list_backups()
        if action in {"restore"}:
            return await self.backup.restore(str(args.get("archive") or ""))
        return ToolResult("Unsupported backup action. Use backup, list, or restore.", ok=False)

    async def security_watch_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.security is None:
            return ToolResult("The security sentinel is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"audit", "score", "how_secure", "hardening", "health"}:
            from .security_audit import run_audit
            return ToolResult(await asyncio.to_thread(run_audit, self.security))
        if action in {"phone_off", "stop_phone", "stop_everything", "phone_control_off"}:
            from . import phone_control
            return ToolResult(await asyncio.to_thread(phone_control.set_enabled, False))
        if action in {"phone_on", "phone_control_on", "allow_phone"}:
            from . import phone_control
            return ToolResult(await asyncio.to_thread(phone_control.set_enabled, True))
        if action in {"phone_activity", "phone_log", "phone_steps", "phone"}:
            from . import phone_control
            return ToolResult(await asyncio.to_thread(phone_control.summary,
                                                      float(args.get("days") or 7)))
        if action in {"private_log", "private_log_on", "private_log_off", "privacy"}:
            from . import session_log
            wanted = args.get("enabled")
            if action.endswith("_off"):
                wanted = False
            elif action.endswith("_on"):
                wanted = True
            if wanted is None:
                state = "on" if session_log.private() else "off"
                return ToolResult(f"The private log is {state}.")
            on = str(wanted).strip().lower() not in {"false", "0", "no", "off"}
            await asyncio.to_thread(session_log.set_private, on)
            return ToolResult(
                "Private log on: my log keeps what I do, never what either of us says."
                if on else "Private log off: my log records the conversation again.")
        if action in {"status", "report", "check"}:
            return self.security.status()
        if action in {"network", "dashboard", "netsec", "posture"}:
            return await asyncio.to_thread(self.security.network_dashboard)
        if action in {"enable", "on"}:
            self.security.set_enabled(True)
            return ToolResult("Cybersecurity monitoring enabled.")
        if action in {"disable", "off"}:
            self.security.set_enabled(False)
            return ToolResult("Cybersecurity monitoring disabled.")
        return ToolResult("Unsupported security_watch action. Use audit, status, network, "
                          "enable, disable, phone_off, phone_on, phone_activity or "
                          "private_log.", ok=False)

    async def breach_check_tool(self, args: dict[str, Any]) -> ToolResult:
        """Check whether a password or account appears in known data breaches (HIBP)."""
        if self.breach_monitor is None:
            return ToolResult("Breach monitoring is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        password = str(args.get("password") or "")
        account = str(args.get("account") or args.get("email") or "").strip()
        if action in {"password", "pwd", "pass"} or (not action and password):
            return await self.breach_monitor.check_password(password)
        if action in {"account", "email"} or (not action and account):
            return await self.breach_monitor.check_account(account)
        return ToolResult(
            "breach_check needs either a 'password' (checked privately via "
            "k-anonymity — only a partial hash is sent) or an 'account'/'email'.",
            ok=False)

    async def antivirus_tool(self, args: dict[str, Any]) -> ToolResult:
        """Windows Defender awareness (antivirus.py) — protection status, a quick
        scan, or recently detected threats. No detection logic of ORION's own;
        entirely delegated to the OS's own antivirus. Degrades to an actionable
        message when Defender isn't available (not Windows, PowerShell/Defender
        module missing, or a third-party AV owns real-time protection)."""
        if self.antivirus is None:
            return ToolResult("Antivirus awareness is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "report", "check"}:
            return await self.antivirus.status()
        if action in {"scan", "quick_scan", "scan_now"}:
            return await self.antivirus.quick_scan()
        if action in {"threats", "recent_threats", "detections"}:
            limit = int(args.get("limit") or 10)
            return await self.antivirus.recent_threats(limit)
        if action in {"file", "scan_file", "check_file"}:
            return await asyncio.to_thread(_scan_one_file, str(args.get("path") or ""))
        if action in {"downloads", "scan_downloads"}:
            return await asyncio.to_thread(_scan_downloads)
        return ToolResult("Unsupported antivirus action. Use status, scan, threats, file "
                          "or downloads.", ok=False)

    async def forge_tool(self, args: dict[str, Any]) -> ToolResult:
        """
        Dynamically forge and activate new tools on the fly.

        Orchestrates: Plan → Code Gen → Sandbox (with self-heal) →
                     Dependency Check → Live Activation.
        """
        if self.forge is None:
            return ToolResult("The Forge capability-forging engine is unavailable.", ok=False)
        action = str(args.get("action") or "forge").lower().strip()

        if action in {"review", "self_review", "improve_now", "self_improve"}:
            # The heartbeat's review, now rather than at its next 25-minute
            # tick: same evidence rules (with nothing failing and nothing
            # unmet it says so instead of inventing work), same forge caps.
            heartbeat = getattr(self, "improvement", None)
            if heartbeat is None:
                return ToolResult("The self-improvement heartbeat is not running.", ok=False)
            entry = await heartbeat.tick()
            lines = [str(entry.get("analysis") or "Nothing actionable right now.")]
            for item in entry.get("forged") or []:
                if not isinstance(item, dict):
                    continue
                name = item.get("tool") or "a tool"
                if item.get("skipped"):
                    lines.append(f"- {name}: skipped ({item['skipped']})")
                else:
                    lines.append(f"- {name}: {'forged' if item.get('ok') else 'failed'}"
                                 f" - {item.get('detail', '')}")
            notes = [a.get("text") for a in entry.get("actions") or []
                     if isinstance(a, dict) and a.get("type") == "note" and a.get("text")]
            lines += [f"- noted: {note}" for note in notes[:3]]
            return ToolResult("\n".join(lines))
        if action in {"gaps", "unmet"}:
            from .capability_gaps import LEDGER
            return ToolResult(LEDGER.report())
        if action == "benchmarks":
            return self.forge.improvement_benchmarks()
        if action == "improvement_status":
            return await self.forge.improvement_status(str(args.get("run_id") or ""))
        if action == "improve":
            return await self.forge.improve_tool(
                str(args.get("tool_name") or "").strip(),
                str(args.get("benchmark") or "").strip(),
                generations=args.get("generations", 3),
                budget_seconds=args.get("budget_seconds", 300),
                tool_plan=args.get("tool_plan", ""))

        if action in {"forge", "build", "create_tool"}:
            tool_name = str(args.get("tool_name") or args.get("name") or "").strip()
            tool_plan = str(args.get("tool_plan") or args.get("plan") or "").strip()
            if not tool_name or not tool_plan:
                return ToolResult(
                    "forge requires 'tool_name' and 'tool_plan'. "
                    "Example: forge(tool_name='summarise_video', tool_plan='...')",
                    ok=False,
                )
            return await self.forge.forge_tool(tool_name, tool_plan)

        if action in {"batch", "forge_batch", "create_tools"}:
            tool_specs = args.get("tool_specs") or args.get("specs") or []
            if not isinstance(tool_specs, list):
                return ToolResult("batch requires 'tool_specs' (list of name + plan dicts).", ok=False)
            return await self.forge.forge_batch(tool_specs)

        if action in {"session", "status", "session_status"}:
            session_id = str(args.get("session_id") or "")
            if not session_id:
                return self.forge.list_sessions()
            return self.forge.session_status(session_id)

        if action in {"health", "quarantine", "held_back", "check"}:
            return self.forge.health()

        return ToolResult(
            f"Unsupported forge action: {action}. Use 'forge' (single tool), "
            "'batch' (multiple), 'session' (track progress), or 'health' "
            "(what is quarantined or held back), 'benchmarks', 'improve', or 'improvement_status'.",
            ok=False,
        )
