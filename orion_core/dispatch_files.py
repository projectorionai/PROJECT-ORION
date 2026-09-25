"""
Dispatch domain — Files, dev workbench, codebase, self-repair, diagnostics, cleanup, backup and forge.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
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


class FilesDispatchMixin:
    """Files, dev workbench, codebase, self-repair, diagnostics, cleanup, backup and forge."""

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
        if action in {"usage", "space", "disk_usage", "biggest_folders", "sizes"}:
            folder = str(args.get("folder") or path_arg or "C:\\")
            rows, cut = await asyncio.to_thread(file_search.folder_usage, folder)
            if not rows:
                return ToolResult(f"Could not read the folders inside {folder}.", ok=False)
            lines = [f"{file_search.human_size(size):>10}  {name}" for name, size in rows]
            return ToolResult(f"Largest folders in {folder}"
                              + (" (stopped at the time limit — partial)" if cut else "")
                              + ":\n" + "\n".join(lines))
        if path_arg and action in {"open", "reveal", "show", "show_in_folder"}:
            problem = (file_search.reveal_path(path_arg) if action != "open"
                       else file_search.open_path(path_arg))
            if problem:
                return ToolResult(f"Could not {action} {path_arg}: {problem}", ok=False)
            return ToolResult(f"{'Opened' if action == 'open' else 'Showing'} {path_arg}.")

        raw_query = SecuritySanitiser.guard_text(
            str(args.get("query") or args.get("name") or ""), "find_files.query").strip()
        args = dict(args, query=raw_query)
        open_first = bool(args.get("open") or args.get("open_first")) or action == "open"
        reveal_first = bool(args.get("reveal")) or action in {"reveal", "show_in_folder"}

        if args.get("contents") or args.get("inside"):
            # Searching INSIDE files is the Windows index's full-text job.
            from . import fast_find
            search = await asyncio.to_thread(
                fast_find.find, raw_query.lower(), contents=True,
                folder=str(args.get("folder") or args.get("in") or ""),
                thorough=bool(args.get("thorough")))
            matches = [hit.path for hit in search.hits]
            if not matches:
                return ToolResult(f"Nothing mentions '{raw_query}' ({search.strategy}).")
            if open_first:
                problem = file_search.open_path(matches[0])
                if problem:
                    return ToolResult(f"Found it but could not open it: {problem}", ok=False)
                return ToolResult(f"Opened {matches[0]}.")
            return ToolResult(f"{search.describe()}\n" + "\n".join(matches[:25]))

        query = file_search.build_query(args)
        if query.is_empty():
            return ToolResult("Tell me what to look for — a name, a type (pdf, "
                              "video…), a size or a date.", ok=False)
        own: list[Any] = []
        if (open_first or reveal_first) and query.name:
            own = await asyncio.to_thread(file_search.find_own_documents, query.name)
        results = await asyncio.to_thread(file_search.search, query)
        seen = {hit.path for hit in own}
        hits = own + [hit for hit in results.hits if hit.path not in seen]
        if not hits:
            return ToolResult(results.summary())
        if open_first or reveal_first:
            best = hits[0].path
            problem = (file_search.open_path(best) if open_first
                       else file_search.reveal_path(best))
            if problem:
                return ToolResult(f"Found {best} but could not "
                                  f"{'open' if open_first else 'show'} it: {problem}",
                                  ok=False)
            others = "\n".join(hit.line() for hit in hits[1:8])
            return ToolResult(
                f"{'Opened' if open_first else 'Showing in File Explorer'}: {best}"
                + (f"\nOther matches:\n{others}" if others else ""))
        return ToolResult(results.summary() + "\n" + "\n".join(
            hit.line() for hit in hits[:query.limit]))

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

    def file_controller(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        path   = self._resolve_user_path(str(args.get("path") or args.get("directory") or BASE_DIR))
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
                first_path = matches[0].split(":", 1)[0]
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
            previous: str | None = None
            too_large = False
            if path.exists() and path.is_file():
                try:
                    if path.stat().st_size > UNDO_MAX_BYTES:
                        too_large = True
                    else:
                        previous = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    too_large = True
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if action == "append_text" else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(text)

            if not too_large:
                def _restore(target=path, before=previous) -> str:
                    if before is None:
                        # The file did not exist before ORION wrote it.
                        target.unlink(missing_ok=True)
                        return "removed the file it created"
                    target.write_text(before, encoding="utf-8")
                    return "restored the previous contents"

                push_undo(f"wrote {path.name}", _restore)
            note = "" if not too_large else (
                f"  (over {UNDO_MAX_BYTES // 1024} KB — this write cannot be undone)")
            return ToolResult(
                f"File {'appended' if mode == 'a' else 'written'}: {path}{note}")
        if action in {"read", "read_text"}:
            if not path.exists() or not path.is_file():
                return ToolResult(f"File not found: {path}", ok=False)
            text = path.read_text(encoding="utf-8", errors="replace")
            return ToolResult(text[:6000])
        if action in {"delete", "remove"}:
            self._ensure_write_safe(path)
            if not path.exists():
                return ToolResult(f"Path already absent: {path}")
            if path.is_dir():
                if any(path.iterdir()):
                    return ToolResult("Refusing to remove a non-empty directory.", ok=False)
                path.rmdir()

                def _recreate(target=path) -> str:
                    target.mkdir(parents=True, exist_ok=True)
                    return "put the empty folder back"

                push_undo(f"removed folder {path.name}", _recreate)
                return ToolResult(f"Removed: {path}")

            # A file. Capture it first, or the deletion is final: there is no
            # "before" to read once it is gone.
            captured: str | None = None
            try:
                if path.stat().st_size <= UNDO_MAX_BYTES:
                    captured = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                captured = None
            path.unlink()
            if captured is not None:
                def _restore(target=path, before=captured) -> str:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(before, encoding="utf-8")
                    return "restored the file"

                push_undo(f"deleted {path.name}", _restore)
                return ToolResult(f"Removed: {path}  (say \"undo\" to restore it)")
            return ToolResult(
                f"Removed: {path}  (too large to hold for an undo — this is final)")
        return ToolResult(f"Unsupported file action: {action}", ok=False)

    async def process_file(self, args: dict[str, Any]) -> ToolResult:
        raw_path = str(args.get("path") or args.get("file_path") or "")
        if not raw_path:
            return ToolResult("No file path supplied.", ok=False)
        prompt = str(args.get("prompt") or args.get("question") or args.get("instruction") or "")
        path   = Path(os.path.expandvars(os.path.expanduser(raw_path)))
        if not path.is_absolute():
            path = BASE_DIR / path
        # inspect_async offloads all synchronous file I/O via asyncio.to_thread()
        result = await self.file_intel.inspect_async(path, prompt=prompt)
        if result.ok:
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
            completed = subprocess.run(
                parts, cwd=str(cwd), capture_output=True, text=True,
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
        self._ensure_write_safe(root)
        package = root / "src" / name
        tests   = root / "tests"
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

    async def diagnostics_tool(self, args: dict[str, Any]) -> ToolResult:
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
            # says yes to it. Only an already-eligible candidate can be
            # promoted, so this cannot invent a shortcut out of nothing.
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
            return ToolResult(learner.report(), ok=True)
        if action in {"routing", "resolver", "tool_routing", "shadow"}:
            # What the tool pre-filter WOULD do, measured on real turns while it
            # is switched off. Recall is the number that decides whether it can
            # safely be enabled; everything else is the benefit.
            try:
                from .resolver_shadow import ShadowEvaluator
                return ToolResult(ShadowEvaluator().report(), ok=True)
            except Exception as exc:
                return ToolResult(f"Shadow evaluation unavailable: {exc}", ok=False)
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
                lines.append(
                    f"- [{c.risk.value}] {c.relpath} — {c.file_type}, "
                    f"{c.size_bytes} bytes, {'tracked' if c.tracked_by_git else 'untracked'}, "
                    f"action: {c.proposed_action.value}")
            if report.gitignore_suggestions:
                lines.append("Suggested .gitignore entries: " + ", ".join(report.gitignore_suggestions))
            lines.append("Nothing has been deleted. Say which files (or the safe group) to remove "
                         "and confirm the on-screen prompt.")
            return ToolResult("\n".join(lines))
        if action in {"plan", "prepare", "select"}:
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
                capture_output=True, text=True, timeout=15, check=False,
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
        return ToolResult("Unsupported security_watch action. Use status, network, "
                          "enable, or disable.", ok=False)

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
        return ToolResult("Unsupported antivirus action. Use status, scan, or threats.",
                          ok=False)

    async def forge_tool(self, args: dict[str, Any]) -> ToolResult:
        """
        Dynamically forge and activate new tools on the fly.

        Orchestrates: Plan → Code Gen → Sandbox (with self-heal) →
                     Dependency Check → Live Activation.
        """
        if self.forge is None:
            return ToolResult("The Forge capability-forging engine is unavailable.", ok=False)
        action = str(args.get("action") or "forge").lower().strip()

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
            "(what is quarantined or held back).",
            ok=False,
        )
