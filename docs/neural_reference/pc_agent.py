from __future__ import annotations

import asyncio
import hashlib
import ntpath
import os
import stat
import subprocess
import threading
import uuid
from contextlib import suppress
from pathlib import Path, PureWindowsPath

from .async_utils import complete_before_cancel
from .contracts import Outcome, Plan


def _is_reserved_windows_path(part: str) -> bool:
    # Python 3.13 added ntpath.isreserved; PurePath.is_reserved is removed in
    # 3.15. Keep the 3.12 fallback used by the optional training environment.
    checker = getattr(ntpath, "isreserved", None)
    if checker is not None:
        return checker(part)
    return PureWindowsPath(part).is_reserved()


class LocalPCAgent:
    """Capability-limited example. Run without administrator privileges.

    The root must be a dedicated local directory with restrictive NTFS ACLs.
    Path checks do not defend against a malicious same-user process racing us.
    A production broker needs handle-based checks and OS-enforced isolation.
    """

    def __init__(
        self,
        root: Path,
        allowed_tools: frozenset[str],
        notepad: Path | None = None,
        nvidia_smi: Path | None = None,
    ):
        self.root = root.absolute()
        self._check_components(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._check_components(self.root)
        self.root = self.root.resolve(strict=True)
        self.allowed_tools = frozenset(allowed_tools)
        # Fixed paths supplied by trusted installation, never model arguments.
        self.notepad = notepad
        self.nvidia_smi = nvidia_smi
        self.lock = threading.Lock()

    @staticmethod
    def _check_components(path: Path):
        for candidate in (path, *path.parents):
            if candidate.exists() or candidate.is_symlink():
                info = candidate.lstat()
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                ):
                    raise PermissionError("Reparse points and symlinks are not allowed")

    def file(self, relative: str, must_exist: bool = False) -> Path:
        win = PureWindowsPath(relative)
        parts = relative.replace("\\", "/").split("/")
        if (
            win.drive
            or win.root
            or any(
                p in {"", ".", ".."}
                or ":" in p
                or p.endswith((" ", "."))
                or any(c in p for c in '<>"|?*')
                or any(ord(c) < 32 for c in p)
                or _is_reserved_windows_path(p)
                for p in parts
            )
        ):
            raise PermissionError("Only ordinary relative paths are allowed")
        target = self.root.joinpath(*parts)
        self._check_components(target)
        resolved = target.resolve(strict=must_exist)
        if not resolved.is_relative_to(self.root):
            raise PermissionError("Path leaves the authorised directory")
        if resolved.suffix.lower() not in {".txt", ".md", ".json", ".csv"}:
            raise PermissionError("Only plain data files are supported")
        if resolved.exists() and (
            not resolved.is_file() or resolved.stat().st_nlink > 1
        ):
            raise PermissionError("Expected an ordinary, unlinked data file")
        return resolved

    def authorise(self, plan: Plan):
        if any(s.action.tool not in self.allowed_tools for s in plan.steps):
            raise PermissionError("Missing server-issued capability")
        for step in plan.steps:
            # Static checks before admission; existence and state are checked
            # again at execution because earlier steps may create the source.
            if step.action.tool in {"read_text", "create_text"}:
                self.file(step.action.path)

    def _execute_sync(self, action):
        with self.lock:
            if action.tool == "resources":
                import psutil

                return {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "ram_percent": psutil.virtual_memory().percent,
                    "disk_free_bytes": psutil.disk_usage(str(self.root)).free,
                }
            if action.tool == "read_text":
                target = self.file(action.path, must_exist=True)
                with target.open("rb") as stream:
                    content = stream.read(65_537)
                if len(content) > 65_536:
                    raise ValueError("File exceeds read budget")
                return {
                    "text": content.decode("utf-8"),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            if action.tool == "create_text":
                target = self.file(action.path)
                content = action.text.encode("utf-8")
                # Exclusive creation: never silently replace an existing file.
                # Parent directories must have been provisioned by the user.
                with target.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                digest = hashlib.sha256(content).hexdigest()
                with target.open("rb") as stream:
                    actual = stream.read(len(content) + 1)
                if hashlib.sha256(actual).hexdigest() != digest:
                    raise OSError("File postcondition failed")
                return {"path": action.path, "sha256": digest, "bytes": len(content)}
            if action.tool == "launch_notepad":
                if not self.notepad or not self.notepad.is_absolute():
                    raise PermissionError("Notepad has not been provisioned")
                process = subprocess.Popen(
                    [str(self.notepad)],
                    shell=False,
                    cwd=str(self.root),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return {"pid": process.pid, "postcondition": "process_created"}
            if action.tool == "screenshot":
                import pyautogui

                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0.1
                # Screenshot capability includes access to visible private data.
                self._check_components(self.root)
                destination = self.root / f"screen_{uuid.uuid4().hex}.png"
                screenshot = pyautogui.screenshot()
                with destination.open("xb") as stream:
                    screenshot.save(stream, format="PNG")
                return {"path": destination.name, "postcondition": "image_saved"}
            raise PermissionError("Unregistered tool")

    async def _gpu_status(self):
        if not self.nvidia_smi or not self.nvidia_smi.is_absolute():
            raise PermissionError("nvidia-smi has not been provisioned")
        process = await asyncio.create_subprocess_exec(
            str(self.nvidia_smi),
            "--query-gpu=memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        except BaseException:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await complete_before_cancel(process.wait())
            raise
        if process.returncode:
            raise RuntimeError("GPU probe failed")
        return {"gpu_csv": output[:4096].decode("ascii", errors="replace")}

    async def execute(self, action) -> Outcome:
        if action.tool not in self.allowed_tools:
            return Outcome(ok=False, error="MissingCapability")
        try:
            data = (
                await self._gpu_status()
                if action.tool == "gpu_status"
                else await complete_before_cancel(
                    asyncio.to_thread(self._execute_sync, action)
                )
            )
            return Outcome(ok=True, data=data)
        except Exception as exc:  # noqa: BLE001 - convert tool failures to receipts
            # Do not turn paths, credentials or file content into telemetry labels.
            return Outcome(ok=False, error=type(exc).__name__)
