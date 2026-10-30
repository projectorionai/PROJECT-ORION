"""
ORION's own MCP servers, run by ORION's own executable.

In the packaged app every bundled server used to be launched as
``python <_internal>/orion_core/mcp_<name>.py``: whatever Python happened to
be on the PATH, pointed at a loose copy of one file. That worked only while
the file imported nothing of ORION's - the copy sits beside none of the other
modules, which live inside the executable's compiled archive. orion_world
imports world_feeds, so in the exe it died on its first import:

    MCP[orion_world]: handshake failed - server closed the connection
    (... _internal\\orion_core\\mcp_world.py, line 15 ... from orion_core.world_feeds import ...)

and on a PC with no Python at all, none of them could start. Frozen, the host
now runs ``ORION.exe --mcp-server orion_world`` instead: the same frozen
modules the app itself uses, no outside interpreter. ``orion.py`` hands that
argument here before it claims the single-instance lock or imports Qt.
"""

from __future__ import annotations

import importlib
import io
import os
import sys
from collections.abc import Sequence

#: server name -> (module, its entry point) - every one is a stdio MCP server.
BUNDLED: dict[str, tuple[str, str]] = {
    "orion_world": ("orion_core.mcp_world", "main"),
    "orion_network": ("orion_core.mcp_network", "main"),
    "orion_telemetry": ("orion_core.mcp_telemetry", "main"),
    "orion_desktop": ("orion_core.mcp_desktop", "run"),
}

#: The bundled module file each server used to be launched from.
FILES: dict[str, str] = {name: module.rsplit(".", 1)[1] + ".py"
                         for name, (module, _entry) in BUNDLED.items()}

FLAG = "--mcp-server"


def launch_spec(name: str) -> tuple[str, list[str]]:
    """(command, args) that start bundled server *name* in this process's
    runtime: the exe itself when frozen, this interpreter on the file when not."""
    if getattr(sys, "frozen", False):
        return sys.executable, [FLAG, name]
    from pathlib import Path
    return sys.executable, [str(Path(__file__).with_name(FILES[name]))]


def _ensure_stdio() -> None:
    """A windowed executable may start with no Python-level std streams even
    when its parent handed it pipes; rebuild them from the inherited handles."""
    if sys.stdin is None:
        sys.stdin = io.TextIOWrapper(io.FileIO(0, "rb", closefd=False), encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = io.TextIOWrapper(io.FileIO(1, "wb", closefd=False), encoding="utf-8",
                                      line_buffering=True)
    if sys.stderr is None:
        try:
            sys.stderr = io.TextIOWrapper(io.FileIO(2, "wb", closefd=False), encoding="utf-8",
                                          line_buffering=True)
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")


def run(argv: Sequence[str]) -> int:
    """Run the bundled server named first in *argv*. Returns an exit code."""
    name = str(argv[0]) if argv else ""
    target = BUNDLED.get(name)
    if target is None:
        return 2
    _ensure_stdio()
    module_name, entry = target
    module = importlib.import_module(module_name)
    result = getattr(module, entry)()
    if hasattr(result, "__await__"):                 # an async entry point
        import asyncio
        asyncio.run(result)
    return 0


__all__ = ["BUNDLED", "FILES", "FLAG", "launch_spec", "run"]
