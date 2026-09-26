"""
Every background task run_application starts is cancelled AND awaited on
shutdown.

The proactivity engine's task was created and never cancelled, and the
housekeeping and diagnostics tasks were cancelled but never awaited, so
teardown went on (closing the services they use) while they still ran. This
reads app.py's own structure, so a task added later cannot be left out.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "orion_core" / "app.py"


def _run_application() -> ast.AsyncFunctionDef:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_application")


def _created_tasks(fn) -> set[str]:
    names = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "asyncio.create_task"):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _loops_over(fn, created: set[str]) -> list[tuple[set[str], ast.For]]:
    loops = []
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            names = {e.id for e in node.iter.elts if isinstance(e, ast.Name)}
            if names & created:
                loops.append((names, node))
    return loops


def _explicitly(fn, method: str) -> set[str]:
    return {ast.unparse(n.func.value) for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == method}


def test_every_background_task_is_cancelled_and_awaited_on_shutdown():
    fn = _run_application()
    created = _created_tasks(fn)
    assert len(created) > 10, "the task inventory could not be read"
    cancelled, awaited = set(_explicitly(fn, "cancel")), set()
    for names, loop in _loops_over(fn, created):
        body = ast.unparse(loop)
        if "task.cancel" in body:
            cancelled |= names
        if "await task" in body:
            awaited |= names
    explicitly_awaited = {ast.unparse(n.value) for n in ast.walk(fn)
                          if isinstance(n, ast.Await) and isinstance(n.value, ast.Name)}
    awaited |= explicitly_awaited
    assert not created - cancelled, f"never cancelled: {sorted(created - cancelled)}"
    assert not created - awaited, f"cancelled but never awaited: {sorted(created - awaited)}"
