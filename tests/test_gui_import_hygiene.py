"""
Invariants that stop the whole test suite dying with no traceback.

Two rules, both learned the hard way:

1. QtWebEngine must not be imported at MODULE scope in new GUI modules.
   Qt requires it to be initialised before the process's QApplication exists.
   pytest imports every test module at collection time, long before any widget
   is built, so a module-scope import initialises WebEngine at the wrong moment
   and corrupts Qt for every real-widget test that runs afterwards — which
   eventually kills the interpreter with no Python traceback at all.

2. runJavaScript must never be called on a page that has not loaded.
   Same failure mode: a native crash the surrounding try/except cannot catch.

``face3d.py`` and ``globe.py`` predate the rule and are grandfathered — they
work in the real app because ``app.py`` imports QtWebEngineWidgets before
creating the QApplication, and nothing in the suite imports them. They are
listed explicitly so the exemption is a deliberate, visible decision rather
than a gap, and so a THIRD offender cannot appear unnoticed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parents[1] / "orion_core" / "gui"

#: Known, deliberate exceptions. Do not add to this list — make the new module
#: lazy instead (see swarm_view for the
#: pattern). Anything here is a module no test may import.
GRANDFATHERED = {"face3d.py", "globe.py"}


def _module_scope_imports(source: str) -> list[tuple[int, str]]:
    """Imports at module level, including inside a top-level try/except."""
    tree = ast.parse(source)
    found: list[tuple[int, str]] = []
    nodes: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
        elif isinstance(node, ast.Try):
            nodes.extend(n for n in node.body
                         if isinstance(n, (ast.Import, ast.ImportFrom)))
            for handler in node.handlers:
                nodes.extend(n for n in handler.body
                             if isinstance(n, (ast.Import, ast.ImportFrom)))
    for node in nodes:
        names = ([alias.name for alias in node.names]
                 if isinstance(node, ast.Import) else [node.module or ""])
        for name in names:
            found.append((node.lineno, str(name)))
    return found


def _gui_modules():
    return sorted(p for p in GUI.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _gui_modules(), ids=lambda p: p.name)
def test_no_new_module_scope_webengine_import(path):
    source = path.read_text(encoding="utf-8")
    offenders = [f"line {line}: {name}"
                 for line, name in _module_scope_imports(source)
                 if "QtWebEngine" in name]
    if path.name in GRANDFATHERED:
        return                      # deliberate, documented exemption
    assert not offenders, (
        f"{path.name} imports QtWebEngine at module scope ({offenders}). "
        "That corrupts Qt for every widget test that runs after it and kills "
        "the interpreter with no traceback. Make it lazy — see "
        "the lazy WebEngine pattern in swarm_view.py.")


def test_the_grandfathered_list_has_not_grown():
    """If this fails, a module was exempted instead of being made lazy."""
    assert GRANDFATHERED == {"face3d.py", "globe.py"}


def test_grandfathered_modules_really_do_still_offend():
    """If one gets fixed, take it off the list rather than leaving it stale."""
    still = {p.name for p in _gui_modules() if p.name in GRANDFATHERED
             and any("QtWebEngine" in name
                     for _line, name in _module_scope_imports(
                         p.read_text(encoding="utf-8")))}
    assert still == GRANDFATHERED, (
        f"these no longer import WebEngine at module scope and should be "
        f"removed from GRANDFATHERED: {GRANDFATHERED - still}")


# ── rule 2: the page must be loaded before it is driven ──────────────────────

@pytest.mark.parametrize("name", ["swarm_view.py"])
def test_heavy_pages_gate_javascript_on_load(name):
    source = (GUI / name).read_text(encoding="utf-8")
    assert "loadFinished" in source, (
        f"{name} must track loadFinished before calling runJavaScript")
    assert "_page_loaded" in source, (
        f"{name} must gate runJavaScript on the page having loaded — calling "
        "it on a freshly-built view crashes the process natively")


@pytest.mark.parametrize("name", ["swarm_view.py"])
def test_heavy_pages_stop_rendering_when_hidden(name):
    source = (GUI / name).read_text(encoding="utf-8")
    for hook in ("def showEvent", "def hideEvent", "def _set_active"):
        assert hook in source, f"{name} is missing {hook}"
    assert "cancelAnimationFrame" in source, (
        f"{name}'s render loop must be genuinely cancellable, not just idled")
