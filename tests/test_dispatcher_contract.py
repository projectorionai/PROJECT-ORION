"""
Dispatcher contract tests (Improvement Pass, Priority 1.1).

The three invariants that keep the 90-tool routing table safe to evolve:

  1. every tool declared to the model has a live handler;
  2. every handler is reachable through dispatch() — including with malformed
     or empty arguments — without ever raising out of the tool layer
     (SecurityViolation is the one deliberate exception);
  3. every declaration is schema-consistent (unique snake_case name,
     description, OBJECT parameters whose required keys exist).

The sweep runs against a bare dispatcher whose services are absent, with the
browser/launcher side-effect seams patched out, so no tool can touch the host.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatcher import TOOL_DECLARATIONS, OrionDispatcher
from orion_core.security import SecurityViolation


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload))


class _StubBus:
    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


def _bare_dispatcher():
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    return d


@pytest.fixture(autouse=True)
def _no_host_side_effects(monkeypatch):
    """No test may open a browser, launch a process or touch the host."""
    import subprocess
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    # ORION opens links through orion_core.browser.open_url now, so stubbing
    # webbrowser alone no longer covers this. (open_url uses subprocess.Popen,
    # which _forbidden below would catch — but loudly, and this fixture's job
    # is to make the host untouchable, not to fail a test for reaching it.)
    from orion_core import browser as _browser
    monkeypatch.setattr(_browser, "open_url", lambda *a, **k: True)
    monkeypatch.setattr(subprocess, "Popen", _forbidden, raising=True)
    monkeypatch.setattr(subprocess, "run", _forbidden, raising=True)
    if hasattr(sys.modules.get("os"), "startfile"):
        import os
        monkeypatch.setattr(os, "startfile", lambda *a, **k: None)


def _forbidden(*args, **kwargs):
    raise AssertionError(f"tool sweep tried to spawn a process: {args!r}")


# ── declaration ⇄ handler consistency ────────────────────────────────────────

def test_every_declared_tool_has_a_live_handler():
    table = _bare_dispatcher().handler_table()
    declared = [decl["name"] for decl in TOOL_DECLARATIONS]
    missing = [name for name in declared if name not in table]
    assert not missing, f"declared tools without a handler: {missing}"


def test_every_handler_is_callable():
    table = _bare_dispatcher().handler_table()
    not_callable = [name for name, fn in table.items() if not callable(fn)]
    assert not_callable == []


def test_declaration_names_are_unique():
    names = [decl["name"] for decl in TOOL_DECLARATIONS]
    assert len(names) == len(set(names)), "duplicate tool declarations"


def test_handler_names_without_declaration_are_known_aliases():
    # Handlers may exceed declarations only for deliberate aliases / internal
    # tools; anything new must be added here consciously, not by accident.
    table = _bare_dispatcher().handler_table()
    declared = {decl["name"] for decl in TOOL_DECLARATIONS}
    undeclared = set(table) - declared
    known_internal = {"image_processor"}         # alias of process_file
    unexpected = undeclared - known_internal
    assert not unexpected, (
        f"handlers with no model-facing declaration: {sorted(unexpected)} — "
        "declare them in TOOL_DECLARATIONS or add to known_internal deliberately"
    )


# ── declaration schema consistency ────────────────────────────────────────────

def test_declaration_schema_shape():
    for decl in TOOL_DECLARATIONS:
        name = decl.get("name", "<missing>")
        assert isinstance(decl.get("name"), str) and decl["name"], decl
        assert decl["name"].replace("_", "").isalnum(), f"bad tool name {name}"
        assert decl["name"] == decl["name"].lower(), f"tool name not snake_case: {name}"
        assert isinstance(decl.get("description"), str) and decl["description"].strip(), \
            f"{name}: missing description"
        params = decl.get("parameters")
        assert isinstance(params, dict), f"{name}: parameters must be a dict"
        assert str(params.get("type", "")).upper() == "OBJECT", \
            f"{name}: parameters.type must be OBJECT"
        properties = params.get("properties", {})
        assert isinstance(properties, dict), f"{name}: properties must be a dict"
        for required_key in params.get("required", []):
            assert required_key in properties, \
                f"{name}: required key '{required_key}' not among properties"


def test_declared_property_types_are_valid():
    valid = {"STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY", "OBJECT"}
    for decl in TOOL_DECLARATIONS:
        for prop_name, prop in decl["parameters"].get("properties", {}).items():
            ptype = str(prop.get("type", "")).upper()
            assert ptype in valid, \
                f"{decl['name']}.{prop_name}: invalid schema type {ptype!r}"


# ── malformed-argument sweep: nothing raises out of dispatch() ────────────────

@pytest.mark.parametrize("tool_name", [decl["name"] for decl in TOOL_DECLARATIONS])
def test_dispatch_with_empty_args_never_raises(tool_name):
    d = _bare_dispatcher()
    result = asyncio.run(d.dispatch(tool_name, {}))
    assert isinstance(result, ToolResult)


@pytest.mark.parametrize("bad_args", [
    None,
    {},
    {"unexpected": object()},
    {"action": 42, "query": ["not", "a", "string"]},
])
def test_dispatch_malformed_args_return_toolresult(bad_args):
    d = _bare_dispatcher()
    result = asyncio.run(d.dispatch("window_control", bad_args))
    assert isinstance(result, ToolResult)


def test_unknown_tool_reports_cleanly():
    d = _bare_dispatcher()
    result = asyncio.run(d.dispatch("definitely_not_a_tool", {}))
    assert not result.ok and "Unknown dispatch target" in result.text


def test_dispatch_records_recent_tools_metadata():
    d = _bare_dispatcher()
    asyncio.run(d.dispatch("system_notify", {"message": "test ping"}))
    assert d.recent_tools and d.recent_tools[-1]["tool"] == "system_notify"
    assert d.active_tools == 0                      # always released


# ── the security firewall guards the tool boundary ────────────────────────────

def test_destructive_args_raise_security_violation():
    d = _bare_dispatcher()
    with pytest.raises(SecurityViolation):
        asyncio.run(d.dispatch("open_app", {"app_name": "cmd /c del *.*"}))


def test_destructive_tool_name_raises_security_violation():
    d = _bare_dispatcher()
    with pytest.raises(SecurityViolation):
        asyncio.run(d.dispatch("del orion_core", {}))


def test_nested_destructive_payload_caught():
    d = _bare_dispatcher()
    with pytest.raises(SecurityViolation):
        asyncio.run(d.dispatch("dev_workbench", {
            "action": "run", "options": {"cmd": "rm -rf /"},
        }))


# ── forged-tool fallthrough ───────────────────────────────────────────────────

def test_forged_tool_route():
    d = _bare_dispatcher()
    d._tool_handlers = {"custom_forged": lambda **kw: f"forged ran with {kw}"}
    result = asyncio.run(d.dispatch("custom_forged", {"a": 1}))
    assert result.ok and "forged ran" in result.text


def test_forged_tool_failure_contained():
    def _boom(**kw):
        raise RuntimeError("forge fault")
    d = _bare_dispatcher()
    d._tool_handlers = {"bad_forged": _boom}
    result = asyncio.run(d.dispatch("bad_forged", {}))
    assert not result.ok and "forge fault" in result.text
