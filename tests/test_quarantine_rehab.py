"""
Letting a quarantined tool out when its blocker is gone.

  "The 'self_repair_agent_tool' is quarantined for some reason, find out why
   this is and fix it."

The reason: it imports scikit-learn at module level, and when it was forged
sklearn was not installed, so the import raised and the file was quarantined as
``.broken``. The loader later learned to install-and-retry a missing dependency
at LOAD time — but nothing ever revisited the graveyard, so a tool condemned
before its package arrived stayed condemned forever. sklearn is installed now;
the tool was still in quarantine.

Rehabilitation closes that gap: each boot, quarantined files are re-checked and
any that now parse, import (installing a still-missing named package) and
satisfy the contract are restored. Anything genuinely broken is left exactly
where it is.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dynamic_loader import ReflectiveModuleLoader  # noqa: E402


class _Bus:
    def __getattr__(self, name):
        return types.SimpleNamespace(
            emit=lambda *a, **k: None, connect=lambda *a, **k: None)


@pytest.fixture
def loader(tmp_path, monkeypatch):
    engine = ReflectiveModuleLoader(_Bus())
    # Redirect the tool tree at the instance, so a test never touches the real
    # config/custom_tools.
    engine.custom_tools_dir = tmp_path
    (tmp_path / "_quarantine").mkdir()
    return engine


def _quarantine(loader, filename: str, source: str, reason: str = "test"):
    qdir = loader.custom_tools_dir / "_quarantine"
    (qdir / filename).write_text(source, encoding="utf-8")
    index = qdir / "index.json"
    data = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    data[filename] = {"tool": filename.split(".")[0], "reason": reason}
    index.write_text(json.dumps(data), encoding="utf-8")


GOOD_TOOL = '''
def get_tool_schema():
    return {"name": "revived", "description": "d",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}},
                           "required": ["x"]}}

def run(x):
    return x
'''

BROKEN_SYNTAX = "def get_tool_schema(:\n    return {}\n"

NEEDS_MISSING = '''
import a_package_that_will_never_exist_xyz
def get_tool_schema():
    return {"name": "n", "description": "d", "parameters": {"type": "object", "properties": {}}}
def run(**kwargs):
    return "ok"
'''


# ── the happy path ───────────────────────────────────────────────────────────

def test_a_now_loadable_tool_is_restored(loader):
    _quarantine(loader, "revived_tool.py.broken", GOOD_TOOL)
    restored = asyncio.run(loader.rehabilitate_quarantined())
    assert restored == ["revived_tool"]
    assert (loader.custom_tools_dir / "revived_tool.py").exists()
    assert not list((loader.custom_tools_dir / "_quarantine").glob("*.broken"))


def test_restoration_clears_the_index(loader):
    _quarantine(loader, "revived_tool.py.broken", GOOD_TOOL)
    asyncio.run(loader.rehabilitate_quarantined())
    index = json.loads(
        (loader.custom_tools_dir / "_quarantine" / "index.json").read_text(encoding="utf-8"))
    assert "revived_tool.py.broken" not in index


def test_a_timestamped_quarantine_name_maps_to_the_right_tool(loader):
    """Repeated quarantines are stored as tool.<timestamp>.py.broken."""
    _quarantine(loader, "revived_tool.1699999999.py.broken", GOOD_TOOL)
    restored = asyncio.run(loader.rehabilitate_quarantined())
    assert restored == ["revived_tool"]
    assert (loader.custom_tools_dir / "revived_tool.py").exists()


# ── the things it must NOT do ────────────────────────────────────────────────

def test_genuinely_broken_code_is_left_alone(loader):
    _quarantine(loader, "bad_tool.py.broken", BROKEN_SYNTAX)
    assert asyncio.run(loader.rehabilitate_quarantined()) == []
    assert (loader.custom_tools_dir / "_quarantine" / "bad_tool.py.broken").exists()


def test_a_still_missing_package_stays_quarantined(loader, monkeypatch):
    async def _no_install(package):
        return False

    monkeypatch.setattr(loader, "_install", _no_install)
    _quarantine(loader, "needy_tool.py.broken", NEEDS_MISSING)
    assert asyncio.run(loader.rehabilitate_quarantined()) == []
    assert (loader.custom_tools_dir / "_quarantine" / "needy_tool.py.broken").exists()


def test_a_live_tool_is_never_clobbered(loader):
    """If a working version already exists, the quarantined copy must not
    overwrite it."""
    (loader.custom_tools_dir / "revived_tool.py").write_text(
        "def get_tool_schema():\n    return {'name':'live'}\ndef run(**k):\n    return 1\n",
        encoding="utf-8")
    _quarantine(loader, "revived_tool.py.broken", GOOD_TOOL)
    assert asyncio.run(loader.rehabilitate_quarantined()) == []
    assert "live" in (loader.custom_tools_dir / "revived_tool.py").read_text(encoding="utf-8")


def test_a_contract_violating_tool_is_not_restored(loader):
    """Imports fine, but run() cannot accept a declared parameter — the same
    gate a fresh forge faces."""
    violating = (
        'def get_tool_schema():\n'
        '    return {"name":"v","description":"d","parameters":{"type":"object",'
        '"properties":{"needed":{"type":"string"}},"required":["needed"]}}\n'
        'def run():\n'          # accepts nothing, but schema requires 'needed'
        '    return 1\n')
    _quarantine(loader, "viol_tool.py.broken", violating)
    assert asyncio.run(loader.rehabilitate_quarantined()) == []


def test_an_empty_quarantine_is_fine(loader):
    assert asyncio.run(loader.rehabilitate_quarantined()) == []


# ── the real inmate ──────────────────────────────────────────────────────────

def test_the_real_self_repair_agent_tool_would_rehabilitate():
    """The actual quarantined file (or its restored form): it must import and
    pass the contract with sklearn present. Skips if neither is on disk."""
    root = Path(__file__).resolve().parents[1] / "config" / "custom_tools"
    candidate = root / "self_repair_agent_tool.py"
    broken = root / "_quarantine" / "self_repair_agent_tool.py.broken"
    source_file = candidate if candidate.exists() else broken
    if not source_file.exists():
        pytest.skip("self_repair_agent_tool is not present in either location")
    pytest.importorskip("sklearn")
    from orion_core.forge_contract import contract_problems

    namespace: dict = {}
    exec(compile(source_file.read_text(encoding="utf-8"), str(source_file), "exec"),
         namespace)
    assert callable(namespace.get("get_tool_schema"))
    assert callable(namespace.get("run"))
    assert not contract_problems(namespace["get_tool_schema"](), namespace["run"])


# ── it is wired into startup ─────────────────────────────────────────────────

def test_startup_rehabilitates_before_loading():
    import inspect

    from orion_core.forge import ForgeOrchestrationManager

    source = inspect.getsource(ForgeOrchestrationManager.reload_persisted_tools)
    assert "rehabilitate_quarantined" in source
    # Must run BEFORE the glob load so a restored tool is picked up same pass.
    assert source.index("rehabilitate_quarantined") < source.index("glob(")
