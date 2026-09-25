"""
A forged tool must never be able to stall ORION's startup at import time.

ORION forges his own tools, so their module-level code runs during boot with
nobody having reviewed it.  `live_camera_analysis_tool.py` ended with

    camera_analysis = LiveCameraAnalysis()

whose __init__ called `cv2.VideoCapture(0)` — so *importing* the tool opened
the webcam.  On Windows that probe blocks or fails hard, producing the
"cv::obsensor ... Camera index out of range" line found in every
orion_startup*.log on this machine.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from orion_core.dynamic_loader import ReflectiveModuleLoader


class _Sig:
    def __init__(self):
        self.messages = []

    def emit(self, *a):
        self.messages.append(a[0] if len(a) == 1 else a)

    def connect(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


def _tool(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


GOOD = '''
def get_tool_schema():
    return {"name": "ok_tool", "description": "d",
            "parameters": {"type": "object", "properties": {}, "required": []}}

def run():
    return "ok"
'''

# 2s rather than 30s: asyncio.to_thread cannot be killed, so asyncio.run()
# waits for the worker at loop shutdown. The point under test is that
# load_and_register RETURNS promptly, not that the thread dies.
BLOCKS_AT_IMPORT = '''
import time
time.sleep(2.0)         # stands in for cv2.VideoCapture(0) on a bad device

def get_tool_schema():
    return {"name": "slow_tool", "description": "d",
            "parameters": {"type": "object", "properties": {}, "required": []}}

def run():
    return "never"
'''


def test_a_tool_that_blocks_at_import_is_timed_out(tmp_path):
    bus = _Bus()
    loader = ReflectiveModuleLoader(bus)
    loader.IMPORT_TIMEOUT_S = 0.3
    path = _tool(tmp_path, "slow_tool.py", BLOCKS_AT_IMPORT)

    async def go():
        started = time.monotonic()
        outcome = await loader.load_and_register(path)
        return outcome, time.monotonic() - started

    outcome, elapsed = asyncio.run(go())

    assert outcome.succeeded is False
    assert elapsed < 1.5, (
        "load_and_register must return as soon as the timeout fires, so the "
        "startup chain continues instead of waiting on the hung import")
    assert any("timed out" in str(m) for m in bus.log.messages)


def test_the_timeout_message_names_the_cause(tmp_path):
    bus = _Bus()
    loader = ReflectiveModuleLoader(bus)
    loader.IMPORT_TIMEOUT_S = 0.2
    asyncio.run(loader.load_and_register(_tool(tmp_path, "slow2.py", BLOCKS_AT_IMPORT)))
    text = " ".join(str(m) for m in bus.log.messages)
    assert "hardware at import" in text, (
        "the log must say what the user has to fix, not just that it failed")


def test_a_timed_out_tool_is_quarantined_so_it_cannot_recur(tmp_path):
    bus = _Bus()
    loader = ReflectiveModuleLoader(bus)
    loader.IMPORT_TIMEOUT_S = 0.2
    path = _tool(tmp_path, "slow3.py", BLOCKS_AT_IMPORT)
    asyncio.run(loader.load_and_register(path))
    assert not path.exists(), "the file must be moved out of the load path"
    # Quarantine suffixes .broken so the file is never re-imported.
    assert (tmp_path / "_quarantine" / "slow3.py.broken").exists()
    index = tmp_path / "_quarantine"
    assert any(index.iterdir()), "the reason must be recorded alongside it"


def test_a_normal_tool_still_loads(tmp_path):
    bus = _Bus()
    loader = ReflectiveModuleLoader(bus)
    outcome = asyncio.run(loader.load_and_register(_tool(tmp_path, "ok_tool.py", GOOD)))
    assert outcome.succeeded is True
    assert outcome.tool_schema["name"] == "ok_tool"


def test_the_import_timeout_is_generous_but_finite():
    assert 2.0 <= ReflectiveModuleLoader.IMPORT_TIMEOUT_S <= 30.0


# ── the specific tool that caused it ──────────────────────────────────────────

def test_the_camera_tool_no_longer_touches_hardware_at_import():
    """Regression: importing it must not construct anything that opens a device."""
    import ast

    source = Path("config/custom_tools/live_camera_analysis_tool.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        # Skip the module docstring: it QUOTES the old broken line to explain
        # the bug, which is exactly the string this test looks for.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, (ast.Assign, ast.Expr)):
            rendered = ast.unparse(node)
            assert "LiveCameraAnalysis()" not in rendered, (
                "the analyser must not be constructed at module level — that is "
                "what opened the webcam during startup")
            assert "VideoCapture" not in rendered


def test_the_camera_tool_still_exposes_the_forge_contract():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cam_tool", "config/custom_tools/live_camera_analysis_tool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = module.get_tool_schema()
    assert schema["name"] == "live_camera_analysis"
    assert callable(module.run)
