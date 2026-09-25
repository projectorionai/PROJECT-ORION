"""
Regression tests for the FORGE loader hardening.

Reproduces the historic failure where a bare ``code`` stub was persisted as a
custom tool and the reflective loader then raised ``name 'code' is not defined``
on every boot.  Two guarantees are locked in:

* A tool file that will not load is *quarantined* (moved to ``_quarantine/``)
  so it never re-errors on subsequent boots.
* The forge refuses to write a placeholder or unparseable tool body at all.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dynamic_loader import ReflectiveModuleLoader
from orion_core.forge import ForgeOrchestrationManager


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, *_a, **_k):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def test_broken_tool_is_quarantined(tmp_path):
    """The exact historic corruption ('code') must be moved out, not retried."""
    loader = ReflectiveModuleLoader(_StubBus())
    bad = tmp_path / "netprobe_tool.py"
    bad.write_text("code\n", encoding="utf-8")

    outcome = asyncio.run(loader.load_and_register(bad))

    assert outcome.succeeded is False
    assert not bad.exists(), "broken tool should be removed from the load path"
    quarantined = tmp_path / "_quarantine" / "netprobe_tool.py.broken"
    assert quarantined.exists(), "broken tool should be quarantined"
    # And it must have cleaned itself out of sys.modules.
    assert "netprobe_tool" not in sys.modules


def test_syntax_error_tool_is_quarantined(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    bad = tmp_path / "broken_syntax_tool.py"
    bad.write_text("def run(:\n    pass\n", encoding="utf-8")

    outcome = asyncio.run(loader.load_and_register(bad))

    assert outcome.succeeded is False
    assert (tmp_path / "_quarantine" / "broken_syntax_tool.py.broken").exists()


def test_valid_tool_still_loads(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    good = tmp_path / "echo_tool.py"
    good.write_text(
        "def get_tool_schema():\n"
        "    return {'name': 'echo', 'description': 'echo', 'parameters': {}}\n\n"
        "def run(**kwargs):\n"
        "    return kwargs\n",
        encoding="utf-8",
    )

    outcome = asyncio.run(loader.load_and_register(good))

    assert outcome.succeeded is True
    assert good.exists()
    assert not (tmp_path / "_quarantine").exists()


def test_forge_refuses_placeholder_source():
    with pytest.raises(RuntimeError):
        ForgeOrchestrationManager._assert_valid_tool_source("code", "net_probe")
    with pytest.raises(RuntimeError):
        ForgeOrchestrationManager._assert_valid_tool_source("", "empty")
    # Long enough to pass the length gate but not valid Python → still refused.
    with pytest.raises(RuntimeError):
        ForgeOrchestrationManager._assert_valid_tool_source(
            "def valid_looking_name_with_padding():\n    return (\n", "trunc")


def test_forge_accepts_real_source():
    ForgeOrchestrationManager._assert_valid_tool_source(
        "def get_tool_schema():\n    return {}\n\ndef run(**k):\n    return 'ok'\n",
        "good",
    )
