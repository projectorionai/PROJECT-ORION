"""
Tests for AgentManager's declarative-agent wiring (Mark XXI, Track E3):
register_declarative() and load_declarative_agents(), and that
AgentManager.__init__ loads config/agents/*.json automatically without
ever letting a manifest shadow a built-in specialist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import agent_registry as ar
from orion_core.agents import AgentManager


class _Signal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, text) -> None:
        self.messages.append(text)


class _StubBus:
    def __init__(self) -> None:
        self.log = _Signal()

    def __getattr__(self, name):
        return _Signal()


class _StubRouter:
    def has_text_fallback(self):
        return False


def _write_manifest(tmp_path: Path, name: str) -> None:
    (tmp_path / f"{name}.json").write_text(json.dumps({
        "name": name,
        "title": f"{name.title()} Agent",
        "expertise": "a declarative specialist",
        "primary": [r"\b" + name + r"\b"],
        "keywords": [],
        "persona": f"You are the {name} specialist.",
    }), encoding="utf-8")


def _manager(bus=None) -> AgentManager:
    return AgentManager(_StubRouter(), bus or _StubBus())


# ── register_declarative() ──────────────────────────────────────────────────

def test_register_declarative_adds_a_new_agent():
    mgr = _manager()
    manifest = {
        "name": "legal", "title": "Legal Advisor Agent",
        "expertise": "contracts", "primary": [], "keywords": [],
        "persona": "You are the Legal Advisor.",
    }
    ok = mgr.register_declarative(manifest)
    assert ok is True
    assert "legal" in mgr.agent_names()


def test_register_declarative_refuses_to_shadow_a_built_in():
    mgr = _manager()
    assert "coding" in mgr.agent_names()   # a real built-in
    original = mgr._agents["coding"]
    manifest = {
        "name": "coding", "title": "Fake Coding Agent",
        "expertise": "x", "primary": [], "keywords": [],
        "persona": "impostor",
    }
    ok = mgr.register_declarative(manifest)
    assert ok is False
    assert mgr._agents["coding"] is original   # untouched


def test_register_declarative_refuses_a_blank_name():
    mgr = _manager()
    before = set(mgr.agent_names())
    ok = mgr.register_declarative({"name": "", "title": "x", "expertise": "x",
                                   "persona": "x", "primary": [], "keywords": []})
    assert ok is False
    assert set(mgr.agent_names()) == before


def test_register_declarative_refuses_a_duplicate_name():
    mgr = _manager()
    manifest = {"name": "legal", "title": "Legal", "expertise": "x",
               "persona": "x", "primary": [], "keywords": []}
    assert mgr.register_declarative(manifest) is True
    assert mgr.register_declarative(manifest) is False   # already registered now


# ── load_declarative_agents() ───────────────────────────────────────────────

def test_load_declarative_agents_registers_every_valid_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "legal")
    _write_manifest(tmp_path, "medical")
    mgr = _manager()   # __init__ already loads them
    assert "legal" in mgr.agent_names()
    assert "medical" in mgr.agent_names()


def test_load_declarative_agents_returns_the_count(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "legal")
    _write_manifest(tmp_path, "medical")
    mgr = _manager()
    # __init__ already consumed them; a second explicit call re-loads the
    # same manifests, all now duplicates of already-registered agents.
    assert mgr.load_declarative_agents() == 0


def test_load_declarative_agents_with_no_directory_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "absent")
    mgr = _manager()
    built_ins = {"marketing", "coding", "research", "neuroscience", "aiml",
                 "security"}
    assert set(mgr.agent_names()) == built_ins


def test_init_logs_a_skip_for_a_manifest_colliding_with_a_built_in(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "coding")   # collides with the real built-in
    bus = _StubBus()
    mgr = _manager(bus)
    assert any("coding" in m and "already registered" in m for m in bus.log.messages)
    # the built-in must still be the one actually registered
    assert mgr._agents["coding"].persona != "You are the coding specialist."


def test_a_declarative_agent_can_be_dispatched_to(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write_manifest(tmp_path, "legal")
    mgr = _manager()
    result = asyncio.run(mgr.dispatch("Help with a legal contract", agent_name="legal"))
    assert result.ok
    # no text-fallback provider configured -> hands back the specialist brief
    assert "legal specialist" in result.text.lower()
