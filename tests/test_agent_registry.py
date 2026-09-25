"""
Tests for agent_registry.py (Mark XXI, Track E3) — declarative specialist
agents from config/agents/*.json, so a new specialist costs a manifest
file, not a code change. A manifest is data: name/title/expertise/regex
routing signals/persona text; build_agent() constructs a real BaseAgent
subclass from it, reusing the exact scoring/handle/investigate machinery
every built-in specialist already has.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import agent_registry as ar
from orion_core.agents import BaseAgent


class _Signal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, text) -> None:
        self.messages.append(text)


class _StubBus:
    def __init__(self) -> None:
        self.log = _Signal()


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_manifest(name: str = "legal") -> dict:
    return {
        "name": name,
        "title": "Legal Advisor Agent",
        "expertise": "contracts, terms of service, compliance basics",
        "primary": [r"\blegal\b", r"\bcontract\b"],
        "keywords": [r"\bclause\b", r"\bliability\b"],
        "persona": "You are ORION's Legal Advisor specialist.",
    }


# ── load_manifest() ──────────────────────────────────────────────────────────

def test_load_manifest_accepts_a_valid_file(tmp_path):
    path = _write(tmp_path, "legal", _valid_manifest())
    manifest, errors = ar.load_manifest(path)
    assert errors == []
    assert manifest["name"] == "legal"


def test_load_manifest_rejects_missing_required_fields(tmp_path):
    data = _valid_manifest()
    del data["persona"]
    path = _write(tmp_path, "legal", data)
    manifest, errors = ar.load_manifest(path)
    assert manifest is None
    assert any("persona" in e for e in errors)


def test_load_manifest_rejects_a_non_snake_case_name(tmp_path):
    data = _valid_manifest(name="Legal Advisor")
    path = _write(tmp_path, "legal", data)
    manifest, errors = ar.load_manifest(path)
    assert manifest is None
    assert any("snake_case" in e for e in errors)


def test_load_manifest_rejects_an_invalid_regex(tmp_path):
    data = _valid_manifest()
    data["primary"] = [r"\b(unclosed"]
    path = _write(tmp_path, "legal", data)
    manifest, errors = ar.load_manifest(path)
    assert manifest is None
    assert any("regex" in e for e in errors)


def test_load_manifest_handles_malformed_json_without_raising(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    manifest, errors = ar.load_manifest(path)
    assert manifest is None
    assert errors


def test_load_manifest_rejects_a_json_array_not_object(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    manifest, errors = ar.load_manifest(path)
    assert manifest is None


def test_load_manifest_primary_must_be_a_list(tmp_path):
    data = _valid_manifest()
    data["primary"] = "not a list"
    path = _write(tmp_path, "legal", data)
    manifest, errors = ar.load_manifest(path)
    assert manifest is None
    assert any("primary" in e for e in errors)


def test_load_manifest_primary_and_keywords_are_optional(tmp_path):
    data = _valid_manifest()
    del data["primary"]
    del data["keywords"]
    path = _write(tmp_path, "legal", data)
    manifest, errors = ar.load_manifest(path)
    assert errors == []
    assert manifest is not None


# ── build_agent() ────────────────────────────────────────────────────────────

class _StubRouter:
    def has_text_fallback(self):
        return False


def test_build_agent_produces_a_real_base_agent():
    manifest = _valid_manifest()
    agent = ar.build_agent(manifest, _StubRouter(), _StubBus())
    assert isinstance(agent, BaseAgent)
    assert agent.name == "legal"
    assert agent.title == "Legal Advisor Agent"
    assert agent.expertise == manifest["expertise"]
    assert agent.persona == manifest["persona"]


def test_build_agent_scoring_uses_the_manifests_regex_signals():
    manifest = _valid_manifest()
    agent = ar.build_agent(manifest, _StubRouter(), _StubBus())
    assert agent.score("I need help with a legal contract") > agent.score("what's the weather")


def test_build_agent_with_no_routing_signals_still_constructs():
    manifest = _valid_manifest()
    manifest["primary"] = []
    manifest["keywords"] = []
    agent = ar.build_agent(manifest, _StubRouter(), _StubBus())
    assert agent.score("anything at all") == 0


# ── load_all() ───────────────────────────────────────────────────────────────

def test_load_all_returns_empty_when_directory_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path / "does_not_exist")
    assert ar.load_all() == []


def test_load_all_loads_every_valid_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write(tmp_path, "legal", _valid_manifest("legal"))
    _write(tmp_path, "medical", _valid_manifest("medical"))
    manifests = ar.load_all()
    assert {m["name"] for m in manifests} == {"legal", "medical"}


def test_load_all_skips_an_invalid_manifest_and_logs_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write(tmp_path, "legal", _valid_manifest("legal"))
    _write(tmp_path, "broken", {"name": "broken"})   # missing required fields
    bus = _StubBus()
    manifests = ar.load_all(bus)
    assert [m["name"] for m in manifests] == ["legal"]
    assert any("broken" in msg for msg in bus.log.messages)


def test_load_all_with_no_bus_never_raises_on_an_invalid_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write(tmp_path, "broken", {"name": "broken"})
    assert ar.load_all(bus=None) == []


def test_load_all_ignores_non_json_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "AGENTS_DIR", tmp_path)
    _write(tmp_path, "legal", _valid_manifest("legal"))
    (tmp_path / "readme.md").write_text("not a manifest", encoding="utf-8")
    manifests = ar.load_all()
    assert [m["name"] for m in manifests] == ["legal"]
