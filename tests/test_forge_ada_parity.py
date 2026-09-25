"""
Tests for the ADA-parity self-improvement upgrades to the Forge.

* parse_missing_module turns import errors into installable package names.
* A sandbox failure naming a missing module triggers a dependency install
  and re-verify WITHOUT consuming an LLM code-repair attempt.
* reload_persisted_tools brings previously forged tools back to life at
  startup, containing per-tool failures.
* The improvement heartbeat consolidates 'note' actions into the durable
  knowledge tier (deduplicated, failure-contained).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.forge import (
    ForgeOrchestrationManager,
    ImprovementHeartbeat,
    parse_missing_module,
)


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


def test_parse_missing_module_variants():
    assert parse_missing_module("ModuleNotFoundError: No module named 'requests'") == "requests"
    assert parse_missing_module('No module named "bs4.element"') == "bs4"
    assert parse_missing_module("ModuleNotFoundError: No module named yaml") == "yaml"
    assert parse_missing_module("SyntaxError: invalid syntax") is None
    assert parse_missing_module("") is None


class _Outcome:
    def __init__(self, passed, errors=()):
        self.passed = passed
        self.error_log = list(errors)


class _DepOutcome:
    def __init__(self, ok):
        self.succeeded = ok
        self.installed_packages = ["x"] if ok else []
        self.error_log = [] if ok else ["pip failed"]


class _LoadOutcome:
    def __init__(self, ok):
        self.succeeded = ok
        self.error_log = [] if ok else ["boom"]


# A realistic tool body — exports the required entry points and parses as
# Python — so the forge's placeholder/ast guard accepts it.  (A bare "code"
# stub used to be persisted here and crashed the loader on every boot.)
_VALID_TOOL_CODE = (
    "def get_tool_schema():\n"
    "    return {'name': 'demo', 'description': 'demo tool', 'parameters': {}}\n\n"
    "def run(**kwargs):\n"
    "    return kwargs\n"
)


def _forge(sandbox_results, dep_ok=True):
    forge = ForgeOrchestrationManager(
        _StubBus(),
        code_generator=lambda name, plan: (_VALID_TOOL_CODE, "def test_demo():\n    assert True\n", []),
    )
    results = list(sandbox_results)

    class _Sandbox:
        async def verify_tool(self, code, test, name):
            return results.pop(0)

    class _Resolver:
        def __init__(self):
            self.installed = []

        async def resolve_and_install(self, requirements):
            self.installed.append(list(requirements))
            return _DepOutcome(dep_ok)

    class _Loader:
        async def load_and_register(self, path):
            return _LoadOutcome(True)

        async def activate_tool(self, outcome, dispatcher):
            from orion_core.data import ToolResult
            return ToolResult("active", ok=True)

    forge.sandbox = _Sandbox()
    forge.resolver = _Resolver()
    forge.loader = _Loader()
    forge.dispatcher = object()
    return forge


def test_missing_module_repair_does_not_consume_code_attempt(tmp_path, monkeypatch):
    # 1st verify: fails on a missing module → install + retry (free).
    # 2nd verify: passes.  No LLM fixer configured, and none was needed.
    forge = _forge([
        _Outcome(False, ["ModuleNotFoundError: No module named 'httpx'"]),
        _Outcome(True),
    ])
    # Redirect the tool write into the temp tree so the test never plants a
    # file in the real config/custom_tools/ (that leak once corrupted net_probe).
    import orion_core.forge as forge_mod
    (tmp_path / "orion_core").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(forge_mod, "CONFIG_DIR", tmp_path / "config")
    result = asyncio.run(forge.forge_tool("net_probe", "probe a url"))
    assert result.ok, result.text
    session = list(forge.sessions.values())[0]
    assert session.sandbox_ok
    # First install is the sandbox repair; Phase 4 then re-resolves the full
    # requirement list (which now carries the discovered module).
    assert forge.resolver.installed[0] == ["httpx"]
    assert "httpx" in session.requirements


def test_repeated_missing_module_is_not_reinstalled():
    # The same module failing twice falls through to normal failure handling
    # instead of an install loop.
    forge = _forge([
        _Outcome(False, ["No module named 'ghost'"]),
        _Outcome(False, ["No module named 'ghost'"]),
        _Outcome(False, ["No module named 'ghost'"]),
        _Outcome(False, ["No module named 'ghost'"]),
    ])
    result = asyncio.run(forge.forge_tool("haunted", "always missing"))
    assert not result.ok
    assert forge.resolver.installed == [["ghost"]]     # exactly one attempt


def test_reload_persisted_tools_counts_and_contains_failures(tmp_path, monkeypatch):
    forge = _forge([])
    tools_dir = tmp_path / "config" / "custom_tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "alpha_tool.py").write_text("x = 1", encoding="utf-8")
    (tools_dir / "beta_tool.py").write_text("x = 2", encoding="utf-8")
    (tools_dir / "not_a_tool.txt").write_text("ignored", encoding="utf-8")

    class _FlakyLoader:
        def __init__(self):
            self.calls = 0

        async def load_and_register(self, path):
            self.calls += 1
            return _LoadOutcome(self.calls > 1)    # first tool fails to load

        async def activate_tool(self, outcome, dispatcher):
            from orion_core.data import ToolResult
            return ToolResult("active", ok=True)

    forge.loader = _FlakyLoader()
    import orion_core.forge as forge_mod
    monkeypatch.setattr(forge_mod, "CONFIG_DIR", tmp_path / "config")
    reloaded = asyncio.run(forge.reload_persisted_tools())
    assert reloaded == 1                            # beta survived, alpha logged
    assert forge.loader.calls == 2                  # .txt never touched


def test_reload_without_dispatcher_is_a_noop():
    forge = _forge([])
    forge.dispatcher = None
    assert asyncio.run(forge.reload_persisted_tools()) == 0


class _Memory:
    def __init__(self):
        self.saved = {}

    def remember_knowledge(self, key, value):
        self.saved[key] = value
        return key


class _NoteRouter:
    def text_available(self):
        return True

    async def generate_text(self, prompt, system_extra=""):
        payload = {
            "analysis": "Logs are clean; the deploy pipeline is the weak spot.",
            "actions": [
                {"type": "note", "text": "Deploy checks take too long."},
                {"type": "note", "text": ""},           # blank → ignored
            ],
        }
        return object(), json.dumps(payload)


def test_heartbeat_notes_consolidate_into_knowledge(tmp_path, monkeypatch):
    import orion_core.forge as forge_mod
    monkeypatch.setattr(forge_mod, "IMPROVE_JOURNAL", tmp_path / "journal.jsonl")
    memory = _Memory()
    heartbeat = ImprovementHeartbeat(
        _StubBus(), _NoteRouter(), _forge([]), memory=memory)
    entry = asyncio.run(heartbeat.tick())
    assert entry["analysis"].startswith("Logs are clean")
    assert len(memory.saved) == 1
    assert list(memory.saved.values())[0] == "Deploy checks take too long."
    # The same note next tick dedupes to the same key — no knowledge spam.
    asyncio.run(heartbeat.tick())
    assert len(memory.saved) == 1


def test_heartbeat_without_memory_still_ticks(tmp_path, monkeypatch):
    import orion_core.forge as forge_mod
    monkeypatch.setattr(forge_mod, "IMPROVE_JOURNAL", tmp_path / "journal.jsonl")
    heartbeat = ImprovementHeartbeat(_StubBus(), _NoteRouter(), _forge([]))
    entry = asyncio.run(heartbeat.tick())          # must not raise
    assert entry["actions"]


def test_reload_leaves_plugin_managed_tools_to_the_plugin_loader(tmp_path, monkeypatch):
    """A *_tool.py with a sibling .plugin.json belongs to load_plugins, which
    honours its enabled state. Reloading it here imported every plugin twice
    and brought DISABLED plugins back at each boot."""
    forge = _forge([])
    tools_dir = tmp_path / "config" / "custom_tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "forged_tool.py").write_text("x = 1", encoding="utf-8")
    (tools_dir / "weather_tool.py").write_text("x = 2", encoding="utf-8")
    (tools_dir / "weather.plugin.json").write_text("{}", encoding="utf-8")
    loaded: list[str] = []

    class _Loader:
        async def load_and_register(self, path):
            loaded.append(path.name)
            return _LoadOutcome(True)

        async def activate_tool(self, outcome, dispatcher):
            from orion_core.data import ToolResult
            return ToolResult("active", ok=True)

    forge.loader = _Loader()
    import orion_core.forge as forge_mod
    monkeypatch.setattr(forge_mod, "CONFIG_DIR", tmp_path / "config")
    asyncio.run(forge.reload_persisted_tools())
    assert loaded == ["forged_tool.py"]
