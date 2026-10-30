"""Regression tests for isolation, repair validation and live activation."""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from orion_core.dynamic_loader import ModuleLoadOutcome, ReflectiveModuleLoader
from orion_core.forge import ForgeOrchestrationManager
from orion_core.sandbox import SandboxVerificationHarness, VerificationOutcome


class Signal:
    def emit(self, *args):
        pass


BUS = SimpleNamespace(log=Signal())
SOURCE = """def get_tool_schema():
    return {'name': 'demo', 'description': 'demo',
            'parameters': {'type': 'object', 'properties': {}}}
def run(**kwargs):
    return 'ok'
"""


def manager(tmp_path, monkeypatch):
    import orion_core.forge as forge
    monkeypatch.setattr(forge, "_custom_tools_dir", lambda: tmp_path)
    obj = ForgeOrchestrationManager(BUS, code_generator=lambda *_: (SOURCE, "assert True", []))
    return obj


async def test_repaired_source_is_guarded_before_execution(tmp_path, monkeypatch):
    obj = manager(tmp_path, monkeypatch)
    seen = []

    async def verify(source, *_):
        seen.append(source)
        return VerificationOutcome(False, "demo", error_log=["ValueError: bad module"])

    obj.sandbox.verify_tool = verify
    obj.llm_fixer = lambda *_: "import orion_core.security\n" + SOURCE
    result = await obj.forge_tool("demo", "demo")
    assert not result.ok
    assert len(seen) == 1, "a repair must not bypass the initial security check"


async def test_activation_failure_preserves_previous_file(tmp_path, monkeypatch):
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / "demo_tool.py"
    old = SOURCE.replace("'ok'", "'old'").encode()
    path.write_bytes(old)

    async def verify(*_):
        return VerificationOutcome(True, "demo")

    async def resolve(*_):
        return SimpleNamespace(succeeded=True, installed_packages=[])

    async def fail_load(*_):
        return ModuleLoadOutcome(False, "demo_tool", error_log=["import failed"])

    obj.sandbox.verify_tool = verify
    obj.resolver.resolve_and_install = resolve
    obj.loader.load_and_register = fail_load
    result = await obj.forge_tool("demo", "replace demo")
    assert not result.ok
    assert path.read_bytes() == old


async def test_handler_and_schema_publish_on_event_loop(tmp_path, monkeypatch):
    loader = ReflectiveModuleLoader(BUS)
    owner = threading.get_ident()
    dispatcher = SimpleNamespace(_tool_handlers={}, TOOL_DECLARATIONS=[])
    observed = []
    original = loader._register_handler

    def register(*args):
        observed.append(threading.get_ident())
        original(*args)

    monkeypatch.setattr(loader, "_register_handler", register)
    schema = {"name": "demo", "description": "Return the new value",
              "parameters": {"type": "object", "properties": {}}}
    outcome = ModuleLoadOutcome(True, "demo_tool", tool_schema=schema, handler_fn=lambda: "new")
    assert (await loader.activate_tool(outcome, dispatcher)).ok
    assert observed == [owner]
    assert dispatcher._tool_handlers["demo"]() == "new"
    assert dispatcher.TOOL_DECLARATIONS == [schema]


async def test_activation_restores_registry_when_schema_publish_fails(monkeypatch):
    loader = ReflectiveModuleLoader(BUS)
    old = lambda: "old"
    schema = {"name": "demo", "description": "old"}
    dispatcher = SimpleNamespace(_tool_handlers={"demo": old}, TOOL_DECLARATIONS=[schema])

    def broken(*_):
        raise ValueError("bad declaration")

    monkeypatch.setattr(loader, "_register_declaration", broken)
    outcome = ModuleLoadOutcome(True, "demo_tool", tool_schema={"name": "demo", "description": "New value",
                                "parameters": {"type": "object", "properties": {}}},
                                handler_fn=lambda: "new")
    assert not (await loader.activate_tool(outcome, dispatcher)).ok
    assert dispatcher._tool_handlers["demo"] is old
    assert dispatcher.TOOL_DECLARATIONS == [schema]


def test_failed_import_restores_previous_module(tmp_path):
    loader = ReflectiveModuleLoader(BUS)
    path = tmp_path / "transaction_demo_tool.py"
    previous = SimpleNamespace(marker="stable")
    sys.modules[path.stem] = previous
    path.write_text("raise RuntimeError('broken')", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError):
            loader._load_module_file(path)
        assert sys.modules[path.stem] is previous
    finally:
        sys.modules.pop(path.stem, None)


async def test_concurrent_verifications_have_private_directories(tmp_path):
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    async def check(value):
        code = SOURCE.replace("'ok'", repr(value))
        test = "import time, demo_tool\ntime.sleep(0.15)\nassert demo_tool.run() == " + repr(value)
        return await harness.verify_tool(code, test, "demo")
    results = await asyncio.gather(check("first"), check("second"))
    assert all(r.passed for r in results), [r.error_log for r in results]
    assert list(tmp_path.iterdir()) == [], "attempt artefacts must be removed after verification"


async def test_pytest_functions_actually_execute(tmp_path):
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    result = await harness.verify_tool(SOURCE, "def test_failure():\n    assert False\n", "demo")
    assert not result.passed, "running a test definition as a plain script executes no assertions"
    assert "AssertionError" in result.stdout


async def test_syntax_error_prevents_import_side_effects(tmp_path):
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    result = await harness.verify_tool(SOURCE, "def broken(:", "demo")
    assert not result.passed
    assert result.stage == "syntax"


async def test_cancellation_stops_child_before_cleaning_stage(tmp_path):
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    started = tmp_path / "started"
    script = f"from pathlib import Path\nimport time\nPath({str(started)!r}).touch()\ntime.sleep(30)\n"
    task = asyncio.create_task(harness.verify_tool(SOURCE, script, "demo"))
    for _ in range(200):
        if started.exists():
            break
        await asyncio.sleep(0.025)
    assert started.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    started.unlink()
    assert not list(tmp_path.iterdir())


def test_same_size_fast_reload_ignores_bytecode_cache(tmp_path):
    loader = ReflectiveModuleLoader(BUS)
    path = tmp_path / "fresh_tool.py"
    path.write_text("value = 'one'\n")
    stamp = path.stat().st_mtime
    import py_compile
    py_compile.compile(str(path))
    path.write_text("value = 'two'\n")
    os.utime(path, (stamp, stamp))
    try:
        assert loader._load_module_file(path).value == "two"
    finally:
        sys.modules.pop(path.stem, None)


async def test_runtime_code_fault_restores_previous_generation(tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.forge import ForgeSession
    obj = manager(tmp_path, monkeypatch)
    dispatcher = OrionDispatcher.__new__(OrionDispatcher)
    dispatcher.bus = BUS
    dispatcher._tool_handlers = {"demo": lambda **_: "stable"}
    dispatcher.TOOL_DECLARATIONS = [{"name": "demo", "description": "old"}]
    obj.dispatcher = dispatcher
    path = tmp_path / "demo_tool.py"
    path.write_text(SOURCE)
    broken = SOURCE.replace("return 'ok'", "raise NameError('misspelt local')")
    session = ForgeSession("runtime_test", "demo", "repair demo", tool_code=broken)
    await obj._activate_verified(session, path)
    result = await dispatcher.dispatch("demo", {})
    assert not result.ok and "previous tool generation restored" in result.text
    assert dispatcher._tool_handlers["demo"]() == "stable"
    assert path.read_text() == SOURCE
    assert dispatcher.TOOL_DECLARATIONS[0]["description"] == "old"


async def test_inflight_request_finishes_using_previous_callable():
    loader = ReflectiveModuleLoader(BUS)
    started, release = asyncio.Event(), asyncio.Event()
    async def old():
        started.set()
        await release.wait()
        return "old completed"
    dispatcher = SimpleNamespace(_tool_handlers={"demo": old}, TOOL_DECLARATIONS=[])
    running = asyncio.create_task(dispatcher._tool_handlers["demo"]())
    await started.wait()
    outcome = ModuleLoadOutcome(True, "demo_tool", tool_schema={"name": "demo", "description": "New value",
                                "parameters": {"type": "object", "properties": {}}}, handler_fn=lambda: "new")
    assert (await loader.activate_tool(outcome, dispatcher)).ok
    release.set()
    assert await running == "old completed"
    assert dispatcher._tool_handlers["demo"]() == "new"


async def test_synchronous_generator_runs_off_the_event_loop(tmp_path, monkeypatch):
    obj = manager(tmp_path, monkeypatch)
    owner = threading.get_ident()
    assert await obj._call(lambda: threading.get_ident()) != owner


async def test_free_autofixes_have_a_total_attempt_bound(tmp_path, monkeypatch):
    import orion_core.forge as forge
    obj = manager(tmp_path, monkeypatch)
    obj.max_attempts = 2
    calls = []
    async def verify(*args):
        calls.append(1)
        return VerificationOutcome(False, "demo", error_log=["CONTRACT: run rejects an argument"])
    obj.sandbox.verify_tool = verify
    monkeypatch.setattr(forge, "autofix_contract", lambda *args: SimpleNamespace(
        changed=True, complete=True, code=SOURCE, summary=lambda: "a repeated ineffective fix"))
    result = await obj.forge_tool("demo", "demo")
    assert not result.ok
    assert len(calls) == obj.max_attempts + forge.MAX_DEPENDENCY_REPAIRS + 5


@pytest.mark.parametrize("same_source", [False, True])
async def test_stale_activation_failure_preserves_newer_registration_module_and_source(tmp_path, monkeypatch, same_source):
    from orion_core.forge import ForgeSession
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / "demo_tool.py"
    path.write_bytes(SOURCE.encode())
    old_handler = lambda: "stable"
    old_schema = {"name": "demo", "description": "stable"}
    dispatcher = SimpleNamespace(_tool_handlers={"demo": old_handler}, TOOL_DECLARATIONS=[old_schema])
    obj.dispatcher = dispatcher
    attempted_handler = lambda: "attempted"
    schema = {"name": "demo", "description": "attempted",
              "parameters": {"type": "object", "properties": {}}}
    attempted_module = SimpleNamespace(run=attempted_handler)
    newer_handler = lambda: "newer"
    newer_schema = dict(schema, description="newer")
    newer_module = SimpleNamespace(run=newer_handler)
    newer_loaded = {"module": newer_module, "handler": newer_handler, "schema": newer_schema}
    newer_source = SOURCE.replace("'ok'", "'attempted'" if same_source else "'newer'").encode()
    history = []
    obj.history = SimpleNamespace(record=lambda **entry: history.append(entry))
    async def load(_path):
        sys.modules[path.stem] = attempted_module
        obj.loader.loaded_modules[path.stem] = {"module": attempted_module, "handler": attempted_handler, "schema": schema}
        return ModuleLoadOutcome(True, path.stem, tool_schema=schema, handler_fn=attempted_handler)
    async def superseded(_outcome, target):
        target._tool_handlers["demo"] = newer_handler
        target.TOOL_DECLARATIONS[:] = [newer_schema]
        sys.modules[path.stem] = newer_module
        obj.loader.loaded_modules[path.stem] = newer_loaded
        path.write_bytes(newer_source)
        from orion_core.data import ToolResult
        return ToolResult("Superseded by a later generation", ok=False)
    monkeypatch.setattr(obj.loader, "load_and_register", load)
    monkeypatch.setattr(obj.loader, "activate_tool", superseded)
    session = ForgeSession("stale", "demo", "new value", tool_code=SOURCE.replace("'ok'", "'attempted'"))
    previous = sys.modules.get(path.stem)
    try:
        with pytest.raises(RuntimeError, match="Superseded"):
            await obj._activate_verified(session, path)
        assert dispatcher._tool_handlers["demo"] is newer_handler
        assert dispatcher.TOOL_DECLARATIONS == [newer_schema]
        assert sys.modules[path.stem] is newer_module
        assert obj.loader.loaded_modules[path.stem] is newer_loaded
        assert path.read_bytes() == newer_source
        assert history[-1]["verification"]["restored"] is False
        assert history[-1]["status"] == "preserved_newer"
    finally:
        if previous is None:
            sys.modules.pop(path.stem, None)
        else:
            sys.modules[path.stem] = previous


async def test_failed_activation_preserves_manual_source_edit(tmp_path, monkeypatch):
    from orion_core.forge import ForgeSession
    from orion_core.data import ToolResult
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / "demo_tool.py"
    path.write_bytes(SOURCE.encode())
    current = SOURCE.encode() + b"# later external edit\n"
    obj.dispatcher = SimpleNamespace(_tool_handlers={}, TOOL_DECLARATIONS=[])
    async def reject(_outcome, _dispatcher):
        path.write_bytes(current)
        return ToolResult("Synthetic activation failure", ok=False)
    monkeypatch.setattr(obj.loader, "activate_tool", reject)
    session = ForgeSession("edited", "demo", "new value", tool_code=SOURCE.replace("'ok'", "'candidate'"))
    with pytest.raises(RuntimeError, match="Synthetic activation"):
        await obj._activate_verified(session, path)
    assert path.read_bytes() == current


async def test_runtime_rollback_preserves_replaced_schema_even_with_same_handler(tmp_path, monkeypatch):
    from orion_core.forge import ForgeSession
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / "demo_tool.py"
    path.write_bytes(SOURCE.encode())
    dispatcher = SimpleNamespace(_tool_handlers={}, TOOL_DECLARATIONS=[])
    obj.dispatcher = dispatcher
    session = ForgeSession("schema_race", "demo", "new value", tool_code=SOURCE.replace("'ok'", "'candidate'"))
    await obj._activate_verified(session, path)
    handler = dispatcher._tool_handlers["demo"]
    newer = dict(dispatcher.TOOL_DECLARATIONS[0], description="Externally refined declaration")
    dispatcher.TOOL_DECLARATIONS[:] = [newer]
    assert not await dispatcher._forge_rollbacks["demo"](handler, "Synthetic runtime fault")
    assert dispatcher._tool_handlers["demo"] is handler
    assert dispatcher.TOOL_DECLARATIONS == [newer]
    assert path.read_bytes() == session.tool_code.encode()


@pytest.mark.parametrize("candidate", [
    "def get_tool_schema():\n    return {'name': 'demo', 'description': 'invalid'}\ndef run(): return 'bad'\n",
    "def run(): return 'missing schema export'\n",
])
async def test_failed_real_loader_contract_restores_the_previous_generation(tmp_path, monkeypatch, candidate):
    from orion_core.forge import ForgeSession
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / "demo_tool.py"
    path.write_bytes(SOURCE.encode())
    stable_handler = lambda: "stable"
    stable_module = SimpleNamespace(run=stable_handler)
    stable_schema = {"name": "demo", "description": "stable"}
    stable_loaded = {"module": stable_module, "handler": stable_handler, "schema": stable_schema}
    obj.loader.loaded_modules[path.stem] = stable_loaded
    obj.dispatcher = SimpleNamespace(_tool_handlers={"demo": stable_handler}, TOOL_DECLARATIONS=[stable_schema])
    previous = sys.modules.get(path.stem)
    sys.modules[path.stem] = stable_module
    try:
        session = ForgeSession("invalid_contract", "demo", "new value", tool_code=candidate)
        with pytest.raises(RuntimeError, match="Failed to load module"):
            await obj._activate_verified(session, path)
        assert path.read_bytes() == SOURCE.encode()
        assert sys.modules[path.stem] is stable_module
        assert obj.loader.loaded_modules[path.stem] is stable_loaded
        assert obj.dispatcher._tool_handlers["demo"] is stable_handler
    finally:
        if previous is None:
            sys.modules.pop(path.stem, None)
        else:
            sys.modules[path.stem] = previous


async def test_failed_load_cannot_undo_a_route_published_during_import(tmp_path, monkeypatch):
    from orion_core.forge import ForgeSession
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / "demo_tool.py"
    path.write_bytes(SOURCE.encode())
    obj.dispatcher = SimpleNamespace(_tool_handlers={}, TOOL_DECLARATIONS=[])
    candidate = SOURCE.replace("'ok'", "'candidate'")
    later_handler = lambda: "a later route"
    later_schema = {"name": "demo", "description": "Later route"}
    imported = SimpleNamespace(run=lambda: "candidate")
    previous = sys.modules.get(path.stem)
    async def failed_load(_path):
        sys.modules[path.stem] = imported
        obj.dispatcher._tool_handlers["demo"] = later_handler
        obj.dispatcher.TOOL_DECLARATIONS[:] = [later_schema]
        return ModuleLoadOutcome(False, path.stem, error_log=["synthetic contract failure"], module_obj=imported)
    monkeypatch.setattr(obj.loader, "load_and_register", failed_load)
    try:
        session = ForgeSession("failed_import", "demo", "new value", tool_code=candidate)
        with pytest.raises(RuntimeError, match="Failed to load module"):
            await obj._activate_verified(session, path)
        assert obj.dispatcher._tool_handlers["demo"] is later_handler
        assert obj.dispatcher.TOOL_DECLARATIONS == [later_schema]
        assert path.read_bytes() == candidate.encode()
        assert sys.modules[path.stem] is imported
    finally:
        if previous is None:
            sys.modules.pop(path.stem, None)
        else:
            sys.modules[path.stem] = previous


async def test_rejected_schema_name_preserves_requested_tool_recovery_identity(tmp_path, monkeypatch):
    from orion_core.forge import ForgeSession
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / "demo_tool.py"
    path.write_bytes(SOURCE.encode())
    stable = lambda: "stable"
    obj.dispatcher = SimpleNamespace(_tool_handlers={"demo": stable},
                                    TOOL_DECLARATIONS=[{"name": "demo", "description": "stable"}])
    candidate = SOURCE.replace("'name': 'demo'", "'name': 'other'")
    session = ForgeSession("wrong_name", "demo", "new value", tool_code=candidate)
    with pytest.raises(ValueError, match="Verified schema name"):
        await obj._activate_verified(session, path)
    assert path.read_bytes() == SOURCE.encode()
    assert obj.dispatcher._tool_handlers == {"demo": stable}
    assert obj.dispatcher.TOOL_DECLARATIONS == [{"name": "demo", "description": "stable"}]
