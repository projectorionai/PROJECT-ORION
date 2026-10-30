"""Regression checks for the supplied plugin adapters' actual contracts."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.jv_plugins import runtime
from orion_core.jv_plugins.catalogue import names
from orion_core.jv_plugins.runner import execute


class Signal:
    def __init__(self):
        self.items = []

    def emit(self, *args):
        self.items.append(args)


def async_test(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        async def invoke():
            runtime.context().loop = asyncio.get_running_loop()
            try:
                return await function(*args, **kwargs)
            finally:
                await runtime.shutdown()
        return asyncio.run(invoke())
    return wrapped


@pytest.fixture
def services(tmp_path, monkeypatch):
    from orion_core.jv_plugins import runner, telegram
    runner._locks.clear()
    telegram._inbox_lock = asyncio.Lock()
    calls = []

    class Dispatcher:
        async def dispatch(self, name, arguments):
            calls.append((name, arguments))
            return ToolResult("Native action completed.")

    bus = SimpleNamespace(dashboard_event=Signal(), speak_request=Signal(), log=Signal())
    runtime.configure(bus=bus, router=None, dispatcher=Dispatcher(), directory=tmp_path,
                      loop=None)
    monkeypatch.setattr(runtime, "secret", lambda *_: "")
    yield SimpleNamespace(bus=bus, calls=calls, directory=tmp_path)
    assert not runtime._tasks


def test_discovery_has_no_device_process_network_or_scheduler_side_effects(tmp_path, monkeypatch):
    import subprocess
    import urllib.request

    from orion_core.forge_contract import schema_problems, signature_problems
    from orion_core.plugin_manifest import PluginManifest
    from orion_core.plugin_registry import validate_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Discovery attempted an external action")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    before = set(runtime._tasks)
    for name in names():
        path = Path("config/custom_tools") / (name + "_tool.py")
        assert not validate_module(path)
        spec = importlib.util.spec_from_file_location(name + "_audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema = module.get_tool_schema()
        problems, properties = schema_problems(schema)
        assert not problems, (name, problems)
        assert not signature_problems(module.run, properties)
        metadata = PluginManifest.from_dict(json.loads(path.with_name(name + ".plugin.json").read_text()))
        assert metadata.name == schema["name"]
        assert metadata.parameters == schema["parameters"]
        assert not metadata.is_scheduled
    assert set(runtime._tasks) == before


def test_free_form_workbook_and_cloud_arguments_are_gemini_compatible():
    from orion_core.gemini_schema import to_gemini_parameters
    from orion_core.jv_plugins.catalogue import schema
    for name, field in (("jv_excel_writer", "workbook"), ("jv_cloud_agent", "arguments")):
        converted = to_gemini_parameters(schema(name)["parameters"])
        assert converted["properties"][field]["type"] == "STRING"


@async_test
async def test_quiz_keeps_reference_hidden_and_handles_choice_labels_and_open_judgement(services):
    result = await execute("jv_quiz", {"action": "start", "topic": "Neuroscience", "questions": [
        {"type": "multiple_choice", "question": "Which ion?", "options": ["Sodium", "Potassium"], "answer": "B"},
        {"type": "open", "question": "Explain it", "answer": "Reference explanation"}]})
    assert result.ok and "Question 1/2" in result.text
    assert "answer" not in services.bus.dashboard_event.items[-1][1]["question"]
    await execute("jv_quiz", {"action": "answer", "answer": "2"})
    result = await execute("jv_quiz", {"action": "answer", "answer": "An alternative valid explanation"})
    assert "1/2 correct; 1 awaiting judgement" in result.text
    result = await execute("jv_quiz", {"action": "grade", "index": 2, "correct": True})
    assert "2/2 correct; 0 awaiting judgement" in result.text
    assert json.loads((services.directory / "quiz.json").read_text())["index"] == 2


@async_test
async def test_bad_quiz_does_not_replace_existing_session(services):
    await execute("jv_quiz", {"action": "start", "questions": [{"question": "First", "answer": "One"}]})
    bad = await execute("jv_quiz", {"action": "start", "questions": [{"type": "multiple_choice", "question": "Bad", "options": ["A", "B"], "answer": "Z"}]})
    assert not bad.ok
    assert "First" in (await execute("jv_quiz", {"action": "status"})).text


@async_test
async def test_hydration_persists_and_duplicate_start_leaves_one_worker(services):
    await execute("jv_water_reminder", {"action": "log", "amount_ml": 350})
    for _ in range(2):
        assert (await execute("jv_water_reminder", {"action": "start", "goal_ml": 2000})).ok
    assert len([task for task in runtime._tasks if task.get_name() == "orion-jv-water"]) == 1
    assert "350/2000" in (await execute("jv_water_reminder", {"action": "status"})).text
    await execute("jv_water_reminder", {"action": "stop"})
    assert not [task for task in runtime._tasks if task.get_name() == "orion-jv-water"]


@async_test
async def test_pomodoro_completion_and_break_follow_monotonic_deadline(services, monkeypatch):
    from orion_core.jv_plugins import wellbeing
    original_sleep = asyncio.sleep
    triggered = asyncio.Event()

    async def fast_sleep(seconds):
        if not triggered.is_set():
            triggered.set()
            return
        await original_sleep(3600)

    await execute("jv_pomodoro", {"action": "start", "work_minutes": 1, "break_minutes": 1})
    monkeypatch.setattr(wellbeing.asyncio, "sleep", fast_sleep)
    await asyncio.wait_for(triggered.wait(), 1)
    for _ in range(30):
        if wellbeing._focus.get("phase") == "break":
            break
        await original_sleep(0.01)
    assert wellbeing._focus["phase"] == "break"
    assert wellbeing._focus["completed"] == 1
    assert (await execute("jv_pomodoro", {"action": "stats"})).text.endswith("today: 1.")
    await execute("jv_pomodoro", {"action": "stop"})


@async_test
async def test_workbook_real_file_chart_formula_and_injection_protection(services):
    from openpyxl import load_workbook
    result = await execute("jv_excel_writer", {"filename": "../Audit", "workbook": json.dumps({"sheets": [{
        "name": "Accounts", "columns": ["=HYPERLINK(\"bad\")", "Amount"],
        "rows": [["April", 12], ["May", 15], ["=WEBSERVICE(\"bad\")", {"formula": "=SUM(B2:B3)"}]],
        "chart": {"type": "bar", "title": "Monthly totals"}}]})})
    assert result.ok, result.text
    output = services.directory / "exports" / "Audit.xlsx"
    book = load_workbook(output)
    try:
        sheet = book["Accounts"]
        assert sheet["A1"].data_type == "s"
        assert sheet["A4"].data_type == "s"
        assert sheet["B4"].data_type == "f" and sheet["B4"].value == "=SUM(B2:B3)"
        assert len(sheet._charts) == 1
    finally:
        book.close()
    duplicate = await execute("jv_excel_writer", {"filename": "Audit", "workbook": {"sheets": [{"columns": ["A"], "rows": [[1]]}]}})
    assert not duplicate.ok and "already exists" in duplicate.text


@async_test
async def test_workbook_rejects_external_formula_without_partial_output(services):
    result = await execute("jv_excel_writer", {"filename": "Unsafe", "workbook": {"sheets": [{"columns": ["A"], "rows": [[{"formula": '=WEBSERVICE("https://example.invalid")'}]]}]}})
    assert not result.ok
    assert not (services.directory / "exports" / "Unsafe.xlsx").exists()


@async_test
async def test_telegram_ignores_other_chats_and_consumes_approval_once(services):
    from orion_core.jv_plugins.telegram import _request_from_update
    message = {"message": {"chat": {"id": 42, "type": "private"},
                           "text": '/tool {"name":"sentinel","arguments":{"action":"status"}}'}}
    assert _request_from_update(message, {"12"}) is None
    request = _request_from_update(message, {"42"})
    assert request is not None
    message["message"]["chat"]["type"] = "group"
    assert _request_from_update(message, {"42"}) is None
    runtime.save("telegram_inbox", {"requests": [request]})
    assert (await execute("jv_telegram_remote", {"action": "approve", "id": request["id"]})).ok
    assert not (await execute("jv_telegram_remote", {"action": "approve", "id": request["id"]})).ok
    assert services.calls == [("sentinel", {"action": "status"})]


@async_test
async def test_unconfigured_network_adapters_fail_honestly_without_starting(services):
    for name, arguments in (("jv_cloud_agent", {"action": "status"}),
                            ("jv_telegram_remote", {"action": "start"}),
                            ("jv_calorie_counter", {}), ("jv_excel_writer", {"request": "Invent a workbook"})):
        assert not (await execute(name, arguments)).ok
    assert not runtime._tasks
    assert not services.calls


@async_test
async def test_cloud_delegation_uses_approved_remote_contract(services, monkeypatch):
    from orion_core.jv_plugins import integrations
    runtime.save("cloud_agent", {"url": "https://orion.example", "token": "test-token"})
    calls = []

    async def fake_request(settings, method, path, data=None):
        calls.append((method, path, data))
        return {"ok": True, "status": "done", "id": "job1"}

    monkeypatch.setattr(integrations, "_cloud_request", fake_request)
    result = await execute("jv_cloud_agent", {"action": "delegate", "tool": "web_search", "arguments": {"query": "test"}})
    assert result.ok
    assert calls == [("POST", "/v1/agent/tasks", {"tool": "web_search", "args": {"query": "test"}})]


@async_test
async def test_forge_trial_runs_once_and_preserves_explicit_rollback(services):
    ctx = runtime.context()
    invoked = []

    class Forge:
        async def forge_tool(self, name, plan):
            invoked.append((name, plan))
            return ToolResult("Verified.")

    ctx.dispatcher.forge = Forge()
    await execute("jv_forge_trial", {"action": "propose", "name": "summary", "plan": "Summarise text; check empty input and fact preservation."})
    state = runtime.load("forge_trials")
    identifier = next(iter(state["trials"]))
    assert (await execute("jv_forge_trial", {"action": "run", "id": identifier})).ok
    assert not (await execute("jv_forge_trial", {"action": "run", "id": identifier})).ok
    assert len(invoked) == 1 and "Never modify ORION's core" in invoked[0][1]


@async_test
async def test_native_reminder_bridge_retains_native_result_and_arguments(services):
    result = await execute("jv_reminder", {"action": "add", "text": "Take a break", "minutes": 15})
    assert result.ok and result.text == "Native action completed."
    assert services.calls == [("reminder", {"action": "add", "text": "Take a break", "minutes": 15})]
