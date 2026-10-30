"""The optional planner is attached to ORION's production composition path."""

from __future__ import annotations

import asyncio

from orion_core import app
from orion_core.data import ToolResult


class _Signal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, value: str) -> None:
        self.messages.append(value)


class _Bus:
    def __init__(self) -> None:
        self.log = _Signal()


class _Dispatcher:
    def __init__(self) -> None:
        self.TOOL_DECLARATIONS = [{
            "name": "autoplan", "description": "Run a supplied plan.",
            "parameters": {"type": "OBJECT", "properties": {}, "required": ["steps"]},
        }]
        self.original: list[dict] = []

    async def autoplan_tool(self, args: dict) -> ToolResult:
        self.original.append(args)
        return ToolResult("original")

    async def dispatch(self, _name: str, _args: dict) -> ToolResult:
        return ToolResult("dispatched")


def test_disabled_planner_leaves_autoplan_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("ORION_NEURAL_PLANNER", "0")
    monkeypatch.setattr(app, "CONFIG_DIR", tmp_path)
    dispatcher = _Dispatcher()
    before = dispatcher.autoplan_tool
    declarations = list(dispatcher.TOOL_DECLARATIONS)

    planner = app._install_optional_neural_planner(
        dispatcher, object(), object(), object(), _Bus(), object())

    assert planner is None
    assert dispatcher.autoplan_tool == before
    assert dispatcher.TOOL_DECLARATIONS == declarations
    assert "neural_planner" not in vars(dispatcher)
    assert not (tmp_path / "neural").exists()


def test_saved_preference_activates_unless_environment_forces_off(monkeypatch, tmp_path):
    from orion_core import neural_planner as npl

    monkeypatch.delenv("ORION_NEURAL_PLANNER", raising=False)
    monkeypatch.setattr(app, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(npl, "NEURAL_DIR", tmp_path / "neural")
    npl.PlannerSettings(enabled=True).save()

    dispatcher = _Dispatcher()
    planner = app._install_optional_neural_planner(
        dispatcher, object(), object(), object(), _Bus(), object())
    assert planner is not None
    assert dispatcher.neural_planner is planner

    monkeypatch.setenv("ORION_NEURAL_PLANNER", "0")
    other = _Dispatcher()
    assert app._install_optional_neural_planner(
        other, object(), object(), object(), _Bus(), object()) is None
    assert "neural_planner" not in vars(other)


def test_enabled_planner_routes_only_objective_calls(monkeypatch, tmp_path):
    from orion_core import neural_planner as npl

    monkeypatch.setenv("ORION_NEURAL_PLANNER", "1")
    monkeypatch.setattr(app, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(npl, "NEURAL_DIR", tmp_path / "neural")
    dispatcher = _Dispatcher()
    bus = _Bus()
    planner = app._install_optional_neural_planner(
        dispatcher, object(), object(), object(), bus, object())

    assert planner is not None
    assert dispatcher.neural_planner is planner
    assert "shadow" in bus.log.messages[0] and "read_only" in bus.log.messages[0]
    assert dispatcher.TOOL_DECLARATIONS[0]["parameters"]["required"] == []

    goals: list[str] = []

    async def propose(goal: str) -> ToolResult:
        goals.append(goal)
        return ToolResult("proposed")

    planner.run = propose
    result = asyncio.run(dispatcher.autoplan_tool({"objective": "check resources"}))
    assert result.text == "proposed"
    assert goals == ["check resources"]

    result = asyncio.run(dispatcher.autoplan_tool({
        "objective": "check resources", "steps": [{"tool": "resource_status"}],
    }))
    assert result.text == "original"
    assert len(dispatcher.original) == 1
    assert goals == ["check resources"]
