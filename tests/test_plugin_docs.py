"""
docs/PLUGINS.md must stay TRUE (Mark XXVI improvement #17).

The plugin contract was previously undocumented, which is why (per the registry's
own docstring) zero third-party manifests were ever written. A guide only helps if
it cannot drift from the code, so every factual claim it makes is checked here.
"""

from __future__ import annotations

import re
import ast
import inspect
import textwrap
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DOC = Path(__file__).resolve().parents[1] / "docs" / "PLUGINS.md"


@pytest.fixture(scope="module")
def doc() -> str:
    assert _DOC.is_file(), "docs/PLUGINS.md is missing"
    return _DOC.read_text(encoding="utf-8")


def test_every_documented_permission_exists(doc):
    from orion_core.plugin_manifest import PERMISSIONS
    for permission in PERMISSIONS:
        assert permission in doc, f"{permission} is undocumented"
    # and nothing invented
    documented = set(re.findall(r"`(network|filesystem|subprocess|input|screen|audio)`", doc))
    assert documented <= PERMISSIONS


def test_every_documented_event_is_subscribable(doc):
    from orion_core.plugin_events import SUBSCRIBABLE
    for event in SUBSCRIBABLE:
        assert event in doc, f"{event} is undocumented"
    claimed = set(re.findall(r"`(state|speaking|log|banner|safety_alert|"
                             r"connection_state|emotion_changed|telemetry_sample|"
                             r"dashboard_event|paused)`", doc))
    assert claimed <= SUBSCRIBABLE, "the guide names an event ORION does not offer"


def test_the_documented_slow_threshold_matches_the_code(doc):
    from orion_core.plugin_events import SLOW_MS
    assert f"**{int(SLOW_MS)} ms**" in doc, (
        "the guide's slow-hook budget has drifted from plugin_events.SLOW_MS")


def test_every_documented_tier_is_real(doc):
    from orion_core.plugin_manifest import CAPABILITY_TIERS
    for tier in CAPABILITY_TIERS:
        assert tier in doc, f"tier {tier} is undocumented"


def test_every_documented_command_is_a_real_action(doc):
    """The command table must not promise an action the tool does not route."""
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    # Inspect complete action branches: the valid deps/reload routes grew
    # beyond the old arbitrary 6,000-character source slice.
    handler = ast.parse(textwrap.dedent(inspect.getsource(ProductivityDispatchMixin.plugin_tool)))
    routed = set()
    for node in ast.walk(handler):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "action":
            for comparator in node.comparators:
                routed.update(item.value for item in ast.walk(comparator)
                              if isinstance(item, ast.Constant) and isinstance(item.value, str))
    documented = set(re.findall(r"\| `plugin ([a-z_]+)", doc))
    assert documented, "no commands found in the guide"
    for action in documented:
        assert action in routed, f"the guide documents '{action}', which is not routed"


@pytest.mark.parametrize("action", ["deps", "reload"])
async def test_documented_management_commands_reach_their_handlers(action):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    from orion_core.data import ToolResult

    expected = ToolResult("handled")
    deps = AsyncMock(return_value=expected)
    reload = AsyncMock(return_value=expected)
    dispatcher = SimpleNamespace(plugins=SimpleNamespace(install_dependencies=deps), _reload_plugins=reload)
    result = await ProductivityDispatchMixin.plugin_tool(dispatcher, {"action": action, "name": "example"})
    assert result is expected
    if action == "deps":
        deps.assert_awaited_once_with("example")
        reload.assert_not_awaited()
    else:
        reload.assert_awaited_once_with()
        deps.assert_not_awaited()


def test_the_documented_file_layout_matches_the_code(doc):
    from orion_core.plugin_registry import _TOOL_SUFFIX
    from orion_core.plugin_manifest import MANIFEST_SUFFIX
    assert _TOOL_SUFFIX in doc
    assert MANIFEST_SUFFIX in doc


def test_the_named_modules_all_exist(doc):
    root = Path(__file__).resolve().parents[1]
    for module in re.findall(r"`orion_core/([a-z_]+\.py)`", doc):
        assert (root / "orion_core" / module).is_file(), f"{module} does not exist"
