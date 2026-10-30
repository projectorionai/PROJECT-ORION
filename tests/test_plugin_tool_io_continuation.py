"""Model-driven registry management must not perform blocking IO on its loop."""
import threading
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.dispatch_productivity import ProductivityDispatchMixin


@pytest.mark.parametrize("action", ["list", "describe", "enable", "disable", "create", "doctor",
                                  "backfill", "export", "audit", "check_update"])
async def test_management_registry_work_leaves_the_dispatch_loop(action):
    owner = threading.get_ident()
    calls = []
    def operation(*_, **__):
        calls.append(threading.get_ident())
        return ToolResult("synthetic receipt")
    def manifests():
        calls.append(threading.get_ident())
        return []
    registry = SimpleNamespace(list_plugins=operation, describe=operation, set_enabled=operation,
                               create=operation, doctor=operation, backfill_manifests=manifests,
                               manifestless=manifests, export_bundle=operation, audit=operation,
                               check_update=operation)
    result = await ProductivityDispatchMixin.plugin_tool(SimpleNamespace(plugins=registry),
                                                        {"action": action, "name": "demo"})
    assert result.ok
    assert calls and owner not in calls
