"""Desktop observations must stay bounded, current and free of side effects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion_core import mcp_desktop, mcp_host


async def request(name, arguments=None):
    return await mcp_desktop.respond({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                      "params": {"name": name, "arguments": arguments or {}}})


async def test_handshake_discovery_does_not_start_capture_or_input(monkeypatch):
    monkeypatch.setattr(mcp_desktop, "vision_service", lambda: pytest.fail("Discovery started desktop capture"))
    reply = await mcp_desktop.respond({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                       "params": {"protocolVersion": "2025-11-25"}})
    assert reply["result"]["protocolVersion"] == "2025-11-25"
    reply = await mcp_desktop.respond({"id": 2, "method": "tools/list"})
    tools = reply["result"]["tools"]
    assert len(tools) == 4
    assert all(tool["annotations"]["readOnlyHint"] for tool in tools)
    assert all(not tool["annotations"]["destructiveHint"] for tool in tools)
    assert not any("click" in tool["name"] for tool in tools)


@pytest.mark.parametrize("value", [None, [], {}, 99, True])
async def test_invalid_protocol_version_cannot_crash_stdio_server(value):
    reply = await mcp_desktop.respond({"id": 1, "method": "initialize", "params": {"protocolVersion": value}})
    assert reply["result"]["protocolVersion"] == "2025-11-25"


@pytest.mark.parametrize("name,args", [
    ("desktop_controls", {"limit": True}), ("desktop_controls", {"limit": 201}),
    ("desktop_controls", {"limit": 0}), ("desktop_controls", {"kinds": []}),
    ("desktop_find_target", {"query": " "}), ("desktop_find_target", {"query": "x" * 257}),
    ("desktop_find_target", {"query": "Save", "automation_id": []}),
    ("desktop_target_status", {"click": True}), ("desktop_controls", {"url": "https://example.com"}),
    ("desktop_click", {}),
])
async def test_bad_arguments_are_refused_before_any_desktop_read(monkeypatch, name, args):
    monkeypatch.setattr(mcp_desktop, "vision_service", lambda: pytest.fail("Invalid input reached the desktop"))
    reply = await request(name, args)
    assert reply["result"]["isError"]


async def test_unavailable_uia_is_explicit_instead_of_no_matching_controls(monkeypatch):
    monkeypatch.setattr(mcp_desktop, "vision_service", lambda: SimpleNamespace(_uia_available=lambda: False))
    result = (await request("desktop_controls"))["result"]
    assert result["isError"] and result["structuredContent"]["status"] == "unavailable"
    status = (await request("desktop_target_status"))["result"]
    assert status["structuredContent"]["available"] is False
    assert status["structuredContent"]["input_owner"] == "desktop_control"


async def test_control_list_is_bounded_keeps_negative_coordinates_and_removes_internal_data(monkeypatch):
    elements = [{"name": "Save", "role": "Button", "automation_id": f"save-{n}",
                 "rect": [-120, 20, 40, 20], "center": [-100, 30], "enabled": True,
                 "source_hwnd": 40, "private": "must-not-return"} for n in range(201)]
    calls = []

    def detect(kinds, max_elements):
        calls.append((kinds, max_elements))
        return elements

    service = SimpleNamespace(_uia_available=lambda: True, _foreground_hwnd=lambda: 40,
                              _detect_elements_sync=detect)
    monkeypatch.setattr(mcp_desktop, "vision_service", lambda: service)
    value = (await request("desktop_controls", {"limit": 2, "kinds": "button"}))["result"]["structuredContent"]
    assert calls == [("button", 201)]
    assert len(value["controls"]) == 2 and value["truncated"]
    assert value["controls"][0]["center"] == [-100, 30]
    assert "private" not in value["controls"][0] and "source_hwnd" not in value["controls"][0]
    assert value["coordinate_space"] == "physical_virtual_desktop"


async def test_control_list_discards_a_foreground_switch_during_enumeration(monkeypatch):
    handles = iter([40, 41])
    service = SimpleNamespace(_uia_available=lambda: True, _foreground_hwnd=lambda: next(handles),
                              _detect_elements_sync=lambda *_args, **_kwargs: [{"name": "Save"}])
    monkeypatch.setattr(mcp_desktop, "vision_service", lambda: service)
    result = (await request("desktop_controls"))["result"]
    assert result["isError"] and result["structuredContent"]["controls"] == []
    assert result["structuredContent"]["status"] == "stale"


async def test_named_target_ambiguity_is_a_receipt_without_a_click(monkeypatch):
    calls = []

    async def resolve(query, kinds, automation_id):
        calls.append((query, kinds, automation_id))
        return {"status": "ambiguous", "element": None, "candidates": [{"name": "Save"}], "detail": "Two controls match"}

    service = SimpleNamespace(_uia_available=lambda: True, _foreground_hwnd=lambda: 40,
                              resolve_element=resolve)
    monkeypatch.setattr(mcp_desktop, "vision_service", lambda: service)
    result = (await request("desktop_find_target", {"query": "Save", "kinds": "button", "automation_id": "save2"}))["result"]
    assert calls == [("Save", "button", "save2")]
    assert result["isError"] and result["structuredContent"]["element"] is None
    assert result["structuredContent"]["candidates"]


async def test_target_transport_failure_never_discloses_exception_content(monkeypatch):
    async def resolve(*_args):
        raise RuntimeError("private-token-must-not-return")

    service = SimpleNamespace(_uia_available=lambda: True, _foreground_hwnd=lambda: 40,
                              resolve_element=resolve)
    monkeypatch.setattr(mcp_desktop, "vision_service", lambda: service)
    reply = await request("desktop_find_target", {"query": "Save"})
    assert reply["result"]["isError"] and "private-token" not in str(reply)


async def test_notification_bad_id_and_invalid_params():
    assert await mcp_desktop.respond({"method": "notifications/initialized"}) is None
    assert (await mcp_desktop.respond({"id": True, "method": "ping"}))["error"]["code"] == -32600
    assert (await mcp_desktop.respond({"id": 3, "method": "tools/list", "params": []}))["error"]["code"] == -32602
    assert (await mcp_desktop.respond({"id": 3, "method": "tools/list", "params": {"cursor": "invented"}}))["error"]["code"] == -32602


def test_new_server_migration_preserves_an_explicit_disabled_choice():
    data = {"_revision": "older", "servers": {}}
    assert mcp_host.migrate_config(data)
    assert data["servers"]["orion_desktop"]["args"][0].endswith("mcp_desktop.py")
    data["servers"]["orion_desktop"]["enabled"] = False
    data["_revision"] = "older"
    mcp_host.migrate_config(data)
    assert data["servers"]["orion_desktop"]["enabled"] is False


def test_bundled_server_uses_the_host_interpreter_when_source_launched(monkeypatch):
    data = {"servers": {"orion_desktop": {"command": "obsolete-python", "args": [str(mcp_host.Path(mcp_host.__file__).with_name("mcp_desktop.py"))]}}}
    monkeypatch.setattr(mcp_host.sys, "frozen", False, raising=False)
    mcp_host._host_runtime_servers(data)
    assert data["servers"]["orion_desktop"]["command"] == mcp_host.sys.executable
