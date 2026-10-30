"""
Tests for the authorized-target store — the persistent scope list
security_recon.py's gate_security_action() calls consult before any real
network action (scan_host, packet_capture, craft_packet) is allowed to run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.authorized_targets import (
    add_authorized_target,
    list_authorized_targets,
    load_authorized_targets,
    remove_authorized_target,
)


def test_missing_file_returns_empty_set(tmp_path):
    assert load_authorized_targets(tmp_path / "nope.json") == set()


def test_add_then_load_round_trips(tmp_path):
    path = tmp_path / "targets.json"
    add_authorized_target("192.168.1.50", path)
    assert load_authorized_targets(path) == {"192.168.1.50"}


def test_add_normalises_case_and_whitespace(tmp_path):
    path = tmp_path / "targets.json"
    add_authorized_target("  LabVM.Local  ", path)
    assert load_authorized_targets(path) == {"labvm.local"}


def test_add_is_idempotent(tmp_path):
    path = tmp_path / "targets.json"
    add_authorized_target("10.0.0.5", path)
    add_authorized_target("10.0.0.5", path)
    assert load_authorized_targets(path) == {"10.0.0.5"}


def test_remove_target(tmp_path):
    path = tmp_path / "targets.json"
    add_authorized_target("10.0.0.5", path)
    add_authorized_target("10.0.0.6", path)
    remove_authorized_target("10.0.0.5", path)
    assert load_authorized_targets(path) == {"10.0.0.6"}


def test_remove_absent_target_is_a_noop(tmp_path):
    path = tmp_path / "targets.json"
    add_authorized_target("10.0.0.5", path)
    remove_authorized_target("not.there", path)
    assert load_authorized_targets(path) == {"10.0.0.5"}


def test_list_is_sorted(tmp_path):
    path = tmp_path / "targets.json"
    add_authorized_target("zeta.local", path)
    add_authorized_target("alpha.local", path)
    assert list_authorized_targets(path) == ["alpha.local", "zeta.local"]


def test_corrupt_json_degrades_to_empty_set(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_authorized_targets(path) == set()


def test_unexpected_json_shape_degrades_to_empty_set(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(["just", "a", "list"]), encoding="utf-8")
    assert load_authorized_targets(path) == set()


def test_add_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "targets.json"
    add_authorized_target("127.0.0.1", path)
    assert path.exists()
    assert load_authorized_targets(path) == {"127.0.0.1"}


# ── "scan my network" from someone at home (lan_devices) ─────────────────────

_ARP = """
Interface: 10.115.15.69 --- 0x5
  Internet Address      Physical Address      Type
  10.115.0.1            48-8f-5a-11-22-33     dynamic
  10.115.2.100          da-a1-19-44-55-66     dynamic
  10.115.255.255        ff-ff-ff-ff-ff-ff     static
  224.0.0.22            01-00-5e-00-00-16     static
  239.255.255.250       01-00-5e-7f-ff-fa     static
"""


def test_the_neighbour_table_becomes_devices_not_multicast():
    """3 Oct: "scan the network real quick" got three clarifying questions and
    a refusal - the only scan needed an authorised target and nmap."""
    from orion_core.lan_devices import describe, parse_arp
    devices = parse_arp(_ARP)
    assert [d.ip for d in devices] == ["10.115.0.1", "10.115.2.100"]
    assert devices[1].private_mac and not devices[0].private_mac
    said = describe(read_table=lambda: _ARP, name_devices=lambda found: None)
    assert "2 devices" in said and "likely your router" in said
    assert "phone or tablet" in said and "passive" in said


def test_lan_devices_needs_no_authorisation_and_is_routed(monkeypatch):
    import asyncio
    import types
    from orion_core import dispatch_schema, lan_devices
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    monkeypatch.setattr(lan_devices, "describe", lambda: "I can see 2 devices.")
    me = types.SimpleNamespace(security_recon=types.SimpleNamespace())  # no targets authorised
    result = asyncio.run(KnowledgeDispatchMixin.security_recon_tool(me, {"action": "lan_devices"}))
    assert result.ok and "2 devices" in str(result.response_payload())
    schema = next(t for t in dispatch_schema.TOOL_DECLARATIONS if t["name"] == "security_recon")
    assert "'lan_devices'" in schema["description"] and "scan my network" in schema["description"]
