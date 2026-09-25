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
