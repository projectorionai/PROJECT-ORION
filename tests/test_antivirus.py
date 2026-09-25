"""
Tests for antivirus.py's pure parse/format functions — no real PowerShell or
Defender install required (mirrors how test_docker_control.py and
test_security_recon.py test subprocess-invoking modules: exercise the
parsing/formatting logic directly with canned output, never a real process).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.antivirus import format_status, parse_protection_status, parse_scan_trigger


def _status_json(**overrides) -> str:
    base = {
        "AMServiceEnabled": True,
        "AntivirusEnabled": True,
        "AntispywareEnabled": True,
        "RealTimeProtectionEnabled": True,
        "NISEnabled": True,
        "AntivirusSignatureVersion": "1.455.474.0",
        "AntivirusSignatureAge": 0,
        "AntivirusSignatureLastUpdated": "2026-08-01T00:00:00",
        "QuickScanAge": 12,
        "FullScanAge": 4294967295,   # UInt32.MaxValue — "never run"
        "AMEngineVersion": "1.1.24010.10",
        "AMProductVersion": "4.18.24010.7",
    }
    base.update(overrides)
    return json.dumps(base)


# ── parse_protection_status ─────────────────────────────────────────────────

def test_parse_status_reports_realtime_protection_on():
    status = parse_protection_status(_status_json())
    assert status is not None
    assert status.realtime_protection is True


def test_parse_status_maps_never_run_full_scan_sentinel_to_none():
    """Regression test: Get-MpComputerStatus reports FullScanAge as
    UInt32.MaxValue (4294967295) when a full scan has never run — which is
    the default on most machines, since only quick scans run automatically.
    Left unguarded this printed as "Last full scan: 4294967295 day(s) ago"."""
    status = parse_protection_status(_status_json(FullScanAge=4294967295))
    assert status.full_scan_age_days is None


def test_parse_status_keeps_a_real_full_scan_age():
    status = parse_protection_status(_status_json(FullScanAge=3))
    assert status.full_scan_age_days == 3


def test_parse_status_maps_never_run_quick_scan_sentinel_to_none():
    status = parse_protection_status(_status_json(QuickScanAge=4294967295))
    assert status.quick_scan_age_days is None


def test_parse_status_handles_empty_output():
    assert parse_protection_status("") is None


def test_parse_status_handles_garbage_output():
    assert parse_protection_status("not json at all") is None


def test_parse_status_handles_powershell_single_object_array():
    # ConvertTo-Json sometimes wraps a lone object in an array.
    status = parse_protection_status(f"[{_status_json()}]")
    assert status is not None
    assert status.realtime_protection is True


# ── format_status ────────────────────────────────────────────────────────────

def test_format_status_never_run_full_scan_reads_as_never_recorded():
    status = parse_protection_status(_status_json(FullScanAge=4294967295))
    text = format_status(status)
    assert "never recorded" in text
    assert "4294967295" not in text


def test_format_status_reports_realtime_protection_off_prominently():
    status = parse_protection_status(_status_json(RealTimeProtectionEnabled=False))
    text = format_status(status)
    assert "OFF" in text
    assert "not actively" in text


def test_format_status_today_for_zero_day_scan_age():
    status = parse_protection_status(_status_json(QuickScanAge=0))
    text = format_status(status)
    assert "Last quick scan: today" in text


# ── parse_scan_trigger ───────────────────────────────────────────────────────

def test_parse_scan_trigger_success():
    ok, error = parse_scan_trigger(json.dumps({"Success": True, "Error": ""}))
    assert ok is True
    assert error == ""


def test_parse_scan_trigger_failure_reports_the_error():
    ok, error = parse_scan_trigger(json.dumps({"Success": False, "Error": "access denied"}))
    assert ok is False
    assert "access denied" in error


def test_parse_scan_trigger_handles_no_response():
    ok, error = parse_scan_trigger("")
    assert ok is False
    assert "no response" in error
