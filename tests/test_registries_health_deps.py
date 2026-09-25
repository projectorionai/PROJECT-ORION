"""
Tests for the widened ModuleRegistry (Mark XX architectural-audit pass,
Track G): dependencies, version, and live health cross-referenced from
Telemetry.health rather than tracked as a second, parallel concept.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.registries import ModuleRegistry, SystemRegistries


class _StubHealth:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def snapshot(self) -> list[dict]:
        return list(self._rows)


class _StubTelemetry:
    def __init__(self, rows: list[dict]) -> None:
        self.health = _StubHealth(rows)


def test_register_defaults_dependencies_and_version_to_empty():
    registry = ModuleRegistry()
    registry.register("evidence", object(), "claims")
    record = registry.get("evidence")
    assert record["dependencies"] == []
    assert record["version"] == ""


def test_register_stores_dependencies_and_version():
    registry = ModuleRegistry()
    registry.register("dispatcher", object(), "tool router",
                       dependencies=["memory", "vision", "desktop"], version="20.0.0")
    record = registry.get("dispatcher")
    assert record["dependencies"] == ["memory", "vision", "desktop"]
    assert record["version"] == "20.0.0"


def test_register_strips_blank_dependency_entries():
    registry = ModuleRegistry()
    registry.register("x", object(), dependencies=["memory", "", "  ", "vision"])
    assert registry.get("x")["dependencies"] == ["memory", "vision"]


def test_health_for_returns_unknown_with_no_telemetry():
    registry = ModuleRegistry()
    registry.register("dispatcher", object())
    assert registry.health_for("dispatcher") == "UNKNOWN"
    assert registry.health_for("dispatcher", telemetry=None) == "UNKNOWN"


def test_health_for_returns_unknown_for_a_module_with_no_heartbeat():
    registry = ModuleRegistry()
    registry.register("dispatcher", object())
    telemetry = _StubTelemetry([{"name": "vision", "status": "OK"}])
    assert registry.health_for("dispatcher", telemetry) == "UNKNOWN"


def test_health_for_cross_references_the_real_telemetry_health():
    registry = ModuleRegistry()
    registry.register("dispatcher", object())
    telemetry = _StubTelemetry([{"name": "dispatcher", "status": "DEGRADED"}])
    assert registry.health_for("dispatcher", telemetry) == "DEGRADED"


def test_health_for_never_raises_on_a_broken_telemetry_object():
    registry = ModuleRegistry()
    registry.register("dispatcher", object())

    class _Broken:
        health = object()   # no .snapshot()

    assert registry.health_for("dispatcher", _Broken()) == "UNKNOWN"


def test_describe_returns_none_for_an_unregistered_module():
    registry = ModuleRegistry()
    assert registry.describe("nope") is None


def test_describe_merges_the_static_record_with_live_health():
    registry = ModuleRegistry()
    registry.register("vision", object(), "screen capture + OCR",
                       dependencies=["desktop"], version="20.0.0")
    telemetry = _StubTelemetry([{"name": "vision", "status": "OK"}])
    described = registry.describe("vision", telemetry)
    assert described["role"] == "screen capture + OCR"
    assert described["dependencies"] == ["desktop"]
    assert described["version"] == "20.0.0"
    assert described["health"] == "OK"


def test_all_described_covers_every_registered_module():
    registry = ModuleRegistry()
    registry.register("a", object())
    registry.register("b", None)
    telemetry = _StubTelemetry([{"name": "a", "status": "OK"}])
    described = registry.all_described(telemetry)
    assert set(described) == {"a", "b"}
    assert described["a"]["health"] == "OK"
    assert described["b"]["health"] == "UNKNOWN"


def test_all_described_with_no_telemetry_is_all_unknown():
    registry = ModuleRegistry()
    registry.register("a", object())
    described = registry.all_described()
    assert described["a"]["health"] == "UNKNOWN"


def test_system_registries_report_lists_unhealthy_modules_when_telemetry_given():
    registries = SystemRegistries()
    registries.modules.register("dispatcher", object(), "router")
    registries.modules.register("vision", object(), "capture")
    telemetry = _StubTelemetry([
        {"name": "dispatcher", "status": "DOWN"},
        {"name": "vision", "status": "OK"},
    ])
    report = registries.report(telemetry=telemetry)
    assert "Unhealthy modules: dispatcher" in report.text
    assert "vision" not in report.text.split("Unhealthy modules:")[-1]


def test_system_registries_report_omits_the_unhealthy_line_with_no_telemetry():
    registries = SystemRegistries()
    registries.modules.register("dispatcher", object(), "router")
    report = registries.report()
    assert "Unhealthy" not in report.text


def test_system_registries_report_omits_the_unhealthy_line_when_all_healthy():
    registries = SystemRegistries()
    registries.modules.register("vision", object(), "capture")
    telemetry = _StubTelemetry([{"name": "vision", "status": "OK"}])
    report = registries.report(telemetry=telemetry)
    assert "Unhealthy" not in report.text
