"""Mark XXXV exposes what a probe established, without inventing readiness."""
from types import SimpleNamespace

from orion_core import capability_health as health
from orion_core.diagnostics import DiagnosticsEngine


def test_installed_dependency_is_not_a_verified_runtime_operation(monkeypatch):
    monkeypatch.setattr(health, "_importable", lambda *_: True)
    row = health._simple("camera", "cv2", tool="vision_analyse", offline=True,
                         tools={"vision_analyse"})
    receipt = row.as_dict()
    assert receipt["evidence_level"] == "dependency"
    assert receipt["readiness"] == "not_verified" and receipt["verified_at"] is None


def test_actual_deep_ocr_result_is_timestamped_and_can_become_stale(monkeypatch):
    monkeypatch.setattr(health, "_OCR_PROBE", SimpleNamespace(available=True, engine_name="synthetic"))
    monkeypatch.setattr(health, "_functional_ocr", lambda _engine: (True, "ORION"))
    row = health._probe_ocr({"vision_analyse"}, deep=True)
    receipt = row.as_dict(now=row.checked_at)
    assert receipt["evidence_level"] == "operation" and receipt["readiness"] == "verified"
    assert row.as_dict(now=row.verified_at + 301)["readiness"] == "stale"
    row.registered = False
    assert row.as_dict(now=row.checked_at)["readiness"] == "unavailable"


async def test_diagnostic_exposes_machine_readable_evidence_and_actual_execution_history(monkeypatch):
    row = health.CapabilityHealth("Synthetic", True, True, True)
    monkeypatch.setattr(health, "capability_matrix", lambda *_: [row])
    engine = DiagnosticsEngine.__new__(DiagnosticsEngine)
    engine.dispatcher = SimpleNamespace(recent_executions=lambda **_: [
        {"tool": "research", "state": "returned", "executor_ok": True}])
    result = await engine.capability_report(deep=False)
    assert result.receipt["capability_health"][0]["readiness"] == "not_verified"
    assert result.receipt["recent_tool_executions"][0]["state"] == "returned"
    assert "not verified" in result.text.lower()


def test_shallow_health_never_initializes_or_downloads_an_ocr_model(monkeypatch):
    import orion_core.ocr_engine as ocr
    monkeypatch.setattr(health, "_importable", lambda *_: True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("health polling must not initialize models")

    monkeypatch.setattr(ocr, "OcrEngine", forbidden)
    row = health._probe_ocr({"vision_analyse"}, deep=False)
    assert row.present and row.readiness() == "not_verified"
