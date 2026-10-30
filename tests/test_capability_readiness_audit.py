"""Presence and routing eligibility cannot stand in for operational readiness."""
from types import SimpleNamespace

from orion_core import capability_health as health
from orion_core.diagnostics import DiagnosticsEngine
from orion_core.dispatch_productivity import ProductivityDispatchMixin


def test_unregistered_present_capability_is_degraded_and_unavailable():
    row = health.CapabilityHealth("orphaned", True, True, True, registered=False)
    assert row.status() == "DEGRADED"
    assert health.unavailable([row]) == [row]
    assert "UNREG" in row.line()


async def test_diagnostics_does_not_return_success_for_a_degraded_capability(monkeypatch):
    monkeypatch.setattr(health, "capability_matrix", lambda *_: [
        health.CapabilityHealth("broken OCR", True, False, True)])
    engine = DiagnosticsEngine.__new__(DiagnosticsEngine)
    result = await engine.capability_report(deep=False)
    assert not result.ok and "degraded" in result.text


def test_dependency_only_probes_state_their_actual_evidence(monkeypatch):
    monkeypatch.setattr(health, "_importable", lambda *_: True)
    row = health._simple("camera", "cv2", tool="vision_analyse", offline=True,
                         tools={"vision_analyse"})
    assert "runtime operation not checked" in row.detail
    assert "cached model and microphone not checked" in health._probe_stt().detail
    assert "end-to-end readiness" in health.render_matrix([row])


def test_self_description_does_not_claim_retired_live_session_or_untested_camera_is_ready():
    dispatcher = ProductivityDispatchMixin()
    dispatcher.live_worker = SimpleNamespace(session=object(), connected=False,
                                             _transport_ready=lambda: False)
    dispatcher.TOOL_DECLARATIONS = [{"name": "camera"}]
    dispatcher._tool_handlers = {"reviewed_plugin": object()}
    dispatcher.ai = SimpleNamespace(router=SimpleNamespace(text_profiles=lambda: [],
                                                           live_profiles=lambda: []))
    report = dispatcher._live_self_description()
    assert "Gemini Live is not connected" in report
    assert "camera readiness not checked" in report
    assert "1 registered" in report and "dynamic handler" in report
    assert "I forged myself" not in report and "camera works" not in report
