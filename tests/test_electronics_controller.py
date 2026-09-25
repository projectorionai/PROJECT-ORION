"""Inspection lifecycle never loses the captured frame or strands its UI."""
import asyncio
from types import SimpleNamespace

from orion_core.data import ToolResult
from orion_core.electronics_controller import ElectronicsInspectionController


class Signal:
    def __init__(self):
        self.events = []
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, payload):
        self.events.append(payload)
        for slot in self.slots:
            slot(payload)


def bus():
    return SimpleNamespace(**{f"electronics_scan_{name}": Signal()
                              for name in ("requested", "started", "result", "error")})


def test_exact_frame_and_request_reach_result():
    async def scenario():
        signals = bus()
        frame = object()

        async def inspect(**kwargs):
            assert kwargs["frame"] is frame
            assert kwargs["prompt"] == "Read U1"
            return ToolResult("Local report", media={"data": b"jpeg"},
                              evidence=[{"status": "local_only", "observations": []}])

        control = ElectronicsInspectionController(signals, SimpleNamespace(inspect_electronics=inspect))
        signals.electronics_scan_requested.emit({"request_id": "scan-1", "frame": frame, "focus": "Read U1"})
        await control._task
        assert signals.electronics_scan_started.events == [{"request_id": "scan-1"}]
        assert signals.electronics_scan_result.events == [{
            "request_id": "scan-1", "status": "local_only", "observations": [], "jpeg": b"jpeg"}]
        assert not signals.electronics_scan_error.events

    asyncio.run(scenario())


def test_concurrent_requests_are_rejected_and_shutdown_cancels():
    async def scenario():
        signals = bus()

        async def inspect(**kwargs):
            await asyncio.Event().wait()

        control = ElectronicsInspectionController(signals, SimpleNamespace(inspect_electronics=inspect))
        control.request({"request_id": "first", "frame": object()})
        await asyncio.sleep(0)
        control.request({"request_id": "second", "frame": object()})
        assert signals.electronics_scan_error.events[-1]["request_id"] == "second"
        await control.shutdown()
        assert control._task.cancelled()
        assert signals.electronics_scan_error.events[-1]["request_id"] == "first"
        assert not signals.electronics_scan_result.events

    asyncio.run(scenario())


def test_provider_errors_remain_actionable_and_do_not_expose_secrets():
    async def scenario():
        signals = bus()

        async def inspect(**kwargs):
            raise RuntimeError("https://provider.test?api_key=private-value")

        control = ElectronicsInspectionController(signals, SimpleNamespace(inspect_electronics=inspect))
        control.request({"request_id": "failed", "frame": object()})
        await control._task
        event = signals.electronics_scan_error.events[-1]
        assert event["request_id"] == "failed"
        assert "provider" in event["message"]
        assert "private-value" not in event["message"]

    asyncio.run(scenario())


def test_missing_frame_and_loop_report_failure():
    signals = bus()
    control = ElectronicsInspectionController(signals, SimpleNamespace())
    control.request({"request_id": "empty"})
    control.request({"request_id": "no-loop", "frame": object()})
    assert [event["request_id"] for event in signals.electronics_scan_error.events] == ["empty", "no-loop"]


def test_timeout_is_reported(monkeypatch):
    async def scenario():
        signals = bus()

        async def inspect(**kwargs):
            return ToolResult("unused")

        async def timed_out(coro, timeout):
            assert timeout == 60.0
            coro.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", timed_out)
        control = ElectronicsInspectionController(signals, SimpleNamespace(inspect_electronics=inspect))
        control.request({"request_id": "slow", "frame": object()})
        await control._task
        assert "timed out" in signals.electronics_scan_error.events[-1]["message"]

    asyncio.run(scenario())
