"""PCB scans keep real image evidence, never fabricate local detections."""

import asyncio
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from orion_core.electronics_inspection import inspect_frame, normalised_box, parse_model_report


def _frame():
    return Image.new("RGB", (640, 480), (40, 80, 35))


def _answer():
    return json.dumps({"summary": "A populated green circuit board is visible.",
                       "observations": [{"label": "Possible IC", "detail": "An eight-pin package.",
                                         "evidence": "Four leads on each side; marking unreadable.",
                                         "confidence": 0.65, "bbox": [0.2, 0.3, 0.2, 0.1]}],
                       "limitations": ["The package marking cannot be read."],
                       "next_steps": ["Take a close-up of the IC marking."]})


def test_offline_checks_do_not_invent_components():
    result = asyncio.run(inspect_frame(_frame(), ocr_reader=lambda image: "U1\nR20"))
    report = result.evidence[0]
    assert result.ok and report["status"] == "local_only"
    assert report["observations"] == []
    assert [item["text"] for item in report["markings"]] == ["U1", "R20"]
    assert report["markings"][0]["confidence"] is None
    assert report["quality"]["width"] == 640
    assert "electrical function" in result.text
    assert set(result.media) == {"data", "mime_type"}
    assert Image.open(io.BytesIO(result.media["data"])).size == (640, 480)


def test_model_gets_real_frame_and_returns_validated_boxes():
    seen = []

    async def generate(jpeg, prompt, **kwargs):
        seen.append((jpeg, prompt, kwargs))
        return SimpleNamespace(name="test-vision"), _answer()

    result = asyncio.run(inspect_frame(_frame(), "Check soldering", router=SimpleNamespace(generate_vision=generate)))
    report = result.evidence[0]
    assert seen[0][0] == result.media["data"]
    assert "Check soldering" in seen[0][1]
    assert "untrusted evidence" in seen[0][2]["instruction"]
    assert report["status"] == "complete"
    assert report["provider"] == "test-vision"
    assert report["observations"][0]["bbox"] == [0.2, 0.3, 0.2, 0.1]


@pytest.mark.parametrize("box", [[100, 30, 40, 20], [-0.1, 0, .5, .5], [.8, .8, .4, .4],
                                 [0, 0, 0, .3], [0, 0, float("nan"), .2], [0, 0, True, .2], None])
def test_invalid_boxes_are_rejected_not_rescaled(box):
    assert normalised_box(box) is None


def test_missing_evidence_drops_observation_and_invalid_box_stays_unlocated():
    parsed = json.loads(_answer())
    parsed["observations"].append({"label": "Invented resistor", "detail": "No visible evidence"})
    parsed["observations"][0]["bbox"] = [10, 20, 30, 40]
    parsed["observations"][0]["confidence"] = float("nan")
    report = parse_model_report("```json\n" + json.dumps(parsed) + "\n```")
    assert len(report["observations"]) == 1
    assert "bbox" not in report["observations"][0]
    assert report["observations"][0]["confidence"] is None


def test_malformed_model_result_is_honest_local_fallback():
    async def generate(*args, **kwargs):
        return None, "All components are definitely working."

    result = asyncio.run(inspect_frame(_frame(), router=SimpleNamespace(generate_vision=generate)))
    assert result.evidence[0]["status"] == "local_only"
    assert result.evidence[0]["observations"] == []
    assert "definitely" not in result.text
    assert "not a valid structured" in result.text


def test_model_timeout_retains_frame_and_local_results():
    async def generate(*args, **kwargs):
        await asyncio.sleep(2)

    result = asyncio.run(inspect_frame(_frame(), router=SimpleNamespace(generate_vision=generate), timeout_s=.01))
    assert result.ok and result.media["data"]
    assert "timed out" in result.text


def test_bgr_frame_is_converted_without_mutating_source():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, :, 2] = 255
    original = frame.copy()
    result = asyncio.run(inspect_frame(frame))
    image = Image.open(io.BytesIO(result.media["data"]))
    r, g, b = image.getpixel((15, 10))
    assert r > 245 and g < 10 and b < 10
    assert np.array_equal(frame, original)


def test_invalid_frame_reports_failure():
    result = asyncio.run(inspect_frame(b"not an image"))
    assert not result.ok
    assert result.media is None


class _Signal:
    def __init__(self):
        self.events = []

    def emit(self, payload):
        self.events.append(payload)


def test_dispatch_pcb_shows_the_actual_scan_in_camera_workbench():
    from orion_core.dispatch_vision import VisionDispatchMixin
    calls = []

    async def inspect(prompt, index, **kwargs):
        calls.append((prompt, index, kwargs))
        return await inspect_frame(_frame())

    bus = SimpleNamespace(**{name: _Signal() for name in
        ("electronics_scan_started", "electronics_scan_result", "electronics_scan_error")})
    dispatcher = SimpleNamespace(bus=bus, vision=SimpleNamespace(inspect_electronics=inspect), router="router")
    result = asyncio.run(VisionDispatchMixin.vision_analyse(dispatcher, {"action": "pcb", "prompt": "Read U1"}))
    assert result.ok
    # Electronics mode, on the default reference grid, so findings name cells.
    assert calls == [("Read U1", None, {"router": "router", "mode": "electronics",
                                        "grid": (8, 6)})]
    event = bus.electronics_scan_result.events[0]
    assert event["request_id"] == bus.electronics_scan_started.events[0]["request_id"]
    assert event["jpeg"] == result.media["data"]
    assert event["status"] == "local_only"


def test_supplied_camera_still_never_opens_another_camera():
    from orion_core.vision import VisionAgent
    agent = VisionAgent.__new__(VisionAgent)
    agent.ocr_engine = SimpleNamespace(image_to_result=lambda image: SimpleNamespace(text="R1", confidence=.8, engine="test"))

    async def forbidden(*args, **kwargs):
        raise AssertionError("Must inspect the user's selected frame")

    agent.capture_live_frame = forbidden
    result = asyncio.run(agent.inspect_electronics(frame=_frame()))
    assert result.ok and result.evidence[0]["markings"][0]["text"] == "R1"
