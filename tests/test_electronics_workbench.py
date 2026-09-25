"""Headless workbench coverage; never opens a physical camera or model API."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from orion_core.gui.electronics_workbench import ElectronicsWorkbench


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, payload):
        for slot in self.slots:
            slot(payload)


class Tracker:
    def __init__(self, *, running=True):
        self.running = running
        self.frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        self.starts = 0
        self.stops = 0
        self.stop_thread = None

    def is_capturing(self):
        return self.running

    def latest_frame(self):
        return self.frame

    def start(self):
        self.starts += 1
        self.running = True
        return SimpleNamespace(ok=True)

    def stop(self):
        self.stops += 1
        self.stop_thread = threading.current_thread()
        self.running = False


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def workbench(app):
    bus = SimpleNamespace(**{f"electronics_scan_{kind}": Signal()
                             for kind in ("requested", "started", "result", "error")})
    widget = ElectronicsWorkbench(bus)
    widget.resize(1100, 720)
    widget.show()
    yield widget
    widget.shutdown()
    widget.close()
    app.processEvents()


def activate(widget, tracker=None):
    tracker = tracker or Tracker()
    widget.attach_tracker(tracker)
    widget.start_camera()
    return tracker


def test_show_does_not_activate_the_camera(workbench):
    tracker = Tracker(running=False)
    workbench.attach_tracker(tracker)
    assert tracker.starts == 0
    assert not workbench._timer.isActive()
    assert not workbench.capture_button.isEnabled()


def test_existing_tracker_is_borrowed_and_never_stopped(workbench):
    tracker = activate(workbench)
    assert tracker.starts == 0
    assert workbench.canvas.image.width() == 160
    workbench.stop_camera()
    assert tracker.stops == 0
    assert tracker.running


def test_owned_camera_released_off_gui_thread_when_hidden(workbench, app):
    tracker = activate(workbench, Tracker(running=False))
    assert tracker.starts == 1
    workbench.hide()
    deadline = time.monotonic() + 2
    while workbench._stopping and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert tracker.stops == 1
    assert tracker.stop_thread is not threading.current_thread()
    assert not workbench._timer.isActive()


def test_capture_is_an_independent_copy_of_the_visible_frame(workbench):
    tracker = activate(workbench)
    requests = []
    workbench.bus.electronics_scan_requested.connect(requests.append)
    workbench.focus_input.setText("Read U3")
    workbench.capture()
    tracker.frame[:] = 0
    assert len(requests) == 1
    assert requests[0]["focus"] == "Read U3"
    assert requests[0]["frame"].mean() == 100
    assert workbench.canvas.captured
    assert workbench.canvas.image.pixelColor(0, 0).red() == 100
    workbench._refresh()
    assert workbench.canvas.image.pixelColor(0, 0).red() == 100
    workbench.capture()
    assert len(requests) == 1


def test_stale_results_cannot_annotate_new_capture(workbench):
    activate(workbench)
    workbench.capture()
    workbench.on_scan_result({"request_id": "older-request", "summary": "Wrong board",
                              "observations": [{"label": "Incorrect"}]})
    assert "Wrong board" not in workbench.report.toPlainText()
    assert workbench._busy
    assert not workbench.canvas.observations


def test_annotations_only_draw_on_retained_capture(workbench):
    activate(workbench)
    workbench.capture()
    result = {"request_id": workbench._request_id, "status": "complete",
              "summary": "Visible integrated circuit", "observations": [
                  {"label": "U3", "detail": "Marking visible", "confidence": .7,
                   "bbox": [.2, .3, .25, .15]}]}
    workbench.bus.electronics_scan_result.emit(result)
    assert len(workbench.canvas.observations) == 1
    assert not workbench._busy
    workbench.show_live()
    assert not workbench.canvas.captured
    assert not workbench.canvas.observations
    workbench.show_live()
    assert workbench.canvas.captured
    assert len(workbench.canvas.observations) == 1


def test_model_html_is_displayed_as_plain_text(workbench):
    activate(workbench)
    workbench.capture()
    workbench.on_scan_result({"request_id": workbench._request_id,
                              "summary": '<img src="https://example.com/tracker">',
                              "observations": []})
    assert '<img src="https://example.com/tracker">' in workbench.report.toPlainText()


def test_local_only_result_is_identified_without_fake_detections(workbench):
    activate(workbench)
    workbench.capture()
    workbench.on_scan_result({"request_id": workbench._request_id, "status": "local_only",
                              "summary": "Vision provider unavailable", "observations": [],
                              "quality": {"warnings": ["Low light"]}})
    assert workbench.report_heading.text() == "LOCAL CHECKS"
    assert workbench.quality_label.text() == "Low light"
    assert not workbench.canvas.observations


def test_stale_tracker_frame_disables_capture(workbench):
    tracker = Tracker()
    tracker.frames = 10
    activate(workbench, tracker)
    workbench._last_frame_time = time.monotonic() - 10
    workbench._refresh()
    assert workbench._frame is None
    assert not workbench.capture_button.isEnabled()
    assert workbench.state_label.text() == "WAITING FOR CAMERA"


def test_scan_error_retains_capture_and_allows_retry(workbench):
    activate(workbench)
    workbench.capture()
    retained = workbench.canvas.image
    workbench.bus.electronics_scan_error.emit({"request_id": workbench._request_id,
                                              "error": "Service is offline"})
    assert not workbench._busy
    assert workbench.canvas.image is retained
    assert "Service is offline" in workbench.report.toPlainText()
    workbench.show_live()
    assert workbench.capture_button.isEnabled()


def test_hidden_shared_camera_has_no_preview_polling(workbench, app):
    tracker = activate(workbench)
    workbench.hide()
    assert not workbench._timer.isActive()
    assert tracker.stops == 0
    workbench.show()
    app.processEvents()
    assert workbench._timer.isActive()


def test_narrow_window_stacks_camera_and_notes(workbench, app):
    workbench.resize(700, 820)
    app.processEvents()
    assert workbench.splitter.orientation() == Qt.Orientation.Vertical
    workbench.resize(1100, 720)
    app.processEvents()
    assert workbench.splitter.orientation() == Qt.Orientation.Horizontal


# ── live geometry on the camera ──────────────────────────────────────────────
#
# "It must show on the camera": the workbench draws locally-measured geometry
# over the live feed continuously, and model findings over the captured frame.
# These are painted in deliberately different colours, and these tests read the
# pixels rather than trusting that the paint code ran.

from orion_core.board_detect import BoardReading  # noqa: E402

READING = BoardReading(
    board=[0.2, 0.2, 0.6, 0.6],
    quad=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
    regions=[[0.3, 0.3, 0.1, 0.1], [0.5, 0.5, 0.12, 0.08]],
    focus=300.0, brightness=0.4, width=160, height=120)


def _repaint(widget, app):
    """Force a real paint and report how many geometry shapes it drew.

    The canvas counts them itself rather than the test sampling pixels: the
    overlay is silver, and so are the canvas's own guide ticks and its "LIVE
    CAMERA" tag, so a colour count cannot tell the overlay from the chrome.
    """
    widget.canvas.repaint()
    app.processEvents()
    return widget.canvas.geometry_drawn


def test_live_view_draws_the_detected_geometry(workbench, app):
    activate(workbench)
    workbench.canvas.reading = READING
    # one board outline + two regions
    assert _repaint(workbench, app) == 3


def test_live_view_without_a_reading_draws_no_geometry(workbench, app):
    activate(workbench)
    workbench.canvas.reading = None
    assert _repaint(workbench, app) == 0


def test_an_unavailable_reading_draws_nothing(workbench, app):
    activate(workbench)
    workbench.canvas.reading = BoardReading(available=False)
    assert _repaint(workbench, app) == 0


def test_a_board_without_four_corners_still_gets_an_outline(workbench, app):
    """No quad (the object is not rectangular enough to claim one) must still
    show the user where ORION is looking."""
    activate(workbench)
    workbench.canvas.reading = BoardReading(board=[0.1, 0.1, 0.5, 0.5], regions=[])
    assert _repaint(workbench, app) == 1


def test_capture_keeps_local_geometry_while_the_model_localises_nothing(workbench, app):
    """Most vision models answer in prose only. The captured frame must not go
    blank through a perfectly good text report."""
    activate(workbench)
    workbench.canvas.reading = READING
    workbench.canvas.set_image(workbench.canvas.image, captured=True)
    workbench.canvas.observations = [{"label": "A capacitor", "detail": "no bbox"}]
    assert _repaint(workbench, app) == 3


def test_model_findings_replace_local_geometry_rather_than_crowd_it(workbench, app):
    activate(workbench)
    workbench.canvas.reading = READING
    workbench.canvas.set_image(workbench.canvas.image, captured=True)
    workbench.canvas.observations = [
        {"label": "Microcontroller", "bbox": [0.3, 0.3, 0.2, 0.2]}]
    assert _repaint(workbench, app) == 0, (
        "local hairlines are competing with the model's findings")


def test_the_overlay_uses_orions_palette_not_a_scanner_cyan(workbench):
    """ORION's identity retired the electric-cyan counter-hue for silver on
    graphite; a scanning overlay is the most tempting place to reintroduce it."""
    from orion_core.constants import C
    from orion_core.gui.electronics_workbench import LIVE_EDGE, LIVE_REGION
    assert LIVE_EDGE.name() == C.ACCENT
    assert LIVE_REGION.name() == C.SILVER


def test_malformed_geometry_never_breaks_the_paint(workbench, app):
    activate(workbench)
    for broken in (
        BoardReading(board="nonsense", regions=[[0.1, 0.1]], focus=1.0),
        BoardReading(board=[0.1, 0.1, float("nan"), 0.2]),
        BoardReading(quad=[[0.1, 0.1], [0.2, 0.2]], board=[0, 0, 1, 1]),
        BoardReading(quad=[["x", "y"], [1, 1], [1, 0], [0, 1]], board=[0, 0, 1, 1]),
        BoardReading(regions=[[0.1, 0.1, -1, 0.2], None, "no"]),
    ):
        workbench.canvas.reading = broken
        _repaint(workbench, app)          # a raise here fails the test


# ── the detector follows the camera ──────────────────────────────────────────

def test_the_scanner_runs_only_while_the_camera_does(workbench):
    activate(workbench)
    assert workbench._scanner.running, "detection never started with the camera"
    workbench.stop_camera()
    assert not workbench._scanner.running, "detection outlived the camera"


def test_shutdown_stops_the_scanner(workbench):
    activate(workbench)
    workbench.shutdown()
    assert not workbench._scanner.running


def test_a_retained_capture_keeps_its_geometry_when_the_camera_stops(workbench):
    activate(workbench)
    workbench.canvas.reading = READING
    workbench._show_capture = True
    workbench.stop_camera()
    assert workbench.canvas.reading is READING


# ── asking again about the frame already captured ────────────────────────────

def _capture(workbench, app):
    tracker = activate(workbench)
    workbench.capture()
    app.processEvents()
    return tracker


def test_ask_is_unavailable_until_something_is_captured(workbench):
    activate(workbench)
    assert not workbench.ask_button.isEnabled()
    workbench.ask_again()                      # must be a no-op, not a crash
    assert workbench._request_id is None


def test_a_follow_up_question_reuses_the_captured_frame(workbench, app):
    """The point of the feature: a second question must be answered about the
    view on screen, not about a board that has since been moved."""
    sent = []
    workbench.bus.electronics_scan_requested.connect(sent.append)
    tracker = _capture(workbench, app)
    assert len(sent) == 1
    first_frame = sent[0]["frame"]

    workbench.on_scan_result({"request_id": workbench._request_id,
                              "summary": "A board.", "observations": []})
    assert workbench.ask_button.isEnabled()

    # The camera has moved on to a different frame in the meantime.
    tracker.frame = np.full((120, 160, 3), 12, dtype=np.uint8)
    workbench.focus_input.setText("what is the chip in the corner")
    workbench.ask_again()
    assert len(sent) == 2
    assert np.array_equal(sent[1]["frame"], first_frame), (
        "the follow-up was answered about a newer frame than the one shown")
    assert sent[1]["focus"] == "what is the chip in the corner"
    assert sent[1]["request_id"] != sent[0]["request_id"]


def test_pressing_enter_in_the_focus_box_asks_again(workbench, app):
    sent = []
    workbench.bus.electronics_scan_requested.connect(sent.append)
    _capture(workbench, app)
    workbench.on_scan_result({"request_id": workbench._request_id,
                              "summary": "A board.", "observations": []})
    workbench.focus_input.setText("read the markings")
    workbench.focus_input.returnPressed.emit()
    assert len(sent) == 2


def test_enter_on_the_live_view_does_not_fire_a_request(workbench):
    sent = []
    workbench.bus.electronics_scan_requested.connect(sent.append)
    activate(workbench)
    workbench.focus_input.setText("anything")
    workbench.focus_input.returnPressed.emit()
    assert sent == []


def test_one_inspection_at_a_time(workbench, app):
    sent = []
    workbench.bus.electronics_scan_requested.connect(sent.append)
    _capture(workbench, app)
    workbench.ask_again()                      # still busy with the capture
    assert len(sent) == 1
    assert not workbench.ask_button.isEnabled()


def test_a_failed_inspection_can_be_retried_on_the_same_frame(workbench, app):
    sent = []
    workbench.bus.electronics_scan_requested.connect(sent.append)
    _capture(workbench, app)
    workbench.on_scan_error({"request_id": workbench._request_id,
                             "message": "The vision provider is unavailable."})
    assert workbench.ask_button.isEnabled(), "a failed scan left no way to retry"
    workbench.ask_again()
    assert np.array_equal(sent[1]["frame"], sent[0]["frame"])


def test_returning_to_live_view_disables_asking(workbench, app):
    _capture(workbench, app)
    workbench.on_scan_result({"request_id": workbench._request_id,
                              "summary": "A board.", "observations": []})
    workbench.show_live()
    assert not workbench.ask_button.isEnabled()
    workbench.show_live()                      # back to the capture
    assert workbench.ask_button.isEnabled()


def test_voice_scan_adopts_request_and_displays_its_frame(workbench):
    import io
    from PIL import Image
    image = Image.new("RGB", (300, 200), "red")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    workbench.bus.electronics_scan_started.emit({"request_id": "voice-scan"})
    assert workbench._busy and workbench._request_id == "voice-scan"
    workbench.bus.electronics_scan_result.emit({
        "request_id": "voice-scan", "status": "local_only", "summary": "Voice capture",
        "jpeg": buffer.getvalue(), "markings": [{"text": "U1 LM358"}], "observations": [],
    })
    assert not workbench._busy
    assert workbench.canvas.captured and workbench.canvas.image.width() == 300
    assert "U1 LM358" in workbench.report.toPlainText()
    assert workbench.ask_button.isEnabled()
    assert workbench._captured_frame == buffer.getvalue()


def test_voice_scan_does_not_replace_an_active_button_scan(workbench):
    activate(workbench)
    workbench.capture()
    original = workbench._request_id
    workbench.bus.electronics_scan_started.emit({"request_id": "competing-scan"})
    workbench.bus.electronics_scan_error.emit({"request_id": "competing-scan", "message": "busy"})
    assert workbench._request_id == original and workbench._busy


def test_geometry_and_capture_use_the_same_processed_frame(workbench, monkeypatch):
    activate(workbench)
    measured = np.full((120, 160, 3), 180, np.uint8)
    monkeypatch.setattr(workbench._scanner, "latest_snapshot", lambda: (measured, READING, time.monotonic()))
    workbench._refresh()
    assert workbench.canvas.reading is READING
    assert workbench.canvas.image.pixelColor(0, 0).red() == 180
    workbench.capture()
    assert np.array_equal(workbench._captured_frame, measured)


def test_old_geometry_is_dropped_for_a_fresh_preview(workbench, monkeypatch):
    tracker = activate(workbench)
    monkeypatch.setattr(workbench._scanner, "latest_snapshot", lambda: (tracker.frame, READING, time.monotonic() - 2))
    workbench._refresh()
    assert workbench.canvas.reading is None
    assert workbench.canvas.image is not None


def test_returning_to_capture_restores_its_own_geometry(workbench):
    activate(workbench)
    workbench.canvas.reading = READING
    workbench.capture()
    workbench.on_scan_result({"request_id": workbench._request_id, "observations": []})
    workbench.show_live()
    workbench.canvas.reading = BoardReading(board=[0, 0, .2, .2])
    workbench.show_live()
    assert workbench.canvas.reading is READING


def test_real_bus_controller_delivers_inspection_to_widget(app):
    import asyncio
    from orion_core.bus import OrionBus
    from orion_core.data import ToolResult
    from orion_core.electronics_controller import ElectronicsInspectionController

    async def scenario():
        bus = OrionBus()
        widget = ElectronicsWorkbench(bus)

        async def inspect(**kwargs):
            assert kwargs["frame"].mean() == 100
            return ToolResult("Inspection ready", evidence=[{
                "status": "local_only", "summary": "Real bus delivered this result.", "observations": [],
            }])

        control = ElectronicsInspectionController(bus, SimpleNamespace(inspect_electronics=inspect))
        try:
            widget.attach_tracker(Tracker())
            widget.show()
            widget.start_camera()
            widget.capture()
            await control._task
            assert not widget._busy
            assert "Real bus delivered this result." in widget.report.toPlainText()
            assert widget.canvas.captured
        finally:
            await control.shutdown()
            widget.shutdown()
            widget.close()

    asyncio.run(scenario())
