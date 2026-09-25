"""A PCB scan must never reuse a stopped/disconnected camera's stale frame."""
import time
from types import SimpleNamespace

from orion_core.face_tracking import FaceTracker


def test_latest_frame_is_fresh_and_defensive():
    tracker = FaceTracker(lambda _: None, camera_index=0)
    tracker.running = True
    tracker._latest_frame = [1, 2, 3]
    tracker._frame_time = time.monotonic()
    frame = tracker.latest_frame()
    frame[0] = 9
    assert tracker._latest_frame == [1, 2, 3]
    tracker._frame_time -= 3
    assert tracker.latest_frame() is None
    tracker._frame_time = time.monotonic()
    tracker.running = False
    assert tracker.latest_frame() is None


def test_stopping_clears_frame_and_restarting_waits_for_driver():
    tracker = FaceTracker(lambda _: None, camera_index=0)
    tracker.available = True
    tracker.running = True
    tracker._latest_frame = [1, 2, 3]
    tracker._frame_time = time.monotonic()
    old_thread = SimpleNamespace(is_alive=lambda: True, join=lambda **_: None)
    tracker._thread = old_thread
    tracker.stop()
    assert tracker.latest_frame() is None
    assert tracker._latest_frame is None
    assert tracker._thread is old_thread
    assert not tracker.start().ok
    assert tracker._stop.is_set()


def test_capture_loop_clears_disconnected_frames_and_accepts_recovery(monkeypatch):
    import numpy as np
    import orion_core.face_tracking as module
    tracker = FaceTracker(lambda sample: None, camera_index=0)
    tracker.running = True
    monkeypatch.setattr(tracker, "_load_detector", lambda: None)
    monkeypatch.setattr(tracker, "_load_cascade", lambda: None)
    monkeypatch.setattr(tracker._stop, "wait", lambda timeout: tracker._stop.is_set())
    first = np.full((24, 32, 3), 30, dtype=np.uint8)
    second = np.full((24, 32, 3), 80, dtype=np.uint8)
    class Capture:
        reads = 0
        released = False
        def isOpened(self): return True
        def read(self):
            self.reads += 1
            if self.reads == 1: return True, first
            if self.reads == 2:
                assert np.array_equal(tracker.latest_frame(), first)
                return False, None
            if self.reads == 3:
                assert tracker.latest_frame() is None
                return True, second
            assert np.array_equal(tracker.latest_frame(), second)
            tracker._stop.set()
            return False, None
        def release(self): self.released = True
        def get(self, *args): return 30.0
        def set(self, *args): return True
    capture = Capture()
    monkeypatch.setattr(module, "open_camera_capture", lambda *args, **kwargs: (capture, "test"))
    tracker._loop()
    assert capture.reads == 4 and capture.released
    assert tracker.latest_frame() is None and not tracker.running
