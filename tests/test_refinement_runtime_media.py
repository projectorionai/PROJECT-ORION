"""Cancelled media watches stop future capture and retain native cleanup."""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from orion_core import video_analysis as video


async def _until(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


async def _settled(analyser):
    async def wait():
        while analyser._capture_workers:
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 3)


def _fake_cv2():
    return SimpleNamespace(
        COLOR_BGR2GRAY=1, IMWRITE_JPEG_QUALITY=2, INTER_AREA=3,
        cvtColor=lambda image, *_: image[:, :, 0],
        resize=lambda _image, size, **_: np.zeros((size[1], size[0]), dtype=np.uint8),
        imencode=lambda *_: (True, b"synthetic-jpeg"),
    )


def test_camera_cancellation_stops_new_reads_and_eventually_releases_owned_handle(monkeypatch):
    from orion_core import utils

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    reads, opens = [], []
    owner = threading.get_ident()
    analyser = video.VideoAnalyser()

    class Camera:
        def read(self):
            reads.append(threading.get_ident())
            entered.set()
            assert release.wait(5)
            return True, np.zeros((20, 30, 3), dtype=np.uint8)

        def release(self):
            closed.set()

    def open_camera(index):
        opens.append(index)
        return Camera(), "fake"

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setattr(utils, "open_camera_capture", open_camera)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    monkeypatch.setattr(analyser, "_describe_camera_sequence",
                        lambda *_: pytest.fail("cancelled capture must not describe frames"))

    async def scenario():
        task = asyncio.create_task(analyser.watch_camera(2, max_frames=2, camera_index=4))
        try:
            await _until(entered)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Cancellation cannot claim the blocked OS call already stopped.
            assert not closed.is_set()
            assert len(analyser._capture_workers) == 1
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 3)
        assert len(reads) == 1, "a second camera read started after cancellation"
        assert reads[0] != owner and opens == [4]
        await _settled(analyser)
        assert not analyser._capture_workers

    asyncio.run(scenario())


def test_screen_cancellation_stops_new_grabs_and_closes_native_context(monkeypatch):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    grabs = []
    analyser = video.VideoAnalyser()

    class Grabber:
        monitors = [{}, {"left": 0, "top": 0, "width": 20, "height": 20}]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.set()

        def grab(self, _target):
            grabs.append(threading.get_ident())
            if len(grabs) > 1:
                raise RuntimeError("a new screenshot started after cancellation")
            entered.set()
            assert release.wait(5)
            return np.zeros((20, 20, 4), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=Grabber))
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    monkeypatch.setattr(analyser, "_describe_frames",
                        lambda *_: pytest.fail("cancelled capture must not describe frames"))
    monkeypatch.setattr(analyser, "_heard_during",
                        lambda *_: pytest.fail("cancelled capture must not transcribe"))

    async def scenario():
        task = asyncio.create_task(analyser.watch_screen(3))
        try:
            await _until(entered)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not closed.is_set()
            assert len(analyser._capture_workers) == 1
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 3)
        assert len(grabs) == 1, "a second screen grab started after cancellation"
        await _settled(analyser)
        assert not analyser._capture_workers

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["screen", "camera"])
def test_cancel_before_worker_starts_does_not_open_native_resources(kind, monkeypatch):
    stop = threading.Event()
    stop.set()
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "mss", None)
    analyser = video.VideoAnalyser()
    if kind == "camera":
        frames, problem = analyser._sample_camera(2, 2, 4, stop)
        assert frames == [] and "cancelled" in problem
    else:
        assert analyser._sample_screen(3, 2, False, stop) == []


@pytest.mark.parametrize("kind", ["screen", "camera"])
def test_capture_interval_wait_is_interruptible_without_another_native_read(kind, monkeypatch):
    from orion_core import utils

    waits, reads, closes = [], [], []

    class Stop:
        def is_set(self):
            return False

        def wait(self, delay):
            waits.append(delay)
            # The initial camera slot has no wait; stop in the long interval.
            return delay > 0.1

    class Native:
        monitors = [{}, {"left": 0, "top": 0, "width": 20, "height": 20}]

        def read(self):
            reads.append(1)
            return True, np.zeros((20, 20, 3), dtype=np.uint8)

        def release(self):
            closes.append(1)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.release()

        def grab(self, _target):
            return self.read()[1]

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=Native))
    monkeypatch.setattr(utils, "open_camera_capture", lambda *_: (Native(), "fake"))
    monkeypatch.setattr(time, "sleep", lambda *_: pytest.fail("capture delay must observe stop"))
    analyser = video.VideoAnalyser()
    if kind == "camera":
        frames, _ = analyser._sample_camera(2, 2, 4, Stop())
    else:
        frames = analyser._sample_screen(3, 2, False, Stop())
    assert len(frames) == 1 and reads == [1] and closes == [1]
    assert waits[-1] > 0.1


def test_cancelled_worker_is_owned_until_settlement_and_does_not_cancel_next_request():
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    ended = [threading.Event(), threading.Event()]
    stops = []
    analyser = video.VideoAnalyser()

    def capture(index, stop):
        stops.append((index, stop))
        entered[index].set()
        try:
            assert release[index].wait(5)
            if index == 0:
                raise RuntimeError("late failure in cancelled fake capture")
            return "second result"
        finally:
            ended[index].set()

    async def scenario():
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: errors.append(context))
        tasks = [asyncio.create_task(analyser._capture_owned(capture, index)) for index in range(2)]
        try:
            await _until(entered[0])
            await _until(entered[1])
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert len(analyser._capture_workers) == 2
            assert dict(stops)[0].is_set() and not dict(stops)[1].is_set()
            release[0].set()
            assert await asyncio.to_thread(ended[0].wait, 3)
            release[1].set()
            assert await tasks[1] == "second result"
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not analyser._capture_workers
            assert errors == []
        finally:
            for event in release:
                event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())
