"""Tests for camera_devices.py — persisted default camera index."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import camera_devices as cd


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "CAMERA_CONFIG_PATH", tmp_path / "camera.json")
    # Not the developer's real webcams: enumeration is exercised separately.
    monkeypatch.setattr(cd, "list_cameras", lambda backend="MSMF": [])


def test_a_camera_is_remembered_by_name_and_found_at_its_new_index(tmp_path, monkeypatch):
    """Renumbering (a virtual camera appearing, a replug) must not change
    which camera ORION opens."""
    _isolate(tmp_path, monkeypatch)
    cams = {"MSMF": [cd.CameraDevice(0, "HD Pro Webcam C920", "MSMF")],
            "DSHOW": [cd.CameraDevice(0, "OBS Virtual Camera", "DSHOW"),
                      cd.CameraDevice(1, "HD Pro Webcam C920", "DSHOW")]}
    monkeypatch.setattr(cd, "list_cameras", lambda backend="MSMF": cams[backend.upper()])
    cd.set_device("C920")
    assert cd.resolve_device(hi_res=True) == (0, "CAP_MSMF")
    assert cd.resolve_device(hi_res=False) == (1, "CAP_DSHOW")


def test_a_virtual_camera_is_never_picked_automatically(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    cams = {"MSMF": [], "DSHOW": [cd.CameraDevice(0, "OBS Virtual Camera", "DSHOW"),
                                  cd.CameraDevice(1, "USB Camera", "DSHOW")]}
    monkeypatch.setattr(cd, "list_cameras", lambda backend="MSMF": cams[backend.upper()])
    assert cd.resolve_device(hi_res=True) == (1, "CAP_DSHOW")


def test_resolve_defaults_to_zero(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert cd.resolve() == 0


def test_set_device_persists_and_resolve_follows(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    message = cd.set_device(1)
    assert "1" in message
    assert cd.resolve() == 1


def test_set_device_rejects_a_negative_index(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    message = cd.set_device(-1)
    assert "must be 0 or greater" in message
    assert cd.resolve() == 0   # unchanged


def test_set_device_rejects_a_non_numeric_spec(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    message = cd.set_device("not-a-number")
    assert "not a valid" in message
    assert cd.resolve() == 0


def test_describe_reports_the_resolved_index(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    cd.set_device(2)
    assert "2" in cd.describe()


def test_open_camera_capture_tries_dshow_first(monkeypatch):
    from orion_core import utils

    calls: list[tuple] = []

    class _FakeCap:
        def __init__(self, opened, readable=True):
            self._opened = opened
            self._readable = readable

        def isOpened(self):
            return self._opened

        def read(self):
            # isOpened() is not the same as working: MSMF opens and then
            # throws on the first grab on some setups, so a capture now has
            # to deliver a frame before it is handed back.
            return (self._readable, object())

        def set(self, *_a):
            return True

        def get(self, *_a):
            return 0

        def release(self):
            pass

    class _FakeCv2:
        CAP_DSHOW = 700
        CAP_MSMF = 1400
        CAP_ANY = 0

        @staticmethod
        def VideoCapture(index, flag=None):
            calls.append((index, flag))
            return _FakeCap(opened=(flag == 700))   # only DSHOW "succeeds"

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())
    cap, backend = utils.open_camera_capture(0)
    assert backend == "CAP_DSHOW"
    assert calls[0] == (0, 700)   # DSHOW tried first, and it succeeded — no fallback needed


def test_a_backend_that_opens_but_cannot_read_is_skipped(monkeypatch):
    """The whole reason a validating read was added. A capture that reports
    itself open and then fails on the first grab used to be handed back, and
    the caller saw zero frames from a camera that looked fine."""
    from orion_core import utils

    calls: list[tuple] = []

    class _Cap:
        def __init__(self, readable):
            self._readable = readable

        def isOpened(self):
            return True

        def read(self):
            return (self._readable, object())

        def set(self, *_a):
            return True

        def get(self, *_a):
            return 0

        def release(self):
            pass

    class _Cv2:
        CAP_DSHOW = 700
        CAP_MSMF = 1400
        CAP_ANY = 0

        @staticmethod
        def VideoWriter_fourcc(*_a):
            return 0

        @staticmethod
        def VideoCapture(index, flag=None):
            calls.append((index, flag))
            return _Cap(readable=(flag == 1400))   # only MSMF can actually read

    monkeypatch.setitem(sys.modules, "cv2", _Cv2())
    _cap, backend = utils.open_camera_capture(0)
    assert backend == "CAP_MSMF", "a capture that cannot read was accepted"
    assert calls[0][1] == 700, "DSHOW should still have been tried first"


def test_high_resolution_flips_the_backend_order(monkeypatch):
    """MEASURED on a C920: DirectShow reports 1080p60 and delivers 5.1 fps
    because it stays on uncompressed YUY2; Media Foundation delivers 30.1."""
    from orion_core import utils

    calls: list[tuple] = []

    class _Cap:
        def isOpened(self):
            return True

        def read(self):
            return (True, object())

        def set(self, *_a):
            return True

        def get(self, *_a):
            return 0

        def release(self):
            pass

    class _Cv2:
        CAP_DSHOW = 700
        CAP_MSMF = 1400
        CAP_ANY = 0
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        CAP_PROP_FOURCC = 6
        CAP_PROP_BUFFERSIZE = 38

        @staticmethod
        def VideoWriter_fourcc(*_a):
            return 0

        @staticmethod
        def VideoCapture(index, flag=None):
            calls.append((index, flag))
            return _Cap()

    monkeypatch.setitem(sys.modules, "cv2", _Cv2())
    _cap, backend = utils.open_camera_capture(0, hi_res=True, size=(1920, 1080))
    assert backend == "CAP_MSMF"
    assert calls[0][1] == 1400, "MSMF must be tried first when resolution matters"
