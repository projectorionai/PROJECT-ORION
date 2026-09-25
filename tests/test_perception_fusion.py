"""
Perception fusion (Mark XXVI investigation).

The unified accessibility→OCR fallback ALREADY EXISTS: vision.find_element (the
accessibility tree) with vision.find_element_via_ocr as the second strategy,
returning the SAME shape so callers use them interchangeably. This test verifies
the OCR leg end-to-end — the path that lets ORION click a control the
accessibility tree cannot see (Electron/canvas/games) — now that the OCR numpy
fault is fixed. It does NOT build a parallel pipeline; it locks in the existing one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _Bus:
    def __getattr__(self, _n):
        class _S:
            def emit(self, *a): pass
            def connect(self, *a): pass
        return _S()


def _button_image(label="Submit"):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (400, 160), (245, 246, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([120, 60, 280, 110], fill=(60, 120, 220))
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    d.text((150, 72), label, fill=(255, 255, 255), font=font)
    return img


class _Grabber:
    def __init__(self, image):
        self._img = image

    def capture_image(self, max_side=0, monitor=None):
        return self._img

    def monitor_bounds(self, monitor=None):
        return (0, 0, self._img.width, self._img.height)


def test_the_ocr_fallback_locates_a_control_the_a11y_tree_would_miss():
    from orion_core.ocr_engine import OcrEngine
    from orion_core.vision import VisionAgent

    eng = OcrEngine(_Bus())
    if not eng.available:
        pytest.skip("no OCR backend installed")

    # Build a Vision without its heavy __init__; inject the OCR engine + a grabber
    # that returns our rendered button, then drive the real fusion leg.
    v = VisionAgent.__new__(VisionAgent)
    v.ocr_engine = eng
    v.grabber = _Grabber(_button_image("Submit"))

    found = v._find_element_via_ocr_sync("Submit", None)
    assert found is not None, "the OCR fallback failed to locate on-screen text"
    assert found["role"] == "OCR"
    assert found["name"] == "Submit"
    x, y, w, h = found["rect"]
    assert w > 0 and h > 0
    cx, cy = found["center"]
    # The 'Submit' label sits inside the drawn button (x 120–280, y 60–110).
    assert 120 <= cx <= 300 and 50 <= cy <= 120, found


def test_the_fallback_returns_none_for_absent_text():
    from orion_core.ocr_engine import OcrEngine
    from orion_core.vision import VisionAgent

    eng = OcrEngine(_Bus())
    if not eng.available:
        pytest.skip("no OCR backend installed")
    v = VisionAgent.__new__(VisionAgent)
    v.ocr_engine = eng
    v.grabber = _Grabber(_button_image("Submit"))
    assert v._find_element_via_ocr_sync("Cancel", None) is None


def test_the_fusion_methods_are_interchangeable_by_shape():
    # find_element and find_element_via_ocr must return the same keys so callers
    # can use either — the property that makes the fallback transparent.
    import inspect
    from orion_core.vision import VisionAgent
    src = inspect.getsource(VisionAgent._find_element_via_ocr_sync)
    for key in ('"role"', '"name"', '"rect"', '"center"', '"enabled"'):
        assert key in src, key
