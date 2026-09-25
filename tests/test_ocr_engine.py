"""
OCR engine (Mark XXVI regression).

The bug this file exists for: the neural adapters did ``_upscaled(raw).convert()``,
which assumed a PIL Image and raised ``'numpy.ndarray' has no attribute 'convert'``
on cv2 frames — a fault the chain SWALLOWED, so video/camera/numpy callers got
empty OCR while ``available`` still reported True. OCR must read text from EVERY
input type a caller might hand it, not just PIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.ocr_engine import OcrEngine  # noqa: E402


class _Bus:
    """Records fault log lines so the test can assert the chain did not swallow one."""

    def __init__(self):
        self.faults: list[str] = []

    def __getattr__(self, _name):
        bus = self

        class _Sig:
            def emit(self, *a):
                msg = " ".join(str(x) for x in a)
                if "fault" in msg.lower():
                    bus.faults.append(msg)

            def connect(self, *a):
                pass
        return _Sig()


@pytest.fixture(scope="module")
def engine():
    eng = OcrEngine(_Bus())
    if not eng.available:
        pytest.skip("no OCR backend installed in this environment")
    return eng


def _text_image(fg=(0, 0, 0), bg=(255, 255, 255), text="ORION reads this"):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (460, 100), bg)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    d.text((14, 30), text, fill=fg, font=font)
    return img


def _reads(out: str) -> bool:
    low = out.lower()
    return "orion" in low and "reads" in low


def test_status_is_coherent(engine):
    st = engine.status
    st = st() if callable(st) else st
    assert st["available"] is True
    assert st["chain"], "an available engine must list a non-empty chain"


def test_ocr_reads_a_pil_image(engine):
    assert _reads(engine.image_to_text(_text_image()))


def test_ocr_reads_a_numpy_array(engine):
    # THE regression: a cv2-style RGB numpy frame must read, not silently fail.
    bus = _Bus()
    eng = OcrEngine(bus)
    out = eng.image_to_text(np.array(_text_image()))
    assert _reads(out)
    assert bus.faults == [], f"the adapter faulted on numpy input: {bus.faults}"


def test_ocr_reads_a_grayscale_array(engine):
    assert _reads(engine.image_to_text(np.array(_text_image().convert("L"))))


def test_ocr_reads_an_rgba_array(engine):
    assert _reads(engine.image_to_text(np.array(_text_image().convert("RGBA"))))


def test_ocr_reads_light_text_on_dark(engine):
    assert _reads(engine.image_to_text(_text_image(fg=(240, 244, 252), bg=(12, 14, 20))))


def test_find_text_box_locates_a_word_on_a_numpy_frame(engine):
    box = engine.find_text_box(np.array(_text_image()), "reads")
    assert box is not None and len(box) == 4
    assert engine.find_text_box(np.array(_text_image()), "absentword") is None


def test_the_rgb_helper_normalises_every_input_type():
    # Pure: no OCR backend needed.
    from PIL import Image
    pil = Image.new("RGB", (40, 20), (0, 0, 0))
    for probe in (pil, np.zeros((20, 40, 3), np.uint8),
                  np.zeros((20, 40), np.uint8), np.zeros((20, 40, 4), np.uint8)):
        arr = OcrEngine._rgb_ndarray(probe)
        assert arr.ndim == 3 and arr.shape[2] == 3, f"bad shape for {type(probe)}"


def test_hex_codes_read_with_a_letter_o_are_corrected():
    """Windows OCR read 0x80070005 as Ox80070005 — useless to search for."""
    assert OcrEngine._clean("Error Ox80070005: Access denied") == "Error 0x80070005: Access denied"
    assert OcrEngine._clean("code 0x8OO7") == "code 0x8007"
    assert OcrEngine._clean("Oxford Street") == "Oxford Street"      # not hex


def test_wide_strips_are_padded_so_rapidocr_detects_text():
    """RapidOCR skips detection above 8:1 — a single line came back empty."""
    strip = np.full((40, 900, 3), 255, dtype=np.uint8)
    ready, top = OcrEngine._rapid_ready(strip)
    assert ready.shape[1] / ready.shape[0] <= 2.0 + 1e-6
    assert top > 0 and ready.shape[1] == 900


def test_dark_mode_is_inverted_for_rapidocr():
    dark = np.full((600, 800, 3), 25, dtype=np.uint8)
    ready, top = OcrEngine._rapid_ready(dark)
    assert top == 0 and int(ready[0, 0, 0]) == 230
