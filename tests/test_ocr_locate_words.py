"""
Tests for OcrEngine.locate_words/find_text_box (Mark XXI, Track D2) — the
word/phrase-level bounding boxes the click-fallback path needs. Distinct
from image_to_result/image_to_text, which return a plain text blob with no
position information — that shape is right for "read the screen", useless
for "click this label".

pytesseract is monkeypatched (no real Tesseract binary needed in CI); the
engine-detection chain is bypassed entirely since these methods call the
engine adapters directly rather than through the probed self._engines list.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.ocr_engine import OcrEngine


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _FakeImage:
    """Minimal PIL.Image stand-in — just enough for .convert()."""

    def convert(self, mode):
        return self


def _install_fake_pytesseract(monkeypatch, data: dict) -> None:
    fake = types.ModuleType("pytesseract")
    fake.Output = types.SimpleNamespace(DICT="dict")

    def image_to_data(image, output_type=None, timeout=None):
        return data

    fake.image_to_data = image_to_data
    monkeypatch.setitem(sys.modules, "pytesseract", fake)


def _engine() -> OcrEngine:
    return OcrEngine(_StubBus())


# ── locate_words via pytesseract ────────────────────────────────────────────

def test_locate_words_returns_boxes_for_recognised_words(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["", "Save", "As"],
        "conf": ["-1", "92", "88"],
        "left": [0, 10, 60],
        "top": [0, 20, 20],
        "width": [0, 40, 20],
        "height": [0, 15, 15],
    })
    words = _engine().locate_words(_FakeImage())
    assert len(words) == 2
    assert words[0] == {"text": "Save", "rect": (10, 20, 40, 15), "confidence": 0.92}
    assert words[1]["text"] == "As"


def test_locate_words_skips_blank_and_low_confidence_entries(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["", "  ", "Cancel"],
        "conf": ["-1", "-1", "80"],
        "left": [0, 0, 5],
        "top": [0, 0, 5],
        "width": [0, 0, 30],
        "height": [0, 0, 12],
    })
    words = _engine().locate_words(_FakeImage())
    assert len(words) == 1
    assert words[0]["text"] == "Cancel"


def test_locate_words_skips_zero_area_boxes(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["Ghost"], "conf": ["90"], "left": [0], "top": [0], "width": [0], "height": [10],
    })
    assert _engine().locate_words(_FakeImage()) == []


def test_locate_words_returns_empty_when_no_engine_available(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytesseract", raising=False)
    monkeypatch.delitem(sys.modules, "rapidocr_onnxruntime", raising=False)
    # Force real imports to fail by making them unresolvable.
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name in {"pytesseract", "rapidocr_onnxruntime"}:
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert _engine().locate_words(_FakeImage()) == []


# ── find_text_box: single-word queries ──────────────────────────────────────

def test_find_text_box_exact_single_word_match(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["Cancel"], "conf": ["90"], "left": [5], "top": [5], "width": [30], "height": [12],
    })
    box = _engine().find_text_box(_FakeImage(), "Cancel")
    assert box == (5, 5, 30, 12)


def test_find_text_box_is_case_insensitive(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["CANCEL"], "conf": ["90"], "left": [5], "top": [5], "width": [30], "height": [12],
    })
    assert _engine().find_text_box(_FakeImage(), "cancel") == (5, 5, 30, 12)


def test_find_text_box_with_no_match_returns_none(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["Cancel"], "conf": ["90"], "left": [5], "top": [5], "width": [30], "height": [12],
    })
    assert _engine().find_text_box(_FakeImage(), "Save") is None


def test_find_text_box_with_blank_query_returns_none(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["Cancel"], "conf": ["90"], "left": [5], "top": [5], "width": [30], "height": [12],
    })
    assert _engine().find_text_box(_FakeImage(), "   ") is None


# ── find_text_box: multi-word queries (adjacent word merging) ──────────────

def test_find_text_box_merges_two_adjacent_words(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["Save", "As"],
        "conf": ["92", "88"],
        "left": [10, 60],
        "top": [20, 20],
        "width": [40, 20],
        "height": [15, 15],
    })
    box = _engine().find_text_box(_FakeImage(), "Save As")
    # union of (10,20,40,15) and (60,20,20,15) -> x0=10,y0=20, x2=80,y2=35
    assert box == (10, 20, 70, 15)


def test_find_text_box_multi_word_query_ignores_unrelated_words_between(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["File", "Save", "As", "Exit"],
        "conf": ["90", "92", "88", "85"],
        "left": [0, 50, 100, 150],
        "top": [0, 0, 0, 0],
        "width": [30, 30, 20, 30],
        "height": [15, 15, 15, 15],
    })
    box = _engine().find_text_box(_FakeImage(), "Save As")
    assert box == (50, 0, 70, 15)


def test_find_text_box_mnemonic_and_ellipsis_are_folded_before_matching(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {
        "text": ["&Save", "As..."],
        "conf": ["90", "88"],
        "left": [10, 60], "top": [20, 20], "width": [40, 20], "height": [15, 15],
    })
    box = _engine().find_text_box(_FakeImage(), "Save As")
    assert box == (10, 20, 70, 15)


# ── graceful degradation ─────────────────────────────────────────────────────

def test_locate_words_never_raises_on_a_malformed_tesseract_response(monkeypatch):
    _install_fake_pytesseract(monkeypatch, {"text": ["Save"]})   # missing conf/left/etc
    # Falls through to the rapidocr attempt (also unavailable) and returns [].
    assert _engine().locate_words(_FakeImage()) == []
