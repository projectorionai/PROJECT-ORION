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


# ── ranking: the best match on screen, not the first in reading order ───────

import pytest

from orion_core import ocr_engine as _ocr


def _words(*items):
    return [{"text": text, "rect": rect, "confidence": 0.9} for text, rect in items]


def _only_reader(monkeypatch, engine, words, calls=None):
    def read(image):
        if calls is not None:
            calls.append(image)
        return list(words)
    monkeypatch.setattr(engine, "word_readers", lambda: (("only", read),))
    return engine


def test_an_exact_label_beats_a_longer_word_read_before_it(monkeypatch):
    """'Save' used to click 'Autosave' whenever Autosave came first (31 of 32
    rendered test screens)."""
    engine = _only_reader(monkeypatch, _engine(), _words(
        ("Autosave", (0, 0, 80, 15)), ("Save", (100, 0, 40, 15))))
    assert engine.find_text_box(_FakeImage(), "Save") == (100, 0, 40, 15)


def test_a_whole_word_inside_a_line_beats_a_longer_word(monkeypatch):
    """Line-level readers (RapidOCR) return 'File Edit View' as one box. The
    query lands on its own part of the line, not on 'Editor' and not on the
    line's middle."""
    engine = _only_reader(monkeypatch, _engine(), _words(
        ("Editor", (0, 100, 60, 16)), ("File Edit View", (200, 50, 140, 16))))
    x, y, w, h = engine.find_text_box(_FakeImage(), "Edit")
    assert (x, y, w, h) == (250, 50, 40, 16)
    assert 245 <= x + w / 2 <= 295


def test_a_longer_word_is_still_the_last_resort_aimed_at_the_matching_letters(monkeypatch):
    """Kept for words OCR ran together ("Closewindow"), aimed at the part
    that matched rather than the middle of the run."""
    engine = _only_reader(monkeypatch, _engine(), _words(("Closewindow", (0, 0, 110, 15))))
    x, y, w, h = engine.find_text_box(_FakeImage(), "Close")
    assert (x, y, h) == (0, 0, 15) and 45 <= w <= 55


def test_a_multi_word_label_never_joins_the_end_of_one_line_to_the_next(monkeypatch):
    engine = _only_reader(monkeypatch, _engine(), _words(
        ("Save", (900, 0, 40, 15)), ("As", (0, 40, 20, 15))))
    assert engine.find_text_box(_FakeImage(), "Save As") is None
    engine = _only_reader(monkeypatch, _engine(), _words(
        ("Save", (900, 0, 40, 15)), ("As", (0, 40, 20, 15)),
        ("Save", (10, 80, 40, 15)), ("As", (60, 80, 20, 15))))
    assert engine.find_text_box(_FakeImage(), "Save As") == (10, 80, 70, 15)


# ── near misses of small text ────────────────────────────────────────────────

@pytest.mark.parametrize("read,wanted,expected", [
    ("fitter", "filter", True), ("fiter", "filter", True), ("filer", "filter", True),
    ("appy", "apply", True), ("edt", "edit", True), ("0pen", "open", True),
    ("clone", "close", False), ("sale", "save", False), ("cbse", "close", False),
    ("delte", "delete", False), ("fnd", "find", True), ("ru", "run", False),
    ("filter", "filter", False),
])
def test_only_thin_stroke_slips_count_as_the_same_word(read, wanted, expected):
    assert _ocr._ocr_confusable(read, wanted) is expected


def test_a_misread_label_is_found_only_by_the_near_miss_pass(monkeypatch):
    engine = _only_reader(monkeypatch, _engine(), _words(
        ("Search", (0, 0, 60, 15)), ("Fitter", (100, 0, 50, 15))))
    assert engine.find_text_box(_FakeImage(), "Filter") is None
    assert engine.find_near_text_box(_FakeImage(), "Filter") == (100, 0, 50, 15)


@pytest.mark.parametrize("items,query", [
    ([("Fitter", (0, 0, 50, 15)), ("Fiter", (300, 0, 40, 15))], "Filter"),   # two places
    ([("Clone", (0, 0, 50, 15))], "Close"),                                  # another word
    ([("Rn", (0, 0, 20, 15))], "Run"),                                       # too short
    ([("Fitter", (0, 0, 50, 15))], "Filter now"),                            # several words
])
def test_a_near_miss_that_could_be_a_guess_is_not_offered(monkeypatch, items, query):
    engine = _only_reader(monkeypatch, _engine(), _words(*items))
    assert engine.find_near_text_box(_FakeImage(), query) is None


def test_one_screenshot_is_read_once_per_reader(monkeypatch):
    """find_text_box then find_near_text_box on the same capture used to be a
    second full recognition pass; a NEW capture is always read afresh."""
    calls: list = []
    engine = _only_reader(monkeypatch, _engine(), _words(("Fitter", (1, 2, 30, 10))), calls)
    shot = _FakeImage()
    assert engine.find_text_box(shot, "Filter") is None
    assert engine.find_near_text_box(shot, "Filter") == (1, 2, 30, 10)
    assert engine.locate_words(shot)
    assert calls == [shot]
    fresh = _FakeImage()
    engine.find_text_box(fresh, "Filter")
    assert calls == [shot, fresh]
