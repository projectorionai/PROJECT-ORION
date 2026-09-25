"""One OCR engine is never enough to find a control on a real desktop.

Two faults found on 2026-09-21, both on the path ORION uses to click something
the accessibility tree cannot see.

THE CRASH. ``locate_tesseract`` reached ``pytesseract.pytesseract`` outside its
own guard, so any build not exposing that submodule raised AttributeError from
a function whose entire contract is "the path, or an empty string". It sits in
front of the word reader, and the caller had no guard either, so the lookup
aborted instead of degrading to the next engine.

THE BLIND SPOT. Tesseract binarises assuming dark text on a light background,
so text LIGHTER than its surroundings is discarded before recognition. Not a
rare case on a desktop: it is every dark-mode window, the taskbar, and most
coloured buttons. Verified across page-segmentation modes 3, 4, 6, 11 and 12,
and not recoverable by inverting the image. RapidOCR reads all of them.

MEASURED on two real 1920x1080 screens: tesseract read 79 and 346 words to
rapidocr's 60 and 135, and was four times faster on the dense one -- but
rapidocr still found 63 and 139 tokens tesseract never saw. Neither engine
dominates, so the order is a preference and the fall-through is what makes it
safe.
"""

from __future__ import annotations

import builtins
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import utils                      # noqa: E402
from orion_core.ocr_engine import OcrEngine       # noqa: E402


class _StubBus:
    def __getattr__(self, name):
        class _Signal:
            def emit(self, *a, **k):
                pass
        return _Signal()


def _engine() -> OcrEngine:
    return OcrEngine(_StubBus())


# -- locate_tesseract never raises -------------------------------------------

@pytest.fixture
def _uncached(monkeypatch):
    monkeypatch.setattr(utils, "_TESSERACT_CMD", None)


def test_a_pytesseract_without_the_submodule_returns_empty(monkeypatch, _uncached):
    """The exact shape the suite's own fake has, and the one that aborted the
    lookup rather than falling through to the next engine."""
    monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
    assert utils.locate_tesseract() == ""


def test_no_pytesseract_at_all_returns_empty(monkeypatch, _uncached):
    monkeypatch.delitem(sys.modules, "pytesseract", raising=False)
    real_import = builtins.__import__

    def _refuse(name, *a, **k):
        if name == "pytesseract":
            raise ImportError("no pytesseract")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _refuse)
    assert utils.locate_tesseract() == ""


def test_a_hostile_pytesseract_still_does_not_raise(monkeypatch, _uncached):
    """Whatever pytesseract's internals do, this returns a string."""
    fake = types.ModuleType("pytesseract")

    class _Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

        def __setattr__(self, name, value):
            raise RuntimeError("boom")

    fake.pytesseract = _Exploding()
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    assert utils.locate_tesseract() == ""


def test_the_answer_is_cached(monkeypatch, _uncached):
    """Called once per screen capture; a miss costs a PATH search plus three
    stats every time otherwise."""
    fake = types.ModuleType("pytesseract")
    fake.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    asked: list[str] = []
    real_which = shutil.which

    def _counting_which(name, *a, **k):
        asked.append(name)
        return real_which(name, *a, **k)

    monkeypatch.setattr(shutil, "which", _counting_which)
    for _ in range(3):
        utils.locate_tesseract()
    assert asked.count("tesseract") <= 1, "the miss was re-probed on every call"


# -- the fall-through --------------------------------------------------------

def _reader(*words):
    def read(image):
        return [{"text": text, "rect": rect, "confidence": 90.0}
                for text, rect in words]
    return read


def _explode(image):
    raise RuntimeError("engine died")


def test_find_text_box_tries_the_next_engine_when_the_query_is_missing(monkeypatch):
    """The condition is "found the query", not "returned anything".

    Tesseract can read 346 words off a screen and still miss the one button
    being looked for, because a button is exactly where its blindness to
    light-on-dark text bites.
    """
    engine = _engine()
    first = _reader(("Chrome", (0, 0, 60, 20)), ("Settings", (70, 0, 80, 20)))
    second = _reader(("WhatsApp", (300, 1040, 90, 24)))
    monkeypatch.setattr(engine, "word_readers",
                        lambda: (("first", first), ("second", second)))

    assert engine.find_text_box(object(), "WhatsApp") == (300, 1040, 90, 24)


def test_the_first_engine_wins_when_it_does_find_the_query(monkeypatch):
    """The slower engine must stay unpaid for in the common case."""
    engine = _engine()
    used: list[str] = []

    def first(image):
        used.append("first")
        return [{"text": "WhatsApp", "rect": (1, 2, 3, 4), "confidence": 90.0}]

    def second(image):
        used.append("second")
        return []

    monkeypatch.setattr(engine, "word_readers",
                        lambda: (("first", first), ("second", second)))
    assert engine.find_text_box(object(), "WhatsApp") == (1, 2, 3, 4)
    assert used == ["first"], "the slower engine ran although the target was found"


def test_locate_words_falls_through_on_an_empty_result(monkeypatch):
    """An engine that ran cleanly and found nothing is indistinguishable from
    one that failed, and there is nothing to lose by asking the next."""
    engine = _engine()
    monkeypatch.setattr(engine, "word_readers", lambda: (
        ("empty", lambda image: []),
        ("useful", _reader(("Submit", (5, 6, 7, 8)))),
    ))
    assert [w["text"] for w in engine.locate_words(object())] == ["Submit"]


def test_a_reader_that_raises_does_not_end_the_search(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "word_readers", lambda: (
        ("broken", _explode),
        ("useful", _reader(("Submit", (5, 6, 7, 8)))),
    ))
    assert engine.find_text_box(object(), "Submit") == (5, 6, 7, 8)
    assert engine.locate_words(object()) != []


def test_everything_failing_is_still_not_an_exception(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "word_readers", lambda: (("broken", _explode),))
    assert engine.find_text_box(object(), "Submit") is None
    assert engine.locate_words(object()) == []


def test_the_fastest_reader_leads_and_the_others_still_follow():
    """Windows OCR leads where it exists (0.2-0.4 s a 1080p screen against
    tesseract's 2.3 s, and it reads dark-mode text); tesseract leads where it
    does not. Either way rapidocr stays in the chain as the fall-through."""
    from orion_core import ocr_engine

    engine = _engine()
    names = [name for name, _ in engine.word_readers()]
    expected_first = ("Windows OCR" if ocr_engine._winrt_modules() is not None
                      else "pytesseract")
    assert names[0] == expected_first
    assert "pytesseract" in names
    assert "rapidocr-onnxruntime" in names, (
        "rapidocr is the fall-through for light-on-dark text")


# -- the real thing ----------------------------------------------------------

def _button(background, foreground, label="Submit"):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (400, 160), (245, 246, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([120, 60, 280, 110], fill=background)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((150, 72), label, fill=foreground, font=font)
    return image


@pytest.mark.parametrize("background,foreground,description", [
    ((245, 246, 250), (20, 20, 20), "dark text on a light background"),
    ((60, 120, 220), (255, 255, 255), "a white label on a coloured button"),
    ((32, 33, 36), (230, 230, 230), "a dark-mode window"),
])
def test_a_control_is_found_whatever_way_round_it_is_drawn(
        background, foreground, description):
    """The last two returned None before the fall-through existed, and they
    are most of a modern desktop."""
    engine = _engine()
    if not engine.available:
        pytest.skip("no OCR backend installed")
    box = engine.find_text_box(_button(background, foreground), "Submit")
    assert box is not None, f"could not read {description}"
    x, y, w, h = box
    assert 120 <= x + w // 2 <= 300 and 50 <= y + h // 2 <= 120, box
