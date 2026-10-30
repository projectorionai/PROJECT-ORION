"""
OCR & vision engine (Phase 8, Section B) — confidence-scored, multi-engine.

Redesigned from a single-backend reader into a fallback CHAIN with real
confidence scoring and OpenCV preprocessing:

    OpenCV preprocess  →  Engine 1  →  (confidence < threshold?)  →  Engine 2  →  …

Every engine returns ``(text, confidence)``; confidence is None when the
engine supplies no score. Usable Windows OCR text retains its native-first
preference without claiming a measured score. Other readers accept the first
scored result above the threshold, or keep the best available fallback.
Unavailable confidence is stated explicitly:

    OCR Result
    Confidence: 98.3%
    Engine Used: rapidocr-onnxruntime

Engine order is "best signal first, least setup first" given what is actually
installed.  The directive's EasyOCR / PaddleOCR / Tesseract are all supported
if present; they are heavy (PyTorch) optional installs documented in the
dependency audit, so the default relies on RapidOCR (ONNX, bundled models,
per-box confidence) and the Windows WinRT engine — both offline, both here.

OpenCV preprocessing (grayscale → upscale small crops → denoise → adaptive
threshold) markedly improves recognition of small UI text; it degrades to the
raw image when OpenCV is absent.  All recognition is synchronous — call it
through ``asyncio.to_thread``.

RapidOCR's text detection and recognition run on the GPU (DirectML) when
accelerator.py finds one, its angle classifier on the CPU (measured there).
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .bus import OrionBus


# ── the Windows OCR engine (Windows.Media.Ocr) ───────────────────────────────
#
# Built into Windows 10/11: no model download, no separate install, fast, and
# — unlike tesseract — it reads light text on dark backgrounds, which is most
# of a modern desktop. It never worked here before: the adapter imported
# `winsdk`, which was never installed and has no Python 3.13 build. The
# maintained `winrt-*` packages are tried first.

_WINRT: Any = None


def _winrt_modules() -> Any:
    """(OcrEngine, BitmapDecoder, InMemoryRandomAccessStream, DataWriter), or None."""
    global _WINRT
    if _WINRT is not None:
        return _WINRT or None
    for prefix in ("winrt.windows", "winsdk.windows"):
        try:
            import importlib

            ocr = importlib.import_module(f"{prefix}.media.ocr")
            imaging = importlib.import_module(f"{prefix}.graphics.imaging")
            streams = importlib.import_module(f"{prefix}.storage.streams")
            _WINRT = (ocr.OcrEngine, imaging.BitmapDecoder,
                      streams.InMemoryRandomAccessStream, streams.DataWriter)
            return _WINRT
        except Exception:
            continue
    _WINRT = False
    return None


def _winrt_recognise(image: Any) -> list[list[dict[str, Any]]]:
    """Lines of {"text", "rect", "confidence"} words, in *image*'s pixel space.

    Synchronous: runs its own event loop, in a helper thread when the caller
    is already inside one (a running loop cannot be re-entered). Returns []
    on any failure.
    """
    modules = _winrt_modules()
    if modules is None:
        return []
    ocr_engine, decoder_cls, stream_cls, writer_cls = modules
    import asyncio
    import io

    from PIL import Image

    if not isinstance(image, Image.Image):
        try:
            import numpy as np
            image = Image.fromarray(np.asarray(image)[:, :, :3][:, :, ::-1])   # BGR frame
        except Exception:
            return []
    rgb = image.convert("RGB")
    limit = int(getattr(ocr_engine, "max_image_dimension", 10000) or 10000)
    # Read at twice the size. MEASURED on rendered UI text: 12 px Segoe UI
    # went from 0.79 accuracy to 0.99 (tesseract: 0.92), light-on-dark
    # included, and the call still takes tens of milliseconds. Boxes are
    # mapped back to the caller's pixels below.
    scale = min(2.0, limit / max(1, max(rgb.size)))
    if abs(scale - 1.0) > 0.01:
        rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))),
                         Image.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="BMP")          # uncompressed: encodes in ~10 ms
    payload = buffer.getvalue()

    async def _recognise() -> list[list[dict[str, Any]]]:
        stream = stream_cls()
        # The writer must be attached to the stream and flushed — a detached
        # DataWriter leaves the stream empty and the decoder fails every call.
        writer = writer_cls(stream.get_output_stream_at(0))
        writer.write_bytes(payload)
        await writer.store_async()
        await writer.flush_async()
        decoder = await decoder_cls.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        engine = ocr_engine.try_create_from_user_profile_languages()
        if engine is None:
            return []
        result = await engine.recognize_async(bitmap)
        # Word boxes are only in image space when TextAngle is 0; otherwise
        # they are in the frame Windows straightened the text into. It
        # decides short lines of perfectly level UI text are rotated — 7°
        # for a three-word label — which put every box ~50 px low and
        # sloping. Each box's centre is turned back, clockwise by the angle
        # about the image centre (y down), keeping the word's own size.
        try:
            angle = float(result.text_angle or 0.0)
        except (TypeError, ValueError):
            angle = 0.0
        cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        mid_x, mid_y = bitmap.pixel_width / 2.0, bitmap.pixel_height / 2.0
        lines: list[list[dict[str, Any]]] = []
        for line in result.lines:
            words = []
            for word in line.words:
                box = word.bounding_rect
                left, top = box.x, box.y
                if abs(angle) >= 0.05:
                    dx = box.x + box.width / 2.0 - mid_x
                    dy = box.y + box.height / 2.0 - mid_y
                    left = mid_x + dx * cos_a - dy * sin_a - box.width / 2.0
                    top = mid_y + dx * sin_a + dy * cos_a - box.height / 2.0
                rect = (max(0, int(round(left / scale))), max(0, int(round(top / scale))),
                        max(1, int(box.width / scale)), max(1, int(box.height / scale)))
                # Windows OCR exposes text and boxes, but no confidence score.
                words.append({"text": str(word.text), "rect": rect,
                              "confidence": None, "confidence_source": "unavailable"})
            if words:
                lines.append(words)
        return lines

    def _run() -> list[list[dict[str, Any]]]:
        try:
            return asyncio.run(_recognise())
        except Exception:
            return []

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run()
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=30)


# ── RapidOCR, its sessions placed by accelerator.py ──────────────────────────

_RAPID_LOCK = threading.RLock()
_WORD_CACHE_LOCK = threading.Lock()

#: RapidOCR's three models, by the file names it ships, as accelerator models.
_RAPID_MODELS = (("_det", "ocr_det"), ("_rec", "ocr_rec"), ("_cls", "ocr_cls"))


def _installed(name: str) -> bool:
    """Whether *name* can be imported, without importing it."""
    if sys.modules.get(name) is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _rapid_model(path: Any) -> str:
    name = Path(str(path)).name.lower()
    for marker, model in _RAPID_MODELS:
        if marker in name:
            return model
    return "ocr_" + Path(name).stem        # not in the policy: the CPU


def _build_rapidocr() -> Any:
    """A RapidOCR engine whose sessions accelerator.create_session built.

    RapidOCR's only device switch is CUDA, and it makes all three sessions
    through one module-level name, rapidocr_onnxruntime.utils.InferenceSession,
    which points at the accelerator while the engine is built (under a lock,
    so two engines never race the swap). Blocking: ~0.3 s on the CPU, ~0.6 s
    with the GPU, most of it holding Python's lock - call it in a worker."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
    try:
        from rapidocr_onnxruntime import utils as rapid_utils  # type: ignore
    except Exception:
        rapid_utils = None
    original = getattr(rapid_utils, "InferenceSession", None)
    if original is None:
        return RapidOCR()
    from . import accelerator

    def session(path: Any, sess_options: Any = None, providers: Any = None,
                **_ignored: Any) -> Any:
        # RapidOCR keeps its CPU memory arena off; the rest is the accelerator's.
        return accelerator.create_session(
            _rapid_model(path), path,
            configure=lambda options: setattr(options, "enable_cpu_mem_arena", False))

    with _RAPID_LOCK:
        rapid_utils.InferenceSession = session
        try:
            return RapidOCR()
        finally:
            rapid_utils.InferenceSession = original


# ── matching a label among recognised words ──────────────────────────────────

_WORD = re.compile(r"\w+")


def _tokens(folded: str) -> list[str]:
    return _WORD.findall(folded)


def _token_span(folded: str, query_tokens: list[str]) -> tuple[int, int] | None:
    """Character span of *query_tokens* as WHOLE consecutive words of
    *folded* ("view" in "file edit view", not "view" in "preview")."""
    pattern = r"(?<!\w)" + r"\W+".join(re.escape(t) for t in query_tokens) + r"(?!\w)"
    found = re.search(pattern, folded)
    return found.span() if found else None


def _part_of_box(rect: tuple[int, int, int, int], folded: str,
                 span: tuple[int, int]) -> tuple[int, int, int, int]:
    """The part of a line's box holding characters *span* of its text, by
    their share of the line (proportional type is close enough to land the
    centre on the word)."""
    x, y, w, h = rect
    length = max(1, len(folded))
    start, end = span
    if start <= 0 and end >= length:
        return rect
    left = x + int(round(w * start / length))
    right = x + int(round(w * end / length))
    return (left, y, max(1, right - left), h)


def _same_line(first: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    """Whether two word boxes sit on one text line (their middles within
    about half a line height of each other)."""
    middle_a = first[1] + first[3] / 2.0
    middle_b = other[1] + other[3] / 2.0
    return abs(middle_a - middle_b) <= max(first[3], other[3]) * 0.6


#: Glyphs OCR drops or swaps on small UI text: the thin vertical strokes
#: (measured on 10-11 px Segoe UI / Arial: "Filter" read as "Fitter",
#: "Fiter" and "Filer", "Apply" as "Appy", "Edit" as "Edt"), and o/0.
_THIN = frozenset("iltj1|!fr")
_ROUND = frozenset("o0")


def _ocr_confusable(read: str, wanted: str) -> bool:
    """*read* is *wanted* with exactly one OCR slip on a thin stroke: one
    such glyph missing or extra, or swapped for another of its kind. Only
    for words of four letters or more, and never a different letter shape
    ("Close" is not "Clone", "Save" is not "Sale")."""
    if len(wanted) < 4 or read == wanted or abs(len(read) - len(wanted)) > 1:
        return False
    if len(read) == len(wanted):
        diffs = [(a, b) for a, b in zip(read, wanted) if a != b]
        if len(diffs) != 1:
            return False
        a, b = diffs[0]
        return (a in _THIN and b in _THIN) or (a in _ROUND and b in _ROUND)
    longer, shorter = (read, wanted) if len(read) > len(wanted) else (wanted, read)
    for i, ch in enumerate(longer):
        if longer[:i] + longer[i + 1:] == shorter:
            return ch in _THIN
    return False


@dataclass
class OcrResult:
    text: str
    confidence: float | None  # engine-provided 0..1 score, or unavailable
    engine: str
    confidence_source: str = "engine"

    def __post_init__(self) -> None:
        if self.confidence is None or not self.engine:
            self.confidence_source = "unavailable"

    def report(self) -> str:
        confidence = ("unavailable (the engine supplied no score)" if self.confidence is None
                      or self.confidence_source == "unavailable" else
                      f"{self.confidence * 100:.1f}% (engine-provided score)")
        return (f"OCR Result\nConfidence: {confidence}\n"
                f"Engine Used: {self.engine or 'none'}\n\n{self.text}")


# A recognised engine: name → callable(preprocessed_pil, raw_pil) → (text, conf)
_Engine = tuple[str, Callable[[Any, Any], tuple[str, float | None]]]


class OcrEngine:
    """Confidence-scored OCR with an automatic multi-engine fallback chain."""

    # Below this confidence, try the next engine (but keep the best result).
    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._engines: list[_Engine] = []
        self._checked = False
        self._name = ""            # primary engine name (compat)

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect(self) -> None:
        if self._checked:
            return
        self._checked = True
        # Tesseract leads when it is installed. MEASURED on two real 1920x1080
        # screens, same captures, same machine:
        #
        #                      sparse screen        dense screen (an IDE)
        #   rapidocr           2176 ms, 124 words   10349 ms, 552 words
        #   pytesseract        2663 ms, 153 words    2399 ms, 598 words
        #
        # Four times faster on the dense case and reading MORE of it, which is
        # the case that matters: the dense screen is the one somebody asks
        # ORION to find something on. RapidOCR stays second because it needs
        # no separate install and is the only engine on a machine without
        # tesseract — the ordering is a preference, not a requirement.
        # Mark XXXI: the Windows OCR engine leads. MEASURED on the same two
        # 1920x1080 screens: 0.18-0.40 s against tesseract's 2.3-2.5 s and
        # rapidocr's 9+ s; and on rendered 12-15 px UI text (read at 2x) it
        # scored 0.97-0.99 against tesseract's 0.90-0.98, light-on-dark
        # included. It was always meant to be in this chain and never ran —
        # the adapter imported a package that was not installed.
        probes = (
            ("Windows OCR", self._probe_winrt),
            ("pytesseract", self._probe_pytesseract),
            ("rapidocr-onnxruntime", self._probe_rapidocr),
            ("easyocr", self._probe_easyocr),
            ("paddleocr", self._probe_paddleocr),
        )
        for name, probe in probes:
            try:
                fn = probe()
                if fn is not None:
                    self._engines.append((name, fn))
            except Exception:
                continue
        if self._engines:
            self._name = self._engines[0][0]
            names = ", ".join(n for n, _ in self._engines)
            self.bus.log.emit(f"OCR: engine chain ready — {names} "
                              f"(primary: {self._name}).")
        else:
            self.bus.log.emit(
                "OCR: no local OCR engine installed. Install one with:  "
                "pip install rapidocr-onnxruntime  (pure pip, bundled models). "
                "Screen text otherwise falls back to the multimodal channel."
            )

    # ── OpenCV preprocessing ──────────────────────────────────────────────────

    def _preprocess(self, image: Any) -> Any:
        """Grayscale → upscale small crops → denoise → adaptive threshold."""
        try:
            import cv2  # type: ignore
            import numpy as np
            arr = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if float(np.median(gray)) < 110:
                # Dark mode: Tesseract wants dark text on light. Thresholding
                # light-on-dark as-is read 16 px dark UI text at 7% (measured).
                gray = 255 - gray
            h, w = gray.shape[:2]
            # Upscale small captures — OCR accuracy on tiny UI text improves a lot.
            if max(h, w) < 1000:
                scale = min(3.0, 1000.0 / max(1, max(h, w)))
                gray = cv2.resize(gray, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            gray = cv2.bilateralFilter(gray, 5, 40, 40)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 11)
            from PIL import Image
            return Image.fromarray(thresh)
        except Exception:
            return image     # OpenCV absent or a colour-critical image: use raw

    @staticmethod
    def _upscaled(image: Any) -> Any:
        """Gently upscale small captures so neural OCR sees legible glyphs
        (no thresholding — neural engines want the natural image)."""
        try:
            w, h = image.size
            if max(w, h) < 1000:
                scale = min(3.0, 1000.0 / max(1, max(w, h)))
                from PIL import Image
                return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        except Exception:
            pass
        return image

    @staticmethod
    def _rgb_ndarray(image: Any, upscale: bool = True) -> Any:
        """An RGB numpy array, whether *image* is a PIL Image OR already a numpy
        array.

        The neural adapters previously did ``_upscaled(raw).convert("RGB")``,
        which assumed a PIL Image and threw ``'numpy.ndarray' has no attribute
        'convert'`` on cv2 frames (video/camera) — a fault the chain swallowed,
        so those callers got empty OCR with no error. This normalises both
        inputs to what RapidOCR/EasyOCR/Paddle expect.

        ``upscale`` enlarges small captures for better RECOGNITION — but callers
        that need the returned bounding-box COORDINATES to map back onto the
        original image (word location for a click) must pass upscale=False, or the
        boxes come back in the enlarged space and the click lands ~scale× off."""
        import numpy as np
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:                                 # grayscale → RGB
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:          # RGBA → RGB
                arr = arr[:, :, :3]
        else:
            try:
                arr = np.array(image.convert("RGB"))
            except Exception:
                arr = np.array(image)
        if upscale:
            try:
                import cv2  # type: ignore
                h, w = arr.shape[:2]
                if max(h, w) < 1000:
                    scale = min(3.0, 1000.0 / max(1, max(h, w)))
                    arr = cv2.resize(arr, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
            except Exception:
                pass
        return arr

    # ── the fallback chain ────────────────────────────────────────────────────

    def image_to_result(self, image: Any) -> OcrResult:
        """Run the engine chain; return the best {text, confidence, engine}."""
        self._detect()
        if not self._engines:
            return OcrResult("", 0.0, "")
        pre = self._preprocess(image)
        best = OcrResult("", 0.0, "")
        unscored = None
        for name, fn in self._engines:
            try:
                text, conf = fn(pre, image)
            except Exception as exc:
                self.bus.log.emit(f"OCR: {name} fault - {str(exc).splitlines()[0][:80]}")
                continue
            text = self._clean(text)
            if not text:
                continue
            try:
                conf = float(conf) if conf is not None else None
                if conf is not None and (not math.isfinite(conf) or not 0 <= conf <= 1):
                    conf = None
            except (TypeError, ValueError):
                conf = None
            if conf is None:
                if name == "Windows OCR":
                    # Native-first is an engine preference, not a fabricated
                    # confidence threshold. Preserve the fast local read;
                    # heavier engines remain fallbacks for empty/failed reads.
                    return OcrResult(text, None, name)
                # Keep native text useful without inventing a threshold score.
                # A measured good read may supersede it; a bad/failed fallback
                # must not erase it merely because None is not comparable.
                if unscored is None:
                    unscored = OcrResult(text, None, name)
                continue
            if not best.text or conf > best.confidence:
                best = OcrResult(text, conf, name)
            if conf >= self.CONFIDENCE_THRESHOLD:
                return best        # good enough — stop, spare the slower engines
        return unscored or best

    def image_to_text(self, image: Any) -> str:
        """Compatibility: just the recognised text (best engine)."""
        return self.image_to_result(image).text

    # ── word-level localisation (Mark XXI, Track D2) ────────────────────────
    #
    # image_to_result/image_to_text return a TEXT BLOB — right for "read the
    # screen", useless for "click this label", which needs a POSITION.
    # Both engine adapters above already compute a per-word/per-line
    # bounding box internally before discarding it down to a plain string;
    # these methods keep that box instead, as a real fallback for the click
    # path when the accessibility tree comes back empty (Electron/canvas
    # apps, games, some installers) — a case that previously had no second
    # strategy at all and simply failed.

    def word_readers(self) -> tuple[tuple[str, Any], ...]:
        """The per-word readers, in the order worth trying.

        Ordered by speed, NOT by quality, because neither engine dominates.
        MEASURED on this machine's two real 1920x1080 screens:

                              sparse desktop        dense screen (an IDE)
          pytesseract          79 words, 1954 ms    346 words, 1873 ms
          rapidocr             60 lines, 2368 ms    135 lines, 8960 ms
          found ONLY by rapidocr    63 tokens            139 tokens

        Tesseract reads far more overall and is four times faster on the
        dense case, so it goes first. But rapidocr still finds a hundred-odd
        tokens it misses, because the two fail in different places:
        **tesseract is structurally blind to light text on a dark
        background.** It binarises assuming dark-on-light, so text lighter
        than its surroundings is thrown away before recognition — verified
        across every page-segmentation mode, and not recoverable by
        inverting the image. On a desktop full of dark-mode windows, a
        taskbar and coloured buttons, that is not an edge case.

        So the order is a preference and the fall-through is what makes it
        safe. RapidOCR is also the only engine on a machine with no separate
        tesseract install.
        """
        readers: list[tuple[str, Any]] = []
        if _winrt_modules() is not None:
            # Fastest by an order of magnitude and reads dark-mode text; see
            # _detect. Tesseract and rapidocr still follow, because a label
            # one engine misses another often reads.
            readers.append(("Windows OCR", self._locate_words_winrt))
        readers += [
            ("pytesseract", self._locate_words_pytesseract),
            ("rapidocr-onnxruntime", self._locate_words_rapidocr),
        ]
        return tuple(readers)

    def locate_words(self, image: Any,
                     readers: Any = None) -> list[dict[str, Any]]:
        """Word/phrase-level {"text", "rect", "confidence"} boxes, in the
        same pixel space as *image*.

        The first reader that returns anything wins. An engine that ran
        cleanly and found NOTHING hands over too, not just one that raised:
        from here those two outcomes are indistinguishable, and when the
        first engine came up empty there is nothing to lose by asking the
        second. Never raises — returns [] if every reader fails.
        """
        for name, read in (self.word_readers() if readers is None
                           else tuple(readers)):
            try:
                words = self._read_words(name, read, image)
            except Exception:
                continue
            if words:
                return words
        return []

    def find_text_box(self, image: Any, query: str,
                      readers: Any = None) -> tuple[int, int, int, int] | None:
        """The bounding box of the OCR word/phrase best matching *query*, or
        None.

        Each reader is tried until one FINDS THE QUERY — which is a stronger
        condition than locate_words' "returned anything", and deliberately.
        Tesseract can read 346 words off a screen and still miss the one
        label being looked for, because the thing being clicked is usually a
        button or a taskbar icon, which is exactly where its blindness to
        light-on-dark text bites. Stopping at the first reader that produced
        *some* output would mean ORION reporting he cannot find a control
        that the next engine reads perfectly well.

        The extra engine costs nothing in the common case: it only runs when
        the target has not been found, which is the case that was otherwise
        about to fail outright.

        *readers* restricts the sweep to a subset of word_readers(). It
        exists for the caller searching SEVERAL screens, which wants the
        fast reader tried on every screen before the slow one is paid for
        on any of them.
        """
        from .utils import fold_control_label
        query_folded = fold_control_label(query)
        if not query_folded:
            return None
        for name, read in (self.word_readers() if readers is None
                           else tuple(readers)):
            try:
                words = self._read_words(name, read, image)
            except Exception:
                continue
            if not words:
                continue
            hit = self._match_in_words(words, query_folded)
            if hit is not None:
                return hit
        return None

    def find_near_text_box(self, image: Any, query: str,
                           readers: Any = None) -> tuple[int, int, int, int] | None:
        """The last resort after find_text_box found nothing: the one word on
        screen that is *query* with a single OCR slip on a thin stroke
        ("Fitter" for "Filter", "Appy" for "Apply" - see _ocr_confusable).

        Only single words of four letters or more, and only when exactly one
        place on screen qualifies; two candidates is a guess, and a guess is
        not clicked. Uses the words find_text_box already read from this same
        image, so it costs no second recognition pass."""
        from .utils import fold_control_label
        query_folded = fold_control_label(query)
        wanted = _tokens(query_folded)
        if len(wanted) != 1 or len(wanted[0]) < 4:
            return None
        for name, read in (self.word_readers() if readers is None
                           else tuple(readers)):
            try:
                words = self._read_words(name, read, image)
            except Exception:
                continue
            hits: list[tuple[int, int, int, int]] = []
            for w in words or ():
                folded = fold_control_label(w["text"])
                for token in _WORD.finditer(folded):
                    if _ocr_confusable(token.group(0), wanted[0]):
                        box = _part_of_box(w["rect"], folded, token.span())
                        centre = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                        if not any(abs(centre[0] - (b[0] + b[2] / 2)) < 4
                                   and abs(centre[1] - (b[1] + b[3] / 2)) < 4 for b in hits):
                            hits.append(box)
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                return None
        return None

    #: Images whose words are remembered (see _read_words).
    WORD_CACHE_IMAGES = 6

    def _read_words(self, name: str, read: Any, image: Any) -> list[dict[str, Any]]:
        """*read*'s words for *image*, recognised once while that image lives.

        A click search asks the same screenshot for one label with each
        reader, then (find_near_text_box) for a near miss: without this every
        question was a fresh recognition pass - 2-9 s a screen for the slower
        readers. Keyed on the image object itself (a weak reference, so a new
        screenshot is always read afresh); objects that cannot be weakly
        referenced are simply not remembered."""
        import weakref
        cache = self.__dict__.setdefault("_word_cache", {})
        key = id(image)
        entry = cache.get(key)
        if entry is not None and entry[0]() is image and name in entry[1]:
            return entry[1][name]
        words = read(image)
        try:
            ref = weakref.ref(image)
        except TypeError:
            return words
        with _WORD_CACHE_LOCK:
            entry = cache.get(key)
            if entry is None or entry[0]() is not image:
                entry = (ref, {})
                cache[key] = entry
                while len(cache) > self.WORD_CACHE_IMAGES:
                    cache.pop(next(iter(cache)))
            entry[1][name] = words
        return words

    def _match_in_words(self, words: list[dict[str, Any]],
                        query_folded: str) -> tuple[int, int, int, int] | None:
        """Where *query_folded* sits among already-recognised *words*.

        Best match first, wherever it is on screen:

            1. an item that IS the query ("Save");
            2. the query as whole words inside one item — RapidOCR's and
               some Windows OCR output is per LINE, so "View" inside "File
               Edit View" — answered with that part of the line's box, not
               the whole line (the line's centre was another menu entry);
            3. consecutive items joined on one line (pytesseract returns
               per-WORD boxes, so "Save As" needs two boxes unioned);
            4. the query inside a longer word, the old rule, kept last (and
               aimed at the matching letters).

        Reading order used to decide alone: "Save" clicked "Autosave"
        whenever Autosave came first, and "Edit" landed on "Editor"
        (measured on rendered UI text: 31 of 32 screens chose Autosave).
        """
        from .utils import fold_control_label
        query_tokens = _tokens(query_folded)
        token_hit = substring_hit = None
        for w in words:
            folded = fold_control_label(w["text"])
            if not folded:
                continue
            if folded == query_folded:
                return w["rect"]
            if token_hit is None and query_tokens:
                span = _token_span(folded, query_tokens)
                if span is not None:
                    token_hit = _part_of_box(w["rect"], folded, span)
            if substring_hit is None and query_folded in folded:
                # Words OCR ran together ("Closewindow"): aim at the part
                # that matched, not the middle of the run.
                start = folded.index(query_folded)
                substring_hit = _part_of_box(w["rect"], folded,
                                             (start, start + len(query_folded)))
        if token_hit is not None:
            return token_hit
        joined = self._join_adjacent(words, query_folded)
        return joined if joined is not None else substring_hit

    def _join_adjacent(self, words: list[dict[str, Any]],
                       query_folded: str) -> tuple[int, int, int, int] | None:
        """A multi-word query spread over consecutive items of ONE line."""
        from .utils import fold_control_label
        query_tokens = query_folded.split()
        if len(query_tokens) < 2:
            return None
        # Multi-word: only start a joining window at a word that could
        # plausibly BE the query's first token — otherwise a match can
        # accidentally span back into an unrelated preceding word (e.g.
        # "File Save As" would wrongly swallow "File" into the box for a
        # "Save As" query, since "file save as" does contain "save as" as
        # a raw substring).
        first_token = query_tokens[0]
        window = len(query_tokens) + 2
        for i, w in enumerate(words):
            piece0 = fold_control_label(w["text"])
            if not piece0 or not (piece0 == first_token
                                  or piece0.startswith(first_token)
                                  or first_token.startswith(piece0)):
                continue
            joined = ""
            rects: list[tuple[int, int, int, int]] = []
            for j in range(i, min(i + window, len(words))):
                piece = fold_control_label(words[j]["text"])
                if not piece:
                    continue
                if rects and not _same_line(rects[0], words[j]["rect"]):
                    # The end of one line and the start of the next are not
                    # one label: their union is a box spanning the screen.
                    break
                joined = (joined + " " + piece).strip()
                rects.append(words[j]["rect"])
                if query_folded in joined:
                    xs = [r[0] for r in rects]
                    ys = [r[1] for r in rects]
                    x2s = [r[0] + r[2] for r in rects]
                    y2s = [r[1] + r[3] for r in rects]
                    x0, y0 = min(xs), min(ys)
                    return (x0, y0, max(x2s) - x0, max(y2s) - y0)
        return None

    @staticmethod
    def _gray_for_tesseract(image: Any) -> Any:
        """*image* as a grayscale PIL Image, from a PIL Image OR a numpy array.

        Both tesseract adapters used to call ``image.convert("L")`` straight
        off, which assumes a PIL Image and raises ``'numpy.ndarray' object has
        no attribute 'convert'`` on a cv2 frame — every camera frame, and every
        caller that grabs the screen through OpenCV. The chain swallowed that
        as a fault and moved on, so it cost a wasted recognition pass and a log
        line rather than an error, which is why it survived: tesseract only
        became the FIRST reader on 2026-09-21, and until then almost nothing
        reached this.

        Size is left alone deliberately. The word reader's bounding boxes have
        to map back onto the original image for a click to land, and scaling
        here would put them in a different space — the trap _rgb_ndarray's
        ``upscale`` flag documents.
        """
        if hasattr(image, "convert"):
            return image.convert("L")
        import numpy as np
        from PIL import Image as _Image

        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[:, :, :3]
        return _Image.fromarray(arr.astype("uint8")).convert("L")

    def _locate_words_pytesseract(self, image: Any) -> list[dict[str, Any]]:
        from .utils import locate_tesseract

        locate_tesseract()
        import pytesseract  # type: ignore
        data = pytesseract.image_to_data(
            self._gray_for_tesseract(image),
            output_type=pytesseract.Output.DICT, timeout=25)
        words: list[dict[str, Any]] = []
        count = len(data.get("text", []))
        for i in range(count):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            if w <= 0 or h <= 0:
                continue
            words.append({"text": text, "rect": (x, y, w, h), "confidence": conf / 100.0})
        return words

    def _rapid(self) -> Any:
        """The RapidOCR engine, built once by the first read that needs it and
        shared by the text reader and the word locator.

        Loading the three models used to happen on EVERY word lookup, then in
        _detect - which runs wherever `available` is first asked, the GUI
        thread included (vision.find_element_via_ocr). Reads run in workers."""
        engine = getattr(self, "_rapid_engine", None)
        if engine is None:
            with _RAPID_LOCK:
                engine = getattr(self, "_rapid_engine", None)
                if engine is None:
                    engine = self._rapid_engine = _build_rapidocr()
        return engine

    def release(self) -> bool:
        """Let RapidOCR's three models go when memory is short (Mark XXXX.4);
        a read in progress keeps its own reference, the next read rebuilds."""
        with _RAPID_LOCK:
            held = getattr(self, "_rapid_engine", None) is not None
            self._rapid_engine = None
        return held

    def _locate_words_rapidocr(self, image: Any) -> list[dict[str, Any]]:
        engine = self._rapid()
        # upscale=False: these boxes are used to locate/click text on the real
        # image, so the coordinates must stay in the original image's space —
        # and the padding _rapid_ready adds above is subtracted back out.
        ready, top = self._rapid_ready(self._rgb_ndarray(image, upscale=False))
        result, _ = engine(ready)
        words: list[dict[str, Any]] = []
        for line in (result or []):
            if len(line) < 2:
                continue
            box, text = line[0], str(line[1]).strip()
            if not text:
                continue
            conf = float(line[2]) if len(line) > 2 and line[2] is not None else None
            try:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x0, y0 = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - x0), int(max(ys) - y0)
                y0 -= top
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            item = {"text": text, "rect": (x0, y0, w, h), "confidence": conf}
            if conf is None:
                item["confidence_source"] = "unavailable"
            words.append(item)
        return words

    # ── engine adapters (each returns (text, confidence 0..1)) ────────────────

    def _probe_rapidocr(self) -> Optional[Callable[[Any, Any], tuple[str, float | None]]]:
        # Installed is enough here: the engine itself is built by the first
        # read (_rapid), in that read's worker thread, never in _detect.
        if not (_installed("rapidocr_onnxruntime") and _installed("onnxruntime")):
            return None

        def run(pre: Any, raw: Any) -> tuple[str, float | None]:
            engine = self._rapid()
            # Neural OCR reads the NATURAL image best; binarisation is for
            # Tesseract.  _rgb_ndarray upscales AND accepts PIL or numpy frames.
            ready, _top = self._rapid_ready(self._rgb_ndarray(raw))
            result, _ = engine(ready)
            if not result:
                return ("", 0.0)
            items, scores = [], []
            for line in result:
                if len(line) < 2:
                    continue
                box, text = line[0], str(line[1])
                score = float(line[2]) if len(line) > 2 and line[2] is not None else None
                scores.append(score)
                try:
                    ys = [p[1] for p in box]; xs = [p[0] for p in box]
                    items.append((min(ys), min(xs), text))
                except Exception:
                    items.append((0.0, 0.0, text))
            items.sort(key=lambda it: (round(it[0] / 12), it[1]))
            text = "\n".join(t for _y, _x, t in items)
            conf = (sum(scores) / len(scores) if scores and all(s is not None for s in scores)
                    else None)
            return (text, conf)
        return run

    def _probe_easyocr(self) -> Optional[Callable[[Any, Any], tuple[str, float]]]:
        import easyocr  # type: ignore
        # Recognition is not setup authorization. EasyOCR honours its existing
        # cache/env paths with downloads disabled, or reports missing models.
        try:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False, download_enabled=False)
        except FileNotFoundError:
            self.bus.log.emit("OCR: EasyOCR needs locally installed models; automatic downloads are disabled.")
            return None

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            results = reader.readtext(self._rgb_ndarray(raw), detail=1)
            if not results:
                return ("", 0.0)
            texts = [str(r[1]) for r in results]
            confs = [float(r[2]) for r in results if len(r) > 2]
            conf = sum(confs) / len(confs) if confs else 0.0
            return ("\n".join(texts), conf)
        return run

    def _probe_paddleocr(self) -> Optional[Callable[[Any, Any], tuple[str, float]]]:
        import importlib
        from importlib import metadata
        from pathlib import Path
        from urllib.parse import urlsplit

        # The legacy adapter below targets PaddleOCR 2.x. Unlike EasyOCR, that
        # constructor has no no-download switch: all three inference model
        # directories must already contain their model/weights. Do not import
        # or initialise a different API and hope its defaults stay offline.
        version = metadata.version("paddleocr")
        if not version.startswith("2."):
            self.bus.log.emit("OCR: PaddleOCR's installed API is not supported by the offline legacy adapter.")
            return None
        from paddleocr import PaddleOCR  # type: ignore
        implementation = importlib.import_module(PaddleOCR.__module__)
        directories = {}
        for kind, language in (("det", "en"), ("rec", "en"), ("cls", "ch")):
            config = implementation.get_model_config(
                "OCR", implementation.DEFAULT_OCR_MODEL_VERSION, kind, language)
            archive_name = Path(urlsplit(config["url"]).path).name
            if not archive_name.endswith(".tar"):
                return None
            root = Path(implementation.BASE_DIR).expanduser() / "whl" / kind
            if kind != "cls":
                root = root / language
            directory = root / archive_name[:-4]
            if not all((directory / name).is_file() and (directory / name).stat().st_size > 0
                       for name in ("inference.pdmodel", "inference.pdiparams")):
                self.bus.log.emit("OCR: PaddleOCR needs locally installed detection, recognition and "
                                  "orientation models; automatic downloads are disabled.")
                return None
            directories[f"{kind}_model_dir"] = str(directory)
        engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, **directories)

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            result = engine.ocr(self._rgb_ndarray(raw), cls=True)
            lines = (result or [[]])[0] or []
            texts, confs = [], []
            for line in lines:
                try:
                    texts.append(str(line[1][0])); confs.append(float(line[1][1]))
                except Exception:
                    continue
            conf = sum(confs) / len(confs) if confs else 0.0
            return ("\n".join(texts), conf)
        return run

    def _probe_pytesseract(self) -> Optional[Callable[[Any, Any], tuple[str, float]]]:
        from .utils import locate_tesseract

        locate_tesseract()
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            data = pytesseract.image_to_data(
                self._gray_for_tesseract(pre),
                output_type=pytesseract.Output.DICT, timeout=25)
            words, confs = [], []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                text = str(text).strip()
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    c = -1.0
                if text and c >= 0:
                    words.append(text); confs.append(c / 100.0)
            return (" ".join(words), sum(confs) / len(confs) if confs else 0.0)
        return run

    def _probe_winrt(self) -> Optional[Callable[[Any, Any], tuple[str, float | None]]]:
        if _winrt_modules() is None:
            return None

        def run(pre: Any, raw: Any) -> tuple[str, float | None]:
            text = self._run_winrt(raw)      # WinRT prefers the natural image
            return (text, None)
        return run

    def _run_winrt(self, image: Any) -> str:
        lines = _winrt_recognise(image)
        return self._clean("\n".join(" ".join(w["text"] for w in line) for line in lines))

    def _locate_words_winrt(self, image: Any) -> list[dict[str, Any]]:
        """Per-word boxes from the Windows OCR engine, in *image*'s pixels."""
        return [word for line in _winrt_recognise(image) for word in line]

    # ── helpers / compat ──────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
        # Hex codes are where OCR's letter/digit confusions hurt most (an
        # error code read wrong is a useless search). "Ox8OO7" is never meant:
        # a 0x prefix followed by hex digits and O's is a hex number.
        return re.sub(r"\b[Oo0][xX]([0-9A-Fa-fOo]{4,16})\b",
                      lambda m: "0x" + re.sub("[Oo]", "0", m.group(1)), text)

    @staticmethod
    def _rapid_ready(arr: Any) -> tuple[Any, int]:
        """(image RapidOCR will actually detect in, rows padded at the top).

        RapidOCR SKIPS text detection when an image is wider than 8:1 or under
        30 px tall and reads the whole thing as one line — so a single line,
        a text box or a strip of a window came back EMPTY (measured: 0% on
        every 11-16 px test strip). Padding to 2:1 with the image's own
        background brings detection back (mean 87% word accuracy across
        11/13/16 px, light and dark; 2:1 measured best of 1.5/2/3/4/6), and
        dark-mode text is inverted first because the models are trained on
        dark text on light (11 px dark: 0% -> 93%)."""
        try:
            import cv2
            import numpy as np
            if arr.ndim == 3 and float(np.median(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))) < 110:
                arr = 255 - arr
            h, w = arr.shape[:2]
            target = max(64, int(np.ceil(w / 2.0)))
            if h >= target:
                return arr, 0
            border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
            colour = [int(v) for v in np.median(border, axis=0)] if border.ndim == 2 \
                else int(np.median(border))
            extra = target - h
            top = extra // 2
            return cv2.copyMakeBorder(arr, top, extra - top, 0, 0,
                                      cv2.BORDER_CONSTANT, value=colour), top
        except Exception:
            return arr, 0

    @property
    def available(self) -> bool:
        self._detect()
        return bool(self._engines)

    @property
    def engine_name(self) -> str:
        self._detect()
        return self._name or "none"

    def status(self) -> dict[str, Any]:
        self._detect()
        return {"available": self.available, "engine": self.engine_name,
                "chain": [n for n, _ in self._engines]}
