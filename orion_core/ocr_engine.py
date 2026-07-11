"""
OCR & vision engine (Phase 8, Section B) — confidence-scored, multi-engine.

Redesigned from a single-backend reader into a fallback CHAIN with real
confidence scoring and OpenCV preprocessing:

    OpenCV preprocess  →  Engine 1  →  (confidence < threshold?)  →  Engine 2  →  …

Every engine returns ``(text, confidence)``; the chain accepts the first
result whose confidence clears the threshold and otherwise keeps the best of
all attempts, so ORION always returns *something* with an honest score:

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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .bus import OrionBus


@dataclass
class OcrResult:
    text: str
    confidence: float          # 0..1
    engine: str

    def report(self) -> str:
        return (f"OCR Result\nConfidence: {self.confidence * 100:.1f}%\n"
                f"Engine Used: {self.engine or 'none'}\n\n{self.text}")


# A recognised engine: name → callable(preprocessed_pil, raw_pil) → (text, conf)
_Engine = tuple[str, Callable[[Any, Any], tuple[str, float]]]


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
        probes = (
            ("rapidocr-onnxruntime", self._probe_rapidocr),
            ("easyocr", self._probe_easyocr),
            ("paddleocr", self._probe_paddleocr),
            ("pytesseract", self._probe_pytesseract),
            ("Windows WinRT OCR", self._probe_winrt),
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

    # ── the fallback chain ────────────────────────────────────────────────────

    def image_to_result(self, image: Any) -> OcrResult:
        """Run the engine chain; return the best {text, confidence, engine}."""
        self._detect()
        if not self._engines:
            return OcrResult("", 0.0, "")
        pre = self._preprocess(image)
        best = OcrResult("", 0.0, "")
        for name, fn in self._engines:
            try:
                text, conf = fn(pre, image)
            except Exception as exc:
                self.bus.log.emit(f"OCR: {name} fault - {str(exc).splitlines()[0][:80]}")
                continue
            text = self._clean(text)
            if not text:
                continue
            if conf > best.confidence:
                best = OcrResult(text, conf, name)
            if conf >= self.CONFIDENCE_THRESHOLD:
                return best        # good enough — stop, spare the slower engines
        return best

    def image_to_text(self, image: Any) -> str:
        """Compatibility: just the recognised text (best engine)."""
        return self.image_to_result(image).text

    # ── engine adapters (each returns (text, confidence 0..1)) ────────────────

    def _probe_rapidocr(self) -> Optional[Callable[[Any, Any], tuple[str, float]]]:
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        engine = RapidOCR()

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            import numpy as np
            # Neural OCR reads the NATURAL image best; binarisation is for
            # Tesseract.  Upscale small captures for legibility though.
            result, _ = engine(np.array(self._upscaled(raw).convert("RGB")))
            if not result:
                return ("", 0.0)
            items, scores = [], []
            for line in result:
                if len(line) < 2:
                    continue
                box, text = line[0], str(line[1])
                score = float(line[2]) if len(line) > 2 else 0.6
                scores.append(score)
                try:
                    ys = [p[1] for p in box]; xs = [p[0] for p in box]
                    items.append((min(ys), min(xs), text))
                except Exception:
                    items.append((0.0, 0.0, text))
            items.sort(key=lambda it: (round(it[0] / 12), it[1]))
            text = "\n".join(t for _y, _x, t in items)
            conf = sum(scores) / len(scores) if scores else 0.0
            return (text, conf)
        return run

    def _probe_easyocr(self) -> Optional[Callable[[Any, Any], tuple[str, float]]]:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            import numpy as np
            results = reader.readtext(np.array(self._upscaled(raw).convert("RGB")), detail=1)
            if not results:
                return ("", 0.0)
            texts = [str(r[1]) for r in results]
            confs = [float(r[2]) for r in results if len(r) > 2]
            conf = sum(confs) / len(confs) if confs else 0.0
            return ("\n".join(texts), conf)
        return run

    def _probe_paddleocr(self) -> Optional[Callable[[Any, Any], tuple[str, float]]]:
        from paddleocr import PaddleOCR  # type: ignore
        engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            import numpy as np
            result = engine.ocr(np.array(self._upscaled(raw).convert("RGB")), cls=True)
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
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            data = pytesseract.image_to_data(
                pre.convert("L"), output_type=pytesseract.Output.DICT, timeout=25)
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

    def _probe_winrt(self) -> Optional[Callable[[Any, Any], tuple[str, float]]]:
        import winsdk.windows.media.ocr as _ocr  # type: ignore  # noqa: F401

        def run(pre: Any, raw: Any) -> tuple[str, float]:
            text = self._run_winrt(raw)      # WinRT prefers the natural image
            # WinRT exposes no per-word score; treat a clean multi-word read as
            # confident, a sparse one as uncertain so the chain can try better.
            words = len(text.split())
            conf = 0.8 if words >= 3 else (0.55 if words else 0.0)
            return (text, conf)
        return run

    def _run_winrt(self, image: Any) -> str:
        import asyncio
        import io
        import winsdk.windows.graphics.imaging as imaging  # type: ignore
        import winsdk.windows.media.ocr as ocr  # type: ignore
        import winsdk.windows.storage.streams as streams  # type: ignore

        async def _recognise() -> str:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            writer = streams.DataWriter()
            writer.write_bytes(list(buf.getvalue()))
            stream = streams.InMemoryRandomAccessStream()
            decoder = await imaging.BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            engine = ocr.OcrEngine.try_create_from_user_profile_languages()
            result = await engine.recognize_async(bitmap)
            return self._clean(result.text)

        try:
            return asyncio.run(_recognise())
        except Exception:
            return ""

    # ── helpers / compat ──────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()

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
