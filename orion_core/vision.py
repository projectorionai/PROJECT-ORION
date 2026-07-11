"""
Vision subsystem.

    VolatileScreenGrabber   — in-memory JPEG capture of the primary monitor
    LocalFileIntelligence   — non-blocking file inspection pipeline (Mark VII)
    VisionAgent             — Mark VIII: image analysis, OCR, screenshot
                              understanding, desktop error detection and
                              visual context awareness.

OCR uses pytesseract when installed (pip install pytesseract + the Tesseract
engine); without it the agent still performs structural analysis and ships
the captured frame to the multimodal live channel, which reads text natively.
All heavy work (PIL decode, OCR inference, window enumeration) is executed
via ``asyncio.to_thread`` so the GUI event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import mimetypes
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mss
from PIL import Image, ImageStat

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

from .bus import OrionBus
from .constants import is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import PIL_RESAMPLE, fold_title


# ──────────────────────────────────────────────────────────────────────────────
# SCREEN GRABBER
# ──────────────────────────────────────────────────────────────────────────────

class VolatileScreenGrabber:
    """
    In-memory screen capture.

    Mark IX: the ``mss`` instance is cached per thread (``threading.local``)
    instead of being re-created on every call — ``mss.mss()`` construction was
    a measurable per-capture cost in the Mark VIII review.  mss objects are not
    thread-safe, so each worker thread keeps its own.
    """

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._local = threading.local()

    def _grabber(self) -> Any:
        grabber = getattr(self._local, "mss", None)
        if grabber is None:
            grabber = mss.mss()
            self._local.mss = grabber
        return grabber

    def _monitor(self, capture: Any, index: int | None) -> Any:
        monitors = capture.monitors
        if index is not None and 1 <= index < len(monitors):
            return monitors[index]
        return monitors[1] if len(monitors) > 1 else monitors[0]

    def capture_jpeg(self, max_side: int = 1024, quality: int = 78,
                     monitor: int | None = None) -> bytes:
        capture = self._grabber()
        raw   = capture.grab(self._monitor(capture, monitor))
        image = Image.frombytes("RGB", raw.size, raw.rgb)
        image.thumbnail((max_side, max_side), PIL_RESAMPLE)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=max(45, min(95, int(quality))), optimize=True)
        return buffer.getvalue()

    def capture_image(self, max_side: int = 1600, monitor: int | None = None) -> Image.Image:
        """Full-quality PIL frame for local analysis (OCR, error detection)."""
        capture = self._grabber()
        raw   = capture.grab(self._monitor(capture, monitor))
        image = Image.frombytes("RGB", raw.size, raw.rgb)
        if max_side:
            image.thumbnail((max_side, max_side), PIL_RESAMPLE)
        return image

    def capture_region(self, region: tuple[int, int, int, int]) -> Image.Image:
        """Capture a virtual-desktop rectangle (x, y, width, height)."""
        x, y, w, h = (int(v) for v in region)
        bbox = {"left": x, "top": y, "width": max(1, w), "height": max(1, h)}
        capture = self._grabber()
        raw = capture.grab(bbox)
        return Image.frombytes("RGB", raw.size, raw.rgb)

    def capture_array(self, region: tuple[int, int, int, int] | None = None,
                      monitor: int | None = None) -> Any:
        """Capture as an (H, W, 3) uint8 numpy array — used for visual diffing."""
        import numpy as np
        capture = self._grabber()
        if region is not None:
            x, y, w, h = (int(v) for v in region)
            target: Any = {"left": x, "top": y, "width": max(1, w), "height": max(1, h)}
        else:
            target = self._monitor(capture, monitor)
        raw = capture.grab(target)
        arr = np.frombuffer(raw.rgb, dtype=np.uint8).reshape(raw.height, raw.width, 3)
        return arr


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL FILE INTELLIGENCE  — all sync I/O wrapped in asyncio.to_thread()
# ──────────────────────────────────────────────────────────────────────────────

class LocalFileIntelligence:
    """
    File inspection pipeline.  Every heavy synchronous branch (PIL decoding,
    CSV sniffing, text tokenisation) is offloaded via asyncio.to_thread() so
    the qasync GUI event loop is never blocked by disk or CPU work.
    """

    TEXT_SUFFIXES = {
        ".txt", ".md", ".rst", ".py", ".js", ".ts", ".html", ".css",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log",
        ".csv", ".tsv", ".xml", ".sql",
    }
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus

    # ── async entry-point ────────────────────────────────────────────────────

    async def inspect_async(self, path: Path, prompt: str = "") -> ToolResult:
        """
        Non-blocking inspection.  Security guard runs synchronously (negligible
        cost); all heavy work is deferred to a thread pool via to_thread().
        """
        SecuritySanitiser.guard_text(str(path), "file_processor.path")
        resolved = path.expanduser().resolve()
        if is_protected_path(resolved):
            raise SecurityViolation("blocked unsafe file operation: core script protected")
        if not resolved.exists() or not resolved.is_file():
            return ToolResult(f"File not found: {resolved}", ok=False)

        suffix = resolved.suffix.lower()

        # Route and offload every synchronous branch to a thread-pool worker.
        if suffix in self.IMAGE_SUFFIXES:
            return await asyncio.to_thread(self._inspect_image, resolved, prompt)
        if suffix == ".pdf":
            return await asyncio.to_thread(self._inspect_pdf, resolved)
        if suffix == ".json":
            return await asyncio.to_thread(self._inspect_json, resolved)
        if suffix in {".csv", ".tsv"}:
            # CSV sniffing (csv.Sniffer) is synchronous I/O — offload.
            return await asyncio.to_thread(self._inspect_table, resolved)
        is_text = await asyncio.to_thread(self._looks_textual, resolved)
        if suffix in self.TEXT_SUFFIXES or is_text:
            return await asyncio.to_thread(self._inspect_text, resolved)
        return await asyncio.to_thread(self._inspect_binary, resolved)

    # ── legacy synchronous wrapper (retained for non-async call sites) ───────

    def inspect(self, path: Path, prompt: str = "") -> ToolResult:
        """Synchronous fallback. Do NOT invoke from the GUI thread during a live session."""
        SecuritySanitiser.guard_text(str(path), "file_processor.path")
        resolved = path.expanduser().resolve()
        if is_protected_path(resolved):
            raise SecurityViolation("blocked unsafe file operation: core script protected")
        if not resolved.exists() or not resolved.is_file():
            return ToolResult(f"File not found: {resolved}", ok=False)
        suffix = resolved.suffix.lower()
        if suffix in self.IMAGE_SUFFIXES:
            return self._inspect_image(resolved, prompt)
        if suffix == ".pdf":
            return self._inspect_pdf(resolved)
        if suffix == ".json":
            return self._inspect_json(resolved)
        if suffix in {".csv", ".tsv"}:
            return self._inspect_table(resolved)
        if suffix in self.TEXT_SUFFIXES or self._looks_textual(resolved):
            return self._inspect_text(resolved)
        return self._inspect_binary(resolved)

    # ── private synchronous workers (called inside to_thread) ────────────────

    def _inspect_image(self, path: Path, prompt: str) -> ToolResult:
        """PIL image decode — synchronous; invoked via asyncio.to_thread()."""
        with Image.open(path) as image:
            original_format = image.format or path.suffix.replace(".", "").upper() or "IMAGE"
            width, height   = image.size
            mode            = image.mode
            rgb             = image.convert("RGB")
            stat_source     = rgb.copy()
            stat_source.thumbnail((512, 512), PIL_RESAMPLE)
            stat      = ImageStat.Stat(stat_source)
            mean      = tuple(int(v) for v in stat.mean[:3])
            extrema   = stat.extrema[:3]
            brightness = sum(stat.mean[:3]) / (3 * 255)
            entropy   = rgb.entropy()
            aspect    = width / max(1, height)
            orientation = (
                "landscape" if aspect > 1.08 else
                "portrait"  if aspect < 0.92 else
                "square"
            )
            exif_count = 0
            try:
                exif_count = len(image.getexif() or {})
            except Exception:
                exif_count = 0
            outbound = rgb.copy()
            outbound.thumbnail((1024, 1024), PIL_RESAMPLE)
            buffer = io.BytesIO()
            outbound.save(buffer, format="JPEG", quality=82, optimize=True)
        size_kb     = path.stat().st_size / 1024
        prompt_line = (
            f"Requested focus: {prompt.strip()}"
            if prompt.strip()
            else "Requested focus: general visual inspection."
        )
        report = (
            f"Image scan complete: {path.name}\n"
            f"Format: {original_format}; dimensions: {width}x{height}; "
            f"orientation: {orientation}; mode: {mode}; size: {size_kb:.1f} KB.\n"
            f"Mean RGB: {mean}; channel ranges: {extrema}; "
            f"brightness index: {brightness:.2f}; entropy: {entropy:.2f}; "
            f"EXIF entries: {exif_count}.\n"
            f"{prompt_line}\n"
            "A volatile JPEG review frame is ready for the live multimodal channel."
        )
        return ToolResult(report, media={"data": buffer.getvalue(), "mime_type": "image/jpeg"})

    def _inspect_json(self, path: Path) -> ToolResult:
        raw  = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        if isinstance(data, dict):
            keys  = list(data.keys())[:30]
            shape = f"object with {len(data)} keys: {', '.join(map(str, keys))}"
        elif isinstance(data, list):
            shape = f"array with {len(data)} items"
            if data and isinstance(data[0], dict):
                shape += f"; first item keys: {', '.join(map(str, list(data[0].keys())[:20]))}"
        else:
            shape = type(data).__name__
        return ToolResult(
            f"JSON scan complete: {path.name}\nShape: {shape}.\n"
            f"Characters: {len(raw)}.\nExcerpt:\n{raw[:2200]}"
        )

    def _inspect_table(self, path: Path) -> ToolResult:
        """CSV/TSV sniffing — synchronous I/O; invoked via asyncio.to_thread()."""
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel_tab if delimiter == "\t" else csv.excel  # type: ignore[assignment]
            reader = csv.reader(handle, dialect)
            for index, row in enumerate(reader):
                rows.append(row)
                if index >= 40:
                    break
        header  = rows[0] if rows else []
        preview = "\n".join(
            " | ".join(cell[:80] for cell in row[:12]) for row in rows[:12]
        )
        return ToolResult(
            f"Table scan complete: {path.name}\n"
            f"Columns detected: {len(header)}; preview rows inspected: {len(rows)}.\n"
            f"Header: {', '.join(header[:20]) if header else 'none'}\nPreview:\n{preview}"
        )

    def _inspect_text(self, path: Path) -> ToolResult:
        raw   = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        words = re.findall(r"[A-Za-z0-9_']+", raw)
        frequencies: dict[str, int] = {}
        for word in words:
            key = word.lower()
            if len(key) < 4 or key.isdigit():
                continue
            frequencies[key] = frequencies.get(key, 0) + 1
        top_terms = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:12]
        terms     = ", ".join(f"{term}:{count}" for term, count in top_terms) or "none"
        return ToolResult(
            f"Text scan complete: {path.name}\n"
            f"Lines: {len(lines)}; words: {len(words)}; characters: {len(raw)}.\n"
            f"Dominant terms: {terms}.\nExcerpt:\n{raw[:4200]}"
        )

    def _inspect_pdf(self, path: Path) -> ToolResult:
        """PDF text extraction via pypdf/PyPDF2 (optional dependency)."""
        reader_cls: Any = None
        try:
            from pypdf import PdfReader as reader_cls  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader as reader_cls  # type: ignore
            except Exception:
                reader_cls = None
        size_kb = path.stat().st_size / 1024
        if reader_cls is None:
            return ToolResult(
                f"PDF detected: {path.name} ({size_kb:.1f} KB). "
                "Install 'pypdf' (pip install pypdf) to enable text extraction, "
                "summarisation and question answering over documents.",
                ok=False,
            )
        reader = reader_cls(str(path))
        page_count = len(reader.pages)
        chunks: list[str] = []
        for page in reader.pages[:12]:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        text  = re.sub(r"[ \t]+", " ", "\n".join(chunks)).strip()
        title = ""
        try:
            title = str((reader.metadata or {}).get("/Title") or "")
        except Exception:
            title = ""
        return ToolResult(
            f"PDF scan complete: {path.name}\n"
            f"Pages: {page_count}; size: {size_kb:.1f} KB; title: {title or 'not set'}.\n"
            f"Extracted text (first {min(page_count, 12)} pages):\n"
            f"{text[:5000] or 'No extractable text — likely a scanned document; convert pages to images for visual analysis.'}"
        )

    def _inspect_binary(self, path: Path) -> ToolResult:
        mime, _ = mimetypes.guess_type(str(path))
        size_kb  = path.stat().st_size / 1024
        with path.open("rb") as handle:
            header = handle.read(32).hex(" ")
        return ToolResult(
            f"Binary file scan complete: {path.name}\n"
            f"MIME: {mime or 'unknown'}; size: {size_kb:.1f} KB; header bytes: {header}.\n"
            "Deep semantic extraction is available for text, JSON, CSV, TSV, "
            "and common image formats."
        )

    def _looks_textual(self, path: Path) -> bool:
        try:
            chunk = path.read_bytes()[:2048]
        except Exception:
            return False
        if not chunk:
            return True
        if b"\x00" in chunk:
            return False
        printable = sum(1 for byte in chunk if byte in b"\r\n\t" or 32 <= byte < 127)
        return (printable / len(chunk)) > 0.82


# ──────────────────────────────────────────────────────────────────────────────
# VISION AGENT  — screenshot understanding, OCR, desktop error detection
# ──────────────────────────────────────────────────────────────────────────────

class VisionAgent:
    """
    ORION's eyes.

    Capabilities:
        analyse_screen()  — capture + structural analysis + OCR excerpt; the
                            JPEG frame is attached so the multimodal channel
                            can read the screen directly.
        ocr()             — extract text from the screen or an image file.
        detect_errors()   — hunt for error dialogs / crash text via window
                            titles (Win32) and OCR keyword heuristics.
        analyse_image()   — file-based image analysis (delegates to
                            LocalFileIntelligence, adds OCR when available).

    OCR is optional: without pytesseract the agent reports what it *can*
    determine and defers text reading to the multimodal model.
    """

    ERROR_KEYWORDS = (
        "error", "exception", "traceback", "failed", "failure", "crash",
        "crashed", "not responding", "fatal", "denied", "cannot", "unable to",
        "0x8", "0xc0", "blue screen", "bsod", "segmentation fault",
    )
    ERROR_TITLE_RE = re.compile(
        r"(?i)\b(error|exception|crash|not responding|problem|failure|fatal)\b"
    )

    def __init__(self, bus: OrionBus, grabber: VolatileScreenGrabber,
                 file_intel: LocalFileIntelligence) -> None:
        self.bus = bus
        self.grabber = grabber
        self.file_intel = file_intel
        self._ocr_checked = False
        self._ocr: Any = None  # pytesseract module when available
        # Optional pluggable multi-backend OCR engine (improvement #2); when
        # attached it takes precedence over the built-in pytesseract path.
        self.ocr_engine: Any = None

    def attach_ocr_engine(self, engine: Any) -> None:
        self.ocr_engine = engine

    # ── OCR availability (lazy, checked once) ────────────────────────────────

    def _ocr_module(self) -> Any:
        if not self._ocr_checked:
            self._ocr_checked = True
            try:
                import pytesseract  # type: ignore
                pytesseract.get_tesseract_version()
                self._ocr = pytesseract
                self.bus.log.emit("VISION: local OCR engine online (pytesseract).")
            except Exception:
                self._ocr = None
                self.bus.log.emit(
                    "VISION: pytesseract not installed; deferring to the pluggable OCR "
                    "engine or the multimodal channel for screen text."
                )
        return self._ocr

    @property
    def ocr_available(self) -> bool:
        if self.ocr_engine is not None and self.ocr_engine.available:
            return True
        return self._ocr_module() is not None

    # ── public async API (all heavy work off-thread) ─────────────────────────

    async def analyse_screen(self, prompt: str = "") -> ToolResult:
        """Capture the screen, analyse structure, extract text, attach the frame."""
        return await asyncio.to_thread(self._analyse_screen_sync, prompt)

    async def ocr(self, path: str = "") -> ToolResult:
        """OCR the screen (no path) or a specific image file."""
        return await asyncio.to_thread(self._ocr_sync, path)

    async def detect_errors(self) -> ToolResult:
        """Scan the desktop for error dialogs, crash text and stuck windows."""
        return await asyncio.to_thread(self._detect_errors_sync)

    async def analyse_image(self, path: str, prompt: str = "") -> ToolResult:
        """Structural + OCR analysis of an image file on disk."""
        result = await self.file_intel.inspect_async(Path(path), prompt=prompt)
        if not result.ok:
            return result
        ocr_text = await asyncio.to_thread(self._ocr_file_text, Path(path))
        if ocr_text:
            result.text += f"\nOCR text extract:\n{ocr_text[:2500]}"
        return result

    async def capture_live_frame(self, camera_index: int = 0, max_side: int = 640) -> ToolResult:
        """Capture a single frame from a webcam and attach it to the multimodal stream."""
        return await asyncio.to_thread(self._capture_live_frame_sync, camera_index, max_side)

    # ── synchronous workers ───────────────────────────────────────────────────

    def _capture_live_frame_sync(self, camera_index: int, max_side: int) -> ToolResult:
        if cv2 is None:
            return ToolResult(
                "OpenCV is not installed; webcam capture is unavailable. Install opencv-python to enable the alternate camera input.",
                ok=False,
            )
        capture = cv2.VideoCapture(int(camera_index))
        try:
            if not capture.isOpened():
                return ToolResult(f"Unable to open camera device {camera_index}.", ok=False)
            ok, frame = capture.read()
            if not ok or frame is None:
                return ToolResult("The camera returned no frame.", ok=False)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side), PIL_RESAMPLE)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=78, optimize=True)
            self.bus.log.emit(f"VISION: live frame captured from camera {camera_index}.")
            return ToolResult(
                "Live webcam frame captured successfully.",
                media={"data": buffer.getvalue(), "mime_type": "image/jpeg"},
            )
        except Exception as exc:
            self.bus.log.emit(f"VISION: camera capture failed - {exc}")
            return ToolResult(f"Camera capture failed: {exc}", ok=False)
        finally:
            capture.release()

    def _analyse_screen_sync(self, prompt: str) -> ToolResult:
        image = self.grabber.capture_image(max_side=1600)
        width, height = image.size
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean[:3]) / (3 * 255)
        entropy = image.entropy()
        dominant = tuple(int(v) for v in stat.mean[:3])
        # Outbound review frame for the multimodal channel.
        outbound = image.copy()
        outbound.thumbnail((1024, 1024), PIL_RESAMPLE)
        buffer = io.BytesIO()
        outbound.save(buffer, format="JPEG", quality=80, optimize=True)
        ocr_text = self._ocr_image_text(image)
        windows = self._visible_window_titles()
        lines = [
            "Screen analysis complete.",
            f"Resolution (captured): {width}x{height}; brightness index {brightness:.2f}; "
            f"visual entropy {entropy:.2f}; mean RGB {dominant}.",
        ]
        if windows:
            lines.append("Foreground window inventory: " + "; ".join(windows[:10]) + ".")
        if ocr_text:
            lines.append(f"On-screen text (OCR excerpt):\n{ocr_text[:2800]}")
        elif not self.ocr_available:
            lines.append(
                "Local OCR unavailable — the attached frame carries the pixels; "
                "read on-screen text directly from it."
            )
        if prompt.strip():
            lines.append(f"Requested focus: {prompt.strip()}")
        lines.append("A volatile JPEG frame of the desktop is attached for direct visual review.")
        return ToolResult(
            "\n".join(lines),
            media={"data": buffer.getvalue(), "mime_type": "image/jpeg"},
        )

    def _ocr_sync(self, path: str) -> ToolResult:
        if path.strip():
            text = self._ocr_file_text(Path(path.strip()))
            source = Path(path.strip()).name
        else:
            text = self._ocr_image_text(self.grabber.capture_image(max_side=1600))
            source = "primary display"
        if not self.ocr_available:
            return ToolResult(
                "Local OCR engine not installed. Install pytesseract and the Tesseract "
                "binary, or use analyse_screen so the multimodal channel reads the "
                "attached frame directly.",
                ok=False,
            )
        if not text:
            return ToolResult(f"OCR of {source} found no legible text.")
        return ToolResult(f"OCR extract from {source}:\n{text[:6000]}")

    def _detect_errors_sync(self) -> ToolResult:
        findings: list[str] = []
        # 1. Window-title sweep — cheapest, most reliable signal on Windows.
        titles = self._visible_window_titles()
        for title in titles:
            if self.ERROR_TITLE_RE.search(title):
                findings.append(f"Window title indicates a fault: '{title}'")
        # 2. OCR keyword sweep across the captured frame.
        image = self.grabber.capture_image(max_side=1600)
        ocr_text = self._ocr_image_text(image)
        if ocr_text:
            lowered = ocr_text.lower()
            hits = sorted({kw for kw in self.ERROR_KEYWORDS if kw in lowered})
            if hits:
                findings.append(
                    "On-screen text contains fault indicators: " + ", ".join(hits) + "."
                )
                # Surface the most relevant OCR lines for context.
                relevant = [
                    line.strip() for line in ocr_text.splitlines()
                    if line.strip() and any(kw in line.lower() for kw in hits)
                ]
                if relevant:
                    findings.append("Relevant lines:\n" + "\n".join(relevant[:8]))
        # 3. Attach the frame so the model can double-check visually.
        outbound = image.copy()
        outbound.thumbnail((1024, 1024), PIL_RESAMPLE)
        buffer = io.BytesIO()
        outbound.save(buffer, format="JPEG", quality=80, optimize=True)
        if not findings:
            report = (
                "Desktop error sweep complete: no error dialogs, crash text or "
                "unresponsive windows detected."
                + ("" if self.ocr_available else
                   " (Local OCR is offline — verdict based on window titles and the attached frame.)")
            )
        else:
            report = "Desktop error sweep found potential faults:\n" + "\n".join(findings)
        return ToolResult(report, media={"data": buffer.getvalue(), "mime_type": "image/jpeg"})

    # ── low-level helpers ─────────────────────────────────────────────────────

    def _ocr_image_text(self, image: Image.Image) -> str:
        # Prefer the pluggable multi-backend engine when present.
        if self.ocr_engine is not None and self.ocr_engine.available:
            text = self.ocr_engine.image_to_text(image)
            if text:
                return text
        module = self._ocr_module()
        if module is None:
            return ""
        try:
            grey = image.convert("L")
            text = module.image_to_string(grey, timeout=20)
            return re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
        except Exception as exc:
            self.bus.log.emit(f"VISION: OCR fault - {str(exc).splitlines()[0][:100]}")
            return ""

    def _ocr_file_text(self, path: Path) -> str:
        module = self._ocr_module()
        if module is None or not path.is_file():
            return ""
        try:
            with Image.open(path) as image:
                return self._ocr_image_text(image.convert("RGB"))
        except Exception:
            return ""

    def _visible_window_titles(self) -> list[str]:
        """Enumerate visible top-level window titles (Windows only)."""
        if sys.platform != "win32":
            return []
        try:
            import ctypes
            import ctypes.wintypes as wintypes
            user32 = ctypes.windll.user32
            titles: list[str] = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            def _collect(hwnd: Any, lparam: Any) -> bool:
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buffer = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buffer, length + 1)
                        titles.append(buffer.value)
                return True

            user32.EnumWindows(_collect, 0)
            return titles[:60]
        except Exception:
            return []

    # ── UI element detection (pywinauto UIA) ──────────────────────────────────

    # Control-type groupings used by the detectors and the verification engine.
    BUTTON_ROLES = {"Button", "SplitButton"}
    MENU_ROLES = {"Menu", "MenuItem", "MenuBar"}
    INPUT_ROLES = {"Edit", "ComboBox", "CheckBox", "RadioButton", "Document"}
    LINK_ROLES = {"Hyperlink"}
    DIALOG_ROLES = {"Window", "Dialog", "Pane"}

    async def detect_elements(self, kinds: str = "all", max_elements: int = 120) -> ToolResult:
        """Enumerate interactive controls in the foreground window (UIA)."""
        elements = await asyncio.to_thread(self._detect_elements_sync, kinds, max_elements)
        if not elements:
            return ToolResult(
                "No UI elements detected (UIA unavailable, or the foreground window "
                "exposes no automation tree). Fall back to vision/OCR for this app.",
                ok=bool(self._uia_available()),
            )
        lines = [f"Detected {len(elements)} UI element(s) in the foreground window:"]
        for el in elements[:60]:
            state = "" if el["enabled"] else " (disabled)"
            lines.append(
                f"- [{el['role']}] '{el['name'][:60]}'{state} "
                f"@ ({el['center'][0]},{el['center'][1]})"
            )
        return ToolResult("\n".join(lines))

    async def detect_dialogs(self) -> ToolResult:
        """Find open dialogs and pop-ups across the desktop."""
        dialogs = await asyncio.to_thread(self._detect_dialogs_sync)
        if not dialogs:
            return ToolResult("No open dialogs or pop-ups detected.")
        lines = ["Open dialogs / pop-ups:"]
        for d in dialogs:
            lines.append(f"- '{d['name'][:70]}' [{d['role']}] @ {d['rect']}")
        return ToolResult("\n".join(lines))

    async def find_element(self, query: str, kinds: str = "all") -> dict[str, Any] | None:
        """Locate the best-matching control; returns its rect + centre or None."""
        return await asyncio.to_thread(self._find_element_sync, query, kinds)

    # ── synchronous UIA workers (run in a COM-initialised thread) ─────────────

    def _uia_available(self) -> bool:
        try:
            import pywinauto  # noqa: F401
            return sys.platform == "win32"
        except Exception:
            return False

    def _foreground_hwnd(self) -> int:
        try:
            import ctypes
            return int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            return 0

    def _roles_for(self, kinds: str) -> set[str] | None:
        kinds = (kinds or "all").lower()
        if kinds in {"all", "any", ""}:
            return None
        mapping = {
            "button": self.BUTTON_ROLES, "buttons": self.BUTTON_ROLES,
            "menu": self.MENU_ROLES, "menus": self.MENU_ROLES,
            "input": self.INPUT_ROLES, "inputs": self.INPUT_ROLES,
            "field": self.INPUT_ROLES, "fields": self.INPUT_ROLES,
            "link": self.LINK_ROLES, "links": self.LINK_ROLES,
            "dialog": self.DIALOG_ROLES, "dialogs": self.DIALOG_ROLES,
        }
        return mapping.get(kinds)

    def _detect_elements_sync(self, kinds: str, max_elements: int) -> list[dict[str, Any]]:
        if not self._uia_available():
            return []
        import pythoncom  # type: ignore
        pythoncom.CoInitialize()
        try:
            from pywinauto import Desktop  # type: ignore
            hwnd = self._foreground_hwnd()
            if not hwnd:
                return []
            window = Desktop(backend="uia").window(handle=hwnd)
            roles = self._roles_for(kinds)
            collected: list[dict[str, Any]] = []
            for ctrl in window.descendants():
                if len(collected) >= max_elements:
                    break
                info = self._element_dict(ctrl)
                if info is None:
                    continue
                if roles is not None and info["role"] not in roles:
                    continue
                # Skip zero-area and unnamed non-input controls (noise).
                if info["rect"][2] <= 0 or info["rect"][3] <= 0:
                    continue
                collected.append(info)
            return collected
        except Exception as exc:
            self.bus.log.emit(f"VISION: UIA enumeration recovered - {str(exc).splitlines()[0][:100]}")
            return []
        finally:
            pythoncom.CoUninitialize()

    def _detect_dialogs_sync(self) -> list[dict[str, Any]]:
        if not self._uia_available():
            return []
        import pythoncom  # type: ignore
        pythoncom.CoInitialize()
        try:
            from pywinauto import Desktop  # type: ignore
            out: list[dict[str, Any]] = []
            for win in Desktop(backend="uia").windows():
                try:
                    role = win.element_info.control_type
                    name = win.window_text() or ""
                    if role not in self.DIALOG_ROLES:
                        continue
                    rect = win.rectangle()
                    w, h = rect.width(), rect.height()
                    # Dialog heuristic: modest-sized, titled, top-level window.
                    if not name or w <= 0 or h <= 0 or (w > 1400 and h > 900):
                        continue
                    if re.search(r"(?i)dialog|confirm|save|open|error|warning|alert|sign in|cookie|consent", name):
                        out.append({"role": role, "name": name,
                                    "rect": (rect.left, rect.top, w, h)})
                except Exception:
                    continue
            return out[:20]
        except Exception:
            return []
        finally:
            pythoncom.CoUninitialize()

    def _find_element_sync(self, query: str, kinds: str) -> dict[str, Any] | None:
        # Fold both sides: control names on the web carry invisible Unicode
        # and non-breaking spaces that defeat naive substring matching.
        query = fold_title(query)
        if not query:
            return None
        elements = self._detect_elements_sync(kinds, max_elements=400)
        if not elements:
            return None
        # Rank: exact name, then startswith, then substring, then token overlap.
        def _score(el: dict[str, Any]) -> tuple[int, int]:
            name = fold_title(el["name"])
            if name == query:
                return (0, len(name))
            if name.startswith(query):
                return (1, len(name))
            if query in name:
                return (2, len(name))
            overlap = len(set(query.split()) & set(name.split()))
            return (3 if overlap else 9, -overlap)
        candidates = [e for e in elements if query in fold_title(e["name"])
                      or set(query.split()) & set(fold_title(e["name"]).split())]
        if not candidates:
            return None
        return sorted(candidates, key=_score)[0]

    def _element_dict(self, ctrl: Any) -> dict[str, Any] | None:
        try:
            info = ctrl.element_info
            rect = ctrl.rectangle()
            x, y = rect.left, rect.top
            w, h = rect.width(), rect.height()
            return {
                "role": str(info.control_type or ""),
                "name": str(info.name or ""),
                "rect": (x, y, w, h),
                "center": (x + w // 2, y + h // 2),
                "enabled": bool(getattr(info, "enabled", True)),
            }
        except Exception:
            return None
