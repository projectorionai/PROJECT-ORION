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

from .bus import OrionBus
from .constants import is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import PIL_RESAMPLE, fold_control_label, fold_title, open_camera_capture


# ──────────────────────────────────────────────────────────────────────────────
# SCREEN GRABBER
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# WHERE A FRAME CAME FROM
# ──────────────────────────────────────────────────────────────────────────────
#
# An image arrives at the model as pixels and nothing else. Without a sentence
# saying what it IS, the model has to guess from content — and ORION's own
# window has a large human face in the middle of it, so a screenshot taken while
# his avatar is on screen reads exactly like a photograph of the user. The model
# then answers questions about "you" by describing its own avatar, confidently
# and completely wrongly.
#
# The fix is a sentence, not a mechanism: every frame says where it came from,
# in the tool result the model reads alongside the image.

SCREEN_FRAME_NOTE = (
    "SOURCE: this image is a SCREENSHOT of the computer's display. Anything in "
    "it is on screen — including ORION's own window, whose avatar is a rendered "
    "face and is NOT a photograph of the user. Do not describe it as a person "
    "in the room."
)

WEBCAM_FRAME_NOTE = (
    "SOURCE: this image is a PHOTOGRAPH from the webcam, looking at the room."
)

FILE_FRAME_NOTE = (
    "SOURCE: this image is a FILE from disk that the user pointed at, not "
    "something ORION captured just now."
)

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
        """The region to grab: one screen if asked, otherwise ALL of them.

        This used to default to ``monitors[1]`` — the primary screen alone —
        so on a two-monitor desk ORION could not see half of what was in front
        of the user. Asked to find something on the second screen he would
        answer, truthfully as far as he could tell, that it was not there.

        ``monitors[0]`` is mss's union of every screen, so the default is now
        the whole virtual desktop and a coordinate found in the capture is
        already a virtual-desktop coordinate. An explicit index still selects
        a single screen, which is what callers wanting resolution should ask
        for.
        """
        monitors = capture.monitors
        if index is not None and 1 <= index < len(monitors):
            return monitors[index]
        return monitors[0]

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
        image = self._unless_blank(image, monitor)
        if max_side:
            image.thumbnail((max_side, max_side), PIL_RESAMPLE)
        return image

    # ── games: what a screenshot cannot see ────────────────────────────────

    @staticmethod
    def _looks_blank(image: Image.Image) -> bool:
        """A solid black (or near-uniform) frame — what GDI capture returns
        for a game drawing straight to the GPU in exclusive fullscreen."""
        try:
            small = image.convert("L").resize((64, 36))
            stat = ImageStat.Stat(small)
            return stat.stddev[0] < 2.0 and stat.mean[0] < 12.0
        except Exception:
            return False

    def _unless_blank(self, image: Image.Image, monitor: int | None) -> Image.Image:
        """Re-take a blank screenshot through DXGI desktop duplication.

        The ordinary capture copies the desktop's device context, and a game
        rendering in exclusive fullscreen is not on it — ORION saw a black
        screen and, reasonably, reported nothing there. Desktop duplication
        reads what the display is actually showing, game included.
        """
        if not self._looks_blank(image):
            return image
        frame = self.capture_dxgi(0 if monitor in (None, 0, 1) else int(monitor) - 1)
        return frame if frame is not None else image

    def capture_dxgi(self, output: int = 0) -> Image.Image | None:
        """One frame of *output* via DXGI desktop duplication, or None."""
        try:
            import dxcam  # type: ignore
        except Exception:
            return None
        cameras = getattr(self, "_dxgi", None)
        if cameras is None:
            cameras = self._dxgi = {}
        try:
            camera = cameras.get(output)
            if camera is None:
                camera = cameras[output] = dxcam.create(output_idx=output, output_color="RGB")
            frame = camera.grab()
            if frame is None:
                # grab() returns None when nothing changed since the last one.
                time.sleep(0.05)
                frame = camera.grab()
            if frame is None:
                frame = getattr(self, "_dxgi_last", {}).get(output)
            if frame is None:
                return None
            self.__dict__.setdefault("_dxgi_last", {})[output] = frame
            return Image.fromarray(frame)
        except Exception:
            return None

    def monitor_bounds(self, monitor: int | None = None) -> tuple[int, int, int, int]:
        """(left, top, width, height) of the monitor capture_image()/
        capture_jpeg() would grab — the offset a caller needs to translate
        a coordinate found WITHIN that capture back to virtual-desktop
        pixels (Mark XXI, Track D2: the OCR click fallback needs this)."""
        capture = self._grabber()
        mon = self._monitor(capture, monitor)
        return (int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"]))

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
            "A volatile JPEG review frame is ready for the live multimodal channel.\n"
            + FILE_FRAME_NOTE
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

def _meaning_first(query: str, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*elements* reordered by how closely each control's name means *query*.

    Unchanged when the sentence encoder is cold or anything fails: the
    lexical order is a perfectly good fallback for a suggestion list.
    """
    try:
        from . import semantic

        named = [el for el in elements if str(el.get("name") or "").strip()]
        if len(named) < 2 or not semantic.ENCODER.ready():
            return elements
        unique = list(dict.fromkeys(str(el["name"]).strip()[:120] for el in named))
        vectors = semantic.ENCODER.encode(unique)
        target = semantic.ENCODER.encode_one(query)
        if vectors is None or target is None:
            return elements
        score = dict(zip(unique, (vectors @ target).tolist()))
        return sorted(named, key=lambda el: -score.get(str(el["name"]).strip()[:120], 0.0))
    except Exception:
        return elements


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
        # Optional borrowed-frame source (e.g. the face tracker's live camera):
        # a zero-arg callable returning the latest BGR frame or None.  Using it
        # avoids opening a SECOND capture on a camera another subsystem already
        # holds — the cause of the -1072873821 MSMF grab failure on Windows.
        self._live_frame_source: Any = None

    def attach_ocr_engine(self, engine: Any) -> None:
        self.ocr_engine = engine

    def set_live_frame_source(self, source: Any) -> None:
        """Register a zero-arg callable returning the latest webcam frame (a BGR
        ndarray) or None, so on-demand vision can borrow an already-open camera
        instead of opening its own handle."""
        self._live_frame_source = source

    # ── OCR availability (lazy, checked once) ────────────────────────────────

    def _ocr_module(self) -> Any:
        if not self._ocr_checked:
            self._ocr_checked = True
            try:
                # Point pytesseract at the binary before asking it anything.
                # It searches PATH and nowhere else, and the Windows installer
                # puts tesseract in Program Files without touching PATH — so
                # a machine with it properly installed still reported "not
                # installed" and quietly fell back to a slower recogniser.
                from .utils import locate_tesseract

                locate_tesseract()
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

    async def capture_live_frame(self, camera_index: int | None = None, max_side: int = 640) -> ToolResult:
        """Capture a single frame from a webcam and attach it to the multimodal stream."""
        if camera_index is None:
            from . import camera_devices as cd
            camera_index = cd.resolve()
        return await asyncio.to_thread(self._capture_live_frame_sync, camera_index, max_side)

    async def analyse_camera(self, prompt: str = "", camera_index: int | None = None) -> ToolResult:
        """Capture a webcam frame and direct the model to describe it forensically.

        The user wants fine detail — not just the obvious objects — so the frame
        is captured at a higher resolution (small text and distant items survive)
        and paired with a structured, systematic instruction covering the
        person, the room, small/partly-hidden objects, any readable text, and
        the spatial layout.  The multimodal channel reads the pixels directly.
        """
        # 1280px (was 768) keeps book spines, labels and background objects
        # legible enough for the model to read and enumerate them.
        result = await self.capture_live_frame(camera_index, max_side=1280)
        if not result.ok:
            return result
        instruction = (
            "Give a thorough, forensic description of this live webcam frame — "
            "go well beyond the obvious items and work through the scene "
            "systematically:\n"
            "• FOREGROUND — the person (if any): posture, clothing, expression, "
            "and anything they are holding or interacting with.\n"
            "• BACKGROUND — the room and setting in detail. Enumerate every "
            "distinct object you can make out, including small, distant, "
            "reflected or partially hidden ones.\n"
            "• TEXT — read any visible writing: titles on book spines, labels, "
            "posters, screens, packaging, signage, handwriting, logos.\n"
            "• DETAIL — colours, materials, textures, brand names, approximate "
            "counts, and the spatial layout (what sits left/right/above/below "
            "what, and roughly how far away).\n"
            "• LIGHTING & STATE — light sources, any time-of-day cues, and the "
            "general tidiness or condition of the space.\n"
            "Prefer specific, concrete nouns over vague ones (say 'a worn "
            "hardback with a red spine' rather than 'a book'). Where something "
            "is too blurry to be certain, say so and give your best inference."
        )
        focus = prompt.strip()
        if focus:
            instruction = f"Focus especially on: {focus}\n\n{instruction}"
        result.text = (
            "Live camera frame captured for detailed visual analysis.\n" + instruction
        )
        return result

    async def inspect_electronics(
        self, prompt: str = "", camera_index: int | None = None,
        frame: Any = None, *, router: Any = None, mode: str = "electronics",
        grid: tuple[int, int] = (0, 0),
    ) -> ToolResult:
        """Inspect the user's exact captured still, or borrow/capture one on demand.

        The structured report lives in ``evidence[0]``; ``media`` keeps its
        existing JPEG-only contract for the voice channel. Heavy image work
        stays in the inspection worker, so the camera preview remains smooth.
        """
        from .electronics_inspection import inspect_frame

        lock = getattr(self, "_electronics_lock", None)
        if lock is None:
            lock = self._electronics_lock = asyncio.Lock()
        if lock.locked():
            return ToolResult("An electronics scan is already in progress. Wait for its result.", ok=False)
        async with lock:
            if frame is None:
                capture = await self.capture_live_frame(camera_index, max_side=1600)
                if not capture.ok:
                    return capture
                frame = (capture.media or {}).get("data")
                if not frame:
                    return ToolResult("The camera did not provide an inspection frame.", ok=False)
            engine = getattr(self, "ocr_engine", None)
            reader = getattr(engine, "image_to_result", None)
            if not callable(reader):
                reader = self._ocr_image_text
            return await inspect_frame(
                frame, prompt, router=router or getattr(self, "electronics_router", None),
                ocr_reader=reader, mode=mode, grid=grid)

    # ── synchronous workers ───────────────────────────────────────────────────

    def _encode_frame(self, frame: Any, max_side: int, source: str) -> ToolResult:
        """Downscale + JPEG-encode a BGR frame into a ToolResult media payload.

        cv2 is imported here rather than at module scope: this module is
        pulled in on every app start, and OpenCV is a heavy import that only
        the camera paths need (same convention as utils.open_camera_capture
        and ocr_engine's preprocessing). Only ever reached once a camera
        frame exists, so cv2 is necessarily installed by this point."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if max_side and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), PIL_RESAMPLE)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        payload = buffer.getvalue()
        self.bus.log.emit(f"VISION: live frame captured ({source}).")
        # Show the user the frame that was actually taken. Without this the
        # capture is invisible — they only ever get ORION's description of a
        # photo of themselves they never saw.
        try:
            self.bus.camera_frame.emit(
                {"kind": "snapshot", "jpeg": payload, "note": source})
        except Exception:
            pass
        return ToolResult(
            "Live webcam frame captured successfully.\n" + WEBCAM_FRAME_NOTE,
            media={"data": payload, "mime_type": "image/jpeg"},
        )

    def _open_camera(self, index: int) -> tuple[Any, str]:
        """Open the webcam, preferring DirectShow (see utils.open_camera_capture)."""
        return open_camera_capture(index)

    def _capture_live_frame_sync(self, camera_index: int, max_side: int) -> ToolResult:
        try:
            import cv2  # noqa: F401 — presence check; the workers below import it themselves
        except Exception:
            return ToolResult(
                "OpenCV is not installed; webcam capture is unavailable. Install opencv-python to enable the alternate camera input.",
                ok=False,
            )

        # 1) BORROW a frame from an already-open camera (e.g. the face tracker).
        #    Opening a SECOND handle on the same webcam is what fails with
        #    -1072873821 (MSMF) on Windows, so when the tracker is live we simply
        #    take its most recent frame — no second capture, and it is instant.
        source = self._live_frame_source
        owner = getattr(source, "__self__", None)
        source_index = getattr(owner, "camera_index", None)
        if source_index is not None and source_index != camera_index:
            source = None   # an explicit USB microscope must not return the face camera
        is_capturing = getattr(owner, "is_capturing", None)
        if source is not None and callable(is_capturing) and not is_capturing():
            source = None   # no 1.2-second warm-up wait for a stopped tracker
        if source is not None:
            try:
                for _ in range(8):                    # the tracker may be warming up
                    frame = source()
                    if frame is not None:
                        return self._encode_frame(frame, max_side, "borrowed from face tracker")
                    time.sleep(0.15)
            except Exception as exc:
                self.bus.log.emit(f"VISION: could not borrow a tracker frame - {exc}")
            if callable(is_capturing) and is_capturing():
                return ToolResult(
                    "The camera is open but has no fresh frame yet. Wait for the "
                    "preview, or restart the camera if it has disconnected.", ok=False)

        # 2) Otherwise open the camera ourselves (DirectShow first), and warm up
        #    before grabbing — the first few frames are often black or dropped.
        capture, backend = self._open_camera(int(camera_index))
        if capture is None:
            return ToolResult(
                f"Unable to open camera {camera_index}. It may be in use by another "
                "app (or by ORION's own face tracking), disabled, or blocked by "
                "Windows camera-privacy settings.", ok=False)
        try:
            frame = None
            for _ in range(12):                       # discard cold/black warm-up frames
                ok, f = capture.read()
                if ok and f is not None:
                    frame = f
                    break
                time.sleep(0.08)
            if frame is None:
                return ToolResult(
                    "The camera opened but returned no frame — it may be held by "
                    "another app or still warming up. Close other camera apps and "
                    "ask me again.", ok=False)
            return self._encode_frame(frame, max_side, f"camera {camera_index} via {backend}")
        except Exception as exc:
            self.bus.log.emit(f"VISION: camera capture failed - {exc}")
            return ToolResult(f"Camera capture failed: {exc}", ok=False)
        finally:
            try:
                capture.release()
            except Exception:
                pass

    def _analyse_screen_sync(self, prompt: str) -> ToolResult:
        # Native resolution for reading, a bounded copy for sending. Across two
        # screens a 1600 px cap scales the desktop to 42%, which is where small
        # labels stop being legible to the recogniser; the frame that goes to
        # the model still gets scaled below, so nothing grows on the wire.
        image = self.grabber.capture_image(max_side=0)
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
        exact = self._exact_window_text(max_chars=12000)
        ocr_text = self._ocr_image_text(image)
        windows = self._visible_window_titles()
        lines = [
            "Screen analysis complete.",
            f"Resolution (captured): {width}x{height}; brightness index {brightness:.2f}; "
            f"visual entropy {entropy:.2f}; mean RGB {dominant}.",
        ]
        if windows:
            lines.append("Foreground window inventory: " + "; ".join(windows[:10]) + ".")
        if exact:
            # The application's own text — character-perfect, including what
            # is scrolled out of view. Trust it over the OCR below.
            lines.append("EXACT TEXT of the window in front (read from the "
                         "application itself, not OCR — authoritative):\n" + exact)
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
        lines.append(SCREEN_FRAME_NOTE)
        return ToolResult(
            "\n".join(lines),
            media={"data": buffer.getvalue(), "mime_type": "image/jpeg"},
        )

    def _exact_window_text(self, window: str = "", max_chars: int = 60000) -> str:
        """The exact text of a window via UI Automation, or "" when it has none.

        *window* names one ("notepad", "edge", "word", part of a title);
        empty means whichever is in front. Never raises.
        """
        try:
            from . import screen_text
        except Exception:
            return ""
        if not screen_text.available():
            return ""
        try:
            info = screen_text.find_window(window) if window else screen_text.foreground_window()
            if info is None:
                return ""
            # ORION's own windows are not what anybody means by "the screen".
            if "ORION" in info.title.upper() and info.process.lower().startswith(("python", "orion")):
                others = [w for w in screen_text.list_windows()
                          if not ("ORION" in w.title.upper()
                                  and w.process.lower().startswith(("python", "orion")))]
                if window or not others:
                    pass
                else:
                    info = others[0]
            reading = screen_text.read_window(info, max_chars=max_chars)
            return reading.render(max_chars) if reading is not None else ""
        except Exception:
            return ""

    async def read_window(self, window: str = "") -> ToolResult:
        """Exactly what a window says — typed text, documents, web pages.

        UI Automation first (the application's own text, character-perfect);
        OCR of that window's pixels when the application exposes none (games,
        canvases, anything drawn straight to the GPU).
        """
        text = await asyncio.to_thread(self._exact_window_text, window)
        if text:
            return ToolResult(text)
        image = await asyncio.to_thread(self.grabber.capture_image, 0)
        ocr_text = await asyncio.to_thread(self._ocr_image_text, image)
        if ocr_text:
            return ToolResult("That window exposes no text of its own (a game or a "
                              "custom-drawn surface), so this is OCR of the screen:\n"
                              + ocr_text[:8000])
        return ToolResult("I could not read any text from that window.", ok=False)

    def _ocr_sync(self, path: str) -> ToolResult:
        if path.strip():
            text = self._ocr_file_text(Path(path.strip()))
            source = Path(path.strip()).name
        else:
            # Native resolution, deliberately. 1600 was fine for one 1920-wide
            # screen; across two it scales a 3840-wide desktop to 42%, and
            # small text stops being legible to the recogniser — which reads
            # as "I cannot see it" rather than as a resolution problem.
            text = self._ocr_image_text(self.grabber.capture_image(max_side=0))
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
        # 2. OCR keyword sweep across the captured frame, at native
        # resolution — an error dialog's text is small, and it is the whole
        # thing being looked for.
        image = self.grabber.capture_image(max_side=0)
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
        return ToolResult(report + "\n" + SCREEN_FRAME_NOTE,
                          media={"data": buffer.getvalue(), "mime_type": "image/jpeg"})

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

    async def find_element_via_ocr(self, query: str, monitor: int | None = None) -> dict[str, Any] | None:
        """OCR-based fallback for when the accessibility tree comes back
        empty — Electron/canvas UIs, games, some installers — a case that
        previously had no second strategy at all (Mark XXI, Track D2).
        Returns the SAME shape as find_element (role/name/rect/center/
        enabled) so callers can use either interchangeably."""
        if self.ocr_engine is None or not self.ocr_engine.available:
            return None
        return await asyncio.to_thread(self._find_element_via_ocr_sync, query, monitor)

    def _screens_to_search(self, monitor: int | None) -> list[int | None]:
        """Which screens to look at, and in what order.

        An explicit monitor is honoured as given. Otherwise every screen is
        searched SEPARATELY rather than as one panorama, for two reasons that
        both matter on a two-monitor desk:

        * speed. Recognising a 3840x1080 capture in one pass took 10.7 s
          here; the primary screen alone took 1.6 s. Searching screen by
          screen and stopping at the first hit usually costs the 1.6 s,
          because what the user is pointing at is usually in front of them.
        * accuracy. A panorama has to be recognised at whatever resolution it
          is handed, and halving the scale is what makes small labels stop
          being read — which reaches the user as "I cannot see it" rather
          than as a resolution problem.

        The primary screen goes first for the same reason: it is where things
        usually are.
        """
        if monitor is not None:
            return [monitor]
        try:
            count = len(self.grabber._grabber().monitors) - 1   # [0] is the union
        except Exception:
            return [None]
        if count <= 1:
            return [None]
        order = list(range(1, count + 1))
        try:
            import mss

            monitors = self.grabber._grabber().monitors
            primary = next((i for i in order
                            if monitors[i]["left"] == 0 and monitors[i]["top"] == 0), 1)
            order.remove(primary)
            order.insert(0, primary)
        except Exception:
            pass
        return list(order)

    def _find_element_via_ocr_sync(self, query: str, monitor: int | None) -> dict[str, Any] | None:
        screens = self._screens_to_search(monitor)
        shots: dict[Any, tuple[int, int, Any] | None] = {}

        def capture(screen: int | None) -> tuple[int, int, Any] | None:
            """The screen, grabbed at most once for the whole search.

            Each reader sweeps every screen, so without this the second
            reader would re-grab what the first already looked at. Reusing
            the capture is also the more honest answer: this is one look at
            the desktop, not several at different moments.
            """
            if screen not in shots:
                try:
                    left, top, _w, _h = self.grabber.monitor_bounds(screen)
                    shots[screen] = (left, top, self.grabber.capture_image(
                        max_side=0, monitor=screen))
                except Exception:
                    shots[screen] = None
            return shots[screen]

        # Reader outer, screen inner: the fast reader is tried on EVERY screen
        # before the slow one is paid for on any of them. MEASURED here —
        # tesseract is ~1.9 s a screen, rapidocr 2.4-9.0 s — so for a target
        # on the second screen that tesseract can read, screen-outer order
        # spends rapidocr's 2.4 s on the first screen for nothing. MEASURED
        # end to end: finding "Terminal" on the right-hand monitor went
        # 7.55 s -> 5.12 s.
        for reader in self._sweep_readers():
            for screen in screens:
                shot = capture(screen)
                if shot is None:
                    continue
                left, top, image = shot
                try:
                    box = (self.ocr_engine.find_text_box(image, query)
                           if reader is None else
                           self.ocr_engine.find_text_box(image, query,
                                                         readers=(reader,)))
                except Exception:
                    continue
                if box is None:
                    continue
                x, y, w, h = box
                # Back into virtual-desktop pixels, which is the only
                # coordinate space a click is meaningful in once there is
                # more than one screen.
                x, y = x + left, y + top
                return {
                    "role": "OCR",
                    "name": query,
                    "rect": (x, y, w, h),
                    "center": (x + w // 2, y + h // 2),
                    "enabled": True,
                    "monitor": screen,
                }
        return None

    def _sweep_readers(self) -> tuple[Any, ...]:
        """The OCR readers to sweep separately, or ``(None,)`` for one pass.

        An engine that cannot name its readers — a stub, or an older one — is
        asked once per screen and left to pick for itself, which is what this
        did before. Reaching for an attribute the collaborator need not have
        is not worth a crash on the path that locates things to click.
        """
        try:
            readers = tuple(self.ocr_engine.word_readers())
        except Exception:
            readers = ()
        return readers or (None,)

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
        # Fold both sides: control names carry mnemonic ampersands ('&Save'),
        # invisible Unicode, non-breaking spaces, and an inconsistently
        # punctuated "..." convention — all of which defeat naive
        # substring matching (Mark XXI, Track D1).
        query = fold_control_label(query)
        if not query:
            return None
        elements = self._detect_elements_sync(kinds, max_elements=400)
        if not elements:
            return None
        candidates = [e for e in elements if query in fold_control_label(e["name"])
                      or set(query.split()) & set(fold_control_label(e["name"]).split())]
        if not candidates:
            return None
        return sorted(candidates, key=lambda el: self._element_score(el, query))[0]

    @staticmethod
    def _element_score(el: dict[str, Any], query: str) -> tuple[int, int]:
        """Rank: exact name, then startswith, then substring, then token
        overlap — lower is better. *query* must already be folded."""
        name = fold_control_label(el["name"])
        if name == query:
            return (0, len(name))
        if name.startswith(query):
            return (1, len(name))
        if query in name:
            return (2, len(name))
        overlap = len(set(query.split()) & set(name.split()))
        return (3 if overlap else 9, -overlap)

    async def nearby_candidates(self, query: str, kinds: str = "all", limit: int = 5) -> list[str]:
        """The *limit* closest-scoring element names to *query*, regardless
        of whether any would actually match — diagnostic-only (Mark XXI,
        Track D5), used to turn a bare "not found" into "the nearest
        matches on screen were X, Y, Z" so a failed lookup is something a
        model can retry intelligently against instead of a dead end."""
        return await asyncio.to_thread(self._nearby_candidates_sync, query, kinds, limit)

    def _nearby_candidates_sync(self, query: str, kinds: str = "all", limit: int = 5) -> list[str]:
        query = fold_control_label(query)
        if not query:
            return []
        elements = self._detect_elements_sync(kinds, max_elements=400)
        if not elements:
            return []
        scored = sorted(elements, key=lambda el: self._element_score(el, query))
        # Meaning first when the local encoder is warm: letter overlap puts
        # arbitrary names in front of "sign in" when the button says "Log
        # in". This list is only SHOWN to the model to retry with — nothing is
        # clicked on a meaning match (measured: 8/13 right, and the misses
        # were confident enough to press the wrong button).
        scored = _meaning_first(query, scored)
        seen: set[str] = set()
        names: list[str] = []
        for el in scored:
            name = str(el.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= limit:
                break
        return names

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
