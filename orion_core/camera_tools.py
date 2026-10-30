"""
Camera tools - what a person expects a camera to do, done on this PC.

The camera could be watched, inspected and asked about, but it could not do
the everyday things: read a QR code or a barcode, scan a page flat, take a
photo or a short clip that ends up in Pictures or Videos, or see in a dim
room. Each of these is plain OpenCV arithmetic on a frame the face tracker
already owns - no second camera handle (Windows refuses one), no model, no
network:

    read_codes(frame)          QR codes and 1-D barcodes, with what they ARE:
                               a link, a Wi-Fi network, a contact, a product
    find_document(frame)       the page's four corners, if a page is in view
    flatten_document(...)      that page warped square-on and cleaned up
    enhance_low_light(frame)   contrast-limited equalisation for dim rooms
    save_image(image, kind)    a photo / scan in Pictures\\ORION
    ClipRecorder               a short MP4 in Videos\\ORION from the live feed

Everything takes and returns BGR numpy arrays and degrades to an empty
result (never an exception) when OpenCV is missing.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def _cv2() -> Any:
    try:
        import cv2
        return cv2
    except Exception:
        return None


# ── codes ─────────────────────────────────────────────────────────────────────

@dataclass
class Code:
    kind: str                     # "QR", "EAN_13", "CODE_128", ...
    text: str
    points: list[list[float]] = field(default_factory=list)   # corners, frame pixels
    meaning: str = "text"         # link / wifi / contact / email / phone / product / text
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "points": self.points,
                "meaning": self.meaning, "summary": self.summary}


_WIFI = re.compile(r"^WIFI:", re.I)


def describe_code_text(text: str, kind: str = "QR") -> tuple[str, str]:
    """(meaning, a one-line summary) for decoded code text. A Wi-Fi code's
    password is never repeated in the summary."""
    value = str(text or "").strip()
    if re.match(r"^https?://", value, re.I):
        host = re.sub(r"^https?://", "", value, flags=re.I).split("/")[0]
        return "link", f"A link to {host}"
    if _WIFI.match(value):
        fields = dict(re.findall(r"([STPH]):((?:\\.|[^;])*)", value[5:]))
        name = fields.get("S", "").replace("\\;", ";").replace("\\:", ":") or "an unnamed network"
        security = fields.get("T", "") or "no password"
        has_password = bool(fields.get("P"))
        return "wifi", (f"Wi-Fi network \"{name}\" ({security}"
                        + (", password included)" if has_password else ")"))
    if value.upper().startswith(("BEGIN:VCARD", "MECARD:")):
        name = _contact_name(value)
        return "contact", f"A contact card for {name}" if name else "A contact card"
    if value.lower().startswith("mailto:"):
        return "email", f"An email address: {value[7:].split('?')[0]}"
    if value.lower().startswith("tel:"):
        return "phone", f"A phone number: {value[4:]}"
    if kind.upper().startswith(("EAN", "UPC")) and value.isdigit():
        return "product", f"Product barcode {value}"
    if kind.upper() != "QR" and value:
        return "product", f"{kind} barcode {value}"
    return "text", value[:120]


def _contact_name(card: str) -> str:
    """The person a vCard or MECARD names. A vCard's FN is the display name;
    failing that its N is "Last;First;…", and a MECARD's N is "Last,First".
    Fields are matched at their own line or field start - "BEGIN:VCARD"
    itself ends in "N:VCARD"."""
    full = re.search(r"(?im)^FN(?:;[^:\n]*)?:(.+)$", card)
    if full and full.group(1).strip():
        return full.group(1).strip()
    if card.upper().startswith("MECARD:"):
        field = re.search(r"(?i)(?:^MECARD:|;)N:((?:\\.|[^;])*)", card)
        parts = field.group(1).split(",") if field else []
    else:
        field = re.search(r"(?im)^N(?:;[^:\n]*)?:(.+)$", card)
        parts = field.group(1).split(";") if field else []
    parts = [p.strip() for p in parts[:2] if p.strip()]
    return " ".join(reversed(parts))


def _corners(points: Any, index: int) -> list[list[float]]:
    try:
        return [[float(x), float(y)] for x, y in points[index]]
    except Exception:
        return []


def read_codes(frame: Any) -> list[Code]:
    """Every QR code and barcode readable in the frame (duplicates merged)."""
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return []
    found: list[Code] = []
    seen: set[str] = set()

    def add(kind: str, text: str, corners: list[list[float]]) -> None:
        text = str(text or "")
        if not text or (kind, text) in seen:
            return
        seen.add((kind, text))          # type: ignore[arg-type]
        meaning, summary = describe_code_text(text, kind)
        found.append(Code(kind, text, corners, meaning, summary))

    attempts = [frame]
    try:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if getattr(frame, "ndim", 2) == 3 else frame
        attempts.append(grey)
    except Exception:
        pass
    for image in attempts:
        # Three QR readers in turn: the plain multi-code detector misses some
        # codes the Aruco-based one and the single-code one read (measured:
        # "https://example.org/a" at 6 px a module).
        for make in ("QRCodeDetector", "QRCodeDetectorAruco"):
            if any(code.kind == "QR" for code in found):
                break
            try:
                ok, texts, points, _ = getattr(cv2, make)().detectAndDecodeMulti(image)
                if ok and points is not None:
                    for index, text in enumerate(texts):
                        add("QR", text, _corners(points, index))
            except Exception:
                pass
        if not any(code.kind == "QR" for code in found):
            try:
                text, points, _ = cv2.QRCodeDetector().detectAndDecode(image)
                if text and points is not None:
                    add("QR", text, _corners(points, 0))
            except Exception:
                pass
        try:
            detector = cv2.barcode.BarcodeDetector()
            result = detector.detectAndDecodeWithType(image)
            ok, texts, kinds, points = result[0], result[1], result[2], result[3]
            if ok and points is not None:
                for index, text in enumerate(texts):
                    kind = str(kinds[index] if index < len(kinds) else "BARCODE") or "BARCODE"
                    add(kind, text, _corners(points, index))
        except Exception:
            pass
        if found:
            break
    return found


# ── documents ─────────────────────────────────────────────────────────────────

def _order(quad: Any) -> Any:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    import numpy as np
    pts = np.asarray(quad, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype="float32")


def find_document(frame: Any, min_fraction: float = 0.18) -> list[list[float]] | None:
    """The four corners of the largest page-like quadrilateral, or None."""
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        scale = 600.0 / max(h, w) if max(h, w) > 600 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        edges = cv2.Canny(grey, 50, 150)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area_min = min_fraction * small.shape[0] * small.shape[1]
        best = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            if cv2.contourArea(contour) < area_min:
                break
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best = approx
                break
        if best is None:
            return None
        return [[float(x) / scale, float(y) / scale] for x, y in best.reshape(4, 2)]
    except Exception:
        return None


def flatten_document(frame: Any, corners: Any, *, clean: bool = True) -> Any:
    """The page square-on; *clean* gives the white-paper scanner look."""
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return None
    import numpy as np
    tl, tr, br, bl = _order(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 40 or height < 40:
        return None
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                      dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), target)
    flat = cv2.warpPerspective(frame, matrix, (width, height))
    return _clean_page(flat) if clean else flat


def _clean_page(flat: Any) -> Any:
    cv2 = _cv2()
    import numpy as np
    grey = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY) if flat.ndim == 3 else flat
    # Divide out uneven lighting (shadow of the phone/hand), then a gentle
    # threshold so text is crisp without destroying photos on the page.
    background = cv2.medianBlur(cv2.dilate(grey, np.ones((7, 7), np.uint8)), 21)
    even = cv2.divide(grey, background, scale=255)
    return cv2.normalize(even, None, 0, 255, cv2.NORM_MINMAX)


def scan_document(frame: Any) -> tuple[Any, bool]:
    """(the scanned page, whether a page outline was found). With no outline
    the whole frame is cleaned, so a close-up of a page still scans."""
    corners = find_document(frame)
    if corners is not None:
        flat = flatten_document(frame, corners)
        if flat is not None:
            return flat, True
    if frame is None or _cv2() is None:
        return None, False
    # Nothing to warp: clean the frame as it is (a warp onto itself would
    # shave a pixel off each edge).
    return _clean_page(frame), False


# ── light ─────────────────────────────────────────────────────────────────────

def mean_brightness(frame: Any) -> float:
    try:
        cv2 = _cv2()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return float(grey.mean())
    except Exception:
        return 128.0


def enhance_low_light(frame: Any) -> Any:
    """A dim frame made readable: contrast-limited equalisation of lightness
    (no colour shift) and a lift of the shadows. Returns the frame unchanged
    when it is not dim or OpenCV is missing."""
    cv2 = _cv2()
    if cv2 is None or frame is None or getattr(frame, "ndim", 0) != 3:
        return frame
    try:
        import numpy as np
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(light)
        out = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
        if out.mean() < 90:                       # still dark: lift shadows
            table = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype="uint8")
            out = cv2.LUT(out, table)
        return out
    except Exception:
        return frame


# ── saving ────────────────────────────────────────────────────────────────────

def media_folder(kind: str) -> Path:
    """Pictures\\ORION or Videos\\ORION - the user's real folders (OneDrive
    redirection included), created on first use."""
    base: Path | None = None
    try:
        from .app_resolver import known_folder
        base = known_folder("videos" if kind == "videos" else "pictures")
    except Exception:
        base = None
    if base is None:
        base = Path.home() / ("Videos" if kind == "videos" else "Pictures")
    folder = base / "ORION"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_image(image: Any, kind: str = "photo", folder: Path | None = None) -> Path | None:
    """Write *image* as a JPEG photo or a PNG scan; returns the path. Encoded
    in memory and written with Python: cv2.imwrite cannot write to paths with
    non-ASCII characters on Windows."""
    cv2 = _cv2()
    if cv2 is None or image is None:
        return None
    extension = ".png" if kind == "scan" else ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 92] if extension == ".jpg" else []
    ok, data = cv2.imencode(extension, image, params)
    if not ok:
        return None
    target_dir = folder or media_folder("pictures")
    stamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    label = kind if kind in ("scan", "watch") else "photo"
    path = target_dir / f"ORION {label} {stamp}{extension}"
    counter = 2
    while path.exists():
        path = target_dir / f"ORION {label} {stamp} ({counter}){extension}"
        counter += 1
    path.write_bytes(data.tobytes())
    return path


class ClipRecorder:
    """Record the live feed to an MP4 in Videos\\ORION.

    Frames come from *source* (the tracker's latest_frame), sampled at *fps*
    on a daemon thread - the capture loop is never touched and no second
    camera handle is opened. ``on_done(path | None, message)`` is called from
    that thread when the clip ends."""

    MAX_SECONDS = 120

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.path: Path | None = None

    @property
    def recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source: Callable[[], Any], seconds: float = 10.0, fps: float = 15.0,
              on_done: Callable[[Path | None, str], None] | None = None,
              folder: Path | None = None) -> bool:
        if self.recording or _cv2() is None:
            return False
        seconds = max(1.0, min(float(seconds), self.MAX_SECONDS))
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(source, seconds, fps, on_done, folder),
            name="orion-clip-recorder", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _run(self, source: Callable[[], Any], seconds: float, fps: float,
             on_done: Callable[[Path | None, str], None] | None, folder: Path | None) -> None:
        cv2 = _cv2()
        writer = None
        size = (0, 0)
        frames = 0
        path: Path | None = None
        message = ""
        try:
            deadline = time.monotonic() + seconds
            interval = 1.0 / max(1.0, fps)
            while time.monotonic() < deadline and not self._stop.is_set():
                started = time.monotonic()
                frame = source()
                if frame is not None and getattr(frame, "ndim", 0) == 3:
                    if writer is None:
                        h, w = frame.shape[:2]
                        size = (w, h)
                        stamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
                        path = (folder or media_folder("videos")) / f"ORION clip {stamp}.mp4"
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError("the video encoder would not open")
                    if (frame.shape[1], frame.shape[0]) != size:
                        # The camera changed resolution mid-clip; a writer
                        # silently drops frames of the wrong size.
                        frame = cv2.resize(frame, size)
                    writer.write(frame)
                    frames += 1
                time.sleep(max(0.0, interval - (time.monotonic() - started)))
            if frames == 0:
                message = "No camera frames arrived, so nothing was recorded."
                path = None
            else:
                message = f"Recorded {frames / fps:.0f} seconds."
        except Exception as exc:
            message = f"Recording failed: {exc}"
            path = None
        finally:
            if writer is not None:
                writer.release()
            if path is not None and (not path.exists() or path.stat().st_size == 0):
                path, message = None, message or "The clip could not be written."
            self.path = path
            if on_done is not None:
                try:
                    on_done(path, message)
                except Exception:
                    pass


def open_file(path: Path) -> bool:
    """Show a saved photo, scan or clip in its default app."""
    from .execution_policy import require_foreground
    if require_foreground("shell_open_camera_artifact") is not None:
        return False
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


__all__ = ["Code", "read_codes", "describe_code_text", "find_document", "flatten_document",
           "scan_document", "enhance_low_light", "mean_brightness", "media_folder",
           "save_image", "ClipRecorder", "open_file"]
