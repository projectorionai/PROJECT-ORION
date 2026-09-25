"""
Shoulder (CAP-03) — "read what I'm looking at."

    "I want ORION to be able to analyse images ... watch and identify what
     course of action to take dependent on the prompt."

The video pipeline already frames and understands moving pictures. This points
the same understanding at a SINGLE still of the active screen, on request, so
"what am I looking at — help me with this" needs no copy-paste.

Two things make it safe rather than creepy:

* **On demand only.** There is no passive loop, no background recording, no
  timer. A capture happens exactly when this is called and the image is never
  written to disk or kept — it is described and discarded within the call. That
  is the whole privacy model, and it is enforced by there being no code that
  captures anywhere else.
* **It says when it looked.** Every read logs that a single screenshot was
  taken on request, so a glance is always visible in the record.

Capture and description are both injected — the module never hard-depends on a
screenshot backend or a vision model — so the flow is tested end to end with a
fake screen and a fake describer, and it degrades to a clear message (rather
than a crash or a lie) when no backend is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

# (image_bytes) -> PNG/JPEG bytes of the current screen
CaptureFn = Callable[[], bytes]
# (image_bytes, prompt) -> (description, how) e.g. ("A code editor…", "model")
DescribeFn = Callable[[bytes, str], Awaitable[tuple[str, str]]]
# (prompt) -> synthesised action text
GenerateFn = Callable[[str], Awaitable[str]]


@dataclass
class ScreenReading:
    ok: bool
    description: str          # what is on screen
    action: str              # what to do about it, for the prompt
    via: str                 # "model" | "ocr" | "none"
    note: str = ""           # why it degraded, when it did

    def describe(self) -> str:
        if not self.ok:
            return self.note or "I couldn't read the screen just now."
        out = self.description.strip()
        if self.action.strip():
            out += "\n\n" + self.action.strip()
        return out


def default_capture() -> bytes:
    """Grab the current screen as PNG bytes. Tries mss, then Pillow's ImageGrab.

    Raises RuntimeError with a plain message when neither backend is available,
    so the caller can degrade rather than crash."""
    # mss is fast and cross-monitor.
    try:
        import mss
        import mss.tools
        factory = getattr(mss, "MSS", None) or mss.mss   # MSS is the newer name
        with factory() as sct:
            shot = sct.grab(sct.monitors[0])          # the full virtual screen
            return mss.tools.to_png(shot.rgb, shot.size)
    except Exception:
        pass
    try:
        import io
        from PIL import ImageGrab
        image = ImageGrab.grab()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:
        raise RuntimeError(
            "no screen-capture backend available (install 'mss' or 'pillow')"
        ) from exc


def capture_available() -> bool:
    for module in ("mss", "PIL.ImageGrab"):
        try:
            __import__(module)
            return True
        except Exception:
            continue
    return False


class ScreenReader:
    """Capture the screen once, describe it, and say what to do about it."""

    def __init__(self, capture: Optional[CaptureFn] = None,
                 describe: Optional[DescribeFn] = None,
                 generate: Optional[GenerateFn] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self._capture = capture or default_capture
        self._describe = describe
        self._generate = generate
        self._log = log

    async def read(self, prompt: str = "") -> ScreenReading:
        # 1. capture — the only place a screenshot is ever taken
        try:
            image = self._capture()
        except Exception as exc:
            return ScreenReading(False, "", "", "none",
                                 note=f"Couldn't capture the screen: {exc}")
        if not image:
            return ScreenReading(False, "", "", "none",
                                 note="The screen capture came back empty.")
        if self._log is not None:
            try:
                self._log("VISION: read the screen once, on request.")
            except Exception:
                pass

        # 2. describe what's there
        description, via = "", "none"
        if self._describe is not None:
            try:
                description, via = await self._describe(image, prompt)
            except Exception:
                description, via = "", "none"
        if not description:
            return ScreenReading(
                False, "", "", "none",
                note="I captured the screen but have no way to see it right now "
                     "(no vision model or OCR available).")

        # 3. decide what to do about it, for the prompt
        action = ""
        if prompt and self._generate is not None:
            try:
                ask = (
                    "You are looking over the user's shoulder at their screen. "
                    f"On screen: {description}\n\nThey asked: {prompt}\n\n"
                    "Give a short, direct answer or the next action to take — "
                    "no preamble.")
                action = str(await self._generate(ask) or "").strip()
            except Exception:
                action = ""

        return ScreenReading(True, description, action, via)


__all__ = ["ScreenReader", "ScreenReading", "default_capture",
           "capture_available", "CaptureFn", "DescribeFn", "GenerateFn"]
