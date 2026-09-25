"""
One way to turn pixels into words, for every path that is not Live.

Gemini Live reads an attached frame itself. Everything else used to pass on
the TEXT that came with the frame — and for the camera that text is the
instruction ("give a thorough, forensic description of this frame…"), so:

* the text fallback (voice channel down) handed a model the instruction and
  no image, and it either invented a room or said it could not see;
* the perception loop stored that same instruction as what it had "seen";
* video analysis called a hard-coded model through its own client, outside
  the router's cooldowns and key rotation.

``describe_image`` routes the picture through ``ProviderRouter.generate_vision``
(which tries the vision-capable text providers and the Gemini key's image
endpoint) and, when nothing can look, says so in words a person understands —
the camera working and no model being able to read it are different facts.
"""

from __future__ import annotations

import base64
from typing import Any

NO_VISION = ("The camera worked, but no vision-capable model is reachable right "
             "now, so I can't tell you what it shows.")

INSTRUCTION = (
    "You are ORION's eyes. Describe only what is actually visible in the image "
    "and answer the request directly. Read any legible text exactly. Be "
    "specific and concrete; say plainly when something is too small or blurred "
    "to be sure. Do not invent anything that is not in the image."
)


def image_bytes(media: Any) -> bytes | None:
    """The JPEG/PNG bytes of a media payload ({"data": bytes|base64, ...})."""
    if not isinstance(media, dict):
        return None
    if not str(media.get("mime_type") or "").startswith("image/"):
        return None
    data = media.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str) and data:
        try:
            return base64.b64decode(data, validate=False)
        except Exception:
            return None
    return None


async def describe_image(router: Any, image: bytes, request: str, *,
                         max_tokens: int = 450) -> tuple[bool, str]:
    """(ok, text): a description of *image* aimed at *request*, or an honest
    reason nothing could look. Never raises."""
    generate = getattr(router, "generate_vision", None)
    if generate is None or not image:
        return False, NO_VISION
    prompt = " ".join(str(request or "").split())[:1200] or "Describe what you see."
    try:
        _profile, text = await generate(image, prompt, instruction=INSTRUCTION,
                                        task="vision", max_tokens=max_tokens)
    except Exception as exc:
        name = type(exc).__name__
        if name == "NoTextProviderError":
            return False, NO_VISION
        return False, ("The camera worked, but the vision models could not analyse "
                       f"the image just now ({name}). Ask me again in a moment.")
    text = str(text or "").strip()
    return (True, text) if text else (False, NO_VISION)


def frame_to_jpeg(frame: Any, max_side: int = 960, quality: int = 80) -> bytes | None:
    """A BGR numpy frame (OpenCV) as JPEG bytes, or None."""
    try:
        import cv2

        h, w = frame.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w) or 1))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return bytes(buf) if ok else None
    except Exception:
        return None


__all__ = ["INSTRUCTION", "NO_VISION", "describe_image", "frame_to_jpeg", "image_bytes"]
