"""
Camera device selection — which webcam PresenceMonitor, FaceTracker,
GestureEngine and VisionAgent's on-demand capture all open by default.

Cameras are remembered by NAME (Mark XXXI). An integer index is only where a
camera happens to sit in one backend's list today: plugging in another
device, starting a virtual camera (OBS registers one) or a Windows update
renumbers the list, and ORION then opened the wrong camera or none — "the
camera lab doesn't recognise the cameras I am using". Names come from
``cv2_enumerate_cameras``, which reports each backend's own index for a
device, so a remembered name resolves to the right index for whichever
backend is opening it. Without that package the stored index is used, as
before.

Media Foundation (MSMF) and DirectShow (DSHOW) number devices differently —
on this machine MSMF lists only the C920 while DSHOW also lists the OBS
Virtual Camera — so resolution is always FOR a backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

CAMERA_CONFIG_PATH = CONFIG_DIR / "camera.json"

#: Names that are software cameras, not lenses. Never chosen automatically.
_VIRTUAL_HINTS = ("virtual", "obs", "droidcam", "snap camera", "xsplit", "ndi",
                  "manycam", "splitcam", "iriun", "epoccam", "camo")


@dataclass
class CameraDevice:
    index: int
    name: str
    backend: str            # "MSMF" or "DSHOW"
    vid: int = 0
    pid: int = 0

    @property
    def virtual(self) -> bool:
        low = self.name.lower()
        return any(hint in low for hint in _VIRTUAL_HINTS)

    def label(self) -> str:
        return self.name + (" (virtual)" if self.virtual else "")


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_config(cfg: dict[str, Any]) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(CAMERA_CONFIG_PATH, json.dumps(cfg, indent=2), encoding="utf-8")
    except OSError:
        pass


def list_cameras(backend: str = "MSMF") -> list[CameraDevice]:
    """Every camera *backend* can open, with its friendly name. [] if unknown."""
    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception:
        return []
    api = getattr(cv2, f"CAP_{backend.upper()}", None)
    if api is None:
        return []
    try:
        found = list(enumerate_cameras(api))
    except Exception:
        return []
    out: list[CameraDevice] = []
    for info in found:
        index = int(getattr(info, "index", 0) or 0)
        # enumerate_cameras reports index + api for some versions; keep the
        # plain per-backend index OpenCV expects.
        if index >= api:
            index -= api
        out.append(CameraDevice(index, str(getattr(info, "name", "") or f"Camera {index}"),
                                backend.upper(), int(getattr(info, "vid", 0) or 0),
                                int(getattr(info, "pid", 0) or 0)))
    return out


def all_cameras() -> list[CameraDevice]:
    """One entry per physical/virtual camera: MSMF's view first (it delivers
    the resolution), then anything only DirectShow can see."""
    msmf = list_cameras("MSMF")
    names = {c.name for c in msmf}
    return msmf + [c for c in list_cameras("DSHOW") if c.name not in names]


def preferred_name() -> str:
    return str(_load_config().get("name") or "")


def resolve_device(hi_res: bool = True) -> tuple[int, str]:
    """(index, backend) to open: the remembered camera if it is present,
    otherwise the first real (non-virtual) camera, otherwise the stored index.

    *backend* is "CAP_MSMF" / "CAP_DSHOW" when known, "" when the caller's
    own order should decide.
    """
    cfg = _load_config()
    wanted = str(cfg.get("name") or "")
    order = ("MSMF", "DSHOW") if hi_res else ("DSHOW", "MSMF")
    listed = {backend: list_cameras(backend) for backend in order}
    if wanted:
        for backend in order:
            for cam in listed[backend]:
                if cam.name == wanted:
                    return cam.index, f"CAP_{backend}"
    for backend in order:
        for cam in listed[backend]:
            if not cam.virtual:
                return cam.index, f"CAP_{backend}"
    try:
        return int(cfg.get("index", 0)), ""
    except (TypeError, ValueError):
        return 0, ""


def resolve() -> int:
    """The camera index every capture site should open (see resolve_device)."""
    return resolve_device(hi_res=True)[0]


def set_device(choice: Any) -> str:
    """Remember a camera by NAME (or index) from now on."""
    cameras = all_cameras()
    chosen: CameraDevice | None = None
    if cameras and isinstance(choice, str) and not choice.strip().lstrip("-").isdigit():
        low = choice.strip().lower()
        chosen = next((c for c in cameras if c.name.lower() == low), None) or \
            next((c for c in cameras if low in c.name.lower()), None)
        if chosen is None:
            names = ", ".join(c.name for c in cameras) or "none found"
            return f"No camera called '{choice}'. Cameras: {names}."
    else:
        try:
            index = int(choice)
        except (TypeError, ValueError):
            return f"'{choice}' is not a valid camera index."
        if index < 0:
            return "Camera index must be 0 or greater."
        chosen = next((c for c in cameras if c.index == index), None)
        if chosen is None:
            cfg = _load_config()
            cfg["index"] = index
            cfg.pop("name", None)
            _save_config(cfg)
            return f"Default camera set to index {index}. Takes effect next time a capture starts."
    cfg = _load_config()
    cfg["name"] = chosen.name
    cfg["index"] = chosen.index
    _save_config(cfg)
    return (f"Default camera set to {chosen.label()} (index {chosen.index}). "
            "Takes effect next time a capture starts.")


def describe() -> str:
    cameras = all_cameras()
    index, backend = resolve_device()
    current = next((c for c in cameras if c.index == index and
                    f"CAP_{c.backend}" == backend), None)
    if not cameras:
        return f"ORION opens camera index {index} by default."
    listing = "; ".join(f"{c.label()} [{c.backend} {c.index}]" for c in cameras)
    return (f"ORION uses {current.label() if current else f'camera index {index}'}. "
            f"Cameras found: {listing}.")


__all__ = ["CAMERA_CONFIG_PATH", "CameraDevice", "all_cameras", "describe",
           "list_cameras", "preferred_name", "resolve", "resolve_device", "set_device"]
