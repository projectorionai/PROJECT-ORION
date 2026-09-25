"""
Downloading model files — once, completely, or not at all.

ORION fetches a few vision models on first use (hand landmarks, body pose,
object detection) into ``CONFIG_DIR/models``. The first of those used
``urllib.request.urlretrieve`` straight onto the final path, so a download cut
off halfway left a truncated file that ``exists()`` then accepted forever:
gesture control was broken until someone found and deleted it by hand.

Here the bytes go to a ``.part`` file beside the target, are checked against
the expected size (and SHA-256 where one is published), and only then renamed
into place — atomically, so there is never a moment when a half-written model
sits under the real name.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable


class ModelDownloadError(RuntimeError):
    """The model could not be fetched intact."""


def download_verified(url: str, target: Path, *, size: int | None = None,
                      sha256: str | None = None,
                      opener: Callable[..., Any] = urllib.request.urlopen,
                      timeout: float = 60.0) -> int:
    """Fetch *url* to *target*, verified. Returns the byte count.

    With no pinned *size* (a "latest" URL whose file may legitimately change),
    completeness is checked against the server's own Content-Length instead.

    Raises ModelDownloadError, leaving nothing behind, if the transfer fails,
    is the wrong size, or does not match *sha256*."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    received = 0
    try:
        with opener(url, timeout=timeout) as response, open(partial, "wb") as out:
            if size is None:
                announced = (getattr(response, "headers", None) or {}).get("Content-Length")
                if announced and str(announced).isdigit():
                    size = int(announced)
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                received += len(block)
                if size is not None and received > size * 2:
                    raise ModelDownloadError("the download is far larger than the model should be")
        if size is not None and received != size:
            raise ModelDownloadError(
                f"the download stopped at {received:,} of {size:,} bytes")
        if sha256 is not None and digest.hexdigest() != sha256.lower():
            raise ModelDownloadError("the download did not match the model's published checksum")
        os.replace(partial, target)
        return received
    except ModelDownloadError:
        _discard(partial)
        raise
    except Exception as exc:
        _discard(partial)
        raise ModelDownloadError(str(exc)) from exc


def usable(path: Path, *, size: int | None = None) -> bool:
    """A model file that exists and — when its size is known — is complete."""
    try:
        actual = Path(path).stat().st_size
    except OSError:
        return False
    return actual > 0 and (size is None or actual == size)


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


__all__ = ["ModelDownloadError", "download_verified", "usable"]
