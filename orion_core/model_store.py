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
import math
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .atomic_io import _replace


class ModelDownloadError(RuntimeError):
    """The model could not be fetched intact."""


def _positive_timeout(value: Any) -> bool:
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0)
    except (OverflowError, TypeError, ValueError):
        return False


def download_verified(url: str, target: Path, *, size: int | None = None,
                      sha256: str | None = None,
                      opener: Callable[..., Any] = urllib.request.urlopen,
                      timeout: float = 60.0, max_bytes: int = 1 << 30,
                      total_timeout: float = 300.0) -> int:
    """Fetch *url* to *target*, verified. Returns the byte count.

    With no pinned *size* (a "latest" URL whose file may legitimately change),
    completeness is checked against the server's own Content-Length instead.

    Unannounced transfers are capped by *max_bytes*. The total deadline is
    checked between reads; a blocked read is also bounded by *timeout*.

    Raises ModelDownloadError on an incomplete or invalid transfer, preserving
    an existing target and removing this attempt's partial file."""
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0
            or not _positive_timeout(total_timeout) or not _positive_timeout(timeout)):
        raise ModelDownloadError("download limits must be finite and positive")
    partial: Path | None = None
    digest = hashlib.sha256()
    received = 0
    deadline = time.monotonic() + total_timeout
    try:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
        partial = Path(temporary)
        with os.fdopen(handle, "wb") as out, opener(url, timeout=timeout) as response:
            if size is None:
                announced = (getattr(response, "headers", None) or {}).get("Content-Length")
                if announced and str(announced).isdigit():
                    size = int(announced)
            if size is not None and (size <= 0 or size > max_bytes):
                raise ModelDownloadError("the announced model size is empty or exceeds the download limit")
            while True:
                if time.monotonic() >= deadline:
                    raise ModelDownloadError("the download exceeded its total time budget")
                block = response.read(1 << 20)
                if time.monotonic() >= deadline:
                    raise ModelDownloadError("the download exceeded its total time budget")
                if not block:
                    break
                if received + len(block) > max_bytes or (size is not None and received + len(block) > size):
                    raise ModelDownloadError("the download is larger than the model should be")
                out.write(block)
                digest.update(block)
                received += len(block)
            out.flush()
            os.fsync(out.fileno())
        if received == 0:
            raise ModelDownloadError("the download returned an empty model")
        if size is not None and received != size:
            raise ModelDownloadError(
                f"the download stopped at {received:,} of {size:,} bytes")
        if sha256 is not None and digest.hexdigest() != sha256.lower():
            raise ModelDownloadError("the download did not match the model's published checksum")
        _replace(str(partial), target)
        return received
    except ModelDownloadError:
        raise
    except Exception as exc:
        raise ModelDownloadError(str(exc)) from exc
    finally:
        # This attempt owns this path, including cancellation and failed checks.
        # Another download can never remove or continue writing our candidate.
        if partial is not None:
            _discard(partial)


def usable(path: Path, *, size: int | None = None) -> bool:
    """A model file that exists and — when its size is known — is complete."""
    try:
        model = Path(path)
        if not model.is_file():
            return False
        actual = model.stat().st_size
    except OSError:
        return False
    return actual > 0 and (size is None or actual == size)


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


__all__ = ["ModelDownloadError", "download_verified", "usable"]
