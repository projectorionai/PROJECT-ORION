"""
Write a file so that a reader sees the old contents or the new ones, never half.

``Path.write_text`` truncates its target first and then writes into it. If the
process dies in between — a crash, a forced exit from the shutdown watchdog, a
power cut — or the write fails part-way because OneDrive or an antivirus
scanner is holding the file, what is left on disk is a truncated file. For
ORION's JSON state that is worse than a failed save: the next start cannot
parse it, and ``api_keys.json`` in that state takes every provider and key with
it. The audit of 2026-09-23 found 45 state files written that way.

``atomic_write_text`` writes to a temporary file in the SAME directory (so the
final step is a rename on one volume, not a copy), flushes it to disk, then
replaces the target in one operation. Its contract is otherwise identical to
``Path.write_text``: same encoding and newline handling, and a failure raises —
the difference is that when it raises, the previous file is still intact.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

#: Windows refuses to replace a file another process holds open without
#: FILE_SHARE_DELETE — OneDrive and antivirus scanners do exactly that, briefly.
#: A short bounded retry rides out the common case; a persistent lock still
#: raises, leaving the old file untouched.
REPLACE_ATTEMPTS = 5
REPLACE_BACKOFF_S = 0.05


def _replace(source: str, target: Path) -> None:
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_BACKOFF_S * (attempt + 1))


def _atomic_write(path: str | os.PathLike[str], data: str | bytes,
                  encoding: str | None) -> None:
    target = Path(path)
    # Same directory as the target, so os.replace is a rename, never a copy
    # across volumes. A missing directory raises here exactly as write_text.
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        mode = "wb" if isinstance(data, bytes) else "w"
        with os.fdopen(handle, mode, encoding=encoding) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str, *,
                      encoding: str = "utf-8") -> None:
    """Replace *path* with *text* atomically. Raises on failure, old file intact."""
    _atomic_write(path, text, encoding)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Replace *path* with *data* atomically. Raises on failure, old file intact."""
    _atomic_write(path, bytes(data), None)


__all__ = ["atomic_write_text", "atomic_write_bytes", "REPLACE_ATTEMPTS"]
