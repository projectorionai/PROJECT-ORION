"""A nonblocking OS lease for one research-ledger writer; never unlink locks."""

from __future__ import annotations

import errno
import os
from pathlib import Path


class ResearchLease:
    def __init__(self, handle):
        self.handle = handle

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def acquire_lease(path: Path) -> ResearchLease | None:
    """None means another writer owns it; all other I/O errors stay explicit.

    The lock file persists. The kernel releases the held descriptor's lock on
    close or process death, so deleting a pathname cannot split ownership.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.with_name(path.name + ".lock").open("a+b")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return ResearchLease(handle)
    except OSError as exc:
        handle.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return None
        raise
