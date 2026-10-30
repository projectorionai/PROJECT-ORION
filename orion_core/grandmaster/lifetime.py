"""Cancellation-safe draining for ownership boundaries (processes and sockets)."""
from __future__ import annotations

import asyncio


async def finish(future):
    """Drain a started operation even if its owner is cancelled repeatedly.

    Return (result, cancellation_received). The owner re-raises cancellation
    after releasing its actual resources. A cancelled child still propagates.
    """
    interrupted = False
    while True:
        try:
            return await asyncio.shield(future), interrupted
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            interrupted = True
