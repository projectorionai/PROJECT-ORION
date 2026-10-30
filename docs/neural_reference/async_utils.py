"""Cancellation barriers for work that cannot safely be abandoned halfway."""

from __future__ import annotations

import asyncio
from functools import wraps


async def complete_before_cancel(awaitable):
    """Finish the owned operation, then propagate even repeated cancellation.

    This is deliberately not a timeout. Use only for bounded operations or
    supervised workers, so cancellation cannot release a still-owned resource.
    """
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


def atomic_async(method):
    @wraps(method)
    async def wrapped(*args, **kwargs):
        return await complete_before_cancel(method(*args, **kwargs))

    return wrapped
