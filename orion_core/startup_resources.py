"""Owned, off-loop construction of thread-safe startup storage resources.

Only plain storage/data factories belong here, never Qt objects. Cancelling
their await cannot stop a filesystem call: its worker retains late cleanup.
"""
from __future__ import annotations

import asyncio
import contextvars
from functools import partial, wraps
import logging
import threading
from typing import Any, Callable

_MISSING = object()
_PENDING: set[Any] = set()
_PENDING_LOCK = threading.Lock()
_SCOPE: contextvars.ContextVar[Any] = contextvars.ContextVar("orion_startup_resources", default=None)
_LOG = logging.getLogger(__name__)


def _forget(item: Any) -> None:
    with _PENDING_LOCK:
        _PENDING.discard(item)


class _Construction:
    def __init__(self, factory: Callable[[], Any], close: Callable[[Any], None] | None) -> None:
        self.factory, self.close = factory, close
        self.lock = threading.Lock()
        self.value = _MISSING
        self.started = self.abandoned = self.closing = False
        with _PENDING_LOCK:
            _PENDING.add(self)

    def _dispose(self, value: Any) -> None:
        try:
            close = self.close or getattr(value, "close", None)
            if close is not None:
                close(value) if self.close is not None else close()
        except Exception as exc:
            _LOG.warning("Startup resource cleanup failed (%s)", type(exc).__name__)
        finally:
            _forget(self)

    def run(self) -> Any:
        with self.lock:
            self.started = True
            abandoned = self.abandoned
        if abandoned:
            _forget(self)
            return None  # cancelled while queued: no storage operation starts
        try:
            value = self.factory()
        except BaseException as exc:
            _forget(self)
            if self.abandoned:
                _LOG.warning("Cancelled startup construction failed late (%s)", type(exc).__name__)
            raise
        with self.lock:
            self.value = value
            dispose = self.abandoned and not self.closing
            if dispose:
                self.closing = True
        if dispose:
            self._dispose(value)  # finish on the constructing thread
        return value

    def abandon(self) -> None:
        with self.lock:
            self.abandoned = True
            value = self.value
            dispose = value is not _MISSING and not self.closing
            if dispose:
                self.closing = True
        if dispose:
            with _PENDING_LOCK:
                _PENDING.add(self)
            # A completed result may exist before its cancelled await resumes.
            # Cleanup must neither block Qt nor depend on a closing loop/pool.
            threading.Thread(target=self._dispose, args=(value,),
                             name="orion-startup-close", daemon=True).start()

    def observed(self, future: asyncio.Future) -> None:
        if future.cancelled():
            self.abandon()
            with self.lock:
                if not self.started:
                    _forget(self)
        else:
            future.exception()  # retrieve failures arriving after caller cancellation


class _StartupScope:
    def __init__(self) -> None:
        self.resources: list[_Construction] = []

    def abort(self) -> None:
        for resource in reversed(self.resources):
            resource.abandon()
        self.resources.clear()


def startup_lifetime(function: Callable) -> Callable:
    """Keep constructed stores owned until normal shutdown takes responsibility."""
    @wraps(function)
    async def owned(*args: Any, **kwargs: Any) -> Any:
        scope = _StartupScope()
        token = _SCOPE.set(scope)
        try:
            return await function(*args, **kwargs)
        finally:
            scope.abort()
            _SCOPE.reset(token)
    return owned


def handoff_startup_resources() -> None:
    """The existing normal shutdown now owns successfully constructed stores."""
    scope = _SCOPE.get()
    if scope is not None:
        scope.resources.clear()


async def construct_startup_resource(factory: Callable[..., Any], *args: Any,
                                     close: Callable[[Any], None] | None = None,
                                     **kwargs: Any) -> Any:
    item = _Construction(partial(factory, *args, **kwargs), close)
    scope = _SCOPE.get()
    if scope is not None:
        scope.resources.append(item)
    context = contextvars.copy_context()
    try:
        future = asyncio.get_running_loop().run_in_executor(None, context.run, item.run)
    except BaseException:
        _forget(item)
        raise
    future.add_done_callback(item.observed)
    try:
        value = await asyncio.shield(future)
    except asyncio.CancelledError:
        item.abandon()
        raise
    _forget(item)
    return value
