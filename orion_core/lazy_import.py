"""
Deferred module imports — startup latency, not laziness for its own sake.

ORION's first paint is gated on ``app.py`` importing ``live_worker``, which
imported ``google.genai`` at module scope. That single import measured
**1.28 s** on this machine (it pulls pydantic, httpx, websockets and the whole
protobuf/type surface) — and every millisecond of it was spent before the
window existed, with the user staring at nothing.

Nothing in ``live_worker`` touches ``genai`` or ``types`` at import time: they
are used inside methods, and the module has ``from __future__ import
annotations`` so type hints never evaluate. So the cost can move off the
critical path to the moment the Live session actually connects — which happens
in a background task, after the GUI is up.

A module-level ``__getattr__`` (PEP 562) does NOT work for this: it only fires
for attribute access on the module from OUTSIDE. A bare ``genai`` inside a
function in the same module resolves through ``globals()`` and would raise
NameError. Hence a proxy object bound to the global name instead.

The proxy is deliberately thin. It resolves the real module on first attribute
access, caches it, and then behaves like it. If the import fails it raises the
original ImportError at the point of use — the same place a missing dependency
would have surfaced anyway, but now without a blank window first.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


class LazyModule:
    """A stand-in for a module, imported on first attribute access.

    >>> np = LazyModule("numpy")        # costs nothing
    >>> np.array([1, 2])                # imports numpy, then delegates
    """

    __slots__ = ("_lazy_name", "_lazy_attr", "_lazy_module")

    def __init__(self, name: str, attr: str | None = None) -> None:
        # object.__setattr__ so we do not recurse through our own __setattr__
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_attr", attr)
        object.__setattr__(self, "_lazy_module", None)

    def _resolve(self) -> Any:
        module = object.__getattribute__(self, "_lazy_module")
        if module is None:
            name = object.__getattribute__(self, "_lazy_name")
            attr = object.__getattribute__(self, "_lazy_attr")
            module = importlib.import_module(name)
            if attr is not None:
                module = getattr(module, attr)
            object.__setattr__(self, "_lazy_module", module)
        return module

    @property
    def loaded(self) -> bool:
        """True once the real module has actually been imported."""
        return object.__getattribute__(self, "_lazy_module") is not None

    def __getattr__(self, item: str) -> Any:
        # Only called for names not found normally, so __slots__/loaded are safe.
        return getattr(self._resolve(), item)

    def __setattr__(self, item: str, value: Any) -> None:
        setattr(self._resolve(), item, value)

    def __dir__(self):
        return dir(self._resolve())

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_lazy_name")
        attr = object.__getattribute__(self, "_lazy_attr")
        full = f"{name}.{attr}" if attr else name
        state = "loaded" if self.loaded else "deferred"
        return f"<LazyModule {full} ({state})>"


class LazyCallable:
    """A stand-in for one callable ATTRIBUTE of a module, resolved on first call.

    ``from aiohttp import ClientSession`` binds a name, not a module, so
    :class:`LazyModule` cannot replace it without rewriting every call site.
    This proxy can: it is constructed and called exactly like the real class.

    Only safe for names used purely as callables. It is deliberately NOT usable
    as an ``except`` target, an ``isinstance`` argument or a base class — those
    need a real class object at the point of use, and Python would fail loudly
    rather than silently misbehave. Every current aiohttp use was audited
    against those three patterns before this was applied.
    """

    __slots__ = ("_lazy_module", "_lazy_attr", "_lazy_target")

    def __init__(self, module: str, attr: str) -> None:
        object.__setattr__(self, "_lazy_module", module)
        object.__setattr__(self, "_lazy_attr", attr)
        object.__setattr__(self, "_lazy_target", None)

    def _resolve(self) -> Any:
        target = object.__getattribute__(self, "_lazy_target")
        if target is None:
            module = importlib.import_module(
                object.__getattribute__(self, "_lazy_module"))
            target = getattr(module, object.__getattribute__(self, "_lazy_attr"))
            object.__setattr__(self, "_lazy_target", target)
        return target

    @property
    def loaded(self) -> bool:
        return object.__getattribute__(self, "_lazy_target") is not None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        module = object.__getattribute__(self, "_lazy_module")
        attr = object.__getattribute__(self, "_lazy_attr")
        state = "loaded" if self.loaded else "deferred"
        return f"<LazyCallable {module}.{attr} ({state})>"


def lazy_attr(module: str, attr: str) -> LazyCallable:
    """Shorthand: ``ClientSession = lazy_attr("aiohttp", "ClientSession")``."""
    return LazyCallable(module, attr)


# ── warming ───────────────────────────────────────────────────────────────────
#
# Deferring an import does not delete its cost, it MOVES it — and moving a
# 1.28 s import from startup onto the first Live connect would be a straight
# downgrade, because connect() runs on the qasync loop: the user would get a
# fast window followed by a freeze mid-sentence. So the deferred modules are
# imported on a daemon thread the moment the window is on screen, where nothing
# is waiting on them. By the time anything asks, sys.modules already has them
# and the proxy resolves instantly.
#
# A background thread rather than asyncio.to_thread: this must not depend on,
# or contend with, the event loop it exists to protect.

#: Everything deferred off the startup path. Kept here (not at the call site)
#: so the warm list cannot drift away from what was actually deferred —
#: test_startup_performance asserts the two agree.
STARTUP_DEFERRED: tuple[str, ...] = (
    "google.genai",        # ~1280 ms — live_worker
    "aiohttp",             # ~178 ms  — providers, utils and 9 others
    "sounddevice",         # ~162 ms  — audio
)

#: ORION's OWN heavy modules, deferred out of ``app.py``'s module scope.
#:
#: Third-party packages were the first 78% of the import cost; once they were
#: gone what remained on the pre-window path was ORION's own tree. None of
#: these is needed to construct or paint the window — they are all built later
#: in ``run_application`` — so each one was pure dead time before first paint.
#:
#: The same warming rule applies, and matters more here: these are needed a
#: few hundred milliseconds later during the boot sequence itself, on the
#: qasync loop. Warmed from the moment the window is visible, the import at
#: the use site is a ``sys.modules`` lookup.
STARTUP_DEFERRED_LOCAL: tuple[str, ...] = (
    "orion_core.live_worker",   # ~103 ms — pulls audio, which pulls numpy
    "orion_core.agents",        # ~61 ms  — the specialist workforce
    "orion_core.gui.globe",     # ~34 ms  — pulls QtWebEngineCore
)

_warm_thread: Any = None


def warm(*modules: str, log: Callable[[str], None] | None = None) -> Any:
    """Import ``modules`` on a daemon thread. Returns the thread (or None).

    Never raises: a module that will not import is exactly the case the caller
    cannot do anything about at this point, and it will surface properly at the
    point of use. Idempotent — a second call while the first is still running
    is a no-op.
    """
    global _warm_thread
    import threading

    if _warm_thread is not None and _warm_thread.is_alive():
        return _warm_thread
    # ORION's own modules first, and the order is not cosmetic. One thread
    # warms these in sequence, and ``google.genai`` alone is ~1.28 s — long
    # enough that anything queued behind it is still cold when the boot
    # sequence asks for it, which puts the import back on the qasync loop as a
    # stall. The local modules are needed DURING boot (live_worker within a
    # second of first paint); genai is not needed until the Live session
    # connects, well after. So the slowest, latest-needed import goes last.
    targets = modules or (STARTUP_DEFERRED_LOCAL + STARTUP_DEFERRED)

    def _run() -> None:
        import time as _time
        for name in targets:
            start = _time.perf_counter()
            try:
                importlib.import_module(name)
            except Exception:
                continue          # surfaces at the point of use, not here
            if log is not None:
                try:
                    log(f"warmed {name} in {(_time.perf_counter()-start)*1000:.0f} ms")
                except Exception:
                    pass

    _warm_thread = threading.Thread(target=_run, name="orion-import-warm", daemon=True)
    _warm_thread.start()
    return _warm_thread


def warmed(module: str) -> bool:
    """True if ``module`` is already imported (so a proxy will resolve free)."""
    import sys
    return module in sys.modules


__all__ = ["LazyModule", "LazyCallable", "lazy_attr", "warm", "warmed",
           "STARTUP_DEFERRED", "STARTUP_DEFERRED_LOCAL"]
