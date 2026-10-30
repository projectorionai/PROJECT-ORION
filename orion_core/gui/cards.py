"""The plain card the Calendar, Focus and Fitness pages are built from, and
the one way they run work off the GUI thread."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from PyQt6 import sip
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from ..constants import SPACE


def card(label: str = "", *, name: str = "card", parent: QWidget | None = None
         ) -> tuple[QFrame, QVBoxLayout]:
    """A card and its layout, with one small label on top when given."""
    frame = QFrame(parent)
    frame.setObjectName(name)
    box = QVBoxLayout(frame)
    box.setContentsMargins(SPACE.LG, SPACE.MD + SPACE.XS, SPACE.LG, SPACE.MD + SPACE.XS)
    box.setSpacing(SPACE.SM)
    if label:
        caption = QLabel(label)
        caption.setObjectName("cardLabel")
        box.addWidget(caption)
    return frame, box


def label(text: str = "", name: str = "", *, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def clear_layout(layout: Any) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def off_thread(owner: QWidget, work: Callable[[], Any], *,
               done: Callable[[Any], None] | None = None,
               failed: Callable[[BaseException], None] | None = None,
               name: str = "orion-deck-page") -> Any:
    """Run *work* on a worker thread; *done* gets its result back on the GUI
    thread (the qasync loop is the Qt thread). Without a running loop - a
    test, a page built on its own - it runs inline."""
    async def wrapped() -> Any:
        return await asyncio.to_thread(work)
    return run_async(owner, wrapped, done=done, failed=failed, name=name)


def run_async(owner: QWidget, factory: Callable[[], Any], *,
              done: Callable[[Any], None] | None = None,
              failed: Callable[[BaseException], None] | None = None,
              name: str = "orion-deck-page") -> Any:
    """Await *factory()* on ORION's loop and hand the result to *done*."""
    def finish(result: Any = None, error: BaseException | None = None) -> None:
        if sip.isdeleted(owner):
            return
        if error is not None:
            if failed is not None:
                failed(error)
        elif done is not None:
            done(result)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            result = asyncio.run(factory())
        except Exception as exc:
            finish(error=exc)
        else:
            finish(result)
        return None

    async def go() -> None:
        try:
            result = await factory()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            finish(error=exc)
        else:
            finish(result)

    from .. import background
    return background.spawn(go(), name=name)


__all__ = ["card", "clear_layout", "label", "off_thread", "run_async"]
