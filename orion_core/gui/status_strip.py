"""
Operational status strip (Mark XXVI, §14) — the single, unobtrusive line that
tells the user what ORION is doing right now.

Renders an OpStatus: a health dot, the mode (cloud/offline), the voice state, the
current activity, and any focus block / mission — one restrained row, not a HUD.
QPainter (so it is render-verifiable head-less), with the degenerate-size guard
from the Mark XXIII black-flicker lesson.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..constants import C
from ..operational_status import OpStatus

# This strip is custom-painted, so no stylesheet ever reached it and it kept
# a palette of its own: a blue-grey ground, cyan, green and amber. Four hues
# on the one line that sits above every page, which is most of what made the
# interface read as "a colour scheme" rather than as one colour on neutral.
# Taken from C now, so it moves when the palette moves.
_BG = QColor(C.INK)         # a recessed rail under the header
_INK = QColor(C.WHITE)      # the voice state: the line's headline
_DIM = QColor(C.FAINT)      # running context, deliberately quiet
_ACCENT = QColor(C.MUTED)   # nominal connectivity — see below
_OK = QColor(C.GOOD)
_WARN = QColor(C.WARN)
_BAD = QColor(C.BAD)
_OFF = QColor(C.FAINT)


def _health_colour(status: OpStatus) -> QColor:
    if not status.online:
        return _OFF
    h = (status.health or "").upper()
    if h.startswith("ONLINE") or h == "OK":
        return _OK
    # A failure used to return the same grey as being offline, so "ORION is
    # broken" and "ORION is not connected" looked identical on the one line
    # meant to tell them apart.
    if "FAIL" in h:
        return _BAD
    if "OFFLINE" in h:
        return _OFF
    return _WARN


class StatusStrip(QWidget):
    """A one-line operational-status surface. Feed it OpStatus via set_status()."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = OpStatus()
        self.setMinimumHeight(28)
        self.setMaximumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_status(self, status: OpStatus) -> None:
        self._status = status or OpStatus()
        self.update()

    def _font(self, size: int, bold: bool = False) -> QFont:
        f = QFont("Segoe UI", size, QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        f.setFamilies(["Segoe UI", "Consolas", "sans-serif"])
        return f

    def paintEvent(self, _event: Any) -> None:
        w, h = self.width(), self.height()
        if w < 8 or h < 8:                 # degenerate size — skip (flicker guard)
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _BG)
        st = self._status
        cy = h / 2

        # health dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_health_colour(st))
        r = 4.0
        p.drawEllipse(QRectF(12 - r, cy - r, r * 2, r * 2))

        # mode chip
        x = 26.0
        p.setFont(self._font(8, bold=True))
        mode = "CLOUD" if (st.online and st.mode != "offline") else "OFFLINE"
        # Connected is the normal case and recedes; not connected steps
        # forward. It was the other way round — cyan for CLOUD, grey for
        # OFFLINE — which spent the loudest colour on the state that needs no
        # attention and hid the one that does.
        p.setPen(_ACCENT if mode == "CLOUD" else _WARN)
        p.drawText(QRectF(x, 0, 60, h), Qt.AlignmentFlag.AlignVCenter, mode)
        x += 58

        # voice state
        p.setFont(self._font(9, bold=True))
        p.setPen(_INK)
        voice = st.voice.upper()
        p.drawText(QRectF(x, 0, 110, h), Qt.AlignmentFlag.AlignVCenter, voice)
        x += 108

        # activity / focus — what he is doing RIGHT NOW, dimmer.
        #
        # The current mission used to be appended here too, which meant the
        # same unchanging words ("Develop ORION") sat on every page of the deck
        # all day. A strip this size is for what changes; a mission that lasts
        # for weeks is not news, and the Mission deck shows it properly.
        tail = []
        if st.activity:
            tail.append(st.activity)
        if st.focus:
            tail.append(f"focus: {st.focus}")
        if tail:
            p.setFont(self._font(9))
            p.setPen(_DIM)
            p.drawText(QRectF(x, 0, max(10.0, w - x - 12), h),
                       Qt.AlignmentFlag.AlignVCenter,
                       "·  " + "   ·   ".join(tail))

    def __probe_state__(self) -> dict[str, Any]:
        """Headless inspection hook for the render test."""
        return self._status.as_dict()


__all__ = ["StatusStrip"]
