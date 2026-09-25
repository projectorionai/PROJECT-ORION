"""
PLUGINS deck page — capability disclosure on screen (Mark XXVI improvement #11).

The registry knows a plugin's version and which capabilities it uses without
declaring them; that fact is only useful if the user SEES it where they decide
whether to keep a plugin enabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _row(record):
    from orion_core.gui.plugin_deck import _PluginRow
    return _PluginRow(record, on_toggle=lambda *a: None)


def _label_text(widget) -> str:
    from PyQt6.QtWidgets import QLabel
    return " ".join(child.text() for child in widget.findChildren(QLabel))


def test_a_row_shows_version_and_declared_capabilities(qapp):
    text = _label_text(_row({
        "name": "lamp", "description": "Flashes a lamp.", "tier": "allow",
        "status": "ready", "version": "2.1.0", "permissions": ["network"],
        "undeclared": [],
    }))
    assert "v2.1.0" in text
    assert "network" in text


def test_a_row_warns_about_undeclared_capabilities(qapp):
    text = _label_text(_row({
        "name": "leaky", "description": "Leaks.", "tier": "confirm",
        "status": "ready", "version": "1.0.0", "permissions": [],
        "undeclared": ["subprocess", "filesystem"],
    }))
    assert "undeclared" in text.lower()
    assert "subprocess" in text and "filesystem" in text


def test_a_clean_plugin_shows_no_warning(qapp):
    text = _label_text(_row({
        "name": "clean", "description": "Clean.", "tier": "allow",
        "status": "ready", "version": "1.0.0", "permissions": [], "undeclared": [],
    }))
    assert "undeclared" not in text.lower()


def test_a_legacy_record_without_the_new_fields_still_renders(qapp):
    # Records predating Mark XXVI have no version/permissions/undeclared keys.
    text = _label_text(_row({"name": "old", "description": "Old.",
                             "tier": "confirm", "status": "ready"}))
    assert "old" in text.lower()


def test_the_page_renders_a_real_frame(qapp):
    from PyQt6.QtGui import QColor, QImage
    from orion_core.bus import OrionBus
    from orion_core.gui.plugin_deck import PluginDeckView
    from orion_core.plugin_registry import PluginRegistry

    view = PluginDeckView(OrionBus(), PluginRegistry(bus=None))
    view.resize(700, 420)
    if hasattr(view, "refresh"):
        view.refresh()
    img = QImage(700, 420, QImage.Format.Format_RGB32)
    img.fill(QColor("#000000"))
    view.render(img)
    colours = {img.pixel(x, y) for x in range(0, 700, 25) for y in range(0, 420, 25)}
    assert len(colours) > 1, "the plugin page rendered flat"
