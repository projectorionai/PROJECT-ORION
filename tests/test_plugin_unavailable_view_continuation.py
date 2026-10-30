"""The panel must distinguish held preferences from user-disabled plugins."""
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QPushButton, QLabel
from PyQt6.QtGui import QTextDocument

from orion_core.bus import OrionBus
from orion_core.gui.plugin_deck import PluginDeckView, _PluginRow


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("records", [[], [
    {"name": "demo", "status": "disabled", "enabled": False, "tier": "confirm"}]],
    ids=["empty_registry", "installed_plugin"])
def test_unavailable_preferences_are_explained_and_toggles_disabled(qapp, records):
    snapshot = {"total": len(records), "enabled": 0, "healthy": 0,
                "plugins": records, "state_error": "synthetic unavailable preferences"}
    view = PluginDeckView(OrionBus(), SimpleNamespace(snapshot=lambda: snapshot))
    try:
        text = " ".join(label.text() for label in view.findChildren(QLabel)).lower()
        assert "unavailable" in text and "held" in text
        assert "no plugins yet" not in text
        for button in view.findChildren(QPushButton):
            if "New Plugin" in button.text() or button.text() in ("Enable", "Disable", "Held", "Manifests"):
                assert not button.isEnabled()
        snapshot["state_error"] = ""
        view._refresh()
        assert all(button.isEnabled() for button in view.findChildren(QPushButton)
                   if "New Plugin" in button.text() or button.text() == "Manifests")
    finally:
        view._timer.stop()
        view.deleteLater()


def test_plugin_metadata_is_literal_text_in_the_rich_row(qapp):
    row = _PluginRow({"name": "a<b>", "description": "Compare <input> & output",
                      "status": "ready", "tier": "confirm", "enabled": True,
                      "last_error": "missing <resource>"}, lambda *_: None)
    try:
        text = " ".join(label.text() for label in row.findChildren(QLabel))
        document = QTextDocument()
        document.setHtml(text)
        visible = document.toPlainText()
        assert "a<b>" in visible
        assert "Compare <input> & output" in visible
        assert "missing <resource>" in visible
    finally:
        row.deleteLater()
