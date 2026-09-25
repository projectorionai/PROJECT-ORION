"""
Test for UnifiedDashboard.page_names() (Mark XX design-spec §8) — the
Command Palette's source of truth for which deck pages to index, instead of
a hardcoded copy of the page list that could drift from the real one.

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from orion_core.bus import OrionBus
from orion_core.gui.unified_dashboard import UnifiedDashboard


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_page_names_returns_every_label_in_tab_order(_app):
    pages = [("MISSION", QWidget()), ("WIDGETS", QWidget()), ("CHESS", QWidget())]
    deck = UnifiedDashboard(OrionBus(), pages)
    assert deck.page_names() == ["MISSION", "WIDGETS", "CHESS"]


def test_page_names_reflects_the_actual_construction_list_not_a_copy(_app):
    pages = [("ONLY", QWidget())]
    deck = UnifiedDashboard(OrionBus(), pages)
    assert deck.page_names() == ["ONLY"]
