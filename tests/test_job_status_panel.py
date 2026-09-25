"""
Tests for JobStatusPanel (orion_core/gui/widgets.py) — the generic
background-job status component (Mark XX design-spec §3/§9, Medium item
11). Replaces the Studio deck's former bespoke AudioCanvas, which drew two
hand-painted bars plus a decorative sine-wave scan line for what was really
just two integers — the design spec's own example of visual novelty over
information density. This is reusable by any future background job
(ingestion, audio processing, a Forge session), not a canvas per job type.

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
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui.widgets import JobStatusPanel


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def test_set_counts_computes_a_completion_percentage(_app):
    panel = JobStatusPanel("Test Job")
    panel.set_counts(3, 10)
    assert panel.bar.value == 30.0


def test_set_counts_with_zero_total_shows_zero_percent(_app):
    panel = JobStatusPanel("Test Job")
    panel.set_counts(0, 0)
    assert panel.bar.value == 0.0


def test_set_counts_clamps_done_to_total(_app):
    panel = JobStatusPanel("Test Job")
    panel.set_counts(15, 10)   # done > total must never exceed 100%
    assert panel.bar.value == 100.0


def test_set_counts_updates_the_status_line_when_given(_app):
    panel = JobStatusPanel("Test Job")
    panel.set_counts(4, 8, "4 raw · 4 processed")
    assert panel.status_label.text() == "4 raw · 4 processed"


def test_set_counts_leaves_the_status_line_alone_when_omitted(_app):
    panel = JobStatusPanel("Test Job")
    panel.status_label.setText("earlier status")
    panel.set_counts(4, 8)
    assert panel.status_label.text() == "earlier status"


def test_title_reaches_the_underlying_bar(_app):
    panel = JobStatusPanel("Ingestion Queue")
    assert panel.bar.label == "Ingestion Queue"


def test_panel_paints_without_crashing(_app):
    panel = JobStatusPanel("Test Job")
    panel.resize(240, 60)
    panel.set_counts(6, 9, "in progress")
    panel.grab()   # must paint (bar + status label) without raising
