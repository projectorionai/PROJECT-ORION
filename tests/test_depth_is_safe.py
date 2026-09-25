"""A shadow must never be able to kill the application.

Measured on the real frozen build, not reasoned about: with drop shadows on,
``dist/ORION/ORION.exe`` died in under forty seconds with
STATUS_HEAP_CORRUPTION (0xC0000374). The identical build with ``ORION_FLAT=1``
ran indefinitely, and the previous build — which had no depth module at all —
was fine. Three runs, one variable.

The cause is a documented Qt constraint rather than a bug here:
``QGraphicsEffect`` renders its widget into an offscreen pixmap, which is
undefined behaviour for a widget that owns or contains a NATIVE window. The
Core Window holds the QWebEngineView face and OpenGL views, so walking the tree
and lifting every panel eventually put an effect over one of them.

Two rules follow, and both are here because the failure mode is a dead process
rather than a wrong colour.

Offline: no QApplication — building one inline hangs this suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.gui import depth  # noqa: E402


def test_shadows_are_off_unless_asked_for(monkeypatch):
    """The default has to be the setting that cannot crash ORION."""
    monkeypatch.delenv("ORION_FLAT", raising=False)
    monkeypatch.delenv("ORION_SHADOWS", raising=False)
    assert depth.shadows_enabled() is False


def test_the_opt_in_works(monkeypatch):
    monkeypatch.delenv("ORION_FLAT", raising=False)
    monkeypatch.setenv("ORION_SHADOWS", "1")
    assert depth.shadows_enabled() is True


def test_flat_still_wins(monkeypatch):
    """Someone who has already been told 'set ORION_FLAT=1' must not have that
    silently overridden by the newer switch."""
    monkeypatch.setenv("ORION_SHADOWS", "1")
    monkeypatch.setenv("ORION_FLAT", "1")
    assert depth.shadows_enabled() is False


def test_lift_refuses_anything_holding_a_native_window(monkeypatch):
    """Even with shadows explicitly on. This is the rule that stops the crash;
    the default merely stops it being reached."""
    monkeypatch.delenv("ORION_FLAT", raising=False)
    monkeypatch.setenv("ORION_SHADOWS", "1")

    class _Native:
        """Stands in for the WebEngine face: reports a native window."""

        def testAttribute(self, _attr):     # noqa: N802  (Qt naming)
            return True

        def windowHandle(self):             # noqa: N802  (Qt naming)
            return object()

        def findChildren(self, _type):      # noqa: N802  (Qt naming)
            return []

    assert depth.lift(_Native()) is False


def test_an_unanswerable_widget_is_treated_as_native():
    """If the check itself raises, the answer is 'do not risk it'. A missing
    shadow is invisible; the alternative took the process down."""

    class _Hostile:
        def testAttribute(self, _attr):     # noqa: N802
            raise RuntimeError("no")

    assert depth._contains_native_window(_Hostile()) is True


def test_the_reason_is_written_down_where_someone_would_turn_it_back_on():
    """The next person to find a switch called ORION_SHADOWS will want to know
    why it is off, and the answer is not guessable from the code."""
    source = (ROOT / "orion_core" / "gui" / "depth.py").read_text(encoding="utf-8")
    assert "HEAP_CORRUPTION" in source
    assert "native" in source.lower()
