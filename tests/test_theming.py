"""
Recolouring the whole shell, not just one button.

``APP_STYLESHEET`` is an f-string built once at import, so ORION's crimson is
baked into a few thousand characters of Qt stylesheet before anything is on
screen. That is the right shape for a fixed palette and the wrong shape for a
colour the user can change — which is why "live theming" was missing even
though every widget already takes its colours from one place.

The alternative to this was making the palette mutable, which would mean
auditing every module that captured a colour into a local and would still
repaint nothing. Deriving a new stylesheet by substitution is one pass, no
global state, and Qt repaints the tree itself.

What the tests hold onto
------------------------
  * a hue is a FAMILY. ORION's crimson is four related values, and swapping
    only the primary leaves every hover state crimson — which reads as a bug
    rather than a theme.
  * surfaces and the status triad are never touched. A background is what makes
    text readable, and a red "bad" badge has to mean bad in every theme.
  * a colour that vanishes into the void black is declined rather than
    silently applied.

Offline: pure strings, no Qt widgets.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.constants import C
from orion_core.gui.style import APP_STYLESHEET
from orion_core.gui.theming import (
    FAMILY,
    derive_family,
    is_readable,
    themed_stylesheet,
)

BLUE = "#2f7fff"


def _method_body(source: str, name: str) -> str:
    """Everything from ``def name`` to the next method at the same indent.

    A fixed character window was the obvious way to write this and broke the
    first time the method grew by twenty lines — the test failed on a slice
    boundary rather than on anything about theming.
    """
    start = source.index(f"def {name}")
    rest = source[start:]
    end = re.search(r"\n    (?:def |@)", rest)
    return rest[:end.start()] if end else rest


# ── the family ───────────────────────────────────────────────────────────────

def test_the_crimson_family_is_four_related_values():
    assert set(FAMILY) == {C.PRI, C.PRI_DIM, C.PRI_HI, C.CORE}


def test_a_chosen_colour_becomes_a_whole_family():
    """Swapping only the primary leaves hover states crimson, which looks like
    a bug rather than a theme."""
    family = derive_family(BLUE)
    assert len(set(family.values())) == 4, "the family collapsed to fewer hues"
    for original, replacement in family.items():
        assert replacement != original


def test_the_family_keeps_its_ordering():
    """dim darker than primary, hi and core progressively lighter — the
    relationships are measured from the originals, not invented."""
    def brightness(value: str) -> int:
        raw = value.lstrip("#")
        return sum(int(raw[i:i + 2], 16) for i in (0, 2, 4))

    family = derive_family(BLUE)
    assert brightness(family[C.PRI_DIM]) < brightness(family[C.PRI])
    assert brightness(family[C.PRI_HI]) > brightness(family[C.PRI])
    assert brightness(family[C.CORE]) > brightness(family[C.PRI_HI])


def test_brightening_keeps_the_hue_rather_than_washing_out():
    """Scaling channels directly desaturates as it brightens — a light red
    drifts toward white instead of staying red."""
    family = derive_family(BLUE)
    raw = family[C.CORE].lstrip("#")
    red, green, blue = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
    assert blue > red, "the lightened blue is no longer blue"


@pytest.mark.parametrize("bad", ["", "not-a-colour", "#12", None, 42])
def test_an_unusable_colour_changes_nothing(bad):
    """A bad hex must not be able to produce an unreadable interface."""
    family = derive_family(bad)
    assert family == {original: original for original in FAMILY}
    assert themed_stylesheet(bad) == APP_STYLESHEET


# ── the stylesheet ───────────────────────────────────────────────────────────

def test_every_crimson_reference_is_replaced():
    sheet = themed_stylesheet(BLUE)
    assert sheet != APP_STYLESHEET
    assert APP_STYLESHEET.lower().count(C.PRI.lower()) > 0, "fixture is wrong"
    assert sheet.lower().count(C.PRI.lower()) == 0, "crimson survived somewhere"
    assert BLUE in sheet


def test_surfaces_are_left_alone():
    """Backgrounds are what make text readable; theming them is how a theme
    becomes unusable."""
    sheet = themed_stylesheet(BLUE)
    for surface in (C.BG, C.INK, C.PANEL, C.PANEL_HI, C.BORDER):
        assert sheet.count(surface) == APP_STYLESHEET.count(surface), surface


def test_the_status_triad_is_left_alone():
    """A red 'bad' badge has to mean bad whatever the accent is."""
    sheet = themed_stylesheet(BLUE)
    for status in (C.GOOD, C.WARN, C.BAD):
        assert sheet.count(status) == APP_STYLESHEET.count(status), status


def test_the_silver_accent_survives():
    sheet = themed_stylesheet(BLUE)
    assert sheet.count(C.ACCENT) == APP_STYLESHEET.count(C.ACCENT)


def test_choosing_orions_own_crimson_is_a_no_op():
    assert themed_stylesheet(C.PRI) == APP_STYLESHEET


# ── readability ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("colour", ["#2f7fff", "#ff1a3c", "#20c997", "#ffffff"])
def test_usable_colours_are_accepted(colour):
    assert is_readable(colour) is True


@pytest.mark.parametrize("colour", ["#0a0a0c", "#111214", "#060709"])
def test_a_colour_that_vanishes_into_the_background_is_declined(colour):
    """Not a hard gate — it is the user's interface — but applying it silently
    would leave them with a shell they cannot read and no obvious way back."""
    assert is_readable(colour) is False


def test_readability_survives_rubbish():
    for bad in ("", "nope", None, 7):
        assert is_readable(bad) is False


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_window_applies_it_to_the_whole_shell():
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert "def apply_accent" in source
    body = _method_body(source, "apply_accent")
    assert "themed_stylesheet" in body, "only the face would be recoloured"
    assert "setStyleSheet" in body
    assert "set_palette_colours" in body, "the face would keep the old colour"


def test_the_hud_is_recoloured_too():
    """The HUD calls setStyleSheet on ITSELF, and a child's own sheet beats an
    inherited one for its own rules. Restyling only the window therefore left
    a crimson HUD inside a themed shell — a half-applied theme, which is the
    thing that reads as a bug."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    body = _method_body(source, "apply_accent")
    assert "hud_stylesheet(accent)" in body, (
        "the HUD would keep the old accent while the shell changed")


def test_the_hud_accent_carries_its_whole_family():
    """The HUD's hover states were hard-coded to C.PRI_HI, so a themed HUD
    still flashed crimson under the pointer."""
    from orion_core.gui.hud_shell import hud_stylesheet

    themed = hud_stylesheet(BLUE)
    for member in FAMILY:
        assert member.lower() not in themed.lower(), (
            f"{member} survived the HUD recolour")


def test_an_unreadable_choice_does_not_half_apply():
    """Refusing after restyling the shell but before the face would leave the
    two disagreeing, which is worse than either outcome."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    body = _method_body(source, "apply_accent")
    readable_at = body.index("is_readable(accent)")
    restyle_at = body.index("setStyleSheet")
    assert readable_at < restyle_at, (
        "the shell is restyled before the colour is checked"
    )
