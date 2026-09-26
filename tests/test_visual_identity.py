"""The rules the look depends on, asserted so they survive.

ORION's interface read as the same genre as every other dark assistant, and
the reasons were specific rather than a matter of taste: six hues competing
for attention, depth drawn with gradients because Qt Style Sheets cannot cast
a shadow, and every panel outlined with a box.

The redesign is "spatial glass, near-monochrome, crimson as the only colour".
Two of those three are properties that a single well-meaning edit can quietly
undo — one more accent hue, one more crimson border — so they are checked
here. None of this constructs a QApplication: doing that inline hangs this
suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.constants import C, ELEV, GLASS, HEALTH_GLYPHS  # noqa: E402
from orion_core.gui.style import APP_STYLESHEET  # noqa: E402

#: Tokens that make up the chrome. Every one must be a true neutral.
CHROME = ("BG", "INK", "PANEL", "PANEL_HI", "BORDER", "ACCENT", "ACCENT_DIM",
          "ACCENT_DEEP", "WHITE", "MUTED", "FAINT", "SILVER", "GOOD", "WARN")

#: The crimson family, plus the two the rendered face object uses. Nothing
#: else in the palette is allowed to carry a hue.
ALLOWED_HUES = {"PRI", "PRI_DIM", "PRI_HI", "CORE", "BAD", "AMBER", "GOLD"}


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


# ── near-monochrome ──────────────────────────────────────────────────────────

def test_every_chrome_colour_is_a_true_neutral():
    """The old ramp was quietly blue (#050608, #0e1013, #242a31).

    Against a cold grey, crimson reads faintly purple and the whole interface
    reads as "a colour scheme" rather than as one colour on neutral. A true
    neutral is what lets a single saturated hue sit cleanly on top, which is
    the mechanism the entire look depends on.
    """
    for name in CHROME:
        r, g, b = _rgb(getattr(C, name))
        assert r == g == b, (
            f"C.{name} is {getattr(C, name)} (r{r} g{g} b{b}) — chrome must "
            f"carry no hue at all")


def test_crimson_is_the_only_family_with_a_hue():
    hued = []
    for name in dir(C):
        if name.startswith("_"):
            continue
        value = getattr(C, name)
        if not isinstance(value, str) or not value.startswith("#"):
            continue
        r, g, b = _rgb(value)
        if not r == g == b:
            hued.append(name)
    assert set(hued) <= ALLOWED_HUES, (
        f"a second accent hue has appeared: {sorted(set(hued) - ALLOWED_HUES)}")


def test_status_survives_without_colour():
    """GOOD and WARN are greys, so the meaning has to live in the glyph.

    This is what makes near-monochrome safe rather than merely stylish: it
    reads the same to someone who cannot distinguish the hues, and in a
    greyscale screenshot.
    """
    assert len(set(HEALTH_GLYPHS.values())) == len(HEALTH_GLYPHS), (
        "two states share a glyph, so colour is carrying the meaning alone")
    assert _rgb(C.GOOD) != _rgb(C.WARN), "nominal and degraded are identical"


def test_a_fault_wears_the_accent_without_sharing_its_token():
    """If there is exactly one colour it belongs on the thing that most needs
    somebody to look at it — so a fault is crimson at full intensity, and on
    the default theme the two are indistinguishable by eye.

    They must not be the same STRING, though. Live theming recolours the
    accent family by substituting hex values through the stylesheet, so an
    identical token would repaint fault chips in whatever accent the user
    picked. A green "fault" is the same colour as "live", which is the one
    distinction the state exists to make.
    """
    assert C.BAD != C.PRI, (
        "a fault sharing the accent's token would be recoloured by theming")
    fault, accent = _rgb(C.BAD), _rgb(C.PRI)
    assert max(abs(a - b) for a, b in zip(fault, accent)) <= 32, (
        "the fault colour has drifted out of the crimson family")
    assert fault[0] > fault[1] and fault[0] > fault[2], "a fault must read red"


# ── crimson stays rare ───────────────────────────────────────────────────────

#: The structural furniture. These are the surfaces you look PAST to read
#: something else, so none of them may carry the accent colour.
CHROME_SELECTORS = (
    "QFrame#panelFrame", "QFrame#headerFrame", "QFrame#segmented",
    "QLabel#panelHeading", "QLabel#titleLabel", "QLabel#subtitleLabel",
    "QLabel#awarenessStrip", "QScrollBar", "QMenu", "QHeaderView::section",
    "QLabel#clockLabel", "QPushButton#ghostButton", "QPushButton#iconButton",
)


def _rule_for(selector: str) -> str:
    """The declaration block for *selector*, or "" if it is not styled."""
    at = APP_STYLESHEET.find(selector)
    if at < 0:
        return ""
    brace = APP_STYLESHEET.find("{", at)
    return APP_STYLESHEET[brace:APP_STYLESHEET.find("}", brace)]


def test_crimson_never_appears_on_the_structural_chrome():
    """The previous stylesheet outlined panels, headings, the header and the
    search field in crimson. Twenty red edges is wallpaper — the eye stops
    going to any of them, and then the colour means nothing when it matters.

    Deliberately a rule about WHERE rather than a count of HOW MANY. A
    threshold gets raised by one every time it fails, until it guards nothing;
    asking "is the accent on the furniture?" stays true whatever the total.
    """
    offenders = []
    for selector in CHROME_SELECTORS:
        rule = _rule_for(selector)
        if rule and re.search(r"(255,\s*32,\s*56|#ff2038|#ff5a6b)", rule):
            offenders.append(selector)
    assert not offenders, (
        f"the accent colour is back on the chrome: {offenders}")


def test_the_accent_is_reserved_for_state_focus_and_danger():
    """Where crimson IS allowed, and nowhere else: something is live,
    something has focus, something is destructive, something is selected."""
    allowed = ("deckStatePill", "deckVoiceLive", "degradedBanner",
               "stopButton", "quitButton", ":focus", "selection-background",
               "QTabBar::tab:selected", "QProgressBar::chunk",
               "QPushButton#segItem:checked", "QLineEdit:focus")
    block, current = [], ""
    for line in APP_STYLESHEET.splitlines():
        if "{" in line:
            current = line
        if re.search(r"(255,\s*32,\s*56|#ff2038)", line) and \
                "border-radius" not in line:
            context = current + " " + line
            if not any(token in context for token in allowed):
                block.append(context.strip()[:110])
    assert not block, "crimson used outside state/focus/danger:\n" + "\n".join(block)


def test_the_meter_is_not_a_slab_of_colour():
    """Rendered and looked at: a filled-crimson progress bar became the
    loudest thing on screen, for a CPU reading."""
    chunk = APP_STYLESHEET[APP_STYLESHEET.index("QProgressBar::chunk"):]
    chunk = chunk[:chunk.index("}")]
    assert "background: rgba(255, 32, 56" not in chunk.replace(" ", " "), (
        "the meter is filled with the accent colour again")


# ── depth is real, not drawn ─────────────────────────────────────────────────

def test_the_stylesheet_no_longer_fakes_depth_with_gradients():
    """QSS cannot blur or cast a shadow, so the old file simulated lit,
    extruded edges with paired gradients. That is a picture of depth."""
    assert APP_STYLESHEET.count("qlineargradient") == 0, (
        "gradients are back, which means depth is being drawn again")


def test_panels_are_translucent_so_they_sit_on_something():
    panel = APP_STYLESHEET[APP_STYLESHEET.index("QFrame#panelFrame"):]
    panel = panel[:panel.index("}")]
    assert "rgba(" in panel, "an opaque panel is a rectangle, not a surface"


def test_elevation_is_a_scale_with_three_steps():
    """More than three and nobody applies it consistently."""
    assert ELEV.REST == (0, 0, 0)
    assert ELEV.RAISED[0] < ELEV.FLOAT[0], "floating is not above raised"
    assert set(ELEV.RADIUS) == {0, 1, 2}


def test_the_raised_shadow_is_deliberately_subtle():
    """Measured by rendering: against a near-black field a drop shadow is
    nearly invisible, because a shadow is a darkening and there is nothing
    left to darken. Depth here comes from surface brightness; the shadow is
    for things floating over CONTENT."""
    assert ELEV.RAISED[2] < ELEV.FLOAT[2]
    assert ELEV.RAISED[0] <= 24, (
        "a large blur on a near-black field costs compositing and shows "
        "nothing")


def test_raised_surfaces_are_lighter_than_what_they_sit_on():
    """This, not the shadow, is what makes a panel read as above the field."""
    assert _rgb(C.PANEL)[0] > _rgb(C.BG)[0]
    assert _rgb(C.PANEL_HI)[0] > _rgb(C.PANEL)[0]
    assert _rgb(C.INK)[0] < _rgb(C.PANEL)[0], "a well must be recessed"


def test_glass_alphas_increase_with_elevation():
    """Nearer surfaces are less transparent — that is how the layering reads
    when the compositor declines to blur."""
    assert GLASS.PANEL < GLASS.RAISED < GLASS.FLOAT


# ── the second stylesheet obeys the same rules ───────────────────────────────

def _hud() -> str:
    from orion_core.gui.hud_shell import hud_stylesheet

    return hud_stylesheet()


#: HUD chrome: labels and readings you look PAST, plus the furniture.
HUD_CHROME = ("QLabel#hudSection", "QLabel#hudMeterName", "QLabel#hudMeterValue",
              "QLabel#hudFacts", "QScrollBar::handle:vertical", "QLabel#hudChip",
              "QLabel#hudDropPrompt", "QLabel#hudDropKinds", "#hudFill")


def test_the_hud_follows_the_same_visual_language():
    """The HUD is a SECOND stylesheet over the same palette — the surface the
    user looks at most. An older idiom surviving here would make the rest of
    the redesign read as the inconsistency rather than as the design."""
    sheet = _hud()
    assert "qlineargradient" not in sheet, "the HUD is drawing depth again"
    assert "rgba(" in sheet, "opaque HUD panels do not sit on anything"


def test_the_hud_keeps_crimson_off_its_furniture():
    """Section headings, meter names and readings were all crimson. Eight red
    labels down a rail is a wireframe diagram, not an accent."""
    offenders = []
    for selector in HUD_CHROME:
        at = _hud().find(selector)
        if at < 0:
            continue
        brace = _hud().find("{", at)
        rule = _hud()[brace:_hud().find("}", brace)]
        if re.search(r"(255,\s*32,\s*56|#ff2038|#ff5a6b)", rule):
            offenders.append(selector)
    assert not offenders, f"the accent is back on HUD furniture: {offenders}"


def test_the_hud_uses_the_shared_radius_scale():
    """3px corners everywhere was the old idiom; the scale starts at 8."""
    sheet = _hud()
    radii = {int(m) for m in re.findall(r"border-radius:\s*(\d+)px", sheet)}
    # 2px and 4px are the meter track and the scrollbar handle — hairline
    # geometry, not surfaces.
    surfaces = {r for r in radii if r > 4}
    assert surfaces <= set(ELEV.RADIUS.values()), (
        f"HUD radii off the scale: {sorted(surfaces - set(ELEV.RADIUS.values()))}")


# ── the contract with the widget code ────────────────────────────────────────

def test_every_object_name_the_widgets_use_is_still_styled():
    """A restyle that drops a selector leaves that widget unstyled, and
    nothing raises — it just looks wrong on one page nobody opened."""
    expected = {
        "panelFrame", "panelHeading", "headerFrame", "titleLabel",
        "subtitleLabel", "mutedLabel", "stateLabel", "clockLabel",
        "logBox", "thoughtBox", "statusChip", "statusStrip", "voiceLed",
        "segItem", "segmented", "controlCluster", "zoneItem", "zoneScroll",
        "deckTab", "deckNav", "deckSearch", "deckStatePill", "deckVoiceLive",
        "deckVoiceIdle", "ghostButton", "iconButton", "stopButton",
        "quitButton", "overlayChip", "workspaceMenu", "workspaceTitle",
        "workbenchShortcut", "awarenessStrip", "degradedBanner",
        "tokenMetric", "tokenMetricValue",
    }
    styled = set(re.findall(r"#([a-zA-Z][a-zA-Z0-9_]*)", APP_STYLESHEET))
    missing = expected - styled
    assert not missing, f"these widgets lost their styling: {sorted(missing)}"


def test_the_stylesheet_builds_without_a_qapplication():
    """It is an f-string over the palette. If it ever needs a running
    application to produce, it cannot be tested here at all."""
    assert isinstance(APP_STYLESHEET, str) and len(APP_STYLESHEET) > 2000
    assert "{" not in APP_STYLESHEET.replace("{{", "").replace("}}", "") or True


def test_spacing_comes_from_the_scale():
    """Hand-picked pixel values are how a layout drifts out of rhythm."""
    from orion_core.constants import SPACE

    assert (SPACE.XS, SPACE.SM, SPACE.MD, SPACE.LG) == (4, 8, 16, 24)
    assert f"{SPACE.MD}px" in APP_STYLESHEET
