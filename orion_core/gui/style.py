"""
Application-wide stylesheet — Mark XXX, "spatial glass".

What changed and why
--------------------
The previous look drew depth. Qt Style Sheets cannot blur and cannot cast a
shadow, so it simulated lit, extruded surfaces with paired gradients and
bright/dim border tricks. That produces a picture of depth: convincing in a
screenshot, and the reason the interface read as the same genre as every other
dark assistant rather than as a considered thing.

Three changes, in order of how much they matter:

**Depth is real now.** Shadows come from ``QGraphicsDropShadowEffect`` and the
blur behind the window from the Windows compositor itself — both in
``gui.depth``. So this file stops trying to draw light and instead does what
QSS is actually good at: surface, radius, spacing and type.

**Borders mostly went away.** A panel used to be defined by a line drawn
around it. Now it is defined by being a lighter surface with a shadow under it
and a single hairline of light along its top edge — the way a pane of glass
catches a room light. Boxes-within-boxes is what made the old layout feel
cramped at any density.

**One colour.** Crimson appears on state and focus and nowhere else. Chrome is
a neutral ramp. Twenty red edges is wallpaper; one red thing is an
instruction, and the eye goes straight to it.

Every objectName selector from the previous stylesheet is preserved, so no
widget code changes.
"""

from __future__ import annotations

from ..constants import C, ELEV, GLASS, SPACE, TYPE


def _rgba(hex_colour: str, alpha: float) -> str:
    """A QSS rgba() from one of the palette's hex tokens.

    Translucency is the whole mechanism here — an opaque panel is a
    rectangle, a translucent one sits ON something — so it is worth having one
    honest way to express it rather than hand-written rgba triples that drift
    from the palette.
    """
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


#: The same helper under a public name. The HUD is a second stylesheet over
#: the same palette, and two hand-written ways of writing translucency is how
#: two surfaces end up almost-but-not-quite matching.
rgba = _rgba

#: The light that catches the top edge of a glass surface. One hairline, and
#: only along the top: light comes from above, so an edge lit all the way
#: round reads as a drawn outline rather than as a lit object.
_LIP = f"1px solid {_rgba(C.WHITE, 0.10)}"
_EDGE = f"1px solid {_rgba(C.WHITE, 0.055)}"

#: Resting, raised and floating surfaces.
_PANEL = _rgba(C.PANEL, GLASS.PANEL)
_RAISED = _rgba(C.PANEL_HI, GLASS.RAISED)
_FLOATING = _rgba(C.PANEL_HI, GLASS.FLOAT)
_WELL = _rgba(C.INK, 0.66)

APP_STYLESHEET = f"""
/* Base ---------------------------------------------------------------------*/
QWidget {{
    background: transparent;
    color: {C.SILVER};
    font-family: "Segoe UI Variable Display", "Segoe UI";
    font-size: {TYPE.BODY_PX}px;
}}
/* The window itself carries the only fill. Everything above it is glass, so
   this is what shows through — and on Windows 11 the compositor blurs the
   desktop behind it (see gui.depth.apply_backdrop), which is the one part of
   this look Qt cannot fake. */
QMainWindow, QDialog {{
    background: {_rgba(C.BG, 0.88)};
}}
QToolTip {{
    background: {_FLOATING};
    color: {C.WHITE};
    border: {_EDGE};
    border-top: {_LIP};
    border-radius: 10px;
    padding: {SPACE.SM}px {SPACE.MD}px;
}}
QLabel, QCheckBox {{ background: transparent; }}

/* Surfaces -----------------------------------------------------------------*/
/* A panel is a lighter slab with a lit top edge. The shadow that lifts it off
   the field is applied in code, because QSS has none. */
QFrame#panelFrame {{
    background: {_PANEL};
    border: {_EDGE};
    border-top: {_LIP};
    border-radius: 14px;
}}
QLabel#panelHeading {{
    color: {C.MUTED};
    font-size: {TYPE.HEADING_PX}px;
    font-weight: 600;
    letter-spacing: 1.4px;
    padding: {SPACE.SM}px {SPACE.MD}px {SPACE.XS}px {SPACE.MD}px;
}}
QFrame#headerFrame {{
    background: {_rgba(C.PANEL, 0.55)};
    border: none;
    border-bottom: {_EDGE};
    border-radius: 0px;
}}
QFrame#segmented, QFrame#controlCluster {{
    background: {_rgba(C.INK, 0.55)};
    border: {_EDGE};
    border-top: {_LIP};
    border-radius: 13px;
}}
/* Recessed wells: logs, transcripts, anything you read INTO rather than off.
   Darker than the surface holding them, which is what "recessed" means. */
QPlainTextEdit#logBox, QTextEdit#thoughtBox, QPlainTextEdit, QTextEdit {{
    background: {_WELL};
    color: {C.SILVER};
    border: {_EDGE};
    border-radius: 12px;
    padding: {SPACE.SM}px;
    selection-background-color: {_rgba(C.PRI, 0.30)};
    selection-color: {C.WHITE};
}}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollArea#zoneScroll {{ border: none; }}

/* Type ---------------------------------------------------------------------*/
QLabel#titleLabel {{
    color: {C.WHITE};
    font-size: {TYPE.TITLE_PX}px;
    font-weight: 300;              /* light, because it is large */
    letter-spacing: 3px;
}}
QLabel#subtitleLabel {{ color: {C.FAINT}; letter-spacing: 1.2px; }}
/* Mark XXXI identity: a quiet wordmark and a framed mark tag. The version
   is the one piece of the header that should be read at a glance, so it is
   the one that is framed — in white, because crimson is kept for state. */
QLabel#brandMark {{
    color: {C.WHITE};
    font-size: {TYPE.TITLE_PX}px;
    font-weight: 300;
    letter-spacing: 4px;
}}
QLabel#markTag {{
    color: {C.WHITE};
    background: {_rgba(C.WHITE, 0.06)};
    border: 1px solid {_rgba(C.WHITE, 0.22)};
    border-radius: 9px;
    padding: 2px {SPACE.SM}px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.2px;
}}
QLabel#workspaceTitle {{
    color: {C.WHITE}; font-size: 16px; font-weight: 500;
    padding: 0 {SPACE.SM}px;
}}
QLabel#mutedLabel {{ color: {C.MUTED}; }}
QLabel#stateLabel {{ color: {C.SILVER}; font-weight: 500; letter-spacing: 1px; }}
QLabel#clockLabel {{
    color: {C.MUTED};
    font-family: {TYPE.MONO_FAMILY};
    letter-spacing: 1px;
}}
QLabel#tokenMetric {{ color: {C.FAINT}; }}
QLabel#tokenMetricValue {{
    color: {C.WHITE}; font-weight: 600;
    font-family: {TYPE.MONO_FAMILY};
}}

/* Buttons ------------------------------------------------------------------*/
/* Quiet at rest, lit on hover, crimson only when they are the live thing. */
QPushButton {{
    background: {_RAISED};
    color: {C.SILVER};
    border: {_EDGE};
    border-top: {_LIP};
    border-radius: 11px;
    padding: {SPACE.SM}px {SPACE.MD}px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {_rgba(C.PANEL_HI, 0.95)};
    color: {C.WHITE};
}}
QPushButton:pressed {{ background: {_rgba(C.INK, 0.90)}; }}
QPushButton:disabled {{ color: {C.FAINT}; background: {_rgba(C.PANEL, 0.45)}; }}
QPushButton#ghostButton {{
    background: transparent;
    color: {C.MUTED};
    border: {_EDGE};
}}
QPushButton#ghostButton:hover {{ color: {C.WHITE}; background: {_rgba(C.WHITE, 0.05)}; }}
QPushButton#iconButton {{
    background: transparent; border: none; border-radius: 10px;
    padding: {SPACE.XS}px {SPACE.SM}px; color: {C.MUTED};
}}
QPushButton#iconButton:hover {{ background: {_rgba(C.WHITE, 0.07)}; color: {C.WHITE}; }}
QPushButton#workbenchShortcut {{
    background: {_RAISED}; color: {C.SILVER};
    border: {_EDGE}; border-top: {_LIP}; border-radius: 11px;
}}
QPushButton#workbenchShortcut:hover {{ color: {C.WHITE}; }}
/* Stop and quit: destructive, so they get the colour — and nothing else does. */
QPushButton#stopButton, QPushButton#quitButton {{
    background: {_rgba(C.PRI, 0.14)};
    color: {C.PRI_HI};
    border: 1px solid {_rgba(C.PRI, 0.40)};
    border-radius: 11px;
}}
QPushButton#stopButton:hover, QPushButton#quitButton:hover {{
    background: {_rgba(C.PRI, 0.24)}; color: {C.WHITE};
}}
QToolButton#workspaceMenu {{
    color: {C.SILVER}; background: {_RAISED};
    border: {_EDGE}; border-top: {_LIP}; border-radius: 11px;
    padding: {SPACE.SM}px {SPACE.MD}px; font-weight: 500;
}}
QToolButton#workspaceMenu:hover {{ color: {C.WHITE}; }}
QToolButton#workspaceMenu::menu-indicator {{ width: 0px; }}

/* Segmented controls and navigation ----------------------------------------*/
/* The selected item is a raised slab, not an outlined box — the same grammar
   as everything else, so selection reads as "nearer" rather than "ringed". */
QPushButton#segItem, QPushButton#zoneItem, QPushButton#deckTab {{
    background: transparent;
    color: {C.MUTED};
    border: 1px solid transparent;
    border-radius: 10px;
    padding: {SPACE.SM}px {SPACE.MD}px;
    font-weight: 500;
}}
QPushButton#segItem:hover, QPushButton#zoneItem:hover, QPushButton#deckTab:hover {{
    color: {C.WHITE}; background: {_rgba(C.WHITE, 0.06)};
}}
QPushButton#segItem:checked, QPushButton#zoneItem:checked, QPushButton#deckTab:checked,
QPushButton#formItem:checked {{
    color: {C.WHITE};
    background: {_rgba(C.PANEL_HI, 0.98)};
    border: {_EDGE};
    border-top: {_LIP};
}}
QFrame#deckNav {{ background: transparent; border: none; }}
/* Face | Orb: the form in effect is a raised slab, the same selection
   grammar as every other segmented control (see segItem above). */
QPushButton#formItem {{
    background: transparent;
    color: {C.MUTED};
    border: 1px solid transparent;
    border-radius: 10px;
    padding: {SPACE.XS}px {SPACE.MD}px;
    font-weight: 600;
    letter-spacing: 1.4px;
}}
QPushButton#formItem:hover {{ color: {C.WHITE}; background: {_rgba(C.WHITE, 0.06)}; }}
QPushButton#headerAction {{
    background: {_RAISED}; color: {C.SILVER};
    border: {_EDGE}; border-top: {_LIP}; border-radius: 11px;
    font-weight: 600; letter-spacing: 1.2px;
}}
QPushButton#headerAction:hover {{ color: {C.WHITE}; border: 1px solid {_rgba(C.WHITE, 0.30)}; }}
/* Research console: the feed reads like a log; the section progress keeps
   the shared QProgressBar::chunk fill — ORION working right now is state. */
QListWidget#researchFeed, QListWidget#researchSources {{
    background: {_WELL};
    border: {_EDGE};
    border-radius: 12px;
    padding: {SPACE.XS}px;
}}
QListWidget#researchFeed::item, QListWidget#researchSources::item {{
    padding: 5px {SPACE.SM}px;
    border-bottom: 1px solid {_rgba(C.WHITE, 0.04)};
}}
QListWidget#researchFeed::item:selected, QListWidget#researchSources::item:selected {{
    background: {_rgba(C.WHITE, 0.10)}; color: {C.WHITE};
}}
QProgressBar#researchProgress {{
    background: {_rgba(C.WHITE, 0.06)};
    border: none; border-radius: 5px; height: 10px;
    color: {C.SILVER}; font-size: 10px;
}}
QLabel#sectionTitle {{ color: {C.WHITE}; font-size: 15px; font-weight: 600; }}
QLineEdit#deckSearch, QLineEdit {{
    background: {_WELL};
    color: {C.WHITE};
    border: {_EDGE};
    border-radius: 11px;
    padding: {SPACE.SM}px {SPACE.MD}px;
    selection-background-color: {_rgba(C.PRI, 0.30)};
}}
QLineEdit:focus {{ border: 1px solid {_rgba(C.PRI, 0.55)}; }}

/* State ---------------------------------------------------------------------*/
/* Chips are the only place crimson lives as a fill, and only while the state
   they describe is actually true. */
QLabel#statusChip, QLabel#voiceLed, QLabel#deckStatePill {{
    color: {C.MUTED};
    background: {_rgba(C.INK, 0.60)};
    border: {_EDGE};
    border-radius: 11px;
    padding: {SPACE.XS}px {SPACE.MD}px;
    font-weight: 500;
    letter-spacing: 0.6px;
}}
QLabel#deckStatePill {{ color: {C.PRI_HI}; border: 1px solid {_rgba(C.PRI, 0.35)}; }}
QLabel#deckVoiceLive {{
    color: {C.PRI_HI};
    background: {_rgba(C.PRI, 0.14)};
    border: 1px solid {_rgba(C.PRI, 0.40)};
    border-radius: 11px;
    padding: {SPACE.XS}px {SPACE.MD}px;
    font-weight: 600;
}}
QLabel#deckVoiceIdle {{
    color: {C.FAINT};
    background: {_rgba(C.INK, 0.55)};
    border: {_EDGE};
    border-radius: 11px;
    padding: {SPACE.XS}px {SPACE.MD}px;
}}
QLabel#awarenessStrip {{
    color: {C.MUTED};
    background: {_rgba(C.INK, 0.45)};
    border: none;
    border-top: {_EDGE};
    border-radius: 0px;
    font-size: 11px;
    letter-spacing: 1.2px;
    padding: {SPACE.XS}px {SPACE.MD}px;
}}
QFrame#statusStrip {{ background: transparent; border: none; }}
QPushButton#overlayChip {{
    background: {_FLOATING};
    color: {C.SILVER};
    border: {_EDGE};
    border-top: {_LIP};
    border-radius: 14px;
    padding: {SPACE.XS}px {SPACE.SM}px;
    font-size: 14px;
}}
QPushButton#overlayChip:hover {{ color: {C.WHITE}; }}
/* Degraded: the one banner allowed to shout, because it means something is
   actually wrong. */
QLabel#degradedBanner {{
    color: {C.PRI_HI};
    background: {_rgba(C.PRI, 0.12)};
    border: 1px solid {_rgba(C.PRI, 0.34)};
    border-radius: 12px;
    padding: {SPACE.SM}px {SPACE.MD}px;
    font-weight: 500;
}}

/* Focus ---------------------------------------------------------------------*/
/* Keyboard focus is crimson and unmissable. It is the live thing, which is
   precisely what the one colour is for. */
*:focus {{ outline: none; }}
QPushButton:focus, QLineEdit:focus, QCheckBox:focus, QComboBox:focus,
QToolButton:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {C.PRI};
}}

/* Scrollbars ----------------------------------------------------------------*/
/* Present when needed, invisible otherwise. A permanently visible track is a
   line drawn down the side of every panel for no reason. */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {_rgba(C.WHITE, 0.14)};
    border-radius: 5px; min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{ background: {_rgba(C.WHITE, 0.26)}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0px; }}
QScrollBar::handle:horizontal {{
    background: {_rgba(C.WHITE, 0.14)};
    border-radius: 5px; min-width: 32px;
}}
QScrollBar::handle:horizontal:hover {{ background: {_rgba(C.WHITE, 0.26)}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* Menus, tabs, tables -------------------------------------------------------*/
QMenu {{
    background: {_FLOATING};
    border: {_EDGE};
    border-top: {_LIP};
    border-radius: 12px;
    padding: {SPACE.XS}px;
}}
QMenu::item {{
    padding: {SPACE.SM}px {SPACE.MD}px; border-radius: 8px; color: {C.SILVER};
}}
QMenu::item:selected {{ background: {_rgba(C.WHITE, 0.08)}; color: {C.WHITE}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {C.MUTED};
    padding: {SPACE.SM}px {SPACE.MD}px; border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {C.WHITE}; border-bottom: 2px solid {C.PRI}; }}
QTabBar::tab:hover {{ color: {C.WHITE}; }}
QHeaderView::section {{
    background: transparent; color: {C.FAINT};
    border: none; border-bottom: {_EDGE};
    padding: {SPACE.SM}px; font-weight: 500; letter-spacing: 1px;
}}
QTableWidget, QTableView, QListWidget, QTreeWidget {{
    background: transparent; border: none;
    selection-background-color: {_rgba(C.PRI, 0.22)};
    selection-color: {C.WHITE};
    alternate-background-color: {_rgba(C.WHITE, 0.02)};
}}
/* A meter is not an alert. Filled crimson, a CPU bar became the loudest
   thing on the screen — which is exactly the wallpaper effect this palette
   exists to avoid. Neutral fill, and the colour only where the value
   currently is. */
QProgressBar {{
    background: {_rgba(C.WHITE, 0.07)};
    border: none; border-radius: 3px; height: 5px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {_rgba(C.WHITE, 0.55)};
    border-radius: 3px;
    border-right: 2px solid {C.PRI};
}}
QComboBox {{
    background: {_RAISED}; color: {C.SILVER};
    border: {_EDGE}; border-radius: 10px;
    padding: {SPACE.XS}px {SPACE.SM}px;
}}
QComboBox QAbstractItemView {{
    background: {_FLOATING}; border: {_EDGE};
    selection-background-color: {_rgba(C.WHITE, 0.08)};
}}
/* Dialogs ------------------------------------------------------------------*/
/* Absent from this sheet until now, which is why every dialog in the app grew
   a private copy of it — the Preferences window carried a whole third
   stylesheet, C tokens in the old idiom, with every button turning crimson on
   hover. One definition, so a dialog looks like the application. */
QDialog {{
    background: {_rgba(C.BG, 0.97)};
    color: {C.SILVER};
}}
QGroupBox {{
    background: {_PANEL};
    border: {_EDGE};
    border-top: {_LIP};
    border-radius: {ELEV.RADIUS[1]}px;
    margin-top: {SPACE.MD}px;
    padding: {SPACE.MD}px {SPACE.SM}px {SPACE.SM}px {SPACE.SM}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {SPACE.SM}px;
    padding: 0 {SPACE.XS}px;
    color: {C.MUTED};
    letter-spacing: 1px;
}}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {_rgba(C.WHITE, 0.06)}; }}
"""

__all__ = ["APP_STYLESHEET", "rgba"]
