"""Application-wide stylesheet shared by both windows (core + dashboard).

Mark X.6 — futuristic glass HUD. Crimson stays the identity colour; electric
cyan (``C.ACCENT``) is the *interaction* colour: focus rings, selections,
active accents and data highlights. Panels are subtly glassy (top-lit
gradients), controls read as machined hardware.
"""

from __future__ import annotations

from ..constants import C

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {C.BG};
    color: {C.WHITE};
    font-family: "Segoe UI";
}}
QToolTip {{
    background: {C.PANEL_HI};
    color: {C.WHITE};
    border: 1px solid {C.ACCENT_DIM};
    border-radius: 4px;
    padding: 6px 8px;
}}
QLabel {{ background: transparent; }}
QCheckBox {{ background: transparent; }}

/* ── Header: top-lit glass with a cyan data underline ──────────────────── */
QFrame#headerFrame {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a1a26, stop:0.5 #101018, stop:1 #08080e);
    border: 1px solid {C.BORDER};
    border-top: 1px solid #2c2c3a;
    border-bottom: 2px solid {C.PRI_DIM};
    border-radius: 10px;
}}
QFrame#panelFrame {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #14141e, stop:1 #0a0a10);
    border: 1px solid #331420;
    border-radius: 10px;
}}

QLabel#titleLabel {{
    color: {C.WHITE};
    font-size: 23px;
    font-weight: 800;
}}
QLabel#subtitleLabel, QLabel#mutedLabel {{
    color: {C.MUTED};
    font-size: 11px;
}}
QLabel#panelHeading {{
    color: {C.ACCENT};
    font-size: 11px;
    font-weight: 800;
    padding-bottom: 3px;
    border-bottom: 1px solid {C.BORDER};
}}
QLabel#clockLabel, QLabel#stateLabel {{
    color: {C.WHITE};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1b1b26, stop:1 #0d0d13);
    border: 1px solid {C.PRI_DIM};
    border-radius: 8px;
    padding: 7px 14px;
    font-family: "Cascadia Mono", "Consolas";
    font-weight: 700;
}}
QLabel#stateLabel {{ color: {C.PRI}; border: 1px solid {C.PRI_DIM}; }}
QLabel#clockLabel {{ color: {C.ACCENT}; border: 1px solid {C.ACCENT_DIM}; }}
QLabel#voiceLed {{
    color: {C.MUTED};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1b1b26, stop:1 #0d0d13);
    border: 1px solid {C.BORDER};
    border-radius: 8px;
    padding: 7px 12px;
    font-weight: 700;
}}
QLabel#voiceLed[speaking="true"] {{
    color: {C.WHITE};
    border: 1px solid {C.PRI};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #7c0d1e, stop:1 #4a0812);
}}

QPlainTextEdit#logBox {{
    background: {C.INK};
    color: {C.WHITE};
    border: 1px solid {C.BORDER};
    border-radius: 8px;
    padding: 10px;
    font-family: "Cascadia Mono", "Consolas";
    font-size: 11px;
    selection-background-color: {C.ACCENT_DIM};
}}
QLineEdit {{
    background: #0a0a10;
    color: {C.WHITE};
    border: 1px solid {C.PRI_DIM};
    border-radius: 8px;
    padding: 9px 11px;
    selection-background-color: {C.ACCENT_DIM};
}}
QLineEdit:focus {{
    border: 1px solid {C.ACCENT};
    background: #0c0e14;
}}

QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #b3132b, stop:1 #7c0d1e);
    color: {C.WHITE};
    border: 1px solid {C.PRI};
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 700;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C.PRI_HI}, stop:1 #b3132b);
    border: 1px solid #ff8a9b;
}}
QPushButton:pressed {{ background: #66091a; }}
QPushButton:disabled {{
    background: #1a1016;
    color: {C.FAINT};
    border: 1px solid {C.BORDER};
}}
/* Secondary / accent action buttons opt in via objectName. */
QPushButton#accentButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0b3d46, stop:1 #072a30);
    border: 1px solid {C.ACCENT_DIM};
    color: {C.ACCENT};
}}
QPushButton#accentButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0e5460, stop:1 #0a3a42);
    border: 1px solid {C.ACCENT};
    color: {C.WHITE};
}}
QPushButton#ghostButton {{
    background: transparent;
    border: 1px solid {C.BORDER};
    color: {C.MUTED};
}}
QPushButton#ghostButton:hover {{
    border: 1px solid {C.ACCENT_DIM};
    color: {C.WHITE};
}}
/* Shutdown control — reads as a real power button, not a routine action. */
QPushButton#quitButton {{
    background: transparent;
    border: 1px solid {C.BORDER};
    color: {C.MUTED};
    padding: 9px 12px;
    font-size: 14px;
}}
QPushButton#quitButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C.PRI}, stop:1 #7c0d1e);
    border: 1px solid #ff8a9b;
    color: {C.WHITE};
}}
/* ── Segmented control — one navigation language app-wide ──────────────── */
QFrame#segmented {{
    background: {C.INK};
    border: 1px solid {C.BORDER};
    border-radius: 10px;
}}
QPushButton#segItem {{
    background: transparent;
    border: none;
    border-radius: 8px;
    color: {C.MUTED};
    padding: 7px 15px;
    font-weight: 700;
    font-size: 12px;
}}
QPushButton#segItem:hover {{ color: {C.WHITE}; }}
QPushButton#segItem:checked {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #b3132b, stop:1 #7c0d1e);
    color: {C.WHITE};
}}

/* ── Compact icon buttons, grouped into a cluster ──────────────────────── */
QFrame#controlCluster {{
    background: {C.INK};
    border: 1px solid {C.BORDER};
    border-radius: 10px;
}}
QPushButton#iconButton {{
    background: transparent;
    border: none;
    border-radius: 8px;
    color: {C.MUTED};
    min-width: 34px;
    max-width: 44px;
    padding: 7px 8px;
    font-size: 15px;
    font-weight: 700;
}}
QPushButton#iconButton:hover {{ color: {C.WHITE}; background: {C.ACCENT_DEEP}; }}
QPushButton#iconButton:checked {{ color: {C.ACCENT}; background: {C.ACCENT_DEEP}; }}

/* Floating glass chips shown over the compact overlay orb. */
QPushButton#overlayChip {{
    background: rgba(10, 12, 18, 205);
    border: 1px solid {C.ACCENT_DIM};
    border-radius: 15px;
    color: {C.WHITE};
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
    padding: 0;
    font-size: 15px;
    font-weight: 800;
}}
QPushButton#overlayChip:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0e5460, stop:1 #0a3a42);
    border: 1px solid {C.ACCENT};
}}

QCheckBox {{
    color: {C.WHITE};
    spacing: 9px;
    font-weight: 700;
}}
QCheckBox::indicator {{ width: 20px; height: 20px; }}
QCheckBox::indicator:unchecked {{
    background: #0a0a10;
    border: 1px solid {C.PRI_DIM};
    border-radius: 5px;
}}
QCheckBox::indicator:unchecked:hover {{ border: 1px solid {C.ACCENT}; }}
QCheckBox::indicator:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {C.ACCENT}, stop:1 {C.ACCENT_DIM});
    border: 1px solid #7ff2ff;
    border-radius: 5px;
}}

QComboBox {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #16161f, stop:1 #0d0d13);
    color: {C.WHITE};
    border: 1px solid {C.PRI_DIM};
    border-radius: 8px;
    padding: 7px 12px;
    font-weight: 700;
    min-width: 160px;
}}
QComboBox:hover {{ border: 1px solid {C.ACCENT}; }}
QComboBox::drop-down {{ border: none; padding-right: 10px; }}
QComboBox QAbstractItemView {{
    background: {C.PANEL};
    color: {C.WHITE};
    border: 1px solid {C.ACCENT_DIM};
    selection-background-color: {C.ACCENT_DIM};
    padding: 4px;
}}

QTableWidget {{
    background: {C.INK};
    color: {C.WHITE};
    gridline-color: {C.BORDER};
    alternate-background-color: #101018;
    border: 1px solid {C.BORDER};
    border-radius: 8px;
    selection-background-color: {C.ACCENT_DIM};
}}
QHeaderView::section {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #16161f, stop:1 #0f0f14);
    color: {C.ACCENT};
    border: 1px solid {C.BORDER};
    padding: 5px;
    font-weight: 700;
}}
QListWidget {{
    background: {C.INK};
    color: {C.WHITE};
    border: 1px solid {C.BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QListWidget::item {{ padding: 5px 6px; border-radius: 5px; }}
QListWidget::item:selected {{ background: {C.ACCENT_DIM}; }}
QListWidget::item:hover {{ background: #14141e; }}

QCalendarWidget QWidget {{ alternate-background-color: #101018; }}
QCalendarWidget QAbstractItemView {{
    background: {C.INK};
    color: {C.WHITE};
    selection-background-color: {C.ACCENT_DIM};
    selection-color: {C.WHITE};
    outline: none;
}}
QCalendarWidget QToolButton {{
    background: transparent;
    color: {C.WHITE};
    font-weight: 700;
    border-radius: 6px;
    padding: 4px 8px;
}}
QCalendarWidget QToolButton:hover {{ background: {C.ACCENT_DIM}; }}
QCalendarWidget QMenu {{ background: {C.PANEL}; color: {C.WHITE}; }}
QCalendarWidget QSpinBox {{
    background: {C.PANEL};
    color: {C.WHITE};
    border: 1px solid {C.ACCENT_DIM};
}}

QSplitter::handle {{ background: {C.BG}; }}
QSplitter::handle:hover {{ background: {C.ACCENT_DEEP}; }}

QTabWidget::pane {{
    border: 1px solid {C.BORDER};
    border-radius: 8px;
}}
QTabBar::tab {{
    background: #101018;
    color: {C.MUTED};
    padding: 8px 18px;
    border: 1px solid {C.BORDER};
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 700;
}}
QTabBar::tab:selected {{
    background: {C.ACCENT_DIM};
    color: {C.WHITE};
    border-bottom: 2px solid {C.ACCENT};
}}
QTabBar::tab:hover:!selected {{ color: {C.WHITE}; }}

QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C.PRI_DIM};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {C.ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 9px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {C.PRI_DIM};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C.ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""
