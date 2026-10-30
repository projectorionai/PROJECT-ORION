"""
House style for names on screen: Title Case for names, sentence case for sentences.

"Vision lab" read as a typo next to "Command Centre". Names - windows, pages,
tabs, group boxes, menu entries and buttons - are written in Title Case; status
lines, empty states, tooltips and placeholders stay in sentence case. These
tests keep it that way: the helper that builds names at run time, the one table
of page names every surface shares, and a scan of the widgets that carry a name.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.gui.labels import PAGE_DISPLAY_NAMES, page_display_name, title_case  # noqa: E402


def test_title_case_capitalises_names_and_keeps_minor_words_down():
    assert title_case("Vision lab") == "Vision Lab"
    assert title_case("Motion and performance") == "Motion and Performance"
    assert title_case("Start at login") == "Start at Login"
    assert title_case("Pin to the top") == "Pin to the Top"
    assert title_case("Depends on") == "Depends On"            # last word is never minor
    assert title_case("the end") == "The End"                  # first word is never minor
    assert title_case("Details — services, actions and usage") == \
        "Details — Services, Actions and Usage"


def test_title_case_leaves_brands_acronyms_and_numbers_alone():
    assert title_case("check ORION health") == "Check ORION Health"
    assert title_case("Connect WhatsApp") == "Connect WhatsApp"
    assert title_case("Wi-Fi and GPU") == "Wi-Fi and GPU"
    assert title_case("Queue NNUE training") == "Queue NNUE Training"
    assert title_case("show 3-D view") == "Show 3-D View"
    assert title_case("technical deep-dive") == "Technical Deep-Dive"


def test_title_case_is_idempotent():
    for text in ("Vision Lab", "Play & Analyse", "Codes (QR & Barcodes)",
                 "Model Usage — Last 24 Hours", "ALERT HISTORY"):
        assert title_case(title_case(text)) == title_case(text)
    assert title_case("") == ""


def test_the_pages_have_one_name_everywhere():
    """The sidebar, header, palette, Overview and placeholder all use this."""
    assert page_display_name("WORKBENCH") == "Vision Lab"
    assert page_display_name("RESEARCH") == "Research Console"
    assert page_display_name("COMMAND CENTRE") == "Command Centre"
    assert page_display_name("OPS") == "Operations"
    assert page_display_name("DIAGNOSTICS") == "Diagnostics"
    assert page_display_name("workbench") == "Vision Lab"
    assert all(name == title_case(name) for name in PAGE_DISPLAY_NAMES.values())

    from orion_core.gui.command_palette import build_page_commands
    from orion_core.gui.unified_dashboard import UnifiedDashboard

    assert UnifiedDashboard.display_name("WORKBENCH") == "Vision Lab"
    titles = {c.name: c.title for c in build_page_commands(["WORKBENCH", "RESEARCH", "TASKS"])}
    assert titles == {"WORKBENCH": "Vision Lab", "RESEARCH": "Research Console",
                      "TASKS": "Tasks"}


def test_the_palette_names_a_tool_without_a_written_title_in_title_case():
    from orion_core.gui.command_palette import command_title

    assert command_title("web_search") == "Search the Web"
    assert command_title("some_new_tool") == "Some New Tool"
    assert command_title("ai_mode") == "AI Mode"                # acronyms stay capitals
    assert command_title("fetch_url") == "Fetch URL"


def test_the_overview_quick_links_use_the_page_names(monkeypatch):
    from PyQt6.QtWidgets import QApplication

    from orion_core.bus import OrionBus
    from orion_core.gui.command_overview import CommandOverview

    app = QApplication.instance() or QApplication([])
    view = CommandOverview(OrionBus())
    assert view._quick_links["COMMAND CENTRE"].text() == "Command Centre"
    assert view._quick_links["RESEARCH"].text() == "Research Console"
    view.deleteLater()
    app.processEvents()


# ── a scan of the widgets that carry a name ───────────────────────────────────
#
# Only the first string literal of constructors/adders that make a NAME: a
# button, a group box, a tab, a menu, a menu entry, a window title. Sentences
# (status lines, tooltips, placeholders) are not scanned - they are sentence
# case on purpose - and neither are the all-capitals HUD tags, which title_case
# leaves untouched anyway.

_NAMING_CALLS = {"QPushButton", "QToolButton", "QGroupBox", "QCheckBox", "addTab",
                 "addMenu", "addAction", "setWindowTitle"}

#: Not names: a whole sentence that happens to sit on a button or checkbox, or
#: an empty-state line (a disabled menu entry saying there is nothing to list).
_SENTENCES = {
    "I agree to save my voiceprint (numbers, never a recording).",
    "Everyone being recorded has agreed to this recording.",
    "Nothing opened yet",
}

#: Labels that carry a live value after a colon, kept lower-case on purpose.
_VALUE_LABELS = ("Automatic Actions: ", "Start at Login: ", "Push to Talk: ")

#: Qt dialog buttons or toggles whose word is a state, not a name.
_SKIP_FILES = {"screen_face.py", "face3d.py", "holo_head.py", "head_mesh.py", "labels.py"}


def _naming_literals():
    gui = ROOT / "orion_core" / "gui"
    for path in sorted(gui.glob("*.py")):
        if path.name in _SKIP_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in _NAMING_CALLS:
                continue
            # addTab(widget, "Title") puts the name second; the rest put it first.
            args = node.args[1:2] if name == "addTab" else node.args[:1]
            for arg in args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield path.name, arg.lineno, arg.value


def test_every_named_widget_is_in_title_case():
    offenders = []
    for filename, line, text in _naming_literals():
        shown = text.replace("&&", "&")
        if shown in _SENTENCES or shown.startswith(_VALUE_LABELS):
            continue
        if not shown.strip() or not any(ch.isalpha() for ch in shown):
            continue
        if title_case(shown) != shown:
            offenders.append(f"{filename}:{line} {shown!r} -> {title_case(shown)!r}")
    assert offenders == [], "names not in Title Case:\n  " + "\n  ".join(offenders)


def test_no_emoji_in_the_names_on_screen():
    """Pictographic emoji draw as colour pictures on Windows, in an interface
    that spends its colour deliberately."""
    pictographic = [(0x1F300, 0x1FAFF), (0x23E9, 0x23F3), (0x231A, 0x231B), (0x2B50, 0x2B50)]
    offenders = []
    for filename, line, text in _naming_literals():
        for char in text:
            if any(lo <= ord(char) <= hi for lo, hi in pictographic):
                offenders.append(f"{filename}:{line} {text!r}")
    assert offenders == []
    gui = ROOT / "orion_core" / "gui"
    for filename in ("entrepreneur.py", "plugin_deck.py", "status_widgets.py", "mission_deck.py"):
        source = (gui / filename).read_text(encoding="utf-8")
        for glyph in ("⏰", "⏳", "⚕", "⚠"):
            assert glyph not in source, f"{filename} still draws {glyph}"


# ── section headings, card headers and shouted names (Mark XXXVII) ───────────
#
# The scan above only reads the FIRST literal of a widget constructor, and
# title_case() leaves an ALL-CAPS word alone (it looks like an acronym), so
# "DEEP FOCUS" and "REFRESH INBOX" passed it - while the Mark XXXV heading filter
# turned them into "Deep focus". Section headings are made by deck helpers
# (_panel("..."), _heading("..."), _panel_into(frame, "...")) and by a QLabel
# given a heading object name; names are Title Case, never sentence case and
# never shouted.

_HEADING_HELPERS = {"_panel", "_panel_into", "_heading", "_section", "_card", "_title",
                    "_group", "_titled", "_make_panel", "_section_label"}
_HEADING_OBJECTS = {"panelHeading", "titleLabel", "sidebarCaption", "sectionTitle",
                    "cardTitle", "pageTitle", "workspaceTitle", "workbenchTitle", "summaryKey"}
_NAME_CALLS = {"QPushButton": (0,), "QToolButton": (0,), "QGroupBox": (0,), "QCheckBox": (0,),
               "QRadioButton": (0,), "addTab": (1,), "addMenu": (0,), "addAction": (0,),
               "setWindowTitle": (0,), "setTitle": (0,), "setTabText": (1,)}
#: Sentences, empty states and values that sit in a heading-shaped place.
_KEEP = {"Full reports are kept in config/crash_reports/", "No tasks yet", "Nothing opened yet",
         "I agree to save my voiceprint (numbers, never a recording).",
         "Everyone being recorded has agreed to this recording.", "Automatic Actions: checking"}


def _shouted(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


def _heading_literals():
    from orion_core.gui.humane import sentence_case  # noqa: F401  (the converter the filter uses)

    for path in sorted((ROOT / "orion_core" / "gui").glob("*.py")):
        if path.name in _SKIP_FILES or path.name == "humane.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        heading_vars = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setObjectName" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in _HEADING_OBJECTS):
                heading_vars.add(ast.unparse(node.func.value))
        for node in ast.walk(tree):
            found = []
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                if getattr(call.func, "id", None) == "QLabel" and call.args \
                        and ast.unparse(node.targets[0]) in heading_vars:
                    found.append(call.args[0])
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in _HEADING_HELPERS:
                    found.extend(node.args[:2])
                elif name in _NAME_CALLS:
                    found.extend(node.args[i] for i in _NAME_CALLS[name] if len(node.args) > i)
            for arg in found:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield path.name, arg.lineno, arg.value


def test_every_section_heading_card_header_and_button_is_title_case_not_shouted():
    from orion_core.gui.humane import heading_case

    offenders = []
    for filename, line, text in _heading_literals():
        if text in _KEEP or not any(ch.isalpha() for ch in text):
            continue
        wanted = heading_case(text.replace("&&", "&")) if _shouted(text) else title_case(text.replace("&&", "&"))
        if wanted != text.replace("&&", "&") and wanted != text:
            offenders.append(f"{filename}:{line} {text!r} -> {wanted!r}")
    assert offenders == [], "headings not in Title Case:\n  " + "\n  ".join(offenders)


def test_the_scan_reaches_the_cognition_headings():
    seen = {(f, t) for f, _l, t in _heading_literals()}
    assert ("cognition_deck.py", "Deep Focus") in seen
    assert ("cognition_deck.py", "Study & Recall") in seen
    assert ("tasks_view.py", "Recent Actions") in seen
    assert ("security_centre.py", "Phone Access") in seen


def test_headings_are_titled_not_sentenced_or_shouted():
    from orion_core.gui.humane import heading_case

    assert heading_case("DEEP FOCUS") == "Deep Focus"
    assert heading_case("Deep focus") == "Deep Focus"
    assert heading_case("STUDY & RECALL") == "Study & Recall"
    assert heading_case("Study & recall") == "Study & Recall"
    assert heading_case("WAITING FOR YOUR APPROVAL") == "Waiting for Your Approval"
    assert heading_case("ORION AI CORE") == "ORION AI Core"
    assert heading_case("CPU & GPU") == "CPU & GPU"
    assert heading_case("Command Centre") == "Command Centre"          # idempotent


def test_the_heading_voice_titles_the_heading_labels_of_a_page():
    """28 Oct 2026: a sweep per window and per deck page replaced the event
    filter on the whole application (Python on every Qt event)."""
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    from orion_core.gui.humane import HEADING_NAMES, apply_heading_voice, install_heading_voice

    app = QApplication.instance() or QApplication([])
    cases = {"panelHeading": ("Deep focus", "Deep Focus"),
             "titleLabel": ("RECENT ACTIONS", "Recent Actions"),
             "sectionTitle": ("study & recall", "Study & Recall"),
             "mutedLabel": ("Make space for one thing.", "Make space for one thing.")}
    page = QWidget()
    layout = QVBoxLayout(page)
    labels = []
    for name, (text, want) in cases.items():
        label = QLabel(text)
        label.setObjectName(name)
        layout.addWidget(label)
        labels.append((label, want))
    assert apply_heading_voice(page) == 3
    for label, want in labels:
        assert label.text() == want
    assert apply_heading_voice(page) == 0                       # idempotent
    assert {"panelHeading", "titleLabel", "sidebarCaption", "sectionTitle"} <= HEADING_NAMES
    voice = install_heading_voice(app)
    assert voice is not None
    later = QLabel("LIVE CHANNEL", page)
    later.setObjectName("panelHeading")
    voice.focus_moved(None, later)                              # the window gains focus
    assert later.text() == "Live Channel"
    voice.setParent(None)


@pytest.mark.parametrize("count,word,plural_form,expected", [
    (0, "card", None, "0 cards"), (1, "card", None, "1 card"), (2, "card", None, "2 cards"),
    (1, "entry", "entries", "1 entry"), (3, "entry", "entries", "3 entries"),
    (1.0, "minute", None, "1 minute"), (2.5, "hour", None, "2.5 hours"),
])
def test_plural_agrees_with_the_number(count, word, plural_form, expected):
    from orion_core.gui.labels import plural

    assert plural(count, word, plural_form) == expected


def test_no_deck_text_pluralises_with_brackets():
    """"1 distraction(s)" and "1 cards" read as unfinished: the noun agrees with
    the number (labels.plural)."""
    import re

    offenders = []
    for path in sorted((ROOT / "orion_core" / "gui").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                first = node.body[0] if node.body else None
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docstrings.add(id(first.value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings and re.search(r"[a-z]\(s\)", node.value)):
                offenders.append(f"{path.name}:{node.lineno} {node.value[:70]!r}")
    assert offenders == [], "unfinished plurals:\n  " + "\n  ".join(offenders)


def test_the_cognition_panels_agree_with_their_numbers(tmp_path):
    from PyQt6.QtWidgets import QApplication

    from orion_core.bus import OrionBus
    from orion_core.gui.cognition_deck import StudyPanel
    from orion_core.study import StudyEngine, StudyStore

    app = QApplication.instance() or QApplication([])
    study = StudyEngine(StudyStore(tmp_path / "s.db"))
    study.add("Q?", "A", deck="D")
    panel = StudyPanel(OrionBus(), study)
    try:
        panel.refresh()
        assert "1 card " in panel.summary.text() + " " and "1 cards" not in panel.summary.text()
        study.add("Q2?", "A2", deck="D")
        panel.refresh()
        assert "2 cards" in panel.summary.text()
    finally:
        panel.deleteLater()
        app.processEvents()
