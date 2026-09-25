"""
Control labelling — ORION's interface being usable by something other than eyes.

An audit of the deck found **119 buttons, 21 of them labelled with nothing but a
symbol** (``|<``, ``⟳``, ``⇄``, ``‹``, ``⏻``) — and not one of those 21 carried
an accessible name. Eight had no tooltip either.

That matters twice over. Qt derives a control's accessible name from its text,
so a screen reader announced the glyph: "less-than" where the user needed
"previous move". And a symbol obvious to whoever built the panel is a guess for
anyone else — four unlabelled chess buttons is a puzzle, not an interface.

The fix is one helper, ``widgets.describe_control``, that sets the accessible
name and the tooltip **together**, because doing only half is the easy mistake:
a tooltip serves a sighted mouse user and nobody else, an accessible name does
the reverse.

The scan at the bottom is the part that compounds. A new symbol-labelled button
added without a spoken name fails the suite, so this cannot silently rot back.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

from orion_core.gui.widgets import describe_control  # noqa: E402

GUI = ROOT / "orion_core" / "gui"

#: A label of up to three non-alphanumeric characters carries no words at all.
SYMBOL_BUTTON = re.compile(r"QPushButton\(\s*(\"[^\"]*\"|'[^']*')\s*[,)]")


class _Bus:
    def __getattr__(self, _name):
        class _Signal:
            def emit(self, *a):
                pass

            def connect(self, *a):
                pass
        return _Signal()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _is_symbol_only(text: str) -> bool:
    return bool(text) and len(text) <= 3 and not text.isalnum()


# ── the helper ───────────────────────────────────────────────────────────────

def test_describe_control_sets_both_halves(qapp):
    button = describe_control(QPushButton("|<"), "First move",
                              "Jump to the start of the game")
    assert button.accessibleName() == "First move"
    assert "First move" in button.toolTip()
    assert "start of the game" in button.toolTip()
    assert button.accessibleDescription() == "Jump to the start of the game"


def test_describe_control_keeps_the_visible_glyph(qapp):
    """The symbol is the point — it must not be replaced by the words."""
    button = describe_control(QPushButton("⟳"), "Refresh")
    assert button.text() == "⟳"


def test_describe_control_returns_the_widget_for_chaining(qapp):
    button = QPushButton("✕")
    assert describe_control(button, "Close") is button


def test_a_name_without_a_hint_still_works(qapp):
    button = describe_control(QPushButton("‹"), "Previous page")
    assert button.accessibleName() == "Previous page"
    assert button.toolTip() == "Previous page"
    assert "—" not in button.toolTip(), "an empty hint left a dangling dash"


def test_an_empty_name_is_a_no_op(qapp):
    button = describe_control(QPushButton("x"), "")
    assert button.accessibleName() == ""
    assert button.toolTip() == ""


def test_a_widget_that_cannot_be_described_still_works():
    """A control that refuses these calls must not take a panel down with it."""
    class _Awkward:
        def setAccessibleName(self, _v):
            raise RuntimeError("no")

    result = describe_control(_Awkward(), "Something")
    assert result is not None


# ── the retrofit, proved at runtime rather than by reading source ───────────

def _symbol_buttons(widget: QWidget) -> list[QPushButton]:
    return [b for b in widget.findChildren(QPushButton)
            if _is_symbol_only(b.text())]


def _build(module_name: str, class_name: str, *args):
    import importlib
    module = importlib.import_module("orion_core.gui." + module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        pytest.skip("%s.%s not present" % (module_name, class_name))
    try:
        return cls(*args)
    except Exception as exc:                       # pragma: no cover
        pytest.skip("%s could not be constructed here: %s" % (class_name, exc))


#: Arguments are built lazily: a QWidget cannot exist before QApplication, and
#: parametrize values are evaluated at collection time.
BUILDABLE = [
    ("camera_preview", "_PreviewWindow", lambda: ("Preview",)),
    ("entrepreneur", "CurrencyConverter", lambda: (_Bus(),)),
    ("chess_view", "ChessPanel", lambda: (_Bus(),)),
    # UnifiedDashboard paginates real widgets; with no pages there is nothing to
    # page between, so the ‹ › buttons would never be built at all.
    ("unified_dashboard", "UnifiedDashboard",
     lambda: (_Bus(), [("ONE", QWidget()), ("TWO", QWidget())])),
]


@pytest.mark.parametrize("module_name,class_name,make_args", BUILDABLE)
def test_every_symbol_button_in_a_real_panel_has_a_spoken_name(
        qapp, module_name, class_name, make_args):
    """Source can be read optimistically; a constructed widget cannot."""
    widget = _build(module_name, class_name, *make_args())
    unnamed = [b.text() for b in _symbol_buttons(widget)
               if not b.accessibleName().strip()]
    assert not unnamed, (
        "%s renders symbol-only buttons a screen reader cannot describe: %s"
        % (class_name, unnamed))


def test_the_chess_navigation_buttons_are_named(qapp):
    """Four adjacent buttons reading |< < > >| were the worst offender."""
    panel = _build("chess_view", "ChessPanel", _Bus())
    names = {b.text(): b.accessibleName() for b in _symbol_buttons(panel)}
    for glyph in ("|<", "<", ">", ">|"):
        assert names.get(glyph), "chess button %r has no spoken name" % glyph
    assert len(set(names.values())) == len(names), (
        "two chess buttons share a name: %s" % names)


def test_named_buttons_also_carry_a_tooltip(qapp):
    """Both halves, or the retrofit only helped one kind of user."""
    panel = _build("chess_view", "ChessPanel", _Bus())
    for button in _symbol_buttons(panel):
        assert button.toolTip().strip(), (
            "%r has a spoken name but no tooltip" % button.text())


# ── the guard that stops this rotting back ──────────────────────────────────

def test_no_symbol_only_button_anywhere_lacks_a_spoken_name():
    """The whole deck, by source scan — the runtime tests above cannot reach
    every panel, and a new unlabelled button should fail immediately."""
    offenders = []
    for path in sorted(GUI.rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        for match in SYMBOL_BUTTON.finditer(src):
            text = match.group(1)[1:-1]
            if not _is_symbol_only(text):
                continue
            window = src[max(0, match.start() - 300):match.end() + 400]
            if "setAccessibleName" in window or "describe_control" in window:
                continue
            line = src[:match.start()].count("\n") + 1
            offenders.append("%s:%d %r" % (path.name, line, text))
    assert not offenders, (
        "these buttons show only a symbol and would be announced as that "
        "symbol — give them a name via widgets.describe_control or "
        "setAccessibleName: " + "; ".join(offenders))


def test_the_audit_still_finds_the_symbol_buttons():
    """Guards the guard: if the pattern stops matching, the test above passes
    vacuously and the protection is gone."""
    found = 0
    for path in sorted(GUI.rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        found += sum(1 for m in SYMBOL_BUTTON.finditer(src)
                     if _is_symbol_only(m.group(1)[1:-1]))
    assert found >= 15, "the scan found only %d symbol buttons — has it broken?" % found


# ── promises the interface makes ────────────────────────────────────────────

SHORTCUT = re.compile(
    r"\b((?:Ctrl|Alt|Shift|Meta)(?:\+(?:Ctrl|Alt|Shift|Meta))*\+[A-Za-z0-9]|F\d{1,2})\b")


def test_every_shortcut_the_ui_advertises_is_actually_registered():
    """Tooltips promise "(Ctrl+K)", "(Ctrl+M)", "(F11)". A promised shortcut
    that was never bound is a misleading label, and the user finds out by
    pressing it and having nothing happen."""
    advertised: dict[str, str] = {}
    registered: set[str] = set()
    for path in sorted(GUI.rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"(?:setToolTip|setText|addAction)\(([^)]*)\)", src):
            for combo in SHORTCUT.findall(match.group(1)):
                advertised.setdefault(combo.replace(" ", "").lower(),
                                      "%s:%d" % (path.name,
                                                 src[:match.start()].count("\n") + 1))
        for match in re.finditer(
                r"QKeySequence\(\s*[\"']([^\"']+)[\"']\s*\)|setShortcut\(\s*[\"']([^\"']+)[\"']",
                src):
            registered.add((match.group(1) or match.group(2)).replace(" ", "").lower())
    missing = {k: v for k, v in advertised.items() if k not in registered}
    assert not missing, (
        "the interface promises shortcuts that are not bound anywhere: "
        + "; ".join("%s (%s)" % (k, v) for k, v in sorted(missing.items())))


def test_some_shortcuts_are_actually_advertised():
    """Guards the guard again — a regex that matches nothing proves nothing."""
    count = 0
    for path in sorted(GUI.rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"setToolTip\(([^)]*)\)", src):
            count += len(SHORTCUT.findall(match.group(1)))
    assert count >= 5, "only %d advertised shortcuts found" % count


# ── two controls that look identical must not act differently ───────────────

def _labelled(widget: QWidget) -> list[QPushButton]:
    return [b for b in widget.findChildren(QPushButton) if b.text().strip()]


def test_buttons_sharing_a_label_in_one_panel_are_distinguishable(qapp):
    """The Development deck carried two "Start" and two "Stop" buttons: one
    pair drove the debugger, the other drove Docker containers. Visually they
    sit in different sections, but a keyboard user tabbing through heard
    "Start, Start, Stop, Stop" — and one of those stops a container.

    Same visible word is fine. Same visible word AND no way to tell them apart
    is not.
    """
    deck = _build("development_deck", "DevelopmentDeckView", _Bus())
    by_text: dict[str, list[str]] = {}
    for button in _labelled(deck):
        by_text.setdefault(button.text().strip(), []).append(
            button.accessibleName().strip())
    ambiguous = {
        text: names for text, names in by_text.items()
        if len(names) > 1 and len({n for n in names if n}) < len(names)
    }
    assert not ambiguous, (
        "these buttons share a label with nothing to tell them apart: "
        + "; ".join("%r x%d" % (t, len(n)) for t, n in ambiguous.items()))


def test_the_debugger_and_docker_controls_are_named_apart(qapp):
    deck = _build("development_deck", "DevelopmentDeckView", _Bus())
    names = {b.accessibleName() for b in _labelled(deck) if b.accessibleName()}
    for expected in ("Start debugger", "Stop debugger",
                     "Start container", "Stop container"):
        assert expected in names, "%r is missing from %s" % (expected, sorted(names))


# ── an action that failed must not report success ───────────────────────────

class _RecordingBus:
    def __init__(self):
        self.messages: list[str] = []

    def __getattr__(self, _name):
        outer = self

        class _Signal:
            def emit(self, *args):
                outer.messages.append(" ".join(str(a) for a in args))

            def connect(self, *a):
                pass
        return _Signal()


class _BrokenMemory:
    def remember(self, *_a, **_k):
        raise RuntimeError("the store is unavailable")

    def query(self, *_a, **_k):
        return []


class _WorkingMemory:
    def __init__(self):
        self.saved: list[tuple] = []

    def remember(self, *args, **_k):
        self.saved.append(args)
        return "ok"

    def query(self, *_a, **_k):
        return []


def test_a_failed_save_does_not_claim_the_idea_was_captured(qapp):
    """The Idea Capture panel swallowed a failing save, then added the idea to
    the list and logged "IDEA: captured" anyway — and it had already cleared the
    input box, so the only copy of what the user typed was gone.
    """
    from orion_core.gui.entrepreneur import IdeaCapture

    bus = _RecordingBus()
    panel = IdeaCapture(bus, _BrokenMemory())
    panel.input.setText("a genuinely good business idea")
    panel._save()

    assert not any("captured" in m.lower() for m in bus.messages), (
        "the panel reported a capture that never happened: %s" % bus.messages)
    assert any("could not save" in m.lower() for m in bus.messages), (
        "the failure was silent: %s" % bus.messages)
    assert panel.input.text() == "a genuinely good business idea", (
        "the input was cleared, so the user's text was lost on failure")
    assert "NOT SAVED" in panel.list.item(0).text()


def test_a_successful_save_still_behaves_normally(qapp):
    from orion_core.gui.entrepreneur import IdeaCapture

    bus = _RecordingBus()
    memory = _WorkingMemory()
    panel = IdeaCapture(bus, memory)
    panel.input.setText("another good idea")
    panel._save()

    assert memory.saved, "the idea was never handed to memory"
    assert panel.input.text() == "", "the input should clear on success"
    assert any("captured" in m.lower() for m in bus.messages)
    assert "NOT SAVED" not in panel.list.item(0).text()


def test_the_failure_path_has_every_name_it_needs(qapp):
    """A NameError inside an error handler turns a handled failure into a
    crash — and only fires when something is already going wrong."""
    import orion_core.gui.entrepreneur as module
    assert hasattr(module, "first_line"), (
        "the failure branch calls first_line() but the module cannot resolve it")
