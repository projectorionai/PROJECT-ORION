"""
Clipboard intelligence — and the reasons it is allowed to exist.

A clipboard watcher is a privacy hazard wearing a convenience hat. People copy
passwords out of password managers, API keys out of dashboards and card numbers
off statements, and a panel that cheerfully offered to send any of those to a
cloud model would be the worst thing in this codebase.

The convenience is real — copy a paragraph of German or a stack trace and what
you want next is nearly always translate, summarise, explain or fix — so the
feature earns its place only if every one of these holds:

  * nothing is ever sent without a press;
  * credentials are never offered at all, and are skipped SILENTLY, because
    "I noticed your password" is its own kind of alarming;
  * it is off until switched on.

Those are the tests. The rest is politeness: short text ignored, repeats
ignored, dismisses itself.

Offline: no clipboard, no model, no network. The panel is exercised separately
in a subprocess because a QApplication here breaks later Qt tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.gui.clipboard_panel import (
    ACTIONS,
    MIN_LENGTH,
    REPEAT_SECONDS,
    ClipboardWatcher,
)

PROSE = ("Der Vertrag laeuft zum Ende des Quartals aus und muss vor dem "
         "dreissigsten neu verhandelt werden.")
PASSWORD = "password = hunter2horses, and some padding so it clears the gate"
API_KEY = "sk-" + "abc123def456ghi789jkl012mno345pqrs plus padding to clear the gate"
CARD = "my card is 4539 1488 0343 6467, remember it for the booking tomorrow"


class _Panel:
    """Records what it was asked to show, and sends nothing."""

    def __init__(self) -> None:
        self.offered: list[str] = []

    def offer(self, text: str, at: object = None) -> None:
        self.offered.append(text)


@pytest.fixture
def watcher():
    panel = _Panel()
    clock = {"t": 1000.0}
    watch = ClipboardWatcher(panel, clock=lambda: clock["t"], enabled=True)
    watch.panel_stub = panel
    watch.clock = clock
    return watch


# ── the privacy gates ────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,text", [
    ("a password assignment", PASSWORD),
    ("an API key", API_KEY),
    ("a payment card", CARD),
])
def test_credentials_are_never_offered(watcher, label, text):
    """THE gate. Long enough, novel, and still refused."""
    assert len(text) >= MIN_LENGTH, "the fixture must clear the length gate"
    assert watcher.should_offer(text) is False, f"{label} was offered"
    assert watcher.consider(text) is False
    assert watcher.panel_stub.offered == [], f"{label} reached the panel"


def test_a_credential_is_skipped_silently(watcher):
    """Not warned about. A panel announcing "I noticed your password" tells the
    user their clipboard is being read, in the most alarming way available."""
    watcher.consider(PASSWORD)
    assert watcher.panel_stub.offered == []


def test_an_unusable_guard_refuses_rather_than_allows(watcher):
    """If the recogniser cannot answer, the safe reading is "not safe"."""
    class _Broken:
        def is_safe(self, _text):
            raise RuntimeError("guard unavailable")

    watcher._guard = _Broken()
    assert watcher.should_offer(PROSE) is False


def test_a_missing_guard_refuses_too():
    panel = _Panel()
    watch = ClipboardWatcher(panel, guard=None, enabled=True)
    watch._guard = None
    assert watch.should_offer(PROSE) is False, (
        "with no recogniser at all it must not offer anything"
    )


# ── consent ──────────────────────────────────────────────────────────────────

def test_it_is_off_until_switched_on():
    """A watcher nobody asked for is a watcher."""
    panel = _Panel()
    assert ClipboardWatcher(panel).should_offer(PROSE) is False


def test_nothing_is_sent_merely_by_appearing(watcher):
    """The panel holds the text; only a press releases it."""
    assert watcher.consider(PROSE) is True
    assert watcher.panel_stub.offered == [PROSE.strip()]
    # Showing is the whole of what happened — no instruction was produced.
    assert not hasattr(watcher, "sent")


# ── politeness ───────────────────────────────────────────────────────────────

def test_short_captures_are_ignored(watcher):
    """A copied filename or number needs no help; offering is noise."""
    assert watcher.should_offer("report_v2.docx") is False
    assert watcher.should_offer("42") is False
    assert watcher.should_offer("") is False


def test_a_repeat_does_not_ask_twice(watcher):
    assert watcher.consider(PROSE) is True
    assert watcher.consider(PROSE) is False


def test_the_same_text_may_be_offered_again_much_later(watcher):
    assert watcher.consider(PROSE) is True
    watcher.clock["t"] += REPEAT_SECONDS + 1
    assert watcher.consider(PROSE) is True


def test_a_different_capture_is_offered_immediately(watcher):
    assert watcher.consider(PROSE) is True
    other = "A completely different paragraph, also long enough to be worth it."
    assert watcher.consider(other) is True


# ── the offers themselves ────────────────────────────────────────────────────

def test_the_four_actions_are_the_ones_people_want():
    labels = [label for label, _key, _instruction in ACTIONS]
    assert labels == ["Translate", "Summarise", "Explain", "Fix"]


def test_every_action_carries_a_real_instruction():
    for label, key, instruction in ACTIONS:
        assert key and instruction
        assert len(instruction) > 25, f"{label} instruction is too vague"


def test_the_fix_action_asks_for_the_text_and_nothing_else():
    """Otherwise it comes back as a critique instead of a correction."""
    instruction = next(i for label, _k, i in ACTIONS if label == "Fix")
    assert "nothing else" in instruction.lower()


def test_translate_targets_the_users_own_language():
    """ORION learns which language that is; the instruction must defer to it
    rather than naming one."""
    instruction = next(i for label, _k, i in ACTIONS if label == "Translate")
    assert "my language" in instruction.lower()


def test_attach_survives_having_no_clipboard():
    """Headless, or a platform with no clipboard, must not raise."""
    watch = ClipboardWatcher(_Panel(), enabled=True)

    class _NoSignal:
        def text(self):
            return ""

    assert watch.attach(_NoSignal()) is False


# ── wiring: a panel nothing opens is no panel at all ─────────────────────────

def _core_window_source() -> str:
    return (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")


def test_the_watcher_is_reachable():
    source = _core_window_source()
    assert "def enable_clipboard_intelligence" in source
    assert '"clipboard", "clipboard_on", "watch_clipboard"' in source, (
        "ORION cannot turn his own clipboard suggestions on"
    )
    assert '"clipboard_off", "stop_clipboard"' in source, (
        "and cannot turn them off again, which matters more"
    )


def test_pressing_a_button_is_what_sends():
    """The text reaches submit_text only from the panel's signal, never from
    the watcher noticing a copy."""
    source = _core_window_source()
    assert "_on_clipboard_request" in source
    start = source.index("def _on_clipboard_request")
    body = source[start:start + 900]
    assert "submit_text" in body, "the pressed action never reaches ORION"

    enable_start = source.index("def enable_clipboard_intelligence")
    enable_body = source[enable_start:source.index("def _on_clipboard_request")]
    assert "submit_text" not in enable_body, (
        "enabling the watcher must not be able to send anything by itself"
    )


def test_it_starts_off():
    """Switched on deliberately, never inherited."""
    source = _core_window_source()
    start = source.index("def enable_clipboard_intelligence")
    signature = source[start:start + 120]
    assert "on: bool = True" in signature
    # Nothing in the window's CONSTRUCTOR turns it on. Deliberately scoped to
    # __init__ rather than "everything above": the voice route quite correctly
    # calls it with True, and an earlier version of this test swept that in and
    # called working code a bug.
    start = source.index("    def __init__(")
    end = source.index("\n    def ", start + 10)
    constructor = source[start:end]
    assert "enable_clipboard_intelligence" not in constructor, (
        "the watcher is switched on while the window is being built"
    )
