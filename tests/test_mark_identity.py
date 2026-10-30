"""
The running application identity must be consistent, current, and derived.

The mark is user-facing — it shows in the window title, the header build line,
the Command Deck's title, the swarm map's caption and the standalone build
stamp — so a bump has to be deliberate and complete rather than half-done.

It had been half-done before. The shell said Mark XXV while the swarm map
still said MARK XXI, because each had its own hard-coded string. So the
version now lives in exactly one constant and everything else derives from it,
and these tests check the derivation as much as the number: pinning the number
alone would let the same drift happen again at the next bump.

Historical references in docstrings — "added in Mark XXI, Track E3" — are
deliberately NOT swept. They record when something arrived, and rewriting them
to match the current version would falsify the provenance of every subsystem
in the codebase.
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orion_core  # noqa: E402
from orion_core.constants import (  # noqa: E402
    APP_BUILD,
    APP_MARK,
    APP_MARK_NUMBER,
    APP_NAME,
)

#: The mark this release is. One number, changed in one place.
MARK = 41
ROMAN = "XXXXI"    # the owner's numeral for forty-one


def test_the_mark_is_the_one_this_release_is():
    assert APP_MARK_NUMBER == MARK
    assert APP_MARK == f"Mark {ROMAN}"


def test_the_application_name_derives_from_the_mark():
    """Derived, not restated — a second copy is a second thing to forget."""
    assert APP_NAME == f"O.R.I.O.N. {APP_MARK}"


def test_the_semantic_version_agrees():
    assert orion_core.__version__.startswith(f"{MARK}.")


def test_the_codename_carries_the_mark():
    assert APP_MARK in orion_core.__codename__


def test_the_build_stamp_carries_the_mark():
    """The header shows this so a running instance can be told from a stale
    one at a glance."""
    assert APP_BUILD.lower().endswith(ROMAN.lower())


def test_the_package_docstring_is_current():
    assert APP_MARK in (orion_core.__doc__ or "")


# ── nothing may hard-code a version ───────────────────────────────────────────

def _code_strings(path: Path) -> list[tuple[int, str]]:
    """String literals in code, excluding docstrings.

    Docstrings are prose and carry historical provenance; a string literal in
    code is something that gets shown.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except Exception:
        return []
    out = []
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        body = token.string.strip()
        if body.startswith('"""') or body.startswith("'''"):
            continue                       # a docstring: prose, not a label
        out.append((token.start[0], body))
    return out


def test_no_ui_caption_hard_codes_a_mark():
    """The regression that made this file necessary.

    The swarm map read "SWARM — MARK XXI NEURAL SUBSYSTEM MAP" for four
    releases, because it was a literal nobody thought to grep for.

    Scoped to the SHOUTY form — an upper-case "MARK <numerals>" is how ORION
    writes a HUD caption, and a caption is a claim about what you are looking
    at right now. Mixed-case "Mark IX" in a changelog entry or a prompt is
    prose about the past and is left alone; a broader rule flagged those and
    would have pushed someone to falsify them.
    """
    offenders = []
    for path in (ROOT / "orion_core").rglob("*.py"):
        if "__pycache__" in path.parts or path.name == "constants.py":
            continue
        for line, body in _code_strings(path):
            if re.search(r"\bMARK\s+[IVXLC]{2,}\b", body):
                offenders.append(f"{path.relative_to(ROOT)}:{line} {body[:60]}")
    assert offenders == [], (
        "these captions state a mark instead of deriving it: " +
        "; ".join(offenders))


def test_the_swarm_map_caption_states_no_mark():
    """Named specifically because it is the one that drifted. Since Mark
    XXXVIII its caption is just "Swarm", so there is no mark left to drift."""
    source = (ROOT / "orion_core" / "gui" / "swarm_view.py").read_text(
        encoding="utf-8")
    assert 'QLabel("Swarm")' in source and "APP_MARK" not in source


def test_the_command_deck_shows_no_mark():
    """The Command Deck carries no Mark at all (owner, 28 Oct 2026).

    Deliberately changed: this test used to pin the deck's "ORION · MARK XXXX ·
    40" tag. The owner asked for the tag to go - the core window's header
    is where the Mark is shown - so no label, and no window title, on the
    deck names one. Nothing left there can drift at the next bump.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget

    from orion_core.bus import OrionBus
    from orion_core.gui.unified_dashboard import UnifiedDashboard

    app = QApplication.instance() or QApplication([])
    deck = UnifiedDashboard(OrionBus(), [("BRAIN", QWidget())])
    try:
        tags = [label for label in deck.findChildren(QLabel)
                if label.objectName() == "markTag"]
        assert tags == []
        texts = [label.text().upper() for label in deck.findChildren(QLabel)]
        texts.append(deck.windowTitle().upper())
        assert not any(APP_MARK.upper() in text or "MARK " in text for text in texts)
    finally:
        deck.deleteLater()
        app.processEvents()


def test_the_window_and_the_deck_share_one_name():
    for name in ("core_window.py", "command_centre.py"):
        source = (ROOT / "orion_core" / "gui" / name).read_text(encoding="utf-8")
        assert "APP_NAME" in source, f"{name} does not use the shared name"


def test_history_in_docstrings_is_left_alone():
    """A guard against a future sweep "tidying" the provenance away.

    "Mark XXI, Track E3" in a docstring says when that subsystem arrived. It
    is not a version claim and must not be rewritten to match this release.
    """
    found = 0
    for path in (ROOT / "orion_core").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found += len(re.findall(r"Mark [IVXLC]{2,}", text))
    assert found > 50, (
        "the historical references have been swept away; they record when "
        "each subsystem arrived and should stay")


def test_the_persona_says_his_elo_is_his_chess_rating():
    """Voice questions reach the Live model directly, not the reflex, so the
    model itself must know that 'your Elo' means chess and where the figures
    live — and must not invent them."""
    from orion_core.bus import OrionBus
    from orion_core.identity import IdentityManager

    text = IdentityManager(OrionBus()).persona_text()
    assert "YOUR ELO IS YOUR CHESS RATING" in text
    assert "action 'rating'" in text and "Never guess" in text


def test_who_made_you_is_answered_with_the_creators_name_everywhere(monkeypatch):
    """Owner's instruction (16 Oct 2026): "Who made you?" -> the creator's full
    name, on every channel - the model persona, the compact local capsule, the
    persona used when no IdentityManager is attached, and the offline brain.
    The name itself comes from this PC's config (a stand-in here)."""
    import asyncio

    from orion_core import identity
    from orion_core.bus import OrionBus
    from orion_core.identity import CREATOR_QUESTION_RE, CoreIdentity, IdentityManager
    from orion_core.local_brain import LocalBrain
    from orion_core.providers import ProviderRouter

    monkeypatch.setattr(identity, "CORE_IDENTITY", CoreIdentity(creator="Alex Morgan"))
    manager = IdentityManager(OrionBus())
    assert "CREATOR RULE" in manager.persona_text()
    assert "Alex Morgan" in manager.persona_text()
    assert "Alex Morgan" in manager.style_capsule()
    assert manager.describe()["creator"] == "Alex Morgan"

    router = ProviderRouter.__new__(ProviderRouter)
    router._identity = None
    assert "Alex Morgan" in router._persona_block()

    for asked in ("Who made you?", "hey orion, who created you", "who actually built you",
                  "Who's your creator?", "who is your developer", "Ryan who made you",
                  "where did you come from"):
        assert CREATOR_QUESTION_RE.search(asked.lower()), asked
    for other in ("who are you", "what made you say that", "who made your dinner",
                  "who is speaking", "make me a coffee"):
        assert not CREATOR_QUESTION_RE.search(other.lower()), other

    brain = LocalBrain(None, None, None, None)
    reply = asyncio.run(brain.respond("Hello, who made you?"))
    assert "Alex Morgan" in reply


def test_the_creators_name_lives_in_the_local_config_never_the_source(tmp_path, monkeypatch):
    """Owner, 28 Oct 2026: nothing personal may reach a public copy of the
    source. The name is read from config/identity.json; a copy with none set
    says it was made by the ORION project."""
    from orion_core import identity
    from orion_core.identity import CoreIdentity, UNNAMED_CREATOR

    monkeypatch.setattr(identity, "IDENTITY_PATH", tmp_path / "identity.json")
    assert CoreIdentity().creator == ""
    monkeypatch.setattr(identity, "CORE_IDENTITY", CoreIdentity())
    assert identity.creator_answer() == f"I was made by {UNNAMED_CREATOR}."
    assert UNNAMED_CREATOR in identity.origin_directive()
    (tmp_path / "identity.json").write_text('{"creator": "  Alex   Morgan ", "honorific": "sir"}',
                                            encoding="utf-8")
    assert CoreIdentity().creator == "Alex Morgan"
    (tmp_path / "identity.json").write_text("not json", encoding="utf-8")
    assert CoreIdentity().creator == ""


def test_he_knows_where_he_began(monkeypatch):
    """Mark XXXX.4: the first ORION on the owner's PC - v0.01, 19 June 2026."""
    from orion_core import identity
    from orion_core.identity import CoreIdentity
    monkeypatch.setattr(identity, "CORE_IDENTITY", CoreIdentity(creator="Alex Morgan"))
    origin = identity.origin_directive()
    assert "19 June 2026" in origin and "v0.01" in origin
    assert "11 June 2026" in origin
    # Owner, 28 Oct 2026: told his story, he credited the tools he was built
    # with; he must say his creator made and developed him.
    assert origin.count("Alex Morgan") == 2
    assert "made and developed you" in origin
    assert "built through" not in origin
    assert "never credit the AI tools or models" in origin
    assert "Mark XXXX" in origin and "Mark X" + "L" not in origin


def test_his_own_life_gets_a_friend_not_an_aide():
    """Owner, 30 Oct 2026: about a movie night with someone he likes ORION
    should say "Oh, you have a date?" - he had answered "As an AI assistant"."""
    from orion_core.bus import OrionBus
    from orion_core.identity import FRIEND_REGISTER, IdentityManager
    persona = IdentityManager(OrionBus()).persona_text()
    assert FRIEND_REGISTER in persona
    assert "Oh, you have a date?" in FRIEND_REGISTER and "Drop 'sir'" in FRIEND_REGISTER
    assert "Never deflect with 'as an AI'" in FRIEND_REGISTER
