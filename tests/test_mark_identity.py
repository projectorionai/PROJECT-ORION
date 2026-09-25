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
MARK = 32
ROMAN = "XXXII"


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
            if re.search(r"MARK\s+[IVXLC]{2,}", body):
                offenders.append(f"{path.relative_to(ROOT)}:{line} {body[:60]}")
    assert offenders == [], (
        "these captions state a mark instead of deriving it: " +
        "; ".join(offenders))


def test_the_swarm_map_derives_its_caption():
    """Named specifically because it is the one that drifted."""
    source = (ROOT / "orion_core" / "gui" / "swarm_view.py").read_text(
        encoding="utf-8")
    assert "APP_MARK" in source


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
