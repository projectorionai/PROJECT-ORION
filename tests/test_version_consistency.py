"""One version, claimed in one way, wherever it is claimed.

The README said 25.0.0 and the capability inventory said "Mark XXV" while
`orion_core/__init__.py` said 30.0.0 and `constants.APP_MARK` said Mark XXX.
Nothing was wrong with the code; the documents had simply been left behind, and
a reader has no way to tell which of two confident statements is the stale one.

This is the kind of drift that is trivial to fix and impossible to notice, so
it is checked rather than remembered.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orion_core  # noqa: E402
from orion_core.constants import APP_MARK, APP_MARK_NUMBER  # noqa: E402

#: Roman numerals, for the marks this project will plausibly reach.
ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
         (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
         (5, "V"), (4, "IV"), (1, "I")]


def _roman(number: int) -> str:
    out = []
    for value, glyph in ROMAN:
        while number >= value:
            out.append(glyph)
            number -= value
    return "".join(out)


def test_the_mark_matches_its_number():
    """APP_MARK is written by hand and APP_MARK_NUMBER is used in comparisons.
    They have disagreed before."""
    assert APP_MARK == f"Mark {_roman(APP_MARK_NUMBER)}"


def test_the_package_version_matches_the_mark():
    """The major version IS the mark number. A release that bumps one and not
    the other leaves two different answers to "which ORION is this"."""
    major = int(orion_core.__version__.split(".")[0])
    assert major == APP_MARK_NUMBER, (
        f"__version__ is {orion_core.__version__} but this is "
        f"{APP_MARK} (mark {APP_MARK_NUMBER})")


def test_the_readme_states_the_current_version():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert orion_core.__version__ in text, (
        f"the README does not mention {orion_core.__version__}")
    assert APP_MARK in text, f"the README does not mention {APP_MARK}"


def test_the_readme_does_not_still_claim_an_old_one():
    """Adding the new version while leaving the old one is worse than either:
    the reader now has two confident statements and no way to choose."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    stale = [m for m in re.findall(r"package version:? \*{0,2}(\d+\.\d+\.\d+)",
                                   text, flags=re.I)
             if m != orion_core.__version__]
    assert not stale, f"the README still claims version(s) {stale}"


def test_a_dated_inventory_says_which_version_it_describes():
    """It is allowed to be out of date — it is a dated snapshot. It is not
    allowed to look current while being stale."""
    path = ROOT / "docs" / "CAPABILITIES_2026-09-18.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    assert "Written against" in text and "Current source" in text, (
        "the dated inventory does not distinguish what it describes from "
        "what the source now is")
    assert orion_core.__version__ in text
