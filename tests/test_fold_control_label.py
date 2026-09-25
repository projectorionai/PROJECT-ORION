"""
Tests for fold_control_label (Mark XXI, Track D1) — UI control label
normalisation for element-click matching: mnemonic ampersands ('&Save')
and the ellipsis "opens a dialog" convention ('Save As…' / 'Save As...')
both defeated exact/substring matching before this. Deliberately a
separate function from fold_title (window-title matching keeps its exact
existing behaviour) — a couple of cross-checks below confirm that.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.utils import fold_control_label, fold_title


# ── basic folding (inherited from fold_title) ───────────────────────────────

def test_lowercases():
    assert fold_control_label("SAVE") == "save"


def test_collapses_whitespace():
    assert fold_control_label("Save   As") == "save as"


def test_strips_invisible_unicode():
    assert fold_control_label("Save​As") == "saveas"


def test_blank_input_folds_to_blank():
    assert fold_control_label("") == ""
    assert fold_control_label(None) == ""


# ── mnemonic ampersands ──────────────────────────────────────────────────────

def test_strips_a_leading_mnemonic_ampersand():
    assert fold_control_label("&Save") == "save"


def test_strips_a_mnemonic_ampersand_mid_label():
    assert fold_control_label("Save &As") == "save as"


def test_double_ampersand_becomes_a_literal_ampersand():
    assert fold_control_label("Ben && Jerry's") == "ben & jerry's"


def test_mnemonic_and_query_without_one_match():
    assert fold_control_label("&Save") == fold_control_label("Save")


# ── ellipsis convention ──────────────────────────────────────────────────────

def test_unicode_ellipsis_is_stripped():
    assert fold_control_label("Save As…") == "save as"


def test_three_dots_are_also_stripped():
    assert fold_control_label("Save As...") == "save as"


def test_plain_label_with_no_ellipsis_at_all_matches_the_same_way():
    assert fold_control_label("Save As") == "save as"
    assert fold_control_label("Save As") == fold_control_label("Save As...")
    assert fold_control_label("Save As") == fold_control_label("Save As…")


def test_ellipsis_only_stripped_at_the_end_not_mid_label():
    # "..." mid-label is meaningful punctuation (e.g. a literal range),
    # not the dialog-opens convention, which is always a trailing marker.
    assert fold_control_label("Page 1...5 of 10") == "page 1...5 of 10"


# ── combined real-world labels ───────────────────────────────────────────────

def test_mnemonic_plus_ellipsis_combined():
    assert fold_control_label("Save &As...") == "save as"


def test_matches_a_bare_query_against_a_decorated_label():
    query = fold_control_label("save as")
    label = fold_control_label("&Save As…")
    assert query == label


# ── fold_title is unaffected (window-title matching keeps its behaviour) ────

def test_fold_title_does_not_strip_ampersands():
    assert fold_title("&Save") == "&save"


def test_fold_title_does_not_strip_ellipsis():
    assert fold_title("Save As...") == "save as..."
    assert fold_title("Save As…") == "save as…"
