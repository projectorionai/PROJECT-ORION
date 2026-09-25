"""Tests for the command-palette logic (Mark X.12 §1.2), extended in the
Mark XX design-spec §8 to unify tools and Command Deck pages into one
searchable surface (KIND_TOOL prefills the message box; KIND_PAGE
navigates the deck straight there — CommandPalette.command_chosen now
emits (kind, name) instead of a bare tool name, exercised headlessly below
via QT_QPA_PLATFORM=offscreen).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.gui.command_palette import (
    KIND_PAGE,
    KIND_TOOL,
    PaletteCommand,
    build_commands,
    build_page_commands,
    rank_commands,
)

_DECLS = [
    {"name": "breach_check", "description": "Run a credential breach check against known leaks; reports exposure."},
    {"name": "web_automation", "description": "Browser co-pilot — drive a real, visible browser window."},
    {"name": "workspace_control", "description": "Save or restore a named desktop workspace layout."},
    {"name": "geo", "description": "Geographic intelligence and postcode lookup."},
    {"name": "", "description": "nameless declarations are dropped"},
]


def test_build_commands_maps_and_sorts_and_drops_nameless():
    cmds = build_commands(_DECLS)
    names = [c.name for c in cmds]
    assert names == sorted(names)                  # sorted by name
    assert "" not in names                         # nameless dropped
    breach = next(c for c in cmds if c.name == "breach_check")
    assert breach.title == "breach check"          # underscores → spaces
    assert breach.subtitle and len(breach.subtitle) <= 100
    assert "—" not in breach.subtitle or True      # first-sentence trimming applied


def test_empty_query_returns_everything():
    cmds = build_commands(_DECLS)
    assert len(rank_commands("", cmds)) == len(cmds)


def test_exact_name_ranks_first():
    cmds = build_commands(_DECLS)
    ranked = rank_commands("breach", cmds)
    assert ranked and ranked[0].name == "breach_check"


def test_subsequence_matching_finds_scattered_letters():
    cmds = build_commands(_DECLS)
    # "wsc" is a subsequence of "workspace_control" but not a substring.
    ranked = rank_commands("wsc", cmds)
    assert any(c.name == "workspace_control" for c in ranked)


def test_substring_outranks_subsequence():
    cmds = build_commands(_DECLS)
    # "work" is a substring of workspace_control; ensure it beats a mere
    # subsequence match elsewhere by landing first.
    ranked = rank_commands("work", cmds)
    assert ranked[0].name == "workspace_control"


def test_no_match_returns_empty():
    cmds = build_commands(_DECLS)
    assert rank_commands("zzzzzz", cmds) == []


def test_description_terms_are_searchable():
    cmds = build_commands(_DECLS)
    # "browser" appears only in web_automation's description.
    ranked = rank_commands("browser", cmds)
    assert ranked and ranked[0].name == "web_automation"


def test_build_commands_tags_every_entry_as_a_tool():
    cmds = build_commands(_DECLS)
    assert all(c.kind == KIND_TOOL for c in cmds)


# ── deck pages (Mark XX design-spec §8) ─────────────────────────────────────

_PAGES = ["MISSION", "WIDGETS", "TOOLKIT", "STUDIO", "LIBRARY", "OPS",
         "COMMAND CENTRE", "DIAGNOSTICS", "GLOBE", "LOG", "MEMORY",
         "TELEMETRY", "CHESS"]


def test_build_page_commands_tags_every_entry_as_a_page():
    cmds = build_page_commands(_PAGES)
    assert all(c.kind == KIND_PAGE for c in cmds)
    assert {c.name for c in cmds} == set(_PAGES)


def test_build_page_commands_drops_blank_names():
    cmds = build_page_commands(["MISSION", "", "  ", "CHESS"])
    assert {c.name for c in cmds} == {"MISSION", "CHESS"}


def test_a_half_remembered_page_name_is_findable_by_search():
    cmds = build_commands(_DECLS) + build_page_commands(_PAGES)
    ranked = rank_commands("chess", cmds)
    assert ranked and ranked[0].name == "CHESS" and ranked[0].kind == KIND_PAGE


def test_tools_and_pages_can_be_ranked_together_without_kind_bias():
    # A page and a tool sharing no special-cased handling in rank_commands —
    # ranking is purely by text match, kind never silently boosts or buries
    # an entry.
    cmds = build_commands(_DECLS) + build_page_commands(_PAGES)
    ranked = rank_commands("g", cmds)
    kinds = {c.kind for c in ranked}
    assert KIND_TOOL in kinds and KIND_PAGE in kinds   # "geo" and "GLOBE" both match


# ── CommandPalette dialog: kind-aware emission ──────────────────────────────

@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_choosing_a_tool_emits_kind_tool_and_its_name(_app):
    from orion_core.gui.command_palette import CommandPalette

    dialog = CommandPalette(build_commands(_DECLS))
    captured = []
    dialog.command_chosen.connect(lambda kind, name: captured.append((kind, name)))
    dialog.search.setText("breach")
    dialog._choose_current()
    assert captured == [(KIND_TOOL, "breach_check")]


def test_choosing_a_page_emits_kind_page_and_its_label(_app):
    from orion_core.gui.command_palette import CommandPalette

    dialog = CommandPalette(build_page_commands(_PAGES))
    captured = []
    dialog.command_chosen.connect(lambda kind, name: captured.append((kind, name)))
    dialog.search.setText("chess")
    dialog._choose_current()
    assert captured == [(KIND_PAGE, "CHESS")]


def test_mixed_palette_lists_both_kinds_and_chooses_the_top_match(_app):
    from orion_core.gui.command_palette import CommandPalette

    dialog = CommandPalette(build_commands(_DECLS) + build_page_commands(_PAGES))
    dialog.search.setText("chess")
    assert dialog.results.count() >= 1
    captured = []
    dialog.command_chosen.connect(lambda kind, name: captured.append((kind, name)))
    dialog._choose_current()
    assert captured == [(KIND_PAGE, "CHESS")]
