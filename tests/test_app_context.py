"""
Tests for app_context.py (Mark XX architectural-audit pass, Track I) — the
generalised "what does having this app open mean" classifier the audit
asked for (VS Code -> programming, Photoshop -> image editing, Chess.com ->
playing chess), with a real fallback instead of silence for anything
unrecognised.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.app_context import describe_context


def test_blank_title_returns_blank():
    assert describe_context("") == ""
    assert describe_context("   ") == ""


def test_vs_code_is_recognised_as_programming():
    assert describe_context("app.py - ORION Main - Visual Studio Code") == "programming"


def test_photoshop_is_recognised_as_editing_an_image():
    assert describe_context("Untitled-1 - Adobe Photoshop 2026") == "editing an image"


def test_chess_dot_com_is_recognised_as_playing_chess():
    assert describe_context("Play Chess vs Computer - Chess.com") == "playing chess"


def test_lichess_is_recognised_as_playing_chess():
    assert describe_context("lichess.org - Free Online Chess") == "playing chess"


def test_pycharm_is_recognised_as_programming():
    assert describe_context("main.py - myproject - PyCharm") == "programming"


def test_terminal_titles_are_recognised():
    assert describe_context("Windows Terminal") == "working in a terminal"
    assert describe_context("Administrator: Windows PowerShell") == "working in a terminal"


def test_office_apps_are_recognised():
    assert describe_context("Q3 Report - Excel") == "working with a spreadsheet"
    assert describe_context("Letter - Word") == "writing a document"
    assert describe_context("Deck - PowerPoint") == "building a presentation"


def test_browser_is_recognised_generically():
    assert describe_context("New Tab - Google Chrome") == "browsing the web"


def test_more_specific_site_beats_the_generic_browser_match():
    # A chess.com tab is still open in a Chrome window titled with "Chrome"
    # in it — the specific pattern must win over the generic browser one.
    assert describe_context("Chess.com - Play Chess Online for Free - Google Chrome") == "playing chess"


def test_github_is_more_specific_than_generic_browsing():
    assert describe_context("torvalds/linux - GitHub.com - Firefox") == "browsing a code repository"


def test_unrecognised_title_falls_back_to_using_the_app_name():
    result = describe_context("Untitled - Notepad")
    assert result == "using Notepad"


def test_unrecognised_title_with_no_separator_uses_the_whole_title():
    result = describe_context("SomeRandomApp")
    assert result == "using SomeRandomApp"


def test_fallback_never_raises_on_a_very_long_title():
    long_title = "A" * 500
    result = describe_context(long_title)
    assert result.startswith("using ")
    assert len(result) < 100   # truncated, not dumped verbatim
