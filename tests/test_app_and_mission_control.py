"""
Two everyday frustrations.

  "ORION wants me to say the exact application name and that's very annoying."
  "I want ORION to be able to get rid of things inside of MISSION without
   restarting."
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from orion_core.app_resolver import (
    ALIASES, AppCatalogue, Candidate, Match, normalise, _score,
)
from orion_core.missions import MissionEngine


# ── the resolver reads ordinary speech ───────────────────────────────────────

@pytest.mark.parametrize("request_text,expected", [
    ("Word", "word"),
    ("Microsoft Word.exe", "microsoft word"),
    ("please open up Microsoft Word for me", "microsoft word"),
    ("open the calculator app", "calculator"),
    ("launch vs code", "vs code"),
    ("start Google Chrome please", "google chrome"),
    ("run notepad.exe", "notepad"),
])
def test_normalise_strips_the_phrasing(request_text, expected):
    assert normalise(request_text) == expected


def test_normalise_never_returns_empty():
    """'open it' is all filler — the caller still needs something to report."""
    assert normalise("open it up please") != ""


def test_normalise_handles_nonsense_safely():
    assert isinstance(normalise(""), str)
    assert isinstance(normalise(None), str)


# ── scoring ──────────────────────────────────────────────────────────────────

def test_exact_beats_prefix_beats_substring():
    exact, _ = _score("word", "word")
    prefix, _ = _score("word", "wordpad")
    substring, _ = _score("word", "microsoft word 2019")
    assert exact > prefix
    assert exact > substring


def test_a_shorter_prefix_match_wins():
    """'word' should prefer 'word' over 'wordpad-something-long'."""
    near, _ = _score("word", "wordpad")
    far, _ = _score("word", "wordperfect professional")
    assert near > far


def test_a_typo_still_matches():
    score, why = _score("powerpint", "powerpnt")
    assert score > 0
    assert "spelling" in why


def test_unrelated_names_score_zero():
    score, _ = _score("word", "steam")
    assert score == 0.0


# ── the catalogue against a controlled set ───────────────────────────────────

def _catalogue(*candidates: Candidate) -> AppCatalogue:
    cat = AppCatalogue()
    cat.candidates = list(candidates)
    cat.built = True
    return cat


def test_an_alias_beats_a_fuzzy_lookalike():
    """'word' must be WINWORD, never WordPad."""
    cat = _catalogue(
        Candidate("winword", r"C:\Office\WINWORD.EXE", "app_paths"),
        Candidate("wordpad", r"C:\Windows\wordpad.exe", "app_paths"),
    )
    match = cat.resolve("word")
    assert match.ok
    assert match.candidate.name == "winword"
    assert match.score == 1.0


def test_the_registry_outranks_a_start_menu_guess():
    cat = _catalogue(
        Candidate("spotify", r"C:\spotify.exe", "app_paths"),
        Candidate("spotify", r"C:\shortcut.lnk", "start_menu"),
    )
    assert cat.resolve("spotify").candidate.source == "app_paths"


def test_filler_words_do_not_prevent_a_match():
    cat = _catalogue(Candidate("discord", r"C:\Discord.lnk", "start_menu"))
    match = cat.resolve("hey can you please open up discord for me")
    assert match.ok and match.confident
    assert match.candidate.name == "discord"


def test_an_unknown_app_is_reported_honestly_with_suggestions():
    cat = _catalogue(Candidate("discord", "d", "start_menu"),
                     Candidate("steam", "s", "start_menu"))
    match = cat.resolve("photoshoop")
    assert match.ok is False or match.confident is False
    assert "photoshoop" in match.why or match.alternatives


def test_nothing_named_is_not_a_match():
    assert _catalogue().resolve("").ok is False


def test_every_alias_target_is_plausible():
    for spoken, target in ALIASES.items():
        assert target, spoken
        assert target == target.lower() or target.endswith(":"), spoken


# ── against the real machine ─────────────────────────────────────────────────

@pytest.mark.parametrize("phrase", ["word", "notepad", "calculator"])
def test_real_machine_resolves_common_apps(phrase):
    """The actual complaint, against the actual installed software."""
    cat = AppCatalogue().build()
    match = cat.resolve(phrase)
    assert match.ok and match.confident, f"{phrase!r} -> {match.why}"


def test_the_catalogue_finds_something():
    cat = AppCatalogue().build()
    assert len(cat.candidates) > 10


# ── open_app uses it, and says what it opened ────────────────────────────────

def test_open_app_uses_the_resolver():
    import inspect

    from orion_core.agents import DesktopAgent

    source = inspect.getsource(DesktopAgent.open_app)
    assert "resolve_app" in source
    assert "matched" in source, (
        "it must say WHAT it opened, so a wrong guess is visible immediately")


def test_the_persona_forbids_narrating_an_action_without_doing_it():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "orion_core" / "providers.py"
              ).read_text(encoding="utf-8")
    assert "NEVER describe an action as done" in source


# ── mission removal ──────────────────────────────────────────────────────────

@pytest.fixture
def board():
    path = Path(tempfile.mkdtemp()) / "missions.json"
    engine = MissionEngine(path=path)
    engine.create("Test Mission", "a trial")
    engine.add_task("Test Mission", "buy milk")
    engine.add_task("Test Mission", "write report")
    engine.add_goal("Test Mission", "ship it")
    return engine


def test_an_item_can_be_removed(board):
    result = board.remove_item("Test", "write report")
    assert result.ok
    assert "write report" in result.text
    tasks = board.load()["missions"]["Test Mission"]["tasks"]
    assert all("write report" not in t["task"] for t in tasks)


def test_removing_something_absent_says_so(board):
    result = board.remove_item("Test", "no such thing")
    assert result.ok is False
    assert "no such thing" in result.text


def test_a_whole_mission_can_be_deleted(board):
    result = board.delete("Test")
    assert result.ok
    assert "2 task" in result.text, "it must say what went with it"
    assert "Test Mission" not in board.load()["missions"]


def test_deleting_the_current_mission_repoints_it(board):
    board.set_current("Test Mission")
    board.delete("Test")
    current = board.load().get("current")
    assert current != "Test Mission"
    assert current in board.load()["missions"] or current == ""


def test_archive_keeps_the_record(board):
    assert board.archive("Test").ok
    record = board.load()["missions"]["Test Mission"]
    assert record["status"] == "archived"
    assert len(record["tasks"]) == 2, "archiving must not destroy anything"


def test_clear_completed_sweeps_only_finished_work(board):
    board.complete("Test", "buy milk")
    result = board.clear_completed("Test")
    assert "1 completed" in result.text
    tasks = board.load()["missions"]["Test Mission"]["tasks"]
    assert len(tasks) == 1 and tasks[0]["task"] == "write report"


def test_clear_completed_across_the_whole_board(board):
    board.complete("Test", "buy milk")
    assert board.clear_completed().ok


def test_removal_persists_immediately_without_a_restart(board):
    """A second engine over the same file must see the change at once."""
    board.remove_item("Test", "buy milk")
    fresh = MissionEngine(path=board.path)
    tasks = fresh.load()["missions"]["Test Mission"]["tasks"]
    assert all("buy milk" not in t["task"] for t in tasks)


def test_removal_emits_so_the_deck_re_renders():
    class Sig:
        def __init__(self):
            self.messages = []

        def emit(self, *a):
            self.messages.append(a)

        def connect(self, *a):
            pass

    class Bus:
        def __getattr__(self, name):
            sig = Sig()
            object.__setattr__(self, name, sig)
            return sig

    bus = Bus()
    engine = MissionEngine(bus=bus, path=Path(tempfile.mkdtemp()) / "m.json")
    engine.create("Doomed")
    engine.delete("Doomed")
    channels = [m[0] for m in bus.dashboard_event.messages]
    assert "mission" in channels, "the deck must be told, or it shows stale data"


# ── the tool surface ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("action", [
    "remove", "delete", "archive", "clear_completed", "remove_item",
])
def test_the_tool_accepts_every_removal_action(board, action):
    result = asyncio.run(board.handle(
        {"action": action, "mission": "Test", "text": "buy milk"}))
    assert isinstance(result.text, str) and result.text


def test_remove_without_text_deletes_the_mission(board):
    result = asyncio.run(board.handle({"action": "remove", "mission": "Test"}))
    assert "Deleted" in result.text


def test_remove_with_text_deletes_only_the_item(board):
    result = asyncio.run(board.handle(
        {"action": "remove", "mission": "Test", "text": "buy milk"}))
    assert "Removed" in result.text
    assert "Test Mission" in board.load()["missions"], "the mission must survive"


def test_the_schema_advertises_removal():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    entry = next(d for d in TOOL_DECLARATIONS if d["name"] == "mission")
    assert "remove" in entry["description"]
    assert "no restart" in entry["description"]
