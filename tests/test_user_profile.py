"""
UserProfile tests — publishable defaults, private overrides.

The point of the module is that a public clone works with neutral names while
this machine keeps the operator's real ones, so the tests pin both halves and,
more importantly, the failure modes: a missing file, a malformed file and a
half-filled file must all still yield a complete, usable profile.  A cleanup
pass that could cost someone their mission board on a stray keystroke would be
worse than the problem it solves.

All offline: a temporary path, no Qt, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import user_profile


@pytest.fixture
def profile_at(tmp_path, monkeypatch):
    """Point the module at a temp profile.json and reset its cache around it."""
    path = tmp_path / "profile.json"
    monkeypatch.setattr(user_profile, "PROFILE_PATH", path)
    monkeypatch.setattr(user_profile, "_cache", None)

    def write(payload):
        if payload is not None:
            path.write_text(json.dumps(payload), encoding="utf-8")
        user_profile.reload()
        return user_profile

    yield write
    user_profile.reload()


def test_with_no_profile_file_the_publishable_defaults_apply(profile_at):
    mod = profile_at(None)
    assert mod.brand() == "ExampleStore"
    assert mod.agency() == "Creator Studio"
    assert mod.missions()[0][0] == "Build Demo Game"


def test_profile_edits_cannot_mutate_shipped_defaults_or_cached_data(profile_at):
    mod = profile_at(None)
    copy = mod.profile()
    copy["missions"][0][0] = "Private sample project"
    copy["brand"] = "Private sample brand"
    assert mod.missions()[0][0] == "Build Demo Game"
    assert mod.brand() == "ExampleStore"
    assert mod.DEFAULTS["missions"][0][0] == "Build Demo Game"


def test_a_private_profile_overrides_every_field(profile_at):
    mod = profile_at({
        "brand": "RealBrand",
        "brand_niche": "kitchenware",
        "agency": "Real Agency",
        "missions": [["Ship It", "Get the thing out of the door."]],
    })
    assert mod.brand() == "RealBrand"
    assert mod.brand_niche() == "kitchenware"
    assert mod.agency() == "Real Agency"
    assert mod.missions() == (("Ship It", "Get the thing out of the door."),)


def test_a_partial_profile_keeps_the_defaults_for_what_it_omits(profile_at):
    mod = profile_at({"brand": "OnlyTheBrand"})
    assert mod.brand() == "OnlyTheBrand"
    assert mod.agency() == "Creator Studio"          # untouched default
    assert len(mod.missions()) == 5                  # untouched default


def test_blank_values_fall_back_rather_than_emptying_a_prompt(profile_at):
    """A hand-edited file with "brand": "" must not put "" into a prompt."""
    mod = profile_at({"brand": "", "missions": []})
    assert mod.brand() == "ExampleStore"
    assert len(mod.missions()) == 5


def test_malformed_json_never_breaks_startup(profile_at, tmp_path):
    (tmp_path / "profile.json").write_text("{not json at all", encoding="utf-8")
    user_profile.reload()
    assert user_profile.brand() == "ExampleStore"
    assert len(user_profile.missions()) == 5


def test_a_single_bad_mission_row_does_not_cost_the_whole_board(profile_at):
    mod = profile_at({"missions": [
        ["Good One", "A proper pair."],
        ["missing its description"],          # too short — skipped
        "not even a list",                    # wrong type — skipped
        ["", "blank title"],                  # no title — skipped
        ["Also Good", "Another proper pair."],
    ]})
    assert mod.missions() == (
        ("Good One", "A proper pair."),
        ("Also Good", "Another proper pair."),
    )


def test_missions_returns_pairs_of_plain_strings(profile_at):
    """missions.py unpacks these as (title, description) — keep that contract."""
    for title, description in profile_at(None).missions():
        assert isinstance(title, str) and title
        assert isinstance(description, str)


def test_malformed_types_and_whitespace_use_defaults(profile_at):
    mod = profile_at({"brand": False, "agency": {"invalid": True}, "brand_niche": "  ",
                      "missions": [[None, "invalid"], ["Title", {}], ["  ", "blank"]]})
    assert mod.brand() == "ExampleStore"
    assert mod.agency() == "Creator Studio"
    assert mod.brand_niche() == "home products"
    assert mod.missions()[0][0] == "Build Demo Game"


def test_profile_missions_are_normalised_and_deduplicated(profile_at):
    mod = profile_at({"missions": [["  Launch  ", "  Plan  "], ["launch", "duplicate"]]})
    assert mod.missions() == (("Launch", "Plan"),)


def test_single_private_starter_mission_seeds_without_index_error(profile_at, tmp_path):
    from orion_core.missions import MissionEngine
    profile_at({"missions": [["One project", "The only starter"]]})
    path = tmp_path / "missions.json"
    engine = MissionEngine(path=path)
    assert engine.current()["name"] == "One project"
    engine.create("Keep my progress")
    profile_at({"missions": [["Different default", "Do not replace the board"]]})
    reopened = MissionEngine(path=path)
    assert reopened.current()["name"] == "One project"
    assert "Keep my progress" in reopened.overview().text


def test_business_advice_uses_private_profile_without_hardcoded_examples(profile_at):
    import asyncio
    from unittest.mock import AsyncMock
    from orion_core.commerce import BusinessAdvisorAgent
    from orion_core.data import ToolResult
    profile_at({"brand": "Sample Business", "brand_niche": "sample products"})
    advisor = BusinessAdvisorAgent.__new__(BusinessAdvisorAgent)
    advisor.advise = AsyncMock(return_value=ToolResult("Prepared"))
    asyncio.run(advisor.brand_strategy())
    prompt = advisor.advise.call_args.args[0]
    assert "Sample Business" in prompt and "sample products" in prompt
    assert "Sample Business" in advisor.BRAND_CONTEXT
    assert "ExampleStore" not in advisor.BRAND_CONTEXT
