"""
ToolResolver (Mark XXVI, Phase 2) — the pre-filter that shrinks the 132-tool
surface per turn.

The load-bearing guarantees:
  * OFF is a provable no-op (returns the input unchanged) — zero regression;
  * ON keeps a safety floor, honours the tool cap, and actually ranks the right
    tool into a small top-k for a keyword-rich query (the golden set);
  * hard filters never hide a tool the user still needs (esp. security_recon,
    whose own 'authorize_target' action would otherwise become unreachable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import tool_resolver as tr  # noqa: E402
from orion_core.dispatch_schema import TOOL_DECLARATIONS  # noqa: E402

DECLS = list(TOOL_DECLARATIONS)
NAMES = {d["name"] for d in DECLS}


# ── the provable no-op (zero regression) ──────────────────────────────────────

def test_disabled_returns_the_input_unchanged():
    out = tr.resolve("anything at all", tr.ResolverState(), DECLS, enabled=False)
    assert [d["name"] for d in out] == [d["name"] for d in DECLS]
    assert len(out) == len(DECLS)


def test_disabled_is_the_default_via_the_flag(monkeypatch):
    monkeypatch.delenv("ORION_TOOL_RESOLVER", raising=False)
    assert tr.resolver_enabled() is False
    monkeypatch.setenv("ORION_TOOL_RESOLVER", "1")
    assert tr.resolver_enabled() is True


# ── the cap + the safety floor ────────────────────────────────────────────────

def test_the_cap_is_honoured():
    out = tr.resolve("play chess", tr.ResolverState(), DECLS, max_tools=10)
    assert len(out) <= 10


def test_the_safety_floor_is_always_present():
    out = tr.resolve("some obscure request", tr.ResolverState(), DECLS, max_tools=12)
    names = {d["name"] for d in out}
    for floor in tr.ALWAYS_INCLUDE:
        if floor in NAMES:
            assert floor in names, f"{floor} must never be filtered out"


def test_the_result_has_no_duplicates():
    out = tr.resolve("study and focus", tr.ResolverState(focus_active=True), DECLS)
    names = [d["name"] for d in out]
    assert len(names) == len(set(names))


# ── the golden retrieval set ──────────────────────────────────────────────────

GOLDEN = [
    ("play a game of chess",                              "chess"),
    ("pomodoro deep work block",                          "focus"),
    ("spaced repetition active recall",                   "study"),
    ("recap what we decided yesterday",                   "rewind"),
    ("nmap port scan host",                               "security_recon"),
    ("summarise a youtube video transcript",              "entertainment"),
    ("back up settings to a timestamped zip",             "backup"),
    ("token usage how much left",                         "token_usage"),
    ("locate a town and list nearby",                     "geo"),
    ("read a folder and answer questions with citations", "read_documents"),
    # The camera workbench: these are the words said while holding a board
    # up to the lens, and each one must reach ORION's eyes.
    ("scan this pcb for damage",                           "vision_analyse"),
    ("inspect this circuit board through the camera",      "vision_analyse"),
    ("why is this motherboard not working",                "vision_analyse"),
    ("read the markings on this chip",                     "vision_analyse"),
    ("check the soldering on these capacitors",            "vision_analyse"),
]


@pytest.mark.parametrize("query,expected", GOLDEN)
def test_the_golden_query_surfaces_the_right_tool(query, expected):
    assert expected in NAMES, f"schema drift: {expected} no longer exists"
    out = tr.resolve(query, tr.ResolverState(), DECLS, max_tools=15)
    assert expected in {d["name"] for d in out}, (
        f"{expected!r} not in top-15 for {query!r}: "
        f"{[d['name'] for d in out]}")


def test_naming_the_tool_ranks_it_near_the_top():
    out = tr.resolve("play chess", tr.ResolverState(), DECLS, max_tools=6)
    assert "chess" in {d["name"] for d in out}


# ── hard filters ──────────────────────────────────────────────────────────────

def test_offline_mode_drops_pure_cloud_tools():
    online = tr.resolve("search the web for news", tr.ResolverState(mode="A"), DECLS)
    assert "web_search" in {d["name"] for d in online}
    offline = tr.resolve("search the web for news", tr.ResolverState(mode="B"), DECLS)
    assert "web_search" not in {d["name"] for d in offline}


def test_offline_flag_also_drops_cloud_tools():
    out = tr.resolve("open the news", tr.ResolverState(online=False), DECLS)
    assert "open_news" not in {d["name"] for d in out}


def test_security_recon_is_never_hidden_by_the_auth_state():
    # 'authorize_target' is an ACTION of security_recon; hiding it when no target
    # is authorised would make authorising one impossible.
    assert "security_recon" not in tr.CLOUD_ONLY
    out = tr.resolve("authorize a target for a port scan",
                     tr.ResolverState(auth_targets=False), DECLS, max_tools=15)
    assert "security_recon" in {d["name"] for d in out}


# ── context boosts ────────────────────────────────────────────────────────────

def test_a_running_focus_block_boosts_study_and_focus():
    # A neutral query with no informative tokens, so the boost — not word
    # overlap — is what surfaces the learning tools.
    out = tr.resolve("okay", tr.ResolverState(focus_active=True), DECLS, max_tools=8)
    names = {d["name"] for d in out}
    assert "focus" in names and "study" in names


def test_the_foreground_page_boosts_its_domain():
    out = tr.resolve("help", tr.ResolverState(active_page="COGNITION"),
                     DECLS, max_tools=8)
    names = {d["name"] for d in out}
    assert "study" in names or "focus" in names


def test_recently_used_tools_are_boosted():
    out = tr.resolve("carry on", tr.ResolverState(recent_tools=("chess",)),
                     DECLS, max_tools=8)
    assert "chess" in {d["name"] for d in out}


# ── domain classification ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name,domain", [
    ("study", "learning"), ("focus", "learning"), ("security_recon", "security"),
    ("rewind", "memory"), ("chess", "games"), ("geo", "geo"),
    ("web_search", "web"), ("forge", "engineering"),
])
def test_tool_classification(name, domain):
    assert tr.classify_tool(name) == domain


def test_unknown_tool_is_general():
    assert tr.classify_tool("no_such_tool_xyz") == "general"


def test_every_real_tool_classifies_to_a_known_domain():
    # Facade grouping (Phase 2b) needs full coverage; a large 'general' bucket
    # would mean tools with no home. Allow a small tail, but not a big one.
    general = [d["name"] for d in DECLS if tr.classify_tool(d["name"]) == "general"
               and not d["name"].startswith("mcp__")]
    assert len(general) <= 12, f"unclassified tools need a domain: {general}"


# ── the dispatcher seam (inert unless the flag is set) ────────────────────────

def test_dispatcher_accessor_passes_through_when_disabled(monkeypatch):
    from types import SimpleNamespace

    from orion_core.dispatcher import OrionDispatcher
    monkeypatch.delenv("ORION_TOOL_RESOLVER", raising=False)
    stub = SimpleNamespace(TOOL_DECLARATIONS=DECLS)
    out = OrionDispatcher.resolve_tool_declarations(stub, "play chess")
    assert len(out) == len(DECLS)


def test_dispatcher_accessor_filters_when_enabled(monkeypatch):
    from types import SimpleNamespace

    from orion_core.dispatcher import OrionDispatcher
    monkeypatch.setenv("ORION_TOOL_RESOLVER", "1")
    stub = SimpleNamespace(TOOL_DECLARATIONS=DECLS)
    out = OrionDispatcher.resolve_tool_declarations(stub, "play chess")
    assert len(out) <= tr.MAX_TOOLS
    assert "chess" in {d["name"] for d in out}
