"""
The reflex offer (Mark XXVI) — closing the learning loop.

Reflex learning without this half is a feature that never fires: candidates
accumulate in a database and nothing ever promotes them, so ORION never actually
gets faster. The offer is what turns an observation into a shortcut.

Promotion stays a deliberate act by a person. A learned reflex bypasses the
model entirely, so the decision to install one is not ORION's to make on his
own — he notices, he offers, the user agrees. The tool path enforces that only
an already-eligible candidate can be promoted, so agreeing to a suggestion
cannot conjure a shortcut out of nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.proactivity_engine import (  # noqa: E402
    REFLEX_OFFER_COOLDOWN_S,
    ProactiveSnapshot,
    Urgency,
    gather,
)
from orion_core.reflex_learning import MIN_OBSERVATIONS, ReflexLearner  # noqa: E402

READ_ONLY = "web_search"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def learner(tmp_path):
    made = ReflexLearner(tmp_path / "learn.db")
    yield made
    made.close()


# ── the nudge ────────────────────────────────────────────────────────────────

def test_a_candidate_produces_an_offer():
    nudges = gather(ProactiveSnapshot(reflex_phrase="check the news", reflex_hits=4))
    offer = next(n for n in nudges if n.kind == "reflex_offer")
    assert "check the news" in offer.text
    assert "4 times" in offer.text


def test_no_candidate_means_no_offer():
    assert not [n for n in gather(ProactiveSnapshot()) if n.kind == "reflex_offer"]


def test_the_offer_is_ambient_so_it_can_never_interrupt():
    """A courtesy, not a reminder: it must sit at the lowest urgency, where the
    speech policy can hold it indefinitely without anything being lost."""
    offer = next(n for n in gather(
        ProactiveSnapshot(reflex_phrase="x", reflex_hits=3)) if n.kind == "reflex_offer")
    assert offer.urgency is Urgency.AMBIENT


def test_the_offer_is_made_at_most_once_a_day():
    offer = next(n for n in gather(
        ProactiveSnapshot(reflex_phrase="x", reflex_hits=3)) if n.kind == "reflex_offer")
    assert offer.cooldown_s == REFLEX_OFFER_COOLDOWN_S
    assert REFLEX_OFFER_COOLDOWN_S >= 12 * 3600


def test_the_offer_does_not_displace_the_actionable_nudges():
    nudges = gather(ProactiveSnapshot(
        focus_break_due=True, focus_label="revision", study_due=40,
        reflex_phrase="check the news", reflex_hits=5))
    kinds = {n.kind for n in nudges}
    assert {"focus_break_due", "study_due", "reflex_offer"} <= kinds


# ── the engine reads real candidates ─────────────────────────────────────────

def _engine(learner=None):
    from orion_core.proactivity_engine import ProactivityEngine
    engine = ProactivityEngine(None)
    engine.reflex_learner = learner
    return engine


def test_the_engine_surfaces_a_real_candidate(learner):
    for _ in range(MIN_OBSERVATIONS):
        learner.observe("check the news", READ_ONLY, {})
    snap = _engine(learner).snapshot()
    assert snap.reflex_phrase == "check the news"
    assert snap.reflex_hits == MIN_OBSERVATIONS


def test_the_engine_says_nothing_before_the_threshold(learner):
    learner.observe("check the news", READ_ONLY, {})
    assert _engine(learner).snapshot().reflex_phrase == ""


def test_a_promoted_phrase_is_no_longer_offered(learner):
    for _ in range(MIN_OBSERVATIONS):
        learner.observe("check the news", READ_ONLY, {})
    learner.promote("check the news")
    assert _engine(learner).snapshot().reflex_phrase == "", (
        "ORION would keep offering a shortcut he already has")


def test_no_learner_is_not_an_error():
    assert _engine(None).snapshot().reflex_phrase == ""


def test_a_broken_learner_never_breaks_a_survey():
    class _Broken:
        def candidates(self):
            raise RuntimeError("database is gone")
    snap = _engine(_Broken()).snapshot()
    assert snap.reflex_phrase == ""          # degraded, not crashed


def test_the_strongest_candidate_is_offered_first(learner):
    for _ in range(MIN_OBSERVATIONS):
        learner.observe("weak phrase", READ_ONLY, {})
    for _ in range(MIN_OBSERVATIONS + 5):
        learner.observe("strong phrase", READ_ONLY, {})
    assert _engine(learner).snapshot().reflex_phrase == "strong phrase"


# ── saying yes ───────────────────────────────────────────────────────────────

def test_the_promote_path_only_accepts_an_eligible_candidate():
    src = (ROOT / "orion_core" / "dispatch_files.py").read_text(encoding="utf-8")
    assert 'args.get("promote")' in src
    assert "learner.promote(promote)" in src, (
        "promotion must go through the learner, which enforces eligibility")


def test_the_forget_path_exists():
    src = (ROOT / "orion_core" / "dispatch_files.py").read_text(encoding="utf-8")
    assert 'args.get("forget")' in src
    assert "learner.demote(forget)" in src


def test_the_schema_declares_promote_and_forget():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "diagnostics")
    props = tool["parameters"]["properties"]
    assert "promote" in props and "forget" in props
    assert "promote" in tool["description"]


def test_promotion_is_never_automatic():
    """Nothing may promote a candidate except an explicit act. If this fails,
    ORION has started installing his own model bypasses unattended."""
    offenders = []
    for path in sorted((ROOT / "orion_core").rglob("*.py")):
        if path.name in {"reflex_learning.py", "dispatch_files.py"}:
            continue                       # the definition and the explicit path
        src = path.read_text(encoding="utf-8", errors="replace")
        if ".promote(" in src:
            offenders.append(path.name)
    assert not offenders, (
        "these promote a learned reflex outside the explicit user-facing path: "
        + ", ".join(offenders))


def test_the_app_gives_the_engine_the_learner():
    src = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "proactivity_engine.reflex_learner" in src


# ── end to end ───────────────────────────────────────────────────────────────

def test_the_whole_loop(learner):
    """Observe three times, get offered, agree, answer instantly."""
    engine = _engine(learner)

    for _ in range(MIN_OBSERVATIONS):
        learner.observe("what is on the news", READ_ONLY, {})

    snap = engine.snapshot()
    offer = next(n for n in gather(snap) if n.kind == "reflex_offer")
    assert "what is on the news" in offer.text

    assert learner.match("what is on the news") is None      # not yet
    assert learner.promote("what is on the news") is True    # the user agrees
    found = learner.match("What is on the news?")            # said again
    assert found is not None and found.tool == READ_ONLY

    assert engine.snapshot().reflex_phrase == ""             # not offered again
    assert learner.demote("what is on the news") is True     # and reversible
    assert learner.match("what is on the news") is None
