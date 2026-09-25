"""
Study & Recall (Mark XXIV — Mastery).

Spaced-repetition memory over any subject: the schedule must match SM-2 (shared
with the language tutor), the store must persist, due-selection must front the
cards about to be lost, and card generation must degrade to REAL extractive cards
when no model is present.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.study import (  # noqa: E402
    MASTERED_DAYS,
    StudyCard,
    StudyEngine,
    StudyStore,
    extractive_cards,
    human_interval,
    parse_generated_cards,
)

NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


@pytest.fixture()
def engine(tmp_path):
    return StudyEngine(StudyStore(tmp_path / "study.db"))


# ── SM-2 scheduling (must match the tutor, since they share sm2_step) ─────────

def test_a_pass_grows_the_interval_1_then_6():
    card = StudyCard(front="q", back="a")
    card.review(5, now=NOW)
    assert card.interval_days == 1.0 and card.repetitions == 1
    card.review(5, now=NOW)
    assert card.interval_days == 6.0 and card.repetitions == 2
    card.review(5, now=NOW)
    assert card.interval_days > 6.0 and card.repetitions == 3


def test_a_lapse_resets_to_daily_and_counts():
    card = StudyCard(front="q", back="a", repetitions=4, interval_days=40.0)
    card.review(1, now=NOW)
    assert card.repetitions == 0
    assert card.interval_days == 1.0
    assert card.lapses == 1


def test_the_matching_tutor_algorithm_is_actually_shared():
    from orion_core.language_tutor import VocabCard, schedule
    vocab = VocabCard(term="hola", translation="hi")
    study = StudyCard(front="hola?", back="hi")
    schedule(vocab, 4, now=NOW)
    study.review(4, now=NOW)
    assert (vocab.ease, vocab.interval_days, vocab.repetitions) == (
        study.ease, study.interval_days, study.repetitions)


def test_mastered_is_a_long_interval():
    card = StudyCard(front="q", back="a", interval_days=MASTERED_DAYS)
    assert card.is_mastered
    assert not StudyCard(front="q", back="a", interval_days=5.0).is_mastered


def test_a_never_reviewed_card_is_due_now_and_new():
    card = StudyCard(front="q", back="a")
    assert card.is_due(NOW)
    assert card.is_new


# ── persistence ──────────────────────────────────────────────────────────────

def test_add_and_read_back(engine):
    card = engine.add("What is the resting potential?", "about -70 mV",
                      deck="Neuro", tags=["exam"])
    assert card.id is not None
    got = engine.store.get(card.id)
    assert got.front.startswith("What is the resting")
    assert got.deck == "Neuro"
    assert got.tags == ["exam"]


def test_a_card_needs_both_sides(engine):
    with pytest.raises(ValueError):
        engine.add("question only", "")


def test_review_state_survives_a_reload(tmp_path):
    store = StudyStore(tmp_path / "s.db")
    engine = StudyEngine(store)
    card = engine.add("q", "a", deck="D")
    engine.next_due("D")
    engine.grade(5, now=NOW)
    store.close()

    reopened = StudyEngine(StudyStore(tmp_path / "s.db"))
    again = reopened.store.all("D")[0]
    assert again.reviews == 1
    assert again.interval_days == 1.0
    assert again.repetitions == 1


def test_delete_removes_the_card(engine):
    card = engine.add("q", "a")
    assert engine.store.delete(card.id)
    assert engine.store.get(card.id) is None


# ── due selection ────────────────────────────────────────────────────────────

def test_due_prefers_new_cards_then_the_most_overdue(engine):
    seen = engine.add("seen", "a", deck="D")
    engine.next_due("D")
    engine.grade(5, now=NOW - timedelta(days=5))     # due again 4 days ago
    fresh = engine.add("fresh", "a", deck="D")        # never reviewed

    order = [c.front for c in engine.due("D", now=NOW)]
    assert order[0] == "fresh", "a never-seen card should lead"
    assert "seen" in order


def test_a_card_not_yet_due_is_held_back(engine):
    engine.add("q", "a", deck="D")
    engine.next_due("D")
    engine.grade(5, now=NOW)                           # next due in 1 day
    assert engine.due("D", now=NOW) == []
    assert len(engine.due("D", now=NOW + timedelta(days=2))) == 1


# ── grading flow + retention ─────────────────────────────────────────────────

def test_grade_advances_the_card_in_play_without_an_id(engine):
    engine.add("q1", "a1", deck="D")
    asked = engine.next_due("D")
    graded = engine.grade(4)
    assert graded.id == asked.id
    assert graded.reviews == 1


def test_grade_with_no_review_started_is_a_clean_miss(engine):
    assert engine.grade(5) is None


def test_retention_reflects_the_review_log(engine):
    engine.add("q", "a", deck="D")
    engine.next_due("D")
    engine.grade(5, now=NOW)          # pass
    engine.add("q2", "a2", deck="D")
    engine.next_due("D")
    engine.grade(1, now=NOW)          # fail
    # one pass, one fail → 50%
    assert engine.store.retention("D", now=NOW) == 50.0


def test_stats_counts_new_learning_and_mastered(engine):
    engine.add("new", "a", deck="D")                               # new
    learning = engine.add("learning", "a", deck="D")
    engine.next_due("D"); engine.grade(5, now=NOW)                 # interval 1 → learning
    mastered = engine.store.get(engine.add("mastered", "a", deck="D").id)
    mastered.interval_days = 40.0
    mastered.reviews = 3
    engine.store.update(mastered)

    s = engine.stats("D")
    assert s["total"] == 3
    assert s["new"] == 1
    assert s["mastered"] == 1
    assert s["learning"] == 1


# ── card generation ──────────────────────────────────────────────────────────

def test_parse_generated_cards_is_lenient_about_separators():
    text = ("Q: What is a neuron? | A: an excitable cell\n"
            "Q: Define synapse - A: a junction between neurons\n"
            "not a card line\n")
    pairs = parse_generated_cards(text)
    assert ("What is a neuron?", "an excitable cell") in pairs
    assert any("synapse" in q.lower() for q, _ in pairs)
    assert len(pairs) == 2


def test_extractive_cards_makes_real_cards_from_definitions():
    text = ("Myelin is a fatty sheath that insulates axons.\n"
            "Action potential: a rapid rise and fall in membrane voltage.\n")
    cards = extractive_cards(text)
    fronts = " ".join(f for f, _ in cards)
    assert "Myelin" in fronts
    assert any("axons" in b for _, b in cards)


def test_extractive_cards_falls_back_to_cloze():
    text = "The hippocampus is strongly implicated in the consolidation of memory."
    cards = extractive_cards(text)
    assert cards
    assert any("_____" in front for front, _ in cards)


async def test_generate_uses_the_model_when_present(engine):
    async def fake(prompt: str) -> str:
        assert "MATERIAL" in prompt
        return "Q: What insulates axons? | A: myelin"
    engine._generate = fake
    cards = await engine.generate_cards("Myelin insulates axons.", deck="Neuro")
    assert cards and cards[0].deck == "Neuro"
    assert any("axons" in c.front.lower() for c in cards)


async def test_generate_falls_back_when_no_model(engine):
    cards = await engine.generate_cards(
        "Dopamine is a neurotransmitter involved in reward and motivation.",
        deck="Neuro", count=5)
    assert cards, "must still produce real cards with no model"
    assert all(c.deck == "Neuro" for c in cards)


async def test_generate_on_empty_material_makes_nothing(engine):
    assert await engine.generate_cards("   ", deck="D") == []


# ── spoken helpers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("days,expected", [
    (0.5, "later today"), (1.0, "1 day"), (6.0, "6 days"),
    (90.0, "3 months"), (400.0, "1.1 years"),
])
def test_human_interval_reads_naturally(days, expected):
    assert human_interval(days) == expected


# ── tool wiring ──────────────────────────────────────────────────────────────

def test_the_study_tool_is_registered():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"study"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "study")
    props = tool["parameters"]["properties"]
    for key in ("action", "front", "back", "deck", "text", "quality"):
        assert key in props


class _Bag:
    """A minimal object carrying only what study_tool touches, so the async tool
    can be exercised without building a whole dispatcher. The tool is called
    unbound. research=None makes generation take the extractive fallback."""

    def __init__(self, dbpath):
        self.research = None
        self.study = StudyEngine(StudyStore(dbpath))


async def _tool(stub, args):
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    return await KnowledgeDispatchMixin.study_tool(stub, args)


async def test_tool_runs_the_add_review_grade_loop(tmp_path):
    stub = _Bag(tmp_path / "s.db")
    r = await _tool(stub, {"action": "add", "front": "Resting potential?",
                           "back": "-70 mV", "deck": "Neuro"})
    assert r.ok and "Neuro" in r.text

    r = await _tool(stub, {"action": "review", "deck": "Neuro"})
    assert "Resting potential?" in r.text and "-70 mV" in r.text

    r = await _tool(stub, {"action": "grade", "quality": 5})
    assert r.ok and "back in" in r.text

    r = await _tool(stub, {"action": "stats", "deck": "Neuro"})
    assert "1 cards" in r.text and "mastered" in r.text


async def test_tool_generates_cards_from_material_offline(tmp_path):
    stub = _Bag(tmp_path / "s.db")
    r = await _tool(stub, {
        "action": "generate", "deck": "Neuro",
        "text": "Myelin is a fatty sheath that insulates axons. "
                "Dopamine is a neurotransmitter involved in reward."})
    assert r.ok and "cards" in r.text
    assert stub.study.stats("Neuro")["total"] >= 1


async def test_tool_review_when_nothing_is_due_is_encouraging(tmp_path):
    stub = _Bag(tmp_path / "s.db")
    r = await _tool(stub, {"action": "review"})
    assert "Nothing's due" in r.text


# ── flashcards from a real document (Mark XXVI) ─────────────────────────────

def _make_pdf(path):
    """A genuinely valid PDF with extractable text, or None if we cannot make one."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig = plt.figure(figsize=(6, 4))
    fig.text(0.05, 0.8, "Myelin is a fatty sheath that insulates axons.", fontsize=11)
    fig.text(0.05, 0.6, "Dopamine is a neurotransmitter involved in reward.", fontsize=11)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


async def test_generate_reads_a_pdf_not_just_plain_text(tmp_path):
    """read_text() on a PDF returns binary noise and silently makes nonsense
    cards; the document reader extracts the real text."""
    pdf = _make_pdf(tmp_path / "paper.pdf")
    if pdf is None:
        pytest.skip("no PDF writer available in this environment")
    stub = _Bag(tmp_path / "s.db")
    result = await _tool(stub, {"action": "generate", "path": str(pdf), "deck": "Neuro"})
    assert result.ok, result.text
    fronts = " ".join(c.front for c in stub.study.store.all()).lower()
    assert "myelin" in fronts or "dopamine" in fronts


async def test_generate_reports_a_missing_file_clearly(tmp_path):
    stub = _Bag(tmp_path / "s.db")
    result = await _tool(stub, {"action": "generate", "path": str(tmp_path / "nope.pdf")})
    assert result.ok is False and "can't find" in result.text


async def test_generate_reports_an_unreadable_document_rather_than_guessing(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 not really a pdf")
    stub = _Bag(tmp_path / "s.db")
    result = await _tool(stub, {"action": "generate", "path": str(broken)})
    assert result.ok is False
    assert stub.study.store.all() == [], "no cards should be invented from a bad file"


async def test_generate_flags_a_document_with_no_extractable_text(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n  ", encoding="utf-8")
    stub = _Bag(tmp_path / "s.db")
    result = await _tool(stub, {"action": "generate", "path": str(empty)})
    assert result.ok is False and "extractable" in result.text


def test_the_schema_documents_document_support():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "study")
    assert "PDF" in tool["description"]


def test_stats_reports_retention_from_the_clock_it_was_given(engine):
    """stats() threads `now` into counts and due_count; retention used to read
    the wall clock instead, so an injected clock gave figures from two
    different dates — and the test that would have caught it only failed once
    the fixed review date aged out of the 30-day window, a month after it was
    written."""
    engine.add("q", "a", deck="D")
    engine.next_due("D")
    engine.grade(5, now=NOW)

    assert engine.stats("D", now=NOW)["retention"] == 100.0
    # Far enough past the review that it falls outside the window.
    later = NOW + timedelta(days=400)
    assert engine.stats("D", now=later)["retention"] is None
