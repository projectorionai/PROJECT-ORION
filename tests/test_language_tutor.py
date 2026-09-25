"""
The immersion tutor (CAP-01) — spaced repetition, pronunciation grading and the
immersion directive.

    "I want ORION to be able to teach me multiple languages, Spanish
     especially."
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import language_tutor as lt  # noqa: E402
from orion_core.language_tutor import (  # noqa: E402
    LanguageTutor, VocabCard, VocabDeck, schedule, score_pronunciation,
    resolve_language, resolve_level,
)


@pytest.fixture()
def deck(tmp_path):
    d = VocabDeck(path=tmp_path / "deck.db")
    yield d
    d.close()


@pytest.fixture()
def tutor(deck):
    return LanguageTutor(deck=deck)


# ── language resolution ──────────────────────────────────────────────────────

def test_spanish_is_the_default_language():
    assert resolve_language(None).code == "es"
    assert resolve_language("").code == "es"
    assert resolve_language("nonsense-language").code == "es"


@pytest.mark.parametrize("value,code", [
    ("Spanish", "es"), ("español", "es"), ("es", "es"), ("castellano", "es"),
    ("French", "fr"), ("français", "fr"),
    ("German", "de"), ("Italian", "it"), ("portuguese", "pt"),
])
def test_language_aliases_resolve(value, code):
    assert resolve_language(value).code == code


def test_level_resolution_defaults_and_clamps():
    assert resolve_level(None) == "A1"
    assert resolve_level("b2") == "B2"
    assert resolve_level("Z9") == "A1"


# ── SM-2 scheduling ──────────────────────────────────────────────────────────

def test_a_pass_grows_the_interval_1_then_6():
    card = VocabCard(term="hola", translation="hello")
    now = datetime(2026, 8, 10, tzinfo=timezone.utc)
    schedule(card, 5, now=now)
    assert card.interval_days == 1.0 and card.repetitions == 1
    schedule(card, 5, now=now)
    assert card.interval_days == 6.0 and card.repetitions == 2
    schedule(card, 5, now=now)
    assert card.interval_days > 6.0 and card.repetitions == 3


def test_a_lapse_resets_to_daily_and_counts():
    card = VocabCard(term="perro", translation="dog", repetitions=4,
                     interval_days=40.0, ease=2.5)
    schedule(card, 1)
    assert card.repetitions == 0
    assert card.interval_days == 1.0
    assert card.lapses == 1


def test_ease_never_falls_below_the_floor():
    card = VocabCard(term="x", translation="y", ease=1.3)
    for _ in range(10):
        schedule(card, 0)          # repeatedly forgotten
    assert card.ease >= 1.3


def test_due_date_advances_by_the_interval():
    card = VocabCard(term="agua", translation="water")
    now = datetime(2026, 8, 10, tzinfo=timezone.utc)
    schedule(card, 5, now=now)     # interval 1 day
    assert not card.is_due(now)
    assert card.is_due(now + timedelta(days=1, seconds=1))


def test_a_never_reviewed_card_is_due_now():
    assert VocabCard(term="nuevo", translation="new").is_due()


# ── the deck ─────────────────────────────────────────────────────────────────

def test_adding_and_reading_back_a_card(deck):
    card = deck.add_card(VocabCard(term="gato", translation="cat", language="es"))
    assert card.id is not None
    fetched = deck.get(card.id)
    assert fetched.term == "gato" and fetched.translation == "cat"


def test_terms_are_unique_per_language(deck):
    deck.add_card(VocabCard(term="hola", translation="hello", language="es"))
    deck.add_card(VocabCard(term="hola", translation="hi there", language="es"))
    cards = [c for c in deck.all_cards("es") if c.term == "hola"]
    assert len(cards) == 1
    assert cards[0].translation == "hi there"   # upsert updated the meaning


def test_due_cards_excludes_ones_scheduled_for_later(deck):
    now = datetime(2026, 8, 10, tzinfo=timezone.utc)
    soon = deck.add_card(VocabCard(term="a", translation="a", language="es"))
    later = VocabCard(term="b", translation="b", language="es")
    schedule(later, 5, now=now)          # due in 1 day
    deck.add_card(later)
    deck.record_review(later)
    due_now = [c.term for c in deck.due_cards("es", now=now)]
    assert "a" in due_now and "b" not in due_now


def test_review_state_persists(deck):
    card = deck.add_card(VocabCard(term="sol", translation="sun", language="es"))
    schedule(card, 5)
    deck.record_review(card)
    again = deck.get(card.id)
    assert again.repetitions == 1
    assert again.interval_days == 1.0


# ── pronunciation ────────────────────────────────────────────────────────────

def test_a_perfect_repeat_scores_high():
    s = score_pronunciation("buenos días", "buenos días")
    assert s.score >= 90
    assert not s.missed


def test_accent_insensitive_by_default():
    s = score_pronunciation("café", "cafe")
    assert s.score >= 90


def test_accent_sensitive_mode_penalises_a_dropped_accent():
    lenient = score_pronunciation("café", "cafe").score
    strict = score_pronunciation("café", "cafe", accent_sensitive=True).score
    assert strict <= lenient


def test_a_missed_word_is_reported():
    s = score_pronunciation("el perro grande", "el grande")
    assert "perro" in s.missed
    assert s.score < 90


def test_gibberish_scores_low():
    s = score_pronunciation("muchas gracias", "zxqw plfgh")
    assert s.score < 40


def test_empty_expected_is_zero_not_a_crash():
    assert score_pronunciation("", "hola").score == 0


# ── seeding ──────────────────────────────────────────────────────────────────

def test_spanish_seeds_a_starter_deck(tutor):
    added = tutor.seed("es")
    assert added >= 30
    terms = {c.term for c in tutor.deck.all_cards("es")}
    assert "hola" in terms and "gracias" in terms


def test_seeding_is_idempotent(tutor):
    first = tutor.seed("es")
    second = tutor.seed("es")
    assert first > 0
    assert second == 0


def test_starting_a_session_seeds_automatically(tutor):
    tutor.start_session("Spanish")
    assert tutor.stats("es")["total"] >= 30


# ── sessions / directive ─────────────────────────────────────────────────────

def test_immersion_directive_pins_target_language_only():
    session = LanguageTutor(deck=None).start_session("Spanish", "A2", immersion=True) \
        if False else None
    # build directly to avoid touching the real deck
    from orion_core.language_tutor import TutorSession, resolve_language
    d = TutorSession(resolve_language("es"), "A2", immersion=True).directive()
    assert "Spanish" in d and "A2" in d
    assert "ONLY" in d and "IMMERSION" in d


def test_bilingual_directive_offers_english_gloss():
    from orion_core.language_tutor import TutorSession, resolve_language
    d = TutorSession(resolve_language("es"), "A1", immersion=False).directive()
    assert "English" in d and "BILINGUAL" in d


def test_grade_advances_and_persists_a_real_card(tutor):
    card = tutor.add_word("libro", "book", language="es")
    graded = tutor.grade(card.id, 5)
    assert graded is not None and graded.repetitions == 1
    assert tutor.deck.get(card.id).repetitions == 1


def test_grade_of_unknown_card_returns_none(tutor):
    assert tutor.grade(999999, 5) is None


# ── the dispatch tool ────────────────────────────────────────────────────────

class _Shim:
    """A minimal stand-in for the dispatcher, carrying a temp-deck tutor so the
    real config/ deck is never touched."""
    pass


@pytest.fixture()
def tool(tmp_path):
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    shim = _Shim()
    shim._language_tutor = LanguageTutor(deck=VocabDeck(path=tmp_path / "tool.db"))
    def call(args):
        return KnowledgeDispatchMixin.language_tutor_tool(shim, args)
    return call


def test_tool_start_returns_the_immersion_directive(tool):
    r = tool({"action": "start", "language": "Spanish", "level": "A2"})
    assert r.ok
    assert "Spanish" in r.text and "A2" in r.text
    assert "IMMERSION" in r.text


def test_tool_start_bilingual_mode(tool):
    r = tool({"action": "start", "language": "es", "mode": "bilingual"})
    assert "BILINGUAL" in r.text


def test_tool_add_then_review_then_grade(tool):
    added = tool({"action": "add", "term": "ventana", "translation": "window",
                  "language": "es"})
    assert added.ok and "ventana" in added.text
    review = tool({"action": "review", "language": "es"})
    assert "ventana" in review.text
    # pull the id out of the review text
    import re
    m = re.search(r"#(\d+)\] ventana", review.text)
    assert m
    graded = tool({"action": "grade", "card": int(m.group(1)), "quality": 5})
    assert graded.ok and "tomorrow" in graded.text


def test_tool_add_requires_both_fields(tool):
    r = tool({"action": "add", "term": "solo"})
    assert not r.ok


def test_tool_pronounce_scores(tool):
    r = tool({"action": "pronounce", "expected": "buenos días",
              "heard": "buenos días"})
    assert r.ok and "/100" in r.text


def test_tool_unknown_action_lists_actions(tool):
    r = tool({"action": "flibbertigibbet"})
    assert not r.ok and "actions" in r.text.lower()


def test_language_tutor_is_registered_in_the_handler_table():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    src = inspect.getsource(OrionDispatcher.__init__) if hasattr(OrionDispatcher, "__init__") else ""
    # handler_table is built in __init__ or a helper; assert the mapping exists
    # by scanning the whole class source.
    whole = inspect.getsource(OrionDispatcher)
    assert '"language_tutor"' in whole


def test_schema_advertises_the_tutor():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "language_tutor")
    desc = tool["description"].lower()
    assert "spanish" in desc
    assert "spaced repetition" in desc or "review" in desc
    props = tool["parameters"]["properties"]
    for key in ("action", "language", "level", "term", "translation", "quality"):
        assert key in props
