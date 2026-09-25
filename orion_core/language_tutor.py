"""
The immersion tutor (CAP-01) — a real language teacher, not a phrasebook.

    "I want ORION to be able to teach me multiple languages, Spanish
     especially."

Three things a phrasebook cannot do, and this does:

**It remembers what you're forgetting.** Every word you learn becomes a card on
an SM-2 spaced-repetition schedule — the same algorithm Anki is built on — so a
word you keep missing comes back tomorrow while one you know cold is not shown
again for months. The schedule is the whole point: it surfaces a word at the
exact moment you're about to lose it, which is when review actually builds
memory.

**It scores how you say it, not just what you type.** ORION already transcribes
speech offline through Whisper; this compares that transcript against the phrase
you were asked to say and grades the match — word by word — so "your R in
*perro* dropped, the rest was clean" is feedback the tutor can give without any
cloud call. The transcription is done by the caller; this module scores
*expected vs heard text*, which keeps the grading deterministic and testable.

**It can refuse to speak English.** An immersion session hands the live model a
directive that pins the target language, the CEFR level and the correction
style, so a genuine A2 Spanish conversation stays in Spanish, stays at A2, and
corrects gently rather than lecturing. The tutor composes the directive; the
existing model routing carries the conversation.

Persistence is a dedicated SQLite deck (``config/language_deck.db``) rather than
the knowledge graph: SRS needs per-card scheduling state (ease, interval, due
date, lapses) that the entity graph has no place for, and the deck is small,
private and self-contained. Spanish ships seeded with a starter A1 deck so
"teach me Spanish" has something to teach from the first minute.
"""

from __future__ import annotations

import difflib
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Iterable

from .constants import CONFIG_DIR
from .utils import utc_stamp
from .db import apply_pragmas

DECK_PATH = CONFIG_DIR / "language_deck.db"


# ── languages ────────────────────────────────────────────────────────────────
# Spanish first and by design; the rest are supported by the same machinery.

@dataclass(frozen=True)
class Language:
    code: str          # ISO 639-1
    name: str          # English name
    native: str        # endonym
    immersion_ok: bool = True   # whether a target-only session makes sense


LANGUAGES: dict[str, Language] = {
    "es": Language("es", "Spanish", "Español"),
    "fr": Language("fr", "French", "Français"),
    "de": Language("de", "German", "Deutsch"),
    "it": Language("it", "Italian", "Italiano"),
    "pt": Language("pt", "Portuguese", "Português"),
    "ja": Language("ja", "Japanese", "日本語"),
}

DEFAULT_LANGUAGE = "es"

# CEFR ladder, easiest first. The level pins vocabulary breadth and how much
# scaffolding the tutor gives.
CEFR = ("A1", "A2", "B1", "B2", "C1", "C2")
DEFAULT_LEVEL = "A1"

# Aliases people actually type/say.
_LANG_ALIASES = {
    "spanish": "es", "español": "es", "espanol": "es", "castellano": "es",
    "french": "fr", "français": "fr", "francais": "fr",
    "german": "de", "deutsch": "de",
    "italian": "it", "italiano": "it",
    "portuguese": "pt", "português": "pt", "portugues": "pt",
    "japanese": "ja", "日本語": "ja", "nihongo": "ja",
}


def resolve_language(value: str | None) -> Language:
    """Map 'Spanish', 'es', 'español' → the Language, defaulting to Spanish."""
    token = str(value or "").strip().lower()
    if not token:
        return LANGUAGES[DEFAULT_LANGUAGE]
    if token in LANGUAGES:
        return LANGUAGES[token]
    if token in _LANG_ALIASES:
        return LANGUAGES[_LANG_ALIASES[token]]
    return LANGUAGES[DEFAULT_LANGUAGE]


def resolve_level(value: str | None) -> str:
    token = str(value or "").strip().upper()
    return token if token in CEFR else DEFAULT_LEVEL


# ── spaced repetition (SM-2) ─────────────────────────────────────────────────

@dataclass
class VocabCard:
    term: str                       # the word/phrase in the target language
    translation: str                # its meaning in English
    language: str = DEFAULT_LANGUAGE
    example: str = ""               # a sentence using it (target language)
    level: str = DEFAULT_LEVEL
    # SM-2 scheduling state
    ease: float = 2.5               # ease factor; floors at 1.3
    interval_days: float = 0.0      # days until next due after last review
    repetitions: int = 0            # consecutive correct reviews
    lapses: int = 0                 # times forgotten (quality < 3)
    due_at: str = ""                # ISO; empty == due now (never reviewed)
    created_at: str = field(default_factory=utc_stamp)
    id: int | None = None

    def is_due(self, now: datetime | None = None) -> bool:
        if not self.due_at:
            return True
        now = now or datetime.now(timezone.utc)
        try:
            return _parse_iso(self.due_at) <= now
        except ValueError:
            return True


# SM-2 first two intervals are fixed; afterwards interval *= ease.
_FIRST_INTERVAL = 1.0
_SECOND_INTERVAL = 6.0
_MIN_EASE = 1.3


def sm2_step(*, ease: float, interval_days: float, repetitions: int,
             lapses: int, quality: int) -> tuple[float, float, int, int]:
    """One SM-2 review step, as pure arithmetic — the scheduling core shared by
    the language tutor's VocabCard and the general Study & Recall cards, so the
    two can never drift apart.

    ``quality`` is 0–5: <3 is a lapse (reset to daily, ease drops, lapse counted);
    3–5 is a pass (interval grows 1 → 6 → ×ease) with the ease nudged by how
    confident the recall was. Returns the new (ease, interval_days, repetitions,
    lapses); the ease is floored so a hard card never spirals to an impossible one.
    """
    quality = max(0, min(5, int(quality)))
    if quality < 3:
        repetitions = 0
        lapses += 1
        interval_days = _FIRST_INTERVAL
    else:
        if repetitions == 0:
            interval_days = _FIRST_INTERVAL
        elif repetitions == 1:
            interval_days = _SECOND_INTERVAL
        else:
            interval_days = round(interval_days * ease, 2)
        repetitions += 1
    ease = max(_MIN_EASE,
               ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    return round(ease, 3), interval_days, repetitions, lapses


def schedule(card: VocabCard, quality: int, now: datetime | None = None) -> VocabCard:
    """Advance *card* by one review under SM-2. ``quality`` is 0–5. Returns the
    SAME card, mutated, for convenience. The arithmetic lives in sm2_step."""
    now = now or datetime.now(timezone.utc)
    card.ease, card.interval_days, card.repetitions, card.lapses = sm2_step(
        ease=card.ease, interval_days=card.interval_days,
        repetitions=card.repetitions, lapses=card.lapses, quality=quality)
    card.due_at = _iso(now + timedelta(days=card.interval_days))
    return card


# ── pronunciation scoring ────────────────────────────────────────────────────

@dataclass
class WordMatch:
    expected: str
    heard: str
    ok: bool


@dataclass
class PronunciationScore:
    score: int                      # 0–100
    words: list[WordMatch]
    verdict: str                    # short human summary
    expected: str
    heard: str

    @property
    def missed(self) -> list[str]:
        return [w.expected for w in self.words if not w.ok]


def _normalise_for_match(text: str, *, strip_accents: bool = False) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[¿¡?!.,;:\"'()…]+", " ", text)
    if strip_accents:
        text = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    return re.sub(r"\s+", " ", text).strip()


def score_pronunciation(expected: str, heard: str,
                        *, accent_sensitive: bool = False) -> PronunciationScore:
    """Grade a spoken attempt: compare the phrase asked for against what
    Whisper heard. Deterministic — no audio and no model needed here.

    ``accent_sensitive`` off (the default) means "cafe" is accepted for "café";
    a learner drilling accents can turn it on.
    """
    exp_words = _normalise_for_match(expected, strip_accents=not accent_sensitive).split()
    heard_norm = _normalise_for_match(heard, strip_accents=not accent_sensitive)
    heard_words = heard_norm.split()

    if not exp_words:
        return PronunciationScore(0, [], "Nothing to say.", expected, heard)

    # Greedy per-word alignment: each expected word is matched to its best
    # remaining heard word above a similarity floor.
    remaining = list(heard_words)
    matches: list[WordMatch] = []
    correct = 0
    for word in exp_words:
        best_i, best_ratio = -1, 0.0
        for i, cand in enumerate(remaining):
            ratio = difflib.SequenceMatcher(None, word, cand).ratio()
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i
        ok = best_ratio >= 0.72
        heard_word = remaining.pop(best_i) if (ok and best_i >= 0) else ""
        matches.append(WordMatch(word, heard_word, ok))
        if ok:
            correct += 1

    # Score blends word accuracy with overall string similarity so word order
    # and extra mumbling both count a little.
    word_acc = correct / len(exp_words)
    seq = difflib.SequenceMatcher(
        None,
        _normalise_for_match(expected, strip_accents=not accent_sensitive),
        heard_norm,
    ).ratio()
    score = int(round(100 * (0.75 * word_acc + 0.25 * seq)))
    score = max(0, min(100, score))

    if score >= 90:
        verdict = "Excellent — that was clear and accurate."
    elif score >= 70:
        missed = [m.expected for m in matches if not m.ok]
        verdict = ("Good. Watch: " + ", ".join(missed)) if missed else "Good and clear."
    elif score >= 40:
        verdict = "Getting there — several words came through unclearly."
    else:
        verdict = "Let's try that again, more slowly."

    return PronunciationScore(score, matches, verdict, expected, heard)


# ── the deck (persistence) ───────────────────────────────────────────────────

class VocabDeck:
    """SQLite-backed spaced-repetition deck. One row per card, per language.

    Follows the ``NewsSignatureCache`` pattern in briefing.py — an RLock around a
    single ``check_same_thread=False`` connection — because the tutor is driven
    from both the asyncio worker and the GUI thread.
    """

    def __init__(self, path: Any = None) -> None:
        # Resolved when called, not bound as a default when the class was
        # defined — otherwise DECK_PATH cannot be redirected (in tests, or by a
        # relocated config) and every store built with no path went to the
        # real one regardless.
        path = DECK_PATH if path is None else path
        self._lock = RLock()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._conn)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                language      TEXT NOT NULL,
                term          TEXT NOT NULL,
                translation   TEXT NOT NULL DEFAULT '',
                example       TEXT NOT NULL DEFAULT '',
                level         TEXT NOT NULL DEFAULT 'A1',
                ease          REAL NOT NULL DEFAULT 2.5,
                interval_days REAL NOT NULL DEFAULT 0,
                repetitions   INTEGER NOT NULL DEFAULT 0,
                lapses        INTEGER NOT NULL DEFAULT 0,
                due_at        TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL,
                UNIQUE(language, term)
            )""")
        self._conn.commit()

    # -- writes --
    def add_card(self, card: VocabCard) -> VocabCard:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO cards
                   (language, term, translation, example, level, ease,
                    interval_days, repetitions, lapses, due_at, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(language, term) DO UPDATE SET
                       translation=excluded.translation,
                       example=excluded.example,
                       level=excluded.level""",
                (card.language, card.term.strip(), card.translation.strip(),
                 card.example.strip(), card.level, card.ease, card.interval_days,
                 card.repetitions, card.lapses, card.due_at,
                 card.created_at or utc_stamp()))
            self._conn.commit()
            card.id = cur.lastrowid or self._card_id(card.language, card.term)
            return card

    def record_review(self, card: VocabCard) -> None:
        if card.id is None:
            card.id = self._card_id(card.language, card.term)
        with self._lock:
            self._conn.execute(
                """UPDATE cards SET ease=?, interval_days=?, repetitions=?,
                       lapses=?, due_at=? WHERE id=?""",
                (card.ease, card.interval_days, card.repetitions,
                 card.lapses, card.due_at, card.id))
            self._conn.commit()

    # -- reads --
    def due_cards(self, language: str, limit: int = 20,
                  now: datetime | None = None) -> list[VocabCard]:
        now = now or datetime.now(timezone.utc)
        stamp = _iso(now)
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM cards
                   WHERE language=? AND (due_at='' OR due_at<=?)
                   ORDER BY (due_at='') DESC, due_at ASC LIMIT ?""",
                (language, stamp, max(1, limit))).fetchall()
        return [_row_to_card(r) for r in rows]

    def all_cards(self, language: str | None = None) -> list[VocabCard]:
        with self._lock:
            if language:
                rows = self._conn.execute(
                    "SELECT * FROM cards WHERE language=? ORDER BY term",
                    (language,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM cards ORDER BY language, term").fetchall()
        return [_row_to_card(r) for r in rows]

    def get(self, card_id: int) -> VocabCard | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
        return _row_to_card(row) if row else None

    def stats(self, language: str) -> dict[str, Any]:
        now = _iso(datetime.now(timezone.utc))
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM cards WHERE language=?", (language,)).fetchone()[0]
            due = self._conn.execute(
                "SELECT COUNT(*) FROM cards WHERE language=? AND (due_at='' OR due_at<=?)",
                (language, now)).fetchone()[0]
            learned = self._conn.execute(
                "SELECT COUNT(*) FROM cards WHERE language=? AND repetitions>=2",
                (language,)).fetchone()[0]
        return {"total": int(total), "due": int(due), "learned": int(learned)}

    def _card_id(self, language: str, term: str) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM cards WHERE language=? AND term=?",
                (language, term.strip())).fetchone()
        return int(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_card(row: sqlite3.Row) -> VocabCard:
    return VocabCard(
        id=int(row["id"]), language=row["language"], term=row["term"],
        translation=row["translation"], example=row["example"], level=row["level"],
        ease=float(row["ease"]), interval_days=float(row["interval_days"]),
        repetitions=int(row["repetitions"]), lapses=int(row["lapses"]),
        due_at=row["due_at"], created_at=row["created_at"])


# ── the tutor ────────────────────────────────────────────────────────────────

@dataclass
class TutorSession:
    language: Language
    level: str
    immersion: bool                 # True = target-language only
    topic: str = ""

    def directive(self) -> str:
        """The instruction ORION follows while conducting the lesson.

        Handed to the existing model routing — it does not call a model here.
        """
        lang = self.language
        lines = [
            f"You are now MODE: LANGUAGE TUTOR for {lang.name} ({lang.native}), "
            f"pitched at CEFR level {self.level}.",
        ]
        if self.immersion:
            lines.append(
                f"IMMERSION: speak ONLY in {lang.name}. Do not translate into "
                "English unless the student explicitly asks 'in English'. Keep "
                f"vocabulary and grammar within {self.level}; simplify rather "
                "than switch languages.")
        else:
            lines.append(
                f"BILINGUAL: reply in {lang.name} first, then a short English "
                "gloss in parentheses so the student can follow.")
        lines.append(
            "Correct gently: restate the student's sentence the natural way "
            "rather than lecturing, and only flag one or two errors at a time.")
        lines.append(
            "End each of your turns with a small question that keeps the "
            "conversation going.")
        if self.topic:
            lines.append(f"Today's topic: {self.topic}.")
        return "\n".join(lines)


class LanguageTutor:
    """Sessions, the deck and pronunciation grading, behind one object."""

    def __init__(self, deck: VocabDeck | None = None) -> None:
        self.deck = deck or VocabDeck()
        self._active: TutorSession | None = None

    # -- sessions --
    def start_session(self, language: str | None = None, level: str | None = None,
                      immersion: bool = True, topic: str = "") -> TutorSession:
        lang = resolve_language(language)
        session = TutorSession(lang, resolve_level(level), bool(immersion), topic.strip())
        self._active = session
        self.seed(lang.code)          # ensure there is something to teach
        return session

    @property
    def active(self) -> TutorSession | None:
        return self._active

    def end_session(self) -> None:
        self._active = None

    # -- vocabulary --
    def add_word(self, term: str, translation: str, language: str | None = None,
                 example: str = "", level: str | None = None) -> VocabCard:
        lang = resolve_language(language)
        card = VocabCard(term=term.strip(), translation=translation.strip(),
                         language=lang.code, example=example.strip(),
                         level=resolve_level(level))
        return self.deck.add_card(card)

    def due(self, language: str | None = None, limit: int = 20) -> list[VocabCard]:
        return self.deck.due_cards(resolve_language(language).code, limit=limit)

    def grade(self, card_id: int, quality: int) -> VocabCard | None:
        card = self.deck.get(card_id)
        if card is None:
            return None
        schedule(card, quality)
        self.deck.record_review(card)
        return card

    def pronounce(self, expected: str, heard: str,
                  accent_sensitive: bool = False) -> PronunciationScore:
        return score_pronunciation(expected, heard, accent_sensitive=accent_sensitive)

    def stats(self, language: str | None = None) -> dict[str, Any]:
        return self.deck.stats(resolve_language(language).code)

    # -- seeding --
    def seed(self, language: str | None = None) -> int:
        """Add the starter deck for a language, idempotently. Returns how many
        NEW cards were added (0 on later calls)."""
        lang = resolve_language(language)
        starter = STARTER_DECKS.get(lang.code)
        if not starter:
            return 0
        existing = {c.term for c in self.deck.all_cards(lang.code)}
        added = 0
        for term, translation, example in starter:
            if term not in existing:
                self.deck.add_card(VocabCard(
                    term=term, translation=translation, language=lang.code,
                    example=example, level="A1"))
                added += 1
        return added


# ── helpers ──────────────────────────────────────────────────────────────────

def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ── starter decks ────────────────────────────────────────────────────────────
# A1 essentials — enough to hold a first conversation. Spanish is complete;
# other languages ship a smaller core and grow as the student adds words.

STARTER_DECKS: dict[str, list[tuple[str, str, str]]] = {
    "es": [
        ("hola", "hello", "Hola, ¿cómo estás?"),
        ("buenos días", "good morning", "Buenos días, señor."),
        ("buenas noches", "good night", "Buenas noches, hasta mañana."),
        ("gracias", "thank you", "Muchas gracias por tu ayuda."),
        ("por favor", "please", "Un café, por favor."),
        ("de nada", "you're welcome", "— Gracias. — De nada."),
        ("sí", "yes", "Sí, me gusta mucho."),
        ("no", "no", "No, gracias."),
        ("perdón", "sorry / excuse me", "Perdón, ¿dónde está el baño?"),
        ("¿cómo estás?", "how are you?", "Hola, ¿cómo estás hoy?"),
        ("bien", "well / fine", "Estoy muy bien, gracias."),
        ("¿cómo te llamas?", "what's your name?", "¿Cómo te llamas?"),
        ("me llamo", "my name is", "Me llamo Sam."),
        ("mucho gusto", "nice to meet you", "Mucho gusto, María."),
        ("agua", "water", "Quiero un vaso de agua."),
        ("comida", "food", "La comida está deliciosa."),
        ("casa", "house / home", "Voy a casa."),
        ("amigo", "friend", "Él es mi amigo."),
        ("tiempo", "time / weather", "No tengo tiempo hoy."),
        ("hoy", "today", "Hoy es lunes."),
        ("mañana", "tomorrow / morning", "Hasta mañana."),
        ("ahora", "now", "Ven aquí ahora."),
        ("¿dónde?", "where?", "¿Dónde vives?"),
        ("¿qué?", "what?", "¿Qué quieres?"),
        ("¿por qué?", "why?", "¿Por qué no vienes?"),
        ("porque", "because", "Porque estoy cansado."),
        ("quiero", "I want", "Quiero aprender español."),
        ("tengo", "I have", "Tengo una pregunta."),
        ("puedo", "I can / may I", "¿Puedo pasar?"),
        ("hablar", "to speak", "Quiero hablar español."),
        ("aprender", "to learn", "Me gusta aprender."),
        ("entender", "to understand", "No entiendo."),
        ("ayuda", "help", "Necesito ayuda, por favor."),
        ("grande", "big", "Es una casa grande."),
        ("pequeño", "small", "Un problema pequeño."),
        ("bueno", "good", "Es un buen libro."),
        ("malo", "bad", "Hace mal tiempo."),
        ("rápido", "fast / quickly", "Habla más rápido."),
        ("despacio", "slowly", "Habla despacio, por favor."),
        ("uno", "one", "Solo uno, gracias."),
    ],
    "fr": [
        ("bonjour", "hello / good day", "Bonjour, ça va?"),
        ("merci", "thank you", "Merci beaucoup."),
        ("s'il vous plaît", "please", "Un café, s'il vous plaît."),
        ("oui", "yes", "Oui, bien sûr."),
        ("non", "no", "Non, merci."),
        ("au revoir", "goodbye", "Au revoir, à demain."),
        ("comment ça va?", "how are you?", "Salut, comment ça va?"),
        ("je m'appelle", "my name is", "Je m'appelle Sam."),
    ],
    "de": [
        ("hallo", "hello", "Hallo, wie geht's?"),
        ("danke", "thank you", "Danke schön."),
        ("bitte", "please / you're welcome", "Ein Kaffee, bitte."),
        ("ja", "yes", "Ja, gern."),
        ("nein", "no", "Nein, danke."),
        ("tschüss", "bye", "Tschüss, bis morgen."),
    ],
    "it": [
        ("ciao", "hi / bye", "Ciao, come stai?"),
        ("grazie", "thank you", "Grazie mille."),
        ("per favore", "please", "Un caffè, per favore."),
        ("sì", "yes", "Sì, certo."),
        ("no", "no", "No, grazie."),
    ],
    "pt": [
        ("olá", "hello", "Olá, tudo bem?"),
        ("obrigado", "thank you", "Muito obrigado."),
        ("por favor", "please", "Um café, por favor."),
        ("sim", "yes", "Sim, claro."),
        ("não", "no", "Não, obrigado."),
    ],
}


__all__ = [
    "Language", "LANGUAGES", "DEFAULT_LANGUAGE", "CEFR", "DEFAULT_LEVEL",
    "resolve_language", "resolve_level",
    "VocabCard", "schedule",
    "PronunciationScore", "WordMatch", "score_pronunciation",
    "VocabDeck", "TutorSession", "LanguageTutor", "STARTER_DECKS",
]
