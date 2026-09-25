"""
Memory recall evaluation — facts ORION stores, and how a person asks for them.

The questions are written the way someone actually speaks, not in the vocabulary
of the stored sentence. That is the whole point: FTS5 treats space-separated
terms as an implicit AND, so before this was fixed a natural question required
every one of its words to appear in one stored row, and retrieved nothing at all.

Split deterministically by index. DEV may be inspected while tuning retrieval;
HOLDOUT may not. Unlike the tool-vocabulary work there is no per-case curation
to overfit here — the fix is one algorithmic change — but the holdout is still
how the number gets reported honestly.

Each case is (question, the key_ref of the fact that answers it).

The people, places, sums and dates below are INVENTED. They stand in for the
kind of thing ORION is actually told, because a retrieval benchmark needs the
awkward shape of real personal facts — a medication, an address, a routine —
to be worth anything. It does not need anyone's real ones, and this file is
published: an earlier version carried a genuine allergy, home area, rent,
gym and named third parties, which is a personal dossier, not a test fixture.
Keep it fictional.
"""

from __future__ import annotations

#: (category, key_ref, stored value) — what ORION has been told over time.
FACTS: list[tuple[str, str, str]] = [
    ("personal", "bicycle",
     "Sam bought a Raleigh bicycle in March for commuting to campus"),
    ("personal", "allergy", "Sam is allergic to ibuprofen"),
    ("personal", "flat",
     "Sam lives in Northgate and pays 700 a month in rent"),
    ("personal", "car",
     "Sam passed the driving test in 2024 but does not own a car"),
    ("personal", "birthday", "Sam's mother's birthday is the 3rd of October"),
    ("personal", "gym", "Sam trains at FitZone on Mill Road, usually early mornings"),
    ("business", "client_rate",
     "The standard client rate for the studio is 450 pounds "
     "per short-form video"),
    ("business", "supplier",
     "The preferred supplier for merch printing is PrintWorks"),
    ("business", "invoice_terms",
     "Invoices for the agency are issued on 30 day payment terms"),
    ("business", "accountant",
     "The agency accountant is Whitfield and Co, contacted every quarter"),
    ("study", "exam_date", "The neural engineering exam is on the 14th of May"),
    ("study", "supervisor",
     "Dr Harper supervises the dissertation on cortical plasticity"),
    ("study", "module",
     "The hardest module this year is computational neuroscience, worth 40 credits"),
    ("study", "library",
     "The Eastgate Library closes at 9pm on weekdays during term time"),
    ("project", "game_stack",
     "Redshift is a browser game built on Three.js, run with python dev_server.py"),
    ("project", "orion_launch",
     "ORION is started from orion.py and its tests run under the venv interpreter"),
]

#: (question as a person would ask it, key_ref of the answering fact)
CASES: list[tuple[str, str]] = [
    # personal
    ("what bicycle did I buy", "bicycle"),
    ("when did I get the bike", "bicycle"),
    ("am I allergic to anything", "allergy"),
    ("what medication should I avoid", "allergy"),
    ("what is my rent", "flat"),
    ("where do I live", "flat"),
    ("do I have a car", "car"),
    ("can I drive", "car"),
    ("when is my mother's birthday", "birthday"),
    ("where do I train", "gym"),
    ("which gym do I go to", "gym"),
    # business
    ("how much do I charge clients", "client_rate"),
    ("what is my rate per video", "client_rate"),
    ("who prints our merch", "supplier"),
    ("which supplier do we use for printing", "supplier"),
    ("what are our payment terms", "invoice_terms"),
    ("how long do clients have to pay", "invoice_terms"),
    ("who is our accountant", "accountant"),
    # study
    ("when is my exam", "exam_date"),
    ("what date is the neural engineering exam", "exam_date"),
    ("who is my supervisor", "supervisor"),
    ("who supervises my dissertation", "supervisor"),
    ("what is my dissertation about", "supervisor"),
    ("which module is hardest", "module"),
    ("how many credits is computational neuroscience", "module"),
    ("when does the library close", "library"),
    # project
    ("what is Redshift built with", "game_stack"),
    ("how do I run the Redshift server", "game_stack"),
    ("how do I start ORION", "orion_launch"),
    ("how do I run ORION's tests", "orion_launch"),
    # single keyword probes — these must never regress
    ("ibuprofen", "allergy"),
    ("PrintWorks", "supplier"),
    ("dissertation", "supervisor"),
    ("Northgate", "flat"),
]


def dev_cases() -> list[tuple[str, str]]:
    """Even indices — may be inspected while tuning retrieval."""
    return [c for i, c in enumerate(CASES) if i % 2 == 0]


def holdout_cases() -> list[tuple[str, str]]:
    """Odd indices — never inspected while tuning; the honest score."""
    return [c for i, c in enumerate(CASES) if i % 2 == 1]


__all__ = ["CASES", "FACTS", "dev_cases", "holdout_cases"]
