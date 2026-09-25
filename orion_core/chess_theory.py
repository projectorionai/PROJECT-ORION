"""
Teaching ORION real opening theory, instead of making him rediscover it.

  "Double check ORION actually has his own chess brain and DOES NOT use the
   engine ... if he doesn't use the engine, can you train him to be extremely
   good at chess with python code like stockfish? Teach him all the openings,
   theory, midgames, endgames."

He does have his own brain (chess_brain.py — negamax with alpha-beta,
quiescence, piece-square tables and an opening book he writes from his own
games). It never touches Stockfish. What it lacked was a HEAD START: the book
began empty, so for the first few hundred games he played the opening from
first principles and lost games a booked engine would not.

Centuries of opening theory already exist. This seeds his book with the main
lines a club player is expected to know — as both colours — so from move one he
plays principled, tested chess instead of working it out. It is exactly how a
human gets good quickly: not by re-deriving the Ruy Lopez, but by learning it.

How the seeding works
---------------------
Each line is a sequence of moves in SAN (how theory is actually written). Every
position along a line records the move that continues it, credited as a strong
result so it clears the book's confidence bar (BookEntry needs a few games and
a winning record before it is trusted). Because the book is keyed on the
position, transpositions between openings share the same entries automatically —
learning the Queen's Gambit also teaches every move order that reaches it.

Seeded entries are marked so ORION's own hard-won experience is never
overwritten by theory, and theory is never re-seeded on top of games he has
actually played.
"""

from __future__ import annotations

from typing import Any

#: The repertoire, in SAN. Chosen for soundness and breadth rather than
#: fashion: a coherent, principled answer to everything a club opponent plays,
#: for both colours. SAN because that is the language theory is written in and
#: it is far less error-prone to transcribe than UCI.
REPERTOIRE: dict[str, list[str]] = {
    # ── 1.e4 e5 — the open games ─────────────────────────────────────────────
    "Ruy Lopez, Morphy":       ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4",
                                 "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O"],
    "Ruy Lopez, Berlin":       ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6", "O-O",
                                 "Nxe4", "d4", "Nd6", "Bxc6", "dxc6", "dxe5", "Nf5"],
    "Italian, Giuoco Piano":   ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3",
                                 "Nf6", "d3", "d6", "O-O", "O-O"],
    "Italian, Two Knights":    ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3",
                                 "Be7", "O-O", "O-O"],
    "Scotch Game":             ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4",
                                 "Nf6", "Nc3", "Bb4", "Nxc6", "bxc6"],
    "Petrov Defence":          ["e4", "e5", "Nf3", "Nf6", "Nxe5", "d6", "Nf3",
                                 "Nxe4", "d4", "d5"],
    # ── 1.e4 c5 — the Sicilian ───────────────────────────────────────────────
    "Sicilian, Najdorf":       ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4",
                                 "Nf6", "Nc3", "a6", "Be2", "e5"],
    "Sicilian, Classical":     ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4",
                                 "Nf6", "Nc3", "d6"],
    "Sicilian, Alapin":        ["e4", "c5", "c3", "d5", "exd5", "Qxd5", "d4",
                                 "Nf6", "Nf3", "e6"],
    # ── 1.e4 other replies ───────────────────────────────────────────────────
    "French, Winawer":         ["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5",
                                 "c5", "a3", "Bxc3+", "bxc3", "Ne7"],
    "French, Classical":       ["e4", "e6", "d4", "d5", "Nc3", "Nf6", "Bg5",
                                 "Be7", "e5", "Nfd7"],
    "Caro-Kann, Main":         ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4",
                                 "Bf5", "Ng3", "Bg6", "h4", "h6"],
    "Caro-Kann, Advance":      ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6"],
    "Scandinavian":            ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4",
                                 "Nf6", "Nf3", "c6"],
    "Pirc Defence":            ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "Nf3",
                                 "Bg7", "Be2", "O-O"],
    "Caro/Alekhine":           ["e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3",
                                 "dxe5", "Nxe5"],
    # ── 1.d4 d5 — the Queen's Gambit ─────────────────────────────────────────
    "Queen's Gambit Declined": ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5",
                                 "Be7", "e3", "O-O", "Nf3", "h6", "Bh4", "b6"],
    "Queen's Gambit Accepted": ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6", "e3",
                                 "e6", "Bxc4", "c5", "O-O", "a6"],
    "Slav Defence":            ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3",
                                 "dxc4", "a4", "Bf5"],
    # ── 1.d4 Nf6 — the Indian defences ───────────────────────────────────────
    "King's Indian, Classical": ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4",
                                  "d6", "Nf3", "O-O", "Be2", "e5", "O-O", "Nc6"],
    "Nimzo-Indian, Rubinstein": ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3",
                                  "O-O", "Bd3", "d5", "Nf3", "c5"],
    "Queen's Indian":          ["d4", "Nf6", "c4", "e6", "Nf3", "b6", "g3",
                                 "Ba6", "b3", "Bb4+"],
    "Grunfeld Defence":        ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5",
                                 "Nxd5", "e4", "Nxc3", "bxc3", "Bg7"],
    # ── flank openings ───────────────────────────────────────────────────────
    "English, Symmetrical":    ["c4", "c5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7"],
    "English, Reversed Sicilian": ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6", "g3", "d5"],
    "Reti Opening":            ["Nf3", "d5", "g3", "Nf6", "Bg2", "e6", "O-O", "Be7"],
    "London System":           ["d4", "d5", "Nf3", "Nf6", "Bf4", "e6", "e3", "c5", "c3", "Nc6"],
    "Catalan":                 ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2",
                                 "Be7", "Nf3", "O-O", "O-O", "dxc4"],
    # ── Mark XXII deepening: more mainline theory so ORION stays in book longer
    # 1.e4 e5
    "Ruy Lopez, Closed":       ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4",
                                 "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6",
                                 "c3", "O-O", "h3", "Na5", "Bc2", "c5", "d4", "Qc7"],
    "Ruy Lopez, Exchange":     ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6",
                                 "dxc6", "O-O", "f6", "d4", "exd4", "Nxd4", "c5"],
    "King's Gambit":           ["e4", "e5", "f4", "exf4", "Nf3", "g5", "h4",
                                 "g4", "Ne5", "Nf6", "d4", "d6", "Nxg4", "Nxe4"],
    "Vienna Game":             ["e4", "e5", "Nc3", "Nf6", "f4", "d5", "fxe5",
                                 "Nxe4", "Nf3", "Be7"],
    "Philidor Defence":        ["e4", "e5", "Nf3", "d6", "d4", "exd4", "Nxd4",
                                 "Nf6", "Nc3", "Be7"],
    # 1.e4 c5 — deeper Sicilians
    "Sicilian, Dragon":        ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4",
                                 "Nf6", "Nc3", "g6", "Be3", "Bg7", "f3", "O-O",
                                 "Qd2", "Nc6"],
    "Sicilian, Sveshnikov":    ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4",
                                 "Nf6", "Nc3", "e5", "Ndb5", "d6", "Bg5", "a6",
                                 "Na3", "b5"],
    "Sicilian, Taimanov":      ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4",
                                 "Nc6", "Nc3", "Qc7", "Be2", "a6"],
    "Sicilian, Scheveningen":  ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4",
                                 "Nf6", "Nc3", "e6", "Be2", "Be7", "O-O", "O-O"],
    # 1.e4 other
    "French, Tarrasch":        ["e4", "e6", "d4", "d5", "Nd2", "Nf6", "e5",
                                 "Nfd7", "Bd3", "c5", "c3", "Nc6", "Ne2", "cxd4",
                                 "cxd4"],
    "French, Advance":         ["e4", "e6", "d4", "d5", "e5", "c5", "c3", "Nc6",
                                 "Nf3", "Qb6"],
    "Caro-Kann, Exchange":     ["e4", "c6", "d4", "d5", "exd5", "cxd5", "Bd3",
                                 "Nc6", "c3", "Nf6"],
    # 1.d4 — deeper QG / Indians
    "Semi-Slav, Meran":        ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3",
                                 "e6", "e3", "Nbd7", "Bd3", "dxc4", "Bxc4", "b5"],
    "QGD, Tartakower":         ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5",
                                 "Be7", "e3", "O-O", "Nf3", "h6", "Bh4", "b6"],
    "King's Indian, Samisch":  ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4",
                                 "d6", "f3", "O-O", "Be3", "e5", "d5"],
    "Benoni, Modern":          ["d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3",
                                 "exd5", "cxd5", "d6", "e4", "g6", "Nf3", "Bg7"],
    "Benko Gambit":            ["d4", "Nf6", "c4", "c5", "d5", "b5", "cxb5",
                                 "a6", "bxa6", "Bxa6", "Nc3", "d6"],
    "Bogo-Indian":             ["d4", "Nf6", "c4", "e6", "Nf3", "Bb4+", "Bd2",
                                 "Qe7", "g3", "O-O", "Bg2", "d5"],
    "Dutch, Leningrad":        ["d4", "f5", "g3", "Nf6", "Bg2", "g6", "Nf3",
                                 "Bg7", "O-O", "O-O", "c4", "d6"],
    # flank / other
    "Trompowsky Attack":       ["d4", "Nf6", "Bg5", "Ne4", "Bf4", "c5", "f3",
                                 "Nf6", "d5"],
    "English, Four Knights":   ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6", "g3",
                                 "d5", "cxd5", "Nxd5"],
    "Nimzo-Larsen Attack":     ["b3", "e5", "Bb2", "Nc6", "e3", "Nf6", "Bb5",
                                 "Bd6"],
    # ── Sep 2026: the main lines a strong club player actually meets, deeper.
    # Every line is played out with python-chess by test_chess_theory, so a
    # typo cannot silently truncate it.
    # 1.e4 e5
    "Ruy Lopez, Marshall Attack": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6",
                                   "O-O", "Be7", "Re1", "b5", "Bb3", "O-O", "c3", "d5",
                                   "exd5", "Nxd5", "Nxe5", "Nxe5", "Rxe5", "c6"],
    "Ruy Lopez, Anti-Marshall": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6",
                                 "O-O", "Be7", "Re1", "b5", "Bb3", "O-O", "a4", "Bb7",
                                 "d3", "d6"],
    "Ruy Lopez, Breyer":       ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6",
                                 "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O",
                                 "h3", "Nb8", "d4", "Nbd7"],
    "Ruy Lopez, Zaitsev":      ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6",
                                 "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O",
                                 "h3", "Bb7", "d4", "Re8", "Nbd2", "Bf8"],
    "Italian, Giuoco Pianissimo": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6",
                                   "d3", "d6", "O-O", "a6", "a4", "Ba7"],
    "Two Knights, 5...Na5":    ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5",
                                 "exd5", "Na5", "Bb5+", "c6", "dxc6", "bxc6", "Be2",
                                 "h6", "Nf3", "e4", "Ne5", "Bd6"],
    "Evans Gambit":            ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "b4", "Bxb4",
                                 "c3", "Ba5", "d4", "d6", "Qb3", "Qd7"],
    "Scotch, Mieses":          ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Nf6",
                                 "Nxc6", "bxc6", "e5", "Qe7", "Qe2", "Nd5", "c4", "Ba6"],
    "Scotch, Classical":       ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Bc5",
                                 "Be3", "Qf6", "c3", "Nge7"],
    "Four Knights, Spanish":   ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "Bb5", "Bb4",
                                 "O-O", "O-O", "d3", "d6"],
    "Petrov, Classical":       ["e4", "e5", "Nf3", "Nf6", "Nxe5", "d6", "Nf3", "Nxe4",
                                 "d4", "d5", "Bd3", "Nc6", "O-O", "Be7"],
    # 1.e4 c5
    "Sicilian, Najdorf English Attack": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4",
                                         "Nf6", "Nc3", "a6", "Be3", "e5", "Nb3", "Be6",
                                         "f3", "Be7", "Qd2", "O-O"],
    "Sicilian, Najdorf 6.Bg5": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6",
                                 "Nc3", "a6", "Bg5", "e6", "f4", "Be7", "Qf3", "Qc7"],
    "Sicilian, Richter-Rauzer": ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6",
                                  "Nc3", "Nc6", "Bg5", "e6", "Qd2", "a6", "O-O-O", "Bd7"],
    "Sicilian, Accelerated Dragon": ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4",
                                     "g6", "Nc3", "Bg7", "Be3", "Nf6", "Bc4", "O-O"],
    "Sicilian, Kan":           ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "a6",
                                 "Bd3", "Nf6", "O-O", "Qc7"],
    "Sicilian, Rossolimo":     ["e4", "c5", "Nf3", "Nc6", "Bb5", "g6", "O-O", "Bg7",
                                 "Re1", "e5"],
    "Sicilian, Smith-Morra":   ["e4", "c5", "d4", "cxd4", "c3", "dxc3", "Nxc3", "Nc6",
                                 "Nf3", "d6", "Bc4", "e6", "O-O", "Nf6"],
    "Sicilian, Closed":        ["e4", "c5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7",
                                 "d3", "d6", "f4", "e6", "Nf3", "Nge7"],
    # 1.e4, other defences
    "French, Rubinstein":      ["e4", "e6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nd7",
                                 "Nf3", "Ngf6", "Nxf6+", "Nxf6"],
    "French, Exchange":        ["e4", "e6", "d4", "d5", "exd5", "exd5", "Nf3", "Nf6",
                                 "Bd3", "Bd6"],
    "Caro-Kann, Classical":    ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5",
                                 "Ng3", "Bg6", "h4", "h6", "Nf3", "Nd7", "h5", "Bh7",
                                 "Bd3", "Bxd3", "Qxd3", "e6"],
    "Caro-Kann, Two Knights":  ["e4", "c6", "Nc3", "d5", "Nf3", "Bg4", "h3", "Bxf3",
                                 "Qxf3", "e6"],
    "Scandinavian, Modern":    ["e4", "d5", "exd5", "Nf6", "d4", "Nxd5", "Nf3", "g6"],
    "Alekhine, Modern":        ["e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3", "Bg4",
                                 "Be2", "e6", "O-O", "Be7"],
    "Pirc, Austrian Attack":   ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "f4", "Bg7",
                                 "Nf3", "O-O"],
    "Modern Defence":          ["e4", "g6", "d4", "Bg7", "Nc3", "d6", "Be3", "a6"],
    # 1.d4
    "QGD, Exchange":           ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "cxd5", "exd5",
                                 "Bg5", "Be7", "e3", "O-O", "Bd3", "Nbd7", "Qc2", "Re8"],
    "QGD, Lasker":             ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7",
                                 "e3", "O-O", "Nf3", "h6", "Bh4", "Ne4"],
    "Slav, Exchange":          ["d4", "d5", "c4", "c6", "cxd5", "cxd5", "Nc3", "Nf6",
                                 "Bf4", "Nc6", "e3", "Bf5"],
    "Slav, Main Line":         ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "dxc4",
                                 "a4", "Bf5", "e3", "e6", "Bxc4", "Bb4", "O-O", "O-O"],
    "Nimzo-Indian, Classical": ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "Qc2", "O-O",
                                 "a3", "Bxc3+", "Qxc3", "b6"],
    "King's Indian, Mar del Plata": ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6",
                                     "Nf3", "O-O", "Be2", "e5", "O-O", "Nc6", "d5",
                                     "Ne7", "Ne1", "Nd7"],
    "Grunfeld, Exchange":      ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5",
                                 "e4", "Nxc3", "bxc3", "Bg7", "Nf3", "c5", "Rb1", "O-O",
                                 "Be2"],
    "Queen's Indian, Petrosian": ["d4", "Nf6", "c4", "e6", "Nf3", "b6", "a3", "Bb7",
                                  "Nc3", "d5", "cxd5", "Nxd5"],
    "Catalan, Open":           ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7",
                                 "Nf3", "O-O", "O-O", "dxc4", "Qc2", "a6", "a4", "Bd7"],
    "London vs King's Indian": ["d4", "Nf6", "Bf4", "g6", "e3", "Bg7", "Nf3", "O-O",
                                 "Be2", "d6", "h3"],
    "Dutch, Stonewall":        ["d4", "f5", "g3", "Nf6", "Bg2", "e6", "Nf3", "d5",
                                 "O-O", "Bd6", "c4", "c6"],
    "Colle System":            ["d4", "d5", "Nf3", "Nf6", "e3", "e6", "Bd3", "c5",
                                 "c3", "Nc6", "Nbd2", "Bd6", "O-O", "O-O"],
    # flank
    "English, Botvinnik":      ["c4", "e5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7",
                                 "e4", "d6", "Nge2"],
    "King's Indian Attack":    ["Nf3", "d5", "g3", "c6", "Bg2", "Bg4", "O-O", "Nd7",
                                 "d3", "Ngf6", "Nbd2", "e5"],
    "Bird Opening":            ["f4", "d5", "Nf3", "Nf6", "e3", "g6"],
}

#: Credited to a seeded move so it clears BookEntry.confidence() (>= 6 games to
#: be fully trusted) and score (> 0.45). A strong-but-not-perfect record: it is
#: theory, not a guarantee, and ORION's real results should still be able to
#: move it.
SEED_WINS = 7
SEED_DRAWS = 2

#: Marks a book that has had theory seeded, so it is not re-seeded every start
#: (which would slowly drown real experience under theory).
THEORY_MARKER = "__theory_seeded__"
#: Bumped when the repertoire grows, so a brain seeded with the smaller one is
#: topped up with the new lines (once) instead of never seeing them.
THEORY_VERSION = 2


def build_theory_book() -> dict[str, dict[str, Any]]:
    """The repertoire as a position→move book. Returns {} if python-chess is
    absent — theory is a bonus, never a dependency."""
    try:
        import chess
    except Exception:
        return {}
    from .chess_brain import BookEntry

    book: dict[str, dict[str, BookEntry]] = {}
    for line in REPERTOIRE.values():
        board = chess.Board()
        for san in line:
            try:
                move = board.parse_san(san)
            except Exception:
                break          # a typo in one line must not poison the rest
            key = " ".join(board.fen().split(" ")[:4])   # matches _book_key
            slot = book.setdefault(key, {})
            entry = slot.get(move.uci())
            if entry is None:
                slot[move.uci()] = BookEntry(wins=SEED_WINS, draws=SEED_DRAWS, losses=0)
            else:
                entry.wins += SEED_WINS
                entry.draws += SEED_DRAWS
            board.push(move)
    return book


def seed_into(brain: Any, force: bool = False) -> int:
    """Merge opening theory into *brain*'s book. Returns positions added.

    Additive and idempotent. Theory tops up a position ORION has not learned,
    and adds to one he has without erasing his own record — his experience and
    the book's are the same currency (wins/draws/losses), so they simply sum.
    Skipped if already seeded, unless forced.
    """
    if not force and getattr(brain, "_theory_seeded", False):
        return 0
    version = _seeded_version(brain)
    if not force and version >= THEORY_VERSION:
        brain._theory_seeded = True
        return 0
    theory = build_theory_book()
    if not theory:
        return 0
    # A brain seeded with an older repertoire already has that theory summed
    # into its record: add only what is new, or the old lines count twice.
    top_up = version > 0 and not force
    added = 0
    for key, moves in theory.items():
        slot = brain.book.setdefault(key, {})
        for uci, entry in moves.items():
            existing = slot.get(uci)
            if existing is None:
                slot[uci] = entry
                added += 1
            elif not top_up:
                existing.wins += entry.wins
                existing.draws += entry.draws
    # Zero-games sentinel entries under a reserved key: mark the book seeded
    # (and with which repertoire) without ever being playable — the key is
    # not a real position.
    from .chess_brain import BookEntry
    brain.book[THEORY_MARKER] = {"seeded": BookEntry(), f"v{THEORY_VERSION}": BookEntry()}
    brain._theory_seeded = True
    return added


def _seeded_version(brain: Any) -> int:
    """0 = never seeded; 1 = the original repertoire (unversioned marker)."""
    marker = getattr(brain, "book", {}).get(THEORY_MARKER)
    if not marker:
        return 0
    versions = [int(key[1:]) for key in marker if key[:1] == "v" and key[1:].isdigit()]
    return max(versions, default=1)


def coverage() -> dict[str, int]:
    """How much theory there is, for 'what openings do you know?'."""
    book = build_theory_book()
    return {"openings": len(REPERTOIRE), "book_positions": len(book)}


# ── explaining moves in plain chess terms ────────────────────────────────────
#
#   "ORION's chess analysis is confusing because it talks about centipawn
#    loss — it must talk in general chess terms: what move could've been
#    better, described, summarised and evaluated WHY."
#
# The engine's numbers decide only the GRADE (how much of the mover's winning
# chances a move gave away — the curve lichess uses, so a pawn lost in a won
# position is not graded like a pawn lost in a level one). Every REASON is read
# off the board: what the refutation actually captures, which pieces a reply
# forks or pins, whether the king's pawn cover was loosened, whether a piece was
# left where it can be taken for nothing. Nothing said to the player contains a
# centipawn figure.

PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}
GRADES = ("best", "excellent", "good", "inaccuracy", "mistake", "blunder")
GRADE_PHRASES = {
    "best": "the best move", "excellent": "an excellent move", "good": "a good move",
    "inaccuracy": "an inaccuracy", "mistake": "a mistake", "blunder": "a blunder",
}
#: Winning-chance drops (0..1) at which a move stops being each grade.
_GRADE_LIMITS = ((0.02, "excellent"), (0.05, "good"), (0.10, "inaccuracy"),
                 (0.15, "mistake"))


def win_chance(score: int) -> float:
    """The mover's chance of winning (0..1) from a search score in the
    mover's view. A forced mate is certain either way."""
    import math
    from .chess_brain import MATE_BOUND
    if score > MATE_BOUND:
        return 1.0
    if score < -MATE_BOUND:
        return 0.0
    return 1.0 / (1.0 + math.exp(-0.00368208 * score))


def grade(best_score: int, played_score: int, is_best: bool = False) -> str:
    """One of GRADES, from how much winning chance the move gave away."""
    if is_best:
        return "best"
    drop = win_chance(best_score) - win_chance(played_score)
    for limit, name in _GRADE_LIMITS:
        if drop < limit:
            return name
    return "blunder"


def _verb(subject: str, base: str) -> str:
    """'you fork' / 'I fork' / 'Black forks'."""
    if subject.lower() in {"you", "i", "we", "they"}:
        return base
    if base.endswith(("s", "sh", "ch", "x")):
        return base + "es"
    return base + "s"


def _possessive(subject: str) -> str:
    return {"you": "your", "i": "my", "we": "our"}.get(subject.lower(), f"{subject}'s")


def _object(subject: str) -> str:
    """'I' as an object is 'me': 'it lets me improve'."""
    return {"i": "me", "we": "us"}.get(subject.lower(), subject)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def move_prefix(board: Any) -> str:
    """'14.' before a White move, '14...' before a Black one."""
    return f"{board.fullmove_number}." if board.turn else f"{board.fullmove_number}..."


def move_label(board: Any, move: Any) -> str:
    """'14. Nf3' or '14... Nf6' — a move as a player would write it."""
    return f"{move_prefix(board)} {board.san(move)}"


def san_line(board: Any, moves: list, limit: int = 4) -> str:
    """A line in proper notation — '14. d4 exd4 15. Nxd4'."""
    probe = board.copy(stack=False)
    parts = []
    for index, move in enumerate(moves[:limit]):
        if move is None or not probe.is_legal(move):
            break
        san = probe.san(move)
        if probe.turn:
            parts.append(f"{probe.fullmove_number}. {san}")
        elif index == 0:
            parts.append(f"{probe.fullmove_number}... {san}")
        else:
            parts.append(san)
        probe.push(move)
    return " ".join(parts)


def _describe_square(board: Any, square: int, owner: str | None = None) -> str:
    """'the knight on e5', or 'your knight on e5' when *owner* is given."""
    import chess
    piece = board.piece_type_at(square)
    name = PIECE_NAMES.get(piece or 0, "piece")
    if piece == chess.KING:
        return f"{_possessive(owner)} king" if owner else "the king"
    article = _possessive(owner) if owner else "the"
    return f"{article} {name} on {chess.square_name(square)}"


def _material_words(pieces: list[int]) -> str:
    """[2, 1, 1] -> 'a knight and two pawns'."""
    if not pieces:
        return "nothing"
    counts: dict[int, int] = {}
    for piece in pieces:
        counts[piece] = counts.get(piece, 0) + 1
    numbers = {2: "two", 3: "three", 4: "four", 5: "five"}
    parts = []
    for piece in sorted(counts, key=lambda p: -_VALUES[p]):
        n = counts[piece]
        name = PIECE_NAMES[piece]
        parts.append(f"a {name}" if n == 1 else f"{numbers.get(n, str(n))} {name}s")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _line_material(board: Any, line: list, max_plies: int = 8) -> tuple[list, list, int]:
    """Play *line* from *board* until it goes quiet. Returns (pieces the side to
    move loses, pieces it wins, net material for it) — counted at the last
    point where the next move is not a capture, so a line that stops halfway
    through an exchange is not read as a win."""
    from .chess_brain import _see
    probe = board.copy(stack=False)
    mover = board.turn
    moves = []
    for move in line[:max_plies]:
        if move is None or not probe.is_legal(move):
            break
        moves.append(move)
        probe.push(move)
    # The line is only as long as the search was deep, so it can stop in the
    # middle of an exchange. Finish THAT exchange — recaptures on the square
    # of the last capture that do not lose material — and nothing else.
    for _ in range(6):
        if not moves:
            break
        square = moves[-1].to_square
        recaptures = [(m, _see(probe, m)) for m in
                      probe.generate_legal_captures(to_mask=1 << square)]
        recaptures = [pair for pair in recaptures if pair[1] >= 0]
        if not recaptures:
            break
        best = max(recaptures, key=lambda pair: pair[1])[0]
        moves.append(best)
        probe.push(best)
    probe = board.copy(stack=False)
    lost: list[int] = []
    won: list[int] = []
    settled = ([], [], 0)
    for index, move in enumerate(moves):
        victim = probe.piece_type_at(move.to_square)
        if victim is None and probe.is_en_passant(move):
            victim = 1
        if victim:
            (won if probe.turn == mover else lost).append(victim)
        if move.promotion:
            (won if probe.turn == mover else lost).append(-move.promotion)
        probe.push(move)
        following = moves[index + 1] if index + 1 < len(moves) else None
        if following is None or not probe.is_capture(following):
            gain = sum(_VALUES[p] if p > 0 else _VALUES[-p] - 100 for p in won)
            loss = sum(_VALUES[p] if p > 0 else _VALUES[-p] - 100 for p in lost)
            # Like for like cancels: "a knight and a pawn for a knight" is a pawn.
            won_left = [p for p in won if p > 0]
            lost_left = [p for p in lost if p > 0]
            for piece in list(won_left):
                if piece in lost_left:
                    won_left.remove(piece)
                    lost_left.remove(piece)
            settled = (lost_left, won_left, gain - loss)
    return settled


def _hanging(board: Any, square: int) -> bool:
    """Can the side to move win the piece on *square* outright?"""
    from .chess_brain import _see
    for move in board.generate_legal_captures(to_mask=1 << square):
        if _see(board, move) > 0:
            return True
    return False


def _motifs(before: Any, move: Any, owner_of_targets: str) -> list[tuple[str, str]]:
    """What *move* does tactically, read off the position after it, as
    (verb, object) pairs so the caller can say who does it: "it forks …",
    "I fork …", "you fork …". *owner_of_targets* names the side whose pieces
    are attacked, for "your knight on e5"."""
    import chess
    after = before.copy(stack=False)
    after.push(move)
    colour = before.turn
    to = move.to_square
    piece = after.piece_type_at(to)
    found: list[tuple[str, str]] = []
    if after.is_checkmate():
        return [("mate", "")]
    enemy = not colour
    # A fork: two or more targets worth attacking at once — the king, or a
    # piece worth more than the attacker, or one nobody defends.
    targets = []
    for square in chess.scan_forward(after.attacks_mask(to) & after.occupied_co[enemy]):
        kind = after.piece_type_at(square)
        if kind == chess.KING:
            targets.insert(0, square)
        elif kind != chess.PAWN and (_VALUES[kind] > _VALUES[piece]
                                     or not after.is_attacked_by(enemy, square)):
            targets.append(square)
    if len(targets) >= 2:
        named = [_describe_square(after, s, owner_of_targets) for s in targets[:2]]
        found.append(("fork", f"{named[0]} and {named[1]}"))
    elif targets and after.piece_type_at(targets[0]) != chess.KING \
            and not after.is_attacked_by(enemy, to):
        # A single threat the opponent now has to answer.
        found.append(("attack", _describe_square(after, targets[0], owner_of_targets)))
    # A pin against the king that this move creates.
    for square in chess.scan_forward(after.occupied_co[enemy] & ~after.kings & ~after.pawns):
        if after.is_pinned(enemy, square) and not before.is_pinned(enemy, square) \
                and after.pin(enemy, square) & (1 << to):
            found.append(("pin", f"{_describe_square(after, square, owner_of_targets)} "
                                 "to the king"))
            break
    # A discovered attack: moving this piece opened a line for another.
    if not found:
        for square in chess.scan_forward(after.occupied_co[colour] & ~(1 << to)
                                         & (after.bishops | after.rooks | after.queens)):
            gained = (after.attacks_mask(square) & ~before.attacks_mask(square)
                      & after.occupied_co[enemy] & (after.queens | after.rooks | after.kings))
            if gained:
                target = gained.bit_length() - 1
                if after.piece_type_at(target) == chess.KING:
                    found.append(("unleash", "a discovered check"))
                else:
                    found.append(("unleash", "a discovered attack on "
                                  + _describe_square(after, target, owner_of_targets)))
                break
    if after.is_check() and not any(verb in ("fork", "unleash") and "king" in obj
                                    or obj == "a discovered check" for verb, obj in found):
        found.append(("give", "check"))
    # A mate threat: were it this side's move again, could it mate at once?
    if not after.is_check():
        threat = _mate_threat(after, colour)
        if threat is not None:
            found.append(("threaten", f"mate with {threat}"))
    return found


def _mate_threat(board: Any, side: bool) -> str | None:
    """If *side* could mate in one were it their move, that move in SAN."""
    import chess
    probe = board.copy(stack=False)
    if probe.turn != side:
        if probe.is_check():
            return None
        probe.push(chess.Move.null())
    for move in list(probe.generate_legal_moves()):
        san = probe.san(move)
        probe.push(move)
        mate = probe.is_checkmate()
        probe.pop()
        if mate:
            return san
    return None


def _say(subject: str, motif: tuple[str, str]) -> str:
    """Render a (verb, object) motif with a subject: 'it forks …'."""
    verb, obj = motif
    if verb == "mate":
        return "it is checkmate" if subject == "it" else f"{subject} {_verb(subject, 'deliver')} mate"
    return f"{subject} {_verb(subject, verb)} {obj}".rstrip()


def _virtues(before: Any, move: Any, line: list, score: int, mover: str,
             opponent: str, support: int | None = None) -> list[str]:
    """Why a move is good, in the order a coach would say it. *support* is
    how much better it scored than the move actually played: material won
    only further down the line is claimed when the scores bear it out."""
    import chess
    from .chess_brain import mate_distance
    after = before.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return ["it is checkmate"]
    notes: list[str] = []
    colour = before.turn
    mate = mate_distance(score)
    if mate is not None and mate > 0:
        notes.append("it forces mate" + (f" in {mate}" if mate > 1 else ""))
    # Parrying a mate threat.
    threat = _mate_threat(before, not colour)
    if threat is not None and _mate_threat(after, not colour) is None:
        notes.append(f"it stops the threat of {threat}")
    if before.is_capture(move):
        # What the capture itself nets, by static exchange on that square.
        from .chess_brain import _see
        victim = before.piece_type_at(move.to_square) or chess.PAWN
        exchange = _see(before, move)
        if exchange >= _VALUES[victim] - 30:
            notes.append(f"it wins the {PIECE_NAMES[victim]} on "
                         f"{chess.square_name(move.to_square)} for nothing")
        elif exchange >= 70:
            notes.append("it comes out ahead in the exchange")
    else:
        lost, won, net = _line_material(before, line or [move])
        # Material won further down the line is only claimed when the scores
        # say the mover really ends up that much better off.
        if net >= 70 and won and support is not None \
                and support >= 0.6 * min(net, 400) and score >= 0.6 * min(net, 400):
            what = _material_words(won) + (f" for {_material_words(lost)}" if lost else "")
            notes.append(f"it leads to winning {what} ({san_line(before, line, 5)})")
    for motif in _motifs(before, move, opponent):
        if mate is not None and mate > 0 and motif[0] == "threaten":
            continue                      # already said: it forces mate
        text = _say("it", motif)
        if text not in notes:
            notes.append(text)
    piece = before.piece_type_at(move.from_square)
    victim = before.piece_type_at(move.to_square)
    # Rescuing a piece that was about to be lost.
    if not notes:
        probe = before.copy(stack=False)
        probe.push(chess.Move.null())
        endangered = [sq for sq in chess.scan_forward(before.occupied_co[colour] & ~before.kings
                                                      & ~before.pawns)
                      if _hanging(probe, sq)]
        if endangered:
            safe = after.copy(stack=False)
            if safe.turn == colour:
                safe.push(chess.Move.null())
            still = {sq for sq in chess.scan_forward(safe.occupied_co[colour] & ~safe.kings
                                                     & ~safe.pawns) if _hanging(safe, sq)}
            rescued = [sq for sq in endangered if (move.to_square if sq == move.from_square
                                                   else sq) not in still]
            if rescued:
                notes.append(f"it saves {_describe_square(before, rescued[0])}, "
                             "which was under attack")
    if before.is_castling(move):
        notes.append("it castles, tucking the king away and connecting the rooks")
    elif victim and not notes:
        notes.append(f"it takes {_describe_square(before, move.to_square)}")
    home = chess.BB_RANK_1 if colour else chess.BB_RANK_8
    if piece in (chess.KNIGHT, chess.BISHOP) and (1 << move.from_square) & home \
            and before.fullmove_number <= 15:
        notes.append(f"it develops the {PIECE_NAMES[piece]}")
    centre = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
    if piece == chess.PAWN and (1 << move.to_square) & centre and before.fullmove_number <= 15:
        notes.append("it stakes a claim in the centre")
    if piece == chess.PAWN and not move.promotion:
        enemy_pawns = before.pawns & before.occupied_co[not colour]
        file_mask = chess.BB_FILES[chess.square_file(move.to_square)]
        adjacent = 0
        f = chess.square_file(move.to_square)
        if f > 0:
            adjacent |= chess.BB_FILES[f - 1]
        if f < 7:
            adjacent |= chess.BB_FILES[f + 1]
        rank = chess.square_rank(move.to_square)
        ahead = 0
        for r in (range(rank + 1, 8) if colour else range(0, rank)):
            ahead |= chess.BB_RANKS[r]
        if not enemy_pawns & (file_mask | adjacent) & ahead:
            notes.append("it pushes a passed pawn closer to queening")
    if move.promotion:
        notes.append(f"it promotes to a {PIECE_NAMES[move.promotion]}")
    if piece == chess.ROOK and not before.pawns & chess.BB_FILES[chess.square_file(move.to_square)] \
            and chess.square_file(move.to_square) != chess.square_file(move.from_square):
        notes.append(f"it puts the rook on the open {chess.FILE_NAMES[chess.square_file(move.to_square)]}-file")
    heavy = before.queens | before.rooks | before.knights | before.bishops
    if piece == chess.KING and heavy.bit_count() <= 4 and not before.is_castling(move):
        notes.append("it brings the king into play, as a king should in the endgame")
    return notes


def _faults(before: Any, move: Any, reply_line: list, played_score: int,
            mover: str, opponent: str, cost: int = 10_000) -> list[str]:
    """What is wrong with a move, from the refutation the search found.

    *cost* is how much worse the move scored than the best one. A material
    loss is only blamed when the scores agree it is real — a line that seems
    to drop a knight in a move graded a mere inaccuracy is a line the search
    did not believe, and saying otherwise would be teaching something false."""
    import chess
    from .chess_brain import mate_distance
    after = before.copy(stack=False)
    after.push(move)
    colour = before.turn
    notes: list[str] = []
    reply = reply_line[0] if reply_line and after.is_legal(reply_line[0]) else None
    reply_text = f"{move_prefix(after)} {after.san(reply)}" if reply is not None else ""
    mate = mate_distance(played_score)
    if mate is not None and mate <= 0:
        if reply is not None:
            probe = after.copy(stack=False)
            probe.push(reply)
            if probe.is_checkmate():
                return [f"it allows {reply_text}, which is checkmate"]
        line = san_line(after, reply_line, 5)
        return [f"it walks into a forced mate ({line})" if line else "it walks into a forced mate"]
    if reply is not None:
        # The refutation, played out until the captures stop: the side
        # "moving" in that line is the opponent.
        lost, won, net = _line_material(after, reply_line)
        if net >= 70 and won and cost >= 0.6 * min(net, 400):
            parts = [f"after {reply_text}"]
            motifs = [m for m in _motifs(after, reply, mover) if m[0] != "give"]
            if motifs:
                parts.append(_say(opponent, motifs[0]))
            outcome = f"{opponent} {_verb(opponent, 'win')} {_material_words(won)}"
            if lost:
                outcome += f" for {_material_words(lost)}"
            notes.append(", ".join(parts) + (", and " if len(parts) > 1 else ", ") + outcome)
    moved_to = move.to_square
    moved_piece = after.piece_type_at(moved_to)
    if not notes and moved_piece not in (None, chess.KING) and _hanging(after, moved_to):
        notes.append(f"it leaves the {PIECE_NAMES[moved_piece]} on "
                     f"{chess.square_name(moved_to)} where it can be taken for nothing")
    if not notes and reply is not None:
        motifs = [m for m in _motifs(after, reply, mover) if m[0] != "give"]
        if motifs:
            notes.append(f"it allows {reply_text}, and {_say(opponent, motifs[0])}")
    if notes:
        return notes
    # Positional reasons.
    piece = before.piece_type_at(move.from_square)
    king = before.king(colour)
    back = 0 if colour else 7
    if king is not None and chess.square_rank(king) == back and piece == chess.PAWN:
        king_file = chess.square_file(king)
        pawn_file = chess.square_file(move.from_square)
        if (king_file >= 5 and pawn_file >= 5) or (king_file <= 2 and pawn_file <= 2):
            notes.append(f"it loosens the pawns in front of {_possessive(mover)} king")
    if piece in (chess.KING, chess.ROOK) and not before.is_castling(move) \
            and before.has_castling_rights(colour) and not after.has_castling_rights(colour):
        notes.append("it gives up the right to castle")
    home = chess.BB_RANK_1 if colour else chess.BB_RANK_8
    undeveloped = (before.knights | before.bishops) & before.occupied_co[colour] & home
    if before.fullmove_number <= 10 and undeveloped.bit_count() >= 2:
        if piece == chess.QUEEN:
            notes.append("it brings the queen out early, where it can be chased around "
                         "while the other pieces are still at home")
        elif piece in (chess.KNIGHT, chess.BISHOP) and not (1 << move.from_square) & home \
                and not before.is_capture(move):
            notes.append("it moves the same piece twice while others are still undeveloped")
    if reply_line:
        probe = after.copy(stack=False)
        for step in reply_line[:4]:
            if not probe.is_legal(step):
                break
            probe.push(step)
        if _passers(probe, not colour) > _passers(before, not colour):
            notes.append(f"it lets {_object(opponent)} get a passed pawn")
    return notes


def _passers(board: Any, colour: bool) -> int:
    import chess
    own = board.pawns & board.occupied_co[colour]
    enemy = board.pawns & board.occupied_co[not colour]
    count = 0
    for square in chess.scan_forward(own):
        f, r = chess.square_file(square), chess.square_rank(square)
        span = 0
        for ff in (f - 1, f, f + 1):
            if 0 <= ff < 8:
                span |= chess.BB_FILES[ff]
        ahead = 0
        for rr in (range(r + 1, 8) if colour else range(0, r)):
            ahead |= chess.BB_RANKS[rr]
        if not enemy & span & ahead:
            count += 1
    return count


def explain_move(board: Any, played: Any, review: dict[str, Any],
                 mover: str = "you", opponent: str = "I") -> dict[str, Any]:
    """A coach's verdict on *played* (made from *board*) given an engine
    *review* (see ChessBrain.review_move): the grade, and in plain words what
    it did, what was wrong with it, and what was better and why.

    Returns {"quality", "text", "drop", "better"}. *mover*/*opponent* are how
    the two sides are named: "you"/"I" when the user plays ORION."""
    best = review.get("best")
    best_score = int(review.get("best_score", 0))
    played_score = int(review.get("played_score", 0))
    is_best = best is None or played == best
    quality = grade(best_score, played_score, is_best)
    drop = max(0.0, win_chance(best_score) - win_chance(played_score))
    prefix = move_prefix(board)
    san = board.san(played)
    head = f"{prefix} {san} — {GRADE_PHRASES[quality]}"
    # A principal variation read back from the hash table is only reliable
    # to the depth actually searched; past that it is stale entries.
    depth = max(1, int(review.get("depth") or 1))
    best_line = list(review.get("best_line") or ([best] if best is not None else []))[:depth]
    reply_line = list(review.get("reply_line") or [])[:max(1, depth - 1)]
    cost = best_score - played_score
    if quality in ("best", "excellent"):
        line = best_line if is_best else [played] + reply_line
        virtues = _virtues(board, played, line, played_score, mover, opponent)
        text = head + (f": {'; '.join(virtues[:2])}." if virtues else ".")
        return {"quality": quality, "text": text, "drop": drop, "better": None}
    better_san = board.san(best) if best is not None else ""
    virtues = (_virtues(board, best, best_line, best_score, mover, opponent, cost)
               if best is not None else [])
    if quality == "good":
        text = head + "."
        if better_san and virtues:
            text += f" {better_san} was a touch stronger — {virtues[0]}."
        return {"quality": quality, "text": text, "drop": drop, "better": better_san or None}
    faults = _faults(board, played, reply_line, played_score, mover, opponent, cost)
    text = head + "."
    if faults:
        text += " " + _cap("; ".join(faults[:2])) + "."
    else:
        text += (f" It lets {_object(opponent)} improve {_possessive(opponent)} position "
                 "without any real counterplay.")
    if better_san:
        text += f" Better was {prefix} {better_san}"
        text += f": {'; '.join(virtues[:2])}." if virtues else "."
        idea = san_line(board, best_line, 4)
        if idea and len(best_line) > 1:
            text += f" (A likely continuation: {idea}.)"
    return {"quality": quality, "text": text, "drop": drop, "better": better_san or None}


def describe_intent(board: Any, move: Any, line: list, score: int,
                    mover: str = "I", opponent: str = "you") -> str:
    """A short 'why I played this' for a move the engine chose itself."""
    virtues = _virtues(board, move, line, score, mover, opponent)
    san = board.san(move)
    if virtues:
        return f"{san} — {virtues[0]}."
    return f"{san}."


def summarise_game(moments: list[dict[str, Any]], result: str,
                   sides: dict[str, str]) -> str:
    """The game in words: how each side played, and the two or three moves
    that decided it. *moments* are reviewed moves: {"mover": name, "quality",
    "drop", "text"}; *sides* maps a mover name to how it is addressed."""
    if not moments:
        return "There is nothing to summarise yet."
    lines = [f"Game summary — {result}." if result and result != "in progress"
             else "Game summary so far."]
    by_mover: dict[str, dict[str, int]] = {}
    for moment in moments:
        counts = by_mover.setdefault(moment["mover"], {})
        counts[moment["quality"]] = counts.get(moment["quality"], 0) + 1
    for name, counts in by_mover.items():
        strong = counts.get("best", 0) + counts.get("excellent", 0)
        errors = []
        for label in ("inaccuracy", "mistake", "blunder"):
            n = counts.get(label, 0)
            if n:
                plural = {"inaccuracy": "inaccuracies"}.get(label, label + "s")
                errors.append(f"{n} {label if n == 1 else plural}")
        total = sum(counts.values())
        who = sides.get(name, name)
        verdict = (f"{_cap(who)} played {total} move(s), {strong} of them best or excellent"
                   + (f", with {', '.join(errors)}." if errors else ", without a real slip."))
        lines.append(verdict)
    turning = sorted((m for m in moments if m["quality"] in ("inaccuracy", "mistake", "blunder")),
                     key=lambda m: -m["drop"])[:3]
    if turning:
        lines.append("The moments that decided it:")
        for moment in sorted(turning, key=lambda m: m.get("ply", 0)):
            lines.append(f"• {moment['text']}")
    else:
        lines.append("No real turning points — neither side gave anything away.")
    return "\n".join(lines)


__all__ = ["GRADES", "GRADE_PHRASES", "PIECE_NAMES", "REPERTOIRE", "SEED_DRAWS",
           "SEED_WINS", "THEORY_MARKER", "THEORY_VERSION", "build_theory_book",
           "coverage", "describe_intent", "explain_move", "grade", "move_label",
           "move_prefix", "san_line", "seed_into", "summarise_game", "win_chance"]
