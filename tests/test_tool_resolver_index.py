"""
Tool resolver inverted index (Mark XXVI performance).

The resolver runs on every turn, on the qasync loop, before ORION can answer.
The original scorer re-tokenised all 136 tool declarations and rebuilt the
document-frequency table each time — **4.34 ms per turn** to recompute something
that only changes when a plugin adds a tool.

It is now backed by a cached inverted index: measured **0.093 ms per turn, 47x
faster**, and ``lexical_scores`` itself 248x faster.

The load-bearing property is that the index is a pure speed change. These tests
pin its scores to a straightforward, unoptimised implementation of the same
algorithm, so a future edit to the index cannot quietly change which tools ORION
can reach.

That oracle models the CURRENT algorithm, which now includes the curated
vocabulary expansion from ``tool_vocabulary`` (name tokens and vocabulary at
weight 2.0, description at 1.0). Changing the algorithm deliberately means
changing the oracle to match; the invariant is index == obvious implementation,
not scores-never-change.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import tool_resolver as tr  # noqa: E402
from orion_core.dispatch_schema import TOOL_DECLARATIONS  # noqa: E402


def reference_scores(query: str, declarations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """A straightforward, unoptimised implementation of the CURRENT scoring
    algorithm, kept as an oracle. Do not optimise it.

    It models name tokens plus curated vocabulary at weight 2.0, description
    tokens at 1.0, and the name-exactness tie-breaker — the same document and
    the same arithmetic the index uses.
    When the algorithm itself changes (as it did when vocabulary expansion was
    added), this oracle must be updated to match; the invariant it guards is
    that the cached inverted index computes the same numbers as the obvious
    implementation, not that the numbers never change.
    """
    q_terms = set(tr._tokens(query))
    if not q_terms:
        return {d["name"]: 0.0 for d in declarations}
    docs: dict[str, set[str]] = {}
    weighted: dict[str, set[str]] = {}
    df: dict[str, int] = {}
    for d in declarations:
        name = d["name"]
        strong = (set(tr._tokens(name.replace("_", " ")))
                  | set(tr._tokens(tr.vocabulary_for(name))))
        terms = strong | set(tr._tokens(d.get("description", "")))
        docs[name] = terms
        weighted[name] = strong
        for t in terms:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(declarations))
    scores: dict[str, float] = {}
    for d in declarations:
        name = d["name"]
        total = 0.0
        for t in q_terms:
            if t in docs[name]:
                idf = math.log((n + 1) / (df.get(t, 0) + 1)) + 1.0
                total += idf * (2.0 if t in weighted[name] else 1.0)
        if total:
            own = set(tr._tokens(name.replace("_", " ")))
            matched = len(q_terms & own)
            if own and matched:
                total += tr.NAME_EXACTNESS_BONUS * (matched / len(own))
        scores[name] = total
    return scores


REAL_QUERIES = [
    "remind me to revise neuroscience tomorrow morning",
    "what is my bank balance this month",
    "start a focus block for two hours",
    "take a screenshot and read the text on it",
    "search the web for transformer architectures",
    "send a discord message", "how am I sleeping lately",
    "build me a tool that renames files", "scan the network",
    "", "   ", "!!!", "study focus finance wellbeing chess",
]


def _fuzz(count: int = 300) -> list[str]:
    vocab = sorted({t for d in TOOL_DECLARATIONS
                    for t in tr._tokens(d["name"] + " " + d.get("description", ""))})
    rng = random.Random(11)
    return [" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6)))
            for _ in range(count)]


# ── the index must not change behaviour ──────────────────────────────────────

def test_scores_are_identical_to_an_unoptimised_implementation():
    for query in REAL_QUERIES + _fuzz():
        expected = reference_scores(query, TOOL_DECLARATIONS)
        actual = tr.lexical_scores(query, TOOL_DECLARATIONS)
        assert set(actual) == set(expected), query
        for name, value in expected.items():
            assert abs(actual[name] - value) < 1e-12, (
                "score drifted for %s on %r: %f vs %f" % (name, query, actual[name], value))


def test_an_empty_query_still_scores_every_tool_zero():
    scores = tr.lexical_scores("", TOOL_DECLARATIONS)
    assert len(scores) == len(TOOL_DECLARATIONS)
    assert set(scores.values()) == {0.0}


def test_a_query_naming_a_tool_ranks_it_first():
    for name in ("study", "focus", "chess"):
        if not any(d["name"] == name for d in TOOL_DECLARATIONS):
            continue
        scores = tr.lexical_scores(name, TOOL_DECLARATIONS)
        assert max(scores, key=lambda k: scores[k]) == name


def test_every_declared_tool_appears_in_the_result():
    scores = tr.lexical_scores("anything at all", TOOL_DECLARATIONS)
    assert {d["name"] for d in TOOL_DECLARATIONS} <= set(scores)


# ── cache correctness (the part that could go stale) ─────────────────────────

def test_the_index_rebuilds_when_a_tool_is_added():
    """A plugin adding a tool at runtime must not be invisible to the resolver."""
    base = list(TOOL_DECLARATIONS)
    tr.reset_index()
    tr.lexical_scores("kumquat", base)                      # builds + caches
    extended = base + [{"name": "kumquat_tool",
                        "description": "handles kumquat harvesting"}]
    scores = tr.lexical_scores("kumquat", extended)
    assert "kumquat_tool" in scores, "the cached index went stale"
    assert scores["kumquat_tool"] > 0


def test_the_index_rebuilds_when_a_tool_is_removed():
    base = list(TOOL_DECLARATIONS)
    tr.lexical_scores("study", base)
    trimmed = base[:-1]
    scores = tr.lexical_scores("study", trimmed)
    assert len(scores) == len(trimmed)


def test_the_index_rebuilds_when_a_tool_is_renamed():
    base = [dict(d) for d in TOOL_DECLARATIONS]
    tr.lexical_scores("study", base)
    renamed = [dict(d) for d in base]
    renamed[0] = dict(renamed[0], name="renamed_probe_tool")
    scores = tr.lexical_scores("renamed probe", renamed)
    assert "renamed_probe_tool" in scores


def test_reset_index_forces_a_rebuild():
    tr.lexical_scores("study", TOOL_DECLARATIONS)
    assert tr._INDEX is not None
    tr.reset_index()
    assert tr._INDEX is None
    assert tr.lexical_scores("study", TOOL_DECLARATIONS)


def test_the_fingerprint_distinguishes_declaration_sets():
    base = list(TOOL_DECLARATIONS)
    assert tr._fingerprint(base) == tr._fingerprint(list(base))
    assert tr._fingerprint(base) != tr._fingerprint(base[:-1])


def test_a_duplicate_tool_name_does_not_crash_the_index():
    doubled = list(TOOL_DECLARATIONS) + [dict(TOOL_DECLARATIONS[0])]
    scores = tr.lexical_scores("study", doubled)
    assert isinstance(scores, dict) and scores


def test_a_declaration_without_a_description_is_handled():
    decls = [{"name": "bare_tool"}, {"name": "other", "description": "does things"}]
    tr.reset_index()
    scores = tr.lexical_scores("bare tool", decls)
    assert scores["bare_tool"] > 0


# ── the speed it exists for ──────────────────────────────────────────────────

def test_scoring_is_far_cheaper_than_rebuilding_every_call():
    """Relative, not absolute: a ratio holds on a loaded machine where a
    millisecond budget would flake."""
    queries = REAL_QUERIES[:6] + _fuzz(20)

    tr.reset_index()
    tr.lexical_scores("warm", TOOL_DECLARATIONS)
    start = time.perf_counter()
    for i in range(120):
        tr.lexical_scores(queries[i % len(queries)], TOOL_DECLARATIONS)
    indexed = (time.perf_counter() - start) / 120

    start = time.perf_counter()
    for i in range(120):
        reference_scores(queries[i % len(queries)], TOOL_DECLARATIONS)
    naive = (time.perf_counter() - start) / 120

    assert indexed < naive / 10, (
        "the index is barely helping: %.3f ms vs %.3f ms — has it stopped "
        "caching?" % (indexed * 1000, naive * 1000))


def test_a_full_resolve_stays_well_inside_a_turn():
    state = tr.ResolverState()
    tr.reset_index()
    tr.resolve("warm", state, TOOL_DECLARATIONS)
    start = time.perf_counter()
    for i in range(100):
        tr.resolve(REAL_QUERIES[i % len(REAL_QUERIES)], state, TOOL_DECLARATIONS)
    per_call = (time.perf_counter() - start) * 1000 / 100
    # It measured 0.093 ms; 3 ms is loose enough for any machine but would catch
    # a regression to the 4.34 ms rebuild-every-time behaviour.
    assert per_call < 3.0, "resolve() costs %.2f ms per turn" % per_call


def test_the_measured_justification_stays_in_the_source():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "tool_resolver.py"
           ).read_text(encoding="utf-8")
    assert "4.34 ms" in src and "inverted index" in src
