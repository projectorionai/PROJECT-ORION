# Retrieval — decision record

Three subsystems in ORION answer "find me the thing that matches this": the tool
resolver, persistent memory, and the knowledge graph. All three were measured
and all three were badly broken, each in a different way, and none of it was
visible because **a retrieval failure looks like a mediocre answer, not like an
error**.

| subsystem | before | after |
|---|---|---|
| tool resolver (recall, held-out) | 72.7% | 100% |
| memory (recall, held-out) | **11.8%** | **88.2%** |
| knowledge graph (recall) | **1/6** | **6/6** |

See also `docs/ORION_TOOL_ROUTING.md` for the resolver, which has its own record.

---

## Memory — FTS5's implicit AND

`OrionMemoryMatrix` built its query as `' '.join(tokens)`. FTS5 treats
space-separated terms as an implicit **AND**, so a question needed *every one of
its words* to appear in one stored row.

Measured against 34 ordinary questions about 16 stored facts: **11.8% recall.**
Only bare single keywords worked. ORION could be told something and then be
unable to retrieve it when asked normally — memory was effectively write-only.

Three changes, each measured on its own:

| change | effect |
|---|---|
| **OR instead of implicit AND**, ranked by FTS5 BM25 (`ORDER BY rank`) | 33% → 83% |
| **Porter stemming** so "supervisor" matches "supervises" | 83% → 92% |
| **Stopword removal** | recall unchanged; sharply less ranking noise |

Stopwords are worth keeping in the design even though they did not move recall:
"when is my exam" returned the right fact *plus two unrelated ones* until "is"
and "my" were dropped. Fewer irrelevant rows reach the model.

**Prefix matching was measured and made no difference at all**, so it was not
added.

The stopword list is deliberately *not* shared with `tool_resolver._STOP`. That
list drops command verbs ("run", "set", "show") because a tool query is an
instruction; in a memory question those verbs are often the point — "what did I
**set** the rate to". Two lists, two jobs.

An all-stopword query keeps its stopwords rather than searching for nothing.

## Knowledge graph — a search index that had never been used

`semantic_retrieve` had **no test coverage** and was broken four ways at once.
The worst was invisible:

```sql
SELECT id, source_type, title, text, at, metadata_json
FROM events_fts JOIN events ON events_fts.rowid = events.rowid
```

`source_type`, `title`, `text` and `at` exist in **both** tables. Every query
raised `ambiguous column name`, straight into a bare `except sqlite3.Error:
rows = []`. **The FTS index was never once consulted in the entire life of the
method.** Every call silently fell through to a LIKE fallback that asked for the
whole question as a substring — so that could not match either.

Fixed: qualified columns, OR semantics, Porter stemming, per-term fallback.
**1/6 → 6/6.**

## The LIKE fallbacks

Both subsystems degraded to `LIKE '%<the entire query>%'` — asking for the whole
question to appear verbatim inside one stored value. That could essentially
never match anything a person actually said. Both now match each meaningful term
separately, capped at six terms so the OR chain cannot turn into a scan.

## The lesson worth keeping

The SQL bugs were easy. The reason they survived is the exception handler.

A swallowed error turned a **total failure** into something that looked like
mild under-performance, and nothing in the system distinguished "search found
nothing" from "search has never worked". So:

* `knowledge_graph.semantic_retrieve` now logs when FTS fails.
* `OrionMemoryMatrix` reports a broken index **once per index per session** —
  once, because a fault that fires on every query would drown the log it is
  trying to appear in.
* `test_knowledge_graph_retrieval.py` asserts the *unqualified* query still
  raises, so the bug cannot be reintroduced as "it looks fine".

## Migrations

Both indexes are FTS5 **external content** tables — every byte lives in
`intelligence`, `episodes` and `events`. Upgrading the tokenizer drops the index
and issues `'rebuild'`, which regenerates it from the content tables. No data is
at risk.

Verified on a copy of the real database: **1,283 facts and 2,905 episodes
preserved exactly, 28 ms one-time, 1 ms on subsequent opens.** The migration is
idempotent and a failure is caught — an unstemmed index still searches, just
slightly less well, and that is never a reason to refuse to start.

## Known limits

Lexical retrieval cannot close a semantic gap. These are honest remaining
misses, not bugs to be patched with a synonym table:

* **"bike" vs "bicycle"** — a true synonym.
* **"what medication should I avoid" vs "allergic to ibuprofen"** — requires
  knowing ibuprofen is a medication.
* **"pay" vs "payment"** — Porter does not conflate these.

Closing them properly needs embeddings, not more string rules. Curating synonyms
for arbitrary user facts would not generalise and was deliberately not done.

## Embeddings (Mark XXXI)

`orion_core/semantic.py` runs a sentence-embedding model locally — BAAI
**bge-small-en-v1.5**, int8 ONNX, 34 MB, 384-d, onnxruntime on two CPU threads.
It never loads on its own: ORION warms it 25 s after start-up in a worker
thread, so tests and machines without the model behave exactly as above.

The benchmark was made harder first. With 16 facts, "found in the top 8" is
half the database, so 106 look-alike distractors were added ("Tom is allergic
to cats", "Alex supports Aston Villa") and **top-1** became the measure.

| memory, 122 facts | dev top-1 | holdout top-1 | unrelated questions answered |
|---|---|---|---|
| words (FTS5 BM25) | 94.1% | 82.4% | 2 / 5 |
| meaning alone (bge-small) | 82.4% | 94.1% | 0 / 5 |
| reciprocal-rank fusion | 88.2% | 94.1% | 0 / 5 |
| **score fusion** | **100%** | **100%** | **0 / 5** |

all-MiniLM-L6-v2 was measured too and was weaker (holdout 88.2% alone).
Rank fusion was a mistake worth recording: it throws away *how much* closer a
meaning match is. "How long do clients have to pay" scored 0.725 against the
payment terms and 0.645 against the client rate, and the word "clients" still
won. **Score fusion** keeps the cosine as the score and adds a small bonus for
the word rank (`0.03 / (rank + 1)`, stable from 0.02 to 0.05), trusts a
meaning-only match above 0.58, and **vetoes** a word match whose meaning is
below 0.45 — which is what removed "boiling POINT → dentist apPOINTment".

The three misses above all resolve: "bike"/"bicycle", "painkiller"/"ibuprofen"
and "pay"/"payment" are close in meaning even where no word is shared.

The same encoder now also serves:

| where | before | after |
|---|---|---|
| library (ingested documents), paraphrased questions | 6/10 top-1 | 10/10 |
| knowledge graph, paraphrases (originals unchanged 6/6) | 3/6 | 4/6 |
| tool routing, 40 unseen requests, recall within 25 tools | 97.5% | 100% |
| proactive recall: typed request brings the fact that bears on it | — | 11/12, 1/12 padded |

Proactive recall (`MemoryAgent.relevant_note`) attaches up to three stored
facts to a typed request, by meaning alone at 0.60 — or 0.55 for health facts
when the request itself is about health or food ("I've got a headache, what
should I take" brings "allergic to ibuprofen"). A z-score against all memories
was tried as the threshold and separated worse at every cut.

Operational notes: encode shortest-first (a batch pads to its longest text;
real memory rows went from 60 ms to 13 ms each), vectors are float16 BLOBs
keyed by a digest of the text they came from, and every store re-embeds only
what changed. The user's 4,408 real rows index once, in the background, in
~45 s.

## Files

| file | role |
|---|---|
| `orion_core/memory.py` | `_sanitise_fts_query`, `_like_terms`, `_ensure_stemmed_index`, `_report_fts_fault` |
| `orion_core/semantic.py` | the local encoder, vector BLOBs, `fuse` (score fusion) |
| `orion_core/memory.py` (Mark XXXI) | `_with_meaning`, `index_semantics`, `relevant` / `relevant_note` |
| `orion_core/fact_memory.py` | learning lasting facts from what the user says |
| `orion_core/knowledge_graph.py` | `semantic_retrieve`, `_ensure_stemmed_events_index` |
| `tests/data/memory_recall_cases.py` | 16 facts, 34 questions, dev/holdout split |
| `tests/test_memory_recall.py` | recall gates, injection safety, migration |
| `tests/test_knowledge_graph_retrieval.py` | the coverage whose absence let this survive |
