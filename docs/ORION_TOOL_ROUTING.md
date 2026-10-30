# Tool routing — decision record

**Status:** resolver still OFF by default. Offline evidence now supports enabling
it; real-turn evidence is being collected. **Do not set `ORION_TOOL_RESOLVER=1`
until `diagnostics(action='routing')` reports SAFE TO ENABLE.**

---

## The problem

ORION declares **136 tools — about 31,600 tokens** of schema. That surface is
paid once per Gemini Live session (tools are fixed at `connect()`), not per turn,
so it is not a latency problem. It is a *selection* problem: the more
near-neighbour capabilities a model is shown, the more often it picks the wrong
one.

`orion_core/tool_resolver.py` was written to pre-filter that surface per turn.
It works, it is fast (0.09 ms), and it was **switched off and wired to nothing**.
`OrionDispatcher.resolve_tool_declarations` existed as a "stable seam" that no
caller used.

The reason it stayed dormant was sound. Filtering risks hiding a tool the model
needed, and a capability that silently disappears mid-conversation is far worse
than a large schema. There was no way to justify enabling it without evidence,
and no way to get evidence without enabling it.

## The method: shadow evaluation

`orion_core/resolver_shadow.py` breaks that circle. On every real turn the
resolver runs *alongside* the live path, is told which tool the model actually
chose, and records whether its selection would have contained that tool.

Nothing is filtered. Nothing changes. Evidence accumulates.

**Recall is the only metric that decides safety.** Not precision, not the token
saving, not the mean number of tools kept — those are the benefit, and a benefit
is worthless if the cost is losing a capability. A miss is a turn where ORION
would have reached for something that was not there.

The verdict is deliberately hard to satisfy: a minimum sample (200 turns),
perfect recall over it, and every miss recorded with its query so a failure is
diagnosable rather than merely counted. A recall figure is never reported
without its sample size beside it.

## What the first measurement showed

Against `tests/data/tool_routing_cases.py` — 67 hand-written phrases in ordinary
speech, split by index into a development half and a **held-out** half:

| | recall | verdict |
|---|---|---|
| dev | 76.5% | one turn in four would have lost its tool |
| holdout | 72.7% | |

The caution was justified. Enabling the resolver would have broken *"quiz me on
neuroscience"*, *"what's the weather in Birmingham"*, *"what did we talk about
yesterday"* and *"summarise this document for me"*.

## Root cause — not the algorithm

The missing words were **not in the schema at all**:

```
"weather"     appears nowhere in the geo tool's schema
"quiz"        appears nowhere in the study tool's schema
"summarise"   appears nowhere in process_file's schema
"yesterday"   appears nowhere in recall_conversation's schema
```

No lexical scorer can match a term that does not exist in the document. Tool
descriptions are written in capability language — precise, technical, aimed at a
model reading a schema — and people ask in ordinary speech.

## The fix — document expansion

`orion_core/tool_vocabulary.py` maps each tool to the ordinary phrases used to
ask for it. This is the standard information-retrieval answer to a query/document
vocabulary mismatch, applied at index time.

It **costs the model nothing**: the terms are used only by ORION's own local
scorers and are never added to the schema sent to a provider. A test asserts the
resolver does not mutate `TOOL_DECLARATIONS`.

| | recall before | recall after |
|---|---|---|
| dev | 76.5% | **100%** |
| **holdout** | **72.7%** | **100%** |

The holdout was never inspected while the vocabulary was written. That is the
number that means something — a scorer tuned until it passes the cases used to
tune it has measured only its own tuning.

While filtering hard: **25 of 136 tools exposed, 82% less schema.**

### Two defects the work introduced and fixed

1. **Name theft.** Vocabulary carries a tool's own name weight, so a term that
   *is* another tool's name lets one capability outrank the tool being asked for.
   Putting "study" in the research tool's vocabulary made the literal query
   `study` rank `research` first. Twelve such collisions existed on the first
   draft. `tool_vocabulary.collisions()` finds them and a test fails on any.
   For multi-word names like `morning_briefing` removal costs nothing — the name
   already tokenises to "morning" and "briefing".

2. **Arbitrary ties.** `research` and `product_research` scored identically on
   the query `research`, and the tie fell to dictionary order. Fixed with a
   name-exactness bonus (`NAME_EXACTNESS_BONUS = 0.5`): ordinary length
   normalisation, where a query accounting for *all* of a tool's name beats one
   accounting for half. Capped well below a single IDF term (~3–5) so it breaks
   ties without ever outweighing real evidence.

## Meaning, not only words (Mark XXXI)

`hybrid_scores` adds sentence meaning to the lexical ranker: each tool's name,
description and vocabulary is embedded once by the local encoder
(`semantic.py`, bge-small-en-v1.5, CPU), and a tool whose purpose stands above
the others for a request gains `SEMANTIC_BOOST` (1.0; 1.5 since 2026-09-26) lexical units per
standard deviation. With the encoder cold it IS `lexical_scores`.

Measured on **40 new everyday requests for tools the golden set never covered**
("kill spotify", "wake me up at 7 tomorrow", "take that last change back",
"what's hogging my RAM"), written after the vocabulary and never tuned on:

| recall within | words | words + meaning |
|---|---|---|
| 5 tools | 87.5% | 87.5% |
| 10 tools | 92.5% | **97.5%** |
| 25 tools | 97.5% | **100%** |

The weight was chosen on the golden dev half, where meaning cost one case at 5
tools (94.7% -> 92.1%) and was level at 10; the golden holdout rose at 5 tools
(94.6% -> 97.3%). The shadow evaluator and `intent_brain.combined_scores` now
use it, so real-turn evidence accumulates on the scorer that would ship.

## The learned network, filler words and plurals (2026-09-26)

Measured on **376 everyday requests covering all 144 tools** — British
phrasing, casual asks, indirect needs ("do I need a brolly today" -> `geo`,
"i'm skint again" -> `finance`), speech/typing slips ("chek my emials",
"shutdwon now") and multi-step asks, each labelled with the right tool and
any other tool that would also be right (89 have one). 320 plain, 32
indirect, 15 slips, 9 multi-step. None copied from a description, a quoted
example or the vocabulary (checked mechanically). Split by index: the even
half was used to choose every setting below; the odd half was never
inspected.

What changed:

* `tool_resolver._tokens` drops the question and helper words of everyday
  speech (how, much, what, can, have, got, any, want, need ...) and folds
  plurals ("emails" = "email"). While they counted, "how much cash have I got
  LEFT" ranked `token_usage` first on "how much ... left".
  who/where/when/why/like still count.
* `SEMANTIC_BOOST` 1.0 -> 1.5.
* `intent_brain`: a new network — words, word pairs, stems and
  letter-trigrams (4,096 hashed) -> 128 hidden -> one output per tool —
  trained on what the tools say about themselves (names, the requests quoted
  in descriptions, description sentences, the vocabulary) plus two copies of
  each wrapped in everyday lead-ins, then on every successful real request.
* It is now USED: `intent_brain.routing_scores` ranks `find_tool` and the text
  path's menu (it was wired only into shadow measurement).
* `find_tool` ranks straight from the scores. Through `resolve()` its safety
  floor pinned `catch_up`, `reason` and `capabilities` and then removed them,
  so `reason` (not a Live core tool) could never be found.
* The text menu ranks on the request only, not on the memory note the live
  worker appends to it.

Ranking (top-k contains an acceptable tool):

| | all 376: top-1 | top-3 | top-10 | holdout 188: top-1 | top-3 | top-10 |
|---|---|---|---|---|---|---|
| **encoder warm** — before (words + meaning) | 74.5% | 87.2% | 93.9% | 79.3% | 88.8% | 94.7% |
| before, with the previous 64-unit network | 74.5% | 88.0% | 93.9% | 78.7% | 89.4% | 94.7% |
| after: new words + meaning x1.5 | 78.2% | 89.6% | 95.5% | 79.3% | 92.0% | 96.3% |
| **after: + new network** | **79.5%** | **89.9%** | **95.5%** | **81.9%** | **92.0%** | **96.3%** |
| **encoder cold** — before (words) | 69.1% | 82.2% | 91.5% | 73.4% | 86.2% | 93.1% |
| before, with the previous network | 68.9% | 83.2% | 90.4% | 73.9% | 86.2% | 92.6% |
| after: new words | 69.9% | 84.8% | 92.8% | 71.3% | 86.7% | 95.2% |
| **after: + new network** | **75.3%** | **88.8%** | **93.4%** | **76.6%** | **91.0%** | **95.2%** |
| network alone — previous / new | 31.1% / 58.8% | 46.3% / 77.4% | 63.0% / 87.8% | | | |

What the user-facing paths actually offer (right tool present):

| | before | after |
|---|---|---|
| `find_tool`'s six results | 88.3% | **93.4%** |
| the text path's ten-tool menu | 92.3% | **94.9%** |
| the same menu when the request carries a memory note | 66.5% | **94.9%** |

Learning and state:

* Taught the 188 development requests (46.7 ms each, learner thread): those
  phrasings 77.1% -> 95.7% top-1, kept across a restart; the untaught half
  unchanged (81.9% -> 81.4%).
* `CONFIG_DIR/brain` was never created and `atomic_write_bytes` will not
  create a folder, so **the previous network never saved**: it was re-seeded
  on every start (~3-4 s CPU) and forgot everything it learned. Now: seeded
  once (~5 s, background, below-normal priority), then loaded in 32 ms plus
  249 ms to refill the replay memory. The replay memory is refilled on every
  start — after a restart the old network sampled only the user's requests.
  Anything learned since the last save is flushed when the learner goes idle.
* The previous network's file is read once for the user's history (a
  migrated history of 188 requests scores 91.5% top-1 on those phrasings).
* A tool the network has no output for (a new plugin, MCP or forged tool) is
  queued by `ensure_known` and learned in the background. With 15 tools
  withheld: before learning, their 41 requests lose 2.5 points at top-1 and
  top-3 against words + meaning alone; after (1.7 s) they are level at top-1
  and +2.4 at top-3, other tools unaffected.

Cost: inference 0.13 ms p50 / 0.19 ms p99; a routed turn 0.25 ms cold,
6.2 ms warm (the query embedding, which words + meaning already paid). ~9 MB
held (table 2.1 MB, its Adam moments 4.2 MB, the phrase corpus 2 MB), 2.1 MB
on disk. Training yields the interpreter between batches: while seeding, a
simulated 60 fps loop thread's worst frame went 24.6 -> 17 ms late.

Measured and not shipped: a sentence-embedding input to the network (+0.5
top-1, -0.6 top-3 on the holdout, for ~30 s CPU and ~22 MB); a confidence
threshold on the boost (p >= 0.2 or 0.3 lowered dev top-1); a 64-unit hidden
layer (1-2 points lower).

## What is still unproven

The 67 golden cases cover roughly 30 of 136 tools; the 40 above add 40 more.
**Recall on the remaining tools is measured only by shadow mode on real
traffic**, which is why the flag stays off.

`tool_vocabulary` covers 99 of 136 tools; the remainder are narrow tools rarely
asked for by name.

## How to decide

```
diagnostics(action='routing')
```

* **NOT YET** — fewer than 200 real turns observed. Keep using ORION.
* **NOT SAFE** — it names the tools being lost. Add their missing vocabulary,
  then `ShadowEvaluator().clear()`: old observations describe a different
  algorithm and would flatter it.
* **SAFE TO ENABLE** — set `ORION_TOOL_RESOLVER=1`.

Note that enabling the flag alone still changes nothing until a caller uses
`OrionDispatcher.resolve_tool_declarations`. Wiring that seam into the turn path
is a separate, deliberate step.

## Files

| file | role |
|---|---|
| `orion_core/tool_resolver.py` | the pre-filter; cached inverted index |
| `orion_core/tool_vocabulary.py` | curated invoking phrases per tool |
| `orion_core/resolver_shadow.py` | measures it on real turns, changes nothing |
| `tests/data/tool_routing_cases.py` | the 75-phrase evaluation set, dev/holdout |
| `tests/test_tool_routing_quality.py` | recall gates, collision guard, wiring |
| `tests/test_tool_resolver_index.py` | index == unoptimised implementation |


## Status on this machine — 2026-09-20

The verdict is **NOT YET**, and it is worth writing down why rather than
enabling the flag anyway.

```
turns observed   4
recall           25.0%  (1 kept, 3 missed)
would expose     25.0 of 138 tools  (82% less schema)
```

Four turns is not evidence of anything. The three misses are more interesting
than the percentage:

```
find_files     'Okay , so , why want you to do is open up a fre sh no te'
execute_plan   'Woul d you be able to up load Ti kTok videos for me?'
self_changes   'I would like you to say hello to you.'
```

Two of those are not routing failures at all — they are the offline recogniser
splitting words mid-syllable. No lexical matcher can score `fre sh no te`
against a tool about notes, and neither could a person. That is a transcription
problem wearing a routing problem's clothes, and enabling the resolver would
turn a bad transcript into a lost tool instead of a confused answer.

So the flag stays off until real turns accumulate. The offline recogniser is
the thing to look at first.

### What the evaluation set says

Separately from live turns, the hand-written set scores:

| split | recall@5 | cases |
|---|---|---|
| DEV | 94.7% | 38 |
| HOLDOUT | **100%** | 37 |

The holdout is the honest number: those phrasings were never looked at while
tuning. The two DEV misses are genuine ambiguities between neighbouring tools
(`research` vs `web_search`, `capture_screen` vs `screen_read`) rather than
missing vocabulary.

### The speaker pair

`speaker_id` and `voice_speaker_id` are easy to confuse and were being confused
by the router. `speaker_id` is a pitch-based guess at whether a voice sounded
male or female; `voice_speaker_id` compares a trained voiceprint against
enrolled individuals. `speaker_id`'s own description claimed it answered "who
was just talking?", which it cannot, and it therefore out-scored the tool that
can on exactly the question people ask most often.

Both descriptions now claim only what their tool does, and the evaluation set
carries both sides of the distinction. Note that the three phrases used to
diagnose this — "who is speaking", "only listen to my voice", "was that a man
or a woman" — are deliberately placed on the DEV side. A phrase used to fix the
scorer cannot also be used to score it.

### Typing slips and nearest examples (6 Oct 2026)

Measured on the 394-request everyday set (`tests/data/intent_routing_cases.py`),
top-1, sentence encoder warm, 3 seeds of the intent network. "Typo'd" is the
same set with one or two seeded slips per request (a letter swapped, dropped,
doubled or replaced by a neighbouring key).

| | dev clean | dev typo'd | HOLDOUT clean | HOLDOUT typo'd |
|---|---|---|---|---|
| before | 77.7% | 59.2% | 75.0% | 53.1% |
| after | **82.1%** | **79.9%** | **80.9%** | **77.7%** |

Three changes, each measured on the development half first:

* **Spelling** (`orion_core/spelling.py`). A word that is in no tool's wording
  and not among the sentence model's ~22k whole English words is put right to
  a tool word one edit away (words people ask with outrank description prose,
  then English frequency). Links, file names and code are left alone, and with
  no model installed nothing is guessed. Applied before the word ranker, the
  meaning ranker and the network (`intent_brain.combined_scores`). Of the 394
  clean requests, 11 change: 8 genuine slips fixed, 1 harmless
  (normalise -> normalises), 2 wrong (spacex -> space, munny -> sunny).
* **More everyday requests** for the tools that had two to five
  (`intent_situations.py`, batch 3; four-word overlap check against the set).
* **Nearest example request** (`tool_resolver.semantic_scores`). A tool scores
  the higher of its document's cosine and its closest known request's: an
  indirect request ("my laptop is crawling") is far from a whole description
  but near one example. 974 requests, embedded in ~3 s by `warm_semantic` on a
  worker; never on the main thread.

The reflex fast path uses the same speller (`reflex.match_reflex_forgiving`),
but only for command words: if the rule would accept any word where the
corrected one stands, it was an argument (a song, a name) and the turn goes to
the model. Its own test commands with one slip: 1/48 handled locally before,
40/48 after, no wrong argument.
