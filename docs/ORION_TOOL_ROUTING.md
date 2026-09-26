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
the others for a request gains `SEMANTIC_BOOST` (1.0) lexical units per
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
