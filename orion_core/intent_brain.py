"""
Learning what you mean — a network that maps YOUR words to ORION's actions.

ORION routes a request to one of ~140 tools. The tool resolver ranks them by
the words in each tool's schema and vocabulary, which is fixed: it knows how a
tool is DESCRIBED, not how this household ASKS for it. This network closes that
gap by learning from experience:

    words ─► hashed features (4,096) ─► 64 hidden units ─► one output per tool

It starts from the tool vocabulary (so it is useful on day one), and every
request that ends in a successful tool call teaches it one more example of how
you phrase that intent. A replay memory keeps the vocabulary and your history
in the mix, so learning today's phrasing does not erase last week's.

Where it is used: as a learned boost on top of the lexical ranker (see
``combined_scores``), which the resolver's shadow evaluation measures on every
real turn — so whether it is helping is a recorded number, not a claim.

Training never runs on the event loop: observations are queued and a single
daemon thread learns from them. State persists in ``CONFIG_DIR/brain``.
"""

from __future__ import annotations

import queue
import re
import threading
import zlib
from pathlib import Path
from typing import Any, Sequence

from .constants import CONFIG_DIR

BRAIN_DIR = CONFIG_DIR / "brain"
FEATURES = 4096
HIDDEN = 64
CAPACITY = 256                  # output units — room for tools added later
LEARNED_BOOST = 6.0             # what a certain prediction adds to a lexical score
USER_WEIGHT = 3.0               # a real request counts more than a vocabulary line
SAVE_EVERY = 5
#: A prediction may outrank the lexical ranker outright only when it is
#: confident AND rests on at least this many of the user's own requests for
#: that tool. A vocabulary-only guess never gets more than LEARNED_BOOST —
#: measured: on its own the seeded network is right only 43% of the time.
TRUSTED_CONFIDENCE = 0.5
TRUSTED_EXAMPLES = 2
#: ...and only for a request that shares most of its meaningful words with
#: something the user actually said for that tool. Without this, three
#: "how much money do I have left" -> finance lessons also dragged "how many
#: tokens have I used" to finance (measured) — confidence alone generalises
#: along the filler words.
TRUSTED_OVERLAP = 0.5
_STOP = frozenset(
    "a an the i me my you your is are was were be do does did can could would will "
    "what whats how much many when where who which why this that these those it its "
    "of in on at to for from with and or have has had any some there here please "
    "tell show give get".split())

_WORD = re.compile(r"[a-z0-9']+")


def features(text: str) -> list[int]:
    """Hashed unigram + bigram indices. Stable across runs (crc32, not hash())."""
    words = _WORD.findall(str(text or "").lower())
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return sorted({zlib.crc32(g.encode("utf-8")) % FEATURES for g in grams})


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP)


def _dense(batch: Sequence[Sequence[int]]) -> Any:
    import numpy as np
    x = np.zeros((len(batch), FEATURES), dtype=np.float32)
    for row, active in enumerate(batch):
        if active:
            x[row, list(active)] = 1.0 / (len(active) ** 0.5)
    return x


class IntentBrain:
    def __init__(self, directory: Path | None = None, seed: int = 0) -> None:
        from .neural import MLP, ReplayMemory
        self.directory = Path(directory) if directory is not None else BRAIN_DIR
        self.path = self.directory / "intent.npz"
        self.labels: list[str] = []
        self.observed = 0
        #: How many of the user's own successful requests taught each tool.
        self.user_counts: dict[str, int] = {}
        #: The meaningful words of those requests, per tool (the overlap gate).
        self.user_phrases: dict[str, list[frozenset[str]]] = {}
        self.net = MLP([FEATURES, HIDDEN, CAPACITY], output="softmax", lr=3e-3,
                       l2=1e-6, seed=seed)
        self.replay = ReplayMemory(capacity=8000, seed=seed)
        self._lock = threading.RLock()
        self._pending: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self.seeded = False
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        from .neural import MLP
        loaded = MLP.load(self.path)
        if loaded is None:
            return
        net, meta = loaded
        if net.sizes != [FEATURES, HIDDEN, CAPACITY]:
            return
        self.net = net
        self.labels = list(meta.get("labels") or [])
        self.observed = int(meta.get("observed", 0))
        self.user_counts = {str(k): int(v) for k, v in (meta.get("user_counts") or {}).items()}
        self.seeded = bool(meta.get("seeded", False))
        for text, label in meta.get("recent") or []:
            if label in self.labels:
                self.replay.add(features(text), self.labels.index(label), USER_WEIGHT)
                self._remember_phrase(text, label)
        self._recent = [tuple(pair) for pair in meta.get("recent") or []]

    def save(self) -> bool:
        with self._lock:
            recent = [(t, l) for (t, l) in getattr(self, "_recent", [])][-400:]
            return self.net.save(self.path, {"labels": self.labels, "observed": self.observed,
                                             "seeded": self.seeded, "recent": recent,
                                             "user_counts": self.user_counts})

    # ── labels ───────────────────────────────────────────────────────────────

    def _label(self, tool: str) -> int | None:
        if tool in self.labels:
            return self.labels.index(tool)
        if len(self.labels) >= CAPACITY:
            return None
        self.labels.append(tool)
        return len(self.labels) - 1

    # ── seeding from what the tools say about themselves ─────────────────────

    def seed(self, vocabulary: dict[str, str], descriptions: dict[str, str] | None = None,
             epochs: int = 40) -> int:
        """Learn the vocabulary as a starting point. Returns examples used."""
        examples: list[tuple[list[int], int]] = []
        with self._lock:
            for tool, phrases in vocabulary.items():
                label = self._label(tool)
                if label is None:
                    continue
                words = str(phrases).split()
                chunks = [" ".join(words[i:i + 4]) for i in range(0, len(words), 3)] or [tool]
                chunks.append(tool.replace("_", " "))
                if descriptions and descriptions.get(tool):
                    first = descriptions[tool].split(".")[0]
                    chunks.append(first[:160])
                for chunk in chunks:
                    active = features(chunk)
                    if active:
                        examples.append((active, label))
                        self.replay.add(active, label, 1.0)
            import random
            rng = random.Random(7)
            for _ in range(max(1, epochs)):
                rng.shuffle(examples)
                for start in range(0, len(examples), 64):
                    batch = examples[start:start + 64]
                    self.net.train_batch(_dense([b[0] for b in batch]),
                                         [b[1] for b in batch])
            self.seeded = True
        return len(examples)

    # ── learning from use ────────────────────────────────────────────────────

    def observe(self, utterance: str, tool: str, ok: bool = True) -> None:
        """Queue one real example. Returns at once; a daemon thread learns it."""
        if not ok or not str(utterance or "").strip() or not tool:
            return
        self._pending.put((str(utterance)[:300], str(tool)))
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._learn_loop, daemon=True,
                                            name="orion-intent-learner")
            self._worker.start()

    def _learn_loop(self) -> None:
        while True:
            try:
                utterance, tool = self._pending.get(timeout=30.0)
            except queue.Empty:
                return
            try:
                self.learn_now(utterance, tool)
            except Exception:
                pass

    def learn_now(self, utterance: str, tool: str, steps: int = 6) -> None:
        with self._lock:
            label = self._label(tool)
            active = features(utterance)
            if label is None or not active:
                return
            self.replay.add(active, label, USER_WEIGHT)
            recent = getattr(self, "_recent", [])
            recent.append((utterance, tool))
            self._recent = recent[-400:]
            for _ in range(steps):
                batch = self.replay.sample(31) + [(active, label, USER_WEIGHT)]
                self.net.train_batch(_dense([b[0] for b in batch]), [b[1] for b in batch],
                                     weights=[b[2] for b in batch])
            self.observed += 1
            self.user_counts[tool] = self.user_counts.get(tool, 0) + 1
            self._remember_phrase(utterance, tool)
            if self.observed % SAVE_EVERY == 0:
                self.save()

    def _remember_phrase(self, text: str, tool: str) -> None:
        words = content_words(text)
        if words:
            phrases = self.user_phrases.setdefault(tool, [])
            phrases.append(words)
            del phrases[:-50]

    def resembles_user_phrasing(self, query: str, tool: str) -> bool:
        words = content_words(query)
        if not words:
            return False
        for said in self.user_phrases.get(tool, ()):
            if len(words & said) / len(words | said) >= TRUSTED_OVERLAP:
                return True
        return False

    # ── predicting ───────────────────────────────────────────────────────────

    def probabilities(self, query: str) -> dict[str, float]:
        active = features(query)
        if not active or not self.labels:
            return {}
        with self._lock:
            import numpy as np
            probs = self.net.forward_sparse(active, np.full(len(active), 1.0 / len(active) ** 0.5))
            return {tool: float(probs[i]) for i, tool in enumerate(self.labels)}

    def top(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        probs = self.probabilities(query)
        return sorted(probs.items(), key=lambda kv: -kv[1])[:k]

    def describe(self) -> str:
        return (f"language→action network: {self.net.parameters:,} weights, "
                f"{len(self.labels)} actions known, learned from {self.observed} of "
                f"your requests{' (seeded from tool vocabulary)' if self.seeded else ''}")


# ── the shared instance and the resolver hook ────────────────────────────────

_BRAIN: IntentBrain | None = None
_BRAIN_LOCK = threading.Lock()


def brain() -> IntentBrain:
    """The process-wide intent network, seeded from the vocabulary on first use."""
    global _BRAIN
    with _BRAIN_LOCK:
        if _BRAIN is None:
            made = IntentBrain()
            if not made.seeded:
                from .dispatch_schema import TOOL_DECLARATIONS
                from .tool_vocabulary import VOCABULARY
                made.seed(VOCABULARY, {d["name"]: d.get("description", "")
                                       for d in TOOL_DECLARATIONS})
                made.save()
            _BRAIN = made
        return _BRAIN


def combined_scores(query: str, declarations: Sequence[dict[str, Any]],
                    intent: IntentBrain | None = None) -> dict[str, float]:
    """Lexical relevance plus what the network has learned. A resolver scorer.

    The base is hybrid_scores: words plus sentence meaning when the encoder
    is warm (semantic.py), exactly lexical_scores when it is not."""
    from .tool_resolver import hybrid_scores
    scores = dict(hybrid_scores(query, declarations))
    net = intent or brain()
    try:
        probs = net.probabilities(query)
    except Exception:
        return scores
    top_lexical = max(scores.values(), default=0.0)
    for decl in declarations:
        name = decl["name"]
        p = probs.get(name, 0.0)
        if not p:
            continue
        boost = LEARNED_BOOST * p
        if (p >= TRUSTED_CONFIDENCE and net.user_counts.get(name, 0) >= TRUSTED_EXAMPLES
                and net.resembles_user_phrasing(query, name)):
            # Learned from YOU, and sure of it: allowed to overrule the words.
            boost = max(boost, p * (top_lexical + 1.0))
        scores[name] = scores.get(name, 0.0) + boost
    return scores




# ── off-the-loop plumbing for the live worker ────────────────────────────────

_QUEUE: "queue.Queue[tuple[str, str, bool]]" = queue.Queue()
_FEEDER: threading.Thread | None = None


def _feed() -> None:
    while True:
        try:
            utterance, tool, ok = _QUEUE.get(timeout=60.0)
        except queue.Empty:
            return
        try:
            brain().learn_now(utterance, tool) if ok else None
        except Exception:
            pass


def observe_async(utterance: str, tool: str, ok: bool = True) -> None:
    """Teach the network from a real turn without touching the caller's thread:
    building (and, the first time, seeding) the network takes ~2 s, which must
    never land on the voice loop."""
    global _FEEDER
    if not ok or not str(utterance or "").strip() or not tool:
        return
    _QUEUE.put((str(utterance)[:300], str(tool), True))
    if _FEEDER is None or not _FEEDER.is_alive():
        _FEEDER = threading.Thread(target=_feed, daemon=True, name="orion-intent-feeder")
        _FEEDER.start()


def warm() -> None:
    """Build the network in the background at start-up."""
    threading.Thread(target=lambda: _safe_brain(), daemon=True,
                     name="orion-intent-warm").start()


def _safe_brain() -> None:
    try:
        brain()
    except Exception:
        pass


def loaded() -> IntentBrain | None:
    """The network if it is already built, else None — never builds it."""
    return _BRAIN


__all__ = ["IntentBrain", "brain", "combined_scores", "features", "loaded",
           "observe_async", "warm"]
