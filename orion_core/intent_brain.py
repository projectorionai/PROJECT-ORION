"""
Learning what you mean — a network that maps YOUR words to ORION's actions.

ORION routes a request to one of ~140 tools. The tool resolver ranks them by
the words in each tool's schema and vocabulary, which is fixed: it knows how a
tool is DESCRIBED, not how this household ASKS for it. This network closes that
gap by learning from experience:

    words ─► hashed features (4,096): words, word pairs, stems and
             letter-trigrams of the meaningful words
          ─► 128 hidden units ─► one output per tool

Letter-trigrams make it forgiving of typing and speech-to-text slips ("chek my
emials" shares most of its trigrams with "check my emails"); stems fold
"emails" into "email"; the pairs keep a little word order.

It starts from what the tools say about themselves — their names, the example
requests quoted in their descriptions, the description sentences and the
curated vocabulary, each also wrapped in everyday lead-ins ("could you ...",
"... for me") so filler is seen with every tool and learned to mean nothing —
so it is useful on day one. Every request that ends in a successful tool call
then teaches it one more example of how you phrase that intent. A replay
memory keeps the tools' own phrases and your history in the mix, so learning
today's phrasing does not erase last week's; it is refilled on every start,
not only the first (before, a restarted network sampled only your requests
and drifted towards those tools).

Measured on 376 everyday requests covering all 144 tools, none taken from the
schema or vocabulary (docs/ORION_TOOL_ROUTING.md has the method and the full
table). Ranking that find_tool and the text path's menu use:

                                        top-1   top-3   top-10
    encoder warm   before               74.5%   87.2%   93.9%
                   after                79.5%   89.9%   95.5%
    encoder cold   before               69.1%   82.2%   91.5%
                   after                75.3%   88.8%   93.4%

(the words-only ranker changed too — filler words and plurals, see
tool_resolver._tokens.) The previous 64-unit network, trained on the
vocabulary alone, was 31% top-1 on its own and left the combined ranking
where it was; this one is 59% on its own. A sentence-embedding input was also built and measured: +0.5 points
top-1 and -0.6 top-3 on the held-out half, for ~30 s of CPU and ~22 MB more
memory — not worth it beside the ranker's own meaning signal, so it is not
used.

Where it is used: ``routing_scores`` — find_tool's search, the text path's
tool menu and the resolver's shadow evaluation — as a learned boost on top of
the word+meaning ranker, once the network is built (never on the caller's
thread).

Training never runs on the event loop: observations are queued and a single
daemon thread learns from them. State persists in ``CONFIG_DIR/brain`` — which
nothing used to create, so until now the network was re-seeded on every start
and forgot everything it had learned. The previous network's file is read once
to carry your learned phrasings across.
"""

from __future__ import annotations

import queue
import random
import re
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

from .constants import CONFIG_DIR

BRAIN_DIR = CONFIG_DIR / "brain"
STATE_FILE = "intent_v2.npz"
#: The 64-unit network's state: read once for the user's history, never written.
LEGACY_FILE = "intent.npz"
FEATURES = 4096
HIDDEN = 128
CAPACITY = 512                  # output units — room for installed and forged tools
EPOCHS = 10
#: When the tools' phrases change (a better description, a new vocabulary
#: line), a network seeded earlier never saw them: it only met them one
#: replay sample at a time. This many passes bring it up to date.
REFRESH_EPOCHS = EPOCHS          # measured: 3 left a toy change unlearnt, 6-10 matched a fresh seed
BATCH = 64
#: Fraction of a training phrase's features dropped at random each time it is
#: seen, so the network cannot lean on any one word.
WORD_DROPOUT = 0.2
LEARNED_BOOST = 6.0             # what a certain prediction adds to a lexical score
USER_WEIGHT = 3.0               # a real request counts more than a vocabulary line
SAVE_EVERY = 5
#: Lessons waiting for the learner thread, at most. Both queues were unbounded,
#: and the first build (seconds of seeding) is exactly when lessons pile up.
#: Past the bound the OLDEST waiting lesson is dropped: a lesson is a
#: replaceable sample — the replay memory already holds its kind, the newest
#: phrasing is the one that reflects current use, and the caller is the voice
#: loop, which must never block on a full queue.
MAX_PENDING_LESSONS = 256
#: The user's own requests kept and learned again whenever the network
#: re-trains (a release that changes the tools' phrases) and put back in the
#: replay memory on every start: distinct phrasings, newest last, a repeat
#: moving its phrasing to the end. Was the last 400 TURNS, repeats included,
#: so the 401st request pushed the oldest lesson out of every re-training
#: and restart, and one request said fifty times took fifty places. ~120 KB
#: in the saved state at the bound.
LESSON_MEMORY = 2000
#: The user's lessons the replay memory samples from (the tools' own
#: phrases are pinned beside them and never evicted).
REPLAY_LESSONS = 4000


def _offer(pending: "queue.Queue[Any]", item: Any) -> Any:
    """Queue *item* without blocking, dropping the oldest when full.
    Returns the dropped item, or None."""
    dropped = None
    for _ in range(4):
        try:
            pending.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                dropped = pending.get_nowait()
            except queue.Empty:
                pass
    return item                         # another producer kept winning: drop this one
#: A prediction may outrank the lexical ranker outright only when it is
#: confident AND rests on at least this many of the user's own requests for
#: that tool. A vocabulary-only guess never gets more than LEARNED_BOOST.
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
    "tell show give get got just im i'm it's".split())

_WORD = re.compile(r"[a-z0-9']+")
_QUOTED = re.compile(r"'([^']{4,90})'")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
#: Everyday lead-ins and tails wrapped round the tools' own phrases.
_LEAD_INS = ("", "can you ", "could you ", "please ", "i want to ", "i need to ", "help me ",
             "hey orion ", "orion ", "would you ", "go on and ")
_TAILS = ("", " please", " for me", " now", " thanks")


def _stem(word: str) -> str:
    word = word.replace("'", "")
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("es", ""), ("s", ""), ("ed", "")):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


def features(text: str) -> list[int]:
    """Hashed words, word pairs, stems and letter-trigrams of the meaningful
    words. Stable across runs (crc32, not hash())."""
    words = _WORD.findall(str(text or "").lower())
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    for word in words:
        if word in _STOP:
            continue
        grams.append("~" + _stem(word))
        if len(word) >= 4:
            marked = "<" + word.replace("'", "") + ">"
            grams.extend("#" + marked[i:i + 3] for i in range(len(marked) - 2))
    return sorted({zlib.crc32(g.encode("utf-8")) % FEATURES for g in grams})


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP)


def exemplars(vocabulary: dict[str, str],
              descriptions: dict[str, str] | None = None) -> dict[str, list[str]]:
    """What each tool says about itself, as short phrases to learn from: its
    name, the requests quoted in its description, the description's
    sentences, and its curated vocabulary in four-word windows — plus the
    everyday phrasings other modules know a tool by (_extra_phrases)."""
    descriptions = descriptions or {}
    extra = _extra_phrases()
    out: dict[str, list[str]] = {}
    for tool in dict.fromkeys(list(descriptions) + list(vocabulary)):
        phrases = [tool.replace("_", " ")]
        text = " ".join(str(descriptions.get(tool) or "").split())
        phrases += [q for q in _QUOTED.findall(text) if len(q.split()) >= 2]
        phrases += [s.strip() for s in _SENTENCE_END.split(text) if 12 <= len(s.strip()) <= 200]
        words = str(vocabulary.get(tool) or "").split()
        phrases += [" ".join(words[i:i + 4]) for i in range(0, len(words), 3)]
        phrases += extra.get(tool, [])
        out[tool] = list(dict.fromkeys(p for p in phrases if p.strip()))
    return out


#: First words that make a category phrase an ACTION ("play music", "take
#: notes") rather than the thing to open — those belong to other tools.
_ACTION_WORDS = frozenset({"play", "listen", "take", "make", "write", "record", "edit",
                           "watch", "read", "draw", "capture", "browse", "surf"})


def _extra_phrases() -> dict[str, list[str]]:
    """Everyday ways of asking for a KIND of application ("open the music
    app", "close the browser"), from the app resolver's categories — the
    network otherwise saw only the tool's own wording. Never raises."""
    out: dict[str, list[str]] = {}
    try:
        from .app_resolver import CATEGORIES
        opens: list[str] = []
        closes: list[str] = []
        for category in CATEGORIES:
            nouns = [p for p in category.phrases if p.split()[0] not in _ACTION_WORDS][:3]
            for index, phrase in enumerate(nouns):
                opens.append(("open the " if index % 2 == 0 else "launch my ") + phrase)
            if nouns:
                closes.append(f"close the {nouns[0]}")
        out = {"open_app": opens, "close_app": closes}
    except Exception:
        pass
    # Situations: the problems and needs each tool answers ("everything is
    # lagging" -> resource_status), see intent_situations.py.
    try:
        from .intent_situations import phrases
        for tool, items in phrases().items():
            out.setdefault(tool, []).extend(items)
    except Exception:
        pass
    return out


def corpus_key(vocabulary: dict[str, str], descriptions: dict[str, str] | None = None) -> int:
    """A fingerprint of everything the network is seeded from."""
    import json
    blob = json.dumps(exemplars(vocabulary, descriptions), sort_keys=True)
    return zlib.crc32(blob.encode("utf-8"))


#: Is the conversation quiet? (app.py hands over the probe the semantic warm-up
#: uses.) None: always train flat out, as tests and tools expect.
_QUIET: list[Any] = [None]
#: While the conversation is busy the trainer waits in steps this long...
PAUSE_POLL_S = 0.05
#: ...but never longer than this between two batches, so a session that never
#: falls quiet still finishes the network (one batch every two minutes).
PAUSE_MAX_S = 120.0
#: The first build waits at most this long for a quiet moment to start in.
QUIET_LATEST_S = 300.0


def set_quiet_probe(quiet: Any) -> None:
    """Install (or, with None, remove) the conversation-quiet probe."""
    _QUIET[0] = quiet


def _calm() -> bool:
    probe = _QUIET[0]
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:
        return True                 # a broken probe must not stall learning


#: Set by shutdown(): training stops at its next step and the learner and
#: feeder threads end, so nothing is mid-way through numpy when the process
#: exits (the native-exit access violations of 11-26 Oct 2026 all had the
#: feeder inside learn_now -> _yield_gil when the exit came).
_STOPPING = threading.Event()


class _Stopping(Exception):
    """Raised inside a training loop once ORION is shutting down."""


def _yield_gil() -> None:
    """Hand the interpreter to other threads between training batches, and
    stand aside entirely while ORION is mid-conversation.

    A training step is mostly short numpy calls and Python loops that keep
    the GIL, so the event loop — which also paints the face — waited on the
    trainer for up to a switch interval at a time. sleep(0) alone was not
    enough: 16 Oct 2026 the first boot of a new build re-learned every tool's
    phrases (a ~30 s refresh, 8,871 rows x 10 epochs measured from source)
    straight through the greeting and the spoken briefing, and session.log
    put the trainer beside four loop blocks of 106-250 ms (14:00:48-14:01:24)
    and an 'input overflow' on the microphone. Measured from source with an
    asyncio heartbeat: loop lag p95 14.5 ms and max 73 ms while it trains,
    against 6.5 / 7.8 ms idle."""
    if _STOPPING.is_set():
        raise _Stopping()
    time.sleep(0)
    if _calm():
        return
    waited = 0.0
    while waited < PAUSE_MAX_S and not _calm():
        if _STOPPING.is_set():
            raise _Stopping()
        time.sleep(PAUSE_POLL_S)
        waited += PAUSE_POLL_S


def _augment(phrases: dict[str, list[str]], seed: int = 7) -> list[tuple[str, str]]:
    """(training text, tool) — each phrase as written plus two copies wrapped
    in everyday lead-ins, deterministically."""
    rng = random.Random(seed)
    rows = []
    for tool, items in phrases.items():
        for phrase in items:
            rows.append((phrase, tool))
            for _ in range(2):
                rows.append((rng.choice(_LEAD_INS) + phrase.lower() + rng.choice(_TAILS), tool))
    return rows


class IntentBrain:
    def __init__(self, directory: Path | None = None, seed: int = 0) -> None:
        from .neural import ReplayMemory, SparseMLP
        self.directory = Path(directory) if directory is not None else BRAIN_DIR
        self.path = self.directory / STATE_FILE
        self._seed = seed
        self.labels: list[str] = []
        self.observed = 0
        #: How many of the user's own successful requests taught each tool.
        self.user_counts: dict[str, int] = {}
        #: The meaningful words of those requests, per tool (the overlap gate).
        self.user_phrases: dict[str, list[frozenset[str]]] = {}
        self.net = SparseMLP(FEATURES, HIDDEN, CAPACITY, lr=3e-3, seed=seed)
        self.replay = ReplayMemory(capacity=REPLAY_LESSONS, seed=seed)
        #: Serialises TRAINING (seeding, new tools, online steps), all of which
        #: runs on worker threads. Predictions do not take it.
        self._lock = threading.RLock()
        self._pending: "queue.Queue[tuple[Any, ...]]" = queue.Queue(maxsize=MAX_PENDING_LESSONS)
        #: Lessons dropped because the learner had fallen that far behind.
        self.dropped = 0
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()  # Ownership only; never training or I/O.
        self.seeded = False
        self._recent: list[tuple[str, str]] = []
        #: The tools' own phrases as (features, tool) — rebuilt every start.
        self._corpus: list[tuple[Any, str]] = []
        self._known_key: Any = None
        self._unsaved = 0
        #: Fingerprint of the phrases it was last trained on (corpus_key).
        self.corpus_key: int | None = None
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        from .neural import MLP, SparseMLP
        loaded = SparseMLP.load(self.path)
        if loaded is not None and loaded[0].sizes[:2] == [FEATURES, HIDDEN] and loaded[0].sizes[2] <= CAPACITY:
            net, meta = loaded
            if net.sizes[2] == CAPACITY:
                self.net = net
            else:
                # Keep learned routes and Adam clocks when expanding the
                # action head. A full 256-output checkpoint formerly stopped
                # all newly installed plugins from ever gaining an output.
                previous = net.sizes[2]
                enlarged = SparseMLP(FEATURES, HIDDEN, CAPACITY, lr=net.lr,
                                     l2=net.l2, seed=self._seed)
                enlarged.table = net.table
                enlarged.b1 = net.b1
                enlarged.W2[:, :previous] = net.W2
                enlarged.b2[:previous] = net.b2
                enlarged._tm, enlarged._tv = net._tm, net._tv
                enlarged._row_steps, enlarged._t = net._row_steps, net._t
                enlarged._m["b1"], enlarged._v["b1"] = net._m["b1"], net._v["b1"]
                for key in ("W2", "b2"):
                    enlarged._m[key][..., :previous] = net._m[key]
                    enlarged._v[key][..., :previous] = net._v[key]
                enlarged.stats = net.stats
                self.net = enlarged
            self.labels = list(meta.get("labels") or [])
            self.seeded = bool(meta.get("seeded", False))
            key = meta.get("corpus_key")
            self.corpus_key = int(key) if isinstance(key, (int, float)) else None
        else:
            # The 64-unit network's weights do not fit this one; what matters
            # in it is the user's history, which is carried across and learned
            # again when this network is seeded.
            legacy = MLP.load(self.directory / LEGACY_FILE)
            if legacy is None:
                return
            meta = legacy[1]
        self.observed = int(meta.get("observed", 0))
        self.user_counts = {str(k): int(v) for k, v in (meta.get("user_counts") or {}).items()}
        self._recent = list(dict.fromkeys(
            (str(t), str(l)) for t, l in reversed(meta.get("recent") or [])))[::-1][-LESSON_MEMORY:]
        for text, label in self._recent:
            self._remember_phrase(text, label)

    def save(self) -> bool:
        with self._lock:
            meta = {"version": 2, "labels": self.labels, "observed": self.observed,
                    "seeded": self.seeded, "recent": self._recent[-LESSON_MEMORY:],
                    "user_counts": self.user_counts, "corpus_key": self.corpus_key}
            try:
                # atomic_write_bytes will not create a missing folder, and
                # nothing else created this one: every save failed silently.
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                return False
            saved = self.net.save(self.path, meta)
            if saved:
                self._unsaved = 0
            return saved

    # ── labels ───────────────────────────────────────────────────────────────

    def _label(self, tool: str) -> int | None:
        if tool in self.labels:
            return self.labels.index(tool)
        if len(self.labels) >= CAPACITY:
            return None
        self.labels.append(tool)
        return len(self.labels) - 1

    # ── the training corpus ──────────────────────────────────────────────────

    @staticmethod
    def _corpus_from(vocabulary: dict[str, str],
                     descriptions: dict[str, str] | None) -> list[tuple[Any, str]]:
        import numpy as np
        rows = []
        for text, tool in _augment(exemplars(vocabulary, descriptions)):
            active = features(text)
            if active:
                rows.append((np.asarray(active, dtype=np.int16), tool))
        return rows

    def _items(self, new_tools: bool = True) -> list[tuple[Any, int, float]]:
        """Everything to train on: the tools' phrases, then the user's own.
        With *new_tools* False, tools without an output yet are left out —
        they are learned properly by learn_tools, not handed an untrained
        output that would also hide them from ensure_known."""
        import numpy as np
        known = set(self.labels)
        items = []
        for active, tool in self._corpus:
            label = self._label(tool) if new_tools or tool in known else None
            if label is not None:
                items.append((active, label, 1.0))
        for text, tool in self._recent:
            label = self._label(tool) if new_tools or tool in known else None
            active = features(text)
            if label is not None and active:
                items.append((np.asarray(active, dtype=np.int16), label, USER_WEIGHT))
        return items

    def _fit(self, items: list[tuple[Any, int, float]], epochs: int) -> None:
        import numpy as np
        rng = np.random.default_rng(self._seed)
        for _ in range(max(1, epochs)):
            order = rng.permutation(len(items))
            for start in range(0, len(order), BATCH):
                _yield_gil()
                batch = [items[i] for i in order[start:start + BATCH]]
                rows = []
                for active, _label, _weight in batch:
                    if len(active) > 3:
                        kept = active[rng.random(len(active)) > WORD_DROPOUT]
                        rows.append(kept if len(kept) else active)
                    else:
                        rows.append(active)
                self.net.train_batch(rows, [b[1] for b in batch], weights=[b[2] for b in batch])

    def _refill_replay(self, items: list[tuple[Any, int, float]]) -> None:
        """The tools' phrases (weight 1.0) pinned, the user's lessons
        (USER_WEIGHT) in the bounded part - see ReplayMemory.pinned."""
        self.replay.clear()
        for active, label, weight in items:
            if weight == USER_WEIGHT:
                self.replay.add(active, label, weight)
            else:
                self.replay.pin(active, label, weight)

    # ── seeding from what the tools say about themselves ─────────────────────

    def seed(self, vocabulary: dict[str, str], descriptions: dict[str, str] | None = None,
             epochs: int = EPOCHS) -> int:
        """Learn the tools' own phrases (and any carried-over history) as a
        starting point. Returns the number of training rows."""
        corpus = self._corpus_from(vocabulary, descriptions)
        with self._lock:
            self._corpus = corpus
            items = self._items()
            self._fit(items, epochs)
            self._refill_replay(items)
            self.seeded = True
            self.corpus_key = corpus_key(vocabulary, descriptions)
        return len(items)

    def refresh(self, vocabulary: dict[str, str], descriptions: dict[str, str] | None = None,
                epochs: int = REFRESH_EPOCHS) -> bool:
        """Re-learn the tools' phrases when they changed since the network
        was last trained (and the user's own requests with them, so nothing
        learned from use is lost). True if it trained."""
        key = corpus_key(vocabulary, descriptions)
        if key == self.corpus_key:
            return False
        corpus = self._corpus_from(vocabulary, descriptions)
        with self._lock:
            self._corpus = corpus
            items = self._items(new_tools=False)
            self._fit(items, epochs)
            self._refill_replay(items)
            self.corpus_key = key
        return True

    def restore_corpus(self, vocabulary: dict[str, str],
                       descriptions: dict[str, str] | None = None) -> None:
        """After a restart: the weights are loaded, but online learning needs
        the tools' phrases back in the replay memory."""
        corpus = self._corpus_from(vocabulary, descriptions)
        with self._lock:
            self._corpus = corpus
            self._refill_replay(self._items(new_tools=False))

    # ── tools the network has never seen ─────────────────────────────────────

    def ensure_known(self, declarations: Sequence[dict[str, Any]]) -> None:
        """Queue any tool it has no output for (a new plugin, MCP server or
        forged tool, or a built-in added since it was seeded) to be learned
        from its own description. Cheap when nothing changed; never trains
        on the caller's thread.

        Measured with 15 tools withheld from seeding: until they are learned
        the network costs their requests 2.5 points at top-1 and top-3 (it
        pushes the tools it knows instead); once learned (1.7 s) they are
        level at top-1 and 2.4 points better at top-3 than words and meaning
        alone, and the other tools are unaffected."""
        names = tuple(str(d.get("name", "")) for d in declarations)
        key = (len(names), hash(names))
        if key == self._known_key:
            return
        self._known_key = key
        known = set(self.labels)
        unseen = [{"name": d["name"], "description": str(d.get("description", ""))}
                  for d in declarations if d.get("name") and d["name"] not in known]
        room = CAPACITY - len(self.labels)
        if unseen and room > 0:
            self._submit(("tools", unseen[:room]))

    def learn_tools(self, declarations: Sequence[dict[str, Any]], epochs: int = EPOCHS) -> int:
        """Add outputs for new tools and learn their phrases, mixed with the
        replay memory so the tools already known are not forgotten."""
        import numpy as np

        from .tool_resolver import vocabulary_for
        wanted = [d for d in declarations if d.get("name") and d["name"] not in self.labels]
        if not wanted:
            return 0
        corpus = self._corpus_from({d["name"]: vocabulary_for(d["name"]) for d in wanted},
                                   {d["name"]: d.get("description", "") for d in wanted})
        rng = np.random.default_rng(self._seed)
        with self._lock:
            self._corpus.extend(corpus)
            fresh = [(a, self._label(t), 1.0) for a, t in corpus]
            fresh = [item for item in fresh if item[1] is not None]
            for active, label, weight in fresh:
                self.replay.pin(active, label, weight)      # a tool's own phrases
            for _ in range(max(1, epochs)):
                order = rng.permutation(len(fresh))
                for start in range(0, len(order), 16):
                    _yield_gil()
                    batch = [fresh[i] for i in order[start:start + 16]] + self.replay.sample(48)
                    self.net.train_batch([b[0] for b in batch], [b[1] for b in batch],
                                         weights=[b[2] for b in batch])
            self.save()
        return len(fresh)

    # ── learning from use ────────────────────────────────────────────────────

    def _submit(self, job: tuple[Any, ...]) -> None:
        with self._worker_lock:
            dropped = _offer(self._pending, job)
            if dropped is not None:
                self.dropped += 1
                if dropped[0] == "tools":
                    # A new-tools job is not a replaceable sample: forget that it
                    # was queued, so the next ensure_known() queues it again.
                    self._known_key = None
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._learn_loop, daemon=True,
                                                name="orion-intent-learner")
                self._worker.start()

    def observe(self, utterance: str, tool: str, ok: bool = True) -> None:
        """Queue one real example. Returns at once; a daemon thread learns it."""
        if not ok or not str(utterance or "").strip() or not tool:
            return
        self._submit(("learn", str(utterance)[:300], str(tool)))

    def _learn_loop(self) -> None:
        while not _STOPPING.is_set():
            try:
                job = self._pending.get(timeout=30.0)
            except queue.Empty:
                with self._worker_lock:
                    if not self._pending.empty():
                        continue
                    self._worker = None
                # Retire before flushing: a producer can now start a successor.
                # save() serialises its snapshot with any new training.
                self.flush()
                return
            try:
                if job[0] == "learn":
                    self.learn_now(job[1], job[2])
                elif job[0] == "tools":
                    self.learn_tools(job[1])
            except Exception:
                pass

    def learn_now(self, utterance: str, tool: str, steps: int = 6) -> None:
        import numpy as np
        with self._lock:
            label = self._label(tool)
            active = features(utterance)
            if label is None or not active:
                return
            row = np.asarray(active, dtype=np.int16)
            self.replay.add(row, label, USER_WEIGHT)
            for _ in range(steps):
                _yield_gil()
                batch = self.replay.sample(31) + [(row, label, USER_WEIGHT)]
                self.net.train_batch([b[0] for b in batch], [b[1] for b in batch],
                                     weights=[b[2] for b in batch])
            lesson = (utterance, tool)
            if lesson in self._recent:          # a repeat: one place, moved to the end
                self._recent.remove(lesson)
            self._recent.append(lesson)
            del self._recent[:-LESSON_MEMORY]
            self.observed += 1
            self.user_counts[tool] = self.user_counts.get(tool, 0) + 1
            self._remember_phrase(utterance, tool)
            self._unsaved += 1
            if self._unsaved >= SAVE_EVERY:
                self.save()

    def unlearn_now(self, utterance: str, tool: str, steps: int = 12) -> bool:
        """A negative lesson: *utterance* did NOT mean *tool* (the owner
        corrected it). The mistake's replayed copies and its trusted-phrasing
        entry are removed, so neither teaches it again, and the network is
        stepped to make that tool less likely for these words. False when the
        tool or the words are unknown - there is nothing to unlearn."""
        import numpy as np
        with self._lock:
            if tool not in self.labels:
                return False
            label = self.labels.index(tool)
            active = features(utterance)
            if not active:
                return False
            row = np.asarray(active, dtype=np.int16)
            items = self.replay.items
            kept = [item for item in items
                    if not (item[1] == label and len(item[0]) == len(row)
                            and np.array_equal(item[0], row))]
            if len(kept) != len(items):
                items.clear()
                items.extend(kept)
            said = str(utterance)[:300]
            self._recent = [pair for pair in self._recent if pair != (said, tool)]
            words = content_words(utterance)
            if tool in self.user_phrases:
                self.user_phrases[tool] = [w for w in self.user_phrases[tool] if w != words]
            if self.user_counts.get(tool, 0) > 0:
                self.user_counts[tool] -= 1
            # The mistake was usually taught seconds ago (observe_for_reflex):
            # its momentum would push on into it before the correction lands.
            self.net.reset_momentum(active)
            for _ in range(steps):
                _yield_gil()
                batch = self.replay.sample(15)
                self.net.train_batch([b[0] for b in batch] + [row],
                                     [b[1] for b in batch] + [label],
                                     weights=[b[2] for b in batch] + [USER_WEIGHT],
                                     away=[False] * len(batch) + [True],
                                     live_outputs=len(self.labels))
            self._unsaved += 1
            if self._unsaved >= SAVE_EVERY:
                self.save()
            return True

    def flush(self) -> None:
        """Save anything learned since the last save. The learner calls this
        when it goes idle, so a restart loses nothing it was taught — saving
        only every SAVE_EVERY lessons left up to four behind."""
        if self._unsaved:
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
        # No lock: this runs on the event loop, and training (seconds, on a
        # worker) must never hold it up. An online step nudges weights in
        # place; reading them mid-nudge moves a boost by a hair, nothing more.
        probs = self.net.predict(active)
        labels = list(self.labels)
        known = probs[:len(labels)]
        total = float(known.sum())
        if total <= 0:
            return {}
        # Unallocated outputs are not actions. Normalising over known labels
        # preserves confidence when the head grows, without retraining old
        # routes or letting spare random outputs dilute their probabilities.
        return {tool: float(known[i] / total) for i, tool in enumerate(labels[:len(known)])}

    def top(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        probs = self.probabilities(query)
        return sorted(probs.items(), key=lambda kv: -kv[1])[:k]

    def describe(self) -> str:
        seeded = " (seeded from the tools' own phrases)" if self.seeded else ""
        return (f"language→action network: {self.net.parameters:,} weights, "
                f"{len(self.labels)} actions known, learned from {self.observed} of "
                f"your requests{seeded}")


# ── the shared instance and the resolver hook ────────────────────────────────

_BRAIN: IntentBrain | None = None
_BRAIN_LOCK = threading.Lock()


def _tool_texts() -> tuple[dict[str, str], dict[str, str]]:
    from .dispatch_schema import TOOL_DECLARATIONS
    from .tool_vocabulary import VOCABULARY
    return VOCABULARY, {d["name"]: d.get("description", "") for d in TOOL_DECLARATIONS}


def brain() -> IntentBrain:
    """The process-wide intent network: loaded, or seeded the first time.
    Seconds of CPU when seeding — call it from a worker thread, never the
    event loop (routing_scores never builds it)."""
    global _BRAIN
    with _BRAIN_LOCK:
        if _BRAIN is None:
            made = IntentBrain()
            vocabulary, descriptions = _tool_texts()
            if not made.seeded:
                made.seed(vocabulary, descriptions)
                made.save()
            elif made.refresh(vocabulary, descriptions):
                # The tools' phrases changed since it last trained: brought
                # up to date here (worker thread, background priority).
                made.save()
            else:
                made.restore_corpus(vocabulary, descriptions)
            _BRAIN = made
        return _BRAIN


def combined_scores(query: str, declarations: Sequence[dict[str, Any]],
                    intent: IntentBrain | None = None) -> dict[str, float]:
    """Lexical relevance plus what the network has learned. A resolver scorer.

    The base is hybrid_scores: words plus sentence meaning when the encoder
    is warm (semantic.py), exactly lexical_scores when it is not. With no
    network built yet this IS hybrid_scores — it never builds one here."""
    from .spelling import correct
    from .tool_resolver import hybrid_scores
    query = correct(query)            # typing slips, for the network too (spelling.py)
    scores = dict(hybrid_scores(query, declarations))
    net = intent or loaded()
    if net is None:
        return scores
    try:
        net.ensure_known(declarations)
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


def routing_scores(query: str, declarations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """The ranking ORION routes by: words + meaning + the learned network once
    it is built (hybrid_scores until then). Safe on the event loop.

    A request that asks for several things is also ranked part by part, so
    each part's tool is near the top (tool_resolver.request_parts); the
    first-ranked tool is the whole request's, as before."""
    from .tool_resolver import merge_part_scores, request_parts
    net = loaded()
    whole = combined_scores(query, declarations, net)
    parts = request_parts(query)
    if not parts:
        return whole
    return merge_part_scores(whole, [combined_scores(part, declarations, net) for part in parts])


# ── off-the-loop plumbing for the live worker ────────────────────────────────

_QUEUE: "queue.Queue[tuple[str, str, bool]]" = queue.Queue(maxsize=MAX_PENDING_LESSONS)
_FEEDER: threading.Thread | None = None
_FEEDER_LOCK = threading.Lock()
#: Lessons the feeder dropped (oldest first) while the network was building.
_DROPPED = [0]


def _feed() -> None:
    global _FEEDER
    while not _STOPPING.is_set():
        try:
            utterance, tool, ok = _QUEUE.get(timeout=60.0)
        except queue.Empty:
            with _FEEDER_LOCK:
                if not _QUEUE.empty():
                    continue
                _FEEDER = None
            if _BRAIN is not None:
                _BRAIN.flush()
            return
        if _STOPPING.is_set():
            return
        try:
            if ok == AWAY:
                brain().unlearn_now(utterance, tool)
            elif ok:
                brain().learn_now(utterance, tool)
        except Exception:
            pass


#: The lesson flag of a queued correction ("these words did not mean this").
AWAY = "away"


def observe_async(utterance: str, tool: str, ok: bool = True, *,
                  negative: bool = False) -> None:
    """Teach the network from a real turn without touching the caller's thread:
    building (and, the first time, seeding) the network takes seconds, which
    must never land on the voice loop. ``negative=True`` queues a correction
    (unlearn_now) instead; a plain failure (ok=False) still teaches nothing,
    because a tool failing is not evidence the words meant another tool."""
    global _FEEDER
    if (not ok and not negative) or not str(utterance or "").strip() or not tool:
        return
    with _FEEDER_LOCK:
        lesson: Any = AWAY if negative else True
        if _offer(_QUEUE, (str(utterance)[:300], str(tool), lesson)) is not None:
            _DROPPED[0] += 1
        if _FEEDER is None or not _FEEDER.is_alive():
            _FEEDER = threading.Thread(target=_feed, daemon=True, name="orion-intent-feeder")
            _FEEDER.start()


def warm() -> None:
    """Build the network in the background at start-up."""
    threading.Thread(target=_safe_brain, daemon=True, name="orion-intent-warm").start()


def _safe_brain() -> None:
    try:
        from .semantic import background_priority
        background_priority()
        brain()
    except Exception:
        pass


def shutdown(timeout: float = 3.0) -> bool:
    """Stop learning before ORION exits: training ends at its next step, the
    learner and feeder threads finish, and what was learned is saved - only
    once they have, so a save never races a training step. True when every
    thread ended in time. Never raises."""
    _STOPPING.set()
    deadline = time.monotonic() + max(0.0, float(timeout))
    # Wake a thread waiting on its queue so it sees the stop at once (a full
    # queue means it is busy, and it checks between training steps instead).
    for pending, item in ((_QUEUE, ("", "", False)),
                          (getattr(_BRAIN, "_pending", None), ("stop",))):
        if pending is not None:
            try:
                pending.put_nowait(item)
            except Exception:
                pass
    threads = [_FEEDER, getattr(_BRAIN, "_worker", None)]
    for thread in threads:
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(max(0.0, deadline - time.monotonic()))
    ended = not any(t is not None and t.is_alive() for t in threads)
    if ended and _BRAIN is not None:
        try:
            _BRAIN.flush()
        except Exception:
            pass
    return ended


def loaded() -> IntentBrain | None:
    """The network if it is already built, else None — never builds it."""
    return _BRAIN


__all__ = ["IntentBrain", "brain", "combined_scores", "exemplars", "features", "loaded", "shutdown",
           "observe_async", "routing_scores", "warm"]
