"""
ORION's sense of meaning — sentence embeddings, computed on this machine.

Everything ORION looked things up with matched WORDS: FTS5 in memory and the
knowledge graph, a hashed bag-of-words in the library, token overlap in the
tool resolver. Words fail exactly where people are most natural. "Which
painkiller can't I take?" shares no word with "Sam is allergic to ibuprofen";
"when's the thing with Dr Harper" shares one weak word with the fact that
answers it. A word index cannot bridge that however well it is tuned.

A sentence-embedding model can: it maps text to a point in a 384-dimension
space where things that MEAN the same sit close together, whatever words they
use. This module runs one locally:

    model      BAAI bge-small-en-v1.5 (MIT), the int8-quantised ONNX export
               — 34 MB, 384-d, CLS pooling. Chosen by measurement, not
               reputation: on ORION's own memory benchmark (34 natural
               questions, 16 facts among 106 look-alike distractors) it took
               held-out top-1 from 82.4% (words) to 94.1% alone and 100%
               fused with the word index; all-MiniLM-L6-v2 managed 88.2%.
    runtime    onnxruntime (already a dependency, for OCR and the voiceprint)
               + HuggingFace `tokenizers`. Questions: CPU, one thread — the
               caller's — so background indexers can lower their own priority
               (background_priority) and never compete with the face or
               audio. Batches of GPU_MIN_ROWS or more: the GPU, when
               accelerator.py has one (DirectML). No torch.
    cost       ~6 ms for a question; ~20-30 ms a row for backfill on the CPU,
               ~5 ms on the GPU; it runs once, in the background. Queries
               never wait on a backfill: there is no lock around CPU
               inference (GPU runs are serialised among the batches only).

Used by memory recall and proactive recall (memory.py), the library
(ingestion.py), the knowledge graph, read_documents (document_intelligence.py),
tool routing (tool_resolver.hybrid_scores), research page ranking and
per-section source choice, fact de-duplication (fact_memory.py) and the
"nearest controls" hint when a UI element is not found (vision.py).

It never downloads or loads anything implicitly. `ready()` is False until
`warm()` has run — which ORION does once, in a background thread, at start-up
— so the test suite and a machine without the model behave exactly as they did
before, on words alone. The files are fetched once, size- and SHA-256-checked
(model_store.download_verified), into CONFIG_DIR/models.

ORION_SEMANTIC=0 turns the whole layer off.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .constants import BASE_DIR, CONFIG_DIR


@dataclass(frozen=True)
class ModelFile:
    remote: str           # path inside the model repository
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    files: dict[str, ModelFile]
    dim: int
    pooling: str          # "cls" or "mean"
    max_tokens: int = 256

    def url(self, local_name: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.files[local_name].remote}"


MODEL = ModelSpec(
    name="bge-small-en-v1.5",
    repo="Xenova/bge-small-en-v1.5",
    files={
        "model_quantized.onnx": ModelFile(
            "onnx/model_quantized.onnx", 34_014_426,
            "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4"),
        "tokenizer.json": ModelFile(
            "tokenizer.json", 711_396,
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    },
    dim=384,
    pooling="cls",
)

#: Where a model may already be. CONFIG_DIR first (downloaded there); the
#: install's own assets second (a build may ship it).
def _search_dirs(spec: ModelSpec) -> list[Path]:
    return [CONFIG_DIR / "models" / spec.name, BASE_DIR / "assets" / "models" / spec.name]


def enabled() -> bool:
    return os.getenv("ORION_SEMANTIC", "1").strip().lower() not in {"0", "false", "no", "off"}


def _no_arena(options: Any) -> None:
    """One short query at a time: an arena only holds memory (-24 MB, Mark XXXX.4)."""
    options.enable_cpu_mem_arena = False


class SemanticEncoder:
    """Text -> unit vectors. Thread-safe; loads once; never on its own."""

    #: Recently asked questions, so a repeated recall costs nothing.
    CACHE_SIZE = 512
    #: Rows from which an encode uses the GPU session (when there is one), and
    #: how many rows it runs at once there. MEASURED (accelerator.POLICY):
    #: 8 rows 234-363 ms -> 139-228 ms, 128 rows 3.6 s -> 0.6 s; a batch of 64
    #: was the fastest size on the GPU, 16 on the CPU.
    #:
    GPU_MIN_ROWS = 8
    GPU_BATCH = 64
    #: The backlog from which the BACKGROUND INDEXERS (memory, knowledge graph)
    #: may use the GPU at all; smaller top-ups pass gpu=False.
    #:
    #: 25 Oct 2026 (Mark 37.2, the owner's sessions): the boot top-up - one to
    #: a few dozen new rows, and the knowledge graph's batches of 16 - built
    #: the DirectML device 20-27 s after every launch, while the face was
    #: drawing on the same GPU. The GUI thread then sat in Qt/native code for
    #: 169+122+325+668 ms (11:50:36-38) and 138+122+325 ms (11:58:04-05),
    #: 1.4 s on 24 Oct - the "lag" felt after the greeting. Saving 0.1-1 s of
    #: background CPU on a small top-up is not worth that, nor the ~1 GB of
    #: committed memory a DirectML session costs. The GPU now serves only a
    #: genuine backlog (a first-run seeding or a re-index), where it saves
    #: seconds to minutes.
    GPU_BACKLOG_ROWS = 256

    def __init__(self, spec: ModelSpec = MODEL) -> None:
        self.spec = spec
        self._session: Any = None
        self._batch_session: Any = None      # the GPU one for big batches, once built
        self._batch_tried = False
        self._batch_lock = threading.Lock()
        self._tokenizer: Any = None
        self._input_names: set[str] = set()
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Any] = OrderedDict()
        # Guards the question cache only — never inference (see encode). Recall
        # runs on worker threads (relevant_note via to_thread, the indexers), and
        # an unguarded get -> move_to_end raced another thread's eviction into a
        # KeyError that silently dropped that question to words-only recall.
        self._cache_lock = threading.Lock()
        self.last_error = ""

    # ── where the model is ───────────────────────────────────────────────────

    def model_dir(self) -> Path | None:
        """The first directory holding every file, complete."""
        from .model_store import usable

        for folder in _search_dirs(self.spec):
            if all(usable(folder / name, size=item.size)
                   for name, item in self.spec.files.items()):
                return folder
        return None

    def installed(self) -> bool:
        return self.model_dir() is not None

    def ready(self) -> bool:
        return self._session is not None and enabled()

    # ── getting it ───────────────────────────────────────────────────────────

    def download(self, opener: Callable[..., Any] | None = None) -> bool:
        """Fetch the model into CONFIG_DIR/models, verified. Blocking."""
        from .model_store import ModelDownloadError, download_verified

        target = _search_dirs(self.spec)[0]
        kwargs = {} if opener is None else {"opener": opener}
        try:
            for name, item in self.spec.files.items():
                path = target / name
                if path.is_file() and path.stat().st_size == item.size:
                    continue
                download_verified(self.spec.url(name), path, size=item.size,
                                  sha256=item.sha256, timeout=120.0, **kwargs)
        except ModelDownloadError as exc:
            self.last_error = f"download failed: {exc}"
            return False
        return True

    def load(self) -> bool:
        """Open the model. Blocking (~0.3 s); call from a worker thread."""
        if self._session is not None:
            return True
        folder = self.model_dir()
        if folder is None:
            self.last_error = "the model is not installed"
            return False
        try:
            from tokenizers import Tokenizer

            from . import accelerator

            tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
            tokenizer.enable_truncation(self.spec.max_tokens)
            tokenizer.enable_padding()
            # ONE thread, so inference runs on the calling thread and inherits
            # its priority: the background indexers lower theirs (see
            # background_priority) and yield to the face, audio and the user.
            # Measured: a question 5.9 ms (4.0 on two threads), a backfill row
            # 19.6 ms (12.1) — the backfill is one-off and in the background.
            # Questions stay here on the CPU (5.4 ms against 8.9 ms on the
            # GPU); big batches use _gpu_batch_session.
            session = accelerator.create_session(
                "semantic_query", folder / "model_quantized.onnx", threads=1, inter_threads=1,
                configure=_no_arena)
        except Exception as exc:
            self.last_error = f"could not load the model: {exc}"
            return False
        with self._lock:
            self._tokenizer = tokenizer
            self._input_names = {i.name for i in session.get_inputs()}
            self._session = session
            self._batch_session = None
            self._batch_tried = False
        self.last_error = ""
        return True

    def _gpu_batch_session(self) -> Any:
        """The GPU session for backfill-sized batches, or None: use the CPU one.

        Built on first use by the calling worker thread, at most once. Never
        on the GUI thread - a big encode there simply stays on the CPU. If its
        GPU run fails, accelerator falls back to the CPU query session."""
        from . import accelerator

        if not self._batch_tried:
            if accelerator.on_gui_thread() or not accelerator.wants_gpu("semantic_batch"):
                return None
            with self._batch_lock:
                if not self._batch_tried:
                    folder, query = self.model_dir(), self._session
                    batch = None
                    if folder is not None and query is not None:
                        try:
                            batch = accelerator.create_session(
                                "semantic_batch", folder / "model_quantized.onnx",
                                threads=1, inter_threads=1, cpu=lambda: query)
                        except Exception:
                            batch = None
                    self._batch_session = batch if batch is not None and batch.on_gpu else None
                    self._batch_tried = True
        batch = self._batch_session
        return batch if batch is not None and batch.on_gpu else None

    def warm(self, allow_download: bool = True) -> bool:
        """Make ready: download if missing (and allowed), then load. Blocking."""
        if not enabled():
            self.last_error = "turned off (ORION_SEMANTIC=0)"
            return False
        if self.ready():
            return True
        if not self.installed() and not (allow_download and self.download()):
            return False
        return self.load()

    # ── using it ─────────────────────────────────────────────────────────────

    def encode(self, texts: Sequence[str], batch: int = 16, *, gpu: bool = True) -> Any:
        """Unit-length float32 vectors, one row per text; None if not ready.

        Encoded shortest-first and put back in the caller's order: a batch is
        padded to its longest text, so mixing a 20-token note with a 256-token
        one made every row cost the long one's attention. On real memory rows
        that was 60 ms a row as stored against 13 ms sorted.
        """
        if not self.ready():
            return None
        import numpy as np

        items = [str(t or " ")[:4000] for t in texts]
        if not items:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        ordered = [items[i] for i in order]
        session, size = self._session, max(1, batch)
        # gpu=False: a background top-up that must never build the DirectML
        # device beside the face (see GPU_BACKLOG_ROWS).
        if gpu and len(items) >= self.GPU_MIN_ROWS:
            fast = self._gpu_batch_session()
            if fast is not None:
                session, size = fast, max(size, self.GPU_BATCH)
        out = []
        for start in range(0, len(ordered), size):
            chunk = ordered[start:start + size]
            # No lock: InferenceSession.run and Tokenizer.encode_batch are both
            # safe to call concurrently. A lock here made a question asked on
            # the event loop queue behind a low-priority backfill batch — a
            # priority inversion, i.e. exactly the stutter this avoids. (The
            # GPU session serialises its own runs; only batches use it.)
            encoded = self._tokenizer.encode_batch(chunk)
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = session.run(None, feed)[0]
            if self.spec.pooling == "cls":
                vectors = hidden[:, 0]
            else:
                weights = mask[..., None].astype(np.float32)
                vectors = (hidden * weights).sum(1) / np.clip(weights.sum(1), 1e-9, None)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            out.append((vectors / np.clip(norms, 1e-9, None)).astype(np.float32))
        sorted_vectors = np.vstack(out)
        result = np.empty_like(sorted_vectors)
        result[order] = sorted_vectors
        return result

    def encode_one(self, text: str) -> Any:
        """One vector, cached by exact text; None if not ready."""
        key = str(text or "")
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        matrix = self.encode([key])
        if matrix is None:
            return None
        vector = matrix[0]
        vector.flags.writeable = False      # shared by every caller of this question
        with self._cache_lock:
            self._cache[key] = vector
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        return vector

    def describe(self) -> dict[str, Any]:
        folder = self.model_dir()
        return {"model": self.spec.name, "enabled": enabled(), "installed": folder is not None,
                "ready": self.ready(), "where": str(folder or ""), "error": self.last_error}


#: The one encoder the process shares.
ENCODER = SemanticEncoder()


# ── vectors in SQLite ────────────────────────────────────────────────────────

def to_blob(vector: Any) -> bytes:
    """float16 bytes: half the space, and far below the model's own precision."""
    import numpy as np
    return np.asarray(vector, dtype=np.float16).tobytes()


def from_blob(blob: bytes, dim: int = MODEL.dim) -> Any:
    import numpy as np
    vector = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    return vector if vector.shape[0] == dim else None


def encode_for_index(texts: Sequence[str], *, backlog: int) -> Any:
    """Encode for a background indexer: the GPU only for a real backlog
    (SemanticEncoder.GPU_BACKLOG_ROWS), so a boot top-up never builds the
    DirectML device beside the face."""
    encoder = ENCODER
    if isinstance(encoder, SemanticEncoder) and backlog < encoder.GPU_BACKLOG_ROWS:
        return encoder.encode(texts, gpu=False)
    return encoder.encode(texts)


def text_digest(text: str) -> str:
    """What a stored vector was computed from, so an edit re-embeds it."""
    import hashlib
    return hashlib.sha1(f"{MODEL.name}\0{text}".encode("utf-8", "replace")).hexdigest()[:16]


# ── fusing word and meaning rankings ─────────────────────────────────────────

#: Measured on the memory benchmark — tuned on the dev half only, then
#: reported on the held-out half (122 facts, 106 of them look-alikes):
#:
#:   words alone                      dev 94.1%  holdout 82.4%  unrelated 2/5
#:   reciprocal-rank fusion           dev 88.2%  holdout 94.1%  unrelated 0/5
#:   score fusion (below)             dev 100%   holdout 100%   unrelated 0/5
#:
#: Rank fusion threw away HOW MUCH closer a meaning match was: "how long do
#: clients have to pay" scored 0.725 against the payment terms and 0.645
#: against the client rate, and the word "clients" still won. So meaning is
#: the score and a word match adds a small, rank-decaying bonus (stable from
#: 0.02 to 0.05; 0.03 is the middle).
LEXICAL_BONUS = 0.03
#: A meaning-only match below this is not trusted. Every unrelated question
#: stayed under 0.54; the weakest genuine answer was 0.57.
SEMANTIC_FLOOR = 0.58
#: A word match whose meaning is this far off is a coincidence of spelling
#: ("boiling POINT" -> "dentist apPOINTment") and is dropped.
LEXICAL_VETO = 0.45


def fuse(lexical: Iterable[Any], semantic: Iterable[tuple[Any, float]], *,
         similarity: dict[Any, float] | None = None, bonus: float = LEXICAL_BONUS,
         floor: float = SEMANTIC_FLOOR, veto: float = LEXICAL_VETO,
         adjust: dict[Any, float] | None = None) -> list[Any]:
    """One ranking from a word ranking and a meaning ranking.

    *lexical* is ids in word-rank order; *semantic* is (id, cosine) pairs;
    *similarity* is id -> cosine for the word matches. A word match scores its
    cosine plus ``bonus / (rank + 1)`` — or is dropped below *veto*; one not
    yet embedded scores as if at the floor, so it is neither lost nor promoted.
    A meaning-only match needs *floor* and scores its cosine. *adjust* adds a
    per-id nudge after both (memory's who-is-it-about preference); it never
    lets a meaning-only match in below the floor.
    """
    score: dict[Any, float] = {}
    similarity = similarity or {}
    for rank, key in enumerate(lexical):
        cosine = similarity.get(key)
        if cosine is not None and cosine < veto:
            continue
        base = floor if cosine is None else cosine
        score[key] = max(score.get(key, -1.0), base + bonus / (rank + 1))
    for key, cosine in semantic:
        if key not in score and cosine >= floor:
            score[key] = cosine
    if adjust:
        for key in score:
            score[key] += adjust.get(key, 0.0)
    return sorted(score, key=lambda key: -score[key])


def background_priority() -> None:
    """Lower the CURRENT thread's scheduling priority (Windows; no-op elsewhere).

    For the indexers: embedding thousands of rows is minutes of CPU, and it
    must never be what makes ORION's face stutter or his voice crackle.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)  # BELOW_NORMAL
    except Exception:
        pass


# ── start-up ─────────────────────────────────────────────────────────────────

def _warm_now(on_ready: Callable[[], Any] | None, log: Callable[[str], Any] | None,
              allow_download: bool, when: str = "") -> None:
    """Load the encoder on THIS (worker) thread, then call *on_ready*."""
    import time
    background_priority()
    started = time.monotonic()
    fetched = not ENCODER.installed()
    if not ENCODER.warm(allow_download=allow_download):
        if log is not None:
            log(f"SEMANTIC: meaning-based recall unavailable — {ENCODER.last_error}; "
                "using word matching.")
        return
    if log is not None:
        log(f"SEMANTIC: meaning-based recall ready ({MODEL.name}"
            f"{', downloaded' if fetched else ''}) in {time.monotonic() - started:.1f}s"
            f"{' - ' + when if when else ''}.")
    if on_ready is not None:
        try:
            on_ready()
        except Exception as exc:
            if log is not None:
                log(f"SEMANTIC: indexing after start-up failed - {exc}")


def warm_in_background(on_ready: Callable[[], Any] | None = None,
                       log: Callable[[str], Any] | None = None,
                       allow_download: bool = True) -> threading.Thread | None:
    """Download (once) and load the encoder off the event loop, then call
    *on_ready* (still in that thread) so callers can index what they hold."""
    if not enabled():
        return None
    thread = threading.Thread(target=_warm_now, args=(on_ready, log, allow_download),
                              name="orion-semantic-warm", daemon=True)
    thread.start()
    return thread


# Building the ONNX session holds Python's lock in one native call, so EVERY
# Python thread waits it out - the event loop, and the microphone and speaker
# callbacks too. MEASURED 13 Oct 2026 (from source, warm disk): 105-114 ms of
# a 150 ms load; the import and the tokenizer hold it under 40 ms, and the
# encoding itself releases it. In the installed app on 12 Oct the load took
# 1.7 s and froze the loop 1184 ms at 11:14:13 - the same second the
# microphone reported "input overflow" while the owner was answering ORION's
# greeting. A fixed 25 s timer cannot know that, so the encoder now starts at
# the first quiet moment after it (never later than WARM_LATEST_S).
#
# 10 s since Mark XXXVII. The 25 s floor also covered re-embedding ~1,000
# routing texts (4.7 s of this thread, a 540 ms loop freeze measured 23 Oct);
# tool_resolver now reads those back from disk, so what remains is the session
# build above - and the quiet gate, not the clock, is what keeps that out of a
# conversation. Meaning-based recall and routing arrive ~15 s sooner.
WARM_EARLIEST_S = 10.0
WARM_LATEST_S = 180.0
#: Quiet: no change of ORION's state for this long, and not mid-turn.
QUIET_S = 6.0
_BUSY_STATES = frozenset({"PROCESSING", "SPEAKING", "CONNECTING"})
_SILENT_STATES = frozenset({"STANDBY", "PAUSED"})       # nothing heard or said


class ConversationQuiet:
    """Is the conversation quiet? Fed ORION's states (bus.state); read from a
    worker thread - a str and a float, each written in one store."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        import time
        self._clock = clock or time.monotonic
        self._state = ""
        self._since = self._clock()

    def note_state(self, state: str) -> None:
        self._state = str(state or "").upper()
        self._since = self._clock()

    def quiet(self) -> bool:
        if self._state in _SILENT_STATES:
            return True
        if self._state in _BUSY_STATES:
            return False
        return self._clock() - self._since >= QUIET_S


def warm_when_quiet(quiet: Callable[[], bool], on_ready: Callable[[], Any] | None = None,
                    log: Callable[[str], Any] | None = None, *, allow_download: bool = True,
                    earliest_s: float = WARM_EARLIEST_S, latest_s: float = WARM_LATEST_S,
                    poll_s: float = 0.5, clock: Callable[[], float] | None = None,
                    sleep: Callable[[float], Any] | None = None) -> threading.Thread | None:
    """``warm_in_background`` at the first moment *quiet()* holds, no sooner
    than *earliest_s* after this call and no later than *latest_s*."""
    if not enabled():
        return None
    import time
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    began = clock()

    def _run() -> None:
        sleep(earliest_s)
        while True:
            waited = clock() - began
            try:
                calm = bool(quiet())
            except Exception:
                calm = True             # a broken probe must not cost recall
            if calm:
                when = f"started at a quiet moment, {waited:.0f} s after launch"
                break
            if waited >= latest_s:
                when = f"started at {waited:.0f} s with no quiet moment"
                break
            sleep(poll_s)
        _warm_now(on_ready, log, allow_download, when)

    thread = threading.Thread(target=_run, name="orion-semantic-warm", daemon=True)
    thread.start()
    return thread


__all__ = ["ENCODER", "LEXICAL_BONUS", "LEXICAL_VETO", "MODEL", "ModelSpec", "SEMANTIC_FLOOR",
           "ConversationQuiet", "background_priority",
           "SemanticEncoder", "enabled", "from_blob", "fuse",
           "text_digest", "to_blob", "warm_in_background", "warm_when_quiet"]
