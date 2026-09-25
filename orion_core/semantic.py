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
               + HuggingFace `tokenizers`. CPU, one thread — the caller's —
               so background indexers can lower their own priority
               (background_priority) and never compete with the face or
               audio. No torch.
    cost       ~6 ms for a question; ~20 ms a row for backfill, which runs
               once, in the background. Queries never wait on a backfill:
               there is no lock around inference.

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


class SemanticEncoder:
    """Text -> unit vectors. Thread-safe; loads once; never on its own."""

    #: Recently asked questions, so a repeated recall costs nothing.
    CACHE_SIZE = 512

    def __init__(self, spec: ModelSpec = MODEL) -> None:
        self.spec = spec
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: set[str] = set()
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Any] = OrderedDict()
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
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
            tokenizer.enable_truncation(self.spec.max_tokens)
            tokenizer.enable_padding()
            options = ort.SessionOptions()
            # ONE thread, so inference runs on the calling thread and inherits
            # its priority: the background indexers lower theirs (see
            # background_priority) and yield to the face, audio and the user.
            # Measured: a question 5.9 ms (4.0 on two threads), a backfill row
            # 19.6 ms (12.1) — the backfill is one-off and in the background.
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            session = ort.InferenceSession(str(folder / "model_quantized.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        except Exception as exc:
            self.last_error = f"could not load the model: {exc}"
            return False
        with self._lock:
            self._tokenizer = tokenizer
            self._input_names = {i.name for i in session.get_inputs()}
            self._session = session
        self.last_error = ""
        return True

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

    def encode(self, texts: Sequence[str], batch: int = 16) -> Any:
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
        out = []
        for start in range(0, len(ordered), max(1, batch)):
            chunk = ordered[start:start + batch]
            # No lock: InferenceSession.run and Tokenizer.encode_batch are both
            # safe to call concurrently. A lock here made a question asked on
            # the event loop queue behind a low-priority backfill batch — a
            # priority inversion, i.e. exactly the stutter this avoids.
            encoded = self._tokenizer.encode_batch(chunk)
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]
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
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        matrix = self.encode([key])
        if matrix is None:
            return None
        vector = matrix[0]
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
         floor: float = SEMANTIC_FLOOR, veto: float = LEXICAL_VETO) -> list[Any]:
    """One ranking from a word ranking and a meaning ranking.

    *lexical* is ids in word-rank order; *semantic* is (id, cosine) pairs;
    *similarity* is id -> cosine for the word matches. A word match scores its
    cosine plus ``bonus / (rank + 1)`` — or is dropped below *veto*; one not
    yet embedded scores as if at the floor, so it is neither lost nor promoted.
    A meaning-only match needs *floor* and scores its cosine.
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

def warm_in_background(on_ready: Callable[[], Any] | None = None,
                       log: Callable[[str], Any] | None = None,
                       allow_download: bool = True) -> threading.Thread | None:
    """Download (once) and load the encoder off the event loop, then call
    *on_ready* (still in that thread) so callers can index what they hold."""
    if not enabled():
        return None

    def _run() -> None:
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
                f"{', downloaded' if fetched else ''}) in {time.monotonic() - started:.1f}s.")
        if on_ready is not None:
            try:
                on_ready()
            except Exception as exc:
                if log is not None:
                    log(f"SEMANTIC: indexing after start-up failed - {exc}")

    thread = threading.Thread(target=_run, name="orion-semantic-warm", daemon=True)
    thread.start()
    return thread


__all__ = ["ENCODER", "LEXICAL_BONUS", "LEXICAL_VETO", "MODEL", "ModelSpec", "SEMANTIC_FLOOR",
           "background_priority",
           "SemanticEncoder", "enabled", "from_blob", "fuse",
           "text_digest", "to_blob", "warm_in_background"]
