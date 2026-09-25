"""
Document intelligence (CAP-06) — "read this, all of it, and cite your sources."

    "File scanning must be absolute and lightning-fast, and ORION must read the
     files to me in full detail."

``fast_find`` already answers *where* a file is. This answers *what a stack of
them says*, together, with a line you can check:

**It reads a whole folder into passages.** Plain text, Markdown, code, CSV/JSON,
and — when the optional readers are installed — PDF and Word. Each file is split
into overlapping passages that remember which file and which lines they came
from, so every later answer can point back at an exact place on disk.

**It retrieves without a cloud.** Ranking is TF-IDF over the passages —
deterministic, offline, instant — plus meaning from ORION's LOCAL sentence
encoder (semantic.py) once it is warm and the passages are embedded. No
embedding service, no API call, nothing that can rate-limit halfway through. That keeps the "lightning-fast" promise and
makes the retrieval something a test can pin down exactly.

**It answers grounded, and cites.** ``ask`` pulls the passages that actually
bear on the question and, if a model is available, has it answer *from those
passages* with inline ``[1]``/``[2]`` markers that map to real file:line
sources. With no model it still answers — extractively, by quoting the
strongest passages verbatim under their citations — so the feature never
silently does nothing.

The corpus is held in memory for the session: a folder digest is a "read these
now" action, not a standing index, and keeping it transient means no stale cache
and no surprise disk growth. The retrieval core is pure and injectable — the
model call is passed in — so all of it is tested without a model, a network, or
a real PDF.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

# Files we can read as text directly.
TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json",
    ".yaml", ".yml", ".ini", ".cfg", ".toml", ".html", ".htm", ".xml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1", ".sql", ".r",
})
PDF_SUFFIXES = frozenset({".pdf"})
DOCX_SUFFIXES = frozenset({".docx"})

# Guards so a runaway folder never exhausts memory.
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PASSAGES = 6000
PASSAGE_WORDS = 110          # target passage length
PASSAGE_OVERLAP = 20        # words shared with the previous passage

_WORD_RE = re.compile(r"[A-Za-z0-9]{2,}")
_STOP = frozenset(
    "the a an and or of to in on for with is are was were be been being as at by "
    "from into this that these those it its he she they we you i not but if then "
    "than so such can could will would may might do does did have has had".split()
)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if t.lower() not in _STOP]


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class Passage:
    source: str                 # file path (as given)
    ordinal: int                # passage index within the file
    line_start: int
    line_end: int
    text: str
    _terms: Counter = field(default_factory=Counter, repr=False)

    @property
    def citation(self) -> str:
        name = Path(self.source).name
        return f"{name}:{self.line_start}-{self.line_end}"


@dataclass
class Scored:
    passage: Passage
    score: float


@dataclass
class Answer:
    text: str
    sources: list[str]          # ordered citations referenced as [1], [2] ...
    passages: list[Passage]
    grounded: bool              # True when a model synthesised it from passages

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "sources": self.sources,
                "grounded": self.grounded}


@dataclass
class IngestReport:
    files_read: int
    passages: int
    skipped: list[str]          # "name — reason"
    total_words: int

    def describe(self) -> str:
        head = (f"Read {self.files_read} file(s) into {self.passages} passages "
                f"({self.total_words:,} words).")
        if self.skipped:
            head += f" Skipped {len(self.skipped)}: " + "; ".join(self.skipped[:6])
            if len(self.skipped) > 6:
                head += f" (+{len(self.skipped) - 6} more)"
        return head


# ── readers ──────────────────────────────────────────────────────────────────

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader          # optional; lazily imported
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx                          # optional; lazily imported
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def read_document(path: Path) -> str:
    """Extract text from one file, or raise with a short reason."""
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _read_pdf(path)
    if suffix in DOCX_SUFFIXES:
        return _read_docx(path)
    if suffix in TEXT_SUFFIXES or suffix == "":
        return _read_text(path)
    raise ValueError(f"unsupported type {suffix or '(none)'}")


# ── chunking ─────────────────────────────────────────────────────────────────

def chunk(source: str, text: str) -> list[Passage]:
    """Split *text* into overlapping, line-anchored passages.

    Line anchoring is what makes a citation checkable — a passage says exactly
    which lines it came from, so 'invoice.txt:40-51' points at a real place.
    """
    lines = text.splitlines()
    if not lines:
        return []
    passages: list[Passage] = []
    ordinal = 0
    i = 0
    n = len(lines)
    while i < n:
        words_here = 0
        j = i
        while j < n and words_here < PASSAGE_WORDS:
            words_here += len(lines[j].split())
            j += 1
        body = "\n".join(lines[i:j]).strip()
        if body:
            p = Passage(source=source, ordinal=ordinal,
                        line_start=i + 1, line_end=j, text=body)
            p._terms = Counter(_tokens(body))
            passages.append(p)
            ordinal += 1
        if j >= n:
            break
        # step back a few lines for overlap so a passage boundary never splits a
        # claim away from its evidence
        back = 0
        step_words = 0
        while j - back - 1 > i and step_words < PASSAGE_OVERLAP:
            step_words += len(lines[j - back - 1].split())
            back += 1
        i = max(i + 1, j - back)
    return passages


# ── the corpus + retrieval ───────────────────────────────────────────────────

class DocumentIntelligence:
    """Read a set of documents, then search and answer across them."""

    def __init__(self) -> None:
        self.passages: list[Passage] = []
        self._df: Counter = Counter()          # document frequency per term
        self._sources: set[str] = set()
        # Meaning (semantic.py): one vector per passage, filled in by a
        # low-priority background thread after each read. None until every
        # passage has one — scores from the two paths are on different
        # scales, so a half-embedded corpus is searched on words alone.
        self._vectors: Any = None
        self._embedding = False

    # -- ingestion --
    def ingest_paths(self, paths: Iterable[str | Path]) -> IngestReport:
        read = 0
        skipped: list[str] = []
        words = 0
        for raw in paths:
            path = Path(raw)
            if not path.is_file():
                skipped.append(f"{path.name} — not a file")
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    skipped.append(f"{path.name} — too large")
                    continue
                text = read_document(path)
            except Exception as exc:
                skipped.append(f"{path.name} — {type(exc).__name__}")
                continue
            new = chunk(str(path), text)
            if len(self.passages) + len(new) > MAX_PASSAGES:
                new = new[: max(0, MAX_PASSAGES - len(self.passages))]
                skipped.append(f"{path.name} — corpus full, truncated")
            for p in new:
                self.passages.append(p)
                for term in p._terms:
                    self._df[term] += 1
            words += sum(len(p.text.split()) for p in new)
            self._sources.add(str(path))
            read += 1
        if read:
            self._embed_in_background()
        return IngestReport(read, len(self.passages), skipped, words)

    # -- meaning --
    def _embed_in_background(self) -> None:
        """Embed every passage off the caller's thread (inert while cold)."""
        from . import semantic

        self._vectors = None
        if self._embedding or not semantic.ENCODER.ready() or not self.passages:
            return
        self._embedding = True

        def _run() -> None:
            try:
                semantic.background_priority()
                while True:
                    count = len(self.passages)
                    vectors = semantic.ENCODER.encode([p.text[:2000] for p in self.passages[:count]])
                    if vectors is None:
                        return
                    if count == len(self.passages):     # nothing read meanwhile
                        self._vectors = vectors
                        return
            except Exception:
                self._vectors = None
            finally:
                self._embedding = False

        import threading
        threading.Thread(target=_run, name="orion-documents-embed", daemon=True).start()

    def _meaning(self, query: str) -> Any:
        """Cosine of *query* against every passage, or None to use words alone."""
        vectors = self._vectors
        if vectors is None or len(vectors) != len(self.passages):
            return None
        try:
            from . import semantic

            question = semantic.ENCODER.encode_one(query)
            return None if question is None else vectors @ question
        except Exception:
            return None

    def ingest_folder(self, folder: str | Path, *, recursive: bool = True,
                      suffixes: Iterable[str] | None = None) -> IngestReport:
        base = Path(folder)
        if not base.is_dir():
            return IngestReport(0, 0, [f"{base} — not a folder"], 0)
        allowed = {s.lower() for s in suffixes} if suffixes else \
            (TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES)
        globber = base.rglob("*") if recursive else base.glob("*")
        files = sorted(p for p in globber
                       if p.is_file() and p.suffix.lower() in allowed)
        return self.ingest_paths(files)

    @property
    def file_count(self) -> int:
        return len(self._sources)

    # -- retrieval --
    def _idf(self, term: str) -> float:
        n = max(1, len(self.passages))
        df = self._df.get(term, 0)
        # smoothed idf; unseen term contributes nothing
        return math.log((n + 1) / (df + 1)) + 1.0 if df else 0.0

    def search(self, query: str, k: int = 6) -> list[Scored]:
        q_terms = Counter(_tokens(query))
        if not q_terms or not self.passages:
            return []
        weights = {t: c * self._idf(t) for t, c in q_terms.items()}
        scored: list[Scored] = []
        for p in self.passages:
            if not p._terms:
                continue
            length = sum(p._terms.values()) or 1
            s = 0.0
            for term, w in weights.items():
                tf = p._terms.get(term, 0)
                if tf:
                    s += w * (tf / length)
            if s > 0:
                scored.append(Scored(p, round(s, 6)))
        scored.sort(key=lambda x: (-x.score, x.passage.source, x.passage.ordinal))
        sims = self._meaning(query)
        if sims is not None:
            return self._fused(scored, sims, k)
        return scored[: max(1, k)]

    def _fused(self, lexical: list[Scored], sims: Any, k: int) -> list[Scored]:
        """Meaning as the score, TF-IDF rank as a small bonus — the fusion
        measured on ORION's memory benchmark (semantic.fuse)."""
        from . import semantic

        index_of = {id(p): i for i, p in enumerate(self.passages)}
        lexical_ids = [index_of[id(h.passage)] for h in lexical]
        ranked = sorted(range(len(self.passages)), key=lambda i: -float(sims[i]))[:max(k, 8) * 3]
        order = semantic.fuse(lexical_ids, [(i, float(sims[i])) for i in ranked],
                              similarity={i: float(sims[i]) for i in lexical_ids})
        return [Scored(self.passages[i], round(float(sims[i]), 6)) for i in order[: max(1, k)]]

    # -- answering --
    async def ask(self, question: str, *, k: int = 6,
                  generate: Callable[[str], Awaitable[str]] | None = None) -> Answer:
        import asyncio

        # Off the loop: ranking thousands of passages (and embedding the
        # question) is not work for the thread that animates the face.
        hits = await asyncio.to_thread(self.search, question, k)
        if not hits:
            return Answer(
                "Nothing in the documents I've read bears on that. Try a "
                "different wording, or point me at more files.",
                [], [], grounded=False)
        passages = [h.passage for h in hits]
        sources = [p.citation for p in passages]
        if generate is not None:
            numbered = "\n\n".join(
                f"[{i + 1}] ({p.citation})\n{p.text}"
                for i, p in enumerate(passages))
            prompt = (
                "Answer the question using ONLY the numbered passages below, "
                "which were retrieved from the user's own documents. Cite the "
                "passages you use inline as [1], [2]. If the passages do not "
                "contain the answer, say so plainly rather than guessing.\n\n"
                f"QUESTION: {question}\n\nPASSAGES:\n{numbered}\n\nANSWER:")
            try:
                body = str(await generate(prompt) or "").strip()
                if body:
                    return Answer(body, sources, passages, grounded=True)
            except Exception:
                pass  # fall through to the extractive answer
        return Answer(self._extractive_answer(question, passages),
                      sources, passages, grounded=False)

    @staticmethod
    def _extractive_answer(question: str, passages: list[Passage]) -> str:
        lines = ["From your documents (quoting the most relevant passages):", ""]
        for i, p in enumerate(passages, 1):
            snippet = re.sub(r"\s+", " ", p.text).strip()
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "…"
            lines.append(f"[{i}] {p.citation}\n    {snippet}")
        return "\n".join(lines)

    async def digest(self, *, generate: Callable[[str], Awaitable[str]] | None = None,
                     max_chars: int = 6000) -> str:
        if not self.passages:
            return "No documents have been read yet."
        # A representative slice: the opening passage of each file, in order.
        leads: list[Passage] = []
        seen: set[str] = set()
        for p in self.passages:
            if p.source not in seen:
                leads.append(p)
                seen.add(p.source)
        if generate is not None:
            corpus = "\n\n".join(
                f"({p.citation})\n{p.text}" for p in leads)[:max_chars]
            prompt = (
                "Write a concise digest of the following documents — the main "
                "themes, what each contributes, and anything notable. Cite files "
                "by name where useful.\n\n" + corpus + "\n\nDIGEST:")
            try:
                body = str(await generate(prompt) or "").strip()
                if body:
                    return body
            except Exception:
                pass
        # extractive digest: one lead line per file + the corpus' top terms
        lines = [f"{self.file_count} document(s), {len(self.passages)} passages. "
                 "Leading content per file:", ""]
        for p in leads[:40]:
            first = re.sub(r"\s+", " ", p.text).strip()
            lines.append(f"• {Path(p.source).name}: {first[:160]}"
                         + ("…" if len(first) > 160 else ""))
        top = [t for t, _ in self._df.most_common(12)]
        if top:
            lines += ["", "Recurring terms: " + ", ".join(top)]
        return "\n".join(lines)


__all__ = [
    "DocumentIntelligence", "Passage", "Scored", "Answer", "IngestReport",
    "chunk", "read_document", "TEXT_SUFFIXES", "PDF_SUFFIXES", "DOCX_SUFFIXES",
]
