"""Measure memory retrieval as the store grows: ranking, rebuilds and recall.

    .venv\\Scripts\\python.exe tools\\bench_retrieval.py --out results.json
        [--sizes 1000,10000,50000] [--queries 200] [--repeats 5]
        [--label before] [--no-encoder]

Everything runs against a temporary store. ``ORION_CONFIG_DIR`` is pointed at
a scratch directory BEFORE ``orion_core`` is imported, so the real config/ and
its databases are never opened. The local sentence encoder is used only when
its files are already on disk; nothing is ever downloaded.

Four sections, each with its own sample count in the JSON:

ranking   pure numpy on synthetic unit vectors (no model): the similarity
          product, full-sort selection with a per-query id map (the original
          method) against partial selection with a cached, sorted id map, and
          recall@k of the latter against an exact full sort.
store     the real ``OrionMemoryMatrix`` over a synthetic corpus, with a
          deterministic stand-in encoder: warm query()/relevant() latency,
          snapshot rebuild cost, the first query after an index change, query
          latency while a backfill is writing vectors, and hit@k for planted
          targets. The stand-in's per-row delay during the backfill is
          SIMULATED encoding time, stated in the output.
quality   the real local encoder (if installed) on the labelled recall set in
          tests/data/memory_recall_cases.py plus fictional look-alike
          distractors: dev/held-out top-1 and hit@8 through query().
meta      CPU, core counts, Python, numpy, SQLite, BLAS and the code variant.

The same script runs against the earlier implementation (it detects which
snapshot API the matrix offers), so before/after numbers come from one method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DIM = 384


# ── helpers ──────────────────────────────────────────────────────────────────

def _percentiles(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        return {"n": 0}
    ordered = sorted(samples_ms)

    def pick(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return round(ordered[index], 4)

    return {"n": len(ordered), "p50_ms": pick(0.50), "p95_ms": pick(0.95),
            "max_ms": round(ordered[-1], 4), "mean_ms": round(statistics.fmean(ordered), 4)}


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = fn()
    return (time.perf_counter() - started) * 1000.0, result


def _unit_rows(rng: Any, n: int) -> Any:
    import numpy as np
    matrix = rng.standard_normal((n, DIM)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def _near(rng: Any, vector: Any, cosine: float = 0.8) -> Any:
    """A unit vector at roughly *cosine* to *vector* (a paraphrase stand-in)."""
    import numpy as np
    noise = rng.standard_normal(DIM).astype(np.float32)
    noise -= noise.dot(vector) * vector
    noise /= np.linalg.norm(noise)
    spread = (1.0 / cosine ** 2 - 1.0) ** 0.5
    out = vector + spread * noise
    return (out / np.linalg.norm(out)).astype(np.float32)


def _meta(label: str, variant: str) -> dict[str, Any]:
    import numpy as np
    import sqlite3
    info: dict[str, Any] = {
        "label": label, "variant": variant,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0], "implementation": platform.python_implementation(),
        "numpy": np.__version__, "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "blas_threads_env": {k: os.environ.get(k, "") for k in
                             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    try:
        import psutil
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    try:
        config = np.show_config(mode="dicts")
        blas = (config.get("Build Dependencies") or {}).get("blas") or {}
        info["blas"] = f"{blas.get('name', '')} {blas.get('version', '')}".strip()
    except Exception:
        pass
    return info


# ── ranking: the selection primitive alone ──────────────────────────────────

def bench_ranking(sizes: list[int], queries: int, seed: int) -> dict[str, Any]:
    import numpy as np

    def top_positions(scores: Any, k: int) -> Any:
        # Same contract as memory._top_positions: k best, best first, ties by
        # position. Local copy so the "legacy" run measures the same method.
        n = scores.shape[0]
        if k >= n:
            return np.argsort(-scores, kind="stable")
        kth = np.partition(scores, n - k)[n - k]
        above = np.flatnonzero(scores > kth)
        ties = np.flatnonzero(scores == kth)[:k - above.shape[0]]
        chosen = np.concatenate([above, ties])
        return chosen[np.argsort(-scores[chosen], kind="stable")]

    out: dict[str, Any] = {}
    for size in sizes:
        rng = np.random.default_rng(seed + size)
        matrix = _unit_rows(rng, size)
        ids = np.sort(rng.choice(size * 4, size=size, replace=False)).astype(np.int64)
        targets = rng.integers(0, size, size=queries)
        qs = [_near(rng, matrix[t]) for t in targets]
        lexical = [ids[rng.integers(0, size, size=8)] for _ in range(queries)]
        timings: dict[str, list[float]] = {"product": [], "legacy_select": [],
                                           "partial_select": []}
        recall = {8: [], 12: [], 32: []}
        for q, lex in zip(qs, lexical):
            ms, sims = _timed(lambda: matrix @ q)
            timings["product"].append(ms)

            def legacy() -> Any:
                top = np.argsort(-sims)[:8]
                position = {int(row_id): index for index, row_id in enumerate(ids)}
                return top, [position.get(int(i)) for i in lex]

            def partial() -> Any:
                top = top_positions(sims, 8)
                found = np.searchsorted(ids, lex)
                return top, found

            timings["legacy_select"].append(_timed(legacy)[0])
            timings["partial_select"].append(_timed(partial)[0])
            exact = np.argsort(-sims, kind="stable")
            for k in recall:
                got = set(top_positions(sims, k).tolist())
                recall[k].append(len(got & set(exact[:k].tolist())) / k)
        out[str(size)] = {
            "vectors": size, "dim": DIM, "queries": queries,
            "product": _percentiles(timings["product"]),
            "legacy_select": _percentiles(timings["legacy_select"]),
            "partial_select": _percentiles(timings["partial_select"]),
            "recall_at_k_vs_exact": {str(k): round(statistics.fmean(v), 6)
                                     for k, v in recall.items()},
            "matrix_mb": round(matrix.nbytes / 2 ** 20, 1),
        }
    return out


# ── store: the real matrix with a stand-in encoder ──────────────────────────

class _Bus:
    def __getattr__(self, _name: str) -> Any:
        class _Signal:
            def emit(self, *_a: Any) -> None:
                pass

            def connect(self, *_a: Any) -> None:
                pass
        return _Signal()


class StandInEncoder:
    """Deterministic text -> unit vector; planted texts map to planted vectors."""

    def __init__(self, per_row_s: float = 0.0) -> None:
        self.fixed: dict[str, Any] = {}
        self.per_row_s = per_row_s

    def ready(self) -> bool:
        return True

    def _vector(self, text: str) -> Any:
        import numpy as np
        planted = self.fixed.get(text)
        if planted is not None:
            return planted
        seed = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
                              "little")
        vector = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def encode(self, texts: Any, batch: int = 16) -> Any:
        import numpy as np
        if self.per_row_s:
            time.sleep(self.per_row_s * len(texts))
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.vstack([self._vector(str(t)) for t in texts])

    def encode_one(self, text: str) -> Any:
        return self._vector(str(text))


def _variant(matrix: Any) -> str:
    return "snapshot" if hasattr(matrix, "_build_snapshot") else "legacy"


def _wait_idle(matrix: Any, timeout: float = 60.0) -> None:
    waiter = getattr(matrix, "wait_semantic_idle", None)
    if waiter is not None:
        waiter(timeout)
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not getattr(matrix, "_semantic_pending", False) and \
                not getattr(matrix, "_semantic_dirty", False):
            return
        time.sleep(0.002)


def _rebuild(matrix: Any, source: str = "intelligence") -> None:
    """Build the vector snapshot from the store once, as a rebuild would."""
    if hasattr(matrix, "_build_snapshot"):
        matrix._build_snapshot(source)
    else:
        matrix._semantic_version += 1
        matrix._semantic_vectors(source)


def _vocabulary(rng: Any, size: int = 5000) -> list[str]:
    letters = "bcdfghjklmnprstvwz"
    vowels = "aeiou"
    words: set[str] = set()
    while len(words) < size:
        parts = rng.integers(0, 18, size=3), rng.integers(0, 5, size=3)
        words.add("".join(letters[c] + vowels[v] for c, v in zip(*parts)) + letters[int(rng.integers(0, 18))])
    return sorted(words)


def bench_store(sizes: list[int], queries: int, repeats: int, seed: int,
                workdir: Path, OrionMemoryMatrix: Any) -> tuple[dict[str, Any], str]:
    import numpy as np
    from orion_core import semantic

    out: dict[str, Any] = {}
    variant = "unknown"
    original_encoder = semantic.ENCODER
    try:
        for size in sizes:
            rng = np.random.default_rng(seed + 7 * size)
            vocab = _vocabulary(rng)
            folder = workdir / f"store_{size}"
            folder.mkdir(parents=True, exist_ok=True)
            semantic.ENCODER = StandInEncoder()
            # Not ready during the bulk load, so save() does not start embedding.
            semantic.ENCODER.ready = lambda: False      # type: ignore[method-assign]
            matrix = OrionMemoryMatrix(folder / "orion_core.db", folder, _Bus())
            variant = _variant(matrix)
            texts = [" ".join(vocab[int(i)] for i in rng.integers(0, len(vocab), size=8))
                     for _ in range(size)]
            load_ms, _ = _timed(lambda: _bulk_load(matrix, texts, rng))
            del semantic.ENCODER.ready                   # back to the class method
            with matrix._lock:
                ids = [int(r[0]) for r in matrix.conn.execute(
                    "SELECT id FROM intelligence ORDER BY id").fetchall()]
                stored = {int(r[0]): np.frombuffer(r[1], dtype=np.float16).astype(np.float32)
                          for r in matrix.conn.execute(
                              "SELECT row_id, vec FROM semantic_vectors WHERE source='intelligence'")}
            first_ms, _ = _timed(lambda: matrix.query(vocab[0], limit=8))   # cold build

            rebuild = [_timed(lambda: _rebuild(matrix))[0] for _ in range(repeats)]
            if variant == "legacy":
                matrix._semantic_vectors("intelligence")       # leave it warm

            # Planted paraphrases: one lexical word from the target (so FTS
            # returns a handful of rows, as a natural question does), a
            # vector near the target's.
            targets = [ids[int(i)] for i in rng.integers(0, len(ids), size=queries)]
            hit_q = []
            for n, target in enumerate(targets):
                text = f"{texts[target - 1].split()[n % 8]} probe{n}"
                semantic.ENCODER.fixed[text] = _near(rng, stored[target])
                hit_q.append((text, target))
            miss_q = []
            for n, target in enumerate(targets[: max(10, queries // 4)]):
                text = f"zqxmissing{n}"
                semantic.ENCODER.fixed[text] = _near(rng, stored[target])
                miss_q.append((text, target))

            def run(cases: list[tuple[str, int]],
                    fn: Callable[[str], list[dict]]) -> tuple[list[float], int, int]:
                times, top1, topk = [], 0, 0
                for text, target in cases:
                    ms, rows = _timed(lambda: fn(text))
                    times.append(ms)
                    wanted = f"k{target}"
                    got = [r.get("key_ref") for r in rows]
                    top1 += bool(got) and got[0] == wanted
                    topk += wanted in got
                return times, top1, topk

            q_ms, q_top1, q_topk = run(hit_q, lambda t: matrix.query(t, limit=8))
            m_ms, m_top1, m_topk = run(miss_q, lambda t: matrix.query(t, limit=8))
            r_ms, r_top1, r_topk = run(hit_q, lambda t: matrix.relevant(t, limit=3))

            # The first query after an index change: a save is embedded in the
            # background, then the next question is timed.
            # Also how long the background work after one save took (save
            # returned -> indexer idle), which competes with everything else.
            after, background = [], []
            for n in range(repeats):
                matrix.save("bench", f"change_{n}", f"fresh fact number {n} {vocab[n]}")
                background.append(_timed(lambda: _wait_idle(matrix))[0])
                after.append(_timed(lambda: matrix.query(hit_q[n % len(hit_q)][0], limit=8))[0])
            one_change = _one_change_publish(matrix, repeats)

            during = _during_backfill(matrix, vocab, rng, hit_q, size)
            out[str(size)] = {
                "records": size, "queries": len(hit_q), "lexical_miss_queries": len(miss_q),
                "bulk_load_ms": round(load_ms, 1),
                "first_query_cold_build_ms": round(first_ms, 3),
                "rebuild": _percentiles(rebuild),
                "query": _percentiles(q_ms),
                "query_lexical_miss": _percentiles(m_ms),
                "relevant": _percentiles(r_ms),
                "query_after_change": _percentiles(after),
                "background_per_save": _percentiles(background),
                "publish_one_change": one_change,
                "during_backfill": during,
                "hit_at_1": round(q_top1 / len(hit_q), 4),
                "hit_at_8": round(q_topk / len(hit_q), 4),
                "lexical_miss_hit_at_1": round(m_top1 / len(miss_q), 4),
                "relevant_hit_at_1": round(r_top1 / len(hit_q), 4),
                "relevant_hit_at_3": round(r_topk / len(hit_q), 4),
            }
            _wait_idle(matrix)
            matrix.close()
    finally:
        semantic.ENCODER = original_encoder
    return out, variant


def _one_change_publish(matrix: Any, repeats: int) -> dict[str, Any]:
    """Cost of bringing a published snapshot up to date after one new vector,
    the work a single save now causes. Only the snapshot implementation has it."""
    import numpy as np
    from orion_core import semantic
    current = getattr(matrix, "_snapshots", {}).get("intelligence")
    if current is None or not hasattr(matrix, "_apply_changes"):
        return {"n": 0, "note": "not applicable: this implementation rebuilds in full"}
    blob = np.asarray(current.matrix[0], dtype=np.float16).tobytes()
    change = [(current.generation + 1, ((int(current.ids[-1]) + 1, semantic.text_digest("x"), blob),), ())]
    return _percentiles([_timed(lambda: matrix._apply_changes(current, change, current.generation + 1))[0]
                         for _ in range(repeats)])


def _bulk_load(matrix: Any, texts: list[str], rng: Any) -> None:
    import numpy as np
    from orion_core import semantic
    stamp = "2026-09-26T00:00:00"
    vectors = _unit_rows(rng, len(texts))
    # A fresh store: ids 1..n, and key_ref names the id so a hit is checkable.
    ids = list(range(1, len(texts) + 1))
    with matrix._lock:
        matrix.conn.executemany(
            "INSERT INTO intelligence(id, category, key_ref, value, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(i, "bench", f"k{i}", t, stamp) for i, t in zip(ids, texts)])
        matrix.conn.executemany(
            "INSERT OR REPLACE INTO semantic_vectors(source, row_id, digest, vec) VALUES (?,?,?,?)",
            [("intelligence", i, semantic.text_digest(t), np.asarray(v, np.float16).tobytes())
             for i, t, v in zip(ids, texts, vectors)])
        matrix.conn.commit()


def _during_backfill(matrix: Any, vocab: list[str], rng: Any,
                     cases: list[tuple[str, int]], size: int) -> dict[str, Any]:
    """Question latency while the indexer writes vectors in the background.

    512 new rows are inserted without vectors, then index_semantics() runs in
    a thread with a SIMULATED 1 ms per row of encoding (the real encoder is
    ~13-20 ms a row, so this is a compressed, harsher schedule of batch
    commits). Every commit used to invalidate the cached matrix, so the next
    question rebuilt it inline.
    """
    from orion_core import semantic
    stamp = "2026-09-26T00:00:00"
    with matrix._lock:
        matrix.conn.executemany(
            "INSERT INTO intelligence(category, key_ref, value, updated_at) VALUES (?, ?, ?, ?)",
            [("backfill", f"b{n}", " ".join(vocab[int(i)] for i in rng.integers(0, len(vocab), 8)),
              stamp) for n in range(512)])
        matrix.conn.commit()
    semantic.ENCODER.per_row_s = 0.001
    worker = threading.Thread(target=matrix.index_semantics, daemon=True)
    times: list[float] = []
    worker.start()
    n = 0
    while worker.is_alive():
        text = cases[n % len(cases)][0]
        times.append(_timed(lambda: matrix.query(text, limit=8))[0])
        n += 1
        time.sleep(0.005)
    worker.join()
    semantic.ENCODER.per_row_s = 0.0
    return {"rows_backfilled": 512, "simulated_encode_ms_per_row": 1.0,
            **_percentiles(times)}


# ── quality: the real encoder on labelled questions ─────────────────────────

_DISTRACTOR_PEOPLE = ("Alex", "Jordan", "Priya", "Tom", "Megan", "Leo", "Aisha", "Callum")
_DISTRACTOR_TEMPLATES = (
    "{p} bought a second-hand {thing} in {month} for weekend trips",
    "{p} is allergic to {allergen} and carries antihistamines",
    "{p} rents a flat in {place} for {amount} a month",
    "{p}'s father's birthday is the {day} of {month}",
    "{p} swims at the {place} leisure centre on Thursday evenings",
    "The {org} charges {amount} pounds per {unit}",
    "The preferred supplier for {goods} is {brand}",
    "Invoices for {org} are paid within {days} days",
    "The {org} bookkeeper is {brand}, contacted every month",
    "The {subject} coursework is due on the {day} of {month}",
    "Dr {surname} teaches the {subject} seminar on Tuesdays",
    "The {subject} module is worth {credits} credits",
    "The {place} gallery closes at {hour}pm at weekends",
    "{brand} is a mobile app built with Flutter and deployed from main",
)
_FILL = {
    "thing": ("kayak", "tent", "scooter", "camera", "guitar"),
    "month": ("January", "April", "June", "August", "November"),
    "allergen": ("penicillin", "peanuts", "latex", "shellfish", "pollen"),
    "place": ("Riverside", "Hillcrest", "Southbank", "Oakfield", "Kingsway"),
    "amount": ("520", "610", "880", "950", "1200"),
    "day": ("2nd", "9th", "17th", "21st", "28th"),
    "org": ("print shop", "design collective", "photo studio", "radio station"),
    "unit": ("poster", "logo", "photo session", "voice-over"),
    "goods": ("stickers", "banners", "packaging", "badges"),
    "brand": ("InkHouse", "Mapleleaf Ltd", "BrightCo", "Northwind", "Pixelforge"),
    "days": ("14", "45", "60"),
    "subject": ("robotics", "microeconomics", "organic chemistry", "art history"),
    "surname": ("Okafor", "Lindqvist", "Moreau", "Tanaka"),
    "credits": ("10", "20", "30"),
    "hour": ("5", "6", "8"),
}


def _distractors(count: int, seed: int) -> list[tuple[str, str, str]]:
    import random
    chooser = random.Random(seed)
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    while len(rows) < count:
        template = chooser.choice(_DISTRACTOR_TEMPLATES)
        values = {k: chooser.choice(v) for k, v in _FILL.items()}
        values["p"] = chooser.choice(_DISTRACTOR_PEOPLE)
        text = template.format(**values)
        if text in seen:
            continue
        seen.add(text)
        rows.append(("distractor", f"d{len(rows)}", text))
    return rows


def _find_local_model() -> Path | None:
    """The encoder's files if they are already on this machine. Never fetches."""
    from orion_core import semantic
    from orion_core.model_store import usable
    spec = semantic.MODEL
    candidates = [ROOT / "config" / "models" / spec.name,
                  ROOT / "assets" / "models" / spec.name,
                  Path.home() / "ORION" / "config" / "models" / spec.name]
    for folder in candidates:
        try:
            if all(usable(folder / name, size=item.size) for name, item in spec.files.items()):
                return folder
        except Exception:
            continue
    return None


def bench_quality(workdir: Path, distractors: int, seed: int,
                  OrionMemoryMatrix: Any) -> dict[str, Any]:
    from orion_core import semantic
    sys.path.insert(0, str(ROOT / "tests"))
    from data.memory_recall_cases import FACTS, dev_cases, holdout_cases

    folder = _find_local_model()
    if folder is None:
        return {"skipped": "the encoder is not installed locally (never downloaded here)"}
    encoder = semantic.SemanticEncoder()
    encoder.model_dir = lambda: folder                    # type: ignore[method-assign]
    if not encoder.load():
        return {"skipped": f"the local encoder would not load: {encoder.last_error}"}
    original = semantic.ENCODER
    semantic.ENCODER = encoder
    try:
        store_dir = workdir / "quality"
        store_dir.mkdir(parents=True, exist_ok=True)
        matrix = OrionMemoryMatrix(store_dir / "orion_core.db", store_dir, _Bus())
        rows = list(FACTS) + _distractors(distractors, seed)
        with matrix._lock:
            matrix.conn.executemany(
                "INSERT INTO intelligence(category, key_ref, value, updated_at) VALUES (?,?,?,?)",
                [(c, k, v, "2026-09-26T00:00:00") for c, k, v in rows])
            matrix.conn.commit()
        embed_ms, embedded = _timed(matrix.index_semantics)
        _wait_idle(matrix)

        def score(cases: list[tuple[str, str]]) -> dict[str, Any]:
            top1 = hit8 = rel = 0
            times = []
            for question, expected in cases:
                ms, found = _timed(lambda: matrix.query(question, limit=8))
                times.append(ms)
                keys = [r["key_ref"] for r in found]
                top1 += bool(keys) and keys[0] == expected
                hit8 += expected in keys
                rel += expected in [r["key_ref"] for r in matrix.relevant(question, limit=3)]
            n = max(1, len(cases))
            return {"cases": len(cases), "top1": round(top1 / n, 4), "hit_at_8": round(hit8 / n, 4),
                    "relevant_hit_at_3": round(rel / n, 4), "query": _percentiles(times)}

        # Uncached question encoding, for scale against the ranking cost.
        encode = [_timed(lambda: encoder.encode([f"benchmark question number {i} about rent"]))[0]
                  for i in range(30)]
        result = {"model": semantic.MODEL.name, "facts": len(FACTS), "distractors": distractors,
                  "embedded_rows": embedded, "embed_pass_ms": round(embed_ms, 1),
                  "dev": score(dev_cases()), "holdout": score(holdout_cases()),
                  "encode_question_uncached": _percentiles(encode)}
        matrix.close()
        return result
    finally:
        semantic.ENCODER = original


# ── entry point ──────────────────────────────────────────────────────────────

def _matrix_class(module_path: str) -> Any:
    """OrionMemoryMatrix from the package, or from another copy of memory.py
    loaded as a module of the same package (so its relative imports resolve)."""
    if not module_path:
        from orion_core.memory import OrionMemoryMatrix
        return OrionMemoryMatrix
    import importlib.util
    import orion_core
    spec = importlib.util.spec_from_file_location("orion_core._memory_under_test",
                                                  Path(module_path).resolve())
    module = importlib.util.module_from_spec(spec)
    module.__package__ = orion_core.__name__
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OrionMemoryMatrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="where to write the JSON results")
    parser.add_argument("--sizes", default="1000,10000,50000")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--distractors", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-encoder", action="store_true",
                        help="skip the real-encoder quality section")
    parser.add_argument("--memory-module", default="",
                        help="measure this copy of orion_core/memory.py instead (e.g. a saved "
                             "earlier version), for a before/after on the same machine")
    args = parser.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    with tempfile.TemporaryDirectory(prefix="orion-bench-retrieval-") as scratch:
        workdir = Path(scratch)
        # Isolation first: orion_core resolves CONFIG_DIR at import time.
        os.environ["ORION_CONFIG_DIR"] = str(workdir / "config")
        sys.path.insert(0, str(ROOT))
        from orion_core import constants
        if Path(constants.CONFIG_DIR).resolve() != (workdir / "config").resolve():
            print("refusing to run: CONFIG_DIR is not the scratch directory", file=sys.stderr)
            return 2

        matrix_class = _matrix_class(args.memory_module)
        results: dict[str, Any] = {}
        print("ranking...", flush=True)
        results["ranking"] = bench_ranking(sizes, args.queries, args.seed)
        print("store...", flush=True)
        results["store"], variant = bench_store(sizes, args.queries, args.repeats,
                                                args.seed, workdir, matrix_class)
        if args.no_encoder:
            results["quality"] = {"skipped": "--no-encoder"}
        else:
            print("quality...", flush=True)
            results["quality"] = bench_quality(workdir, args.distractors, args.seed,
                                               matrix_class)
        results["meta"] = _meta(args.label, variant)
        results["meta"]["argv"] = {"sizes": sizes, "queries": args.queries,
                                   "repeats": args.repeats, "seed": args.seed,
                                   "memory_module": args.memory_module or "orion_core/memory.py"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
