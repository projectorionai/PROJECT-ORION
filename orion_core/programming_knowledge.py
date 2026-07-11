"""
Programming knowledge base — extensive software-engineering expertise.

Mirrors the neuroscience knowledge base pattern: a curated corpus of real
programming knowledge (languages, paradigms, data structures, algorithms,
complexity, design patterns, concurrency, testing, security and common bugs)
seeded idempotently into the KNOWLEDGE memory tier, plus a persona boost that
sharpens ORION's coding register and a query API for offline recall.

This gives the LocalBrain and the offline model real, retrievable engineering
knowledge with no internet, and grounds ORION's answers in fundamentals rather
than guesswork.
"""

from __future__ import annotations

import re
from typing import Any, Optional

SEED_MARKER = "programming_corpus_seeded_v1"

# (topic, fact) — real, load-bearing software-engineering knowledge.
CORPUS: list[tuple[str, str]] = [
    ("Big-O complexity", "Big-O describes how work grows with input size n. O(1) constant, O(log n) "
     "binary search, O(n) linear scan, O(n log n) good sorts, O(n^2) nested loops, O(2^n) brute-force "
     "subsets. Optimise the dominant term; constants matter in practice but not asymptotically."),
    ("Data structures", "Arrays: O(1) index, O(n) insert-middle. Hash maps: average O(1) lookup, "
     "worst O(n) on collisions. Balanced trees: O(log n) ordered ops. Heaps: O(log n) push/pop, O(1) "
     "peek — ideal for priority queues. Choose by the operations you do most."),
    ("Sorting", "Quicksort averages O(n log n) but O(n^2) worst-case on bad pivots; mergesort is stable "
     "O(n log n) with O(n) space; timsort (Python's sort) exploits existing runs. Prefer the library sort."),
    ("Recursion vs iteration", "Recursion expresses divide-and-conquer cleanly but costs stack frames; "
     "deep recursion risks stack overflow. Convert to iteration with an explicit stack, or use "
     "tail-call/accumulator style, when depth is unbounded."),
    ("Dynamic programming", "DP solves problems with overlapping subproblems and optimal substructure by "
     "memoising results. Define the state, the recurrence and the base case; top-down memoisation or "
     "bottom-up tabulation both work."),
    ("Concurrency vs parallelism", "Concurrency is dealing with many things at once (interleaving, I/O "
     "bound); parallelism is doing many things at once (CPU bound, multiple cores). Async/await suits "
     "I/O; threads share memory (mind the GIL in CPython); processes give true parallelism."),
    ("Race conditions & locks", "A race condition is when correctness depends on timing of unsynchronised "
     "access to shared state. Guard shared mutable state with a lock, prefer immutability, or pass "
     "messages between tasks. Hold locks briefly; acquire in a consistent order to avoid deadlock."),
    ("Deadlock", "Four Coffman conditions: mutual exclusion, hold-and-wait, no pre-emption, circular wait. "
     "Break any one — e.g. always acquire locks in a global order, or use timeouts."),
    ("Memory management", "Stack holds call frames (automatic); heap holds dynamic objects. Leaks come from "
     "unreleased references; in GC languages, from lingering references (caches, listeners). Bound caches "
     "and detach event handlers."),
    ("Pointers & references", "A pointer/reference is an address of a value. Aliasing means two names refer "
     "to one object; mutating through one is visible through the other. Copy when you need independence."),
    ("Design patterns", "Creational (factory, builder, singleton), structural (adapter, decorator, facade), "
     "behavioural (observer, strategy, command). Patterns name recurring solutions; don't force them — "
     "reach for one when the problem shape matches."),
    ("SOLID", "Single-responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency "
     "inversion. They reduce coupling and ease change; apply pragmatically, not dogmatically."),
    ("DRY / KISS / YAGNI", "Don't Repeat Yourself (one source of truth), Keep It Simple, You Aren't Gonna "
     "Need It (don't build for imagined futures). Duplication is cheaper than the wrong abstraction."),
    ("REST APIs", "Resources as nouns, HTTP verbs as actions: GET (read, safe), POST (create), PUT/PATCH "
     "(update), DELETE. Use status codes meaningfully (2xx ok, 4xx client error, 5xx server). Make GET "
     "idempotent and cacheable."),
    ("HTTP & status codes", "200 OK, 201 Created, 204 No Content, 301/302 redirects, 400 Bad Request, 401 "
     "Unauthorised, 403 Forbidden, 404 Not Found, 429 Too Many Requests, 500 Server Error, 502/503 "
     "upstream/unavailable."),
    ("Databases & ACID", "ACID: Atomicity, Consistency, Isolation, Durability. Index the columns you filter "
     "or join on (trades write speed and space for read speed). Normalise to remove redundancy; denormalise "
     "deliberately for read performance."),
    ("SQL vs NoSQL", "Relational DBs give joins, transactions and schema; document/key-value stores give "
     "flexible schema and horizontal scale. Choose by access patterns and consistency needs, not hype."),
    ("Version control (git)", "Commit small, logical units; write imperative messages. Branch per feature; "
     "rebase to tidy local history, merge to integrate. Never rewrite shared history. A conflict marks "
     "where two edits touched the same lines."),
    ("Testing", "Unit tests isolate a function; integration tests check components together; end-to-end "
     "tests exercise the whole flow. Aim for fast, deterministic tests. Test behaviour and edge cases, "
     "not implementation details."),
    ("Debugging method", "Reproduce reliably, isolate by bisection, read the actual error and stack trace, "
     "form one hypothesis at a time and test it, and check your assumptions before the code. Rubber-duck "
     "the problem aloud."),
    ("Common bugs", "Off-by-one errors, null/None dereferences, mutable default arguments, integer overflow, "
     "floating-point equality, unclosed resources, and timezone/encoding assumptions. Most bugs live at "
     "boundaries."),
    ("Security basics", "Never trust input: validate and parameterise (defeats SQL injection). Escape output "
     "(defeats XSS). Store passwords hashed with a slow salted algorithm (bcrypt/argon2). Least privilege; "
     "keep secrets out of source; keep dependencies patched."),
    ("Python specifics", "The GIL serialises bytecode so threads don't parallelise CPU work — use processes "
     "or async for that. Everything is an object; default args evaluate once (never use mutable defaults). "
     "List/dict/set comprehensions are idiomatic; generators stream lazily."),
    ("Async programming", "An event loop runs coroutines cooperatively; await yields control on I/O. Never "
     "block the loop with sync CPU or blocking I/O — offload to a thread/process. One slow await starves "
     "everything on that loop."),
    ("Functional concepts", "Pure functions (no side effects, same input → same output) are easy to test and "
     "parallelise. Prefer immutability; map/filter/reduce express transforms; higher-order functions take or "
     "return functions."),
    ("Type systems", "Static typing catches errors before running and documents intent; dynamic typing is "
     "flexible and terse. Gradual typing (type hints) gives static checks where they pay off without full "
     "rigidity."),
    ("Clean code", "Name things for what they mean; small functions that do one thing; comment the why, not "
     "the what; keep functions at one level of abstraction; fail fast with clear errors. Readability beats "
     "cleverness."),
    ("Refactoring", "Change structure without changing behaviour, in small safe steps backed by tests. Extract "
     "function/variable, rename, inline, and remove duplication. Refactor before adding a feature to a messy "
     "area, not after."),
    ("Networking basics", "TCP is reliable, ordered, connection-based; UDP is fast, best-effort. DNS resolves "
     "names to addresses. TLS encrypts and authenticates. Latency and bandwidth are different limits."),
    ("Caching", "Cache to trade memory for speed, but invalidation is hard. Set TTLs; key precisely; beware "
     "stale reads and thundering herds. The two hard things: naming, cache invalidation, and off-by-one errors."),
]

PROGRAMMING_PERSONA_BOOST = (
    "You carry deep, current software-engineering expertise across Python, JavaScript/TypeScript, "
    "systems programming, algorithms and data structures, complexity analysis, concurrency, "
    "databases, networking, security, testing and design patterns. When coding: reason from "
    "fundamentals, state complexity and trade-offs, prefer minimal correct idiomatic solutions, "
    "never invent APIs, and when debugging start from the actual error and stack trace."
)


class ProgrammingKnowledgeBase:
    """Seeds and serves the programming corpus (mirrors NeuroKnowledgeBase)."""

    KEYWORDS = (
        "code", "coding", "program", "algorithm", "complexity", "big-o", "data structure",
        "recursion", "concurrency", "async", "thread", "deadlock", "race condition", "pattern",
        "solid", "rest", "http", "sql", "database", "git", "test", "debug", "python", "javascript",
        "typescript", "rust", "memory leak", "pointer", "cache", "refactor", "security", "api",
    )

    def __init__(self, telemetry: Any | None = None) -> None:
        self.telemetry = telemetry

    def seed(self, memory: Any) -> int:
        """Idempotently write the corpus into the KNOWLEDGE tier."""
        try:
            existing = memory.query(SEED_MARKER, limit=1)
            if existing and any(SEED_MARKER in str(r.get("value", "")) for r in existing):
                return 0
        except Exception:
            pass
        count = 0
        for topic, fact in CORPUS:
            try:
                key = "prog_" + re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:44]
                memory.remember("knowledge", key, f"{topic}: {fact}")
                count += 1
            except Exception:
                continue
        try:
            memory.remember("knowledge", "prog_seed_marker", SEED_MARKER)
        except Exception:
            pass
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("programming.corpus", float(count))
        return count

    def is_programming_query(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(k in low for k in self.KEYWORDS)

    def answer(self, query: str) -> Optional[str]:
        low = str(query or "").lower()
        best: Optional[tuple[int, str, str]] = None
        for topic, fact in CORPUS:
            score = sum(1 for w in re.findall(r"[a-z0-9+]+", low) if w in (topic + " " + fact).lower())
            if score and (best is None or score > best[0]):
                best = (score, topic, fact)
        if best is None:
            return None
        return f"{best[1]}: {best[2]}"
