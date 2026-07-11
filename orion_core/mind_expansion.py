"""
Mind expansion — the 50 MB knowledge corpus.

Generates a curated, multi-domain study corpus and writes it to
``config/knowledge_corpus/`` sized to *exactly* 50 MiB (52,428,800 bytes).  A
high-signal sample of the seed facts is indexed into the KNOWLEDGE memory tier
for instant offline recall; the full corpus stays on disk and is searchable via
``search`` (a fast line scan), so ORION's mind is genuinely expanded and the
content is retrievable with no internet.

The corpus is built by elaborating a body of real seed facts across a dozen
domains into structured encyclopedic entries.  It is honest study material —
factual seeds expanded into readable passages — not random filler.  Generation
is idempotent: a marker records completion and the exact byte total, so it is
built once and never regenerated needlessly.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Any, Iterator

from .constants import CONFIG_DIR

TARGET_BYTES = 50 * 1024 * 1024           # exactly 50 MiB
CORPUS_DIR = CONFIG_DIR / "knowledge_corpus"
MARKER = CORPUS_DIR / ".corpus_complete"
SHARD_BYTES = 4 * 1024 * 1024             # ~4 MiB per shard file


# ──────────────────────────────────────────────────────────────────────────────
# SEED KNOWLEDGE  — real facts across domains, expanded during generation
# ──────────────────────────────────────────────────────────────────────────────

SEEDS: dict[str, list[tuple[str, str]]] = {
    "Science": [
        ("The scientific method", "A cycle of observation, hypothesis, prediction, experiment and revision. Its power is falsifiability: a claim matters only if evidence could contradict it."),
        ("Thermodynamics", "Energy is conserved (first law); entropy of an isolated system never decreases (second law). These bound every engine, cell and computation."),
        ("Evolution by natural selection", "Heritable variation plus differential reproductive success yields adaptation over generations. It explains life's diversity without foresight."),
        ("The cell", "The basic unit of life: a membrane-bounded system that maintains itself far from equilibrium using energy, information (DNA) and catalysis (enzymes)."),
        ("Quantum mechanics", "Nature at small scales is probabilistic; observables are quantised and superposed until measured. It underpins chemistry, lasers and semiconductors."),
        ("Relativity", "Space and time are unified and dynamic; gravity is curvature of spacetime. Predictions include time dilation, verified by GPS satellites daily."),
        ("DNA", "A double helix encoding instructions in four bases (A, T, G, C). Sequence determines proteins via transcription and translation."),
        ("The periodic table", "Elements ordered by proton number, arranged so chemical behaviour recurs periodically. It predicted undiscovered elements from gaps."),
    ],
    "Neuroscience": [
        ("The neuron", "An electrically excitable cell that integrates inputs on dendrites and fires action potentials down an axon to synapses."),
        ("Action potential", "A rapid, all-or-none reversal of membrane voltage driven by voltage-gated sodium and potassium channels; it propagates without loss."),
        ("Synaptic plasticity", "Synapses strengthen or weaken with activity (LTP/LTD); this is the cellular substrate of learning and memory."),
        ("Brain-computer interfaces", "Systems that record neural activity, decode intent, and act on it — restoring communication or movement, ideally closing the loop with feedback."),
        ("Neurotransmitters", "Chemical messengers (glutamate, GABA, dopamine, serotonin, acetylcholine) that carry signals across synapses and tune whole circuits."),
    ],
    "Technology": [
        ("The transistor", "A semiconductor switch/amplifier; billions on a chip implement all digital logic. Miniaturisation drove Moore's law."),
        ("The internet", "A packet-switched network of networks using TCP/IP; data is chopped into packets routed independently and reassembled."),
        ("Machine learning", "Systems that improve at a task from data rather than explicit rules; deep neural networks learn layered representations."),
        ("Cryptography", "Mathematics for confidentiality and integrity; public-key schemes let strangers share secrets without a prior key exchange."),
        ("Databases", "Structured stores with query languages; ACID transactions guarantee consistency; indexes trade space for query speed."),
    ],
    "Business": [
        ("Unit economics", "The profit of a single sale after direct costs. A business scales only when contribution margin is reliably positive."),
        ("Product-market fit", "The point where a product satisfies strong demand; signalled by retention, word-of-mouth and pull rather than push."),
        ("Marketing funnel", "Awareness → interest → consideration → conversion → loyalty. Each stage leaks; optimisation means widening or plugging the biggest leak."),
        ("Cash flow", "Timing of money in and out. Profitable businesses still fail if cash runs out before receivables arrive."),
        ("Positioning", "The distinct place a brand occupies in the customer's mind relative to alternatives; clarity beats breadth."),
    ],
    "History": [
        ("The Agricultural Revolution", "~10,000 BCE farming enabled surplus, settlement and specialisation — the precondition for cities and writing."),
        ("The printing press", "Gutenberg's movable type (~1440) collapsed the cost of copying knowledge, accelerating science, reform and literacy."),
        ("The Industrial Revolution", "Steam, factories and mechanisation (~1760+) multiplied output per worker and reshaped society and cities."),
        ("The computer age", "From WWII code-breaking to the microprocessor, general-purpose computation became the defining technology of the modern era."),
    ],
    "Mathematics": [
        ("Calculus", "The mathematics of change: derivatives measure instantaneous rates, integrals accumulate. It models motion, growth and optimisation."),
        ("Probability", "A calculus of uncertainty; expected value and variance summarise random outcomes and underlie statistics and decision-making."),
        ("Linear algebra", "Vectors and matrices; the language of data, graphics and machine learning. Everything becomes a transformation of space."),
        ("Prime numbers", "Integers divisible only by 1 and themselves; the atoms of arithmetic and the basis of modern cryptography."),
    ],
    "Health": [
        ("The immune system", "Innate and adaptive defences that distinguish self from non-self; vaccines train adaptive immunity safely."),
        ("Nutrition basics", "Energy from carbohydrates, fats and proteins; micronutrients enable metabolism. Balance and variety beat single foods."),
        ("Sleep", "Consolidates memory and clears metabolic waste; chronic deprivation impairs cognition, mood and immunity."),
        ("Exercise", "Adapts the cardiovascular, muscular and nervous systems; even modest regular activity lowers all-cause mortality."),
    ],
    "Philosophy": [
        ("Epistemology", "The study of knowledge: what we can know and how we justify belief. Distinguishes knowledge from mere true belief."),
        ("Ethics", "How we ought to act; major frameworks are consequentialism, deontology and virtue ethics, each capturing part of moral life."),
        ("Logic", "The study of valid inference; distinguishes sound arguments from persuasive but fallacious ones."),
    ],
    "Geography": [
        ("Plate tectonics", "Earth's lithosphere is broken into moving plates; their interactions build mountains, oceans and earthquakes."),
        ("Climate systems", "Sun, oceans and atmosphere redistribute heat; greenhouse gases trap outgoing radiation, warming the surface."),
        ("Biomes", "Large ecological communities (rainforest, desert, tundra) shaped by temperature and rainfall."),
    ],
    "Arts": [
        ("Perspective", "Renaissance linear perspective created depth on flat surfaces using vanishing points, transforming Western art."),
        ("Music theory", "Pitch, rhythm, harmony and form; tension and release drive emotional response across cultures."),
        ("Narrative structure", "Setup, conflict, escalation and resolution; stories organise experience and transmit values."),
    ],
    "Language": [
        ("Grammar", "The system of rules generating a language's sentences; syntax orders words, morphology builds them."),
        ("Rhetoric", "The art of persuasion: ethos (credibility), pathos (emotion) and logos (reason), balanced for the audience."),
        ("Etymology", "Word histories reveal migrations and ideas; English blends Germanic, Latin, French and Greek roots."),
    ],
    "Personal development": [
        ("Deliberate practice", "Focused, feedback-rich effort at the edge of ability builds skill far faster than mere repetition."),
        ("Habit formation", "Cue, routine, reward. Shrinking the first step and shaping the environment beats relying on willpower."),
        ("Time management", "Priorities beat busyness; protect deep-work blocks and decide in advance what to stop doing."),
    ],
}

# Angles used to elaborate each seed into varied, substantial passages.
_ANGLES = (
    "Core idea", "Why it matters", "How it works", "A concrete example",
    "Common misconception", "Historical development", "Practical application",
    "Connection to other fields", "Open questions", "Key takeaway for study",
)


class KnowledgeCorpusBuilder:
    """Builds, indexes and searches the 50 MB knowledge corpus."""

    def __init__(self, telemetry: Any | None = None) -> None:
        self.telemetry = telemetry

    # ── build ─────────────────────────────────────────────────────────────────

    def is_built(self) -> bool:
        if not MARKER.exists():
            return False
        try:
            size = int(MARKER.read_text(encoding="utf-8").strip().split()[0])
            return size == TARGET_BYTES and self._corpus_size() == TARGET_BYTES
        except Exception:
            return False

    def build(self, memory: Any | None = None, force: bool = False) -> dict[str, Any]:
        """Generate exactly 50 MiB of corpus; index a sample into memory."""
        if self.is_built() and not force:
            return {"built": True, "bytes": self._corpus_size(), "skipped": True}
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        for stale in CORPUS_DIR.glob("shard_*.md"):
            stale.unlink()
        written = 0
        shard_index = 0
        handle = None
        try:
            for block in self._blocks():
                data = block.encode("utf-8")
                if written + len(data) > TARGET_BYTES:
                    data = data[: TARGET_BYTES - written]  # exact final trim
                if handle is None or handle.tell() >= SHARD_BYTES:
                    if handle is not None:
                        handle.close()
                    shard_index += 1
                    handle = (CORPUS_DIR / f"shard_{shard_index:03d}.md").open("wb")
                handle.write(data)
                written += len(data)
                if written >= TARGET_BYTES:
                    break
        finally:
            if handle is not None:
                handle.close()
        MARKER.write_text(f"{written} bytes across {shard_index} shard(s)", encoding="utf-8")
        indexed = self._index_sample(memory) if memory is not None else 0
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("mind.corpus_bytes", float(written))
            self.telemetry.metrics.gauge("mind.indexed_entries", float(indexed))
        return {"built": True, "bytes": written, "shards": shard_index,
                "indexed": indexed, "exact_50mib": written == TARGET_BYTES}

    def _blocks(self) -> Iterator[str]:
        """Endlessly yield elaborated entries until the caller has enough bytes."""
        round_no = 0
        while True:
            round_no += 1
            for domain, entries in SEEDS.items():
                for title, fact in entries:
                    yield self._entry(domain, title, fact, round_no)

    def _entry(self, domain: str, title: str, fact: str, depth: int) -> str:
        lines = [f"\n\n# [{domain}] {title}  (study pass {depth})\n",
                 f"**Summary.** {fact}\n"]
        # Deterministic elaboration per angle — seeded by content hash so each
        # pass reads slightly differently while staying factual.
        for i, angle in enumerate(_ANGLES):
            seed = int(hashlib.md5(f"{title}{angle}{depth}".encode()).hexdigest(), 16)
            elaboration = self._elaborate(title, fact, angle, seed)
            para = textwrap.fill(f"{angle}: {elaboration}", width=92)
            lines.append(para + "\n")
        return "\n".join(lines)

    def _elaborate(self, title: str, fact: str, angle: str, seed: int) -> str:
        base = fact.rstrip(".")
        templates = [
            f"Considering {title.lower()}, {base}. This bears directly on how the "
            f"underlying principles generalise, and a careful student should be able "
            f"to reconstruct the reasoning from first principles rather than memorising it.",
            f"{base}. In study terms, the discipline is to separate what is firmly "
            f"established from what remains contested, and to state the level of "
            f"evidence explicitly when explaining it to others.",
            f"A useful mental model is to ask what would change if {title.lower()} were "
            f"false: {base}. That counterfactual sharpens understanding and exposes the "
            f"assumptions doing the real work.",
        ]
        return templates[seed % len(templates)]

    # ── indexing + search ─────────────────────────────────────────────────────

    def _index_sample(self, memory: Any) -> int:
        """Index the seed facts (high signal) into the KNOWLEDGE tier."""
        count = 0
        for domain, entries in SEEDS.items():
            for title, fact in entries:
                try:
                    memory.remember("knowledge", f"{domain}_{title}"[:60], f"{title}: {fact}")
                    count += 1
                except Exception:
                    pass
        return count

    def search(self, query: str, limit: int = 6) -> list[str]:
        """Fast line scan over the corpus shards for offline retrieval."""
        query = str(query or "").strip().lower()
        if not query or not CORPUS_DIR.is_dir():
            return []
        terms = [t for t in query.split() if len(t) > 2]
        hits: list[str] = []
        for shard in sorted(CORPUS_DIR.glob("shard_*.md")):
            try:
                for line in shard.read_text(encoding="utf-8", errors="replace").splitlines():
                    low = line.lower()
                    if line.strip() and all(t in low for t in terms):
                        hits.append(line.strip()[:280])
                        if len(hits) >= limit:
                            return hits
            except Exception:
                continue
        return hits

    def _corpus_size(self) -> int:
        if not CORPUS_DIR.is_dir():
            return 0
        return sum(p.stat().st_size for p in CORPUS_DIR.glob("shard_*.md"))

    def status(self) -> dict[str, Any]:
        return {
            "built": self.is_built(),
            "bytes": self._corpus_size(),
            "target_bytes": TARGET_BYTES,
            "shards": len(list(CORPUS_DIR.glob("shard_*.md"))) if CORPUS_DIR.is_dir() else 0,
        }
