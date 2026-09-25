"""
Creator Studio Intelligence Suite (Phase 3).

Business intelligence for a short-form content operation — TikTok, Instagram
Reels, YouTube Shorts and Facebook Reels — covering UGC production, creator
management and performance optimisation. This suite replaces the earlier
storefront-era commerce focus with the agency's actual business.

    HookAnalyzer           — deterministic 0-10 scoring of opening hooks
    ScriptEvaluator        — hook/body/CTA breakdown, pacing, retention loops
    CreatorManager         — creator roster + submission reviews (durable JSON)
    CreatorPerformanceTracker — per-creator score history and trends
    ProductResearchAgent   — product angles for short-form commerce
    ViralAnalysisAgent     — recurring patterns across example scripts
    ContentStrategyAgent   — platform mix, cadence and testing plans
    CreatorResearchAgent   — creator/brand fit assessment
    CreatorIntelSuite      — the dispatcher-facing facade

Every scorer is deterministic and offline-first; when the ProviderRouter has
a model available the suite adds a strategic enrichment paragraph on top of
— never instead of — the measured result.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import threading
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .security import SecuritySanitiser
from . import user_profile
from .utils import first_line, utc_stamp
from .atomic_io import atomic_write_text

CREATORS_PATH = CONFIG_DIR / "creators.json"

#: Publishable default; config/profile.json carries the real agency name.
BRAND = user_profile.agency()

# ── hook heuristics ──────────────────────────────────────────────────────────

_CURIOSITY = ("secret", "nobody", "no one", "stop", "wait", "before you",
              "the truth", "why you", "what happens", "you won't", "hidden",
              "mistake", "wrong", "never", "warning")
_PATTERN_INTERRUPT = ("pov", "watch this", "look at", "did you know",
                      "i tested", "i tried", "we found")
_WEAK_OPENERS = ("hey guys", "hi everyone", "welcome back", "in this video",
                 "today i", "so basically", "hello everyone")
_CTA_MARKERS = ("follow", "comment", "share", "link in bio", "shop", "grab",
                "get yours", "save this", "try it", "dm ", "order")
_EMOTION_TRIGGERS = {
    "curiosity": _CURIOSITY,
    "urgency": ("now", "today", "before", "last chance", "running out"),
    "relatability": ("you", "your", "we all", "everyone", "me when"),
    "proof": ("results", "tested", "proof", "before and after", "review",
              "%", "sold"),
    "novelty": ("new", "never seen", "first time", "finally", "just dropped"),
}


class HookAnalyzer:
    """Deterministic 0-10 scoring of an opening hook."""

    def score(self, hook: str) -> tuple[float, list[str]]:
        hook = str(hook or "").strip()
        if not hook:
            return 0.0, ["No hook supplied."]
        lowered = hook.lower()
        words = lowered.split()
        score, notes = 5.0, []
        if len(words) <= 12:
            score += 1.0
            notes.append(f"Tight length ({len(words)} words) — good.")
        else:
            score -= 1.5
            notes.append(f"Too long at {len(words)} words; hooks land in "
                         "under ~12.")
        if any(w in ("you", "your") for w in words):
            score += 1.0
            notes.append("Direct address ('you') — pulls the viewer in.")
        else:
            notes.append("No direct address; consider speaking to 'you'.")
        if any(term in lowered for term in _CURIOSITY):
            score += 1.5
            notes.append("Opens a curiosity gap.")
        if any(term in lowered for term in _PATTERN_INTERRUPT):
            score += 0.5
            notes.append("Pattern-interrupt phrasing present.")
        if "?" in hook:
            score += 0.5
            notes.append("Question format invites a mental answer.")
        if re.search(r"\d", hook):
            score += 0.5
            notes.append("Contains a number — specificity performs.")
        for weak in _WEAK_OPENERS:
            if lowered.startswith(weak):
                score -= 2.5
                notes.append(f"Weak opener ('{weak}') — the scroll won't stop "
                             "for it.")
                break
        return round(max(0.0, min(10.0, score)), 1), notes


class ScriptEvaluator:
    """Hook/body/CTA breakdown with pacing and retention analysis."""

    def __init__(self, hooks: HookAnalyzer | None = None) -> None:
        self.hooks = hooks or HookAnalyzer()

    def evaluate(self, script: str) -> dict[str, Any]:
        script = str(script or "").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", script)
                     if s.strip()]
        if not sentences:
            return {"ok": False, "note": "Empty script."}
        hook = sentences[0]
        hook_score, hook_notes = self.hooks.score(hook)
        lowered = script.lower()
        has_cta = any(marker in lowered for marker in _CTA_MARKERS)
        cta_line = next((s for s in reversed(sentences)
                         if any(m in s.lower() for m in _CTA_MARKERS)), "")
        word_counts = [len(s.split()) for s in sentences]
        avg_len = statistics.mean(word_counts)
        pacing = ("fast" if avg_len <= 9 else
                  "moderate" if avg_len <= 15 else "slow")
        loops = sum(1 for s in sentences
                    if re.search(r"\b(but|here's the thing|until|then i|"
                                 r"what happened next|the problem)\b",
                                 s.lower()))
        total_words = sum(word_counts)
        est_seconds = total_words / 2.6  # conversational speaking rate
        structure_score = 5.0
        structure_notes: list[str] = []
        if loops >= 2:
            structure_score += 2.0
            structure_notes.append(f"{loops} retention loops keep watch-time up.")
        elif loops == 1:
            structure_score += 1.0
            structure_notes.append("One retention loop; a second mid-script "
                                   "would help.")
        else:
            structure_notes.append("No retention loops — nothing re-hooks the "
                                   "viewer mid-video.")
        if pacing == "slow":
            structure_score -= 1.5
            structure_notes.append(f"Pacing is slow (avg {avg_len:.0f} "
                                   "words/sentence); cut filler.")
        else:
            structure_notes.append(f"Pacing is {pacing}.")
        if est_seconds > 60:
            structure_score -= 1.0
            structure_notes.append(f"~{est_seconds:.0f}s spoken — trim towards "
                                   "30-45s for short-form.")
        cta_score = 7.0 if has_cta else 2.0
        cta_notes = ([f"CTA present: '{cta_line[:60]}'"] if has_cta else
                     ["No call-to-action — the view converts to nothing."])
        overall = round(hook_score * 0.45 + min(10.0, structure_score) * 0.35
                        + cta_score * 0.20, 1)
        return {
            "ok": True, "overall": overall,
            "hook": {"text": hook[:120], "score": hook_score, "notes": hook_notes},
            "structure": {"score": round(min(10.0, structure_score), 1),
                          "notes": structure_notes, "pacing": pacing,
                          "est_seconds": round(est_seconds)},
            "cta": {"score": cta_score, "notes": cta_notes},
            "priority_fix": self._priority_fix(hook_score, structure_score,
                                               cta_score),
        }

    @staticmethod
    def _priority_fix(hook: float, structure: float, cta: float) -> str:
        weakest = min(("hook", hook), ("structure", structure), ("cta", cta),
                      key=lambda p: p[1])
        return {
            "hook": "Rewrite the first line — it decides 80% of performance. "
                    "Lead with the outcome or the tension, not the greeting.",
            "structure": "Add a mid-script re-hook ('but here's the thing…') "
                         "and cut every sentence that doesn't earn its second.",
            "cta": "End with ONE specific action (follow, comment a word, tap "
                   "the link) tied to what they just watched.",
        }[weakest[0]]


class CreatorPerformanceTracker:
    """Score history per creator, computed from stored reviews."""

    def __init__(self, store: "CreatorManager") -> None:
        self.store = store

    def report(self, creator: str = "") -> list[str]:
        data = self.store.load()
        names = ([creator] if creator else sorted(data))
        lines: list[str] = []
        for name in names:
            record = data.get(name)
            if not record:
                continue
            reviews = record.get("reviews") or []
            if not reviews:
                lines.append(f"- {name}: no reviews yet.")
                continue
            scores = [float(r.get("overall") or 0.0) for r in reviews]
            trend = ""
            if len(scores) >= 2:
                delta = scores[-1] - scores[0]
                trend = (" — improving" if delta > 0.5 else
                         " — declining" if delta < -0.5 else " — steady")
            lines.append(f"- {name}: {len(reviews)} review(s), avg "
                         f"{statistics.mean(scores):.1f}/10, latest "
                         f"{scores[-1]:.1f}{trend}")
        return lines or ["No creators on the roster yet."]


class CreatorManager:
    """Creator roster + submission reviews, durable in config/creators.json."""

    def __init__(self, path: Path | None = None,
                 evaluator: ScriptEvaluator | None = None) -> None:
        self.path = Path(path) if path is not None else CREATORS_PATH
        self.evaluator = evaluator or ScriptEvaluator()
        self.tracker = CreatorPerformanceTracker(self)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(data, indent=2), encoding="utf-8")

    def add_creator(self, name: str, handle: str = "", niche: str = "") -> str:
        name = SecuritySanitiser.guard_text(str(name or ""), "creator.name")[:80]
        if not name:
            return ""
        with self._lock:
            data = self.load()
            record = data.setdefault(name, {"added_at": utc_stamp(),
                                            "reviews": []})
            if handle:
                record["handle"] = str(handle)[:80]
            if niche:
                record["niche"] = str(niche)[:120]
            self._save(data)
        return name

    def review_submission(self, creator: str, script: str) -> dict[str, Any]:
        review = self.evaluator.evaluate(script)
        if not review.get("ok"):
            return review
        creator = self.add_creator(creator) or "unassigned"
        with self._lock:
            data = self.load()
            record = data.setdefault(creator, {"added_at": utc_stamp(),
                                               "reviews": []})
            record["reviews"] = (record.get("reviews") or [])[-19:] + [{
                "at": utc_stamp(), "overall": review["overall"],
                "hook": review["hook"]["score"],
                "structure": review["structure"]["score"],
                "cta": review["cta"]["score"],
            }]
            self._save(data)
        review["creator"] = creator
        return review


class ViralAnalysisAgent:
    """Recurring patterns across example hooks/scripts (deterministic)."""

    def analyse(self, examples: list[str]) -> list[str]:
        examples = [str(e or "").strip() for e in examples if str(e or "").strip()]
        if not examples:
            return ["Provide example hooks or scripts to analyse."]
        lines: list[str] = [f"Patterns across {len(examples)} example(s):"]
        trigger_counts: dict[str, int] = {}
        for name, terms in _EMOTION_TRIGGERS.items():
            count = sum(1 for e in examples
                        if any(t in e.lower() for t in terms))
            if count:
                trigger_counts[name] = count
        for name, count in sorted(trigger_counts.items(), key=lambda p: -p[1]):
            lines.append(f"- {name.title()} trigger in {count}/{len(examples)} "
                         "examples.")
        questions = sum(1 for e in examples if "?" in e.split("\n")[0])
        if questions:
            lines.append(f"- {questions} open with a question.")
        numbers = sum(1 for e in examples if re.search(r"\d", e.split("\n")[0]))
        if numbers:
            lines.append(f"- {numbers} lead with a specific number.")
        first_words = [e.lower().split()[0] for e in examples if e.split()]
        common = {w: first_words.count(w) for w in set(first_words)
                  if first_words.count(w) > 1}
        if common:
            lines.append("- Recurring opening words: "
                         + ", ".join(f"'{w}' x{n}" for w, n in
                                     sorted(common.items(), key=lambda p: -p[1])[:3]))
        avg_words = statistics.mean(len(e.split()) for e in examples)
        lines.append(f"- Average length {avg_words:.0f} words.")
        return lines

    def hook_ideas(self, topic: str) -> list[str]:
        topic = str(topic or "your product").strip()[:60]
        return [
            f"I tested {topic} for 7 days — here's what nobody tells you.",
            f"Stop buying {topic} before you watch this.",
            f"The {topic} mistake 90% of people make.",
            f"POV: you finally found a {topic} that actually works.",
            f"3 signs your {topic} is costing you money.",
            f"Wait — this is why your {topic} isn't working.",
        ]


class ProductResearchAgent:
    """Product angles and evaluation for short-form commerce."""

    def analyse(self, product: str) -> list[str]:
        product = str(product or "").strip()[:120]
        if not product:
            return ["Name the product (or paste its URL/description)."]
        return [
            f"Product analysis — {product}:",
            "Positioning checklist:",
            "- What visible problem does it solve in the first 2 seconds on camera?",
            "- Is the result demonstrable in one shot (before/after, live demo)?",
            "- Price point vs impulse threshold (~£30 for cold TikTok traffic)?",
            "- What do the 1-star reviews of competitors complain about?",
            "Content angles to test:",
            "- Problem-agitate-demo: show the pain, then the fix in real time.",
            "- UGC testimonial: creator's honest first reaction, unpolished.",
            "- Before/after with a hard cut at the reveal.",
            "- 'I tested the viral X' — ride existing search intent.",
            "- Founder story: why this exists, filmed handheld.",
            "Next step: pick the two cheapest angles, brief one creator each, "
            "and compare 3-second retention after 48 hours.",
        ]


class ContentStrategyAgent:
    """Platform mix, cadence and testing plans."""

    def plan(self, focus: str = "") -> list[str]:
        focus = str(focus or "the brand").strip()[:80]
        return [
            f"Short-form strategy — {focus}:",
            "- TikTok: 1-2 posts/day; native, unpolished, hook-first. The "
            "testing ground.",
            "- Instagram Reels: repost top 30% of TikToks after 24-48h; "
            "polish the caption.",
            "- YouTube Shorts: evergreen winners only; titles carry search "
            "intent.",
            "- Facebook Reels: older demographic; lead with proof and price.",
            "Cadence rule: volume on TikTok discovers winners; the other "
            "platforms amplify them.",
            "Iteration rule: when a video outperforms 2x median, ship 3 "
            "variations of its hook within 72 hours.",
            "Measurement: 3-second retention > completion rate > saves > "
            "follows, in that order of diagnostic value.",
        ]


class CreatorResearchAgent:
    """Creator/brand fit assessment from supplied profile facts."""

    def assess(self, profile: str) -> list[str]:
        profile = str(profile or "").strip()
        if not profile:
            return ["Paste the creator's profile facts (followers, niche, "
                    "engagement, example content)."]
        lines = ["Creator fit assessment:"]
        followers = re.search(r"([\d,.]+)\s*[km]?\s*(followers|subs)",
                              profile.lower())
        engagement = re.search(r"([\d.]+)\s*%", profile)
        if engagement:
            rate = float(engagement.group(1))
            verdict = ("strong" if rate >= 5 else
                       "acceptable" if rate >= 2 else "weak")
            lines.append(f"- Engagement ~{rate:.1f}% — {verdict} (rate beats "
                         "reach for UGC).")
        elif followers:
            lines.append("- Follower count noted, but ask for engagement rate "
                         "— reach without engagement is vanity.")
        lines.extend([
            "- Check their last 10 videos: consistent hooks, native platform "
            "style, comments answered?",
            "- Fit test: brief one paid test video before any retainer.",
            "- Rights: confirm usage terms for ads (spark ads / whitelisting) "
            "in writing.",
        ])
        return lines


class CTAAnalyzer:
    """Deterministic 0-10 scoring of a call-to-action, with rewrites."""

    _STRONG_VERBS = ("grab", "shop", "comment", "save", "follow", "tap",
                     "order", "try", "dm", "get")
    _SPECIFICITY = ("link in bio", "code", "%", "today", "below", "first",
                    "free", "before")

    def score(self, cta: str) -> tuple[float, list[str]]:
        cta = str(cta or "").strip()
        if not cta:
            return 0.0, ["No CTA supplied."]
        lowered = cta.lower()
        words = lowered.split()
        score, notes = 5.0, []
        actions = [v for v in self._STRONG_VERBS if v in lowered]
        if len(actions) == 1:
            score += 2.0
            notes.append(f"One clear action ('{actions[0]}') — decisive.")
        elif len(actions) > 1:
            score -= 1.0
            notes.append(f"{len(actions)} competing actions — pick ONE; "
                         "split attention converts to nothing.")
        else:
            score -= 2.0
            notes.append("No action verb — the viewer is never told what to do.")
        if any(term in lowered for term in self._SPECIFICITY):
            score += 1.5
            notes.append("Specific destination/incentive present.")
        else:
            notes.append("Add a specific destination or incentive "
                         "('link in bio', a code, 'comment WORD').")
        if len(words) <= 10:
            score += 1.0
            notes.append("Tight — CTAs land in under ~10 words.")
        else:
            score -= 1.0
            notes.append(f"{len(words)} words is long for a CTA; trim it.")
        if any(term in lowered for term in _EMOTION_TRIGGERS["urgency"]):
            score += 0.5
            notes.append("Urgency cue raises action rate.")
        return round(max(0.0, min(10.0, score)), 1), notes

    def improve(self, cta: str, product: str = "") -> list[str]:
        product = str(product or "it").strip()[:50]
        return [
            f"Comment 'MORE' and I'll send you the link to {product}.",
            f"Tap the link in bio before today's stock of {product} goes.",
            f"Save this for when you're ready to try {product}.",
            f"Follow for part 2 — I'm testing {product} for 30 days.",
        ]


class CompetitorAnalysisAgent:
    """Deterministic teardown of competitor content or a competitor page."""

    def __init__(self, hooks: HookAnalyzer, ctas: CTAAnalyzer) -> None:
        self.hooks = hooks
        self.ctas = ctas

    def analyse(self, content: str) -> list[str]:
        content = str(content or "").strip()
        if not content:
            return ["Paste the competitor's script, caption or page copy."]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", content)
                     if s.strip()]
        lines = ["Competitor teardown:"]
        hook_score, _ = self.hooks.score(sentences[0] if sentences else "")
        lines.append(f"- Their hook scores {hook_score}/10 — "
                     + ("beat it with a sharper curiosity gap."
                        if hook_score >= 7 else
                        "beatable; lead with a stronger first line."))
        cta_line = next((s for s in reversed(sentences)
                         if any(m in s.lower() for m in _CTA_MARKERS)), "")
        if cta_line:
            cta_score, _ = self.ctas.score(cta_line)
            lines.append(f"- Their CTA ('{cta_line[:50]}') scores "
                         f"{cta_score}/10.")
        else:
            lines.append("- No visible CTA — they're leaking conversions; "
                         "yours should always close.")
        triggers = [name for name, terms in _EMOTION_TRIGGERS.items()
                    if any(t in content.lower() for t in terms)]
        if triggers:
            lines.append("- Emotional triggers they lean on: "
                         + ", ".join(triggers) + ".")
        gaps = [name for name in _EMOTION_TRIGGERS if name not in triggers]
        if gaps:
            lines.append(f"- Unused angles you can own: {', '.join(gaps[:3])}.")
        return lines


class TrendTracker:
    """Findings over time — durable JSON so patterns emerge across sessions."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (Path(path) if path is not None
                     else CONFIG_DIR / "creator_trends.json")
        self._lock = threading.Lock()

    def load(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    def record(self, kind: str, subject: str, findings: dict[str, Any]) -> None:
        with self._lock:
            entries = self.load()
            entries.append({"at": utc_stamp(), "kind": str(kind)[:40],
                            "subject": str(subject)[:120],
                            "findings": findings})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, json.dumps(entries[-400:], indent=2),
                                 encoding="utf-8")

    def report(self, kind: str = "") -> list[str]:
        entries = self.load()
        if kind:
            entries = [e for e in entries if e.get("kind") == kind]
        if not entries:
            return ["No trend data recorded yet — run analyses and I'll "
                    "track patterns over time."]
        lines = [f"Trend history — {len(entries)} recorded analysis(es):"]
        trigger_totals: dict[str, int] = {}
        hook_scores: list[float] = []
        for e in entries:
            f = e.get("findings") or {}
            for name in f.get("triggers") or []:
                trigger_totals[name] = trigger_totals.get(name, 0) + 1
            if isinstance(f.get("hook_score"), (int, float)):
                hook_scores.append(float(f["hook_score"]))
        if trigger_totals:
            top = sorted(trigger_totals.items(), key=lambda p: -p[1])[:3]
            lines.append("Recurring emotional triggers: "
                         + ", ".join(f"{n} (x{c})" for n, c in top) + ".")
        if len(hook_scores) >= 2:
            recent = statistics.mean(hook_scores[-5:])
            overall = statistics.mean(hook_scores)
            drift = ("improving" if recent > overall + 0.3 else
                     "declining" if recent < overall - 0.3 else "steady")
            lines.append(f"Hook quality: avg {overall:.1f}/10, recent "
                         f"{recent:.1f}/10 — {drift}.")
        recent_subjects = [str(e.get("subject", ""))[:40]
                           for e in entries[-3:]]
        lines.append("Latest subjects: " + "; ".join(recent_subjects))
        return lines


class ProductIntelligencePipeline:
    """One command, full teardown: product → analysis → creative package.

    Feed it whatever exists — a product name/URL/description, competitor
    content, a creator's script — and it runs every analyser, then generates
    the improvement package: better hooks, better CTAs, a creator brief,
    script variations, the testing plan and scaling gates.  Findings are
    stored in the trend tracker (and long-term memory when available) so
    intelligence compounds over time.
    """

    def __init__(self, suite: "CreatorIntelSuite") -> None:
        self.suite = suite

    def run(self, product: str, competitor: str = "",
            script: str = "") -> dict[str, Any]:
        product = str(product or "").strip()
        result: dict[str, Any] = {"product": product[:120]}
        sections: list[str] = [f"PRODUCT INTELLIGENCE — {product[:80]}"]
        # 1 product angles
        sections.append("\n".join(self.suite.products.analyse(product)))
        # 2 competitor teardown
        triggers: list[str] = []
        if competitor.strip():
            sections.append("\n".join(
                self.suite.competitors.analyse(competitor)))
            triggers = [name for name, terms in _EMOTION_TRIGGERS.items()
                        if any(t in competitor.lower() for t in terms)]
        # 3 supplied script/creator video review
        hook_score = None
        if script.strip():
            review = self.suite.evaluator.evaluate(script)
            if review.get("ok"):
                hook_score = review["hook"]["score"]
                sections.append(
                    f"SUPPLIED SCRIPT — overall {review['overall']}/10, "
                    f"hook {hook_score}/10, priority fix: "
                    f"{review['priority_fix']}")
        # 4 the improvement package
        hooks = self.suite.viral.hook_ideas(product)
        sections.append("IMPROVED HOOKS:\n" + "\n".join(f"- {h}" for h in hooks))
        ctas = self.suite.ctas.improve("", product)
        sections.append("IMPROVED CTAs:\n" + "\n".join(f"- {c}" for c in ctas))
        sections.append(self._creator_brief(product, hooks[0], ctas[0]))
        sections.append(self._testing_plan(product))
        result["report"] = "\n\n".join(sections)
        result["hook_score"] = hook_score
        result["triggers"] = triggers
        # 5 compound the intelligence
        self.suite.trends.record("product_intel", product, {
            "hook_score": hook_score, "triggers": triggers,
            "competitor": bool(competitor.strip())})
        return result

    @staticmethod
    def _creator_brief(product: str, hook: str, cta: str) -> str:
        product = product[:60]
        return (
            "CREATOR BRIEF:\n"
            f"- Deliverable: one 30-45s vertical video featuring {product}.\n"
            f"- Open EXACTLY with: \"{hook}\"\n"
            "- Structure: problem (0-3s) → demo/result (3-25s) → one "
            "re-hook mid-way ('but here's the thing…') → close.\n"
            f"- Close with: \"{cta}\"\n"
            "- Style: native, handheld, captions on, no logo intro.\n"
            "- Deliver raw + edited; usage rights for paid amplification.")

    @staticmethod
    def _testing_plan(product: str) -> str:
        return (
            "TESTING & SCALING:\n"
            "- Week 1: 2 hook variants x 2 creators; judge on 3-second "
            "retention only.\n"
            "- Gate 1: any video ≥2x median retention → ship 3 hook "
            "variations within 72h.\n"
            "- Gate 2: organic winner sustained 48h → whitelist as spark "
            "ad at £10/day.\n"
            "- Gate 3: paid ROAS ≥1.5 over 7 days → scale spend 2x, brief "
            "two more creators on the winning structure.\n"
            "- Kill rule: below-median retention twice → retire the angle, "
            "not the product.")


class CreatorIntelSuite:
    """Facade behind the 'creator_intel' tool."""

    def __init__(self, bus: OrionBus, router: Any | None = None,
                 memory: Any | None = None, telemetry: Any | None = None,
                 creators_path: Path | None = None) -> None:
        self.bus = bus
        self.router = router
        self.memory = memory
        self.telemetry = telemetry
        self.hooks = HookAnalyzer()
        self.evaluator = ScriptEvaluator(self.hooks)
        self.creators = CreatorManager(creators_path, self.evaluator)
        self.viral = ViralAnalysisAgent()
        self.products = ProductResearchAgent()
        self.strategy = ContentStrategyAgent()
        self.creator_research = CreatorResearchAgent()
        self.ctas = CTAAnalyzer()
        self.competitors = CompetitorAnalysisAgent(self.hooks, self.ctas)
        self.trends = TrendTracker(
            Path(creators_path).parent / "creator_trends.json"
            if creators_path is not None else None)
        self.pipeline = ProductIntelligencePipeline(self)
        if telemetry is not None:
            telemetry.health.register("creator_intel")

    async def handle(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").lower().strip()
        text = str(args.get("script") or args.get("text") or "").strip()
        if action in {"hook", "score_hook"}:
            score, notes = self.hooks.score(str(args.get("hook") or text))
            return ToolResult(f"Hook score: {score}/10\n"
                              + "\n".join(f"- {n}" for n in notes))
        if action in {"review_script", "script", "review"}:
            if not text:
                return ToolResult("Paste the script to review.", ok=False)
            review = await asyncio.to_thread(
                self.creators.review_submission,
                str(args.get("creator") or ""), text)
            if not review.get("ok"):
                return ToolResult(str(review.get("note") or "Review failed."),
                                  ok=False)
            result = self._format_review(review)
            enriched = await self._enrich(
                f"Script:\n{text[:2500]}\n\nMeasured review:\n{result[:800]}",
                "Add ONE paragraph of coaching a creator manager would give. "
                "Specific, warm, no fluff.")
            if enriched:
                result += f"\n\nCoaching note:\n{enriched}"
            if self.telemetry is not None:
                self.telemetry.metrics.incr("creator.reviews")
            return ToolResult(result)
        if action in {"performance", "creators", "roster"}:
            lines = await asyncio.to_thread(
                self.creators.tracker.report, str(args.get("creator") or ""))
            return ToolResult("Creator performance:\n" + "\n".join(lines))
        if action in {"add_creator", "signup"}:
            name = self.creators.add_creator(
                str(args.get("creator") or args.get("name") or ""),
                str(args.get("handle") or ""), str(args.get("niche") or ""))
            if not name:
                return ToolResult("A creator name is required.", ok=False)
            return ToolResult(f"Creator '{name}' added to the "
                              f"{BRAND} roster.")
        if action in {"viral", "patterns", "analyse_viral"}:
            examples = args.get("examples")
            if isinstance(examples, str):
                examples = [e for e in re.split(r"\n{2,}|\|\|", examples) if e.strip()]
            if not isinstance(examples, list):
                examples = [text] if text else []
            return ToolResult("\n".join(self.viral.analyse(examples)))
        if action in {"hooks", "hook_ideas", "ideas"}:
            topic = str(args.get("topic") or text or "")
            ideas = self.viral.hook_ideas(topic)
            return ToolResult("Hook ideas:\n" + "\n".join(f"- {h}" for h in ideas))
        if action in {"product", "product_research"}:
            product = str(args.get("product") or args.get("url") or text)
            lines = self.products.analyse(product)
            result = "\n".join(lines)
            enriched = await self._enrich(
                f"Product: {product[:300]}", "In under 120 words: the single "
                "biggest risk with this product for TikTok Shop-style "
                "commerce, and the strongest angle. British English.")
            if enriched:
                result += f"\n\nStrategic view:\n{enriched}"
            return ToolResult(result)
        if action in {"strategy", "plan", "content_plan"}:
            return ToolResult("\n".join(
                self.strategy.plan(str(args.get("focus") or text))))
        if action in {"creator_fit", "assess_creator"}:
            return ToolResult("\n".join(self.creator_research.assess(
                str(args.get("profile") or text))))
        if action in {"cta", "score_cta"}:
            score, notes = self.ctas.score(str(args.get("cta") or text))
            improved = self.ctas.improve(text, str(args.get("product") or ""))
            return ToolResult(
                f"CTA score: {score}/10\n" + "\n".join(f"- {n}" for n in notes)
                + "\nStronger variants:\n"
                + "\n".join(f"- {c}" for c in improved))
        if action in {"competitor", "teardown"}:
            return ToolResult("\n".join(self.competitors.analyse(
                str(args.get("competitor") or text))))
        if action in {"pipeline", "product_intel", "full_analysis"}:
            product = str(args.get("product") or args.get("url") or text)
            if not product.strip():
                return ToolResult("Give me the product (name, URL or "
                                  "description).", ok=False)
            result = await asyncio.to_thread(
                self.pipeline.run, product,
                str(args.get("competitor") or ""),
                str(args.get("script") or ""))
            report = result["report"]
            enriched = await self._enrich(
                f"Product intelligence report:\n{report[:2500]}",
                "In under 150 words: the sharpest strategic read on this "
                "product for short-form commerce — the one risk and the one "
                "move. British English.")
            if enriched:
                report += f"\n\nSTRATEGIC VIEW:\n{enriched}"
            self._remember(f"Product intelligence — {result['product']}",
                           report)
            if self.telemetry is not None:
                self.telemetry.metrics.incr("creator.pipeline")
            return ToolResult(report)
        if action in {"trends", "trend_report"}:
            lines = await asyncio.to_thread(
                self.trends.report, str(args.get("kind") or ""))
            return ToolResult("\n".join(lines))
        return ToolResult(
            "Unsupported creator_intel action. Use hook, review_script, "
            "performance, add_creator, viral, hook_ideas, product, strategy, "
            "creator_fit, cta, competitor, pipeline, or trends.", ok=False)

    @staticmethod
    def _format_review(review: dict[str, Any]) -> str:
        hook = review["hook"]
        structure = review["structure"]
        cta = review["cta"]
        lines = [
            f"Script review — overall {review['overall']}/10"
            + (f" (creator: {review['creator']})"
               if review.get("creator", "unassigned") != "unassigned" else ""),
            f"HOOK {hook['score']}/10: \"{hook['text']}\"",
        ]
        lines.extend(f"  - {n}" for n in hook["notes"][:3])
        lines.append(f"STRUCTURE {structure['score']}/10 "
                     f"(~{structure['est_seconds']}s, {structure['pacing']} pacing)")
        lines.extend(f"  - {n}" for n in structure["notes"][:3])
        lines.append(f"CTA {cta['score']}/10")
        lines.extend(f"  - {n}" for n in cta["notes"][:2])
        lines.append(f"PRIORITY FIX: {review['priority_fix']}")
        return "\n".join(lines)

    def _remember(self, key: str, report: str) -> None:
        """Persist a pipeline finding to long-term memory (best effort)."""
        if self.memory is None:
            return
        try:
            self.memory.save("KNOWLEDGE", key[:120], report[:2000])
        except Exception:
            pass

    async def _enrich(self, context: str, instruction: str) -> str:
        if self.router is None or not getattr(
                self.router, "has_text_fallback", lambda: False)():
            return ""
        try:
            _profile, text = await self.router.generate_text(
                f"{context}\n\n{instruction}",
                system_extra=f"You are the strategy lead at {BRAND}, a "
                             "short-form content and UGC agency.")
            return text.strip()
        except Exception as exc:
            self.bus.log.emit(f"CREATOR: enrichment skipped - {first_line(exc)}")
            return ""


__all__ = [
    "BRAND", "CTAAnalyzer", "CompetitorAnalysisAgent", "ContentStrategyAgent",
    "CreatorIntelSuite", "CreatorManager", "CreatorPerformanceTracker",
    "CreatorResearchAgent", "HookAnalyzer", "ProductIntelligencePipeline",
    "ProductResearchAgent", "ScriptEvaluator", "TrendTracker",
    "ViralAnalysisAgent",
]
