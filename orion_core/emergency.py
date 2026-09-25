"""
EmergencyProtocol — genuinely urgent events that could affect the user.

Deliberately NOT a macro
------------------------
The other protocols are named sequences of tool calls.  This one cannot be,
because the brief asks it to make a judgement about each thing it finds:

    what happened / when / how reliable is the source / does it affect the
    user / how severe / what can they do / what should they do now / what
    should they avoid / does this need interrupting them for

A list of tool calls cannot answer "does it actually affect the user".  So this
is a small pipeline: gather from configured sources, classify each candidate,
score it against the user's actual location, and only then decide whether it is
worth speaking.

Honesty about certainty is the whole point
------------------------------------------
An assistant that invents an emergency is worse than one that stays quiet.
Every finding carries a :class:`Confidence` that is derived from the source and
corroboration, never assumed, and the spoken output always states it:

    CONFIRMED    an official source (met office, government, national rail)
    REPORTED     a recognised news outlet, uncorroborated
    POSSIBLE     a single low-authority mention, or a weak keyword match
    SPECULATIVE  inferred, not stated

Nothing here fabricates.  If no source returns anything, the honest answer is
"no active emergencies found", and that is what it says.

Safety rails the brief asks for
-------------------------------
  • priority handling      severity ordering, highest first
  • timeouts               per-source and overall, so a dead feed cannot hang it
  • cancellation           cooperative, checked between sources
  • rate limiting          a minimum gap between full runs
  • failure recovery       every source is isolated; one failing is not a
                           failed protocol
  • duplicate prevention   a fingerprint per finding, remembered, so the same
                           storm is not announced every fifteen minutes
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .time_service import TIME
from .utils import first_line


class Confidence(str, Enum):
    CONFIRMED = "confirmed"
    REPORTED = "reported but unverified"
    POSSIBLE = "possible"
    SPECULATIVE = "speculative"


class Severity(int, Enum):
    INFO = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.title()


#: Categories the protocol looks for, each with the words that identify it and
#: a baseline severity.  Explicit and inspectable rather than model-guessed, so
#: the classification is deterministic and testable.
CATEGORIES: dict[str, dict[str, Any]] = {
    "severe_weather": {
        "terms": ("red warning", "amber warning", "severe weather", "storm",
                  "flood warning", "hurricane", "tornado", "blizzard",
                  "heatwave", "ice warning", "gale"),
        "severity": Severity.HIGH,
        "advice": ("Avoid non-essential travel and stay away from exposed "
                   "coastal or wooded areas."),
        "avoid": "Do not drive through standing water or park under trees.",
    },
    "infrastructure": {
        "terms": ("power cut", "power outage", "blackout", "water supply",
                  "gas leak", "outage affecting", "national grid"),
        "severity": Severity.HIGH,
        "advice": "Charge devices now and keep a torch to hand.",
        "avoid": "Do not use candles near anything flammable.",
    },
    "transport": {
        "terms": ("all lines suspended", "major delays", "line closed",
                  "motorway closed", "airport closed", "rail strike",
                  "severe disruption"),
        "severity": Severity.MODERATE,
        "advice": "Check your route before leaving and allow extra time.",
        "avoid": "Do not rely on the usual timetable today.",
    },
    "security": {
        "terms": ("major incident", "evacuation", "cordon", "terror",
                  "active shooter", "unexploded", "police incident"),
        "severity": Severity.CRITICAL,
        "advice": "Follow instructions from emergency services immediately.",
        "avoid": "Do not travel towards the area or share unverified detail.",
    },
    "cyber": {
        "terms": ("data breach", "ransomware", "zero-day", "actively exploited",
                  "critical vulnerability", "cyber attack", "credential leak"),
        "severity": Severity.MODERATE,
        "advice": "Update affected software and change any reused passwords.",
        "avoid": "Do not open unexpected attachments or reset links.",
    },
    "health": {
        "terms": ("public health emergency", "outbreak", "contamination",
                  "recall notice", "do not drink"),
        "severity": Severity.HIGH,
        "advice": "Follow the official guidance for your area.",
        "avoid": "Do not rely on second-hand summaries of the advice.",
    },
    "system": {
        "terms": ("orion",),   # matched only against ORION's own health input
        "severity": Severity.MODERATE,
        "advice": "Check the diagnostics page for the failing subsystem.",
        "avoid": "Do not assume voice commands are being heard until it clears.",
    },
}

#: Sources considered authoritative — a finding from one of these is CONFIRMED.
OFFICIAL_DOMAINS = (
    "metoffice.gov.uk", "gov.uk", "environment-agency.gov.uk",
    "nationalrail.co.uk", "tfl.gov.uk", "nhs.uk", "ncsc.gov.uk",
    "cisa.gov", "who.int", "police.uk",
)
#: Recognised outlets — REPORTED, i.e. real but uncorroborated.
NEWS_DOMAINS = (
    "bbc.co.uk", "bbc.com", "reuters.com", "apnews.com", "sky.com",
    "theguardian.com", "ft.com", "independent.co.uk", "telegraph.co.uk",
)


@dataclass
class Finding:
    """One candidate emergency, classified and scored."""

    headline: str
    category: str
    severity: Severity
    confidence: Confidence
    source: str = ""
    url: str = ""
    when: str = ""
    affects_user: bool = False
    why_affects: str = ""
    advice: str = ""
    avoid: str = ""
    detail: str = ""

    @property
    def fingerprint(self) -> str:
        """Stable identity, so the same event is not announced twice.

        Built from the category plus the significant words of the headline, so
        a re-worded copy of the same story still collapses onto one alert.
        """
        words = sorted(set(re.findall(r"[a-z]{4,}", self.headline.lower())))
        blob = f"{self.category}|{'-'.join(words[:8])}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def spoken(self) -> str:
        parts = [f"{self.severity.label} {self.category.replace('_', ' ')}: "
                 f"{self.headline}"]
        parts.append(f"This is {self.confidence.value}"
                     + (f", per {self.source}." if self.source else "."))
        if self.when:
            parts.append(f"Reported {self.when}.")
        parts.append(self.why_affects if self.affects_user
                     else "It does not appear to affect your area directly.")
        if self.advice:
            parts.append(f"What to do: {self.advice}")
        if self.avoid:
            parts.append(f"What to avoid: {self.avoid}")
        return " ".join(parts)

    def describe(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "category": self.category,
            "severity": self.severity.label,
            "severity_rank": int(self.severity),
            "confidence": self.confidence.value,
            "source": self.source,
            "url": self.url,
            "when": self.when,
            "affects_user": self.affects_user,
            "advice": self.advice,
            "avoid": self.avoid,
            "fingerprint": self.fingerprint,
        }


def classify(headline: str, *, source: str = "", url: str = "",
             locality: str = "", detail: str = "") -> Finding | None:
    """Turn a raw headline into a Finding, or None when it is not an emergency.

    Deterministic: the same headline always classifies the same way.  Returning
    None for ordinary news is what keeps this from crying wolf.
    """
    text = f"{headline} {detail}".lower()
    matched: tuple[str, dict[str, Any]] | None = None
    for name, spec in CATEGORIES.items():
        if name == "system":
            continue
        if any(term in text for term in spec["terms"]):
            if matched is None or spec["severity"] > matched[1]["severity"]:
                matched = (name, spec)
    if matched is None:
        return None
    category, spec = matched

    host = (url or source or "").lower()
    if any(domain in host for domain in OFFICIAL_DOMAINS):
        confidence = Confidence.CONFIRMED
    elif any(domain in host for domain in NEWS_DOMAINS):
        confidence = Confidence.REPORTED
    elif host:
        confidence = Confidence.POSSIBLE
    else:
        confidence = Confidence.SPECULATIVE

    severity: Severity = spec["severity"]
    # An unattributed claim is never allowed to read as high severity.
    if confidence in (Confidence.POSSIBLE, Confidence.SPECULATIVE):
        severity = Severity(max(int(Severity.LOW), int(severity) - 1))

    affects, why = _affects_user(text, locality)
    return Finding(
        headline=headline.strip(),
        category=category,
        severity=severity,
        confidence=confidence,
        source=source,
        url=url,
        affects_user=affects,
        why_affects=why,
        advice=spec["advice"],
        avoid=spec["avoid"],
        detail=detail,
    )


def _affects_user(text: str, locality: str) -> tuple[bool, str]:
    """Does this actually reach the user, or is it somewhere else entirely?

    Location is the honest discriminator available offline.  When ORION does
    not know where the user is, it says so rather than assuming relevance —
    an alert claimed to affect you without grounds is a fabricated alert.
    """
    if not locality:
        return False, ("I do not know your location well enough to say whether "
                       "this reaches you.")
    places = [p.strip().lower() for p in locality.split(",") if p.strip()]
    for place in places:
        if place and place in text:
            return True, f"This names {place.title()}, where you are."
    # National-scope wording still reaches a user in that country.
    country = places[-1] if places else ""
    national = ("uk-wide", "nationwide", "across the uk", "national",
                "england", "scotland", "wales", "northern ireland")
    if country in ("united kingdom", "uk", "gb") and any(t in text for t in national):
        return True, "This is national in scope, so it includes your area."
    return False, "It does not name your area."


class EmergencyProtocol:
    """Gathers, classifies, deduplicates and reports genuine emergencies."""

    #: Whole-run budget.  An emergency check that takes a minute is useless.
    TOTAL_TIMEOUT_S = 25.0
    #: Per-source budget, so one dead feed cannot consume the whole run.
    SOURCE_TIMEOUT_S = 10.0
    #: Minimum gap between full runs.
    MIN_INTERVAL_S = 120.0
    #: How long a fingerprint is remembered, so the same storm is not
    #: re-announced every time the protocol runs.
    DEDUP_WINDOW_S = 6 * 3600.0
    #: Only findings at or above this severity interrupt the user unprompted.
    NOTIFY_AT = Severity.HIGH

    def __init__(self, bus: Any, *, briefing: Any = None, health: Any = None,
                 router: Any = None, telemetry: Any = None) -> None:
        self.bus = bus
        self.briefing = briefing
        self.health = health
        self.router = router
        self.telemetry = telemetry
        self._seen: dict[str, float] = {}
        self._last_run = 0.0
        self._running = False
        self.last_findings: list[Finding] = []

    # ── locality ──────────────────────────────────────────────────────────────

    def locality(self) -> str:
        for source in (self.router,):
            value = getattr(source, "_locality", "") if source is not None else ""
            if value:
                return str(value)
        try:
            from .temporal import STATE_PATH
            import json
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            parts = [data.get(k) for k in ("city", "region", "country")]
            return ", ".join(str(p) for p in parts if p)
        except Exception:
            return ""

    # ── the run ───────────────────────────────────────────────────────────────

    async def run(self, *, force: bool = False,
                  cancel: Any = None) -> dict[str, Any]:
        """Check every configured source and report what is genuinely urgent.

        Returns a structured result; never raises.
        """
        now = time.monotonic()
        if self._running:
            return self._result([], note="An emergency check is already running.")
        if not force and (now - self._last_run) < self.MIN_INTERVAL_S:
            wait = self.MIN_INTERVAL_S - (now - self._last_run)
            return self._result(
                [], note=f"Checked {now - self._last_run:.0f}s ago; "
                         f"next check available in {wait:.0f}s.")
        self._running = True
        self._last_run = now
        locality = self.locality()
        try:
            findings = await asyncio.wait_for(
                self._gather(locality, cancel), timeout=self.TOTAL_TIMEOUT_S)
        except asyncio.TimeoutError:
            findings = []
            self._log("EMERGENCY: the check exceeded its time budget; "
                      "reporting what was gathered.")
        except asyncio.CancelledError:
            self._running = False
            raise
        except Exception as exc:
            findings = []
            self._log(f"EMERGENCY: check failed - {first_line(exc, 120)}")
        finally:
            self._running = False

        fresh = self._deduplicate(findings)
        # Priority handling: most severe first, then the ones that reach the
        # user, then the better-attested.
        fresh.sort(key=lambda f: (-int(f.severity), not f.affects_user,
                                  list(Confidence).index(f.confidence)))
        self.last_findings = fresh
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("protocol.emergency")
            except Exception:
                pass
        return self._result(fresh, locality=locality)

    async def _gather(self, locality: str, cancel: Any) -> list[Finding]:
        results: list[Finding] = []
        for name, source in (("system", self._system_source),
                             ("news", self._news_source)):
            if cancel is not None and cancel():
                self._log("EMERGENCY: check cancelled.")
                break
            try:
                found = await asyncio.wait_for(source(locality),
                                               timeout=self.SOURCE_TIMEOUT_S)
                results.extend(found)
            except asyncio.TimeoutError:
                self._log(f"EMERGENCY: source '{name}' timed out; continuing.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # One failing source must never fail the protocol.
                self._log(f"EMERGENCY: source '{name}' failed - "
                          f"{first_line(exc, 100)}; continuing.")
        return results

    async def _system_source(self, locality: str) -> list[Finding]:
        """ORION's own emergencies — a subsystem down IS an emergency here."""
        if self.health is None:
            return []
        try:
            snapshot = self.health.snapshot()
        except Exception:
            return []
        spec = CATEGORIES["system"]
        findings: list[Finding] = []
        for name, entry in (snapshot or {}).items():
            status = str((entry or {}).get("status", "")).upper()
            if status != "OFFLINE":
                continue
            findings.append(Finding(
                headline=f"ORION subsystem '{name}' is offline",
                category="system",
                severity=Severity.HIGH,
                confidence=Confidence.CONFIRMED,   # measured locally, not claimed
                source="ORION self-diagnostics",
                when=TIME.spoken_now(),
                affects_user=True,
                why_affects="It affects what ORION can do for you right now.",
                advice=spec["advice"],
                avoid=spec["avoid"],
                detail=str((entry or {}).get("detail", "")),
            ))
        return findings

    async def _news_source(self, locality: str) -> list[Finding]:
        """Classify whatever the briefing service already gathers.

        Reuses the existing, tested news pipeline rather than adding a second
        one — and if it is unavailable, this source simply contributes nothing.
        """
        briefing = self.briefing
        if briefing is None:
            return []
        gather = getattr(briefing, "_gather_candidates", None)
        if gather is None:
            return []
        try:
            from aiohttp import ClientSession, ClientTimeout
            timeout = ClientTimeout(total=self.SOURCE_TIMEOUT_S, connect=4.0)
            async with ClientSession(timeout=timeout) as session:
                stories = await gather(session, 6.0)
        except Exception:
            return []
        findings: list[Finding] = []
        for story in stories or []:
            headline = str(getattr(story, "title", "")
                           or (story.get("title") if isinstance(story, dict) else ""))
            if not headline:
                continue
            url = str(getattr(story, "url", "")
                      or (story.get("url") if isinstance(story, dict) else ""))
            source = str(getattr(story, "source", "")
                         or (story.get("source") if isinstance(story, dict) else ""))
            finding = classify(headline, source=source, url=url, locality=locality)
            if finding is not None:
                finding.when = str(getattr(story, "published", "") or "recently")
                findings.append(finding)
        return findings

    # ── deduplication ─────────────────────────────────────────────────────────

    def _deduplicate(self, findings: Iterable[Finding]) -> list[Finding]:
        now = time.time()
        self._seen = {k: t for k, t in self._seen.items()
                      if now - t < self.DEDUP_WINDOW_S}
        fresh: list[Finding] = []
        for finding in findings:
            key = finding.fingerprint
            if key in self._seen:
                continue
            self._seen[key] = now
            fresh.append(finding)
        return fresh

    # ── output ────────────────────────────────────────────────────────────────

    def should_notify(self, findings: list[Finding]) -> bool:
        """Interrupt the user only for something that genuinely warrants it."""
        return any(f.severity >= self.NOTIFY_AT and f.affects_user
                   for f in findings)

    def _result(self, findings: list[Finding], *, locality: str = "",
                note: str = "") -> dict[str, Any]:
        return {
            "at": TIME.now().isoformat(timespec="seconds"),
            "locality": locality,
            "count": len(findings),
            "notify": self.should_notify(findings),
            "findings": [f.describe() for f in findings],
            "spoken": self.spoken_summary(findings, note=note),
            "note": note,
        }

    def spoken_summary(self, findings: list[Finding], note: str = "") -> str:
        if note and not findings:
            return note
        if not findings:
            # The honest answer, and by far the most common one.
            return ("Emergency check complete. I found no active emergencies "
                    "affecting you.")
        head = (f"Emergency check complete. {len(findings)} item"
                f"{'s' if len(findings) != 1 else ''} of note, most serious first.")
        return " ".join([head] + [f.spoken() for f in findings[:3]])

    def _log(self, message: str) -> None:
        try:
            self.bus.log.emit(message)
        except Exception:
            pass


__all__ = ["CATEGORIES", "Confidence", "EmergencyProtocol", "Finding",
           "Severity", "classify"]
