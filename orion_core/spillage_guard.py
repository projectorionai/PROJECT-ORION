"""
The spillage guard (CAP-08) — catch a secret before it leaves the building.

    "Safety concerns — damage to the PC, private data spillage, harm to humans —
     must be voiced to me directly, overriding standby."

Damage to the PC is already watched by ``SentinelAgent`` (CPU, memory, disk,
battery, unknown heavyweight processes). The missing half of that sentence was
*private data spillage*, and that is what this does: it reads a piece of text
about to leave ORION — a message, a web post, a log line, something spoken — and
flags anything in it that looks like a credential, a private key or a card
number before it goes anywhere.

Two design choices keep it honest rather than noisy:

**Recognition, not guessing.** Findings come from shapes that are genuinely
secret — an ``sk-`` OpenAI key, an ``AKIA`` AWS id, a PEM private-key block, a
``password =`` assignment, a Luhn-valid card number — plus, when the caller
supplies them, ORION's *own* known secrets echoed back verbatim. High-entropy
paranoia is deliberately narrow so a UUID or a git hash doesn't trip it.

**It can override the quiet.** ``guard()`` raises a critical finding on the
``safety_alert`` bus — the same channel the sentinel uses to pierce standby —
because a key about to be leaked is exactly the kind of thing the user said must
be spoken even when he's asked for silence. Everything is pure and injectable
(the bus and the known-secret set are passed in), so the detection is tested
without a real bus and without ever handling a real key.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable


class Severity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class SecretPattern:
    kind: str
    severity: Severity
    regex: re.Pattern
    # which capture group holds the actual secret to redact (0 == whole match)
    group: int = 0


# Ordered most-specific first so a token is labelled by its true kind.
_PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern("private key block", Severity.CRITICAL,
                  re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
                             r".*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
                             re.DOTALL)),
    SecretPattern("OpenAI API key", Severity.CRITICAL,
                  re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    SecretPattern("AWS access key id", Severity.CRITICAL,
                  re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    SecretPattern("Google API key", Severity.CRITICAL,
                  re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    SecretPattern("GitHub token", Severity.CRITICAL,
                  re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    SecretPattern("Slack token", Severity.HIGH,
                  re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b")),
    SecretPattern("Anthropic API key", Severity.CRITICAL,
                  re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    SecretPattern("bearer token", Severity.HIGH,
                  re.compile(r"(?i)authorization:\s*bearer\s+([A-Za-z0-9._\-]{12,})", ), group=1),
    SecretPattern("credential assignment", Severity.HIGH,
                  re.compile(r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|token)\b"
                             r"\s*[:=]\s*[\"']?([^\s\"',]{6,})", ), group=1),
)

# A token this long, this random and not pure-hex is probably a secret even
# without a recognisable prefix. Kept conservative to avoid false alarms.
_ENTROPY_MIN_LEN = 32
_ENTROPY_THRESHOLD = 4.2
_HEXish = re.compile(r"\A[0-9a-fA-F]+\Z")
_UUID = re.compile(r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _shannon(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total, parity = 0, len(nums) % 2
    for i, d in enumerate(nums):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class Finding:
    kind: str
    severity: Severity
    preview: str            # redacted — never the raw secret
    span: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "severity": self.severity.label,
                "preview": self.preview}


@dataclass
class SpillageReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        """Safe to send == nothing HIGH or worse was found."""
        return all(f.severity < Severity.HIGH for f in self.findings)

    @property
    def worst(self) -> Severity | None:
        return max((f.severity for f in self.findings), default=None)

    def describe(self) -> str:
        if not self.findings:
            return "No secrets or credentials detected — safe to send."
        head = f"Withhold — {len(self.findings)} sensitive item(s) detected:"
        rows = [f"  • {f.kind} ({f.severity.label}): {f.preview}" for f in self.findings]
        return head + "\n" + "\n".join(rows)


def _mask(secret: str) -> str:
    secret = secret.strip()
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:3]}…{secret[-2:]} ({len(secret)} chars)"


class SpillageGuard:
    """Scan outbound text for credentials, and raise the alarm when it matters."""

    def __init__(self, known_secrets: Iterable[str] | None = None) -> None:
        # ORION's own secrets, so echoing a real key back is caught even if its
        # shape isn't recognised. Short values are ignored to avoid matching
        # ordinary words.
        self._known = {s for s in (known_secrets or []) if len(str(s)) >= 8}

    # ── detection ─────────────────────────────────────────────────────────────

    def scan(self, text: str) -> SpillageReport:
        text = str(text or "")
        report = SpillageReport()
        claimed: list[tuple[int, int]] = []

        def overlaps(a: int, b: int) -> bool:
            return any(not (b <= s or a >= e) for s, e in claimed)

        def add(kind, severity, raw, span):
            if overlaps(*span):
                return
            claimed.append(span)
            report.findings.append(Finding(kind, severity, _mask(raw), span))

        # 1. ORION's own secrets, verbatim — highest priority.
        for secret in self._known:
            idx = text.find(secret)
            if idx >= 0:
                add("ORION's own credential", Severity.CRITICAL, secret,
                    (idx, idx + len(secret)))

        # 2. Known secret shapes.
        for pat in _PATTERNS:
            for m in pat.regex.finditer(text):
                raw = m.group(pat.group) if pat.group else m.group(0)
                if not raw:
                    continue
                span = m.span(pat.group) if pat.group else m.span(0)
                add(pat.kind, pat.severity, raw, span)

        # 3. Luhn-valid card numbers.
        for m in _CARD.finditer(text):
            if _luhn_ok(m.group(0)):
                add("payment card number", Severity.HIGH, m.group(0), m.span())

        # 4. Narrow high-entropy fallback for un-prefixed secrets.
        for m in _TOKEN.finditer(text):
            tok = m.group(0)
            if len(tok) < _ENTROPY_MIN_LEN:
                continue
            if _HEXish.match(tok) or _UUID.match(tok):
                continue                      # hashes / UUIDs are not secrets
            if _shannon(tok) >= _ENTROPY_THRESHOLD:
                add("high-entropy token", Severity.MEDIUM, tok, m.span())

        report.findings.sort(key=lambda f: (-int(f.severity), f.span[0]))
        return report

    def is_safe(self, text: str) -> bool:
        return self.scan(text).safe

    def redact(self, text: str) -> str:
        """Return *text* with every detected secret blanked out."""
        report = self.scan(text)
        out = str(text or "")
        for f in sorted(report.findings, key=lambda x: -x.span[0]):
            s, e = f.span
            out = out[:s] + f"[REDACTED:{f.kind}]" + out[e:]
        return out

    # ── the standby-piercing alarm ────────────────────────────────────────────

    def guard(self, text: str, bus: Any = None, channel: str = "output") -> SpillageReport:
        """Scan, and if anything critical is present raise it on the safety
        channel — the one that overrides standby, per the user's rule."""
        report = self.scan(text)
        critical = [f for f in report.findings if f.severity >= Severity.CRITICAL]
        if critical and bus is not None:
            kinds = ", ".join(sorted({f.kind for f in critical}))
            message = (f"Stop — the {channel} contains what looks like a live "
                       f"credential ({kinds}). I'm holding it back rather than "
                       "letting it leave.")
            for signal in ("safety_alert", "log"):
                emitter = getattr(bus, signal, None)
                emit = getattr(emitter, "emit", None)
                if emit is not None:
                    try:
                        emit(message)
                        break
                    except Exception:
                        continue
        return report


__all__ = ["Severity", "Finding", "SpillageReport", "SpillageGuard"]
