"""
Capability gaps — what the user asked for that ORION could not do.

The self-improvement heartbeat (forge.ImprovementHeartbeat) decides what to
forge from evidence: a log tail, diagnostics, per-tool failure COUNTS and the
self-repair journal. None of that says what the user actually WANTED. A
request ORION declined ("I can't control your lights") left no trace at all
beyond a spoken sentence, and a failed tool left a counter with no words
attached — so the forge was steered by ORION's own faults, never by the
things the person kept asking for and not getting.

This ledger keeps those, structured:

* **failures** — every tool call that did not succeed, from the dispatcher
  (so voice, text, phone and plans alike), with the words that asked for it
  when they are known. Classified first: a clarifying question ("which
  one?"), a pending confirmation or a security guard is ORION working as
  designed, not a missing capability, and is not counted.
* **declines** — a reply in which ORION said it cannot do something
  without having tried a tool. The plainest evidence of a missing capability
  there is.

Similar events are clustered (same tool and the same failure, or the same
content words for a decline) so "asked three times this week" is visible as
one need, not three anecdotes. A cluster the heartbeat has been shown three
times without it recurring is retired, so a need nobody has since repeated
cannot drive forge attempts forever. Persisted to ``CONFIG_DIR/capability_gaps
.json`` (bounded, atomic). Never raises into its callers.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

GAPS_PATH = CONFIG_DIR / "capability_gaps.json"
MAX_CLUSTERS = 200
MAX_EXAMPLES = 3
#: How long a spoken request stays attached to the tool calls it caused.
UTTERANCE_TTL_S = 90.0
#: A cluster shown to the heartbeat this often with no new occurrence retires.
RETIRE_AFTER_PRESENTED = 3

#: ORION saying it cannot do something. Matched on his own reply.
_DECLINE_RE = re.compile(
    r"(?i)\b(?:i(?:'|’)m\s+(?:not\s+able|unable)|"
    r"i\s+(?:can(?:'|’)?t|cannot|am\s+(?:not\s+able|unable)|"
    r"don(?:'|’)?t\s+have\s+(?:a\s+way|the\s+ability|access|any\s+way|a\s+tool|the\s+tools?)|"
    r"have\s+no\s+(?:way|tool|ability))|"
    r"(?:that|this)(?:'|’)?s\s+not\s+something\s+i\s+can|"
    r"(?:that|this)\s+isn(?:'|’)?t\s+something\s+i\s+can|"
    r"i\s+(?:currently\s+)?(?:lack|don(?:'|’)?t\s+(?:currently\s+)?support))")

#: Outcomes that are ORION working as designed rather than unable. A policy
#: deferral ("needs a foreground session") is one: 26 Oct 2026 two of them
#: were read as a missing capability and the self-improver set out to forge
#: a "session_manager" tool for a rule, not a gap.
_NOT_A_GAP_RE = re.compile(
    r"(?i)(confirm|approval|approve|cancelled|canceled|which (?:one|application|app)|"
    r"refused - spoken go-ahead|not the owner|say which|did you mean|"
    r"needs your|waiting for you|already (?:open|running)|nothing to (?:do|undo)|"
    r"no (?:running process|visible window)|i won't start|i won't press|"
    r"needs a foreground session|left your desktop unchanged)")

#: Outcomes that say a capability is missing or unavailable.
_MISSING_RE = re.compile(
    r"(?i)(not installed|unavailable|not available|isn(?:'|’)?t available|unsupported|"
    r"not supported|unknown dispatch target|no tool|could not find an application|"
    r"is not configured|not configured|no provider|requires? .* (?:key|token|account))")

_STOP = frozenset("""
a an the and or but to of in on at for from with by about into over please can could
would will you me my i it is are be do does did just now up this that these those
some any orion hey hi okay ok so what how when where why who which like want need
""".split())


def _content_words(text: str, limit: int = 6) -> list[str]:
    words = [w for w in re.findall(r"[a-z][a-z0-9']+", str(text or "").lower())
             if w not in _STOP and len(w) > 2]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:limit]


#: The words of saying no, as opposed to the capability being declined.
_DECLINE_WORDS = frozenset({
    "can't", "cannot", "unable", "afraid", "sorry", "yet", "currently", "able",
    "don't", "access", "way", "tool", "tools", "ability", "isn't", "something",
    "that's", "i'm", "here", "from", "right", "now", "really", "unfortunately"})


def _ready_encoder() -> Any:
    """The local sentence encoder if it is ALREADY loaded, else None."""
    try:
        from .semantic import ENCODER
        return ENCODER if ENCODER.ready() else None
    except Exception:
        return None


def _stems(words: list[str]) -> list[str]:
    """Crude plural folding so 'light' and 'lights' are one word."""
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in words]


def _signature(text: str) -> str:
    """A failure message with its specifics removed, so recurrences cluster."""
    text = str(text or "").splitlines()[0] if text else ""
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", text)
    text = re.sub(r"\d+(?:\.\d+)?", "#", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:80]


def classify(outcome: str) -> str:
    """'designed' (a question, confirmation or guard), 'missing' or 'failure'."""
    text = str(outcome or "")
    if _NOT_A_GAP_RE.search(text):
        return "designed"
    if _MISSING_RE.search(text):
        return "missing"
    return "failure"


def is_decline(reply: str) -> bool:
    return bool(_DECLINE_RE.search(str(reply or "")))


class GapLedger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.clusters: dict[str, dict[str, Any]] = {}
        self.version = 0                  # bumps on every new occurrence
        self._lock = threading.Lock()
        self._loaded = False
        self._utterance = ("", 0.0)
        self._tool_at = 0.0

    def _file(self) -> Path:
        return self.path if self.path is not None else GAPS_PATH

    # ── persistence ──────────────────────────────────────────────────────────

    def load(self) -> "GapLedger":
        if self._loaded:
            return self
        self._loaded = True
        try:
            data = json.loads(self._file().read_text(encoding="utf-8"))
            clusters = data.get("clusters") if isinstance(data, dict) else None
            if isinstance(clusters, dict):
                self.clusters = {str(k): v for k, v in clusters.items() if isinstance(v, dict)}
        except Exception:
            pass
        return self

    def _save(self) -> None:
        try:
            from .atomic_io import atomic_write_text
            self._file().parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._file(), json.dumps({"clusters": self.clusters}, indent=1))
        except Exception:
            pass

    # ── what was asked ───────────────────────────────────────────────────────

    def note_utterance(self, text: str) -> None:
        """The words of the request now being handled (any origin)."""
        self._utterance = (" ".join(str(text or "").split())[:240], time.monotonic())

    def current_utterance(self) -> str:
        text, at = self._utterance
        return text if text and time.monotonic() - at <= UTTERANCE_TTL_S else ""

    # ── recording ────────────────────────────────────────────────────────────

    def _add(self, key: str, kind: str, label: str, example: str, detail: str) -> None:
        now = time.time()
        with self._lock:
            self.load()
            cluster = self.clusters.get(key)
            if cluster is None:
                cluster = {"kind": kind, "label": label, "count": 0, "first": now,
                           "examples": [], "detail": detail[:200], "presented": 0}
                self.clusters[key] = cluster
            cluster["count"] = int(cluster.get("count", 0)) + 1
            cluster["last"] = now
            cluster["presented"] = 0           # it happened again: it is live
            cluster.pop("retired", None)
            if cluster.get("addressed_by") and now > float(cluster.get("addressed_at", 0)):
                # A tool was built for this and it is STILL being asked for:
                # the tool did not meet the need (or is not being found).
                cluster["still_unmet_after"] = cluster["addressed_by"]
            if example and example not in cluster["examples"]:
                cluster["examples"] = (cluster["examples"] + [example[:160]])[-MAX_EXAMPLES:]
            if detail:
                cluster["detail"] = detail[:200]
            if len(self.clusters) > MAX_CLUSTERS:
                oldest = sorted(self.clusters, key=lambda k: self.clusters[k].get("last", 0))
                for stale in oldest[: len(self.clusters) - MAX_CLUSTERS]:
                    del self.clusters[stale]
            self.version += 1
            self._save()

    def record_failure(self, tool: str, args: dict[str, Any] | None, outcome: str) -> str:
        """A tool call that did not succeed. Returns its classification."""
        try:
            self._tool_at = time.monotonic()
            kind = classify(outcome)
            if kind == "designed":
                return kind
            utterance = self.current_utterance()
            key = f"tool:{tool}:{_signature(outcome)}"
            example = utterance or _args_summary(args)
            self._add(key, kind, f"{tool} - {str(outcome).splitlines()[0][:120] if outcome else ''}",
                      example, str(outcome or ""))
            return kind
        except Exception:
            return "failure"

    def note_tool_call(self) -> None:
        self._tool_at = time.monotonic()

    def _similar_decline(self, words: list[str], reply_words: list[str] | None = None,
                         request: str = "") -> str:
        """An existing declined cluster asking for the same thing in other
        words. Three signals, any one enough:

        * the request's words overlap ("turn on the lights" / "switch the
          living room lights on");
        * ORION named the same missing capability ("I can't convert units"
          for miles, stone and celsius alike);
        * the requests MEAN the same, when the local sentence encoder is
          already loaded (it is never loaded just for this).
        """
        mine = set(_stems(words))
        said = set(_stems(reply_words or []))
        encoder = _ready_encoder()
        query = encoder.encode_one(request) if (encoder is not None and request) else None
        best, best_score = "", 0.0
        self.load()
        for key, cluster in list(self.clusters.items()):
            if cluster.get("kind") != "declined":
                continue
            theirs = set(cluster.get("words") or _stems(str(cluster.get("label", "")).split()))
            score = len(mine & theirs) / len(mine | theirs) if theirs else 0.0
            their_said = set(cluster.get("reply_words") or [])
            if len(said) >= 2 and their_said:
                if len(said & their_said) / len(said | their_said) >= 0.6:
                    score = max(score, 0.6)
            if query is not None and cluster.get("examples"):
                try:
                    other = encoder.encode_one(cluster["examples"][-1])
                    if other is not None and float(query @ other) >= 0.78:
                        score = max(score, 0.6)
                except Exception:
                    pass
            if score > best_score:
                best, best_score = key, score
        return best if best_score >= 0.4 else ""

    def note_reply(self, request: str, reply: str, *, tool_used: bool | None = None) -> bool:
        """ORION's reply to *request*. Records a decline made WITHOUT trying a
        tool (a tool that failed is already recorded as a failure)."""
        try:
            if not request or not is_decline(reply):
                return False
            if tool_used is None:
                tool_used = time.monotonic() - self._tool_at < 30.0
            if tool_used:
                return False
            words = _content_words(request)
            if not words:
                return False
            sentence = next((s for s in re.split(r"(?<=[.!?])\s+", reply) if is_decline(s)), reply)
            reply_words = [w for w in _content_words(sentence, limit=8) if w not in _DECLINE_WORDS]
            key = (self._similar_decline(words, reply_words, request)
                   or "declined:" + " ".join(sorted(words[:4])))
            self._add(key, "declined", " ".join(words[:4]), request, sentence.strip())
            with self._lock:
                cluster = self.clusters.get(key)
                if cluster is not None:
                    merged = list(dict.fromkeys(list(cluster.get("words") or []) + _stems(words)))
                    cluster["words"] = merged[:12]
                    if reply_words and not cluster.get("reply_words"):
                        cluster["reply_words"] = _stems(reply_words)[:8]
                    self._save()
            return True
        except Exception:
            return False

    # ── reading ──────────────────────────────────────────────────────────────

    def recurring(self, minimum: int = 2, limit: int = 8) -> list[dict[str, Any]]:
        """Live clusters seen at least *minimum* times, most frequent first."""
        self.load()
        rows = [dict(c, key=k) for k, c in list(self.clusters.items())
                if int(c.get("count", 0)) >= minimum and not c.get("retired")]
        rows.sort(key=lambda c: (-int(c.get("count", 0)), -float(c.get("last", 0))))
        return rows[:limit]

    def mark_presented(self, keys: list[str]) -> None:
        """The heartbeat was shown these; one not recurring by the third
        showing retires."""
        with self._lock:
            self.load()
            for key in keys:
                cluster = self.clusters.get(key)
                if cluster is None:
                    continue
                cluster["presented"] = int(cluster.get("presented", 0)) + 1
                if cluster["presented"] >= RETIRE_AFTER_PRESENTED:
                    cluster["retired"] = time.time()
            self._save()

    def match_tool(self, tool_name: str, plan: str = "") -> str:
        """The live cluster a newly forged tool most plausibly answers, or ''."""
        self.load()
        tool_words = set(_stems(_content_words(
            f"{str(tool_name).replace('_', ' ')} {plan}", limit=24)))
        best, best_overlap = "", 0
        for key, cluster in list(self.clusters.items()):
            if cluster.get("retired") or cluster.get("kind") == "designed":
                continue
            words = set(cluster.get("words") or []) | set(cluster.get("reply_words") or [])
            for example in cluster.get("examples") or []:
                words |= set(_stems(_content_words(example)))
            overlap = len(tool_words & words)
            if overlap > best_overlap:
                best, best_overlap = key, overlap
        return best if best_overlap >= 1 else ""

    def mark_addressed(self, key: str, tool_name: str) -> dict[str, Any] | None:
        """A tool was forged for this need: stop steering the forge with it
        (until it recurs). Returns the cluster."""
        with self._lock:
            self.load()
            cluster = self.clusters.get(key)
            if cluster is None:
                return None
            cluster["addressed_by"] = tool_name
            cluster["addressed_at"] = time.time()
            cluster["retired"] = time.time()
            cluster.pop("still_unmet_after", None)
            self._save()
            return dict(cluster)

    def report(self, limit: int = 10) -> str:
        """For the user: what they asked for that ORION could not do."""
        self.load()
        rows = sorted((c for c in list(self.clusters.values()) if c.get("kind") != "designed"),
                      key=lambda c: (-int(c.get("count", 0)), -float(c.get("last", 0))))[:limit]
        if not rows:
            return "Nothing on record - I haven't been unable to do anything you've asked."
        lines = ["What you've asked for that I couldn't do (most frequent first):"]
        for row in rows:
            when = time.strftime("%d %b", time.localtime(float(row.get("last", 0))))
            example = (row.get("examples") or [""])[-1]
            what = f"'{example}'" if example else row.get("label", "")
            why = row.get("detail", "")
            state = (f" [built: {row['addressed_by']}]" if row.get("addressed_by")
                     and not row.get("still_unmet_after")
                     else " [no longer pursued]" if row.get("retired") else "")
            lines.append(f"- {int(row.get('count', 0))}x ({row.get('kind')}, last {when}): "
                         f"{what} - {why[:110]}{state}")
        return "\n".join(lines)

    def context_for_review(self, limit: int = 8) -> tuple[str, list[str]]:
        """The heartbeat's evidence section and the cluster keys shown in it."""
        rows = self.recurring(limit=limit)
        if not rows:
            return "", []
        lines = ["=== UNMET REQUESTS (the user asked; ORION could not) ==="]
        for row in rows:
            examples = "; ".join(f"'{e}'" for e in row.get("examples") or [])
            lines.append(f"- {int(row['count'])}x {row.get('kind')}: {row.get('label')}"
                         + (f" | asked as: {examples}" if examples else "")
                         + (f" | ORION's answer: {row.get('detail', '')[:140]}"
                            if row.get("kind") == "declined" else "")
                         + (f" | NOTE: tool '{row['still_unmet_after']}' was forged for this "
                            "and it is still being asked for — check that it works or is "
                            "findable before building another" if row.get("still_unmet_after")
                            else ""))
        lines.append("A recurring unmet request is the strongest reason to forge a tool — "
                     "but only when a self-contained tool could genuinely do it (no "
                     "hardware, account or service ORION does not have).")
        return "\n".join(lines), [row["key"] for row in rows]


def _args_summary(args: dict[str, Any] | None) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    parts = []
    for key, value in list(args.items())[:4]:
        text = " ".join(str(value).split())
        parts.append(f"{key}={text[:40]}")
    return ", ".join(parts)


LEDGER = GapLedger()

__all__ = ["GAPS_PATH", "GapLedger", "LEDGER", "classify", "is_decline"]
