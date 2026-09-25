"""
Vision rules — "when the camera sees X, do Y".

A rule joins something the local instruments can measure to something ORION
can do:

    when                         then
    ───────────────────────────  ─────────────────────────────────────────
    object      a cup appears    announce   say it (or your own message)
    object_gone the keys leave   workflow   start one of your workflows,
    count       2+ people        log        just record it
    pose        a hand is raised
    colour      red fills a zone
    motion      movement (in a zone)
    scene_change the view changes
    dark        the lights go off
    covered     the lens is covered

Rules are EDGE-triggered. A rule fires once when its condition becomes true
and has held for ``hold_s``, then re-arms only after the condition has been
false for RELEASE_S — so "when someone appears, say hello" greets each
arrival once rather than every minute they sit there. ``cooldown_s`` is a
floor between firings on top of that, so a flickering condition can never
become a storm of workflow runs.

An instrument that did not run on a given frame (object detection and pose
are rate-limited) reports "unknown", which changes nothing: a rule is never
reset — or fired — by the absence of a measurement.

Rules persist in ``config/vision_rules.json`` (written atomically). Workflows
run through the workflow engine, so a camera-triggered action has exactly the
permissions — and the confirmation gates — of the same workflow run by hand.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR
from .object_detection import canonical_label
from .pose_tracking import POSE_GESTURES, canonical_gesture
from .utils import utc_stamp
from .vision_lab import COLOUR_NAMES, ZONES, normalise_zone

RULES_PATH = CONFIG_DIR / "vision_rules.json"

TRIGGERS = {
    "object": "a kind of object is in view",
    "object_gone": "an object that was in view has not been seen for the hold time",
    "count": "at least a number of one kind of object are in view",
    "pose": "a body gesture (hand_raised, both_hands_up, arms_out, person)",
    "colour": "a colour covers part of the view",
    "motion": "movement, optionally in one zone",
    "scene_change": "the whole view changes",
    "dark": "the view goes dark",
    "covered": "the lens is covered",
}
TRIGGER_ALIASES = {"color": "colour", "gone": "object_gone", "leaves": "object_gone",
                   "missing": "object_gone", "gesture": "pose", "body": "pose",
                   "movement": "motion", "moves": "motion", "lights_off": "dark",
                   "darkness": "dark", "blocked": "covered", "scene": "scene_change",
                   "people": "count", "sees": "object", "see": "object"}
ACTIONS = ("announce", "workflow", "log")

RELEASE_S = 3.0          # a condition must be false this long before the rule re-arms
MIN_COOLDOWN_S = 5.0
MAX_RULES = 40

#: Which instrument each trigger needs. Rules that need nothing extra run on
#: the always-on light/colour/motion reading.
_NEEDS = {"object": "objects", "object_gone": "objects", "count": "objects", "pose": "pose"}


@dataclass
class VisionRule:
    id: str
    name: str
    trigger: str
    target: str = ""
    zone: str | None = None
    count: int = 1
    min_confidence: float = 0.45
    share: float = 0.15
    hold_s: float = 1.0
    cooldown_s: float = 60.0
    action: str = "announce"
    workflow: str = ""
    message: str = ""
    enabled: bool = True
    created: str = ""
    fired: int = 0
    last_fired_at: str = ""
    # Runtime state — never persisted.
    _holding_since: float | None = field(default=None, repr=False, compare=False)
    _false_since: float | None = field(default=None, repr=False, compare=False)
    _armed: bool = field(default=True, repr=False, compare=False)
    _last_fire: float | None = field(default=None, repr=False, compare=False)
    _seen_at: float | None = field(default=None, repr=False, compare=False)

    def persisted(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def describe(self) -> str:
        what = {
            "object": f"a {self.target} is in view",
            "object_gone": f"the {self.target} is gone for {self.hold_s:g}s",
            "count": f"{self.count}+ {self.target}s are in view",
            "pose": f"someone shows '{self.target}'",
            "colour": f"{self.target} covers {self.share:.0%}",
            "motion": "something moves",
            "scene_change": "the view changes",
            "dark": "it goes dark",
            "covered": "the lens is covered",
        }[self.trigger]
        where = f" ({self.zone})" if self.zone else ""
        then = (f"run workflow '{self.workflow}'" if self.action == "workflow"
                else "log it" if self.action == "log"
                else f"say \"{self.message}\"" if self.message else "tell you")
        state = "" if self.enabled else " [off]"
        fired = f", fired {self.fired}×" if self.fired else ""
        return f"{self.id} · {self.name}: when {what}{where}, {then}{fired}{state}"


@dataclass(frozen=True)
class Observation:
    """What the instruments said on one frame. ``None`` = did not run."""

    now: float
    reading: Any = None                      # vision_lab.FrameReading
    detections: Sequence[Any] | None = None  # object_detection.Detection
    pose: Any = None                         # pose_tracking.PoseReading
    events: Sequence[Any] = ()               # perception.PerceptionEvent


@dataclass(frozen=True)
class Firing:
    rule: VisionRule
    detail: str


class VisionRules:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else RULES_PATH
        self.rules: list[VisionRule] = []
        self.watch_on_startup = False
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.watch_on_startup = bool(raw.get("watch_on_startup", False))
        known = set(VisionRule.__dataclass_fields__)
        for item in raw.get("rules", []):
            if not isinstance(item, dict) or item.get("trigger") not in TRIGGERS:
                continue
            try:
                self.rules.append(VisionRule(**{k: v for k, v in item.items()
                                                if k in known and not k.startswith("_")}))
            except TypeError:
                continue

    def save(self) -> None:
        payload = {"watch_on_startup": self.watch_on_startup,
                   "rules": [rule.persisted() for rule in self.rules]}
        atomic_write_text(self.path, json.dumps(payload, indent=2))

    # ── editing ──────────────────────────────────────────────────────────────

    def add(self, *, trigger: str, target: str = "", zone: str | None = None,
            count: int | None = None, hold_s: float | None = None,
            cooldown_s: float | None = None, action: str = "announce",
            workflow: str = "", message: str = "", name: str = "",
            min_confidence: float | None = None, share: float | None = None) -> VisionRule:
        """Validate and store a rule. Raises ValueError with a readable reason."""
        if len(self.rules) >= MAX_RULES:
            raise ValueError(f"there are already {MAX_RULES} vision rules — remove one first")
        kind = str(trigger or "").strip().lower().replace(" ", "_").replace("-", "_")
        kind = TRIGGER_ALIASES.get(kind, kind)
        if kind not in TRIGGERS:
            raise ValueError(f"'{trigger}' is not something I can watch for — use one of: "
                             + ", ".join(TRIGGERS))
        canonical = ""
        if kind in ("object", "object_gone", "count"):
            canonical = canonical_label(target) or ""
            if not canonical:
                raise ValueError(f"I can't recognise '{target}' as an object — I know the 80 "
                                 "everyday COCO objects (person, cup, cell phone, laptop, "
                                 "bottle, dog, book …)")
        elif kind == "pose":
            canonical = canonical_gesture(target or "hand_raised") or ""
            if not canonical:
                raise ValueError(f"'{target}' is not a gesture I can read — use one of: "
                                 + ", ".join(POSE_GESTURES))
        elif kind == "colour":
            canonical = str(target or "").strip().lower().replace("gray", "grey")
            if canonical not in COLOUR_NAMES:
                raise ValueError(f"'{target}' is not a colour I name — use one of: "
                                 + ", ".join(COLOUR_NAMES))
        zone_name = normalise_zone(zone)
        verb = str(action or "announce").strip().lower()
        verb = {"say": "announce", "speak": "announce", "tell": "announce",
                "run": "workflow", "run_workflow": "workflow"}.get(verb, verb)
        if verb not in ACTIONS:
            raise ValueError(f"'{action}' is not an action — use announce, workflow or log")
        if verb == "workflow" and not str(workflow or "").strip():
            raise ValueError("a workflow rule needs the name of the workflow to run")
        rule = VisionRule(
            id=self._next_id(),
            name=str(name or "").strip()[:60] or f"{kind.replace('_', ' ')} {canonical}".strip(),
            trigger=kind,
            target=canonical,
            zone=zone_name,
            count=max(1, int(count)) if count else (2 if kind == "count" else 1),
            min_confidence=min(0.95, max(0.2, float(min_confidence))) if min_confidence else 0.45,
            share=min(0.95, max(0.02, float(share))) if share else 0.15,
            # A scene change is an instant, true for one frame: any hold would
            # mean it could never fire. Absence needs a real wait, or a person
            # stepping in front of the keys would count as the keys leaving.
            hold_s=max(0.0, float(hold_s)) if hold_s is not None
            else {"object_gone": 10.0, "scene_change": 0.0}.get(kind, 1.0),
            cooldown_s=max(MIN_COOLDOWN_S, float(cooldown_s)) if cooldown_s is not None else 60.0,
            action=verb,
            workflow=str(workflow or "").strip(),
            message=str(message or "").strip()[:200],
            created=utc_stamp(),
        )
        self.rules.append(rule)
        self.save()
        return rule

    def remove(self, ident: str) -> VisionRule | None:
        key = str(ident or "").strip().lower()
        for rule in list(self.rules):
            if key in (rule.id.lower(), rule.name.lower()):
                self.rules.remove(rule)
                self.save()
                return rule
        return None

    def set_enabled(self, ident: str, enabled: bool) -> VisionRule | None:
        key = str(ident or "").strip().lower()
        for rule in self.rules:
            if key in (rule.id.lower(), rule.name.lower()):
                rule.enabled = bool(enabled)
                self.save()
                return rule
        return None

    def set_watch_on_startup(self, on: bool) -> None:
        self.watch_on_startup = bool(on)
        self.save()

    def _next_id(self) -> str:
        used = {rule.id for rule in self.rules}
        n = len(self.rules) + 1
        while f"r{n}" in used:
            n += 1
        return f"r{n}"

    # ── reading ──────────────────────────────────────────────────────────────

    def enabled_rules(self) -> list[VisionRule]:
        return [rule for rule in self.rules if rule.enabled]

    def needs(self) -> set[str]:
        """Instruments the enabled rules depend on ('objects', 'pose')."""
        return {_NEEDS[rule.trigger] for rule in self.enabled_rules() if rule.trigger in _NEEDS}

    def describe(self) -> str:
        if not self.rules:
            return ("No vision rules yet. Example: 'when you see a cup on the left, "
                    "tell me' or 'when I raise my hand, run my focus workflow'.")
        lines = [rule.describe() for rule in self.rules]
        lines.append("Watching automatically at start-up." if self.watch_on_startup
                     else "They run while I am watching (not armed at start-up).")
        return "\n".join(lines)

    # ── evaluating ───────────────────────────────────────────────────────────

    def evaluate(self, observation: Observation) -> list[Firing]:
        firings = []
        for rule in self.enabled_rules():
            met, detail = self._condition(rule, observation)
            if met is None:
                continue                         # instrument didn't run: change nothing
            now = observation.now
            if not met:
                rule._holding_since = None
                if rule._false_since is None:
                    rule._false_since = now
                if not rule._armed and now - rule._false_since >= RELEASE_S:
                    rule._armed = True
                continue
            rule._false_since = None
            if rule._holding_since is None:
                rule._holding_since = now
            if not rule._armed or now - rule._holding_since < rule.hold_s:
                continue
            if rule._last_fire is not None and now - rule._last_fire < rule.cooldown_s:
                continue
            rule._armed = False
            rule._last_fire = now
            rule.fired += 1
            rule.last_fired_at = utc_stamp()
            if rule.trigger == "object_gone":
                rule._seen_at = None             # must be seen again before it can go again
            firings.append(Firing(rule, detail))
        if firings:
            try:
                self.save()                      # keep the fired counts
            except OSError:
                pass
        return firings

    def _condition(self, rule: VisionRule, obs: Observation) -> tuple[bool | None, str]:
        kind = rule.trigger
        if kind in ("object", "count", "object_gone"):
            if obs.detections is None:
                return None, ""
            matches = [d for d in obs.detections
                       if d.label == rule.target and d.confidence >= rule.min_confidence
                       and (rule.zone is None or d.zone == rule.zone)]
            if kind == "object_gone":
                if matches:
                    rule._seen_at = obs.now
                    return False, ""
                return (rule._seen_at is not None,
                        f"the {rule.target} has gone from view")
            needed = rule.count if kind == "count" else 1
            where = ", ".join(sorted({d.zone for d in matches}))
            noun = rule.target if len(matches) == 1 else f"{len(matches)} {rule.target}s"
            return len(matches) >= needed, f"{noun} in view ({where})"
        if kind == "pose":
            if obs.pose is None:
                return None, ""
            from .vision_lab import zone_of
            people = [p for p in obs.pose.people if rule.target in p.gestures
                      and (rule.zone is None or zone_of(*p.centre) == rule.zone)]
            return bool(people), f"{rule.target.replace('_', ' ')} seen"
        if kind == "scene_change":
            return any(getattr(e, "kind", "") == "scene_changed" for e in obs.events), \
                "the whole view changed"
        reading = obs.reading
        if reading is None:
            return None, ""
        if kind == "colour":
            share = reading.colour_share(rule.target, rule.zone)
            return share >= rule.share, f"{rule.target} covers {share:.0%} of the " + \
                (f"{rule.zone} of the view" if rule.zone else "view")
        if kind == "motion":
            motion = reading.motion
            if motion is None:
                return None, ""
            if rule.zone is not None:
                moving = motion.zones[ZONES.index(rule.zone)] >= 0.05
            else:
                moving = motion.is_moving
            return moving, f"movement {('in the ' + rule.zone) if rule.zone else 'in view'}"
        if kind == "dark":
            return reading.dark and not reading.covered, "the view has gone dark"
        if kind == "covered":
            return reading.covered, "the camera lens is covered"
        return None, ""


def act(firing: Firing, *, say: Callable[[str], Any], log: Callable[[str], Any],
        run_workflow: Callable[[str, str], Any] | None) -> str:
    """Carry out a fired rule. Returns what was done, for the event log."""
    rule = firing.rule
    if rule.action == "announce":
        say(rule.message or f"Heads up — {firing.detail}.")
        return f"said it ({firing.detail})"
    if rule.action == "workflow":
        if run_workflow is None:
            log(f"VISION RULE: {rule.name} fired but workflows are unavailable")
            return "workflows unavailable"
        result = run_workflow(rule.workflow, f"{rule.name}: {firing.detail}")
        text = getattr(result, "text", str(result))
        log(f"VISION RULE: {rule.name} -> workflow '{rule.workflow}': {text}")
        return f"ran workflow '{rule.workflow}'"
    log(f"VISION RULE: {rule.name} — {firing.detail}")
    return "logged"


__all__ = ["ACTIONS", "Firing", "Observation", "TRIGGERS", "VisionRule", "VisionRules", "act"]
