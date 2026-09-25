"""
Sound watch — ORION noticing the sounds that matter without being asked.

``sound_sense`` answers when asked ("what was that noise?"). A smoke alarm in
the next room, a siren outside or a knock at the door is worth hearing when
nobody thought to ask, so this runs the same local YAMNet classifier over the
microphone's rolling buffer about once a second and raises an alert for a
short list of sounds a person would want to be told about.

It is opt-in, local and cheap:

* no audio leaves the machine and nothing is recorded — each window is read
  from ``audio.RECENT_AUDIO`` (which the capture thread already fills),
  classified and dropped;
* a silent window is skipped before the model runs, and one 0.96 s patch
  costs a few milliseconds of CPU, so an idle room costs almost nothing;
* ORION's own voice is excluded, so his speech cannot trigger an alert.

An alert needs the sound in ``hits`` of the last ``window`` classifications
(one loud frame is not an alarm), then that category stays quiet for its
cooldown so a ringing alarm is announced once, not every second.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

SAMPLE_RATE = 16_000
#: One YAMNet patch: 96 frames of 10 ms hop plus the 25 ms window.
WINDOW_S = 0.975
#: Below this RMS (int16 full scale = 1.0) a window is treated as silence.
SILENCE_RMS = 0.004
#: Half a window of int16 audio: less than this is too little to classify
#: (YAMNet pads a short window, but not one that is mostly padding).
MIN_WINDOW_BYTES = int(WINDOW_S * SAMPLE_RATE)


@dataclass(frozen=True)
class AlertRule:
    """One kind of sound worth announcing."""

    category: str
    labels: frozenset[str]
    threshold: float
    hits: int = 2              # of the last `window` classifications …
    window: int = 3            # … this many must carry the sound
    cooldown_s: float = 60.0
    urgent: bool = False
    #: Broader categories the same sound also triggers. A smoke alarm scores
    #: high as "Alarm" too; one alarm should be announced once.
    quiets: frozenset[str] = frozenset()


#: Deliberately short. Every entry is something a person would want to hear
#: about when they did not ask; music, speech and ordinary noise are not.
#: Specific rules come before the generic ones they quiet.
DEFAULT_RULES: tuple[AlertRule, ...] = (
    AlertRule("smoke alarm", frozenset({"Smoke detector, smoke alarm", "Fire alarm"}),
              0.30, hits=2, window=3, cooldown_s=45.0, urgent=True,
              quiets=frozenset({"alarm"})),
    AlertRule("siren", frozenset({"Siren", "Civil defense siren", "Police car (siren)",
                                  "Ambulance (siren)", "Fire engine, fire truck (siren)"}),
              0.35, hits=2, window=4, cooldown_s=120.0, quiets=frozenset({"alarm"})),
    AlertRule("alarm", frozenset({"Alarm", "Car alarm", "Buzzer"}),
              0.40, hits=2, window=3, cooldown_s=90.0),
    AlertRule("glass breaking", frozenset({"Shatter", "Breaking"}),
              0.40, hits=1, window=1, cooldown_s=60.0, urgent=True),
    AlertRule("baby crying", frozenset({"Baby cry, infant cry"}),
              0.40, hits=2, window=3, cooldown_s=120.0),
    AlertRule("screaming", frozenset({"Screaming"}),
              0.45, hits=2, window=3, cooldown_s=90.0, urgent=True),
    AlertRule("doorbell", frozenset({"Doorbell", "Ding-dong"}),
              0.40, hits=1, window=1, cooldown_s=30.0),
    AlertRule("knocking", frozenset({"Knock"}),
              0.45, hits=2, window=3, cooldown_s=30.0),
)


@dataclass
class Alert:
    category: str
    label: str
    score: float
    at: float
    urgent: bool = False

    def sentence(self) -> str:
        opener = "I can hear" if not self.urgent else "Heads up — I can hear"
        return f"{opener} what sounds like a {self.category}."

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "label": self.label,
                "score": round(self.score, 3), "urgent": self.urgent}


@dataclass
class SoundWatcher:
    """The decision part: class scores in, alerts out. No model, no thread."""

    labels: list[str]
    rules: tuple[AlertRule, ...] = DEFAULT_RULES
    _index: dict[str, list[int]] = field(default_factory=dict, init=False)
    _recent: dict[str, deque] = field(default_factory=dict, init=False)
    _quiet_until: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        where = {name: i for i, name in enumerate(self.labels)}
        for rule in self.rules:
            self._index[rule.category] = [where[n] for n in rule.labels if n in where]
            self._recent[rule.category] = deque(maxlen=max(1, rule.window))

    def observe(self, scores: np.ndarray, now: float | None = None) -> list[Alert]:
        """Feed one window's class scores (shape (521,)); returns new alerts."""
        now = time.monotonic() if now is None else now
        alerts: list[Alert] = []
        for rule in self.rules:
            indices = self._index.get(rule.category) or []
            if not indices:
                continue
            best = max(indices, key=lambda i: float(scores[i]))
            score = float(scores[best])
            recent = self._recent[rule.category]
            recent.append(score >= rule.threshold)
            if sum(recent) < rule.hits:
                continue
            if now < self._quiet_until.get(rule.category, 0.0):
                continue
            self._quiet_until[rule.category] = now + rule.cooldown_s
            for broader in rule.quiets:
                self._quiet_until[broader] = max(self._quiet_until.get(broader, 0.0),
                                                 now + rule.cooldown_s)
            recent.clear()
            alerts.append(Alert(rule.category, self.labels[best], score, now, rule.urgent))
        return alerts

    def reset(self) -> None:
        for recent in self._recent.values():
            recent.clear()
        self._quiet_until.clear()


def window_rms(pcm: bytes) -> float:
    """RMS of int16 PCM on a 0..1 scale (0 for an empty window)."""
    if len(pcm) < 2:
        return 0.0
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(np.square(samples / 32768.0))))


class SoundWatch:
    """Runs a SoundWatcher over the microphone's rolling buffer."""

    def __init__(self, on_alert: Callable[[Alert], None], *,
                 source: Any = None,
                 classify: Callable[[np.ndarray], np.ndarray] | None = None,
                 labels: list[str] | None = None,
                 interval_s: float = 1.0,
                 log: Callable[[str], None] | None = None) -> None:
        self.on_alert = on_alert
        self._source = source
        self._classify = classify
        self._labels = labels
        self.interval_s = max(0.25, float(interval_s))
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watcher: SoundWatcher | None = None
        self.windows = 0           # windows classified
        self.skipped = 0           # silent or missing windows skipped
        self.alerts: deque[Alert] = deque(maxlen=20)
        self.error = ""

    # -- lifecycle -------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> tuple[bool, str]:
        if self.running:
            return True, "already listening for alarms, sirens and the door."
        if not self._prepare():
            return False, self.error or "the sound classifier is not available."
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="orion-sound-watch",
                                        daemon=True)
        self._thread.start()
        return True, ("listening for smoke alarms, sirens, alarms, breaking glass, "
                      "a crying baby, screaming, the doorbell and knocking.")

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._thread = None

    # -- the work --------------------------------------------------------------
    def _prepare(self) -> bool:
        if self._source is None:
            from .audio import RECENT_AUDIO
            self._source = RECENT_AUDIO
        if self._classify is None or self._labels is None:
            from .sound_sense import _Yamnet
            if not _Yamnet.load():
                self.error = "the YAMNet model or onnxruntime is missing."
                return False
            self._labels = list(_Yamnet._labels)
            self._classify = lambda wave: _Yamnet.scores(wave)[0][0]
        self._watcher = SoundWatcher(self._labels)
        return True

    def step(self, now: float | None = None) -> list[Alert]:
        """Classify the latest window once; returns (and delivers) new alerts."""
        assert self._watcher is not None and self._classify is not None
        pcm = self._source.snapshot(WINDOW_S) if self._source is not None else b""
        if len(pcm) < MIN_WINDOW_BYTES or window_rms(pcm) < SILENCE_RMS:
            self.skipped += 1
            # Silence is evidence too: a sound that stopped is not still going.
            self._watcher.observe(np.zeros(len(self._labels or [])), now)
            return []
        wave = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        scores = np.asarray(self._classify(wave), dtype=np.float32)
        self.windows += 1
        alerts = self._watcher.observe(scores, now)
        for alert in alerts:
            self.alerts.append(alert)
            try:
                self.on_alert(alert)
            except Exception as exc:          # a broken consumer must not stop the watch
                self._say(f"SOUND: alert delivery failed - {exc}")
        return alerts

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.step()
            except Exception as exc:
                self.error = str(exc)
                self._say(f"SOUND: watch stopped - {exc}")
                return

    def _say(self, message: str) -> None:
        if self.log is not None:
            try:
                self.log(message)
            except Exception:
                pass

    def describe(self) -> str:
        if not self.running:
            return "The sound watch is off."
        heard = ", ".join(f"{a.category} ({a.score:.2f})" for a in list(self.alerts)[-5:])
        return (f"The sound watch is on: {self.windows} windows classified, "
                f"{self.skipped} quiet ones skipped."
                + (f" Recent alerts: {heard}." if heard else " No alerts so far."))


#: The process-wide watch, created on first use.
_WATCH: SoundWatch | None = None


def watch(on_alert: Callable[[Alert], None] | None = None,
          log: Callable[[str], None] | None = None) -> SoundWatch:
    """The shared watch; *on_alert*/*log* replace its consumers when given."""
    global _WATCH
    if _WATCH is None:
        _WATCH = SoundWatch(on_alert or (lambda _a: None), log=log)
    else:
        if on_alert is not None:
            _WATCH.on_alert = on_alert
        if log is not None:
            _WATCH.log = log
    return _WATCH


def rules_for(categories: Iterable[str]) -> tuple[AlertRule, ...]:
    wanted = {c.strip().lower() for c in categories}
    return tuple(r for r in DEFAULT_RULES if r.category in wanted)


__all__ = ["Alert", "AlertRule", "DEFAULT_RULES", "SoundWatch", "SoundWatcher",
           "rules_for", "watch", "window_rms"]
