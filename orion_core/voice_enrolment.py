"""
Learning the owner's voice, and knowing on every turn who is talking.

The voiceprint encoder existed and nobody was enrolled, because enrolling
meant one five-second clip recorded on a second microphone stream: one
sentence, one mood, one distance from the microphone. A profile built from
that recognises you in exactly the conditions you recorded it in. This module
is the five-minute alternative, and the per-turn use of what it learns.

The guided session
------------------
ORION asks for about twenty short lines - his name, the owner's name, ordinary
sentences, a few said quietly and a few louder, as if from across the room -
and takes each finished utterance as it arrives from the SAME capture path
conversation uses (no second microphone stream). Each one is checked before it
is kept, and a rejected one is explained in plain words: silent, too short,
clipped, too quiet, too noisy, or not the same voice as the others (someone
else, or the TV). The kept samples become a profile in ``voiceprint`` - the
mean plus sub-centroids and a personal threshold - stored as EMBEDDINGS ONLY.
No audio is written anywhere, at any point; "forget my voice" deletes the
profile and switches the voice gates off.

Started by voice through the ``voice_speaker_id`` tool ("learn my voice") or
from Preferences. Both reach the same :class:`EnrolmentSession`.

Where utterances come from
--------------------------
Every complete utterance - on the Live path (``live_presence`` assembles it
from the Silero-VAD-gated capture stream) and on the offline path - is read by
``VoicePresence``, which asks ``SpeakerIdentificationService.identify``. That
call routes here through :func:`observe_utterance`: while a session is
learning, the utterance becomes an enrolment sample; otherwise it is
identified and recorded as this turn's speaker. The live gate also asks
``voiceprint.judge`` part-way through each utterance (at 1.6 s of speech), and
that provisional verdict is recorded here too. Everything runs on worker
threads; the capture thread only appends bytes.

What a verdict may decide
-------------------------
Two things, and only two: whether ORION answers (with "owner only" on, a voice
that is confidently NOT the owner's is not answered) and how he addresses the
speaker (the owner's name only when it is the owner talking). A match never
authorises anything - no payment, message, deletion or remote action is ever
released because a voice matched. The only effect a verdict can have on an
action is ``speaker_gate`` REFUSING a spoken go-ahead, which already existed.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SAMPLE_RATE = 16000

# ── what counts as a usable sample ───────────────────────────────────────────

#: Seconds of actual speech a sample needs. The encoder can embed less, but
#: an enrolment is the one place ORION can ask for more, so it does.
MIN_SPEECH_SECONDS = 1.4

#: Below this RMS (about -80 dBFS) there is nothing there at all.
SILENT_RMS = 1e-4

#: Speech whose active level is below this is lost in the microphone's own
#: noise; the encoder would be fingerprinting the hiss.
TOO_QUIET_DBFS = -55.0

#: Samples at or above this fraction of full scale count as clipped, and more
#: than MAX_CLIPPED_FRACTION of them means the sample is distorted.
CLIP_LEVEL = 0.985
MAX_CLIPPED_FRACTION = 0.001

#: Peak speech frames over the quiet floor, in dB. Below this the floor is
#: background noise rather than the gaps between syllables.
MIN_SNR_DB = 12.0

#: Once a few samples are in, a new one must sit at least this close to their
#: average. A different person - or the TV - talking mid-enrolment otherwise
#: becomes part of "your" voice.
CONSISTENCY_FLOOR = 0.58
CONSISTENCY_AFTER = 4

# ── the session ──────────────────────────────────────────────────────────────

TARGET_SAMPLES = 20
MIN_SAMPLES = 12
SESSION_SECONDS = 8 * 60

#: After a finished enrolment, "owner only" is switched on when the profile's
#: own leave-one-out check recognises at least this share of its samples.
OWNER_ONLY_SELF_CHECK = 0.95

#: (style, line with the owner's name, line without it).
PHRASES: tuple[tuple[str, str, str], ...] = (
    ("normal", "Orion, this is my voice. Please remember how I sound.",
     "Orion, this is my voice. Please remember how I sound."),
    ("normal", "Hello Orion, it's {name} here.", "Hello Orion, it's me here."),
    ("normal", "My name is {name}, and this computer is mine.",
     "This is my computer, and you work for me."),
    ("normal", "The quick brown fox jumps over the lazy dog by the river.",
     "The quick brown fox jumps over the lazy dog by the river."),
    ("normal", "I usually ask about the weather before I leave the house.",
     "I usually ask about the weather before I leave the house."),
    ("quiet", "Orion, I'm speaking quietly now, so listen carefully.",
     "Orion, I'm speaking quietly now, so listen carefully."),
    ("loud", "Orion! Can you hear me from across the room?",
     "Orion! Can you hear me from across the room?"),
    ("normal", "Twenty three, forty seven, ninety one, and one hundred and six.",
     "Twenty three, forty seven, ninety one, and one hundred and six."),
    ("normal", "A large size in stockings is hard to sell.",
     "A large size in stockings is hard to sell."),
    ("normal", "Later today I need to phone the garage about the car.",
     "Later today I need to phone the garage about the car."),
    ("quiet", "This is how I sound late at night when everyone else is asleep.",
     "This is how I sound late at night when everyone else is asleep."),
    ("loud", "This is how I sound when I'm calling you from the kitchen!",
     "This is how I sound when I'm calling you from the kitchen!"),
    ("normal", "The birch canoe slid on the smooth planks.",
     "The birch canoe slid on the smooth planks."),
    ("normal", "Glue the sheet to the dark blue background.",
     "Glue the sheet to the dark blue background."),
    ("normal", "It's easy to tell the depth of a well.",
     "It's easy to tell the depth of a well."),
    ("normal", "Orion, I'm going to read you a few more sentences.",
     "Orion, I'm going to read you a few more sentences."),
    ("normal", "These days a chicken leg is a rare dish.",
     "These days a chicken leg is a rare dish."),
    ("normal", "Rice is often served in round bowls.",
     "Rice is often served in round bowls."),
    ("normal", "The juice of lemons makes fine punch.",
     "The juice of lemons makes fine punch."),
    ("normal", "Four hours of steady work faced us.",
     "Four hours of steady work faced us."),
    ("quiet", "Orion, this one is just a quiet test between us.",
     "Orion, this one is just a quiet test between us."),
    ("loud", "Orion, I'm a bit further from the microphone now!",
     "Orion, I'm a bit further from the microphone now!"),
    ("normal", "Orion, you work for me, {name}, and this is what I sound like.",
     "Orion, you work for me, and this is what I sound like."),
    ("normal", "That's the last one. Thank you, Orion.",
     "That's the last one. Thank you, Orion."),
)

STYLE_HINTS = {
    "quiet": "Quietly, as if someone nearby is asleep",
    "loud": "A little louder, as if from across the room",
    "normal": "",
}

REASONS = {
    "silent": "I didn't hear any speech in that one.",
    "too_short": "That was too short for me to learn from - the whole line, please.",
    "too_quiet": ("That was too quiet to learn from. A little louder, or a "
                  "little closer to the microphone."),
    "clipped": ("That was so loud it distorted. Back off from the microphone a "
                "little, or speak a bit softer."),
    "noisy": ("There was too much background noise in that one. Could you "
              "try it again when it's quieter?"),
    "inconsistent": ("That didn't sound like the same voice as the others - "
                     "was someone else talking, or the TV? Let's try that "
                     "one again."),
    "unreadable": "I couldn't get a clear voiceprint from that one.",
    "no_encoder": ("The voice encoder isn't available on this machine, so I "
                   "can't learn a voice."),
    "invalid": "That sample could not be read as finite audio at a valid sample rate.",
}


@dataclass(frozen=True)
class Assessment:
    """Whether one utterance is good enough to learn from, and why not."""

    ok: bool
    code: str = "ok"
    reason: str = ""
    speech_seconds: float = 0.0
    level_dbfs: float = -120.0
    snr_db: float = 0.0
    clipped: float = 0.0


def _as_float(samples: Any) -> Any:
    """16-bit PCM bytes, int16 arrays, or float arrays -> float32 in [-1, 1]."""
    import numpy as np

    if isinstance(samples, (bytes, bytearray, memoryview)):
        usable = len(samples) - (len(samples) % 2)
        return np.frombuffer(bytes(samples[:usable]), dtype=np.int16).astype(
            np.float32) / 32768.0
    wav = np.asarray(samples)
    if wav.dtype.kind in "iu":
        return wav.astype(np.float32).flatten() / 32768.0
    wav = wav.astype(np.float32).flatten()
    if wav.size and float(np.abs(wav).max()) > 1.5:      # int16 as floats
        wav = wav / 32768.0
    return wav


def assess(samples: Any, sample_rate: int = SAMPLE_RATE) -> Assessment:
    """Check one utterance before learning from it. Cheap: numpy, no model.

    Measured on 20 ms frames: the loudest frames give the speech level, the
    quietest give the floor, and their difference estimates the signal to
    noise ratio. Order matters - a silent clip is reported as silent, not as
    too short, because "I didn't hear you" is the useful thing to say.
    """
    import numpy as np

    wav = _as_float(samples)
    rate = int(sample_rate or SAMPLE_RATE)
    if rate <= 0 or not np.isfinite(wav).all():
        return Assessment(False, "invalid", REASONS["invalid"])
    if wav.size == 0 or float(np.sqrt(np.mean(np.square(wav)))) < SILENT_RMS:
        return Assessment(False, "silent", REASONS["silent"])
    clipped = float(np.mean(np.abs(wav) >= CLIP_LEVEL))
    frame = max(1, int(0.02 * rate))
    count = wav.size // frame
    if count < 5:
        return Assessment(False, "too_short", REASONS["too_short"],
                          clipped=clipped)
    frames = wav[:count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9
    db = 20.0 * np.log10(rms)
    top = float(np.percentile(db, 95))
    active = db > top - 30.0
    speech_seconds = float(active.sum()) * frame / rate
    level = float(10.0 * np.log10(np.mean(np.square(rms[active])) + 1e-18))
    snr = max(0.0, min(60.0, top - float(np.percentile(db, 10))))
    details = dict(speech_seconds=round(speech_seconds, 2),
                   level_dbfs=round(level, 1), snr_db=round(snr, 1),
                   clipped=round(clipped, 5))
    if speech_seconds < MIN_SPEECH_SECONDS:
        return Assessment(False, "too_short", REASONS["too_short"], **details)
    if clipped > MAX_CLIPPED_FRACTION:
        return Assessment(False, "clipped", REASONS["clipped"], **details)
    if level < TOO_QUIET_DBFS:
        return Assessment(False, "too_quiet", REASONS["too_quiet"], **details)
    if snr < MIN_SNR_DB:
        return Assessment(False, "noisy", REASONS["noisy"], **details)
    return Assessment(True, **details)


@dataclass(frozen=True)
class Outcome:
    """What happened to one offered utterance."""

    accepted: bool
    code: str
    message: str
    good: int
    target: int


@dataclass(frozen=True)
class Progress:
    """A snapshot of a session, safe to read from any thread."""

    state: str
    good: int
    target: int
    minimum: int
    rejected: int
    phrase: str
    hint: str
    last: str
    seconds_left: int
    heard_anything: bool
    outcomes: int

    @property
    def listening(self) -> bool:
        return self.state == "listening"

    def prompt(self) -> str:
        """The next line to say, with its style hint."""
        if not self.listening:
            return ""
        return (f"{self.hint}: \"{self.phrase}\"" if self.hint
                else f"\"{self.phrase}\"")

    def describe(self) -> str:
        if self.state == "listening":
            head = f"{self.good} of {self.target} good samples so far."
            if self.last:
                head = f"{self.last} {head}"
            return f"{head} Next, say: {self.prompt()}"
        return self.last or f"Voice learning is {self.state}."


class EnrolmentSession:
    """One guided enrolment. Fed from a worker thread, read from anywhere."""

    def __init__(self, name: str, *, owner: bool = True, store: Any = None,
                 user_name: str = "", target: int = TARGET_SAMPLES,
                 minimum: int = MIN_SAMPLES, time_limit: float = SESSION_SECONDS,
                 origin: str = "voice", owner_only_after: bool = True,
                 clock: Callable[[], float] = time.monotonic,
                 on_event: Callable[[str], None] | None = None) -> None:
        self.name = str(name or "").strip() or "Owner"
        self.owner = bool(owner)
        self._store = store
        self.user_name = str(user_name or "").strip()
        self.target = max(1, int(target))
        self.minimum = max(1, min(int(minimum), self.target))
        self.time_limit = float(time_limit)
        if not math.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("voice learning time_limit must be finite and positive")
        self.origin = origin if origin in ("voice", "screen") else "voice"
        self.owner_only_after = bool(owner_only_after)
        self.clock = clock
        self.on_event = on_event
        self.started = clock()
        self.state = "listening"
        self._lock = threading.RLock()
        self._embeddings: list = []
        self._good = 0
        self._styles: list[str] = []
        self._rejected: dict[str, int] = {}
        self._outcomes = 0
        self._last = ""
        self._phrase = 0
        self.result = ""

    # -- reading -------------------------------------------------------------
    @property
    def store(self) -> Any:
        if self._store is None:
            from . import voiceprint

            self._store = voiceprint.store()
        return self._store

    def _line(self, index: int) -> tuple[str, str]:
        style, named, plain = PHRASES[index % len(PHRASES)]
        text = named.format(name=self.user_name) if self.user_name else plain
        return style, text

    def progress(self) -> Progress:
        self.check_deadline()
        with self._lock:
            style, text = self._line(self._phrase)
            left = max(0, int(self.time_limit - (self.clock() - self.started)))
            return Progress(state=self.state, good=self._good,
                            target=self.target, minimum=self.minimum,
                            rejected=sum(self._rejected.values()),
                            phrase=text, hint=STYLE_HINTS.get(style, ""),
                            last=self._last, seconds_left=left,
                            heard_anything=self._outcomes > 0,
                            outcomes=self._outcomes)

    def describe(self) -> str:
        return self.progress().describe()

    @property
    def listening(self) -> bool:
        self.check_deadline()
        with self._lock:
            return self.state == "listening"

    def check_deadline(self) -> None:
        """End an abandoned session even when no further utterance arrives.

        The capture and screen paths consult listening/progress regularly;
        neither needs to start a timer or retain raw audio to expire a session.
        """
        message = ""
        with self._lock:
            if self.state == "listening" and self.clock() - self.started >= self.time_limit:
                self._expire()
                if self.state == "failed":
                    message = self.result
        self._emit(message)

    def _clear_samples(self) -> None:
        """Release biometric samples after finishing, cancelling or forgetting."""
        for vector in self._embeddings:
            try:
                vector.fill(0)
            except (AttributeError, ValueError):
                pass
        self._embeddings.clear()
        self._styles.clear()

    # -- learning ------------------------------------------------------------
    def offer(self, samples: Any, sample_rate: int = SAMPLE_RATE,
              embedding: Any = None) -> Outcome:
        """Consider one finished utterance. Runs the encoder if no embedding
        is supplied, so call it off the GUI thread."""
        from . import voiceprint

        self.check_deadline()
        with self._lock:
            if self.state != "listening":
                return Outcome(False, "closed", self._last, len(self._embeddings),
                               self.target)
        check = assess(samples, sample_rate)
        if check.ok and embedding is None:
            if not voiceprint.available():
                check = Assessment(False, "no_encoder", REASONS["no_encoder"])
            else:
                embedding = voiceprint.embed(_as_float(samples), sample_rate)
        if check.ok and embedding is None:
            check = Assessment(False, "unreadable", REASONS["unreadable"])
        if check.ok:
            embedding = voiceprint._unit(embedding)
            if (embedding is None or (self._embeddings
                    and embedding.size != self._embeddings[0].size)):
                check = Assessment(False, "unreadable", REASONS["unreadable"])
        self.check_deadline()
        with self._lock:
            if self.state != "listening":
                return Outcome(False, "closed", self._last, len(self._embeddings),
                               self.target)
            self._outcomes += 1
            if check.ok and len(self._embeddings) >= CONSISTENCY_AFTER:
                centre = voiceprint._unit(
                    sum(voiceprint._unit(e) for e in self._embeddings))
                if voiceprint.similarity(embedding, centre) < CONSISTENCY_FLOOR:
                    check = Assessment(False, "inconsistent", REASONS["inconsistent"])
            if not check.ok:
                self._rejected[check.code] = self._rejected.get(check.code, 0) + 1
                self._last = check.reason
                outcome = Outcome(False, check.code, check.reason,
                                  len(self._embeddings), self.target)
                if check.code == "no_encoder":
                    self.state = "failed"
                    self.result = check.reason
            else:
                style, _ = self._line(self._phrase)
                self._embeddings.append(voiceprint._unit(embedding))
                self._styles.append(style)
                self._phrase += 1
                good = len(self._embeddings)
                self._good = good
                self._last = f"Got it - {good} of {self.target}."
                outcome = Outcome(True, "ok", self._last, good, self.target)
            done = self.state == "listening" and len(self._embeddings) >= self.target
        prompt = self.progress().prompt() if not done else ""
        self._emit(outcome.message + (f" Next, say: {prompt}" if prompt else ""))
        if done:
            self.finish()
        return outcome

    def finish(self) -> tuple[bool, str]:
        """Build and store the profile from what has been collected. Called by
        itself at the target; callable earlier once the minimum is in."""

        with self._lock:
            if self.state not in ("listening", "expired"):
                return self.state == "complete", self.result
            embeddings = list(self._embeddings)
            styles = list(self._styles)
            if len(embeddings) < self.minimum:
                self.state = "failed"
                self.result = (f"I only got {len(embeddings)} good samples and need "
                               f"at least {self.minimum}, so I haven't saved "
                               "anything. Say \"learn my voice\" to try again.")
                message = self.result
                ok = False
            else:
                kept, styles, dropped = _drop_outliers(embeddings, styles)
                if len(kept) < self.minimum:
                    ok, stored = False, (f"only {len(kept)} consistent samples remained; "
                                         f"at least {self.minimum} are needed")
                else:
                    try:
                        ok, stored = self.store.enrol_embeddings(
                            self.name, kept, owner=self.owner, styles=styles,
                            method="guided")
                    except Exception as exc:
                        ok, stored = False, str(exc)
                if not ok:
                    self.state = "failed"
                    self.result = f"I couldn't save the voiceprint: {stored}"
                else:
                    self.state = "complete"
                    self._good = len(kept)
                    self.result = self._completion(kept, dropped)
                message = self.result
            self._last = message
            self._clear_samples()
        self._emit(message)
        return ok, message

    def _completion(self, kept: list, dropped: int) -> str:
        from . import voiceprint

        record = {}
        try:
            record = self.store._load()["profiles"].get(self.name) or {}
        except Exception:
            record = {}
        agreement = _self_check(kept)
        parts = [f"Done - I know your voice now, from {len(kept)} samples."]
        if dropped:
            parts.append(f"I set aside {dropped} that didn't match the rest.")
        parts.append("I've kept a voiceprint - numbers that describe your voice, "
                     "never a recording - and \"forget my voice\" deletes it.")
        if self.owner and self.owner_only_after and agreement >= OWNER_ONLY_SELF_CHECK:
            try:
                ok, _ = self.store.set_only_owner(True)
            except Exception:
                ok = False
            if ok:
                parts.append("From now on I'll only answer your voice, so the TV "
                             "and other people won't set me off. Say \"answer "
                             "anyone\" if you want me to talk to everyone again.")
            else:
                parts.append("Your voiceprint is saved, but I couldn't switch the owner-only setting on.")
        elif self.owner:
            parts.append("I'm still answering anyone; say \"only listen to me\" "
                         "when you want me to ignore the TV and other people.")
        threshold = voiceprint.profile_threshold(record)
        parts.append(f"(Self-check {agreement:.0%}, threshold {threshold:.2f}.)")
        return " ".join(parts)

    def cancel(self) -> str:
        with self._lock:
            if self.state in ("listening", "expired"):
                self.state = "cancelled"
                self.result = "Voice learning cancelled - nothing was saved."
                self._last = self.result
            self._clear_samples()
            return self.result or self._last

    def _expire(self) -> None:
        """Out of time: keep what is there if it is enough."""
        if len(self._embeddings) >= self.minimum:
            self.state = "expired"
            threading.Thread(target=self.finish, name="orion-voice-enrol-finish",
                             daemon=True).start()
            return
        self.state = "failed"
        self.result = (f"Time's up - I only got {len(self._embeddings)} good "
                       f"samples, and I need {self.minimum}. Nothing was saved.")
        self._last = self.result
        self._clear_samples()

    def _emit(self, message: str) -> None:
        if self.on_event is None or not message:
            return
        try:
            self.on_event(message)
        except Exception:
            pass


def _drop_outliers(embeddings: list, styles: list) -> tuple[list, list, int]:
    """Set aside samples that sit far from the rest (the leave-one-out mean).
    Never drops more than a quarter: a profile is not rebuilt around noise."""
    import numpy as np

    from . import voiceprint

    if len(embeddings) < 6:
        return embeddings, styles, 0
    stack = np.stack(embeddings)
    total = stack.sum(axis=0)
    scores = [voiceprint.similarity(stack[i], voiceprint._unit(total - stack[i]))
              for i in range(len(stack))]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    drop = {i for i in order[:len(scores) // 4] if scores[i] < CONSISTENCY_FLOOR}
    kept = [e for i, e in enumerate(embeddings) if i not in drop]
    kept_styles = [s for i, s in enumerate(styles) if i not in drop]
    return kept, kept_styles, len(drop)


def _self_check(embeddings: list) -> float:
    """Share of samples a profile built WITHOUT them would recognise."""

    from . import voiceprint

    if len(embeddings) < 3:
        return 0.0
    hits = 0
    for i in range(len(embeddings)):
        rest = [e for j, e in enumerate(embeddings) if j != i]
        record = voiceprint.build_profile(rest)
        if record is None:
            continue
        limit = voiceprint.profile_threshold(record)
        hits += int(voiceprint.score_profile(embeddings[i], record) >= limit)
    return hits / float(len(embeddings) or 1)


# ── the session registry ─────────────────────────────────────────────────────

_SESSION_LOCK = threading.Lock()
_SESSION: EnrolmentSession | None = None


def start(name: str, **kwargs: Any) -> EnrolmentSession:
    """Begin a guided enrolment, replacing any session still running."""
    global _SESSION
    session = EnrolmentSession(name, **kwargs)
    with _SESSION_LOCK:
        previous, _SESSION = _SESSION, session
    if previous is not None and previous.listening:
        previous.cancel()
    return session


def current() -> EnrolmentSession | None:
    """The running session, or the last one to finish."""
    return _SESSION


def active() -> EnrolmentSession | None:
    session = _SESSION
    return session if session is not None and session.listening else None


def learning() -> bool:
    """Whether utterances are currently enrolment samples rather than turns."""
    return active() is not None


def holding_model() -> bool:
    """Hold model forwarding while the local guide collects its samples.

    The initial tool result reads the first prompt; on_event reads subsequent
    prompts. Sending those samples to Gemini too would trigger replies to the
    practice sentences rather than let the guide learn from them.
    """
    return active() is not None


def cancel() -> str:
    session = active()
    return session.cancel() if session is not None else "No voice learning is running."


def forget_my_voice(store: Any = None) -> tuple[bool, str]:
    """Delete the owner's voiceprint, switch both voice gates off, and drop
    every in-memory trace (turn history, cached embeddings, the guard's last
    verdict)."""
    from . import speaker_gate, voiceprint

    global _SESSION
    prints = store if store is not None else voiceprint.store()
    with _SESSION_LOCK:
        session, _SESSION = _SESSION, None
    if session is not None:
        session.cancel()
    turns().clear()
    speaker_gate.clear()
    voiceprint.clear_cache()
    owner = prints.owner
    if not owner or not prints.enrolled(owner):
        prints.set_only_owner(False)
        prints.set_guard_actions(False)
        return False, "I don't have a voiceprint of yours to forget."
    try:
        ok, message = prints.forget(owner)
    except Exception as exc:
        return False, f"The saved voiceprint could not be deleted: {exc}"
    if not ok:
        return False, message
    return True, ("Done - I've deleted your voiceprint. I'll answer anyone "
                  "again, and I won't use your voice to decide anything.")


# ── who is speaking, turn by turn ────────────────────────────────────────────

#: How long a recognised voice keeps deciding how ORION addresses the room.
CONTINUITY_S = 90.0

#: How long the last judged utterance decides whether a reply is allowed.
REPLY_WINDOW_S = 15.0

OWNER_NOTE = ("[voice] The person speaking now is the owner - recognised by "
              "their voice. You may use their name as usual.")
OTHER_NOTE = ("[voice] The person speaking now is NOT the owner - their voice "
              "does not match the owner's voiceprint{who}. Do not call them by "
              "the owner's name; address them neutrally.")


@dataclass(frozen=True)
class Turn:
    """Who one utterance belonged to."""

    kind: str
    name: str
    score: float
    confident: bool
    stage: str
    seconds: float
    at: float
    reason: str = ""

    @property
    def determined(self) -> bool:
        """Owner, or confidently someone else. Doubtful and unreadable turns
        neither confirm the owner nor revoke them."""
        return self.kind == "owner" or (
            self.kind in ("known", "unknown") and self.confident)

    @property
    def not_owner(self) -> bool:
        return self.kind in ("known", "unknown") and self.confident

    def describe(self) -> str:
        if self.kind == "owner":
            return f"That sounded like you (similarity {self.score:.2f})."
        if self.kind == "known":
            return f"That sounded like {self.name}, not you."
        if self.kind == "unknown" and self.confident:
            return ("That wasn't a voice I know - not yours "
                    f"(similarity {self.score:.2f}).")
        if self.kind == "uncertain":
            return f"I couldn't be sure (similarity {self.score:.2f}), so I assumed it was you."
        if self.kind == "nobody":
            return "Nobody's voice is enrolled yet."
        return "That was too short to tell whose voice it was."


class SpeakerTurns:
    """The recent run of speaker verdicts, and what follows from them."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 size: int = 24) -> None:
        self.clock = clock
        self._turns: deque[Turn] = deque(maxlen=size)
        self._lock = threading.Lock()
        self._noted = ""
        self._listeners: list[Callable[[str], None]] = []
        self._address = ""

    def record(self, verdict: Any, *, stage: str = "final", seconds: float = 0.0,
               now: float | None = None) -> Turn:
        kind = str(getattr(verdict, "category", "") or getattr(verdict, "kind", "")
                   or ("owner" if getattr(verdict, "is_owner", True) else "unknown"))
        name = str(getattr(verdict, "speaker", "") or "")
        score = float(getattr(verdict, "score", 0.0) or 0.0)
        if kind == "known":
            score = float(getattr(verdict, "match", 0.0) or score)
        turn = Turn(kind=kind, name=name, score=score,
                    confident=bool(getattr(verdict, "confident", False)),
                    stage=stage, seconds=float(seconds or 0.0),
                    at=self.clock() if now is None else now,
                    reason=str(getattr(verdict, "reason", "") or ""))
        self._append(turn)
        return turn

    def note_unjudged(self, seconds: float = 0.0, now: float | None = None) -> Turn:
        """An utterance too short to judge. It still ends a previous voice's
        hold on whether replies are allowed: never silence a short "stop"."""
        turn = Turn(kind="unreadable", name="", score=0.0, confident=False,
                    stage="final", seconds=float(seconds or 0.0),
                    at=self.clock() if now is None else now,
                    reason="too short to tell")
        self._append(turn)
        return turn

    def _append(self, turn: Turn) -> None:
        with self._lock:
            self._turns.append(turn)
            address = self._address_locked(turn.at)
            changed = address != self._address
            self._address = address
            listeners = list(self._listeners) if changed else []
        if listeners:
            note = self.listener_note(turn.at)
            for listener in listeners:
                try:
                    listener(note)
                except Exception:
                    pass

    def latest(self) -> Turn | None:
        with self._lock:
            return self._turns[-1] if self._turns else None

    def history(self) -> list[Turn]:
        with self._lock:
            return list(self._turns)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._address = ""
            self._noted = ""

    # -- how to address the room ---------------------------------------------
    def _address_locked(self, now: float) -> str:
        for turn in reversed(self._turns):
            if now - turn.at > CONTINUITY_S:
                break
            if turn.determined:
                return "owner" if turn.kind == "owner" else "other"
        return ""

    def address(self, now: float | None = None) -> str:
        """'owner', 'other', or '' when nothing recent was determined."""
        with self._lock:
            return self._address_locked(self.clock() if now is None else now)

    def address_by_name(self, now: float | None = None) -> bool:
        """Whether ORION may use the owner's name: only while the most recent
        determined voice was the owner's."""
        return self.address(now) == "owner"

    def listener_note(self, now: float | None = None) -> str:
        """A line for the model's context, or '' when there is nothing new to
        say. Doubt says nothing: the model keeps its last instruction."""
        address = self.address(now)
        if address == "owner":
            return OWNER_NOTE
        if address == "other":
            last = next((t for t in reversed(self.history()) if t.not_owner), None)
            who = (f" (it sounds like {last.name})"
                   if last is not None and last.kind == "known" and last.name else "")
            return OTHER_NOTE.format(who=who)
        return ""

    def note_if_changed(self, now: float | None = None) -> str:
        """The note, the first time it differs from the last one handed out."""
        note = self.listener_note(now)
        with self._lock:
            if not note or note == self._noted:
                return ""
            self._noted = note
            return note

    def add_listener(self, callback: Callable[[str], None]) -> None:
        """Called (on the recording thread) when how to address the room
        changes. Hop to your own thread inside it."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    # -- whether to answer at all --------------------------------------------
    def reply_allowed(self, now: float | None = None,
                      only_owner: bool | None = None) -> bool:
        """False only while "owner only" is on AND the latest utterance, heard
        within REPLY_WINDOW_S, was confidently not the owner's. Every doubt,
        every unjudged utterance and every stale verdict allows the reply."""
        if only_owner is None:
            try:
                from . import voiceprint

                only_owner = voiceprint.store().only_owner
            except Exception:
                only_owner = False
        if not only_owner:
            return True
        current = self.clock() if now is None else now
        latest = self.latest()
        if latest is None or current - latest.at > REPLY_WINDOW_S:
            return True
        return not latest.not_owner


_TURNS = SpeakerTurns()


def turns() -> SpeakerTurns:
    return _TURNS


# ── the per-utterance seam ───────────────────────────────────────────────────

def observe_utterance(samples: Any, sample_rate: int = SAMPLE_RATE, *,
                      stage: str = "final", embedding: Any = None,
                      profiles: dict | None = None, store: Any = None) -> Any:
    """The one call a capture path makes per finished utterance.

    Off the real-time and GUI threads (it runs the encoder). While a session
    is learning, a final utterance becomes an enrolment sample and None is
    returned. Otherwise the speaker is judged, recorded as this turn's, noted
    for the action guard when that is on, and the Verdict returned.

    *profiles* lets a caller with its own view of the profile file (the
    voice_speaker_id service) be judged against exactly that view.
    """
    from . import speaker_gate, voiceprint

    session = active()
    if session is not None:
        if stage != "final":
            return None
        session.offer(samples, sample_rate, embedding=embedding)
        return None
    prints = store if store is not None else voiceprint.store()
    try:
        data = prints._load()
    except Exception:
        data = {"profiles": {}, "owner": ""}
    pool = profiles if profiles is not None else data.get("profiles") or {}
    seconds = _seconds(samples, sample_rate)
    if not pool:
        return voiceprint.Verdict(True, reason="nobody is enrolled", kind="nobody")
    if embedding is None:
        embedding = (voiceprint.embed(_as_float(samples), sample_rate)
                     if voiceprint.available() else None)
    if embedding is None:
        turns().note_unjudged(seconds)
        return voiceprint.Verdict(True, reason="not enough speech to tell",
                                  kind="unreadable")
    owner = voiceprint.resolve_owner(pool, str(data.get("owner") or ""))
    verdict = voiceprint.decide(embedding, pool, owner, voiceprint._env_threshold())
    turns().record(verdict, stage=stage, seconds=seconds)
    try:
        if data.get("guard_actions"):
            speaker_gate.note_verdict(verdict)
    except Exception:
        pass
    return verdict


def note_short_utterance(seconds: float = 0.0) -> None:
    """For a capture path that also reports utterances too short to judge."""
    turns().note_unjudged(seconds)


def _seconds(samples: Any, sample_rate: int) -> float:
    try:
        if isinstance(samples, (bytes, bytearray, memoryview)):
            return len(samples) / 2.0 / float(sample_rate or SAMPLE_RATE)
        return float(len(samples)) / float(sample_rate or SAMPLE_RATE)
    except Exception:
        return 0.0


def summary(store: Any = None) -> str:
    """Plain words: whether the owner's voice is known, and what follows."""
    from . import voiceprint

    prints = store if store is not None else voiceprint.store()
    session = active()
    if session is not None:
        return "Learning your voice right now: " + session.describe()
    try:
        data = prints._load()
    except Exception:
        return "Voice recognition settings could not be read."
    owner = prints.owner
    record = data["profiles"].get(owner) if owner else None
    if not record:
        return "ORION doesn't know your voice yet."
    when = ""
    try:
        when = time.strftime(" on %d %b %Y", time.localtime(float(record["enrolled_at"])))
    except Exception:
        when = ""
    count = int(record.get("clips") or record.get("sample_count") or 0)
    gate = ("Answering only your voice." if data.get("only_owner")
            else "Answering anyone.")
    return f"ORION knows your voice ({count} samples{when}). {gate}"


def _isfinite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


__all__ = [
    "PHRASES",
    "Assessment",
    "EnrolmentSession",
    "Outcome",
    "Progress",
    "SpeakerTurns",
    "Turn",
    "active",
    "assess",
    "cancel",
    "current",
    "forget_my_voice",
    "holding_model",
    "learning",
    "note_short_utterance",
    "observe_utterance",
    "start",
    "summary",
    "turns",
]
