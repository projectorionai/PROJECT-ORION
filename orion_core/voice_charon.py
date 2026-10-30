"""ORION's own voice - Charon - when the Live channel is not there to speak.

Live speaks in Gemini's prebuilt "Charon" voice. Everything ORION said on any
other path (the Live channel down, quota, a reconnect, the offline loop) went
to the Windows SAPI voice, which sounds nothing like him: "I don't want to be
stuck with system default voice all the time" (owner, 10 Oct 2026).

Gemini's text-to-speech models offer the same prebuilt voice. MEASURED with
ORION's own key, "Good evening.": gemini-3.8-flash-lite-tts 1.1-1.3 s,
gemini-3.8-flash-tts 1.2-1.7 s, gemini-2.5-flash-preview-tts 1.3-1.7 s.

With no internet at all there is nobody to ask, so every short line Charon
speaks is kept (config/voice_cache/charon): greetings, goodbyes, acknowledgements
- the lines ORION says most - then come back in his voice offline, and only an
unusual sentence falls through to the local voice.

Voice continuity (Mark XXXV, W2.3). MEASURED on 11 Oct 2026 with a fake
transport: that cache alone gave an offline session 0 of 6 lines in Charon on
the owner's real (empty) cache, and still only 29 of 120 after forty lines had
been cached the old way - because ORION's own lines rotate (voice_variety), so
the phrasing he picks offline is rarely the one he happened to say online, and
a greeting or goodbye is one long utterance that never matches word for word.
So now:

  * ``CharonPrewarm`` fills the cache in the background with every line ORION
    composes himself (voice_variety.offline_lines), most important first:
    one request every few seconds, a daily cap, a cap on the cache's size,
    stopping for the day on a quota error, only once the app is ready, only
    while ORION is neither speaking nor hearing the owner, and only with a
    key and the internet. Warmed lines are never evicted by everyday ones.
  * Offline, an utterance is matched sentence by sentence (several sentences
    can be one cached line), and a sentence never cached word for word is
    said as a cached phrasing of the same intent ("See you later, sir." ->
    "Goodbye, sir.") rather than in another voice. ``offline_plan`` says
    which parts Charon can say; the synthesiser gives only the rest to the
    local voices (voice_piper, then SAPI).

Why Charon said nothing at all (16 Oct 2026). The owner's sessions of 12-16
Oct logged "0 of 164 ... in his voice" every day, and on 16 Oct "today's 120
requests are used" with an empty cache: every goodbye and every notice was
the SAPI voice. REPRODUCED with ORION's own key and code: every request was
refused before any audio, ``400 INVALID_ARGUMENT: Manually set deadline 8s is
too short. Minimum allowed deadline is 10s`` - the SDK turns the client's
``timeout`` (8000 ms, meant to fail fast offline) into an ``X-Server-Timeout``
header, and the service now refuses one under 10 s. That 400 was classed
"other", so the warm-up counted it against the day, put the line on the skip
list and moved on - 120 refusals, nothing cached - while the speaking path
stood down for two minutes after each one and handed the line to SAPI. Fixed
here: the deadline is sent explicitly (``SERVER_DEADLINE_S``), apart from the
client's own short per-operation timeout; a refusal is logged with its reason
the first time; only lines that worked count against the day; a run of
failures stops the warm-up instead of skipping every line.

Stand-downs (17 Oct 2026). Any failure but a quota stood the speaking path
down for two to fifteen minutes, and the system voice said every new line
meanwhile. Now only what will not fix itself soon - a spent quota, billing,
a refused or missing key - takes its whole cooldown at once. Anything else may
be one blip: seconds at first (``BLIP_COOLDOWN_S``), doubling with each failure
in a row up to the cooldown for its kind, and a line that works ends it. The
one CharonVoice is shared by the speaking path, the warm-up and render_pcm
(``shared_voice``), and a line already cached is said in his voice throughout.

Streaming (MEASURED 16 Oct, same key): a 10 s line streamed from
gemini-3.8-flash-lite-tts gave its first audio after 0.44 s and finished in
1.8 s, so the synthesiser now plays Charon as he arrives instead of after the
whole line. Several lines in one request (to stretch the quota) was tried and
rejected: the model read the instruction aloud (23.4 s of audio for ~8 s of
lines), and its gaps between lines (1.0-1.2 s) are not separable from its
pauses between sentences (0.83-0.86 s). One line per request.

Quota. Each model has its own quota, so a model whose quota is spent waits
until its reset (remembered across restarts) and the next model speaks
instead - still Charon. The warm-up only ever uses the first model, so it can
never spend the quota that real-time speech falls back on.

``render_pcm`` / ``render_wav`` are the public "say this in Charon's voice"
calls (optionally with a style prompt), shared with anything else that needs
his voice (the singing lane): same client, same quota state, same off-switch.

The cache format is unchanged: ``sha1("Charon|<lower-case, single-spaced
text>").pcm``, raw 24 kHz mono 16-bit PCM.

Same interface as ElevenLabsVoice (SAMPLE_RATE, available, stream_pcm), so the
SpeechSynthesiser streams it into ORION's one verified output stream.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import socket
import threading
import time
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SAMPLE_RATE = 24000
#: Fastest first; the next is tried when one is unavailable or its quota is spent.
MODELS = ("gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts")
#: Lines this short are cached for offline use.
CACHE_MAX_CHARS = 160
#: The cache never grows past this many lines (oldest first out)...
CACHE_MAX_FILES = 400
#: ...or this many bytes (an average line is ~100 KB: 2 s of 24 kHz PCM).
CACHE_MAX_BYTES = 64 * 1024 * 1024
#: A failure stands the backend down, so every line does not wait on it: at
#: most this long for a failure of no known kind...
COOLDOWN_S = 120.0
#: ...a dropped network at most this long: it is usually back within a minute.
NETWORK_COOLDOWN_S = 30.0
#: ...and a refused request (bad configuration) or a refused key this long.
CONFIG_COOLDOWN_S = 900.0
#: The first failure in a row stands Charon down only this long: one dropped
#: connection or one refused line must not hand his voice to the system voice
#: for minutes. Each further failure in a row doubles it, up to the cooldown
#: for its kind; a line that works ends the run. Quota, billing and a refused
#: key are not blips - they take their whole cooldown at once.
BLIP_COOLDOWN_S = 5.0
CHUNK_BYTES = 9600            # 0.2 s at 24 kHz mono 16-bit

#: Per network operation (the connect, each read of the stream): a dead
#: network fails fast and never holds ORION's voice.
CLIENT_TIMEOUT_MS = 8000
#: The deadline the SERVICE is given, sent explicitly so it is not derived
#: from CLIENT_TIMEOUT_MS: MEASURED 16 Oct 2026, a derived 8 s deadline is
#: refused outright ("Minimum allowed deadline is 10s"). Long enough for a
#: long line read at the pace it is played.
SERVER_DEADLINE_S = 300

#: Per-day quotas reset at midnight Pacific time: 08:00 UTC in winter (an hour
#: late in summer, never early).
QUOTA_RESET_UTC_HOUR = 8
#: A per-minute limit, when the refusal does not say how long to wait.
RATE_LIMIT_WAIT_S = 60.0

#: Offline, consecutive sentences up to this many are tried as one cached line
#: ("Good evening. Welcome back." is one line of the bank).
MAX_RUN = 4
#: The breath between two cached lines said one after the other.
SEGMENT_GAP_S = 0.18

#: Background pre-warm (CharonPrewarm). One request at most every 30 s, a daily
#: cap, and a size cap; the whole catalogue is ~200 lines, ~25 MB.
#:
#: Was 6 s - ten requests a minute against a free tier that allows THREE. The
#: owner's logs showed Charon 429s on five days (18-23 Oct) and the offline
#: bank stuck at 23 of 188 lines: the warm-up spent the minute's quota, was
#: refused, stood down, and the next real sentence found the quota gone too.
#: Two a minute leaves room for live speech in the same minute.
PREWARM_INTERVAL_S = 30.0
PREWARM_DAILY_CAP = 120
PREWARM_MAX_LINES = 300
PREWARM_MAX_BYTES = 48 * 1024 * 1024
#: Warmed lines may be longer than everyday ones (the offline refusal is ~210).
PREWARM_MAX_CHARS = 320
#: Never at startup: this long after the voice starts, and only once the app
#: reports itself ready (startup_budget's "command_ready").
PREWARM_START_DELAY_S = 90.0
#: ORION must have been quiet - not speaking, not hearing the owner - this long.
PREWARM_IDLE_S = 4.0
#: No internet, or the service struggling: look again this much later.
PREWARM_RETRY_S = 300.0
#: This many lines failing in a row is the service or the setup, not the
#: lines: the warm-up pauses (skipping nothing) instead of working through
#: the whole catalogue. 16 Oct 2026: 120 lines failed in a row, all skipped.
PREWARM_MAX_FAILURES = 3
#: The longest a failing warm-up waits before trying again.
PREWARM_MAX_BACKOFF_S = 6 * 3600.0
#: The day's state file. Version 1 counted failed requests against the day.
PREWARM_STATE_VERSION = 2


_ON = False


def enable() -> None:
    """Switch Charon on for a real run (orion.py). Off by default, so nothing
    built by the tests ever calls Gemini or writes the voice cache. The local
    neural voices that stand in for him offline (voice_kokoro, voice_piper)
    are switched on with him: they are part of the same voice, and equally
    never on in tests."""
    global _ON
    _ON = True
    try:
        from . import voice_kokoro
        voice_kokoro.enable()
    except Exception:
        pass
    try:
        from . import voice_piper
        voice_piper.enable()
    except Exception:
        pass


def enabled() -> bool:
    return _ON and os.getenv("ORION_CHARON_VOICE", "1").strip().lower() not in {"0", "false", "no", "off"}


def cache_dir() -> Path:
    from .constants import CONFIG_DIR
    return Path(CONFIG_DIR) / "voice_cache" / "charon"


def _cache_key(text: str, voice: str) -> str:
    norm = " ".join(str(text).lower().split())
    return hashlib.sha1(f"{voice}|{norm}".encode("utf-8")).hexdigest()


def spoken_form(text: str) -> str:
    """*text* exactly as it reaches the voice: what SpeechQueueManager.speak_text
    and SpeechSynthesiser.speak do to it first, so a warmed line and the same
    line said later share one cache key."""
    try:
        from .utils import normalise_for_speech
        text = normalise_for_speech(text)
    except Exception:
        text = str(text or "")
    return re.sub(r"[*_`#]+", "", text).strip()


def _resample(samples: Any, rate: int) -> Any:
    import numpy as np
    if rate == SAMPLE_RATE or not samples.size:
        return samples
    positions = np.linspace(0, samples.size - 1, int(samples.size * SAMPLE_RATE / rate))
    return np.interp(positions, np.arange(samples.size), samples).astype("<i2")


def to_pcm(data: bytes, mime: str) -> bytes:
    """24 kHz mono 16-bit PCM from what the model returned (raw L16 or WAV)."""
    if not data:
        return b""
    if "wav" not in (mime or "").lower() and not data[:4] == b"RIFF":
        rate = re.search(r"rate=(\d+)", mime or "")
        if rate and int(rate.group(1)) != SAMPLE_RATE:
            import numpy as np
            samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
            return _resample(samples, int(rate.group(1))).tobytes()
        return data
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        return b""
    import numpy as np
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype("<i2")
    return _resample(samples, rate).tobytes()


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Mono 16-bit PCM in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(rate))
        wav.writeframes(pcm or b"")
    return buffer.getvalue()


_SENTENCE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence in *text*, so unmatched parts are passed
    on exactly as written ("3.5" must not become "3. 5")."""
    spans = []
    position = 0
    for match in _SENTENCE.finditer(text or ""):
        start, end = position, match.end()
        position = end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    start, end = position, len(text or "")
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append((start, end))
    return spans


def _owner_honorific() -> str:
    """How the owner likes to be addressed (identity.json), for the farewells."""
    try:
        from . import identity
        data = json.loads(Path(identity.IDENTITY_PATH).read_text(encoding="utf-8"))
        return str(data.get("honorific") or "sir").strip()
    except Exception:
        return "sir"


# ── why a request failed ─────────────────────────────────────────────────────

class CharonUnavailable(RuntimeError):
    """Charon cannot say it now. ``kind`` is one of 'off' (not a real run -
    the tests - or ORION_CHARON_VOICE=0), 'no-key', 'quota', 'auth' (the key,
    its permissions or its billing refused), 'network', 'config' (the service
    refused the request itself) or 'other'."""

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind


def _status(exc: BaseException) -> int:
    for name in ("code", "status_code"):
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    match = re.match(r"\s*(\d{3})\b", str(exc))
    return int(match.group(1)) if match else 0


def failure_kind(exc: BaseException) -> str:
    """'quota', 'auth' (incl. billing not enabled), 'network' (incl. a
    struggling service), 'config' (the request itself refused - the same for
    every line) or 'other'."""
    if isinstance(exc, CharonUnavailable) and exc.kind in (
            "quota", "auth", "network", "config", "no-key", "other"):
        return exc.kind
    status = _status(exc)
    text = f"{type(exc).__name__} {exc}".lower()
    if status == 429 or any(t in text for t in ("resource_exhausted", "resource exhausted",
                                                "quota", "rate limit", "ratelimit")):
        return "quota"
    # Billing after quota: a spent quota's message also says "check your plan
    # and billing details", and that one waits for its reset like any quota.
    if status in (401, 402, 403) or any(t in text for t in ("permission_denied", "unauthenticated",
                                                            "api key not valid", "api_key_invalid",
                                                            "billing")):
        return "auth"
    if status >= 500 or isinstance(exc, (OSError, TimeoutError, ConnectionError)) or any(
            t in text for t in ("timeout", "timed out", "connect", "getaddrinfo", "network",
                                "unreachable", " 500", " 502", " 503", " 504",
                                "unavailable", "internal", "overloaded")):
        return "network"
    if status == 400 or any(t in text for t in ("invalid_argument", "failed_precondition")):
        return "config"
    return "other"


def _retired(exc: BaseException) -> bool:
    """The model is gone (or never existed for this key)."""
    return _status(exc) == 404 or "not found" in str(exc).lower()


def describe_failure(exc: BaseException, limit: int = 160) -> str:
    """One readable line: the service's own message where there is one."""
    text = str(exc)
    message = re.search(r"['\"]message['\"]\s*:\s*['\"](.+?)['\"]\s*[,}]", text)
    if message:
        text = f"{_status(exc) or ''} {message.group(1)}".strip()
    return " ".join(text.split())[:limit] or type(exc).__name__


def next_quota_reset(now: float | None = None) -> float:
    """The next per-day quota reset, as a wall-clock timestamp."""
    now = time.time() if now is None else float(now)
    reset = datetime.fromtimestamp(now, timezone.utc).replace(
        hour=QUOTA_RESET_UTC_HOUR, minute=0, second=0, microsecond=0)
    if reset.timestamp() <= now:
        reset += timedelta(days=1)
    return reset.timestamp()


def quota_wait(exc: BaseException, now: float | None = None) -> tuple[float, str]:
    """(when the model may be asked again - a wall-clock timestamp - and the
    quota's own description) from a quota refusal. A per-day quota waits for
    the reset even though the refusal also carries a short retryDelay."""
    now = time.time() if now is None else float(now)
    text = str(exc)
    flat = re.sub(r"[^a-z]", "", text.lower())
    quota_id = re.search(r"quotaId['\"]?\s*[:=]\s*['\"]([\w.\-/]+)", text)
    value = re.search(r"quotaValue['\"]?\s*[:=]\s*['\"]?(\d+)", text)
    label = ((quota_id.group(1) if quota_id else "quota")
             + (f" = {value.group(1)}" if value else ""))
    if "perday" in flat or "requestsperday" in flat:
        return next_quota_reset(now), label
    delay = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text)
    return now + (float(delay.group(1)) if delay else RATE_LIMIT_WAIT_S), label


def compose_prompt(text: str, style: str = "") -> str:
    """What the model is asked: the text, or "<style>: <text>" - Gemini TTS
    takes its direction in plain words ("Say warmly: Good evening.")."""
    style = " ".join(str(style or "").split()).rstrip(":").strip()
    return f"{style}: {text}" if style else str(text)


@dataclass(frozen=True)
class Segment:
    """One part of an utterance, offline: what was asked, and - when Charon
    can say it - his PCM and the cached line it came from."""
    text: str
    pcm: bytes | None = None
    said: str = ""


class ReadyVoice:
    """PCM already in hand (a cached line), streamed like any cloud voice."""

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, pcm: bytes) -> None:
        self._pcm = pcm

    def available(self) -> bool:
        return bool(self._pcm)

    def stream_pcm(self, text: str, emotion: str = "neutral", *,
                   on_chunk: Callable[[bytes], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> bytes | None:
        del text, emotion
        _play(self._pcm, on_chunk, should_stop)
        return self._pcm or None


def _play(pcm: bytes | None, on_chunk: Callable[[bytes], None] | None,
          should_stop: Callable[[], bool] | None) -> None:
    if on_chunk is None or not pcm:
        return
    for start in range(0, len(pcm), CHUNK_BYTES):
        if should_stop is not None and should_stop():
            break
        on_chunk(pcm[start:start + CHUNK_BYTES])


#: A stand-in for the network in tests: (model, contents, voice name, emit,
#: should_stop) -> True when the whole line arrived; raises as the service would.
Transport = Callable[[str, str, str, Callable[[bytes], None], Callable[[], bool] | None], bool]


class CharonVoice:
    """Gemini TTS in ORION's Live voice, with an offline cache of his lines."""

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, *, key_source: Callable[[], str] | None = None,
                 online: Callable[[], bool] | None = None,
                 synthesise: Callable[[str], bytes | None] | None = None,
                 transport: Transport | None = None,
                 cache: Path | None = None, honorific: str | None = None,
                 log: Callable[[str], Any] | None = None) -> None:
        if key_source is None:
            from .web_search_backends import gemini_key
            key_source = gemini_key
        self._key_source = key_source
        self._online = online or (lambda: True)
        self._synthesise_override = synthesise
        self._transport = transport
        self._cache = cache
        self._honorific = honorific
        self._log = log
        self._cooldown_until = 0.0
        #: Failures in a row since a line last worked (see BLIP_COOLDOWN_S).
        self._failures = 0
        self._retired: set[str] = set()
        self._blocked: dict[str, float] | None = None     # model -> wall clock
        #: True once a refusal has named a free-tier quota (10 lines a day a
        #: model): then nothing but his live speech may spend it (on_free_tier).
        self._free_tier: bool | None = None
        self._client: Any = None
        self._client_key = ""
        self._lock = threading.Lock()
        self._quota_lock = threading.Lock()
        self._pinned: frozenset[str] | None = None
        self._recent_variants: dict[str, list[str]] = {}
        self._plan_memo: tuple[str, list[Segment]] | None = None
        self._logged: set[str] = set()
        #: Where the last stream_pcm's audio came from: "cache", "gemini",
        #: "cache-variant" (offline, all of it from cached lines) or "".
        self.last_source = ""
        #: Why the last stream_pcm could not reach Gemini ("" when it did).
        self.last_reason = ""
        #: Seconds to the first audio of the last line Gemini spoke.
        self.first_audio_s: float | None = None
        #: The model that spoke last, and whether the whole line arrived.
        self.last_model = ""
        self.last_complete = False

    def attach(self, *, online: Callable[[], bool] | None = None,
               log: Callable[[str], Any] | None = None) -> None:
        """Give the voice the app's connectivity and log (SpeechSynthesiser)."""
        if online is not None:
            self._online = online
        if log is not None:
            self._log = log

    def _say_once(self, key: str, message: str) -> None:
        if key in self._logged or self._log is None:
            return
        self._logged.add(key)
        try:
            self._log(message)
        except Exception:
            pass

    @property
    def voice_name(self) -> str:
        # Charon unless the owner chose another prebuilt voice (voice_settings).
        # The offline cache is keyed by it, so a chosen voice never plays lines
        # cached in another.
        try:
            from .voice_settings import cloud_voice
            return cloud_voice()
        except Exception:
            from .constants import VOICE_PROFILE
            return VOICE_PROFILE.gemini_voice_name

    @property
    def honorific(self) -> str:
        if self._honorific is None:
            self._honorific = _owner_honorific()
        return self._honorific

    # ── the cache ────────────────────────────────────────────────────────────

    def _cache_dir(self) -> Path:
        return self._cache or cache_dir()

    def _cache_path(self, text: str) -> Path:
        return self._cache_dir() / f"{_cache_key(text, self.voice_name)}.pcm"

    def is_cached(self, text: str) -> bool:
        try:
            return self._cache_path(text).is_file()
        except OSError:
            return False

    def cached(self, text: str) -> bytes | None:
        path = self._cache_path(text)
        try:
            return path.read_bytes() if path.is_file() else None
        except OSError:
            return None

    def cache_stats(self) -> dict[str, int]:
        """{"files": n, "bytes": n} for the whole cache."""
        files = size = 0
        try:
            with os.scandir(self._cache_dir()) as entries:
                for entry in entries:
                    if entry.name.endswith(".pcm"):
                        files += 1
                        size += entry.stat().st_size
        except OSError:
            pass
        return {"files": files, "bytes": size}

    def _has_cache(self) -> bool:
        try:
            with os.scandir(self._cache_dir()) as entries:
                return any(entry.name.endswith(".pcm") for entry in entries)
        except OSError:
            return False

    def _pinned_keys(self) -> frozenset[str]:
        """Warmed lines: never evicted to make room for everyday ones."""
        if self._pinned is None:
            try:
                from .voice_variety import offline_lines
                self._pinned = frozenset(
                    _cache_key(spoken_form(line), self.voice_name)
                    for _intent, line in offline_lines(self.honorific))
            except Exception:
                self._pinned = frozenset()
        return self._pinned

    def _remember(self, text: str, pcm: bytes, *, limit: int = CACHE_MAX_CHARS) -> None:
        if len(text) > limit or not pcm:
            return
        path = self._cache_path(text)
        try:
            from .atomic_io import atomic_write_bytes

            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, pcm)
            self._evict(path.parent)
        except OSError:
            pass

    def _evict(self, directory: Path) -> None:
        """Oldest everyday lines out until both caps hold; warmed lines stay."""
        pinned = self._pinned_keys()
        entries = []
        total_files = total_bytes = 0
        for path in directory.glob("*.pcm"):
            try:
                stat = path.stat()
            except OSError:
                continue
            total_files += 1
            total_bytes += stat.st_size
            if path.stem not in pinned:
                entries.append((stat.st_mtime, stat.st_size, path))
        entries.sort()
        for _mtime, size, path in entries:
            if total_files <= CACHE_MAX_FILES and total_bytes <= CACHE_MAX_BYTES:
                break
            path.unlink(missing_ok=True)          # ORION's own cache, regenerable
            total_files -= 1
            total_bytes -= size

    # ── quota, per model, remembered across restarts ─────────────────────────

    def _quota_path(self) -> Path:
        return self._cache_dir() / "quota.json"

    def _blocks(self) -> dict[str, float]:
        """model -> when it may be asked again (wall clock), expired ones gone."""
        with self._quota_lock:
            if self._blocked is None:
                self._blocked = {}
                try:
                    saved = json.loads(self._quota_path().read_text(encoding="utf-8"))
                    for model, until in (saved.get("blocked") or {}).items():
                        if isinstance(model, str) and isinstance(until, (int, float)):
                            self._blocked[model] = float(until)
                    if self._free_tier is None and saved.get("free_tier") is True:
                        self._free_tier = True
                except (OSError, ValueError, AttributeError):
                    pass
            now = time.time()
            for model in [m for m, until in self._blocked.items() if until <= now]:
                del self._blocked[model]
            return dict(self._blocked)

    def _block(self, model: str, exc: BaseException) -> None:
        until, label = quota_wait(exc)
        self._blocks()
        free = "freetier" in f"{label} {exc}".replace("_", "").replace("-", "").lower()
        with self._quota_lock:
            if self._blocked is None:
                self._blocked = {}
            self._blocked[model] = until
            blocked = dict(self._blocked)
            if free:
                self._free_tier = True
            saved: dict[str, Any] = {"blocked": blocked}
            if self._free_tier:
                saved["free_tier"] = True
        try:
            from .atomic_io import atomic_write_text
            self._quota_path().parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._quota_path(), json.dumps(saved), encoding="utf-8")
        except Exception:
            pass
        when = datetime.fromtimestamp(until).strftime("%H:%M on %a %d %b")
        self._say_once(f"quota:{model}:{int(until)}",
                       f"VOICE: Charon's {model} quota is spent ({label}) - it is asked "
                       f"again after {when}; the next model speaks for him meanwhile.")

    def on_free_tier(self) -> bool:
        """Whether the key is on Google's free tier (10 lines a day a model),
        learnt from a refusal that said so and remembered across restarts. On
        it, the warm-up and the singer leave Charon to his live speech: 26 Oct
        2026 the warm-up spent the first model's day by 13:35 and the voice
        dropped to the offline one for the afternoon."""
        self._blocks()                           # loads the remembered state
        return bool(self._free_tier) and os.getenv(
            "ORION_CHARON_FREE_TIER_PREWARM", "").strip().lower() not in {"1", "true", "yes", "on"}

    def models_to_try(self) -> list[str]:
        """The models that may be asked now, fastest first."""
        blocked = self._blocks()
        return [m for m in MODELS if m not in self._retired and m not in blocked]

    def primary_model(self) -> str | None:
        """The one model the warm-up uses (the speaking path's first choice),
        or None while its quota is spent - the others are kept for speech."""
        live = [m for m in MODELS if m not in self._retired]
        if not live or live[0] in self._blocks():
            return None
        return live[0]

    # ── online ───────────────────────────────────────────────────────────────

    def available(self) -> bool:
        # enabled() first: in tests nothing else here is ever consulted.
        return enabled() and bool(self._synthesise_override or self._transport
                                  or self._key_source() or self._has_cache())

    def reachable(self) -> bool:
        """Whether Gemini is worth asking now (not standing down, not offline)."""
        if time.monotonic() < self._cooldown_until:
            return False
        return self.network_available()

    def network_available(self) -> bool:
        """Shared nonblocking connectivity/mode policy, independent of cooldown."""
        try:
            return bool(self._online())
        except Exception:
            return False

    def _client_for(self, key: str) -> Any:
        with self._lock:
            if self._client is None or self._client_key != key:
                from google import genai
                self._client = genai.Client(api_key=key, http_options={
                    "timeout": CLIENT_TIMEOUT_MS,
                    "headers": {"X-Server-Timeout": str(SERVER_DEADLINE_S)}})
                self._client_key = key
            return self._client

    def _stream_from(self, model: str, contents: str, voice_name: str,
                     emit: Callable[[bytes], None],
                     should_stop: Callable[[], bool] | None) -> bool:
        """One streamed request. True when the whole line arrived."""
        if self._transport is not None:
            return bool(self._transport(model, contents, voice_name, emit, should_stop))
        key = self._key_source()
        if not key:
            raise CharonUnavailable("no Gemini key is configured", "no-key")
        from google.genai import types
        client = self._client_for(key)
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name))))
        stream = client.models.generate_content_stream(model=model, contents=contents,
                                                       config=config)
        try:
            for reply in stream:
                if should_stop is not None and should_stop():
                    return False
                for part in _audio_parts(reply):
                    emit(to_pcm(part.data, str(getattr(part, "mime_type", "") or "")))
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        return True

    def _generate(self, contents: str, *, on_pcm: Callable[[bytes], None] | None = None,
                  should_stop: Callable[[], bool] | None = None,
                  models: Iterable[str] | None = None,
                  voice_name: str | None = None) -> bytes:
        """Charon's audio for *contents* (see _run_models)."""
        return self._run_models(contents, on_pcm=on_pcm, should_stop=should_stop,
                                models=models, voice_name=voice_name)[0]

    def _run_models(self, contents: str, *, on_pcm: Callable[[bytes], None] | None = None,
                    should_stop: Callable[[], bool] | None = None,
                    models: Iterable[str] | None = None,
                    voice_name: str | None = None) -> tuple[bytes, bool]:
        """(Charon's audio for *contents*, whether all of it came), handed to
        *on_pcm* as it arrives. Models are tried in order: a retired one (404)
        is dropped for good, one whose quota is spent waits for its reset.
        Raises when none could say it (CharonUnavailable 'quota' when every
        quota is spent); once audio has started, a failure returns what came
        (never restarted on another model). Returned, not kept on self: the
        speaking path and the warm-up call this from different threads."""
        if not self.network_available():
            raise CharonUnavailable("offline", "network")
        out = bytearray()
        carry = bytearray()

        def emit(pcm: bytes) -> None:
            carry.extend(pcm)
            even = len(carry) // 2 * 2
            if not even:
                return
            piece = bytes(carry[:even])
            del carry[:even]
            out.extend(piece)
            if on_pcm is not None and not (should_stop is not None and should_stop()):
                on_pcm(piece)

        voice_name = voice_name or self.voice_name
        spent: BaseException | None = None
        for model in list(models) if models is not None else self.models_to_try():
            try:
                complete = self._stream_from(model, contents, voice_name, emit, should_stop)
            except Exception as exc:
                if out:
                    self.last_model = model
                    self._say_once(f"partial:{describe_failure(exc, 60)}",
                                   f"VOICE: Charon stopped mid-line - {describe_failure(exc)}")
                    return bytes(out), False
                if _retired(exc):
                    self._retired.add(model)          # retired model: try the next
                    continue
                if failure_kind(exc) == "quota":
                    self._block(model, exc)
                    spent = exc
                    continue
                raise
            self.last_model = model
            if complete and out:
                self._worked()              # spoken, warmed or rendered
            return bytes(out), complete and not (should_stop is not None and should_stop())
        if spent is not None or not self.models_to_try():
            raise CharonUnavailable(
                "quota: every Charon model's quota is spent"
                + (f" ({describe_failure(spent, 80)})" if spent is not None else ""), "quota")
        raise CharonUnavailable("no Charon text-to-speech model is available", "config")

    def _synthesise(self, text: str) -> bytes | None:
        """The whole of *text* (no streaming): the legacy override, else Gemini."""
        if self._synthesise_override is not None:
            return self._synthesise_override(text)
        return self._generate(text) or None

    def prewarm_one(self, text: str) -> int:
        """Synthesise *text* into the cache (the pre-warm's one request), on
        the warm-up model only. Bytes kept; raises what the service raised.
        Never touches the speaking path's cooldown: a failed warm-up is the
        pre-warm's business."""
        cached = self.cached(text)
        if cached:
            return len(cached)
        if not self.network_available():
            raise CharonUnavailable("offline", "network")
        if self._synthesise_override is not None:
            pcm = self._synthesise_override(text)
        else:
            model = self.primary_model()
            if model is None:
                raise CharonUnavailable("quota: the warm-up model's quota is spent", "quota")
            pcm, complete = self._run_models(text, models=[model])
            if not complete:
                return 0
        if not pcm:
            return 0
        self._remember(text, pcm, limit=PREWARM_MAX_CHARS)
        return len(pcm) if self.is_cached(text) else 0

    def _worked(self) -> None:
        """A whole line came from the service: the run of failures and its
        stand-down are over."""
        self._failures = 0
        self._cooldown_until = 0.0

    def _stand_down(self, exc: BaseException) -> None:
        """After a failed line: how long not to ask, and say why - once.

        Quota, billing and a refused or missing key take their whole cooldown
        at once: they do not fix themselves within minutes. Anything else may
        be one blip (17 Oct 2026: one failure of any kind used to stand him
        down for 2-15 minutes, and the system voice said every new line
        meanwhile), so the first failure in a row stands him down for
        BLIP_COOLDOWN_S and each further one doubles that, up to the cooldown
        for its kind. Throughout, a line already in his cache is still said in
        his voice: stream_pcm looks there before anything else."""
        kind = failure_kind(exc)
        if kind == "quota":
            blocks = self._blocks()
            wait = max(RATE_LIMIT_WAIT_S, min(blocks.values()) - time.time()) if blocks \
                else COOLDOWN_S
        elif kind in ("auth", "no-key"):
            wait = CONFIG_COOLDOWN_S
        else:
            self._failures += 1
            ceiling = (NETWORK_COOLDOWN_S if kind == "network" else
                       CONFIG_COOLDOWN_S if kind == "config" else COOLDOWN_S)
            wait = min(ceiling, BLIP_COOLDOWN_S * 2 ** min(self._failures - 1, 16))
        self._cooldown_until = time.monotonic() + wait
        self.last_reason = f"{kind}: {describe_failure(exc, 100)}"
        if kind != "network":
            span = f"{wait / 60:.0f} min" if wait >= 90 else f"{wait:.0f} s"
            self._say_once(f"down:{kind}:{describe_failure(exc, 60)}",
                           f"VOICE: Charon (Gemini TTS) could not speak - {kind}: "
                           f"{describe_failure(exc)}. His cache speaks for him; he is "
                           f"asked again in {span}"
                           + (", longer while it keeps failing." if kind not in
                              ("quota", "auth", "no-key") else "."))

    # ── offline: cached lines, and cached phrasings of the same intent ───────

    def variant(self, text: str) -> tuple[str, bytes] | None:
        """A cached line meaning the same as *text* (one of ORION's own lines
        - voice_variety's catalogue - said another way), or None. The same
        line, however punctuated, comes first; then one not just used, so a
        partly warmed cache does not say one phrasing every time."""
        try:
            from .voice_variety import intent_of, loose, phrasings
        except Exception:
            return None
        intent = intent_of(text, self.honorific)
        if intent is None:
            return None
        wanted = loose(text)
        want_words = set(wanted.split())
        honorific = loose(self.honorific)
        recent = self._recent_variants.get(intent, [])
        options = []
        for order, line in enumerate(phrasings(intent, self.honorific)):
            said = spoken_form(line)
            if not self.is_cached(said):
                continue
            plain = loose(said)
            words = set(plain.split())
            # Never 'sir' where ORION did not say it (or drop it where he did).
            mismatch = bool(honorific) and ((f" {honorific} " in f" {wanted} ")
                                            != (f" {honorific} " in f" {plain} "))
            overlap = len(want_words & words) / max(1, len(want_words | words))
            options.append((plain != wanted, mismatch, said in recent, -overlap, order, said))
        if not options:
            return None
        said = min(options)[-1]
        pcm = self.cached(said)
        if pcm is None:
            return None
        self._recent_variants[intent] = (recent + [said])[-2:]
        return said, pcm

    def offline_plan(self, text: str) -> list[Segment]:
        """*text* as Charon can say it with no internet: each run of sentences
        that is a cached line (or a cached phrasing of the same intent) comes
        with its PCM; whatever is left is a Segment without PCM, kept exactly
        as written, for the local voice."""
        memo, self._plan_memo = self._plan_memo, None
        if memo is not None and memo[0] == text:
            return memo[1]                  # just planned by offline_pcm
        spans = sentence_spans(text)
        segments: list[Segment] = []
        gap: tuple[int, int] | None = None
        i = 0
        while i < len(spans):
            hit: tuple[int, str, bytes, str] | None = None
            for j in range(min(len(spans), i + MAX_RUN), i, -1):
                piece = text[spans[i][0]:spans[j - 1][1]]
                pcm = self.cached(piece)
                if pcm is not None:
                    hit = (j, piece, pcm, piece)
                    break
                found = self.variant(piece)
                if found is not None:
                    hit = (j, piece, found[1], found[0])
                    break
            if hit is None:
                gap = (gap[0] if gap else spans[i][0], spans[i][1])
                i += 1
                continue
            if gap is not None:
                segments.append(Segment(text[gap[0]:gap[1]]))
                gap = None
            j, piece, pcm, said = hit
            segments.append(Segment(piece, pcm, said))
            i = j
        if gap is not None:
            segments.append(Segment(text[gap[0]:gap[1]]))
        return segments

    def offline_pcm(self, text: str) -> bytes | None:
        """All of *text* from the cache, or None if any part is missing (the
        plan is kept for the synthesiser's next offline_plan of this text)."""
        segments = self.offline_plan(text)
        if not segments or any(segment.pcm is None for segment in segments):
            self._plan_memo = (text, segments)
            return None
        gap = b"\x00\x00" * int(SEGMENT_GAP_S * SAMPLE_RATE)
        return gap.join(segment.pcm or b"" for segment in segments)

    def stream_pcm(self, text: str, emotion: str = "neutral", *,
                   on_chunk: Callable[[bytes], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> bytes | None:
        """Speak *text*: from the cache, else streamed from Gemini as it is
        generated, else - offline - from cached lines meaning the same. None =
        Charon cannot say all of it (offline_plan says which parts he can)."""
        del emotion                                   # Charon's delivery is the model's
        self.last_source = ""
        self.last_reason = ""
        pcm = self.cached(text)
        if pcm is not None:
            self.last_source = "cache"
            _play(pcm, on_chunk, should_stop)
            return pcm
        if not self.reachable():
            self.last_reason = ("standing down after a failure" if
                                time.monotonic() < self._cooldown_until else "offline")
        else:
            started = time.perf_counter()
            first: list[float] = []

            def arrived(piece: bytes) -> None:
                if not first:
                    first.append(time.perf_counter() - started)
                if on_chunk is not None:
                    on_chunk(piece)

            complete = False
            try:
                if self._synthesise_override is not None:
                    pcm = self._synthesise_override(text) or None   # whole line, then played
                    complete = bool(pcm)
                    if pcm:
                        self._worked()
                        first.append(time.perf_counter() - started)
                        _play(pcm, on_chunk, should_stop)
                else:
                    audio, complete = self._run_models(text, on_pcm=arrived,
                                                       should_stop=should_stop)
                    pcm = audio or None
            except Exception as exc:
                self._stand_down(exc)
                pcm = None
            if pcm:
                self.first_audio_s = first[0] if first else None
                self.last_source = "gemini"
                self.last_complete = complete
                if complete:
                    self._remember(text, pcm)
                return pcm
        pcm = self.offline_pcm(text)
        if not pcm:
            return None
        self.last_source = "cache-variant"
        _play(pcm, on_chunk, should_stop)
        return pcm


def _audio_parts(reply: Any) -> list[Any]:
    """The inline audio parts of one streamed reply (none on a text-only one)."""
    try:
        parts = reply.candidates[0].content.parts or []
    except Exception:
        return []
    return [part.inline_data for part in parts
            if getattr(part, "inline_data", None) is not None and part.inline_data.data]


# ── his voice for anything else that needs it ────────────────────────────────

_SHARED: CharonVoice | None = None
_SHARED_LOCK = threading.Lock()


def shared_voice() -> CharonVoice:
    """The one CharonVoice of this process: the speaking path, the warm-up and
    render_pcm share its client, its quota state and its stand-downs."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = CharonVoice()
        return _SHARED


def render_pcm(text: str, *, style: str = "", voice_name: str | None = None,
               on_chunk: Callable[[bytes], None] | None = None,
               should_stop: Callable[[], bool] | None = None) -> bytes:
    """*text* in Charon's voice (the Live channel's voice): 24 kHz mono 16-bit
    PCM, handed to *on_chunk* as it arrives when given (first audio ~0.4 s).

    *style* is a direction in plain words, put before the text the way Gemini
    TTS takes it ("Sing slowly and warmly" -> "Sing slowly and warmly: <text>").
    *voice_name* picks another prebuilt Gemini voice (default: Charon).

    Blocking network work: call it off the GUI/event-loop thread. Shares the
    speaking path's client and per-model quota state. Never caches a styled
    line; an unstyled Charon line already in his cache costs no request.
    Raises CharonUnavailable (``.kind``: 'off' - always in tests, Charon is
    switched on only by orion.py - 'no-key', 'quota', 'auth', 'network',
    'config', 'other')."""
    if not enabled():
        raise CharonUnavailable("Charon is off (orion.py switches him on for a real run)", "off")
    voice = shared_voice()
    if not style and voice_name in (None, voice.voice_name):
        cached = voice.cached(text)
        if cached is not None:
            _play(cached, on_chunk, should_stop)
            return cached
    try:
        pcm = voice._generate(compose_prompt(text, style), on_pcm=on_chunk,
                              should_stop=should_stop, voice_name=voice_name)
    except CharonUnavailable:
        raise
    except Exception as exc:
        raise CharonUnavailable(describe_failure(exc), failure_kind(exc)) from exc
    if not pcm:
        raise CharonUnavailable("no audio came back", "other")
    return pcm


def render_wav(text: str, *, style: str = "", voice_name: str | None = None) -> bytes:
    """render_pcm in a WAV container (24 kHz mono 16-bit)."""
    return pcm_to_wav(render_pcm(text, style=style, voice_name=voice_name))


# ── keeping the cache warm ───────────────────────────────────────────────────

_PROBE_HOSTS = (("1.1.1.1", 443), ("8.8.8.8", 443))


def network_up(timeout: float = 1.5) -> bool:
    """A raw TCP connect, no DNS (as ConnectivityMonitor does). Blocking:
    for the pre-warm's own thread only."""
    for host in _PROBE_HOSTS:
        try:
            with socket.create_connection(host, timeout=timeout):
                return True
        except OSError:
            continue
    return False


def app_ready() -> bool:
    """True once the app has reported itself ready for commands."""
    try:
        from . import startup_budget
        budget = startup_budget.current()
        return budget is not None and "command_ready" in getattr(budget, "milestones", {})
    except Exception:
        return False


class CharonPrewarm:
    """Fills Charon's cache with every line ORION says himself, in the
    background, so offline he still says them in his own voice.

    Rate-limited (one request every ``interval_s``), capped per day
    (``daily_cap`` lines that worked, remembered across restarts - a failed
    request is not counted), bounded in lines and in the cache's total bytes,
    skipping whatever is already cached, stopping for the day on a quota or
    key error, pausing (skipping nothing) when several lines in a row fail.
    Only the first model is used, so real-time speech keeps the others. Each
    request waits until ORION has been quiet - not speaking, not hearing the
    owner - for ``idle_s``; nothing starts until ``start_delay_s`` has passed
    and the app is ready; nothing is asked without a key and the internet.
    Runs on its own daemon thread, never on the Qt/asyncio thread; ``fill()``
    is the same work, synchronously."""

    def __init__(self, voice: CharonVoice, *,
                 lines: Callable[[], list[str]] | None = None,
                 busy: Callable[[], bool] | None = None,
                 ready: Callable[[], bool] | None = None,
                 online: Callable[[], bool] | None = None,
                 log: Callable[[str], Any] | None = None,
                 interval_s: float = PREWARM_INTERVAL_S,
                 daily_cap: int = PREWARM_DAILY_CAP,
                 max_lines: int = PREWARM_MAX_LINES,
                 max_bytes: int = PREWARM_MAX_BYTES,
                 start_delay_s: float = PREWARM_START_DELAY_S,
                 idle_s: float = PREWARM_IDLE_S,
                 retry_s: float = PREWARM_RETRY_S,
                 poll_s: float = 0.25,
                 state_path: Path | None = None,
                 today: Callable[[], str] | None = None) -> None:
        self.voice = voice
        self._lines_source = lines
        self._busy = busy or (lambda: False)
        self._ready = ready or app_ready
        self._online = online or network_up
        self._log = log or (lambda message: None)
        self.interval_s = float(interval_s)
        self.daily_cap = int(daily_cap)
        self.max_lines = int(max_lines)
        self.max_bytes = int(max_bytes)
        self.start_delay_s = float(start_delay_s)
        self.idle_s = float(idle_s)
        self.retry_s = float(retry_s)
        self.poll_s = float(poll_s)
        self._state_path = state_path
        self._today = today or (lambda: date.today().isoformat())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_at = 0.0
        self._backoff_s = 0.0
        #: The outcome of the last fill(), for the log and the tests.
        self.outcome = ""

    # ── the lines ────────────────────────────────────────────────────────────

    def lines(self) -> list[str]:
        """What to keep warm, most important first, as the voice will hear it."""
        if self._lines_source is not None:
            raw = list(self._lines_source())
        else:
            from .voice_variety import offline_lines
            raw = [line for _intent, line in offline_lines(self.voice.honorific)]
        seen: set[str] = set()
        kept: list[str] = []
        for line in raw:
            said = spoken_form(line)
            if said and len(said) <= PREWARM_MAX_CHARS and said.lower() not in seen:
                seen.add(said.lower())
                kept.append(said)
        return kept[:self.max_lines]

    # ── the day's budget, remembered across restarts ─────────────────────────

    def _path(self) -> Path:
        return self._state_path or (self.voice._cache_dir() / "prewarm.json")

    def _load_state(self) -> dict[str, Any]:
        today = self._today()
        try:
            state = json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if not isinstance(state, dict) or state.get("day") != today:
            state = {"v": PREWARM_STATE_VERSION, "day": today, "requests": 0,
                     "blocked": "", "skip": []}
        requests = state.get("requests", 0)
        skipped = state.get("skip")
        if not isinstance(requests, int) or isinstance(requests, bool) or requests < 0:
            requests = self.daily_cap    # malformed counters cannot reset today's budget
        elif state.get("v") != PREWARM_STATE_VERSION:
            # Version 1 counted every attempt and skipped every failed line:
            # on 16 Oct 2026, 120 refused requests and nothing cached. What the
            # cache holds bounds what really worked; the skips were the bug's.
            requests = min(requests, self.voice.cache_stats()["files"])
            skipped = []
        state["v"] = PREWARM_STATE_VERSION
        state["requests"] = requests
        state["skip"] = ([line for line in skipped if isinstance(line, str)]
                         if isinstance(skipped, list) else [])
        return state

    def _save_state(self, state: dict[str, Any]) -> bool:
        try:
            from .atomic_io import atomic_write_text
            self._path().parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._path(), json.dumps(state), encoding="utf-8")
            return True
        except Exception:
            return False

    # ── waiting ──────────────────────────────────────────────────────────────

    def _wait(self, seconds: float) -> bool:
        """Sleep; False if asked to stop meanwhile."""
        return not self._stop.wait(max(0.0, seconds))

    def _wait_for_quiet(self) -> bool:
        quiet_since: float | None = None
        while not self._stop.is_set():
            try:
                busy = bool(self._busy())
            except Exception:
                busy = True
            now = time.monotonic()
            if busy:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            if quiet_since is not None and now - quiet_since >= self.idle_s:
                return True
            if not self._wait(self.poll_s):
                return False
        return False

    def _progress(self, cached: int, total: int, note: str = "") -> None:
        size = self.voice.cache_stats()["bytes"] / (1024 * 1024)
        self._log(f"VOICE: Charon offline cache - {cached} of {total} of ORION's own lines "
                  f"in his voice ({size:.1f} MB){note}.")

    # ── the work ─────────────────────────────────────────────────────────────

    def fill(self) -> str:
        """Warm whatever is missing, within today's budget. Returns why it
        stopped: 'complete', 'daily-cap', 'quota', 'skipped' (only lines that
        failed today are left), 'failing' (several lines in a row failed:
        nothing skipped, try later), 'bytes', 'offline', 'no-key' or 'stopped'."""
        self.outcome = self._fill()
        return self.outcome

    def _fill(self) -> str:
        state = self._load_state()
        lines = self.lines()
        skip = set(state["skip"])
        todo = [line for line in lines
                if not self.voice.is_cached(line) and line.lower() not in skip]
        cached = sum(1 for line in lines if self.voice.is_cached(line))
        if not todo:
            if cached == len(lines):
                self._progress(cached, len(lines), " - complete")
                return "complete"
            return "skipped"                # lines that failed today: tomorrow
        if state.get("blocked"):
            return "quota"
        free_tier = getattr(self.voice, "on_free_tier", None)
        if callable(free_tier) and free_tier():
            # Ten lines a day a model: warming the bank would take three weeks
            # and spend the day's first model before lunch. The lines already
            # cached keep working; a paid key warms the rest.
            self._progress(cached, len(lines), " - paused: on the free tier, today's "
                           "Charon lines are kept for his live speech")
            return "free-tier"
        self._progress(cached, len(lines),
                       f"; warming {len(todo)} more in the background, one every "
                       f"{self.interval_s:g} s")
        failed: list[str] = []              # the current run of failures
        reported = False
        for line in todo:
            if self._stop.is_set():
                return "stopped"
            if state["requests"] >= self.daily_cap:
                self._progress(cached, len(lines), f" - today's {self.daily_cap} lines "
                               "are warmed; the rest tomorrow")
                return "daily-cap"
            if self.voice.cache_stats()["bytes"] >= self.max_bytes:
                self._progress(cached, len(lines), " - size cap reached")
                return "bytes"
            try:
                has_key = bool(self.voice._synthesise_override or self.voice._transport
                               or self.voice._key_source())
            except Exception:
                has_key = False
            if not has_key:
                return "no-key"
            if not self._wait(self._next_at - time.monotonic()):
                return "stopped"
            if not self._wait_for_quiet():
                return "stopped"
            try:
                online = bool(self._online())
            except Exception:
                online = False
            if not online:
                return "offline"
            self._next_at = time.monotonic() + self.interval_s
            # Reserved before asking, so a crash mid-request cannot exceed the
            # cap; given back below when the line did not come.
            state["requests"] += 1
            if not self._save_state(state):
                self._log("VOICE: Charon cache warm-up stopped: the daily budget could not be saved.")
                return "state-unavailable"
            error: BaseException | None = None
            try:
                kept = self.voice.prewarm_one(line)
            except Exception as exc:
                error, kept = exc, 0
            if not kept:
                state["requests"] = max(0, state["requests"] - 1)
                self._save_state(state)
            if error is not None:
                kind = failure_kind(error)
                if not reported:
                    reported = True
                    self._log(f"VOICE: Charon warm-up request refused - {kind}: "
                              f"{describe_failure(error)}")
                if kind in ("quota", "auth"):
                    state["blocked"] = kind
                    self._save_state(state)
                    self._progress(cached, len(lines),
                                   f" - stopped for today ({'quota reached' if kind == 'quota' else 'the key was refused'})")
                    return "quota"
                if kind == "network":
                    return "offline"
            if not kept:
                failed.append(line)
                if len(failed) >= PREWARM_MAX_FAILURES:
                    self._progress(cached, len(lines),
                                   f" - paused: {len(failed)} lines in a row failed, so "
                                   "it is the service, not the lines; nothing skipped")
                    return "failing"
                continue
            if failed:
                # Something works, so the lines before it were the problem.
                skip.update(item.lower() for item in failed)
                state["skip"] = sorted(skip)
                self._save_state(state)
                failed.clear()
            self._backoff_s = 0.0
            cached += 1
            if cached % 10 == 0:
                self._progress(cached, len(lines))
        if failed:
            skip.update(item.lower() for item in failed)
            state["skip"] = sorted(skip)
            self._save_state(state)
        done = sum(1 for line in lines if self.voice.is_cached(line))
        self._progress(done, len(lines), " - complete" if done == len(lines) else "")
        return "complete" if done == len(lines) else "skipped"

    # ── the thread ───────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="orion-charon-prewarm",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        started = time.monotonic()
        while not self._stop.is_set():
            try:
                ready = time.monotonic() - started >= self.start_delay_s and bool(self._ready())
            except Exception:
                ready = False
            if ready:
                break
            if not self._wait(1.0):
                return
        while not self._stop.is_set():
            try:
                outcome = self.fill()
            except Exception as exc:          # never take the voice down with it
                self._log(f"VOICE: Charon offline cache warm-up fault - {str(exc)[:100]}")
                return
            if outcome in ("complete", "bytes", "no-key", "stopped", "state-unavailable"):
                return
            if outcome == "offline":
                if not self._wait(self.retry_s):
                    return
                continue
            if outcome == "failing":
                self._backoff_s = min(PREWARM_MAX_BACKOFF_S,
                                      max(self.retry_s, self._backoff_s * 2))
                if not self._wait(self._backoff_s):
                    return
                continue
            day = self._today()       # daily cap, quota or failed lines: tomorrow
            while self._today() == day:
                if not self._wait(600.0):
                    return


__all__ = [
    "MODELS",
    "SAMPLE_RATE",
    "CharonPrewarm",
    "CharonUnavailable",
    "CharonVoice",
    "ReadyVoice",
    "Segment",
    "app_ready",
    "cache_dir",
    "compose_prompt",
    "describe_failure",
    "enable",
    "enabled",
    "failure_kind",
    "network_up",
    "next_quota_reset",
    "pcm_to_wav",
    "quota_wait",
    "render_pcm",
    "render_wav",
    "sentence_spans",
    "shared_voice",
    "spoken_form",
    "to_pcm",
]
