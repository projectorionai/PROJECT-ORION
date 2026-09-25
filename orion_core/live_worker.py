"""
GenAILiveWorker — the realtime session brain.

Native audio uses Gemini Live.  If that channel is unavailable because of
quota, rate limits, authentication, network failure, or missing tokens,
manual text turns and file reviews are routed through configured
OpenAI-compatible providers, and the offline voice loop (local STT →
provider router → local voice) keeps ORION conversational.

Mark VIII voice integration: every byte ORION speaks flows through the
SpeechQueueManager, which is also the single authority the microphone gate
consults — so ORION always finishes speaking before listening resumes, on
both the native and the fallback channel.  The Gemini voice is locked to the
frozen VOICE_PROFILE and can never be switched by configuration drift.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
import traceback
from datetime import datetime
from typing import Any

# google.genai costs ~1.28 s to import and is only ever touched inside
# methods (see lazy_import). Importing it eagerly delayed ORION's first
# paint by that much, because app.py imports this module at startup.
from .lazy_import import LazyModule

genai = LazyModule("google.genai")
types = LazyModule("google.genai.types")

from .audio import (
    AudioPlaybackThread,
    LocalSpeechRecogniser,
    MicrophoneEngine,
    SileroVADGatekeeper,
    SpeechQueueManager,
    SpeechSynthesiser,
)
from .audio_state import AudioStateMachine
from .bus import OrionBus
from .concurrency import describe_plan, plan_batches
from .connection_state import ConnectionState
from .constants import (
    CANCEL_WORDS,
    INPUT_SWITCH_WORDS,
    LIVE_MODEL,
    LIVE_MODEL_FALLBACKS,
    MIC_QUEUE_LIMIT,
    OUTPUT_SWITCH_WORDS,
    PAUSE_WORDS,
    PRESENCE_GRACE_SECONDS,
    PRESENCE_PROMPTS,
    RESUME_WORDS,
    SEND_SAMPLE_RATE,
    STARTUP_GREETINGS,
    VOICE_PROFILE,
    WAKE_WINDOW_SECONDS,
    WAKE_WORDS,
)
from . import background
from .dispatcher import TOOL_DECLARATIONS, OrionDispatcher
from .request_trace import TRACES, Stage
from .memory import MemoryAgent
from .providers import AIProviderProfile, OrionProviderSettings, ProviderRouter
from .security import SecuritySanitiser, SecurityViolation
from .utils import clean_transcript, correct_identity, first_line
from .voice_interrupt import VoiceInterruptManager


def _extract_genai_error_context(exc: Exception) -> tuple[str, str]:
    """
    Extract meaningful error context from google-genai exceptions.
    Returns (error_summary, error_type) for better logging.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc).strip()
    
    # Try to extract useful info from exception attributes
    if hasattr(exc, "code") or hasattr(exc, "status"):
        code = getattr(exc, "code", getattr(exc, "status", "?"))
        if code == 1011:
            return (f"WebSocket error 1011 (server error) — {exc_str or 'no message'}", exc_type)
        return (f"Error code {code} — {exc_str or 'no message'}", exc_type)
    
    if hasattr(exc, "message"):
        return (str(exc.message), exc_type)
    
    # Check for common error patterns
    if "api_key" in exc_str.lower() or "authentication" in exc_str.lower():
        return ("Authentication failed — check your Gemini API key in config/api_keys.json", exc_type)
    if "quota" in exc_str.lower():
        return ("Quota exceeded — rate limited by Gemini API", exc_type)
    if "invalid" in exc_str.lower() or "unsupported" in exc_str.lower():
        return (f"Invalid request — {exc_str[:100]}", exc_type)
    if "timeout" in exc_str.lower() or "deadline" in exc_str.lower():
        return ("Connection timeout — network may be slow or unreachable", exc_type)
    
    # Fallback: use first line or exception type
    summary = exc_str.split("\n")[0] if exc_str else exc_type
    return (summary[:160], exc_type)


#: WebSocket close codes that say the CONNECTION ended, not that the provider
#: is broken: normal close, going away, abnormal close (a network drop or a
#: missed keepalive, which the SDK reports as 1006), server error, service
#: restart, try again later, bad gateway. All of them used to fall through to
#: router.mark_failure, which stood Live down for 45 s per drop — ORION
#: "cutting out" until he was restarted.
_TRANSIENT_CLOSE_CODES = frozenset({1000, 1001, 1006, 1011, 1012, 1013, 1014})
#: The server refused something WE sent mid-session (an oversized or
#: malformed frame). A fresh connection fixes it; the provider is fine.
_OUR_PAYLOAD_CLOSE_CODES = frozenset({1007, 1009})


def _close_code(exc: BaseException) -> int | None:
    """The WebSocket close code a google-genai APIError carries, if any."""
    code = getattr(exc, "code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    return code if 1000 <= code <= 4999 else None


#: Every Live connect and drop is written here, because the in-app log is
#: never saved and "he cut out" otherwise leaves no evidence at all.
_LIVE_DIAG_LIMIT = 1_000_000
from .constants import CONFIG_DIR as _CONFIG_DIR  # noqa: E402
LIVE_DIAG_PATH = _CONFIG_DIR / "diagnostics" / "live_sessions.jsonl"


def _live_diag(event: str, **fields: Any) -> None:
    """Append one Live lifecycle event to diagnostics/live_sessions.jsonl."""
    try:
        import json

        path = LIVE_DIAG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _LIVE_DIAG_LIMIT:
            path.replace(path.with_suffix(".jsonl.1"))
        record = {"at": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


#: A control phrase is SHORT — see GenAILiveWorker.CONTROL_PHRASE_MAX_WORDS.
#: A module-level function, not a method, so it works for any worker-shaped
#: object including the stand-ins the standby tests drive the real handlers
#: with.
def _is_control_length(text: str, max_words: int = 12) -> bool:
    """Whether *text* is short enough to be a control phrase at all."""
    return len(str(text or "").split()) <= max_words


def _consume_outcome(task: "asyncio.Task[Any]") -> None:
    """Mark a finished task's exception as read, so asyncio does not report it
    as never retrieved. Usable directly or as a done-callback."""
    if not task.cancelled():
        task.exception()


def _audio_recovery(worker: Any) -> Any:
    """The standby/wake owner for *worker*, built on demand.

    Normally constructed in ``__init__``.  Resolved lazily here as well
    because standby and wake are the two commands that MUST work even if the
    worker was assembled through an unusual path — being unable to wake ORION
    back up is the one failure the user has no way out of.

    A module-level function rather than a method so it works for any
    worker-shaped object, including the lightweight stand-ins the standby
    tests drive the real command handler with.
    """
    recovery = getattr(worker, "recovery", None)
    if recovery is None:
        from .audio_recovery import AudioRecovery
        recovery = AudioRecovery(worker)
        try:
            worker.recovery = recovery
        except AttributeError:      # a frozen/slotted stand-in: use it once
            pass
    return recovery


#: Said on the way out. Varied so that shutting ORION down twice in an
#: evening does not produce the same sentence twice — a fixed sign-off is
#: the single most machine-like thing an assistant can do, because it is
#: the last thing you hear every time.
_FAREWELLS_LATE = (
    "Goodnight{h}.",
    "Goodnight{h}. Rest well.",
    "Sleep well{h}. I'll be here.",
    "Goodnight{h} — I'll see you in the morning.",
)
_FAREWELLS_DAY = (
    "Okay{h}, I'll see you soon.",
    "Goodbye{h}.",
    "See you later{h}.",
    "Right you are{h}. Until next time.",
    "Very good{h}. I'll be here when you need me.",
    "Take care{h}. See you shortly.",
    "Until later{h}.",
)


async def _with_memory_note(worker: Any, text: str) -> str:
    """*text* with any stored facts that bear on it, by meaning.

    ORION used to answer from what the model happened to ask memory for,
    plus sixteen memories chosen at connect time by category and recency.
    Now a typed request carries the few facts that are actually about it
    (MemoryAgent.relevant_note: meaning-based, a 0.60 cosine floor — lower
    for health facts when the request is about health — at most three), so
    "book me in after my exam" meets the exam date and "I've got a headache,
    what should I take" meets the ibuprofen allergy, without
    the model first having to think of looking it up. Only the payload
    changes: what is shown, logged and matched as the turn stays exactly
    what was typed. Bounded to half a second; any failure sends the text
    alone.
    """
    note_of = getattr(getattr(worker, "memory", None), "relevant_note", None)
    if note_of is None:
        return text
    try:
        note = await asyncio.wait_for(asyncio.to_thread(note_of, text), timeout=0.5)
    except Exception:
        return text
    if not note:
        return text
    worker.bus.log.emit(f"MEMORY: brought {note.count(';') + 1} relevant fact(s) "
                        "to this request.")
    return text + "\n\n" + note


# "refs" holds the source dicts so their ids cannot be recycled while cached.
_SAFE_DECLARATIONS_CACHE: dict[str, Any] = {"key": None, "value": None, "refs": None}


def _live_safe_declarations(declarations: list[Any], bus: Any = None) -> list[dict[str, Any]]:
    """The declarations the Live server will actually accept.

    Every tool passes through gemini_schema (an MCP server's raw JSON Schema —
    $defs, oneOf, const — once failed EVERY connect with "35 validation
    errors for LiveConnectConfig"), then each is validated ON ITS OWN against
    the SDK's FunctionDeclaration, so one bad tool (a forged tool, a future
    MCP server) is dropped and named in the log instead of taking the whole
    voice channel down with it. Cached on the list's identity: a reconnect
    with the same tools costs nothing."""
    from .gemini_schema import sanitise_declarations
    key = tuple((str(d.get("name", "")), id(d)) for d in declarations if isinstance(d, dict))
    if _SAFE_DECLARATIONS_CACHE["key"] == key and _SAFE_DECLARATIONS_CACHE["value"] is not None:
        return _SAFE_DECLARATIONS_CACHE["value"]
    clean, dropped = sanitise_declarations(list(declarations))
    validator = getattr(getattr(types, "FunctionDeclaration", None), "model_validate", None)
    if callable(validator):
        kept = []
        for declaration in clean:
            try:
                validator(declaration)
            except Exception:
                dropped.append(declaration["name"])
                continue
            kept.append(declaration)
        clean = kept
    if dropped and bus is not None:
        try:
            bus.log.emit(
                f"NET: {len(dropped)} tool declaration(s) the live channel cannot "
                f"accept were left out of it: {', '.join(dropped[:8])}"
                + ("…" if len(dropped) > 8 else ""))
        except Exception:
            pass
    _SAFE_DECLARATIONS_CACHE["key"] = key
    _SAFE_DECLARATIONS_CACHE["value"] = clean
    _SAFE_DECLARATIONS_CACHE["refs"] = list(declarations)
    return clean


def _declaration_fingerprint(declarations: list[Any]) -> int:
    """Cheap identity of a tool set: its names, in order."""
    return hash(tuple(str(d.get("name", "")) for d in declarations if isinstance(d, dict)))


def _is_local_config_error(exc: BaseException) -> bool:
    """The SDK refused ORION's own config before any socket opened (a pydantic
    ValidationError from LiveConnectConfig). Says nothing about the provider."""
    module = type(exc).__module__ or ""
    return type(exc).__name__ == "ValidationError" and module.startswith("pydantic")


def _context_window_compression() -> Any:
    """Sliding-window context compression for the Live session.

    Without it Gemini caps a native-audio session at ~15 minutes of audio and
    ~2 minutes once camera/screen frames are streaming, then closes it — a
    hard, silent end every conversation eventually ran into ("he cuts out",
    worst with the camera on). With it the session has no length limit; the
    server trims the oldest turns instead, and the ~10-minute connection
    rotation is carried over by session resumption."""
    try:
        return types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow())
    except Exception:
        return None


class GenAILiveWorker:
    """Realtime provider session with speech-queue-governed voice output."""

    #: Bounds on stop(): how long a cancelled helper task may take to unwind,
    #: and how long the provider gets to acknowledge the session close.
    STOP_TASK_SECONDS = 1.0
    SESSION_CLOSE_SECONDS = 2.0

    def __init__(
        self,
        settings: OrionProviderSettings,
        bus: OrionBus,
        memory: MemoryAgent,
        dispatcher: OrionDispatcher,
        router: ProviderRouter,
        telemetry: Any | None = None,
        local_brain: Any | None = None,
    ) -> None:
        self.settings   = settings
        self.router     = router
        self.bus        = bus
        self.memory     = memory
        self.dispatcher = dispatcher
        self.telemetry  = telemetry
        self.local_brain = local_brain
        # Spoken/button pause state — while paused ORION is silent and only
        # listens for a resume word (or the wake word) to "zone back in".
        self.paused = False
        # Anti-repetition guards (offline voice loop): remember what ORION last
        # said and last acted on, so a transcribed echo of his own voice or a
        # duplicated recogniser result never triggers a second answer.
        self._last_spoken_norm = ""
        self._spoken_at = 0.0
        self._last_cmd_norm = ""
        self._last_cmd_at = 0.0
        # Proactive-announcement dedup (announce()): every source funnels
        # through one channel, so one guard here catches a repeat regardless
        # of which upstream sentinel/reminder/protocol produced it.
        self._last_announced_norm = ""
        self._last_announced_at = 0.0
        # Presence check + standby (Mark X.10).  When the mic input flatlines
        # while ORION is listening, he asks ONCE whether the user is still there
        # instead of silently rotating microphones; no answer within the grace
        # window drops him to a quiet STANDBY that simply waits for a command.
        self._presence_active = False        # a "are you there?" check is open
        self._presence_task: asyncio.Task | None = None
        self._standby = False                # dropped out after no response
        self._presence_prompt_ix = 0         # rotate the prompt wording
        # Quiet mode — "go on standby", "be quiet", "give me a minute". Kept
        # SEPARATE from _standby above (which is the automatic no-response
        # drop-out) because this one is a deliberate instruction from the
        # user and must not be cleared by ORION's own presence logic. He
        # keeps listening and still answers when asked; he just stops
        # volunteering.
        self.quiet_mode = False
        # Deeper than quiet_mode: in standby he answers NOTHING until the
        # wake phrase. Asked for explicitly — quiet_mode alone still replied.
        self.standby_mode = False
        # "Finish speaking" mode (Mark XXII): when ON, ORION always completes his
        # utterance instead of being cut off mid-sentence. This has TWO effects:
        #   • server-side — the Gemini Live session is configured with
        #     activity_handling=NO_INTERRUPTION, so the server stops truncating
        #     his audio when its VAD hears faint echo/room noise (the real cause
        #     of "the text finished but the voice didn't"); see _build_config();
        #   • client-side — a server-signalled interruption is ignored rather
        #     than flushing playback; see _honour_server_interruption().
        # ON by default now (the user asked for it); ORION_FINISH_SPEAKING=0 or a
        # spoken "you can interrupt me" restores classic barge-in.
        self.finish_speaking = os.getenv(
            "ORION_FINISH_SPEAKING", "1").strip().lower() not in {"0", "false", "off", "no"}
        # Startup briefing consent: ORION asks first, then waits for a yes/no.
        self.awaiting_briefing = False
        self._pending_shutdown: dict | None = None   # armed by _ask_to_confirm_shutdown
        self._last_farewell = ""
        # When ORION last delivered a full greeting — the briefing must never
        # open with a second "Good evening" minutes after the first.
        self._greeted_at = 0.0
        # ── unified speech pipeline (event-driven audio state machine) ────────
        self.audio_state = AudioStateMachine(bus, telemetry)
        self.playback = AudioPlaybackThread(bus, self.audio_state, telemetry)
        self.tts      = SpeechSynthesiser(bus, self.audio_state, telemetry)
        self.speech   = SpeechQueueManager(bus, self.playback, self.tts, self.audio_state, telemetry)
        self.speech.state_cb = self._on_speaking_changed_threadsafe
        # Speaker gender recognition (#14): ORION distinguishing male vs female
        # voices on input.  Opt-out via ORION_SPEAKER_ID=0.  The tracker is fed
        # voiced user chunks by the microphone gatekeeper.
        self.speaker_tracker: Any = None
        # Who is speaking, and how they sound. Set by app.py once the reader
        # exists, and handed to the microphone engine when a session opens.
        # Gender is a pitch guess and lives in speaker_tracker above; this is
        # the trained voiceprint, the tone reading, and the only-my-voice gate
        # — all of which used to work on the offline path and nowhere else.
        self.voice_presence: Any = None
        if os.getenv("ORION_SPEAKER_ID", "1").strip().lower() not in {"0", "false", "no", "off"}:
            try:
                from .voice_gender import SpeakerGenderTracker
                self.speaker_tracker = SpeakerGenderTracker(
                    bus, sample_rate=SEND_SAMPLE_RATE, telemetry=telemetry)
            except Exception:
                self.speaker_tracker = None
        # ── session state ─────────────────────────────────────────────────────
        self.session: Any = None
        self.mic: MicrophoneEngine | None = None
        self.out_queue: asyncio.Queue = asyncio.Queue(maxsize=MIC_QUEUE_LIMIT)
        self.stop_event         = asyncio.Event()
        # ── outbound send discipline (Mark X.11, 2026-07-15) ─────────────────
        # A single live websocket underlies every send path: the queued-media
        # pump (orion-send-realtime), GUI-scheduled text turns and tool
        # responses.  Driving them concurrently, or driving one while the
        # channel is being torn down on a go_away, raced the transport and
        # surfaced as `RuntimeError: Cannot enter into task
        # <Task cancelling name='orion-send-realtime'>`.  One lock serialises
        # every outbound call so exactly one send is ever in flight, and a
        # closed-transport flag makes a send scheduled just as the channel drops
        # a clean no-op instead of a reentrancy fault.
        self._send_lock         = asyncio.Lock()
        self._send_closed       = True   # no live transport until connected
        self._session_loop_active = False
        self.stop_event.clear()
        # Explicit connection/failover state machine (Section 4).  Observability
        # only — it publishes each transition on the bus for the diagnostics UI
        # and concise user status; the reconnect control flow is unchanged.
        from .connection_state import ConnectionStateMachine
        self.conn_state = ConnectionStateMachine(on_change=self._on_conn_state)
        self.microphone_enabled = True
        self.connected          = False
        self.tool_busy          = False
        # defer=True: loading Silero costs ~1.5 s and ran on the startup path.
        # The deterministic amplitude gate works from the first chunk, so
        # nothing is lost while the model warms in the background.
        self.vad                = SileroVADGatekeeper(bus, threshold=0.50, defer=True)
        self.recogniser         = LocalSpeechRecogniser(bus)
        # True voice interruption (Mark X.5): a grammar-constrained listener
        # that stays live while ORION speaks — "ORION stop" always works.
        self.interrupts         = VoiceInterruptManager(bus, self.recogniser, telemetry)
        # Wake-word standby is opt-in (ORION_WAKE_MODE=1); by default the
        # microphone is always live, exactly like JARVIS.
        self.wake_mode_enabled = os.getenv("ORION_WAKE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        self._live_model_shift: dict[str, int] = {}
        self.wake_active_until = 0.0
        self._resumption_handle: str | None = None
        # Which language the user speaks, remembered across sessions. Loaded
        # here rather than on first use so the very first thing ORION says in a
        # session — a greeting, a briefing — is already in the right language.
        from .language_memory import LanguageMemory

        self._language = LanguageMemory()
        self._search_tool_enabled = True
        self._last_state       = ""
        self.active_live_provider: AIProviderProfile | None = None
        self._no_live_notice_sent = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # ── turn lifecycle + stall watchdog (Mark X.9+, 2026-07-14) ──────────
        # A request sent to the live channel had no owner once a go_away, a hung
        # tool or a provider stall interrupted it: the reply was silently
        # dropped and the GUI sat in PROCESSING forever — the "stuck
        # processing" bug.  We now track the in-flight request, heartbeat its
        # progress, re-drive it on a clean reconnect, and a watchdog recovers
        # anything that stalls so ORION always returns to LISTENING.
        self._pending_turn: str | None = None    # request awaiting a reply
        self._turn_active = False                 # a turn is in flight
        self._turn_progress_at = 0.0              # monotonic of last progress
        self._pending_turn_retries = 0            # live re-sends attempted
        self._drop_live_output = False            # cancel_turn mutes the reply
        self._turn_watchdog_task: asyncio.Task | None = None
        # Standby / wake owner.  Every flag that standby touches is set AND
        # cleared here, in one deterministic idempotent sequence, so a wake can
        # never miss one and leave ORION alive-but-deaf (see audio_recovery.py).
        self._microphone_before_standby: bool | None = None
        # Per-request trace (request_trace.py) and a latch so the first audio
        # chunk of a reply stamps PLAYBACK_START exactly once.
        self._trace: Any = None
        self._traced_playback = False
        from .audio_recovery import AudioRecovery
        self.recovery = AudioRecovery(self)
        # The dispatcher's morning_briefing tool re-enters through this hook.
        dispatcher.on_briefing_request = self.deliver_briefing_on_demand
        # Proactive-voice channel: reminders, sentinel, protocols, presence.
        self.bus.speak_request.connect(self.announce)
        self.bus.safety_alert.connect(self.alert)
        # Manual audio-device selection from the GUI (the user picks the mic /
        # speaker directly when ORION locks onto the wrong one).
        self.bus.audio_device_request.connect(self._on_audio_device_request)

    def _on_audio_device_request(self, kind: Any = "input", spec: Any = None) -> None:
        """Apply a user-chosen microphone/speaker to the live streams.

        Runs on the GUI/qasync thread (where the streams were opened), so it is
        safe to reopen them directly.  Never raises — a bad pick just logs.
        """
        try:
            kind_s = "input" if str(kind).lower().startswith("in") else "output"
            if kind_s == "input":
                if self.mic is not None:
                    name = self.mic.switch_to_input(spec)
                else:
                    from . import audio_devices as ad
                    ad.set_device("input", str(spec))
                    name = "the chosen microphone (applies when listening resumes)"
                if name:
                    self.bus.speak_request.emit(f"Listening on {name} now.")
            else:
                name = self.speech.set_output_device(spec)
                if name:
                    self.bus.speak_request.emit(f"Speaking through {name} now.")
        except Exception as exc:
            try:
                self.bus.log.emit(f"AUDIO: manual device request failed - {exc}")
            except Exception:
                pass

    def _on_conn_state(self, transition: Any) -> None:
        """Publish each connection-state transition for the diagnostics UI."""
        try:
            self.bus.connection_state.emit(self.conn_state.describe())
        except Exception:
            pass

    # ── state plumbing ────────────────────────────────────────────────────────

    def _emit_state(self, state: str) -> None:
        """Emit state only on change — audio streaming otherwise floods the GUI
        with hundreds of identical SPEAKING signals per second."""
        if state != self._last_state:
            previous = self._last_state
            self._last_state = state
            self.bus.state.emit(state)
            # Returning to LISTENING closes the request out.  A trace that
            # never reaches this stage is precisely a request ORION dropped —
            # that is what TRACES.stalled() reports on.
            trace = getattr(self, "_trace", None)
            if trace is not None and state == "LISTENING" and not trace.complete:
                if previous == "SPEAKING":
                    trace.stage(Stage.PLAYBACK_COMPLETE)
                trace.stage(Stage.LISTENING_RESTORED)

    # ── turn lifecycle (stall recovery) ───────────────────────────────────────

    def _mark_turn(self, text: str) -> None:
        """Record a request now in flight on the live channel so the watchdog
        and reconnect logic can recover it if the reply never arrives."""
        self._pending_turn = text
        self._turn_active = True
        self._pending_turn_retries = 0
        self._drop_live_output = False
        self._turn_progress_at = time.monotonic()
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace.stage(Stage.API_REQUEST_START, text[:80])

    def _mark_turn_progress(self) -> None:
        """Heartbeat: streamed audio, a tool call or turn text all count as the
        turn making progress, so the watchdog only fires on a genuine stall."""
        if self._turn_active:
            self._turn_progress_at = time.monotonic()

    def _clear_turn(self) -> None:
        self._pending_turn = None
        self._turn_active = False
        self._pending_turn_retries = 0

    async def _turn_watchdog(self) -> None:
        """Recover any live turn that stalls with no reply, tool progress or
        speech.  This is the safety net behind every hang vector the live
        channel exposes — a dropped go_away turn the reconnect could not
        re-drive, a tool that never returns, a provider that accepts a turn
        then goes silent.  Without it the GUI stays in PROCESSING forever."""
        TURN_STALL = 28.0    # no reply/output for this long → recover
        TOOL_STALL = 120.0   # a tool may legitimately run long; be generous
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(4.0)
            except asyncio.CancelledError:
                return
            if not self._turn_active or self.paused:
                continue
            if self.speech.output_active():
                self._turn_progress_at = time.monotonic()  # speaking IS progress
                continue
            elapsed = time.monotonic() - self._turn_progress_at
            limit = TOOL_STALL if self.tool_busy else TURN_STALL
            if elapsed < limit:
                continue
            text = self._pending_turn
            self.bus.log.emit(
                f"NET: turn stalled {elapsed:.0f}s with no reply - recovering."
            )
            self.tool_busy = False
            self._clear_turn()
            if text and not self.paused and self.router.has_text_fallback():
                # Best-effort spoken answer via the cloud/local text providers,
                # so the request is never silently lost.  _submit_text_fallback
                # fully manages the state transition back to LISTENING.
                await self._submit_text_fallback(text, reason="Recovered a stalled turn")
            elif not self.speech.output_active():
                self._emit_state("LISTENING" if self._offline_voice_ready() else "STANDBY")

    async def _redrive_pending_turn(self) -> None:
        """After a clean reconnect, re-send a request that a go_away interrupted
        so it is never silently dropped.  One live retry (which preserves tool
        calling), then the text providers as a fallback."""
        if not self._turn_active or not self._pending_turn or self.paused:
            return
        if self.session is None:
            return
        if self._pending_turn_retries >= 1:
            text = self._pending_turn
            self._clear_turn()
            if self.router.has_text_fallback():
                await self._submit_text_fallback(text, reason="Interrupted request")
            return
        self._pending_turn_retries += 1
        self._turn_progress_at = time.monotonic()
        self._drop_live_output = False
        self.bus.log.emit("NET: re-sending the interrupted request on the new channel.")
        self._emit_state("PROCESSING")
        await self._send_text_turn(self._pending_turn)

    def cancel_turn(self) -> None:
        """Hard-stop the current turn so the user can rephrase — distinct from
        pause (which preserves position for a seamless resume).  Cuts playback,
        drops the in-flight request, mutes any reply still streaming from the
        live model, frees the mic and returns to listening."""
        self.speech.interrupt_all()          # destructive: cut playback now
        self._drop_live_output = self.connected  # silence a mid-flight live reply
        self.tool_busy = False
        self._clear_turn()
        if self.paused:
            self.paused = False
            self.bus.paused.emit(False)
        self._refresh_wake_window()
        if self.mic is not None:
            self.mic._drain()
        state = "LISTENING" if (self.connected or self._offline_voice_ready()) else "STANDBY"
        self._emit_state(state)
        self.bus.banner.emit("STOPPED — go ahead, say it again", 2)
        self.bus.log.emit("VOICE: turn cancelled by user - ready for a new request.")

    def _output_active(self) -> bool:
        """True while ORION is audibly speaking through either voice path."""
        return self.speech.output_active()

    def _offline_voice_ready(self) -> bool:
        return (
            self.mic is not None
            and self.recogniser.available
            and self.router.has_text_fallback()
        )

    def _on_speaking_changed_threadsafe(self, active: bool) -> None:
        """SpeechQueueManager monitor-thread callback → marshal to the loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._on_speaking_changed, active)
        except RuntimeError:
            pass  # loop shut down mid-flight

    def _on_speaking_changed(self, active: bool) -> None:
        if getattr(self, "standby_mode", False) and not active:
            # The acknowledgement has finished. Stay dormant rather than
            # letting the ordinary speech lifecycle re-open listening.
            self._emit_state("STANDBY")
            return
        if self.paused:
            return  # hold the PAUSED indicator; do not flip to LISTENING
        if active:
            self._emit_state("SPEAKING")
        elif not self.stop_event.is_set():
            # Speech queue fully drained — only NOW may listening resume.
            if self.connected or self._offline_voice_ready():
                self._emit_state("LISTENING")
            else:
                self._emit_state("STANDBY")

    def set_microphone_enabled(self, enabled: bool) -> None:
        self.microphone_enabled = bool(enabled)
        if self.mic is not None:
            self.mic.set_enabled(self.microphone_enabled)

    def _persist_episode(self, role: str, text: str) -> None:
        """
        Fire-and-forget episodic write.  Mark VIII wrote to SQLite on the
        event-loop thread inside the receive loop, stalling audio on disk I/O.
        Mark IX offloads the write to a worker thread so the loop never blocks.
        """
        try:
            background.spawn(asyncio.to_thread(self.memory.log_episode, role, text))
        except RuntimeError:
            # No running loop (shutdown) — fall back to a direct write.
            self.memory.log_episode(role, text)
        # Companion continuity: silently harvest "working on X" signals from
        # the user's own turns (optional, attached post-construction).
        observer = getattr(self, "companion_agent", None)
        if observer is not None and role == "user":
            try:
                background.spawn(asyncio.to_thread(observer.observe, text))
            except RuntimeError:
                pass

    # ── anti-repetition helpers ───────────────────────────────────────────────

    @staticmethod
    def _normalise_phrase(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", str(text or "").lower()).strip()

    @staticmethod
    def _phrase_overlap(a: str, b: str) -> float:
        """Word-set overlap ratio (0..1) — cheap fuzzy match for echo detection."""
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _say(self, text: str) -> None:
        """Speak via the local voice AND record it, so ORION never re-answers
        his own words if the recogniser transcribes the tail/echo."""
        self._last_spoken_norm = self._normalise_phrase(text)
        self._spoken_at = time.monotonic()
        self.speech.speak_text(text)

    # How long a repeat of the same (or near-identical) proactive note is
    # suppressed. Each *source* (SecuritySentinel, SentinelAgent, reminders,
    # protocols, presence...) already tries to dedupe its own alerts, but
    # they all funnel through this one channel — a guard here catches a
    # repeat regardless of which upstream source produced it, which is what
    # "announces it twice" reports actually need: a floor that can't be
    # bypassed by a bug in any one caller.
    ANNOUNCE_DEDUP_SECONDS = 12.0

    def announce(self, text: str) -> None:
        """
        Proactive, unprompted speech (reminders, sentinel alerts, protocol
        completion) — the JARVIS "Sir, …" channel.  These notices are spoken
        directly through ORION's queued voice rather than submitted as a model
        turn: a model can acknowledge an instruction to relay a notice (for
        example, "Proactive has been relayed") instead of delivering the notice
        itself.  Direct delivery is deterministic, preserves the user-facing
        wording, and keeps a proactive event out of the active conversation.
        Suppressed while paused so a pause is truly silent.
        """
        text = str(text or "").strip()
        if not text or self.paused or self.stop_event.is_set():
            return
        # Quiet mode: the user asked to be left alone while they concentrate
        # on something else. Only UNPROMPTED speech is gated — this method is
        # the sole path for it (reminders, sentinel, presence, protocols), so
        # anything the user actually asks for still gets answered normally
        # through the live session. Being told to be quiet and then being
        # unable to get an answer would be a worse failure than the
        # interruptions it was meant to stop.
        if self.quiet_mode:
            self.bus.log.emit(f"ORION: held back while quiet — {text[:80]}")
            return
        # A volunteered remark that goes unanswered is the user telling ORION
        # something without saying it. Three in a row and he stops offering
        # until they next speak to him (proactive_policy.note_ignored).
        from .proactive_policy import POLICY
        POLICY.note_ignored()
        norm = self._normalise_phrase(text)
        now = time.monotonic()
        if (norm and norm == self._last_announced_norm
                and (now - self._last_announced_at) < self.ANNOUNCE_DEDUP_SECONDS):
            self.bus.log.emit(f"NET: suppressed a repeat proactive announcement: {text[:80]}")
            return
        self._last_announced_norm = norm
        self._last_announced_at = now
        self.bus.log.emit(f"ORION (proactive): {text}")
        self._say(text)

    #: A safety alert repeats at most this often, so a stuck sensor cannot
    #: turn the override channel into a siren. Far longer than the ordinary
    #: announce dedup: these are rare by construction.
    ALERT_DEDUP_SECONDS = 120.0

    def alert(self, text: str, reason: str = "") -> bool:
        """Say something that MUST reach the user, whatever state ORION is in.

        "If there are things that are damaging, concerning to my PC, spilling
        of private data, anything that harms human / humans he must inform me
        directly with voice and this is predominantly important even if he is
        on standby and it overrides everything."

        So this is the one path that ignores standby, quiet mode and pause.
        Every other channel is correctly silenced by those states —
        :meth:`announce` deliberately drops unprompted remarks while quiet —
        and that is exactly why a separate channel is needed rather than a flag
        on the existing one. Danger is not a louder version of a reminder; it
        is a different thing, and it needs a route that no "be quiet" can close.

        Kept deliberately narrow. It does not wake ORION, does not resume the
        conversation and does not open the microphone: it says the thing, logs
        it, and leaves the user in the state they asked for. If they want to
        respond they will speak, and the ordinary wake path handles that.

        Returns whether it was spoken (False only for an empty message or an
        immediate repeat).
        """
        text = str(text or "").strip()
        if not text:
            return False
        norm = self._normalise_phrase(text)
        now = time.monotonic()
        if (norm and norm == getattr(self, "_last_alert_norm", "")
                and now - getattr(self, "_last_alert_at", 0.0) < self.ALERT_DEDUP_SECONDS):
            self.bus.log.emit(f"SAFETY: repeat alert suppressed - {text[:80]}")
            return False
        self._last_alert_norm = norm
        self._last_alert_at = now
        state = []
        if getattr(self, "standby_mode", False):
            state.append("standby")
        if getattr(self, "quiet_mode", False):
            state.append("quiet")
        if getattr(self, "paused", False):
            state.append("paused")
        speech = getattr(self, "speech", None)
        muted = bool(getattr(speech, "muted", False))
        if muted:
            state.append("muted")
        where = f" (overriding {', '.join(state)})" if state else ""
        self.bus.log.emit(
            f"SAFETY{where}: {text}" + (f" [{reason}]" if reason else ""))
        try:
            self.bus.dashboard_event.emit("safety_alert", text[:200])
        except Exception:
            pass
        # Straight to the voice, bypassing announce()'s quiet/pause gates —
        # and a mute: the flag only gates what is QUEUED, so lifting it for
        # this one utterance lets the alert play without unmuting him.
        if muted:
            speech.muted = False
        try:
            self._say(text)
        finally:
            if muted:
                speech.muted = True
        return True

    def _is_own_echo(self, command_norm: str) -> bool:
        """True if a transcript looks like ORION's own recent speech."""
        if not command_norm or not self._last_spoken_norm:
            return False
        if (time.monotonic() - self._spoken_at) > 6.0:
            return False
        return (
            command_norm in self._last_spoken_norm
            or self._phrase_overlap(command_norm, self._last_spoken_norm) >= 0.6
        )

    def _is_duplicate_command(self, command_norm: str) -> bool:
        """True if the same command was just submitted (double recogniser hit)."""
        if command_norm and command_norm == self._last_cmd_norm \
                and (time.monotonic() - self._last_cmd_at) < 4.0:
            return True
        return False

    # ── manual text / file turns ──────────────────────────────────────────────

    async def submit_text(self, text: str) -> None:
        try:
            SecuritySanitiser.guard_text(text, "manual_command")
        except SecurityViolation as exc:
            self.bus.log.emit(f"SEC: {exc}")
            return
        lowered = text.lower().strip()
        # Explicit standby/wake commands must never be delegated to the model.
        # Standby is not the softer pause state: it closes audio capture and
        # moves the UI to ORION's compact orb.
        if self._handle_quiet_command(lowered):
            return
        if self._held_in_standby(lowered):
            return
        # "I'm busy" / "I'm done" — quiet without going dormant.
        if self._handle_focus_command(lowered):
            return
        # A typed command proves the user is here — clear any automatic
        # presence standby, then honour a microphone/speaker request.
        self._note_user_present()
        if self._handle_device_command(lowered):
            return
        # A power/shutdown command is resolved deterministically here so the
        # model can never mis-route "shut down" into powering off the whole PC.
        # A reply to "did you mean that?" is checked BEFORE anything else can
        # claim it — a bare "yes" is not a power command and would otherwise
        # sail past into a normal model turn.
        if self._resolve_shutdown_confirmation(text):
            return
        if await self._handle_power_command(lowered):
            return
        # An explicit typed/sent command re-engages ORION if he was paused.
        if self.paused:
            self.resume()
        # If ORION offered a briefing and is awaiting consent, resolve it here.
        # An offer nobody answered lapses — it must not sit armed for hours
        # waiting to swallow the next thing said.
        offered_at = getattr(self, "_briefing_offered_at", None)
        if (self.awaiting_briefing and offered_at is not None
                and time.monotonic() - offered_at > self.BRIEFING_OFFER_WINDOW_S):
            self.awaiting_briefing = False
        if self.awaiting_briefing:
            consent = self._briefing_consent(text)
            if consent is True:
                self.awaiting_briefing = False
                self._persist_episode("user", text)
                # A startup-offered briefing accepted while it's actually
                # morning is exactly the overnight catch-up case — give it
                # the morning treatment rather than the generic composition.
                from .briefing import MorningBriefingService
                period = ("morning" if MorningBriefingService.greeting_period() == "morning"
                          else "general")
                await self.deliver_briefing_on_demand(period=period)
                return
            if consent is False:
                self.awaiting_briefing = False
                self._persist_episode("user", text)
                self._say("Very good. I'll hold the briefing — just ask when you're ready.")
                return
            # Neither a clear yes nor no: treat as a normal command, stop waiting.
            self.awaiting_briefing = False
        # Reflex fast-path (Mark XXVI): an unambiguous command is dispatched
        # locally with no model round-trip — instant response. Conversation and
        # anything uncertain returns False and flows on to the model unchanged.
        if await self._handle_reflex(lowered, text):
            return
        # Manual text opens the wake window so voice follow-ups flow immediately.
        self.wake_active_until = time.monotonic() + WAKE_WINDOW_SECONDS
        self._persist_episode("user", text)
        # Typed words carry feeling too — let the face respond empathetically.
        try:
            from .emotion import SentimentAnalyser
            SentimentAnalyser.broadcast(self.bus, text, origin="user")
        except Exception:
            pass
        if not self.session:
            await self._submit_text_fallback(text, reason="Live channel offline")
            return
        self._mark_turn(text)
        self._emit_state("PROCESSING")
        await self._send_text_turn(text)

    async def _handle_reflex(self, lowered: str, text: str) -> bool:
        """Dispatch an unambiguous command locally — no model turn. Returns True
        when handled; False (including on ANY dispatch error) falls through to the
        model, so nothing is ever stranded by the fast-path."""
        from .reflex import match_reflex, reflex_enabled
        if not reflex_enabled():
            return False
        m = match_reflex(lowered)
        if m is None:
            # No hand-written rule. A LEARNED reflex may still cover it: a
            # phrase this user has said before that reached the same read-only
            # tool three times running. Exact normalised match only — see
            # reflex_learning for why no generalisation happens here.
            learned = self._learned_reflex(lowered)
            if learned is None:
                return False
            from .reflex import ReflexMatch
            m = ReflexMatch(learned.tool, learned.args(), learned.why())
        self._persist_episode("user", text)
        try:
            result = await self.dispatcher.dispatch_chain(m.tool, dict(m.args), max_depth=2)
        except Exception as exc:
            self.bus.log.emit(f"REFLEX: {m.tool} deferred to model - {first_line(exc)}")
            return False
        reply = (getattr(result, "text", "") or "").strip()
        if reply:
            self._say(reply)
            self._persist_episode("assistant", reply)
        self.bus.log.emit(f"REFLEX: {m.why} handled locally (no model turn)")
        return True

    def reflex_learner(self):
        """The learned-reflex store, built on first use (it opens a database).

        Never raises: reflex learning is an optimisation, and a broken store
        must degrade to "no learned reflexes", never to a failed turn.
        """
        learner = getattr(self, "_reflex_learner", None)
        if learner is None and not getattr(self, "_reflex_learner_failed", False):
            try:
                from .reflex_learning import ReflexLearner
                learner = ReflexLearner()
                self._reflex_learner = learner
            except Exception as exc:
                self._reflex_learner_failed = True
                self.bus.log.emit(f"REFLEX: learning unavailable - {first_line(exc)}")
        return learner

    def _learned_reflex(self, text: str):
        learner = self.reflex_learner()
        if learner is None:
            return None
        try:
            return learner.match(text)
        except Exception:
            return None

    def observe_for_reflex(self, utterance: str, tool: str,
                           args: dict[str, Any] | None = None, ok: bool = True) -> None:
        """Note that an utterance reached a tool, so ORION can learn the shortcut.

        Deliberately silent on failure and never awaited on the turn path: this
        is bookkeeping, and bookkeeping must not be able to break a reply.
        """
        # The language→action network learns from every successful turn, on
        # its own thread (intent_brain.observe_async never blocks this one).
        try:
            from . import intent_brain
            intent_brain.observe_async(utterance, tool, ok=ok)
        except Exception:
            pass
        learner = self.reflex_learner()
        if learner is None:
            return
        try:
            learner.observe(utterance, tool, args, ok=ok)
        except Exception:
            pass

    def resolver_shadow(self):
        """The tool-resolver shadow evaluator, built on first use.

        Never raises: shadow evaluation is measurement, and measurement must
        never be able to affect the thing it measures.
        """
        evaluator = getattr(self, "_resolver_shadow", None)
        if evaluator is None and not getattr(self, "_resolver_shadow_failed", False):
            try:
                from .resolver_shadow import ShadowEvaluator
                evaluator = ShadowEvaluator()
                self._resolver_shadow = evaluator
            except Exception as exc:
                self._resolver_shadow_failed = True
                self.bus.log.emit(f"ROUTER: shadow evaluation unavailable - {first_line(exc)}")
        return evaluator

    def observe_resolver_shadow(self, query: str, tool: str) -> None:
        """Record what the resolver would have done on this turn. Silent."""
        evaluator = self.resolver_shadow()
        if evaluator is None:
            return
        try:
            evaluator.observe(query, tool)
        except Exception:
            pass

    async def submit_file_for_review(self, path: str, prompt: str = "") -> None:
        self._emit_state("PROCESSING")
        try:
            result = await self.dispatcher.dispatch_chain(
                "process_file", {"path": path, "prompt": prompt}, max_depth=2
            )
        except SecurityViolation as exc:
            self.bus.log.emit(f"SEC: {exc}")
            self._emit_state("STANDBY")
            return
        except Exception as exc:
            self.bus.log.emit(f"FILE: scan failed - {exc}")
            self._emit_state("STANDBY")
            return
        self.bus.log.emit("FILE: " + result.text.splitlines()[0][:180])
        instruction = (
            "Review this local file scan and give concise operational input. "
            "If an image frame was attached, inspect the visual content directly when the active provider supports it.\n\n"
            f"{result.text[:7000]}"
        )
        if prompt.strip():
            instruction += f"\n\nUser focus: {prompt.strip()}"
        if not self.session:
            await self._submit_text_fallback(instruction, reason="Live file review unavailable")
            return
        try:
            if result.media:
                await self._send_media(result.media)
            await self._send_text_turn(instruction)
        except Exception as exc:
            self.bus.log.emit(f"FILE: live review dispatch failed - {exc}")
            await self._submit_text_fallback(instruction, reason="Live file review failed")

    async def _text_path_dispatch(self, name: str, args: dict[str, Any],
                                  request: str) -> Any:
        """Run a tool for the text path, turning any IMAGE it returns into words.

        Live reads an attached frame itself; a text model cannot. It used to
        receive only the text beside the frame — for the camera, the
        instruction to describe it — and so answered without having seen
        anything. The image now goes through the router's vision models, and
        the description (or an honest "nothing could look") is the result."""
        result = await self.dispatcher.dispatch_chain(name, args)
        from .vision_describe import describe_image, image_bytes
        image = image_bytes(getattr(result, "media", None))
        if image is None:
            return result
        ok, described = await describe_image(
            self.router, image,
            request + "\n\n" + str(getattr(result, "text", ""))[:600])
        from .data import ToolResult
        return ToolResult(("What the image shows: " if ok else "") + described,
                          ok=bool(getattr(result, "ok", True)) and ok)

    async def _submit_text_fallback(self, text: str, reason: str = "") -> None:
        if self.paused:
            return
        if reason:
            self.bus.log.emit(f"NET: {reason}; routing to text fallback.")
        self._emit_state("PROCESSING")
        # 1) Cloud/local text providers first, when any are available.
        if self.router.has_text_fallback():
            try:
                request = await _with_memory_note(self, text)
                dispatcher = getattr(self, "dispatcher", None)
                if dispatcher is not None and hasattr(dispatcher, "dispatch_chain"):
                    # The text path ACTS, not just talks: the same tools and
                    # guards as Live, through text_tools' act/observe loop.
                    from . import text_tools
                    declarations = (getattr(dispatcher, "TOOL_DECLARATIONS", None)
                                    or TOOL_DECLARATIONS)
                    turn = await text_tools.run(
                        request, declarations,
                        lambda prompt, extra: self.router.generate_text(prompt, extra),
                        lambda name, args: self._text_path_dispatch(name, args, text),
                        log=self.bus.log.emit)
                    profile, response = turn.provider, turn.reply
                    if profile is None:
                        raise RuntimeError("no text provider answered")
                else:
                    profile, response = await self.router.generate_text(request)
                    from .text_tools import strip_markup
                    response = strip_markup(response)
                response = correct_identity(response)
                self.bus.log.emit(f"ORION[{profile.name}]: {response}")
                self._persist_episode("orion", response)
                self._say(response)
                return
            except Exception as exc:
                self.bus.log.emit(f"NET: text providers exhausted - {first_line(exc, 160)}")
        # 2) No provider (or all failed): the offline LocalBrain keeps ORION
        #    conversational and task-capable with zero API calls — never a zombie.
        if self.local_brain is not None:
            try:
                reply = await self.local_brain.respond(text)
                if reply:
                    reply = correct_identity(reply)
                    self.bus.log.emit(f"ORION[local]: {reply}")
                    self._persist_episode("orion", reply)
                    self._say(reply)
                    return
            except Exception as exc:
                self.bus.log.emit(f"BRAIN: local brain fault - {first_line(exc, 160)}")
        else:
            self.bus.log.emit("NET: no text provider and no local brain configured.")
        if not self.speech.output_active():
            self._emit_state("LISTENING" if self._offline_voice_ready() else "STANDBY")

    # ── spoken / button pause control ─────────────────────────────────────────

    def toggle_pause(self) -> None:
        self.resume() if self.paused else self.pause()

    def pause(self) -> None:
        """
        Silence ORION and enter a listening-only PAUSED state.

        Mark X.5: this is a HOLD, not a stop — the playback queue position and
        the unspoken utterance remainder are preserved, so 'resume' continues
        from the exact interruption point with no regeneration.
        """
        if self.paused:
            return
        self.paused = True
        was_speaking = self.speech.hold_all()
        if self.mic is not None:
            self.mic._drain()
        self.bus.paused.emit(True)
        self._emit_state("PAUSED")
        self.bus.banner.emit("PAUSED — say 'Orion resume' to continue where I left off", 2)
        self.bus.log.emit(
            "VOICE: paused"
            + (" mid-speech — position preserved for resume." if was_speaking
               else " — awaiting a resume word.")
        )

    def resume(self) -> None:
        """Zone back in from PAUSED, continuing any held speech in place."""
        if not self.paused:
            return
        self.paused = False
        self._refresh_wake_window()
        self.bus.paused.emit(False)
        self.bus.banner.emit("RESUMED", 1)
        resumed = self.speech.resume_all()
        if resumed:
            # Held speech continues from the exact point — no acknowledgement
            # is spoken over it, and the state machine flips to SPEAKING the
            # instant the renderer writes the preserved remainder.
            self.bus.log.emit("VOICE: resumed from the held position.")
            self._emit_state("SPEAKING")
            return
        self.bus.log.emit("VOICE: resumed.")
        if self.connected:
            self._emit_state("LISTENING")
        else:
            # Offline: a brief local-voice acknowledgement to confirm re-engagement.
            self._emit_state("LISTENING" if self._offline_voice_ready() else "STANDBY")
            self.speech.speak_text("Back with you.")

    def _on_interrupt_phrase(self, action: str) -> None:
        """A spoken interruption command matched by the always-on listener."""
        self.interrupts.note_trigger()
        if action == VoiceInterruptManager.ACTION_PAUSE:
            self.bus.log.emit("VOICE: interruption command honoured — holding speech.")
            self.pause()
        elif action == VoiceInterruptManager.ACTION_RESUME:
            self.bus.log.emit("VOICE: continuation command honoured.")
            self.resume()

    @staticmethod
    def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
        return any(p in text for p in phrases)

    _AFFIRM = ("yes", "yeah", "yep", "yup", "sure", "please", "go ahead", "go on",
               "ok", "okay", "do it", "absolutely", "brief me", "of course",
               "affirmative", "sounds good", "let's", "lets", "i would", "i'd like")
    _DECLINE = ("no", "nope", "not now", "later", "skip", "don't", "dont",
                "no thanks", "no thank you", "not right now", "maybe later", "hold off")
    #: Words that may surround a yes/no without making it a new request.
    _CONSENT_FILLER = frozenset(
        "orion hey sir please thanks thank you then now right go on ahead it do "
        "that that's thats would be great lovely fine good just all ok okay yes "
        "yeah sure me brief briefing update the a an i'd like love to hear not "
        "now later maybe thanks no nope for moment at the minute cheers thing".split())
    #: How long a briefing offer waits for its answer before lapsing.
    BRIEFING_OFFER_WINDOW_S = 300.0

    def _briefing_consent(self, text: str) -> bool | None:
        """Return True (yes), False (no), or None (not an answer) to a yes/no offer.

        The answer must be the WHOLE reply — a yes/no phrase and nothing but
        filler around it. A word merely somewhere in a sentence is not consent:
        "call me via Twilio please" once counted as "yes" to a briefing offered
        at startup (the "please"), and this same test answers "shut down — did
        you mean that?". Whole-word matching as before, so "do it now" and
        "I know" are not read as declines just for containing "no"."""
        low = re.sub(r"[^\w\s']", " ", str(text or "").lower())
        words = low.split()
        while words and words[0] in ("orion", "hey"):
            words = words[1:]          # the wake word is not part of the answer
        if not words or len(words) > 8:
            return None
        joined = " ".join(words)

        def leads(phrases: tuple[str, ...]) -> str | None:
            for phrase in sorted(phrases, key=len, reverse=True):
                if joined == phrase or joined.startswith(phrase + " "):
                    return phrase
            return None

        for phrases, verdict in ((self._DECLINE, False), (self._AFFIRM, True)):
            phrase = leads(phrases)
            if phrase is None:
                continue
            rest = joined[len(phrase):].split()
            if all(word in self._CONSENT_FILLER for word in rest):
                return verdict
            return None       # "no, call me instead" — a new request, not an answer
        return None

    async def _send_text_turn(self, text: str) -> None:
        if not self._transport_ready():
            # The live channel dropped between scheduling and running this turn;
            # route it through the text providers rather than sending on a dead
            # transport (the reentrancy fault) or silently losing the request.
            await self._submit_text_fallback(text, reason="Live channel unavailable")
            return
        fallback_reason: str | None = None
        payload = await _with_memory_note(self, text)
        async with self._send_lock:
            session = self.session
            if session is None or self._send_closed:
                fallback_reason = "Live channel closed"
            else:
                try:
                    await session.send_client_content(
                        turns={"parts": [{"text": payload}]}, turn_complete=True
                    )
                except asyncio.CancelledError:
                    raise
                except TypeError:
                    content = types.Content(role="user", parts=[types.Part(text=payload)])
                    await session.send_client_content(turns=content, turn_complete=True)
                except Exception as exc:
                    self.bus.log.emit(f"NET: manual command rejected - {exc}")
                    fallback_reason = "Native provider rejected manual command"
        # Run any provider fallback OUTSIDE the send lock so a slow HTTP call can
        # never block the next websocket send.
        if fallback_reason is not None:
            # This live turn has definitively handed off to the text path.
            # Leaving it active made the stall watchdog retry an answer that had
            # already been spoken after a transport failure.
            if self._turn_active and self._pending_turn == text:
                self._clear_turn()
            await self._submit_text_fallback(text, reason=fallback_reason)

    # ── farewell (Mark XXI) ──────────────────────────────────────────────────

    def compose_farewell(self) -> str:
        """A time-varying, honorific-aware farewell, mirroring
        _compose_greeting's shape — used on every shutdown path, not just
        the voice-initiated one, so closing the window is not silent.

        Avoids repeating the previous sign-off: with a handful of options,
        random choice alone lands on the same one back-to-back often enough to
        be noticeable, which defeats the point of having several.
        """
        honorific = ""
        identity = getattr(self.dispatcher, "identity", None)
        if identity is not None:
            try:
                frequency = str(identity.preferences.get("honorific_frequency") or "occasional")
                if frequency != "never":
                    honorific = f", {identity.preferences.get('honorific') or 'sir'}"
            except Exception:
                pass
        from .time_service import TIME
        hour = TIME.now().hour           # ORION's clock, like his greetings
        pool = _FAREWELLS_LATE if (hour >= 22 or hour < 5) else _FAREWELLS_DAY
        options = [line for line in pool if line != getattr(self, "_last_farewell", "")]
        chosen = random.choice(options or list(pool))
        self._last_farewell = chosen
        return chosen.format(h=honorific)

    # ── shutdown confirmation ────────────────────────────────────────────────

    #: How long an unanswered "did you mean that?" stays open. Long enough to
    #: reply properly, short enough that a stray "yes" ten minutes later in an
    #: unrelated conversation cannot power the machine down.
    SHUTDOWN_CONFIRM_WINDOW_S = 45.0

    def _ask_to_confirm_shutdown(self, is_restart: bool = False) -> None:
        """Check the user meant it, instead of acting on the first mention.

        "Shut down" is a phrase that turns up inside ordinary sentences — a
        question about shutting down a service, thinking out loud, a phrase
        caught mid-conversation by an open microphone. Acting on it
        immediately makes ORION something you have to be careful in front of,
        and the cost of being wrong is asymmetric: a needless confirmation
        costs two seconds, a wrong shutdown costs the whole session.
        """
        # Saying it twice IS the confirmation. Someone who repeats "shut down"
        # after being asked has answered the question, and making them find the
        # word "yes" would be pedantic.
        if self._shutdown_confirmation_open():
            pending = self._pending_shutdown
            self._pending_shutdown = None
            self._begin_shutdown(is_restart=bool(pending.get("restart")))
            return
        self._pending_shutdown = {
            "at": time.monotonic(),
            "restart": bool(is_restart),
        }
        word = "restart" if is_restart else "shut down"
        self._say(random.choice([
            f"Just to confirm — you'd like me to {word}?",
            f"You want me to {word}. Did you mean that?",
            f"Shall I {word}? Say the word and I will.",
        ]))

    def _shutdown_confirmation_open(self) -> bool:
        pending = getattr(self, "_pending_shutdown", None)
        if not pending:
            return False
        if time.monotonic() - pending["at"] > self.SHUTDOWN_CONFIRM_WINDOW_S:
            # Expired rather than answered. Let it lapse silently — announcing
            # "I won't shut down then" for a question the user has already
            # moved on from is noise.
            self._pending_shutdown = None
            return False
        return True

    def _resolve_shutdown_confirmation(self, text: str) -> bool:
        """Handle a reply to "did you mean that?". True if it was consumed."""
        if not self._shutdown_confirmation_open():
            return False
        pending = self._pending_shutdown
        answer = self._briefing_consent(text)
        if answer is None:
            return False          # not a yes/no — let it be a normal request
        self._pending_shutdown = None
        if not answer:
            self.bus.log.emit("SYS: shutdown declined by user.")
            self._say(random.choice([
                "Understood — staying put.",
                "Right, I'll stay where I am.",
                "Standing by, then.",
            ]))
            return True
        self._begin_shutdown(is_restart=bool(pending.get("restart")))
        return True

    def _begin_shutdown(self, is_restart: bool = False) -> None:
        """Say goodbye properly, THEN go."""
        if is_restart:
            self.bus.log.emit("SYS: self-restart requested by user.")
            self._farewell_then_go(
                "Restarting now - I'll be right back.",
                restart=True, task_name="orion-self-restart")
            return
        self.bus.log.emit("SYS: self-shutdown requested by user.")
        self._farewell_spoken = True
        self._farewell_then_go(
            f"Very good. {self.compose_farewell()}",
            restart=False, task_name="orion-self-shutdown")

    def _farewell_then_go(self, farewell: str, restart: bool,
                          task_name: str) -> None:
        """Say goodbye, then shut down — in his real voice where there is one.

        The goodbye and the shutdown both live in the spawned coroutine so the
        farewell can be routed through the live session and awaited. That only
        works if there IS a loop: ``background.spawn`` closes the coroutine and
        returns None when there is not, which would lose the goodbye AND the
        shutdown with it. So a missing loop falls back to exactly what this
        used to do — speak locally, emit, done.
        """
        task = background.spawn(
            self._shutdown_after_farewell(restart=restart, farewell=farewell),
            name=task_name)
        if task is not None:
            return
        self._say(farewell)
        if restart:
            self.bus.request_restart.emit()
        else:
            self.bus.request_shutdown.emit()

    async def speak_farewell(self, text: str) -> None:
        """Say *text* in ORION's real voice if there is one, else the local one.

        The goodbye was always spoken by the built-in SAPI voice, on every
        path — the one that sounds nothing like him. `_say()` goes straight to
        the local synthesiser, so the model ORION actually talks through was
        never asked, even while its session was still open.

        Sent as an instruction rather than a script: given the situation he
        signs off in his own words, which is the point of it being his voice.
        The local voice stays as the fallback, because a goodbye in the wrong
        voice is much better than silence.
        """
        if self.session is not None:
            try:
                await self._send_text_turn(
                    "You are shutting down now, at the user's request. Say a "
                    "short goodbye in your own words — one or two sentences, "
                    "warm, no questions, and call no tools. "
                    f"For tone, the line you would otherwise have said is: {text}")
                self._last_spoken_norm = self._normalise_phrase(text)
                self._spoken_at = time.monotonic()
                return
            except Exception as exc:
                self.bus.log.emit(
                    f"SYS: the live voice could not say goodbye ({first_line(exc)}) "
                    "- using the local voice.")
        self._say(text)

    async def _shutdown_after_farewell(self, restart: bool,
                                       farewell: str = "") -> None:
        """Wait for the goodbye to actually FINISH, then shut down.

        The previous version slept a flat 2.4 s and fired. That is not long
        enough for the longer sign-offs, and because this path sets
        _farewell_spoken, app.py's own _await_farewell is skipped — so nothing
        downstream was waiting either, and ORION cut himself off mid-goodbye.

        Polls the speech pipeline's real is_busy() instead, with a hard ceiling
        so a wedged audio device can never hold shutdown open.
        """
        if farewell:
            await self.speak_farewell(farewell)
        await self._await_speech()
        if restart:
            self.bus.request_restart.emit()
        else:
            self.bus.request_shutdown.emit()

    # ── startup briefing ──────────────────────────────────────────────────────

    async def _compose_greeting(self) -> str:
        """
        Phase 3 — dynamic temporal presence.  When the TemporalPresence
        service is attached (``worker.temporal``, wired in app.py) the
        greeting carries the day, season, weather, calendar load and time
        since the last conversation; otherwise the classic greeting rotation
        stands in, so startup can never fail on a missing feed.
        """
        temporal = getattr(self, "temporal", None)
        if temporal is not None:
            try:
                greeting = await temporal.compose_greeting()
                if greeting:
                    return greeting
            except Exception as exc:
                self.bus.log.emit(f"TEMPORAL: greeting degraded - {first_line(exc)}")
        from .time_service import TIME
        # Shared band table — the classic rotation used to say "good morning"
        # at half past midnight because it only ever asked `hour < 12`.
        return random.choice(STARTUP_GREETINGS).format(period=TIME.greeting_period())

    async def offer_startup_briefing(self) -> None:
        """
        At startup ORION ASKS whether the user wants their briefing rather than
        launching into it — UNLESS one was already delivered today and nothing
        has genuinely broken since, in which case it just greets normally
        without asking again (re-asking for a rerun of the same briefing was
        the reported annoyance).  On the live channel the model is instructed
        to ask and only call the morning_briefing tool on consent; offline,
        ORION asks by voice and the next affirmative reply (handled in
        submit_text) runs it.
        """
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline and not self.connected and not self.stop_event.is_set():
            await asyncio.sleep(0.5)
        if self.stop_event.is_set():
            return
        self._refresh_wake_window()
        greeting = await self._compose_greeting()
        self._greeted_at = time.monotonic()

        skip_offer = False
        recency = ""
        fresh = False
        briefing = self.dispatcher.briefing
        # OR'd with the recency check so a briefing given just before midnight
        # is not forgotten just after it — see briefed_recently().
        try:
            counts = briefing.already_briefed_today() or briefing.briefed_recently()
        except AttributeError:
            counts = briefing.already_briefed_today()
        if counts:
            try:
                fresh = await briefing.has_fresh_news_since_last_briefing()
            except Exception:
                fresh = False
            skip_offer = not fresh
            try:
                recency = briefing.briefing_recency_phrase()
            except Exception:
                recency = ""

        if skip_offer:
            # SAY that he remembers, rather than silently not offering. The
            # previous instruction was "do NOT offer or mention the briefing",
            # which is correct about not re-offering and wrong about staying
            # silent: from the user's side an assistant who says nothing is
            # indistinguishable from one who has forgotten. Naming when it
            # happened is the whole difference between "he knows" and "he
            # doesn't".
            if recency:
                remembered = (
                    f"Then tell them they already had their briefing {recency}, "
                    f"and that nothing new has broken since. Use the phrase "
                    f"'{recency}' EXACTLY as written — it is measured from the "
                    f"clock and is accurate, so do not round it, rephrase it, or "
                    f"turn it into a vaguer time. Say it once, briefly.")
            else:
                remembered = (
                    "Then tell them, in your own words, that they have already "
                    "had their briefing today and nothing new has broken since. "
                    "Say it once and briefly. Do NOT guess at when it was.")
            if self.session is not None:
                instruction = (
                    f'Open with this exact greeting, word for word: "{greeting}" '
                    "Do not paraphrase it, do not shorten it, and do not replace "
                    "'morning'/'afternoon'/'evening' with a vaguer word like 'day' — "
                    "it was composed from the actual clock, so it is already correct. "
                    + remembered +
                    " Do NOT offer another briefing and do NOT summarise the old one."
                )
                self._emit_state("PROCESSING")
                await self._send_text_turn(instruction)
            else:
                spoken = (f"{greeting} You've already had your briefing "
                          f"{recency}, and nothing new has broken since."
                          if recency else
                          f"{greeting} You've already had your briefing today, "
                          "and nothing new has broken since.")
                self._say(spoken)
            return

        self.awaiting_briefing = True
        self._briefing_offered_at = time.monotonic()
        # Two different offers. Having already been briefed and there being
        # something NEW is a specific situation, and asking the same generic
        # "would you like your briefing?" throws away the one useful fact ORION
        # has — that he checked, and something has actually changed since.
        if fresh:
            ask = (
                f"Then say that they had their briefing "
                f"{recency or 'earlier today'} but that something new has broken "
                "since, and ask whether they want the update. Make it clear this "
                "is NEW material, not a repeat. One short sentence."
            )
            spoken = (f"{greeting} You had your briefing {recency}, but something "
                      "new has come in. Want the update?" if recency else
                      f"{greeting} You've had your briefing today, but something "
                      "new has come in. Want the update?")
        else:
            ask = (
                "Then ask, in one short sentence, whether they would like their "
                "briefing now (AI and technology news, markets, calendar, tasks and "
                "priority email)."
            )
            spoken = f"{greeting} Would you like your briefing?"
        if self.session is not None:
            instruction = (
                f'Open with this exact greeting, word for word: "{greeting}" '
                "Do not paraphrase it, do not shorten it, and do not replace "
                "'morning'/'afternoon'/'evening' with a vaguer word like 'day' — "
                "it was composed from the actual clock, so it is already correct. "
                + ask +
                " Do NOT deliver the briefing yet. Only if they agree, "
                "call the morning_briefing tool. If they decline, acknowledge briefly "
                "and stand by."
            )
            self._emit_state("PROCESSING")
            await self._send_text_turn(instruction)
        else:
            self._say(spoken)

    # Legacy alias retained for any external caller.
    async def deliver_startup_briefing(self) -> None:
        await self.offer_startup_briefing()

    async def deliver_briefing_on_demand(self, period: str = "general") -> None:
        """Dispatcher hook: the user asked for the briefing mid-session."""
        self.awaiting_briefing = False
        await self._deliver_briefing(wait_for_connection=2.0, period=period)

    async def _deliver_briefing(self, wait_for_connection: float, period: str = "general") -> None:
        try:
            briefing = await self.dispatcher.briefing.compose_source_material(period=period)
        except Exception as exc:
            self.bus.log.emit(f"BRIEF: briefing failed - {first_line(exc)}")
            return
        self.bus.log.emit("BRIEF: intelligence briefing prepared.")
        deadline = time.monotonic() + max(0.0, wait_for_connection)
        while time.monotonic() < deadline and not self.connected and not self.stop_event.is_set():
            await asyncio.sleep(0.5)
        if self.stop_event.is_set():
            return
        self._refresh_wake_window()
        # Greet only if this session hasn't greeted recently — the startup
        # offer already opened with the full temporal greeting, and repeating
        # it word-for-word at the top of the briefing sounded broken.
        if (time.monotonic() - self._greeted_at) > 600.0:
            greeting = await self._compose_greeting()
            self._greeted_at = time.monotonic()
        else:
            greeting = ""
        instruction = self.dispatcher.briefing.delivery_instruction(greeting, briefing)
        # A briefing runs for a minute or more, and it is exactly when the user
        # is most likely to have something urgent — "not now, I need X". Normal
        # barge-in stays off (his own voice clears the same VAD bar as a real
        # interruption, and a false trigger mid-answer is worse than waiting),
        # but for a monologue that trade reverses. The hardening is untouched:
        # this only makes the path reachable while he is reading the briefing.
        self._set_interruptible(True)
        try:
            if self.session is not None:
                self._emit_state("PROCESSING")
                await self._send_text_turn(instruction)
            else:
                self.bus.log.emit(f"BRIEF: {briefing}")
                await self._submit_text_fallback(
                    instruction, reason="Live channel offline for briefing")
            await self._await_speech(end_limit=240.0)
        finally:
            self._set_interruptible(False)

    def _set_interruptible(self, on: bool) -> None:
        """Let the user cut in while ORION is mid-monologue. Never raises: a
        missing microphone must not stop a briefing being delivered."""
        mic = getattr(self, "mic", None)
        if mic is None:
            return
        try:
            mic.set_interruptible(bool(on))
        except Exception:
            pass

    async def _await_speech(self, start_limit: float = 6.0,
                            end_limit: float = 14.0, tail: float = 1.3) -> None:
        """Wait for ORION to START speaking, then for him to FINISH.

        Waiting only for "not busy" is a trap whenever the words come from the
        MODEL rather than the local synthesiser. ``speak_text`` queues
        instantly, so a short fixed grace was enough to see is_busy() go true;
        a live turn takes a second or three to come back, so the first poll
        finds silence, concludes he has finished, and shuts down before he has
        said anything. That is the clipped goodbye again, by a different route.

        Both waits are bounded, and the second is measured from when he
        actually starts — so a slow model costs start-up latency, not the
        whole ceiling. A tail lets the output buffer drain: is_busy() drops
        when the pipeline stops FEEDING the device, with the last word still
        on its way out.
        """
        speech = getattr(self, "speech", None)
        if speech is None or not hasattr(speech, "is_busy"):
            return
        # Read through getattr rather than self.stop_event directly. This runs
        # on the teardown path, and reaching for an attribute that may not be
        # there yet is the exact shape of the bug this whole wait exists to
        # fix — app._await_farewell polled a method SpeechQueueManager did not
        # have, and its bare except turned that into an instant return.
        stopping = getattr(self, "stop_event", None)

        def _stopping() -> bool:
            try:
                return bool(stopping is not None and stopping.is_set())
            except Exception:
                return False

        def _busy() -> bool | None:
            try:
                return bool(speech.is_busy())
            except Exception:
                return None               # no usable pipeline — do not stall

        deadline = time.monotonic() + max(0.0, start_limit)
        while time.monotonic() < deadline and not _stopping():
            busy = _busy()
            if busy is None:
                return
            if busy:
                break                     # he has started
            await asyncio.sleep(0.1)

        deadline = time.monotonic() + max(0.0, end_limit)
        while time.monotonic() < deadline and not _stopping():
            busy = _busy()
            if busy is None:
                return
            if not busy:
                await asyncio.sleep(tail)
                return
            await asyncio.sleep(0.1)

    # ── session lifecycle ─────────────────────────────────────────────────────

    def _warn_if_key_malformed(self) -> None:
        """A standard Google AI Studio key looks like 'AIza…' (39 chars).  A
        short-lived OAuth / ephemeral token (e.g. 'AQ.…') authenticates for a
        while, then EXPIRES — dropping the live channel to offline every so
        often with a 1008 'missing authentication credential'.  Detect a
        non-standard credential at startup and say so once, clearly, so the
        recurring offline drops are never a mystery."""
        try:
            for profile in self.router.live_profiles():
                key = (self.router.active_key(profile)
                       if hasattr(self.router, "active_key")
                       else getattr(profile, "api_key", "") or "").strip()
                if not key or re.fullmatch(r"AIza[0-9A-Za-z_\-]{35}", key):
                    continue
                if key.startswith("AQ."):
                    self.bus.log.emit(
                        f"NET: the {profile.name} credential is an EPHEMERAL token "
                        "('AQ.…'). It authenticates the live channel now but will "
                        "expire within hours, dropping voice to offline until it is "
                        "regenerated. For a permanent link, replace it with a free "
                        "AI Studio key (https://aistudio.google.com/apikey) in "
                        "config/api_keys.json.")
                else:
                    self.bus.log.emit(
                        f"NET: the {profile.name} credential is not a standard Google "
                        "AI Studio API key (expected 'AIza…', 39 chars). Live voice will "
                        "keep dropping to offline until it is replaced — get a free key "
                        "at https://aistudio.google.com/apikey and paste it into "
                        "config/api_keys.json.")
                break
        except Exception:
            pass

    def _announce_credential_problem(self, profile: Any) -> None:
        """Explain an authentication failure to the user ONCE — out loud and on
        the banner — so recurring offline drops are never a silent 'policy
        violation' mystery."""
        if getattr(self, "_credential_warned", False):
            return
        self._credential_warned = True
        key = (self.router.active_key(profile)
               if hasattr(self.router, "active_key")
               else getattr(profile, "api_key", "") or "").strip()
        if key.startswith("AQ."):
            self.bus.log.emit(
                "NET: the Gemini ephemeral token has EXPIRED (they only last hours). "
                "Regenerate it, or replace it with a permanent Google AI Studio key "
                "(https://aistudio.google.com/apikey) in config/api_keys.json so the "
                "live channel stops dropping. Running on local voice meanwhile.")
            spoken = (
                "My live voice token has expired — ephemeral tokens only last a "
                "few hours. A permanent AI Studio key in my config would stop these "
                "drops; I'll keep using my local voice until then.")
        else:
            self.bus.log.emit(
                "NET: Gemini rejected the API key (authentication error). Replace the "
                "Gemini key in config/api_keys.json with a valid Google AI Studio key "
                "(https://aistudio.google.com/apikey). Running on local voice meanwhile.")
            spoken = (
                "My live voice key has been rejected — it appears invalid or "
                "expired. Please replace the Gemini key in my config, and I'll keep "
                "using my local voice until then.")
        try:
            self.bus.banner.emit(
                "⚠ Gemini credential invalid/expired — replace it in config/api_keys.json", 8)
        except Exception:
            pass
        if not self.paused:
            try:
                self._say(spoken)
            except Exception:
                pass

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.speech.start()
        # Stall watchdog: recovers any live turn that hangs (go_away drop, hung
        # tool, silent provider) so the GUI never sticks in PROCESSING.
        if self._turn_watchdog_task is None:
            self._turn_watchdog_task = asyncio.create_task(
                self._turn_watchdog(), name="orion-turn-watchdog"
            )
        self.bus.log.emit(
            "VOICE: wake-word standby "
            + ("enabled - say 'Orion' to open the channel."
               if self.wake_mode_enabled
               else "disabled - microphone always live (set ORION_WAKE_MODE=1 to change).")
        )
        self._warn_if_key_malformed()
        live_index = 0
        backoff    = 2.0
        consecutive_failures = 0
        while not self.stop_event.is_set():
            retry_delay = 0.0
            profiles = self.router.live_profiles()
            if not profiles:
                self.connected = False
                self.session   = None
                self._ensure_fallback_mic()
                self.conn_state.set(
                    ConnectionState.LOCAL_FALLBACK if self._offline_voice_ready()
                    else ConnectionState.DEGRADED_OFFLINE,
                    "no live audio provider")
                self._emit_state("LISTENING" if self._offline_voice_ready() else "STANDBY")
                if not self._no_live_notice_sent:
                    if self.router.has_text_fallback():
                        self.bus.log.emit("NET: no live audio provider available; text fallback is armed.")
                        if self._offline_voice_ready():
                            self.bus.log.emit(
                                "VOICE: offline voice loop active - speak normally; "
                                "local recognition routes to the text providers and replies aloud."
                            )
                    else:
                        self.bus.log.emit("NET: no live audio or text fallback provider is currently available.")
                    self._no_live_notice_sent = True
                await asyncio.sleep(2.0)
                continue
            self._no_live_notice_sent = False
            profile = profiles[live_index % len(profiles)]
            live_index += 1
            candidates = [profile.model or LIVE_MODEL] + [
                m for m in LIVE_MODEL_FALLBACKS if m != (profile.model or LIVE_MODEL)
            ]
            live_model = candidates[self._live_model_shift.get(profile.name, 0) % len(candidates)]
            live_key = self.router.active_key(profile) if hasattr(self.router, "active_key") \
                else profile.api_key
            # Ephemeral tokens ('AQ.…', minted via auth_tokens.create) only
            # authenticate the Live WebSocket on the v1alpha surface; sending
            # one over v1beta yields 1008 'missing required authentication
            # credential' even while the token is still valid.
            api_version = "v1alpha" if live_key.startswith("AQ.") else "v1beta"
            client = genai.Client(api_key=live_key, http_options={"api_version": api_version})
            session_established = False
            connected_at = 0.0
            try:
                self.conn_state.provider = profile.name
                self.conn_state.set(ConnectionState.CONNECTING, f"{profile.name}")
                self._emit_state("CONNECTING")
                self.bus.log.emit(
                    f"NET: initialising {profile.name} live channel ({live_model.rsplit('/', 1)[-1]})."
                )
                config = self._build_config()
                resuming = bool(self._resumption_handle)
                async with client.aio.live.connect(model=live_model, config=config) as session:
                    session_established = True
                    connected_at = time.monotonic()
                    _live_diag("connected", provider=profile.name, model=live_model,
                               resumed=resuming, stage=getattr(self, "_live_config_stage", 0))
                    self.session     = session
                    self._send_closed = False   # transport open — sends permitted
                    self.connected   = True
                    self.active_live_provider = profile
                    self.active_live_model = live_model
                    self.conn_state.set(ConnectionState.CONNECTED, f"{profile.name} synchronised")
                    self._emit_state("LISTENING")
                    self.bus.log.emit(f"NET: {profile.name} live channel synchronised.")
                    backoff = 2.0
                    consecutive_failures = 0
                    live_index -= 1  # keep the working profile on clean reconnects
                    # A go_away mid-turn drops the request; re-send it on the
                    # fresh channel so the notepad edit (or whatever was asked)
                    # actually completes instead of hanging in PROCESSING.
                    await self._redrive_pending_turn()
                    await self._session_loop()
                    self.bus.log.emit("NET: live channel closed; re-establishing.")
                    _live_diag("closed", provider=profile.name,
                               seconds=round(time.monotonic() - connected_at, 1),
                               reason=getattr(self, "_last_close_reason", "") or "receive ended")
                    self._last_close_reason = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.session   = None
                self.active_live_provider = None
                self._emit_state("STANDBY")
                message = str(exc)
                error_summary, error_type = _extract_genai_error_context(exc)
                close_code = _close_code(exc)
                # A connection that WORKED and then dropped says nothing bad
                # about the provider or the resumption handle.
                dropped = session_established and (
                    close_code in _TRANSIENT_CLOSE_CODES
                    or close_code in _OUR_PAYLOAD_CLOSE_CODES
                    or error_type in {"ConnectionError", "TimeoutError", "OSError",
                                      "ConnectionClosed", "ConnectionClosedError",
                                      "ConnectionClosedOK", "IncompleteReadError"})
                transient = dropped or close_code in _TRANSIENT_CLOSE_CODES
                _live_diag("error", provider=profile.name, code=close_code,
                           type=error_type, established=session_established,
                           seconds=round(time.monotonic() - connected_at, 1)
                           if session_established else 0.0,
                           summary=error_summary[:240])
                self.bus.log.emit(f"NET: {error_summary}")
                # Log full traceback at debug level for troubleshooting
                try:
                    tb = traceback.format_exc(limit=3)
                    if "Traceback" in tb:
                        self.bus.log.emit(f"DEBUG: Live channel exception trace: {tb[:400]}")
                except Exception:
                    pass
                # Our OWN config refused — by the SDK before any socket
                # opened, or by the server during setup — says nothing about
                # the provider. Cooling it for 45 s, again on every retry, WAS
                # the "live voice channel is unstable" outage. Step down to a
                # plainer config and reconnect at once instead.
                stage = getattr(self, "_live_config_stage", 0)
                local_config_error = stage < 2 and (
                    _is_local_config_error(exc)
                    or (not session_established
                        and bool(re.search(r"(?i)\b1007\b|invalid argument|"
                                           r"invalid json payload|function.?declaration",
                                           message))))
                if local_config_error:
                    self._live_config_stage = stage + 1
                    self._live_config_fp = _declaration_fingerprint(
                        getattr(self.dispatcher, "TOOL_DECLARATIONS", None) or TOOL_DECLARATIONS)
                    _SAFE_DECLARATIONS_CACHE["key"] = None
                    self.bus.log.emit(
                        "NET: live config refused; reconnecting on the built-in tools only."
                        if stage == 0 else
                        "NET: live config still refused; reconnecting without context compression.")
                elif self._search_tool_enabled and re.search(r"(?i)google_search", message):
                    self._search_tool_enabled = False
                    self.bus.log.emit("NET: search grounding rejected by provider; reconnecting without it.")
                elif transient:
                    # The channel dropped; the provider did not fail. Reconnect
                    # (resuming the conversation) instead of benching Live.
                    if close_code in _OUR_PAYLOAD_CLOSE_CODES:
                        self.bus.log.emit(
                            f"NET: the live channel refused a frame ({close_code}); "
                            "reconnecting on a fresh channel.")
                    else:
                        self.bus.log.emit(
                            "NET: live channel dropped"
                            + (f" (code {close_code})" if close_code else "")
                            + "; resuming the conversation on a new channel.")
                elif re.search(r"(?i)not.?found|404|does not exist|unsupported|invalid model", message):
                    # Bad/retired model id — rotate through known live models
                    # instead of cooling the whole provider.
                    self._live_model_shift[profile.name] = (
                        self._live_model_shift.get(profile.name, 0) + 1
                    )
                    self.bus.log.emit(
                        f"NET: live model {live_model} unavailable; rotating to the next candidate."
                    )
                elif error_type in {"ConnectionError", "TimeoutError", "OSError"}:
                    # Network-level issues shouldn't cool the whole provider for 45s;
                    # just back off and retry.
                    self.bus.log.emit(f"NET: temporary connection issue; retrying shortly.")
                elif "1011" in error_summary or "server error" in error_summary.lower():
                    # WebSocket 1011: generic server error. Could be transient.
                    self.bus.log.emit(f"NET: Gemini service temporarily unavailable; retrying with backoff.")
                elif error_summary.startswith("Error code 1000"):
                    # WebSocket 1000 is a NORMAL closure — the far end (or our own
                    # shutdown path) ended the session cleanly, most often the Live
                    # API's own session-duration limit on a long conversation. It is
                    # not a fault, so it must never draw the same 45s punitive
                    # cooldown as a real error would — that 45s of silence with no
                    # obvious cause was exactly the "ORION randomly pauses" report.
                    # Reconnect promptly instead, same as a network blip.
                    self.bus.log.emit(
                        "NET: live session ended normally (code 1000) — reconnecting.")
                elif re.search(r"(?i)missing required authentication|unauthenticated|"
                               r"expected oauth|oauth ?2|access token|login cookie|"
                               r"authentication credential|api[ _-]?key not valid|"
                               r"invalid api[ _-]?key|api key expired|permission denied", message):
                    # 1008 AUTHENTICATION failure — an invalid or expired key (an
                    # 'AQ.…' OAuth/ephemeral token instead of an 'AIza…' key is the
                    # usual cause).  Explain it in plain language once, then cool the
                    # provider so ORION runs on local voice until the key is fixed —
                    # rather than a cryptic 'policy violation' every few minutes.
                    self._announce_credential_problem(profile)
                    self.router.mark_failure(profile, exc)
                else:
                    self.router.mark_failure(profile, exc)
                # A stale resumption handle can poison every reconnect attempt;
                # drop it so the next connection starts fresh. But only then:
                # dropping it after an ordinary disconnect is what made ORION
                # come back with no memory of the conversation. A handle that
                # is itself the problem fails at setup, which lands here with
                # session_established False and is dropped.
                if not dropped:
                    self._resumption_handle = None
                if local_config_error:
                    retry_delay = 0.5    # nothing remote to wait out
                elif dropped:
                    retry_delay = 0.5    # it was working a moment ago
                else:
                    consecutive_failures += 1
                    retry_delay = backoff
                    backoff = min(12.0, backoff * 1.5)
                # Reflect the failure class in the explicit connection state so
                # the diagnostics UI shows WHY the channel dropped.
                from .connection_state import ConnectionStateMachine as _CSM
                go_away = _CSM.classify_go_away(message)
                self.conn_state.set(_CSM.next_state_for(go_away), error_summary[:120])
                if self.conn_state.in_reconnect_loop():
                    self.conn_state.set(ConnectionState.SWITCHING_PROVIDER, "reconnect loop — rotating")
            finally:
                # Close the transport gate BEFORE dropping the session handle so
                # any send racing this teardown re-checks the flag and no-ops
                # instead of driving a half-closed websocket (the go_away
                # reentrancy path).
                self._send_closed = True
                self.connected = False
                self.session   = None
                self.active_live_provider = None
                if self.mic is not None:
                    self.mic.stop()
                    self.mic = None
            if retry_delay and not self.stop_event.is_set():
                # The live channel is down (1011 server errors, network drops,
                # quota) — ORION must never go quiet while it recovers.  Bring
                # the offline voice loop up so local recognition and the text
                # providers keep him conversational through the outage, and
                # tell the user ONCE what happened instead of silent STANDBY.
                self._ensure_fallback_mic()
                if self._offline_voice_ready():
                    self._emit_state("LISTENING")
                if consecutive_failures == 3 and not self.paused:
                    self._say(
                        "My Gemini Live connection has dropped and I'm "
                        "reconnecting now. Until it's back I'm on my local "
                        "voice — keep talking to me as normal; I can still "
                        "act on what you ask."
                    )
                await asyncio.sleep(retry_delay)
        self.speech.stop()

    async def stop(self) -> None:
        # Idempotent: a second stop (double-clicked quit, shutdown + restart)
        # must not double-cancel tasks or re-close a dead transport.
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.conn_state.set(ConnectionState.SHUTTING_DOWN, "worker stop")
        self._send_closed = True   # forbid any further outbound send immediately
        from .shutdown_trace import TRACE
        if self.mic is not None:
            with TRACE.phase("microphone"):
                self.mic.stop()
        with TRACE.phase("speech"):
            self.speech.stop()
        # Await the long-lived helper tasks so none is destroyed pending at
        # shutdown ("Task was destroyed but it is pending!") — each with a
        # bound, because a task that is cancelled mid-way through a tool call
        # or a text fallback can take its time to unwind, and nothing here may
        # hold shutdown open.
        for attr in ("_turn_watchdog_task", "_presence_task"):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=self.STOP_TASK_SECONDS)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception:
                    pass
            setattr(self, attr, None)
        # Closing the live session is a network round trip (a websocket close
        # handshake with the provider). It had no bound at all, so a provider
        # that never answered the close held the whole of shutdown here until
        # the 25 s watchdog killed the process.
        try:
            if self.session is not None and hasattr(self.session, "close"):
                with TRACE.phase("live session close"):
                    maybe = self.session.close()
                    if asyncio.iscoroutine(maybe):
                        await asyncio.wait_for(maybe, timeout=self.SESSION_CLOSE_SECONDS)
        except Exception:           # includes the TimeoutError
            pass
        self.session = None

    def _ensure_fallback_mic(self) -> None:
        """
        JARVIS-style offline voice loop: when no live audio provider is
        available, keep the microphone alive so local recognition can still
        hear commands; replies are routed through the text providers and
        spoken with the local voice.
        """
        if self.mic is not None or not self.microphone_enabled or self.stop_event.is_set():
            return
        try:
            loop = asyncio.get_running_loop()
            self.mic = MicrophoneEngine(
                loop, asyncio.Queue(maxsize=8), self.bus,
                lambda: False,  # nothing consumes forwarded audio offline
                self.vad,
                recogniser=self.recogniser,
                on_transcript=self._on_local_transcript,
                speaking_check=self._output_active,
                on_barge_in=self._on_barge_in,
                interrupt_feed=self.interrupts.feed,
                on_interrupt=self._on_interrupt_phrase,
                on_input_silent=self._on_input_silent,
                on_input_restored=self._on_input_restored,
                speaker_tracker=self.speaker_tracker,
                voice_presence=self.voice_presence,
                is_own_echo=self._is_own_echo,
            )
            self.mic.set_enabled(self.microphone_enabled)
            self.mic.start()
        except Exception as exc:
            self.mic = None
            self.bus.log.emit(f"AUDIO: offline microphone unavailable - {exc}")

    async def _session_loop(self) -> None:
        if self.mic is not None:
            # Replace the offline fallback microphone with the live pipeline.
            self.mic.stop()
            self.mic = None
        self.out_queue = asyncio.Queue(maxsize=MIC_QUEUE_LIMIT)
        loop           = asyncio.get_running_loop()
        self.mic       = MicrophoneEngine(
            loop, self.out_queue, self.bus, self._can_capture_microphone, self.vad,
            recogniser=self.recogniser,
            on_transcript=self._on_local_transcript,
            speaking_check=self._output_active,
            on_barge_in=self._on_barge_in,
            interrupt_feed=self.interrupts.feed,
            on_interrupt=self._on_interrupt_phrase,
            # Keep the command listener alive while paused, so "ORION resume" is
            # heard even though the interaction channel is held. Hearing and
            # interaction, kept separate — exactly what pause should mean.
            command_listen_check=lambda: self.paused,
            on_input_silent=self._on_input_silent,
            on_input_restored=self._on_input_restored,
            speaker_tracker=self.speaker_tracker,
            voice_presence=self.voice_presence,
            is_own_echo=self._is_own_echo,
        )
        self.mic.set_enabled(self.microphone_enabled)
        self.mic.start()
        # Guard against a second session loop ever driving the same queue/session
        # concurrently — start/reconnect must be idempotent.
        if self._session_loop_active:
            self.bus.log.emit("NET: session loop already active; refusing a duplicate sender.")
            return
        self._session_loop_active = True
        send_task = asyncio.create_task(self._send_realtime(),    name="orion-send-realtime")
        recv_task = asyncio.create_task(self._receive_realtime(), name="orion-receive-realtime")
        try:
            # FIRST_COMPLETED: a receive loop that ends cleanly must also tear down
            # the send loop, otherwise the session hangs on out_queue.get() forever.
            done, pending = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
            # Shut the transport gate before cancelling the pump so neither the
            # cancelling send task nor any GUI-scheduled turn drives the socket
            # mid-teardown (the 'Cannot enter into task' reentrancy).
            self._send_closed = True
            for task in pending:
                task.cancel()
            if pending:
                # Await the cancelled tasks so no pending task is destroyed and
                # any CancelledError is fully settled before the session closes.
                await asyncio.gather(*pending, return_exceptions=True)
            # Retrieve EVERY outcome before raising any of them. Raising on the
            # first exceptional task left any other completed task's exception
            # unread, and Python then reported it from the collector at an
            # arbitrary later moment — which is where
            #
            #   Task exception was never retrieved
            #   future: <Task finished name='orion-receive-realtime'
            #            exception=APIError('1011 ... Internal error')>
            #
            # came from on shutdown. The 1011 itself is an ordinary upstream
            # drop that the reconnect logic already handles; only the unread
            # exception was noise.
            failures = []
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    failures.append(exc)
            if failures:
                raise failures[0]
        finally:
            self._session_loop_active = False
            # Cancelled while still in the wait above, none of that ran. That
            # is the ordinary shutdown order: worker.stop() closes the session
            # — which ends the receive loop with APIError('1000 None.'), the
            # normal-close code — and app.py cancels this task on the very
            # next line, before the wait has woken. The receive task's
            # exception then went unread and was printed at exit as "Task
            # exception was never retrieved" (seen 2026-09-23, on a clean quit).
            for task in (send_task, recv_task):
                if task.done():
                    _consume_outcome(task)
                else:
                    task.cancel()
                    task.add_done_callback(_consume_outcome)

    def _can_capture_microphone(self) -> bool:
        # The half-duplex speak-then-listen rule is enforced upstream in the
        # AudioGateThread via speaking_check; here only mute, tool execution,
        # pause, shutdown, and a closed wake gate block capture.  (Local
        # recognition still runs while paused so the resume word is heard.)
        return (
            self.microphone_enabled
            and not self.tool_busy
            and not self.paused
            and not self.standby_mode
            and not self.stop_event.is_set()
            and self._wake_gate_open()
        )

    def _wake_gate_open(self) -> bool:
        if not self.wake_mode_enabled or not self.recogniser.available:
            return True
        return time.monotonic() < self.wake_active_until

    def _refresh_wake_window(self) -> None:
        self.wake_active_until = time.monotonic() + WAKE_WINDOW_SECONDS

    def _on_local_transcript(self, text: str) -> None:
        lowered = text.lower().strip()
        if not lowered:
            # An empty transcription is one of the ways "he ignored me" happens
            # and it used to leave no evidence at all.  Record it.
            trace = TRACES.begin(origin="voice")
            trace.stage(Stage.CAPTURE_COMPLETE)
            trace.failed(Stage.TRANSCRIPTION_START, "empty transcription",
                         recovery="kept listening")
            return
        # From here the request has an identity that every later stage stamps,
        # so a request that stops anywhere can be attributed to a subsystem
        # instead of just being silence (see request_trace.py).
        self._trace = TRACES.begin(origin="voice")
        self._trace.stage(Stage.TRANSCRIPTION_COMPLETE, lowered[:80])
        # Explicit standby disables microphone capture. A final queued local
        # transcript can still arrive as it closes, so only an explicit wake
        # phrase may act on it; ambient speech cannot revive ORION.
        if self.standby_mode:
            if self._handle_quiet_command(lowered):
                return
            if self._held_in_standby(lowered):
                return
        if self._handle_quiet_command(lowered):
            return
        # "I'm busy" / "I'm done" — quiet without going dormant.
        if self._handle_focus_command(lowered):
            return
        # "Finish your sentences" / "you can interrupt me" — toggle finish-speaking.
        if self._handle_finish_speaking_command(lowered):
            return
        # Any recognised speech proves the user is present and the mic is live:
        # cancel a presence check and leave automatic standby before anything else.
        self._note_user_present()
        wake_hit = any(word in lowered for word in WAKE_WORDS)
        # ── explicit interruption commands take precedence over everything ───
        interrupt_action = VoiceInterruptManager.classify(lowered)
        if interrupt_action:
            self._on_interrupt_phrase(interrupt_action)
            return
        # ── audio-device changes: "I can't hear you" / "switch microphone" ───
        if self._handle_device_command(lowered):
            return
        # ── power / shutdown: resolve deterministically BEFORE the live channel
        #    handles the turn natively (below), so a spoken "shut down" brings
        #    ORION down by default and can never be mis-routed by the model into
        #    powering off the whole PC.  Only an utterance that names the
        #    computer / PC / machine reaches the guarded host-shutdown path.
        if self._is_power_command(lowered):
            self.bus.log.emit("VOICE: power command intercepted deterministically.")
            background.spawn(self._handle_power_command(lowered))
            return
        # ── spoken cancel: scrap the current request and re-listen ───────────
        # Checked before pause so "cancel"/"never mind" always aborts rather
        # than holding.  Only meaningful while a turn is actually in flight or
        # ORION is speaking; otherwise it is just conversational filler.
        if self._matches_any(lowered, CANCEL_WORDS) and (
            self._turn_active or self.speech.output_active() or self.tool_busy
        ):
            self.bus.log.emit("VOICE: cancel phrase heard - dropping the request.")
            self.cancel_turn()
            return
        # ── spoken pause / resume control (works online and offline) ─────────
        if self.paused:
            # While paused, ORION only listens for a way back in.
            if wake_hit or self._matches_any(lowered, RESUME_WORDS):
                self.resume()
            return
        if self._matches_any(lowered, PAUSE_WORDS):
            self.pause()
            return
        if wake_hit:
            was_closed = not self._wake_gate_open()
            self._refresh_wake_window()
            if was_closed:
                self.bus.log.emit("SR: wake word detected - channel open.")
                self.bus.banner.emit("WAKE WORD ACKNOWLEDGED", 2)
            if self.connected:
                self._emit_state("LISTENING")
        if self.connected:
            return  # live channel handles the conversation natively
        # ── offline voice loop: local STT → provider router → local voice ────
        command = self._strip_wake_words(lowered) if wake_hit else lowered
        if wake_hit and not command:
            self.bus.log.emit("YOU (voice): [wake]")
            self._say("Yes?")
            return
        if self.wake_mode_enabled and not wake_hit and not self._wake_gate_open():
            return  # standby: ignore ambient speech until the wake word
        if not wake_hit and len(command.split()) < 2:
            return  # single stray words are almost always noise
        # ── anti-repetition: never answer ORION's own echo or a duplicate ────
        command_norm = self._normalise_phrase(command)
        if self._is_own_echo(command_norm):
            self.bus.log.emit("SR: ignored an echo of ORION's own speech.")
            return
        if self._is_duplicate_command(command_norm):
            self.bus.log.emit("SR: ignored a duplicate command.")
            return
        if not self.router.has_text_fallback() and self.local_brain is None:
            self.bus.log.emit(f"SR (local): {text}")
            return
        self._last_cmd_norm = command_norm
        self._last_cmd_at = time.monotonic()
        self.bus.log.emit(f"YOU (voice): {command}")
        self._refresh_wake_window()
        background.spawn(self.submit_text(command))

    def _strip_wake_words(self, text: str) -> str:
        for word in WAKE_WORDS:
            text = text.replace(word, " ")
        return re.sub(r"\s+", " ", text).strip(" ,.!?")

    def _on_barge_in(self) -> None:
        """Explicit voice interruption — only reachable with ORION_ALLOW_BARGE_IN=1."""
        if self.speech.interrupt_all():
            self.bus.log.emit("AUDIO: user interruption - playback halted.")
            if self.connected:
                self._emit_state("LISTENING")

    def _honour_server_interruption(self) -> bool:
        """Whether to act on a server-signalled interruption by cutting speech.

        Normally yes — that is how the user barges in. But in finish-speaking
        mode we do NOT: the user reported his voice being clipped mid-sentence
        (the text log shows the full reply, the audio stops early), which is the
        classic self-barge-in — his own native-audio voice bleeding into the mic
        and tripping the server's VAD. When finish-speaking is on he always
        completes the utterance. Isolated here so the decision is testable
        without the whole receive loop.
        """
        return not self.finish_speaking

    def set_finish_speaking(self, on: bool) -> str:
        """Toggle finish-speaking mode (also reachable by voice)."""
        self.finish_speaking = bool(on)
        if on:
            return ("I'll finish what I'm saying from now on, rather than stopping "
                    "when I think you've cut in. Say 'you can interrupt me' to change "
                    "it back.")
        return "You can interrupt me again — I'll stop when you start talking."

    _FINISH_ON_RE = re.compile(
        r"\b(finish (your|the|each) (sentence|sentences|thought)|finish speaking|"
        r"finish what you(?:'re| are) saying|don'?t cut yourself off|"
        r"stop cutting yourself off|say the whole thing|"
        r"let me hear the whole (thing|sentence)|"
        r"don'?t stop (until|till) you(?:'re| are) (finished|done))\b",
        re.IGNORECASE)
    _FINISH_OFF_RE = re.compile(
        r"\b(you can interrupt me|you can interrupt|i can interrupt you|"
        r"let me interrupt|barge[- ]?in is (fine|ok|okay)|interrupt mode)\b",
        re.IGNORECASE)

    def _handle_finish_speaking_command(self, lowered: str) -> bool:
        """"Finish your sentences" / "you can interrupt me" — toggle the mode."""
        text = str(lowered or "")
        if not _is_control_length(text):
            return False
        if self._FINISH_OFF_RE.search(text):
            self._say(self.set_finish_speaking(False))
            return True
        if self._FINISH_ON_RE.search(text):
            self._say(self.set_finish_speaking(True))
            return True
        return False

    # ── presence check + standby (dead-mic handling) ──────────────────────────

    def _on_input_silent(self) -> None:
        """The microphone input has flatlined while ORION is listening.  Runs on
        the event loop (marshalled by the mic engine).  Rather than hopping
        devices, ask once whether the user is still there."""
        self._begin_presence_check()

    def _on_input_restored(self) -> None:
        """The microphone is hearing sound again — the user is clearly present,
        so clear any open presence check and leave standby."""
        self._note_user_present()

    def _begin_presence_check(self) -> None:
        """Ask, once, whether the user is still there; after the grace window
        with no response, drop to STANDBY.  Never fires while paused, already in
        standby, mid-check, or while ORION himself is speaking."""
        if (self.paused or self._standby or self._presence_active
                or self.stop_event.is_set() or self.speech.output_active()
                or self.tool_busy or self._turn_active or self.awaiting_briefing):
            return
        self._presence_active = True
        prompt = PRESENCE_PROMPTS[self._presence_prompt_ix % len(PRESENCE_PROMPTS)]
        self._presence_prompt_ix += 1
        self.bus.log.emit("AUDIO: input has gone quiet — checking you're still there.")
        self.bus.banner.emit("ARE YOU STILL THERE?", 3)
        # "Are you still there?" is AMBIENT by policy: a quiet microphone is
        # not an emergency, and asking about it is exactly the pestering the
        # user objected to. The banner still shows it; the room stays quiet,
        # and the grace window below still drops him to standby on its own.
        from .proactive_policy import POLICY
        if POLICY.should_speak("presence_check").speak:
            self.announce(prompt)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            self._presence_task = asyncio.create_task(self._presence_grace())

    async def _presence_grace(self) -> None:
        """Wait the grace window; if the presence check is still open (no reply,
        no restored signal) settle into standby."""
        try:
            await asyncio.sleep(PRESENCE_GRACE_SECONDS)
        except asyncio.CancelledError:
            return
        if self._presence_active and not self.stop_event.is_set():
            self._enter_standby()

    def _note_user_present(self) -> None:
        """Any sign of the user (a transcript, restored mic signal, a typed
        command) confirms presence: cancel a check and leave standby.

        Also the strongest possible evidence that they are available to be
        spoken to, so it releases a FOCUS that ORION entered on his own."""
        from .proactive_policy import POLICY
        POLICY.user_spoke()
        if self._presence_active:
            self._presence_active = False
            if self._presence_task is not None:
                self._presence_task.cancel()
                self._presence_task = None
            self.bus.log.emit("AUDIO: you're still there — carrying on.")
        if self._standby:
            self._exit_standby()

    def _enter_standby(self) -> None:
        """No response to the presence check — go quiet and simply wait.  ORION
        keeps listening (so a wake word or 'I can't hear you' still reaches him)
        but stops prompting an empty room."""
        if self._standby:
            return
        self._standby = True
        self._presence_active = False
        self.bus.log.emit("AUDIO: no response — standing by until you need me.")
        self.bus.banner.emit("STANDING BY — say 'Orion' when you need me", 4)
        if not self.speech.output_active():
            self._emit_state("STANDBY")

    def _exit_standby(self) -> None:
        if not self._standby:
            return
        self._standby = False
        self.bus.log.emit("AUDIO: leaving standby.")
        if not self.speech.output_active():
            self._emit_state("LISTENING" if (self.connected or self._offline_voice_ready())
                             else "STANDBY")

    # ── explicit audio-device switching ("I can't hear you") ──────────────────

    def _handle_device_command(self, lowered: str) -> bool:
        """If *lowered* is a spoken/typed request to change an audio device,
        rotate it and return True; otherwise return False.  'I can't hear you'
        rotates the OUTPUT (the user can't hear ORION); 'switch microphone'
        rotates the INPUT."""
        if self._matches_any(lowered, OUTPUT_SWITCH_WORDS):
            self._switch_output_device()
            return True
        if self._matches_any(lowered, INPUT_SWITCH_WORDS):
            self._switch_input_device()
            return True
        return False

    def _is_power_command(self, lowered: str) -> bool:
        """True if *lowered* is a shutdown / restart / power command that must be
        resolved deterministically — so it never reaches the live model, which
        could otherwise mis-route a bare 'shut down' into powering off the whole
        PC.  Covers destructive host intents (guarded), the self-directed
        CLOSE_APP / STOP_ORION, and the bare/ambiguous 'shut down' or 'restart'
        which, addressed to ORION in conversation, mean *him*."""
        try:
            from .system_guard import (
                ActionIntent, DESTRUCTIVE_INTENTS, classify_intent,
            )
        except Exception:
            return False
        intent = classify_intent(lowered)
        return intent in DESTRUCTIVE_INTENTS or intent in (
            ActionIntent.CLOSE_APP, ActionIntent.STOP_ORION, ActionIntent.AMBIGUOUS,
        )

    # Deliberately narrow. These must not fire on ordinary conversation that
    # merely mentions being quiet ("it's quiet in here", "the quiet part"),
    # so each pattern requires an imperative addressed to ORION.
    _QUIET_ON_RE = re.compile(
        r"\b(go (?:on |into )?standby|(?:on|into) standby(?: mode)?|"
        r"standby mode|"
        r"stand by(?: for now)?|"
        r"(?:be|stay|keep) (?:quiet|silent)|"
        r"(?:don'?t|do not) (?:speak|talk|interrupt|disturb)(?: me)?|"
        r"quiet mode|silence yourself|leave me (?:alone|to it)|"
        r"give me (?:a minute|a moment|some space|some quiet))\b"
        # A bare "standby" counts only when it ENDS the utterance ("orion,
        # standby"). Matching it anywhere would fire on ordinary questions
        # like "what does standby power mean".
        r"|\bstandby\s*[.!]?\s*$")
    _QUIET_OFF_RE = re.compile(
        r"\b(?:you can |ok(?:ay)? )?("
        # The phrase the user actually uses to bring him back. Listed first
        # because it is the one that has to work every time: standby is now
        # FULL silence, so nothing else he says will release it.
        r"i need you|i need your help|are you (?:there|awake)|"
        r"wake up|come back|i'?m back|resume|carry on|"
        r"(?:stop|exit|end|leave|cancel) (?:being )?(?:quiet|silent|standby)|"
        r"(?:exit|leave|end) standby|"
        r"(?:speak|talk) (?:to me )?(?:again|freely)|"
        r"you can (?:speak|talk)(?: again)?)\b")

    # FOCUS is the state between "talking normally" and "asleep": ORION still
    # HEARS you and answers instantly, he just stops volunteering. The user
    # asked for exactly this — "he keeps pestering me when I'm there and I am
    # busy doing a task". Standby (dormant, mic closed) was the only thing on
    # offer before, and it is far too blunt for being merely busy.
    # The first-person prefix is REQUIRED on the adjective forms. Without it a
    # bare "busy" matched anywhere in a sentence, so asking "how busy is the
    # processor" silenced him — the opposite of what was wanted.
    #: "I need to focus" is a STRONGER request than "I'm busy", and the user
    #: asked for it to mean full standby: microphone closed, nothing said, only
    #: inward work continuing. Checked BEFORE _FOCUS_ON_RE, which would
    #: otherwise swallow it on the word "focus" and leave him merely quiet —
    #: still listening, which is exactly what was not wanted.
    _NEED_FOCUS_RE = re.compile(
        r"\b(?:i )?need to (?:focus|concentrate)\b"
        r"|\bi need (?:to get )?(?:some )?(?:focus|quiet|peace|deep work)\b"
        r"|\b(?:let me|i want to) (?:focus|concentrate)\b"
        r"|\bstop listening(?: to me)?\b"
        r"|\bgo (?:on |into )?standby\b")

    _FOCUS_ON_RE = re.compile(
        r"\b(?:i'?m|i am|i'?ll be)\s+(?:busy|concentrating|working|"
        r"focus(?:sed|ing|sing)?|in the middle of something|on a call|"
        r"in a meeting|heads down)\b"
        r"|\b(?:don'?t|do not) (?:volunteer|offer|suggest|bother me|interrupt me)\b"
        r"|\b(?:no|stop) (?:interruptions?|suggestions?|updates?)\b"
        r"|\bfocus mode\b|\bdo not disturb\b|\bhold your thoughts\b"
        r"|\bgive me (?:a minute|a moment|some space)\b")
    _FOCUS_OFF_RE = re.compile(
        r"\b(?:i'?m )?(?:done|finished|free now|back)\b"
        r"|\b(?:end|exit|stop|leave|cancel) (?:focus|do not disturb)(?: mode)?\b"
        r"|\byou can (?:talk|speak|volunteer|interrupt)(?: again)?\b"
        r"|\bwhat did i miss\b")

    def _enter_focus_standby(self) -> None:
        """Full standby, entered because the user said they need to focus.

        Distinct from "I'm busy" (which only stops him volunteering) in that
        the microphone closes too — he genuinely stops listening. Research, the
        forge, the thought stream and anything already assigned keep running,
        and are written to the standby ledger so they show up in the logs; only
        the outward-facing half stops. Danger still reaches the user, via
        alert(), which does not consult any of this.
        """
        from .standby_work import LEDGER
        from .proactive_policy import POLICY, Attention

        LEDGER.start()
        POLICY.set_attention(Attention.STANDBY, "user needs to focus")
        self._say("Understood — I'll stop listening and leave you to it. "
                  "I'll keep working on what you've already given me, and I'll "
                  "still speak up if something's genuinely wrong. "
                  "Say ‘ORION, I need you’ when you want me back.")
        self.bus.log.emit(
            "STANDBY: entered because the user needs to focus - microphone "
            "closed, speech held; research, assigned tasks, thinking and "
            "self-improvement continue (see the standby ledger).")
        try:
            self._audio_recovery().enter_standby("user needs to focus")
        except Exception as exc:
            self.bus.log.emit(f"STANDBY: could not fully enter standby - {exc}")

    def _handle_focus_command(self, lowered: str) -> bool:
        """Enter or leave FOCUS — quiet, but still listening.

        "I need to focus" is handled first and goes further than this: it means
        full standby, because that is what was asked for.
        """
        from .proactive_policy import POLICY, Attention
        text = str(lowered or "")
        if not _is_control_length(text):
            return False
        if self._NEED_FOCUS_RE.search(text) and not self._FOCUS_OFF_RE.search(text):
            self._enter_focus_standby()
            return True
        # Leaving is matched first for the same reason quiet mode does it:
        # "stop focus mode" contains "focus mode".
        if self._FOCUS_OFF_RE.search(text):
            if POLICY.attention is Attention.FOCUS:
                held = sum(POLICY.suppressed.values())
                POLICY.leave_focus("user said they are free")
                self._say(f"Right — I held back {held} thing{'s' if held != 1 else ''} "
                          "while you were busy." if held else "Good. I'm here.")
                return True
            return False
        if self._FOCUS_ON_RE.search(text):
            POLICY.enter_focus("user said they are busy")
            self._say("Understood. I'll keep quiet unless it's urgent — "
                      "just speak and I'm here.")
            return True
        return False

    #: A control phrase is SHORT. "Orion, standby" is three words; a prompt you
    #: pasted in is not a command however many of these words happen to appear
    #: inside it.
    #:
    #: This is the paste bug: pasting a paragraph containing "…give me a minute
    #: to think about how standby should work… I am busy most days" matched BOTH
    #: the standby pattern and the focus pattern, so ORION went dormant instead
    #: of answering the thing he had just been handed. Verified against the real
    #: patterns before the guard existed.
    #:
    #: Word count, not characters: a long single-clause sentence is still
    #: plausibly spoken, whereas forty words never is.
    CONTROL_PHRASE_MAX_WORDS = 12


    # MUTE silences ORION's VOICE and nothing else: he still listens, still
    # works, and his replies still reach the screen. Distinct from standby
    # (dormant, deaf) and focus (stops volunteering), and from the host
    # "mute" (the PC's master volume, via the peripherals tool). Every
    # pattern names HIM — "mute yourself", "mute your voice", "orion, mute" —
    # so muting a video or asking what "mute" means cannot silence him.
    _MUTE_OFF_RE = re.compile(
        r"\b(un-?mute(?: yourself| your voice| orion)?|"
        r"(?:turn|switch) (?:your )?voice (?:back )?on|voice on|"
        r"(?:speak|talk) (?:out loud|aloud)(?: again)?|"
        r"you can (?:speak|talk) (?:out loud|aloud)(?: again)?)\b")
    _MUTE_ON_RE = re.compile(
        r"\b(mute (?:yourself|your voice|orion)|orion,? mute|"
        r"(?:turn|switch) (?:off )?your voice(?: off)?|voice off|"
        r"(?:stop|no) (?:talking|speaking) (?:out loud|aloud)|"
        r"(?:reply|answer|respond) (?:in|with|by) text(?: only)?|text only)\b")

    def set_voice_muted(self, muted: bool, announce: bool = True) -> None:
        """Mute or unmute ORION's voice — the HUD button and the spoken
        command both land here. ``announce=False`` on the Live path, where the
        model's own reply is the acknowledgement."""
        speech = getattr(self, "speech", None)
        if speech is None or not hasattr(speech, "set_muted"):
            return
        if bool(muted) == bool(getattr(speech, "muted", False)):
            return
        speech.set_muted(muted)
        if muted:
            self.bus.log.emit("ORION: voice muted — my replies will appear as text. "
                              "Say 'unmute' or press the speaker button to hear me again.")
        else:
            self.bus.log.emit("ORION: voice back on.")
            if announce:
                self._say("Voice back on.")

    def _handle_mute_command(self, lowered: str, announce: bool = True) -> bool:
        text = str(lowered or "")
        if not _is_control_length(text):
            return False
        # Unmute first, for the same reason quiet mode checks its release
        # first: "don't mute yourself" must not be read as a mute.
        if self._MUTE_OFF_RE.search(text):
            self.set_voice_muted(False, announce=announce)
            return True
        if self._MUTE_ON_RE.search(text) and not re.search(r"\b(?:don'?t|do not)\b", text):
            self.set_voice_muted(True, announce=announce)
            return True
        return False

    def _handle_quiet_command(self, lowered: str) -> bool:
        """Enter or leave quiet mode on a spoken instruction.

        A standby instruction is an actual dormant state: microphone capture
        closes, the current turn is discarded, and the core window becomes
        ORION's compact orb. Returns True when the utterance was such an
        instruction, so it is not also handed to the model.
        """
        text = str(lowered or "")
        # A pasted prompt is not a command, however many command words it
        # happens to contain (see CONTROL_PHRASE_MAX_WORDS).
        if not _is_control_length(text):
            return False
        # Mute is resolved here because every path that could carry it —
        # typed, local voice, and ahead of the power classifier — already
        # calls this first.
        if self._handle_mute_command(text):
            return True
        recovery = _audio_recovery(self)
        # Leaving is checked first: "stop being quiet" contains "be quiet",
        # and matching the ON pattern there would trap the user in silence
        # with the very phrase meant to release it.
        if self._QUIET_OFF_RE.search(text):
            was_quiet = bool(getattr(self, "quiet_mode", False) or
                              getattr(self, "standby_mode", False))
            was_standby = bool(getattr(self, "standby_mode", False))
            if was_standby:
                # The full deterministic wake ladder (audio_recovery.py):
                # cancel standby -> verify output -> verify input -> reopen
                # streams -> reset playback/interruption/mic flags -> listen.
                # It is idempotent, and it is the ONLY thing that clears
                # _drop_live_output on this path — without that, the entire
                # first reply after waking was discarded and ORION appeared
                # mute.
                report = recovery.wake("spoken wake command")
                # What he got through while you were focused. Said on WAKE
                # because that is the only moment it is useful — assembled from
                # the ledger, since it cannot be reconstructed from log lines
                # that have already scrolled past. Silent when he did nothing:
                # "while you were away I did nothing" helps no one.
                from .standby_work import LEDGER
                from .proactive_policy import POLICY, Attention
                done = LEDGER.summary()
                self._say(report.spoken() + (f" {done}" if done else ""))
                if LEDGER.entries or LEDGER.blocked:
                    self.bus.log.emit(LEDGER.report())
                POLICY.set_attention(Attention.OPEN, "user came back")
            else:
                self.quiet_mode = False
                self.standby_mode = False
                self.bus.state.emit("LISTENING")
                self._say("Back with you." if was_quiet else "I'm here.")
            self.bus.log.emit("ORION: standby released — speaking freely again.")
            return True
        if self._QUIET_ON_RE.search(text):
            # Standby entry is owned by the same object that owns the wake, so
            # what is torn down and what is restored can never drift apart.
            recovery.enter_standby("spoken standby command")
            # STANDBY collapses the avatar and explicitly requests the compact
            # orb shell, so the visual and audio behaviour agree.
            self.bus.state.emit("STANDBY")
            try:
                self.bus.gui_command.emit({"action": "standby", "target": ""})
            except Exception:
                pass
            self.bus.log.emit(
                "ORION: standby — microphone disabled; use wake up or the orb to return.")
            # One short acknowledgement, then silence. Saying nothing at all
            # would leave the user unsure whether he heard the instruction.
            self._say("Standing by.")
            return True
        return False

    def _held_in_standby(self, lowered: str) -> bool:
        """True when this utterance must be met with silence.

        Standby is not "stop volunteering" — it is "say nothing at all until I
        ask for you". The two exceptions are deliberate and both are about not
        trapping the user: the wake phrase has to work, and a power command has
        to work, or standby would be a state he could not be brought out of.
        """
        if not self.standby_mode:
            return False
        text = str(lowered or "")
        if self._QUIET_OFF_RE.search(text) and _is_control_length(text):
            return False
        if re.search(r"\b(shut ?down|power off|restart|reboot|exit|quit)\b", text):
            return False
        self.bus.log.emit(f"ORION: silent in standby — {text[:60]}")
        return True

    async def _handle_power_command(self, lowered: str) -> bool:
        """Resolve a shutdown / restart / power command deterministically.

        The default meaning of a power command addressed to ORION is to bring
        *ORION himself* down (or restart him) — NOT to power off the physical
        computer.  Only an utterance that explicitly names the computer / PC /
        machine reaches a destructive host action, and even then it is armed
        behind the token-bound on-screen confirmation, never executed here.

        Returns True when the utterance was a power command (and has been
        handled), so the caller must not pass it on to the model."""
        # Quiet mode is resolved BEFORE the power classifier: "stand by" and
        # "standby" would otherwise be read as a stop/close intent and shut
        # ORION down when the user only wanted him to stop volunteering.
        if self._handle_quiet_command(lowered):
            return True

        try:
            from .system_guard import (
                ActionIntent, DESTRUCTIVE_INTENTS, classify_intent,
            )
        except Exception:
            return False
        intent = classify_intent(lowered)
        is_restart = bool(re.search(r"\b(restart|reboot)\b", lowered))

        # Explicit host power-off / reboot / sleep etc. → guarded confirmation.
        if intent in DESTRUCTIVE_INTENTS:
            guard = getattr(self.dispatcher, "system_guard", None)
            if guard is None:
                return False
            decision = guard.request_confirmation(intent, origin="voice_command")
            if decision.token:
                self.bus.confirm_action.emit({
                    "token":   decision.token,
                    "intent":  intent.value,
                    "message": decision.message,
                    "machine": decision.machine,
                })
            self._say(decision.message)
            return True

        # Self-directed: shut ORION down (or restart him).  This covers
        # CLOSE_APP / STOP_ORION and the bare, ambiguous "shut down" / "restart"
        # that has no host object — in conversation the subject is ORION.
        self_directed = intent in (
            ActionIntent.CLOSE_APP, ActionIntent.STOP_ORION, ActionIntent.AMBIGUOUS,
        )
        if not self_directed:
            return False
        # ASK FIRST. This used to power the machine down on the first mention
        # of "shut down", which made ORION something you had to watch your
        # words around. The confirmation is answered by the next yes/no (see
        # _resolve_shutdown_confirmation) and lapses on its own if ignored.
        self._ask_to_confirm_shutdown(
            is_restart=is_restart and intent is not ActionIntent.CLOSE_APP)
        return True

    def _switch_output_device(self) -> None:
        """Move ORION's voice to the next available speaker/headphone, then
        confirm on the NEW device so the user can tell it worked."""
        self._note_user_present()
        name = self.speech.switch_output_device()
        if name is None:
            self._say("That's the only speaker I can find — please check the "
                      "output device in your sound settings.")
            return
        self._say(f"Switching my voice to {name}. Can you hear me now?")

    def _switch_input_device(self) -> None:
        """Rotate the microphone to the next available input device."""
        self._note_user_present()
        if self.mic is None:
            self._say("I don't have a microphone open to switch.")
            return
        self.mic.switch_to_next_input(explicit=True)

    # ── realtime pump ─────────────────────────────────────────────────────────

    async def _send_realtime(self) -> None:
        while not self.stop_event.is_set():
            try:
                media = await asyncio.wait_for(self.out_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self.session is None:
                continue
            await self._send_media(media)

    def _record_live_usage(self, usage: Any) -> None:
        """Record the live channel's authoritative usage into the token ledger
        so Gemini Live shows on the command-deck graph like every text
        provider.  Deduped by a per-turn request id; never raises."""
        ledger = getattr(self.router, "token_ledger", None)
        profile = self.active_live_provider
        if ledger is None or usage is None or profile is None:
            return
        try:
            import uuid as _uuid

            from .token_usage import UsageRecord, mask_key
            key = (self.router.active_key(profile)
                   if hasattr(self.router, "active_key") else profile.api_key)
            ledger.record(UsageRecord(
                request_id=f"live-{_uuid.uuid4().hex}",
                provider=profile.name,
                model=getattr(self, "active_live_model", "") or profile.model,
                key_alias=mask_key(key),
                task="live_audio",
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "response_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                streaming=True,
            ))
        except Exception:
            pass

    async def _receive_realtime(self) -> None:
        in_buffer:  list[str] = []
        out_buffer: list[str] = []
        turn_usage: Any = None
        while not self.stop_event.is_set() and self.session is not None:
            async for response in self.session.receive():
                if self.stop_event.is_set():
                    return
                # usage_metadata arrives cumulatively across a turn; keep the
                # latest and record once at turn_complete.
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    turn_usage = usage
                if getattr(response, "go_away", None) is not None:
                    reason = str(getattr(getattr(response, "go_away", None), "reason", "") or "")
                    self.conn_state.set(ConnectionState.RECONNECTING,
                                        f"go_away{(': ' + reason) if reason else ''}")
                    self.bus.log.emit("NET: server requested reconnect (go_away); rotating channel.")
                    self._last_close_reason = "go_away" + (f": {reason}" if reason else "")
                    return
                resumption = getattr(response, "session_resumption_update", None)
                if resumption is not None:
                    handle = getattr(resumption, "new_handle", None)
                    if getattr(resumption, "resumable", False) and handle:
                        self._resumption_handle = str(handle)
                data = getattr(response, "data", None)
                if data:
                    self._mark_turn_progress()
                    if not self._drop_live_output:
                        # A cancelled response may carry audio AND turn_complete
                        # in this same message.  Do not skip the completion:
                        # it clears the mute before the next request arrives.
                        # While paused the renderer is HELD, preserving audio.
                        self.speech.enqueue_native_audio(data)
                        trace = getattr(self, "_trace", None)
                        if trace is not None and not self._traced_playback:
                            self._traced_playback = True
                            trace.stage(Stage.API_RESPONSE_RECEIVED)
                            trace.stage(Stage.PLAYBACK_START)
                        if not self.paused:
                            self._emit_state("SPEAKING")
                server_content = getattr(response, "server_content", None)
                if server_content is not None:
                    if getattr(server_content, "interrupted", False):
                        # Server-signalled interruption (barge-in mode) — the
                        # ONLY path allowed to cut speech short mid-utterance.
                        if self._honour_server_interruption():
                            self.speech.interrupt_all()
                            out_buffer = []
                            self.bus.log.emit("AUDIO: response interrupted by user.")
                            self._refresh_wake_window()
                            if self.connected:
                                self._emit_state("LISTENING")
                        else:
                            # Finish-speaking mode: let him complete the sentence
                            # instead of cutting it (his own echo was clipping it).
                            self.bus.log.emit(
                                "AUDIO: interruption ignored — finishing the sentence "
                                "(finish-speaking mode).")
                    output_text = self._extract_transcription(server_content, "output_transcription")
                    input_text  = self._extract_transcription(server_content, "input_transcription")
                    if output_text:
                        out_buffer.append(output_text)
                        # Feed the echo comparator (_is_own_echo) so hardened
                        # barge-in (Mark X.9) can tell a real interruption
                        # apart from ORION's own native-audio voice bleeding
                        # into the mic — _say() already does this for the
                        # local-voice fallback path; the native channel is
                        # the primary voice and was previously blind here.
                        self._last_spoken_norm = self._normalise_phrase(
                            " ".join(out_buffer)
                        )
                        self._spoken_at = time.monotonic()
                    if input_text:
                        in_buffer.append(input_text)
                        # The Brain page's Hear lane: the Live path has no
                        # other sign that the user is mid-sentence. No text.
                        try:
                            self.bus.dashboard_event.emit("voice_heard", None)
                        except Exception:
                            pass
                        # The Live model hears the words directly, so a mute
                        # is caught on the transcript as it streams in — before
                        # the reply starts playing, not after it has finished.
                        self._handle_mute_command(" ".join(in_buffer).lower(),
                                                  announce=False)
                        self._refresh_wake_window()
                        self._note_user_present()  # the user is speaking to me
                    if getattr(server_content, "turn_complete", False):
                        if in_buffer:
                            user_text = " ".join(in_buffer).strip()
                            self.bus.log.emit(f"YOU: {user_text}")
                            self._persist_episode("user", user_text)
                            # Held so a tool call later in this turn can be
                            # attributed to the words that caused it — the raw
                            # material for learned reflexes.
                            self._reflex_utterance = user_text
                            # Empathy: the USER's words drive expression too —
                            # stress in your voice reads as concern on his face.
                            try:
                                from .emotion import SentimentAnalyser
                                SentimentAnalyser.broadcast(self.bus, user_text,
                                                            origin="user")
                            except Exception:
                                pass
                            # Which language this household actually speaks.
                            # A live reply already adapts on its own; what does
                            # not is everything ORION opens — a briefing, an
                            # alert, a greeting — because those have no user
                            # turn to take a cue from. Learned silently, and
                            # only after several turns agree.
                            try:
                                if self._language.observe(user_text):
                                    self.bus.log.emit(
                                        f"LANG: you speak {self._language.name} "
                                        f"— I'll open in that from now on.")
                            except Exception:
                                pass
                            in_buffer = []
                        if out_buffer:
                            orion_text = " ".join(out_buffer).strip()
                            self.bus.log.emit(f"ORION: {orion_text}")
                            self._persist_episode("orion", orion_text)
                            # The native voice channel drives ORION's facial
                            # expression: classify what he just said and
                            # broadcast it (local, instant, no model call).
                            try:
                                from .emotion import SentimentAnalyser
                                SentimentAnalyser.broadcast(self.bus, orion_text,
                                                            origin="orion")
                            except Exception:
                                pass
                            out_buffer = []
                        self._refresh_wake_window()
                        self._record_live_usage(turn_usage)
                        turn_usage = None
                        # The reply arrived — this turn is done; clear it so the
                        # watchdog and reconnect logic stop tracking it, and drop
                        # any lingering cancel-mute.
                        self._clear_turn()
                        self._drop_live_output = False
                        self._traced_playback = False
                        # Playback usually outlives turn_complete; only flip to
                        # LISTENING when the speech queue is already drained —
                        # otherwise the SpeechQueueManager transition handles it
                        # the moment the final syllable finishes.
                        if (self.connected and not self.standby_mode
                                and not self.speech.output_active()):
                            self._emit_state("LISTENING")
                tool_call = getattr(response, "tool_call", None)
                if tool_call is not None:
                    self._mark_turn_progress()
                    await self._handle_tool_call(tool_call)

    async def _handle_tool_call(self, tool_call: Any) -> None:
        """Execute the turn's tool calls, running independent ones together.

        The Live channel routinely asks for several tools at once. This loop
        used to await each in strict sequence, so three independent two-second
        lookups cost six seconds. They are now grouped by
        ``concurrency.plan_batches``: adjacent read-only calls run in one
        ``gather``, while anything that touches the machine keeps its own batch
        and its original position. Ordering of side-effecting work is therefore
        unchanged — only the waiting between calls that cannot interfere is
        removed.
        """
        calls = list(getattr(tool_call, "function_calls", []) or [])
        self.tool_busy = True
        self._mark_turn_progress()
        if self.mic is not None:
            self.mic._drain()
        self._emit_state("PROCESSING")
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace.stage(Stage.TOOL_EXECUTION_START,
                        ",".join(str(getattr(fc, "name", "")) for fc in calls)[:80])
        try:
            names = [str(getattr(fc, "name", "")) for fc in calls]
            # Arguments matter to the plan: a nominally read-only tool steered
            # onto a write branch (awareness add_task, second_brain ingest)
            # must keep its own batch.
            call_args = [dict(getattr(fc, "args", {}) or {}) for fc in calls]
            batches = plan_batches(names, args=call_args)
            if any(len(batch) > 1 for batch in batches):
                self.bus.log.emit(
                    f"TOOL: {len(calls)} calls — {describe_plan(names, batches)}")

            # Results are collected by ORIGINAL index: the model matched each
            # response to its call id, and a reordered response list would pair
            # answers with the wrong questions.
            outcomes: list[tuple[dict[str, Any], dict[str, Any] | None]] = [
                ({}, None) for _ in calls
            ]
            for batch in batches:
                if len(batch) == 1:
                    index = batch[0]
                    outcomes[index] = await self._run_tool_call(calls[index])
                    continue
                gathered = await asyncio.gather(
                    *(self._run_tool_call(calls[i]) for i in batch),
                    return_exceptions=True,
                )
                for index, outcome in zip(batch, gathered):
                    if isinstance(outcome, BaseException):
                        if isinstance(outcome, asyncio.CancelledError):
                            raise outcome
                        # gather() with return_exceptions must never lose a
                        # call: an unanswered id stalls the model's turn.
                        self.bus.log.emit(
                            f"TOOL: {names[index]} failed - {first_line(outcome)}")
                        outcomes[index] = (
                            {"ok": False,
                             "result": f"{names[index]} failed: {first_line(outcome, 200)}"},
                            None,
                        )
                    else:
                        outcomes[index] = outcome
                self._mark_turn_progress()

            # Learn the shortcut — but only from an UNAMBIGUOUS turn. A turn
            # that used two tools does not tell us which one the phrase meant,
            # and guessing is exactly what this design refuses to do. The
            # read-only gate and the three-time threshold live in the learner.
            utterance_seen = getattr(self, "_reflex_utterance", "")
            if len(calls) == 1:
                utterance = getattr(self, "_reflex_utterance", "")
                if utterance:
                    payload = outcomes[0][0] if outcomes and outcomes[0] else None
                    succeeded = bool((payload or {}).get("ok", True))
                    self.observe_for_reflex(utterance, names[0],
                                            dict(getattr(calls[0], "args", {}) or {}),
                                            ok=succeeded)
                    self._reflex_utterance = ""

            # Shadow-evaluate the tool resolver on this real turn. It changes
            # nothing — it records whether the pre-filter WOULD have kept the
            # tool the model actually chose, which is the evidence needed to
            # decide whether ORION_TOOL_RESOLVER can safely be switched on.
            # Every call is scored, including multi-tool turns: recall is a
            # per-tool property.
            shadow_query = getattr(self, "_reflex_utterance", "") or utterance_seen
            if shadow_query:
                for used in names:
                    self.observe_resolver_shadow(shadow_query, used)

            # Media is streamed AFTER the batch, in call order, so concurrent
            # tools never interleave frames on the wire.
            function_responses: list[Any] = []
            for fc, (payload, media) in zip(calls, outcomes):
                if media:
                    await self._send_media(media)
                    # An IMAGE (screen/camera/photo) is delivered as an ambient
                    # video frame; the model tends to answer from the tool's text
                    # alone and claim it "cannot see" the image until a SECOND
                    # prompt puts the frame in context. Nudge it, in the tool
                    # response, to look at the frame it has just been handed and
                    # describe it now — the fix for "he searched but couldn't see
                    # it the first time".
                    payload = self._nudge_payload_for_image(
                        payload, str(media.get("mime_type", "")))
                function_responses.append(
                    self._function_response(
                        getattr(fc, "id", None), str(getattr(fc, "name", "")), payload))

            if function_responses and self._transport_ready():
                async with self._send_lock:
                    session = self.session
                    if session is not None and not self._send_closed:
                        try:
                            await session.send_tool_response(function_responses=function_responses)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            self.bus.log.emit(f"NET: tool response not delivered - {first_line(exc)}")
        finally:
            self.tool_busy = False
            if trace is not None:
                trace.stage(Stage.TOOL_EXECUTION_COMPLETE)
            if self.connected and not self.speech.output_active():
                self._emit_state("LISTENING")

    async def _run_tool_call(self, fc: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Run one function call, returning its (payload, media).

        Media is returned rather than sent, so the caller can stream frames in
        call order even when the calls themselves ran concurrently. Every
        failure becomes a payload: a call that produced no response leaves the
        model waiting on an id that will never be answered.
        """
        name = str(getattr(fc, "name", ""))
        args = dict(getattr(fc, "args", {}) or {})
        self.bus.log.emit(f"TOOL: {name} requested.")
        # The receive loop waits on this call, and the model waits on its
        # answer. A tool that never returned (a desktop action waiting on a
        # window, a hung page) therefore froze the Live channel for good: the
        # watchdog recovered the GUI but nothing ever read the socket again,
        # and only a restart brought ORION back. Past the limit the model is
        # answered now and the tool finishes in the background — nothing is
        # cancelled half-way through.
        task = asyncio.ensure_future(self._dispatch_live_tool(name, args))
        started = time.monotonic()
        try:
            done, _ = await asyncio.wait({task}, timeout=self.LIVE_TOOL_ANSWER_S)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task in done:
            elapsed = time.monotonic() - started
            if elapsed >= 10.0:
                _live_diag("tool_slow", tool=name, seconds=round(elapsed, 1))
            return task.result()
        self.bus.log.emit(
            f"TOOL: {name} still running after {self.LIVE_TOOL_ANSWER_S:.0f}s - "
            "answering now and finishing it in the background.")
        _live_diag("tool_detached", tool=name, seconds=self.LIVE_TOOL_ANSWER_S)
        task.add_done_callback(lambda t: self._report_detached_tool(name, t))
        return {
            "ok": True,
            "result": (f"{name} is still running in the background. Tell the user "
                       "briefly that it is in progress and that you will say when "
                       "it is done. Do not call it again for this request."),
        }, None

    #: How long a Live tool call may hold the model's turn before it is
    #: answered and left to finish in the background (ORION_LIVE_TOOL_ANSWER_S).
    try:
        LIVE_TOOL_ANSWER_S = max(5.0, float(os.getenv("ORION_LIVE_TOOL_ANSWER_S", "45")))
    except ValueError:
        LIVE_TOOL_ANSWER_S = 45.0

    async def _dispatch_live_tool(self, name: str, args: dict[str, Any]
                                  ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        from . import speaker_gate

        try:
            with speaker_gate.voice_turn():
                result = await self.dispatcher.dispatch_chain(name, args)
            return result.response_payload(), result.media
        except asyncio.CancelledError:
            raise
        except SecurityViolation as exc:
            self.bus.log.emit(f"SEC: {exc}")
            return {"ok": False, "result": str(exc)}, None
        except Exception as exc:
            self.bus.log.emit(f"TOOL: {name} failed - {exc}")
            return {"ok": False, "result": f"{name} failed: {exc}"}, None

    def _report_detached_tool(self, name: str, task: "asyncio.Future[Any]") -> None:
        """Say how a tool that outlived its Live turn finally ended."""
        if task.cancelled():
            return
        try:
            payload, _media = task.result()
        except Exception as exc:          # _dispatch_live_tool already caught most
            payload = {"ok": False, "result": first_line(exc, 200)}
        _live_diag("tool_finished_late", tool=name,
                   ok=bool((payload or {}).get("ok", True)))
        text = first_line(str((payload or {}).get("result", "") or ""), 400)
        label = name.replace("_", " ")
        if (payload or {}).get("ok", True):
            self.announce(f"The {label} task has finished. {text}".strip())
        else:
            self.announce(f"The {label} task failed. {text}".strip())

    @staticmethod
    def _nudge_payload_for_image(payload: Any, mime_type: str) -> Any:
        """Append a 'look at the attached frame now' instruction to an image
        tool's response, so the model describes what it was just shown instead
        of answering from the text and claiming it cannot see it. Pure and
        idempotent; leaves non-image or non-dict payloads untouched."""
        if not str(mime_type or "").startswith("image/"):
            return payload
        if not isinstance(payload, dict):
            return payload
        note = ("[An image frame has been attached to this turn. Look at it now "
                "and describe what you actually see in it to answer — do not say "
                "you are unable to see it.]")
        result = str(payload.get("result", "") or "")
        if note in result:
            return payload
        updated = dict(payload)
        updated["result"] = (result + " " + note).strip()
        return updated

    def _function_response(self, call_id: Any, name: str, payload: dict[str, Any]) -> Any:
        try:
            return types.FunctionResponse(id=call_id, name=name, response=payload)
        except Exception:
            return {"id": call_id, "name": name, "response": payload}

    def _transport_ready(self) -> bool:
        """True when a live session is established and its transport is open.
        Every outbound path checks this so a send scheduled just as the channel
        drops (go_away, reconnect, shutdown) becomes a clean no-op."""
        return self.session is not None and not self._send_closed and not self.stop_event.is_set()

    async def _send_media(self, media: dict[str, Any]) -> None:
        """Stream one media frame to the live session on the correct typed input.

        This MUST route by mime type.  The generic ``media=`` chunk is treated by
        the server as AUDIO by default, so pushing a camera/screen JPEG through it
        made the native-audio model reject the whole turn with
        ``1007 — CONTENT_TYPE_AUDIO is not supported`` and drop the channel.
        Image frames are delivered as a single ``video=`` frame (the Live API's
        camera/screen path); audio goes to ``audio=``; anything else falls back
        to the generic chunk."""
        if not self._transport_ready():
            return
        if not isinstance(media, dict) or "data" not in media:
            # Only a typed frame can go down this channel. Anything else (a
            # tool once returned a file PATH here) is dropped, not sent.
            return
        session = self.session
        async with self._send_lock:
            if session is None or session is not self.session or not self._transport_ready():
                return
            mime = str(media.get("mime_type", "") or "")
            blob = types.Blob(
                data=media["data"],
                mime_type=mime or "application/octet-stream",
            )
            try:
                if mime.startswith("audio/"):
                    await session.send_realtime_input(audio=blob)
                elif mime.startswith("image/"):
                    # Camera / screen frames are VIDEO input, never a generic
                    # (audio-defaulted) media chunk — see the docstring.
                    await session.send_realtime_input(video=blob)
                else:
                    await session.send_realtime_input(media=blob)
            except asyncio.CancelledError:
                raise
            except TypeError:
                # Older SDK without the typed params: fall back to a raw chunk.
                await session.send_realtime_input(media=blob)

    def _extract_transcription(self, server_content: Any, attr: str) -> str:
        item = getattr(server_content, attr, None)
        text = getattr(item, "text", "") if item is not None else ""
        return clean_transcript(text) if text else ""

    # ── live config (voice permanently locked to the profile) ────────────────

    def _build_config(self) -> Any:
        """Build a Pylance-clean Gemini LiveConnectConfig."""
        system_instruction = self.router.system_instruction()
        from . import tool_gateway
        # Mark XXI, Track E1: the dispatcher's own per-instance list, when
        # present, includes whatever MCP servers connected + forged tools
        # activated after boot — the static module-level list is the
        # fallback for a bare/test dispatcher that never got one.
        declarations = getattr(self.dispatcher, "TOOL_DECLARATIONS", None) or TOOL_DECLARATIONS
        stage = getattr(self, "_live_config_stage", 0)
        if stage and _declaration_fingerprint(declarations) != getattr(
                self, "_live_config_fp", None):
            # The tool set changed since a config was refused (a server
            # reconnected, a tool was forged or fixed): give the full set
            # another chance rather than staying on the fallback all session.
            stage = self._live_config_stage = 0
        if stage >= 1:
            # A config refused at setup (see run()) — the built-in tools are
            # known-good, so the channel comes up on those rather than not
            # at all.
            declarations = TOOL_DECLARATIONS
        # The core set plus find_tool/use_tool (see tool_gateway): ~40k fewer
        # prompt tokens on EVERY spoken turn, measured; nothing unreachable.
        gated = tool_gateway.live_declarations(declarations)
        if len(gated) < len(declarations):
            system_instruction = f"{system_instruction}\n{tool_gateway.GATEWAY_NOTE}"
        declarations = _live_safe_declarations(gated, getattr(self, "bus", None))
        tools: list[Any] = [{"function_declarations": declarations}]
        if self._search_tool_enabled:
            # Google Search grounding: real-time knowledge alongside local tools.
            tools.insert(0, {"google_search": {}})
        # The voice name comes from the frozen VOICE_PROFILE — the single
        # source of truth.  No configuration file or runtime path can vary it.
        voice_name = VOICE_PROFILE.gemini_voice_name
        # Finish-speaking (Mark XXII): tell the SERVER not to interrupt ORION's
        # turn when its VAD hears activity. This is the real fix for "the text
        # log shows he finished but his voice cut off" — Gemini's default VAD was
        # ending his audio turn early on faint echo/room noise. NO_INTERRUPTION
        # lets him complete the utterance. Barge-in mode leaves the default.
        realtime_input_config = None
        if getattr(self, "finish_speaking", False):
            try:
                realtime_input_config = types.RealtimeInputConfig(
                    activity_handling=types.ActivityHandling.NO_INTERRUPTION)
            except Exception:
                realtime_input_config = None
        try:
            kwargs: dict[str, Any] = dict(
                response_modalities=["AUDIO"],
                # Transcriptions are what feed the YOU:/ORION: console log and
                # episodic memory — without requesting them the server sends none.
                input_audio_transcription={},
                output_audio_transcription={},
                system_instruction=system_instruction,
                tools=tools,
                session_resumption=types.SessionResumptionConfig(handle=self._resumption_handle),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                ),
            )
            if realtime_input_config is not None:
                kwargs["realtime_input_config"] = realtime_input_config
            compression = _context_window_compression() if stage < 2 else None
            if compression is not None:
                kwargs["context_window_compression"] = compression
            return types.LiveConnectConfig(**kwargs)
        except Exception:
            config = {
                "response_modalities": ["AUDIO"],
                "input_audio_transcription": {},
                "output_audio_transcription": {},
                "system_instruction": system_instruction,
                "tools": tools,
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": voice_name}
                    }
                },
            }
            if getattr(self, "finish_speaking", False):
                config["realtime_input_config"] = {"activity_handling": "NO_INTERRUPTION"}
            if stage < 2:
                config["context_window_compression"] = {"sliding_window": {}}
            return config
