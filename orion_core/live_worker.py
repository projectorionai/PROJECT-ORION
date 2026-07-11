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

from google import genai
from google.genai import types

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
from .constants import (
    LIVE_MODEL,
    LIVE_MODEL_FALLBACKS,
    MIC_QUEUE_LIMIT,
    PAUSE_WORDS,
    RESUME_WORDS,
    STARTUP_GREETINGS,
    VOICE_PROFILE,
    WAKE_WINDOW_SECONDS,
    WAKE_WORDS,
)
from .dispatcher import TOOL_DECLARATIONS, OrionDispatcher
from .memory import MemoryAgent
from .providers import AIProviderProfile, OrionProviderSettings, ProviderRouter
from .security import SecuritySanitiser, SecurityViolation
from .utils import clean_transcript, first_line
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


class GenAILiveWorker:
    """Realtime provider session with speech-queue-governed voice output."""

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
        # Startup briefing consent: ORION asks first, then waits for a yes/no.
        self.awaiting_briefing = False
        # ── unified speech pipeline (event-driven audio state machine) ────────
        self.audio_state = AudioStateMachine(bus, telemetry)
        self.playback = AudioPlaybackThread(bus, self.audio_state, telemetry)
        self.tts      = SpeechSynthesiser(bus, self.audio_state, telemetry)
        self.speech   = SpeechQueueManager(bus, self.playback, self.tts, self.audio_state, telemetry)
        self.speech.state_cb = self._on_speaking_changed_threadsafe
        # ── session state ─────────────────────────────────────────────────────
        self.session: Any = None
        self.mic: MicrophoneEngine | None = None
        self.out_queue: asyncio.Queue = asyncio.Queue(maxsize=MIC_QUEUE_LIMIT)
        self.stop_event         = asyncio.Event()
        self.microphone_enabled = True
        self.connected          = False
        self.tool_busy          = False
        self.vad                = SileroVADGatekeeper(bus, threshold=0.50)
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
        self._search_tool_enabled = True
        self._last_state       = ""
        self.active_live_provider: AIProviderProfile | None = None
        self._no_live_notice_sent = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # The dispatcher's morning_briefing tool re-enters through this hook.
        dispatcher.on_briefing_request = self.deliver_briefing_on_demand
        # Proactive-voice channel: reminders, sentinel, protocols, presence.
        self.bus.speak_request.connect(self.announce)

    # ── state plumbing ────────────────────────────────────────────────────────

    def _emit_state(self, state: str) -> None:
        """Emit state only on change — audio streaming otherwise floods the GUI
        with hundreds of identical SPEAKING signals per second."""
        if state != self._last_state:
            self._last_state = state
            self.bus.state.emit(state)

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
            asyncio.create_task(asyncio.to_thread(self.memory.log_episode, role, text))
        except RuntimeError:
            # No running loop (shutdown) — fall back to a direct write.
            self.memory.log_episode(role, text)

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

    def announce(self, text: str) -> None:
        """
        Proactive, unprompted speech (reminders, sentinel alerts, protocol
        completion) — the JARVIS "Sir, …" channel.  When the live channel is up
        the note is relayed through the model so it comes out in ORION's native
        voice and register; otherwise the local voice speaks it.  Suppressed
        while paused so a pause is truly silent.
        """
        text = str(text or "").strip()
        if not text or self.paused or self.stop_event.is_set():
            return
        self.bus.log.emit(f"ORION (proactive): {text}")
        if self.session is not None and self.connected:
            instruction = (
                "Relay this to the user now, briefly and in your own voice, as a "
                f"proactive note — do not add commentary beyond it: {text}"
            )
            try:
                asyncio.create_task(self._send_text_turn(instruction))
            except RuntimeError:
                self._say(text)
        else:
            self._say(text)

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
        # An explicit typed/sent command re-engages ORION if he was paused.
        if self.paused:
            self.resume()
        # If ORION offered a briefing and is awaiting consent, resolve it here.
        if self.awaiting_briefing:
            consent = self._briefing_consent(text)
            if consent is True:
                self.awaiting_briefing = False
                self._persist_episode("user", text)
                await self.deliver_briefing_on_demand()
                return
            if consent is False:
                self.awaiting_briefing = False
                self._persist_episode("user", text)
                self._say("Very good, sir. I'll hold the briefing — just ask when you're ready.")
                return
            # Neither a clear yes nor no: treat as a normal command, stop waiting.
            self.awaiting_briefing = False
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
        self._emit_state("PROCESSING")
        await self._send_text_turn(text)

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

    async def _submit_text_fallback(self, text: str, reason: str = "") -> None:
        if self.paused:
            return
        if reason:
            self.bus.log.emit(f"NET: {reason}; routing to text fallback.")
        self._emit_state("PROCESSING")
        # 1) Cloud/local text providers first, when any are available.
        if self.router.has_text_fallback():
            try:
                profile, response = await self.router.generate_text(text)
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
            self.speech.speak_text("Back with you, sir.")

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

    def _briefing_consent(self, text: str) -> bool | None:
        """Return True (yes), False (no), or None (ambiguous) for the briefing offer."""
        low = f" {text.lower().strip()} "
        if any(f"{w}" in low for w in self._DECLINE):
            return False
        if any(f"{w}" in low for w in self._AFFIRM):
            return True
        return None

    async def _send_text_turn(self, text: str) -> None:
        try:
            await self.session.send_client_content(
                turns={"parts": [{"text": text}]}, turn_complete=True
            )
        except TypeError:
            content = types.Content(role="user", parts=[types.Part(text=text)])
            await self.session.send_client_content(turns=content, turn_complete=True)
        except Exception as exc:
            self.bus.log.emit(f"NET: manual command rejected - {exc}")
            await self._submit_text_fallback(text, reason="Native provider rejected manual command")

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
        hour = datetime.now().hour
        period = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        return random.choice(STARTUP_GREETINGS).format(period=period)

    async def offer_startup_briefing(self) -> None:
        """
        At startup ORION ASKS whether the user wants their briefing rather than
        launching into it.  On the live channel the model is instructed to ask
        and only call the morning_briefing tool on consent; offline, ORION asks
        by voice and the next affirmative reply (handled in submit_text) runs it.
        """
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline and not self.connected and not self.stop_event.is_set():
            await asyncio.sleep(0.5)
        if self.stop_event.is_set():
            return
        self._refresh_wake_window()
        greeting = await self._compose_greeting()
        self.awaiting_briefing = True
        if self.session is not None:
            instruction = (
                f'Greet the user warmly and naturally, incorporating: "{greeting}" '
                "Then ask, in one short sentence, whether they would like their "
                "briefing now (AI and technology news, markets, calendar, tasks and "
                "priority email). Do NOT deliver the briefing yet. Only if they agree, "
                "call the morning_briefing tool. If they decline, acknowledge briefly "
                "and stand by."
            )
            self._emit_state("PROCESSING")
            await self._send_text_turn(instruction)
        else:
            self._say(f"{greeting} Would you like your briefing, sir?")

    # Legacy alias retained for any external caller.
    async def deliver_startup_briefing(self) -> None:
        await self.offer_startup_briefing()

    async def deliver_briefing_on_demand(self) -> None:
        """Dispatcher hook: the user asked for the briefing mid-session."""
        self.awaiting_briefing = False
        await self._deliver_briefing(wait_for_connection=2.0)

    async def _deliver_briefing(self, wait_for_connection: float) -> None:
        try:
            briefing = await self.dispatcher.briefing.compose_source_material()
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
        greeting = await self._compose_greeting()
        instruction = self.dispatcher.briefing.delivery_instruction(greeting, briefing)
        if self.session is not None:
            self._emit_state("PROCESSING")
            await self._send_text_turn(instruction)
        else:
            self.bus.log.emit(f"BRIEF: {briefing}")
            await self._submit_text_fallback(instruction, reason="Live channel offline for briefing")

    # ── session lifecycle ─────────────────────────────────────────────────────

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.speech.start()
        self.bus.log.emit(
            "VOICE: wake-word standby "
            + ("enabled - say 'Orion' to open the channel."
               if self.wake_mode_enabled
               else "disabled - microphone always live (set ORION_WAKE_MODE=1 to change).")
        )
        live_index = 0
        backoff    = 2.0
        while not self.stop_event.is_set():
            profiles = self.router.live_profiles()
            if not profiles:
                self.connected = False
                self.session   = None
                self._ensure_fallback_mic()
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
            client = genai.Client(api_key=profile.api_key, http_options={"api_version": "v1beta"})
            try:
                self._emit_state("CONNECTING")
                self.bus.log.emit(
                    f"NET: initialising {profile.name} live channel ({live_model.rsplit('/', 1)[-1]})."
                )
                config = self._build_config()
                async with client.aio.live.connect(model=live_model, config=config) as session:
                    self.session   = session
                    self.connected = True
                    self.active_live_provider = profile
                    self._emit_state("LISTENING")
                    self.bus.log.emit(f"NET: {profile.name} live channel synchronised.")
                    backoff = 2.0
                    live_index -= 1  # keep the working profile on clean reconnects
                    await self._session_loop()
                    self.bus.log.emit("NET: live channel closed; re-establishing.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.session   = None
                self.active_live_provider = None
                self._emit_state("STANDBY")
                message = str(exc)
                error_summary, error_type = _extract_genai_error_context(exc)
                self.bus.log.emit(f"NET: {error_summary}")
                # Log full traceback at debug level for troubleshooting
                try:
                    tb = traceback.format_exc(limit=3)
                    if "Traceback" in tb:
                        self.bus.log.emit(f"DEBUG: Live channel exception trace: {tb[:400]}")
                except Exception:
                    pass
                if self._search_tool_enabled and re.search(r"(?i)google_search", message):
                    self._search_tool_enabled = False
                    self.bus.log.emit("NET: search grounding rejected by provider; reconnecting without it.")
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
                else:
                    self.router.mark_failure(profile, exc)
                # A stale resumption handle can poison every reconnect attempt;
                # drop it so the next connection starts fresh.
                self._resumption_handle = None
                await asyncio.sleep(backoff)
                backoff = min(12.0, backoff * 1.5)
            finally:
                self.connected = False
                self.session   = None
                self.active_live_provider = None
                if self.mic is not None:
                    self.mic.stop()
                    self.mic = None
        self.speech.stop()

    async def stop(self) -> None:
        self.stop_event.set()
        if self.mic is not None:
            self.mic.stop()
        self.speech.stop()
        try:
            if self.session is not None and hasattr(self.session, "close"):
                maybe = self.session.close()
                if asyncio.iscoroutine(maybe):
                    await maybe
        except Exception:
            pass

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
        )
        self.mic.set_enabled(self.microphone_enabled)
        self.mic.start()
        send_task = asyncio.create_task(self._send_realtime(),    name="orion-send-realtime")
        recv_task = asyncio.create_task(self._receive_realtime(), name="orion-receive-realtime")
        # FIRST_COMPLETED: a receive loop that ends cleanly must also tear down
        # the send loop, otherwise the session hangs on out_queue.get() forever.
        done, pending = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc:
                raise exc

    def _can_capture_microphone(self) -> bool:
        # The half-duplex speak-then-listen rule is enforced upstream in the
        # AudioGateThread via speaking_check; here only mute, tool execution,
        # pause, shutdown, and a closed wake gate block capture.  (Local
        # recognition still runs while paused so the resume word is heard.)
        return (
            self.microphone_enabled
            and not self.tool_busy
            and not self.paused
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
            return
        wake_hit = any(word in lowered for word in WAKE_WORDS)
        # ── explicit interruption commands take precedence over everything ───
        interrupt_action = VoiceInterruptManager.classify(lowered)
        if interrupt_action:
            self._on_interrupt_phrase(interrupt_action)
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
            self._say("Yes, sir?")
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
        asyncio.create_task(self.submit_text(command))

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

    async def _receive_realtime(self) -> None:
        in_buffer:  list[str] = []
        out_buffer: list[str] = []
        while not self.stop_event.is_set() and self.session is not None:
            async for response in self.session.receive():
                if self.stop_event.is_set():
                    return
                if getattr(response, "go_away", None) is not None:
                    self.bus.log.emit("NET: server requested reconnect (go_away); rotating channel.")
                    return
                resumption = getattr(response, "session_resumption_update", None)
                if resumption is not None:
                    handle = getattr(resumption, "new_handle", None)
                    if getattr(resumption, "resumable", False) and handle:
                        self._resumption_handle = str(handle)
                data = getattr(response, "data", None)
                if data:
                    # Always enqueue — while paused the renderer is HELD, so
                    # audio streamed during a pause is preserved silently and
                    # resumes from the exact interruption point.  Dropping it
                    # here (the old behaviour) lost the rest of the response.
                    self.speech.enqueue_native_audio(data)
                    if not self.paused:
                        self._emit_state("SPEAKING")
                server_content = getattr(response, "server_content", None)
                if server_content is not None:
                    if getattr(server_content, "interrupted", False):
                        # Server-signalled interruption (barge-in mode) — the
                        # ONLY path allowed to cut speech short mid-utterance.
                        self.speech.interrupt_all()
                        out_buffer = []
                        self.bus.log.emit("AUDIO: response interrupted by user.")
                        self._refresh_wake_window()
                        if self.connected:
                            self._emit_state("LISTENING")
                    output_text = self._extract_transcription(server_content, "output_transcription")
                    input_text  = self._extract_transcription(server_content, "input_transcription")
                    if output_text:
                        out_buffer.append(output_text)
                    if input_text:
                        in_buffer.append(input_text)
                        self._refresh_wake_window()
                    if getattr(server_content, "turn_complete", False):
                        if in_buffer:
                            user_text = " ".join(in_buffer).strip()
                            self.bus.log.emit(f"YOU: {user_text}")
                            self._persist_episode("user", user_text)
                            # Empathy: the USER's words drive expression too —
                            # stress in your voice reads as concern on his face.
                            try:
                                from .emotion import SentimentAnalyser
                                SentimentAnalyser.broadcast(self.bus, user_text,
                                                            origin="user")
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
                        # Playback usually outlives turn_complete; only flip to
                        # LISTENING when the speech queue is already drained —
                        # otherwise the SpeechQueueManager transition handles it
                        # the moment the final syllable finishes.
                        if self.connected and not self.speech.output_active():
                            self._emit_state("LISTENING")
                tool_call = getattr(response, "tool_call", None)
                if tool_call is not None:
                    await self._handle_tool_call(tool_call)

    async def _handle_tool_call(self, tool_call: Any) -> None:
        function_responses: list[Any] = []
        self.tool_busy = True
        if self.mic is not None:
            self.mic._drain()
        self._emit_state("PROCESSING")
        try:
            for fc in getattr(tool_call, "function_calls", []) or []:
                name    = getattr(fc, "name", "")
                args    = dict(getattr(fc, "args", {}) or {})
                call_id = getattr(fc, "id", None)
                self.bus.log.emit(f"TOOL: {name} requested.")
                try:
                    result = await self.dispatcher.dispatch_chain(name, args)
                    if result.media:
                        await self._send_media(result.media)
                    payload = result.response_payload()
                except SecurityViolation as exc:
                    self.bus.log.emit(f"SEC: {exc}")
                    payload = {"ok": False, "result": str(exc)}
                except Exception as exc:
                    self.bus.log.emit(f"TOOL: {name} failed - {exc}")
                    payload = {"ok": False, "result": f"{name} failed: {exc}"}
                function_responses.append(self._function_response(call_id, name, payload))
            if function_responses and self.session is not None:
                await self.session.send_tool_response(function_responses=function_responses)
        finally:
            self.tool_busy = False
            if self.connected and not self.speech.output_active():
                self._emit_state("LISTENING")

    def _function_response(self, call_id: Any, name: str, payload: dict[str, Any]) -> Any:
        try:
            return types.FunctionResponse(id=call_id, name=name, response=payload)
        except Exception:
            return {"id": call_id, "name": name, "response": payload}

    async def _send_media(self, media: dict[str, Any]) -> None:
        if self.session is None:
            return
        try:
            await self.session.send_realtime_input(media=media)
            return
        except TypeError:
            pass
        blob = types.Blob(
            data=media["data"],
            mime_type=media.get("mime_type", "application/octet-stream"),
        )
        mime = media.get("mime_type", "")
        try:
            if mime.startswith("audio/"):
                await self.session.send_realtime_input(audio=blob)
            elif mime.startswith("image/"):
                await self.session.send_realtime_input(image=blob)
            else:
                await self.session.send_realtime_input(media=blob)
        except TypeError:
            await self.session.send_realtime_input(media=blob)

    def _extract_transcription(self, server_content: Any, attr: str) -> str:
        item = getattr(server_content, attr, None)
        text = getattr(item, "text", "") if item is not None else ""
        return clean_transcript(text) if text else ""

    # ── live config (voice permanently locked to the profile) ────────────────

    def _build_config(self) -> Any:
        """Build a Pylance-clean Gemini LiveConnectConfig."""
        system_instruction = self.router.system_instruction()
        tools: list[Any] = [{"function_declarations": TOOL_DECLARATIONS}]
        if self._search_tool_enabled:
            # Google Search grounding: real-time knowledge alongside local tools.
            tools.insert(0, {"google_search": {}})
        # The voice name comes from the frozen VOICE_PROFILE — the single
        # source of truth.  No configuration file or runtime path can vary it.
        voice_name = VOICE_PROFILE.gemini_voice_name
        try:
            return types.LiveConnectConfig(
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
        except Exception:
            return {
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
