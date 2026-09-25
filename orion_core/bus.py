"""
OrionBus — the Qt signal hub that decouples every subsystem.

Audio threads, the live worker, agents and integrations never touch widgets
directly; they emit signals here and any window (core or dashboard) that
cares connects to them.  This is the single seam that lets the dual-window
GUI, the HUD and the headless services evolve independently.

New in Mark VIII:
    speaking        — True while ORION is audibly speaking (either channel);
                      drives the half-duplex microphone gate indicator.
    agent_activity  — (agent_name, summary) whenever a specialist agent runs.
    dashboard_event — (channel, payload) generic feed for dashboard panels,
                      e.g. ("emails", [...]), ("tasks", [...]), ("briefing", str).
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from .data import ToolResult  # re-export for legacy imports  # noqa: F401


class OrionBus(QObject):
    log              = pyqtSignal(str)
    state            = pyqtSignal(str)
    amplitude        = pyqtSignal(float)
    # Mark X.14: spectral profile of the SAME audio amplitude already covers —
    # a dict of {"low","mid","high"} band-energy ratios, so the avatar's mouth
    # can shape itself to what's actually being said instead of just how loud
    # it is. Kept as a separate signal (not a change to amplitude's float
    # signature) since amplitude has other consumers (mini-orb) that only
    # want a loudness pulse and shouldn't need to change.
    voice_spectrum   = pyqtSignal(object)
    banner           = pyqtSignal(str, int)
    mic_enabled      = pyqtSignal(bool)
    request_shutdown = pyqtSignal()
    # Mark X.6: ORION can restart himself — clean shutdown, then the launcher
    # respawns the process (also how an applied self-repair gets loaded).
    request_restart  = pyqtSignal()

    # Mark VIII additions
    speaking         = pyqtSignal(bool)
    agent_activity   = pyqtSignal(str, str)
    dashboard_event  = pyqtSignal(str, object)

    # Mark IX+: spoken/button pause control (True = paused)
    paused           = pyqtSignal(bool)

    # Globe: ORION (or the dispatcher) asks the GUI globe to fly to a place and
    # show regional news.  Payload is a place name string (a city, region,
    # country, or a postcode / ZIP).
    globe_request    = pyqtSignal(str)

    # Globe camera control: zoom the on-screen globe without changing location.
    # Payload is a directive string — "in", "out", or "reset".
    globe_zoom       = pyqtSignal(str)

    # Globe maximum zoom (max_zoom_in_globe): drive the camera to the closest
    # zoom the active backend supports, under bounded/cancellable control.
    # Payload is an options dict (may be empty) forwarded to the zoom controller.
    globe_max_zoom   = pyqtSignal(object)

    # Self-GUI navigation: ORION drives his OWN interface (switch core view,
    # open a deck page — widgets/toolkit/studio/command/globe/diagnostics —
    # toggle the dashboard, refresh the environment, etc.) programmatically
    # rather than by pixel-clicking his own window.  Payload is a dict
    # {"action": str, "target": str}.  Handled on the GUI thread by CoreWindow.
    gui_command      = pyqtSignal(object)

    # Proactive voice: any subsystem (reminders, sentinel, protocols, presence)
    # can make ORION speak unprompted by emitting the text to announce.
    speak_request    = pyqtSignal(str)
    #: Mute ORION's VOICE (not the PC, not his hearing): the HUD button asks
    #: through voice_mute_request; voice_muted reports the state to every view.
    voice_mute_request = pyqtSignal(bool)
    voice_muted      = pyqtSignal(bool)
    #: Danger that must reach the user whatever state ORION is in — routed to
    #: GenAILiveWorker.alert(), which ignores standby, quiet mode and pause.
    #: Deliberately separate from speak_request: that lands on announce(),
    #: which is correctly silenced by exactly those states.
    safety_alert     = pyqtSignal(str)

    # Host telemetry samples (cpu/ram/net_percent/...), emitted every ~0.75s by
    # the Core Window's background loop.  The TELEMETRY deck page self-wires to
    # this rather than the Core Window holding a direct widget reference —
    # the face-only Core Window doesn't own that view any more.
    telemetry_sample = pyqtSignal(object)

    # Manual audio-device selection: the user picks ORION's microphone or speaker
    # directly from the GUI (he sometimes locks onto the wrong device).  Payload
    # is (kind, spec) where kind is "input"/"output" and spec is a device index,
    # a name fragment, or "default" to follow the system default.  The live
    # worker applies it to the running capture/playback streams.
    audio_device_request = pyqtSignal(str, object)

    # Connection state (Section 4): emitted on every live-channel/failover state
    # transition (connecting, connected, reconnecting, cooling down, rotating
    # credential, switching provider, local fallback, degraded/offline, shutting
    # down).  Payload is the state machine's describe() dict for the diagnostics
    # UI and concise user-facing status.
    connection_state = pyqtSignal(object)

    # Provider degraded mode (Section 2): emitted when NO text provider is
    # currently usable (so model-dependent work is paused) and again on
    # recovery.  Payload is a dict {"degraded": bool, "reason": str,
    # "configured": [names], "alternatives": [hints]}.  Deterministic tools,
    # monitoring and the UI stay responsive throughout.
    provider_degraded = pyqtSignal(object)

    # Provider diagnosis (Section 5): a classified, actionable fault
    # {"category","detail","remediation","provider","at"} for the diagnostics UI.
    provider_diagnosed = pyqtSignal(object)

    # Destructive-action confirmation (Section 10): a pending OS shutdown/
    # restart/logout/sleep/hibernate (or other destructive command) needs
    # explicit human approval.  Payload is a dict
    # {"token", "intent", "message", "machine"}; the GUI raises an accessible
    # confirmation dialog and, only on approval, calls back with the token.
    confirm_action   = pyqtSignal(object)

    # Creative Audio Workspace: real-time rendering/telemetry stream.
    # Payload: (phase, data) e.g. ("processing", {"file": ..., "progress": ...}).
    audio_studio_activity = pyqtSignal(str, object)

    # Autonomous control activity — makes the visible cursor halo flare when
    # ORION moves the mouse, clicks or types.  Payload: a short action label.
    control_activity = pyqtSignal(str)
    #: Showing his working: a caption for the pointer overlay while ORION
    #: drives the desktop. Payload: {text, x, y, region} (region = x,y,w,h
    #: of a window he is switching to or typing into, or None).
    control_narration = pyqtSignal(object)

    # Browser co-pilot "showing his workings": one human-readable step per
    # action as ORION drives a live browser (navigate/scroll/highlight/read/
    # click/research).  Payload dict: {"step": str, "detail": str, "url": str}.
    browser_step = pyqtSignal(object)

    # ── Mark X.7: sentiment-driven expression (Phases 4–5) ────────────────────
    # Emitted by whichever channel produced or received language (ProviderRouter
    # text path, the live worker's spoken turns, the user's own words):
    # (sentiment, confidence 0..1), e.g. ("analytical", 0.72).
    sentiment_changed = pyqtSignal(str, float)
    # The full specification payload for the same event:
    # {"sentiment", "confidence", "intensity", "reason", "origin"}.
    # The EmotionStateManager consumes this richer form.
    sentiment_payload = pyqtSignal(object)
    # Emitted by the EmotionStateManager whenever ORION's emotional state
    # changes: (emotion_name, parameter_dict). The avatar widgets subscribe
    # and re-render — services never touch a widget, widgets never poll.
    emotion_changed  = pyqtSignal(str, object)

    # ── speaker gender recognition (#14) ──────────────────────────────────────
    # Emitted by the SpeakerGenderTracker when the voice it is hearing settles
    # on a likely gender: {"label" ("male"|"female"|"unknown"), "f0", "confidence",
    # "at"}.  Input-side only — ORION recognising who is speaking, never changing
    # his own locked voice.
    speaker_identified = pyqtSignal(object)

    # ── free-thinking stream ──────────────────────────────────────────────────
    # Emitted by the ThoughtStream whenever ORION forms an inner thought:
    # {"at", "kind" ("reflection"|"decision"), "text", "model"}.  Read-only
    # introspection — a thought never triggers an action by itself.
    thought          = pyqtSignal(object)
    # Streaming inner voice: emitted token-by-token WHILE a thought is being
    # generated, so the GUI types it out in true real time rather than pasting a
    # finished paragraph.  Payload dict:
    #   {"id": int, "phase": "start"|"delta"|"end", "text": str,
    #    "kind": "reflection"|"decision", "model": str, "at": str}
    # The complete thought still arrives once on `thought` (for journals/mirrors).
    thought_delta    = pyqtSignal(object)

    # ── avatar face tracking ──────────────────────────────────────────────────
    # One sample per webcam frame from the FaceTracker daemon thread:
    # {"present": bool, "x": float, "y": float, "size": float} with x/y in
    # [-1, 1].  Queued delivery moves it onto the GUI thread for the
    # AvatarController's head-follow.
    face_tracking    = pyqtSignal(object)

    # ── lip-sync (Mark XXVI) ──────────────────────────────────────────────────
    # A mouth posture for the current speech sound, emitted by the viseme source
    # (SAPI events or a text-driven schedule) and consumed by whichever face is
    # live. Payload: {"viseme": str, "weight": float}.
    viseme           = pyqtSignal(object)

    # ── camera visibility ─────────────────────────────────────────────────────
    # A frame ORION actually captured, so the user can SEE what he saw rather
    # than only reading a description of it.  Payload dict:
    # {"kind": "snapshot", "jpeg": bytes, "note": str} — emitted by the vision
    # layer at capture time and rendered by gui.camera_preview.
    camera_frame     = pyqtSignal(object)

    # ── found material ────────────────────────────────────────────────────────
    # Rows that have a title and usually a link — news stories, research
    # sources — on their way to the content panel. ORION already fetched these;
    # before the panel existed they were spoken aloud and written to the log,
    # where a headline cannot be clicked and the third one has already pushed
    # the first out of sight. Payload: a list of dicts carrying at least
    # "title", and normally "url" and "topic".
    content_results  = pyqtSignal(object)

    # Electronics inspection: one request owns one immutable captured frame.
    # Lifecycle events carry request_id so a late result cannot replace a
    # newer capture. Results contain the report and exact submitted JPEG.
    electronics_scan_requested = pyqtSignal(object)
    electronics_scan_started = pyqtSignal(object)
    electronics_scan_result = pyqtSignal(object)
    electronics_scan_error = pyqtSignal(object)

    # ── phone hand-off ────────────────────────────────────────────────────────
    # A native action ORION wants the paired phone to perform through its app
    # bridge (window.OrionNative): dial, text, e-mail, navigate, share, open.
    # Payload dict: {"kind": "call"|"sms"|"email"|"navigate"|"map"|"openUrl"
    #   |"share", plus the relevant fields: number/body/to/subject/query/url/
    #   text}.  The remote gateway fans it out to connected phones over SSE;
    #   the phone opens the app pre-filled and the user confirms the final tap.
    phone_action     = pyqtSignal(object)
    #: The Command Deck asks the core window to open the search palette. The
    #: palette lives on the core window (it prefills its message box); the deck
    #: is a separate window, so a signal is how its visible Search bar reaches
    #: it without the two windows holding references to each other.
    open_palette     = pyqtSignal()
