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
    banner           = pyqtSignal(str, int)
    mic_enabled      = pyqtSignal(bool)
    request_shutdown = pyqtSignal()

    # Mark VIII additions
    speaking         = pyqtSignal(bool)
    agent_activity   = pyqtSignal(str, str)
    dashboard_event  = pyqtSignal(str, object)

    # Mark IX+: spoken/button pause control (True = paused)
    paused           = pyqtSignal(bool)

    # Globe: ORION (or the dispatcher) asks the GUI globe to fly to a place and
    # show regional news.  Payload is a place name string.
    globe_request    = pyqtSignal(str)

    # Proactive voice: any subsystem (reminders, sentinel, protocols, presence)
    # can make ORION speak unprompted by emitting the text to announce.
    speak_request    = pyqtSignal(str)

    # Creative Audio Workspace: real-time rendering/telemetry stream.
    # Payload: (phase, data) e.g. ("processing", {"file": ..., "progress": ...}).
    audio_studio_activity = pyqtSignal(str, object)

    # Autonomous control activity — makes the visible cursor halo flare when
    # ORION moves the mouse, clicks or types.  Payload: a short action label.
    control_activity = pyqtSignal(str)

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
