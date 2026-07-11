"""
O.R.I.O.N. Mark VIII — Open Resolution Intelligence Overt Network.

A modular personal operating-system AI. The package is organised into
independent services and managers so each capability can evolve without
touching the others:

    constants     — application constants, colour palette, voice profile
    utils         — small shared helpers
    security      — regex/AST firewall for all OS action payloads
    bus           — Qt signal hub decoupling every subsystem
    memory        — persistent SQLite FTS5 matrix + MemoryAgent (session layer)
    audio         — capture, VAD, recognition, playback, SpeechQueueManager
    vision        — screen grabber, file intelligence, VisionAgent
    providers     — provider profiles, configuration, ProviderRouter
    outlook       — Outlook COM integration (read/draft/summarise/send)
    notion        — Notion REST integration (tasks/calendar/projects)
    agents        — BaseAgent, AgentManager, DesktopAgent, specialists
    briefing      — Morning Briefing System
    dispatcher    — tool dispatcher + Gemini function declarations
    live_worker   — Gemini Live session worker and offline voice loop
    remote        — opt-in browser/mobile uplink
    gui           — dual-window Qt shell (core window + widget dashboard)
    app           — application bootstrap
    web_automation — Playwright-style browser automation wrapper
    peripherals   — audio, power and network control helpers
    messaging     — outbound WhatsApp/Telegram routing gateway
    gaming        — local Steam/Epic gaming client indexer
    entertainment — YouTube summaries and trending helpers

Entry point:  python orion.py   (thin launcher kept for compatibility)
"""

__version__ = "10.9.0"
__codename__ = "Mark X.9 — Sharper Senses"
