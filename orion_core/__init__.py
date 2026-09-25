"""
O.R.I.O.N. Mark XXXII — Open Resolution Intelligence Overt Network.

Copyright (C) 2026 Project ORION contributors.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your
option) any later version. It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
General Public License (LICENSE) for more details.

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
    reasoning     — budget tiers, specialist panel, red-team critique,
                    evidence-grounded verification
    strategy      — decision-space search: thousands of candidates scored
                    locally, Pareto frontier, Copeland tournament
    perception    — continuous local frame worker, motion/change events, scene
                    memory with object permanence, adaptive cloud sampling
    briefing      — Morning Briefing System
    dispatcher    — tool dispatcher + Gemini function declarations
    live_worker   — Gemini Live session worker and offline voice loop
    remote        — opt-in browser/mobile uplink
    gui           — dual-window Qt shell (core window + widget dashboard)
    app           — application bootstrap
    browser_copilot — real visible-browser CDP co-pilot (drives Chrome/Edge)
    peripherals   — audio, power and network control helpers
    messaging     — outbound WhatsApp/Telegram routing gateway
    gaming        — local Steam/Epic gaming client indexer
    entertainment — YouTube summaries and trending helpers

Entry point:  python orion.py   (thin launcher kept for compatibility)
"""

__version__ = "32.0.0"
__codename__ = "Mark XXXII — Iris & Signal Lanes"
