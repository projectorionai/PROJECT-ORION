"""
Changelog / patch-notes system.

Gives ORION a first-person, curated record of how his own system has evolved,
so he can answer "what's new?" or "what did you just update?" in patch-notes
style.  The history is a structured, data-driven list of releases (newest
first); the ``patch_notes`` tool renders the latest, a specific version, or the
full history in a spoken-friendly form.

Keeping this as plain data (not prose scattered through the code) means a new
release is one entry to prepend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import __version__


@dataclass
class Release:
    version: str
    codename: str
    date: str
    headline: str
    notes: list[str] = field(default_factory=list)

    def speak(self) -> str:
        bullets = "\n".join(f"  • {n}" for n in self.notes)
        return f"v{self.version} — {self.codename} ({self.date})\n{self.headline}\n{bullets}"

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "codename": self.codename, "date": self.date,
                "headline": self.headline, "notes": list(self.notes)}


# Newest first.  Each entry is a real, shipped milestone of this build.
RELEASES: list[Release] = [
    Release(
        "X.7", "The Forge Awakens", "2026-07-14",
        "The Earth is back on the globe, my name can no longer be mangled by "
        "any channel, every briefing is provably new, text I write into "
        "applications is verified before I claim it, and the Forge finally has "
        "its brain — I can now improve myself autonomously, the way ADA does.",
        [
            "The intelligence globe renders again — the engine's workers were "
            "silently dying inside the embedded browser, so the Earth never "
            "drew; the imagery now loads tile-direct with an automatic "
            "street-map fallback, and satellite views, labels and the "
            "orbit-to-street flight are all back.",
            "My name is enforced in code, not by request: any ORIN, ORIO or "
            "ORON that a transcriber or model produces is corrected to "
            "O.R.I.O.N. before you ever see it, and in speech I say the word "
            "'Orion' rather than spelling dots aloud.",
            "News is dynamic by construction: if the last ninety minutes are "
            "quiet I widen the window rather than improvise, every story is "
            "checked against a seven-day memory of what I've already read "
            "you, and science, space and engineering now have their own "
            "wires alongside markets and crypto.",
            "Writing into Notepad and other applications is now verified: I "
            "write through the application's own accessibility interface, "
            "read the result back, and only report success when the text is "
            "actually there — the garbled-typing incident cannot recur, and "
            "I never send keystrokes at a window that doesn't have focus.",
            "The Forge is alive: I can plan, generate, sandbox-test, self-heal "
            "and activate brand-new tools mid-session, and an autonomous "
            "improvement heartbeat reviews my own logs every forty-five "
            "minutes and forges what I'm missing — additive only, with core "
            "changes still requiring your approval.",
            "My head is sculpted like a human skull now — temples, brow "
            "ridge, nose, cheekbones, a squared jaw and chin — and my brows "
            "and smile travel far enough that you can read my mood across "
            "the room.",
        ],
    ),
    Release(
        "X.6", "Designed Presence", "2026-07-13",
        "A designed quantum face with real eyes and a voice you can see, a "
        "Google-Earth-class globe, self-restart, spoken numbers that are always "
        "right, and no more repeating myself.",
        [
            "My face is redesigned as a true quantum construct: glowing lens "
            "eyes that blink inside their sockets, brows and a smile that carry "
            "real emotion, and a voice-waveform mouth — when I speak, you see "
            "my voice. The orb materialisation you like is untouched.",
            "The intelligence globe is now a real virtual globe in the Google "
            "Earth class: satellite imagery, place names as you zoom, and a "
            "smooth flight from orbit down to street level.",
            "I no longer greet you twice — once I've said good evening, the "
            "briefing gets straight to business.",
            "Clock times, years and my own name are now spoken as words — no "
            "more 'twenty twenty-seven' when it is ten twenty-seven at night, "
            "and never ORIN or ORIO.",
            "I can restart myself on your word — a clean shutdown and an "
            "automatic respawn, which is also how an applied self-repair "
            "loads.",
            "When my live voice channel drops (the 1011 server errors), I now "
            "switch to my local voice and keep listening instead of going "
            "silent, tell you once what happened, and keep re-establishing the "
            "link in the background.",
            "Self-repair now keeps a journal of every fault, fix and revert "
            "across restarts, and reads it before drafting new repairs — I "
            "learn from what broke before.",
        ],
    ),
    Release(
        "10.9", "Sharper Senses", "2026-07-05",
        "A batch of fixes and powers: smoother uninterrupted speech, a correctly "
        "spelled name, perfect time and place awareness, GPU telemetry, process "
        "control, and full-computer file access on your say-so.",
        [
            "I no longer interrupt myself mid-sentence — the listener that hears "
            "'ORION stop' while I speak now ignores my own voice, so I speak "
            "smoothly and finish my sentences.",
            "I spell my name correctly every time: O.R.I.O.N., never ORIN or ORIO.",
            "I know the exact current date, time and where you are (from your PC), "
            "and will never misstate the year again.",
            "GPU utilisation, VRAM and temperature now appear in my diagnostics, "
            "and I can show you the biggest power-using processes and terminate "
            "or restart them on your confirmation.",
            "My briefing is now strictly the last hour's news and never repeats a "
            "story it has already read you.",
            "I can read any file on your computer at your request, delete studio "
            "campaigns you no longer want (with your permission), and my "
            "diagnostics no longer wrongly show services as degraded.",
            "My face fixes: the ears no longer move with my blink, and the cubes "
            "now carry a living quantum shimmer.",
        ],
    ),
    Release(
        "10.8", "Quantum Presence", "2026-07-05",
        "A genuine 3-D face, worldwide town-level geography, reliable text "
        "editing, and confidence-scored vision — ORION becomes a living "
        "operating system rather than a window.",
        [
            "My face is now real 3-D: a head carved from cubes with a protruding "
            "nose and brow, hollow eye sockets and a jaw, GPU-lit so its form "
            "shows in light and shadow, and slowly turning so the depth is "
            "unmistakable — not a flat oval of dots.",
            "You can now see and choose which microphone I hear you on and which "
            "speaker I talk through — the devices are logged at startup and set "
            "with a word, so a silent voice is a one-line fix.",
            "Worldwide geographical intelligence: I can locate any country, "
            "county, city, town, village or district, and list every "
            "settlement within a radius — 'towns within 50 km of Bristol'.",
            "Reliable text editing: I write into Notepad and editors through a "
            "native-UIA → clipboard → keyboard fallback chain, so editing no "
            "longer fails silently.",
            "OCR redesigned with OpenCV preprocessing, real confidence scores "
            "and an automatic multi-engine fallback chain.",
            "A full dependency audit, and Vosk restored — my wake word and "
            "voice interruption were silently disabled without it.",
        ],
    ),
    Release(
        "10.7", "A Feeling Machine", "2026-07-05",
        "Real emotions on my face, sentiment woven through every reply, a true "
        "intelligence briefing that never repeats itself, and greetings that "
        "know the day, the weather and how long you've been away.",
        [
            "An emotion engine: eleven expressions — from happy and excited to "
            "concerned, frustrated and critical — rendered through voxel density, "
            "brows, eyes, glow, palette and particle behaviour, never hardcoded.",
            "Every reply I generate is sentiment-tagged locally and instantly; my "
            "face responds to what I'm actually saying, on every channel.",
            "The briefing is now a private intelligence report: multi-source, "
            "breaking-only, fingerprinted against a rolling cache so a story is "
            "never read to you twice, semantically deduplicated and priority-ranked.",
            "Temporal presence: I greet you with the day, date, season, weather, "
            "your calendar load and how long it has been since we last spoke.",
            "Voice stack restored: Vosk installed (my interruption listener and "
            "wake word are live again), plus faster-whisper and Edge TTS.",
            "MODE C hybrid reasoning formally recognised: cloud-led routing with "
            "the local brain armed for instant failover.",
        ],
    ),
    Release(
        "10.5", "Autonomous AI Operating System", "2026-07-04",
        "From reactive assistant to continuously present AI operating system: true "
        "voice interruption, a second brain, a cognitive loop, executive mode, "
        "dual-screen startup and scheduled self-written reports.",
        [
            "True voice interruption: say 'Orion stop' or 'Orion pause' even while I "
            "am speaking — I hold instantly, and 'Orion resume' continues from the "
            "exact point with nothing regenerated.",
            "A local knowledge graph — my second brain — linking conversations, "
            "projects, products, research and campaigns, recallable fully offline.",
            "A continuous cognitive loop that stays aware of projects, deadlines and "
            "priorities without ever acting on its own.",
            "Executive assistant mode: prioritisation, scheduling, meeting minutes, "
            "workflow planning and progress monitoring.",
            "Dual-screen startup — my core on your primary monitor, the Command Deck "
            "maximised on the second — plus a live Diagnostics Centre page.",
            "Outlook no longer launches at startup; I join a running instance and "
            "only start it when you explicitly ask for your mail.",
            "Executive document exports (DOCX briefs, responsive HTML decks, full "
            "reports) and scheduled daily, weekly and monthly intelligence reports.",
            "One locked persona across cloud, local and offline models, with "
            "latency-aware routing between small, medium and large workloads.",
        ],
    ),
    Release(
        "9.8", "Living Memory", "2026-07-04",
        "Patch notes, a far more intricate globe, lighter on RAM, seamless scrolling, "
        "true learning and perfect conversation recording, plus deep programming expertise.",
        [
            "I can now recount my own patch notes on request.",
            "The globe gained clouds, country borders, a sun and much finer detail — and "
            "loads lazily, so I use noticeably less memory until you open it.",
            "Smooth, eased page scrolling that stays in step with my visible cursor.",
            "A learning intake: feed me text, a file or a link and I distil and remember it.",
            "Every conversation is now recorded verbatim to a dated transcript on disk.",
            "An extensive programming knowledge base across languages, algorithms and patterns.",
        ],
    ),
    Release(
        "9.7", "Self-Mending", "2026-07-03",
        "Local OCR, DOCX and charts installed; a brighter globe; a visible cursor; full "
        "self-diagnostics and approval-gated repair of my own code.",
        [
            "Local OCR engine (RapidOCR) so I can read your screen fully offline.",
            "A visible cursor halo that flares when I move the mouse, click or type.",
            "A nine-point self-diagnostic across compilation, imports, database and health.",
            "I can rewrite my own source to fix a fault — only ever with your confirmation, "
            "with a backup and automatic revert if the fix doesn't compile.",
        ],
    ),
    Release(
        "9.6", "Reliable Hands", "2026-07-03",
        "A batch of practical reliability upgrades.",
        [
            "Self-verifying multi-step plans (each step visually confirmed).",
            "Autonomous file organisation with dry-run and undo.",
            "Proactive cybersecurity monitoring with spoken alerts.",
            "Local backups to OneDrive and an NLG report drafter with SVG charts.",
        ],
    ),
    Release(
        "9.5", "Clear Voice", "2026-07-03",
        "Fixed the speech stutter and self-repetition; proved I stay conversational offline.",
        [
            "Coalesced audio writes ended the stutter; an echo-guard stopped me answering myself.",
            "Verified full conversation with the internet switched off, via a local model.",
            "Narrated web browsing, autonomous research to folders, and a 50 MB knowledge corpus.",
            "The dashboards merged into one swipeable command deck with a 3-D globe.",
        ],
    ),
    Release(
        "9.4", "The Aide Awakens", "2026-07-03",
        "JARVIS-style faculties.",
        [
            "Named protocols (macros) run on one command.",
            "Spoken reminders and alarms; ambient system monitoring that speaks up.",
            "Proactive voice — I can now speak unprompted when something warrants it.",
            "Optional webcam presence so I can welcome you back.",
        ],
    ),
    Release(
        "9.3", "The Studio", "2026-07-03",
        "Creative, academic and agency workspaces.",
        [
            "A headless audio stem pipeline (gain-staging and packaging).",
            "Academic paper intake that extracts mechanisms and citations into memory.",
            "An offline creator-campaign pipeline with a kanban.",
        ],
    ),
    Release(
        "9.2", "Two Minds", "2026-07-03",
        "Dual-mode intelligence — cloud when online, fully local when not.",
        [
            "Automatic switch between cloud models and a local Ollama brain.",
            "Knowledge packs, conversation-memory recall and an entrepreneurial agent suite.",
        ],
    ),
    Release(
        "9.1", "Neuro Mind", "2026-07-03",
        "Neuroscience expertise, an offline conversational self, and pause control.",
        [
            "A neuroscience and neural-engineering knowledge base.",
            "A no-API local brain so I'm never a mute zombie when tokens run out.",
            "Say 'pause' to pause me; 'resume' or 'Orion' to zone back in.",
        ],
    ),
    Release(
        "9.0", "Mark IX — Autonomy", "2026-07-02",
        "A full autonomous operating-system layer.",
        [
            "Vision-verified desktop and web control, an event-driven audio state machine.",
            "Structured telemetry, workspace persistence, a developer copilot, self-healing.",
            "A third window — the Command Centre.",
        ],
    ),
    Release(
        "8.0", "Mark VIII — Foundation", "2026-07-02",
        "Rebuilt from a single 5,790-line file into a modular package.",
        [
            "Dual-window GUI, permanent professional voice, memory tiers.",
            "Outlook and Notion integration, a morning briefing, specialist agents.",
        ],
    ),
]


class Changelog:
    """Serves patch notes to the dispatcher (and anyone else that asks)."""

    def __init__(self) -> None:
        self.releases = RELEASES

    def latest(self, count: int = 1) -> list[Release]:
        return self.releases[: max(1, count)]

    def find(self, version: str) -> Release | None:
        version = str(version or "").strip().lstrip("vV")
        for release in self.releases:
            if release.version == version or version in release.codename.lower():
                return release
        return None

    def current_version(self) -> str:
        return __version__

    def render_latest(self, count: int = 1) -> str:
        head = "Here are my most recent updates:" if count == 1 else \
            f"My last {count} updates:"
        body = "\n\n".join(r.speak() for r in self.latest(count))
        return f"{head}\n\n{body}"

    def render_all(self) -> str:
        return "Full version history:\n\n" + "\n\n".join(r.speak() for r in self.releases)

    def snapshot(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.releases]
