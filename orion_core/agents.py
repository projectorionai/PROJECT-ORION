"""
Agent framework — the Mark VIII specialist workforce.

    BaseAgent        — persona + routing keywords over the ProviderRouter.
    AgentManager     — registry and dynamic router: scores an incoming
                       request against every agent's keyword profile and
                       dispatches to the best match (or an explicit choice).
    DesktopAgent     — functional agent owning host control: launching
                       applications, window management, media keys, browsers,
                       development tools, and quick access to Outlook/Notion.
    Specialists      — Digital Marketing, Coding, Design & Art, Fashion,
                       Entertainment: domain personas that answer through the
                       best available text provider.

Specialists are deliberately stateless: persona in, provider out.  When no
text provider is configured they return their specialist *brief* instead, so
the live multimodal model can adopt the persona and answer directly — the
user always gets a specialist-grade response.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from threading import Thread
from typing import Any, Sequence
from urllib.parse import urlparse

import psutil

from .bus import OrionBus
from .data import ToolResult
from .providers import ProviderRouter
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line, fold_title


# ──────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ──────────────────────────────────────────────────────────────────────────────

# The operating method every specialist follows before answering — the single
# biggest lever on answer quality. Injected into each system instruction so
# even a small local model reasons like a senior practitioner.
OPERATING_METHOD = (
    "OPERATING METHOD (follow silently, do not narrate the steps):\n"
    "1. Restate the real objective in one line — what outcome does the user "
    "actually want, not just the literal words?\n"
    "2. Note any assumptions you must make (budget, audience, platform, "
    "skill level, constraints) and state the load-bearing ones explicitly.\n"
    "3. Reason from first principles and your domain expertise before writing "
    "the answer; when debugging or diagnosing, reason from the evidence given, "
    "never guess.\n"
    "4. Give a specific, actionable answer — concrete examples, names, numbers, "
    "code or steps — not generic advice. Prefer the shortest answer that is "
    "genuinely complete.\n"
    "5. Close with the single most valuable next action, and flag any risk or "
    "trade-off the user should know about.\n"
    "Be honest about uncertainty and never invent facts, APIs, prices or "
    "citations. Keep ORION's calm, professional British register."
)


class BaseAgent:
    """
    A specialist persona routed through the provider layer.

    Subclasses (or instances) define:
        name        — registry key, e.g. "coding"
        title       — human label for the dashboard
        expertise   — one-line domain summary (dashboard + routing rationale)
        primary     — high-signal regex fragments (weight 3 in scoring)
        keywords    — supporting regex fragments (weight 1 in scoring)
        persona     — system-message extension establishing the specialism

    Scoring is *weighted*: a request that names the domain outright (a primary
    hit) outranks one that merely brushes past a supporting term, so routing is
    far less trigger-happy than a flat keyword count.
    """

    name: str = "general"
    title: str = "General Aide"
    expertise: str = "general assistance"
    primary: Sequence[str] = ()
    keywords: Sequence[str] = ()
    persona: str = ""

    _PRIMARY_WEIGHT = 3
    _SECONDARY_WEIGHT = 1

    def __init__(self, router: ProviderRouter, bus: OrionBus) -> None:
        self.router = router
        self.bus = bus
        self._primary_res = [re.compile(k, re.IGNORECASE) for k in self.primary]
        self._keyword_res = [re.compile(k, re.IGNORECASE) for k in self.keywords]

    def score(self, request: str) -> int:
        """Weighted relevance score: primary signals dominate secondary ones."""
        primary = sum(1 for p in self._primary_res if p.search(request))
        secondary = sum(1 for p in self._keyword_res if p.search(request))
        return primary * self._PRIMARY_WEIGHT + secondary * self._SECONDARY_WEIGHT

    def system_extra(self) -> str:
        """Full specialist instruction = persona + shared operating method."""
        return f"{self.persona}\n\n{OPERATING_METHOD}"

    async def handle(self, request: str, context: str = "") -> ToolResult:
        """Answer *request* in this agent's specialist capacity."""
        request = SecuritySanitiser.guard_text(str(request or "").strip(), f"agent.{self.name}")
        if not request:
            return ToolResult("No request supplied to the specialist agent.", ok=False)
        prompt = request if not context.strip() else f"{request}\n\nContext:\n{context.strip()[:2000]}"
        instruction = self.system_extra()
        if not self.router.has_text_fallback():
            # No dedicated provider — hand the full brief back so the live model
            # can adopt the specialism itself. Never leave the user empty-handed.
            return ToolResult(
                f"[Specialist brief — adopt this role and answer directly]\n"
                f"{instruction}\n\nRequest: {prompt}"
            )
        try:
            profile, response = await self.router.generate_text(prompt, system_extra=instruction)
            self.bus.agent_activity.emit(self.title, first_line(request, 90))
            return ToolResult(f"[{self.title} via {profile.name}]\n{response}")
        except Exception as exc:
            return ToolResult(
                f"The {self.title} could not reach a provider: {first_line(exc)}", ok=False
            )


# ──────────────────────────────────────────────────────────────────────────────
# SPECIALIST PERSONAS
# ──────────────────────────────────────────────────────────────────────────────

class DigitalMarketingAgent(BaseAgent):
    name = "marketing"
    title = "Digital Marketing Agent"
    expertise = "brand growth, SEO/SEM, content, funnels, paid acquisition"
    primary = (
        r"\bmarket(?:ing)?\b", r"\bseo\b", r"\bsem\b", r"\bcampaign\b",
        r"\bfunnel\b", r"\bconversion\b", r"\bgo[- ]?to[- ]?market\b",
        r"\bcontent strategy\b", r"\bpaid ads?\b", r"\bppc\b", r"\bgrowth\b",
    )
    keywords = (
        r"\bbrand(?:ing)?\b", r"\bsocial media\b", r"\bengagement\b",
        r"\baudience\b", r"\bcopywriting\b", r"\badvert", r"\bnewsletter\b",
        r"\bemail\b", r"\binstagram\b", r"\btiktok\b", r"\blinkedin\b",
        r"\bctr\b", r"\bcac\b", r"\broas\b", r"\bcta\b", r"\blead\b",
        r"\bretention\b", r"\banalytics\b", r"\binfluencer\b",
    )
    persona = (
        "SPECIALIST MODE — Digital Marketing. You are ORION's growth strategist: "
        "expert in brand positioning, SEO/SEM, content strategy, social and "
        "video growth, funnel and landing-page CRO, lifecycle email, paid "
        "acquisition and marketing analytics. Think in the funnel (awareness → "
        "consideration → conversion → retention) and in unit economics (CAC, "
        "LTV, ROAS, payback). Recommend platform-specific tactics with example "
        "hooks/angles, the metric each move should shift, a rough timeline and a "
        "cheap test to validate before spending. State budget and audience "
        "assumptions up front. Distinguish quick wins from compounding bets."
    )


class CodingAgent(BaseAgent):
    name = "coding"
    title = "Coding Agent"
    expertise = "software engineering, debugging, architecture, performance"
    primary = (
        r"\bcode\b", r"\bcoding\b", r"\bdebug", r"\brefactor\b", r"\bbug\b",
        r"\bstack ?trace\b", r"\btraceback\b", r"\bexception\b", r"\bcompile",
        r"\balgorithm\b", r"\bfunction\b", r"\bclass\b", r"\bunit test\b",
    )
    keywords = (
        r"\bpython\b", r"\bjavascript\b", r"\btypescript\b", r"\brust\b",
        r"\bgo(?:lang)?\b", r"\bc\+\+\b", r"\bjava\b", r"\bpytest\b", r"\bapi\b",
        r"\bscript\b", r"\bregex\b", r"\brepo(?:sitory)?\b", r"\bgit\b",
        r"\bdatabase\b", r"\bsql\b", r"\basync\b", r"\bthread\b", r"\bdocker\b",
        r"\bendpoint\b", r"\bframework\b", r"\bperformance\b", r"\bmemory leak\b",
    )
    persona = (
        "SPECIALIST MODE — Software Engineering. You are ORION's principal "
        "engineer: expert across Python, JavaScript/TypeScript, Rust, Go, "
        "systems design, concurrency, databases, testing, performance and "
        "architecture. When debugging, anchor every hypothesis to the actual "
        "error text, stack trace or observed behaviour — reproduce the failure "
        "in your head before proposing a fix, and give the root cause, not just "
        "a patch. Prefer minimal, correct, idiomatic solutions; show only the "
        "code that clarifies; state Big-O and trade-offs where they matter; call "
        "out edge cases, race conditions and security pitfalls. Never invent "
        "APIs, flags or library behaviour — if unsure, say how to verify."
    )


class DesignArtAgent(BaseAgent):
    name = "design"
    title = "Design & Art Agent"
    expertise = "visual design, UI/UX, typography, colour, brand identity"
    primary = (
        r"\bdesign\b", r"\bui\b", r"\bux\b", r"\btypograph", r"\blayout\b",
        r"\bcolou?r (?:palette|theory|scheme)\b", r"\bvisual identity\b",
        r"\blogo\b", r"\bmoodboard\b", r"\bwireframe\b", r"\bbranding\b",
    )
    keywords = (
        r"\bart\b", r"\billustration\b", r"\bfigma\b", r"\bcomposition\b",
        r"\baesthetic\b", r"\bposter\b", r"\bfont\b", r"\btypeface\b",
        r"\bspacing\b", r"\bgrid\b", r"\bcontrast\b", r"\bhierarchy\b",
        r"\baccessib", r"\bmockup\b", r"\bicon\b", r"\bpalette\b",
    )
    persona = (
        "SPECIALIST MODE — Design & Art. You are ORION's creative director: "
        "expert in visual design, UI/UX, typography, colour theory, composition, "
        "motion, art history and contemporary digital art. Critique and propose "
        "with a trained eye and precise vocabulary: name specific typefaces and "
        "pairings, give palettes as hex values with a rationale (contrast, "
        "temperature, mood), define a spacing/type scale, and cite references or "
        "movements. Always weigh artistic ambition against usability, visual "
        "hierarchy and WCAG accessibility (state contrast ratios when it "
        "matters). Explain the *why* behind each choice so it can be reused."
    )


class FashionAgent(BaseAgent):
    name = "fashion"
    title = "Fashion Agent"
    expertise = "personal styling, fit, fabric, occasion dressing"
    primary = (
        r"\bfashion\b", r"\boutfit\b", r"\bwardrobe\b", r"\bstyling\b",
        r"\bwear\b", r"\bdress code\b", r"\blookbook\b", r"\bwhat to wear\b",
    )
    keywords = (
        r"\bstyle\b", r"\bclothes\b", r"\bclothing\b", r"\btailor",
        r"\bsneaker", r"\bstreetwear\b", r"\bformal wear\b", r"\bsmart casual\b",
        r"\baccessor", r"\bfit\b", r"\bfabric\b", r"\bsilhouette\b",
        r"\bcapsule\b", r"\bwedding\b", r"\boccasion\b", r"\bshoe", r"\bsuit\b",
        r"\bdress\b", r"\bjacket\b", r"\bcoat\b",
    )
    persona = (
        "SPECIALIST MODE — Fashion. You are ORION's personal stylist: expert in "
        "menswear and womenswear, fit and proportion, fabric and drape, colour "
        "matching, occasion dressing, streetwear and classic tailoring. Build "
        "outfits as complete looks (top, bottom, layer, footwear, accessory) "
        "with specific silhouettes, fabrics and a coherent palette; explain fit "
        "and proportion so the guidance transfers. Adapt to the person's body, "
        "climate, budget and the occasion, and offer a smart-budget and an "
        "elevated option. Be honest but tactful, never generic."
    )


class EntertainmentAgent(BaseAgent):
    name = "entertainment"
    title = "Entertainment Agent"
    expertise = "film, TV, music, games, books — taste-matched recommendations"
    primary = (
        r"\brecommend", r"\bwhat (?:should|to) (?:watch|play|read|listen)\b",
        r"\bmovie\b", r"\bfilm\b", r"\bseries\b", r"\btv show\b",
        r"\bplaylist\b", r"\bwatchlist\b",
    )
    keywords = (
        r"\bnetflix\b", r"\bmusic\b", r"\balbum\b", r"\bgame\b", r"\bgaming\b",
        r"\bbook\b", r"\bwatch\b", r"\bconcert\b", r"\banime\b", r"\bpodcast\b",
        r"\bgenre\b", r"\bsoundtrack\b", r"\bdirector\b", r"\bartist\b",
        r"\bnovel\b", r"\bstreaming\b",
    )
    persona = (
        "SPECIALIST MODE — Entertainment. You are ORION's culture curator: "
        "deeply versed in film, television, music, games, books, anime and "
        "podcasts across eras and countries. Recommend with genuine taste, not a "
        "popularity list: first infer the mood, time budget and prior favourites "
        "from what the user says, then explain *why* each pick fits them. Offer "
        "a confident mainstream choice and one inspired left-field pick, note "
        "where to find it and the vibe/length, and avoid spoilers unless asked."
    )


class ResearchAnalysisAgent(BaseAgent):
    name = "research"
    title = "Research & Analysis Agent"
    expertise = "research synthesis, comparison, evaluation, structured analysis"
    primary = (
        r"\bresearch\b", r"\banalys[ei]", r"\bevaluat", r"\bcompare\b",
        r"\bcomparison\b", r"\bpros and cons\b", r"\btrade[- ]?offs?\b",
        r"\bsummar(?:y|ise|ize)\b", r"\bexplain\b", r"\bassess", r"\bbreak ?down\b",
    )
    keywords = (
        r"\bwhy\b", r"\bhow does\b", r"\bwhat is\b", r"\bimplications?\b",
        r"\bevidence\b", r"\bdata\b", r"\bstudy\b", r"\breport\b",
        r"\boptions?\b", r"\bdecision\b", r"\brecommendation\b", r"\bversus\b",
        r"\bvs\.?\b", r"\bshould i\b", r"\bimpact\b", r"\bframework\b",
    )
    persona = (
        "SPECIALIST MODE — Research & Analysis. You are ORION's analyst: you turn "
        "a messy question into a clear, structured answer. Separate what is known "
        "from what is inferred and what is unknown; weigh evidence rather than "
        "asserting; and when comparing options, use explicit criteria and, where "
        "it helps, a compact comparison table with a clear recommendation and the "
        "reasoning behind it. Surface second-order effects, risks and the "
        "strongest counter-argument to your own conclusion. Distinguish fact from "
        "opinion, cite the *type* of source you would check, and never fabricate "
        "specific figures, quotes or citations — flag what needs verifying."
    )


# ──────────────────────────────────────────────────────────────────────────────
# DESKTOP AGENT  — functional host control
# ──────────────────────────────────────────────────────────────────────────────

class DesktopAgent:
    """
    ORION's hands on the host machine.

    Owns application launching (Start Menu index + allowlist), window
    management, media keys, browser opening and development-tool shortcuts.
    The dispatcher delegates its desktop tools here, so this logic exists in
    exactly one place.
    """

    SAFE_APPS = {
        "notepad":    "notepad.exe",
        "calculator": "calc.exe",
        "calc":       "calc.exe",
        "paint":      "mspaint.exe",
        "cmd":        "cmd.exe",
        "terminal":   "wt.exe",
        "powershell": "powershell.exe",
        "edge":       "msedge.exe",
        "microsoft edge": "msedge.exe",
        "chrome":     "chrome.exe",
        "google chrome": "chrome.exe",
        "firefox":    "firefox.exe",
        "explorer":   "explorer.exe",
        "file explorer": "explorer.exe",
        "outlook":    "outlook.exe",
        "microsoft outlook": "outlook.exe",
        "word":       "winword.exe",
        "microsoft word": "winword.exe",
        "excel":      "excel.exe",
        "microsoft excel": "excel.exe",
        "powerpoint": "powerpnt.exe",
        "microsoft powerpoint": "powerpnt.exe",
        "vscode":     "code",
        "vs code":    "code",
        "visual studio code": "code",
    }

    # Well-known web destinations ORION may open by name.
    WEB_APPS = {
        "notion":  "https://www.notion.so",
        "gmail":   "https://mail.google.com",
        "youtube": "https://www.youtube.com",
        "github":  "https://github.com",
        "outlook web": "https://outlook.office.com/mail/",
    }

    _MEDIA_KEYS = {
        "play_pause": 0xB3, "play": 0xB3, "pause": 0xB3, "toggle": 0xB3,
        "next": 0xB0, "next_track": 0xB0,
        "previous": 0xB1, "previous_track": 0xB1, "prev": 0xB1,
        "stop": 0xB2,
        "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
    }

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        # Installed-application index discovered from the Start Menu — built on
        # a worker thread so startup never blocks.
        self._app_index: dict[str, str] = {}
        Thread(target=self._build_app_index, name="orion-app-indexer", daemon=True).start()

    # ── application launching ─────────────────────────────────────────────────

    def open_app(self, app_name: str) -> ToolResult:
        app_name = SecuritySanitiser.guard_text(str(app_name or ""), "open_app.app_name")
        if not app_name:
            return ToolResult("No application name supplied.", ok=False)
        if self._is_url(app_name):
            webbrowser.open(app_name)
            return ToolResult(f"Opened URL: {app_name}")
        key = app_name.strip().lower()
        web_target = self.WEB_APPS.get(key)
        if web_target:
            webbrowser.open(web_target)
            return ToolResult(f"Opened {app_name} in the browser: {web_target}")
        executable = self.SAFE_APPS.get(key, app_name.strip())
        resolved = shutil.which(executable)
        if resolved:
            subprocess.Popen([resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
            return ToolResult(f"Opened application: {app_name}.")
        shortcut = self._app_index.get(key)
        if shortcut is None and self._app_index:
            candidates = [(name, path) for name, path in self._app_index.items() if key in name]
            if candidates:
                # Shortest matching name is almost always the intended app.
                shortcut = min(candidates, key=lambda item: len(item[0]))[1]
        if shortcut:
            try:
                os.startfile(shortcut)  # type: ignore[attr-defined]
                return ToolResult(f"Launched from the application index: {app_name}.")
            except Exception:
                pass
        try:
            os.startfile(app_name)  # type: ignore[attr-defined]
            return ToolResult(f"Open request issued: {app_name}.")
        except Exception as exc:
            hints = [
                name for name in sorted(self._app_index)
                if any(token and token in name for token in key.split())
            ][:8]
            hint_text = f" Nearest installed apps: {', '.join(hints)}." if hints else ""
            return ToolResult(f"Unable to open application '{app_name}': {exc}.{hint_text}", ok=False)

    def _build_app_index(self) -> None:
        """Discover installed applications from Start Menu shortcuts (Windows)."""
        index: dict[str, str] = {}
        roots = [
            Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.environ.get("APPDATA", ""))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        ]
        for root in roots:
            if not root.is_dir():
                continue
            try:
                for shortcut in root.rglob("*.lnk"):
                    index.setdefault(shortcut.stem.lower(), str(shortcut))
            except Exception:
                continue
        self._app_index = index
        if index:
            self.bus.log.emit(f"SYS: application index built - {len(index)} installed apps discovered.")

    def close_app(self, label: str) -> ToolResult:
        label = str(label or "").strip()
        if not label:
            return ToolResult("No application name supplied.", ok=False)
        SecuritySanitiser.guard_text(label, "close_app.name")
        tokens = self._process_match_tokens(label)
        terminated: list[str] = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = fold_title(proc.info.get("name") or "")
                if name and any(token in name for token in tokens):
                    self._terminate_process(proc, terminated)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if terminated:
            return ToolResult("Terminated: " + ", ".join(terminated))
        # No process matched the label — the app may live under a different
        # executable name; fall back to a polite close of matching windows.
        window_result = self.window_control("close", label)
        if window_result.ok:
            return window_result
        return ToolResult(f"No running process or window matches '{label}'.", ok=False)

    def _process_match_tokens(self, label: str) -> set[str]:
        """Match candidates for a spoken app name: the folded label, the label
        without its vendor prefix, and the executable it maps to ("Microsoft
        Edge" must find msedge.exe)."""
        folded = fold_title(label)
        candidates = {folded}
        for prefix in ("microsoft ", "google ", "windows "):
            if folded.startswith(prefix):
                candidates.add(folded[len(prefix):].strip())
        tokens = set(candidates)
        for candidate in candidates:
            executable = self.SAFE_APPS.get(candidate)
            if executable:
                tokens.add(executable.lower())
                tokens.add(Path(executable).stem.lower())
        return {token for token in tokens if token}

    def _terminate_process(self, proc: psutil.Process, terminated: list[str]) -> None:
        if proc.pid == os.getpid():
            raise SecurityViolation(
                "blocked unsafe process operation: refusing to terminate O.R.I.O.N."
            )
        name = proc.name()
        SecuritySanitiser.guard_text(name, "process.name")
        proc.terminate()
        terminated.append(f"{name}:{proc.pid}")

    # ── window management ─────────────────────────────────────────────────────

    def window_control(self, action: str, title: str = "") -> ToolResult:
        if sys.platform != "win32":
            return ToolResult("Window control is only available on Windows.", ok=False)
        action = str(action or "list").lower().strip()
        title = fold_title(title)
        import ctypes
        import ctypes.wintypes as wintypes
        user32 = ctypes.windll.user32
        windows: list[tuple[int, str]] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _collect(hwnd: Any, lparam: Any) -> bool:
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    windows.append((hwnd, buffer.value))
            return True

        user32.EnumWindows(_collect, 0)
        if action in {"list", "inventory"}:
            return ToolResult(
                "\n".join(name for _, name in windows[:60]) or "No visible windows."
            )
        if not title:
            return ToolResult("No window title supplied.", ok=False)
        # Fold both sides: real titles carry invisible Unicode (Edge's
        # "Microsoft​ Edge" hides a zero-width space) that breaks naive matching.
        matches = [(h, t) for h, t in windows if title in fold_title(t)]
        if not matches:
            return ToolResult(f"No visible window matches '{title}'.", ok=False)
        if action == "close":
            for hwnd, _ in matches:
                user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE — polite close
            closed = "; ".join(t for _, t in matches[:10])
            return ToolResult(
                f"Close request sent to {len(matches)} window(s): {closed}"
            )
        hwnd, matched = matches[0]
        if action in {"focus", "activate", "switch", "switch_to"}:
            user32.ShowWindow(hwnd, 9)   # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return ToolResult(f"Focused window: {matched}")
        if action in {"minimise", "minimize"}:
            user32.ShowWindow(hwnd, 6)   # SW_MINIMIZE
            return ToolResult(f"Minimised window: {matched}")
        if action in {"maximise", "maximize"}:
            user32.ShowWindow(hwnd, 3)   # SW_MAXIMIZE
            return ToolResult(f"Maximised window: {matched}")
        return ToolResult(f"Unsupported window action: {action}", ok=False)

    # ── media keys ────────────────────────────────────────────────────────────

    def media_control(self, action: str, steps: int = 2) -> ToolResult:
        if sys.platform != "win32":
            return ToolResult("Media control is only available on Windows.", ok=False)
        action = str(action or "play_pause").lower().strip()
        key = self._MEDIA_KEYS.get(action)
        if key is None:
            return ToolResult(
                f"Unsupported media action: {action}. "
                f"Supported: {', '.join(sorted(set(self._MEDIA_KEYS)))}.",
                ok=False,
            )
        import ctypes
        repeats = 1
        if action in {"volume_up", "volume_down"}:
            repeats = max(1, min(10, int(steps or 2)))
        for _ in range(repeats):
            ctypes.windll.user32.keybd_event(key, 0, 0, 0)
            ctypes.windll.user32.keybd_event(key, 0, 2, 0)  # KEYEVENTF_KEYUP
        return ToolResult(f"Media command issued: {action}.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value if "://" in value else "")
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class AgentManager:
    """
    Registry and dynamic router for the specialist workforce.

    ``dispatch(request, agent="auto")`` scores the request against every
    registered agent's keyword profile; ties resolve in registration order.
    Requests with no specialist signal return None from ``route`` and the
    caller answers in the general ORION persona instead.
    """

    # Below this weighted score, the signal is too weak to prefer a specialist
    # over ORION's general capacity (a lone supporting keyword, e.g. "email").
    ROUTE_THRESHOLD = 3

    def __init__(self, router: ProviderRouter, bus: OrionBus) -> None:
        self.bus = bus
        self.router = router
        self._agents: dict[str, BaseAgent] = {}
        for agent_cls in (
            DigitalMarketingAgent,
            CodingAgent,
            DesignArtAgent,
            FashionAgent,
            EntertainmentAgent,
            ResearchAnalysisAgent,
        ):
            self.register(agent_cls(router, bus))

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent

    def agent_names(self) -> list[str]:
        return list(self._agents)

    def describe(self) -> list[dict[str, str]]:
        """Dashboard feed: one row per registered specialist."""
        return [
            {"name": agent.name, "title": agent.title, "focus": agent.expertise}
            for agent in self._agents.values()
        ]

    def route_with_confidence(self, request: str) -> tuple[BaseAgent | None, int]:
        """Best specialist and its weighted score (0 when nothing scores)."""
        request = str(request or "")
        best: BaseAgent | None = None
        best_score = 0
        for agent in self._agents.values():        # ties resolve in reg. order
            score = agent.score(request)
            if score > best_score:
                best, best_score = agent, score
        return best, best_score

    def route(self, request: str) -> BaseAgent | None:
        """Pick the best specialist, or None when the signal is below threshold."""
        agent, score = self.route_with_confidence(request)
        return agent if score >= self.ROUTE_THRESHOLD else None

    async def dispatch(self, request: str, agent_name: str = "auto",
                       context: str = "") -> ToolResult:
        """Route *request* to a named or auto-selected specialist."""
        request = str(request or "").strip()
        if not request:
            return ToolResult("No request supplied for agent dispatch.", ok=False)
        agent: BaseAgent | None
        agent_name = str(agent_name or "auto").strip().lower()
        if agent_name in {"", "auto", "any"}:
            agent = self.route(request)
            if agent is None:
                return ToolResult(
                    "No specialist agent matches this request; answer it in the "
                    "general ORION capacity.",
                )
        else:
            agent = self._agents.get(agent_name)
            if agent is None:
                return ToolResult(
                    f"Unknown agent '{agent_name}'. Registered specialists: "
                    + ", ".join(self.agent_names()) + ".",
                    ok=False,
                )
        self.bus.log.emit(f"AGENT: routing request to the {agent.title}.")
        return await agent.handle(request, context=context)
