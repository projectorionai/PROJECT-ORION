"""
AppResolver — "open Word" should open Word.

The complaint
-------------
"I want ORION to have common sense as he wants me to say the exact application
name and that's very annoying for me."

He was right. The old path was: look the phrase up in a 20-entry dictionary,
then ``shutil.which`` it, then check the Start Menu index for an EXACT key, then
fall back to a naive ``key in name`` substring scan. So "Word" missed (the
executable is ``WINWORD.EXE``), "open up word please" missed (the filler words
were part of the key), and "Microsoft Word.exe" missed (the suffix was).

Where applications actually live
--------------------------------
Four sources, in decreasing authority:

  1. **App Paths** — ``HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App
     Paths``. This is the registry Windows itself consults when you type
     "winword" into Run, and it is how a program declares "this is my name and
     this is where I am". Verified present on this machine and it contains
     ``Winword.exe`` and ``excel.exe`` — precisely the cases that were failing.
  2. **Start Menu shortcuts** — human-facing names ("Word", "Visual Studio
     Code"), which is what a person actually says.
  3. **PATH** — anything directly runnable.
  4. **Aliases** — the handful of cases where what people say and what the
     binary is called have no textual relationship at all (word/WINWORD,
     vscode/Code, teams/ms-teams).

Matching
--------
Requests are normalised first (filler words, ``.exe``, punctuation), then
scored against every candidate. Exact and alias hits win outright; after that
it is prefix, whole-token containment, substring and finally a similarity
ratio, so "powerpint" still finds PowerPoint.

Nothing here launches anything — it answers "what did they mean?" and returns
its reasoning, so the caller can say *what* it opened and the whole thing is
testable without starting programs.
"""

from __future__ import annotations

import difflib
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

#: Words that carry no identifying information in a request to open something.
FILLER = frozenset({
    "open", "opens", "opening", "up", "the", "a", "an", "my", "please",
    "launch", "launches", "start", "run", "go", "to", "for", "me", "app",
    "apps", "application", "applications", "program", "programme", "software",
    "window", "can", "you", "could", "would", "will", "just", "now",
})

#: Only where the spoken name and the binary have no textual relationship.
#: Anything resolvable by the registry or the Start Menu does NOT belong here.
ALIASES: dict[str, str] = {
    "word": "winword",
    "microsoft word": "winword",
    "ms word": "winword",
    "excel": "excel",
    "microsoft excel": "excel",
    "powerpoint": "powerpnt",
    "microsoft powerpoint": "powerpnt",
    "ppt": "powerpnt",
    "outlook": "outlook",
    "access": "msaccess",
    "publisher": "mspub",
    "onenote": "onenote",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "photoshop": "photoshop",
    "task manager": "taskmgr",
    "control panel": "control",
    "device manager": "devmgmt.msc",
    "registry editor": "regedit",
    "notepad": "notepad",
    "calculator": "calc",
    "paint": "mspaint",
    "file explorer": "explorer",
    "files": "explorer",
    "terminal": "wt",
    "command prompt": "cmd",
    "settings": "ms-settings:",
    "snipping tool": "snippingtool",
}


def normalise(request: str) -> str:
    """Reduce a spoken request to the words that identify the application.

    "please open up Microsoft Word.exe for me" -> "microsoft word"
    """
    text = str(request or "").strip().lower()
    for suffix in (".exe", ".lnk", ".msc", ".app"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    cleaned = []
    for word in text.replace("-", " ").replace("_", " ").split():
        stripped = word.strip(".,!?;:'\"()")
        if stripped and stripped not in FILLER:
            cleaned.append(stripped)
    # Everything was filler ("open it") — fall back to the raw text so the
    # caller gets a sensible "I don't know what you meant" rather than "".
    return " ".join(cleaned) or text.strip()


@dataclass(frozen=True)
class Candidate:
    """One thing that could be launched."""

    name: str            # human-facing, lower-case
    target: str          # path or command to hand the OS
    source: str          # app_paths | start_menu | path | alias | builtin


@dataclass
class Match:
    candidate: Candidate | None
    score: float
    why: str
    alternatives: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.candidate is not None

    @property
    def confident(self) -> bool:
        """High enough to act without asking."""
        return self.score >= 0.62


def _score(needle: str, name: str) -> tuple[float, str]:
    """How well *name* answers *needle*, and why. 0..1."""
    if not needle or not name:
        return 0.0, ""
    if needle == name:
        return 1.0, "exact name"
    n_tokens = set(needle.split())
    m_tokens = set(name.split())
    if n_tokens and n_tokens == m_tokens:
        return 0.97, "same words"
    if name.startswith(needle) or needle.startswith(name):
        # "word" vs "wordpad" must not beat "word" vs "word": shorter is better.
        penalty = min(0.22, abs(len(name) - len(needle)) * 0.02)
        return 0.90 - penalty, "name starts with it"
    if n_tokens and n_tokens <= m_tokens:
        return 0.84, "contains all of those words"
    if needle in name:
        return 0.74 - min(0.18, (len(name) - len(needle)) * 0.012), "contains it"
    if m_tokens and m_tokens <= n_tokens:
        return 0.70, "you named it and more"
    ratio = difflib.SequenceMatcher(None, needle, name).ratio()
    if ratio >= 0.72:
        return ratio * 0.80, f"close spelling ({ratio:.0%})"
    return 0.0, ""


class AppCatalogue:
    """Every launchable application this machine knows about."""

    def __init__(self) -> None:
        self.candidates: list[Candidate] = []
        self.built = False

    # ── sources ───────────────────────────────────────────────────────────────

    @staticmethod
    def _from_app_paths() -> list[Candidate]:
        """The registry Windows itself uses to resolve a bare program name."""
        found: list[Candidate] = []
        try:
            import winreg
        except ImportError:
            return found
        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, key_path) as root:
                    count = winreg.QueryInfoKey(root)[0]
                    for index in range(count):
                        try:
                            entry = winreg.EnumKey(root, index)
                            with winreg.OpenKey(root, entry) as sub:
                                target = winreg.QueryValueEx(sub, "")[0]
                        except OSError:
                            continue
                        stem = entry.lower()
                        for suffix in (".exe", ".msc"):
                            if stem.endswith(suffix):
                                stem = stem[: -len(suffix)]
                        if stem and target:
                            found.append(Candidate(stem, str(target), "app_paths"))
            except OSError:
                continue
        return found

    @staticmethod
    def _from_start_menu() -> list[Candidate]:
        """Human-facing names — what a person actually says out loud."""
        found: list[Candidate] = []
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
                    found.append(Candidate(shortcut.stem.lower(),
                                           str(shortcut), "start_menu"))
            except OSError:
                continue
        return found

    @staticmethod
    def _from_apps_folder() -> list[Candidate]:
        r"""Everything on the Start screen — including Store apps.

        This is the source that was missing, and it is the reason "open
        WhatsApp" failed while "open Discord" worked. A Microsoft Store app is
        not an executable on disk that you can find: it has no App Paths
        registry entry, nothing on PATH, and no .lnk in the Start Menu
        folders. It exists only in the shell's AppsFolder namespace, keyed by
        an AppUserModelID like
        ``5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App``.

        All three existing sources therefore missed every Store app on the
        machine, and the user had to name a path for something sitting on
        their taskbar.

        Launched through ``shell:AppsFolder\<id>``, which is how Explorer
        itself starts them.
        """
        found: list[Candidate] = []
        if sys.platform != "win32":
            return found
        try:
            import pythoncom
            import win32com.client
        except Exception:
            return found
        initialised = False
        try:
            try:
                pythoncom.CoInitialize()
                initialised = True
            except Exception:
                pass          # already initialised on this thread; fine
            shell = win32com.client.Dispatch("Shell.Application")
            folder = shell.NameSpace("shell:AppsFolder")
            if folder is None:
                return found
            items = folder.Items()
            for index in range(items.Count):
                try:
                    item = items.Item(index)
                    name = str(item.Name or "").strip()
                    app_id = str(item.Path or "").strip()
                except Exception:
                    continue
                # A real AppUserModelID contains "!"; anything else in here is
                # an ordinary shortcut the other sources already cover.
                if not name or "!" not in app_id:
                    continue
                found.append(Candidate(name.lower(),
                                       f"shell:AppsFolder\\{app_id}",
                                       "apps_folder"))
        except Exception:
            return found
        finally:
            if initialised:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
        return found

    @staticmethod
    def _from_aliases() -> list[Candidate]:
        return [Candidate(name, target, "alias") for name, target in ALIASES.items()]

    def build(self) -> "AppCatalogue":
        """Gather every source. Safe to call repeatedly; never raises."""
        seen: dict[tuple[str, str], Candidate] = {}
        for source in (self._from_app_paths, self._from_start_menu,
                       self._from_apps_folder, self._from_aliases):
            try:
                for candidate in source():
                    seen.setdefault((candidate.name, candidate.source), candidate)
            except Exception:
                continue
        self.candidates = list(seen.values())
        self.built = True
        return self

    # ── resolution ────────────────────────────────────────────────────────────

    def resolve(self, request: str) -> Match:
        """What did they mean, and how sure are we?"""
        needle = normalise(request)
        if not needle:
            return Match(None, 0.0, "nothing was named")
        if not self.built:
            self.build()

        # An alias is a deliberate human decision and outranks fuzzy matching:
        # "word" must become WINWORD, never WordPad.
        aliased = ALIASES.get(needle)
        if aliased:
            resolved = self._launchable(aliased)
            if resolved is not None:
                return Match(resolved, 1.0,
                             f"'{needle}' is {aliased}", [])

        scored: list[tuple[float, str, Candidate]] = []
        for candidate in self.candidates:
            score, why = _score(needle, candidate.name)
            if score > 0:
                # The registry is authoritative; a Start Menu folder name is a
                # guess by comparison.
                if candidate.source == "app_paths":
                    score = min(1.0, score + 0.05)
                scored.append((score, why, candidate))

        if not scored:
            # Last resort: is it simply on PATH? ("git", "python")
            direct = shutil.which(needle)
            if direct:
                return Match(Candidate(needle, direct, "path"), 0.70,
                             "found on PATH")
            return Match(None, 0.0, f"nothing installed matches '{needle}'",
                         self.suggest(needle))

        scored.sort(key=lambda row: (-row[0], len(row[2].name)))
        best_score, best_why, best = scored[0]
        alternatives = []
        for score, _why, candidate in scored[1:]:
            if candidate.name != best.name and candidate.name not in alternatives:
                alternatives.append(candidate.name)
            if len(alternatives) >= 4:
                break
        return Match(best, best_score, best_why, alternatives)

    def _launchable(self, stem: str) -> Candidate | None:
        """Turn an alias target into something the OS can actually start."""
        for candidate in self.candidates:
            if candidate.name == stem and candidate.source == "app_paths":
                return candidate
        direct = shutil.which(stem if stem.endswith(".exe") else stem + ".exe")
        if direct:
            return Candidate(stem, direct, "path")
        if stem.endswith(":") or stem.endswith(".msc"):
            return Candidate(stem, stem, "builtin")     # ms-settings:, devmgmt.msc
        for candidate in self.candidates:
            if candidate.name == stem:
                return candidate
        return None

    def suggest(self, request: str, limit: int = 6) -> list[str]:
        """Nearest installed names, for an honest 'did you mean'."""
        needle = normalise(request)
        if not self.built:
            self.build()
        names = sorted({c.name for c in self.candidates})
        close = difflib.get_close_matches(needle, names, n=limit, cutoff=0.5)
        if close:
            return close
        return [name for name in names
                if any(token in name for token in needle.split())][:limit]


#: One shared catalogue — building it walks the registry and the Start Menu.
CATALOGUE = AppCatalogue()


def resolve_app(request: str) -> Match:
    return CATALOGUE.resolve(request)


__all__ = ["ALIASES", "AppCatalogue", "CATALOGUE", "Candidate", "FILLER",
           "Match", "normalise", "resolve_app"]
