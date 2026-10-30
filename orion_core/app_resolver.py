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
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .constants import CONFIG_DIR

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


#: Start Menu entries that are not the application: "open Zoom" picked
#: "Uninstall Zoom Workplace" at 0.84 on 4 Oct 2026, and "open the web" picked
#: a CapCut shortcut. Such an entry is only a candidate when the request itself
#: says the word ("uninstall zoom").
JUNK_WORDS = frozenset({
    "uninstall", "uninstaller", "readme", "manual", "manuals", "documentation",
    "dokumentation", "license", "licence", "faq", "homepage", "website",
    "samples", "changelog", "release", "reference", "updates", "maintenance",
    "help", "localization", "docs", "what", "telemetry",
})


def is_junk(name: str, needle: str = "") -> bool:
    """A shortcut to an uninstaller, manual or web page — not an application."""
    asked = set(needle.split())
    words = set(str(name).replace("(", " ").replace(")", " ").split())
    return bool((words & JUNK_WORDS) - asked)


#: What a request means when it names a KIND of application rather than one:
#: "open a browser", "my music", "something to write code in". Each member is
#: one application written as its catalogue synonyms, first = display name, in
#: the order to try when nothing better is known. ``default`` names the
#: Windows association whose handler is the user's own choice for the kind
#: ("https" = the default browser); ``prefer_default`` puts that choice ahead
#: of the static order.
@dataclass(frozen=True)
class Category:
    name: str
    phrases: tuple[str, ...]
    members: tuple[tuple[str, ...], ...]
    default: str = ""
    prefer_default: bool = False


CATEGORIES: tuple[Category, ...] = (
    Category("browser",
             ("browser", "web browser", "internet browser", "internet", "the internet",
              "web", "the web", "browse", "browsing", "surf the web", "internet explorer"),
             (("google chrome", "chrome"), ("microsoft edge", "msedge", "edge"),
              ("firefox", "mozilla firefox"), ("brave", "brave browser"),
              ("opera", "opera gx browser", "opera browser"), ("vivaldi",),
              ("tor browser",)),
             default="https", prefer_default=True),
    Category("email",
             ("email", "e-mail", "emails", "my email", "email app", "email client",
              "inbox", "my inbox", "mail app"),
             (("outlook", "olk", "microsoft outlook"), ("mail",), ("thunderbird",))),
    Category("music",
             ("music", "music player", "music app", "songs", "audio player",
              "some music", "my music", "play music", "listen to music"),
             (("spotify",), ("itunes", "apple music"), ("media player",),
              ("windows media player legacy", "wmplayer"), ("vlc", "vlc media player"),
              ("foobar2000",), ("musicbee",)),
             default=".mp3"),
    Category("video player",
             ("video player", "videos", "movie player", "media player", "watch a video",
              "watch videos", "play a video"),
             (("vlc", "vlc media player"), ("media player",), ("films & tv", "movies & tv"),
              ("windows media player legacy", "wmplayer"), ("potplayer",)),
             default=".mp4", prefer_default=True),
    Category("photos",
             ("photos", "pictures", "my photos", "my pictures", "gallery", "photo viewer",
              "image viewer", "picture viewer"),
             (("photos",),), default=".jpg", prefer_default=True),
    Category("image editor",
             ("photo editor", "image editor", "edit photos", "edit a photo", "drawing",
              "drawing app", "draw", "graphics editor"),
             (("adobe photoshop 2020", "photoshop", "adobe photoshop"), ("gimp",),
              ("paint.net",), ("krita",), ("paint", "mspaint"))),
    Category("text editor",
             ("text editor", "editor", "plain text", "text file", "notepad app"),
             (("notepad",), ("notepad++",), ("visual studio code", "code")),
             default=".txt", prefer_default=True),
    Category("notes",
             ("notes", "note taking", "note-taking", "notes app", "take notes",
              "a note", "make a note"),
             (("onenote",), ("sticky notes", "sticky notes (new)"), ("notion",),
              ("obsidian",), ("evernote",))),
    Category("code editor",
             ("code editor", "ide", "coding", "programming", "write code", "code",
              "editor for code", "development environment"),
             (("visual studio code", "code"), ("cursor",), ("pycharm 2026.1.1", "pycharm"),
              ("visual studio",), ("android studio",), ("sublime text",), ("notepad++",))),
    Category("terminal",
             ("terminal", "command line", "the command line", "command prompt", "console",
              "shell", "cmd", "powershell window", "cli"),
             (("terminal", "wt"), ("windows powershell", "powershell"),
              ("command prompt", "cmd"))),
    Category("file manager",
             ("file manager", "file browser", "files", "my files", "folders", "my folders",
              "file explorer", "explorer"),
             (("file explorer", "explorer"),)),
    Category("calculator", ("calculator", "calc", "a calculator", "maths", "math"),
             (("calculator", "calc"),)),
    Category("clock", ("clock", "alarm", "alarms", "timer", "stopwatch", "an alarm"),
             (("clock",),)),
    Category("calendar", ("calendar", "my calendar", "schedule", "diary", "agenda"),
             (("calendar",), ("outlook", "olk"), ("notion calendar",))),
    Category("settings", ("settings", "system settings", "pc settings", "windows settings",
                          "preferences"),
             (("settings", "ms-settings:"),)),
    Category("word processor",
             ("word processor", "document editor", "a document",
              "write a document", "write a letter", "writing"),
             (("word", "winword", "microsoft word"), ("libreoffice writer",),
              ("wordpad",))),
    Category("spreadsheet",
             ("spreadsheet", "spreadsheets", "sheets", "a spreadsheet", "budget sheet"),
             (("excel", "microsoft excel"), ("libreoffice calc",))),
    Category("presentation",
             ("slides", "presentation", "presentations", "slideshow", "slide deck",
              "a presentation", "slide show"),
             (("powerpoint", "powerpnt", "microsoft powerpoint"), ("libreoffice impress",))),
    Category("messaging",
             ("messages", "messaging", "chat", "chat app", "texting", "text messages",
              "messenger app", "dms"),
             (("whatsapp",), ("discord",), ("messenger",), ("microsoft teams", "ms-teams"),
              ("telegram",), ("signal",), ("phone link",))),
    Category("video call",
             ("video call", "video calls", "meeting", "meetings", "conference call",
              "call app", "video chat", "a call"),
             (("zoom workplace", "zoom"), ("microsoft teams", "ms-teams"),
              ("skype", "skype for business"))),
    Category("games",
             ("games", "game launcher", "gaming", "my games", "play games", "a game",
              "game library"),
             (("steam",), ("epic games launcher",), ("xbox",), ("battle.net",),
              ("riot client",), ("ubisoft connect",), ("ea app",))),
    Category("screenshot",
             ("screenshot", "screenshot tool", "screen capture", "snip", "screen snip",
              "snipping", "capture the screen"),
             (("snipping tool", "snippingtool"),)),
    Category("screen recorder",
             ("screen recorder", "record my screen", "record the screen",
              "screen recording", "streaming software", "recording software"),
             (("obs studio (64bit)", "obs studio", "obs"), ("game bar",),
              ("snipping tool", "snippingtool"))),
    Category("voice recorder",
             ("voice recorder", "sound recorder", "audio recorder", "record audio",
              "record my voice", "dictaphone", "voice memo"),
             (("sound recorder",), ("audacity",))),
    Category("video editor",
             ("video editor", "edit videos", "edit a video", "video editing"),
             (("adobe premiere pro 2023", "adobe premiere pro", "premiere pro"),
              ("microsoft clipchamp", "clipchamp"), ("capcut",), ("davinci resolve",))),
    Category("pdf reader",
             ("pdf", "pdf reader", "pdf viewer", "a pdf", "read a pdf"),
             (("adobe acrobat", "acrobat reader", "adobe acrobat reader"),
              ("microsoft edge", "msedge")),
             default=".pdf", prefer_default=True),
    Category("app store",
             ("app store", "store", "windows store", "the store"),
             (("microsoft store", "store"),)),
    Category("ai chat",
             ("ai", "ai chat", "chatbot", "ai assistant", "chat gpt"),
             (("chatgpt",), ("gemini",), ("microsoft 365 copilot", "copilot"))),
    Category("cloud storage",
             ("cloud storage", "cloud drive", "my cloud", "cloud files"),
             (("onedrive",), ("megasync",), ("google drive",), ("dropbox",))),
    Category("vpn",
             ("vpn", "a vpn"),
             (("windscribe",), ("nordvpn",), ("protonvpn", "proton vpn"),
              ("expressvpn",), ("tailscale",))),
    Category("to do",
             ("todo", "to do list", "todo list", "tasks app", "task list"),
             (("microsoft to do",), ("notion",))),
)

#: Words that make a request vague rather than naming anything: "open an app",
#: "launch something".
GENERIC = frozenset({"something", "anything", "it", "that", "this", "one", "stuff",
                     "thing", "whatever", "some", "programs", "something else"})


def identifying_words(request: str) -> str:
    """normalise() without its fallback: '' when nothing identifying was said."""
    text = str(request or "").strip().lower()
    words = []
    for word in text.replace("-", " ").replace("_", " ").split():
        stripped = word.strip(".,!?;:'\"()")
        if stripped and stripped not in FILLER:
            words.append(stripped)
    return " ".join(words)


def _phrase_index() -> dict[str, Category]:
    """Category phrases in the same normalised form a request is reduced to
    ("the internet" -> "internet"), so the two can be compared at all."""
    index: dict[str, Category] = {}
    for category in CATEGORIES:
        for phrase in category.phrases:
            key = identifying_words(phrase)
            if len(key) > 2:            # "to do" -> "do" would match everything
                index.setdefault(key, category)
    return index


_CATEGORY_BY_PHRASE: dict[str, Category] = _phrase_index()

#: Every catalogue synonym of a category member -> that member's display name,
#: so usage counted under "chrome" and "google chrome" is one application.
_CANONICAL: dict[str, str] = {
    synonym: member[0]
    for category in CATEGORIES for member in category.members for synonym in member}


def canonical(name: str) -> str:
    """The one name an application's launches are counted under."""
    return _CANONICAL.get(str(name).lower(), str(name).lower())


def category_of_app(name: str) -> Category | None:
    """The kind an application chiefly is ("google chrome" -> browser): the
    category that lists it earliest, so VLC is a video player before it is a
    music player."""
    name = canonical(name)
    best: tuple[int, Category] | None = None
    for category in CATEGORIES:
        for rank, member in enumerate(category.members):
            if member[0] == name and (best is None or rank < best[0]):
                best = (rank, category)
    return best[1] if best else None


def pretty_name(label: str, candidate: "Candidate | None" = None) -> str:
    """How to SAY an application's name: its own capitalisation when the
    catalogue has it ("Microsoft Edge"), otherwise capitalised words."""
    title = getattr(candidate, "title", "") if candidate is not None else ""
    if title and title.lower() == label.lower():
        return title
    try:      # the catalogue knows how the system writes it ("PowerShell")
        catalogue = globals().get("CATALOGUE")
        if catalogue is not None and catalogue.built:
            wanted = label.lower()
            for entry in catalogue.candidates:
                if entry.title and entry.name == wanted:
                    return entry.title
    except Exception:
        pass
    return " ".join(word[:1].upper() + word[1:] if word.isalpha() else word
                    for word in label.split())


def is_generic(request: str) -> bool:
    """'open an app', 'launch something' — a request that names no application."""
    words = identifying_words(request).split()
    return all(word in GENERIC for word in words)


def category_for(request: str) -> Category | None:
    """The kind of application a request names, if it names a kind.

    Whole request first ("a code editor" -> code editor), then the longest
    category phrase it contains ("play some music" -> music).
    """
    needle = identifying_words(request)
    if not needle:
        return None
    if needle in _CATEGORY_BY_PHRASE:
        return _CATEGORY_BY_PHRASE[needle]
    padded = f" {needle} "
    for phrase in sorted(_CATEGORY_BY_PHRASE, key=len, reverse=True):
        if f" {phrase} " in padded:
            return _CATEGORY_BY_PHRASE[phrase]
    return None


# ── the user's own folders ───────────────────────────────────────────────────

#: Spoken folder -> its value name under User Shell Folders. Read from the
#: registry because OneDrive redirects Documents/Desktop/Pictures away from
#: %USERPROFILE%, which is the case on this machine.
KNOWN_FOLDERS: dict[str, str] = {
    "documents": "Personal", "documents folder": "Personal",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "downloads folder": "{374DE290-123F-4565-9164-39C4925E467B}",
    "desktop folder": "Desktop",
    "pictures folder": "My Pictures", "music folder": "My Music",
    "videos folder": "My Video", "video folder": "My Video",
}
_FALLBACK_FOLDERS = {"Personal": "Documents", "Desktop": "Desktop",
                     "My Pictures": "Pictures", "My Music": "Music", "My Video": "Videos",
                     "{374DE290-123F-4565-9164-39C4925E467B}": "Downloads"}
_LOCATION_ALIASES = {
    "desktop": "desktop folder", "document": "documents", "docs": "documents",
    "download": "downloads", "pictures": "pictures folder", "photos": "pictures folder",
    "photo folder": "pictures folder", "music": "music folder", "videos": "videos folder",
    "video": "videos folder",
}


def known_folder(request: str, *, location: bool = False) -> Path | None:
    """The real path of a folder the request names ("my downloads"), or None."""
    words = identifying_words(request)
    # Bare music/photos denote apps in an app request and locations when used
    # by file search. Keep that distinction at the caller's explicit seam.
    if location:
        words = _LOCATION_ALIASES.get(words, words)
        if words in {"onedrive", "onedrive folder", "personal onedrive", "work onedrive",
                     "school onedrive"}:
            variables = ({"personal onedrive": ("OneDriveConsumer",),
                          "work onedrive": ("OneDriveCommercial",),
                          "school onedrive": ("OneDriveCommercial",)}.get(words)
                         or ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"))
            roots = {os.path.normcase(os.path.realpath(raw)): Path(raw)
                     for variable in variables
                     if (raw := os.environ.get(variable, "").strip()) and Path(raw).is_dir()}
            return next(iter(roots.values())) if len(roots) == 1 else None
    value = KNOWN_FOLDERS.get(words)
    if not value:
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer"
                            r"\User Shell Folders") as key:
            raw = str(winreg.QueryValueEx(key, value)[0])
        folder = Path(os.path.expandvars(raw))
        if folder.is_dir():
            return folder
    except Exception:
        pass
    folder = Path.home() / _FALLBACK_FOLDERS.get(value, "")
    return folder if folder.is_dir() else None


# ── Windows Settings pages ───────────────────────────────────────────────────
#
# "Open bluetooth settings" used to open Settings at its home page at best:
# every page has its own ms-settings: address. Only matched when the request
# says "settings" (or options/preferences), so "turn on bluetooth" is not
# taken for "show me the bluetooth page".

SETTINGS_PAGES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("bluetooth", "devices"), "Bluetooth & devices", "ms-settings:bluetooth"),
    (("wifi", "wi-fi", "wireless", "wi fi"), "Wi-Fi", "ms-settings:network-wifi"),
    (("network", "internet"), "Network & internet", "ms-settings:network-status"),
    (("ethernet",), "Ethernet", "ms-settings:network-ethernet"),
    (("hotspot",), "Mobile hotspot", "ms-settings:network-mobilehotspot"),
    (("airplane", "aeroplane", "flight mode"), "Airplane mode", "ms-settings:network-airplanemode"),
    (("vpn",), "VPN", "ms-settings:network-vpn"),
    (("proxy",), "Proxy", "ms-settings:network-proxy"),
    (("display", "screen", "resolution", "brightness", "monitor"), "Display", "ms-settings:display"),
    (("night light", "blue light"), "Night light", "ms-settings:nightlight"),
    (("sound", "audio", "volume", "speaker", "speakers", "headphones"), "Sound", "ms-settings:sound"),
    (("microphone permission", "microphone privacy", "mic privacy"), "Microphone privacy",
     "ms-settings:privacy-microphone"),
    (("camera permission", "camera privacy", "webcam privacy"), "Camera privacy",
     "ms-settings:privacy-webcam"),
    (("location",), "Location", "ms-settings:privacy-location"),
    (("privacy",), "Privacy & security", "ms-settings:privacy"),
    (("update", "updates", "windows update"), "Windows Update", "ms-settings:windowsupdate"),
    (("default apps", "default app", "default browser"), "Default apps", "ms-settings:defaultapps"),
    (("startup", "startup apps"), "Startup apps", "ms-settings:startupapps"),
    (("apps", "installed apps", "programs", "uninstall"), "Installed apps", "ms-settings:appsfeatures"),
    (("notification", "notifications"), "Notifications", "ms-settings:notifications"),
    (("focus", "do not disturb"), "Focus", "ms-settings:quiethours"),
    (("storage", "disk space", "space"), "Storage", "ms-settings:storagesense"),
    (("battery", "power", "sleep"), "Power & battery", "ms-settings:powersleep"),
    (("mouse", "touchpad", "trackpad"), "Mouse", "ms-settings:mousetouchpad"),
    (("keyboard", "typing"), "Typing", "ms-settings:typing"),
    (("printer", "printers", "scanner"), "Printers & scanners", "ms-settings:printers"),
    (("date", "time", "clock", "time zone"), "Date & time", "ms-settings:dateandtime"),
    (("language", "region"), "Language & region", "ms-settings:regionlanguage"),
    (("background", "wallpaper"), "Background", "ms-settings:personalization-background"),
    (("theme", "themes"), "Themes", "ms-settings:themes"),
    (("colour", "colours", "color", "colors", "dark mode", "light mode"), "Colours",
     "ms-settings:colors"),
    (("personalisation", "personalization"), "Personalisation", "ms-settings:personalization"),
    (("account", "accounts", "profile"), "Accounts", "ms-settings:yourinfo"),
    (("sign in", "sign-in", "password", "pin", "windows hello"), "Sign-in options",
     "ms-settings:signinoptions"),
    (("about", "pc info", "system info", "specs"), "About this PC", "ms-settings:about"),
    (("clipboard",), "Clipboard", "ms-settings:clipboard"),
    (("game mode", "gaming"), "Game Mode", "ms-settings:gaming-gamemode"),
    (("multitasking", "snap"), "Multitasking", "ms-settings:multitasking"),
    (("backup",), "Windows Backup", "ms-settings:backup"),
    (("recovery", "reset this pc"), "Recovery", "ms-settings:recovery"),
    (("troubleshoot", "troubleshooter"), "Troubleshoot", "ms-settings:troubleshoot"),
    (("accessibility", "ease of access"), "Accessibility", "ms-settings:easeofaccess"),
)
_SETTINGS_WORDS = ("settings", "setting", "options", "preferences", "control panel")


def settings_page(request: str) -> tuple[str, str] | None:
    """(page name, ms-settings URI) for "open bluetooth settings", or None."""
    text = " " + " ".join(re.findall(r"[a-z0-9-]+", str(request or "").lower())) + " "
    if not any(f" {word} " in text for word in _SETTINGS_WORDS):
        return None
    best: tuple[int, str, str] | None = None
    for words, name, uri in SETTINGS_PAGES:
        for word in words:
            if f" {word} " in text and (best is None or len(word) > best[0]):
                best = (len(word), name, uri)
    return (best[1], best[2]) if best else None


# ── the user's own choices: Windows default apps ─────────────────────────────

_DEFAULTS: dict[str, tuple[str, str] | None] = {}


def default_handler(association: str) -> tuple[str, str] | None:
    """(target, identity) of the app Windows opens for *association*.

    ``association`` is a URL scheme ("https", "mailto") or an extension
    (".pdf"). The identity is an executable stem ("msedge") or an
    AppUserModelID; the target is what to launch. Cached; never raises.
    """
    if association in _DEFAULTS:
        return _DEFAULTS[association]
    found: tuple[str, str] | None = None
    try:
        import winreg

        if association.startswith("."):
            key = (r"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts"
                   f"\\{association}\\UserChoice")
        else:
            key = (r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations"
                   f"\\{association}\\UserChoice")
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
            prog_id = str(winreg.QueryValueEx(handle, "ProgId")[0])
        try:      # a Store app: its ProgId names an AppUserModelID
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, f"{prog_id}\\Application") as app:
                aumid = str(winreg.QueryValueEx(app, "AppUserModelID")[0])
            found = (f"shell:AppsFolder\\{aumid}", aumid.lower())
        except OSError:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
                                f"{prog_id}\\shell\\open\\command") as command:
                line = str(winreg.QueryValueEx(command, "")[0]).strip()
            exe = line[1:line.index('"', 1)] if line.startswith('"') else line.split()[0]
            if exe.lower().endswith(".exe"):
                found = (exe, Path(exe).stem.lower())
    except Exception:
        found = None
    _DEFAULTS[association] = found
    return found


# ── what the user actually does: usage and taught meanings ───────────────────

USAGE_PATH = CONFIG_DIR / "app_usage.json"
#: A pick the user overrules within this window teaches the category.
CORRECTION_WINDOW_S = 120.0


class AppUsage:
    """Launch counts per application, and meanings the user has taught.

    ``launches``: display name -> {"count", "last"} — the most-used member of a
    kind wins ("browser" -> the one actually opened every day).
    ``taught``: phrase -> display name — "when I say browser I mean Chrome",
    either said outright or learned when the user immediately overrules a
    guess by opening another app of the same kind.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.launches: dict[str, dict[str, float]] = {}
        self.taught: dict[str, str] = {}
        self.last_pick: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._loaded = False

    def _file(self) -> Path:
        return self.path if self.path is not None else USAGE_PATH

    def load(self) -> "AppUsage":
        if self._loaded:
            return self
        self._loaded = True
        try:
            data = json.loads(self._file().read_text(encoding="utf-8"))
            launches = data.get("launches") or {}
            taught = data.get("taught") or {}
            self.launches = {str(k): {"count": float(v.get("count", 0)),
                                      "last": float(v.get("last", 0))}
                             for k, v in launches.items() if isinstance(v, dict)}
            self.taught = {str(k): str(v) for k, v in taught.items() if v}
        except Exception:
            pass
        return self

    def _save(self) -> None:
        try:
            from .atomic_io import atomic_write_text
            self._file().parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._file(), json.dumps(
                {"launches": self.launches, "taught": self.taught}, indent=1))
        except Exception:
            pass

    def count(self, name: str) -> float:
        self.load()
        return float(self.launches.get(name, {}).get("count", 0.0))

    def record(self, name: str, *, category: str = "", phrase: str = "",
               via_category: bool = False, now: float | None = None) -> str:
        """Count a launch of *name*. ``category`` is the kind it belongs to;
        ``via_category`` says the request named the kind (``phrase``) rather
        than the app. Returns a sentence when the launch taught something."""
        self.load()
        now = time.time() if now is None else now
        learned = ""
        with self._lock:
            entry = self.launches.setdefault(name, {"count": 0.0, "last": 0.0})
            entry["count"] += 1
            entry["last"] = now
            last = self.last_pick
            if via_category:
                self.last_pick = {"category": category, "app": name, "at": now,
                                  "phrase": identifying_words(phrase) or category}
            elif (last and category and last["category"] == category
                    and last["app"] != name and now - last["at"] <= CORRECTION_WINDOW_S):
                # Asked for a kind, got one app, then opened another of the same
                # kind by name within moments: that is a correction.
                self.taught[last["phrase"]] = name
                self.taught[f"kind:{category}"] = name
                learned = (f"Noted - I'll open {pretty_name(name)} when you ask for "
                           f"'{last['phrase']}' from now on.")
                self.last_pick = None
            self._save()
        return learned

    def teach(self, phrase: str, name: str) -> None:
        """'When I say <phrase>, open <name>.' A kind's phrase teaches the kind."""
        self.load()
        key = identifying_words(phrase) or phrase.strip().lower()
        with self._lock:
            self.taught[key] = name
            kind = _CATEGORY_BY_PHRASE.get(key)
            if kind is not None:
                self.taught[f"kind:{kind.name}"] = name
            self._save()

    def taught_for(self, phrase: str) -> str:
        self.load()
        return self.taught.get(identifying_words(phrase), "")

    def most_used(self, limit: int = 5) -> list[str]:
        self.load()
        ranked = sorted(self.launches.items(),
                        key=lambda item: (-item[1].get("count", 0), -item[1].get("last", 0)))
        return [name for name, _ in ranked[:limit]]


USAGE = AppUsage()


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
    title: str = ""      # the name as the system writes it ("Microsoft Edge")


@dataclass
class Match:
    candidate: Candidate | None
    score: float
    why: str
    alternatives: list[str] = field(default_factory=list)
    #: Set when the request named a KIND of application ("a browser").
    category: str = ""
    #: The name to say ("microsoft edge"), when nicer than the catalogue key
    #: ("msedge").
    display: str = ""

    @property
    def ok(self) -> bool:
        return self.candidate is not None

    @property
    def label(self) -> str:
        if self.display:
            return self.display
        return canonical(self.candidate.name) if self.candidate is not None else ""

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
        self.built_at = 0.0         # time.monotonic() of the last build

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
                                           str(shortcut), "start_menu", shortcut.stem))
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
                                       "apps_folder", name))
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
        self.built_at = time.monotonic()
        return self

    #: A failed lookup rebuilds a catalogue at least this old, so an app
    #: installed while ORION is running is found without a restart.
    STALE_AFTER_S = 120.0

    def refresh_if_stale(self) -> bool:
        """Rebuild when older than STALE_AFTER_S. True if it rebuilt."""
        if self.built and time.monotonic() - self.built_at < self.STALE_AFTER_S:
            return False
        self.build()
        return True

    # ── resolution ────────────────────────────────────────────────────────────

    def resolve(self, request: str, usage: AppUsage | None = None) -> Match:
        """What did they mean, and how sure are we?

        In order: a meaning the user taught; an alias; an application with
        exactly that name; a KIND of application named outright ("a browser");
        a strong name match; a kind mentioned inside a longer request ("play
        some music"); then the best weaker name match.
        """
        needle = normalise(request)
        if not needle:
            return Match(None, 0.0, "nothing was named")
        if not self.built:
            self.build()
        usage = USAGE if usage is None else usage
        words = identifying_words(request)

        # Their own word for it outranks everything.
        taught = usage.taught_for(request) if words else ""
        if taught:
            found = self._find_named(self._synonyms(taught))
            if found is not None:
                kind = category_of_app(taught)
                return Match(found, 1.0, f"you told me '{words}' means {pretty_name(taught)}",
                             category=kind.name if kind else "", display=taught)

        # An alias is a deliberate human decision and outranks fuzzy matching:
        # "word" must become WINWORD, never WordPad.
        aliased = ALIASES.get(needle)
        if aliased:
            resolved = self._launchable(aliased)
            if resolved is not None:
                return Match(resolved, 1.0,
                             f"'{needle}' is {aliased}", [])

        exact = any(c.name == needle and c.source != "alias" and not is_junk(c.name, needle)
                    for c in self.candidates)
        named_kind = _CATEGORY_BY_PHRASE.get(words) if words else None
        if named_kind is not None and not exact:
            kind_match = self._resolve_category(named_kind, usage)
            if kind_match is not None:
                return kind_match

        by_name = self._by_name(needle, usage)
        if by_name.ok and by_name.score >= 0.84:
            return by_name
        mentioned = None if exact else (named_kind or category_for(request))
        if mentioned is not None:
            kind_match = self._resolve_category(mentioned, usage)
            if kind_match is not None:
                return kind_match
            if not (by_name.ok and by_name.confident):
                # A kind of app that is not installed: say so, rather than
                # letting the caller go hunting for a document called "vpn".
                return Match(None, 0.0, f"no {mentioned.name} is installed",
                             by_name.alternatives, category=mentioned.name)
        if not (by_name.ok and by_name.confident):
            # A well-known application that simply is not on this PC ("open
            # firefox"): say that, and offer what does the same job.
            kind = category_of_app(needle) if needle in _CANONICAL else None
            if kind is not None:
                instead = self._resolve_category(kind, usage)
                offers = ([instead.label] + instead.alternatives) if instead else []
                return Match(None, 0.0, f"{pretty_name(_CANONICAL[needle])} is not installed",
                             offers[:4], category=kind.name)
        return by_name

    # ── kinds of application ─────────────────────────────────────────────────

    @staticmethod
    def _synonyms(name: str) -> tuple[str, ...]:
        display = canonical(name)
        for category in CATEGORIES:
            for member in category.members:
                if member[0] == display:
                    return member
        return (name.lower(),)

    _SOURCE_RANK = {"app_paths": 0, "apps_folder": 1, "start_menu": 2, "path": 3,
                    "builtin": 4, "default": 5, "alias": 6}

    def _find_named(self, synonyms: Iterable[str]) -> Candidate | None:
        """The launchable entry for an application known by these names."""
        for synonym in synonyms:
            hits = [c for c in self.candidates
                    if c.name == synonym and c.source != "alias" and not is_junk(c.name)]
            if hits:
                return min(hits, key=lambda c: self._SOURCE_RANK.get(c.source, 9))
            alias_target = ALIASES.get(synonym)
            if alias_target:
                found = self._launchable(alias_target)
                if found is not None:
                    return found
            if synonym.endswith(":"):
                return Candidate(synonym, synonym, "builtin")
        return None

    def _default_member(self, category: Category,
                        installed: list[tuple[str, Candidate]]) -> tuple[str, Candidate] | None:
        """The member Windows itself opens for this kind (the default browser)."""
        if not category.default:
            return None
        handler = default_handler(category.default)
        if handler is None:
            return None
        target, identity = handler
        display = ""
        for member in category.members:
            if identity in member:
                display = member[0]
                break
        if not display:           # a Store app: match its AppUserModelID
            for candidate in self.candidates:
                if candidate.source == "apps_folder" and \
                        candidate.target.lower().endswith(identity):
                    display = canonical(candidate.name)
                    break
        for name, candidate in installed:
            if name == display:
                return name, candidate
        if not display:
            display = identity if "!" not in identity else f"your default {category.name}"
        return display, Candidate(display, target, "default")

    def _resolve_category(self, category: Category, usage: AppUsage) -> Match | None:
        """Pick the member of *category* this user means. None if none installed."""
        installed: list[tuple[str, Candidate]] = []
        for member in category.members:
            found = self._find_named(member)
            if found is not None and all(name != member[0] for name, _ in installed):
                installed.append((member[0], found))
        default = self._default_member(category, installed)
        options = list(installed)
        if default is not None and all(name != default[0] for name, _ in options):
            options.append(default)
        if not options:
            return None
        usage.load()
        chosen: tuple[str, Candidate] | None = None
        why = ""
        preferred = usage.taught.get(f"kind:{category.name}", "")
        if preferred:
            chosen = next((o for o in options if o[0] == preferred), None)
            why = f"you prefer {pretty_name(preferred)} for this"
        if chosen is None:
            busiest = max(options, key=lambda o: usage.count(o[0]))
            if usage.count(busiest[0]) > 0:
                chosen, why = busiest, f"the {category.name} you use most"
        if chosen is None and default is not None and category.prefer_default:
            chosen, why = default, f"your default {category.name}"
        if chosen is None:
            chosen, why = options[0], f"the {category.name} you have installed"
        alternatives = [name for name, _ in options if name != chosen[0]][:4]
        return Match(chosen[1], 0.9, why, alternatives,
                     category=category.name, display=chosen[0])

    def _by_name(self, needle: str, usage: AppUsage) -> Match:
        """The best fuzzy match on application names."""
        scored: list[tuple[float, str, Candidate]] = []
        for candidate in self.candidates:
            if is_junk(candidate.name, needle):
                continue
            score, why = _score(needle, candidate.name)
            if score > 0:
                # The registry is authoritative; a Start Menu folder name is a
                # guess by comparison.
                if candidate.source == "app_paths":
                    score = min(1.0, score + 0.05)
                # What this user actually opens breaks near-ties.
                used = usage.count(canonical(candidate.name))
                if used and score < 1.0:
                    score = min(0.99, score + min(0.06, 0.02 * math.log2(1 + used)))
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

_LNK_TARGETS: dict[str, str] = {}


def _lnk_target(path: str) -> str:
    """The executable a Start Menu shortcut points at ('' if unknown)."""
    if path in _LNK_TARGETS:
        return _LNK_TARGETS[path]
    target = ""
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        try:
            target = str(win32com.client.Dispatch("WScript.Shell")
                         .CreateShortcut(path).TargetPath or "")
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        target = ""
    _LNK_TARGETS[path] = target
    return target


def process_stems(display: str, candidate: Candidate | None) -> set[str]:
    """Executable names (lower-case, no .exe) an application runs as."""
    stems = set()
    for synonym in AppCatalogue._synonyms(display) + (display,):
        compact = synonym.replace(" ", "")
        if compact and not compact.endswith(":"):
            stems.add(compact)
    target = candidate.target if candidate is not None else ""
    if target.lower().endswith(".lnk"):
        target = _lnk_target(target)
    if target.lower().endswith(".exe"):
        stems.add(Path(target).stem.lower())
    return stems


def close_targets(request: str, catalogue: AppCatalogue | None = None,
                  usage: AppUsage | None = None) -> tuple[str, list[tuple[str, set[str]]]]:
    """What a CLOSE request names: (kind or '', [(display, process stems)]).

    "close the browser" -> every installed browser, so the caller can close
    whichever is running; "close word" -> Word alone.
    """
    catalogue = CATALOGUE if catalogue is None else catalogue
    usage = USAGE if usage is None else usage
    if not catalogue.built:
        catalogue.build()
    words = identifying_words(request)
    exact = any(c.name == words and c.source != "alias" for c in catalogue.candidates)
    kind = None if exact else (_CATEGORY_BY_PHRASE.get(words) or category_for(request))
    if kind is not None:
        apps = []
        for member in kind.members:
            found = catalogue._find_named(member)
            if found is not None:
                apps.append((member[0], process_stems(member[0], found)))
        if apps:
            return kind.name, apps
    match = catalogue.resolve(request, usage)
    if match.ok and match.confident:
        return "", [(match.label, process_stems(match.label, match.candidate))]
    return "", []


def resolve_app(request: str) -> Match:
    return CATALOGUE.resolve(request)


_WARM_LOCK = threading.Lock()


def warm() -> None:
    """Build the catalogue ahead of the first request (1.1-1.3 s measured, most
    of it the Store-app enumeration) and read the default apps. Worker thread;
    once per process however many callers ask."""
    with _WARM_LOCK:
        try:
            if not CATALOGUE.built:
                CATALOGUE.build()
            for category in CATEGORIES:
                if category.default:
                    default_handler(category.default)
            USAGE.load()
        except Exception:
            pass


__all__ = ["ALIASES", "AppCatalogue", "AppUsage", "CATALOGUE", "CATEGORIES", "Candidate",
           "Category", "FILLER", "Match", "USAGE", "canonical", "category_for",
           "category_of_app", "default_handler", "identifying_words", "is_generic",
           "is_junk", "known_folder", "normalise", "resolve_app", "warm"]
