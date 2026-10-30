"""
"Play <song / artist / playlist / video>" - find it, start it, check it started.

media_control could play, pause and skip whatever was already open, but had
no way to play something BY NAME: "play about time by Loyle Carner", "play my
Going out playlist" or "put lofi beats on YouTube" left the model to open an
app and hope.

  * Spotify (desktop app): ORION does what a person does - opens Spotify's
    own search for the request, reads the results through UI Automation
    (title, "Song • Artist", the row's Play button and, for the top row, its
    spotify:track: link), picks the best match (songs before podcast chapters
    and audiobooks), starts it, and confirms through Windows' media session
    that the requested track is what is now playing. No API key, no login,
    no web scraping - search engines answered a burst of lookups with a bot
    challenge. "My <name> playlist" is played from the Library sidebar.
  * YouTube: the top video is read from the results page and opened in the
    default browser.

Every outcome is checked; an unconfirmed one is reported as such rather than
"playing". Blocking (UI Automation, network, a few seconds of confirmation):
call it off the UI thread. Buttons are pressed through UIA Invoke, so the
user's mouse never moves.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .data import ToolResult

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/130.0 Safari/537.36")
_TIMEOUT_S = 8.0
CONFIRM_S = 8.0
RESULTS_WAIT_S = 6.0
LAUNCH_WAIT_S = 15.0

_STOP_WORDS = {"the", "a", "an", "by", "song", "songs", "track", "music", "some", "me", "please",
               "play", "on", "from", "of", "and", "feat", "ft", "video", "official", "my",
               "playlist", "album", "spotify", "youtube", "to", "for"}
_SPOKEN_KINDS = {"song", "music video"}
_URI = re.compile(r"spotify:(track|album|playlist|artist|episode|show):[A-Za-z0-9]{22}")


@dataclass
class Found:
    service: str            # "spotify" | "youtube"
    id: str = ""
    title: str = ""
    artist: str = ""
    kind: str = ""          # Spotify row kind: Song, Music video, Playlist, Album, Episode...
    uri: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def link(self) -> str:
        if self.uri:
            return self.uri
        if self.service == "youtube":
            return f"https://www.youtube.com/watch?v={self.id}"
        return ""

    def describe(self) -> str:
        return f"{self.title} by {self.artist}" if self.artist else self.title


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", str(text).lower().replace("’", "'"))
            if w not in _STOP_WORDS}


def score(query: str, found: Found) -> float:
    """Share of the request's words that the title and artist account for."""
    wanted = _words(query)
    if not wanted:
        return 0.0
    have = _words(found.title) | _words(found.artist)
    return len(wanted & have) / len(wanted)


def best(query: str, found: list[Found]) -> Found | None:
    """The best match; songs beat podcast chapters and audiobooks at equal
    score, and the service's own order (relevance) breaks remaining ties."""
    if not found:
        return None
    asked = _words(query)

    def rank(pair: tuple[int, Found]) -> tuple[float, int, int, int, int]:
        index, item = pair
        spoken = 0 if (not item.kind or item.kind.lower() in _SPOKEN_KINDS) else 1
        title = _words(item.title)
        # "Bohemian Rhapsody - Live at Amway Center..." is not what "play
        # bohemian rhapsody" means; nor is a karaoke, cover or sped-up cut.
        variant = 1 if (title & _VARIANTS) - asked else 0
        extra = len(title - asked)
        return (-round(score(query, item), 3), spoken, variant, extra, index)

    return sorted(enumerate(found), key=rank)[0][1]


# Not "remastered", "edit" or "mix": those are in canonical titles
# ("Bohemian Rhapsody - Remastered 2011", "Radio Edit").
_VARIANTS = {"live", "remix", "cover", "karaoke", "instrumental", "sped", "slowed", "acoustic",
             "reverb", "nightcore", "demo", "8d"}


# ── YouTube ──────────────────────────────────────────────────────────────────

def _get(url: str, cookie: str = "") -> str:
    headers = {"User-Agent": _UA, "Accept-Language": "en-GB,en;q=0.9"}
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return response.read().decode("utf-8", "replace")


def youtube_search(query: str, limit: int = 6) -> list[Found]:
    """Videos for *query* from YouTube's own results page (no API key).
    The consent cookie keeps a UK/EU request off the cookie wall."""
    page = _get("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query),
                cookie="SOCS=CAI; CONSENT=YES+1")
    match = re.search(r"var ytInitialData = (\{.*?\});</script>", page, re.S)
    if not match:
        return []
    found: list[Found] = []

    def walk(node: object) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            video = node.get("videoRenderer")
            if isinstance(video, dict) and video.get("videoId"):
                title = "".join(run.get("text", "") for run in
                                (video.get("title") or {}).get("runs") or [])
                owner = ((video.get("ownerText") or {}).get("runs") or [{}])[0].get("text", "")
                found.append(Found("youtube", video["videoId"], title, owner))
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    try:
        walk(json.loads(match.group(1)))
    except ValueError:
        return []
    return found


# ── Spotify (the desktop app, through its own search) ────────────────────────

def spotify_installed() -> bool:
    """The spotify: protocol is registered (classic installer or Store app)."""
    try:
        import winreg
        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "spotify"))
        return True
    except Exception:
        return False


def _spotify_window(timeout: float = 0.0):
    from . import screen_text
    deadline = time.monotonic() + timeout
    while True:
        window = screen_text.find_window("spotify")
        if window is not None and window.process.lower() == "spotify.exe":
            return window
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.4)


_ROW_KINDS = {"song", "music video", "artist", "album", "playlist", "podcast", "episode",
              "audiobook", "profile", "single", "ep", "show", "chapter"}


def _row_details(row: Any, auto: Any) -> tuple[Found, Any]:
    """(what a results row is, its Play button or None).

    The row's OWN name can be stale - Chromium kept "Mr. Brightside" on the
    row that had become the Loyle Carner artist card - so the title comes
    from the row's link and the kind from its subtitle, both re-rendered."""
    item = Found("spotify")
    controls = []
    for ctrl, _depth in auto.WalkControl(row, maxDepth=6):
        controls.append((ctrl.ControlTypeName, str(ctrl.Name or "").strip(), ctrl))
        match = _URI.search(str(ctrl.AutomationId or ""))
        if match and not item.uri:
            item.uri = match.group(0)
    for kind, name, _ctrl in controls:
        if kind == "HyperlinkControl" and name:
            item.title = name
            break
    if not item.title:
        item.title = next((name for kind, name, _c in controls if kind == "GroupControl" and name),
                          str(row.Name or "").strip())
    for kind, name, _ctrl in controls:
        if kind != "DataItemControl" or item.kind:
            continue
        if re.search(r"\s[•·]\s", name):
            parts = re.split(r"\s*[•·]\s*", name, maxsplit=1)
            item.kind, item.artist = parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        elif name.lower() in _ROW_KINDS:
            item.kind = name
    button = next((ctrl for kind, name, ctrl in controls
                   if kind == "ButtonControl" and name in {"Play", f"Play {item.title}"}), None)
    return item, button


_FILLER = re.compile(
    r"^(?:(?:can you |could you |please )?play\s+)?(?:me\s+)?(?:some(?:thing)?|a bit of|any|anything|"
    r"a song|a track|songs?|tracks?|music)(?:\s+(?:by|from))?\s+", re.I)


def _clean_request(query: str) -> str:
    """'some loyle carner' -> 'loyle carner': what a search box should get."""
    return _FILLER.sub("", query.strip(), count=1).strip()


def acceptable(query: str, item: Found) -> bool:
    """Good enough to play: most of the request is accounted for, or half of
    it with a title made only of requested words. A stale row (an old title
    under a new artist) covers half the request with a title nobody asked for."""
    value = score(query, item)
    if value >= 0.75:
        return True
    title = _words(item.title)
    return value >= 0.5 and bool(title) and title <= _words(query)


def _same_text(a: str, b: str) -> bool:
    """Spotify shows a searched phrase with '+' for its spaces."""
    def plain(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9']+", " ", str(text).lower()).split())
    return plain(a) == plain(b)


def _read_search(root: Any, auto: Any) -> tuple[str | None, list[tuple[Found, Any]]]:
    """(the search box's text, the results rows) on Spotify's search page.

    Elements vanish while Spotify re-renders its results, and UI Automation
    then raises COMError mid-read; such a read is simply not ready yet."""
    box: str | None = None
    grid = None
    try:
        for ctrl, _depth in auto.WalkControl(root, maxDepth=30):
            kind = ctrl.ControlTypeName
            if box is None and kind == "ComboBoxControl" and ctrl.Name == "What do you want to play?":
                try:
                    box = str(ctrl.GetValuePattern().Value or "")
                except Exception:
                    box = ""
            elif kind == "DataGridControl" and ctrl.Name == "Search results":
                grid = ctrl
                break
    except Exception:
        return None, []
    rows: list[tuple[Found, Any]] = []
    if grid is not None:
        try:
            children = grid.GetChildren()[:12]
        except Exception:
            return box, []
        for row in children:
            try:
                if row.ControlTypeName == "DataItemControl":
                    rows.append(_row_details(row, auto))
            except Exception:
                return box, []          # mid re-render: read it again shortly
    return box, rows


def _invoke(button: Any) -> bool:
    from .execution_policy import require_foreground
    if require_foreground("media_app_control") is not None:
        return False
    try:
        button.GetInvokePattern().Invoke()
        return True
    except Exception:
        return False


def _now_playing() -> dict:
    from . import media_session
    return media_session.now_playing() or {}


def _confirm(title: str = "", timeout: float = CONFIRM_S, *, changed_from: dict | None = None,
             app: str = "", artist: str = "") -> dict | None:
    """The media session once it shows *title* (or a track by *artist*)
    playing - or, with neither, anything other than *changed_from* - else
    None. Every session is checked: a browser tab starting while Spotify
    still plays is not the "current" session."""
    from . import media_session
    wanted = _words(title)
    by = _words(artist)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for info in media_session.sessions() or []:
            if info.get("status") != "playing":
                continue
            if app and app.lower() not in str(info.get("app") or "").lower():
                continue
            have = _words(info.get("title") or "")
            if wanted and len(wanted & have) >= max(1, (len(wanted) + 1) // 2):
                return info
            if by and by <= _words(info.get("artist") or ""):
                return info
            if not wanted and not by and (changed_from is None or changed_from.get("status") != "playing"
                               or info.get("title") != changed_from.get("title")):
                return info
        time.sleep(0.25)
    return None


def _quiet_others(started: dict) -> None:
    """Pause whatever ELSE is playing, so the new track is not on top of
    the old one (a YouTube tab over Spotify)."""
    from . import media_session
    for info in media_session.sessions() or []:
        if info.get("status") == "playing" and info.get("app") != started.get("app"):
            media_session.control("pause", app=str(info.get("app") or ""))


def _spotify_library_play(query: str, auto: Any, root: Any) -> Found | None:
    """Press Play on the Library playlist/album the request names."""
    candidates: list[tuple[Found, Any]] = []
    for ctrl, _depth in auto.WalkControl(root, maxDepth=32):
        name = str(ctrl.Name or "")
        if ctrl.ControlTypeName == "ButtonControl" and name.startswith("Play ") and len(name) > 5:
            candidates.append((Found("spotify", title=name[5:].strip(), kind="Playlist"), ctrl))
    if not candidates:
        return None
    choice = best(query, [item for item, _ in candidates])
    if choice is None or score(query, choice) < 0.75:
        return None
    for item, button in candidates:
        if item is choice and _invoke(button):
            return choice
    return None


def _play_spotify(query: str, *, prefer_library: bool) -> ToolResult | None:
    """Play *query* in the Spotify app. None: Spotify had nothing that
    matches (the caller may try YouTube)."""
    from .execution_policy import require_foreground
    blocked = require_foreground("spotify_visible_app")
    if blocked is not None:
        return blocked
    from . import screen_text
    if not screen_text.available():
        return None
    import uiautomation as auto

    before = _now_playing()
    window = _spotify_window()
    if window is not None and prefer_library:
        screen_text.enable_chromium_accessibility(window.hwnd)
        with auto.UIAutomationInitializerInThread():
            root = auto.ControlFromHandle(window.hwnd)
            chosen = _spotify_library_play(query, auto, root) if root is not None else None
        if chosen is not None:
            info = _confirm(timeout=CONFIRM_S, changed_from=before, app="spotify")
            if info is None:
                return ToolResult(f"I pressed play on {chosen.title} in Spotify, but it "
                                  "hasn't started.", ok=False)
            _quiet_others(info)
            from . import media_session
            track = media_session.describe_track(info)
            return ToolResult(f"Playing your {chosen.title} playlist on Spotify"
                              + (f" - {track}." if track else "."))

    search = re.sub(r"[:\\/?#]", " ", _clean_request(query)).strip() or query
    # Encoded: with raw spaces Spotify searched for the FIRST WORD only.
    search_link = "spotify:search:" + urllib.parse.quote(search)
    blocked = require_foreground("spotify_shell_search")
    if blocked is not None:
        return blocked
    os.startfile(search_link)  # type: ignore[attr-defined]
    window = window or _spotify_window(LAUNCH_WAIT_S)
    if window is None:
        return ToolResult("I asked Spotify to search, but its window never appeared.", ok=False)
    screen_text.enable_chromium_accessibility(window.hwnd)

    with auto.UIAutomationInitializerInThread():
        started_at = time.monotonic()
        deadline = started_at + RESULTS_WAIT_S + (0 if before else LAUNCH_WAIT_S / 2)
        choice, button, previous, reissued = None, None, None, False
        while time.monotonic() < deadline:
            root = auto.ControlFromHandle(window.hwnd)
            box, rows = _read_search(root, auto) if root is not None else (None, [])
            # Spotify can keep the PREVIOUS search on screen (or drop the
            # link entirely); rows only count once the box shows this query,
            # and a half re-rendered row - an old title under the new
            # artist - must not pass for a match.
            if box is not None and _same_text(box, search):
                pick = best(query, [item for item, _ in rows])
                if pick is not None and acceptable(query, pick):
                    key = (pick.title, pick.artist, pick.kind)
                    if key == previous:
                        choice = pick
                        button = next(b for item, b in rows if item is pick)
                        break
                    previous = key
            elif not reissued and time.monotonic() - started_at > 3.0:
                blocked = require_foreground("spotify_shell_search")
                if blocked is not None:
                    return blocked
                os.startfile(search_link)  # type: ignore[attr-defined]
                reissued = True
            time.sleep(0.35)
        if choice is None:
            return None
        # The row's own Play button first: it works whether or not something
        # is playing, and leaves the results on screen. A spotify:track: link
        # is the fallback - Spotify ignores one opened while a track is
        # PLAYING (measured), so it pauses first.
        started = button is not None and _invoke(button)
    # What "it started" looks like: the track's title; for an artist, one of
    # their tracks; for an album, its artist; for a playlist, anything new.
    kind = choice.kind.lower()
    if kind == "artist":
        check: dict[str, Any] = {"artist": choice.title}
    elif kind == "album" and choice.artist:
        check = {"artist": choice.artist}
    elif kind and kind not in _SPOKEN_KINDS:
        check = {"changed_from": before}
    else:
        check = {"title": choice.title}
    info = _confirm(timeout=4.0, app="spotify", **check) if started else None
    if info is None and choice.uri.startswith("spotify:track:"):
        from . import media_session
        if any(i.get("app") == "Spotify" and i.get("status") == "playing"
               for i in media_session.sessions() or []):
            media_session.control("pause", app="spotify")
            time.sleep(0.3)
        blocked = require_foreground("spotify_shell_play")
        if blocked is not None:
            return blocked
        os.startfile(choice.uri)  # type: ignore[attr-defined]
        started = True
        info = _confirm(app="spotify", **check)
    if not started:
        return ToolResult(f"I found {choice.describe()} in Spotify but couldn't press play.",
                          ok=False)
    if info is None:
        return ToolResult(f"I've started {choice.describe()} in Spotify, but I can't "
                          "confirm it's playing.", ok=False)
    _quiet_others(info)
    from . import media_session
    return ToolResult(f"Playing {media_session.describe_track(info) or choice.describe()} "
                      "on Spotify.")


def _play_youtube(query: str) -> ToolResult:
    try:
        candidates = youtube_search(query)
    except Exception as exc:
        return ToolResult(f"I couldn't search YouTube: {exc}", ok=False)
    choice = best(query, candidates)
    if choice is None:
        return ToolResult(f"YouTube found nothing for '{query}'.", ok=False)
    from . import browser
    if not browser.open_url(choice.link):
        return ToolResult(f"I found {choice.describe()} on YouTube but couldn't open a browser.",
                          ok=False)
    info = _confirm(choice.title, timeout=6.0)
    if info is not None:
        _quiet_others(info)
        return ToolResult(f"Playing {choice.describe()} on YouTube.")
    # Not every browser publishes a media session (and some block autoplay
    # with sound): the video is open, which is what can be confirmed.
    return ToolResult(f"Opened {choice.describe()} on YouTube.")


def play(query: str, service: str = "auto") -> ToolResult:
    """Find *query* and play it on *service*: "spotify", "youtube" or "auto"
    (Spotify when it is installed and has a match, otherwise YouTube)."""
    from .execution_policy import require_foreground
    blocked = require_foreground("visible_media_playback")
    if blocked is not None:
        return blocked
    query = re.sub(r"\s+", " ", str(query or "")).strip(" .?!")
    if not query:
        return ToolResult("Tell me what to play.", ok=False)
    service = str(service or "auto").lower().strip()
    lowered = query.lower()
    if service == "auto":
        if re.search(r"\b(on|from) youtube\b|\byoutube\b", lowered):
            service = "youtube"
        elif "spotify" in lowered:
            service = "spotify"
    query = re.sub(r"\s*\b(?:on|from|in) (?:youtube|spotify)\b", "", query, flags=re.I).strip()
    if service in {"spotify", "auto"}:
        if spotify_installed():
            prefer_library = bool(re.search(r"\bmy\b|\bplaylist\b", lowered))
            try:
                result = _play_spotify(query, prefer_library=prefer_library)
            except Exception as exc:
                result = ToolResult(f"Spotify didn't respond: {exc}", ok=False)
            if result is not None:
                return result
            if service == "spotify":
                return ToolResult(f"I couldn't find '{query}' on Spotify.", ok=False)
            fallback = _play_youtube(query)
            return ToolResult(f"Spotify had no match for '{query}'. {fallback.text}",
                              ok=fallback.ok)
        elif service == "spotify":
            return ToolResult("Spotify isn't installed on this PC.", ok=False)
    return _play_youtube(query)
