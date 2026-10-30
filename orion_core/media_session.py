"""
What is actually playing — Windows' media sessions, not blind media keys.

media_control used to press VK_MEDIA_PLAY_PAUSE for everything. That key is a
TOGGLE: "play" while music was already playing PAUSED it; "pause" with
nothing playing started whatever app last held the media keys; and every
call reported "Media command issued" whether or not anything was listening.

Windows keeps a list of media sessions (System Media Transport Controls —
the overlay that appears with the volume flyout): which app is playing, its
status and the track. Through it ORION can play when asked to play, pause
when asked to pause, say "nothing is playing" truthfully, and answer "what's
playing?". Measured: 13 ms to reach the session manager on this PC.

Needs ``winrt-Windows.Media.Control`` (with the winrt runtime ORION's OCR
already uses). Without it every function here returns None and the caller
falls back to the media keys. Synchronous: call from a worker thread (each
call runs its own short event loop for the WinRT async operations).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

#: Operation names -> the session method that performs them.
_OPERATIONS = {
    "play": "try_play_async",
    "pause": "try_pause_async",
    "toggle": "try_toggle_play_pause_async",
    "next": "try_skip_next_async",
    "previous": "try_skip_previous_async",
    "stop": "try_stop_async",
}


def available() -> bool:
    try:
        import winrt.windows.media.control  # noqa: F401
        return True
    except Exception:
        return False


def _run(coro: Any) -> Any:
    """Run a coroutine on a private event loop (this is a worker thread)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=4.0))
    finally:
        loop.close()


def _app_name(aumid: str) -> str:
    """'Spotify.exe' / 'SpotifyAB.SpotifyMusic_zpd...!Spotify' -> 'Spotify'.

    The Start-screen name first (Microsoft.ZuneMusic_8wekyb3d8bbwe!... is
    "Media Player"), from the app catalogue when it is already built."""
    try:
        from .app_resolver import CATALOGUE
        if CATALOGUE.built and aumid:
            wanted = f"shell:appsfolder\\{str(aumid).lower()}"
            for candidate in CATALOGUE.candidates:
                if candidate.target.lower() == wanted and (candidate.title or candidate.name):
                    return candidate.title or candidate.name
    except Exception:
        pass
    text = str(aumid or "")
    if "!" in text:
        text = text.split("!", 1)[1] or text.split("!", 1)[0]
    text = text.rsplit("\\", 1)[-1]
    if text.lower().endswith(".exe"):
        text = text[:-4]
    if "." in text and "_" in text:          # Publisher.App_hash
        text = text.split("_", 1)[0].split(".")[-1]
    return text or "an app"


async def _sessions() -> tuple[Any, list[Any]]:
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as Manager)
    manager = await Manager.request_async()
    return manager, list(manager.get_sessions())


def _status_name(session: Any) -> str:
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as Status)
    try:
        return Status(session.get_playback_info().playback_status).name.lower()
    except Exception:
        return "unknown"


async def _describe(session: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"app": _app_name(session.source_app_user_model_id),
                            "status": _status_name(session), "title": "", "artist": ""}
    try:
        props = await session.try_get_media_properties_async()
        info["title"] = str(props.title or "")
        info["artist"] = str(props.artist or "")
    except Exception:
        pass
    return info


async def _pick(manager: Any, sessions: list[Any]) -> Any:
    """The session a command means: one that is playing, else Windows'
    current session, else the first."""
    playing = [s for s in sessions if _status_name(s) == "playing"]
    if playing:
        return playing[0]
    current = manager.get_current_session()
    return current or (sessions[0] if sessions else None)


def now_playing() -> dict[str, Any] | None:
    """{'app','status','title','artist'} for the relevant session, {} when
    there is none, None when the API is unavailable. Never raises."""
    if not available():
        return None

    async def go() -> dict[str, Any]:
        manager, sessions = await _sessions()
        session = await _pick(manager, sessions)
        return await _describe(session) if session is not None else {}

    try:
        return _run(go())
    except Exception:
        return None


def sessions() -> list[dict[str, Any]] | None:
    """Every media session on the PC (a browser tab and Spotify can both
    have one), or None when the API is unavailable. Never raises."""
    if not available():
        return None

    async def go() -> list[dict[str, Any]]:
        _manager, found = await _sessions()
        return [await _describe(session) for session in found]

    try:
        return _run(go())
    except Exception:
        return None


def control(operation: str, app: str = "") -> dict[str, Any] | None:
    """Perform *operation* on the relevant session - or on *app*'s session
    when one is named ("Spotify", "Microsoft Edge").

    Returns {'done': bool, 'before': {...}, 'after': {...}} — 'before' empty
    when no session exists — or None when the API is unavailable.
    """
    method = _OPERATIONS.get(operation)
    if method is None or not available():
        return None

    async def go() -> dict[str, Any]:
        manager, sessions = await _sessions()
        if app:
            wanted = app.lower()
            sessions = [s for s in sessions
                        if wanted in _app_name(s.source_app_user_model_id).lower()]
            session = sessions[0] if sessions else None
        else:
            session = await _pick(manager, sessions)
        if session is None:
            return {"done": False, "before": {}, "after": {}}
        before = await _describe(session)
        done = bool(await getattr(session, method)())
        await asyncio.sleep(0.35)        # let the app report its new state
        after = await _describe(session)
        return {"done": done, "before": before, "after": after}

    try:
        return _run(go())
    except Exception:
        return None


def wait_for_session(timeout: float = 6.0) -> dict[str, Any] | None:
    """Poll until any media session exists (an app just opened), or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = now_playing()
        if found is None:
            return None
        if found:
            return found
        time.sleep(0.5)
    return None


def describe_track(info: dict[str, Any]) -> str:
    title, artist = info.get("title") or "", info.get("artist") or ""
    if title and artist:
        return f"'{title}' by {artist}"
    return f"'{title}'" if title else ""


__all__ = ["available", "control", "describe_track", "now_playing", "wait_for_session"]
