"""Task-local policy for background work that must not disturb the desktop.

This is independent of concurrency and retry classifications. It protects the
cooperative host routes, not arbitrary Python or raw native calls in plugins.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time
from typing import Iterator

from .data import ToolResult


@dataclass
class BackgroundScope:
    origin: str
    run_id: str = ""
    parent: "BackgroundScope | None" = None
    _revoked: threading.Event = field(default_factory=threading.Event, repr=False)

    _detached: threading.Event = field(default_factory=threading.Event, repr=False)
    #: A plan the owner started (attended_scope): until this monotonic time it
    #: stays his turn even after its Live call is answered and left to finish.
    attended_until: float = 0.0

    @property
    def active(self) -> bool:
        return not self._revoked.is_set() and (self.parent is None or self.parent.active)

    def revoke(self) -> None:
        self._revoked.set()

    @property
    def detached(self) -> bool:
        """Answered and left to finish in the background (a Live tool that
        outran its turn): still allowed its own work, no longer the owner's
        conversation - except a plan he started, while it is attended."""
        if self.attended_until and time.monotonic() < self.attended_until:
            return False
        return self._detached.is_set() or (self.parent is not None and self.parent.detached)

    def detach(self) -> None:
        self._detached.set()


_BACKGROUND: ContextVar[BackgroundScope | None] = ContextVar("orion_background_scope", default=None)
_FOREGROUND: ContextVar[BackgroundScope | None] = ContextVar("orion_foreground_authorization", default=None)


@contextmanager
def foreground_scope(origin: str) -> Iterator[None]:
    """Host-only opt-in for explicit user desktop controls, never model args."""
    if is_background():
        raise RuntimeError("A background task cannot upgrade itself to foreground control")
    if not str(origin).strip():
        raise ValueError("Foreground authorization needs an explicit host origin")
    grant = BackgroundScope(str(origin))
    token = _FOREGROUND.set(grant)
    try:
        yield
    finally:
        grant.revoke()
        _FOREGROUND.reset(token)


def foreground_authorized() -> bool:
    grant = _FOREGROUND.get()
    return grant is not None and grant.active and not is_background()


@contextmanager
def background_scope(origin: str, run_id: str = "") -> Iterator[BackgroundScope]:
    """Restrict descendants and copied worker contexts, revoking on every exit."""
    scope = BackgroundScope(str(origin), str(run_id), parent=_BACKGROUND.get())
    token = _BACKGROUND.set(scope)
    try:
        yield scope
    finally:
        scope.revoke()
        _BACKGROUND.reset(token)


#: The origin of a tool call made while answering the owner's own turn.
CONVERSATION = "conversation"


@contextmanager
def conversation_scope(how: str) -> Iterator[BackgroundScope]:
    """A tool call made while answering the owner's own turn - spoken, or typed
    in ORION's window (*how*: "voice", "typed", "reflex", ...).

    It IS a background scope: shared mouse and keyboard, clipboard writes,
    other applications and shell-open effects still defer
    (require_foreground), exactly as for any other tool call. The one
    difference is require_own_window: effects confined to ORION's own windows
    - a Command Deck page, his face, his dock - may run, because the owner is
    the one asking and nothing of his is touched. Mark XXXVI put every tool
    call in the background, which left ORION unable to show a deck page or a
    facial expression when asked (owner's log, 26 Oct 2026)."""
    with background_scope(CONVERSATION, run_id=str(how)) as scope:
        yield scope


#: Effects the owner's own live turn may have on the shared desktop: what he
#: asked for and can see happen (an application opening, a window moving,
#: music pausing, a file or a Settings page opening) and the confirmation
#: dialogs that ask HIM before anything irreversible. Mark XXXVI deferred all
#: of them for every tool call, so "open WhatsApp", "minimise this", "pause
#: the music" and "forget that" were refused when the owner said them (owner's
#: log and a direct check, 26 Oct 2026). Still deferred even then, because a
#: model choosing them is how a web page or a document could act through
#: him: synthetic keyboard and mouse input, the clipboard, browser and
#: messaging automation, gesture pointing, unrestricted processes, debugging
#: and plugins' own desktop effects. Every other background keeps all of them.
OWNER_TURN_EFFECTS: frozenset[str] = frozenset({
    # applications and windows he names
    "application_open", "application_launch", "application_shell_open",
    "application_close", "application_terminate", "game_application_launch",
    "outlook_application_launch", "workspace_restore",
    # media he asks to play, pause or skip
    "media_control", "media_keyboard", "media_session_control",
    "media_application_open", "media_app_control", "visible_media_playback",
    "spotify_visible_app", "spotify_shell_search", "spotify_shell_play",
    # files, folders, pages and settings he asks to see
    "shell_open_file", "shell_open_file_fallback", "shell_reveal_file",
    "shell_open_folder", "shell_open_camera_artifact", "browser_window",
    "shared_browser_open", "network_settings_window", "bluetooth_settings_window",
    "open messaging compose window in the default browser",
    "open messaging account setup",
    # his own machine settings and his own process
    "host_system_command", "screen_off", "orion_shutdown", "orion_restart",
    # dialogs that ask the owner first
    "process_confirmation_dialog", "file_confirmation_dialog",
    "cleanup_confirmation_dialog", "forget_confirmation_dialog",
    "telephony_confirmation_dialog", "mcp_confirmation_dialog", "messaging_confirmation_dialog",
    "mcp_configuration_dialog", "host_power_confirmation",
    # the pointer and the keyboard, when he asks (Mark XXXX.2). "Can you just
    # press Start Camera?" was refused as "needs a foreground session" four
    # times running and he was asked to do it himself (27 Oct 2026). Still
    # his live turn only: jobs, schedules, plugins, agents and the phone keep
    # every restriction, and the caption beside the pointer says what and why.
    "move_cursor", "keyboard_input", "edit_text_focus", "focus_window", "clipboard_write",
    # a robot he asks to move, and firmware he asks to flash (Mark XXXX.6);
    # background work never moves a robot or reprograms a board.
    "robot_motion", "robot_firmware_flash",
    # erasing his data asks the owner on screen first (erase_all.py)
    "erase_confirmation_dialog",
})
#: Window actions arrive as window_<verb> (minimise, maximise, close, focus...).
_OWNER_TURN_PREFIXES: tuple[str, ...] = ("window_",)


def _owner_turn(scope: BackgroundScope | None) -> bool:
    """Whether *scope* is the owner's own turn, still live: a conversation
    scope with nothing but conversation scopes above it, not cancelled
    (revoked) and not answered and left to finish in the background
    (detached)."""
    if scope is None:
        return False
    chain: BackgroundScope | None = scope
    while chain is not None:
        if chain.origin != CONVERSATION:
            return False
        chain = chain.parent
    return scope.active and not scope.detached


def owner_turn_allows(effect: str) -> bool:
    """Whether *effect* may run now because the owner is asking for it."""
    name = str(effect or "")
    return (_owner_turn(_BACKGROUND.get())
            and (name in OWNER_TURN_EFFECTS or name.startswith(_OWNER_TURN_PREFIXES)))


#: How long a plan the owner started keeps his turn's permissions once its
#: Live call has been answered and left to finish in the background.
PLAN_ATTENDED_S = 600.0


@contextmanager
def attended_scope(how: str = "plan", seconds: float = PLAN_ATTENDED_S) -> Iterator[BackgroundScope | None]:
    """A multi-step plan the owner asked for, run as his own turn.

    "Open Spotify and play my focus playlist" is one plan: Live answers after
    45 s and leaves the rest to finish, which used to cost the remaining steps
    the owner's permissions (require_foreground deferred them). Inside this
    scope the steps keep them for *seconds* (PLAN_ATTENDED_S) - every step that
    needs approval still asks, and Stop / a cancelled turn still ends it
    (revocation reaches it through the parent). Started outside a live owner
    turn (a job, a workflow, a schedule) it changes nothing."""
    if not _owner_turn(_BACKGROUND.get()):
        yield None
        return
    with background_scope(CONVERSATION, run_id=str(how)) as scope:
        scope.attended_until = time.monotonic() + max(0.0, float(seconds))
        yield scope


def require_own_window(effect: str) -> ToolResult | None:
    """For an effect confined to ORION's own windows. Allowed in the
    foreground and in a live conversation_scope; deferred in every other
    background (jobs, workflows, plugins, schedules, agents), once a
    conversational call has been cancelled (revoked) or answered and left to
    finish in the background (detached), and in a conversation started
    inside other background work."""
    if _owner_turn(_BACKGROUND.get()):
        return None
    return require_foreground(effect)


def is_background() -> bool:
    # A worker which outlives cancellation must retain the restriction, even
    # though the corresponding coroutine's scope is now revoked.
    return _BACKGROUND.get() is not None


def current_scope() -> BackgroundScope | None:
    return _BACKGROUND.get()


def require_foreground(effect: str) -> ToolResult | None:
    """Return a typed deferral before an action touches shared user input.

    Ordinary explicit foreground controls retain their existing behavior.
    There is no API that upgrades a nested background call to foreground.
    The owner's own live turn may have the effects in OWNER_TURN_EFFECTS.
    """
    if owner_turn_allows(effect):
        return None
    scope = _BACKGROUND.get()
    if scope is None:
        grant = _FOREGROUND.get()
        if grant is None or grant.active:
            return None
        scope = grant
    return ToolResult(
        f"'{effect}' needs a foreground session; the background task left your desktop unchanged.",
        ok=False, receipt={"status": "requires_foreground", "effect": str(effect),
                           "origin": scope.origin, "run_id": scope.run_id,
                           "background_scope_active": scope.active, "desktop_mutated": False})


__all__ = ["BackgroundScope", "CONVERSATION", "OWNER_TURN_EFFECTS", "PLAN_ATTENDED_S",
           "attended_scope", "background_scope",
           "conversation_scope", "current_scope", "is_background", "owner_turn_allows",
           "require_foreground", "require_own_window", "foreground_scope",
           "foreground_authorized"]
