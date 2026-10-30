"""
ORION's doors in the Windows shell: orion:// links, "Ask ORION" in File
Explorer, the taskbar jump list, and the channel that carries a click on any
of them to the ORION that is already running.

Phase 1 of the Windows integration (Mark XXXVII.1), shell entry points lane.

Why a command channel
---------------------
Every one of these doors works by LAUNCHING ORION.exe with an argument:
Explorer runs ``ORION.exe --ask-file "C:\\file.pdf"``, the jump list runs
``ORION.exe --do briefing``, a browser runs ``ORION.exe "orion://show"``. The
single-instance guard refuses that second ORION, as it must, so on its own
every click would merely raise the window and drop the request. The refused
launch therefore forwards its request over a per-user local pipe to the
running instance before it exits (``forward``), and the running instance
listens on that pipe (``CommandServer``).

Why links get so little
-----------------------
Any web page can open an orion:// link; the browser asks once and the click
does the rest. So a link can only do what a stranger may do to ORION: bring
him forward, open a deck page, start the briefing, mute or unmute him, change
his face, put text in his input box (never sent - the owner presses Enter),
or answer a notification button whose single-use token the presence lane
issued. Nothing a link carries is ever run, opened or read. File commands are
CLI flags only: Explorer's menu and the jump list are the owner's own clicks.

Why nothing here touches the registry unless frozen
---------------------------------------------------
A developer checkout running from source must not repoint the owner's
Explorer menu and orion:// links at a python.exe in a work folder.
``launch_command()`` returns None unless this is the installed ORION.exe (or
ORION_SHELL_REGISTER=1 says otherwise), and everything that writes is driven
from that. Every key written carries "OrionManaged"="1", and a key without
that marker is never modified or deleted: it belongs to somebody else.

Everything is injectable (``reg=`` a winreg-like object, ``com=`` a COM
factory) so the tests never write the real registry or a real jump list.
Nothing here raises into its caller: each entry point returns a status.

The command schema (the contract app.py wires)
----------------------------------------------
Every command is a dict with "action", "origin" ("link" or "cli") and:

  show / briefing / mute / unmute / last_task   (no other fields)
  page      "page":  deck page name, letters only, lower case
  face      "form":  "screen" (the only face in this build; "classic" | "orb"
                      stay defined but are refused - see enabled_face_forms)
  dock      "state": "show" | "hide"                              (cli only)
  prefill   "text":  up to 500 characters; fills the input box, NEVER sent
  toast     "token": 16-64 url-safe characters, "verb": letters, lower case
  ask_file  "path":  an existing file or folder, absolute        (cli only)
  summarise "path":  an existing file or folder, absolute        (cli only)
  organise  "path":  an existing folder, absolute                (cli only)
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qsl, urlsplit

# ── limits and vocabulary ────────────────────────────────────────────────────

SCHEME = "orion"
PIPE_PREFIX = "orion-commands-"

#: One command line on the pipe, newline included.
MAX_PAYLOAD_BYTES = 8 * 1024
#: A whole link. Generous because 500 characters of non-Latin text are about
#: 4,500 characters once a browser has percent-encoded them.
MAX_LINK_CHARS = 8192
MAX_PREFILL_CHARS = 500
MAX_PATH_CHARS = 1024
MAX_PAGE_CHARS = 32
MAX_VERB_CHARS = 32

#: Every face the app can draw. Only the ones enabled_face_forms() returns may
#: be asked for (Mark XXXVII offers the Screen Face alone): the others stay
#: defined here, dormant, exactly as appearance.FACE_FORMS keeps them.
FACE_FORMS = ("screen", "classic", "orb")


def enabled_face_forms() -> tuple[str, ...]:
    """The faces a link, a --do verb or a jump-list task may ask for in this
    build - appearance.enabled_forms() (the Screen Face only; ORION_FACE_FORM
    re-enables one for development). Never raises."""
    try:
        from .appearance import enabled_forms

        return tuple(form for form in enabled_forms() if form in FACE_FORMS)
    except Exception:
        return ("screen",)

#: What a link may ask for. Anything else from a link is refused.
LINK_ACTIONS = frozenset({"show", "page", "briefing", "mute", "unmute",
                          "face", "prefill", "toast"})
#: Path-carrying actions: CLI only, never from a link.
PATH_ACTIONS = {"ask_file": "any", "summarise": "any", "organise": "dir"}
#: The dock is the owner's own click (jump list), never a link.
DOCK_STATES = ("show", "hide")
ALL_ACTIONS = LINK_ACTIONS | frozenset(PATH_ACTIONS) | frozenset({"last_task", "dock"})

#: CLI flags and the action each one carries (None: --do picks the action).
CLI_FLAGS: dict[str, str | None] = {
    "--ask-file": "ask_file",
    "--summarise": "summarise", "--summarize": "summarise",
    "--organise": "organise", "--organize": "organise",
    "--do": None,
}

#: --do verbs (the jump list's tasks) and the command each one becomes.
DO_VERBS: dict[str, dict[str, str]] = {
    "show": {"action": "show"},
    "briefing": {"action": "briefing"},
    "mute": {"action": "mute"},
    "unmute": {"action": "unmute"},
    "face:screen": {"action": "face", "form": "screen"},
    "face:classic": {"action": "face", "form": "classic"},
    "face:orb": {"action": "face", "form": "orb"},
    "dock:show": {"action": "dock", "state": "show"},
    "dock:hide": {"action": "dock", "state": "hide"},
    "last-task": {"action": "last_task"},
}

#: The fields each action carries besides "action" and "origin".
_FIELDS: dict[str, frozenset[str]] = {
    "show": frozenset(), "briefing": frozenset(), "mute": frozenset(),
    "unmute": frozenset(), "last_task": frozenset(),
    "page": frozenset({"page"}), "face": frozenset({"form"}),
    "dock": frozenset({"state"}),
    "prefill": frozenset({"text"}), "toast": frozenset({"token", "verb"}),
    "ask_file": frozenset({"path"}), "summarise": frozenset({"path"}),
    "organise": frozenset({"path"}),
}

_ACTION_WORD = re.compile(r"[A-Za-z]{1,16}")
_PAGE = re.compile(r"[A-Za-z]{1,%d}" % MAX_PAGE_CHARS)
_TOKEN = re.compile(r"[A-Za-z0-9_-]{16,64}")
_VERB = re.compile(r"[A-Za-z]{1,%d}" % MAX_VERB_CHARS)
#: Invisible or direction-changing characters, and lone surrogates: removed
#: from prefilled text so a link cannot hide words in the input box.
_INVISIBLE = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u00ad"
    "\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff"
    "\ud800-\udfff]")
_LINE_BREAKS = re.compile("[\t\r\n\u0085\u2028\u2029]+")
_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\", "\\??\\")


def _brief(exc: BaseException, limit: int = 160) -> str:
    text = (str(exc) or type(exc).__name__).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:limit]


# ── orion:// links ───────────────────────────────────────────────────────────

def _strict_query(query: str, names: set[str]) -> tuple[dict[str, str] | None, str]:
    """Exactly the parameters *names*, each once, nothing else."""
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=False,
                          max_num_fields=len(names) + 1)
    except ValueError:
        return None, "too many parameters"
    found: dict[str, str] = {}
    for key, value in pairs:
        if key not in names:
            return None, "unexpected parameter"
        if key in found:
            return None, "repeated parameter"
        found[key] = value
    if set(found) != names:
        return None, "missing parameter"
    return found, ""


def _clean_text(raw: Any) -> tuple[str | None, str]:
    """Prefill text: one line, visible characters only, 500 at most."""
    if not isinstance(raw, str):
        return None, "text is not a string"
    if len(raw) > MAX_PREFILL_CHARS:
        return None, "text is too long"
    text = _INVISIBLE.sub("", _LINE_BREAKS.sub(" ", raw))
    text = re.sub(" {2,}", " ", text).strip()
    if not text:
        return None, "text is empty"
    return text, ""


def _valid_token(raw: Any) -> bool:
    return isinstance(raw, str) and _TOKEN.fullmatch(raw) is not None


def _valid_verb(raw: Any) -> bool:
    return isinstance(raw, str) and _VERB.fullmatch(raw) is not None


def check_link(url: Any) -> tuple[dict[str, str] | None, str]:
    """Parse an orion:// link against the allowlist.

    Returns (command, "") or (None, reason). The reason is a fixed phrase,
    never the link itself, so it is safe to write to a log: a link is text
    from a stranger's web page.
    """
    if not isinstance(url, str):
        return None, "not text"
    if len(url) > MAX_LINK_CHARS:
        return None, "link is too long"
    raw = url.strip()
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in raw):
        return None, "control characters in link"
    if not raw.lower().startswith(SCHEME + "://"):
        return None, "not an orion:// link"
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None, "malformed link"
    if parts.scheme.lower() != SCHEME:
        return None, "not an orion:// link"
    # The "host" is the action: letters only, so user@host, :port, %xx and
    # an empty host (orion:///x) are all refused here.
    if _ACTION_WORD.fullmatch(parts.netloc or "") is None:
        return None, "malformed action"
    if parts.fragment:
        return None, "fragments are not accepted"
    action = parts.netloc.lower()
    rest = parts.path[1:] if parts.path.startswith("/") else parts.path
    if rest.endswith("/"):
        rest = rest[:-1]                    # orion://show/ from some browsers
    query = parts.query

    if action in ("show", "briefing", "mute", "unmute"):
        if rest or query:
            return None, "this action takes no arguments"
        return {"action": action, "origin": "link"}, ""
    if action == "page":
        if query:
            return None, "this action takes no parameters"
        # Raw path, not unquoted: %2e%2e is refused as surely as "..".
        if _PAGE.fullmatch(rest) is None:
            return None, "page name must be letters only"
        return {"action": "page", "page": rest.lower(), "origin": "link"}, ""
    if action == "face":
        if query:
            return None, "this action takes no parameters"
        form = rest.lower()
        if form not in FACE_FORMS:
            return None, "unknown face"
        if form not in enabled_face_forms():
            return None, "that face is not part of this version"
        return {"action": "face", "form": form, "origin": "link"}, ""
    if action == "ask":
        if rest:
            return None, "unexpected path"
        params, why = _strict_query(query, {"q"})
        if params is None:
            return None, why
        text, why = _clean_text(params["q"])
        if text is None:
            return None, why
        return {"action": "prefill", "text": text, "origin": "link"}, ""
    if action == "toast":
        if rest:
            return None, "unexpected path"
        params, why = _strict_query(query, {"t", "a"})
        if params is None:
            return None, why
        if not _valid_token(params["t"]):
            return None, "invalid token"
        if not _valid_verb(params["a"]):
            return None, "invalid verb"
        return {"action": "toast", "token": params["t"],
                "verb": params["a"].lower(), "origin": "link"}, ""
    return None, "unknown action"


def parse_link(url: Any) -> dict[str, str] | None:
    """The command an orion:// link asks for, or None (see check_link)."""
    return check_link(url)[0]


# ── paths ────────────────────────────────────────────────────────────────────

def _strip_long_prefix(text: str) -> str:
    """Python's own resolve() may hand back \\\\?\\ forms for long paths."""
    if text.upper().startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    if text.startswith("\\\\?\\"):
        return text[4:]
    return text


def _path_syntax(text: str) -> str:
    """Why *text* is not an acceptable path string, or ''."""
    if not text:
        return "empty path"
    if len(text) > MAX_PATH_CHARS:
        return "path is too long"
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in text):
        return "control characters in path"
    norm = text.replace("/", "\\")
    if norm.startswith(_DEVICE_PREFIXES):
        return "device paths are not accepted"
    if os.name == "nt":
        if re.match(r"^[A-Za-z]:(?![\\])", norm):
            return "drive-relative paths are not accepted"
        if ":" in norm[2:]:
            return "alternate data streams are not accepted"
    return ""


def validate_path(raw: Any, *, kind: str = "any", cwd: str | os.PathLike | None = None,
                  allow_relative: bool = True,
                  check_exists: bool = True) -> tuple[str | None, str]:
    """An existing, absolute, ordinary path - or (None, reason).

    *kind* is "file", "dir" or "any". Relative input resolves against *cwd*
    (the launching process's directory) when *allow_relative*; the running
    instance re-validates forwarded paths with ``allow_relative=False``.
    ``check_exists=False`` is the syntax-only check the running instance uses:
    the launching process has already touched the disk, and a stat of an
    unreachable network share on ORION's GUI thread would freeze him.
    """
    if not isinstance(raw, str):
        return None, "path is not a string"
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    why = _path_syntax(text)
    if why:
        return None, why
    candidate = Path(text)
    if not candidate.is_absolute():
        if not allow_relative:
            return None, "path is not absolute"
        candidate = Path(cwd or os.getcwd()) / candidate
    if not check_exists:
        result = os.path.normpath(str(candidate))
        if result != str(candidate) or not Path(result).is_absolute():
            return None, "path is not normalised"
        return result, ""
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None, "path does not exist"
    result = _strip_long_prefix(str(resolved))
    why = _path_syntax(result)
    if why:
        return None, why
    if not Path(result).is_absolute():
        return None, "path is not absolute"
    try:
        if kind == "file" and not resolved.is_file():
            return None, "not a file"
        if kind == "dir" and not resolved.is_dir():
            return None, "not a folder"
    except OSError:
        return None, "path cannot be read"
    return result, ""


# ── CLI flags ────────────────────────────────────────────────────────────────

def _scan_flags(argv: Iterable[Any]) -> tuple[list[tuple[str, Any]], str]:
    args = list(argv)
    found: list[tuple[str, Any]] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if not isinstance(arg, str):
            continue
        name, eq, inline = arg.partition("=")
        key = name.strip().lower()
        if key not in CLI_FLAGS:
            continue
        if eq:
            found.append((key, inline))
        elif index < len(args):
            found.append((key, args[index]))
            index += 1
        else:
            return found, "a shell flag is missing its value"
    return found, ""


def check_argv(argv: Iterable[Any], *,
               cwd: str | os.PathLike | None = None) -> tuple[dict[str, str] | None, str]:
    """Parse ORION's shell CLI flags. Returns (command, "") or (None, reason).

    These come from Explorer's menu and the jump list - the owner's own
    clicks - so they may carry paths. One shell flag per launch.
    """
    try:
        found, why = _scan_flags(argv)
    except TypeError:
        return None, "argv is not a list"
    if why:
        return None, why
    if not found:
        return None, "no shell command"
    if len(found) > 1:
        return None, "more than one shell command"
    flag, value = found[0]
    if flag == "--do":
        verb = value.strip().lower() if isinstance(value, str) else ""
        command = DO_VERBS.get(verb)
        if command is None:
            return None, "unknown --do verb"
        if command["action"] == "face" and command["form"] not in enabled_face_forms():
            return None, "that face is not part of this version"
        return dict(command, origin="cli"), ""
    action = CLI_FLAGS[flag] or ""
    path, why = validate_path(value, kind=PATH_ACTIONS[action], cwd=cwd)
    if path is None:
        return None, why
    return {"action": action, "path": path, "origin": "cli"}, ""


def parse_argv(argv: Iterable[Any]) -> dict[str, str] | None:
    """The command ORION's CLI flags ask for, or None (see check_argv)."""
    return check_argv(argv)[0]


def _links_in(argv: Iterable[Any]) -> list[str]:
    return [a for a in argv if isinstance(a, str)
            and a.strip().lower().startswith(SCHEME + ":")]


def check_command_argv(argv: Iterable[Any], *,
                       cwd: str | os.PathLike | None = None) -> tuple[dict[str, str] | None, str]:
    """The shell command a whole launch carries: a link, else CLI flags.

    A launch that carries a link is a LINK launch and nothing else in it is
    read. A browser builds that command line from a web page's URL, so if
    one ever passed quotes through unescaped, ``orion://x" --ask-file "C:\\...``
    would arrive as extra arguments - which are therefore never trusted.
    """
    try:
        args = list(argv)
    except TypeError:
        return None, "argv is not a list"
    links = _links_in(args)
    if links:
        if len(links) > 1:
            return None, "more than one link"
        return check_link(links[0])
    return check_argv(args, cwd=cwd)


def command_from_argv(argv: Iterable[Any]) -> dict[str, str] | None:
    """The shell command this launch carries, or None."""
    return check_command_argv(argv)[0]


def argv_may_carry_command(argv: Iterable[Any]) -> bool:
    """Cheap test: is there anything in *argv* worth parsing?"""
    try:
        args = list(argv)
    except TypeError:
        return False
    for arg in args:
        if not isinstance(arg, str):
            continue
        low = arg.strip().lower()
        if low.startswith(SCHEME + ":") or low.partition("=")[0] in CLI_FLAGS:
            return True
    return False


def strip_shell_args(argv: Iterable[Any]) -> list[Any]:
    """*argv* without links or shell flags (and their values).

    For a self-restart: relaunching with the same argv would replay the
    request that started this ORION (a second briefing, a second prefill).
    """
    args = list(argv)
    kept: list[Any] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if isinstance(arg, str):
            low = arg.strip().lower()
            if low.startswith(SCHEME + ":"):
                continue
            name, eq, _ = low.partition("=")
            if name in CLI_FLAGS:
                if not eq and index < len(args):
                    index += 1
                continue
        kept.append(arg)
    return kept


# ── re-validation on the receiving side ──────────────────────────────────────

def check_command(obj: Any, *, check_exists: bool = False) -> tuple[dict[str, str] | None, str]:
    """Re-validate a command dict (one that came off the pipe).

    Rebuilt field by field with the same validators the parsers use; unknown
    fields are refused. "origin" defaults to "cli"; a "link" origin may only
    carry link actions.
    """
    if not isinstance(obj, dict):
        return None, "command is not an object"
    action = obj.get("action")
    if not isinstance(action, str) or action not in ALL_ACTIONS:
        return None, "unknown action"
    origin = obj.get("origin", "cli")
    if origin not in ("link", "cli"):
        return None, "unknown origin"
    if origin == "link" and action not in LINK_ACTIONS:
        return None, "action not allowed from a link"
    extra = set(obj) - {"action", "origin"} - _FIELDS[action]
    if extra:
        return None, "unexpected field"
    missing = _FIELDS[action] - set(obj)
    if missing:
        return None, "missing field"
    out: dict[str, str] = {"action": action, "origin": origin}
    if action == "page":
        page = obj["page"]
        if not isinstance(page, str) or _PAGE.fullmatch(page) is None:
            return None, "page name must be letters only"
        out["page"] = page.lower()
    elif action == "face":
        if obj["form"] not in FACE_FORMS:
            return None, "unknown face"
        if obj["form"] not in enabled_face_forms():
            return None, "that face is not part of this version"
        out["form"] = obj["form"]
    elif action == "dock":
        if obj["state"] not in DOCK_STATES:
            return None, "unknown dock state"
        out["state"] = obj["state"]
    elif action == "prefill":
        text, why = _clean_text(obj["text"])
        if text is None:
            return None, why
        out["text"] = text
    elif action == "toast":
        if not _valid_token(obj["token"]):
            return None, "invalid token"
        if not _valid_verb(obj["verb"]):
            return None, "invalid verb"
        out["token"] = obj["token"]
        out["verb"] = obj["verb"].lower()
    elif action in PATH_ACTIONS:
        path, why = validate_path(obj["path"], kind=PATH_ACTIONS[action],
                                  allow_relative=False, check_exists=check_exists)
        if path is None:
            return None, why
        out["path"] = path
    return out, ""


def validate_command(obj: Any, *, check_exists: bool = False) -> dict[str, str] | None:
    return check_command(obj, check_exists=check_exists)[0]


def decode_command(line: bytes) -> tuple[dict[str, str] | None, str]:
    """One pipe line (without its newline) -> command."""
    if len(line) > MAX_PAYLOAD_BYTES:
        return None, "payload is too large"
    try:
        obj = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, "payload is not JSON"
    return check_command(obj)


def encode_command(command: dict[str, Any]) -> bytes | None:
    """command -> one pipe line, or None if it is invalid or too large."""
    checked = validate_command(command)
    if checked is None:
        return None
    payload = json.dumps(checked, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8") + b"\n"
    return payload if len(payload) <= MAX_PAYLOAD_BYTES else None


# ── the command channel ──────────────────────────────────────────────────────

def pipe_name(user: str | None = None) -> str:
    """This user's pipe. Named pipes are machine-wide, so the user is in the
    name; the server also restricts the pipe to this user's account."""
    if user is None:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
        if not user:
            try:
                import getpass
                user = getpass.getuser()
            except Exception:
                user = "owner"
    try:
        from .single_instance import MUTEX_NAME
    except Exception:
        MUTEX_NAME = "ORION"
    digest = hashlib.sha256(f"{user.lower()}|{MUTEX_NAME}".encode("utf-8")).hexdigest()
    return PIPE_PREFIX + digest[:16]


def _qt() -> tuple[Any, Any]:
    from PyQt6 import QtCore, QtNetwork
    return QtCore, QtNetwork


def forward(command: dict[str, Any], timeout_s: float = 2.0, *,
            name: str | None = None) -> bool:
    """Send *command* to the running ORION. True only if he accepted it.

    Plain blocking QLocalSocket: the refused second launch has no event loop
    and no application object, and needs neither (checked across two real
    processes). Never raises.
    """
    try:
        payload = encode_command(command)
        if payload is None:
            return False
        _QtCore, QtNetwork = _qt()
        deadline = time.monotonic() + max(0.05, float(timeout_s))

        def remaining_ms() -> int:
            return max(1, int((deadline - time.monotonic()) * 1000))

        sock = QtNetwork.QLocalSocket()
        try:
            sock.connectToServer(name or pipe_name())
            if not sock.waitForConnected(remaining_ms()):
                return False
            sock.write(payload)
            while sock.bytesToWrite() > 0:
                if time.monotonic() >= deadline or not sock.waitForBytesWritten(remaining_ms()):
                    return False
            reply = bytearray()
            while b"\n" not in reply and len(reply) < 256:
                if sock.bytesAvailable() <= 0:
                    if time.monotonic() >= deadline or not sock.waitForReadyRead(remaining_ms()):
                        break
                reply.extend(bytes(sock.readAll().data()))
            reply.extend(bytes(sock.readAll().data()))
        finally:
            try:
                sock.abort()
            except Exception:
                pass
        line = bytes(reply).split(b"\n", 1)[0]
        answer = json.loads(line.decode("utf-8")) if line else {}
        return isinstance(answer, dict) and answer.get("ok") is True
    except Exception:
        return False


def allow_foreground_handoff() -> bool:
    """Let the running ORION take the foreground.

    The process the owner just launched holds the right to come to the
    front; the running ORION does not. Handing it over is what lets him
    raise a window that was hidden to the tray, where raise_existing_window
    (which looks for a VISIBLE window) cannot find it.
    """
    if os.name != "nt":
        return False
    try:
        ASFW_ANY = 0xFFFFFFFF
        return bool(ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY))
    except Exception:
        return False


def forward_launch(argv: Iterable[Any] | None = None, *,
                   sender: Callable[[dict[str, Any]], bool] | None = None,
                   allow_foreground: Callable[[], Any] | None = None) -> bool | None:
    """The refused second launch's half: forward its shell command, if any.

    None: the launch carried no shell command (behave exactly as before).
    True/False: whether the running ORION accepted it. Never raises.
    """
    try:
        args = list(sys.argv if argv is None else argv)
        if not argv_may_carry_command(args):
            return None
        command, _why = check_command_argv(args)
        if command is None:
            return False
        try:
            (allow_foreground or allow_foreground_handoff)()
        except Exception:
            pass
        return bool((sender or forward)(command))
    except Exception:
        return False


def _build_command_server() -> type:
    QtCore, QtNetwork = _qt()

    class CommandServer(QtCore.QObject):
        """The running ORION's end of the pipe.

        One JSON object per line, at most 8 KB; each connection gets
        {"ok": true|false} and is closed. Valid commands are emitted on
        ``command``; refusals on ``rejected`` with a log-safe reason.
        """

        command = QtCore.pyqtSignal(dict)
        rejected = QtCore.pyqtSignal(str)

        def __init__(self, name: str | None = None, parent: Any = None, *,
                     max_bytes: int = MAX_PAYLOAD_BYTES, idle_ms: int = 3000,
                     max_clients: int = 8) -> None:
            super().__init__(parent)
            self._name = name or pipe_name()
            self._max_bytes = int(max_bytes)
            self._idle_ms = int(idle_ms)
            self._max_clients = int(max_clients)
            self._server: Any = None
            self._clients: dict[Any, bytearray] = {}
            self._error = ""

        @property
        def name(self) -> str:
            return self._name

        def is_listening(self) -> bool:
            return self._server is not None and self._server.isListening()

        def error(self) -> str:
            return self._error

        def start(self) -> bool:
            if self.is_listening():
                return True
            try:
                server = QtNetwork.QLocalServer(self)
                server.setSocketOptions(QtNetwork.QLocalServer.SocketOption.UserAccessOption)
                server.setMaxPendingConnections(self._max_clients)
                server.newConnection.connect(self._accept)
                ok = server.listen(self._name)
                if not ok and os.name != "nt":
                    # A stale socket file from a crash (POSIX only; Windows
                    # pipes die with their process).
                    QtNetwork.QLocalServer.removeServer(self._name)
                    ok = server.listen(self._name)
                if not ok:
                    self._error = server.errorString() or "listen failed"
                    server.deleteLater()
                    return False
                self._server = server
                self._error = ""
                return True
            except Exception as exc:
                self._error = _brief(exc)
                return False

        def stop(self) -> None:
            for sock in list(self._clients):
                self._discard(sock)
            self._clients.clear()
            server, self._server = self._server, None
            if server is not None:
                try:
                    server.close()
                    server.deleteLater()
                except Exception:
                    pass

        # -- connections ---------------------------------------------------

        def _accept(self) -> None:
            server = self._server
            if server is None:
                return
            try:
                while server.hasPendingConnections():
                    sock = server.nextPendingConnection()
                    if sock is None:
                        break
                    if len(self._clients) >= self._max_clients:
                        self._discard(sock)
                        continue
                    self._clients[sock] = bytearray()
                    sock.readyRead.connect(lambda s=sock: self._read(s))
                    sock.disconnected.connect(lambda s=sock: self._gone(s))
                    timer = QtCore.QTimer(sock)
                    timer.setSingleShot(True)
                    timer.timeout.connect(lambda s=sock: self._finish(s, False, "connection idle"))
                    timer.start(self._idle_ms)
                    if sock.bytesAvailable() > 0:
                        self._read(sock)
            except Exception as exc:
                self._error = _brief(exc)

        def _read(self, sock: Any) -> None:
            try:
                buffer = self._clients.get(sock)
                if buffer is None:
                    return
                buffer.extend(bytes(sock.readAll().data()))
                newline = buffer.find(b"\n")
                if newline < 0:
                    if len(buffer) >= self._max_bytes:
                        self._finish(sock, False, "payload is too large")
                    return
                line = bytes(buffer[:newline])
                if newline + 1 > self._max_bytes:
                    self._finish(sock, False, "payload is too large")
                    return
                command, why = decode_command(line)
                self._finish(sock, command is not None, why)
                if command is not None:
                    self.command.emit(command)
            except Exception as exc:
                self._finish(sock, False, f"read failed ({_brief(exc, 60)})")

        def _finish(self, sock: Any, ok: bool, why: str = "") -> None:
            if self._clients.pop(sock, None) is None:
                return
            try:
                sock.write(b'{"ok": true}\n' if ok else b'{"ok": false}\n')
                sock.flush()
                if sock.bytesToWrite() > 0:
                    sock.waitForBytesWritten(100)
                sock.disconnected.connect(sock.deleteLater)
                sock.disconnectFromServer()
                if sock.state() == QtNetwork.QLocalSocket.LocalSocketState.UnconnectedState:
                    sock.deleteLater()
            except Exception:
                self._discard(sock)
            if not ok and why:
                self.rejected.emit(why)

        def _gone(self, sock: Any) -> None:
            if self._clients.pop(sock, None) is not None:
                self._discard(sock)

        @staticmethod
        def _discard(sock: Any) -> None:
            try:
                sock.abort()
                sock.deleteLater()
            except Exception:
                pass

    return CommandServer


def _command_server_class() -> type:
    cls = globals().get("CommandServer")
    if cls is None:
        cls = _build_command_server()
        globals()["CommandServer"] = cls
    return cls


def __getattr__(name: str) -> Any:
    # CommandServer is built on first use so that importing this module -
    # which the refused second launch does to parse its argv - costs no Qt.
    if name == "CommandServer":
        return _command_server_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── what Windows should launch ───────────────────────────────────────────────

@dataclass(frozen=True)
class LaunchTarget:
    """The program Windows runs for ORION's shell entries."""

    path: str                 # the executable
    args: str = ""            # arguments that precede the shell flag
    icon: str = ""            # icon file (index 0)

    @property
    def command(self) -> str:
        quoted = f'"{self.path}"'
        return f"{quoted} {self.args}" if self.args else quoted


def launch_target(*, frozen: bool | None = None, executable: str | None = None,
                  env: dict[str, str] | None = None) -> LaunchTarget | None:
    """The installed ORION.exe, or None for a developer checkout.

    Frozen, sys.executable IS ORION.exe. From source it is a python.exe in a
    work folder, and registering that would hijack the owner's real Explorer
    menu and links - so None, unless ORION_SHELL_REGISTER=1 asks for it.
    """
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    exe = str(Path(executable or sys.executable).absolute())
    if frozen:
        return LaunchTarget(exe, "", exe)
    environ = os.environ if env is None else env
    if str(environ.get("ORION_SHELL_REGISTER", "")).strip() != "1":
        return None
    try:
        from .constants import BASE_DIR
        from .desktop_app import ICON_PATH
        script = BASE_DIR / "orion.py"
        icon = str(ICON_PATH) if ICON_PATH.exists() else exe
    except Exception:
        script = Path(__file__).resolve().parents[1] / "orion.py"
        icon = exe
    interpreter = Path(exe)
    windowed = interpreter.with_name(interpreter.name.lower().replace("python", "pythonw"))
    if "python" in interpreter.name.lower() and windowed.exists():
        interpreter = windowed
    return LaunchTarget(str(interpreter), f'"{script}"', icon)


def launch_command(*, frozen: bool | None = None, executable: str | None = None,
                   env: dict[str, str] | None = None) -> str | None:
    """The quoted command prefix Windows should run, or None (see launch_target)."""
    target = launch_target(frozen=frozen, executable=executable, env=env)
    return target.command if target else None


def _split_command(command: str) -> tuple[str, str]:
    """'"C:\\a b\\x.exe" rest' -> ('C:\\a b\\x.exe', 'rest')."""
    text = str(command or "").strip()
    if text.startswith('"'):
        end = text.find('"', 1)
        if end > 0:
            return text[1:end], text[end + 1:].strip()
        return text[1:], ""
    head, _, rest = text.partition(" ")
    return head, rest.strip()


def _default_icon(command: str) -> str:
    return f"{_split_command(command)[0]},0"


# ── registration (HKCU only) ─────────────────────────────────────────────────

CLASSES_KEY = "Software\\Classes"
MARKER_NAME = "OrionManaged"
MARKER_VALUE = "1"
PROTOCOL_KEY = SCHEME


@dataclass(frozen=True)
class ContextVerb:
    key: str            # under HKCU\Software\Classes
    label: str          # MUIVerb, Title Case
    flag: str
    placeholder: str    # %1 (the item) or %V (the folder behind the click)


CONTEXT_VERBS: tuple[ContextVerb, ...] = (
    ContextVerb("*\\shell\\ORION.Ask", "Ask ORION About This File", "--ask-file", "%1"),
    ContextVerb("Directory\\shell\\ORION.Ask", "Ask ORION About This Folder", "--ask-file", "%1"),
    ContextVerb("Directory\\shell\\ORION.Organise", "Organise with ORION", "--organise", "%1"),
    ContextVerb("Directory\\Background\\shell\\ORION.Ask", "Ask ORION About This Folder",
                "--ask-file", "%V"),
)


def _winreg(reg: Any) -> Any:
    if reg is not None:
        return reg
    import winreg
    return winreg


def _full(subkey: str) -> str:
    return f"{CLASSES_KEY}\\{subkey}"


def _ownership(reg: Any, subkey: str) -> str:
    """'absent', 'ours' (carries the marker) or 'foreign'."""
    try:
        key = reg.OpenKey(reg.HKEY_CURRENT_USER, _full(subkey), 0, reg.KEY_READ)
    except OSError:
        return "absent"
    with key:
        try:
            value, _kind = reg.QueryValueEx(key, MARKER_NAME)
        except OSError:
            return "foreign"
    return "ours" if str(value) == MARKER_VALUE else "foreign"


def _apply(reg: Any, root: str, plan: list[tuple[str, str, str]]) -> tuple[str, int]:
    """Write *plan* (subkey, value name, text) under a key ORION owns.

    The root's marker is written first, so a crash part-way leaves a key
    ORION still recognises as his to finish or remove. Values already equal
    are not rewritten.
    """
    state = _ownership(reg, root)
    if state == "foreign":
        return "foreign", 0
    writes = 0
    marked: set[str] = set()
    for subkey, name, text in plan:
        with reg.CreateKeyEx(reg.HKEY_CURRENT_USER, _full(subkey), 0,
                             reg.KEY_READ | reg.KEY_WRITE) as key:
            wanted = [(MARKER_NAME, MARKER_VALUE)] if subkey not in marked else []
            marked.add(subkey)
            for value_name, value in wanted + [(name, text)]:
                try:
                    current, kind = reg.QueryValueEx(key, value_name)
                except OSError:
                    current, kind = None, None
                if current != value or kind != reg.REG_SZ:
                    reg.SetValueEx(key, value_name, 0, reg.REG_SZ, value)
                    writes += 1
    return ("registered" if writes else "unchanged"), writes


def _delete_tree(reg: Any, subkey: str) -> None:
    with reg.OpenKey(reg.HKEY_CURRENT_USER, _full(subkey), 0, reg.KEY_READ) as key:
        children: list[str] = []
        index = 0
        while True:
            try:
                children.append(reg.EnumKey(key, index))
            except OSError:
                break
            index += 1
    for child in children:
        _delete_tree(reg, f"{subkey}\\{child}")
    reg.DeleteKey(reg.HKEY_CURRENT_USER, _full(subkey))


def _remove(reg: Any, root: str) -> str:
    state = _ownership(reg, root)
    if state != "ours":
        return state                    # absent, or somebody else's: untouched
    _delete_tree(reg, root)
    return "removed"


def _protocol_plan(command: str, icon: str) -> list[tuple[str, str, str]]:
    return [
        (PROTOCOL_KEY, "", "URL:ORION Protocol"),
        (PROTOCOL_KEY, "URL Protocol", ""),
        (f"{PROTOCOL_KEY}\\DefaultIcon", "", icon),
        (f"{PROTOCOL_KEY}\\shell\\open\\command", "", f'{command} "%1"'),
    ]


def _verb_plan(verb: ContextVerb, command: str, icon: str) -> list[tuple[str, str, str]]:
    return [
        (verb.key, "MUIVerb", verb.label),
        (verb.key, "Icon", icon),
        (f"{verb.key}\\command", "", f'{command} {verb.flag} "{verb.placeholder}"'),
    ]


def register_protocol(exe_cmd: str, *, reg: Any = None, icon: str | None = None) -> str:
    """orion:// -> *exe_cmd* "%1". Returns registered/unchanged/foreign/error."""
    try:
        if not exe_cmd:
            return "skipped"
        reg = _winreg(reg)
        return _apply(reg, PROTOCOL_KEY,
                      _protocol_plan(exe_cmd, icon or _default_icon(exe_cmd)))[0]
    except Exception as exc:
        return f"error: {_brief(exc)}"


def unregister_protocol(*, reg: Any = None) -> str:
    """Remove orion:// if ORION registered it. removed/absent/foreign/error."""
    try:
        return _remove(_winreg(reg), PROTOCOL_KEY)
    except Exception as exc:
        return f"error: {_brief(exc)}"


def _combine(states: list[str]) -> str:
    for state in states:
        if state.startswith("error"):
            return state
    if "foreign" in states:
        others = sorted({s for s in states if s != "foreign"})
        return "foreign" + (f" ({', '.join(others)} for the rest)" if others else "")
    for state in ("registered", "removed"):
        if state in states:
            return state
    return states[0] if len(set(states)) == 1 else "unchanged"


def register_context_menu(exe_cmd: str, *, reg: Any = None, icon: str | None = None) -> str:
    """The four "Ask ORION" / "Organise with ORION" Explorer entries."""
    try:
        if not exe_cmd:
            return "skipped"
        reg = _winreg(reg)
        icon = icon or _default_icon(exe_cmd)
        return _combine([_apply(reg, verb.key, _verb_plan(verb, exe_cmd, icon))[0]
                         for verb in CONTEXT_VERBS])
    except Exception as exc:
        return f"error: {_brief(exc)}"


def unregister_context_menu(*, reg: Any = None) -> str:
    try:
        reg = _winreg(reg)
        states = [_remove(reg, verb.key) for verb in CONTEXT_VERBS]
        if all(state == "absent" for state in states):
            return "absent"
        return _combine([s for s in states if s != "absent"])
    except Exception as exc:
        return f"error: {_brief(exc)}"


_AUTO: Any = object()


def sync_registration(settings: dict[str, Any] | None = None, *, reg: Any = None,
                      exe_cmd: Any = _AUTO, icon: str | None = None) -> dict[str, Any]:
    """Make the registry match windows_settings (protocol, explorer_menu).

    Returns {"command", "protocol", "explorer_menu"}. A developer checkout
    (no launch command) touches nothing at all: "skipped".
    """
    status: dict[str, Any] = {"command": None, "protocol": "skipped",
                              "explorer_menu": "skipped"}
    try:
        if settings is None:
            from . import windows_settings
            settings = windows_settings.snapshot()
        if exe_cmd is _AUTO:
            target = launch_target()
            exe_cmd = target.command if target else None
            if target is not None and icon is None and target.icon:
                icon = f"{target.icon},0"
        status["command"] = exe_cmd
        if not exe_cmd:
            return status
        status["protocol"] = (register_protocol(exe_cmd, reg=reg, icon=icon)
                              if settings.get("protocol", True)
                              else unregister_protocol(reg=reg))
        status["explorer_menu"] = (register_context_menu(exe_cmd, reg=reg, icon=icon)
                                   if settings.get("explorer_menu", True)
                                   else unregister_context_menu(reg=reg))
    except Exception as exc:
        status["error"] = _brief(exc)
    return status


def registration_state(*, reg: Any = None) -> dict[str, str]:
    """Read-only: who owns each key ORION would write (for diagnostics)."""
    try:
        reg = _winreg(reg)
        state = {PROTOCOL_KEY: _ownership(reg, PROTOCOL_KEY)}
        for verb in CONTEXT_VERBS:
            state[verb.key] = _ownership(reg, verb.key)
        return state
    except Exception as exc:
        return {"error": _brief(exc)}


# ── the taskbar jump list ────────────────────────────────────────────────────

@dataclass(frozen=True)
class JumpTask:
    title: str
    arguments: str
    description: str = ""


#: What this build puts in the jump list. The Screen Face is the only face, so
#: there is nothing to switch between; "Show Dock" is how the docked face is
#: asked for (it is not on screen at start).
DEFAULT_TASKS: tuple[JumpTask, ...] = (
    JumpTask("Show ORION", "--do show", "Bring ORION to the front."),
    JumpTask("Morning Briefing", "--do briefing", "Hear the briefing now."),
    JumpTask("Mute", "--do mute", "Silence ORION's voice."),
    JumpTask("Unmute", "--do unmute", "Let ORION speak again."),
    JumpTask("Show Dock", "--do dock:show", "Park ORION's face on the screen edge."),
)

#: The face-switching tasks, kept dormant: offered by default_tasks() only for a
#: form enabled by appearance.ENABLED_FORMS (or ORION_FACE_FORM in development).
DORMANT_FACE_TASKS: tuple[JumpTask, ...] = (
    JumpTask("Screen Face", "--do face:screen", "Switch to the screen face."),
    JumpTask("Classic Face", "--do face:classic", "Switch to the classic face."),
    JumpTask("Orb", "--do face:orb", "Switch to the orb."),
)


def default_tasks() -> tuple[JumpTask, ...]:
    """DEFAULT_TASKS, plus a face task per form beyond the Screen Face that
    this build offers (none in Mark XXXVII)."""
    offered = enabled_face_forms()
    if len(offered) < 2:
        return DEFAULT_TASKS
    extra = tuple(task for task in DORMANT_FACE_TASKS
                  if task.arguments.rpartition(":")[2] in offered)
    return DEFAULT_TASKS + extra

VT_LPWSTR = 31


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_text(cls, text: str) -> "_GUID":
        value = uuid.UUID(text)
        return cls(value.time_low, value.time_mid, value.time_hi_version,
                   (ctypes.c_ubyte * 8)(*value.bytes[8:]))

    def __str__(self) -> str:
        return "{%s}" % str(uuid.UUID(bytes=(
            self.Data1.to_bytes(4, "big") + self.Data2.to_bytes(2, "big")
            + self.Data3.to_bytes(2, "big") + bytes(self.Data4)))).upper()


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]


class PROPVARIANT(ctypes.Structure):
    """Only the VT_LPWSTR shape is used: 8-byte header + a 16-byte union
    on 64-bit (24 bytes), 8 + 8 on 32-bit (16 bytes), as in propidl.h."""

    _fields_ = [("vt", ctypes.c_ushort), ("wReserved1", ctypes.c_ushort),
                ("wReserved2", ctypes.c_ushort), ("wReserved3", ctypes.c_ushort),
                ("pwszVal", ctypes.c_wchar_p), ("_pad", ctypes.c_void_p)]


#: PKEY_Title (propkey.h) - the text a jump-list task shows.
PKEY_TITLE_FMTID = "{F29F85E0-4FF9-1068-AB91-08002B27B3D9}"
PKEY_TITLE_PID = 2


def pkey_title() -> PROPERTYKEY:
    return PROPERTYKEY(_GUID.from_text(PKEY_TITLE_FMTID), PKEY_TITLE_PID)


def propvariant_text(text: str) -> PROPVARIANT:
    value = PROPVARIANT()
    value.vt = VT_LPWSTR
    value.pwszVal = text
    return value


# The real IIDs and CLSIDs (shobjidl_core.h, propsys.h, objectarray.h).
IID_IObjectArray = "{92CA9DCD-5622-4BBA-A805-5E9F541BD8C9}"
IID_IObjectCollection = "{5632B1A4-E38A-400A-928A-D4CD63230295}"
IID_IShellLinkW = "{000214F9-0000-0000-C000-000000000046}"
IID_IPropertyStore = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
IID_ICustomDestinationList = "{6332DEBF-87B5-4670-90C0-5E57B408A49E}"
CLSID_DestinationList = "{77F10CF0-3DB5-4966-B520-B7C54FD35ED6}"
CLSID_EnumerableObjectCollection = "{2D3468C1-36A7-43B6-AC24-D3F02FD9607A}"
CLSID_ShellLink = "{00021401-0000-0000-C000-000000000046}"

_INTERFACES: dict[str, Any] = {}


def _interfaces() -> dict[str, Any]:
    """The five COM interfaces, declared with comtypes in vtable order."""
    if _INTERFACES:
        return _INTERFACES
    from ctypes import POINTER, c_int, c_uint, c_void_p, c_wchar_p
    from ctypes.wintypes import DWORD, HWND, WORD

    from comtypes import GUID, HRESULT, STDMETHOD, IUnknown

    class IObjectArray(IUnknown):
        _iid_ = GUID(IID_IObjectArray)
        _methods_ = [
            STDMETHOD(HRESULT, "GetCount", [POINTER(c_uint)]),
            STDMETHOD(HRESULT, "GetAt", [c_uint, POINTER(GUID), POINTER(c_void_p)]),
        ]

    class IObjectCollection(IObjectArray):
        _iid_ = GUID(IID_IObjectCollection)
        _methods_ = [
            STDMETHOD(HRESULT, "AddObject", [POINTER(IUnknown)]),
            STDMETHOD(HRESULT, "AddFromArray", [POINTER(IObjectArray)]),
            STDMETHOD(HRESULT, "RemoveObjectAt", [c_uint]),
            STDMETHOD(HRESULT, "Clear", []),
        ]

    class IShellLinkW(IUnknown):
        _iid_ = GUID(IID_IShellLinkW)
        _methods_ = [
            STDMETHOD(HRESULT, "GetPath", [c_wchar_p, c_int, c_void_p, DWORD]),
            STDMETHOD(HRESULT, "GetIDList", [POINTER(c_void_p)]),
            STDMETHOD(HRESULT, "SetIDList", [c_void_p]),
            STDMETHOD(HRESULT, "GetDescription", [c_wchar_p, c_int]),
            STDMETHOD(HRESULT, "SetDescription", [c_wchar_p]),
            STDMETHOD(HRESULT, "GetWorkingDirectory", [c_wchar_p, c_int]),
            STDMETHOD(HRESULT, "SetWorkingDirectory", [c_wchar_p]),
            STDMETHOD(HRESULT, "GetArguments", [c_wchar_p, c_int]),
            STDMETHOD(HRESULT, "SetArguments", [c_wchar_p]),
            STDMETHOD(HRESULT, "GetHotkey", [POINTER(WORD)]),
            STDMETHOD(HRESULT, "SetHotkey", [WORD]),
            STDMETHOD(HRESULT, "GetShowCmd", [POINTER(c_int)]),
            STDMETHOD(HRESULT, "SetShowCmd", [c_int]),
            STDMETHOD(HRESULT, "GetIconLocation", [c_wchar_p, c_int, POINTER(c_int)]),
            STDMETHOD(HRESULT, "SetIconLocation", [c_wchar_p, c_int]),
            STDMETHOD(HRESULT, "SetRelativePath", [c_wchar_p, DWORD]),
            STDMETHOD(HRESULT, "Resolve", [HWND, DWORD]),
            STDMETHOD(HRESULT, "SetPath", [c_wchar_p]),
        ]

    class IPropertyStore(IUnknown):
        _iid_ = GUID(IID_IPropertyStore)
        _methods_ = [
            STDMETHOD(HRESULT, "GetCount", [POINTER(DWORD)]),
            STDMETHOD(HRESULT, "GetAt", [DWORD, POINTER(PROPERTYKEY)]),
            STDMETHOD(HRESULT, "GetValue", [POINTER(PROPERTYKEY), POINTER(PROPVARIANT)]),
            STDMETHOD(HRESULT, "SetValue", [POINTER(PROPERTYKEY), POINTER(PROPVARIANT)]),
            STDMETHOD(HRESULT, "Commit", []),
        ]

    class ICustomDestinationList(IUnknown):
        _iid_ = GUID(IID_ICustomDestinationList)
        _methods_ = [
            STDMETHOD(HRESULT, "SetAppID", [c_wchar_p]),
            STDMETHOD(HRESULT, "BeginList",
                      [POINTER(c_uint), POINTER(GUID), POINTER(POINTER(IObjectArray))]),
            STDMETHOD(HRESULT, "AppendCategory", [c_wchar_p, POINTER(IObjectArray)]),
            STDMETHOD(HRESULT, "AppendKnownCategory", [c_int]),
            STDMETHOD(HRESULT, "AddUserTasks", [POINTER(IObjectArray)]),
            STDMETHOD(HRESULT, "CommitList", []),
            STDMETHOD(HRESULT, "GetRemovedDestinations", [POINTER(GUID), POINTER(c_void_p)]),
            STDMETHOD(HRESULT, "DeleteList", [c_wchar_p]),
            STDMETHOD(HRESULT, "AbortList", []),
        ]

    _INTERFACES.update(
        IObjectArray=IObjectArray, IObjectCollection=IObjectCollection,
        IShellLinkW=IShellLinkW, IPropertyStore=IPropertyStore,
        ICustomDestinationList=ICustomDestinationList, GUID=GUID)
    return _INTERFACES


class ComtypesFactory:
    """The real COM objects. Tests pass a fake with the same five methods."""

    _CREATE = {
        "destination_list": (CLSID_DestinationList, "ICustomDestinationList"),
        "object_collection": (CLSID_EnumerableObjectCollection, "IObjectCollection"),
        "shell_link": (CLSID_ShellLink, "IShellLinkW"),
    }
    _QUERY = {"property_store": "IPropertyStore", "object_array": "IObjectArray"}

    def __init__(self) -> None:
        import comtypes
        self._comtypes = comtypes
        self._ifaces = _interfaces()
        self._initialised = False

    def initialize(self) -> None:
        try:
            self._comtypes.CoInitializeEx(self._comtypes.COINIT_APARTMENTTHREADED)
            self._initialised = True
        except OSError:
            self._initialised = False       # already initialised another way

    def uninitialize(self) -> None:
        if self._initialised:
            self._initialised = False
            gc.collect()                    # release every pointer first
            self._comtypes.CoUninitialize()

    def create(self, kind: str) -> Any:
        clsid, iface = self._CREATE[kind]
        return self._comtypes.CoCreateInstance(
            self._ifaces["GUID"](clsid), interface=self._ifaces[iface],
            clsctx=self._comtypes.CLSCTX_INPROC_SERVER)

    def query(self, obj: Any, kind: str) -> Any:
        return obj.QueryInterface(self._ifaces[self._QUERY[kind]])

    def begin_list(self, destination: Any) -> int:
        array_type = self._ifaces["IObjectArray"]
        slots = ctypes.c_uint(0)
        removed = ctypes.POINTER(array_type)()
        destination.BeginList(ctypes.byref(slots), ctypes.byref(array_type._iid_),
                              ctypes.byref(removed))
        del removed
        return int(slots.value)


def _app_user_model_id() -> str:
    try:
        from .desktop_app import APP_USER_MODEL_ID
        return APP_USER_MODEL_ID
    except Exception:
        return "ProjectORION.Assistant"


def _coerce_tasks(tasks: Iterable[Any] | None) -> list[JumpTask]:
    out: list[JumpTask] = []
    for task in default_tasks() if tasks is None else tasks:
        if isinstance(task, JumpTask):
            out.append(task)
        elif isinstance(task, dict):
            out.append(JumpTask(str(task["title"]), str(task["arguments"]),
                                str(task.get("description", ""))))
        else:
            title, arguments, *rest = task
            out.append(JumpTask(str(title), str(arguments), str(rest[0]) if rest else ""))
    return out


def _with_com(factory: Any, work: Callable[[Any], Any]) -> tuple[Any, str]:
    """Run *work* inside COM initialisation; (result, "") or (None, reason).

    The except clause ends (dropping the traceback and every COM pointer it
    holds) before the finally uninitialises COM.
    """
    error = ""
    result: Any = None
    factory.initialize()
    try:
        try:
            result = work(factory)
        except Exception as exc:
            error = _brief(exc)
    finally:
        try:
            factory.uninitialize()
        except Exception:
            pass
    return result, error


def set_jump_list(exe: str, tasks: Iterable[Any] | None = None, *, com: Any = None,
                  app_id: str | None = None, icon: str | None = None) -> dict[str, Any]:
    """Replace ORION's taskbar tasks. *exe* is a launch command (see
    launch_command). Returns {"ok", "tasks", "reason"}; never raises."""
    status: dict[str, Any] = {"ok": False, "tasks": 0, "reason": ""}
    try:
        if not exe:
            status["reason"] = "no launch command"
            return status
        if com is None and os.name != "nt":
            status["reason"] = "not Windows"
            return status
        factory = com if com is not None else ComtypesFactory()
        path, base_args = _split_command(exe)
        icon_file, _, icon_index = (icon or f"{path},0").rpartition(",")
        if not icon_file or not icon_index.strip().lstrip("-").isdigit():
            icon_file, icon_index = (icon or path), "0"
        workdir = str(Path(path).parent)
        built = _coerce_tasks(tasks)
        app_id = app_id or _app_user_model_id()

        def work(factory: Any) -> int:
            destination = factory.create("destination_list")
            destination.SetAppID(app_id)
            factory.begin_list(destination)
            committed = False
            try:
                collection = factory.create("object_collection")
                for task in built:
                    link = factory.create("shell_link")
                    link.SetPath(path)
                    link.SetArguments(f"{base_args} {task.arguments}".strip())
                    link.SetWorkingDirectory(workdir)
                    link.SetIconLocation(icon_file, int(icon_index))
                    if task.description:
                        link.SetDescription(task.description)
                    store = factory.query(link, "property_store")
                    store.SetValue(pkey_title(), propvariant_text(task.title))
                    store.Commit()
                    collection.AddObject(link)
                destination.AddUserTasks(factory.query(collection, "object_array"))
                destination.CommitList()
                committed = True
            finally:
                if not committed:
                    try:
                        destination.AbortList()
                    except Exception:
                        pass
            return len(built)

        count, error = _with_com(factory, work)
        if error:
            status["reason"] = error
        else:
            status.update(ok=True, tasks=int(count or 0))
    except Exception as exc:
        status["reason"] = _brief(exc)
    return status


def clear_jump_list(*, com: Any = None, app_id: str | None = None) -> dict[str, Any]:
    """Take ORION's tasks back off the taskbar. Never raises."""
    status: dict[str, Any] = {"ok": False, "reason": ""}
    try:
        if com is None and os.name != "nt":
            status["reason"] = "not Windows"
            return status
        factory = com if com is not None else ComtypesFactory()
        target = app_id or _app_user_model_id()

        def work(factory: Any) -> bool:
            factory.create("destination_list").DeleteList(target)
            return True

        _result, error = _with_com(factory, work)
        status.update(ok=not error, reason=error)
    except Exception as exc:
        status["reason"] = _brief(exc)
    return status


# ── start-up, for app.py ─────────────────────────────────────────────────────

class WindowsShellHandle:
    """What start_windows_shell returns: stop() at shutdown."""

    def __init__(self, server: Any, startup_command: dict[str, str] | None) -> None:
        self.server = server
        self.startup_command = startup_command
        self.worker: threading.Thread | None = None
        self.status: dict[str, Any] = {"channel": "stopped", "registration": None,
                                       "jump_list": None}
        self._stopped = False

    @property
    def listening(self) -> bool:
        try:
            return bool(self.server is not None and self.server.is_listening())
        except Exception:
            return False

    def join(self, timeout: float | None = None) -> bool:
        worker = self.worker
        if worker is None:
            return True
        worker.join(timeout)
        return not worker.is_alive()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if self.server is not None:
                self.server.stop()
        except Exception:
            pass
        self.status["channel"] = "stopped"
        global _CURRENT
        if _CURRENT is self:
            _CURRENT = None


_CURRENT: WindowsShellHandle | None = None


def _logger(bus: Any) -> Callable[[str], None]:
    def log(line: str) -> None:
        try:
            bus.log.emit(line)
        except Exception:
            pass
    return log


def _describe_registration(status: dict[str, Any]) -> list[str]:
    words = {"registered": "registered", "unchanged": "already registered",
             "removed": "removed (turned off)", "absent": "off",
             "skipped": "skipped"}
    lines = []
    for key, label in (("protocol", "orion:// links"), ("explorer_menu", "Explorer menu")):
        state = str(status.get(key) or "skipped")
        if state.startswith("foreign"):
            text = "left alone - another program owns that key"
        else:
            text = words.get(state, state)
        lines.append(f"WINDOWS: {label} {text}.")
    if status.get("error"):
        lines.append(f"WINDOWS: registration failed - {status['error']}.")
    return lines


def start_windows_shell(bus: Any, on_command: Callable[[dict[str, str]], Any], *,
                        argv: Iterable[Any] | None = None,
                        settings: dict[str, Any] | None = None,
                        reg: Any = None, com: Any = None, exe_cmd: Any = _AUTO,
                        server_name: str | None = None,
                        dispatch_startup: bool = True) -> WindowsShellHandle:
    """Open ORION's shell entry points. Call on the GUI thread.

    * Starts the command channel; every valid command reaches
      ``on_command(dict)`` on the GUI thread (exceptions are logged, never
      raised into Qt).
    * The command THIS launch carried (Explorer/jump list/link when no ORION
      was running) is delivered the same way on the next event-loop turn.
    * When this is the installed ORION.exe, registration and the jump list
      are brought in line with windows_settings on a worker thread.
    """
    log = _logger(bus)

    def deliver(command: dict[str, str]) -> None:
        try:
            on_command(dict(command))
        except Exception as exc:
            log(f"WINDOWS: {command.get('action', 'a command')} could not be "
                f"carried out - {_brief(exc)}")

    startup, _why = check_command_argv(sys.argv if argv is None else argv)
    server = None
    try:
        server = _command_server_class()(server_name)
    except Exception as exc:
        log(f"WINDOWS: command channel unavailable - {_brief(exc)}.")
    handle = WindowsShellHandle(server, startup)
    global _CURRENT
    _CURRENT = handle

    if server is not None:
        server.command.connect(deliver)
        server.rejected.connect(lambda why: log(f"WINDOWS: refused a shell command ({why})."))
        if server.start():
            handle.status["channel"] = "listening"
            log("WINDOWS: command channel listening - Explorer, the jump list and "
                "orion:// links reach this ORION.")
        else:
            handle.status["channel"] = f"unavailable: {server.error()}"
            log(f"WINDOWS: command channel unavailable - {server.error()}.")

    if startup is not None and dispatch_startup:
        try:
            QtCore, _ = _qt()
            QtCore.QTimer.singleShot(0, lambda: deliver(startup))
        except Exception:
            deliver(startup)

    try:
        if settings is None:
            from . import windows_settings
            settings = windows_settings.snapshot()
    except Exception:
        settings = {}
    icon = None
    if exe_cmd is _AUTO:
        target = launch_target()
        exe_cmd = target.command if target else None
        icon = f"{target.icon},0" if target and target.icon else None
    if not exe_cmd:
        handle.status["registration"] = {"protocol": "skipped", "explorer_menu": "skipped"}
        handle.status["jump_list"] = "skipped"
        log("WINDOWS: shell registration skipped - this is not the installed "
            "ORION.exe (ORION_SHELL_REGISTER=1 overrides).")
        return handle

    snapshot = dict(settings)

    def register() -> None:
        status = sync_registration(snapshot, reg=reg, exe_cmd=exe_cmd, icon=icon)
        handle.status["registration"] = status
        for line in _describe_registration(status):
            log(line)
        if snapshot.get("jump_list", True):
            jump = set_jump_list(exe_cmd, com=com, icon=icon)
            handle.status["jump_list"] = jump
            log(f"WINDOWS: jump list set ({jump['tasks']} tasks)." if jump["ok"]
                else f"WINDOWS: jump list not set - {jump['reason']}.")
        else:
            jump = clear_jump_list(com=com)
            handle.status["jump_list"] = dict(jump, cleared=True)
            log("WINDOWS: jump list off." if jump["ok"]
                else f"WINDOWS: jump list could not be cleared - {jump['reason']}.")

    worker = threading.Thread(target=register, name="orion-windows-shell", daemon=True)
    handle.worker = worker
    worker.start()
    return handle


def status() -> dict[str, Any]:
    """In-memory status of the running shell integration (no registry reads)."""
    handle = _CURRENT
    target = None
    try:
        target = launch_command()
    except Exception:
        pass
    return {
        "pipe": pipe_name(),
        "listening": bool(handle and handle.listening),
        "launch_command": target,
        "frozen": bool(getattr(sys, "frozen", False)),
        "registration": dict(handle.status.get("registration") or {}) if handle else {},
        "jump_list": (handle.status.get("jump_list") if handle else None),
    }


def describe_status() -> str:
    """One paragraph for diagnostics."""
    info = status()
    parts = [f"Command channel {'listening' if info['listening'] else 'not listening'} "
             f"({info['pipe']})."]
    if not info["launch_command"]:
        parts.append("Shell registration off: not the installed ORION.exe.")
    reg_status = info["registration"]
    if reg_status:
        parts.append(f"orion:// links: {reg_status.get('protocol', 'unknown')}; "
                     f"Explorer menu: {reg_status.get('explorer_menu', 'unknown')}.")
    jump = info["jump_list"]
    if isinstance(jump, dict):
        if jump.get("cleared"):
            parts.append("Jump list: off.")
        elif jump.get("ok"):
            parts.append(f"Jump list: {jump.get('tasks', 0)} tasks.")
        else:
            parts.append(f"Jump list: not set ({jump.get('reason') or 'unknown'}).")
    elif jump:
        parts.append(f"Jump list: {jump}.")
    return " ".join(parts)


__all__ = [
    "ALL_ACTIONS", "CLI_FLAGS", "CONTEXT_VERBS", "DEFAULT_TASKS", "DO_VERBS",
    "DOCK_STATES", "DORMANT_FACE_TASKS", "FACE_FORMS", "default_tasks",
    "enabled_face_forms", "LINK_ACTIONS", "MARKER_NAME", "MARKER_VALUE", "MAX_PAYLOAD_BYTES",
    "CommandServer", "ComtypesFactory", "ContextVerb", "JumpTask", "LaunchTarget",
    "WindowsShellHandle", "argv_may_carry_command", "check_argv", "check_command",
    "check_command_argv", "check_link", "clear_jump_list", "command_from_argv",
    "decode_command", "describe_status", "encode_command", "forward", "forward_launch",
    "launch_command", "launch_target", "parse_argv", "parse_link", "pipe_name",
    "register_context_menu", "register_protocol", "registration_state", "set_jump_list",
    "start_windows_shell", "status", "strip_shell_args", "sync_registration",
    "unregister_context_menu", "unregister_protocol", "validate_command",
    "validate_path",
]
