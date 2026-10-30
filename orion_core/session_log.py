"""ORION's own log, kept on disk.

Every NET:, VOICE:, PERF:, FACE: and TOOL: line went to the window and nowhere
else. On 8 Oct the Live voice froze for ten minutes after the camera started
and the face flashed black, and afterwards there was nothing to read: no line
said what the channel or the face had done. This writes the same lines to
``config/diagnostics/session.log`` (rotated at MAX_BYTES, KEEP files).

Each line is stamped with the full local date and time::

    2026-10-09 23:41:07  NET: live channel closed; re-establishing.

The date used to appear only in the "===== session ... =====" header, so
"23:41" was ambiguous across midnight, sessions and the rotated files. Older
files in the time-only form are still read back (self_diagnosis.py tracks the
header dates); a multi-line message keeps one stamp and its other lines are
indented, so a reader never mistakes them for new entries.

Secrets and e-mail addresses are redacted before anything reaches the disk
(:func:`redact`). The log is read back to a model and quoted to the owner, so
an API key echoed in an error message must never be in it.

Writes happen on one background thread: the log signal fires on the GUI
thread, which is also the asyncio loop, and must never wait on a disk. The
caller only reads the clock; formatting and redaction run on the writer.

Off unless :func:`enable` was called (orion.py does, for a real run), so a
window built by the tests writes nothing.

Private log (Mark XXXX.3): with it on, the file keeps what ORION DID - tools,
timings, connections, errors - and never what was SAID. The owner's words,
ORION's replies, the transcript in a trace line, the phrase an emotion was
read from and a specialist's answer are replaced with PRIVATE_MARK before the
line reaches the disk (:func:`private_line`). The window still shows them.
Saved in config/privacy.json; Preferences > Privacy or "private log on".
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any

MAX_BYTES = 2_000_000
KEEP = 3
#: strftime form of every line's stamp (local time).
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
#: What a redacted secret / e-mail address is replaced with.
SECRET_MARK = "<redacted>"
EMAIL_MARK = "<email>"

_enabled = False
_private = False
#: What a private line's words are replaced with.
PRIVATE_MARK = "[private]"
_path: Path | None = None
_queue: "SimpleQueue[tuple[float, str]]" = SimpleQueue()
_thread: threading.Thread | None = None
_attached: set[int] = set()
_lock = threading.Lock()


# ── redaction ────────────────────────────────────────────────────────────────
#
# Recognition, not guessing: a secret is redacted when its SHAPE is a known
# credential (a provider prefix, a JWT, a Bearer value) or it sits in a
# credential CONTEXT (token=, "password": ...). A bare long hex string is left
# alone - hashes in file names, commit ids and request ids are ordinary log
# text, and mangling them would make the log useless for the thing it is for.

_TOKEN_SHAPES = re.compile(
    r"\bAIza[0-9A-Za-z_\-]{30,}"                     # Google API key
    r"|\bAQ\.[0-9A-Za-z_\-.]{20,}"                   # Gemini / Google "AQ." access token
    r"|\bya29\.[0-9A-Za-z_\-.]{20,}"                 # Google OAuth access token
    r"|\b1//0[0-9A-Za-z_\-]{20,}"                    # Google OAuth refresh token
    r"|\bsk-(?:ant-|proj-)?[0-9A-Za-z_\-]{16,}"      # Anthropic / OpenAI
    r"|\bsk_[0-9A-Za-z]{20,}"                        # ElevenLabs-style
    r"|\b[rp]k_(?:live|test)_[0-9A-Za-z]{16,}"       # Stripe
    r"|\bgsk_[0-9A-Za-z]{20,}"                       # Groq
    r"|\bhf_[0-9A-Za-z]{20,}"                        # Hugging Face
    r"|\bxai-[0-9A-Za-z]{20,}"                       # xAI
    r"|\bgh[pousr]_[0-9A-Za-z]{20,}"                 # GitHub
    r"|\bgithub_pat_[0-9A-Za-z_]{20,}"               # GitHub fine-grained
    r"|\bxox[abposr]-[0-9A-Za-z\-]{10,}"             # Slack
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"                # AWS access key id
    r"|\bauth_tokens/[0-9A-Za-z_\-]{8,}"             # Gemini Live ephemeral token
    r"|\beyJ[0-9A-Za-z_\-]{8,}\.[0-9A-Za-z_\-]{8,}\.[0-9A-Za-z_\-]{4,}"  # JWT
)
# Authorization values can use token credentials, parameter lists (Digest),
# or an entire quoted value in a JSON/repr log. Consume the complete value;
# matching only the first word would leave Negotiate/NTLM credentials behind.
_AUTH_QUOTED = r'''(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*')'''
_AUTH_TOKEN = r"[!#$%&'*+.^_`|~0-9A-Za-z-]+"
_AUTH_OPAQUE = r'''[^\s"',;:{}<>\[\]]+'''
_AUTH_PARAM = rf"{_AUTH_TOKEN}[ \t]*=[ \t]*(?:{_AUTH_QUOTED}|{_AUTH_OPAQUE})"
_AUTH_HEADER = re.compile(
    r"(?i)(\b(?:proxy-)?authorization[\"']?[ \t]*[:=][ \t]*)"
    rf"({_AUTH_QUOTED}|(?:{_AUTH_TOKEN}[ \t]+)?"
    rf"(?:{_AUTH_PARAM}(?:[ \t]*,[ \t]*{_AUTH_PARAM})*|{_AUTH_OPAQUE}))")
# A bare "Bearer <token>": the token must look like one (a digit or a
# separator in it), so prose such as "bearer instruments" is left alone.
_BEARER = re.compile(
    r"(?i)(\bbearer\s+)(?=[0-9A-Za-z._~+/=\-]*[0-9._~+/=\-])([0-9A-Za-z._~+/=\-]{12,})")
# key=value, key: value and "key": "value" for credential-named fields. The
# name must stand alone ("max_tokens=", "token_usage:" and "tokens:" do not
# match), so ordinary counters and tool names are untouched.
_SECRET_FIELD = re.compile(
    r"(?i)(?<![0-9A-Za-z_\-])"
    r"([\"']?(?:x-goog-api-key|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|"
    r"id[_-]?token|auth[_-]?token|session[_-]?token|client[_-]?secret|private[_-]?key|"
    r"secret[_-]?key|secret|token|password|passwd|passphrase)[\"']?\s*[:=]\s*)"
    r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s\"'&,;}<>\])]+)")
# ?key=... in a URL, and a JSON "key": "..." field (a bare "key:" is a
# keyboard key far more often than a credential).
_QUERY_KEY = re.compile(r"(?i)([?&](?:key|sig|signature|code)=)([^&\s\"'#]{6,})")
_JSON_KEY = re.compile(r"(?i)([\"']key[\"']\s*:\s*[\"'])([^\"']{8,})")
_EMAIL = re.compile(
    r"(?<![\w.+\-])[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9\-]+\.)+([A-Za-z]{2,24})\b")
# "face@2x.png" and "module@v2.py" are file names, not addresses.
_FILE_SUFFIXES = frozenset(
    "py pyw pyc js mjs ts json txt log md html htm css xml yaml yml toml ini cfg "
    "db exe dll png jpg jpeg gif svg webp ico wav mp3 mp4 ogg onnx bin pt pth zip".split())


def _email(match: re.Match) -> str:
    return match.group(0) if match.group(1).lower() in _FILE_SUFFIXES else EMAIL_MARK


def _secret_field(match: re.Match) -> str:
    # W1.5 continuation, 11 Oct: consume the whole quoted credential rather
    # than just its first word, and redact short explicit values too.
    value = match.group(2)
    quote = value[0] if value[0] in "\"'" else ""
    return match.group(1) + quote + SECRET_MARK + quote


def redact(text: str) -> str:
    """*text* with credentials and e-mail addresses replaced. Never raises."""
    try:
        text = str(text)
        text = _AUTH_HEADER.sub(_secret_field, text)
        text = _TOKEN_SHAPES.sub(SECRET_MARK, text)
        text = _BEARER.sub(lambda m: m.group(1) + SECRET_MARK, text)
        text = _SECRET_FIELD.sub(_secret_field, text)
        text = _QUERY_KEY.sub(lambda m: m.group(1) + SECRET_MARK, text)
        text = _JSON_KEY.sub(lambda m: m.group(1) + SECRET_MARK, text)
        return _EMAIL.sub(_email, text)
    except Exception:
        return "<unloggable line>"


# ── private log ──────────────────────────────────────────────────────────────

#: Lines that ARE words: the owner's, ORION's, a remote sender's.
_SPOKEN = re.compile(r"^((?:YOU|ORION|USER|PHONE)(?:\[[^\]]*\])?(?: \([^)]*\))?:\s*).+", re.S)
#: "VOICE: not answering '<what was heard>' - not addressed to him".
_HEARD = re.compile(r"^(VOICE: not answering )'[^']*'(.*)$", re.S)
#: Trace stages whose detail is the owner's words.
_TRACE_WORDS = re.compile(r"^(TRACE: \S+ (?:TRANSCRIPTION_COMPLETE|API_REQUEST_START))(?: — .*)?$", re.S)
#: Lines that end in words after a dash: what an emotion was read from, a
#: specialist's answer, a remembered research topic.
_TAIL_WORDS = re.compile(r"^((?:EMOTION|SPECIALIST|AGENT|BRIEF|REFLEX): [^—]*?)\s+—\s+.+$", re.S)
_RESEARCH = re.compile(r"^(COG: .*?Research sessions:).+$", re.S)


def private_line(text: str) -> str:
    """*text* with what was said replaced by PRIVATE_MARK; what was done kept."""
    match = _SPOKEN.match(text)
    if match:
        return match.group(1) + PRIVATE_MARK
    match = _TRACE_WORDS.match(text)
    if match:
        return match.group(1)
    match = _HEARD.match(text)
    if match:
        return f"{match.group(1)}{PRIVATE_MARK}{match.group(2)}"
    for pattern in (_TAIL_WORDS, _RESEARCH):
        match = pattern.match(text)
        if match:
            return f"{match.group(1)} {PRIVATE_MARK}"
    return text


def _privacy_path() -> Path:
    from .constants import CONFIG_DIR
    return Path(CONFIG_DIR) / "privacy.json"


def private() -> bool:
    """Is the private log on?"""
    return _private


def load_private() -> bool:
    """Read the saved switch (config/privacy.json)."""
    global _private
    try:
        import json
        data = json.loads(_privacy_path().read_text(encoding="utf-8"))
        _private = bool(data.get("private_log")) if isinstance(data, dict) else False
    except (OSError, ValueError):
        _private = False
    return _private


def set_private(on: bool) -> bool:
    """Switch the private log on or off and save it. Returns the new state."""
    global _private
    _private = bool(on)
    try:
        import json
        from .atomic_io import atomic_write_text
        path = _privacy_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data = data if isinstance(data, dict) else {}
        data["private_log"] = _private
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    write(f"SYS: private log {'on - what is said is no longer written here' if _private else 'off'}.")
    return _private


def format_line(at: float, line: str) -> str:
    """One on-disk line: full local date and time, two spaces, the redacted
    text. Continuation lines of a multi-line message are indented."""
    stamp = time.strftime(STAMP_FORMAT, time.localtime(at))
    text = redact(str(line).rstrip())
    if _private:
        text = private_line(text)
    if "\n" in text or "\r" in text:
        text = "\n    ".join(part.rstrip() for part in text.replace("\r\n", "\n")
                             .replace("\r", "\n").split("\n"))
    return f"{stamp}  {text}"


# ── the file ─────────────────────────────────────────────────────────────────

def enable(path: Path | None = None) -> Path:
    """Switch the file log on (once). Returns where it writes."""
    global _enabled, _path
    if path is None:
        from .constants import CONFIG_DIR
        path = Path(CONFIG_DIR) / "diagnostics" / "session.log"
    _path = Path(path)
    _enabled = True
    load_private()
    import atexit
    atexit.register(flush)          # the last lines before exit are the useful ones
    write(f"===== session {time.strftime('%Y-%m-%d %H:%M:%S')} (pid {os.getpid()}) =====")
    return _path


def attach(bus: Any) -> bool:
    """Copy *bus*'s log lines to the file. No-op unless enabled."""
    if not _enabled or bus is None or id(bus) in _attached:
        return False
    try:
        bus.log.connect(write)
    except Exception:
        return False
    _attached.add(id(bus))
    return True


def write(line: str) -> None:
    """Queue one line. Cheap and thread-safe; never raises.

    Only the clock is read here - this runs on the GUI/asyncio thread. The
    stamp is formatted and the line redacted on the writer thread."""
    if not _enabled:
        return
    try:
        _queue.put((time.time(), str(line)))
    except Exception:
        return
    _ensure_writer()


def _ensure_writer() -> None:
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_run, name="orion-session-log", daemon=True)
        _thread.start()


def _rotate(path: Path) -> None:
    for index in range(KEEP - 1, 0, -1):
        older = path.with_name(f"{path.name}.{index}")
        if older.exists():
            os.replace(older, path.with_name(f"{path.name}.{index + 1}"))
    os.replace(path, path.with_name(f"{path.name}.1"))


def _render(item: tuple[float, str]) -> str:
    try:
        return format_line(*item)
    except Exception:
        return f"{time.strftime(STAMP_FORMAT)}  <unloggable line>"


def _run() -> None:
    while True:
        try:
            first = _queue.get(timeout=30.0)
        except Empty:
            return                          # idle: the next write restarts it
        batch = [first]
        while len(batch) < 500:
            try:
                batch.append(_queue.get_nowait())
            except Empty:
                break
        path = _path
        if path is None:
            continue
        lines = [_render(item) for item in batch]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > MAX_BYTES:
                _rotate(path)
            with open(path, "a", encoding="utf-8", errors="replace") as handle:
                handle.write("\n".join(lines) + "\n")
        except OSError:
            time.sleep(1.0)                 # a locked file: drop this batch, keep going


def flush(timeout: float = 2.0) -> None:
    """Wait (bounded) for queued lines to reach the disk - for shutdown."""
    deadline = time.monotonic() + timeout
    while not _queue.empty() and time.monotonic() < deadline:
        time.sleep(0.05)


__all__ = ["enable", "attach", "write", "flush", "redact", "format_line",
           "private", "private_line", "set_private", "load_private",
           "MAX_BYTES", "KEEP", "PRIVATE_MARK", "STAMP_FORMAT"]
