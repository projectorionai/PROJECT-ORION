"""
ORION's shell entry points: orion:// links, Explorer's "Ask ORION", the
taskbar jump list, and the channel that carries a click to the running ORION.

Nothing here touches the real registry, a real jump list, the real config or
a real ORION: the registry is a fake winreg, COM is a fake factory, settings
live in tmp_path, and every pipe has a name of its own test.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop
from PyQt6.QtWidgets import QApplication  # noqa: F401  (the session owns the app)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import single_instance as si  # noqa: E402
from orion_core import windows_settings  # noqa: E402
from orion_core import windows_shell as ws  # noqa: E402

TOKEN = "Ab3_-" * 4                     # 20 url-safe characters


@pytest.fixture(autouse=True)
def _private_names(monkeypatch, tmp_path):
    """A mutex (and so a pipe) name of this test's own, settings in tmp_path,
    and no developer opt-in leaking in from the environment."""
    monkeypatch.setattr(si, "MUTEX_NAME", si.MUTEX_NAME + "-shell-test-" + uuid.uuid4().hex[:10])
    monkeypatch.setattr(windows_settings, "SETTINGS_PATH", tmp_path / "windows_integration.json")
    monkeypatch.delenv("ORION_SHELL_REGISTER", raising=False)
    for key in windows_settings.DEFAULTS:
        monkeypatch.delenv(f"ORION_WIN_{key.upper()}", raising=False)
    monkeypatch.setattr(ws, "allow_foreground_handoff", lambda: True)
    yield
    if ws._CURRENT is not None:
        ws._CURRENT.stop()


@pytest.fixture
def qapp():
    app = QCoreApplication.instance() or QApplication([])
    return app


def _unique_pipe() -> str:
    return "orion-test-" + uuid.uuid4().hex[:12]


def _pump_until(predicate, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _in_thread(fn, timeout: float = 8.0):
    """Run a blocking client in a thread while this (GUI) thread serves."""
    box: dict = {}
    thread = threading.Thread(target=lambda: box.setdefault("value", fn()), daemon=True)
    thread.start()
    assert _pump_until(lambda: not thread.is_alive(), timeout), "the client never finished"
    return box.get("value")


def _raw_send(name: str, data: bytes, timeout: float = 3.0) -> bytes:
    """A deliberately unhelpful client: sends *data* as is, returns the reply."""
    from PyQt6.QtNetwork import QLocalSocket

    sock = QLocalSocket()
    sock.connectToServer(name)
    if not sock.waitForConnected(int(timeout * 1000)):
        return b"<no connection>"
    if data:
        sock.write(data)
        sock.waitForBytesWritten(int(timeout * 1000))
    reply = bytearray()
    deadline = time.monotonic() + timeout
    while b"\n" not in reply and time.monotonic() < deadline:
        if sock.bytesAvailable() <= 0 and not sock.waitForReadyRead(100):
            if sock.state() == QLocalSocket.LocalSocketState.UnconnectedState:
                break
            continue
        reply.extend(bytes(sock.readAll().data()))
    reply.extend(bytes(sock.readAll().data()))
    sock.abort()
    return bytes(reply)


# ── orion:// links: the allowlist ────────────────────────────────────────────

@pytest.mark.parametrize("url, expected", [
    ("orion://show", {"action": "show"}),
    ("orion://show/", {"action": "show"}),
    ("ORION://SHOW", {"action": "show"}),
    ("orion://briefing", {"action": "briefing"}),
    ("orion://mute", {"action": "mute"}),
    ("orion://unmute/", {"action": "unmute"}),
    ("orion://page/tasks", {"action": "page", "page": "tasks"}),
    ("orion://page/CHESS/", {"action": "page", "page": "chess"}),
    ("orion://face/screen", {"action": "face", "form": "screen"}),
    ("orion://face/Screen", {"action": "face", "form": "screen"}),
    ("orion://ask?q=hello%20world", {"action": "prefill", "text": "hello world"}),
    ("orion://ask/?q=what+time+is+it", {"action": "prefill", "text": "what time is it"}),
    (f"orion://toast?t={TOKEN}&a=Snooze", {"action": "toast", "token": TOKEN, "verb": "snooze"}),
    (f"orion://toast?a=open&t={TOKEN}", {"action": "toast", "token": TOKEN, "verb": "open"}),
])
def test_allowlisted_links_parse(url, expected):
    assert ws.parse_link(url) == dict(expected, origin="link")


HOSTILE_LINKS = [
    "javascript:alert(document.cookie)",
    "file:///C:/Windows/System32/cmd.exe",
    "http://evil.example/orion://show",
    "orion:show",
    "orion:///show",
    "orion://",
    "orion://show@evil.example",
    "orion://show:8080",
    "orion://sh%6Fw",
    "orion://show?x=1",
    "orion://show/extra",
    "orion://show#fragment",
    "orion://mute?q=1",
    "orion://unknown",
    "orion://ask_file?path=C:/Windows/win.ini",
    "orion://summarise?path=C:/secret.txt",
    "orion://organise/C:/Users",
    "orion://run?cmd=calc.exe",
    "orion://page/../../Windows",
    "orion://page/%2e%2e",
    "orion://page/a%2Fb",
    "orion://page/tasks/extra",
    "orion://page/tasks1",
    "orion://page/",
    "orion://page/" + "a" * 33,
    "orion://page/tasks?evil=1",
    "orion://face/evil",
    "orion://face/",
    "orion://face/orb?x=1",
    "orion://ask",
    "orion://ask?q=",
    "orion://ask?q=%00%01",
    "orion://ask?q=a&q=b",
    "orion://ask?q=a&send=1",
    "orion://ask/extra?q=a",
    "orion://ask?q=" + "a" * 501,
    "orion://ask?q=" + "%E6%97%A5" * 501,
    "orion://toast",
    "orion://toast?t=short&a=open",
    f"orion://toast?t={'A' * 65}&a=open",
    f"orion://toast?t={TOKEN[:-5]}.....&a=open",
    f"orion://toast?t={TOKEN}",
    f"orion://toast?t={TOKEN}&a=",
    f"orion://toast?t={TOKEN}&a=rm%20-rf",
    f"orion://toast?t={TOKEN}&a=open1",
    f"orion://toast?t={TOKEN}&a=open&x=1",
    f"orion://toast?t={TOKEN}&t={TOKEN}&a=open",
    "orion://show\x00",
    "orion://show\n--ask-file",
    "orion://" + "a" * 10_000,
    "",
    "   ",
]


@pytest.mark.parametrize("url", HOSTILE_LINKS, ids=[f"hostile{i}" for i in range(len(HOSTILE_LINKS))])
def test_hostile_links_are_refused(url):
    assert ws.parse_link(url) is None


@pytest.mark.parametrize("value", [None, 42, b"orion://show", ["orion://show"], {"a": 1}])
def test_a_link_that_is_not_text_is_refused(value):
    assert ws.parse_link(value) is None


@pytest.mark.parametrize("url", HOSTILE_LINKS, ids=[f"hostile{i}" for i in range(len(HOSTILE_LINKS))])
def test_refusal_reasons_are_safe_to_log(url):
    """The reason is a fixed phrase; the link is a stranger's text."""
    command, reason = ws.check_link(url)
    assert command is None
    assert reason and len(reason) <= 60
    assert "\n" not in reason and "\x00" not in reason
    for fragment in ("cookie", "evil", "calc", "Windows", "secret", "aaaa", "--ask"):
        assert fragment not in reason


def test_prefill_text_is_one_visible_line():
    """Hidden and direction-changing characters would let a page put words
    in the input box that the owner cannot see before pressing Enter."""
    command = ws.parse_link("orion://ask?q=hi%0Athere%E2%80%AE%E2%80%8Bfriend%09!")
    assert command == {"action": "prefill", "text": "hi therefriend !", "origin": "link"}


def test_prefill_is_never_marked_to_send():
    command = ws.parse_link("orion://ask?q=delete%20everything")
    assert command["action"] == "prefill"
    assert set(command) == {"action", "text", "origin"}


def test_five_hundred_characters_is_the_limit():
    assert ws.parse_link("orion://ask?q=" + "a" * 500)["text"] == "a" * 500
    assert ws.parse_link("orion://ask?q=" + "%E6%97%A5" * 500)["text"] == "\u65e5" * 500
    assert ws.parse_link("orion://ask?q=" + "a" * 501) is None


# ── CLI flags ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("verb, expected", [
    ("show", {"action": "show"}),
    ("briefing", {"action": "briefing"}),
    ("mute", {"action": "mute"}),
    ("unmute", {"action": "unmute"}),
    ("face:screen", {"action": "face", "form": "screen"}),
    ("dock:show", {"action": "dock", "state": "show"}),
    ("dock:hide", {"action": "dock", "state": "hide"}),
    ("last-task", {"action": "last_task"}),
    ("BRIEFING", {"action": "briefing"}),
])
def test_do_verbs(verb, expected):
    assert ws.parse_argv(["ORION.exe", "--do", verb]) == dict(expected, origin="cli")
    assert ws.parse_argv(["ORION.exe", f"--do={verb}"]) == dict(expected, origin="cli")


@pytest.mark.parametrize("argv", [
    ["ORION.exe"],
    ["ORION.exe", "--offline"],
    ["ORION.exe", "--do"],
    ["ORION.exe", "--do", "format-c"],
    ["ORION.exe", "--do", "face:evil"],
    ["ORION.exe", "--do", "--ask-file"],
    ["ORION.exe", "--do", "mute", "--do", "unmute"],
])
def test_bad_or_missing_cli_commands(argv):
    assert ws.parse_argv(argv) is None


def test_file_commands_carry_a_resolved_absolute_path(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    folder = tmp_path / "Downloads"
    folder.mkdir()
    real_note, real_folder = str(note.resolve()), str(folder.resolve())

    assert ws.parse_argv(["ORION.exe", "--ask-file", str(note)]) == {
        "action": "ask_file", "path": real_note, "origin": "cli"}
    assert ws.parse_argv(["ORION.exe", "--ask-file", str(folder)])["path"] == real_folder
    assert ws.parse_argv(["ORION.exe", "--summarise", str(note)]) == {
        "action": "summarise", "path": real_note, "origin": "cli"}
    assert ws.parse_argv(["ORION.exe", "--summarize", str(note)])["action"] == "summarise"
    assert ws.parse_argv(["ORION.exe", "--organise", str(folder)]) == {
        "action": "organise", "path": real_folder, "origin": "cli"}
    assert ws.parse_argv(["ORION.exe", f"--organize={folder}"])["path"] == real_folder
    assert ws.parse_argv(["ORION.exe", "--ask-file", f'"{note}"'])["path"] == real_note


def test_organise_wants_a_folder_and_nothing_wants_a_missing_path(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    assert ws.parse_argv(["ORION.exe", "--organise", str(note)]) is None
    assert ws.parse_argv(["ORION.exe", "--ask-file", str(tmp_path / "missing.txt")]) is None
    assert ws.parse_argv(["ORION.exe", "--ask-file", str(note), "--do", "mute"]) is None


def test_a_relative_path_resolves_against_the_launching_directory(tmp_path, monkeypatch):
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    command = ws.parse_argv(["ORION.exe", "--ask-file", "note.txt"])
    assert command["path"] == str((tmp_path / "note.txt").resolve())
    assert Path(command["path"]).is_absolute()


@pytest.mark.skipif(os.name != "nt", reason="Windows path forms")
@pytest.mark.parametrize("raw", [
    "\\\\?\\C:\\Windows\\win.ini",
    "\\\\.\\PIPE\\orion",
    "\\\\.\\C:",
    "//?/C:/Windows/win.ini",
    "//./PhysicalDrive0",
    "\\??\\C:\\Windows",
    "C:Windows\\win.ini",
    "C:",
    "C:\\a\x00b",
    "C:\\a\nb",
    "C:\\" + "a" * 2000,
    "",
    '""',
])
def test_device_and_malformed_paths_are_refused(raw):
    path, reason = ws.validate_path(raw)
    assert path is None and reason


@pytest.mark.skipif(os.name != "nt", reason="NTFS streams")
def test_alternate_data_streams_are_refused(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    assert ws.validate_path(f"{note}:hidden")[0] is None


def test_a_link_launch_ignores_every_other_argument(tmp_path):
    """Should a browser pass a crafted URL's quotes through, the extra
    arguments must not become a file command."""
    secret = tmp_path / "secret.txt"
    secret.write_text("x", encoding="utf-8")
    argv = ["ORION.exe", "orion://ask?q=hi", "--ask-file", str(secret)]
    assert ws.command_from_argv(argv) == {"action": "prefill", "text": "hi", "origin": "link"}
    assert ws.command_from_argv(["ORION.exe", "orion://evil", "--do", "mute"]) is None
    assert ws.command_from_argv(["ORION.exe", "orion://show", "orion://mute"]) is None
    assert ws.command_from_argv(["ORION.exe", "--do", "mute"]) == {"action": "mute", "origin": "cli"}


def test_the_launcher_confines_a_link_launch_before_any_mode_runs():
    source = (ROOT / "orion.py").read_text(encoding="utf-8")
    start = source.index("def _confine_link_launch(")
    end = source.index('\nif __name__ == "__main__":\n    sys.argv[:] = _confine_link_launch')
    namespace: dict = {}
    exec(compile(source[start:end], "orion.py", "exec"), namespace)
    confine = namespace["_confine_link_launch"]
    assert confine(["ORION.exe", 'orion://x"', "--forge-verify", "evil.py"]) == [
        "ORION.exe", 'orion://x"']
    assert confine(["ORION.exe", "--offline"]) == ["ORION.exe", "--offline"]
    assert confine([]) == []
    # Before the first mode dispatch reads argv.
    assert end < source.index('"--mcp-server" in sys.argv[1:]')
    assert end < source.index('"--offline" in sys.argv[1:]')


def test_a_restart_can_drop_the_command_that_started_orion():
    argv = ["ORION.exe", "--offline", "--do", "briefing", "orion://show", "--ask-file=C:\\x"]
    assert ws.strip_shell_args(argv) == ["ORION.exe", "--offline"]


def test_the_cheap_check_finds_only_shell_commands():
    assert ws.argv_may_carry_command(["ORION.exe", "--do", "mute"])
    assert ws.argv_may_carry_command(["ORION.exe", "ORION://show"])
    assert ws.argv_may_carry_command(["ORION.exe", "--ask-file=C:\\x"])
    assert not ws.argv_may_carry_command(["ORION.exe", "--offline", "--background"])
    assert not ws.argv_may_carry_command(None)


# ── re-validation on the receiving side ──────────────────────────────────────

def test_every_parsed_command_survives_revalidation(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    commands = [ws.parse_link(url) for url in (
        "orion://show", "orion://page/tasks", "orion://face/screen",
        "orion://ask?q=hello", f"orion://toast?t={TOKEN}&a=open")]
    commands += [ws.parse_argv(["ORION.exe", "--ask-file", str(note)]),
                 ws.parse_argv(["ORION.exe", "--organise", str(tmp_path)]),
                 ws.parse_argv(["ORION.exe", "--do", "last-task"]),
                 ws.parse_argv(["ORION.exe", "--do", "dock:show"])]
    for command in commands:
        assert command is not None
        assert ws.validate_command(command) == command
        assert ws.validate_command(command, check_exists=True) == command


@pytest.mark.parametrize("obj", [
    None, "show", ["show"],
    {},
    {"action": "format"},
    {"action": "show", "origin": "web"},
    {"action": "show", "extra": 1},
    {"action": "ask_file", "path": "C:\\Windows\\win.ini", "origin": "link"},
    {"action": "organise", "path": "C:\\Users", "origin": "link"},
    {"action": "last_task", "origin": "link"},
    {"action": "ask_file", "path": "relative.txt"},
    {"action": "ask_file", "path": "C:\\a\\..\\b"},
    {"action": "ask_file", "path": "\\\\?\\C:\\x"},
    {"action": "ask_file"},
    {"action": "ask_file", "path": 7},
    {"action": "face", "form": "evil"},
    {"action": "face", "form": ["orb"]},
    {"action": "face", "form": "orb"},                    # real, but not offered in this build
    {"action": "face", "form": "classic", "origin": "link"},
    {"action": "dock", "state": "sideways"},
    {"action": "dock", "state": "show", "origin": "link"},  # the dock is never a link
    {"action": "page", "page": "../x"},
    {"action": "prefill", "text": "x" * 501},
    {"action": "prefill", "text": 5},
    {"action": "toast", "token": "short", "verb": "open"},
    {"action": "toast", "token": TOKEN, "verb": "rm -rf"},
])
def test_invalid_commands_are_refused_on_receipt(obj):
    assert ws.validate_command(obj) is None


def test_an_oversize_or_garbled_line_is_refused():
    assert ws.decode_command(b"{" + b" " * ws.MAX_PAYLOAD_BYTES + b"}")[0] is None
    assert ws.decode_command(b"not json")[0] is None
    assert ws.decode_command(b"\xff\xfe")[0] is None
    assert ws.decode_command(b'{"action": "show"}')[0] == {"action": "show", "origin": "cli"}


# ── registration, against a fake winreg ──────────────────────────────────────

class FakeKey:
    def __init__(self, path: str) -> None:
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeWinreg:
    """Enough of winreg, with its semantics: missing keys raise OSError and
    DeleteKey refuses a key that still has subkeys."""

    HKEY_CURRENT_USER = "HKCU"
    REG_SZ = 1
    KEY_READ = 0x20019
    KEY_WRITE = 0x20006

    def __init__(self) -> None:
        self.keys: dict[str, dict[str, tuple[str, int]]] = {}
        self.writes: list[tuple[str, str, str]] = []
        self.deleted: list[str] = []

    @staticmethod
    def _norm(root: str, sub: str) -> str:
        assert root == "HKCU", "ORION must only ever write HKCU"
        return sub.lower()

    def CreateKeyEx(self, root, sub, reserved=0, access=0):
        parts = sub.split("\\")
        for i in range(1, len(parts) + 1):
            self.keys.setdefault("\\".join(parts[:i]).lower(), {})
        return FakeKey(self._norm(root, sub))

    def OpenKey(self, root, sub, reserved=0, access=0):
        path = self._norm(root, sub)
        if path not in self.keys:
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return FakeKey(path)

    def QueryValueEx(self, key, name):
        values = self.keys[key.path]
        if name.lower() not in values:
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return values[name.lower()]

    def SetValueEx(self, key, name, reserved, kind, value):
        self.keys[key.path][name.lower()] = (value, kind)
        self.writes.append((key.path, name, value))

    def EnumKey(self, key, index):
        prefix = key.path + "\\"
        children = sorted({p[len(prefix):].split("\\")[0] for p in self.keys
                           if p.startswith(prefix)})
        if index >= len(children):
            raise OSError(259, "No more data is available")
        return children[index]

    def DeleteKey(self, root, sub):
        path = self._norm(root, sub)
        if path not in self.keys:
            raise FileNotFoundError(2, "missing")
        if any(p.startswith(path + "\\") for p in self.keys):
            raise PermissionError(5, "Access is denied")
        del self.keys[path]
        self.deleted.append(path)

    # -- helpers for the assertions
    def value(self, sub: str, name: str = "") -> str | None:
        entry = self.keys.get(("Software\\Classes\\" + sub).lower(), {}).get(name.lower())
        return entry[0] if entry else None

    def has(self, sub: str) -> bool:
        return ("Software\\Classes\\" + sub).lower() in self.keys


EXE = '"C:\\Program Files\\ORION\\ORION.exe"'


def test_the_protocol_is_registered_under_hkcu_with_a_marker():
    reg = FakeWinreg()
    assert ws.register_protocol(EXE, reg=reg) == "registered"
    assert reg.value("orion") == "URL:ORION Protocol"
    assert reg.value("orion", "URL Protocol") == ""
    assert reg.value("orion", "OrionManaged") == "1"
    assert reg.value("orion\\DefaultIcon") == "C:\\Program Files\\ORION\\ORION.exe,0"
    assert reg.value("orion\\shell\\open\\command") == f'{EXE} "%1"'
    assert all(path.startswith("software\\classes\\orion") for path, _n, _v in reg.writes)


def test_registering_twice_writes_nothing_and_a_move_rewrites_only_what_changed():
    reg = FakeWinreg()
    ws.register_protocol(EXE, reg=reg)
    ws.register_context_menu(EXE, reg=reg)
    before = len(reg.writes)
    assert ws.register_protocol(EXE, reg=reg) == "unchanged"
    assert ws.register_context_menu(EXE, reg=reg) == "unchanged"
    assert len(reg.writes) == before

    moved = '"D:\\Apps\\ORION\\ORION.exe"'
    assert ws.register_protocol(moved, reg=reg) == "registered"
    changed = reg.writes[before:]
    assert {name for _path, name, _value in changed} == {""}       # icon + command only
    assert len(changed) == 2
    assert reg.value("orion\\shell\\open\\command") == f'{moved} "%1"'


def test_the_explorer_entries():
    reg = FakeWinreg()
    assert ws.register_context_menu(EXE, reg=reg) == "registered"
    expected = {
        "*\\shell\\ORION.Ask": ("Ask ORION About This File", '--ask-file "%1"'),
        "Directory\\shell\\ORION.Ask": ("Ask ORION About This Folder", '--ask-file "%1"'),
        "Directory\\shell\\ORION.Organise": ("Organise with ORION", '--organise "%1"'),
        "Directory\\Background\\shell\\ORION.Ask": ("Ask ORION About This Folder",
                                                   '--ask-file "%V"'),
    }
    for key, (label, tail) in expected.items():
        assert reg.value(key, "MUIVerb") == label
        assert reg.value(key, "Icon") == "C:\\Program Files\\ORION\\ORION.exe,0"
        assert reg.value(key, "OrionManaged") == "1"
        assert reg.value(key + "\\command") == f"{EXE} {tail}"


def test_menu_and_task_labels_are_in_house_title_case():
    from orion_core.gui.labels import title_case

    for label in [verb.label for verb in ws.CONTEXT_VERBS] + [t.title for t in ws.DEFAULT_TASKS]:
        assert title_case(label) == label, label


def test_unregistering_removes_only_orions_keys():
    reg = FakeWinreg()
    # Somebody else's verbs live beside ORION's.
    with reg.CreateKeyEx("HKCU", "Software\\Classes\\*\\shell\\VSCode\\command") as key:
        reg.SetValueEx(key, "", 0, reg.REG_SZ, "code.exe %1")
    ws.register_protocol(EXE, reg=reg)
    ws.register_context_menu(EXE, reg=reg)

    assert ws.unregister_protocol(reg=reg) == "removed"
    assert ws.unregister_context_menu(reg=reg) == "removed"
    assert not reg.has("orion")
    for verb in ws.CONTEXT_VERBS:
        assert not reg.has(verb.key)
    assert reg.value("*\\shell\\VSCode\\command") == "code.exe %1"
    assert reg.has("*\\shell") and reg.has("Directory\\shell")
    # Idempotent.
    assert ws.unregister_protocol(reg=reg) == "absent"
    assert ws.unregister_context_menu(reg=reg) == "absent"


def test_a_key_without_the_marker_is_never_touched():
    """Another program's orion:// handler (or the owner's own) is theirs."""
    reg = FakeWinreg()
    with reg.CreateKeyEx("HKCU", "Software\\Classes\\orion\\shell\\open\\command") as key:
        reg.SetValueEx(key, "", 0, reg.REG_SZ, "other.exe %1")
    with reg.CreateKeyEx("HKCU", "Software\\Classes\\Directory\\shell\\ORION.Organise") as key:
        reg.SetValueEx(key, "MUIVerb", 0, reg.REG_SZ, "Mine")
    before = list(reg.writes)

    assert ws.register_protocol(EXE, reg=reg) == "foreign"
    assert ws.unregister_protocol(reg=reg) == "foreign"
    assert reg.value("orion\\shell\\open\\command") == "other.exe %1"
    state = ws.register_context_menu(EXE, reg=reg)
    assert state.startswith("foreign")
    assert reg.value("Directory\\shell\\ORION.Organise", "MUIVerb") == "Mine"
    assert reg.value("Directory\\shell\\ORION.Organise", "OrionManaged") is None
    assert ws.unregister_context_menu(reg=reg).startswith("foreign")
    assert reg.has("Directory\\shell\\ORION.Organise")
    assert not any(path.endswith("orion.organise") for path, _n, _v in reg.writes[len(before):])
    assert ws.registration_state(reg=reg)["orion"] == "foreign"


def test_sync_follows_the_settings_switches():
    reg = FakeWinreg()
    on = {"protocol": True, "explorer_menu": True}
    status = ws.sync_registration(on, reg=reg, exe_cmd=EXE)
    assert status == {"command": EXE, "protocol": "registered", "explorer_menu": "registered"}
    assert ws.sync_registration(on, reg=reg, exe_cmd=EXE)["protocol"] == "unchanged"
    status = ws.sync_registration({"protocol": False, "explorer_menu": True}, reg=reg, exe_cmd=EXE)
    assert status["protocol"] == "removed" and status["explorer_menu"] == "unchanged"
    assert not reg.has("orion")
    status = ws.sync_registration({"protocol": False, "explorer_menu": False}, reg=reg,
                                  exe_cmd=EXE)
    assert status["protocol"] == "absent" and status["explorer_menu"] == "removed"


def test_sync_reads_windows_settings_by_default():
    reg = FakeWinreg()
    assert windows_settings.set_value("explorer_menu", False)
    assert windows_settings.SETTINGS_PATH.exists()
    status = ws.sync_registration(reg=reg, exe_cmd=EXE)
    assert status["protocol"] == "registered"
    assert status["explorer_menu"] == "absent"


def test_a_developer_checkout_never_registers(monkeypatch):
    reg = FakeWinreg()
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert ws.launch_command() is None
    status = ws.sync_registration({"protocol": True, "explorer_menu": True}, reg=reg)
    assert status["protocol"] == "skipped" and status["explorer_menu"] == "skipped"
    assert reg.writes == [] and reg.keys == {}
    assert ws.register_protocol(None, reg=reg) == "skipped"


def test_the_launch_command():
    exe = "C:\\Program Files\\ORION\\ORION.exe"
    assert ws.launch_command(frozen=True, executable=exe) == f'"{exe}"'
    assert ws.launch_command(frozen=False, executable=exe, env={}) is None
    assert ws.launch_command(frozen=False, env={"ORION_SHELL_REGISTER": "0"}) is None
    opted = ws.launch_command(frozen=False, env={"ORION_SHELL_REGISTER": "1"})
    assert opted.startswith('"') and opted.endswith('orion.py"')


def test_registry_failures_become_a_status_not_an_exception():
    class Broken(FakeWinreg):
        def CreateKeyEx(self, *a, **k):
            raise PermissionError(5, "Access is denied")

    assert ws.register_protocol(EXE, reg=Broken()).startswith("error")
    assert ws.register_context_menu(EXE, reg=Broken()).startswith("error")


# ── the jump list, against a fake COM factory ────────────────────────────────

class FakeLink:
    def __init__(self) -> None:
        self.description = ""
        self.title = None
        self.committed = False

    def SetPath(self, path):
        self.path = path

    def SetArguments(self, arguments):
        self.arguments = arguments

    def SetWorkingDirectory(self, workdir):
        self.workdir = workdir

    def SetIconLocation(self, path, index):
        self.icon = (path, index)

    def SetDescription(self, text):
        self.description = text


class FakeStore:
    def __init__(self, link: FakeLink) -> None:
        self.link = link

    def SetValue(self, key, value):
        self.link.key = (str(key.fmtid), key.pid)
        self.link.vt = value.vt
        self.link.title = value.pwszVal

    def Commit(self):
        self.link.committed = True


class FakeCollection:
    def __init__(self) -> None:
        self.items: list = []

    def AddObject(self, item):
        self.items.append(item)


class FakeDestination:
    def __init__(self, com: "FakeCom") -> None:
        self.com = com

    def SetAppID(self, app_id):
        self.com.calls.append(("SetAppID", app_id))

    def AddUserTasks(self, array):
        self.com.calls.append(("AddUserTasks", array))

    def CommitList(self):
        if self.com.fail_commit:
            raise OSError("Commit refused")
        self.com.calls.append(("CommitList",))

    def AbortList(self):
        self.com.calls.append(("AbortList",))

    def DeleteList(self, app_id):
        self.com.calls.append(("DeleteList", app_id))


class FakeCom:
    def __init__(self, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.calls: list[tuple] = []
        self.links: list[FakeLink] = []
        self.collection = FakeCollection()
        self.initialised = self.uninitialised = 0

    def initialize(self):
        self.initialised += 1

    def uninitialize(self):
        self.uninitialised += 1

    def create(self, kind):
        if kind == "destination_list":
            return FakeDestination(self)
        if kind == "object_collection":
            return self.collection
        if kind == "shell_link":
            link = FakeLink()
            self.links.append(link)
            return link
        raise AssertionError(kind)

    def query(self, obj, kind):
        if kind == "property_store":
            return FakeStore(obj)
        if kind == "object_array":
            return ("array", obj)
        raise AssertionError(kind)

    def begin_list(self, destination):
        self.calls.append(("BeginList",))
        return 10


def test_the_jump_list_tasks():
    from orion_core.desktop_app import APP_USER_MODEL_ID

    com = FakeCom()
    status = ws.set_jump_list(EXE, com=com)
    assert status == {"ok": True, "tasks": 5, "reason": ""}
    # Mark XXXVII: no Classic Face / Orb tasks (and with one face, nothing to
    # switch); "Show Dock" is how the docked face is asked for.
    assert [link.title for link in com.links] == [
        "Show ORION", "Morning Briefing", "Mute", "Unmute", "Show Dock"]
    assert [link.arguments for link in com.links] == [
        "--do show", "--do briefing", "--do mute", "--do unmute", "--do dock:show"]
    for link in com.links:
        assert link.path == "C:\\Program Files\\ORION\\ORION.exe"
        assert link.workdir == "C:\\Program Files\\ORION"
        assert link.icon == ("C:\\Program Files\\ORION\\ORION.exe", 0)
        assert link.key == ("{F29F85E0-4FF9-1068-AB91-08002B27B3D9}", 2)   # PKEY_Title
        assert link.vt == ws.VT_LPWSTR and link.committed
        assert link.description
        assert ws.parse_argv(["ORION.exe", *link.arguments.split()]) is not None
    assert com.collection.items == com.links
    names = [call[0] for call in com.calls]
    assert names == ["SetAppID", "BeginList", "AddUserTasks", "CommitList"]
    assert com.calls[0] == ("SetAppID", APP_USER_MODEL_ID)
    assert com.calls[2] == ("AddUserTasks", ("array", com.collection))
    assert com.initialised == com.uninitialised == 1


def test_a_developer_launch_command_keeps_its_script_argument():
    com = FakeCom()
    ws.set_jump_list('"C:\\py\\pythonw.exe" "C:\\src\\orion.py"', [("Mute", "--do mute")],
                     com=com, icon="C:\\src\\assets\\orion.ico,0")
    link = com.links[0]
    assert link.path == "C:\\py\\pythonw.exe"
    assert link.arguments == '"C:\\src\\orion.py" --do mute'
    assert link.icon == ("C:\\src\\assets\\orion.ico", 0)


def test_a_failed_commit_aborts_and_reports_instead_of_raising():
    com = FakeCom(fail_commit=True)
    status = ws.set_jump_list(EXE, com=com)
    assert status["ok"] is False and "Commit refused" in status["reason"]
    assert ("AbortList",) in com.calls
    assert com.uninitialised == 1


def test_clearing_the_jump_list():
    from orion_core.desktop_app import APP_USER_MODEL_ID

    com = FakeCom()
    assert ws.clear_jump_list(com=com) == {"ok": True, "reason": ""}
    assert com.calls == [("DeleteList", APP_USER_MODEL_ID)]


def test_a_broken_com_layer_never_raises():
    class Exploding(FakeCom):
        def create(self, kind):
            raise OSError("CoCreateInstance failed")

    assert ws.set_jump_list(EXE, com=Exploding())["ok"] is False
    assert ws.clear_jump_list(com=Exploding())["ok"] is False
    assert ws.set_jump_list("", com=FakeCom())["ok"] is False


def test_the_com_structures_match_the_windows_headers():
    import ctypes

    expected = 24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16
    assert ctypes.sizeof(ws.PROPVARIANT) == expected
    assert ctypes.sizeof(ws.PROPERTYKEY) == 20
    assert str(ws.pkey_title().fmtid) == ws.PKEY_TITLE_FMTID
    assert ws.propvariant_text("Orb").pwszVal == "Orb"


# ── the command channel ──────────────────────────────────────────────────────

def test_a_command_round_trips_through_the_pipe(qapp):
    name = _unique_pipe()
    server = ws.CommandServer(name)
    received: list[dict] = []
    server.command.connect(received.append)
    assert server.start(), server.error()
    try:
        command = {"action": "page", "page": "tasks", "origin": "link"}
        assert _in_thread(lambda: ws.forward(command, 3.0, name=name)) is True
        assert _pump_until(lambda: received)
        assert received == [command]
    finally:
        server.stop()
    assert not server.is_listening()


def test_the_pipe_is_named_for_this_user():
    assert ws.pipe_name("alice") != ws.pipe_name("bob")
    assert ws.pipe_name("Alice") == ws.pipe_name("alice")
    assert ws.pipe_name("alice").startswith(ws.PIPE_PREFIX)


@pytest.mark.parametrize("payload, reason", [
    (b"not json at all\n", "payload is not JSON"),
    (b'{"action": "format_c"}\n', "unknown action"),
    (b'{"action": "ask_file", "path": "C:\\\\Windows\\\\win.ini", "origin": "link"}\n',
     "action not allowed from a link"),
    (b"a" * (ws.MAX_PAYLOAD_BYTES + 500), "payload is too large"),
    (b'{"action": "prefill", "text": "' + b"x" * 9000 + b'"}\n', "payload is too large"),
], ids=["garbled", "unknown-action", "link-file-command", "oversize-no-newline", "oversize-line"])
def test_invalid_and_oversize_payloads_are_rejected(qapp, payload, reason):
    name = _unique_pipe()
    server = ws.CommandServer(name)
    received: list[dict] = []
    rejected: list[str] = []
    server.command.connect(received.append)
    server.rejected.connect(rejected.append)
    assert server.start(), server.error()
    try:
        reply = _in_thread(lambda: _raw_send(name, payload))
        assert reply.startswith(b'{"ok": false}'), reply
        assert _pump_until(lambda: rejected)
        assert rejected == [reason]
        assert received == []
    finally:
        server.stop()


def test_a_silent_connection_is_closed(qapp):
    name = _unique_pipe()
    server = ws.CommandServer(name, idle_ms=150)
    rejected: list[str] = []
    server.rejected.connect(rejected.append)
    assert server.start(), server.error()
    try:
        reply = _in_thread(lambda: _raw_send(name, b"", timeout=3.0))
        assert reply.startswith(b'{"ok": false}'), reply
        assert _pump_until(lambda: rejected) and rejected == ["connection idle"]
    finally:
        server.stop()


def test_forwarding_with_nobody_listening_fails_quickly(qapp):
    started = time.monotonic()
    assert ws.forward({"action": "show"}, 0.5, name=_unique_pipe()) is False
    assert time.monotonic() - started < 3.0


def test_an_invalid_command_is_never_sent():
    assert ws.forward({"action": "rm"}, 0.5, name=_unique_pipe()) is False
    assert ws.encode_command({"action": "show"}) == b'{"action":"show","origin":"cli"}\n'


# ── the refused second launch ────────────────────────────────────────────────

def _refused(monkeypatch):
    monkeypatch.setattr(si, "claim", lambda directory=None: si.Claim(False, "held"))
    monkeypatch.setattr(si, "raise_existing_window", lambda *a, **k: True)


def test_a_second_launch_with_a_link_forwards_it(monkeypatch):
    _refused(monkeypatch)
    sent: list[dict] = []
    messages: list[str] = []
    result = si.enforce(on_refused=messages.append, argv=["ORION.exe", "orion://face/screen"],
                        forward=lambda command: sent.append(command) or True)
    assert result.granted is False
    assert sent == [{"action": "face", "form": "screen", "origin": "link"}]
    assert "already running" in messages[0] and "Passed your request" in messages[0]


def test_a_plain_second_launch_is_unchanged(monkeypatch):
    _refused(monkeypatch)
    sent: list[dict] = []
    messages: list[str] = []
    si.enforce(on_refused=messages.append, argv=["ORION.exe", "--offline"],
               forward=lambda command: sent.append(command) or True)
    assert sent == []
    assert messages == ["O.R.I.O.N. is already running. Bringing his window to the front."]


def test_a_second_launch_with_a_bad_command_sends_nothing(monkeypatch):
    _refused(monkeypatch)
    sent: list[dict] = []
    messages: list[str] = []
    si.enforce(on_refused=messages.append, argv=["ORION.exe", "orion://run?cmd=calc"],
               forward=lambda command: sent.append(command) or True)
    assert sent == []
    assert "could not take that request" in messages[0]


def test_a_granted_launch_never_forwards(monkeypatch):
    monkeypatch.setattr(si, "claim", lambda directory=None: si.Claim(True, "lock held"))
    sent: list[dict] = []
    assert si.enforce(argv=["ORION.exe", "--do", "mute"],
                      forward=lambda c: sent.append(c) or True).granted
    assert sent == []


def test_the_second_launch_reaches_the_running_orion_end_to_end(qapp, monkeypatch):
    """The real pipe, named from the (test-private) mutex, as orion.py runs it."""
    _refused(monkeypatch)
    server = ws.CommandServer()           # the default, per-user name
    assert server.name == ws.pipe_name()
    received: list[dict] = []
    server.command.connect(received.append)
    assert server.start(), server.error()
    messages: list[str] = []
    try:
        _in_thread(lambda: si.enforce(on_refused=messages.append,
                                      argv=["ORION.exe", "--do", "briefing"]))
        assert _pump_until(lambda: received)
        assert received == [{"action": "briefing", "origin": "cli"}]
        assert "Passed your request" in messages[0]
    finally:
        server.stop()


_REFUSED_CHILD = '''
import json, os, sys
sys.path.insert(0, r"{root}")
import orion_core.single_instance as si
si.MUTEX_NAME = os.environ["ORION_SHELL_TEST_MUTEX"]
si.SOCKET_NAME = si.SOCKET_NAME + "-shell-test"
si.raise_existing_window = lambda *a, **k: False
import orion_core.windows_shell as ws
ws.allow_foreground_handoff = lambda: True
got = si.enforce(argv=["ORION.exe", "orion://page/tasks"])
print(json.dumps({{"granted": got.granted, "widgets": "PyQt6.QtWidgets" in sys.modules}}))
'''


def test_a_real_refused_process_forwards_with_no_event_loop(qapp):
    """A separate process, refused by a (test-private) mutex this one holds,
    with no Qt application or event loop - exactly the second ORION."""
    import json
    import subprocess

    held = si.claim()
    assert held.granted
    server = ws.CommandServer()
    received: list[dict] = []
    server.command.connect(received.append)
    assert server.start(), server.error()
    env = dict(os.environ, ORION_SHELL_TEST_MUTEX=si.MUTEX_NAME, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.Popen(
        [sys.executable, "-c", _REFUSED_CHILD.format(root=str(ROOT))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(ROOT))
    try:
        assert _pump_until(lambda: proc.poll() is not None, 90)
        out, err = proc.communicate(timeout=10)
        assert json.loads(out.strip().splitlines()[-1]) == {"granted": False, "widgets": False}
        assert "Passed your request" in err
        assert _pump_until(lambda: received)
        assert received == [{"action": "page", "page": "tasks", "origin": "link"}]
    finally:
        if proc.poll() is None:
            proc.kill()
        server.stop()
        si.release()


# ── start_windows_shell, as app.py will call it ──────────────────────────────

class FakeSignal:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, line):
        self.lines.append(line)


class FakeBus:
    def __init__(self) -> None:
        self.log = FakeSignal()


def test_start_in_a_developer_checkout(qapp, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    bus, got = FakeBus(), []
    handle = ws.start_windows_shell(bus, got.append, argv=["orion.py"],
                                    server_name=_unique_pipe())
    try:
        assert handle.listening
        assert handle.worker is None
        assert any("registration skipped" in line for line in bus.log.lines)
        assert all(line.startswith("WINDOWS: ") for line in bus.log.lines)
        command = {"action": "mute", "origin": "cli"}
        assert _in_thread(lambda: ws.forward(command, 3.0, name=handle.server.name))
        assert _pump_until(lambda: got) and got == [command]
        assert "listening" in ws.describe_status()
    finally:
        handle.stop()
    assert not handle.listening
    assert ws._CURRENT is None


def test_the_command_this_launch_carried_is_delivered(qapp):
    got: list[dict] = []
    handle = ws.start_windows_shell(FakeBus(), got.append, argv=["ORION.exe", "--do", "unmute"],
                                    server_name=_unique_pipe(), exe_cmd=None)
    try:
        assert handle.startup_command == {"action": "unmute", "origin": "cli"}
        assert _pump_until(lambda: got)
        assert got == [{"action": "unmute", "origin": "cli"}]
    finally:
        handle.stop()


def test_a_failing_handler_is_logged_not_raised(qapp):
    bus = FakeBus()

    def explode(_command):
        raise RuntimeError("deck not attached")

    handle = ws.start_windows_shell(bus, explode, argv=["ORION.exe", "orion://show"],
                                    server_name=_unique_pipe(), exe_cmd=None)
    try:
        assert _pump_until(lambda: any("could not be carried out" in l for l in bus.log.lines))
    finally:
        handle.stop()


def test_the_installed_exe_registers_and_sets_the_jump_list_on_a_worker(qapp):
    bus, reg, com = FakeBus(), FakeWinreg(), FakeCom()
    handle = ws.start_windows_shell(
        bus, lambda c: None, argv=["ORION.exe"], server_name=_unique_pipe(),
        settings={"protocol": True, "explorer_menu": True, "jump_list": True},
        reg=reg, com=com, exe_cmd=EXE)
    try:
        assert handle.join(10)
        assert reg.value("orion\\shell\\open\\command") == f'{EXE} "%1"'
        assert reg.value("Directory\\shell\\ORION.Organise", "MUIVerb") == "Organise with ORION"
        assert len(com.links) == 5
        lines = bus.log.lines
        assert "WINDOWS: orion:// links registered." in lines
        assert "WINDOWS: Explorer menu registered." in lines
        assert "WINDOWS: jump list set (5 tasks)." in lines
        assert handle.status["jump_list"]["ok"] is True
        assert "5 tasks" in ws.describe_status()
    finally:
        handle.stop()


def test_switched_off_entries_are_taken_back_out(qapp):
    bus, reg, com = FakeBus(), FakeWinreg(), FakeCom()
    ws.register_protocol(EXE, reg=reg)
    handle = ws.start_windows_shell(
        bus, lambda c: None, argv=["ORION.exe"], server_name=_unique_pipe(),
        settings={"protocol": False, "explorer_menu": False, "jump_list": False},
        reg=reg, com=com, exe_cmd=EXE)
    try:
        assert handle.join(10)
        assert not reg.has("orion")
        assert com.calls and com.calls[0][0] == "DeleteList"
        assert "WINDOWS: orion:// links removed (turned off)." in bus.log.lines
        assert "WINDOWS: jump list off." in bus.log.lines
    finally:
        handle.stop()


# ── Mark XXXVII: the Screen Face is the only face ────────────────────────────

@pytest.mark.parametrize("url", [
    "orion://face/classic", "orion://face/Classic", "orion://face/orb", "orion://face/ORB/",
])
def test_links_for_the_classic_face_and_the_orb_are_refused(url):
    command, why = ws.check_link(url)
    assert "not part of this version" in why
    assert command is None


@pytest.mark.parametrize("verb", ["face:classic", "face:orb", "FACE:ORB"])
def test_do_verbs_for_the_classic_face_and_the_orb_are_refused(verb):
    assert ws.parse_argv(["ORION.exe", "--do", verb]) is None
    command, why = ws.check_argv(["ORION.exe", "--do", verb])
    assert command is None and "not part of this version" in why


def test_the_dormant_faces_are_still_defined_and_the_env_re_enables_one(monkeypatch):
    assert ws.FACE_FORMS == ("screen", "classic", "orb")
    assert ws.enabled_face_forms() == ("screen",)
    monkeypatch.setenv("ORION_FACE_FORM", "orb")
    assert ws.enabled_face_forms() == ("screen", "orb")
    assert ws.parse_link("orion://face/orb") == {"action": "face", "form": "orb", "origin": "link"}
    assert ws.parse_argv(["ORION.exe", "--do", "face:orb"]) == {
        "action": "face", "form": "orb", "origin": "cli"}
    assert ws.parse_link("orion://face/classic") is None          # only the named form
    assert ws.validate_command({"action": "face", "form": "orb"}) is not None


def test_the_jump_list_gains_face_tasks_only_for_extra_enabled_forms(monkeypatch):
    assert [t.title for t in ws.default_tasks()] == [t.title for t in ws.DEFAULT_TASKS]
    assert not any("Classic" in t.title or t.title == "Orb" for t in ws.default_tasks())
    from orion_core import appearance

    monkeypatch.setattr(appearance, "ENABLED_FORMS", appearance.FACE_FORMS)
    titles = [t.title for t in ws.default_tasks()]
    assert titles[-3:] == ["Screen Face", "Classic Face", "Orb"]
    for task in ws.default_tasks():
        assert ws.parse_argv(["ORION.exe", *task.arguments.split()]) is not None


def test_the_dock_task_and_command_round_trip():
    command = ws.parse_argv(["ORION.exe", "--do", "dock:show"])
    assert command == {"action": "dock", "state": "show", "origin": "cli"}
    line = ws.encode_command(command)
    assert ws.decode_command(line[:-1])[0] == command
    assert "dock" in ws.ALL_ACTIONS and "dock" not in ws.LINK_ACTIONS
    assert ws.parse_link("orion://dock/show") is None
