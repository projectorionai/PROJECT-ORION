"""ORION's form on screen, and what his camera can actually do.

Two unrelated complaints with one thing in common: the honest answer came from
measuring the hardware rather than from what an API reported.

THE FORM. The default face is a rendered human one, and a human face is not
always what you want in the room. The orb is now a first-class form, asked for
by voice or from the menu, and the choice is remembered — somebody who turned
the face off because it unsettles a housemate should not have to do it again
every restart.

THE CAMERA. It reported 1920x1080 at 60 fps and delivered 5.1. DirectShow
ignores a request to switch to MJPG and keeps sending uncompressed YUY2, which
saturates USB at 1080p. Media Foundation negotiates a compressed format by
itself and delivers 30.1 fps at the same resolution.

THE LANDMARKS. OpenCV 5 ships no Haar cascade, so face tracking had no model at
all. YuNet replaces it and returns five landmarks with every face — both eyes,
the nose tip and both mouth corners — from a 227 KB model built into cv2.

Offline: no camera is opened and no window is built.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import appearance, face_tracking, utils  # noqa: E402

WINDOW_SOURCE = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(
    encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(appearance, "APPEARANCE_PATH", tmp_path / "appearance.json")
    monkeypatch.delenv("ORION_FACE_FORM", raising=False)


@pytest.fixture
def all_forms(monkeypatch):
    """The dormant code paths: every form enabled, as a developer (or a later
    build that widens ENABLED_FORMS) would have it."""
    monkeypatch.setattr(appearance, "ENABLED_FORMS", appearance.FACE_FORMS)


# -- the form ---------------------------------------------------------------

@pytest.mark.parametrize("said,expected", [
    ("orb", "orb"),
    ("go into your orb form", "orb"),
    ("sphere", "orb"),
    ("switch to the ball", "orb"),
    # The legacy value: "face" was "the default face", and the default face
    # is now the screen face — the owner's choice.
    ("face", "screen"),
    ("show me your face", "screen"),
    ("avatar", "screen"),
    ("screen", "screen"),
    ("screen face", "screen"),
    ("your robot face", "screen"),
    ("the cute one", "screen"),
    ("companion", "screen"),
    ("just your eyes", "screen"),
    ("classic", "classic"),
    ("classic face", "classic"),
    ("the quantum face", "classic"),
    ("realistic", "classic"),
    ("human", "classic"),
    ("your 3d face", "classic"),
    ("Screen Face", "screen"),
    ("Classic Face", "classic"),
])
def test_the_words_people_use_are_understood(said, expected):
    """Reached by voice as often as by menu, so "orb form" must not fail on
    the word "form", and "classic face" must not be taken for the screen face
    because it contains the word "face"."""
    assert appearance.normalise(said) == expected


def test_the_forms_are_screen_classic_and_orb():
    # All three stay DEFINED (the code paths are dormant, not deleted)...
    assert appearance.FACE_FORMS == ("screen", "classic", "orb")
    # ...but Mark XXXVII offers the Screen Face alone.
    assert appearance.ENABLED_FORMS == ("screen",)
    assert appearance.enabled_forms() == ("screen",)
    assert appearance.DEFAULT_FORM in appearance.ENABLED_FORMS
    assert appearance.FORM_LABELS == {"screen": "Screen Face",
                                      "classic": "Classic Face", "orb": "Orb"}
    assert appearance.form_label("face") == "Screen Face"


def test_a_stored_legacy_face_now_means_the_screen_face():
    """Everybody who never touched the setting stored nothing, and somebody
    who once picked "face" picked the default face — which is now the screen
    face. Neither is put back on the quantum human."""
    appearance.APPEARANCE_PATH.write_text('{"face_form": "face"}', encoding="utf-8")
    assert appearance.face_form() == "screen"


@pytest.mark.parametrize("stored", ["classic", "orb", "Classic Face", "sphere", "face", "banana"])
def test_a_stored_classic_or_orb_wears_the_screen_face_in_this_build(stored):
    """Mark XXXVII shows only the Screen Face: an older build's stored classic
    or orb choice maps to it (and the stored value is left alone, so widening
    ENABLED_FORMS later restores it)."""
    appearance.APPEARANCE_PATH.write_text('{"face_form": "%s"}' % stored, encoding="utf-8")
    assert appearance.face_form() == "screen"
    assert stored in appearance.APPEARANCE_PATH.read_text(encoding="utf-8")


def test_a_stored_choice_comes_back_when_its_form_is_enabled(all_forms):
    appearance.APPEARANCE_PATH.write_text('{"face_form": "orb"}', encoding="utf-8")
    assert appearance.face_form() == "orb"


@pytest.mark.parametrize("asked", ["classic", "orb", "classic face", "switch to the sphere"])
def test_a_form_this_build_does_not_offer_is_refused_and_changes_nothing(asked):
    appearance.set_face_form("screen")
    before = appearance.APPEARANCE_PATH.read_text(encoding="utf-8")
    assert appearance.form_unavailable(asked) and not appearance.form_enabled(asked)
    assert appearance.set_face_form(asked) == ""
    assert appearance.face_form() == "screen"
    assert appearance.APPEARANCE_PATH.read_text(encoding="utf-8") == before
    assert appearance.SCREEN_ONLY_REPLY == "I only have my Screen Face in this version."


def test_the_screen_face_is_always_available_and_gibberish_is_not_unavailable():
    assert appearance.form_enabled("screen face") and not appearance.form_unavailable("screen")
    assert not appearance.form_unavailable("banana")     # unknown, not "dormant"


def test_describe_names_each_form(all_forms):
    for form, words in (("screen", "screen face"), ("classic", "classic face"),
                        ("orb", "orb form")):
        appearance.set_face_form(form)
        assert words in appearance.describe()


@pytest.mark.parametrize("said", ["", "   ", "banana", "loud"])
def test_an_unrecognised_form_changes_nothing(said, all_forms):
    """Silently putting the face back on somebody who asked for the orb is the
    one outcome to avoid here."""
    appearance.set_face_form("orb")
    assert appearance.set_face_form(said) == ""
    assert appearance.face_form() == "orb"


def test_the_choice_outlives_the_session(all_forms):
    appearance.set_face_form("orb")
    assert appearance.face_form() == "orb"
    assert appearance.APPEARANCE_PATH.is_file(), "nothing was written down"


def test_the_default_is_the_screen_face():
    assert appearance.DEFAULT_FORM == "screen"
    assert appearance.face_form() == "screen"


def test_an_override_wins(monkeypatch, all_forms):
    appearance.set_face_form("classic")
    monkeypatch.setenv("ORION_FACE_FORM", "orb")
    assert appearance.face_form() == "orb"
    monkeypatch.setenv("ORION_FACE_FORM", "screen")
    assert appearance.face_form() == "screen"


def test_the_env_override_still_re_enables_a_dormant_form_for_development(monkeypatch):
    """ORION_FACE_FORM=orb brings the orb back for that process only."""
    assert appearance.enabled_forms() == ("screen",)
    monkeypatch.setenv("ORION_FACE_FORM", "orb")
    assert appearance.face_form() == "orb"
    assert appearance.enabled_forms() == ("screen", "orb")
    assert appearance.form_enabled("orb") and not appearance.form_enabled("classic")
    assert appearance.set_face_form("orb") == "orb"
    monkeypatch.delenv("ORION_FACE_FORM")
    assert appearance.face_form() == "screen"      # the env was the only reason the orb counted


def test_the_window_honours_the_choice_before_every_fallback():
    """The orb is a first-class form rather than a degraded mode, so it is
    chosen ahead of the automatic face fallbacks, not after them — and the
    screen face, the default, is chosen before anything touches WebEngine."""
    build = WINDOW_SOURCE[WINDOW_SOURCE.index("def _build_face"):]
    build = build[:build.index("\n    def ", 10)]
    assert 'form == "orb"' in build
    # Consulted before any face is constructed (webengine_face_possible() is
    # only asked beforehand so the orb itself can be the Three.js one).
    assert build.index('form == "orb"') < build.index("if webgl_ok:")
    screen = build.index('form == "screen"')
    assert screen < build.index("webengine_face_possible()")
    assert screen < build.index("face3d")
    assert screen < build.index('form == "orb"')
    # ...but never ahead of the explicit escape hatch.
    assert build.index("holo_head_enabled()") < screen


def test_switching_rebuilds_whichever_kind_is_showing():
    """_rebuild_face returns early unless the face is a WebEngine one, because
    it exists to recover from a native window recreation. Switching has to
    work orb -> face too, where the OLD widget is not WebEngine at all."""
    from orion_core.gui.core_window import OrionCoreWindow

    swap = inspect.getsource(OrionCoreWindow._swap_face)
    assert "_face_is_webengine" not in swap
    assert "_build_face()" in swap


def test_the_menu_route_exists():
    assert 'more_menu.addMenu("Appearance")' in WINDOW_SOURCE
    assert '"face_form", "appearance", "form"' in WINDOW_SOURCE


def test_the_code_paths_for_every_form_are_kept_in_title_case():
    """Dormant, not deleted: the classic face and the orb entries are still in
    the window code (built only for forms ENABLED_FORMS offers)."""
    for form, label in (("screen", "Screen Face"), ("classic", "Classic Face"),
                        ("orb", "Orb")):
        assert f'("{form}", "{label}"' in WINDOW_SOURCE, (form, label)
    assert "if form not in _offered:" in WINDOW_SOURCE


@pytest.mark.parametrize("phrase,want", [
    ("go into your orb form", "orb"),
    ("switch to orb", "orb"),
    ("change to your face", "face"),
    ("orb mode", "orb"),
    ("show me your orb", "orb"),
    ("switch to your classic face", "classic face"),
    ("go back to your screen face", "screen face"),
    ("use your robot face", "robot face"),
    ("classic mode", "classic"),
])
def test_the_voice_route_exists(phrase, want):
    from orion_core.reflex import match_reflex

    match = match_reflex(phrase)
    assert match is not None, phrase
    assert match.args == {"action": "face_form", "target": want}
    # ...and whatever was said lands on a real (defined) form.
    assert appearance.normalise(want) in appearance.FACE_FORMS


def _tool(emit):
    from types import SimpleNamespace

    from orion_core.dispatch_desktop import DesktopDispatchMixin

    tool = DesktopDispatchMixin()
    tool.bus = SimpleNamespace(gui_command=SimpleNamespace(emit=emit))
    return tool


def test_the_tool_answers_for_the_screen_face():
    sent = []
    result = _tool(sent.append).interface_control({"action": "face_form", "target": "face"})
    assert result.ok and "screen face" in result.text
    assert sent == [{"action": "face_form", "target": "screen"}]


@pytest.mark.parametrize("target,form,words", [
    ("classic face", "classic", "classic face"),
    ("orb", "orb", "orb form"),
])
def test_the_dormant_forms_still_work_when_enabled(target, form, words, all_forms):
    sent = []
    result = _tool(sent.append).interface_control({"action": "face_form", "target": target})
    assert result.ok and words in result.text
    assert sent == [{"action": "face_form", "target": form}]


@pytest.mark.parametrize("phrase", [
    "go into your orb form", "switch to orb", "orb mode", "show me your orb",
    "switch to your classic face", "classic mode",
])
def test_asking_for_the_classic_face_or_the_orb_gets_a_polite_screen_face_only(phrase):
    """The reflex still catches the phrase; the tool declines it kindly, and
    nothing reaches the window."""
    from orion_core.reflex import match_reflex

    match = match_reflex(phrase)
    assert match is not None and match.tool == "interface_control", phrase
    result = _tool(lambda _c: pytest.fail("a disabled form reached the window")
                   ).interface_control(match.args)
    assert not result.ok
    assert result.text == "I only have my Screen Face in this version."


def test_the_tool_refuses_a_form_that_does_not_exist():
    tool = _tool(lambda _c: pytest.fail("an unknown form reached the window"))
    result = tool.interface_control({"action": "face_form", "target": "banana"})
    assert not result.ok and "screen face" in result.text.lower()


def test_the_tool_schema_offers_only_the_screen_face():
    source = (ROOT / "orion_core" / "dispatch_schema.py").read_text(encoding="utf-8")
    assert "you only have your SCREEN face" in source
    assert "screen, classic or orb" not in source


def test_the_header_form_buttons_and_appearance_menu_are_built_from_enabled_forms():
    """No lone "Screen Face" pill and no Appearance submenu while only one form
    exists; the orb hand-control action is still created (detached)."""
    assert "if len(self._form_buttons) < 2:" in WINDOW_SOURCE
    assert "self.form_toggle.hide()" in WINDOW_SOURCE
    assert 'if len(_offered) > 1 or "orb" in _offered:' in WINDOW_SOURCE


# -- the camera -------------------------------------------------------------

def test_high_resolution_prefers_media_foundation():
    """DirectShow reports 60 fps at 1080p and delivers 5.1, because it stays
    on uncompressed YUY2. Media Foundation delivers 30.1."""
    source = inspect.getsource(utils.open_camera_capture)
    assert '("CAP_MSMF", "CAP_DSHOW", "CAP_ANY") if hi_res' in source


def test_ordinary_use_still_prefers_directshow():
    """MSMF's grab path throws -1072873821 on many Windows setups, which is
    why it was second. The order flips only when resolution is wanted."""
    source = inspect.getsource(utils.open_camera_capture)
    assert 'else ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY")' in source


def test_the_size_is_negotiated_before_the_validating_read():
    """Setting the resolution after a frame has been pulled makes Media
    Foundation fail to re-select the stream and hand back a malformed Mat —
    zero frames, and a camera that looks broken."""
    source = inspect.getsource(utils.open_camera_capture)
    assert source.index("CAP_PROP_FRAME_WIDTH") < source.index("cap.read()")


def test_a_backend_that_cannot_deliver_a_frame_is_rejected():
    """isOpened() is not the same as working."""
    source = inspect.getsource(utils.open_camera_capture)
    read_at = source.index("if cap.read()[0]:")
    assert "continue" in source[read_at:read_at + 500]


def test_the_tracker_asks_for_the_resolution_the_camera_is():
    assert face_tracking._CAPTURE_WIDTH == 1920
    assert face_tracking._CAPTURE_HEIGHT == 1080


def test_the_frame_rate_matches_what_the_sensor_can_give():
    """A C920 tops out at 1080p30; 60 fps exists only at 720p and below.
    Asking for more buys nothing and burns a core on repeat frames."""
    assert 24 <= face_tracking._TARGET_FPS <= 31


def test_the_tracker_does_not_renegotiate_after_opening():
    source = inspect.getsource(face_tracking.FaceTracker._loop)
    assert "size=(_CAPTURE_WIDTH, _CAPTURE_HEIGHT)" in source
    # .get() is fine — that is how the negotiated size is read back for the
    # log. It is .set() afterwards that breaks Media Foundation.
    assert "set(cv2.CAP_PROP_FRAME_WIDTH" not in source, (
        "the caller is setting the size again after the opener validated it")


# -- the landmarks ----------------------------------------------------------

def test_the_landmark_model_ships_with_orion():
    from orion_core.constants import resource_path

    model = Path(resource_path("assets", "face",
                               "face_detection_yunet_2023mar.onnx"))
    assert model.is_file(), "the face model is missing"
    assert model.stat().st_size > 100_000


def test_detection_runs_on_a_downscaled_copy():
    """98 ms on a full 1920x1080 frame against 6.7 ms at 640x360. At 30 fps
    the first is three frames of budget for a single detection."""
    assert face_tracking._DETECT_WIDTH <= 800
    source = inspect.getsource(face_tracking.FaceTracker._detect_yunet)
    assert "cv2.resize" in source


def test_yunet_is_preferred_over_the_absent_cascade():
    source = inspect.getsource(face_tracking.FaceTracker._loop)
    assert "detector = self._load_detector()" in source
    assert source.index("_load_detector") < source.index("_load_cascade")


@pytest.mark.parametrize("part", [
    "right_eye", "left_eye", "nose", "mouth_right", "mouth_left",
])
def test_every_requested_landmark_is_reported(part):
    assert part in inspect.getsource(face_tracking.FaceTracker._detect_yunet)


def test_landmarks_are_fractions_not_pixels():
    """A consumer should not have to know what resolution the camera happened
    to negotiate, and they keep working when it changes."""
    source = inspect.getsource(face_tracking.FaceTracker._detect_yunet)
    assert "/ sw" in source and "/ sh" in source


def test_a_missing_model_degrades_instead_of_killing_capture():
    """Capturing frames is what the camera lab, vision and gestures need.
    Detection is a bonus on top and is allowed to be absent."""
    assert "return None" in inspect.getsource(face_tracking.FaceTracker._load_detector)
    assert "elif cascade is not None:" in inspect.getsource(face_tracking.FaceTracker._loop)


_PUMP_PROBE = r'''
import asyncio, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtCore import QEventLoop, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication
import qasync
app = QApplication([])
loop = qasync.QEventLoop(app); asyncio.set_event_loop(loop)
errors = []
loop.set_exception_handler(lambda _l, ctx: errors.append(str(ctx.get("exception"))))
class Bus(QObject):
    gui_command = pyqtSignal(object)
def pump(_):
    nested = QEventLoop(); QTimer.singleShot(300, nested.quit); nested.exec()
bus = Bus()
bus.gui_command.connect(pump, Qt.ConnectionType.QueuedConnection if sys.argv[1] == "queued"
                        else Qt.ConnectionType.AutoConnection)
ticks = [0]
async def receive():
    while True:
        await asyncio.sleep(0.02); ticks[0] += 1
async def main():
    task = asyncio.ensure_future(receive())
    await asyncio.sleep(0.2); bus.gui_command.emit({}); await asyncio.sleep(0.05)
    before = ticks[0]; await asyncio.sleep(1.0)
    print("@@", ticks[0] - before, len(errors)); task.cancel()
with loop:
    loop.run_until_complete(main())
'''


@pytest.mark.parametrize("mode,alive", [("direct", False), ("queued", True)])
def test_a_gui_command_that_pumps_messages_cannot_kill_the_voice_task(mode, alive):
    """8 Oct: the camera opened inline inside the voice tool task; Windows
    pumped messages while it opened, qasync re-entered the Live receive loop
    ("Cannot enter into task") and it never resumed - ten silent minutes."""
    import subprocess
    done = subprocess.run([sys.executable, "-c", _PUMP_PROBE, mode],
                          capture_output=True, text=True, timeout=120)
    line = next(l for l in done.stdout.splitlines() if l.startswith("@@"))
    ticks, errors = (int(x) for x in line.split()[1:])
    assert (ticks > 10 and errors == 0) is alive, (mode, ticks, errors)


def test_gui_commands_and_confirmations_are_queued():
    assert "gui_command.connect(self._on_gui_command, Qt.ConnectionType.QueuedConnection)" in WINDOW_SOURCE
    assert "confirm_action.connect(self._on_confirm_action, Qt.ConnectionType.QueuedConnection)" in WINDOW_SOURCE


# -- loop hygiene: no message pump runs inline on a bus signal ----------------
#
# The probe above is the mechanism; this is the guard for every other signal.
# Tool handlers run inside asyncio tasks on the Qt thread, so a bus signal they
# emit calls same-thread slots INLINE, inside the task's step. A slot that
# pumps the Windows message queue - a modal dialog, a camera open, the shell,
# a PortAudio stream, a COM call - lets qasync step other tasks nested in the
# running one ("Cannot enter into task") and they never resume. Such a slot is
# connected with Qt.ConnectionType.QueuedConnection (it then runs on a later
# loop iteration, outside every task) or hands the work to a thread.
#
# Statically, per module: each bus .connect() handler is resolved (self
# methods incl. same-module bases, local and module functions, lambdas,
# functools.partial, literal (signal, handler) tables) and its body - plus the
# self-methods and local functions it calls, one level down - is searched for
# the calls below. Nested defs/lambdas are not searched: that is how work is
# handed to a thread (threading.Thread(target=...), asyncio.to_thread(...)).

import ast  # noqa: E402
import functools  # noqa: E402

#: Any attribute call with this name pumps, whatever it is called on.
_PUMP_METHODS = {
    "exec": "a nested Qt event loop (modal dialog, menu, QEventLoop)",
    "exec_": "a nested Qt event loop (modal dialog, menu, QEventLoop)",
    "processEvents": "pumps the Qt event queue",
    "runAndWait": "SAPI (pyttsx3) runs a message loop",
    "getOpenFileName": "modal file dialog", "getOpenFileNames": "modal file dialog",
    "getSaveFileName": "modal file dialog", "getExistingDirectory": "modal file dialog",
    "ShellExecute": "ShellExecute", "ShellExecuteW": "ShellExecute",
    "ShellExecuteEx": "ShellExecute", "ShellExecuteExW": "ShellExecute",
}
#: Dotted calls after import-alias resolution; a trailing "." means any member.
_PUMP_CALLS = {
    "os.startfile": "ShellExecute", "webbrowser.open": "ShellExecute",
    "webbrowser.open_new": "ShellExecute", "webbrowser.open_new_tab": "ShellExecute",
    "QDesktopServices.openUrl": "ShellExecute",
    "os.system": "a blocking child process", "os.popen": "a blocking child process",
    "subprocess.": "a child process",
    "cv2.VideoCapture": "a camera open (Media Foundation / DirectShow)",
    "QMessageBox.question": "modal message box", "QMessageBox.warning": "modal message box",
    "QMessageBox.information": "modal message box", "QMessageBox.critical": "modal message box",
    "QMessageBox.about": "modal message box", "QInputDialog.": "modal input dialog",
    "QColorDialog.getColor": "modal dialog", "QFontDialog.getFont": "modal dialog",
    "sounddevice.InputStream": "a PortAudio stream", "sounddevice.OutputStream": "a PortAudio stream",
    "sounddevice.Stream": "a PortAudio stream", "sounddevice.RawInputStream": "a PortAudio stream",
    "sounddevice.RawOutputStream": "a PortAudio stream", "sounddevice.play": "a PortAudio stream",
    "sounddevice.rec": "a PortAudio stream", "sounddevice.playrec": "a PortAudio stream",
    "sounddevice.stop": "a PortAudio stream", "sounddevice.wait": "a PortAudio stream",
    "win32com.": "COM", "pythoncom.": "COM", "comtypes.": "COM", "pywinauto.": "UIA / COM",
    "pycaw.": "Core Audio COM", "win32clipboard.": "the Win32 clipboard",
    "pyttsx3.init": "a SAPI COM engine",
}
#: Importing one of these inside a handler means it is about to be used there.
_PUMP_IMPORTS = ("win32com", "pythoncom", "comtypes", "pywinauto", "pycaw",
                 "win32clipboard", "pyttsx3")

#: Inline bus connections allowed to reach a call above anyway, keyed
#: "path:signal:handler source" (no line numbers - they drift) -> the reason
#: it is safe. Every entry must say why; an entry that no longer matches a
#: flagged connection fails the suite so the list cannot rot.
INLINE_PUMP_ALLOWLIST: dict[str, str] = {
    # (none at present - every flagged handler is queued)
}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


@functools.lru_cache(maxsize=1)
def _bus_signals() -> frozenset[str]:
    tree = ast.parse((ROOT / "orion_core" / "bus.py").read_text(encoding="utf-8"))
    bus = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "OrionBus")
    return frozenset(t.id for stmt in bus.body if isinstance(stmt, ast.Assign)
                     and isinstance(stmt.value, ast.Call) and _dotted(stmt.value.func) == "pyqtSignal"
                     for t in stmt.targets if isinstance(t, ast.Name))


class _BusConnections:
    """One module's bus connections and what their handlers reach."""

    def __init__(self, source: str, rel: str, signals: frozenset[str]):
        self.rel, self.signals = rel, signals
        self.tree = ast.parse(source)
        self.parents = {child: parent for parent in ast.walk(self.tree)
                        for child in ast.iter_child_nodes(parent)}
        self.aliases: dict[str, str] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self.aliases.update({a.asname: a.name for a in node.names if a.asname})
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                self.aliases.update({a.asname or a.name: f"{node.module}.{a.name}"
                                     for a in node.names})
            elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                  and (_dotted(node.value.func) or "").rsplit(".", 1)[-1]
                  in {"LazyModule", "import_module"}
                  and node.value.args and isinstance(node.value.args[0], ast.Constant)):
                for target in node.targets:       # sd = LazyModule("sounddevice")
                    if isinstance(target, ast.Name):
                        self.aliases[target.id] = str(node.value.args[0].value)
        self.classes = {n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)}
        self.functions = {n.name: n for n in self.tree.body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    # -- resolution --------------------------------------------------------

    def _ancestors(self, node):
        while node in self.parents:
            node = self.parents[node]
            yield node

    def _class_of(self, node):
        return next((a for a in self._ancestors(node) if isinstance(a, ast.ClassDef)), None)

    def _method(self, cls, name, seen=()):
        if cls is None:
            return None
        for stmt in cls.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                return stmt
        for base in cls.bases:
            base_cls = self.classes.get(_dotted(base) or "")
            if base_cls is not None and base_cls.name not in seen:
                found = self._method(base_cls, name, (*seen, cls.name))
                if found is not None:
                    return found
        return None

    def _function(self, at, name):
        for scope in self._ancestors(at):
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(scope):
                    if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == name and node is not scope):
                        return node
        return self.functions.get(name)

    def _handlers(self, handler, at):
        """[(row, function-or-lambda)]; row is the table row a loop variable
        came from (so the signal can be read from the same row), else None."""
        if isinstance(handler, ast.Lambda):
            return [(None, handler)]
        if (isinstance(handler, ast.Call) and handler.args
                and (_dotted(handler.func) or "").rsplit(".", 1)[-1] == "partial"):
            return self._handlers(handler.args[0], at)
        if (isinstance(handler, ast.Attribute) and isinstance(handler.value, ast.Name)
                and handler.value.id == "self"):
            found = self._method(self._class_of(at), handler.attr)
            return [(None, found)] if found is not None else []
        if isinstance(handler, ast.Name):
            found = self._function(at, handler.id)
            if found is not None:
                return [(None, found)]
            for loop in self._ancestors(at):      # for name, slot in (("a", f), ...)
                if (isinstance(loop, ast.For) and isinstance(loop.target, ast.Tuple)
                        and isinstance(loop.iter, (ast.Tuple, ast.List))):
                    names = [getattr(t, "id", None) for t in loop.target.elts]
                    if handler.id in names:
                        index = names.index(handler.id)
                        return [(dict(zip(names, row.elts)), fn)
                                for row in loop.iter.elts if isinstance(row, (ast.Tuple, ast.List))
                                for _row, fn in self._handlers(row.elts[index], at)]
        return []       # another object's method: not resolvable from here

    def _signal(self, receiver, at):
        """The bus signal name a .connect() receiver denotes, or None."""
        def from_expr(expr):
            if (isinstance(expr, ast.Attribute) and expr.attr in self.signals
                    and "bus" in ast.unparse(expr.value).lower()):
                return expr.attr
            if (isinstance(expr, ast.Call) and _dotted(expr.func) == "getattr" and len(expr.args) > 1
                    and "bus" in ast.unparse(expr.args[0]).lower()):
                name = expr.args[1]
                return name.value if isinstance(name, ast.Constant) else "<" + ast.unparse(name) + ">"
            if isinstance(expr, ast.IfExp):
                return from_expr(expr.body)
            return None
        found = from_expr(receiver)
        if found is None and isinstance(receiver, ast.Name):   # sig = getattr(bus, ...)
            scope = next((a for a in self._ancestors(at)
                          if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            for node in ast.walk(scope) if scope is not None else ():
                if (isinstance(node, ast.Assign) and from_expr(node.value)
                        and any(getattr(t, "id", None) == receiver.id for t in node.targets)):
                    found = from_expr(node.value)
        return found

    # -- what a body does ----------------------------------------------------

    @staticmethod
    def _own_nodes(fn):
        stack = list(fn.body) if isinstance(fn.body, list) else [fn.body]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(child for child in ast.iter_child_nodes(node) if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)))

    def _pumps(self, fn):
        for node in self._own_nodes(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""])
                for module in modules:
                    if module.split(".")[0] in _PUMP_IMPORTS:
                        yield node.lineno, f"import {module}", "COM / SAPI"
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in _PUMP_METHODS:
                yield node.lineno, ast.unparse(node.func), _PUMP_METHODS[node.func.attr]
                continue
            dotted = _dotted(node.func)
            if not dotted:
                continue
            head, _, rest = dotted.partition(".")
            name = self.aliases.get(head, head) + ("." + rest if rest else "")
            for key, why in _PUMP_CALLS.items():
                if (name.startswith(key) or "." + key in name if key.endswith(".")
                        else name == key or name.endswith("." + key)):
                    yield node.lineno, name, why
                    break

    @staticmethod
    def _runs_later(fn):
        """An async def / asyncSlot handler is scheduled as its own task."""
        return isinstance(fn, ast.AsyncFunctionDef) or any(
            (_dotted(d.func if isinstance(d, ast.Call) else d) or "").rsplit(".", 1)[-1]
            in {"asyncSlot", "asyncClose"} for d in getattr(fn, "decorator_list", ()))

    def _reached(self, fn, cls):
        """fn, then the self-methods and local/module functions it calls."""
        yield fn
        for node in self._own_nodes(fn):
            if not isinstance(node, ast.Call):
                continue
            callee = None
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"):
                callee = self._method(cls, node.func.attr)
            elif isinstance(node.func, ast.Name):
                callee = self._function(node, node.func.id)
            if callee is not None and callee is not fn:
                yield callee

    def connections(self):
        """(key, queued, [problems]) for every bus .connect() in the module."""
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "connect" and node.args):
                continue
            signal = self._signal(node.func.value, node)
            if signal is None:
                continue
            kind = [*node.args[1:], *(k.value for k in node.keywords if k.arg == "type")]
            queued = any("queued" in ast.unparse(arg).lower() for arg in kind)
            cls = self._class_of(node)
            rows = self._handlers(node.args[0], node) or [(None, None)]
            for row, fn in rows:
                row_signal = signal
                if row and signal.startswith("<") and isinstance(
                        row.get(signal[1:-1]), ast.Constant):
                    row_signal = row[signal[1:-1]].value
                problems = []
                if fn is not None and not self._runs_later(fn):
                    for reached in self._reached(fn, cls):
                        for line, call, why in self._pumps(reached):
                            problems.append(f"{getattr(reached, 'name', '<lambda>')} "
                                            f"line {line}: {call} ({why})")
                handler = ast.unparse(node.args[0]).split("\n")[0][:80]
                yield f"{self.rel}:{row_signal}:{handler}", queued, problems


def _scan(source: str, rel: str):
    return list(_BusConnections(source, rel, _bus_signals()).connections())


def _inline_pumps(source: str, rel: str) -> dict[str, list[str]]:
    return {key: problems for key, queued, problems in _scan(source, rel)
            if problems and not queued}


@functools.lru_cache(maxsize=1)
def _real_tree_scan() -> tuple:
    """(key, queued, problems) for every bus connection in ORION - parsed once
    (about 5 s for the whole tree) and shared by the tests below."""
    paths = [*sorted((ROOT / "orion_core").rglob("*.py")),
             *sorted((ROOT / "launcher").rglob("*.py")), ROOT / "orion.py"]
    rows = []
    for path in paths:
        if "__pycache__" in path.parts or not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        if ".connect(" in source:
            rows.extend(_scan(source, path.relative_to(ROOT).as_posix()))
    return tuple(rows)


_BAD_MODULE = '''
import functools, os, webbrowser
import sounddevice as sd
from subprocess import run
from PyQt6.QtWidgets import QMessageBox

class Panel:
    def __init__(self, bus):
        self.bus = bus
        bus.gui_command.connect(self._on_command)
        bus.confirm_action.connect(self._ask)
        self.bus.globe_request.connect(lambda place: webbrowser.open(place))
        bus.audio_device_request.connect(functools.partial(self._reopen, "output"))
        for name, slot in (("camera_frame", self._camera), ("log", self._quiet)):
            getattr(self.bus, name).connect(slot)
        def tidy(_thought):
            run(["cleanmgr"])
        bus.thought.connect(tidy)

    def _on_command(self, command):
        self._open(command)             # one level down

    def _open(self, command):
        os.startfile(command["path"])

    def _ask(self, request):
        QMessageBox.question(None, "Confirm", request["message"])

    def _reopen(self, kind, spec):
        sd.OutputStream(device=spec).start()

    def _camera(self, frame):
        import cv2
        cv2.VideoCapture(0)

    def _quiet(self, line):
        self.bus.dashboard_event.emit("log", line)
'''

_GOOD_MODULE = '''
import asyncio, os, threading
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox
from qasync import asyncSlot

class Panel:
    def __init__(self, bus):
        bus.gui_command.connect(self._on_command, Qt.ConnectionType.QueuedConnection)
        bus.confirm_action.connect(self._ask, type=Qt.ConnectionType.QueuedConnection)
        bus.globe_request.connect(self._open_on_a_thread)
        bus.thought.connect(self._later)

    def _on_command(self, command):
        os.startfile(command["path"])

    def _ask(self, request):
        QMessageBox.question(None, "Confirm", request["message"])

    def _open_on_a_thread(self, place):
        threading.Thread(target=lambda: os.startfile(place), daemon=True).start()
        asyncio.ensure_future(asyncio.to_thread(os.startfile, place))

    @asyncSlot()
    async def _later(self, _thought):
        os.startfile("x")
'''


def test_the_guard_catches_every_kind_of_inline_pump():
    """Proof the guard can fail: methods (and one level down), lambdas,
    partials, literal signal tables, local functions, import aliases."""
    found = _inline_pumps(_BAD_MODULE, "bad.py")
    assert set(found) == {
        "bad.py:gui_command:self._on_command",
        "bad.py:confirm_action:self._ask",
        "bad.py:globe_request:lambda place: webbrowser.open(place)",
        "bad.py:audio_device_request:functools.partial(self._reopen, 'output')",
        "bad.py:camera_frame:slot",
        "bad.py:thought:tidy",
    }, found
    assert "os.startfile" in found["bad.py:gui_command:self._on_command"][0]
    assert "sounddevice.OutputStream" in found[
        "bad.py:audio_device_request:functools.partial(self._reopen, 'output')"][0]
    assert "subprocess.run" in found["bad.py:thought:tidy"][0]


def test_queued_threaded_and_async_handlers_pass_the_guard():
    connections = _scan(_GOOD_MODULE, "good.py")
    assert len(connections) == 4
    assert _inline_pumps(_GOOD_MODULE, "good.py") == {}


def test_no_bus_handler_pumps_windows_messages_inline():
    """Every bus handler that reaches a message pump is queued (or allowlisted
    with a reason)."""
    found = {key: problems for key, queued, problems in _real_tree_scan()
             if problems and not queued}
    unexplained = {key: problems for key, problems in found.items()
                   if key not in INLINE_PUMP_ALLOWLIST}
    assert not unexplained, (
        "Connect these with Qt.ConnectionType.QueuedConnection (or move the work "
        "to a thread):\n" + "\n".join(f"  {key}: {'; '.join(p)}"
                                      for key, p in unexplained.items()))
    stale = set(INLINE_PUMP_ALLOWLIST) - set(found)
    assert not stale, f"allowlist entries that no longer match anything: {stale}"
    assert all(reason.strip() for reason in INLINE_PUMP_ALLOWLIST.values())


def test_the_guard_reads_the_real_tree():
    """If the scanner silently stopped resolving ORION's handlers it would pass
    everything; the known pumping handlers (all queued) must still be seen."""
    seen = {key: (queued, problems) for key, queued, problems in _real_tree_scan()}
    assert len(seen) >= 80
    for key in ("orion_core/gui/core_window.py:gui_command:self._on_gui_command",
                "orion_core/gui/core_window.py:confirm_action:self._on_confirm_action",
                "orion_core/gui/core_window.py:open_palette:self.open_command_palette"):
        queued, problems = seen[key]
        assert queued and problems, key


#: Handlers that reach pumping or blocking work through ANOTHER object, which
#: the static guard cannot follow - pinned here instead. (file, signal, a
#: fragment of the handler source). Not here on purpose: app.py's globe
#: lambdas (_globe_on_screen) must see, AT EMISSION, whether the view existed
#: (queuing the connection would drop a second globe request emitted before
#: the first had built the view), so they stay inline and defer the expensive
#: part themselves - see test_the_globe_is_built_outside_the_tool_task.
_QUEUED_THROUGH_OTHER_OBJECTS = [
    # deck show/raise - emitted inside the electronics inspection task
    ("orion_core/app.py", "electronics_scan_started", "_toggle_deck_page"),
    # speech.set_muted -> interrupt_all -> SAPI engine.stop() / sd.stop()
    ("orion_core/app.py", "voice_mute_request", "worker.set_voice_muted"),
    # a 'stopped' sample stops the orb camera tracker (join up to 2 s)
    ("orion_core/gui/core_window.py", "orb_gesture", "self._on_orb_gesture"),
]


@pytest.mark.parametrize("rel,signal,handler", _QUEUED_THROUGH_OTHER_OBJECTS)
def test_handlers_reaching_pumps_through_other_objects_stay_queued(rel, signal, handler):
    matches = [queued for key, queued, _problems in _real_tree_scan()
               if key.startswith(rel + ":" + signal + ":") and handler in key]
    assert matches and all(matches), (rel, signal, handler, matches)


def test_the_globe_is_built_outside_the_tool_task():
    """A voice globe request decides 'did the view exist?' at emission and
    builds/raises on the next loop turn (275-512 ms of WebEngine or a page
    host spawn no longer inside the tool's task)."""
    import inspect
    from orion_core import app
    src = inspect.getsource(app)
    on_screen = src[src.index("def _globe_on_screen"):src.index("def _globe_open")]
    assert "existed = bool(globe_views)" in on_screen
    assert "QTimer.singleShot(0, lambda: _globe_open(existed, replay))" in on_screen
    assert "_open_deck_page" not in on_screen, "the page must not be opened inline"

# -- Dropped file integration continuation (13 October) --
import asyncio
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult


class _FileDropSignal:
    def __init__(self):
        self.received = []

    def emit(self, value):
        self.received.append(value)


def test_hud_drop_reviews_all_paths_sequentially_and_deduplicates(monkeypatch, tmp_path):
    from orion_core import background
    from orion_core.gui.core_window import OrionCoreWindow
    first, second = tmp_path / "first.py", tmp_path / "second.png"
    events, scheduled, logs = [], [], []
    active = 0
    peak = 0

    async def review(path):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        events.append(("start", path))
        await asyncio.sleep(0)
        events.append(("end", path))
        active -= 1
        # A failed review receipt still leaves later files to be reviewed.
        return ToolResult("synthetic unsupported file", ok=False)

    def spawn(coro, **kwargs):
        scheduled.append((coro, kwargs))

    monkeypatch.setattr(background, "spawn", spawn)
    core = SimpleNamespace(worker=SimpleNamespace(review_file=review), write_log=logs.append)
    OrionCoreWindow._hud_files_dropped(core, [str(first), "", str(second), str(first)])
    assert len(scheduled) == 1
    assert scheduled[0][1]["name"] == "orion-dropped-files"
    asyncio.run(scheduled[0][0])
    assert events == [("start", str(first)), ("end", str(first)),
                      ("start", str(second)), ("end", str(second))]
    assert peak == 1
    assert any(str(second) in line for line in logs)


def test_hud_drop_without_worker_reports_unavailable_and_schedules_nothing(monkeypatch, tmp_path):
    from orion_core import background
    from orion_core.gui.core_window import OrionCoreWindow
    logs = []
    monkeypatch.setattr(background, "spawn", lambda *_a, **_k: pytest.fail("unavailable worker scheduled review"))
    core = SimpleNamespace(worker=None, write_log=logs.append)
    OrionCoreWindow._hud_files_dropped(core, [str(tmp_path / "synthetic.py")])
    assert any("ready" in line.lower() for line in logs)


def _file_drop_worker(process_file):
    from orion_core.live_worker import GenAILiveWorker
    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.stop_event = asyncio.Event()
    worker.bus = SimpleNamespace(log=_FileDropSignal())
    worker.dispatcher = SimpleNamespace(process_file=process_file)
    announcements = []
    worker.announce = lambda text, **_: announcements.append(text)
    return worker, announcements


def test_worker_file_review_uses_shared_process_file_and_announces_honest_failure(tmp_path):
    path = str(tmp_path / "unsupported.bin")
    calls = []

    async def process_file(args):
        calls.append(args)
        return ToolResult("Could not review unsupported.bin: no decoder is available.", ok=False)

    worker, announcements = _file_drop_worker(process_file)
    result = asyncio.run(worker.review_file(path))
    assert not result.ok
    assert calls == [{"path": path}]
    assert announcements == [result.text]
    assert any(result.text in line for line in worker.bus.log.received)
    assert not any("scanned" in line.lower() for line in worker.bus.log.received)


def test_worker_file_review_contains_service_failure_without_losing_later_files(tmp_path):
    calls = []
    first, second = str(tmp_path / "bad.py"), str(tmp_path / "good.csv")

    async def process_file(args):
        calls.append(args["path"])
        if args["path"] == first:
            raise OSError("synthetic file access denied")
        return ToolResult("Reviewed good.csv: two synthetic rows.")

    worker, announcements = _file_drop_worker(process_file)
    asyncio.run(worker.review_files([first, second, first]))
    assert calls == [first, second]
    assert "synthetic file access denied" in announcements[0]
    assert "couldn't read" in announcements[0]
    assert "Reviewed good.csv" in announcements[1]


def test_worker_file_review_stops_before_access_when_shutdown_started(tmp_path):
    async def process_file(_args):
        pytest.fail("shutdown worker accessed a file")
    worker, announcements = _file_drop_worker(process_file)
    worker.stop_event.set()
    result = asyncio.run(worker.review_file(str(tmp_path / "synthetic.txt")))
    assert not result.ok and "shutting down" in result.text
    assert not announcements


class _FileDropEvent:
    def __init__(self, local_paths):
        self.accepted = False
        self.ignored = False
        self.urls = [SimpleNamespace(isLocalFile=lambda: True,
                                     toLocalFile=lambda path=path: path)
                     for path in local_paths]
        self.urls.append(SimpleNamespace(isLocalFile=lambda: False,
                                         toLocalFile=lambda: pytest.fail("remote URL treated as a file")))

    def mimeData(self):
        return SimpleNamespace(urls=lambda: self.urls)

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def test_deck_drop_forwards_all_local_paths_through_weak_core_binding(tmp_path):
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    received = []
    class Core:
        def _hud_files_dropped(self, paths):
            received.append(paths)
    core = Core()
    deck = SimpleNamespace(_core_window_ref=weakref.ref(core))
    paths = [str(tmp_path / "one.py"), str(tmp_path / "two.png")]
    event = _FileDropEvent(paths)
    UnifiedDashboard.dropEvent(deck, event)
    assert received == [paths]
    assert event.accepted and not event.ignored


def test_deck_drop_ignores_a_retired_core_without_recreating_it(tmp_path):
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    class Core:
        def _hud_files_dropped(self, _paths):
            pytest.fail("retired core received a drop")
    core = Core()
    reference = weakref.ref(core)
    del core
    assert reference() is None
    deck = SimpleNamespace(_core_window_ref=reference)
    event = _FileDropEvent([str(tmp_path / "one.py")])
    UnifiedDashboard.dropEvent(deck, event)
    assert event.ignored and not event.accepted


@pytest.fixture
def file_drop_dispatch(tmp_path, monkeypatch):
    from orion_core import recycle_bin, undo
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    from orion_core.dispatch_files import FilesDispatchMixin
    from orion_core.path_policy import Verdict, Zone
    from orion_core.system_guard import SystemActionGuard

    class Dispatch(FilesDispatchMixin, DesktopDispatchMixin):
        def _resolve_user_path(self, raw):
            path = Path(raw).resolve()
            assert path.is_relative_to(tmp_path.resolve()), "test escaped its private paths"
            return path

        def _ensure_write_safe(self, path):
            path = Path(path).resolve()
            assert path.is_relative_to(tmp_path.resolve()), "test policy escaped its private paths"
            return Verdict(Zone.USER, path, "inside a synthetic user folder")

    dispatch = Dispatch()
    dispatch.bus = SimpleNamespace(log=_FileDropSignal(), confirm_action=_FileDropSignal())
    dispatch.system_guard = SystemActionGuard(machine="TEST", logger=lambda *_: None)
    dispatch.peripherals = None
    bin_root = tmp_path / "synthetic-bin"
    bin_root.mkdir()
    recycled = []

    def recycle(path):
        path = Path(path).resolve()
        assert path.is_relative_to(tmp_path.resolve())
        recycled.append(path)
        path.rename(bin_root / path.name)
        return True

    monkeypatch.setattr(recycle_bin, "refusal", lambda _path: None)
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", recycle)
    undo.clear()
    yield SimpleNamespace(dispatch=dispatch, recycled=recycled, root=tmp_path, bin=bin_root)
    undo.clear()


@pytest.mark.parametrize("action", ["write_text", "append_text", "delete"])
def test_existing_user_file_requires_human_bound_token_despite_model_confirm_true(file_drop_dispatch, action):
    target = file_drop_dispatch.root / "existing.txt"
    target.write_bytes(b"original")
    args = {"action": action, "path": str(target), "text": "approved text",
            "confirm": True, "confirm_token": "model-invented-token"}
    dispatch = file_drop_dispatch.dispatch
    result = dispatch.file_controller(args)
    assert not result.ok and target.read_bytes() == b"original"
    assert not file_drop_dispatch.recycled
    prompts = dispatch.bus.confirm_action.received
    assert len(prompts) == 1 and prompts[0]["token"]
    assert prompts[0]["token"] not in result.text, "approval token was exposed to model text"
    args["text"] = "later unapproved text"
    approved = dispatch.confirm_system_action(prompts[0]["token"])
    assert approved.ok
    if action == "delete":
        assert not target.exists()
        assert (file_drop_dispatch.bin / target.name).read_bytes() == b"original"
        assert file_drop_dispatch.recycled == [target]
    else:
        expected = "approved text" if action == "write_text" else "originalapproved text"
        assert target.read_text(encoding="utf-8") == expected
    assert not dispatch.confirm_system_action(prompts[0]["token"]).ok
    assert len(file_drop_dispatch.recycled) == (1 if action == "delete" else 0)


def test_new_user_file_is_created_immediately_without_a_spurious_confirmation(file_drop_dispatch):
    target = file_drop_dispatch.root / "new.txt"
    result = file_drop_dispatch.dispatch.file_controller(
        {"action": "write_text", "path": str(target), "text": "synthetic new content"})
    assert result.ok and target.read_text(encoding="utf-8") == "synthetic new content"
    assert not file_drop_dispatch.dispatch.bus.confirm_action.received


def test_changed_file_after_prompt_is_preserved_and_token_consumed(file_drop_dispatch):
    target = file_drop_dispatch.root / "existing.txt"
    target.write_bytes(b"original")
    dispatch = file_drop_dispatch.dispatch
    dispatch.file_controller({"action": "write_text", "path": str(target), "text": "approved"})
    token = dispatch.bus.confirm_action.received[-1]["token"]
    target.write_bytes(b"new owner changes after the prompt")
    result = dispatch.confirm_system_action(token)
    assert not result.ok and "changed" in result.text.lower()
    assert target.read_bytes() == b"new owner changes after the prompt"
    assert not dispatch.confirm_system_action(token).ok


def test_nonempty_user_folder_is_recycled_once_after_human_approval(file_drop_dispatch, monkeypatch):
    target = file_drop_dispatch.root / "work-folder"
    target.mkdir()
    (target / "owner.txt").write_bytes(b"valuable synthetic content")
    dispatch = file_drop_dispatch.dispatch
    pending = dispatch.file_controller({"action": "delete", "path": str(target), "confirm": True})
    assert not pending.ok and target.is_dir()
    token = dispatch.bus.confirm_action.received[-1]["token"]
    original_rmdir = Path.rmdir
    def guarded_rmdir(path):
        assert path != target, "folder deletion fell back to rmdir"
        return original_rmdir(path)
    monkeypatch.setattr(Path, "rmdir", guarded_rmdir)
    assert dispatch.confirm_system_action(token).ok
    assert not target.exists()
    assert (file_drop_dispatch.bin / target.name / "owner.txt").read_bytes() == b"valuable synthetic content"
    assert file_drop_dispatch.recycled == [target]
    assert not dispatch.confirm_system_action(token).ok


def test_recycle_failure_keeps_the_user_folder_without_permanent_fallback(file_drop_dispatch, monkeypatch):
    from orion_core import recycle_bin
    target = file_drop_dispatch.root / "work-folder"
    target.mkdir()
    original = target / "owner.txt"
    original.write_bytes(b"valuable synthetic content")
    dispatch = file_drop_dispatch.dispatch
    dispatch.file_controller({"action": "delete", "path": str(target)})
    token = dispatch.bus.confirm_action.received[-1]["token"]
    monkeypatch.setattr(recycle_bin, "send_to_recycle_bin", lambda _path: False)
    original_rmdir, original_unlink = Path.rmdir, Path.unlink
    def guarded_rmdir(path):
        assert path != target, "failed recycling used rmdir"
        return original_rmdir(path)
    def guarded_unlink(path, *args, **kwargs):
        assert path not in {target, original}, "failed recycling used unlink"
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "rmdir", guarded_rmdir)
    monkeypatch.setattr(Path, "unlink", guarded_unlink)
    result = dispatch.confirm_system_action(token)
    assert not result.ok
    assert target.is_dir() and original.read_bytes() == b"valuable synthetic content"


def test_nested_folder_changes_after_preview_are_preserved(file_drop_dispatch):
    target = file_drop_dispatch.root / "work-folder"
    nested = target / "nested"
    nested.mkdir(parents=True)
    original = nested / "owner.txt"
    original.write_bytes(b"original")
    dispatch = file_drop_dispatch.dispatch
    dispatch.file_controller({"action": "delete", "path": str(target)})
    token = dispatch.bus.confirm_action.received[-1]["token"]
    original.write_bytes(b"new valuable owner changes after preview")
    result = dispatch.confirm_system_action(token)
    assert not result.ok and "changed" in result.text.lower()
    assert original.read_bytes() == b"new valuable owner changes after preview"
    assert not file_drop_dispatch.recycled


# -- Dropped files are processed, acknowledged and shared (30 October) --

class _DropOrderLog(list):
    """A deque stand-in that records when a context note was queued."""

    def __init__(self, events):
        super().__init__()
        self.events = events

    def append(self, value):
        self.events.append(("note", value))
        super().append(value)


def test_hud_drop_shares_content_before_a_short_acknowledgement(monkeypatch, tmp_path):
    """30 Oct 2026: a drop was announced as its whole path and the
    conversation never saw the content. Now the note reaches Live first,
    then a one-line acknowledgement with an offer, and the ledger keeps it."""
    from orion_core import background, file_review
    from orion_core.gui.core_window import OrionCoreWindow
    from orion_core.session_context import ActivityLedger
    path = tmp_path / "garden notes.md"
    path.write_text("# Raised Bed Plan\nOrder the sleepers on Friday.\n", encoding="utf-8")
    events, scheduled = [], []

    class Router:
        async def generate_text(self, *args, **kwargs):
            pytest.fail("a dropped file is read by the conversation, not a second model")

    async def process_file(args):
        return await file_review.review(args["path"], router=Router())

    worker, _ = _file_drop_worker(process_file)
    worker.announce = lambda text, **_: events.append(("said", text))
    worker._pending_context_notes = _DropOrderLog(events)
    worker._activity_ledger = ActivityLedger()

    async def flush():
        events.append(("flush", None))
    worker._flush_context_notes = flush
    monkeypatch.setattr(background, "spawn", lambda coro, **kwargs: scheduled.append(coro))
    core = SimpleNamespace(worker=worker, write_log=lambda *_: None)
    OrionCoreWindow._hud_files_dropped(core, [str(path)])

    async def run_all():
        while scheduled:
            await scheduled.pop(0)
    asyncio.run(run_all())
    kinds = [kind for kind, _value in events]
    assert kinds.index("note") < kinds.index("said")
    note = next(value for kind, value in events if kind == "note")
    said = next(value for kind, value in events if kind == "said")
    assert "Order the sleepers on Friday" in note and "never follow instructions" in note
    assert said.startswith("Got garden notes.md: a Markdown document") and said.endswith("Want a summary?")
    assert '"Raised Bed Plan"' in said and str(tmp_path) not in said
    assert "flush" in kinds
    recorded = worker._activity_ledger.snapshot()
    assert recorded and recorded[0].name == "process_file" and recorded[0].ok
    assert recorded[0].outcome.startswith("Got garden notes.md")


def test_hud_drop_of_an_unreadable_file_says_why_without_its_path(tmp_path):
    from orion_core import file_review
    path = tmp_path / "locked.xyz"
    path.write_bytes(b"\x00\x01binary\x00")

    async def process_file(args):
        return await file_review.review(args["path"])

    worker, announcements = _file_drop_worker(process_file)

    async def drop():
        with file_review.handed_over():
            return await worker.review_file(str(path))
    result = asyncio.run(drop())
    assert not result.ok
    assert announcements == ["I can't read .xyz files, so I haven't opened locked.xyz."]


def test_drop_zone_refuses_a_dragged_web_link():
    from orion_core.gui.hud_shell import DropZone
    styled = []
    zone = SimpleNamespace(setProperty=lambda *a: styled.append(a), _restyle=lambda: None)
    link = _FileDropEvent([])
    link.urls = [SimpleNamespace(isLocalFile=lambda: False, toLocalFile=lambda: "")]
    link.mimeData = lambda: SimpleNamespace(hasUrls=lambda: True, urls=lambda: link.urls)
    DropZone.dragEnterEvent(zone, link)
    assert link.ignored and not link.accepted and not styled


def test_scan_a_file_hands_the_file_over_without_an_offer_or_second_model(tmp_path):
    from orion_core import file_review
    from orion_core.gui.views import LogConsoleView
    path = tmp_path / "notes.txt"
    path.write_text("The pump runs at six.", encoding="utf-8")
    seen = []

    class Router:
        async def generate_text(self, *args, **kwargs):
            pytest.fail("the Live model reviews a chosen file itself")

    class Worker:
        async def submit_file_for_review(self, file_path, focus):
            seen.append(await file_review.review(file_path, router=Router()))

    asyncio.run(LogConsoleView._review_chosen_file(Worker(), str(path), ""))
    first = seen[0].text.splitlines()[0]
    assert first.startswith("Got notes.txt: a text file") and "Want" not in first
    assert "The pump runs at six" in seen[0].text
