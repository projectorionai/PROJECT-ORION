"""
The screen face — ORION's default form — verified by rendering real frames.

What is held here, and why it is held with pixels
-------------------------------------------------
The screen face replaced the rendered human as the default because the owner
found the human one unsettling. What it has to get right is therefore visible
rather than structural: an eye that blinks, a mouth that opens with speech and
stays shut on /m/, ORION's colour (and the owner's accent when he changes it),
every state and every emotion producing a face rather than a blank or a crash,
at the overlay's 120 px and on a 1500 px monitor. So frames are rendered and
measured, the same way test_holo_head.py measures the sculpted head.

And the reason it exists at all must hold too: it is drawn on the face
pipeline's worker thread, never on the qasync loop; its frame rate drops when
nothing moves; and building it never loads WebEngine, Three.js or the face
process.

Rendered in subprocesses: a QApplication inside this suite has been observed
to break later Qt tests (see test_holo_head.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Steady cost of one frame at 640 x 440, the size the Core Window usually
#: gives the face. Measured ~2 ms; the headroom is for a busy machine. It is
#: paid on the worker thread, but that thread still holds the GIL while it
#: paints, so it is a real ceiling.
PAINT_BUDGET_MS = 12.0
PAINT_BUDGET_LARGE_MS = 24.0


def _probe_env(config_dir: str) -> dict:
    """The probe's environment: its own config directory, so a probe never
    reads (or creates) anything in the user's real ORION config."""
    import os

    env = dict(os.environ)
    env["ORION_CONFIG_DIR"] = config_dir
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def _run(probe: str, *args: str) -> dict:
    import tempfile

    script = probe.replace("{root}", str(ROOT).replace("\\", "/"))
    with tempfile.TemporaryDirectory(prefix="orion-screen-face-") as config_dir:
        proc = subprocess.run([sys.executable, "-c", script, *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=300,
                              env=_probe_env(config_dir))
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced no result:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")


# ── the rig, frame by frame ──────────────────────────────────────────────────

_RIG_PROBE = r'''
import json, os, statistics, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.constants import C
from orion_core import emotion as emotion_module
from orion_core.gui import screen_face as sf

PRI, ACC, BG = QColor(C.PRI), QColor(C.ACCENT), QColor(C.BG)
CYAN = QColor("#25bce8")
W, H = 640, 440
out = {}


def face(state="IDLE", emotion=None, seconds=1.2, amp=0.0, speaking=False,
         viseme=None, seed=5, boot=False, design=None):
    f = sf.ScreenFace(seed=seed)
    if design:
        f.set_design(design)
    if not boot:
        f.skip_boot()
    if emotion:
        f.apply_emotion(emotion, {"intensity": 1.0})
    for _ in range(int(seconds * 30)):
        if viseme:
            f.set_viseme(*viseme)
        # Hold the blink off so frames compare like with like.
        f._blink_at = f._t + 10.0
        f._blink, f._blink_age = 0.0, -1.0
        f.step(1 / 30, amp, speaking=speaking, state=state)
    return f


def frame(f, w=W, h=H, pri=PRI):
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(BG)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cx, cy, r = sf.face_geometry(w, h)
    f.paint(p, cx, cy, r, pri, ACC, BG)
    p.end()
    return img


def lit(img, step=3):
    bg = (BG.red(), BG.green(), BG.blue())
    n = 0
    for y in range(0, img.height(), step):
        for x in range(0, img.width(), step):
            c = img.pixelColor(x, y)
            if abs(c.red() - bg[0]) + abs(c.green() - bg[1]) + abs(c.blue() - bg[2]) > 60:
                n += 1
    return n


def region(img, x0, y0, x1, y1, test):
    n = 0
    for y in range(max(0, int(y0)), min(img.height(), int(y1))):
        for x in range(max(0, int(x0)), min(img.width(), int(x1))):
            if test(img.pixelColor(x, y)):
                n += 1
    return n


cx, cy, r = sf.face_geometry(W, H)
EYE = (cx + 0.52 * r - 0.3 * r, cy - 0.1 * r - 0.45 * r,
       cx + 0.52 * r + 0.3 * r, cy - 0.1 * r + 0.45 * r)
MOUTH = (cx - 0.3 * r, cy + 0.3 * r, cx + 0.3 * r, cy + 0.8 * r)
red_eye = lambda c: c.red() > 150 and c.red() > c.green() + 60
cyan_eye = lambda c: c.blue() > 150 and c.blue() > c.red() + 60
filled = lambda c: c.red() > 55

# Every state and every emotion, at the overlay's size, the window's, and a
# big monitor's.
states = ["INITIALISING", "LISTENING", "SPEAKING", "THINKING", "PROCESSING",
          "STANDBY", "IDLE", "PAUSED", "OFFLINE", "RECONNECTING", "CONNECTING",
          "MUTED", "FOCUS", "ERROR", "SLEEPING", "SHUTTING DOWN",
          "SOMETHING NOBODY DEFINED", ""]
emotions = sorted(set(sf.EMOTIONS) | set(emotion_module.EMOTIONS)
                  | {"focused", "determined", "alert", "alarmed", "joy",
                     "surprised", "sleepy", "love", "grateful", "angry",
                     "curious", "no-such-feeling"})
failures, blank = [], []
for size in ((120, 100), (400, 300), (1500, 900)):
    for state in states:
        try:
            f = face(state, seconds=0.6, amp=0.4, speaking=state == "SPEAKING")
            img = frame(f, *size)
            if state != "SHUTTING DOWN" and lit(img, 2 if size[0] < 200 else 4) < 20:
                blank.append(f"{state}@{size}")
        except Exception as exc:
            failures.append(f"{state}@{size}: {exc!r}")
    for name in emotions:
        try:
            f = face("IDLE", emotion=name, seconds=0.6)
            img = frame(f, *size)
            if lit(img, 2 if size[0] < 200 else 4) < 20:
                blank.append(f"{name}@{size}")
        except Exception as exc:
            failures.append(f"{name}@{size}: {exc!r}")
out["failures"] = failures
out["blank"] = blank
out["state_modes"] = {s: sf.state_mode(s) for s in states}
out["emotion_presets"] = {e: sf.emotion_preset(e) for e in emotion_module.EMOTIONS}

# Recolouring: the same face in crimson and in a user accent.
f = face("IDLE")
crimson, cyan = frame(f), frame(f, pri=CYAN)
out["eye_red_crimson"] = region(crimson, *EYE, red_eye)
out["eye_cyan_crimson"] = region(crimson, *EYE, cyan_eye)
out["eye_red_cyan"] = region(cyan, *EYE, red_eye)
out["eye_cyan_cyan"] = region(cyan, *EYE, cyan_eye)

# Blink: the same face, eyes open and mid-blink.
f = face("IDLE")
open_img = frame(f)
f._blink_at = f._t
for _ in range(40):
    f.step(1 / 60, 0.0, state="IDLE")
    if f._blink > 0.9:
        break
out["blink_value"] = f._blink
shut_img = frame(f)
out["eye_open"] = region(open_img, *EYE, red_eye)
out["eye_blink"] = region(shut_img, *EYE, red_eye)
# ...and it opens again, by itself.
for _ in range(30):
    f.step(1 / 60, 0.0, state="IDLE")
out["eye_after"] = region(frame(f), *EYE, red_eye)

# The mouth: shut at rest, open on a viseme, open on amplitude alone while
# speaking, shut on /m/ at full volume, shut while LISTENING to a loud user.
# Measured on the Companion design, whose open mouth is a filled D.
CM = "companion"
out["mouth_rest"] = region(frame(face("IDLE", design=CM)), *MOUTH, filled)
out["mouth_viseme"] = region(frame(face("SPEAKING", amp=0.7, speaking=True,
                                        viseme=(0.9, 0.0, 0.0), design=CM)), *MOUTH, filled)
out["mouth_amp"] = region(frame(face("SPEAKING", amp=0.7, speaking=True, design=CM)),
                          *MOUTH, filled)
out["mouth_pp"] = region(frame(face("SPEAKING", amp=0.9, speaking=True,
                                    viseme=(0.0, 0.0, 1.0), design=CM)), *MOUTH, filled)
out["mouth_listening_loud"] = region(frame(face("LISTENING", amp=0.9, design=CM)),
                                     *MOUTH, filled)
# Sentinel (the default, Mark XXXIX): a light bar that becomes a level meter.
bright = lambda c: c.red() > 120
out["bar_rest"] = region(frame(face("IDLE")), *MOUTH, bright)
out["bar_speaking"] = region(frame(face("SPEAKING", amp=0.7, speaking=True,
                                        viseme=(0.9, 0.0, 0.0))), *MOUTH, bright)
out["bar_pp"] = region(frame(face("SPEAKING", amp=0.9, speaking=True,
                                  viseme=(0.0, 0.0, 1.0))), *MOUTH, bright)
out["designs"] = {key: lit(frame(face("IDLE", emotion="happy", design=key)), 4)
                  for key in sf.DESIGNS}

# Expressions are actually different pictures.
def diff(a, b):
    n = 0
    for y in range(0, a.height(), 3):
        for x in range(0, a.width(), 3):
            if a.pixel(x, y) != b.pixel(x, y):
                n += 1
    return n
neutral = frame(face("IDLE"))
out["diff"] = {name: diff(neutral, frame(face("IDLE", emotion=name)))
               for name in ("happy", "sad", "surprised", "curious", "concerned",
                            "love", "angry")}
out["diff_states"] = {s: diff(neutral, frame(face(s, seconds=1.5)))
                      for s in ("LISTENING", "THINKING", "PAUSED", "OFFLINE")}

# The power-on: the very first frame is already a visor, not a blank.
f = sf.ScreenFace(seed=1)
f.step(0.0, 0.0, state="IDLE")
out["boot_first_lit"] = lit(frame(f))
out["boot_first_eyes"] = region(frame(f), *EYE, red_eye)
for _ in range(int(sf.BOOT_SECONDS * 60) + 6):
    f.step(1 / 60, 0.0, state="IDLE")
out["boot_done_eyes"] = region(frame(f), *EYE, red_eye)
out["boot_done_lit"] = lit(frame(f))

# Sleep shows the z's.
f = face("PAUSED", seconds=4.0)
out["sleep_zs"] = len(f._zs)
out["sleep_open"] = f._p["open"]

# Resting (the dormant avatar) dozes off after a quiet while, and wakes when
# it hears something.
f = sf.ScreenFace(seed=9)
f.skip_boot()
f.set_morph(0.0)
f.step(0.05, 0.0, state="IDLE")
out["rest_mode"] = f._mode
for _ in range(int(50 / 0.25)):
    f.step(0.25, 0.0, state="IDLE")
out["dozed_mode"] = f._mode
f.step(0.05, 0.4, state="IDLE")
out["woken_mode"] = f._mode

# Speech beats a sleepy emotion: the mouth never freezes mid-sentence.
f = face("SPEAKING", emotion="sleepy", amp=0.7, speaking=True, viseme=(0.9, 0.0, 0.0))
out["sleepy_speaking_mode"] = f._mode
out["sleepy_speaking_open"] = f._open
f = face("IDLE", emotion="sleepy", seconds=2.0)
out["sleepy_idle_mode"] = f._mode

# Cost: median of warm frames.
def cost(w, h):
    f = face("SPEAKING", amp=0.7, speaking=True, viseme=(0.8, 0.0, 0.0), emotion="happy")
    frame(f, w, h)
    times = []
    for _ in range(9):
        f.step(1 / 30, 0.7, speaking=True, state="SPEAKING")
        frame(f, w, h)
        times.append(f.last_paint_ms)
    return statistics.median(times)
out["paint_ms"] = cost(640, 440)
out["paint_ms_large"] = cost(1500, 900)
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def rig() -> dict:
    return _run(_RIG_PROBE)


def test_every_state_and_emotion_paints_at_every_size(rig):
    assert rig["failures"] == [], rig["failures"][:10]
    assert rig["blank"] == [], f"blank frames: {rig['blank'][:10]}"


def test_unknown_states_rest_at_idle(rig):
    modes = rig["state_modes"]
    assert modes["SOMETHING NOBODY DEFINED"] == "idle"
    assert modes[""] == "idle"
    assert modes["LISTENING"] == "listening"
    assert modes["PROCESSING"] == "thinking"
    assert modes["PAUSED"] == "sleep"
    assert modes["SHUTTING DOWN"] == "off"
    assert modes["STANDBY"] == "rest"


def test_every_emotion_orion_emits_has_an_expression(rig):
    """EmotionStateManager's names are what actually arrive; none of them
    may fall through to a blank neutral face by accident."""
    unmapped = [name for name, preset in rig["emotion_presets"].items()
                if preset == "neutral" and name != "neutral"]
    assert unmapped == []


def test_the_face_wears_the_colour_it_is_given(rig):
    """ORION's crimson by default, the owner's accent when he picks one —
    never a colour of the renderer's own."""
    assert rig["eye_red_crimson"] > 300 and rig["eye_cyan_crimson"] == 0
    assert rig["eye_cyan_cyan"] > 300 and rig["eye_red_cyan"] == 0


def test_a_blink_closes_the_eye_and_it_reopens(rig):
    assert rig["blink_value"] > 0.85
    assert rig["eye_blink"] < rig["eye_open"] * 0.3, (rig["eye_open"], rig["eye_blink"])
    assert rig["eye_after"] > rig["eye_open"] * 0.9


def test_the_mouth_opens_with_speech(rig):
    assert rig["mouth_viseme"] > rig["mouth_rest"] * 2.5 + 40, rig
    assert rig["mouth_amp"] > rig["mouth_rest"] * 2.0 + 30, rig


def test_a_lip_closure_keeps_the_mouth_shut(rig):
    """/m/ /b/ /p/ at full volume: closure wins over loudness."""
    assert rig["mouth_pp"] < rig["mouth_viseme"] * 0.5


def test_the_sentinel_mouth_is_a_meter_that_still_closes(rig):
    assert rig["bar_speaking"] > rig["bar_rest"] * 1.5 + 30, rig
    assert rig["bar_pp"] < rig["bar_speaking"] * 0.6, rig


def test_every_design_draws_a_face(rig):
    from orion_core import appearance
    from orion_core.gui import screen_face as sf
    assert set(rig["designs"]) == set(appearance.FACE_DESIGNS)
    assert sf.DEFAULT_DESIGN == appearance.DEFAULT_FACE_DESIGN == "sentinel"
    for key, count in rig["designs"].items():
        assert count > 200, (key, count)


def test_a_loud_user_does_not_open_orions_mouth(rig):
    """The microphone shares the amplitude signal with playback. Hearing you
    must not look like talking over you."""
    assert rig["mouth_listening_loud"] < rig["mouth_viseme"] * 0.5


def test_expressions_and_states_are_different_pictures(rig):
    for name, changed in {**rig["diff"], **rig["diff_states"]}.items():
        assert changed > 150, f"{name} looks the same as neutral ({changed} px)"


def test_the_first_frame_is_already_a_face_screen(rig):
    """Instant first paint: the visor is up on frame one and the eyes draw on
    from the scanline within the power-on."""
    # The visor, rim and brackets are already up: a substantial share of a
    # finished frame's light (the rest is the eyes and their glow).
    assert rig["boot_first_lit"] > rig["boot_done_lit"] * 0.3, (
        rig["boot_first_lit"], rig["boot_done_lit"])
    assert rig["boot_first_eyes"] == 0
    assert rig["boot_done_eyes"] > 300


def test_sleep_shows_closed_eyes_and_zs(rig):
    assert rig["sleep_open"] < 0.1
    assert rig["sleep_zs"] >= 1


def test_resting_dozes_off_and_a_sound_wakes_him(rig):
    assert rig["rest_mode"] == "rest"
    assert rig["dozed_mode"] == "sleep"
    assert rig["woken_mode"] == "rest"


def test_speech_beats_a_sleepy_emotion(rig):
    assert rig["sleepy_speaking_mode"] == "speaking"
    assert rig["sleepy_speaking_open"] > 0.5
    assert rig["sleepy_idle_mode"] == "sleep"


def test_paint_stays_within_budget(rig):
    assert rig["paint_ms"] < PAINT_BUDGET_MS, rig["paint_ms"]
    assert rig["paint_ms_large"] < PAINT_BUDGET_LARGE_MS, rig["paint_ms_large"]


# ── the panel: off the loop, and quiet when nothing moves ────────────────────

_PIPELINE_PROBE = r'''
import json, os, sys, threading, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6 import sip
from PyQt6.QtWidgets import QApplication
app = QApplication([])
import orion_core.gui.screen_face as sf
# Offscreen counts as headless, which (correctly) keeps the face off a worker
# nobody can see. Pretend there is a screen so the real path runs.
sf.headless = lambda: False
sf.face_thread_enabled = lambda: True

out = {}
main = threading.get_ident()
panel = sf.ScreenFacePanel()
calls = []
real_paint = panel._head.paint
def spy(*a, **k):
    calls.append(threading.get_ident())
    return real_paint(*a, **k)
panel._head.paint = spy
panel.resize(480, 360)
panel.show()


def pump(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


pump(1.2)
pipe = panel._pipeline
out["threaded"] = pipe is not None
out["rendered"] = getattr(pipe, "rendered", 0)
out["paints"] = len(calls)
out["paints_on_gui"] = sum(1 for c in calls if c == main)


def sample(img):
    red = cyan = lit = 0
    for y in range(0, img.height(), 3):
        for x in range(0, img.width(), 3):
            c = img.pixelColor(x, y)
            if c.red() > 150 and c.red() > c.green() + 60:
                red += 1
            if c.blue() > 150 and c.blue() > c.red() + 60:
                cyan += 1
            if c.red() + c.green() + c.blue() > 90:
                lit += 1
    return red, cyan, lit


out["grab_red"], out["grab_cyan"], out["grab_lit"] = sample(panel.grab().toImage())

# Every protocol call, through the queue, none of them raising.
errors = []
for name, args in (("set_state", ("LISTENING",)), ("set_amplitude", (0.5,)),
                   ("set_speaking", (True,)), ("set_viseme", ("PP", 1.0)),
                   ("set_viseme", ("aa",)), ("set_viseme", (0.5, 0.1, 0.0)),
                   ("glance", (0.0, -0.65, 1.2)), ("glance", ()),
                   ("apply_emotion", ("happy", {"glow_pulse": 0.1})),
                   ("apply_emotion", ("not-a-feeling", None)),
                   ("set_label", ("ORION · LISTENING",)), ("set_pose", (0.1, -0.05)),
                   ("set_spectrum", (0.2, 0.3, 0.4)), ("set_morph", (1.0,)),
                   ("set_orb_hands", ({},)), ("set_orb_camera_enabled", (True,)),
                   ("set_speaking", (False,)), ("set_amplitude", ("junk",)),
                   ("apply_emotion", ("happy", {"intensity": 1.0}))):
    try:
        getattr(panel, name)(*args)
    except Exception as exc:
        errors.append(f"{name}{args}: {exc!r}")
out["protocol_errors"] = errors
pump(0.5)
out["emotion_applied"] = panel._head._emotion
out["label_applied"] = panel._head._label

panel.set_palette_colours("#25bce8", "#ededed")
pump(0.5)
out["recolour_red"], out["recolour_cyan"], _ = sample(panel.grab().toImage())
out["paints_on_gui_after"] = sum(1 for c in calls if c == main)

# Hidden: the worker stops instead of drawing for nobody, and comes back on
# show without a black frame in between.
first_thread = pipe._thread
panel.hide()
pump(0.3)
out["hidden_stopped"] = panel._pipeline is None and not first_thread.is_alive()
panel.show()
out["shown_has_frame"] = panel._last_frame is not None
pump(0.4)
out["shown_running"] = panel._pipeline is not None and panel._pipeline.running
out["shown_fps"] = panel._pipeline.fps if panel._pipeline is not None else 0.0

thread = panel._pipeline._thread
panel.stop_pipeline()
out["stopped"] = panel._pipeline is None and not thread.is_alive()

# A face retired the way core_window does it (_retire_face: setParent(None),
# deleteLater, no close event) must take its worker with it, and leave
# nothing behind that Qt or the GC can trip over later.
import gc
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from orion_core import appearance
from orion_core.gui.core_window import OrionCoreWindow
hook_errors = []
sys.excepthook = lambda kind, value, tb: hook_errors.append(f"{kind.__name__}: {value}")
listeners_before = len(appearance._motion_listeners)
container = QWidget()
layout = QVBoxLayout(container)
second = sf.ScreenFacePanel()
layout.addWidget(second)
container.resize(320, 260)
container.show()
pump(0.4)
worker = second._pipeline._thread
out["second_alive_before"] = worker.is_alive()
OrionCoreWindow._retire_face(second, layout)
worker.join(2.0)
out["second_alive_after"] = worker.is_alive()
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
del second
for _ in range(3):
    third = sf.ScreenFacePanel()
    third.show()
    pump(0.1)
    third.setParent(None)
    third.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    del third
gc.collect()
pump(0.2)
out["listeners_leaked"] = len(appearance._motion_listeners) - listeners_before
out["live_panels"] = len(sf._LIVE_PANELS)
out["hook_errors"] = hook_errors
container.close()
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def threaded() -> dict:
    return _run(_PIPELINE_PROBE)


def test_the_face_renders_on_the_worker_thread(threaded):
    """Under qasync the Qt thread IS the asyncio loop: the face may blit
    there, but it must never be painted there."""
    assert threaded["threaded"] is True
    assert threaded["rendered"] > 5 and threaded["paints"] > 5
    assert threaded["paints_on_gui"] == 0
    assert threaded["paints_on_gui_after"] == 0


def test_the_blit_shows_the_face(threaded):
    assert threaded["grab_red"] > 100 and threaded["grab_lit"] > 1000


def test_every_protocol_call_is_accepted_and_reaches_the_rig(threaded):
    assert threaded["protocol_errors"] == []
    assert threaded["emotion_applied"] == "happy"
    assert threaded["label_applied"] == "ORION · LISTENING"


def test_set_palette_colours_recolours_the_live_face(threaded):
    assert threaded["recolour_cyan"] > 100
    assert threaded["recolour_red"] < threaded["recolour_cyan"] * 0.1


def test_a_hidden_face_renders_nothing_and_resumes_on_show(threaded):
    assert threaded["hidden_stopped"] is True
    assert threaded["shown_has_frame"] is True, "un-hiding would flash black"
    assert threaded["shown_running"] is True
    assert threaded["shown_fps"] >= 12.0


def test_the_worker_stops_when_asked_and_when_the_face_is_retired(threaded):
    assert threaded["stopped"] is True
    assert threaded["second_alive_before"] is True
    assert threaded["second_alive_after"] is False


def test_retiring_faces_raises_nothing_and_leaks_nothing(threaded):
    """Per-panel hooks on ``destroyed`` first raised TypeError inside Qt (a
    lambda default PyQt drops) and then crashed the process from inside the
    garbage collector. There are none now; this holds it."""
    assert threaded["hook_errors"] == []
    assert threaded["listeners_leaked"] == 0
    assert threaded["live_panels"] <= 1          # only the first panel


_BUDGET_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.gui.screen_face import MAX_RENDER_PIXELS, ScreenFacePanel

panel = ScreenFacePanel()
panel.resize(480, 360)
panel.show()
panel.set_state("LISTENING")
out = {"start_ms": panel.timer.interval(), "headless_threaded": panel._threaded}
for _ in range(40):
    panel._tick()
out["idle_ms"] = panel.timer.interval()
out["idling"] = bool(panel._budget.idling)

panel.set_speaking(True)
panel.set_amplitude(0.85)
panel.set_viseme(0.9, 0.0, 0.0)
panel._tick()
out["speaking_ms"] = panel.timer.interval()

panel.set_speaking(False)
panel.set_amplitude(0.0)
for _ in range(80):
    panel._tick()
out["resettled_ms"] = panel.timer.interval()

# A blink about to start wakes the rate BEFORE it starts.
panel._head._blink_at = panel._head._t + 0.05
panel._tick()
out["before_blink_ms"] = panel.timer.interval()

# The synchronous path (headless or ORION_FACE_THREAD=0) still draws the face,
# reusing its buffer, and caps a huge frame.
img = QImage(480, 360, QImage.Format.Format_RGB32)
panel.render(img)
first = panel._buffer
panel.render(img)
out["buffer_reused"] = first is panel._buffer
lit = 0
for y in range(0, 360, 3):
    for x in range(0, 480, 3):
        c = img.pixelColor(x, y)
        if c.red() > 150 and c.red() > c.green() + 60:
            lit += 1
out["sync_red"] = lit
panel.resize(2600, 1500)
w, h, cx, cy, r = panel._draw_geometry()
out["capped"] = [w, h]
out["cap_pixels"] = MAX_RENDER_PIXELS
out["min_size"] = [panel.minimumWidth(), panel.minimumHeight()]
out["has_timer"] = hasattr(panel, "timer")
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def budget() -> dict:
    return _run(_BUDGET_PROBE)


def test_headless_never_starts_a_worker(budget):
    assert budget["headless_threaded"] is False


def test_an_idle_face_drops_its_frame_rate(budget):
    assert budget["start_ms"] <= 34
    assert budget["idling"] is True
    assert budget["idle_ms"] > budget["start_ms"] * 1.5
    assert budget["idle_ms"] <= 125, "breathing would visibly step"


def test_speech_restores_full_rate_immediately(budget):
    """Never lower the ACTIVE rate while speaking."""
    assert budget["speaking_ms"] <= 34


def test_it_settles_again_and_a_blink_wakes_it_first(budget):
    assert budget["resettled_ms"] == budget["idle_ms"]
    assert budget["before_blink_ms"] <= 34


def test_the_synchronous_fallback_draws_and_caps(budget):
    assert budget["buffer_reused"] is True
    assert budget["sync_red"] > 100
    w, h = budget["capped"]
    assert w * h <= budget["cap_pixels"] * 1.01
    assert w < 2600 and h < 1500


def test_it_fits_the_compact_sizes(budget):
    assert budget["min_size"][0] <= 120 and budget["min_size"][1] <= 120
    assert budget["has_timer"] is True       # overlay mode retimes it by name


# ── the window builds it, and builds nothing heavy doing so ──────────────────

_BUILD_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["ORION_HOLO_HEAD"] = "0"
os.environ["ORION_FACE_FORM"] = sys.argv[1]
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.constants import C
from orion_core.gui import core_window
before = set(sys.modules)
face = core_window.OrionCoreWindow._build_face(object())
new = sorted(set(sys.modules) - before)
heavy = [m for m in new if "WebEngine" in m or m.endswith(("face3d", "face_host",
         "holo_head", "head_mesh", "native_face")) or m.split(".")[0] == "numpy"]
print("@@" + json.dumps({
    "kind": type(face).__name__,
    "heavy": heavy,
    "primary": face._primary.name(),
    "accent": face._accent.name(),
    "PRI": C.PRI.lower(), "ACCENT": C.ACCENT.lower(),
}))
'''


@pytest.mark.parametrize("form", ["screen", "face"])
def test_the_window_builds_the_screen_face_without_webengine(form):
    """The performance win: no Chromium face process, no Three.js page and
    no numpy head at startup. A legacy stored "face" gets it too."""
    built = _run(_BUILD_PROBE, form)
    assert built["kind"] == "ScreenFacePanel"
    assert built["heavy"] == [], built["heavy"]
    assert built["primary"] == built["PRI"]
    assert built["accent"] == built["ACCENT"]


# ── switching forms (window logic, no widgets) ───────────────────────────────

def _window_stub(face, tmp_path, monkeypatch, all_forms=False):
    """all_forms=True exercises the DORMANT classic/orb code paths (Mark XXXVII
    enables only the Screen Face; appearance.ENABLED_FORMS gates the rest)."""
    from orion_core import appearance
    from orion_core.gui.core_window import OrionCoreWindow

    monkeypatch.setattr(appearance, "APPEARANCE_PATH", tmp_path / "appearance.json")
    monkeypatch.delenv("ORION_FACE_FORM", raising=False)
    if all_forms:
        monkeypatch.setattr(appearance, "ENABLED_FORMS", appearance.FACE_FORMS)
    events = []
    stub = SimpleNamespace(
        face=face, _orb_hands_enabled=False, avatar=None,
        bus=SimpleNamespace(log=SimpleNamespace(emit=events.append)),
        _WEBGL_FORMS=OrionCoreWindow._WEBGL_FORMS)
    stub._current_face_form = lambda: OrionCoreWindow._current_face_form(stub)
    stub._swap_face = lambda: events.append("swap")
    return OrionCoreWindow, stub, events


def test_the_window_knows_which_form_is_on_screen(tmp_path, monkeypatch):
    screen = type("ScreenFacePanel", (), {})()
    window, stub, _ = _window_stub(screen, tmp_path, monkeypatch)
    assert window._current_face_form(stub) == "screen"
    stub.face = SimpleNamespace(form="face")          # the Three.js face
    assert window._current_face_form(stub) == "classic"
    stub.face = SimpleNamespace(form="orb")
    assert window._current_face_form(stub) == "orb"
    stub.face = type("CentralHud", (), {})()
    assert window._current_face_form(stub) == "orb"


def test_screen_to_classic_rebuilds_and_classic_to_orb_morphs(tmp_path, monkeypatch):
    screen = type("ScreenFacePanel", (), {})()
    window, stub, events = _window_stub(screen, tmp_path, monkeypatch, all_forms=True)
    assert window.set_face_form(stub, "classic") == "classic"
    assert "swap" in events, "the screen face cannot morph; it must be rebuilt"

    morphs = []
    events.clear()
    stub.face = SimpleNamespace(form="face", set_form=morphs.append)
    assert window.set_face_form(stub, "orb") == "orb"
    assert morphs == ["orb"] and "swap" not in events

    morphs.clear()
    assert window.set_face_form(stub, "screen") == "screen"
    assert morphs == [] and "swap" in events, "the Three.js face cannot become a screen"
    assert any("Screen Face" in str(line) for line in events)


def test_the_behaviour_engine_is_told_a_form_it_understands(tmp_path, monkeypatch):
    told = []
    screen = type("ScreenFacePanel", (), {})()
    window, stub, _ = _window_stub(screen, tmp_path, monkeypatch, all_forms=True)
    stub.avatar = SimpleNamespace(set_form=told.append)
    for form in ("classic", "orb", "screen"):
        window.set_face_form(stub, form)
    assert told == ["face", "orb", "auto"]


def test_this_build_will_not_switch_away_from_the_screen_face(tmp_path, monkeypatch):
    """Mark XXXVII: asking the window for the classic face or the orb changes
    nothing - no rebuild, no morph, nothing stored."""
    screen = type("ScreenFacePanel", (), {})()
    window, stub, events = _window_stub(screen, tmp_path, monkeypatch)
    stub.avatar = SimpleNamespace(set_form=lambda form: events.append(("avatar", form)))
    for form in ("classic", "orb", "Classic Face", "sphere"):
        assert window.set_face_form(stub, form) == ""
    assert "swap" not in events and not any(isinstance(e, tuple) for e in events)
    assert window.set_face_form(stub, "screen") == "screen"
