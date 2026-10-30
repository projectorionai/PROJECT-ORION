"""Execute the actual WebGL scheduling code without a GPU or network."""

from __future__ import annotations

import shutil
import subprocess
import json
import re
from pathlib import Path

import pytest

from orion_core.gui.face3d import FACE_HTML, three_sourced

ROOT = Path(__file__).resolve().parents[1]


def _run_scheduler(assertions: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the renderer scheduler")
    source = FACE_HTML[FACE_HTML.index("const ACTIVE_FPS="):FACE_HTML.index("function animate()")]
    source = source.replace("__IDLE_FPS__", "30")
    harness = """
const assert=require('node:assert/strict');
const window={};
let now=0,lastT=0,mode='IDLE',morphT=0,morphTarget=0,speakEnv=0;
const performance={now:()=>now};
const st={amp:0},tg={amp:0};
const orbCtl={drag:null,spread:0,ts:0,x:0,tx:0,y:0,ty:0};
const orbReady=()=>true;
"""
    result = subprocess.run(
        [node, "-e", harness + source + assertions],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_slightly_early_frames_do_not_halve_the_refresh_rate():
    _run_scheduler("""
mode='LISTENING'; let painted=0;
for(let i=0;i<61;i++){now=i*16;if(takeRenderFrame(now))painted++;}
assert.ok(painted>=57&&painted<=61,`only ${painted} frames were painted`);
assert.equal(idleRate,false);
""")


def test_idle_dwell_is_timed_and_listening_wakes_immediately():
    _run_scheduler("""
for(let i=0;i<61;i++){now=i*8;targetFrameMs();}
assert.equal(idleRate,false); // 61 callbacks are only 480 ms on this display.
now=1001;assert.equal(targetFrameMs(),1000/30);
mode='LISTENING';now=1002;assert.equal(targetFrameMs(),1000/60);
mode='IDLE';now=1003;targetFrameMs();now=2100;targetFrameMs();
assert.equal(idleRate,true);
tg.amp=0.2;now=2101;assert.equal(targetFrameMs(),1000/60);
""")


def test_orb_help_does_not_overlay_the_avatar():
    assert 'id="orb-help"' not in FACE_HTML
    assert "Drag: move" not in FACE_HTML


def test_reversing_materialisation_uses_the_current_position():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the renderer timeline")
    setter = next(line for line in FACE_HTML.splitlines() if line.startswith("window.orionMorph="))
    advance = next(line for line in FACE_HTML.splitlines() if line.strip().startswith("morphT+=Math.sign("))
    script = """
const assert=require('node:assert/strict'),window={};
let morphT=0,morphTarget=1,dt=0.05;
""" + setter + "\nconst step=()=>{" + advance + "};\n" + """
for(let i=0;i<20;i++)step();
assert.ok(morphT>0.3&&morphT<0.5);
const halfway=morphT;
window.orionMorph(0);step();
assert.ok(morphT<halfway&&morphT>0);
window.orionMorph(1);step();
assert.ok(Math.abs(morphT-halfway)<1e-9);
window.orionMorph('not a number');assert.equal(morphTarget,1);
for(let i=0;i<100;i++)step();assert.equal(morphT,1);
window.orionMorph(0);for(let i=0;i<100;i++)step();assert.equal(morphT,0);
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_complete_webgl_module_has_valid_javascript_syntax():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to validate the embedded module")
    page = three_sourced(FACE_HTML)
    module = re.search(r'<script type="module">(.*?)</script>', page, re.S).group(1)
    result = subprocess.run(
        [node, "--check", "--input-type=module"], input=module,
        capture_output=True, text=True, encoding="utf-8", timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_a_resize_redraws_at_once_instead_of_presenting_a_cleared_canvas():
    """renderer.setSize clears the canvas; with the next frame up to an idle
    interval away, every resize showed a black frame. The handler now redraws
    in the same task, and ignores events that did not change the size."""
    from orion_core.gui.face3d import FACE_HTML
    handler = FACE_HTML[FACE_HTML.index("addEventListener('resize'"):]
    handler = handler[:handler.index("});") + 3]
    assert "setSize" in handler and "composer.render()" in handler
    assert handler.index("setSize") < handler.index("composer.render()")
    assert "innerWidth===sizedW" in handler


def test_chromium_occlusion_tracking_is_switched_off_for_orion_pages(monkeypatch):
    """A covered window stopped producing frames and could present a blank one
    when uncovered. The feature is merged into any --disable-features list."""
    import os
    from orion_core import app
    monkeypatch.delenv("ORION_WEB_OCCLUSION", raising=False)
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--js-flags=--x --disable-features=Foo")
    app._steady_web_surfaces()
    assert os.environ["QTWEBENGINE_CHROMIUM_FLAGS"].split() == [
        "--js-flags=--x", "--disable-features=Foo,CalculateNativeWinOcclusion"]
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "")
    app._steady_web_surfaces()
    assert os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] == "--disable-features=CalculateNativeWinOcclusion"


def test_a_black_flash_leaves_a_trace():
    """8-10 Oct: black flashes and a frozen voice left nothing to read - the
    log went only to the window. The page counts lost WebGL contexts, resizes
    and renderer stalls; Python logs what changed."""
    from orion_core.gui.face3d import FACE_HTML, health_change
    assert "webglcontextlost" in FACE_HTML and "e.preventDefault()" in FACE_HTML
    assert "orionHealth.resizes++" in FACE_HTML
    seen: dict = {}
    assert health_change(seen, '{"lost":0,"restored":0,"resizes":0,"gaps":0,"maxGap":0}') == ""
    line = health_change(seen, '{"lost":1,"restored":1,"resizes":0,"gaps":2,"maxGap":950}')
    assert line == "WebGL context lost x1; WebGL context restored x1; renderer stalled x2 (longest 950 ms)"
    assert health_change(seen, '{"lost":1,"restored":1,"resizes":0,"gaps":2,"maxGap":950}') == ""
    assert health_change(seen, "null") == "" and health_change(seen, "not json") == ""


def test_the_session_log_writes_only_when_enabled(tmp_path):
    import time
    from orion_core import session_log
    from orion_core.bus import OrionBus
    bus = OrionBus()
    assert session_log.attach(bus) is False            # tests never enable it
    path = session_log.enable(tmp_path / "session.log")
    try:
        assert session_log.attach(bus) is True
        bus.log.emit("NET: live channel closed; re-establishing.")
        session_log.flush()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and "re-establishing" not in (
                path.read_text(encoding="utf-8") if path.exists() else ""):
            time.sleep(0.05)
        assert "NET: live channel closed" in path.read_text(encoding="utf-8")
    finally:
        session_log._enabled = False


# ── the face in its own process (gui/face_host.py) ───────────────────────────

def test_the_host_runs_only_the_face_api():
    """MEASURED 10 Oct with the real face and ORION's thread busy 40% of the
    time: in-process 10 stutters >= 50 ms in 5 s, in its own process none."""
    from orion_core.gui import face_host
    calls, replies, quit_flag = [], [], []

    class Face:
        def set_state(self, state): calls.append(("set_state", state))
        def check_alive(self, callback): callback(True)

    face = Face()
    face_host.apply(face, {"m": "set_state", "a": ["LISTENING"]}, replies.append, lambda: None)
    face_host.apply(face, {"m": "check_alive", "id": 7}, replies.append, lambda: None)
    face_host.apply(face, {"m": "deleteLater"}, replies.append, lambda: None)      # not the API
    face_host.apply(face, {"m": "__quit__"}, replies.append, lambda: quit_flag.append(1))
    assert calls == [("set_state", "LISTENING")]
    assert replies == [{"t": "reply", "id": 7, "value": True}]
    assert quit_flag == [1]
    line = face_host.encode("set_avatar_frame", ({"amp": 0.5, "odd": object()},), rid=3)
    decoded = json.loads(line)
    assert decoded["m"] == "set_avatar_frame" and decoded["id"] == 3 and decoded["a"][0]["amp"] == 0.5


def test_remote_face_never_blocks_and_falls_back(monkeypatch):
    import io, os, queue, threading
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from orion_core.gui import face_host

    class Stuck:
        """A host that never reads its stdin and never answers."""
        def __init__(self, *a, **k):
            self.stdin = io.StringIO()
            self.stdout = iter(threading.Event().wait, True)    # blocks forever
            self.pid = 1
        def wait(self, timeout=None): raise TimeoutError
        def kill(self): pass

    monkeypatch.setenv("ORION_FACE_PROCESS", "1")
    face = face_host.RemoteFace(command=["host"], popen=Stuck)
    face._outbox = queue.Queue(maxsize=4)
    for i in range(50):                                   # far more than fits
        face.set_avatar_frame({"js_state": "LISTENING", "i": i})
    face.set_form("orb")
    assert face.form == "orb" and face._sticky["set_form"] == ("orb",)
    answers = []
    face.check_alive(answers.append)                      # pending: no reply yet
    face._give_up()
    assert answers == [False] and face.failed
    assert os.environ["ORION_FACE_PROCESS"] == "0"        # the rebuild draws in-process
    face.check_alive(answers.append)
    assert answers == [False, False]
    face.shutdown()
    app.processEvents()


def test_the_face_is_drawn_in_its_own_process_by_default():
    from orion_core.gui import face_host
    from orion_core.gui.core_window import _new_webgl_face  # noqa: F401
    assert face_host.FLAG in (ROOT / "orion.py").read_text(encoding="utf-8")
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert source.count("_new_webgl_face()") >= 2 and '"RemoteFace"' in source


def _startup_face_without_a_process(monkeypatch):
    from orion_core.gui import face_host, page_host
    app = _qt()
    monkeypatch.setattr(page_host.RemoteSurface, "_start", lambda self: None)
    face = face_host.RemoteFace(command=["private-fake-face-host"])
    face.resize(400, 400)
    face.show()
    app.processEvents()
    return face, app


def test_remote_face_starts_with_real_software_orb_pixels_and_live_state(monkeypatch):
    from PyQt6.QtGui import QImage
    from orion_core.gui.hud import CentralHud
    face, app = _startup_face_without_a_process(monkeypatch)
    try:
        orb = face._startup_orb
        assert isinstance(orb, CentralHud) and face._proc is None
        assert orb.isVisible() and orb.timer.isActive()
        face.set_form("face")
        face.set_avatar_frame({"js_state": "SPEAKING", "amplitude": 0.7})
        assert face.form == "face" and orb.state_name == "SPEAKING"
        assert orb.target_amplitude == 0.7
        image = QImage(400, 400, QImage.Format.Format_RGB32)
        image.fill(0)
        face.render(image)
        colours = {image.pixel(x, y) for x in range(100, 300, 10)
                   for y in range(100, 300, 10)}
        assert len(colours) > 16, "the startup avatar is a flat blank rectangle"
    finally:
        face.shutdown()
        face.close()
        app.processEvents()


def test_remote_face_placeholder_stops_when_embedded_and_returns_on_restart(monkeypatch):
    face, app = _startup_face_without_a_process(monkeypatch)
    try:
        orb = face._startup_orb
        face._on_embedded("main")
        assert orb.isHidden() and not orb.timer.isActive()
        assert face._layout.indexOf(orb) == -1
        face.set_state("LISTENING")
        face.set_amplitude(0.4)
        assert not orb.timer.isActive()
        face._on_restarting()
        assert face._layout.indexOf(orb) >= 0 and orb.isVisible()
        assert orb.state_name == "LISTENING" and orb.target_amplitude == 0.4
        face.hide()
        assert not orb.timer.isActive()
        face.show()
        assert orb.timer.isActive()
    finally:
        face.shutdown()
        assert not face._startup_orb.timer.isActive()
        face.close()
        app.processEvents()


def test_a_failed_face_host_keeps_a_static_visible_orb_until_owner_recovery(monkeypatch):
    monkeypatch.setenv("ORION_FACE_PROCESS", "1")
    face, app = _startup_face_without_a_process(monkeypatch)
    try:
        face._give_up()
        assert face.failed and face._startup_orb.isVisible()
        assert not face._startup_orb.timer.isActive()
    finally:
        face.close()
        app.processEvents()


def test_a_stutter_is_written_down():
    from orion_core.gui.face3d import health_change
    seen: dict = {}
    health_change(seen, '{"hitches":0,"maxHitch":0}')
    assert health_change(seen, '{"hitches":4,"maxHitch":210}') == "stuttered x4 (longest 210 ms)"


# ── the globe and the Brain instruments in their own process (page_host) ────
#
# MEASURED 11 Oct 2026, real windows, ORION's thread blocked 200 ms of every
# 500 ms (scratch harness, embedded and screenshotted):
#   face   in-process 19 stutters >= 50 ms / 10 s      own process 0, 60 fps
#   brain  in-process 15 fps, 16 stutters / 6 s        own process 62.5 fps, 0
#   globe  in-process 55 fps, 4 stutters / 6 s (183)   own process 60 fps, 2 in
#          70 s - and 1 in 20 s with nothing blocked: Cesium's own, not ORION's.

class _NoHost:
    """A host process that never starts talking (nothing is spawned)."""
    def __init__(self, *a, **k):
        import io
        import threading
        self.stdin = io.StringIO()
        self.stdout = iter(threading.Event().wait, True)
        self.pid = 1
    def wait(self, timeout=None): raise TimeoutError
    def kill(self): pass


def _qt():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controlled_page_host(monkeypatch):
    """Exercise lifecycle callbacks synchronously, without workers or hosts."""
    import io
    from types import SimpleNamespace
    from orion_core.gui import page_host

    app = _qt()
    threads, timers, processes = [], [], []

    class CapturedThread:
        def __init__(self, *, target, args=(), **kwargs):
            self.target, self.args = target, args
            self.name = kwargs.get("name", "")
            threads.append(self)

        def start(self):
            pass

    class Process:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = []
            self.pid = len(processes) + 10
            self.killed = False
            processes.append(self)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.killed = True

    monkeypatch.setattr(page_host.threading, "Thread", CapturedThread)
    monkeypatch.setattr(page_host.QTimer, "singleShot", lambda delay, callback: timers.append((delay, callback)))
    monkeypatch.setenv("ORION_WEB_PROCESS", "1")
    surface = page_host.RemoteWebView(command=["synthetic-host"], popen=lambda *_a, **_k: Process())
    monkeypatch.setattr(surface, "_hide_host_windows", lambda: None)
    monkeypatch.setattr(surface, "_log", lambda _line: None)
    state = SimpleNamespace(app=app, surface=surface, threads=threads, timers=timers,
                            processes=processes, process_type=Process)
    try:
        yield state
    finally:
        surface.shutdown()
        from PyQt6 import sip
        if not sip.isdeleted(surface):
            surface.close()
            surface.deleteLater()


def test_page_host_pending_restart_cannot_launch_after_shutdown(controlled_page_host):
    state = controlled_page_host
    surface = state.surface
    surface._on_exit()
    restarts = [callback for delay, callback in state.timers if delay == 1000]
    assert len(restarts) == 1
    surface.shutdown()

    restarts[0]()

    assert len(state.processes) == 1, "a retired surface launched an unowned replacement host"
    assert surface._closing


def test_page_host_retiring_writer_cannot_consume_replacement_commands(controlled_page_host):
    import queue
    from types import SimpleNamespace
    from orion_core.gui.page_host import encode
    state = controlled_page_host
    surface = state.surface
    old = state.processes[0]
    writer = next(thread for thread in state.threads if thread.name.endswith("-write"))
    # Remove constructor setup; the old writer reaches a flush boundary.
    while not surface._outbox.empty():
        surface._outbox.get_nowait()
    surface._outbox.put_nowait(encode("old-command"))
    writes, replacement_commands = [], []

    def restart_during_flush():
        surface._on_exit()
        surface._start()
        surface._send("replacement-command")
        replacement_commands.extend(list(surface._outbox.queue))

    old.stdin = SimpleNamespace(write=writes.append, flush=restart_during_flush)
    writer.target(*writer.args)

    remaining = []
    while True:
        try:
            remaining.append(surface._outbox.get_nowait())
        except queue.Empty:
            break
    assert writes == [encode("old-command")]
    assert remaining == replacement_commands, "the old writer stole its replacement's command"
    assert encode("replacement-command") in remaining


@pytest.mark.parametrize("kind", ["wid", "failed", "event", "reply"])
def test_page_host_old_reader_frames_cannot_affect_a_replacement(controlled_page_host, monkeypatch, kind):
    state = controlled_page_host
    surface = state.surface
    old = state.processes[0]
    observed, queued = [], []
    monkeypatch.setattr(surface, "_embed", lambda *args: observed.append(("embed", args)))
    monkeypatch.setattr(surface, "_on_event", lambda *args: observed.append(("event", args)))
    surface._callbacks[91] = lambda value: observed.append(("reply", value))
    message = {
        "wid": {"t": "wid", "wid": 12345, "pid": 9001},
        "failed": {"t": "failed", "reason": "retired host"},
        "event": {"t": "event", "name": "loaded", "args": [True]},
        "reply": {"t": "reply", "id": 91, "value": "retired reply"},
    }[kind]
    old.stdout = ["@@" + json.dumps(message) + "\n"]
    surface._inbox.disconnect()
    surface._inbox.connect(queued.append)
    reader = next(thread for thread in state.threads if thread.name.endswith("-read"))
    reader.target(*reader.args)
    surface._proc = state.process_type()

    for frame in queued:
        surface._on_message(frame)

    assert observed == [] and not surface.failed
    assert surface.host_pid is None
    assert 91 in surface._callbacks


@pytest.mark.parametrize("retire", ["shutdown", "restart", "give_up"])
def test_page_host_retirement_resolves_each_query_with_its_dead_answer(controlled_page_host, retire):
    surface = controlled_page_host.surface
    answers = []
    surface._query("check_alive", lambda value: answers.append(("alive", value)), dead=False)
    surface._query("check_state", lambda value: answers.append(("state", value)), dead="failed")
    surface._query("report", lambda value: answers.append(("report", value)), dead=None)

    {"shutdown": surface.shutdown, "restart": surface._on_exit, "give_up": surface._give_up}[retire]()

    assert answers == [("alive", False), ("state", "failed"), ("report", None)]
    assert surface._callbacks == {}
    if retire == "shutdown":
        surface._query("check_alive", lambda value: answers.append(("closed", value)), dead=False)
        assert answers[-1] == ("closed", False)
        assert surface._callbacks == {}


def test_page_host_current_reader_frames_and_query_replies_still_deliver(controlled_page_host, monkeypatch):
    state = controlled_page_host
    surface = state.surface
    current = state.processes[0]
    observed, queued = [], []
    monkeypatch.setattr(surface, "_embed", lambda *args: observed.append(("embed", args)))
    monkeypatch.setattr(surface, "_on_event", lambda *args: observed.append(("event", args)))
    surface._query("check_alive", lambda value: observed.append(("reply", value)), dead=False)
    rid = surface._next_id - 1
    messages = [{"t": "wid", "wid": 12345, "pid": current.pid},
                {"t": "event", "name": "loaded", "args": [True]},
                {"t": "reply", "id": rid, "value": True}]
    current.stdout = ["@@" + json.dumps(message) + "\n" for message in messages]
    surface._inbox.disconnect()
    surface._inbox.connect(queued.append)
    reader = next(thread for thread in state.threads if thread.name.endswith("-read"))
    reader.target(*reader.args)

    for frame in queued[:-1]:  # The final synthetic EOF is a separate lifecycle event.
        surface._on_message(frame)

    assert observed == [("embed", (12345, "main")), ("event", ("loaded", [True])), ("reply", True)]
    assert surface.host_pid == current.pid
    assert surface._callbacks == {} and surface._callback_dead == {}
    surface.shutdown()
    assert observed[-1] == ("reply", True), "a completed query was resolved twice"


def test_page_host_shutdown_ignores_queued_events_and_callback_errors(controlled_page_host, monkeypatch):
    surface = controlled_page_host.surface
    observed = []
    monkeypatch.setattr(surface, "_embed", lambda *args: observed.append(("embed", args)))
    monkeypatch.setattr(surface, "_on_event", lambda *args: observed.append(("event", args)))

    def bad_callback(value):
        observed.append(("failed-callback", value))
        raise RuntimeError("synthetic callback failure")

    surface._query("check_alive", bad_callback, dead=False)
    surface._query("report", lambda value: observed.append(("report", value)), dead=None)
    surface.shutdown()
    surface.shutdown()
    surface._on_message({"t": "wid", "wid": 12345, "pid": 9001})
    surface._on_message({"t": "event", "name": "loaded", "args": [True]})
    surface._on_message({"t": "reply", "id": 1, "value": "late reply"})
    surface._query("check_alive", bad_callback, dead=False)

    assert observed == [("failed-callback", False), ("report", None), ("failed-callback", False)]
    assert surface.host_pid is None
    assert surface._callbacks == {} and surface._callback_dead == {}


def test_page_host_full_retired_queue_releases_its_writer(controlled_page_host):
    import queue
    surface = controlled_page_host.surface
    old_queue = surface._outbox
    while not old_queue.full():
        old_queue.put_nowait("superseded frame")

    surface._on_exit()

    assert old_queue.get_nowait() is None
    with pytest.raises(queue.Empty):
        old_queue.get_nowait()
    assert surface._outbox is not old_queue


def test_page_host_deleted_widget_cannot_run_its_pending_restart(controlled_page_host):
    from PyQt6 import sip
    state = controlled_page_host
    surface = state.surface
    surface._on_exit()
    restart = next(callback for delay, callback in state.timers if delay == 1000)
    sip.delete(surface)

    restart()

    assert len(state.processes) == 1


def test_the_page_bridge_is_mirrored_from_qts_own_meta_object():
    _qt()
    from orion_core.gui.globe import _GlobeBridge
    from orion_core.gui.page_host import bridge_spec
    spec = bridge_spec(_GlobeBridge(None))
    assert spec == {"setWorldLayer": ["str", "bool"], "refreshWorld": [],
                    "inspectPoint": ["float", "float"], "resetView": [],
                    "routeTo": ["str", "str"]}


def test_a_page_reaches_only_the_real_slots_of_its_bridge():
    app = _qt()
    from orion_core.gui.globe import _GlobeBridge
    from orion_core.gui.page_host import RemoteWebView
    calls = []
    owner = type("Owner", (), {"inspect_point": lambda self, la, lo: calls.append((la, lo)),
                               "set_world_layer": lambda self, n, on: calls.append((n, on))})()
    bridge = _GlobeBridge(None)
    bridge.owner = owner
    view = RemoteWebView(command=["host"], popen=_NoHost, bridges={"globeBridge": bridge})
    view._on_event("bridge", ["globeBridge", "inspectPoint", ["51.5", -0.12]])
    view._on_event("bridge", ["globeBridge", "setWorldLayer", ["earthquakes", 1]])
    view._on_event("bridge", ["globeBridge", "deleteLater", []])          # not a slot
    view._on_event("bridge", ["globeBridge", "inspectPoint", [1.0]])      # wrong arity
    view._on_event("bridge", ["otherBridge", "resetView", []])            # not registered
    assert calls == [(51.5, -0.12), ("earthquakes", True)]
    view.shutdown()
    app.processEvents()


def test_the_globe_asks_its_process_for_a_disk_cache_that_outlives_it(monkeypatch):
    from types import SimpleNamespace
    """28 Oct 2026: the page process used Qt's off-the-record profile, so each
    restart (every open after a release, every crash) fetched Cesium and every
    map tile again. The globe now names a cache folder in its setup."""
    app = _qt()
    from orion_core.gui import globe, page_host
    from orion_core.gui.page_host import RemoteWebView
    view = RemoteWebView(command=["host"], popen=_NoHost, cache="C:/x/web_cache/globe")
    assert view._sticky["setup"][0]["cache"] == "C:/x/web_cache/globe"
    view.shutdown()
    plain = RemoteWebView(command=["host"], popen=_NoHost)
    assert "cache" not in plain._sticky["setup"][0]
    plain.shutdown()
    made = []
    monkeypatch.setattr(page_host, "enabled", lambda _page="face": True)
    monkeypatch.setattr(page_host, "RemoteWebView",
                        lambda **kwargs: made.append(kwargs) or SimpleNamespace(on_failed=None))
    owner = SimpleNamespace(_world_bridge=None, _on_js_error=lambda *_a: None)
    globe.GlobeView._remote_view(owner)
    assert made and made[0]["cache"].replace("\\", "/").endswith("web_cache/globe")
    app.processEvents()


def test_page_events_become_the_signals_the_globe_listens_to(monkeypatch):
    app = _qt()
    from orion_core.gui import page_host
    errors, opened, loaded, crashed = [], [], [], []
    view = page_host.RemoteWebView(command=["host"], popen=_NoHost,
                                   on_js_error=errors.append, on_open_url=opened.append)
    view.loadFinished.connect(loaded.append)
    view.renderProcessTerminated.connect(lambda status, code: crashed.append((status, code)))
    view._on_event("loaded", [True])
    view._on_event("crashed", ["AbnormalTerminationStatus", 5])
    view._on_event("console", ["WEBGL context lost @ globe:12"])
    for url in ("https://example.org/a", "javascript:alert(1)", "file:///C:/x"):
        view._on_event("open_url", [url])
    assert loaded == [True] and crashed == [("AbnormalTerminationStatus", 5)]
    assert errors == ["WEBGL context lost @ globe:12"]
    assert opened == ["https://example.org/a"], "only web links leave the page"
    view.setHtml("<p>x</p>", "https://cdn.example/base.html")
    view.page().runJavaScript("1+1")
    import time
    end = time.monotonic() + 5
    while time.monotonic() < end and view._proc.stdin.getvalue().count("\n") < 3:
        time.sleep(0.02)                         # the writer thread delivers them
    sent = [json.loads(line) for line in view._proc.stdin.getvalue().splitlines()]
    assert [m["m"] for m in sent] == ["setup", "set_html", "run_js"]
    assert view._sticky["set_html"] == ("<p>x</p>", "https://cdn.example/base.html")
    view.shutdown()
    app.processEvents()


def test_a_replaced_host_window_is_embedded_again_and_a_dead_one_removed(monkeypatch):
    """WebEngine's GPU surface makes Qt REPLACE the host's top-level window;
    the handle embedded first is then dead (11 Oct: face and globe were drawn
    off screen). Each announced handle is embedded, the old container goes."""
    app = _qt()
    from PyQt6.QtWidgets import QWidget
    from orion_core.gui import page_host
    made = []

    class FakeWindow:
        @staticmethod
        def fromWinId(wid):
            return ("window", wid)

    def container(window, parent):
        widget = QWidget(parent)
        made.append((window[1], widget))
        return widget
    monkeypatch.setattr(page_host, "QWindow", FakeWindow)
    monkeypatch.setattr(page_host.QWidget, "createWindowContainer", staticmethod(container))
    view = page_host.RemoteWebView(command=["host"], popen=_NoHost)
    view._on_message({"t": "wid", "wid": 101, "pid": 9, "part": "main"})
    view._on_message({"t": "wid", "wid": 101, "pid": 9, "part": "main"})      # same: no-op
    view._on_message({"t": "wid", "wid": 202, "pid": 9, "part": "main"})      # replaced
    assert [wid for wid, _w in made] == [101, 202]
    assert view.host_wids == {"main": 202} and view._layout.count() == 1
    view._drop_containers()                                                   # host died
    assert view._layout.count() == 0 and not view.host_wids
    view.shutdown()
    app.processEvents()


def test_a_retired_face_host_is_told_to_quit():
    """A rebuilt face left its old host process (~350 MB) running until ORION
    exited: deleting the widget does not stop the process."""
    app = _qt()
    from PyQt6.QtWidgets import QVBoxLayout, QWidget
    from orion_core.gui import face_host
    from orion_core.gui.core_window import OrionCoreWindow
    holder = QWidget()
    layout = QVBoxLayout(holder)
    face = face_host.RemoteFace(command=["host"], popen=_NoHost)
    plain = QWidget()
    for widget in (face, plain):
        layout.addWidget(widget)
    OrionCoreWindow._retire_face(face, layout)
    OrionCoreWindow._retire_face(plain, layout)         # no host: just removed
    assert face._closing and layout.count() == 0
    app.processEvents()


def test_shutdown_stops_every_page_host_first_and_nothing_shows_again():
    """12 Oct: "his face screen pops up for a split second right after
    shutting down". Taking ORION off screen now stops every page host first
    (hide + quit), and any window shown after that is hidden again."""
    app = _qt()
    from PyQt6.QtWidgets import QWidget
    from orion_core import app as orion_app
    from orion_core.gui import page_host
    face = page_host.RemoteWebView(command=["host"], popen=_NoHost)
    window = QWidget()
    window.show()
    app.processEvents()
    orion_app._take_off_screen(window)
    assert face._closing, "the page host was not asked to quit"
    assert not window.isVisible()
    stray = QWidget()
    stray.show()                         # something tries to come back on screen
    for _ in range(5):
        app.processEvents()
    assert not stray.isVisible(), "a window stayed on screen after shutdown"
    app.removeEventFilter(app._orion_stay_off_screen)
    del app._orion_stay_off_screen
    app.processEvents()


def test_a_failing_child_mode_writes_its_traceback_and_never_raises(capfd):
    """A windowed exe shows PyInstaller's "Unhandled exception in script"
    dialog for an uncaught error - from a BACKGROUND child, on the owner's
    screen (11 Oct 2026, a broken orion_world). The child modes report on
    stderr and exit 1 instead; their parent reads and logs the pipe."""
    import os
    import sys
    source = (ROOT / "orion.py").read_text(encoding="utf-8")
    namespace = {"os": os, "sys": sys}
    exec(compile(source[source.index("def _headless("):source.index("def _argv_after(")],
                 "orion.py", "exec"), namespace)

    def broken() -> int:
        raise ModuleNotFoundError("No module named 'orion_core.world_feeds'")
    assert namespace["_headless"](broken) == 1
    assert "No module named 'orion_core.world_feeds'" in capfd.readouterr().err
    assert namespace["_headless"](lambda: 0) == 0
    with pytest.raises(SystemExit):
        namespace["_headless"](lambda: sys.exit(3))
    for mode in ("--mcp-server", "--face-host", "--page-host", "--forge-verify"):
        branch = source[source.index(f'"{mode}" in sys.argv[1:]'):]
        assert branch.split("\n", 2)[1].strip().startswith("raise SystemExit(_headless("), mode


def test_a_brain_host_process_shows_both_instruments_and_quits_with_orion(tmp_path):
    """The real `orion.py --page-host brain`: two windows announced, events and
    a frame report answered, and it exits by itself when its stdin closes."""
    import os
    import sys
    import threading
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ORION_CONFIG_DIR=str(tmp_path))
    proc = subprocess.Popen([sys.executable, str(ROOT / "orion.py"), "--page-host", "brain"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                            bufsize=1, env=env)
    messages: list = []
    got = threading.Event()

    def read():
        for line in proc.stdout:
            if line.startswith("@@"):
                messages.append(json.loads(line[2:]))
                got.set()
    threading.Thread(target=read, daemon=True).start()
    try:
        deadline = 60.0
        import time
        end = time.monotonic() + deadline
        while time.monotonic() < end and sum(m.get("t") == "wid" for m in messages) < 2:
            got.wait(0.2)
            got.clear()
        parts = {m["part"] for m in messages if m.get("t") == "wid"}
        assert parts == {"iris", "lanes"}, messages
        for line in ({"m": "set_state", "a": ["PROCESSING"]},
                     {"m": "set_visible", "a": [True]},
                     {"m": "report", "a": [], "id": 7}):
            proc.stdin.write(json.dumps(line) + "\n")
        proc.stdin.flush()
        end = time.monotonic() + 20
        while time.monotonic() < end and not any(m.get("id") == 7 for m in messages):
            time.sleep(0.05)
        reply = next(m for m in messages if m.get("id") == 7)
        assert set(reply["value"]) >= {"frames", "fps", "hitches", "longest_ms"}
        proc.stdin.close()
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows messages")
def test_a_host_answers_the_pointer_itself_instead_of_waiting_on_orion():
    """An embedded host window is a child of ORION's; Windows hands a child's
    WM_SETCURSOR to the parent and waits, so with the pointer on the face every
    busy moment of ORION's thread stuttered the face (13 Oct: 37 in 21 s). The
    host answers it itself. Run in a child process through the same dispatcher
    call Qt's window procedure makes: a wrong 'not handled' answer is a fatal
    Python error there, which must fail this test, not end the test run."""
    import inspect
    import os
    import sys
    import textwrap
    from orion_core.gui import page_host
    probe = textwrap.dedent("""
        import sys
        from ctypes import addressof, wintypes
        from PyQt6 import sip
        from PyQt6.QtCore import QAbstractEventDispatcher
        from PyQt6.QtWidgets import QApplication
        app = QApplication([sys.argv[0]])
        from orion_core.gui import page_host
        own_cursor = page_host._cursor_filter()
        app.installNativeEventFilter(own_cursor)

        def answer(message, lparam):
            msg = wintypes.MSG(None, message, 0, lparam, 0, wintypes.POINT(0, 0))
            handled, result = QAbstractEventDispatcher.instance().filterNativeEvent(
                b"windows_generic_MSG", sip.voidptr(addressof(msg)))
            return int(bool(handled)), int(result) if result is not None else None

        moved = 0x0200 << 16
        print(answer(page_host.WM_SETCURSOR, moved | page_host.HTCLIENT),
              answer(page_host.WM_SETCURSOR, moved | 0xFFFE)[0],  # HTERROR: Windows' beep
              answer(0x000F, 0)[0])                               # WM_PAINT and the rest
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines()[0] == "(1, 1) 0 0", result.stdout
    assert "installNativeEventFilter(own_cursor)" in inspect.getsource(page_host.run_host)
