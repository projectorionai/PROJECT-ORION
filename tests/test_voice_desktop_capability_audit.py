"""Offline reproductions from the 15 October voice/desktop capability audit."""

import asyncio
import ctypes
import sys
from threading import RLock
from types import SimpleNamespace

import pytest
from test_control_layer import _FakeKB, _FakePG, _StubBus, _StubDisplay

from orion_core.agents import DesktopAgent
from orion_core.control import AutonomousControlLayer
from orion_core.live_worker import GenAILiveWorker


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", _FakePG())
    monkeypatch.setitem(__import__("sys").modules, "keyboard", _FakeKB())
    monkeypatch.setenv("ORION_AUTONOMY", "1")
    value = AutonomousControlLayer(_StubBus(), _StubDisplay())
    value._windows = lambda: SimpleNamespace(getAllWindows=lambda: [
        SimpleNamespace(title="Report - Word"), SimpleNamespace(title="Report - Notepad")])
    return value


@pytest.mark.parametrize("action,args", [
    ("focus_window", ()), ("resize_window", (700, 500)), ("move_window", (200, 100)),
    ("minimise_window", ()), ("maximise_window", ()),
])
def test_ambiguous_title_never_mutates_an_arbitrary_window(control, action, args):
    result = getattr(control, action)("Report", *args)
    assert not result.ok and "Several windows" in result.text
    assert "Report - Word" in result.text and "Report - Notepad" in result.text


def test_ambiguous_title_never_uses_another_launch_or_typing_fallback(control, monkeypatch):
    control.desktop = SimpleNamespace(window_control=lambda *_: pytest.fail("ambiguity fallback"))
    monkeypatch.setattr(control, "_narrate_typing", lambda *_a, **_k: None)
    monkeypatch.setattr(control, "_edit_via_uia", lambda *_: pytest.fail("wrote into ambiguous editor"))
    assert not control.focus_window("Report").ok
    assert not control.edit_text("replacement", title="Report", replace=True).ok


def test_one_exact_title_wins_over_a_longer_partial_title(control):
    windows = [SimpleNamespace(title="Report - Word"), SimpleNamespace(title="Report")]
    control._windows = lambda: SimpleNamespace(getAllWindows=lambda: windows)
    assert control._find_window("Report") is windows[1]
    assert control._find_window("Word") is windows[0]


def test_uia_named_editor_refuses_ambiguous_partial_titles(control):
    windows = [SimpleNamespace(window_text=lambda title=title: title)
               for title in ("Report - Word", "Report - Notepad")]
    desktop = SimpleNamespace(windows=lambda: windows)
    assert control._uia_window(desktop, "Report") is None
    assert control._uia_window(desktop, "Report - Word") is windows[0]


@pytest.fixture
def native_windows(monkeypatch):
    titles, calls = {1: "Report - Word", 2: "Report - Notepad"}, []

    def text(hwnd, buffer, _length):
        buffer.value = titles[hwnd]

    user32 = SimpleNamespace(
        IsWindowVisible=lambda _h: True,
        GetWindowTextLengthW=lambda hwnd: len(titles[hwnd]), GetWindowTextW=text,
        EnumWindows=lambda visit, value: [visit(handle, value) for handle in titles],
        PostMessageW=lambda *args: calls.append(("close", *args)),
        ShowWindow=lambda *args: calls.append(("show", *args)))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=user32))
    monkeypatch.setattr("orion_core.agents.force_foreground", lambda hwnd: calls.append(("focus", hwnd)) or True)
    agent = DesktopAgent.__new__(DesktopAgent)
    agent._protected_pids = lambda: set()
    return agent, titles, calls


@pytest.mark.parametrize("action", ["focus", "close", "minimise", "maximise"])
def test_native_window_tools_show_candidates_before_any_ambiguous_action(native_windows, action):
    agent, _titles, calls = native_windows
    result = agent.window_control(action, "Report")
    assert not result.ok and "Several windows" in result.text
    assert calls == []


def test_native_exact_window_selection_preserves_its_handle(native_windows):
    agent, titles, calls = native_windows
    titles[2] = "Report"
    assert agent.window_control("focus", "Report").ok
    assert calls == [("focus", 2)]


def fallback_worker():
    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.stop_event = asyncio.Event()
    worker.paused, worker.connected, worker.tool_busy = False, False, False
    worker._turn_active = False
    worker.mic, worker.local_brain = None, None
    worker.bus = _StubBus()
    worker.speech = SimpleNamespace(interrupt_all=lambda: None, output_active=lambda: False)
    worker._refresh_wake_window = lambda: None
    worker._offline_voice_ready = lambda: True
    worker._emit_state = lambda *_: None
    worker._persist_episode = lambda *_: None
    worker.said = []
    worker._say = worker.said.append
    return worker


@pytest.mark.parametrize("provider_swallows_cancel", [False, True])
def test_cancelled_fallback_cannot_speak_and_a_later_request_still_works(provider_swallows_cancel):
    async def scenario():
        worker = fallback_worker()
        entered = asyncio.Event()
        profile = SimpleNamespace(name="offline-test")

        async def generate(*_args):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not provider_swallows_cancel:
                    raise
            return profile, "Late cancelled answer"

        worker.router = SimpleNamespace(has_text_fallback=lambda: True, generate_text=generate)
        task = asyncio.create_task(worker._submit_text_fallback("old request"))
        await entered.wait()
        worker.cancel_turn()
        await asyncio.gather(task, return_exceptions=True)
        assert worker.said == [] and not worker._text_fallback_tasks

        async def answer(*_args):
            return profile, "New requested answer"

        worker.router.generate_text = answer
        await worker._submit_text_fallback("new request")
        assert worker.said == ["New requested answer"]

    asyncio.run(scenario())


def test_cancelled_text_tool_loop_cannot_start_a_later_action(monkeypatch):
    from orion_core import text_tools

    async def scenario():
        worker = fallback_worker()
        entered = asyncio.Event()
        worker.router = SimpleNamespace(has_text_fallback=lambda: True)
        worker.dispatcher = SimpleNamespace(dispatch_chain=lambda *_: None, TOOL_DECLARATIONS=[{}])

        async def model_then_action(_request, _declarations, _generate, act, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            await act("open_app", {"app_name": "Notepad"})
            pytest.fail("a cancelled tool turn continued")

        async def forbidden(*_args):
            pytest.fail("a tool ran after its request was cancelled")

        monkeypatch.setattr(text_tools, "run", model_then_action)
        worker._text_path_dispatch = forbidden
        task = asyncio.create_task(worker._submit_text_fallback("open Notepad"))
        await entered.wait()
        worker.cancel_turn()
        outcomes = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert worker.said == [] and not worker._text_fallback_tasks

    asyncio.run(scenario())


@pytest.mark.parametrize("cached", [False, True])
def test_offline_and_startup_vosk_paths_never_use_the_network_downloader(tmp_path, monkeypatch, cached):
    from orion_core.audio import LocalSpeechRecogniser
    from orion_core.speech_offline import OfflineTranscriber

    model_path = tmp_path / "vosk-model-small-en-us-0.15"
    if cached:
        (model_path / "am").mkdir(parents=True)
        (model_path / "conf").mkdir()
        (model_path / "am" / "final.mdl").write_bytes(b"synthetic")
        (model_path / "conf" / "mfcc.conf").write_text("synthetic", encoding="utf-8")
    calls = []

    def model(*args, **kwargs):
        assert not kwargs, "Vosk Model(lang=...) can download over the network"
        assert args == (str(model_path),)
        calls.append(args[0])
        return object()

    monkeypatch.delenv("ORION_VOSK_MODEL", raising=False)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "whisper", None)
    monkeypatch.setitem(sys.modules, "vosk", SimpleNamespace(
        MODEL_DIRS=[None, tmp_path], Model=model, SetLogLevel=lambda *_: None,
        KaldiRecognizer=lambda *_: SimpleNamespace(SetWords=lambda *_: None)))
    transcriber = OfflineTranscriber(defer=True)
    transcriber._select_engine()
    assert transcriber.engine == ("vosk" if cached else "")
    recogniser = LocalSpeechRecogniser.__new__(LocalSpeechRecogniser)
    recogniser.bus = _StubBus()
    recogniser._lock = RLock()
    recogniser._sample_rate = 16000
    recogniser.available = False
    recogniser._initialise()
    assert recogniser.available is cached
    assert calls == ([str(model_path)] * 2 if cached else [])
    if not cached:
        assert "ORION_VOSK_MODEL" in recogniser.bus.log.emitted[-1][0]


def test_explicit_vosk_model_is_honoured_and_invalid_config_does_not_download(tmp_path, monkeypatch):
    from orion_core.vosk_models import cached_vosk_model
    path = tmp_path / "my chosen model"
    (path / "am").mkdir(parents=True)
    (path / "conf").mkdir()
    (path / "am" / "final.mdl").write_bytes(b"synthetic")
    (path / "conf" / "mfcc.conf").write_text("synthetic", encoding="utf-8")
    monkeypatch.setenv("ORION_VOSK_MODEL", str(path))
    assert cached_vosk_model(SimpleNamespace(MODEL_DIRS=[])) == str(path)
    monkeypatch.setenv("ORION_VOSK_MODEL", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="complete local"):
        cached_vosk_model(SimpleNamespace(MODEL_DIRS=[]))
