"""Mark XXXV regressions: fake native windows, OCR engines and media only."""

import asyncio
import ctypes
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from orion_core import agents, display, ocr_engine
from orion_core.control import AutonomousControlLayer
from orion_core.data import ToolResult
from orion_core.dispatch_desktop import DesktopDispatchMixin
from orion_core.dispatch_vision import VisionDispatchMixin
from orion_core.ocr_engine import OcrEngine
from orion_core.scene_analysis import _group_words
from orion_core.video_analysis import FrameNote, VideoAnalyser, VideoReport


def test_native_window_dispatch_leaves_event_loop_responsive():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        loop_thread = threading.get_ident()
        calls = []

        def native(*args):
            calls.append((threading.get_ident(), args))
            entered.set()
            assert release.wait(2), "event loop did not release native worker"
            return ToolResult("Native state confirmed")

        dispatcher = DesktopDispatchMixin()
        dispatcher.desktop = SimpleNamespace(window_control=native)
        task = asyncio.create_task(dispatcher.window_control(
            {"action": "move", "title": "Exact editor", "monitor": "second"}))
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(.001)
            assert entered.is_set() and not task.done()
            assert calls == [(calls[0][0], ("move", "Exact editor", "second"))]
            assert calls[0][0] != loop_thread
        finally:
            release.set()
        assert (await task).ok

    asyncio.run(scenario())


def test_cancelled_native_dispatch_never_replays_the_os_call():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        calls = []

        def native(*args):
            calls.append(args)
            entered.set()
            assert release.wait(2)
            return ToolResult("Completed native action")

        dispatcher = DesktopDispatchMixin()
        dispatcher.desktop = SimpleNamespace(window_control=native)
        task = asyncio.create_task(dispatcher.window_control({"action": "minimise", "title": "Editor"}))
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(.001)
            assert entered.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        # The caller is cancelled; an already-issued native operation can still
        # complete. It must never be automatically replayed or return success.
        await asyncio.to_thread(lambda: None)
        assert calls == [("minimise", "Editor", "")]

    asyncio.run(scenario())


@pytest.fixture
def fake_control(monkeypatch):
    # DPI changes are native operations too: this suite fakes even that seam.
    monkeypatch.setattr(display, "_physical_pixel_context", nullcontext)
    monkeypatch.setattr("orion_core.control.time.sleep", lambda _seconds: None)
    layer = AutonomousControlLayer.__new__(AutonomousControlLayer)
    layer._guard = lambda *args, **kwargs: None
    layer._resolve = lambda x, y, monitor: (x, y)
    layer.records = []
    layer._record = lambda *args, **kwargs: layer.records.append(args)
    return layer


class NativeWindow:
    def __init__(self, apply=True):
        self.title = "Exact editor"
        self.left, self.top, self.width, self.height = 10, 20, 640, 480
        self.isMinimized, self.isMaximized = False, False
        self.apply, self.calls = apply, []

    def resizeTo(self, width, height):
        self.calls.append(("resize", width, height))
        if self.apply:
            self.width, self.height = width, height

    def moveTo(self, left, top):
        self.calls.append(("move", left, top))
        if self.apply:
            self.left, self.top = left, top

    def minimize(self):
        self.calls.append(("minimise",))
        if self.apply:
            self.isMinimized = True

    def maximize(self):
        self.calls.append(("maximise",))
        if self.apply:
            self.isMaximized = True


@pytest.mark.parametrize("action,args,call", [
    ("resize_window", (900, 700), ("resize", 900, 700)),
    ("move_window", (-1200, 400), ("move", -1200, 400)),
    ("minimise_window", (), ("minimise",)),
    ("maximise_window", (), ("maximise",)),
])
def test_ignored_native_window_request_never_claims_success(fake_control, action, args, call):
    win = NativeWindow(apply=False)
    fake_control._find_window = lambda _title: win
    result = getattr(fake_control, action)("Exact editor", *args)
    assert not result.ok and "confirm" in result.text.lower()
    assert win.calls == [call], "uncertain OS mutation must never be replayed"
    assert not fake_control.records


@pytest.mark.parametrize("action,args", [
    ("resize_window", (900, 700)), ("move_window", (-1200, 400)),
    ("minimise_window", ()), ("maximise_window", ()),
])
def test_actual_native_window_readback_confirms_effect(fake_control, action, args):
    win = NativeWindow()
    fake_control._find_window = lambda _title: win
    result = getattr(fake_control, action)("Exact editor", *args)
    assert result.ok and len(win.calls) == 1 and len(fake_control.records) == 1


def test_resize_reports_applied_minimum_geometry(fake_control):
    win = NativeWindow()
    fake_control._find_window = lambda _title: win
    result = fake_control.resize_window("Exact editor", 10, 20)
    assert result.ok and "120x80" in result.text
    assert win.calls == [("resize", 120, 80)]


def test_unreadable_geometry_never_reports_confirmed_move(fake_control):
    calls = []
    win = SimpleNamespace(title="Exact editor", moveTo=lambda *xy: calls.append(xy))
    fake_control._find_window = lambda _title: win
    result = fake_control.move_window("Exact editor", 200, 400)
    assert not result.ok and "confirm" in result.text.lower()
    assert calls == [(200, 400)]


def test_size_constrained_by_application_is_reported_as_actual_geometry(fake_control):
    win = NativeWindow()

    def constrained(width, height):
        win.calls.append((width, height))
        win.width, win.height = 1000, 720

    win.resizeTo = constrained
    fake_control._find_window = lambda _title: win
    result = fake_control.resize_window("Exact editor", 900, 700)
    assert not result.ok and "1000x720" in result.text and "900x700" in result.text
    assert win.calls == [(900, 700)]


def test_delayed_readback_is_polled_without_reissuing_input(fake_control):
    win = NativeWindow()
    fake_control._find_window = lambda _title: win
    states = iter([(10, 20, 640, 480), (-1200, 400, 640, 480)])
    fake_control._window_region = lambda _win: next(states)
    assert fake_control.move_window("Exact editor", -1200, 400).ok
    assert win.calls == [("move", -1200, 400)]


@pytest.mark.parametrize("action,args", [
    ("focus_window", {}), ("resize_window", {"width": 900, "height": 700}),
    ("move_window", {"x": 100, "y": 200}), ("minimise_window", {}), ("maximise_window", {}),
])
def test_visual_check_cannot_replay_a_verified_window_operation(action, args):
    async def scenario():
        dispatcher = DesktopDispatchMixin()
        dispatcher.control = SimpleNamespace(**{action: lambda *_: ToolResult("Native state confirmed")})
        dispatcher.verifier = object()
        checks = []

        async def check(call, verify, **kwargs):
            checks.append((verify, kwargs))
            return call()

        dispatcher._maybe_verify = check
        result = await dispatcher.desktop_control({"action": action, "title": "Editor", **args})
        assert result.ok and checks == [(True, {"repeatable": False})]

    asyncio.run(scenario())


@pytest.mark.parametrize("action,state", [("minimise", "IsIconic"), ("maximise", "IsZoomed")])
@pytest.mark.parametrize("changed", [False, True])
def test_native_window_control_verifies_minimise_and_maximise(monkeypatch, action, state, changed):
    calls = []
    title = "Exact editor"

    def get_text(_hwnd, buffer, _length):
        buffer.value = title

    native = SimpleNamespace(
        IsWindowVisible=lambda _h: True,
        GetWindowTextLengthW=lambda _h: len(title), GetWindowTextW=get_text,
        EnumWindows=lambda visitor, value: visitor(42, value),
        ShowWindow=lambda *args: calls.append(args),
        IsIconic=lambda _h: changed if state == "IsIconic" else False,
        IsZoomed=lambda _h: changed if state == "IsZoomed" else False)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=native))
    monkeypatch.setattr(agents.time, "sleep", lambda _seconds: None)
    agent = agents.DesktopAgent.__new__(agents.DesktopAgent)
    result = agent.window_control(action, title)
    assert result.ok is changed
    assert calls == [(42, 6 if action == "minimise" else 3)]
    if not changed:
        assert "confirm" in result.text.lower()


@pytest.mark.parametrize("actual_geometry,expected", [
    ((1100, 190, 1740, 670), True),
    ((1000, 190, 1640, 670), False),  # Right monitor, wrong requested position.
    (None, False),  # Right monitor, unreadable geometry.
])
def test_native_screen_move_requires_actual_geometry(monkeypatch, actual_geometry, expected):
    calls = []
    current = {"monitor": 1, "rect": (10, 20, 650, 500)}

    def get_rect(_hwnd, pointer):
        if current["rect"] is None:
            return False
        rect = pointer._obj
        rect.left, rect.top, rect.right, rect.bottom = current["rect"]
        return True

    def move(*args):
        calls.append(args)
        current.update(monitor=2, rect=actual_geometry)
        return True

    native = SimpleNamespace(MonitorFromWindow=lambda *_: current["monitor"],
        IsZoomed=lambda _h: False, IsIconic=lambda _h: False,
        GetWindowRect=get_rect, SetWindowPos=move)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=native))
    monkeypatch.setattr(agents.time, "sleep", lambda _seconds: None)
    from orion_core import screen_text
    monkeypatch.setattr(screen_text, "list_windows", lambda: [SimpleNamespace(hwnd=42, title="Editor", pid=2)])
    agent = agents.DesktopAgent.__new__(agents.DesktopAgent)
    agent._protected_pids = lambda: set()
    agent._pid_of = lambda _hwnd: 2
    agent._screens = lambda: [
        {"handle": 1, "left": 0, "top": 0, "right": 840, "bottom": 860, "primary": True},
        {"handle": 2, "left": 1000, "top": 0, "right": 1840, "bottom": 860, "primary": False}]
    result = agent._move_to_screen("editor", "second")
    assert result.ok is expected and len(calls) == 1
    assert calls[0] == (42, 0, 1100, 190, 640, 480, 0x0014)
    if not expected:
        assert "confirm" in result.text.lower()


def test_minimise_all_reports_only_confirmed_effects(monkeypatch):
    iconic, calls = set(), []

    def minimize(hwnd, _state):
        calls.append(hwnd)
        if hwnd == 1:
            iconic.add(hwnd)

    def process_id(hwnd, pointer):
        pointer._obj.value = 2

    native = SimpleNamespace(GetWindowThreadProcessId=process_id,
        GetClassNameW=lambda _h, buffer, _n: setattr(buffer, "value", "Editor"),
        IsIconic=lambda hwnd: hwnd in iconic, ShowWindow=minimize)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=native))
    monkeypatch.setattr(agents.time, "sleep", lambda _seconds: None)
    agent = agents.DesktopAgent.__new__(agents.DesktopAgent)
    agent._protected_pids = lambda: set()
    result = agent._minimise_all([(1, "One"), (2, "Two")])
    assert not result.ok and "Minimised 1" in result.text and "Two" in result.text
    assert calls == [1, 2]


class Bus:
    log = SimpleNamespace(emit=lambda *args: None)


def test_windows_ocr_unknown_score_preserves_native_first_latency(monkeypatch):
    monkeypatch.setattr(ocr_engine, "_winrt_modules", lambda: (object(),) * 4)
    engine = OcrEngine(Bus())
    monkeypatch.setattr(engine, "_run_winrt", lambda _raw: "three wrong words")
    winrt = engine._probe_winrt()
    assert winrt(None, None) == ("three wrong words", None)
    engine._engines = [("Windows OCR", winrt), ("measured", lambda *_:
        pytest.fail("usable native text must not pay for heavier OCR"))]
    monkeypatch.setattr(engine, "_detect", lambda: None)
    monkeypatch.setattr(engine, "_preprocess", lambda image: image)
    result = engine.image_to_result(object())
    assert result.text == "three wrong words"
    assert result.confidence is None and result.confidence_source == "unavailable"


@pytest.mark.parametrize("failed", [False, True])
def test_empty_or_failed_native_ocr_still_uses_scored_fallback(monkeypatch, failed):
    def native(*_args):
        if failed:
            raise RuntimeError("native reader unavailable")
        return "", None

    engine = OcrEngine(Bus())
    engine._engines = [("Windows OCR", native), ("measured", lambda *_: ("fallback words", .94))]
    monkeypatch.setattr(engine, "_detect", lambda: None)
    monkeypatch.setattr(engine, "_preprocess", lambda image: image)
    result = engine.image_to_result(object())
    assert result.text == "fallback words" and result.confidence == .94
    assert result.confidence_source == "engine"


def test_unknown_ocr_confidence_keeps_native_text_when_other_readers_fail(monkeypatch):
    engine = OcrEngine(Bus())
    engine._engines = [("Windows OCR", lambda *_: ("native useful text", None)),
                       ("low-scoring", lambda *_: ("bad", .1))]
    monkeypatch.setattr(engine, "_detect", lambda: None)
    monkeypatch.setattr(engine, "_preprocess", lambda image: image)
    result = engine.image_to_result(object())
    assert result.text == "native useful text" and result.confidence is None
    assert result.confidence_source == "unavailable"
    assert "unavailable" in result.report().lower() and "85" not in result.report()


def test_winrt_word_boxes_do_not_invent_a_confidence(monkeypatch):
    async def done(value=None):
        return value

    word = SimpleNamespace(text="Submit", bounding_rect=SimpleNamespace(x=2, y=4, width=8, height=6))
    result = SimpleNamespace(text_angle=0, lines=[SimpleNamespace(words=[word])])
    bitmap = SimpleNamespace(pixel_width=40, pixel_height=40)
    native = SimpleNamespace(max_image_dimension=100,
        try_create_from_user_profile_languages=lambda: SimpleNamespace(
            recognize_async=lambda _bitmap: done(result)))
    decoder = SimpleNamespace(create_async=lambda _stream: done(SimpleNamespace(
        get_software_bitmap_async=lambda: done(bitmap))))
    stream = lambda: SimpleNamespace(get_output_stream_at=lambda _offset: object())
    writer = lambda _stream: SimpleNamespace(write_bytes=lambda _data: None,
        store_async=lambda: done(), flush_async=lambda: done())
    monkeypatch.setattr(ocr_engine, "_winrt_modules", lambda: (native, decoder, stream, writer))
    words = ocr_engine._winrt_recognise(Image.new("RGB", (20, 20)))
    assert words[0][0]["text"] == "Submit" and words[0][0]["rect"] == (1, 2, 4, 3)
    assert words[0][0]["confidence"] is None
    assert words[0][0]["confidence_source"] == "unavailable"
    lines = _group_words(words[0], 20, 20)
    assert lines[0]["text"] == "Submit" and lines[0]["confidence_source"] == "unavailable"


def test_scene_unknown_confidence_is_not_a_zero_score():
    lines = _group_words([
        {"text": "Native", "rect": (1, 2, 20, 10), "confidence": None,
         "confidence_source": "unavailable"},
        {"text": "garbage", "rect": (50, 2, 20, 10), "confidence": .1},
    ], 100, 50)
    assert [line["text"] for line in lines] == ["Native"]


def test_missing_rapidocr_score_is_not_replaced_with_sixty_percent(monkeypatch):
    box = [(1, 2), (11, 2), (11, 8), (1, 8)]
    reader = lambda _image: ([[box, "unscored text"]], None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=lambda: reader))
    engine = OcrEngine(Bus())
    engine._rapid_ready = lambda array: (array, 0)
    engine._rgb_ndarray = lambda image, **kw: image
    adapter = engine._probe_rapidocr()
    assert adapter(None, object()) == ("unscored text", None)
    word = engine._locate_words_rapidocr(object())[0]
    assert word["confidence"] is None and word["confidence_source"] == "unavailable"


@pytest.mark.parametrize("cached", [False, True])
def test_easyocr_probe_explicitly_forbids_model_downloads(monkeypatch, cached):
    calls = []

    def reader(languages, *, gpu=False, verbose=False, download_enabled=True):
        calls.append((languages, download_enabled))
        assert download_enabled is False, "model lookup could download unexpectedly"
        if not cached:
            raise FileNotFoundError("cached model missing")
        return SimpleNamespace(readtext=lambda *_a, **_k: [(None, "local words", .92)])

    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=reader))
    engine = OcrEngine(Bus())
    engine._rgb_ndarray = lambda raw: raw
    run = engine._probe_easyocr()
    assert calls == [(["en"], False)]
    if cached:
        assert run(None, object()) == ("local words", .92)
    else:
        assert run is None


@pytest.mark.parametrize("cached", [False, True])
def test_paddleocr_legacy_probe_never_constructs_without_all_local_models(monkeypatch, tmp_path, cached):
    from importlib import metadata
    monkeypatch.setattr(metadata, "version", lambda package: "2.10.0")
    calls = []

    class Paddle:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            for kind in ("det", "rec", "cls"):
                directory = Path(kwargs[f"{kind}_model_dir"])
                assert all((directory / name).read_bytes() == b"local model"
                           for name in ("inference.pdmodel", "inference.pdiparams"))

        def ocr(self, _image, cls=True):
            return [[(None, ("cached paddle text", .9))]]

    Paddle.__module__ = "paddleocr.test_implementation"
    implementation = SimpleNamespace(BASE_DIR=str(tmp_path), DEFAULT_OCR_MODEL_VERSION="PP-OCRv4",
        get_model_config=lambda _domain, _version, kind, _language: {"url": f"https://invalid.test/{kind}_model.tar"})
    monkeypatch.setitem(sys.modules, "paddleocr.test_implementation", implementation)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=Paddle))
    if cached:
        for kind, language in (("det", "en"), ("rec", "en"), ("cls", "")):
            directory = tmp_path / "whl" / kind / language / f"{kind}_model"
            directory.mkdir(parents=True)
            for name in ("inference.pdmodel", "inference.pdiparams"):
                (directory / name).write_bytes(b"local model")
    engine = OcrEngine(Bus())
    engine._rgb_ndarray = lambda raw: raw
    run = engine._probe_paddleocr()
    if cached:
        assert len(calls) == 1 and run(None, object()) == ("cached paddle text", .9)
    else:
        assert run is None and calls == []


def test_unverified_paddle_api_is_not_initialized_by_detection(monkeypatch):
    from importlib import metadata
    monkeypatch.setattr(metadata, "version", lambda _package: "3.3.0")
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(
        PaddleOCR=lambda **kw: pytest.fail("unverified API could auto-download")))
    assert OcrEngine(Bus())._probe_paddleocr() is None


def test_failed_transcriber_readiness_is_not_silence(tmp_path):
    wav = tmp_path / "decoded.wav"
    wav.write_bytes(b"fake decoded audio")
    transcriber = SimpleNamespace(ensure_ready=lambda: False,
        transcribe_wav=lambda _path: pytest.fail("unready recognizer was invoked"))
    analyser = VideoAnalyser(transcriber=transcriber)
    analyser._extract_audio = lambda _source: (wav, "")
    transcript, note = asyncio.run(analyser._transcribe(Path("fake.mp4")))
    assert transcript == "" and "unavailable" in note
    assert "no discernible speech" not in note and not wav.exists()


@pytest.mark.parametrize("answer,raises,expected_note", [
    ("narration", False, ""), ("   ", False, "audio had no discernible speech"),
    ("", True, "transcription failed"),
])
def test_ready_transcriber_result_and_failure_are_distinct(tmp_path, answer, raises, expected_note):
    wav = tmp_path / "decoded.wav"
    wav.write_bytes(b"fake decoded audio")

    def recognize(_path):
        if raises:
            raise RuntimeError("recognizer crashed")
        return answer

    analyser = VideoAnalyser(transcriber=SimpleNamespace(ensure_ready=lambda: True, transcribe_wav=recognize))
    analyser._extract_audio = lambda _source: (wav, "")
    transcript, note = asyncio.run(analyser._transcribe(Path("fake.mp4")))
    assert transcript == ("" if raises else answer.strip())
    assert expected_note in note and not wav.exists()


def test_missing_decoder_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    path, note = VideoAnalyser()._extract_audio(Path("fake.mp4"))
    assert path is None and "PyAV not installed" in note


def test_decode_failure_removes_its_partial_wav_and_preserves_source(monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"unchanged original")
    closed = []

    def decode(**_kwargs):
        raise RuntimeError("decoder failed midway")

    container = SimpleNamespace(streams=[SimpleNamespace(type="audio")], decode=decode,
        close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: container,
        AudioResampler=lambda **kw: object()))
    monkeypatch.setattr("orion_core.video_analysis.tempfile.gettempdir", lambda: str(tmp_path))
    path, note = VideoAnalyser()._extract_audio(source)
    assert path is None and "audio extraction failed" in note and closed == [True]
    assert source.read_bytes() == b"unchanged original"
    assert not list(tmp_path.glob("orion_vaudio_*.wav"))


@pytest.mark.parametrize("phase", ["extract", "recognize"])
def test_cancelled_video_audio_owns_late_temporary_cleanup(monkeypatch, tmp_path, phase):
    async def scenario():
        entered, release, cleaned, file_open = (threading.Event() for _ in range(4))
        source, wav = tmp_path / "original.mp4", tmp_path / "owned.wav"
        source.write_bytes(b"preserved source")
        real_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == wav and file_open.is_set():
                raise PermissionError("Windows recognizer still holds the WAV handle")
            result = real_unlink(path, *args, **kwargs)
            if path == wav:
                cleaned.set()
            return result

        def extract(_source):
            if phase == "extract":
                entered.set()
                assert release.wait(2)
            wav.write_bytes(b"owned decoded audio")
            return wav, ""

        def recognize(_path):
            if phase == "recognize":
                file_open.set()
                entered.set()
                try:
                    assert release.wait(2)
                finally:
                    file_open.clear()
            return "late transcript must be discarded"

        monkeypatch.setattr(Path, "unlink", unlink)
        analyser = VideoAnalyser(transcriber=SimpleNamespace(ensure_ready=lambda: True,
                                                             transcribe_wav=recognize))
        analyser._extract_audio = extract
        task = asyncio.create_task(analyser._transcribe(source))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, .25)
            assert not release.is_set(), "cancel must return before native/media work settles"
        finally:
            release.set()
        assert await asyncio.to_thread(cleaned.wait, 2)
        assert not wav.exists() and source.read_bytes() == b"preserved source"

    asyncio.run(scenario())


@pytest.mark.parametrize("note,expected", [
    ("no transcriber available", "unavailable"),
    ("PyAV not installed — cannot read the audio track", "unavailable"),
    ("transcription failed (recognizer crashed)", "unavailable"),
    ("the video has no audio track", "no_audio"),
    ("the audio track was empty", "no_audio"),
    ("audio had no discernible speech", "no_speech"),
])
def test_video_pipeline_preserves_audio_evidence(tmp_path, note, expected):
    source = tmp_path / "fake.mp4"
    source.write_bytes(b"not a real clip; extractors below are fakes")
    prompts = []

    async def synthesize(prompt):
        prompts.append(prompt)
        return None, "SUMMARY: visible slide\nACTION: Read the slide"

    analyser = VideoAnalyser(router=SimpleNamespace(generate_text=synthesize))
    analyser._extract_keyframes = lambda *_: ([], 12)

    async def describe(*_):
        return [FrameNote(1, "visible slide")]

    async def transcribe(*_):
        return "", note

    analyser._describe_frames, analyser._transcribe = describe, transcribe
    report = asyncio.run(analyser.analyse(str(source)))
    assert report.audio_state == expected and report.audio_note == note
    assert note in prompts[0] and note in report.announcement()
    if expected == "unavailable":
        assert "no speech" not in prompts[0].lower()
        assert "no speech" not in report.announcement().lower()
    assert "sampled" in report.announcement().lower()


def test_legacy_report_without_audio_evidence_cannot_assert_silence():
    said = VideoReport(path="clip.mp4", frames=[FrameNote(0, "slide")]).announcement()
    assert "unavailable" in said and "no speech" not in said


def test_no_video_evidence_never_calls_a_synthesis_provider():
    async def forbidden(*args, **kwargs):
        pytest.fail("no observed frames/transcript can support model synthesis")

    report = VideoReport(path="bad.mp4", notes=["decoder unavailable"])
    asyncio.run(VideoAnalyser(router=SimpleNamespace(generate_text=forbidden))._synthesise(report, "summarise"))
    assert "could not be read" in report.summary
    assert "could not analyse" in report.announcement()


@pytest.mark.parametrize("frames,transcript,expected", [
    ([], "", False), ([FrameNote(0, "a slide")], "", True), ([], "narration", True),
])
def test_video_tool_success_requires_actual_content_evidence(monkeypatch, frames, transcript, expected):
    report = VideoReport(path="fake.mp4", frames=frames, transcript=transcript,
                         audio_state="transcribed" if transcript else "unavailable",
                         audio_note="" if transcript else "decoder unavailable")

    async def analyse(*args, **kwargs):
        return report

    monkeypatch.setattr(VideoAnalyser, "analyse", analyse)
    dispatcher = VisionDispatchMixin()
    dispatcher.vision = object()
    spoken = []
    dispatcher.bus = SimpleNamespace(speak_request=SimpleNamespace(emit=spoken.append))
    dispatcher._video_api_key = lambda _router: ""
    result = asyncio.run(dispatcher._analyse_video("fake.mp4", "summarise", {}))
    assert result.ok is expected
    assert result.receipt == {"analysis_state": "completed" if expected else "failed",
                              "frame_count": len(frames), "audio_state": report.audio_state,
                              "audio_note": report.audio_note}
    assert spoken == [report.announcement()]
    if not expected:
        assert "could not analyse" in result.text

