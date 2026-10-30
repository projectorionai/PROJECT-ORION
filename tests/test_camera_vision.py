"""
Camera background-identification tests (new ask).

ORION should be able to look through the webcam and identify what's in view —
the person and, especially, the background.  These tests cover the dispatch
routing and the text-shaping, with the actual capture stubbed (no hardware).
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher


class _Sig:
    def emit(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


class _FakeVision:
    """Stands in for VisionAgent; records the camera call."""

    def __init__(self):
        self.calls = []

    async def analyse_camera(self, prompt="", camera_index=0):
        self.calls.append((prompt, camera_index))
        return ToolResult(
            f"camera[{camera_index}] {prompt}",
            media={"data": b"jpeg", "mime_type": "image/jpeg"})

    async def analyse_screen(self, prompt=""):
        return ToolResult("screen")


def _dispatcher_with_vision(vision):
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _Bus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    d.vision = vision
    return d


def test_camera_action_routes_to_analyse_camera():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    result = asyncio.run(d.dispatch("vision_analyse", {"action": "camera"}))
    assert result.ok
    # None, not 0: no index named means "the remembered camera", which the
    # vision agent resolves (and can borrow from the face tracker). A forced 0
    # opened the wrong device and refused the tracker's frame.
    assert vision.calls == [("", None)]
    assert result.media and result.media["mime_type"] == "image/jpeg"


def test_camera_index_is_passed_through():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    asyncio.run(d.dispatch("vision_analyse",
                            {"action": "background", "camera_index": 2, "prompt": "what's behind me"}))
    assert vision.calls == [("what's behind me", 2)]


def test_bad_camera_index_falls_back_to_the_remembered_camera():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    asyncio.run(d.dispatch("vision_analyse", {"action": "webcam", "camera_index": "oops"}))
    assert vision.calls == [("", None)]


def test_camera_index_zero_is_still_honoured_when_named():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    asyncio.run(d.dispatch("vision_analyse", {"action": "camera", "camera_index": 0}))
    assert vision.calls == [("", 0)]


def test_analyse_camera_shapes_identification_prompt(monkeypatch):
    from orion_core.vision import VisionAgent

    agent = VisionAgent.__new__(VisionAgent)

    async def _fake_capture(camera_index=0, max_side=768):
        return ToolResult("raw frame", media={"data": b"x", "mime_type": "image/jpeg"})

    agent.capture_live_frame = _fake_capture           # type: ignore
    result = asyncio.run(agent.analyse_camera())
    assert result.ok
    assert "background" in result.text.lower()
    assert result.media["mime_type"] == "image/jpeg"


def test_analyse_camera_propagates_capture_failure():
    from orion_core.vision import VisionAgent

    agent = VisionAgent.__new__(VisionAgent)

    async def _fail(camera_index=0, max_side=768):
        return ToolResult("no camera", ok=False)

    agent.capture_live_frame = _fail                   # type: ignore
    result = asyncio.run(agent.analyse_camera())
    assert not result.ok


def _bare_vision():
    from orion_core.vision import VisionAgent
    agent = VisionAgent.__new__(VisionAgent)
    agent.bus = _Bus()
    agent._live_frame_source = None
    return agent


def test_capture_borrows_tracker_frame_and_never_opens_a_second_camera(monkeypatch):
    """When a live frame source (the face tracker) is available, on-demand
    capture must BORROW its frame — never open a second handle on the same
    webcam (the -1072873821 MSMF grab failure on Windows)."""
    import cv2
    import numpy as np

    def _boom(*_a, **_k):
        raise AssertionError("VideoCapture must not be opened when a frame is borrowable")

    # Patched on the cv2 module itself rather than through orion_core.vision:
    # vision.py now imports cv2 lazily inside the worker methods (so importing
    # it on every app start doesn't drag OpenCV in), and sys.modules caching
    # means those local imports resolve to this very object.
    monkeypatch.setattr(cv2, "VideoCapture", _boom)

    agent = _bare_vision()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    agent._live_frame_source = lambda: frame

    result = agent._capture_live_frame_sync(0, 640)
    assert result.ok
    assert result.media["mime_type"] == "image/jpeg"
    assert result.media["data"]


def test_capture_falls_back_to_camera_when_no_source(monkeypatch):
    """With no borrowable frame, capture opens the camera itself (DirectShow
    path) and warms up before grabbing."""
    import numpy as np

    agent = _bare_vision()

    class _FakeCap:
        def __init__(self):
            self.reads = 0

        def read(self):
            self.reads += 1
            if self.reads < 2:          # first frame cold/black → dropped
                return False, None
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(agent, "_open_camera", lambda idx: (_FakeCap(), "CAP_DSHOW"))
    result = agent._capture_live_frame_sync(0, 640)
    assert result.ok
    assert result.media["mime_type"] == "image/jpeg"


def test_capture_reports_busy_camera_clearly(monkeypatch):
    """If the camera can't be opened at all, the message must name the likely
    cause (in use / privacy) rather than a bare failure."""
    agent = _bare_vision()
    monkeypatch.setattr(agent, "_open_camera", lambda idx: (None, ""))
    result = agent._capture_live_frame_sync(0, 640)
    assert not result.ok
    assert "in use" in result.text.lower()


# Detailed scene analysis and text memory.
"""Detailed scene memory uses synthetic pixels and mocked providers/instruments."""



import asyncio
import io
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from orion_core import scene_analysis as scenes
from orion_core.data import ToolResult


class _SceneSignal:
    def emit(self, *_args):
        pass

    def connect(self, *_args):
        pass


class _SceneBus:
    def __getattr__(self, _name):
        return _SceneSignal()


@pytest.fixture
def _scene_no_models(monkeypatch):
    from orion_core import semantic
    monkeypatch.setattr(semantic.ENCODER, "ready", lambda: False)
    monkeypatch.setenv("ORION_SCENE_MEMORY", "1")
    monkeypatch.setenv("ORION_SCENE_CLOUD", "1")
    monkeypatch.delenv("ORION_CAMERA_HISTORY", raising=False)


@pytest.fixture
def scene_matrix(tmp_path):
    from orion_core.memory import OrionMemoryMatrix
    memory = OrionMemoryMatrix(tmp_path / "memory.db", tmp_path / "config", _SceneBus())
    yield memory
    memory.close()


def _scene_picture(size=(320, 240)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], 12):
        for x in range(0, size[0], 12):
            if (x // 12 + y // 12) % 2:
                draw.rectangle((x, y, x + 8, y + 8), fill="blue")
    return image


def _scene_reply():
    return {
        "summary": "A desk is in the foreground. Behind it, a bookcase holds red books and a plant.",
        "foreground": ["a wooden desk and mug"],
        "people": [{"region": "centre near", "description": "a person wearing a blue shirt"}],
        "background_by_region": {"left far": ["a bookcase with three red hardbacks"],
                                 "right far": ["a green plant beside a window"]},
        "objects": [{"name": "book", "count": 3, "region": "left far", "detail": "red hardbacks"},
                    {"name": "mug", "count": 1, "region": "right near", "detail": "white ceramic"}],
        "text": [{"text": "History of Flight", "where": "book spine", "region": "left far"}],
        "screens": [{"where": "desk monitor", "region": "centre near", "shows": "a code editor"}],
        "lighting": {"description": "daylight", "sources": ["window on the right"], "time_cues": []},
        "changes": [], "uncertain": ["the smallest book titles are blurred"],
    }


class _SceneRouter:
    def __init__(self, answer=None, online=True):
        self.calls = []
        self.answer = json.dumps(_scene_reply()) if answer is None else answer
        self.online = online

    def is_online(self):
        return self.online

    async def generate_vision(self, jpeg, prompt, **options):
        self.calls.append((jpeg, prompt, options))
        return SimpleNamespace(name="configured-vision"), self.answer


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_tiles_spend_pixels_on_the_background_and_overlap():
    tiles = scenes.plan_tiles(1920, 1080, person_boxes=[(.30, .0, .70, 1.0)])
    assert len(tiles) == 9
    assert any(tile.weight == 0 and "person" in tile.reason for tile in tiles)
    assert any(tile.weight > 0 and tile.region == "left-far" for tile in tiles)
    assert tiles[0].box[2] > tiles[1].box[0]
    assert all(0 <= tile.box[0] < tile.box[2] <= 1920 for tile in tiles)


@pytest.mark.parametrize("size", [(1920, 1080), (100, 4096), (4096, 100)])
@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_contact_sheet_is_bounded_including_labels(size):
    image = Image.new("RGB", size, "white")
    encoded, layout = scenes.contact_sheet(image, scenes.plan_tiles(*size),
                                           max_side=512, max_pixels=100000)
    with Image.open(io.BytesIO(encoded)) as sheet:
        assert max(sheet.size) <= 512
        assert sheet.width * sheet.height <= 100000
        assert list(sheet.size) == layout["sheet"]
    assert encoded.startswith(b"\xff\xd8")


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_array_channels_follow_opencv_convention():
    import numpy as np
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[..., 0] = 255
    assert scenes.load_image(frame).getpixel((0, 0)) == (0, 0, 255)
    assert scenes.load_image(frame, bgr=False).getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_detector_deduplicates_the_same_object_in_overview_and_crop():
    calls = []

    def detect(frame, **_kwargs):
        calls.append(frame.shape)
        return [SimpleNamespace(label="book", confidence=.9, box=(.1, .1, .3, .3))]

    image = _scene_picture()
    tile = scenes.Tile("A1", (0, 0, 320, 240), "left-far", detail=.2)
    found, note = scenes.detect_objects(image, SimpleNamespace(available=True, detect=detect), [tile])
    assert len(calls) == 2 and len(found) == 1
    assert found[0]["label"] == "book" and not note


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_local_detail_pass_does_not_repeat_the_whole_frame_detector():
    sizes = []

    def detect(frame, **_kwargs):
        sizes.append(frame.shape[:2])
        return []

    scenes.analyse_local(_scene_picture(), detector=SimpleNamespace(available=True, detect=detect),
                         build_sheet=False)
    assert sizes.count((240, 320)) == 1
    assert 2 <= len(sizes) <= 7


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_ocr_detail_crops_map_readable_book_text_back_to_a_region():
    calls = []

    def locate(image):
        calls.append(image.size)
        text = "Desk" if image.size == (320, 240) else "History of Flight"
        return [{"text": text, "rect": (10, 10, 60, 12), "confidence": .9}]

    local = scenes.analyse_local(_scene_picture(), ocr=SimpleNamespace(locate_words=locate),
                                 build_sheet=False)
    book = next(line for line in local.text if line["text"] == "History of Flight")
    assert book["region"] in scenes.REGIONS
    assert len(book["box"]) == 4 and all(0 <= v <= 1 for v in book["box"])
    assert len(calls) > 1
    assert sum(line["text"] == "History of Flight" for line in local.text) == 1


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_one_provider_request_returns_detailed_searchable_text_without_pixels():
    router = _SceneRouter()
    analysis = asyncio.run(scenes.describe_image(_scene_picture(), router=router, prompt="what is behind me"))
    assert analysis.status == "complete" and analysis.provider == "configured-vision"
    assert len(router.calls) == 1
    jpeg, request, options = router.calls[0]
    assert jpeg.startswith(b"\xff\xd8")
    assert "CONTACT SHEET" in request and "background" in options["instruction"].lower()
    assert options["max_tokens"] == scenes.CLOUD_MAX_TOKENS
    record = analysis.to_record()
    encoded = json.dumps(record)
    assert "History of Flight" in analysis.searchable_text()
    assert "bookcase" in analysis.report()
    assert not any(key in record for key in ("image", "frame", "jpeg", "sheet"))
    assert "data:image/" not in encoded


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_offline_and_disabled_cloud_never_call_a_provider(monkeypatch):
    router = _SceneRouter(online=False)
    offline = asyncio.run(scenes.describe_image(_scene_picture(), router=router))
    assert offline.status == "local_only" and "Offline" in " ".join(offline.limitations)
    router.online = True
    monkeypatch.setenv("ORION_SCENE_CLOUD", "0")
    disabled = asyncio.run(scenes.describe_image(_scene_picture(), router=router))
    assert disabled.status == "local_only" and not router.calls


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_malformed_model_reply_falls_back_to_honest_measurements():
    analysis = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter("not JSON")))
    assert analysis.ok and analysis.status == "local_only"
    assert any("not usable" in item for item in analysis.limitations)
    assert scenes.LIMITED_NOTE in analysis.limitations


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_timeout_is_bounded_and_camera_pixels_are_not_required():
    class Router:
        async def generate_vision(self, *_args, **_kwargs):
            await asyncio.sleep(.05)

    result = asyncio.run(scenes.describe_image(_scene_picture(), router=Router(), timeout_s=.001))
    assert result.status == "local_only" and any("timed out" in item for item in result.limitations)
    invalid = asyncio.run(scenes.describe_image(b"not an image", allow_cloud=False))
    assert invalid.status == "failed"


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_parser_bounds_bad_collection_types_and_nonfinite_counts():
    reply = _scene_reply()
    reply.update(people=42, text={"bad": "shape"}, screens=True)
    reply["objects"][0]["count"] = float("inf")
    parsed = scenes.parse_reply(json.dumps(reply))
    assert not parsed["people"] and not parsed["text"] and not parsed["screens"]
    assert parsed["objects"][0]["count"] == 1


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_sensitive_person_guesses_are_removed_from_stored_scene():
    reply = _scene_reply()
    reply["people"][0]["description"] = "a young woman, wearing a blue shirt"
    reply["summary"] = "A young woman is seated. A red book is on the shelf."
    result = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter(json.dumps(reply))))
    assert "young" not in result.summary and "woman" not in result.summary
    assert "blue shirt" in result.people[0]["description"]
    assert "red book" in result.summary


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_real_memory_keeps_recallable_text_and_can_forget_it(scene_matrix):
    now = time.time()
    analysis = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter(), epoch=now))
    row_id = scenes.store(SimpleNamespace(matrix=scene_matrix), analysis)
    assert row_id == analysis.record_id
    assert len(scene_matrix.scene_memories()) == 1
    assert "History of Flight" in scenes.recall(scene_matrix, "books on the shelf", when="today")
    assert scenes.latest_record(scene_matrix)["id"] == row_id
    removed, message = scenes.forget(scene_matrix, "all")
    assert removed == 1 and "Forgotten" in message and not scene_matrix.scene_memories()


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_scene_memory_source_time_order_and_retention(scene_matrix):
    now = time.time()

    def save(epoch, source, summary):
        return scene_matrix.save_scene(captured_at="test", epoch=epoch, source=source, summary=summary,
                                 searchable=summary, record=json.dumps({"summary": summary, "epoch": epoch}))

    save(now - 31 * 86400, "camera", "expired shelf")
    first = save(now - 100, "camera", "red books")
    last = save(now, "perception", "green plant")
    assert [row["id"] for row in scene_matrix.scene_memories()] == [last, first]
    assert scene_matrix.scene_memories(source="camera")[0]["id"] == first
    assert scene_matrix.scene_memories(since=now - 50)[0]["id"] == last
    assert scene_matrix.scene_memories(oldest_first=True)[0]["id"] == first
    assert not scene_matrix.scene_memories("expired shelf")


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_memory_rejects_image_payloads_and_forgetting_unknown_time_changes_nothing(scene_matrix):
    with pytest.raises(ValueError):
        scene_matrix.save_scene(captured_at="test", epoch=time.time(), record='{"jpeg": "pixels"}')
    analysis = asyncio.run(scenes.describe_image(_scene_picture(), allow_cloud=False))
    scenes.store(scene_matrix, analysis)
    removed, message = scenes.forget(scene_matrix, "the blue hour of my dreams")
    assert removed == 0 and "haven't forgotten" in message
    assert len(scene_matrix.scene_memories()) == 1


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_time_windows_respect_explicit_timezone_and_do_not_guess_unknown_phrases():
    now = datetime(2026, 10, 12, 15, 30, tzinfo=timezone(timedelta(hours=1)))
    start, end, _label, understood = scenes.time_window("this week", now)
    assert start == now.replace(hour=0, minute=0).timestamp()
    assert end == now.timestamp() and understood
    start, _end, _label, understood = scenes.time_window("yesterday", now)
    assert start == (now - timedelta(days=1)).replace(hour=0, minute=0).timestamp()
    assert understood
    assert scenes.time_window("the blue hour of my dreams", now)[3] is False


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_today_includes_fractional_seconds_and_yesterday_excludes_todays_midnight():
    now = datetime(2026, 10, 12, 15, 30, 45, 999999)
    today_start, today_end, _label, _ok = scenes.time_window("today", now)
    assert today_start <= now.timestamp() <= today_end
    _start, yesterday_end, _label, _ok = scenes.time_window("yesterday", now)
    assert yesterday_end < now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_compare_names_object_text_and_lighting_changes():
    old = {"objects": [{"name": "books", "count": 2}], "text": [{"text": "Old title"}],
           "measured": {"brightness": 80}, "people": []}
    new = {"objects": [{"name": "book", "count": 3}, {"name": "plant", "count": 1}],
           "text": [{"text": "New title"}], "measured": {"brightness": 150}, "people": []}
    changes = " ".join(scenes.compare_records(old, new))
    assert "plant" in changes and "2 then, 3 now" in changes
    assert "New title" in changes and "brighter" in changes


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_recorder_limits_pending_work_and_snapshots_mutable_camera_frames(monkeypatch):
    async def run():
        import numpy as np
        started, finish = asyncio.Event(), asyncio.Event()
        observed = []

        async def describe(frame, *_args, **_kwargs):
            observed.append(int(frame[0, 0, 0]))
            started.set()
            await finish.wait()
            return scenes.SceneAnalysis("local_only", "scene", "camera", time.time(), "blue desk")

        monkeypatch.setattr(scenes, "describe_image", describe)
        saved = []
        memory = SimpleNamespace(save_scene=lambda **record: saved.append(record) or len(saved),
                                 scene_memories=lambda **_kwargs: [])
        recorder = scenes.SceneRecorder(memory)
        frame = np.full((8, 8, 3), 10, dtype=np.uint8)
        task = recorder.remember_in_background(frame)
        frame[:] = 99
        assert recorder.remember_in_background(frame) is None
        await started.wait()
        assert observed == [10]
        finish.set()
        await task
        assert recorder.stored == 1 and len(saved) == 1

    asyncio.run(run())


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_forget_prevents_an_inflight_look_from_resurrecting_memories(scene_matrix, monkeypatch):
    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def describe(*_args, **_kwargs):
            entered.set()
            await finish.wait()
            return scenes.SceneAnalysis("local_only", "scene", "camera", time.time(), "private desk")

        monkeypatch.setattr(scenes, "describe_image", describe)
        recorder = scenes.SceneRecorder(scene_matrix)
        task = recorder.remember_in_background(_scene_picture())
        await entered.wait()
        await recorder.forget("all")
        finish.set()
        await task
        assert not scene_matrix.scene_memories() and recorder.last is None and recorder.stored == 0

    asyncio.run(run())


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_perception_memories_are_rate_limited_local_only_and_skip_empty_events(monkeypatch):
    async def run():
        calls = []
        now = [100.0]
        recorder = scenes.SceneRecorder(SimpleNamespace(save_scene=lambda **_kwargs: 1),
                                        clock=lambda: now[0])

        async def look(_frame, **options):
            calls.append(options)

        monkeypatch.setattr(recorder, "look", look)
        assert recorder.perception_event(_scene_picture(), []) is False
        assert recorder.perception_event(_scene_picture(), ["first_frame"])
        await recorder.wait_pending()
        now[0] += 10
        assert not recorder.perception_event(_scene_picture(), ["arrival"])
        now[0] += 120
        assert recorder.perception_event(_scene_picture(), ["scene_changed"], description="a plant appeared")
        await recorder.wait_pending()
        assert all(call["allow_cloud"] is False for call in calls)

    asyncio.run(run())


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_camera_look_uses_original_frame_and_keeps_existing_media_contract(scene_matrix, monkeypatch):
    from orion_core.vision import VisionAgent
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    agent = VisionAgent.__new__(VisionAgent)
    agent.bus = _SceneBus()
    agent.scene_recorder = scenes.SceneRecorder(scene_matrix, router=_SceneRouter())
    captured = []

    async def grab(index):
        captured.append(index)
        return frame, ToolResult("captured")

    monkeypatch.setattr(agent, "grab_frame", grab)
    result = asyncio.run(agent.analyse_camera("the bookcase", 2, place="office"))
    assert captured == [2] and result.ok
    assert result.media["data"].startswith(b"\xff\xd8")
    assert "History of Flight" in result.text and "Remembered as text" in result.text
    assert scenes.latest_record(scene_matrix)["location"]["place"] == "office"
    assert not any("frame" in item for item in result.evidence)


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_camera_memory_dispatch_works_without_a_camera_and_validates_delete_time(scene_matrix):
    from orion_core.dispatch_vision import VisionDispatchMixin

    class Dispatcher(VisionDispatchMixin):
        memory = scene_matrix
        perception = None

    analysis = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter()))
    scenes.store(scene_matrix, analysis)
    dispatcher = Dispatcher()
    result = asyncio.run(dispatcher.vision_analyse({"action": "recall_camera", "query": "book"}))
    assert result.ok and "History of Flight" in result.text
    invalid = asyncio.run(dispatcher.vision_analyse({"action": "forget_camera", "when": "nonsense"}))
    assert not invalid.ok and scene_matrix.scene_memories()
    result = asyncio.run(dispatcher.perception_tool({"action": "forget_memory", "when": "all"}))
    assert result.ok and not scene_matrix.scene_memories()


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_local_watch_does_not_retain_raw_frames_without_opt_in():
    from orion_core.perception import PerceptionLoop
    import numpy as np

    loop = PerceptionLoop(_SceneBus(), lab=SimpleNamespace(reset=lambda: None), clock=lambda: 100)
    loop._task = SimpleNamespace(done=lambda: False)
    loop._remember_camera_frame(np.zeros((20, 20, 3), dtype=np.uint8), 100)
    assert not loop._camera_history and loop.recent_camera_frames() == []
    loop.set_frame_history(True)
    loop._remember_camera_frame(np.zeros((20, 20, 3), dtype=np.uint8), 100)
    assert loop._camera_history
    loop.set_frame_history(False)
    assert not loop._camera_history


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_perception_tick_remembers_initial_frame_without_cloud_calls():
    from orion_core.perception import PerceptionLoop
    import numpy as np

    saved = []
    recorder = SimpleNamespace(perception_event=lambda frame, kinds, **kwargs:
                               saved.append((kinds, kwargs)))
    loop = PerceptionLoop(_SceneBus(), frame_source=lambda: np.zeros((24, 32, 3), dtype=np.uint8),
                          lab=SimpleNamespace(read=lambda _frame: None), scene_recorder=recorder)
    assert asyncio.run(loop.tick()) == []
    assert saved[0][0] == {"first_frame"}
    assert loop.cloud_calls_this_grant == 0


@pytest.mark.parametrize("action", ["pcb", "scan", "count", "read_label"])
@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_camera_scans_remember_existing_frame_and_report_without_a_second_model_call(scene_matrix, monkeypatch, action):
    from orion_core import electronics_inspection
    from orion_core.dispatch_vision import VisionDispatchMixin
    from orion_core.vision import VisionAgent

    buffer = io.BytesIO()
    _scene_picture().save(buffer, format="JPEG")
    jpeg = buffer.getvalue()
    router = _SceneRouter()
    calls = []

    async def inspect(frame, prompt, **options):
        calls.append((frame, prompt, options))
        return ToolResult("Three resistors and a legible label R12.",
                          media={"data": jpeg, "mime_type": "image/jpeg"},
                          evidence=[{"summary": "Three resistors", "status": "complete"}])

    async def capture(*_args, **_kwargs):
        return ToolResult("captured", media={"data": jpeg, "mime_type": "image/jpeg"})

    monkeypatch.setattr(electronics_inspection, "inspect_frame", inspect)
    vision = VisionAgent.__new__(VisionAgent)
    vision.bus = _SceneBus()
    vision.scene_recorder = scenes.SceneRecorder(scene_matrix, router=router)
    monkeypatch.setattr(vision, "capture_live_frame", capture)

    class Dispatcher(VisionDispatchMixin):
        bus = _SceneBus()

    dispatcher = Dispatcher()
    dispatcher.vision, dispatcher.router = vision, router
    result = asyncio.run(dispatcher.vision_analyse({"action": action, "prompt": "read the labels"}))
    assert result.ok and result.evidence[0]["summary"] == "Three resistors"
    assert len(calls) == 1 and not router.calls
    record = scenes.latest_record(scene_matrix)
    assert record["source"] == "camera_inspection" and "R12" in record["summary"]
    assert "no additional vision request" in " ".join(record["limitations"])


@pytest.mark.parametrize("action,source", [("watch_camera", "camera_watch"),
                                           ("watch_camera_past", "camera_recent")])
@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_camera_sequences_remember_timeline_text_without_new_inference(scene_matrix, monkeypatch, action, source):
    from orion_core import video_analysis
    from orion_core.dispatch_vision import VisionDispatchMixin

    report = video_analysis.VideoReport(path="the camera", duration=6, summary="The mug moved to the left.",
                                         frames=[video_analysis.FrameNote(0, "mug on the right", "model"),
                                                 video_analysis.FrameNote(6, "mug on the left", "model")])
    router = _SceneRouter()
    seen = []

    class Analyser:
        def __init__(self, **_options):
            pass

        async def watch_camera(self, *_args, **_options):
            seen.append("watch")
            return report

        async def analyse_recent_camera(self, *_args, **_options):
            seen.append("recent")
            return report

    monkeypatch.setattr(video_analysis, "VideoAnalyser", Analyser)

    class Dispatcher(VisionDispatchMixin):
        pass

    dispatcher = Dispatcher()
    dispatcher.router = router
    dispatcher.scene_recorder = scenes.SceneRecorder(scene_matrix, router=router)
    dispatcher.vision = SimpleNamespace(scene_recorder=dispatcher.scene_recorder)
    dispatcher.perception = SimpleNamespace(running=True, retain_camera_history=True,
                                             recent_camera_frames=lambda *_args: [(0, b"first"), (6, b"last")])
    result = asyncio.run(dispatcher.vision_analyse({"action": action}))
    assert result.ok and len(seen) == 1 and not router.calls
    record = scenes.latest_record(scene_matrix)
    assert record["source"] == source and "mug moved" in record["summary"]
    assert "At 6.0 seconds: mug on the left" in record["foreground"]
    assert not any(key in record for key in ("image", "jpeg", "frames"))


@pytest.mark.parametrize("instrument,source", [("read_now", "camera_measure"),
                                               ("detect_now", "camera_detect"),
                                               ("pose_now", "camera_pose")])
@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_on_demand_local_instruments_write_text_without_cloud(scene_matrix, monkeypatch, instrument, source):
    from orion_core import vision_lab
    from orion_core.object_detection import Detection
    from orion_core.perception import PerceptionLoop
    import numpy as np

    class Lab:
        def __init__(self, **_options):
            pass

        def read(self, _frame):
            return SimpleNamespace(summary=lambda: "The camera view is bright and mostly blue.")

    monkeypatch.setattr(vision_lab, "VisionLab", Lab)
    router = _SceneRouter()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    recorder = scenes.SceneRecorder(scene_matrix, router=router)
    loop = PerceptionLoop(_SceneBus(), scene_recorder=recorder,
                          objects=SimpleNamespace(available=True,
                                                  detect=lambda _frame: [Detection("book", .9, (.1, .1, .3, .3))]),
                          pose=SimpleNamespace(available=True, error="",
                                               read=lambda _frame: SimpleNamespace(describe=lambda: "A person is seated.")))

    async def grab(count=1, **_options):
        return [frame] * count

    monkeypatch.setattr(loop, "grab", grab)
    asyncio.run(getattr(loop, instrument)())
    record = scenes.latest_record(scene_matrix)
    assert record["source"] == source and not router.calls
    if instrument == "detect_now":
        assert record["objects"][0]["name"] == "book"


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_stop_revokes_raw_history_opt_in():
    from orion_core.perception import PerceptionLoop

    loop = PerceptionLoop(_SceneBus(), lab=SimpleNamespace(reset=lambda: None))
    loop.set_frame_history(True)
    loop._camera_history.append((100, b"old jpeg"))
    asyncio.run(loop.stop())
    assert not loop.retain_camera_history and not loop._camera_history


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_forgetting_camera_memories_clears_cached_perception_descriptions(scene_matrix):
    from orion_core.dispatch_vision import VisionDispatchMixin
    from orion_core.perception import PerceptionEvent, PerceptionLoop

    loop = PerceptionLoop(_SceneBus(), lab=SimpleNamespace(reset=lambda: None))
    loop.last_description = "A secret book title is visible."
    loop.events.append(PerceptionEvent("described", "today", loop.last_description))
    loop.scene.tracks[1] = SimpleNamespace(label=loop.last_description)
    loop._camera_history.append((100, b"old jpeg"))

    class Dispatcher(VisionDispatchMixin):
        memory = scene_matrix
        perception = loop

    result = asyncio.run(Dispatcher().vision_analyse({"action": "forget_camera", "when": "all"}))
    assert result.ok and not loop.last_description and not loop.events
    assert not loop.scene.tracks and not loop._camera_history


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_forgetting_cached_perception_rejects_a_cloud_description_still_in_flight():
    from orion_core.perception import PerceptionLoop

    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def describe(_frame):
            entered.set()
            await finish.wait()
            return "A secret book title is visible."

        loop = PerceptionLoop(_SceneBus(), describe=describe, cloud_enabled=True,
                              lab=SimpleNamespace(reset=lambda: None))
        task = asyncio.create_task(loop._understand(None, "scene changed", 100))
        await entered.wait()
        loop.clear_scene_memories()
        finish.set()
        await task
        assert not loop.last_description and not loop.events

    asyncio.run(run())


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_text_only_measurement_does_not_claim_previously_seen_objects_vanished(scene_matrix):
    recorder = scenes.SceneRecorder(scene_matrix)
    report = asyncio.run(recorder.remember_text("Light level 100/255.", source="camera_measure"))
    changes = scenes.compare_records({"objects": [{"name": "book", "count": 1}]}, report.to_record())
    assert not any("no longer seen" in item for item in changes)


@pytest.mark.parametrize("prefix", ["no longer seen:", "text no longer visible:", "people in view:"])
@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_local_measurements_do_not_invent_disappearances_without_instruments(prefix):
    before = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter())).to_record()
    after = asyncio.run(scenes.describe_image(_scene_picture(), allow_cloud=False,
                                               previous=before)).to_record()
    assert after["status"] == "local_only"
    assert after["measured"]["brightness"] > 0
    assert not any(item.startswith(prefix) for item in after["changes"])


@pytest.mark.parametrize("missing,prefix", [("objects", "no longer seen:"),
                                            ("text", "text no longer visible:"),
                                            ("people", "people in view:")])
@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_missing_model_collections_do_not_become_empty_observations(missing, prefix):
    before = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter())).to_record()
    answer = _scene_reply()
    del answer[missing]
    after = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter(json.dumps(answer)),
                                               previous=before)).to_record()
    assert after["status"] == "complete"
    assert not any(item.startswith(prefix) for item in after["changes"])


@pytest.mark.parametrize("empty,prefix", [("objects", "no longer seen:"),
                                          ("text", "text no longer visible:"),
                                          ("people", "people in view:")])
@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_completed_empty_model_inventory_can_report_disappearances(empty, prefix):
    before = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter())).to_record()
    answer = _scene_reply()
    answer[empty] = []
    after = asyncio.run(scenes.describe_image(_scene_picture(), router=_SceneRouter(json.dumps(answer)),
                                               previous=before)).to_record()
    assert after["status"] == "complete"
    assert any(item.startswith(prefix) for item in after["changes"])


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_internal_provider_type_error_does_not_repeat_a_started_request():
    class Router:
        def __init__(self):
            self.calls = []

        async def generate_vision(self, _jpeg, _prompt, **options):
            self.calls.append(options)
            raise TypeError("Provider response conversion failed after the request started")

    router = Router()
    result = asyncio.run(scenes.describe_image(_scene_picture(), router=router))
    assert result.status == "local_only"
    assert len(router.calls) == 1
    assert router.calls[0]["max_tokens"] == scenes.CLOUD_MAX_TOKENS
    assert any("TypeError" in item for item in result.limitations)


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_legacy_provider_signature_is_checked_before_one_request():
    class Router:
        def __init__(self):
            self.calls = 0

        async def generate_vision(self, _jpeg, _prompt, *, instruction, task):
            self.calls += 1
            assert "background" in instruction.lower() and task == "scene_analysis"
            return SimpleNamespace(name="legacy-vision"), json.dumps(_scene_reply())

    router = Router()
    result = asyncio.run(scenes.describe_image(_scene_picture(), router=router))
    assert result.status == "complete" and result.provider == "legacy-vision"
    assert router.calls == 1


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_offline_look_skips_contact_sheet_encoding(monkeypatch):
    encoded = []
    original = scenes.contact_sheet

    def contact_sheet(*args, **kwargs):
        jpeg, layout = original(*args, **kwargs)
        encoded.append(len(jpeg))
        return jpeg, layout

    monkeypatch.setattr(scenes, "contact_sheet", contact_sheet)
    router = _SceneRouter(online=False)
    result = asyncio.run(scenes.describe_image(_scene_picture(), router=router))
    assert result.status == "local_only" and result.measured["brightness"] > 0
    assert not router.calls and any("Offline" in item for item in result.limitations)
    assert not encoded and result.request_bytes == 0


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_online_look_keeps_one_encoded_contact_sheet(monkeypatch):
    encoded = []
    original = scenes.contact_sheet

    def contact_sheet(*args, **kwargs):
        jpeg, layout = original(*args, **kwargs)
        encoded.append(len(jpeg))
        return jpeg, layout

    monkeypatch.setattr(scenes, "contact_sheet", contact_sheet)
    router = _SceneRouter()
    result = asyncio.run(scenes.describe_image(_scene_picture(), router=router))
    assert result.status == "complete" and len(router.calls) == 1
    assert encoded == [result.request_bytes] and result.request_bytes > 0


@pytest.mark.usefixtures("_scene_no_models")
def test_scene_detail_provider_cancellation_is_propagated_without_retry():
    async def run():
        entered = asyncio.Event()
        calls = []

        class Router:
            async def generate_vision(self, *_args, **_kwargs):
                calls.append("started")
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(scenes.describe_image(_scene_picture(), router=Router()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == ["started"]

    asyncio.run(run())
