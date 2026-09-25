"""
Watching a video: its pictures, its voices, and what to do about it.

  "ORION must be able to analyse images, analyse voices and transcribe the
   voices too within those videos - I want ORION to be able to watch videos and
   identify what course of action to take dependent on the prompt."

A video is two streams read differently then read together: keyframes described
(by a vision model where reachable, else OCR), the audio track transcribed
offline, and both handed to the model with the prompt so the recommended action
is grounded in what was seen AND said. Every stage degrades to a plain note
rather than failing the whole analysis.

The real cv2/PyAV extraction is exercised against a genuine synthesised clip;
the model stages use stubs so the test needs no network.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.video_analysis import (  # noqa: E402
    VIDEO_SUFFIXES, FrameNote, VideoAnalyser, VideoReport,
)

cv2 = pytest.importorskip("cv2")
av = pytest.importorskip("av")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def clip():
    """A real 2-second clip: text-bearing frames + a sine audio track."""
    path = Path(tempfile.mkdtemp()) / "clip.mp4"
    container = av.open(str(path), mode="w")
    vs = container.add_stream("libx264", rate=10)
    vs.width, vs.height, vs.pix_fmt = 320, 240, "yuv420p"
    astream = container.add_stream("aac", rate=44100)
    for i in range(20):
        img = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(img, f"STEP {i}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        for pkt in vs.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            container.mux(pkt)
    for pkt in vs.encode():
        container.mux(pkt)
    sr = 44100
    t = np.arange(int(sr * 2)) / sr
    sig = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    frame = av.AudioFrame.from_ndarray(sig.reshape(1, -1), format="fltp", layout="mono")
    frame.rate = sr
    for pkt in astream.encode(frame):
        container.mux(pkt)
    for pkt in astream.encode():
        container.mux(pkt)
    container.close()
    return path


@pytest.fixture(scope="module")
def silent_clip():
    """A clip with NO audio track — the degradation case."""
    path = Path(tempfile.mkdtemp()) / "silent.mp4"
    container = av.open(str(path), mode="w")
    vs = container.add_stream("libx264", rate=10)
    vs.width, vs.height, vs.pix_fmt = 320, 240, "yuv420p"
    for _ in range(15):
        img = np.zeros((240, 320, 3), np.uint8)
        for pkt in vs.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            container.mux(pkt)
    for pkt in vs.encode():
        container.mux(pkt)
    container.close()
    return path


class _Router:
    def __init__(self):
        self.prompts = []

    async def generate_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return (None, "SUMMARY: A step-by-step tutorial.\nACTION: Follow the steps.")


class _Vision:
    def _ocr_file_text(self, path):
        return "STEP text visible"


class _Transcriber:
    def ensure_ready(self):
        return True

    def transcribe_wav(self, path):
        return "this is the narration"


# ── real extraction ──────────────────────────────────────────────────────────

def test_keyframes_are_extracted_with_a_duration(clip):
    frames, duration = VideoAnalyser()._extract_keyframes(clip, 5)
    assert len(frames) == 5
    assert 1.5 <= duration <= 2.5
    assert all(isinstance(jpeg, bytes) and jpeg[:2] == b"\xff\xd8" for _t, jpeg in frames)


def test_the_audio_track_is_extracted_to_wav(clip):
    wav, note = VideoAnalyser()._extract_audio(clip)
    assert wav is not None and wav.suffix == ".wav"
    assert wav.stat().st_size > 1000
    assert note == ""
    wav.unlink()


def test_a_video_with_no_audio_is_reported_not_crashed(silent_clip):
    wav, note = VideoAnalyser()._extract_audio(silent_clip)
    assert wav is None
    assert "no audio" in note.lower() or "empty" in note.lower()


def test_a_missing_file_yields_a_note_not_an_exception():
    report = asyncio.run(VideoAnalyser().analyse("does-not-exist.mp4"))
    assert "file not found" in report.notes


# ── the full pipeline ────────────────────────────────────────────────────────

def test_frames_transcript_and_action_all_come_together(clip):
    router = _Router()
    analyser = VideoAnalyser(router=router, vision=_Vision(), transcriber=_Transcriber())
    report = asyncio.run(analyser.analyse(str(clip), prompt="what should I do?"))
    assert len(report.frames) > 0
    assert report.transcript == "this is the narration"
    assert report.summary
    assert report.action == "Follow the steps."
    # The synthesis must have seen BOTH streams.
    assert any("KEYFRAMES" in p and "TRANSCRIPT" in p for p in router.prompts)


def test_frame_description_falls_back_to_ocr_without_a_key(clip):
    analyser = VideoAnalyser(router=_Router(), vision=_Vision(),
                             transcriber=_Transcriber())   # no api_key
    report = asyncio.run(analyser.analyse(str(clip)))
    assert report.frames and all(fn.via == "ocr" for fn in report.frames)


def test_no_transcriber_degrades_but_keeps_the_frames(clip):
    report = asyncio.run(
        VideoAnalyser(router=_Router(), vision=_Vision()).analyse(str(clip)))
    assert report.frames
    assert "no transcriber available" in report.notes
    assert report.transcript == ""


def test_no_router_still_produces_an_offline_summary(clip):
    report = asyncio.run(
        VideoAnalyser(vision=_Vision(), transcriber=_Transcriber()).analyse(str(clip)))
    assert report.summary          # offline summary from frames + transcript
    assert "narration" in report.summary


def test_synthesis_failure_does_not_lose_the_analysis(clip):
    class _Broken(_Router):
        async def generate_text(self, prompt, **kwargs):
            raise RuntimeError("provider down")

    report = asyncio.run(
        VideoAnalyser(router=_Broken(), vision=_Vision(),
                      transcriber=_Transcriber()).analyse(str(clip)))
    assert any("synthesis failed" in n for n in report.notes)
    assert report.summary          # fell back to the offline summary


# ── reporting ────────────────────────────────────────────────────────────────

def test_the_announcement_states_what_was_done():
    report = VideoReport(path="lecture.mp4", duration=185.0,
                         frames=[FrameNote(1.0, "a slide")], transcript="hello",
                         action="Take notes on section 2.")
    said = report.announcement()
    assert "lecture.mp4" in said
    assert "3m05s" in said
    assert "transcribed" in said
    assert "Take notes" in said


def test_the_split_is_robust():
    s, a = VideoAnalyser._split_summary_action(
        "SUMMARY: it shows X.\n\nACTION: do Y.")
    assert s == "it shows X."
    assert a == "do Y."


# ── the tool ─────────────────────────────────────────────────────────────────

def test_the_vision_tool_routes_video():
    import inspect

    from orion_core.dispatch_vision import VisionDispatchMixin

    source = inspect.getsource(VisionDispatchMixin.vision_analyse)
    assert '"video"' in source
    assert "_analyse_video" in source


def test_the_schema_advertises_the_video_action():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "vision_analyse")
    assert "video" in tool["description"].lower()
    assert "frames" in tool["parameters"]["properties"]
