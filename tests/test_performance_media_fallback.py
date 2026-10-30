import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from orion_core import video_analysis as media


def _sdk(monkeypatch, *, fail=False, release=None, construction_release=None, close_fails=False):
    import google
    state = SimpleNamespace(clients=[], started=threading.Event(), constructed=threading.Event())
    module = ModuleType("google.genai")
    types = ModuleType("google.genai.types")
    types.Part = SimpleNamespace(from_bytes=lambda **kw: kw)
    types.HttpOptions = lambda **kw: SimpleNamespace(**kw)
    types.HttpRetryOptions = lambda **kw: SimpleNamespace(**kw)
    class Client:
        def __init__(self, **kwargs):
            self.kwargs, self.calls, self.closed = kwargs, 0, 0
            self.models = self
            state.clients.append(self)
            state.constructed.set()
            if construction_release is not None:
                assert construction_release.wait(2)
        def generate_content(self, **kwargs):
            self.calls += 1
            state.started.set()
            if release is not None:
                assert release.wait(2)
            if fail:
                raise RuntimeError("synthetic provider error")
            return SimpleNamespace(text="A visible test frame.")
        def close(self):
            self.closed += 1
            if close_fails:
                raise OSError("synthetic close failure")
    module.Client, module.types = Client, types
    monkeypatch.setattr(google, "genai", module, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    return state


@pytest.mark.parametrize("fail", [False, True])
def test_frame_client_closes_after_success_or_provider_failure(monkeypatch, fail):
    state = _sdk(monkeypatch, fail=fail)
    value = asyncio.run(media.VideoAnalyser(api_key="synthetic")._describe_with_model(b"frame", ""))
    assert value == ("" if fail else "A visible test frame.")
    assert len(state.clients) == 1 and state.clients[0].closed == 1


def test_frame_fallback_has_a_transport_deadline_and_no_hidden_retries(monkeypatch):
    state = _sdk(monkeypatch)
    asyncio.run(media.VideoAnalyser(api_key="synthetic")._describe_with_model(b"frame", ""))
    options = state.clients[0].kwargs.get("http_options")
    assert options is not None and 0 < options.timeout <= 12_000
    assert options.retry_options.attempts == 1


def test_client_cleanup_failure_preserves_observed_frame_text(monkeypatch):
    state = _sdk(monkeypatch, close_fails=True)
    value = asyncio.run(media.VideoAnalyser(api_key="synthetic")._describe_with_model(b"frame", ""))
    assert value == "A visible test frame."
    assert state.clients[0].closed == 1


def test_timed_out_frame_reaches_ocr_while_late_client_retains_cleanup(monkeypatch):
    release = threading.Event()
    state = _sdk(monkeypatch, release=release)
    monkeypatch.setattr(media, "MODEL_FALLBACK_TIMEOUT_S", .025, raising=False)
    analyser = media.VideoAnalyser(api_key="synthetic")
    async def ocr(*args):
        return "Verified OCR frame"
    analyser._describe_with_ocr = ocr
    async def scenario():
        task = asyncio.create_task(analyser._describe_one(b"frame", ""))
        try:
            assert await asyncio.to_thread(state.started.wait, 1)
            await asyncio.sleep(.09)
            finished_before_native_return = task.done()
            close_before_return = state.clients[0].closed
        finally:
            release.set()
        result = await task
        for _ in range(50):
            if state.clients[0].closed:
                break
            await asyncio.sleep(.002)
        assert finished_before_native_return
        assert close_before_return == 0
        assert result == ("Verified OCR frame", "ocr")
        assert state.clients[0].closed == 1
        assert not analyser._capture_workers
    asyncio.run(scenario())


def test_cancelled_frame_keeps_client_owned_until_request_settles(monkeypatch):
    release = threading.Event()
    state = _sdk(monkeypatch, release=release, fail=True)
    analyser = media.VideoAnalyser(api_key="synthetic")
    async def scenario():
        task = asyncio.create_task(analyser._describe_with_model(b"frame", ""))
        try:
            assert await asyncio.to_thread(state.started.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert state.clients[0].closed == 0
            assert len(analyser._capture_workers) == 1
        finally:
            release.set()
        for _ in range(100):
            if not analyser._capture_workers:
                break
            await asyncio.sleep(.002)
        assert state.clients[0].closed == 1 and not analyser._capture_workers
    asyncio.run(scenario())


def test_cancel_during_client_creation_prevents_late_generation(monkeypatch):
    release = threading.Event()
    state = _sdk(monkeypatch, construction_release=release)
    analyser = media.VideoAnalyser(api_key="synthetic")
    async def scenario():
        task = asyncio.create_task(analyser._describe_with_model(b"frame", ""))
        try:
            assert await asyncio.to_thread(state.constructed.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        for _ in range(100):
            if state.clients[0].closed:
                break
            await asyncio.sleep(.002)
        assert state.clients[0].calls == 0
        assert state.clients[0].closed == 1
        assert not analyser._capture_workers
    asyncio.run(scenario())
