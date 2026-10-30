"""Exercise latency fixes through real loopback HTTP, without API accounts."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from orion_core import provider_transport
from orion_core.providers import ProviderDeadlineError, _CallScope
from test_local_provider_memory import _cloud, _router


async def test_catalogue_read_is_shared_by_profiles_and_concurrent_callers():
    calls = []

    async def catalogue(request):
        calls.append(request.headers.get("Authorization"))
        await asyncio.sleep(0.03)
        return web.json_response({"data": [{"id": "gemini-3.5-flash-lite"}, {"id": "gemini-3.8-flash"}]})

    app = web.Application()
    app.router.add_get("/v1/models", catalogue)
    async with TestServer(app) as server:
        first = _cloud("routine", str(server.make_url("/v1")))
        second = replace(first, name="reasoning", base_url=first.base_url + "/chat/completions")
        router = _router(first, second)
        try:
            results = await asyncio.gather(*(router._list_models(profile) for profile in
                                            (first, second, first, second)))
            assert results == [["gemini-3.5-flash-lite", "gemini-3.8-flash"]] * 4
            results[0].clear()
            assert await router._list_models(second) == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
            assert len(calls) == 1
        finally:
            await router.aclose()


async def test_catalogue_cache_is_scoped_to_current_credential_and_expires():
    calls = []

    async def catalogue(request):
        key = request.headers.get("Authorization")
        calls.append(key)
        return web.json_response({"data": [{"id": f"catalogue-{len(calls)}"}]})

    app = web.Application()
    app.router.add_get("/v1/models", catalogue)
    async with TestServer(app) as server:
        first = _cloud("one", str(server.make_url("/v1")))
        second = replace(first, name="two", api_key="another-test-key")
        router = _router(first, second)
        try:
            assert await router._list_models(first) == ["catalogue-1"]
            assert await router._list_models(second) == ["catalogue-2"]
            assert await router._list_models(first) == ["catalogue-1"]
            assert len(calls) == 2
            for key, (_, rows) in list(router._model_catalogues.items()):
                router._model_catalogues[key] = (time.monotonic() - 301, rows)
            assert await router._list_models(first) == ["catalogue-3"]
            assert "another-test-key" not in repr(router._model_catalogues)
        finally:
            await router.aclose()


async def test_cancelled_catalogue_waiter_does_not_cancel_the_active_reader():
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def catalogue(_request):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return web.json_response({"data": [{"id": "usable"}]})

    app = web.Application()
    app.router.add_get("/v1/models", catalogue)
    async with TestServer(app) as server:
        profile = _cloud("one", str(server.make_url("/v1")))
        router = _router(profile)
        reader = asyncio.create_task(router._list_models(profile))
        try:
            await started.wait()
            waiter = asyncio.create_task(router._list_models(profile))
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            release.set()
            assert await reader == ["usable"]
            assert await router._list_models(profile) == ["usable"]
            assert calls == 1
        finally:
            release.set()
            await router.aclose()


async def test_boot_validation_and_healing_do_not_fetch_the_same_catalogue_twice(monkeypatch):
    calls = 0

    async def catalogue(_request):
        nonlocal calls
        calls += 1
        return web.json_response({"data": [{"id": "gemini-3.5-flash-lite"}]})

    app = web.Application()
    app.router.add_get("/v1/models", catalogue)
    async with TestServer(app) as server:
        profile = _cloud("retired", str(server.make_url("/v1")))
        profile.model = "retired-model"
        router = _router(profile)
        monkeypatch.setattr(router, "_persist_model", lambda *_, **__: None)
        try:
            assert await router.validate_models() == {"retired": "gemini-3.5-flash-lite"}
            assert calls == 1
        finally:
            await router.aclose()


@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_completion_marker_releases_a_retained_response_promptly(kind):
    release = asyncio.Event()

    async def chat(_request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(_request)
        if kind == "openai":
            body = b'data: {"choices":[{"delta":{"content":"complete"},"finish_reason":"stop"}]}\n\n'
            body += b"data: [DONE]\n\n"
        else:
            body = json.dumps({"message": {"content": "complete"}, "done": True}).encode() + b"\n"
        await response.write(body)
        await release.wait()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions" if kind == "openai" else "/api/chat", chat)
    async with TestServer(app) as server:
        pool = provider_transport.SessionPool()
        transport = (provider_transport.OpenAIChatTransport(str(server.make_url("/v1")))
                     if kind == "openai" else provider_transport.OllamaChatTransport(str(server.make_url(""))))
        try:
            session = await pool.get()
            req = provider_transport.ChatRequest("test", "system", "hello", 200, stream=True)
            # An unanswered read after [DONE] used to hold the result for 500 ms.
            result = await asyncio.wait_for(transport.chat(
                session, req, timeout=provider_transport._aiohttp().ClientTimeout(total=1), trace={}), 0.25)
            assert result.text == "complete"
        finally:
            release.set()
            await pool.aclose()


async def test_completed_streams_with_normal_eof_still_reuse_the_connection():
    peers = []

    async def chat(request):
        peers.append(request.transport.get_extra_info("peername"))
        return web.Response(body=b'data: {"choices":[{"delta":{"content":"complete"}}]}\n\n'
                                b"data: [DONE]\n\n", content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    async with TestServer(app) as server:
        pool = provider_transport.SessionPool()
        transport = provider_transport.OpenAIChatTransport(str(server.make_url("/v1")))
        try:
            session = await pool.get()
            for _ in range(3):
                result = await transport.chat(session, provider_transport.ChatRequest(
                    "test", "system", "hello", 100, stream=True),
                    timeout=provider_transport._aiohttp().ClientTimeout(total=1), trace={})
                assert result.text == "complete"
            assert len(set(peers)) == 1
        finally:
            await pool.aclose()


async def test_excess_trailing_data_is_bounded_without_losing_the_finished_answer():
    class Content:
        total = 0

        async def read(self, count):
            self.total += count
            return b"x" * count

    closed = []
    content = Content()
    response = SimpleNamespace(content=content, close=lambda: closed.append(True))
    await provider_transport._drain(response)
    assert content.total == 65_536 and closed == [True]


async def test_prompt_preparation_does_not_block_the_loop_and_reaches_the_provider():
    captured = []
    worker_thread = []

    class Memory:
        def prompt_context(self, limit):
            worker_thread.append(threading.get_ident())
            time.sleep(0.12)
            return "A grounded fact about the user's project."

    async def chat(request):
        captured.append(await request.json())
        return web.json_response({"choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    async with TestServer(app) as server:
        router = _router(_cloud("one", str(server.make_url("/v1"))))
        router.memory = Memory()
        main_thread = threading.get_ident()
        pulses = 0
        running = True

        async def pulse():
            nonlocal pulses
            while running:
                pulses += 1
                await asyncio.sleep(0.01)

        heartbeat = asyncio.create_task(pulse())
        try:
            _, answer = await router.generate_text("Hello")
            assert answer == "complete"
            assert pulses >= 5
            assert worker_thread and all(value != main_thread for value in worker_thread)
            assert "A grounded fact about the user's project." in captured[0]["messages"][0]["content"]
        finally:
            running = False
            await heartbeat
            await router.aclose()


async def test_context_worker_respects_the_turn_deadline():
    router = _router(_cloud("one", "https://unused.invalid/v1"))
    try:
        with pytest.raises(ProviderDeadlineError, match="context preparation"):
            await router._prompt_work(lambda: time.sleep(0.08),
                                      _CallScope(deadline=time.monotonic() + 0.01))
    finally:
        await router.aclose()
