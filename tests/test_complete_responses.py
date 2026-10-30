"""Long replies are resumed through real local HTTP exchanges, without replays."""
import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from test_local_provider_memory import _cloud, _router
from orion_core.remote import _PhoneHistory


@pytest.mark.parametrize("stream", [False, True])
def test_output_limit_resumes_the_same_answer_and_keeps_final_words(stream):
    async def run():
        calls = []
        fragments = ["A comprehensive answer includes inter", "operability and the FINAL WORDS."]
        async def chat(request):
            payload = await request.json()
            calls.append(payload)
            index = len(calls) - 1
            reason = "length" if index == 0 else "stop"
            if payload["stream"]:
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                for chunk in [
                    {"choices": [{"delta": {"content": fragments[index]}}]},
                    {"choices": [{"delta": {}, "finish_reason": reason}]},
                ]:
                    await response.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                await response.write(b"data: [DONE]\n\n")
                return response
            return web.json_response({"choices": [{"message": {"content": fragments[index]},
                                                    "finish_reason": reason}]})
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        async with TestServer(app) as server:
            router = _router(_cloud("long-answer", str(server.make_url("/v1"))))
            deltas = []
            try:
                provider, text = await router.generate_text(
                    "Give me a comprehensive answer", on_delta=deltas.append if stream else None)
                assert text == "".join(fragments)
                assert provider.name == "long-answer"
                assert len(calls) == 2
                assert calls[0]["max_tokens"] == 4096
                assert calls[1]["messages"][-2] == {"role": "assistant", "content": fragments[0]}
                assert calls[1]["messages"][-1]["role"] == "user"
                if stream:
                    assert "".join(deltas) == text
            finally:
                await router.aclose()
    asyncio.run(run())


def test_failed_continuation_preserves_the_answer_and_reports_incompletion():
    async def run():
        calls = 0
        async def chat(request):
            nonlocal calls
            calls += 1
            if calls > 1:
                return web.Response(status=503)
            return web.json_response({"choices": [{"message": {"content": "The completed first section."},
                                                    "finish_reason": "length"}]})
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        async with TestServer(app) as server:
            router = _router(_cloud("continuation", str(server.make_url("/v1"))))
            try:
                _, text = await router.generate_text("Explain this")
                assert text.startswith("The completed first section.")
                assert "answer is incomplete" in text
                assert calls == 2  # A failed paid request is not replayed.
            finally:
                await router.aclose()
    asyncio.run(run())


def test_strict_json_is_not_extended_and_repeated_limits_are_bounded():
    async def run():
        calls = []
        async def chat(request):
            calls.append(await request.json())
            return web.json_response({"choices": [{"message": {"content": "partial "},
                                                    "finish_reason": "length"}]})
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        async with TestServer(app) as server:
            router = _router(_cloud("bounded", str(server.make_url("/v1"))))
            try:
                await router.generate_text("Return JSON", instruction="Only JSON")
                assert len(calls) == 1
                _, text = await router.generate_text("Explain this")
                assert len(calls) == 7  # Initial answer plus five continuations.
                assert text.count("partial") == 6
                assert "answer is incomplete" in text
            finally:
                await router.aclose()
    asyncio.run(run())


def test_history_keeps_full_long_answers_with_a_device_memory_budget():
    history = _PhoneHistory()
    reply = "Detailed answer. " * 2500 + "THE CONCLUSION."
    history.add("phone", "orion", reply)
    assert history.page("phone", None, 10)[0][0]["text"] == reply
    assert not history.page("phone", None, 10)[0][0]["truncated"]
    for _ in range(200):
        history.add("phone", "orion", reply)
    entries = history.page("phone", None, 100)[0]
    assert entries[-1]["text"].endswith("THE CONCLUSION.")
    assert sum(len(entry["text"]) for entry in entries) <= history.CHAR_BUDGET


def test_quick_tunnel_frames_deliver_text_before_completion(tmp_path):
    from test_remote_gateway import _StubBus, _StubMemory, _StreamingRouter, _paired, _client
    from orion_core.remote import RemoteGateway
    async def run():
        finish = asyncio.Event()
        async def body(on_delta, on_restart):
            on_delta("Beginning of the answer. ")
            await finish.wait()
            return "Beginning of the answer. FINAL WORDS."
        gateway = RemoteGateway(_StreamingRouter(body), _StubMemory(), _StubBus(), config_dir=tmp_path)
        _, headers = _paired(gateway)
        client = await _client(gateway)
        try:
            response = await client.post("/api/chat/frames", json={"message": "Explain TLS"}, headers=headers)
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("application/x-ndjson")
            assert json.loads(await response.content.readline())["type"] == "start"
            delta = json.loads(await asyncio.wait_for(response.content.readline(), 2))
            assert delta["type"] == "delta" and not finish.is_set()
            finish.set()
            events = [json.loads(line) for line in (await response.content.read()).splitlines() if line]
            assert events[-1]["type"] == "done"
            assert events[-1]["reply"].endswith("FINAL WORDS.")
            denied = await client.post("/api/chat/frames", json={"message": "Explain TLS"})
            assert denied.status == 401
        finally:
            finish.set()
            await client.close()
    asyncio.run(run())
