"""A truncated or over-budget model reply must never be accepted as complete."""
import json
from types import SimpleNamespace

import pytest

from orion_core.provider_transport import ChatRequest, OllamaChatTransport, OpenAIChatTransport


class Content:
    def __init__(self, lines):
        self.lines = lines

    async def __aiter__(self):
        for line in self.lines:
            yield line

    async def read(self):
        return b""


def response(lines):
    return SimpleNamespace(content=Content(lines))


def event(content=None, finish_reason=None):
    choice = {"delta": {"content": content}, "finish_reason": finish_reason}
    return b"data: " + json.dumps({"choices": [choice]}, ensure_ascii=False).encode() + b"\n\n"


async def test_openai_eof_without_completion_is_not_success():
    req = ChatRequest("model", "system", "prompt", 20)
    with pytest.raises(RuntimeError, match="complete"):
        await OpenAIChatTransport("http://unused")._read_events(response([event("partial")]), req, 0)


@pytest.mark.parametrize("kind", ["openai", "ollama"])
async def test_stream_limit_counts_utf8_bytes_before_emitting(kind):
    pieces = []
    req = ChatRequest("model", "system", "prompt", 20, max_response_bytes=3, on_delta=pieces.append)
    if kind == "openai":
        lines = [event("\U0001f680"), b"data: [DONE]\n"]
        read = OpenAIChatTransport("http://unused")._read_events
    else:
        lines = [json.dumps({"message": {"content": "\U0001f680"}, "done": True}, ensure_ascii=False).encode()]
        read = OllamaChatTransport("http://unused")._read_stream
    with pytest.raises(RuntimeError, match="size limit"):
        await read(response(lines), req, 0)
    assert pieces == []


async def test_compatible_finish_reason_without_done_marker_is_complete():
    req = ChatRequest("model", "system", "prompt", 20)
    result = await OpenAIChatTransport("http://unused")._read_events(
        response([event("complete"), event(finish_reason="stop")]), req, 0)
    assert result.text == "complete"


@pytest.mark.parametrize("ending", [b"", b'data: {"error":{"message":"quota exhausted"}}\n'])
async def test_live_router_rejects_incomplete_or_error_http_streams(ending):
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter
    async def reply(_):
        return web.Response(body=event("partial") + ending, content_type="text/event-stream")
    app = web.Application()
    app.router.add_post("/v1/chat/completions", reply)
    async with TestServer(app) as server:
        profile = AIProviderProfile(name="probe", kind="openai_compatible", model="test",
                                    base_url=str(server.make_url("/v1")), api_key="local")
        settings = OrionProviderSettings(active_provider="probe", provider_order=["probe"],
                                         providers={"probe": profile})
        router = ProviderRouter(settings, SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)), object())
        with pytest.raises(RuntimeError, match="complete|quota"):
            await router._openai_compatible_chat_stream(profile, "probe", instruction="system")
