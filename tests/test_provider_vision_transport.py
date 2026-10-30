"""Verify multimodal HTTP payloads and literal evidence against a local server."""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter


class _Remote(AIProviderProfile):
    """Routed as a cloud provider; served from 127.0.0.1."""

    is_local = property(lambda self: False)


def _run(body: bytes, *, native: bool = False):
    sent = []

    async def openai(request):
        sent.append(await request.json())
        return web.Response(body=body, content_type="application/json")

    async def ollama_version(_request):
        return web.json_response({"version": "0.32.6"})

    async def ollama_chat(request):
        sent.append(await request.json())
        reply = json.loads(body)["choices"][0]["message"]["content"] if len(body) < 262144 else body.decode()
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        await response.write(json.dumps({"message": {"content": reply}, "done": False}).encode() + b"\n")
        await response.write(json.dumps({"message": {"content": ""}, "done": True,
                                         "eval_count": 1, "prompt_eval_count": 1}).encode() + b"\n")
        await response.write_eof()
        return response

    async def flow():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", openai)
        app.router.add_get("/api/version", ollama_version)
        app.router.add_post("/api/chat", ollama_chat)
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        try:
            base = str(server.make_url("/v1"))
            if native:
                profile = AIProviderProfile(name="local_ollama", kind="openai_compatible",
                                            model="llava", base_url=base, api_key="local")
            else:
                profile = _Remote(name="test", kind="openai_compatible", model="gpt-4o-mini",
                                  base_url=base, api_key="fixture")
            bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
            router = ProviderRouter(OrionProviderSettings(
                active_provider=profile.name, provider_order=[profile.name],
                providers={profile.name: profile}), bus, memory=object())
            try:
                return await router._openai_compatible_chat(
                    profile, "Inspect", instruction="JSON", image_jpeg=b"fixture-image"), sent
            finally:
                await router.aclose()
        finally:
            await server.close()

    return asyncio.run(flow())


RAW_REPORT = json.dumps({"summary": "ORIN marking visible", "observations": []})
BODY = json.dumps({"choices": [{"message": {"content": RAW_REPORT}}]}).encode()


def test_image_is_sent_as_pixels_and_component_markings_are_not_rewritten():
    result, sent = _run(BODY)
    assert result == RAW_REPORT
    content = sent[0]["messages"][1]["content"]
    assert content[1]["image_url"]["url"] == \
        "data:image/jpeg;base64," + base64.b64encode(b"fixture-image").decode()
    assert sent[0]["max_tokens"] > 0


def test_an_oversized_vision_reply_is_refused():
    with pytest.raises(RuntimeError, match="size limit"):
        _run(b"x" * 262145)


def test_ollama_receives_the_image_in_its_native_field():
    result, sent = _run(BODY, native=True)
    assert result == RAW_REPORT                       # evidence kept literally
    user = sent[0]["messages"][1]
    assert user["content"] == "Inspect"
    assert user["images"] == [base64.b64encode(b"fixture-image").decode()]
