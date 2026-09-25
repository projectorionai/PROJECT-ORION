"""Verify multimodal HTTP payloads and literal evidence without a network."""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter


def test_image_is_sent_as_pixels_and_component_markings_are_not_rewritten(monkeypatch):
    async def run(oversized=False):
        raw_report = json.dumps({"summary": "ORIN marking visible", "observations": []})
        body = json.dumps({"choices": [{"message": {"content": raw_report}}]}).encode()
        if oversized:
            body = b"x" * 262145
        sent = []

        class Response:
            status = 200

            def __init__(self):
                self.content = self

            async def iter_chunked(self, size):
                for start in range(0, len(body), size):
                    yield body[start:start + size]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, endpoint, **kwargs):
                sent.append(kwargs["json"])
                return Response()

        import orion_core.providers as module
        monkeypatch.setattr(module, "ClientSession", lambda **kwargs: Session())
        bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
        profile = AIProviderProfile(name="test", kind="openai_compatible", model="gpt-4o-mini",
                                    base_url="https://fixture.invalid/v1", api_key="fixture")
        router = ProviderRouter(OrionProviderSettings(active_provider="test", provider_order=["test"],
                                                      providers={"test": profile}), bus, memory=object())
        result = await router._openai_compatible_chat(profile, "Inspect", instruction="JSON",
                                                     image_jpeg=b"fixture-image")
        assert result == raw_report
        content = sent[0]["messages"][1]["content"]
        assert content[1]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"fixture-image").decode()
        assert sent[0]["max_tokens"] > 0

    asyncio.run(run())
    with pytest.raises(RuntimeError, match="size limit"):
        asyncio.run(run(oversized=True))
