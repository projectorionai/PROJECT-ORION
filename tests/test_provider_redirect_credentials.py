"""A hosted model must never forward its API key through an HTTP redirect."""

import asyncio
from types import SimpleNamespace

import pytest

from orion_core.provider_transport import _send
from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter


def test_provider_request_leaves_a_redirect_for_the_caller():
    calls = []

    class Session:
        async def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            class Response:
                status = 302

                def close(self):
                    calls.append("closed")

            return Response()

    with pytest.raises(RuntimeError, match="redirected"):
        asyncio.run(_send(
            Session(), "POST", "https://provider.example/chat/completions",
            trace={}, headers={"Authorization": "Bearer secret"}))

    assert len(calls) == 2
    assert calls[0][2]["allow_redirects"] is False
    assert calls[1] == "closed"


def test_model_discovery_does_not_forward_a_provider_key():
    calls = []

    class Response:
        status = 302

        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass

    class Session:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    async def session():
        return Session()

    profile = AIProviderProfile(
        name="probe", kind="openai_compatible", model="test",
        base_url="https://provider.example/v1", api_key="secret")
    settings = OrionProviderSettings(
        active_provider="probe", provider_order=["probe"],
        providers={"probe": profile})
    router = ProviderRouter(
        settings, SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)), object())
    router._pool = SimpleNamespace(get=session)

    assert asyncio.run(router._list_models(profile)) == []
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.parametrize("url, expected", [
    ("http://127.0.0.1:11434/v1", True),
    ("http://localhost:1234/v1", True),
    ("http://[::1]:11434/v1", True),
    ("http://localhost.example.com/v1", False),
    ("http://127.0.0.1@outside.example/v1", False),
    ("https://localhost:1234/v1", False),
])
def test_local_provider_classification_uses_the_url_hostname(url, expected):
    profile = AIProviderProfile(name="probe", kind="openai_compatible",
                                model="test", base_url=url)
    assert profile.is_local is expected
