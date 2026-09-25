"""Neither thought transport may silently spend money without opt-in."""
import asyncio
from types import SimpleNamespace

import pytest

from orion_core import local_mind
from orion_core.providers import ProviderRouter, NoTextProviderError


def router(profiles):
    instance = ProviderRouter.__new__(ProviderRouter)
    instance._thought_profiles = lambda prompt: profiles
    instance.mark_failure = lambda *args: None
    calls = []
    async def chat(profile, *args, **kwargs):
        calls.append(profile.name)
        return "a reflection"
    instance._openai_compatible_chat = chat
    instance._openai_compatible_chat_stream = chat
    return instance, calls


@pytest.mark.parametrize("stream", [False, True])
def test_no_cloud_thought_call_without_opt_in(monkeypatch, stream):
    monkeypatch.delenv("ORION_PAID_THOUGHTS", raising=False)
    instance, calls = router([SimpleNamespace(name="cloud", is_local=False)])
    method = instance.generate_thought_stream if stream else instance.generate_thought
    with pytest.raises(NoTextProviderError):
        asyncio.run(method("reflect"))
    assert calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_local_model_is_used_and_cloud_requires_explicit_opt_in(monkeypatch, stream):
    monkeypatch.delenv("ORION_PAID_THOUGHTS", raising=False)
    instance, calls = router([SimpleNamespace(name="cloud", is_local=False), SimpleNamespace(name="local", is_local=True)])
    method = instance.generate_thought_stream if stream else instance.generate_thought
    asyncio.run(method("reflect"))
    assert calls == ["local"]
    monkeypatch.setenv("ORION_PAID_THOUGHTS", "1")
    asyncio.run(method("reflect"))
    assert calls == ["local", "cloud"]


def test_policy_failure_cannot_authorise_paid_work(monkeypatch):
    instance, calls = router([SimpleNamespace(name="cloud", is_local=False)])
    def broken():
        raise RuntimeError("configuration unavailable")
    monkeypatch.setattr(local_mind, "local_only_thoughts", broken)
    with pytest.raises(NoTextProviderError):
        asyncio.run(instance.generate_thought_stream("reflect"))
    assert calls == []
