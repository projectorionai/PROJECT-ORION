"""Real routing, permission grounding and host-generated phone speech contracts."""

import asyncio
import io
import wave
from types import SimpleNamespace

from orion_core.data import ToolResult
from orion_core.remote import RemoteGateway
from orion_core.remote_capability import RemoteConfirmationRegistry, RemoteToolGate
from orion_core.remote_speech import SpeechUnavailable, audio_to_wav
from orion_core.remote_tasks import is_action_request, notepad_plan
from tests.test_remote_hardware_audio import _Bus, _Memory, _client, _token


def test_charon_audio_validates_pcm_and_rejects_truncation():
    audio = audio_to_wav(b"\x00\x01" * 100, "audio/L16;rate=24000")
    with wave.open(io.BytesIO(audio), "rb") as decoded:
        assert decoded.getframerate() == 24000
        assert decoded.getnframes() == 100
    for data, mime in ((b"odd", "audio/l16"), (audio[:-30], "audio/wav"),
                       (b"\x00\x00", "audio/mp3")):
        try:
            audio_to_wav(data, mime)
        except SpeechUnavailable:
            pass
        else:
            raise AssertionError("Invalid speech audio was accepted")


def test_paired_speech_auth_validation_and_busy_cleanup(tmp_path, monkeypatch):
    async def flow():
        from orion_core import remote_speech
        calls = []
        async def speak(router, text):
            calls.append(text)
            return audio_to_wav(b"\x00\x00" * 100, "audio/l16;rate=24000")
        monkeypatch.setattr(remote_speech, "synthesise_charon", speak)
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        client = await _client(gateway)
        try:
            assert (await client.post("/api/speech", json={"text": "Hello"})).status == 401
            token = await _token(client, gateway)
            headers = {"Authorization": "Bearer " + token}
            assert (await client.post("/api/speech", headers=headers, json={"text": 23})).status == 400
            response = await client.post("/api/speech", headers=headers, json={"text": "Hello"})
            assert response.status == 200
            assert response.headers["X-Orion-Voice"] == "Charon"
            assert (await response.read()).startswith(b"RIFF")
            assert calls == ["Hello"]
            assert not gateway._speech_devices
        finally:
            await client.close()
    asyncio.run(flow())


def test_notepad_task_is_reviewed_as_a_whole_and_stops_on_failure():
    class Desktop:
        TOOL_DECLARATIONS = [{"name": "open_app"}, {"name": "desktop_control"}]
        def __init__(self): self.calls = []
        async def dispatch(self, tool, args):
            self.calls.append((tool, args))
            return ToolResult("Verified text" if tool == "desktop_control" else "Opened")
    async def flow():
        desktop = Desktop()
        registry = RemoteConfirmationRegistry()
        gate = RemoteToolGate(desktop, registry)
        steps = notepad_plan('Open Notepad and write "Remote control works"')
        with gate.for_device("owner"):
            result = await gate.dispatch_plan(steps)
        assert "Nothing has run yet" in result.text
        assert desktop.calls == []
        pending = registry.pending_for("owner")[0]
        approved = registry.resolve(pending.id, pending.token, True, device_id="owner")
        result = await gate.run_approved(approved)
        assert result.ok and "2. desktop_control: Verified text" in result.text
        assert len(desktop.calls) == 2
        assert not (await gate.run_approved(approved)).ok

        async def fail(tool, args):
            desktop.calls.append((tool, args))
            return ToolResult("Launch failed", ok=False)
        desktop.calls.clear()
        desktop.dispatch = fail
        with gate.for_device("owner"):
            await gate.dispatch_plan(steps)
        pending = registry.pending_for("owner")[0]
        approved = registry.resolve(pending.id, pending.token, True, device_id="owner")
        result = await gate.run_approved(approved)
        assert not result.ok and "later steps have not run" in result.text
        assert len(desktop.calls) == 1
    asyncio.run(flow())


def test_phone_capabilities_come_from_this_devices_live_status(tmp_path):
    gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
    gateway.mobile_viewport = SimpleNamespace(device_status=lambda device: {
        "connected": True, "accessibility": False, "microphone": True,
    } if device == "owner" else {"connected": False})
    reply, source = asyncio.run(gateway._answer("What can I do on my phone?", "owner"))
    assert source == "phone-capabilities"
    assert "ACCESS: off" in reply and "microphone permission: on" in reply
    assert "opening apps" in reply
    assert "disconnected" in gateway._phone_capabilities("other")
    assert is_action_request('Please open Notepad and write "Hello"')
    assert not is_action_request("What can I do on my phone?")


def test_quick_tunnel_phone_actions_are_private_and_delivered_once(tmp_path):
    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        client = await _client(gateway)
        try:
            token = await _token(client, gateway)
            owner = gateway.auth.verify_access(token)
            event = {"type": "action", "action": {"kind": "map", "query": "Birmingham"}}
            gateway._push_device_event(owner, event)
            assert (await client.get("/api/phone-actions")).status == 401
            headers = {"Authorization": "Bearer " + token}
            response = await client.get("/api/phone-actions", headers=headers)
            assert (await response.json())["events"] == [event]
            response = await client.get("/api/phone-actions", headers=headers)
            assert (await response.json())["events"] == []
        finally:
            await client.close()
    asyncio.run(flow())
