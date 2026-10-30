"""Regressions for real task creation, complete validation and visible control results."""

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.remote_capability import RemoteConfirmationRegistry, RemoteToolGate
from orion_core.remote_tasks import is_action_request


def test_invalid_later_step_prevents_an_earlier_action():
    class Desktop:
        TOOL_DECLARATIONS = [{"name": "open_app", "parameters": {
            "type": "OBJECT", "properties": {"app_name": {"type": "STRING"}},
            "required": ["app_name"]}}]
        async def dispatch(self, *_args):
            raise AssertionError("An invalid whole task must never dispatch")
    async def flow():
        gate = RemoteToolGate(Desktop(), RemoteConfirmationRegistry())
        with gate.for_device("phone"):
            result = await gate.dispatch_plan([
                {"tool": "open_app", "args": {"app_name": "notepad"}},
                {"tool": "open_app", "args": {}},
            ])
        assert not result.ok and "missing required" in result.text
        assert not gate.confirmations.pending_for("phone")
    asyncio.run(flow())


def test_politely_worded_commands_are_actions():
    for message in ('Hey ORION, could you please open Notepad?',
                    'I want you to write "Hello" in Notepad', 'Please pause the music'):
        assert is_action_request(message)
    assert not is_action_request("Why did you open Notepad yesterday?")


def test_tasks_composer_creates_a_durable_plan_without_replaying_a_step():
    from orion_core.admission import Admission
    from orion_core.gui.tasks_view import TasksView
    from orion_core.job_store import JobStore
    from orion_core.plan_executor import PlanExecutor
    app = QApplication.instance() or QApplication([])
    async def flow():
        store = JobStore(None)
        calls = []
        class Desktop:
            TOOL_DECLARATIONS = [{"name": "open_app"}, {"name": "desktop_control"}]
            async def dispatch(self, tool, args):
                calls.append((tool, args))
                return ToolResult("Verified result")
        desktop = Desktop()
        bus = OrionBus()
        desktop.plan_executor = PlanExecutor(bus, desktop.dispatch, store=store,
                                              admission=Admission(quiet_s=0))
        jobs = SimpleNamespace(snapshot=lambda **kw: store.snapshot(**kw))
        view = TasksView(bus, jobs, desktop)
        try:
            view.task_entry.setText('Open Notepad and write "Task submission works"')
            view._prepare_task()
            for _ in range(10):
                await asyncio.sleep(0)
            assert view.run_button.isEnabled()
            assert not calls
            view._start_task()
            for _ in range(30):
                await asyncio.sleep(0)
            assert [tool for tool, _ in calls] == ["open_app", "desktop_control"]
            records = store.snapshot()["jobs"]
            assert len(records) == 1 and records[0]["state"] == "completed"
            assert records[0]["total"] == 2
            view._start_task()
            assert len(calls) == 2
        finally:
            view.close()
            store.close()
    asyncio.run(flow())
    assert app is not None


def test_chess_action_failures_are_shown_and_buttons_reenabled():
    from orion_core.gui.chess_remote import RemoteChessWidget
    app = QApplication.instance() or QApplication([])
    async def flow():
        view = RemoteChessWidget(SimpleNamespace())
        async def unavailable():
            return ToolResult("Lichess needs a bot token", ok=False)
        try:
            view._launch(unavailable(), button=view.connect_button)
            assert not view.connect_button.isEnabled()
            for _ in range(5):
                await asyncio.sleep(0)
            assert view.connect_button.isEnabled()
            assert "Failed: Lichess needs a bot token" in view.label.text()
            assert "bot token" in view.history.toPlainText()
        finally:
            view.close()
    asyncio.run(flow())
    assert app is not None


def test_remote_challenge_authorises_the_named_opponent_before_challenging():
    from orion_core.gui.chess_remote import RemoteChessWidget
    app = QApplication.instance() or QApplication([])
    calls = []

    async def authorise(handle, entity, *, elo):
        calls.append(("authorise", handle, entity, elo))

    async def challenge(username, *, rated):
        calls.append(("challenge", username, rated))
        return ToolResult("Challenge sent")

    service = SimpleNamespace(
        lichess=SimpleNamespace(authorise=authorise),
        lichess_challenge=challenge,
    )

    async def flow():
        view = RemoteChessWidget(service)
        try:
            view.username.setText("Artemis_42")
            view._challenge()
            for _ in range(5):
                await asyncio.sleep(0)
            assert calls == [
                ("authorise", "Artemis_42", "Artemis_42", 2604),
                ("challenge", "Artemis_42", False),
            ]
            assert view.challenge_button.isEnabled()
            assert "Challenge sent" in view.history.toPlainText()
        finally:
            view.close()

    asyncio.run(flow())
    assert app is not None


def test_operations_has_no_second_security_panel():
    from orion_core.gui.ops_deck import OperationsDeckView
    app = QApplication.instance() or QApplication([])
    view = OperationsDeckView(OrionBus())
    try:
        assert not hasattr(view, "security_panel")
    finally:
        view.close()
    assert app is not None


def test_background_wake_word_can_precede_the_request(tmp_path):
    from orion_core.remote import RemoteGateway
    from tests.test_remote_hardware_audio import _Memory, _Bus, _client, _token
    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        transcripts = iter(["Hey Orion", "check status"])
        gateway._transcribe_blocking = lambda _wav: (next(transcripts), "")
        received = []
        async def answer(text, device, *, on_widget=None):
            received.append(text)
            return "Ready", "test"
        gateway._answer = answer
        client = await _client(gateway)
        try:
            token = await _token(client, gateway)
            socket = await client.ws_connect("/v1/audio/stream", headers={"Authorization": "Bearer " + token})
            await socket.send_bytes(b"\x00\x00" * 1600)
            await socket.send_json({"type": "commit"})
            assert (await socket.receive_json(timeout=2))["type"] == "transcript"
            assert (await socket.receive_json(timeout=2))["type"] == "status"
            await asyncio.sleep(0)
            await socket.send_bytes(b"\x00\x00" * 1600)
            await socket.send_json({"type": "commit"})
            assert (await socket.receive_json(timeout=2))["type"] == "transcript"
            assert (await socket.receive_json(timeout=2))["text"] == "Ready"
            assert received == ["check status"]
            await socket.close()
        finally:
            await client.close()
    asyncio.run(flow())


def test_audio_diagnostics_are_paired_private_and_count_the_real_stream(tmp_path):
    from orion_core.remote import RemoteGateway
    from tests.test_remote_hardware_audio import _Memory, _Bus, _client, _token
    async def flow():
        gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
        gateway._transcribe_blocking = lambda _wav: ("Orion hello", "")
        async def answer(text, device, *, on_widget=None):
            return "Hello", "test"
        gateway._answer = answer
        client = await _client(gateway)
        try:
            assert (await client.get("/api/audio/status")).status == 401
            token = await _token(client, gateway)
            headers = {"Authorization": "Bearer " + token}
            other_token = await _token(client, gateway)
            socket = await client.ws_connect("/v1/audio/stream", headers=headers)
            await socket.send_bytes(b"\x00\x00" * 1600)
            await socket.send_json({"type": "commit"})
            assert (await socket.receive_json(timeout=2))["type"] == "transcript"
            assert (await socket.receive_json(timeout=2))["type"] == "tts_text"
            response = await client.get("/api/audio/status", headers=headers)
            data = await response.json()
            assert response.headers["Cache-Control"] == "no-store"
            assert data["connected"] and data["pcm_bytes"] == 3200
            assert data["utterances"] == data["transcriptions"] == data["requests"] == 1
            assert "hello" not in str(data).lower()
            other = await (await client.get("/api/audio/status", headers={
                "Authorization": "Bearer " + other_token})).json()
            assert not other["connected"] and other["pcm_bytes"] == 0
            await socket.close()
        finally:
            await client.close()
    asyncio.run(flow())


def test_mobile_starts_with_terminal_and_listening_box_and_hidden_pc_statistics():
    from orion_core.remote import REMOTE_PAGE_HTML
    from html.parser import HTMLParser
    class Sections(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sections = {}
        def handle_starttag(self, tag, attrs):
            props = dict(attrs)
            if tag == "section" and "id" in props:
                self.sections[props["id"]] = props
    page = Sections()
    page.feed(REMOTE_PAGE_HTML)
    assert "hidden" in page.sections["hardware"]
    assert page.sections["stage"]["aria-label"] == "ORION listening box"
    assert "hidden||document.hidden" in REMOTE_PAGE_HTML
