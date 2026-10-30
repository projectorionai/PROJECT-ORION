"""
The local agent (orion_core/local_agent.py): a local model choosing and using
ORION's tools, with a safety gate the model cannot talk its way past.

Nothing here starts Ollama, ORION, a microphone or the network. The model is a
scripted fake (ChatResult objects, or scripted NDJSON chunks for the
transport), and the dispatcher is a fake that records calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import local_agent as la  # noqa: E402
from orion_core import provider_transport as pt  # noqa: E402
from orion_core.data import ToolResult  # noqa: E402
from orion_core.local_agent import (  # noqa: E402
    ACT, HIGH, READ, LocalAgent, classify_risk, is_affirmative, is_negative, open_ended,
    to_ollama_tool, to_json_schema)

# ── fixtures ────────────────────────────────────────────────────────────────

DECLS = [
    {"name": "open_app", "description": "Launch a host application. " * 30,
     "parameters": {"type": "OBJECT", "properties": {
         "app_name": {"type": "STRING", "description": "What to open."}},
         "required": ["app_name"]}},
    {"name": "resource_status", "description": "Report CPU and memory.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "shutdown_orion", "description": "Shut ORION down.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "process_governor", "description": "Monitor and terminate processes.",
     "parameters": {"type": "OBJECT", "properties": {
         "action": {"type": "STRING", "description": "top, list, terminate, or restart."},
         "name": {"type": "STRING"},
         "confirm": {"type": "BOOLEAN", "description": "Must be true to terminate."}}}},
    {"name": "messaging", "description": "Send a chat message.",
     "parameters": {"type": "OBJECT", "properties": {
         "action": {"type": "STRING"}, "contact": {"type": "STRING"},
         "message": {"type": "STRING"}}, "required": ["action"]}},
    {"name": "desktop_control", "description": "Type and click.",
     "parameters": {"type": "OBJECT", "properties": {
         "action": {"type": "STRING"}, "text": {"type": "STRING"}}}},
    {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}},
                    "required": ["query"]}},
    {"name": "use_tool", "description": "Run any tool by name.",
     "parameters": {"type": "OBJECT", "properties": {"name": {"type": "STRING"}}}},
]


def _scorer(text, declarations):
    """Every offered tool scores, in declared order (deterministic shortlist)."""
    return {d["name"]: 10.0 - i * 0.1 for i, d in enumerate(declarations)}


class FakeChat:
    """Scripted model replies: a ChatResult (or exception) per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def __call__(self, messages, tools, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "tools": [t["function"]["name"] for t in tools], **kwargs})
        if not self.replies:
            return pt.ChatResult(text="Nothing more to add.")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _call(tool_name, **args):
    return pt.ChatResult(text="", tool_calls=[{"name": tool_name, "args": args, "id": ""}],
                         first_token_ms=5.0)


def _say(text):
    return pt.ChatResult(text=text, first_token_ms=5.0)


class FakeDispatch:
    def __init__(self, results=None):
        self.calls: list[tuple[str, dict]] = []
        self.voice: list[bool] = []
        self.results = results or {}

    async def __call__(self, name, args):
        from orion_core import speaker_gate
        self.calls.append((name, dict(args)))
        self.voice.append(speaker_gate.VOICE_TURN.get())
        result = self.results.get(name)
        if callable(result):
            return result(args)
        return result or ToolResult(f"[{name}] {name} done.")


def _agent(chat, dispatch=None, **kw):
    return LocalAgent(None, None, chat=chat, dispatch=dispatch or FakeDispatch(),
                      declarations=lambda: DECLS, scorer=_scorer, **kw)


def run(coro):
    return asyncio.run(coro)


# ── schema conversion ───────────────────────────────────────────────────────

def test_gemini_declarations_become_json_schema_without_approval_fields():
    tool = to_ollama_tool(DECLS[3])
    fn = tool["function"]
    assert tool["type"] == "function" and fn["name"] == "process_governor"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["action"]["type"] == "string"
    assert "confirm" not in params["properties"]          # the model never sees approval
    assert len(to_ollama_tool(DECLS[0])["function"]["description"]) <= la.DESCRIPTION_CHARS + 1
    assert to_ollama_tool(DECLS[0])["function"]["parameters"]["required"] == ["app_name"]


def test_enums_items_and_nested_objects_convert():
    schema = to_json_schema({"type": "OBJECT", "properties": {
        "mode": {"type": "STRING", "enum": ["a", "b"]},
        "list": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "inner": {"type": "OBJECT", "properties": {"x": {"type": "NUMBER"},
                                                   "consent": {"type": "BOOLEAN"}},
                  "required": ["x", "consent"]}},
        "required": ["mode", "missing"]})
    assert schema["properties"]["mode"]["enum"] == ["a", "b"]
    assert schema["properties"]["list"] == {"type": "array", "items": {"type": "integer"}}
    inner = schema["properties"]["inner"]
    assert inner["properties"] == {"x": {"type": "number"}} and inner["required"] == ["x"]
    assert schema["required"] == ["mode"]


def test_every_real_declaration_converts_to_lowercase_json_schema():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    def walk(node):
        if isinstance(node, dict):
            if "type" in node and isinstance(node["type"], str):
                assert node["type"] == node["type"].lower(), node
            for key, value in node.items():
                if key == "properties":
                    assert not any(la.is_approval_key(k) for k in value), value.keys()
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for declaration in TOOL_DECLARATIONS:
        tool = to_ollama_tool(declaration)
        json.dumps(tool)
        walk(tool["function"]["parameters"])


# ── the Ollama tool transport (scripted NDJSON) ─────────────────────────────

class _Content:
    def __init__(self, lines):
        self._lines = [json.dumps(line).encode() + b"\n" if not isinstance(line, bytes)
                       else line for line in lines]

    def __aiter__(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()

    async def read(self, _n=-1):
        return b""


class _Response:
    def __init__(self, lines, status=200):
        self.status = status
        self.headers = {"Content-Type": "application/x-ndjson"}
        self.content = _Content(lines)
        self.closed = False

    def close(self):
        self.closed = True

    def release(self):
        pass


def test_streamed_ndjson_tool_calls_are_parsed_and_normalised():
    lines = [
        {"message": {"role": "assistant", "content": ""}, "done": False},
        {"message": {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "open_app",
                                                  "arguments": {"app_name": "notepad"}}}]},
         "done": False},
        {"message": {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "web_search",
                                                  "arguments": "{\"query\": \"rain\"}"}}]},
         "done": False},
        b"\n",
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 40, "eval_count": 9, "load_duration": 2_000_000},
    ]
    transport = pt.OllamaToolTransport("http://127.0.0.1:11434/v1")
    req = pt.ToolChatRequest(model="llama3.1", messages=[{"role": "user", "content": "x"}])
    result = run(transport._read_stream(_Response(lines), req, 0.0))
    assert [c["name"] for c in result.tool_calls] == ["open_app", "web_search"]
    assert result.tool_calls[0]["args"] == {"app_name": "notepad"}
    assert result.tool_calls[1]["args"] == {"query": "rain"}
    assert result.first_token_ms is not None
    assert result.usage["completion_tokens"] == 9 and result.load_ms == 2.0


def test_tool_payload_carries_tools_history_context_and_keep_alive():
    transport = pt.OllamaToolTransport("http://127.0.0.1:11434")
    req = pt.ToolChatRequest(model="llama3.1", messages=[{"role": "system", "content": "s"},
                                                         {"role": "user", "content": "u"}],
                             tools=[to_ollama_tool(DECLS[0])], num_ctx=8192,
                             keep_alive="10m", temperature=0.2, max_tokens=300)
    payload = transport.payload(req)
    assert transport.endpoint.endswith("/api/chat")
    assert payload["tools"][0]["function"]["name"] == "open_app"
    assert payload["options"] == {"temperature": 0.2, "num_predict": 300, "num_ctx": 8192}
    assert payload["keep_alive"] == "10m" and payload["stream"] is True
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_an_endless_stream_of_tool_calls_is_bounded():
    call = {"function": {"name": "open_app", "arguments": {"app_name": "x"}}}
    lines = [{"message": {"content": "", "tool_calls": [call]}, "done": False}] * 20
    transport = pt.OllamaToolTransport("http://127.0.0.1:11434")
    req = pt.ToolChatRequest(model="m", messages=[])
    with pytest.raises(RuntimeError, match="size limit"):
        run(transport._read_stream(_Response(lines), req, 0.0))


def test_router_local_tool_chat_uses_native_ollama_and_records_residency(monkeypatch):
    from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter

    profile = AIProviderProfile(name="local_ollama", kind="openai_compatible",
                                model="llama3.1:latest", base_url="http://127.0.0.1:11434/v1",
                                api_key="local", enabled=True)
    settings = OrionProviderSettings(active_provider="local_ollama",
                                     provider_order=["local_ollama"],
                                     providers={"local_ollama": profile})
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_a: None),
                          dashboard_event=SimpleNamespace(emit=lambda *_a: None))
    router = ProviderRouter(settings, bus, memory=SimpleNamespace(prompt_context=lambda *_: ""))
    sent = {}

    class Session:
        async def request(self, method, url, **kwargs):
            sent.update(url=url, json=kwargs["json"])
            return _Response([
                {"message": {"content": "", "tool_calls": [
                    {"function": {"name": "open_app", "arguments": {"app_name": "notepad"}}}]},
                 "done": False},
                {"message": {"content": ""}, "done": True, "done_reason": "stop"}])

    async def is_ollama(root, scope=None):
        return True

    async def pool_get():
        return Session()

    monkeypatch.setattr(router, "_is_ollama", is_ollama)
    monkeypatch.setattr(router._pool, "get", pool_get)

    async def flow():
        try:
            return await router.local_tool_chat(
                [{"role": "user", "content": "open notepad"}], [to_ollama_tool(DECLS[0])],
                deadline_s=10.0, keep_alive="10m")
        finally:
            router._resident_snapshot = dict(router._resident_until)
            await router.aclose()

    _profile, result = run(flow())
    assert sent["url"].endswith("/api/chat")
    assert sent["json"]["tools"][0]["function"]["name"] == "open_app"
    assert sent["json"]["keep_alive"] == "600s"
    assert sent["json"]["options"]["num_ctx"] == ProviderRouter.LOCAL_AGENT_NUM_CTX == 8192
    assert result.tool_calls[0]["name"] == "open_app"
    # Residency is recorded, so shutdown's unload_local_models releases it.
    assert ("http://127.0.0.1:11434", "llama3.1:latest") in router._resident_snapshot


def test_router_refuses_tool_turns_without_a_local_model():
    from orion_core.providers import NoTextProviderError, OrionProviderSettings, ProviderRouter
    router = ProviderRouter(OrionProviderSettings(active_provider="", provider_order=[],
                                                  providers={}),
                            SimpleNamespace(log=SimpleNamespace(emit=lambda *_a: None)),
                            memory=SimpleNamespace(prompt_context=lambda *_: ""))
    with pytest.raises(NoTextProviderError):
        run(router.local_tool_chat([{"role": "user", "content": "x"}], []))


# ── shortlist ───────────────────────────────────────────────────────────────

def test_shortlist_is_bounded_and_never_offers_meta_tools():
    agent = _agent(FakeChat())
    names = [d["name"] for d in agent.shortlist("anything")]
    assert "use_tool" not in names
    assert la.MIN_SHORTLIST <= agent.shortlist_size <= la.MAX_SHORTLIST
    assert len(names) <= agent.shortlist_size


def test_shortlist_with_the_real_resolver_ranks_the_obvious_tool():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.tool_resolver import lexical_scores
    agent = LocalAgent(None, None, chat=FakeChat(), declarations=lambda: TOOL_DECLARATIONS,
                       scorer=lexical_scores)
    names = [d["name"] for d in agent.shortlist("open spotify")]
    assert "open_app" in names and len(names) <= la.MAX_SHORTLIST
    assert not set(names) & la.NEVER_OFFER


def test_cloud_ai_tools_are_withheld_from_the_local_only_brain(monkeypatch):
    from orion_core import brain_mode
    decls = DECLS + [{"name": "antigravity_research", "description": "Gemini research.",
                      "parameters": {"type": "OBJECT", "properties": {}}}]
    agent = LocalAgent(None, None, chat=FakeChat(), declarations=lambda: decls, scorer=_scorer)
    assert "antigravity_research" in [d["name"] for d in agent.shortlist("research")]
    monkeypatch.setenv("ORION_BRAIN_MODE", "local")
    assert brain_mode.effective_mode() == "local"
    assert "antigravity_research" not in [d["name"] for d in agent.shortlist("research")]


# ── the loop ────────────────────────────────────────────────────────────────

def test_an_action_runs_through_the_dispatcher_and_its_own_words_answer():
    chat = FakeChat(_call("open_app", app_name="notepad"))
    dispatch = FakeDispatch({"open_app": ToolResult("[open_app] Opening Notepad.")})
    reply = run(_agent(chat, dispatch).respond("open notepad"))
    assert dispatch.calls == [("open_app", {"app_name": "notepad"})]
    assert reply.text == "Opening Notepad." and reply.handled
    assert len(chat.calls) == 1                 # no extra generation after a clean action
    assert chat.calls[0]["keep_alive"] is None and chat.calls[0]["num_ctx"] == 8192
    assert chat.calls[0]["temperature"] == 0.2
    assert reply.tool_calls[0]["status"] == "ran" and reply.tool_calls[0]["risk"] == ACT
    assert {"first_token_ms", "total_ms", "model_ms", "tools_ms"} <= set(reply.timings)


def test_a_lookup_goes_back_to_the_model_for_a_short_answer():
    chat = FakeChat(_call("resource_status"), _say("Your CPU is at 23 percent."))
    dispatch = FakeDispatch({"resource_status": ToolResult("CPU 23%\nRAM 61%")})
    reply = run(_agent(chat, dispatch).respond("how busy is my pc"))
    assert reply.text == "Your CPU is at 23 percent."
    tool_message = chat.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_name"] == "resource_status"
    assert json.loads(tool_message["content"]) == {"ok": True, "result": "CPU 23%\nRAM 61%"}
    assert chat.calls[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "resource_status"


def test_rounds_are_bounded_and_the_last_call_offers_no_tools():
    replies = [_call("web_search", query=f"q{i}") for i in range(10)]
    chat = FakeChat(*replies)
    dispatch = FakeDispatch()
    agent = _agent(chat, dispatch, max_rounds=3)
    reply = run(agent.respond("search for several things"))
    assert len(dispatch.calls) == 3
    assert len(chat.calls) == 4 and chat.calls[-1]["tools"] == []
    assert reply.text


def test_the_total_deadline_stops_further_rounds():
    clock = SimpleNamespace(t=0.0)

    class SlowChat(FakeChat):
        async def __call__(self, messages, tools, **kwargs):
            clock.t += 4.0
            return await super().__call__(messages, tools, **kwargs)

    chat = SlowChat(*[_call("web_search", query=f"q{i}") for i in range(10)])
    agent = _agent(chat, FakeDispatch(), deadline_s=10.0, clock=lambda: clock.t)
    run(agent.respond("keep searching"))
    assert len(chat.calls) <= 3
    assert all(call["deadline_s"] <= 10.0 for call in chat.calls)


def test_cancellation_propagates_and_nothing_is_dispatched():
    started = asyncio.Event()

    class Hanging(FakeChat):
        async def __call__(self, messages, tools, **kwargs):
            started.set()
            await asyncio.Event().wait()

    dispatch = FakeDispatch()

    async def flow():
        task = asyncio.ensure_future(_agent(Hanging(), dispatch).respond("open notepad"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(flow())
    assert dispatch.calls == []


def test_a_model_failure_is_reported_and_escalates_in_hybrid():
    from orion_core.providers import NoTextProviderError
    reply = run(_agent(FakeChat(NoTextProviderError("no local model is available")))
                .respond("open notepad", allow_escalate=True))
    assert reply.unavailable and reply.escalate and not reply.handled
    reply = run(_agent(FakeChat(TimeoutError("slow"))).respond("open notepad"))
    assert not reply.unavailable and "too long" in reply.reason


def test_no_success_is_claimed_without_a_successful_tool():
    claim = _say("I've opened Notepad for you.")
    local = run(_agent(FakeChat(claim)).respond("open notepad"))
    assert "opened" not in local.text.lower() and "haven't" in local.text
    hybrid = run(_agent(FakeChat(_say("I've opened Notepad for you.")))
                 .respond("open notepad", allow_escalate=True))
    assert hybrid.escalate and not hybrid.text


def test_a_failed_action_is_reported_as_failed():
    chat = FakeChat(_call("open_app", app_name="nonsense"), _say("Done, it is open."))
    dispatch = FakeDispatch({"open_app": ToolResult("[open_app] No app called nonsense.",
                                                    ok=False)})
    reply = run(_agent(chat, dispatch).respond("open nonsense"))
    assert reply.text.startswith("I couldn't do that.") and "nonsense" in reply.text
    assert "Done" not in reply.text


def test_a_call_the_model_invents_or_was_not_offered_never_runs():
    chat = FakeChat(_call("use_tool", name="shutdown_orion"),
                    _call("format_disk", drive="C"), _say("I can't do that."))
    dispatch = FakeDispatch()
    reply = run(_agent(chat, dispatch).respond("use the use_tool function"))
    assert dispatch.calls == []
    assert [c["status"] for c in reply.tool_calls] == ["rejected", "rejected"]


def test_a_call_written_as_json_text_is_recognised():
    chat = FakeChat(_say('{"name": "open_app", "parameters": {"app_name": "paint"}}'))
    dispatch = FakeDispatch()
    run(_agent(chat, dispatch).respond("open paint"))
    assert dispatch.calls == [("open_app", {"app_name": "paint"})]


def test_the_default_dispatch_is_the_real_dispatcher_chain():
    seen = []

    class Dispatcher:
        TOOL_DECLARATIONS = DECLS

        async def dispatch_chain(self, name, args):
            seen.append((name, args))
            return ToolResult(f"[{name}] Opening Notepad.")

    agent = LocalAgent(None, Dispatcher(), chat=FakeChat(_call("open_app", app_name="notepad")),
                       scorer=_scorer)
    reply = run(agent.respond("open notepad"))
    assert seen == [("open_app", {"app_name": "notepad"})] and reply.text == "Opening Notepad."


def test_a_spoken_turn_dispatches_under_the_owner_voice_gate():
    dispatch = FakeDispatch()
    run(_agent(FakeChat(_call("open_app", app_name="notepad")), dispatch)
        .respond("open notepad", origin="voice"))
    run(_agent(FakeChat(_call("open_app", app_name="paint")), dispatch)
        .respond("open paint", origin="text"))
    assert dispatch.voice == [True, False]


# ── the safety gate ─────────────────────────────────────────────────────────

def test_a_high_risk_call_waits_for_the_owner_and_yes_runs_it_once():
    chat = FakeChat(_call("shutdown_orion"), _say("Anything else?"))
    dispatch = FakeDispatch({"shutdown_orion": ToolResult("Shutting down.")})
    agent = _agent(chat, dispatch)
    first = run(agent.respond("shut yourself down"))
    assert first.pending and dispatch.calls == []
    assert "Shall I go ahead" in first.text and "shut ORION down" in first.text
    assert first.tool_calls[-1]["status"] == "pending_confirmation"
    second = run(agent.respond("yes"))
    assert dispatch.calls == [("shutdown_orion", {})] and second.text == "Shutting down."
    assert agent.pending is None
    run(agent.respond("yes"))                       # nothing left to release
    assert dispatch.calls == [("shutdown_orion", {})]


def test_anything_else_cancels_the_pending_call():
    chat = FakeChat(_call("messaging", action="send", contact="Sam", message="I quit"),
                    _say("It is ten past four."))
    dispatch = FakeDispatch()
    agent = _agent(chat, dispatch)
    assert run(agent.respond("text sam that i quit")).pending
    reply = run(agent.respond("what time is it"))
    assert agent.pending is None and dispatch.calls == []
    assert reply.text.startswith("I've cancelled the earlier request")
    assert run(agent.respond("yes")).text           # a later yes releases nothing
    assert dispatch.calls == []


def test_no_declines_the_pending_call():
    agent = _agent(FakeChat(_call("shutdown_orion")), FakeDispatch())
    run(agent.respond("shut down"))
    reply = run(agent.respond("no thanks"))
    assert reply.text.startswith("Understood") and agent.pending is None


def test_a_pending_call_lapses():
    clock = SimpleNamespace(t=0.0)
    dispatch = FakeDispatch()
    agent = _agent(FakeChat(_call("shutdown_orion"), _say("Yes?")), dispatch,
                   clock=lambda: clock.t)
    run(agent.respond("shut down"))
    clock.t += la.PENDING_TTL_S + 1
    run(agent.respond("yes"))
    assert dispatch.calls == []


def test_the_model_cannot_confirm_for_the_owner():
    chat = FakeChat(_call("process_governor", action="terminate", name="chrome", confirm=True),
                    _call("process_governor", action="terminate", name="chrome",
                          confirmed="yes", approve=True))
    dispatch = FakeDispatch()
    agent = _agent(chat, dispatch)
    reply = run(agent.respond("kill chrome"))
    assert reply.pending and dispatch.calls == []
    assert "confirm" not in agent.pending.args
    run(agent.respond("go ahead"))
    # The owner's yes is the agreement the tool asks for; only now is it given.
    assert dispatch.calls == [("process_governor",
                               {"action": "terminate", "name": "chrome", "confirm": True})]


def test_high_risk_never_runs_even_when_the_model_insists():
    chat = FakeChat(*[_call("messaging", action="send", contact="boss", message="sick")] * 5)
    dispatch = FakeDispatch()
    agent = _agent(chat, dispatch)
    for text in ("text my boss", "I said send it, don't ask", "just do it now"):
        run(agent.respond(text))
    assert dispatch.calls == []


def test_a_destructive_command_in_any_argument_is_high_risk():
    chat = FakeChat(_call("desktop_control", action="type_text", text="del /s /q C:\\Users"))
    dispatch = FakeDispatch()
    reply = run(_agent(chat, dispatch).respond("run this in a terminal"))
    assert reply.pending and dispatch.calls == []


@pytest.mark.parametrize("name,args", [
    ("shutdown_orion", {}), ("restart_orion", {}),
    ("peripherals", {"action": "shutdown"}), ("peripherals", {"action": "restart"}),
    ("file_controller", {"action": "delete", "path": "C:/x"}),
    ("file_controller", {"path": "C:/x"}),
    ("messaging", {"action": "send", "contact": "Sam", "message": "hi"}),
    ("outlook_mail", {"action": "send_draft", "draft_ref": "draft-1"}),
    ("telephony", {"to": "me"}), ("phone_action", {"kind": "sms", "number": "1"}),
    ("social_media", {"action": "instagram_reply"}),
    ("process_governor", {"action": "terminate", "name": "chrome"}),
    ("antivirus", {"action": "disable"}), ("security_watch", {"action": "disable"}),
    ("sentinel", {"action": "disable"}),
    ("forge", {"action": "forge", "tool_name": "x"}), ("self_repair", {"action": "repair"}),
    ("plugin", {"action": "install", "source": "x.py"}), ("skill", {"action": "install"}),
    ("system_startup", {"action": "enable"}), ("peripherals", {"action": "wifi"}),
    ("organise_files", {"action": "apply", "folder": "Downloads"}),
    ("undo", {"action": "last"}), ("web_control", {"action": "fill_form", "submit": True}),
    ("docker", {"action": "remove", "container": "db"}),
    ("some_forged_tool", {}), ("mcp__gmail__send", {}),
    ("open_app", {"app_name": "cmd /c del /s /q C:\\"}),
])
def test_every_high_risk_category_needs_the_owner(name, args):
    assert classify_risk(name, args)[0] == HIGH


@pytest.mark.parametrize("name,args,risk", [
    ("web_search", {"query": "x"}, READ), ("resource_status", {}, READ),
    ("file_controller", {"action": "list", "path": "C:/"}, READ),
    ("organise_files", {"action": "preview"}, READ), ("organise_files", {}, READ),
    ("process_governor", {"action": "top"}, READ), ("antivirus", {"action": "status"}, READ),
    ("plugin", {"action": "list"}, READ),
    ("open_app", {"app_name": "notepad"}, ACT), ("close_app", {"app_name": "spotify"}, ACT),
    ("peripherals", {"action": "volume_down"}, ACT), ("reminder", {"action": "add"}, ACT),
    ("antivirus", {"action": "scan"}, ACT), ("ai_mode", {"action": "set_brain"}, ACT),
])
def test_ordinary_calls_run_without_asking(name, args, risk):
    assert classify_risk(name, args)[0] == risk


# ── escalation and utterances ───────────────────────────────────────────────

def test_open_ended_requests_escalate_in_hybrid_without_a_local_generation():
    chat = FakeChat()
    reply = run(_agent(chat).respond("write me a poem about autumn", allow_escalate=True))
    assert reply.escalate and chat.calls == []
    # Local only: the model answers itself, with no tools to misuse.
    local = run(_agent(FakeChat(_say("Leaves fall softly."))).respond("write me a poem about autumn"))
    assert local.text == "Leaves fall softly."


def test_an_unsure_local_model_escalates_in_hybrid():
    reply = run(_agent(FakeChat(_say("I'm not sure I can help with that.")))
                .respond("what's the capital of peru", allow_escalate=True))
    assert reply.escalate


def test_remote_origins_are_not_served():
    chat = FakeChat(_call("open_app", app_name="x"))
    reply = run(_agent(chat).respond("open x", origin="remote"))
    assert not reply.handled and chat.calls == []


@pytest.mark.parametrize("text", ["yes", "Yes.", "yeah go ahead", "go ahead", "do it",
                                  "Orion, yes please", "confirm", "ok"])
def test_whole_utterance_yes(text):
    assert is_affirmative(text)


@pytest.mark.parametrize("text", ["yes but send it to Tom instead", "yesterday", "no",
                                  "open notepad", "yes and delete the other one too"])
def test_not_a_bare_yes(text):
    assert not is_affirmative(text)


@pytest.mark.parametrize("text,expected", [("no", True), ("cancel that", True),
                                           ("no thanks", True), ("stop the music", False),
                                           ("nothing", False)])
def test_whole_utterance_no(text, expected):
    assert is_negative(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("explain how transformers work", True), ("write an essay on Rome", True),
    ("why is the sky blue", True), ("open notepad", False), ("what's my cpu usage", False),
    ("set a timer for five minutes", False), ("where is my cv", False),
])
def test_open_ended_classifier(text, expected):
    assert open_ended(text) is expected


def test_an_unavailable_model_says_so_without_trying(monkeypatch):
    agent = LocalAgent(SimpleNamespace(local_text_profiles=lambda: [],
                                       local_tool_chat=lambda *a, **k: None),
                       None, declarations=lambda: DECLS)
    reply = run(agent.respond("open notepad"))
    assert reply.unavailable and "Ollama" in reply.reason
