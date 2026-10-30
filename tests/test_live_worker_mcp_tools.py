"""
Tests for _build_config reading the dispatcher's per-instance
TOOL_DECLARATIONS (Mark XXI, Track E1) — so a connected MCP server's tools
(and any freshly forged tool) actually reach the live Gemini session's
function-calling schema, not only the static module-level list frozen at
import time.
"""

from __future__ import annotations

import sys
import types as pytypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.live_worker as lw
from orion_core.live_worker import GenAILiveWorker, TOOL_DECLARATIONS


def _worker(dispatcher_declarations=None) -> pytypes.SimpleNamespace:
    w = pytypes.SimpleNamespace()
    w.router = pytypes.SimpleNamespace(system_instruction=lambda: "persona")
    w._search_tool_enabled = False
    w._resumption_handle = None
    if dispatcher_declarations is None:
        w.dispatcher = pytypes.SimpleNamespace()   # no TOOL_DECLARATIONS attribute at all
    else:
        w.dispatcher = pytypes.SimpleNamespace(TOOL_DECLARATIONS=dispatcher_declarations)
    return w


def _force_dict_fallback(monkeypatch) -> None:
    # _build_config prefers the real google-genai LiveConnectConfig (a
    # Pydantic-style model, not subscriptable) when the SDK is installed;
    # this test only cares about the plain "tools" list *passed in* to
    # that constructor, so force the method's own except-path, which
    # builds the equivalent plain dict this test can inspect directly.
    def _boom(*a, **k):
        raise RuntimeError("forced for test")
    monkeypatch.setattr(lw.types, "LiveConnectConfig", _boom)
    # These tests are about WHICH source list is used; the core/gateway filter
    # (tool_gateway) is tested separately below.
    monkeypatch.setenv("ORION_LIVE_TOOLS", "full")


def test_falls_back_to_the_static_list_when_dispatcher_has_none(monkeypatch):
    _force_dict_fallback(monkeypatch)
    w = _worker(dispatcher_declarations=None)
    config = GenAILiveWorker._build_config(w)
    assert config["tools"][0]["function_declarations"] == lw._live_safe_declarations(TOOL_DECLARATIONS)


def test_uses_the_dispatchers_per_instance_list_when_present(monkeypatch):
    _force_dict_fallback(monkeypatch)
    widened = list(TOOL_DECLARATIONS) + [{"name": "mcp__gmail__send_email", "description": "x",
                                          "parameters": {"type": "OBJECT", "properties": {}}}]
    w = _worker(dispatcher_declarations=widened)
    config = GenAILiveWorker._build_config(w)
    names = {t["name"] for t in config["tools"][0]["function_declarations"]}
    assert "mcp__gmail__send_email" in names


def test_falls_back_when_dispatcher_declarations_is_falsy(monkeypatch):
    _force_dict_fallback(monkeypatch)
    # An empty list is falsy — must not silently produce a schema with zero
    # tools; fall back to the real static list instead.
    w = _worker(dispatcher_declarations=[])
    config = GenAILiveWorker._build_config(w)
    assert config["tools"][0]["function_declarations"] == lw._live_safe_declarations(TOOL_DECLARATIONS)


def test_dict_fallback_keeps_session_resumption_and_compression(monkeypatch):
    _force_dict_fallback(monkeypatch)
    worker = _worker()
    worker._resumption_handle = "resume-test"
    config = GenAILiveWorker._build_config(worker)
    assert config["session_resumption"] == {"handle": "resume-test"}
    assert config["context_window_compression"] == {"sliding_window": {}}


def test_live_schema_cache_observes_in_place_parameter_repairs():
    declaration = {"name": "repaired_tool", "description": "Read a report",
                   "parameters": {"type": "OBJECT", "properties": {
                       "path": {"type": "STRING"}}, "required": ["path"]}}
    before = lw._live_safe_declarations([declaration])
    declaration["parameters"]["properties"]["path"]["type"] = "INTEGER"
    after = lw._live_safe_declarations([declaration])
    assert before[0]["parameters"]["properties"]["path"]["type"] == "STRING"
    assert after[0]["parameters"]["properties"]["path"]["type"] == "INTEGER"


def test_repaired_schema_retries_live_tool_set_even_with_the_same_names(monkeypatch):
    _force_dict_fallback(monkeypatch)
    declaration = {"name": "repaired_tool", "description": "Original",
                   "parameters": {"type": "OBJECT", "properties": {}}}
    worker = _worker([declaration])
    worker._live_config_stage = 1
    worker._live_config_fp = lw._declaration_fingerprint([declaration])
    declaration["parameters"]["properties"]["contact"] = {"type": "STRING"}
    config = GenAILiveWorker._build_config(worker)
    assert worker._live_config_stage == 0
    assert config["tools"][0]["function_declarations"][0]["name"] == "repaired_tool"


# ── the Live tool gateway (core tools + find_tool/use_tool) ─────────────────

def test_live_messaging_accepts_saved_contacts_and_email_only_arguments(monkeypatch):
    """The voice session must carry the same usable contract as dispatch."""
    _force_dict_fallback(monkeypatch)
    monkeypatch.delenv("ORION_LIVE_TOOLS", raising=False)
    config = GenAILiveWorker._build_config(_worker(TOOL_DECLARATIONS))
    declarations = config["tools"][0]["function_declarations"]
    messaging = next(row for row in declarations if row["name"] == "messaging")
    parameters = messaging["parameters"]
    # Email has no chat platform/contact/message; send validates its own fields.
    assert parameters["required"] == ["action"]
    assert "contact name" in parameters["properties"]["contact"]["description"]
    assert "phone number is not required" in messaging["description"]


def test_discord_named_contact_is_discoverable_without_a_running_intent_model():
    from orion_core.tool_resolver import lexical_scores

    scores = lexical_scores("send a Discord message to my contact", TOOL_DECLARATIONS)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert "messaging" in ranked[:5]

def test_the_live_session_carries_the_core_set_and_the_gateway(monkeypatch):
    monkeypatch.setattr(lw.types, "LiveConnectConfig",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("dict path")))
    monkeypatch.delenv("ORION_LIVE_TOOLS", raising=False)
    widened = list(TOOL_DECLARATIONS) + [{"name": "mcp__notion__search", "description": "x",
                                          "parameters": {"type": "OBJECT", "properties": {}}}]
    w = _worker(dispatcher_declarations=widened)
    config = GenAILiveWorker._build_config(w)
    names = {t["name"] for t in config["tools"][0]["function_declarations"]}
    from orion_core.tool_gateway import LIVE_CORE_TOOLS
    assert {"find_tool", "use_tool"} <= names
    assert names <= LIVE_CORE_TOOLS
    assert "mcp__notion__search" not in names and "chess" not in names
    assert "find_tool" in config["system_instruction"]


async def test_a_gated_tool_is_still_reachable_through_use_tool():
    from orion_core.data import ToolResult
    from orion_core import tool_gateway
    calls = []

    class _D:
        TOOL_DECLARATIONS = list(TOOL_DECLARATIONS) + [
            {"name": "mcp__notion__search", "description": "Search Notion.",
             "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}}}]

        async def dispatch_chain(self, name, args):
            calls.append((name, args))
            return ToolResult("found 3 pages")

    found = tool_gateway.find_tool(_D(), {"need": "search my notion pages"})
    assert "mcp__notion__search" in found.text
    result = await tool_gateway.use_tool(_D(), {"name": "mcp__notion__search",
                                                "arguments": '{"query": "budget"}'})
    assert result.ok and calls == [("mcp__notion__search", {"query": "budget"})]
    refused = await tool_gateway.use_tool(_D(), {"name": "use_tool", "arguments": "{}"})
    assert not refused.ok


def test_find_tool_can_return_the_tools_the_resolvers_floor_pins():
    """find_tool went through resolve(), whose safety floor pins catch_up,
    reason and capabilities to the front, then dropped those three: so
    reason — not a Live core tool — could never be found, however well it
    matched. It now ranks straight from the scores."""
    from orion_core import tool_gateway

    class _D:
        TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)

    found = tool_gateway.find_tool(_D(), {"need": "think hard about this difficult question"})
    assert "- reason:" in found.text


# ── Mark XXXVII: what the voice session holds directly, and lossless find_tool ─

def test_the_tools_the_owner_asks_for_by_voice_are_direct():
    """'What changed / your updates' was answered from resource_status and old
    notes ~19 times, and 'sing me a song' got 'not quite equipped': each of
    these was a find_tool hop away."""
    from orion_core.tool_gateway import LIVE_CORE_TOOLS
    assert {"sing", "patch_notes", "self_changes", "code_changes"} <= LIVE_CORE_TOOLS
    declared = {d["name"] for d in TOOL_DECLARATIONS}
    assert LIVE_CORE_TOOLS <= declared, "a core name with no declaration is a dead slot"


def test_the_core_voice_schema_stays_within_its_prompt_budget():
    """Every core declaration rides in every spoken turn (tool_gateway's
    docstring: 42k tokens -> ~10.5k bought ~0.5 s of first audio). 23 Oct
    2026: 64,886 characters before this pass, 69,936 after adding sing and
    the three change tools (sing's description trimmed) and widening
    execute_plan/web_search. Growth past the bound is a decision to make,
    not something to drift into."""
    import json as _json
    from orion_core.tool_gateway import LIVE_CORE_TOOLS
    core = [d for d in TOOL_DECLARATIONS if d["name"] in LIVE_CORE_TOOLS]
    assert len(_json.dumps(core)) <= 71_000


def test_find_tool_shows_its_best_matches_whole():
    from orion_core import tool_gateway

    class _D:
        TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)

    found = tool_gateway.find_tool(_D(), {"need": "chess"}).text
    chess = next(d for d in TOOL_DECLARATIONS if d["name"] == "chess")
    block = found.split("\n- ")[1]
    assert block.startswith("chess:")
    # Every action, including the ones the old 80-character cut hid ...
    for action in ("rating", "practice", "lichess_start", "lichess_resign", "challenge"):
        assert action in block, action
    # ... and every parameter, with its type.
    for name, spec in chess["parameters"]["properties"].items():
        assert f"    {name}" in block and f"({spec['type'].lower()})" in block, name
    assert "    action* (string)" in block


def test_find_tool_shows_choices_and_keeps_the_rest_compact():
    from orion_core import tool_gateway

    def decl(name, description, **props):
        return {"name": name, "description": description,
                "parameters": {"type": "OBJECT", "properties": props}}

    class _D:
        TOOL_DECLARATIONS = [
            decl("lamp_switch", "Switch the desk lamp. Second sentence detail.",
                 mode={"type": "STRING", "description": "How bright.",
                       "enum": ["off", "dim", "full"]},
                 rooms={"type": "ARRAY", "items": {"type": "STRING"}}),
            decl("lamp_colour", "Colour of the desk lamp. More detail here."),
            *[decl(f"lamp_extra_{n}", f"Another desk lamp helper {n}. Hidden detail.",
                   level={"type": "INTEGER", "description": "x" * 200}) for n in range(4)],
        ]

    text = tool_gateway.find_tool(_D(), {"need": "desk lamp switch"}).text
    assert "One of: off, dim, full." in text and "rooms (array of string)" in text
    assert "Second sentence detail." in text, "the best match is not cut to one sentence"
    compact = [line for line in text.splitlines() if line.startswith("- lamp_extra_")]
    assert compact and all("Hidden detail" not in line and "Arguments:" in line
                           for line in compact)


# ── listening hints: the words to expect (owner, 7 Oct 2026) ─────────────────

class _Memory:
    """A memory store holding a few facts with names in them."""

    def __init__(self):
        import sqlite3
        import threading
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("CREATE TABLE intelligence (key_ref TEXT, value TEXT, updated_at TEXT)")
        self.conn.executemany("INSERT INTO intelligence VALUES (?, ?, ?)", [
            ("best_friend", "The user's best friend is Kieran, an electrician", "2026-10-01"),
            ("garage", "The car goes to Kwik Fit on Station Road", "2026-10-02"),
            ("graph_event_1", "Ignored mirror mentions Zebulon", "2026-10-03")])
        self._lock = threading.Lock()


def test_the_live_transcript_is_told_the_language_and_the_names(monkeypatch):
    monkeypatch.setattr("orion_core.listening_vocabulary._common_words",
                        lambda: frozenset({"station", "road"}))
    w = _worker()
    w.memory = _Memory()
    config = GenAILiveWorker._build_config(w)
    heard = config.input_audio_transcription
    assert heard.language_hints.language_codes == ["en-GB"]
    assert "ORION" in heard.adaptation_phrases
    assert "Kieran" in heard.adaptation_phrases and "Kwik Fit" in heard.adaptation_phrases
    assert "Zebulon" not in heard.adaptation_phrases, "graph mirrors are not facts"
    assert w._transcription_hints_used is True


def test_refused_hints_are_dropped_alone(monkeypatch):
    w = _worker()
    w.memory = _Memory()
    w._transcription_hints_off = True
    config = GenAILiveWorker._build_config(w)
    assert not config.input_audio_transcription.adaptation_phrases
    assert w._transcription_hints_used is False


def test_whisper_hears_the_same_words_but_never_recites_them():
    from orion_core.speech_offline import OfflineTranscriber, _recites
    stt = OfflineTranscriber(defer=True)
    stt.vocabulary_source = lambda: ["ORION", "Kieran", "Kwik Fit"]
    assert stt._prompt() == "ORION, Kieran, Kwik Fit."
    assert _recites("ORION, Kieran, Kwik Fit.", stt._prompt())
    assert not _recites("ask Kieran if Kwik Fit is open", stt._prompt())


# ── background work landing (owner's log, 7 Oct 2026) ────────────────────────

def test_a_finished_user_job_is_announced_once_and_housekeeping_is_not():
    said = []
    w = pytypes.SimpleNamespace(_announced_jobs=set(), announce=said.append,
                                ANNOUNCED_JOB_CLASSES=GenAILiveWorker.ANNOUNCED_JOB_CLASSES)
    job = {"id": "job-7", "state": "completed", "class": "user", "label": "AI model analysis"}
    GenAILiveWorker._on_dashboard_event(w, "job", job)
    GenAILiveWorker._on_dashboard_event(w, "job", job)            # progress repeats
    GenAILiveWorker._on_dashboard_event(w, "job", {**job, "id": "job-8", "class": "background"})
    GenAILiveWorker._on_dashboard_event(w, "job", {**job, "id": "job-9", "state": "running"})
    assert said == ["That AI model analysis has finished - ask me what it found whenever you like."]
    GenAILiveWorker._on_dashboard_event(w, "job", {"id": "job-10", "state": "failed", "class": "user",
                                                   "label": "competitor sweep",
                                                   "result": "the site refused the request"})
    assert said[-1] == "That competitor sweep didn't finish: the site refused the request"


def test_noise_alone_never_sets_off_an_action():
    from orion_core.live_worker import _heard_only_noise
    assert _heard_only_noise("<noise>") and _heard_only_noise("<noise> [music]")
    assert not _heard_only_noise("")                         # not heard yet: unknown
    assert not _heard_only_noise("Uh <noise> shut down")


def test_a_reply_cut_off_mid_sentence_is_recognised():
    from orion_core.live_worker import _reply_was_cut
    assert _reply_was_cut("with desktop and web control being among the most")
    assert _reply_was_cut("For HTTPS connections, the standard port is")
    assert _reply_was_cut("Your calendar is clear for the next two days. However,")
    assert not _reply_was_cut("Would you like to see that?")
    assert not _reply_was_cut("He said \"it is done.\"")
    assert not _reply_was_cut("Okay")                         # too short to judge
    assert not _reply_was_cut("Run it with this: ```pip install pynput```")


def test_a_spoken_question_cut_mid_answer_is_carried_on():
    # 27 Oct 2026: "...From an educational perspective," then silence. The
    # request's own transcript shares the turn, which used to block this.
    from orion_core.live_worker import MAX_CUT_CONTINUATIONS, _should_finish_cut_reply
    cut = "It allows you to reuse code without writing it multiple times, which makes"
    base = dict(interrupted=False, talked_over=False, continuations=0, held=False)
    assert _should_finish_cut_reply(cut, **base)
    assert not _should_finish_cut_reply(cut, **{**base, "talked_over": True})
    assert not _should_finish_cut_reply(cut, **{**base, "interrupted": True})
    assert not _should_finish_cut_reply(cut, **{**base, "held": True})
    assert _should_finish_cut_reply(cut, **{**base, "continuations": MAX_CUT_CONTINUATIONS - 1})
    assert not _should_finish_cut_reply(cut, **{**base, "continuations": MAX_CUT_CONTINUATIONS})
    assert not _should_finish_cut_reply("That is the whole answer, sir.", **base)


def test_the_receive_loop_judges_talking_over_not_the_request_itself():
    import inspect
    from orion_core.live_worker import GenAILiveWorker
    src = inspect.getsource(GenAILiveWorker._receive_realtime_stream)
    assert "_should_finish_cut_reply(" in src and "talked_over=talked_over" in src
    assert "and not user_spoke" not in src


def test_a_cut_reply_is_asked_to_finish_once():
    import asyncio
    sent, logged = [], []

    class _Session:
        async def send_client_content(self, turns=None, turn_complete=False):
            sent.append((turns, turn_complete))

    async def run():
        w = pytypes.SimpleNamespace(
            _send_lock=asyncio.Lock(), session=_Session(), _send_closed=False,
            bus=pytypes.SimpleNamespace(log=pytypes.SimpleNamespace(emit=logged.append)),
            _turn_active=False, _turn_progress_at=0.0)
        await GenAILiveWorker._finish_cut_reply(
            w, "the trends show consistent operation across various tools, with desktop "
               "and web control being among the most", 150)
        return w
    w = asyncio.run(run())
    assert len(sent) == 1 and sent[0][1] is True
    text = sent[0][0]["parts"][0]["text"]
    assert "being among the most" in text and "not from the user" in text
    assert w._continuing_cut_reply is True and w._turn_active is True
    assert "150 tokens" in logged[0]


# ── Mark XXXX.1: finishing his sentences, and a tone that follows his ─────────

def test_the_channel_waits_for_a_real_pause_before_answering(monkeypatch):
    # 27 Oct 2026: "I can be talking and he can cut off listening and
    # instantly start replying ... I want to be able to finish my sentences".
    from orion_core import voice_settings as vs
    monkeypatch.setattr(vs, "SETTINGS_PATH", vs.SETTINGS_PATH.parent / "pause_test.json")
    monkeypatch.delenv("ORION_REPLY_PAUSE", raising=False)
    w = _worker()
    config = GenAILiveWorker._build_config(w)
    detection = config.realtime_input_config.automatic_activity_detection
    assert detection.silence_duration_ms == vs.REPLY_PAUSES["natural"] == 1200
    assert detection.end_of_speech_sensitivity == lw.types.EndSensitivity.END_SENSITIVITY_LOW
    monkeypatch.setenv("ORION_REPLY_PAUSE", "patient")
    assert GenAILiveWorker._build_config(w).realtime_input_config \
        .automatic_activity_detection.silence_duration_ms == 1700
    monkeypatch.setenv("ORION_REPLY_PAUSE", "brisk")
    brisk = GenAILiveWorker._build_config(w).realtime_input_config.automatic_activity_detection
    assert brisk.silence_duration_ms == 700
    assert brisk.end_of_speech_sensitivity == lw.types.EndSensitivity.END_SENSITIVITY_HIGH


def test_expressive_voice_is_asked_for_and_peeled_off_when_refused(monkeypatch):
    monkeypatch.delenv("ORION_EXPRESSIVE_VOICE", raising=False)
    w = _worker()
    assert GenAILiveWorker._build_config(w).enable_affective_dialog is True
    assert w._affective_used is True
    w._affective_off = True
    assert not GenAILiveWorker._build_config(w).enable_affective_dialog
    import inspect
    run = inspect.getsource(GenAILiveWorker.run)
    assert run.index("expressive voice refused") < run.index("transcription hints refused")
    assert "self._affective_wanted()" in run and '"v1alpha"' in run


def test_two_silent_replies_set_the_expressive_voice_aside():
    logged = []
    w = pytypes.SimpleNamespace(bus=pytypes.SimpleNamespace(log=pytypes.SimpleNamespace(
        emit=logged.append)), _affective_used=True, SILENT_TURNS_BEFORE_PLAIN=2)
    GenAILiveWorker._note_silent_turn(w)
    assert not getattr(w, "_affective_off", False)
    GenAILiveWorker._note_silent_turn(w)
    assert w._affective_off is True and "set aside" in logged[-1]


def test_choosing_expressive_again_brings_it_back_this_session():
    """Mark XXXX.4: set aside after two silent replies, it stayed off until a
    restart even when the owner switched it on again."""
    w = pytypes.SimpleNamespace(_affective_off=True, _silent_turns=2, _refresh_pending=True)
    GenAILiveWorker.request_channel_refresh(w, "expressive voice on")
    assert w._affective_off is False and w._silent_turns == 0
    w = pytypes.SimpleNamespace(_affective_off=True, _silent_turns=2, _refresh_pending=True)
    GenAILiveWorker.request_channel_refresh(w, "new reply pause setting")
    assert w._affective_off is True


def test_a_settings_refresh_waits_until_he_is_quiet(monkeypatch):
    import asyncio
    closed = []

    async def run():
        w = pytypes.SimpleNamespace(
            _turn_active=True, tool_busy=False, session=object(), CHANNEL_REFRESH_WAIT_S=5.0,
            speech=pytypes.SimpleNamespace(output_active=lambda: False),
            stop_event=asyncio.Event())

        async def end(reason):
            closed.append(reason)
        w._end_live_session = end
        monkeypatch.setattr(lw.asyncio, "sleep", _fast_sleep)
        GenAILiveWorker.request_channel_refresh(w, "new reply pause")
        GenAILiveWorker.request_channel_refresh(w, "again")      # one at a time
        await _real_sleep(0.01)
        assert closed == []                                     # mid-turn: waits
        w._turn_active = False
        for _ in range(20):
            await _real_sleep(0.005)
        return w
    w = asyncio.run(run())
    assert closed == ["new reply pause"] and w._refresh_pending is False


async def _fast_sleep(_seconds):
    await _real_sleep(0.001)

import asyncio as _asyncio_for_sleep  # noqa: E402
_real_sleep = _asyncio_for_sleep.sleep
