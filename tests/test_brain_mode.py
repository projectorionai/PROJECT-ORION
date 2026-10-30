"""
Brain modes (orion_core/brain_mode.py): cloud, hybrid and local — the setting,
what the router withholds, how the live worker routes a turn in each mode, the
ai_mode tool, the mode report and the Preferences selector.

Everything is faked: no Ollama, no Live session, no microphone, no network.
conftest points brain_mode.SETTINGS_PATH at a per-test temporary file.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication  # noqa: E402,F401  (GUI test: see conftest)

from orion_core import brain_mode  # noqa: E402
from orion_core.data import ToolResult  # noqa: E402
from orion_core.live_worker import GenAILiveWorker, _TURN_ORIGIN  # noqa: E402
from orion_core.local_agent import AgentReply  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ── the setting ─────────────────────────────────────────────────────────────

def test_cloud_is_the_default_and_a_choice_is_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(brain_mode, "SETTINGS_PATH", tmp_path / "brain_mode.json")
    assert brain_mode.mode() == brain_mode.effective_mode() == "cloud"
    assert brain_mode.set_mode("Local Only") == "local"
    stored = json.loads((tmp_path / "brain_mode.json").read_text(encoding="utf-8"))
    assert stored["brain_mode"] == "local" and stored["schema"] == "orion.brain_mode.v1"
    assert brain_mode.mode() == "local"
    assert brain_mode.set_mode("sideways") is None and brain_mode.mode() == "local"
    assert brain_mode.get("keep_alive") == "10m" and brain_mode.get("local_model") == ""
    with pytest.raises(KeyError):
        brain_mode.get("nonsense")


@pytest.mark.parametrize("words,mode", [("cloud", "cloud"), ("Hybrid", "hybrid"),
                                        ("local only", "local"), ("offline", "local"),
                                        ("local_only", "local"), ("", None), ("fast", None)])
def test_loose_words_name_a_mode(words, mode):
    assert brain_mode.normalise(words) == mode


def test_the_environment_overrides_and_offline_forces_local(monkeypatch):
    brain_mode.set_mode("cloud")
    monkeypatch.setenv("ORION_BRAIN_MODE", "hybrid")
    assert brain_mode.mode() == "hybrid" and brain_mode.env_override_active()
    monkeypatch.setenv("ORION_MODE", "offline")          # --offline
    assert brain_mode.effective_mode() == "local" and brain_mode.forced_offline()
    assert brain_mode.stored_mode() == "cloud"           # the saved choice is untouched


def test_every_mode_has_a_title_case_label_and_one_line():
    from orion_core.gui.labels import title_case
    for mode in brain_mode.MODES:
        assert title_case(brain_mode.LABELS[mode]) == brain_mode.LABELS[mode]
        assert "\n" not in brain_mode.EXPLANATIONS[mode]


# ── the router in the local brain ───────────────────────────────────────────

def _router():
    from orion_core.providers import AIProviderProfile, OrionProviderSettings, ProviderRouter

    class Cloud(AIProviderProfile):
        is_local = property(lambda self: False)

    cloud = Cloud(name="gemini", kind="openai_compatible", model="gemini-x",
                  base_url="https://example.invalid/v1", api_key="k", enabled=True)
    local = AIProviderProfile(name="local_ollama", kind="openai_compatible", model="llama3.1",
                              base_url="http://127.0.0.1:11434/v1", api_key="local",
                              enabled=True)
    settings = OrionProviderSettings(active_provider="gemini",
                                     provider_order=["gemini", "local_ollama"],
                                     providers={"gemini": cloud, "local_ollama": local})
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_a: None))
    router = ProviderRouter(settings, bus, memory=SimpleNamespace(prompt_context=lambda *_: ""))
    router.mode_override = "cloud"                       # "online", without probing anything
    return router


def test_the_local_brain_withholds_every_cloud_model_but_not_connectivity():
    router = _router()
    assert [p.name for p in router.cloud_text_profiles()] == ["gemini"]
    router.set_local_only(True)
    assert router.is_online()                            # web tools still work
    assert router.cloud_text_profiles() == [] and router.live_profiles() == []
    assert [p.name for p in router.text_profiles()] == ["local_ollama"]
    assert [p.name for p in router.select_text_profiles("hello")] == ["local_ollama"]
    assert router.current_mode().startswith("MODE L") and not router.cloud_allowed()
    router.set_local_only(False)
    assert "gemini" in [p.name for p in router.text_profiles()]


# ── the ai_mode tool and the mode report ────────────────────────────────────

class _Host:
    def __init__(self, ai=None, apply=None):
        from orion_core.dispatch_productivity import ProductivityDispatchMixin
        self._mixin = ProductivityDispatchMixin
        self.ai = ai
        if apply is not None:
            self.apply_brain_mode = apply

    def __getattr__(self, name):
        from orion_core.dispatch_productivity import ProductivityDispatchMixin
        return getattr(ProductivityDispatchMixin, name).__get__(self)


def test_ai_mode_reads_and_sets_the_brain_by_voice():
    applied = []
    host = _Host(apply=lambda mode: (applied.append(mode) or True, f"{mode} brain now"))
    result = host.ai_mode({"action": "set_brain", "mode": "local brain"})
    assert result.ok and applied == ["local"] and "local" in result.text
    assert not host.ai_mode({"action": "set_brain"}).ok
    # Without a running worker the choice is still saved.
    plain = _Host()
    saved = plain.ai_mode({"action": "set_brain", "mode": "hybrid"})
    assert saved.ok and brain_mode.stored_mode() == "hybrid"
    report = plain.ai_mode({"action": "brain"})
    assert report.ok and "Hybrid" in report.text


def test_the_schema_offers_the_brain_actions():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "ai_mode")
    assert "set_brain" in tool["parameters"]["properties"]["action"]["description"]
    assert "mode" in tool["parameters"]["properties"]
    assert "local brain" in tool["description"]


def test_the_vocabulary_and_situations_know_the_brain_switch():
    from orion_core.intent_situations import phrases
    from orion_core.tool_vocabulary import terms_for
    assert any("local brain" in phrase for phrase in phrases()["ai_mode"])
    assert "hybrid" in terms_for("ai_mode")


def test_the_mode_report_names_the_brain_and_the_local_model():
    from orion_core.local_models import AIModeInfo
    brain_mode.set_mode("hybrid")
    router = SimpleNamespace(provider_snapshot=lambda: {"mode": "MODE A"},
                             local_text_profiles=lambda: [])
    ollama = SimpleNamespace(status=lambda: {"available": True, "best_model": "llama3.1:latest",
                                             "models": ["llama3.1:latest"]},
                             recommendations=lambda: [])
    text = AIModeInfo(router, ollama=ollama).report().text
    assert "Brain mode: Hybrid" in text
    assert "Local agent model (tool use without an AI API): llama3.1:latest" in text


# ── how the live worker routes a turn ───────────────────────────────────────

class FakeAgent:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.pending = None

    async def respond(self, text, *, origin="text", allow_escalate=False, dispatch=None):
        self.calls.append({"text": text, "origin": origin, "allow_escalate": allow_escalate})
        return self.replies.pop(0) if self.replies else AgentReply()

    def cancel_pending(self):
        pending, self.pending = self.pending, None
        return pending

    def model_ready(self):
        return True, ""


class _Bus:
    def __init__(self):
        self.lines = []
        self.log = SimpleNamespace(emit=self.lines.append)


def _worker(agent, *, online=True, cloud_reply=None, cloud_fails=False, session=None,
            rules="A rules answer."):
    spoken, cloud_calls, rule_calls, sent = [], [], [], []

    async def generate_text(prompt, *_a, **_k):
        cloud_calls.append(prompt)
        if cloud_fails:
            raise RuntimeError("all text providers failed")
        return SimpleNamespace(name="gemini"), cloud_reply or "A cloud answer."

    async def respond(text):
        rule_calls.append(text)
        return rules

    async def known(_text):
        return None

    async def send_text_turn(text):
        sent.append(text)

    worker = SimpleNamespace(
        paused=False, stop_event=threading.Event(), _fallback_epoch=0, bus=_Bus(),
        router=SimpleNamespace(is_online=lambda: online, has_text_fallback=lambda: True,
                               generate_text=generate_text),
        local_brain=SimpleNamespace(respond=respond, try_known_response=known),
        local_agent=agent, session=session, dispatcher=None,
        _emit_state=lambda *_: None, _say=spoken.append, _persist_episode=lambda *_: None,
        _overheard_filler=lambda *_: False, _offline_voice_ready=lambda: False,
        speech=SimpleNamespace(output_active=lambda: False),
        _mark_turn=lambda *_a, **_k: None, _send_text_turn=send_text_turn)
    return worker, spoken, cloud_calls, rule_calls, sent


def _fallback(worker, text="open notepad"):
    run(GenAILiveWorker._run_text_fallback(worker, text))


def test_cloud_mode_is_unchanged_while_the_cloud_answers():
    agent = FakeAgent(AgentReply("Opened."))
    worker, spoken, cloud, rules, _sent = _worker(agent)
    _fallback(worker)
    assert spoken == ["A cloud answer."] and agent.calls == [] and rules == []


def test_cloud_mode_tries_the_local_agent_before_the_rules_when_the_cloud_fails():
    agent = FakeAgent(AgentReply("Opening Notepad.",
                                 [{"name": "open_app", "status": "ran", "ok": True,
                                   "args": {"app_name": "notepad"}}]))
    worker, spoken, cloud, rules, _sent = _worker(agent, cloud_fails=True)
    worker.observe_for_reflex = lambda *a, **k: None
    _fallback(worker)
    assert cloud and spoken == ["Opening Notepad."] and rules == []
    assert agent.calls == [{"text": "open notepad", "origin": "text", "allow_escalate": False}]


def test_cloud_mode_falls_to_the_rules_when_the_local_model_is_down():
    agent = FakeAgent(AgentReply(unavailable=True, reason="unavailable: Ollama is not answering"))
    worker, spoken, _cloud, rules, _sent = _worker(agent, cloud_fails=True)
    _fallback(worker)
    assert spoken == ["A rules answer."] and rules == ["open notepad"]


def test_hybrid_answers_locally_first_and_never_bills_the_cloud_for_it():
    brain_mode.set_mode("hybrid")
    agent = FakeAgent(AgentReply("Opening Notepad."))
    worker, spoken, cloud, rules, _sent = _worker(agent)
    _fallback(worker)
    assert spoken == ["Opening Notepad."] and cloud == [] and rules == []
    assert agent.calls[0]["allow_escalate"] is True


def test_hybrid_escalates_open_ended_turns_to_the_cloud_text_model():
    brain_mode.set_mode("hybrid")
    agent = FakeAgent(AgentReply(escalate=True, reason="open-ended request"))
    worker, spoken, cloud, rules, _sent = _worker(agent)
    _fallback(worker, "write me a poem")
    assert len(cloud) == 1 and spoken == ["A cloud answer."] and rules == []


def test_hybrid_escalates_a_typed_turn_to_live_when_live_is_up():
    brain_mode.set_mode("hybrid")
    agent = FakeAgent(AgentReply(escalate=True, reason="open-ended request"))
    worker, spoken, cloud, _rules, sent = _worker(agent, session=object())
    _fallback(worker, "write me a poem")
    assert sent == ["write me a poem"] and cloud == [] and spoken == []


def test_local_mode_never_calls_the_cloud_text_path():
    brain_mode.set_mode("local")
    agent = FakeAgent(AgentReply(""))                    # nothing to say
    worker, spoken, cloud, rules, _sent = _worker(agent)
    _fallback(worker)
    assert cloud == [] and rules == ["open notepad"] and spoken == ["A rules answer."]
    assert agent.calls[0]["allow_escalate"] is False


def test_local_mode_says_plainly_when_the_local_model_is_missing():
    brain_mode.set_mode("local")
    agent = FakeAgent(AgentReply(unavailable=True,
                                 reason="unavailable: no local model is available"))
    worker, spoken, cloud, rules, _sent = _worker(agent)
    _fallback(worker)
    assert cloud == [] and rules == ["open notepad"]
    assert spoken[0].startswith("My local model isn't available")
    assert spoken[0].endswith("A rules answer.")


def test_local_mode_with_the_real_agent_and_no_ollama_falls_back_to_the_rules():
    from orion_core.local_agent import LocalAgent
    brain_mode.set_mode("local")
    agent = LocalAgent(SimpleNamespace(local_text_profiles=lambda: [],
                                       local_tool_chat=lambda *a, **k: pytest.fail("no model")),
                       None)
    worker, spoken, cloud, rules, _sent = _worker(agent)
    _fallback(worker)
    assert cloud == [] and rules == ["open notepad"] and "isn't available" in spoken[0]


def test_offline_voice_turns_reach_the_agent_as_spoken():
    agent = FakeAgent(AgentReply("Opening Notepad."))
    worker, spoken, *_ = _worker(agent, online=False)

    async def flow():
        _TURN_ORIGIN.set("voice")
        await GenAILiveWorker._run_text_fallback(worker, "open notepad")

    run(flow())
    assert agent.calls[0]["origin"] == "voice" and spoken == ["Opening Notepad."]
    assert agent.calls[0]["allow_escalate"] is False      # offline: nothing to escalate to


def test_a_yes_for_a_pending_call_goes_to_the_agent_in_every_mode():
    agent = FakeAgent(AgentReply("Shutting down."))
    agent.pending = SimpleNamespace(name="shutdown_orion")
    worker, spoken, cloud, rules, _sent = _worker(agent)
    routed = []

    async def submit_fallback(text, reason=""):
        routed.append(text)
        await GenAILiveWorker._run_text_fallback(worker, text)

    worker._submit_text_fallback = submit_fallback
    worker.resume = lambda: None
    assert run(GenAILiveWorker._answer_local_confirmation(worker, "yes")) is True
    assert routed == ["yes"] and spoken == ["Shutting down."] and cloud == []


def test_anything_else_cancels_the_pending_call_before_other_routes():
    agent = FakeAgent()
    agent.pending = SimpleNamespace(name="messaging")
    worker, *_ = _worker(agent)
    assert run(GenAILiveWorker._answer_local_confirmation(worker, "what's the weather")) is False
    assert agent.pending is None


def test_a_spoken_yes_with_live_up_goes_to_the_agent_not_live():
    agent = FakeAgent()
    agent.pending = SimpleNamespace(name="shutdown_orion")
    submitted = []

    async def submit(text):
        submitted.append((text, _TURN_ORIGIN.get()))

    worker = SimpleNamespace(local_agent=agent, submit_text=submit, bus=_Bus(),
                             _strip_wake_words=lambda t: t)

    async def flow():
        assert GenAILiveWorker._local_confirmation_heard(worker, "go ahead", False) is True
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert GenAILiveWorker._local_confirmation_heard(worker, "open spotify", False) is False

    run(flow())
    assert submitted == [("go ahead", "voice")] and agent.pending is None


def test_switching_to_the_local_brain_applies_at_once():
    closed, prepared = [], []

    class Session:
        async def close(self):
            closed.append(True)

    router = SimpleNamespace(local_only=False)
    router.set_local_only = lambda active: setattr(router, "local_only", active)
    agent = FakeAgent()

    async def prepare():
        prepared.append(True)

    agent._prepare = prepare
    agent.model_ready = lambda: (False, "not yet")
    worker = SimpleNamespace(router=router, local_agent=agent, session=Session(), bus=_Bus(),
                             SESSION_CLOSE_SECONDS=1.0)
    worker._apply_brain_runtime = lambda effective: GenAILiveWorker._apply_brain_runtime(
        worker, effective)
    worker._prepare_local_brain = lambda: GenAILiveWorker._prepare_local_brain(worker)
    worker._end_live_session = lambda reason: GenAILiveWorker._end_live_session(worker, reason)

    async def flow():
        ok, message = GenAILiveWorker.apply_brain_mode(worker, "local")
        for _ in range(5):
            await asyncio.sleep(0)
        return ok, message

    ok, message = run(flow())
    assert ok and message.startswith("Local Only brain")
    assert router.local_only is True and brain_mode.stored_mode() == "local"
    assert closed == [True] and prepared == [True]
    ok, _ = GenAILiveWorker.apply_brain_mode(worker, "cloud")
    assert ok and router.local_only is False
    assert GenAILiveWorker.apply_brain_mode(worker, "nonsense")[0] is False


# ── the Preferences selector ────────────────────────────────────────────────

def _selector():
    from orion_core.gui.preferences import _BrainSelector
    globals().setdefault("_QT_APP", QApplication.instance() or QApplication([]))  # held, or it is collected
    return _BrainSelector()


def test_the_brain_selector_offers_three_brains_with_explanations():
    group = _selector()
    try:
        labels = [group.combo.itemText(i) for i in range(group.combo.count())]
        assert labels == ["Cloud", "Hybrid", "Local Only"]
        assert group.title() == "Brain" and group.selected() == "cloud"
        assert group.explanation.text() == brain_mode.EXPLANATIONS["cloud"]
        assert group.combo.accessibleName()
    finally:
        group.deleteLater()


def test_choosing_a_brain_saves_it_and_reports_it():
    group = _selector()
    seen = []
    group.changed.connect(seen.append)
    try:
        group.combo.setCurrentIndex(group.combo.findData("local"))
        assert brain_mode.stored_mode() == "local" and seen == ["local"]
        assert group.explanation.text() == brain_mode.EXPLANATIONS["local"]
        group.combo.setCurrentIndex(group.combo.findData("hybrid"))
        assert brain_mode.stored_mode() == "hybrid" and seen == ["local", "hybrid"]
    finally:
        group.deleteLater()


def test_the_selector_shows_what_is_stored_and_reports_a_failed_save(monkeypatch):
    brain_mode.set_mode("hybrid")
    group = _selector()
    try:
        assert group.selected() == "hybrid"
        monkeypatch.setattr(brain_mode, "set_mode", lambda value: None)
        group.combo.setCurrentIndex(group.combo.findData("local"))
        assert "could not be saved" in group.status.text()
    finally:
        group.deleteLater()


def test_preferences_applies_the_brain_through_the_worker():
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "gui"
              / "core_window.py").read_text(encoding="utf-8")
    assert "OrionCoreWindow._apply_brain_mode(self, mode)" in source
    assert "apply(mode, persist=False)" in source


def test_the_window_hands_a_saved_brain_to_the_worker():
    from orion_core.gui.core_window import OrionCoreWindow
    applied, lines = [], []
    core = SimpleNamespace(
        worker=SimpleNamespace(apply_brain_mode=lambda mode, persist=True: (
            applied.append((mode, persist)) or True, f"{mode} applied")),
        bus=SimpleNamespace(log=SimpleNamespace(emit=lines.append)))
    OrionCoreWindow._apply_brain_mode(core, "local")
    assert applied == [("local", False)] and lines == ["BRAIN: local applied"]
