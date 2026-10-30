"""ORION learns from how a turn went: praise and uncorrected success are
rewards, corrections and failures are penalties, a rewarded read-only route
becomes an instant reflex, and a rewarded multi-step method is remembered and
offered back to the model the next time the same words come."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from orion_core import intent_brain as _intent_brain
from orion_core import reward_learning as rl
from orion_core.reflex_learning import ReflexLearner

READ_ONLY = "web_search"       # PARALLEL and not FORBID (as test_reflex_learning)
#: Captured at import, before conftest stubs it for every test.
_REAL_OBSERVE_ASYNC = _intent_brain.observe_async


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float = 5.0) -> None:
        self.now += seconds


@pytest.fixture
def rig(tmp_path):
    clock, lessons, logs, hints, steps = _Clock(), [], [], [], {}
    ledger = ReflexLearner(tmp_path / "reflex.db")
    learner = rl.RewardLearner(
        rl.RewardStore(tmp_path / "reward.db"),
        teach=lambda u, t, positive: lessons.append((u, t, positive)),
        reflexes=lambda: ledger, steps_for=lambda rid: steps.get(rid, []),
        hint=hints.append, log=logs.append, clock=clock)
    rig = SimpleNamespace(learner=learner, store=learner.store, ledger=ledger, clock=clock,
                          lessons=lessons, logs=logs, hints=hints, steps=steps)
    yield rig
    learner.store.close()
    ledger.close()


def _typed(rig, text, tool=None, args=None, ok=True, rid=0):
    """A typed turn: the words are stored, then the tool returns."""
    rig.learner.note_utterance(text)
    if tool:
        rig.clock.tick(1)
        if ok is not None:
            rig.ledger.observe(text, tool, args or {}, ok=ok)   # as observe_for_reflex does
        rig.learner.note_result(text, tool, args or {}, ok, request_id=rid)
    rig.clock.tick(5)


# ── judging the owner's words ────────────────────────────────────────────────

@pytest.mark.parametrize("text,verdict", [
    ("thanks", "praise"), ("Perfect, thank you Orion!", "praise"), ("well done", "praise"),
    ("that's right", "praise"), ("brilliant, now open spotify", "praise"),
    ("no, I meant Spotify", "correction"), ("that's wrong", "correction"),
    ("not that one", "correction"), ("try again", "correction"),
    ("thanks but that's the wrong one", "correction"), ("no no, the other one", "correction"),
    ("no", "neutral"), ("no thanks", "neutral"), ("never mind", "neutral"), ("stop", "neutral"),
    ("open spotify", "other"), ("what's the weather", "other"), ("ok", "other"),
])
def test_the_owners_words_are_judged(text, verdict):
    assert rl.judge(text) == verdict


# ── rewards ──────────────────────────────────────────────────────────────────

def test_praise_is_a_reward_taught_twice_and_logged(rig):
    _typed(rig, "search the news", READ_ONLY)
    rig.learner.note_utterance("perfect, thanks")
    assert rig.store.counts("search the news", READ_ONLY) == {
        "rewards": 1, "praise": 1, "penalties": 0, "corrections": 0, "failures": 0}
    assert rig.lessons == [("search the news", READ_ONLY, True)]
    assert any(line.startswith("LEARN: rewarded web_search for 'search the news' (1 time, praised)")
               for line in rig.logs)


def test_moving_on_or_silence_is_a_quiet_reward_but_asking_again_is_not(rig):
    _typed(rig, "search the news", READ_ONLY)
    _typed(rig, "what's on my calendar")                      # moved on: it worked
    assert rig.store.counts("search the news", READ_ONLY)["rewards"] == 1
    _typed(rig, "search the news", READ_ONLY)
    rig.clock.tick(rl.WINDOW_S + 1)
    rig.learner.note_utterance("search the news please")      # long after: settled first
    assert rig.store.counts("search the news", READ_ONLY)["rewards"] == 2
    _typed(rig, "look up the football scores", READ_ONLY)
    assert rig.learner.note_utterance("look up football scores") == "other"   # asked again
    assert rig.store.counts("look up the football scores", READ_ONLY)["rewards"] == 0
    assert rig.lessons == []                                  # only praise teaches twice


def test_no_and_stop_decide_nothing(rig):
    _typed(rig, "search the news", READ_ONLY)
    rig.learner.note_utterance("no thanks")
    assert rig.store.counts("search the news", READ_ONLY)["rewards"] == 0
    assert rig.store.counts("search the news", READ_ONLY)["penalties"] == 0


# ── mistakes ─────────────────────────────────────────────────────────────────

def test_a_correction_penalises_poisons_and_the_retry_teaches_the_right_route(rig):
    _typed(rig, "play my workout mix", "media", {"action": "play"})
    rig.learner.note_utterance("no, I meant open spotify")
    assert rig.store.counts("play my workout mix", "media", "play")["corrections"] == 1
    rig.clock.tick(1)
    rig.learner.note_result("no, I meant open spotify", "open_app", {"app": "spotify"}, True)
    assert rig.lessons == [("play my workout mix", "media", False),
                           ("play my workout mix", "open_app", True)]
    assert rig.store.counts("play my workout mix", "open_app")["rewards"] == 1
    assert any("means open_app, not media.play" in line for line in rig.logs)
    # The retry was rewarded already: moving on does not count it twice.
    rig.clock.tick(5)
    rig.learner.note_utterance("what's the time")
    assert rig.store.counts("play my workout mix", "open_app")["rewards"] == 1


def test_a_retry_with_the_same_tool_was_a_detail_not_a_wrong_route(rig):
    _typed(rig, "open steam", "open_app", {"app": "steam"})
    rig.learner.note_utterance("no, I meant the steam website")
    rig.learner.note_result("no, I meant the steam website", "open_app", {"app": "steam.com"}, True)
    assert rig.lessons == []                                  # routing left alone
    assert rig.store.counts("open steam", "open_app")["rewards"] == 1
    assert any("worked on the second try" in line for line in rig.logs)


def test_an_unrelated_request_after_a_correction_is_not_taken_as_the_retry(rig):
    _typed(rig, "play my workout mix", "media", {"action": "play"})
    rig.learner.note_utterance("that's wrong")
    _typed(rig, "what's the weather", "geo", {"action": "weather"})
    assert rig.lessons == []
    assert rig.store.counts("play my workout mix", "geo", "weather")["rewards"] == 0


def test_a_correction_withdraws_a_learned_shortcut(rig):
    for _ in range(3):
        rig.ledger.observe("search the news", READ_ONLY, {})
    assert rig.ledger.promote("search the news")
    found = rig.ledger.match("search the news")
    rig.learner.note_utterance("search the news", reflex=(found.tool, found.args()))
    rig.clock.tick(3)
    rig.learner.note_utterance("not that")
    assert rig.ledger.match("search the news") is None        # back to the model
    assert any("instant shortcut is withdrawn" in line for line in rig.logs)


def test_a_failed_tool_is_a_penalty_but_not_a_routing_lesson(rig):
    _typed(rig, "check the news", READ_ONLY, ok=False)
    assert rig.store.counts("check the news", READ_ONLY)["failures"] == 1
    rig.learner.note_utterance("thanks")                      # not a reward for a failure
    assert rig.store.counts("check the news", READ_ONLY)["rewards"] == 0
    assert rig.lessons == []


def test_a_long_job_ending_late_does_not_steal_the_current_turn(rig):
    _typed(rig, "search the news", READ_ONLY)
    _typed(rig, "what's the weather", "geo", {"action": "weather"})
    rig.learner.note_result("research quantum batteries", "research", {}, ok=False, live=False)
    assert rig.learner.pending.tool == "geo"
    assert rig.store.counts("research quantum batteries", "research")["failures"] == 1


# ── effortless repeats ───────────────────────────────────────────────────────

def _offers(rig):
    return [h for h in rig.hints if h.startswith("[Learned: ")]


def test_three_rewards_earn_an_offer_and_only_the_owners_yes_promotes(rig):
    for _ in range(3):
        _typed(rig, "search the news", READ_ONLY)
        _typed(rig, "thanks")
    offers = _offers(rig)
    assert len(offers) == 1 and "promote='search the news'" in offers[0]
    assert "Only if they say yes" in offers[0]
    assert rig.ledger.match("search the news") is None        # nothing installed unasked
    assert rig.store.stats()["offered"] == 1
    assert any("has earned an instant shortcut" in line for line in rig.logs)
    _typed(rig, "search the news", READ_ONLY)                  # a fourth reward:
    _typed(rig, "thanks")
    assert len(_offers(rig)) == 1                               # offered once, not nagged
    assert rig.ledger.promote("search the news")               # the explicit path, on yes
    assert rig.ledger.match("search the news") is not None


def test_a_call_with_details_is_never_offered_as_a_reflex(rig):
    for _ in range(3):
        _typed(rig, "search paris weather", READ_ONLY, {"query": "paris weather"})
        _typed(rig, "thanks")
    assert rig.store.counts("search paris weather", READ_ONLY)["rewards"] == 3
    assert _offers(rig) == []                                   # a reflex would drop "paris"


def test_a_corrected_route_is_never_offered(rig):
    _typed(rig, "search the news", READ_ONLY)
    rig.learner.note_utterance("that's wrong")
    for _ in range(3):
        _typed(rig, "search the news", READ_ONLY)
        _typed(rig, "thanks")
    assert _offers(rig) == []


def test_a_rewarded_multi_step_method_is_remembered_and_offered_back(rig):
    rig.steps[7] = [("web_search", "", True), ("find_tool", "", True),
                    ("document_export", "pdf", True)]
    _typed(rig, "research heat pumps and make a pdf", "web_search", rid=7)
    _typed(rig, "well done")
    assert rig.store.sequence("research heat pumps and make a pdf") == [
        "web_search", "document_export.pdf"]
    # Its first tool is not the meaning: no one-tool reward, no reflex.
    assert rig.store.counts("research heat pumps and make a pdf", "web_search")["rewards"] == 0
    rig.clock.tick(30)
    rig.learner.note_utterance("research heat pumps and make a pdf")
    assert len(rig.hints) == 1 and "web_search -> document_export.pdf" in rig.hints[0]
    assert "do not repeat any action with side effects" in rig.hints[0]


def test_a_corrected_method_is_forgotten(rig):
    rig.store.remember_sequence("tidy my downloads", ["find_files", "file_ops.move"])
    _typed(rig, "tidy my downloads", "find_files")
    rig.learner.note_utterance("that's not what i asked")
    assert rig.store.sequence("tidy my downloads") == []


def test_a_spoken_turns_transcript_after_its_tool_call_is_the_same_turn(rig):
    # Live voice: the tool call (with the partial words) comes first.
    rig.learner.note_result("search the", READ_ONLY, {}, True, request_id=3, live=True)
    rig.clock.tick(2)
    assert rig.learner.note_utterance("search the news", voice=True) == "same"
    assert rig.store.counts("search the", READ_ONLY)["rewards"] == 0
    rig.clock.tick(5)
    rig.learner.note_utterance("thank you", voice=True)
    assert rig.store.counts("search the", READ_ONLY)["praise"] == 1


def test_counts_persist_across_restarts(tmp_path):
    store = rl.RewardStore(tmp_path / "reward.db")
    store.record("praise", "search the news", READ_ONLY)
    store.record("corrected", "play music", "media", "play")
    store.close()
    again = rl.RewardStore(tmp_path / "reward.db")
    stats = again.stats()
    again.close()
    assert stats["rewards"] == 1 and stats["praise"] == 1 and stats["corrections"] == 1
    assert stats["week"] == {"rewards": 1, "penalties": 1}


# ── the live worker's hooks ──────────────────────────────────────────────────

def _worker(tmp_path, said=""):
    logs, sent = [], []
    ledger = ReflexLearner(tmp_path / "reflex.db")

    async def flush():
        sent.append(True)

    guard = SimpleNamespace(receipts=lambda rid: [])
    return SimpleNamespace(
        bus=SimpleNamespace(log=SimpleNamespace(emit=logs.append)), logs=logs,
        reflex_learner=lambda: ledger, _learned_reflex=ledger.match, _turn_said=said,
        _turn_request_id=4, _said_by_me=lambda text: text == "i am orion",
        _research_commitments=lambda: guard, _transport_ready=lambda: True,
        _pending_context_notes=deque(maxlen=24), _flush_context_notes=flush, ledger=ledger)


def test_the_hooks_judge_turns_on_a_worker_and_never_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "DEFAULT_PATH", tmp_path / "reward.db")
    worker = _worker(tmp_path)
    rl.note_utterance(worker, "search the news")
    rl.note_result(worker, "search the news", READ_ONLY, {}, True)
    rl.note_utterance(worker, "i am orion")                  # his own voice: ignored
    rl.note_utterance(worker, "brilliant")
    assert any("praised" in line for line in worker.logs)
    assert worker._reward_learner.store.counts("search the news", READ_ONLY)["praise"] == 1
    broken = SimpleNamespace()                                # no bus, nothing
    rl.note_utterance(broken, "thanks")
    rl.note_result(broken, "x", "y", {}, True)
    for learner in (worker._reward_learner, getattr(broken, "_reward_learner", None)):
        if learner is not None:
            learner.store.close()
    worker.ledger.close()


# ── the intent network's negative lesson ─────────────────────────────────────

def test_an_away_row_makes_its_label_less_likely():
    from orion_core.neural import SparseMLP
    net = SparseMLP(64, 16, 4, lr=0.05, seed=1)
    row = [3, 9, 20]
    for _ in range(40):
        net.train_batch([row], [2])
    before = float(net.predict(row)[2])
    for _ in range(40):
        net.train_batch([row], [2], away=[True])
    after = float(net.predict(row)[2])
    assert before > 0.9 and after < before / 2
    with pytest.raises(ValueError):
        net.train_batch([row], [2], away=[True, False])


def test_unlearn_removes_the_mistake_and_lowers_it(tmp_path, monkeypatch):
    import threading
    from orion_core import intent_brain
    monkeypatch.setattr(intent_brain, "_STOPPING", threading.Event())
    monkeypatch.setattr(intent_brain, "_QUIET", [None])     # no conversation to wait for
    brain = intent_brain.IntentBrain(directory=tmp_path / "brain")
    brain.labels = ["media", "open_app", "geo"]
    for _ in range(4):
        brain.learn_now("play my workout mix", "media")
    before = brain.probabilities("play my workout mix")["media"]
    assert brain.user_counts["media"] == 4 and brain.user_phrases["media"]
    assert brain.unlearn_now("play my workout mix", "media")
    after = brain.probabilities("play my workout mix")["media"]
    assert after < before
    assert not any(label == 0 for _x, label, _w in brain.replay.items)   # no replayed copy
    assert brain.user_counts["media"] == 3 and not brain.user_phrases["media"]
    assert ("play my workout mix", "media") not in brain._recent
    assert not brain.unlearn_now("anything", "never_seen_tool")


def test_a_queued_correction_reaches_unlearn(monkeypatch):
    import queue
    from orion_core import intent_brain

    class _Now(queue.Queue):
        def get(self, block=True, timeout=None):
            return super().get(block=False)

    pending = _Now()
    done = []
    fake = SimpleNamespace(learn_now=lambda u, t: done.append(("learn", u, t)),
                           unlearn_now=lambda u, t: done.append(("unlearn", u, t)),
                           flush=lambda: None)
    monkeypatch.setattr(intent_brain, "_QUEUE", pending)
    monkeypatch.setattr(intent_brain, "_FEEDER", None)
    monkeypatch.setattr(intent_brain, "_BRAIN", fake)
    monkeypatch.setattr(intent_brain, "brain", lambda: fake)
    pending.put(("open spotify", "media", intent_brain.AWAY))
    pending.put(("open spotify", "open_app", True))
    intent_brain._feed()
    assert done == [("unlearn", "open spotify", "media"), ("learn", "open spotify", "open_app")]


def test_observe_async_queues_a_correction_but_a_plain_failure_teaches_nothing(monkeypatch):
    import queue
    from orion_core import intent_brain
    pending = queue.Queue()
    monkeypatch.setattr(intent_brain, "_QUEUE", pending)
    monkeypatch.setattr(intent_brain, "_FEEDER", SimpleNamespace(is_alive=lambda: True))
    _REAL_OBSERVE_ASYNC("open spotify", "media", ok=False)
    _REAL_OBSERVE_ASYNC("open spotify", "media", negative=True)
    _REAL_OBSERVE_ASYNC("open spotify", "open_app")
    assert [pending.get_nowait() for _ in range(pending.qsize())] == [
        ("open spotify", "media", intent_brain.AWAY), ("open spotify", "open_app", True)]


def test_the_live_worker_calls_both_hooks(monkeypatch):
    from orion_core import live_worker
    seen = []
    monkeypatch.setattr(rl, "note_result", lambda *a: seen.append(("result", a[1:])))
    monkeypatch.setattr(rl, "note_utterance", lambda *a: seen.append(("utterance", a[1:])))
    fake = SimpleNamespace(reflex_learner=lambda: None,
                           memory=SimpleNamespace(log_episode=lambda role, text: None))
    live_worker.GenAILiveWorker.observe_for_reflex(fake, "search the news", READ_ONLY, {}, ok=True)
    live_worker.GenAILiveWorker._persist_episode(fake, "user", "thanks")
    live_worker.GenAILiveWorker._persist_episode(fake, "orion", "You're welcome.")
    assert seen == [("result", ("search the news", READ_ONLY, {}, True)),
                    ("utterance", ("thanks",))]


# ── what he reports about his learning ───────────────────────────────────────

def test_learning_stats_and_lines_report_rewards_reflexes_and_trend(tmp_path):
    store = rl.RewardStore(tmp_path / "reward.db")
    for _ in range(3):
        store.record("praise", "search the news", READ_ONLY)
    store.record("failed", "check mail", "email")
    store.close()
    stats = rl.learning_stats(tmp_path / "reward.db")
    assert stats["rewards"] == 3 and stats["penalties"] == 1
    lines = rl.learning_lines(stats, {"rewards": 1, "penalties": 1})
    text = "\n".join(lines)
    assert "Learning: 3 rewards (3 praised), 1 mistakes (0 corrected, 1 failed)" in text
    assert "(+2 rewards, +0 mistakes since last run)" in text
    assert "Turns judged right this week: 75%" in text
    assert "Instant reflexes learned:" in text
    assert rl.summary(tmp_path / "reward.db").startswith("What I've learned")
