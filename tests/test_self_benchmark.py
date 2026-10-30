"""Mark XXXX.2: ORION benchmarks himself - scoring is code, model-written code
is refused unless provably harmless, and a run is read against the last."""
from __future__ import annotations

import asyncio

from orion_core import self_benchmark as sb


def test_answers_are_read_from_the_answer_line_and_matched_fairly():
    assert sb.extract_answer("so 18 pens...\nANSWER: 18") == "18"
    assert sb.extract_answer("It is 12:15 then.\nANSWER: 12:15.") == "12:15"
    assert sb.extract_answer("no tag but 42 at the end") == "42"
    assert sb.answer_matches("18.0", "18") and sb.answer_matches("Friday.", "Friday")
    assert sb.answer_matches("5 pence", "5") and not sb.answer_matches("10", "5")


def test_instructions_are_checked_exactly():
    assert sb.check_instruction("Pale, cratered, silent, distant, ancient, bright, cold", "word_count", 7)
    assert not sb.check_instruction("six words only here right now", "word_count", 7)
    assert sb.check_instruction('{"city": "Tokyo", "country": "Japan"}', "json_keys",
                                {"city": "tokyo", "country": "japan"})
    assert not sb.check_instruction("Dark waves crash on sand.", "no_letter", "e")
    assert sb.check_instruction("- red\n- blue\n- yellow", "bullets", 3)
    assert sb.check_instruction("NEBULA NEBULA NEBULA", "exact", "NEBULA NEBULA NEBULA")


def test_plans_must_hold_the_tools_in_order():
    assert sb.plan_tools('["geo", "reminder"]') == ["geo", "reminder"]
    assert sb.plan_ok(["geo", "reminder"], ("geo", "reminder"))
    assert not sb.plan_ok(["reminder", "geo"], ("geo", "reminder"))
    assert sb.plan_tools("First geo, then reminder.") == ["geo", "reminder"]


def test_model_code_that_reaches_the_system_is_refused():
    assert sb.unsafe_reason("import os\nos.remove('x')") == "imports os"
    assert sb.unsafe_reason("def f():\n    return open('x').read()") == "uses open"
    assert sb.unsafe_reason("def f(x):\n    return x.__class__") == "reaches __class__"
    assert sb.unsafe_reason("import math\ndef f(x):\n    return math.sqrt(x)") == ""
    assert sb.unsafe_reason("def f(:") .startswith("does not parse")


def test_code_is_found_in_any_fence_or_after_prose():
    assert sb.code_from("Here:\n```py\ndef f():\n    return 1\n```") == "def f():\n    return 1"
    assert sb.code_from("Sure - this works:\ndef f():\n    return 1") == "def f():\n    return 1"


def test_safe_code_runs_in_its_own_process_and_is_scored():
    good = "def double(x):\n    return 2 * x"
    assert sb.run_code(good, ("double(4) == 8",)) == (True, "passed")
    passed, why = sb.run_code("def double(x):\n    return x", ("double(4) == 8",))
    assert not passed and "AssertionError" in why
    assert sb.run_code("import subprocess", ()) == (False, "refused: imports subprocess")


def test_a_run_is_scored_saved_and_compared(tmp_path):
    answers = {q: f"ANSWER: {a}" for q, a in sb.REASONING}

    async def ask(prompt: str) -> str:
        for question, reply in answers.items():
            if question in prompt:
                return reply
        if "JSON list of the tool names" in prompt:
            return '["geo", "reminder", "web_search", "document_export", "antivirus", ' \
                   '"security_watch", "open_app", "focus"]'
        return ""

    def route(request: str) -> list[str]:
        return [next(iter(right)) for req, right in sb.ROUTING if req == request]

    run = asyncio.run(sb.run_benchmark(ask, categories=("reasoning", "agentic"), route=route))
    assert run.categories["reasoning"].percent == 100.0
    assert run.categories["agentic"].passed == run.categories["agentic"].total
    path = tmp_path / "bench.jsonl"
    sb.save(run, path)
    saved = sb.history(path)
    assert saved and saved[-1]["overall"] == 100.0
    text = sb.report(run, {"overall": 80.0, "categories": {"reasoning": {"passed": 6, "total": 12}}})
    assert text.startswith("Self-benchmark: 100% overall (+20)")
    assert "Reasoning: 12/12 (100%) (+50)" in text


def test_a_run_reports_how_he_learns_and_the_change_since_last_time(tmp_path):
    async def ask(prompt: str) -> str:
        return ""

    def route(request: str) -> list[str]:
        return [next(iter(right)) for req, right in sb.ROUTING if req == request]

    run = asyncio.run(sb.run_benchmark(ask, categories=("agentic",), route=route))
    total = len(sb.ROUTING)
    assert run.learning["routing"] == {"passed": total, "total": total}
    run.learning.update({"rewards": 9, "praise": 4, "penalties": 1, "corrections": 1,
                         "failures": 0, "offered": 1, "sequences": 2,
                         "reflexes": {"learned": 2, "candidates": 1},
                         "week": {"rewards": 9, "penalties": 1}})
    sb.save(run, tmp_path / "bench.jsonl")
    assert sb.history(tmp_path / "bench.jsonl")[-1]["learning"]["rewards"] == 9
    previous = {"learning": {"routing": {"passed": total - 2, "total": total},
                             "rewards": 5, "penalties": 1}}
    text = sb.report(run, previous)
    assert f"Routing (resolver + learned network): {total}/{total} (100%) (+" in text
    assert "Learning: 9 rewards (4 praised), 1 mistakes (1 corrected, 0 failed) " \
           "(+4 rewards, +0 mistakes since last run)" in text
    assert "Turns judged right this week: 90%" in text
    assert "Instant reflexes learned: 2, 1 candidate(s) (1 offered after rewards); " \
           "2 multi-step method(s) remembered" in text


def test_benchmarking_himself_on_request_includes_his_learning(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from orion_core import intent_brain, reward_learning
    from orion_core.dispatch_files import FilesDispatchMixin
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    monkeypatch.setattr(reward_learning, "DEFAULT_PATH", tmp_path / "reward.db")
    monkeypatch.setattr(sb, "HISTORY_PATH", tmp_path / "bench.jsonl")
    routed = []
    real_scores = intent_brain.routing_scores
    monkeypatch.setattr(intent_brain, "routing_scores",
                        lambda q, d: routed.append(q) or real_scores(q, d))
    store = reward_learning.RewardStore()            # conftest: the test data folder
    store.record("praise", "search the news", "web_search")
    store.close()

    class _Router:
        async def generate_text(self, prompt, **_kw):
            return SimpleNamespace(name="stub"), '["geo", "reminder"]'

    host = SimpleNamespace(ai=SimpleNamespace(router=_Router()), TOOL_DECLARATIONS=TOOL_DECLARATIONS,
                           bus=SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    result = asyncio.run(FilesDispatchMixin._self_benchmark(host, {"category": "agentic"}))
    assert len(routed) == len(sb.ROUTING)            # routed as he really routes
    assert "Routing (resolver + learned network):" in result.text
    assert "Learning: 1 rewards (1 praised)" in result.text
    assert sb.history()[-1]["learning"]["rewards"] == 1


def test_the_benchmark_is_reachable_from_his_diagnostics_tool():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    diag = next(d for d in TOOL_DECLARATIONS if d["name"] == "diagnostics")
    assert "benchmark" in diag["parameters"]["properties"]["action"]["description"]
    assert "category" in diag["parameters"]["properties"]


def test_the_known_ways_round_the_code_check_are_refused():
    """Mark XXXX.4: these passed the allow-list and ran with full file access."""
    from orion_core.self_benchmark import unsafe_reason
    for code in ('__builtins__.eval("1+1")',
                 'import operator\nf = operator.attrgetter("__globals__")',
                 'x = "{0.__globals__}".format(len)',
                 'import math\nmath.eval = 1'):
        assert unsafe_reason(code), code
    assert unsafe_reason("def add(a, b):\n    return a + b") == ""
    assert unsafe_reason('def f():\n    return 1\n\nif __name__ == "__main__":\n    print(f())') == ""
    assert unsafe_reason("from functools import reduce\nimport operator\n"
                         "total = reduce(operator.add, [1, 2, 3])") == ""
