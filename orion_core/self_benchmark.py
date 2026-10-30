"""ORION benchmarks himself: reasoning, coding, instruction following and
agentic skill, every answer scored by code - never by a model's opinion of
itself.

  reasoning     word problems and logic with one right answer ("ANSWER: 42")
  coding        a function written from a spec, run against hidden tests in
                a separate, time-limited process
  instructions  constraints that can be checked exactly (word counts, JSON
                shapes, forbidden letters)
  agentic       choosing the right tool for a request (ORION's own resolver,
                no model call) and planning multi-step work as tool calls

Results are kept (CONFIG_DIR/benchmarks/self_benchmark.jsonl) so a run is
read against the last one: a provider swap, a persona change or a new Mark
shows up as a number, not an impression.

Model-written code never runs in ORION's process. It is parsed first and
refused unless it imports only pure standard modules and calls nothing that
reaches the system (open, eval, exec, __import__, getattr...); then it runs
under the forge's interpreter (forge_runtime.verifier_command: this Python
from source, ORION.exe --forge-verify when installed) with a timeout, in an
empty temporary folder, with a scrubbed environment.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .constants import CONFIG_DIR

HISTORY_PATH = CONFIG_DIR / "benchmarks" / "self_benchmark.jsonl"
CATEGORIES = ("reasoning", "coding", "instructions", "agentic")
CODE_TIMEOUT_S = 10.0
#: Model calls in flight at once - a benchmark must not starve conversation.
CONCURRENCY = 3

Ask = Callable[[str], Awaitable[str]]


# ── the tasks (ORION's own; every answer checkable) ─────────────────────────

REASONING: tuple[tuple[str, str], ...] = (
    ("A shop sells pens at 3 for 2 pounds. How many pounds do 27 pens cost?", "18"),
    ("A train leaves at 09:40 and the journey takes 2 hours 35 minutes. At what time "
     "does it arrive? Give the time as HH:MM.", "12:15"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines take "
     "to make 100 widgets?", "5"),
    ("A bat and a ball cost 1.10 pounds together. The bat costs 1.00 pound more than "
     "the ball. How much does the ball cost, in pence?", "5"),
    ("What is the sum of all whole numbers from 1 to 200 that are divisible by 7?", "2842"),
    ("Anna is older than Ben. Ben is older than Cara. Dev is younger than Cara. Who is "
     "the second youngest? Answer with the name only.", "Cara"),
    ("A price rises by 20 percent and then falls by 20 percent. By what percent is the "
     "final price lower than the original?", "4"),
    ("How many times does the digit 1 appear when writing all whole numbers from 1 to "
     "100?", "21"),
    ("A rectangle has a perimeter of 36 cm and its length is twice its width. What is "
     "its area in square centimetres?", "72"),
    ("It is Wednesday today. What day of the week will it be 100 days from now? Answer "
     "with the day name only.", "Friday"),
    ("A tank fills in 6 hours with pipe A and in 3 hours with pipe B. With both pipes "
     "open, how many hours does it take to fill?", "2"),
    ("What is the smallest positive whole number that leaves a remainder of 1 when "
     "divided by 2, 3, 4, 5 and 6, and is divisible by 7?", "301"),
)

#: (function name, specification, hidden tests as Python expressions)
CODING: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("is_palindrome", "is_palindrome(text) returns True when text reads the same "
     "backwards, ignoring case, spaces and punctuation.",
     ("is_palindrome('A man, a plan, a canal: Panama') is True",
      "is_palindrome('Orion') is False", "is_palindrome('') is True")),
    ("merge_intervals", "merge_intervals(pairs) takes a list of [start, end] pairs and "
     "returns the merged, sorted list of overlapping or touching intervals as lists.",
     ("merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
      "merge_intervals([[1,4],[4,5]]) == [[1,5]]", "merge_intervals([]) == []")),
    ("roman_to_int", "roman_to_int(numeral) converts a Roman numeral string (I to MMMCMXCIX) "
     "to an int.",
     ("roman_to_int('XL') == 40", "roman_to_int('XXXIX') == 39",
      "roman_to_int('MCMXCIV') == 1994")),
    ("top_k_words", "top_k_words(text, k) returns the k most frequent lowercase words "
     "(letters only), most frequent first, ties broken alphabetically.",
     ("top_k_words('b a b c a b', 2) == ['b', 'a']",
      "top_k_words('Dog cat dog CAT bird', 3) == ['cat', 'dog', 'bird']")),
    ("shortest_path_len", "shortest_path_len(grid) takes a list of equal-length strings "
     "of '.' (open) and '#' (wall) and returns the number of steps of the shortest "
     "4-directional path from the top-left to the bottom-right cell, or -1 if none.",
     ("shortest_path_len(['..', '..']) == 2",
      "shortest_path_len(['.#', '#.']) == -1",
      "shortest_path_len(['...', '##.', '...']) == 4")),
    ("break_even_roas", "break_even_roas(margin) returns the return on ad spend at which "
     "an ad breaks even for a product with gross margin *margin* (a fraction, 0 < margin "
     "<= 1), rounded to 2 decimal places.",
     ("break_even_roas(0.5) == 2.0", "break_even_roas(0.25) == 4.0",
      "break_even_roas(0.3) == 3.33")),
)

#: (prompt, checker name, checker argument)
INSTRUCTIONS: tuple[tuple[str, str, Any], ...] = (
    ("Describe the Moon in exactly seven words. Reply with the seven words only.",
     "word_count", 7),
    ("Reply with a JSON object with the keys \"city\" and \"country\" for the capital "
     "of Japan, and nothing else.", "json_keys", {"city": "tokyo", "country": "japan"}),
    ("Write one sentence about the sea without using the letter 'e'.", "no_letter", "e"),
    ("List three primary colours as a bulleted list, one per line, each line starting "
     "with '- ', and nothing else.", "bullets", 3),
    # A neutral word: his own name has a written form of its own (O.R.I.O.N.),
    # which tested his brand rule, not his instruction following.
    ("Reply with the word NEBULA in capital letters, repeated exactly three times, "
     "separated by single spaces, and nothing else.", "exact", "NEBULA NEBULA NEBULA"),
)

#: (request, tools that may come first) - routed by ORION's own resolver.
ROUTING: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("what's the weather like tomorrow", ("geo",)),
    ("remind me to call mum at six", ("reminder", "awareness")),
    ("open spotify", ("open_app", "desktop_control", "entertainment")),
    ("how secure is my computer", ("security_watch", "antivirus")),
    ("sing me a lullaby", ("sing",)),
    ("scan my downloads for viruses", ("antivirus", "security_watch")),
    ("show me paris on the globe", ("globe", "max_zoom_in_globe", "geo")),
    ("play my focus playlist", ("entertainment", "media_control")),
    ("research the latest breakthroughs in fusion energy", ("research", "antigravity_research",
                                                            "web_search")),
    ("what's on my calendar today", ("morning_briefing", "briefing", "awareness", "notion_workspace")),
    ("start a 25 minute focus block", ("focus",)),
    ("how have i been feeling this week", ("emotion", "wellbeing")),
)

#: (goal, tools the plan must contain, in this order)
PLANNING: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Find out tomorrow's weather in Colchester and remind me at 7am to take a coat if "
     "it will rain.", ("geo", "reminder")),
    ("Research TikTok Shop fees for UK sellers and save a short report as a document.",
     ("web_search", "document_export")),
    ("Check whether my Downloads folder has anything dangerous, then tell me how secure "
     "my PC is overall.", ("antivirus", "security_watch")),
    ("Open Spotify and play my focus playlist, then start a 25 minute focus block.",
     ("open_app", "focus")),
)
PLAN_MENU = ("geo", "reminder", "web_search", "document_export", "antivirus",
             "security_watch", "open_app", "entertainment", "focus", "morning_briefing",
             "outlook_mail", "globe", "sing", "emotion", "calculate")


# ── scoring ──────────────────────────────────────────────────────────────────

_ANSWER = re.compile(r"ANSWER\s*[:=]\s*(.+)", re.I)


def extract_answer(reply: str) -> str:
    """The value after the last 'ANSWER:' line (or the last number)."""
    found = _ANSWER.findall(str(reply or ""))
    if found:
        return found[-1].strip().strip(".*` ")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(reply or ""))
    return numbers[-1] if numbers else ""


def answer_matches(got: str, expected: str) -> bool:
    got = str(got or "").strip().lower().rstrip(".")
    want = str(expected).strip().lower()
    if got == want:
        return True
    nums = re.findall(r"-?\d+(?:\.\d+)?", got)
    if nums and re.fullmatch(r"-?\d+(?:\.\d+)?", want):
        return abs(float(nums[0]) - float(want)) < 1e-6
    return bool(re.search(rf"\b{re.escape(want)}\b", got))


def check_instruction(reply: str, kind: str, arg: Any) -> bool:
    text = str(reply or "").strip()
    if kind == "word_count":
        return len(re.findall(r"[A-Za-z0-9'-]+", text)) == int(arg)
    if kind == "json_keys":
        match = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(match.group(0)) if match else None
        except ValueError:
            return False
        return isinstance(data, dict) and all(
            str(data.get(k, "")).strip().lower() == v for k, v in dict(arg).items())
    if kind == "no_letter":
        return bool(text) and str(arg).lower() not in text.lower()
    if kind == "bullets":
        lines = [line for line in text.splitlines() if line.strip()]
        return len(lines) == int(arg) and all(line.startswith("- ") for line in lines)
    if kind == "exact":
        return text.strip("`\" ") == str(arg)
    return False


def plan_tools(reply: str) -> list[str]:
    """Tool names from a plan reply: a JSON list, or names in order of mention."""
    match = re.search(r"\[[^\[\]]*\]", str(reply or ""), re.S)
    if match:
        try:
            items = json.loads(match.group(0))
            names = [str(i.get("tool") if isinstance(i, dict) else i).strip() for i in items]
            return [n for n in names if n]
        except ValueError:
            pass
    return re.findall(r"\b(" + "|".join(PLAN_MENU) + r")\b", str(reply or ""))


def plan_ok(tools: list[str], required: Iterable[str]) -> bool:
    """Every required tool present, in the required order."""
    position = -1
    for name in required:
        try:
            position = tools.index(name, position + 1)
        except ValueError:
            return False
    return True


# ── running model-written code ──────────────────────────────────────────────

SAFE_IMPORTS = frozenset({"math", "itertools", "functools", "collections", "re", "string",
                          "heapq", "bisect", "typing", "dataclasses", "statistics", "fractions",
                          "decimal", "operator"})
FORBIDDEN_NAMES = frozenset({"open", "eval", "exec", "compile", "__import__", "input",
                             "globals", "locals", "vars", "getattr", "setattr", "delattr",
                             "breakpoint", "help", "exit", "quit", "memoryview"})
#: Mark XXXX.4: reached as attributes too - ``__builtins__.eval(...)`` and
#: ``operator.attrgetter("__globals__")`` both passed the name check.
FORBIDDEN_ATTRS = FORBIDDEN_NAMES | {"attrgetter", "methodcaller"}
_DUNDER_TEXT = re.compile(r"__\w+__")
#: The one dunder idiom ordinary code uses: ``if __name__ == "__main__":``.
_MAIN_GUARD = ("__name__", "__main__")


def code_from(reply: str) -> str:
    """The code in a reply: a fenced block (any language tag), else everything
    from the first line that starts Python code - prose around it is not code."""
    text = str(reply or "")
    match = re.search(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", text, re.S)
    if match:
        return match.group(1).strip()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"(?:def |import |from |class |@)", line):
            return "\n".join(lines[index:]).strip().rstrip("`").strip()
    return text.strip()


def unsafe_reason(code: str) -> str:
    """Why *code* may not run, or "" when it may."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"does not parse ({exc.msg})"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in SAFE_IMPORTS:
                    return f"imports {name}"
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES
                                             or (node.id.startswith("__")
                                                 and node.id != _MAIN_GUARD[0])):
            return f"uses {node.id}"
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__")
                                                  or node.attr in FORBIDDEN_ATTRS):
            return f"reaches {node.attr}"
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and node.value != _MAIN_GUARD[1] and _DUNDER_TEXT.search(node.value)):
            # "{0.__globals__}".format(f) and attrgetter strings walk to the
            # interpreter's internals without a single dunder attribute.
            return "names an interpreter internal in a string"
    return ""


def run_code(code: str, tests: Iterable[str], timeout: float = CODE_TIMEOUT_S) -> tuple[bool, str]:
    """Run *code* plus its tests in a child process. (passed, why)."""
    reason = unsafe_reason(code)
    if reason:
        return False, f"refused: {reason}"
    checks = "\n".join(f"assert {t}, {t!r}" for t in tests)
    script = f"{code}\n\n{checks}\nprint('ALL_PASS')\n"
    from .forge_runtime import verifier_command
    with tempfile.TemporaryDirectory(prefix="orion-bench-") as folder:
        path = Path(folder) / "bench_task.py"
        path.write_text(script, encoding="utf-8")
        env = {k: v for k, v in os.environ.items()
               if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PYTHONIOENCODING"}}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            done = subprocess.run(verifier_command(str(path)), cwd=folder, env=env,
                                  capture_output=True, text=True, errors="replace",
                                  timeout=timeout, stdin=subprocess.DEVNULL,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return False, "timed out"
        except OSError as exc:
            return False, f"could not run ({exc})"
    if done.returncode == 0 and "ALL_PASS" in done.stdout:
        return True, "passed"
    last = (done.stderr.strip().splitlines() or ["failed"])[-1]
    return False, last[:160]


# ── the run ──────────────────────────────────────────────────────────────────

@dataclass
class CategoryScore:
    name: str
    passed: int = 0
    total: int = 0
    seconds: float = 0.0
    misses: list[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        return 100.0 * self.passed / self.total if self.total else 0.0


@dataclass
class BenchmarkRun:
    at: float
    provider: str
    categories: dict[str, CategoryScore]
    #: How he learns, not how he answers: his routing network's score on the
    #: fixed routing set, rewards, mistakes and reflexes (reward_learning).
    learning: dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        scored = [c for c in self.categories.values() if c.total]
        return sum(c.percent for c in scored) / len(scored) if scored else 0.0

    def to_record(self) -> dict[str, Any]:
        record = {"at": round(self.at, 1), "provider": self.provider,
                  "overall": round(self.overall, 1),
                  "categories": {k: {"passed": c.passed, "total": c.total,
                                     "seconds": round(c.seconds, 1)}
                                 for k, c in self.categories.items()}}
        if self.learning:
            record["learning"] = self.learning
        return record


_REASONING_RULE = ("Solve this carefully. Work it out step by step, check your result, "
                   "then finish with a final line exactly 'ANSWER: <answer>'.\n\n")
_CODE_RULE = ("Write a Python function. Standard library only (math, itertools, "
              "collections, re and similar); no input, no files, no printing. Reply with "
              "only the code in one ```python block.\n\n")
_PLAN_RULE = ("Plan this as tool calls. Tools available: {menu}. Reply with only a JSON "
              "list of the tool names in the order you would call them.\n\nGoal: ")


async def _timed(ask: Ask, prompt: str) -> tuple[str, float]:
    started = time.monotonic()
    try:
        reply = await ask(prompt)
    except Exception as exc:                      # a provider failing is a miss, not a crash
        reply = f"[error: {exc}]"
    return str(reply or ""), time.monotonic() - started


async def run_benchmark(ask: Ask, *, categories: Iterable[str] = CATEGORIES,
                        route: Callable[[str], list[str]] | None = None,
                        provider: str = "", clock: Callable[[], float] = time.time,
                        say: Callable[[str], Any] | None = None) -> BenchmarkRun:
    """Run the chosen categories. *ask* sends one prompt to ORION's text brain;
    *route* ranks tools for a request (ORION's resolver) for the agentic part."""
    wanted = [c for c in categories if c in CATEGORIES] or list(CATEGORIES)
    gate = asyncio.Semaphore(CONCURRENCY)
    scores = {name: CategoryScore(name) for name in wanted}

    async def one(prompt: str) -> tuple[str, float]:
        async with gate:
            return await _timed(ask, prompt)

    def note(message: str) -> None:
        if say is not None:
            try:
                say(message)
            except Exception:
                pass

    if "reasoning" in scores:
        note("benchmark: reasoning")
        replies = await asyncio.gather(*(one(_REASONING_RULE + q) for q, _ in REASONING))
        for (question, expected), (reply, seconds) in zip(REASONING, replies):
            score = scores["reasoning"]
            score.total += 1
            score.seconds += seconds
            got = extract_answer(reply)
            if answer_matches(got, expected):
                score.passed += 1
            else:
                score.misses.append(f"{question[:50]}... wanted {expected}, got {got or '-'}")
    if "coding" in scores:
        note("benchmark: coding")
        replies = await asyncio.gather(*(one(_CODE_RULE + spec) for _n, spec, _t in CODING))
        for (name, _spec, tests), (reply, seconds) in zip(CODING, replies):
            score = scores["coding"]
            score.total += 1
            score.seconds += seconds
            passed, why = await asyncio.to_thread(run_code, code_from(reply), tests)
            if passed:
                score.passed += 1
            else:
                score.misses.append(f"{name}: {why}")
    if "instructions" in scores:
        note("benchmark: instructions")
        replies = await asyncio.gather(*(one(p) for p, _k, _a in INSTRUCTIONS))
        for (prompt, kind, arg), (reply, seconds) in zip(INSTRUCTIONS, replies):
            score = scores["instructions"]
            score.total += 1
            score.seconds += seconds
            if check_instruction(reply, kind, arg):
                score.passed += 1
            else:
                score.misses.append(f"{kind}: {reply[:60]!r}")
    learning: dict[str, Any] = {}
    if "agentic" in scores:
        note("benchmark: agentic")
        score = scores["agentic"]
        if route is not None:
            routed = 0
            for request, right in ROUTING:
                score.total += 1
                started = time.monotonic()
                try:
                    ranked = route(request)
                except Exception:
                    ranked = []
                score.seconds += time.monotonic() - started
                if ranked and ranked[0] in right:
                    score.passed += 1
                    routed += 1
                else:
                    score.misses.append(f"route '{request}': got {ranked[:1] or '-'}")
            learning["routing"] = {"passed": routed, "total": len(ROUTING)}
        menu = ", ".join(PLAN_MENU)
        replies = await asyncio.gather(*(one(_PLAN_RULE.format(menu=menu) + goal)
                                         for goal, _r in PLANNING))
        for (goal, required), (reply, seconds) in zip(PLANNING, replies):
            score.total += 1
            score.seconds += seconds
            tools = plan_tools(reply)
            if plan_ok(tools, required):
                score.passed += 1
            else:
                score.misses.append(f"plan '{goal[:40]}...': {tools}")
    return BenchmarkRun(clock(), provider, scores, learning)


# ── keeping and reading results ─────────────────────────────────────────────

def save(run: BenchmarkRun, path: Path | None = None) -> None:
    target = Path(path or HISTORY_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(run.to_record()) + "\n")
    except OSError:
        pass


def history(path: Path | None = None, limit: int = 20) -> list[dict[str, Any]]:
    try:
        lines = Path(path or HISTORY_PATH).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    found = []
    for line in lines[-limit:]:
        try:
            found.append(json.loads(line))
        except ValueError:
            continue
    return found


def report(run: BenchmarkRun, previous: dict[str, Any] | None = None) -> str:
    """The run as a short report, with the change since the last one."""
    def delta(now: float, before: Any) -> str:
        if not isinstance(before, (int, float)):
            return ""
        change = now - float(before)
        return f" ({'+' if change >= 0 else ''}{change:.0f})" if abs(change) >= 0.5 else " (=)"

    head = (f"Self-benchmark: {run.overall:.0f}% overall"
            + delta(run.overall, (previous or {}).get("overall"))
            + (f", answered by {run.provider}" if run.provider else "") + ".")
    lines = [head]
    before = (previous or {}).get("categories", {})
    for name, score in run.categories.items():
        if not score.total:
            continue
        old = before.get(name) or {}
        old_pct = 100.0 * old["passed"] / old["total"] if old.get("total") else None
        lines.append(f"  {name.title()}: {score.passed}/{score.total} ({score.percent:.0f}%)"
                     + delta(score.percent, old_pct)
                     + f", {score.seconds / max(1, score.total):.1f} s each")
        for miss in score.misses[:3]:
            lines.append(f"    missed - {miss}")
    lines.extend(learning_report(run.learning, (previous or {}).get("learning") or {}))
    return "\n".join(lines)


def learning_report(learning: dict[str, Any], previous: dict[str, Any] | None = None) -> list[str]:
    """How he is learning: his routing network on the fixed set (with the
    change since the last run) and reward_learning's stats."""
    previous = previous or {}
    lines: list[str] = []
    routing = learning.get("routing") or {}
    if routing.get("total"):
        pct = 100.0 * routing["passed"] / routing["total"]
        old = previous.get("routing") or {}
        change = ""
        if old.get("total"):
            diff = pct - 100.0 * old["passed"] / old["total"]
            change = f" ({'+' if diff >= 0 else ''}{diff:.0f})" if abs(diff) >= 0.5 else " (=)"
        lines.append(f"  Routing (resolver + learned network): {routing['passed']}/{routing['total']} "
                     f"({pct:.0f}%){change}")
    try:
        from .reward_learning import learning_lines
        lines.extend(learning_lines(learning, previous))
    except Exception:
        pass
    return lines


__all__ = ["BenchmarkRun", "CATEGORIES", "CategoryScore", "HISTORY_PATH", "answer_matches",
           "check_instruction", "code_from", "extract_answer", "history", "plan_ok",
           "plan_tools", "report", "run_benchmark", "run_code", "save", "unsafe_reason"]
