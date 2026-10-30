"""
Keeping the measuring stick out of the training set.

A fine-tuned model is promoted only if it beats the current one on ORION's
evaluation sets. If the sentences it was trained on ARE the evaluation
sentences, the comparison measures memorisation and the promotion gate means
nothing. So before anything is written, every candidate row's user text is
compared with every evaluation string, and a row that is the same request (or
nearly) is dropped:

* ``tools/planner_eval_set.json`` - states "never train on these";
* the hostile and multi-step lists in ``tools/eval_local_agent.py``;
* the held-out (odd-index) half of ``tests/data/intent_routing_cases.py`` and
  ``tests/data/tool_routing_cases.py`` - their own docstrings reserve that half
  as "the honest score".

Two requests are "the same" when their content words (stop-words and
politeness removed) overlap by at least ``THRESHOLD`` (Jaccard), or are equal.
Training on the routing sets' development half is allowed and intended.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .toolcalls import sha256_text

THRESHOLD = 0.75

_STOP = frozenset("""
a an the to of for me my mine i you your it its is are was be on in at by with and or
please can could would will orion hey ok okay just quickly now thanks thank some any
that this these those so then do does did have has had up out
""".split())


def content_tokens(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9']+", str(text or "").lower().replace("’", "'"))
    return frozenset(w for w in words if w not in _STOP)


def similar(a: frozenset[str], b: frozenset[str], threshold: float = THRESHOLD) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    inter = len(a & b)
    if not inter:
        return False
    return inter / len(a | b) >= threshold


class EvalGuard:
    """Flags text that duplicates an evaluation string."""

    def __init__(self, strings: Iterable[str]) -> None:
        self.strings = [s for s in dict.fromkeys(str(x).strip() for x in strings) if s]
        self._tokens = [content_tokens(s) for s in self.strings]
        self._index: dict[str, list[int]] = {}
        for i, tokens in enumerate(self._tokens):
            for token in tokens:
                self._index.setdefault(token, []).append(i)
        self.fingerprint = sha256_text("\n".join(sorted(self.strings)))

    def __len__(self) -> int:
        return len(self.strings)

    def overlaps(self, text: str) -> bool:
        tokens = content_tokens(text)
        if not tokens:
            return False
        seen: set[int] = set()
        for token in tokens:
            for i in self._index.get(token, ()):
                if i not in seen:
                    seen.add(i)
                    if similar(tokens, self._tokens[i]):
                        return True
        return False


# ── loading the evaluation strings ───────────────────────────────────────────────

def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planner_strings(path: Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for case in data.get("cases", []) if isinstance(data, dict) else []:
        if isinstance(case, dict):
            out.append(str(case.get("goal") or ""))
    return out


def agent_eval_strings(path: Path) -> list[str]:
    """The request sentences in tools/eval_local_agent.py (EXTRA_SAFETY and
    MULTI_STEP), read from the source without importing it."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    wanted = {"EXTRA_SAFETY", "MULTI_STEP"}
    found: list[str] = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if wanted & set(targets):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and " " in sub.value:
                    found.append(sub.value)
    return found


def routing_split_cases(path: Path, which: str) -> list[tuple[str, ...]]:
    """The 'dev' (even-index) or 'holdout' (odd-index) cases of a routing set;
    'all' is every case. Cases are tuples whose first item is the request."""
    module = _load_module(path, f"_orion_routing_{abs(hash(str(path)))}")
    if which == "dev":
        return list(module.dev_cases())
    if which == "holdout":
        return list(module.holdout_cases())
    return list(module.CASES)


def collect_eval_strings(root: Path, *, include_routing_holdout: bool = True,
                         planner: Path | None = None, agent_eval: Path | None = None,
                         routing: Path | None = None, tool_routing: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    """(strings, report). Missing files are reported, not fatal: a build run
    from an installed copy has no tests folder, and says so in its manifest."""
    root = Path(root)
    report: dict[str, Any] = {"sources": {}, "missing": []}
    strings: list[str] = []
    jobs = [("planner_eval_set", planner or root / "tools" / "planner_eval_set.json", planner_strings),
            ("eval_local_agent", agent_eval or root / "tools" / "eval_local_agent.py", agent_eval_strings)]
    if include_routing_holdout:
        jobs += [("routing_holdout", routing or root / "tests" / "data" / "intent_routing_cases.py",
                  lambda p: [c[0] for c in routing_split_cases(p, "holdout")]),
                 ("tool_routing_holdout", tool_routing or root / "tests" / "data" / "tool_routing_cases.py",
                  lambda p: [c[0] for c in routing_split_cases(p, "holdout")])]
    for name, path, loader in jobs:
        path = Path(path)
        if not path.is_file():
            report["missing"].append(name)
            continue
        try:
            found = [s for s in loader(path) if s]
        except Exception as exc:                     # noqa: BLE001 - reported, never fatal
            report["missing"].append(f"{name} ({type(exc).__name__})")
            continue
        report["sources"][name] = len(found)
        strings += found
    return strings, report


__all__ = ["EvalGuard", "THRESHOLD", "agent_eval_strings", "collect_eval_strings",
           "content_tokens", "planner_strings", "routing_split_cases", "similar"]
