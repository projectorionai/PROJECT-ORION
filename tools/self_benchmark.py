"""Run ORION's self-benchmark outside the app, on his real text providers.

    .venv\\Scripts\\python.exe tools\\self_benchmark.py [reasoning coding instructions agentic]

The same tasks and scoring ORION runs when asked "benchmark yourself"
(orion_core/self_benchmark.py): his configured providers answer, his own tool
resolver routes, and the result is saved beside his own runs. Reads the
provider settings; prints no key.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Quiet:
    def emit(self, *_a: Any, **_k: Any) -> None:
        pass

    def connect(self, *_a: Any, **_k: Any) -> None:
        pass


class _QuietBus:
    def __getattr__(self, _name: str) -> _Quiet:
        return _Quiet()


class _NoMemory:
    def prompt_context(self, limit: int = 16) -> str:
        return ""

    def query(self, *_a: Any, **_k: Any) -> list[Any]:
        return []

    def remember(self, *_a: Any, **_k: Any) -> None:
        return None


async def main(categories: list[str]) -> int:
    from orion_core import self_benchmark
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.providers import ProviderRouter, read_provider_settings
    from orion_core.tool_resolver import hybrid_scores

    router = ProviderRouter(read_provider_settings(), _QuietBus(), memory=_NoMemory())
    used: list[str] = []

    async def ask(prompt: str) -> str:
        profile, text = await router.generate_text(prompt, task="self_benchmark",
                                                   budget="background", deadline_s=120.0)
        used.append(str(getattr(profile, "name", "") or ""))
        return str(text or "")

    menu = [d for d in TOOL_DECLARATIONS if d["name"] not in {"find_tool", "use_tool"}]

    def route(request: str) -> list[str]:
        scores = hybrid_scores(request, menu)
        return [n for n in sorted(scores, key=lambda n: (-scores[n], n)) if scores[n] > 0][:3]

    previous = (self_benchmark.history(None, 1) or [None])[-1]
    run = await self_benchmark.run_benchmark(
        ask, categories=categories or self_benchmark.CATEGORIES, route=route,
        say=lambda m: print(m, flush=True))
    run.provider = max(set(used), key=used.count) if used else ""
    self_benchmark.save(run)
    print(self_benchmark.report(run, previous))
    print("\nEvery miss:")
    for score in run.categories.values():
        for miss in score.misses:
            print(f"  {score.name}: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main([a.lower() for a in sys.argv[1:]])))
