"""ORION acts on what he notices, without spending a token to decide.

The thought stream narrated what ORION observed and did nothing about it. He
would note that eight megabytes of his memory were sitting in write-ahead logs
and leave them there.

The constraint shapes the design: acting must cost no tokens, which rules out
handing the thought back to a model and asking what it meant. So nothing here
reads a thought's WORDS. Every action is keyed to a fact ORION can measure —
which is also the safer design, because a measurement cannot be hallucinated
and a generated sentence can.

Offline: a temporary directory, no databases of consequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.thought_actions import (  # noqa: E402
    WAL_BYTES_BEFORE_CHECKPOINT,
    ThoughtAction,
    ThoughtActor,
)


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, line: str) -> None:
        self.lines.append(line)


class _Bus:
    def __init__(self) -> None:
        self.log = _Log()


class _Housekeeper:
    """Stands in for the real one; records that it was asked."""

    def __init__(self, on_checkpoint=None) -> None:
        self.calls = 0
        self._on_checkpoint = on_checkpoint

    def checkpoint_all(self) -> int:
        self.calls += 1
        if self._on_checkpoint is not None:
            self._on_checkpoint()
        return 3


def _wal(directory: Path, name: str, size: int) -> Path:
    path = directory / f"{name}.db-wal"
    path.write_bytes(b"\0" * size)
    return path


def test_a_small_write_ahead_log_is_left_alone(tmp_path):
    """Acting on every tick would be churn, not attention."""
    _wal(tmp_path, "small", 1024)
    actor = ThoughtActor(bus=_Bus(), config_dir=tmp_path,
                         housekeeper=_Housekeeper())
    assert actor.run_once() == []


def test_a_large_one_is_folded_back_in(tmp_path):
    """This is the memory ORION loses: a .db restored without its WAL is the
    database as it stood at the last checkpoint."""
    big = _wal(tmp_path, "big", WAL_BYTES_BEFORE_CHECKPOINT + 1)
    housekeeper = _Housekeeper(on_checkpoint=lambda: big.write_bytes(b""))
    bus = _Bus()
    actor = ThoughtActor(bus=bus, config_dir=tmp_path, housekeeper=housekeeper)

    outcomes = actor.run_once()
    assert len(outcomes) == 1
    assert outcomes[0].ok and outcomes[0].name == "checkpoint-wal"
    assert housekeeper.calls == 1
    assert bus.log.lines, "he acted silently"
    assert "MB" in bus.log.lines[0]


def test_a_standing_condition_does_not_become_a_loop(tmp_path):
    """The WAL stays large if the checkpoint cannot shrink it. Without a
    cooldown that is a checkpoint every tick, forever."""
    _wal(tmp_path, "stubborn", WAL_BYTES_BEFORE_CHECKPOINT * 2)
    housekeeper = _Housekeeper()          # never actually shrinks it
    actor = ThoughtActor(bus=_Bus(), config_dir=tmp_path, housekeeper=housekeeper)

    assert len(actor.run_once(now=0.0)) == 1
    assert actor.run_once(now=1.0) == []
    assert actor.run_once(now=60.0) == []
    assert len(actor.run_once(now=10_000.0)) == 1, "it never came back"
    assert housekeeper.calls == 2


def test_a_failing_action_is_reported_not_swallowed(tmp_path):
    """Self-maintenance that quietly fails is worse than none: it looks like
    it is being handled."""
    _wal(tmp_path, "big", WAL_BYTES_BEFORE_CHECKPOINT + 1)

    class _Broken:
        def checkpoint_all(self):
            raise RuntimeError("database is locked")

    bus = _Bus()
    outcomes = ThoughtActor(bus=bus, config_dir=tmp_path,
                            housekeeper=_Broken()).run_once()
    assert len(outcomes) == 1 and outcomes[0].ok is False
    assert "could not act" in bus.log.lines[0]


def test_a_broken_condition_never_stops_him_thinking():
    """This runs inside the thought loop."""

    def _explode() -> str:
        raise RuntimeError("no")

    actor = ThoughtActor(bus=_Bus(), config_dir=Path("."))
    actor._actions = [ThoughtAction("boom", _explode, lambda: "never")]
    assert actor.run_once() == []


def test_no_action_consults_a_model():
    """The whole point. If any of this reached a router or a provider it
    would be spending tokens to decide what to do about a thought.

    Read as CODE rather than as text — the first version of this check
    matched the word "unprompted" in a docstring, which is the usual fate of
    grepping prose for intent.
    """
    import ast

    tree = ast.parse((ROOT / "orion_core" / "thought_actions.py")
                     .read_text(encoding="utf-8"))
    banned = {"router", "llm", "openai", "genai", "anthropic", "provider",
              "completion", "complete", "generate", "chat"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr.lower())
        elif isinstance(node, ast.Import):
            seen.update(a.name.split(".")[0].lower() for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0].lower())
            seen.update(a.name.lower() for a in node.names)
    offenders = sorted(seen & banned)
    assert not offenders, (
        f"{offenders} would mean spending tokens to decide what to do about "
        f"a thought, which is the one thing this may not do")


def test_it_is_wired_into_the_thought_loop():
    """An actor nothing runs is a thought nobody acts on."""
    stream = (ROOT / "orion_core" / "thought_stream.py").read_text(encoding="utf-8")
    assert "_act_without_tokens" in stream
    assert stream.index("await self._act_without_tokens()") < stream.index(
        "await self._reflection_thought()")

    app = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "thoughts.actor = ThoughtActor(" in app
