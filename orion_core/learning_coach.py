"""Small, learner-controlled lessons with durable position, not health records.

This service returns reviewed text only. It never executes an answer, sets a
timer, marks curriculum modules complete, or infers a medical diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from .atomic_io import atomic_write_text
from .constants import CONFIG_DIR
from .data import ToolResult
from .electronics_teaching import construction_boundary
from .learning_steps import LearningPlan, LearningStep, resolve_plan

STATE_PATH = CONFIG_DIR / "learning_coach.json"
_LOCK = RLock()
_MAX_STATE_BYTES = 128 * 1024
_MAX_LEARNERS = 64
_CONTROLS = 'Say "next", "simpler", "repeat", or "pause my lesson".'


class _StateError(ValueError):
    pass


class LearningCoach:
    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = Path(state_path) if state_path is not None else STATE_PATH

    @staticmethod
    def _setting(value: Any, minimum: int, maximum: int, name: str) -> int:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be a whole number from {minimum} to {maximum}.")
        return value

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"version": 1, "learners": {}}
        try:
            if self.state_path.stat().st_size > _MAX_STATE_BYTES:
                raise _StateError("Saved lesson state is too large.")
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or type(data.get("version")) is not int
                    or data["version"] != 1 or not isinstance(data.get("learners"), dict)
                    or len(data["learners"]) > _MAX_LEARNERS):
                raise _StateError("Saved lesson state has an unsupported format.")
            for user, record in data["learners"].items():
                if not isinstance(user, str) or not 1 <= len(user) <= 64:
                    raise _StateError("Saved learner identifier is invalid.")
                if not isinstance(record, dict):
                    raise _StateError("Saved lesson entry is invalid.")
                plan = resolve_plan(record.get("topic", ""))
                if plan is None or plan.topic != record.get("topic"):
                    raise _StateError("Saved lesson topic is unavailable.")
                if (type(record.get("index")) is not int
                        or not 0 <= record["index"] < len(plan.steps)
                        or type(record.get("paused")) is not bool
                        or type(record.get("complete")) is not bool):
                    raise _StateError("Saved lesson position is invalid.")
                if record["complete"] and record["index"] != len(plan.steps) - 1:
                    raise _StateError("Saved lesson completion is invalid.")
                if type(record.get("started", True)) is not bool:
                    raise _StateError("Saved lesson start state is invalid.")
                if not record.get("started", True) and (
                        record["index"] != 0 or record["paused"] or record["complete"]
                        or record.get("segment_steps") != 0):
                    raise _StateError("Saved lesson preparation is invalid.")
                self._setting(record.get("pace_minutes"), 0, 10, "Pace minutes")
                self._setting(record.get("steps_per_session"), 1, 5, "Steps per session")
                self._setting(record.get("segment_steps"), 0, 5, "Steps in this session")
                if record["segment_steps"] > record["steps_per_session"]:
                    raise _StateError("Saved session length is invalid.")
            return data
        except (ValueError, TypeError, UnicodeError) as exc:
            raise _StateError("I couldn't read the saved lesson position. The original file "
                              "has been left intact; use a full lesson while it is repaired.") from exc

    def _save(self, data: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.state_path, json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _plan(record: dict[str, Any]) -> LearningPlan:
        plan = resolve_plan(record["topic"])
        assert plan is not None  # validated by _load or start
        return plan

    @staticmethod
    def _question(step: LearningStep) -> str:
        text = step.question
        if step.choices:
            text += "\n" + "\n".join(f"{letter}. {choice}"
                                       for letter, choice in zip("ABC", step.choices))
            text += "\nReply with A, B or C, or say next to skip this check."
        return text

    def _render(self, record: dict[str, Any], *, simpler: bool = False,
                recap: bool = False) -> str:
        plan = self._plan(record)
        step = plan.steps[record["index"]]
        if not record.get("started", True):
            return f"Ready to begin {plan.title}. Say \"teach me in small steps\" when ready."
        if record["complete"]:
            return (f"{plan.title}: all {len(plan.steps)} steps visited.\n"
                    f"Last point: {step.recap}\nYou can stop here or restart to review. "
                    "This is lesson progress, not a qualification or a mastery score.")
        if record["paused"]:
            return (f"Lesson paused at step {record['index'] + 1}/{len(plan.steps)}.\n"
                    f"Last point: {step.recap}\nSay \"resume my lesson\" whenever you want.")
        content = step.simpler if simpler else step.content
        lines = [f"{plan.title} — step {record['index'] + 1}/{len(plan.steps)}",
                 f"Goal: {step.goal}"]
        if recap:
            lines.append(f"Where we were: {step.recap}")
        lines += ["", content, "", self._question(step), "", _CONTROLS]
        return "\n".join(lines)

    @staticmethod
    def _pace(record: dict[str, Any]) -> str:
        if record["pace_minutes"]:
            return (f"Pace suggestion: about {record['pace_minutes']} minutes per step; "
                    "take as long as you need. There is no timer.")
        return "No time target: one small step at your own pace."

    def handle(self, action: str, *, user: str = "default", topic: str = "",
               answer: str = "", pace_minutes: int | None = None,
               steps_per_session: int | None = None, restart: bool = False) -> ToolResult:
        """Advance only on an explicit next; save before reporting a change."""
        if not isinstance(user, str) or not user.strip() or len(user) > 64:
            return ToolResult("Choose a learner id of 1 to 64 characters.", ok=False)
        if not isinstance(topic, str) or len(topic) > 500:
            return ToolResult("Use a short lesson topic, such as Python basics.", ok=False)
        if not isinstance(answer, str) or len(answer) > 2000:
            return ToolResult("Please send a short answer, or say next to skip this check.", ok=False)
        if type(restart) is not bool:
            return ToolResult("Restart must be true or false.", ok=False)
        boundary = construction_boundary(topic)
        if boundary:
            return ToolResult(boundary, ok=False)
        user = user.strip()
        action = str(action or "start").lower().strip().removeprefix("coach_")
        if action not in {"start", "next", "answer", "simpler", "repeat", "pause",
                          "resume", "status", "settings"}:
            return ToolResult("Lesson controls: start, next, answer, simpler, repeat, "
                              "pause, resume, status, settings.", ok=False)
        try:
            if pace_minutes is not None:
                self._setting(pace_minutes, 0, 10, "Pace minutes")
            if steps_per_session is not None:
                self._setting(steps_per_session, 1, 5, "Steps per session")
            with _LOCK:
                data = self._load()
                record = data["learners"].get(user)
                changed = False
                if action == "start" or (action == "settings" and record is None):
                    plan = resolve_plan(topic or (record["topic"] if record else "python_basics"))
                    if plan is None:
                        return ToolResult("Choose a short lesson: Python basics, secure SQL, "
                                          "hardware protection, or ask for the lesson catalogue.", ok=False)
                    if record is None or restart or record["topic"] != plan.topic:
                        if record is None and len(data["learners"]) >= _MAX_LEARNERS:
                            return ToolResult("The saved learner list is full. Use an existing learner id.",
                                              ok=False)
                        record = {
                            "topic": plan.topic, "index": 0, "paused": False, "complete": False,
                            "started": action == "start",
                            "segment_steps": 0 if action == "settings" else 1,
                            "pace_minutes": record["pace_minutes"] if record else 5,
                            "steps_per_session": record["steps_per_session"] if record else 3,
                        }
                        data["learners"][user] = record
                        changed = True
                    elif action == "start" and not record.get("started", True):
                        record["started"] = True
                        record["segment_steps"] = 1
                        changed = True
                elif record is None:
                    return ToolResult("No saved lesson yet. Say \"teach me in small steps\" "
                                      "and a topic, or start with Python basics.", ok=False)

                assert record is not None
                if not record.get("started", True) and action not in {"start", "settings", "status"}:
                    return ToolResult('Your preferences are saved. Say "teach me in small steps" '
                                      "to begin the first step.", ok=False)
                if pace_minutes is not None:
                    record["pace_minutes"] = pace_minutes
                    changed = True
                if steps_per_session is not None:
                    record["steps_per_session"] = steps_per_session
                    record["segment_steps"] = min(record["segment_steps"], steps_per_session)
                    changed = True
                plan = self._plan(record)
                step = plan.steps[record["index"]]

                if action == "next" and not record["paused"] and not record["complete"]:
                    if record["index"] + 1 == len(plan.steps):
                        record["complete"] = True
                    elif record["segment_steps"] >= record["steps_per_session"]:
                        record["paused"] = True
                    else:
                        record["index"] += 1
                        record["segment_steps"] += 1
                    changed = True
                elif action == "pause":
                    record["paused"] = True
                    changed = True
                elif action == "resume" and record["paused"]:
                    record["paused"] = False
                    record["segment_steps"] = 0
                    changed = True

                if changed:
                    self._save(data)

                if action == "settings":
                    return ToolResult(self._pace(record)
                                      + f"\nA pause point after {record['steps_per_session']} new steps. "
                                      "You control when to continue.")
                if action == "status":
                    state = ("ready to begin" if not record.get("started", True) else
                             "complete" if record["complete"] else "paused" if record["paused"] else "active")
                    return ToolResult(f"{plan.title}: {state}, step {record['index'] + 1}/{len(plan.steps)}.\n"
                                      f"Last point: {step.recap}\n" + self._pace(record))
                if action == "answer" and not record["paused"] and not record["complete"]:
                    if not answer.strip():
                        return ToolResult(self._question(step), ok=False)
                    if step.answer:
                        choice = answer.strip().upper()
                        choice = {"1": "A", "2": "B", "3": "C"}.get(choice, choice)
                        if choice not in {"A", "B", "C"}:
                            return ToolResult("Use A, B or C for this check; your answer is never run as code.\n"
                                              + self._question(step), ok=False)
                        if choice != step.answer:
                            return ToolResult("Let's try a smaller explanation.\n" + step.simpler
                                              + "\n\n" + self._question(step))
                        return ToolResult(f"That's it. {step.feedback or step.recap}\n"
                                          'Say "next" when ready, or pause here.')
                    return ToolResult(f"One way to think about it: {step.feedback or step.recap}\n"
                                      'That reflection is not graded. Say "next" when ready.')

                text = self._render(record, simpler=action == "simpler", recap=action == "resume")
                if action == "next" and record["paused"]:
                    text = "A good pause point after a few small steps.\n" + text
                if action == "start" and not record["paused"] and not record["complete"]:
                    text = self._pace(record) + "\n\n" + text
                return ToolResult(text)
        except _StateError as exc:
            return ToolResult(str(exc), ok=False)
        except OSError:
            return ToolResult("I couldn't read or save your lesson position. Your previous checkpoint "
                              "has been kept; no progress change was recorded. Try again when ready.", ok=False)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)


__all__ = ["LearningCoach", "STATE_PATH"]
