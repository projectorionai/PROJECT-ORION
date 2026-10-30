"""Document review and stateful quizzes, without placeholder UI claims."""

from __future__ import annotations

import asyncio
import unicodedata
from typing import Any

from ..data import ToolResult
from . import runtime


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    return " ".join("".join(c if c.isalnum() or c.isspace() else " "
                            for c in text if not unicodedata.combining(c)).split())


def _clean_questions(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
        raise ValueError("Supply between one and twenty questions")
    cleaned = []
    for row in raw:
        if not isinstance(row, dict):
            raise TypeError("Each question must be an object")
        question = str(row.get("question") or "").strip()[:2000]
        answer = str(row.get("answer") or "").strip()[:2000]
        kind = str(row.get("type") or "open")
        if not question or not answer or kind not in {"open", "multiple_choice", "true_false", "fill_blank"}:
            raise ValueError("Every question needs a supported type, question and answer")
        options = list(dict.fromkeys(str(s).strip()[:500] for s in row.get("options", []) if str(s).strip()))
        if kind == "true_false" and not options:
            options = ["True", "False"]
        if kind in {"multiple_choice", "true_false"}:
            if not 2 <= len(options) <= 6:
                raise ValueError("Choice questions need two to six distinct options")
            token = _norm(answer)
            index = ord(token) - ord("a") if len(token) == 1 and "a" <= token <= "f" else -1
            if token.isdigit():
                index = int(token) - 1
            if 0 <= index < len(options):
                answer = options[index]
            if _norm(answer) not in {_norm(o) for o in options}:
                raise ValueError("A choice answer must identify one of its options")
        cleaned.append({"type": kind, "question": question, "answer": answer,
                        "options": options, "explanation": str(row.get("explanation") or "")[:2000]})
    return cleaned


def _shown(data: dict) -> dict:
    rows = data.get("questions", [])
    index = data.get("index", 0)
    active = rows[index] if index < len(rows) else None
    return {"topic": data.get("topic", "Quiz"), "index": index, "total": len(rows),
            "question": ({k: v for k, v in active.items() if k != "answer" and k != "explanation"}
                         if active else None), "answers": data.get("answers", [])}


async def quiz(args: dict) -> ToolResult:
    action = str(args.get("action") or "status").lower()
    if action == "start":
        rows = _clean_questions(args.get("questions"))
        data = {"topic": str(args.get("topic") or "Quiz")[:200], "index": 0,
                "questions": rows, "answers": []}
    else:
        data = await asyncio.to_thread(runtime.load, "quiz")
    if action == "stop":
        data = {}
    elif action == "answer":
        index = data.get("index", 0)
        rows = data.get("questions", [])
        if index >= len(rows):
            return ToolResult("There is no unanswered quiz question.", ok=False)
        given = str(args.get("answer") or "").strip()[:2000]
        if not given:
            return ToolResult("Give your answer to the current question.", ok=False)
        row = rows[index]
        if row["type"] == "multiple_choice":
            token = _norm(given)
            choice = int(token) - 1 if token.isdigit() else ord(token) - ord("a") \
                if len(token) == 1 and "a" <= token <= "f" else -1
            if 0 <= choice < len(row["options"]):
                given = row["options"][choice]
        correct: bool | None = _norm(given) == _norm(row["answer"])
        if row["type"] in {"open", "fill_blank"} and not correct:
            correct = None  # valid alternative wording needs a real judgement
        data.setdefault("answers", []).append({"given": given, "correct": correct,
            "reference": row["answer"], "explanation": row["explanation"]})
        data["index"] = index + 1
    elif action == "grade":
        index = int(args.get("index", 0)) - 1
        if not 0 <= index < len(data.get("answers", [])):
            return ToolResult("Choose an answered question by its one-based index.", ok=False)
        if not isinstance(args.get("correct"), bool):
            return ToolResult("A manual grade requires correct=true or correct=false.", ok=False)
        data["answers"][index]["correct"] = args["correct"]
    elif action not in {"start", "status", "results", "stop"}:
        return ToolResult("Quiz actions: start, answer, grade, status, results or stop.", ok=False)
    await asyncio.to_thread(runtime.save, "quiz", data)
    shown = _shown(data)
    runtime.emit("jv_quiz", shown)
    if shown["question"]:
        row = shown["question"]
        options = "\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(row["options"])) if row["options"] else ""
        return ToolResult(f"Question {shown['index']+1}/{shown['total']}: {row['question']}{options}")
    answers = shown["answers"]
    correct = sum(a["correct"] is True for a in answers)
    pending = sum(a["correct"] is None for a in answers)
    return ToolResult("Quiz stopped." if action == "stop" else
                      f"Quiz complete: {correct}/{len(answers)} correct; {pending} awaiting judgement.",
                      evidence=answers)


async def document_review(args: dict) -> ToolResult:
    title = str(args.get("title") or "Document review").strip()[:200]
    text = str(args.get("text") or "").strip()
    if text:
        if len(text) > 60000:
            return ToolResult("Review a section of up to 60,000 characters at a time.", ok=False)
        result = await runtime.text_json(
            "Review the following supplied document. Identify errors, unclear wording and actionable improvements. "
            "Do not invent facts or claim to have checked references you cannot access. Return "
            '{"summary":"...","findings":[{"severity":"high|medium|low","location":"...",'
            '"issue":"...","suggestion":"..."}],"unclear":["..."]}.\nDOCUMENT:\n' + text,
            task="document_review")
    else:
        result = args
        if not args.get("findings") and not str(args.get("summary") or "").strip():
            return ToolResult("Supply the document text or existing review findings.", ok=False)
    findings = result.get("findings", [])
    if not isinstance(findings, list):
        raise TypeError("Findings must be an array")
    lines = [title, str(result.get("summary") or "Review findings supplied by the caller.")[:12000]]
    clean = []
    for row in findings[:60]:
        if not isinstance(row, dict):
            continue
        item = {key: str(row.get(key) or "")[:2000] for key in ("severity", "location", "issue", "suggestion")}
        clean.append(item)
        lines.append(f"• {item['severity'] or 'Review'} · {item['location'] or 'Document'}: {item['issue']}\n  {item['suggestion']}")
    unclear = result.get("unclear", [])
    if isinstance(unclear, list):
        lines.extend("Needs clarification: " + str(x)[:2000] for x in unclear[:20])
    runtime.emit("jv_document_review", {"title": title, "findings": clean, "summary": lines[1]})
    return ToolResult("\n".join(lines), evidence=clean)
