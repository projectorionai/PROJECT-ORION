"""Static beginner cybersecurity lessons with fixed-choice self-checks.

ORION only returns text from this module. It does not run the displayed code,
accept user programs, inspect a target, read logs, or modify files. The snippets
use Python's standard library and synthetic, in-memory practice data. Tests run
the bundled examples offline to keep their documented output accurate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class Exercise:
    """One reviewed example and its static multiple-choice question."""

    topic: str
    title: str
    aliases: tuple[str, ...]
    code: str
    explanation: tuple[str, ...]
    question: str
    choices: tuple[str, str, str]
    answer: str
    expected_output: str
    feedback: str
    sources: tuple[tuple[str, str], ...]


EXERCISES: tuple[Exercise, ...] = (
    Exercise(
        topic="sql_injection",
        title="Secure SQL: keep query instructions separate from data",
        aliases=("sql", "sqlite", "secure sql", "sql injection", "parameterised queries",
                 "parameterized queries", "sql injection prevention"),
        code=dedent("""\
            import sqlite3

            # :memory: keeps this synthetic database off the filesystem.
            db = sqlite3.connect(":memory:")
            try:
                db.execute("CREATE TABLE learners (username TEXT)")
                db.executemany("INSERT INTO learners VALUES (?)", [("alice",), ("bob",)])

                def find_learner(supplied_name):
                    # The ? receives a value; supplied_name never becomes SQL syntax.
                    rows = db.execute(
                        "SELECT username FROM learners WHERE username = ?",
                        (supplied_name,),
                    ).fetchall()
                    return [row[0] for row in rows]

                print("Known username:", find_learner("alice"))
                print("SQL-looking text:", find_learner("alice' OR '1'='1"))
            finally:
                db.close()
            """),
        explanation=(
            "A database executes SQL instructions. Bind untrusted values with placeholders "
            "instead of joining them into the instruction string.",
            "The first lookup matches alice. The second input looks like SQL, but the "
            "parameterised query compares that entire literal string with usernames, so it "
            "matches nothing. No vulnerable query is executed.",
            "(supplied_name,) is a one-item tuple; its comma matters. Placeholders bind values, "
            "not table names or SQL keywords. A real application still needs authentication "
            "and access controls.",
        ),
        question="What does the second lookup return, and why?",
        choices=(
            "Both learners, because the input changes the SQL instructions.",
            "An empty list, because the input is compared as a literal username.",
            "The database file path, because the input opens a file.",
        ),
        answer="B",
        expected_output="Known username: ['alice']\nSQL-looking text: []",
        feedback="The bound parameter is data. No username equals the complete second "
                 "input, so the result is an empty list. Keep using placeholders for values.",
        sources=(
            ("Python sqlite3: placeholders", "https://docs.python.org/3/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries"),
            ("OWASP SQL injection prevention", "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html"),
        ),
    ),
    Exercise(
        topic="xss",
        title="HTML escaping: display text without interpreting markup",
        aliases=("xss", "escaping", "html escaping", "cross site scripting",
                 "xss prevention", "output encoding"),
        code=dedent('''\
            from html import escape

            # Synthetic message, destined for an HTML text node.
            message = 'Hello "Ada" & <friends>'
            safe_text = escape(message, quote=True)
            print(f"<p>{safe_text}</p>")
            '''),
        explanation=(
            "HTML uses characters such as < and & as markup. escape() encodes those "
            "characters so this synthetic message is displayed as text inside a paragraph.",
            "quote=True also encodes quotation marks. The printed result is a string; "
            "ORION does not open a browser or render it.",
            "Escaping is context dependent. This example covers HTML text, not JavaScript, "
            "CSS or URL values. Use a framework's autoescaping and context-appropriate safe "
            "APIs in a real application; avoid concatenating untrusted text into executable contexts.",
        ),
        question="Which step makes this message suitable for this HTML text context?",
        choices=(
            "Call html.escape before placing the message inside the paragraph.",
            "Remove spaces from the message.",
            "Put the original message inside a script element.",
        ),
        answer="A",
        expected_output="<p>Hello &quot;Ada&quot; &amp; &lt;friends&gt;</p>",
        feedback="HTML escaping changes markup characters into text representations. "
                 "This protects the demonstrated HTML text context; other contexts need "
                 "their own safe handling.",
        sources=(
            ("Python html.escape", "https://docs.python.org/3/library/html.html#html.escape"),
            ("OWASP XSS prevention", "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html"),
        ),
    ),
    Exercise(
        topic="access_control",
        title="Object access control: authorise each requested record",
        aliases=("access control", "idor", "object level access control",
                 "object access control", "authorization", "authorisation"),
        code=dedent("""\
            # Synthetic records exist only in memory.
            reports = {
                101: {"owner": "learner_a", "text": "Learner A's practice report"},
                102: {"owner": "learner_b", "text": "Learner B's practice report"},
            }

            def read_report(authenticated_user, report_id):
                # In a real service this identity comes from trusted authentication.
                report = reports.get(report_id)
                if report is None or report["owner"] != authenticated_user:
                    return "DENIED"
                return report["text"]

            print(read_report("learner_a", 101))
            print(read_report("learner_a", 102))
            print(read_report("learner_a", 999))
            """),
        explanation=(
            "Authentication establishes identity; authorisation decides what that identity "
            "may access. Knowing a report's ID does not give permission to read it.",
            "read_report checks ownership before returning the record's content. It denies "
            "both an unknown record and a record belonging to another learner.",
            "A real server must obtain the current user from trusted authentication and "
            "apply the permission check on every request. Do not trust a user ID sent by "
            "the caller. This example is a simple ownership policy, not a complete login system.",
        ),
        question="Which check prevents learner_a from reading learner_b's report?",
        choices=(
            "Check that the report ID is a number.",
            "Check that the authenticated user is allowed to access that specific report.",
            "Hide report IDs in the user interface and skip the server-side check.",
        ),
        answer="B",
        expected_output="Learner A's practice report\nDENIED\nDENIED",
        feedback="Authorise the specific object for the trusted current identity before "
                 "returning it. Predictable or hidden identifiers do not replace this check.",
        sources=(
            ("OWASP authorisation", "https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html"),
        ),
    ),
    Exercise(
        topic="log_analysis",
        title="Defensive log analysis: identify events that need review",
        aliases=("log analysis", "detection", "login logs", "defensive log analysis",
                 "failed logins", "security logging"),
        code=dedent("""\
            from collections import Counter

            # Supplied synthetic events from one five-minute practice window.
            # No passwords, real users, log files or network connections.
            events = [
                ("learner_a", "failure"),
                ("learner_b", "failure"),
                ("learner_a", "failure"),
                ("learner_b", "success"),
                ("learner_a", "failure"),
            ]
            failures = Counter(account for account, result in events if result == "failure")
            # Three is a teaching threshold, not a universal detection rule.
            review = sorted(account for account, count in failures.items() if count >= 3)
            print("Accounts to review:", review)
            """),
        explanation=(
            "Counter groups the supplied failed-login events by account. The threshold "
            "selects learner_a for review because that account has three failures in this "
            "practice window.",
            "A single indicator is not proof of an attack. Mistyped passwords and client "
            "faults can also cause failures. Review timing, surrounding successes and other "
            "relevant evidence before drawing conclusions.",
            "Real detection needs an explicit time window, validated events and thresholds "
            "suited to the service. Collect only necessary context, protect logs and avoid "
            "logging passwords or other secrets. This example takes no automated response action.",
        ),
        question="What is the appropriate interpretation of the printed result?",
        choices=(
            "learner_a is certainly an attacker.",
            "The later success means learner_b's events can never matter.",
            "Review learner_a's events and context; the threshold alone proves no attack.",
        ),
        answer="C",
        expected_output="Accounts to review: ['learner_a']",
        feedback="The threshold is a review signal, not a verdict. Correlate events and "
                 "consider benign causes before deciding whether an incident occurred.",
        sources=(
            ("OWASP logging", "https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html"),
            ("Python Counter", "https://docs.python.org/3/library/collections.html#collections.Counter"),
        ),
    ),
)

_BY_TOPIC = {item.topic: item for item in EXERCISES}
_TEXT_ONLY_NOTE = (
    "Text-only lesson: ORION displays the code and checks A/B/C answers; it never runs "
    "your code or these snippets. All practice data is synthetic and in memory. "
    "No network connections are made and no files are modified."
)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", " ").replace("-", " "))


_ALIASES = {
    _normalise(alias): item.topic
    for item in EXERCISES
    for alias in (item.topic, *item.aliases)
}


def catalogue() -> list[dict[str, str]]:
    """Return fresh topic/title rows, without running any lesson code."""
    return [{"topic": item.topic, "title": item.title} for item in EXERCISES]


def resolve_topic(query: str) -> str | None:
    """Resolve exact reviewed aliases and simple lesson prefixes; never guess."""
    if not isinstance(query, str):
        return None
    normalised = _normalise(query)
    direct = _ALIASES.get(normalised)
    if direct is not None:
        return direct
    for prefix in (
        "show code for ", "teach me ", "code lesson on ", "code lesson for ",
        "lesson on ", "lesson for ", "exercise on ", "exercise for ",
    ):
        if normalised.startswith(prefix):
            return _ALIASES.get(normalised[len(prefix):])
    return None


def _unknown_topic() -> str:
    available = "; ".join(f"{item.topic}: {item.title}" for item in EXERCISES)
    return "Choose one of the beginner code topics: " + available + "."


def _question_text(item: Exercise) -> str:
    choices = "\n".join(f"{letter}. {text}" for letter, text in zip("ABC", item.choices))
    return f"Question: {item.question}\n{choices}\nReply with A, B or C (or 1, 2 or 3)."


def code_lesson(query: str) -> str:
    """Present the reviewed snippet, explanation, expected output and question."""
    topic = resolve_topic(query)
    if topic is None:
        return _unknown_topic()
    item = _BY_TOPIC[topic]
    explanation = "\n".join(f"{number}. {text}" for number, text in enumerate(item.explanation, 1))
    sources = "; ".join(f"[{title}]({url})" for title, url in item.sources)
    return (
        f"{item.title}\n\n{_TEXT_ONLY_NOTE}\n\n"
        f"Python example:\n```python\n{item.code.rstrip()}\n```\n\n"
        f"How it works:\n{explanation}\n\n"
        f"Expected output if you run this example in your own local practice environment:\n"
        f"```text\n{item.expected_output}\n```\n\n"
        f"{_question_text(item)}\n\nSources: {sources}"
    )


def exercise(query: str) -> str:
    """Present a fixed-choice self-check; no supplied programs are accepted."""
    topic = resolve_topic(query)
    if topic is None:
        return _unknown_topic()
    item = _BY_TOPIC[topic]
    return (
        f"Practice: {item.title}\n\n{_TEXT_ONLY_NOTE}\n\n"
        f"Read this Python example:\n```python\n{item.code.rstrip()}\n```\n\n"
        f"{_question_text(item)}"
    )


def check_answer(query: str, answer: str) -> str:
    """Compare one fixed-choice token; never parse or execute user code."""
    topic = resolve_topic(query)
    if topic is None:
        return _unknown_topic()
    item = _BY_TOPIC[topic]
    choice = answer.strip().upper() if isinstance(answer, str) else ""
    choice = {"1": "A", "2": "B", "3": "C"}.get(choice, choice)
    if choice not in ("A", "B", "C"):
        return (
            "Please send only A, B or C (or 1, 2 or 3). This self-check accepts fixed "
            "choices; ORION does not parse or execute submitted code.\n\n"
            + _question_text(item)
        )
    verdict = "Correct." if choice == item.answer else f"Not quite. The answer is {item.answer}."
    return (
        f"{verdict} {item.feedback}\n\nExpected output:\n"
        f"```text\n{item.expected_output}\n```"
    )


def learning_path() -> str:
    """Return an ordered beginner route towards authorised security practice."""
    return (
        "Beginner cybersecurity learning path\n\n"
        "1. Programming foundations: Python variables, types, loops, functions and tests; "
        "then version control and reading code before running it.\n"
        "2. Networks and operating systems: TCP/IP, DNS, HTTP, TLS, permissions and "
        "authentication. Start with diagrams and supplied examples.\n"
        "3. Scope and permission: establish written authorisation, approved targets, "
        "methods, time limits and stop conditions before any testing.\n"
        "4. Secure coding: study sql_injection, xss and access_control code lessons, "
        "then log_analysis; use their A/B/C exercises to check your reasoning.\n"
        "5. Authorised lab practice: use an isolated local lab, synthetic data and "
        "snapshots. Practise only activities covered by its agreed scope.\n"
        "6. Security reporting: record reproducible evidence, impact, remediation "
        "and verification; follow the agreed disclosure process.\n"
        "7. Defence: practise least privilege, patching, protected logging, incident "
        "triage and backup recovery. For electronics, start with low-voltage simulation, "
        "component basics and protection against electrical faults.\n\n"
        + _TEXT_ONLY_NOTE
    )
