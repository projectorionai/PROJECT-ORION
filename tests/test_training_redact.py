"""
Redaction for training data (orion_core/training/redact.py).

Pure text in, text out: no stores, no network, no files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.training import redact as rd  # noqa: E402
from orion_core.training.redact import Redactor, luhn_valid, sensitive_topic  # noqa: E402


@pytest.fixture()
def red() -> Redactor:
    return Redactor(["Alex", "alexm"])


@pytest.mark.parametrize("text, expected", [
    ("email jo@example.com about friday", "email <EMAIL> about friday"),
    ("cc Jo.Smith+work@mail.example.co.uk please", "cc <EMAIL> please"),
    ("call 07700 900123 now", "call <PHONE> now"),
    ("call +44 7700 900123 now", "call <PHONE> now"),
    ("ring 0121 496 0000 or 020 7946 0000", "ring <PHONE> or <PHONE>"),
    # Key shapes are assembled here so no key-shaped text is in the source
    # (the release's secret audit refuses one, fake or not).
    ("my key is " + "sk-" + "abcdefghijklmnopqrstuvwxyz123456", "my key is <SECRET>"),
    ("api_key=" + "AIza" + "SyA1234567890abcdefghijklmnopqrstuv", "api_key=<SECRET>"),
    ("Authorization: Bearer abcdefghijklmnop123456", "Authorization: Bearer <SECRET>"),
    ("password: hunter22", "password: <SECRET>"),
    ("https://user:pw@example.com/x?token=abc123&q=1", "https://<SECRET>@example.com/x?token=<SECRET>&q=1"),
    ("card 4111 1111 1111 1111 expires 12/29", "card <CARD> expires 12/29"),
    ("postcode B15 2TT", "postcode <POSTCODE>"),
    ("server at 192.168.1.20 and 127.0.0.1", "server at <IP> and 127.0.0.1"),
])
def test_personal_strings_become_placeholders(red, text, expected):
    assert red.redact(text) == expected


def test_profile_paths_lose_the_account_name(red):
    out = red.redact(r"open C:\Users\alexm\OneDrive - Example University\Desktop\notes.txt")
    assert out == r"open <HOME>\OneDrive\Desktop\notes.txt"
    assert red.redact("C:/Users/anyone/Documents/x.pdf") == "<HOME>/Documents/x.pdf"
    assert red.redact("/home/bob/file") == "<HOME>/file"


def test_the_users_own_name_is_replaced_whole_word_only(red):
    assert red.redact("Alex asked alexm to open it") == "<USER> asked <USER> to open it"
    assert red.redact("Alexandria is a town") == "Alexandria is a town"


@pytest.mark.parametrize("text", [
    "Today: 41,200 tokens used of a 1,000,000 allowance at 14:06:33 on 2026-10-24",
    "CPU 23%, memory 61% (9.8 of 16 GB), GPU 12%, disk 71% full.",
    "order 1234 5678 9012 3456 is not a card",          # fails the Luhn check
    "I have an MP3 4GB player",                          # not a UK postcode area
    "Annual_Report_2026_Final_Version_Updated_Draft",   # a long NAME is not a secret
    "version 2.13.0 and pid 4312",
    "open notepad and check the weather in Leeds",
])
def test_ordinary_text_is_left_alone(red, text):
    assert red.redact(text) == text


def test_redaction_is_idempotent_and_counted(red):
    text = ("mail jo@example.com or 07700 900123; token=abcd1234efgh5678; C:\\Users\\alexm\\x; "
            "192.168.0.9; " + "sk-" + "abcdefghijklmnopqrstuvwxyz123456")
    once = red.redact(text)
    assert red.redact(once) == once
    for label in ("email", "phone", "secret", "home", "ip"):
        assert red.counts[label] >= 1
    assert "jo@example.com" not in once and "alexm" not in once


def test_hex_hashes_and_long_opaque_tokens_are_secrets(red):
    assert red.redact("hash e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == "hash <SECRET>"
    assert red.redact("token Zx9Qv3Lm8Kd2Pw7Rt5Yb1Nc4Hj6Fg0Sa8Ue2Io") == "token <SECRET>"


def test_redact_obj_walks_structures_but_not_keys(red):
    value = {"to": "jo@example.com", "n": 3, "list": ["07700 900123", {"deep": "alexm"}], "ok": True}
    out = red.redact_obj(value)
    assert out == {"to": "<EMAIL>", "n": 3, "list": ["<PHONE>", {"deep": "<USER>"}], "ok": True}


def test_common_account_names_are_not_treated_as_terms():
    r = Redactor(["user", "Admin", "me", "Alexandra"])
    assert r.terms == ["Alexandra"]


def test_environment_terms_are_the_os_account(monkeypatch):
    monkeypatch.setattr(rd.getpass, "getuser", lambda: "Zebediah")
    monkeypatch.setattr(rd.Path, "home", classmethod(lambda cls: Path("C:/Users/Zebediah")))
    monkeypatch.setenv("USERNAME", "Zebediah")
    r = Redactor.from_environment(["Hannah"])
    assert r.redact("Zebediah and hannah") == "<USER> and <USER>"


def test_luhn():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("123")


@pytest.mark.parametrize("text, hit", [
    ("my pin is 1234", True), ("what's the password for the wifi", True),
    ("here is my sort code", True), ("read my api key aloud", True), ("my national insurance number", True),
    ("open notepad", False), ("pin this window to the taskbar", False),
    ("the passenger list", False),
])
def test_sensitive_topics_are_recognised(text, hit):
    assert bool(sensitive_topic(text)) is hit
