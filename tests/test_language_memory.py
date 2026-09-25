"""
Learning which language the user speaks, without ever asking.

Why it is needed at all
-----------------------
A live model replies in whatever language it was just spoken to, so the
conversation adapts by itself. What does not adapt is everything ORION opens —
the morning briefing, a proactive check-in, an alert, the greeting after a
restart. Those have no user turn to take a cue from, so without this they come
out in the language the prompt happens to be written in, whoever is listening.

Two properties carry the whole design, and both are tested here:

  * it refuses to guess. UNKNOWN costs nothing, because the caller simply keeps
    what it had. A confident wrong answer has ORION greeting an English speaker
    in Portuguese every morning, which is worse than never adapting at all.

  * one sentence never decides. People quote, swear, paste and borrow, and a
    single "merci" must not move an entire assistant into French.

Dependency-free by choice: adding a language-detection package to tell apart a
handful of European languages would be a poor trade. Scripts are decided by
character range; Latin languages by the small closed class of function words
that appear in nearly every sentence and that nobody borrows.

Offline: pure text and a temp file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.language_memory import (
    SWITCH_AFTER,
    LanguageMemory,
    detect_language,
    detect_script,
    language_name,
)

ENGLISH = "I was hoping you could have a look at the report for me"
FRENCH = "Je ne trouve pas le fichier et ce n est pas normal du tout"
GERMAN = "Ich habe die Datei nicht gefunden und das ist ein Problem"


@pytest.fixture
def memory(tmp_path):
    return LanguageMemory(tmp_path / "language.json")


# ── detection ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,text", [
    ("en", ENGLISH),
    ("de", GERMAN),
    ("fr", FRENCH),
    ("es", "No encuentro el archivo y esto no es lo que yo esperaba"),
    ("it", "Non trovo il file e questo non e quello che mi aspettavo"),
    ("tr", "Bu dosyayi bulamadim ve bu benim icin cok buyuk bir sorun"),
    ("nl", "Ik kan het bestand niet vinden en dat is een probleem voor mij"),
])
def test_latin_languages_are_told_apart_by_function_words(code, text):
    """Content words are useless here — 'computer' and 'internet' are the same
    in a dozen languages. The closed class is what identifies one."""
    assert detect_language(text) == code


@pytest.mark.parametrize("code,text", [
    ("ru", "Я не могу найти этот файл и это большая проблема"),
    ("el", "Δεν μπορώ να βρω αυτό το αρχείο και αυτό είναι πρόβλημα"),
    ("ja", "ファイルが見つかりません。これは大きな問題です。"),
    ("zh", "我找不到这个文件这是一个很大的问题"),
    ("ar", "لا أستطيع العثور على هذا الملف وهذه مشكلة كبيرة"),
    ("he", "אני לא מוצא את הקובץ הזה וזו בעיה גדולה"),
    ("ko", "파일을 찾을 수 없습니다 이것은 큰 문제입니다"),
    ("th", "ฉันหาไฟล์นี้ไม่เจอและนี่เป็นปัญหาใหญ่"),
])
def test_non_latin_scripts_are_decided_on_sight(code, text):
    """No word list needed: the writing system settles it."""
    assert detect_language(text) == code
    assert detect_script(text) == code


@pytest.mark.parametrize("text", ["", "   ", "yes", "ok thanks", "merci",
                                  "12345 !!!", "hmm", "aaaa bbbb cccc dddd"])
def test_it_refuses_to_guess_from_too_little(text):
    """UNKNOWN is free; a wrong answer is not."""
    assert detect_language(text) == ""


def test_a_quoted_foreign_phrase_does_not_flip_the_script_test():
    """Two Cyrillic words inside an English sentence is a quotation."""
    text = "He kept saying спасибо which I think means thank you in Russian"
    assert detect_script(text) == ""


def test_language_names_are_human_readable():
    assert language_name("fr") == "French"
    assert language_name("") == ""


# ── stability ────────────────────────────────────────────────────────────────

def test_a_single_foreign_sentence_does_not_switch_him(memory):
    """THE property. People quote, swear, paste and use loanwords."""
    for _ in range(SWITCH_AFTER):
        memory.observe(ENGLISH)
    assert memory.name == "English"
    assert memory.observe(FRENCH) is False
    assert memory.name == "English", "one French line moved the whole assistant"


def test_a_genuine_switch_is_picked_up(memory):
    for _ in range(SWITCH_AFTER):
        memory.observe(ENGLISH)
    changed = [memory.observe(FRENCH) for _ in range(SWITCH_AFTER)]
    assert any(changed), "consistent French was never adopted"
    assert memory.name == "French"


def test_an_interrupted_run_does_not_count(memory):
    """Alternating languages is not evidence for either."""
    for _ in range(SWITCH_AFTER):
        memory.observe(ENGLISH)
    memory.observe(FRENCH)
    memory.observe(ENGLISH)
    memory.observe(FRENCH)
    assert memory.name == "English"


def test_unreadable_utterances_are_simply_skipped(memory):
    for _ in range(SWITCH_AFTER):
        memory.observe(ENGLISH)
    for _ in range(10):
        memory.observe("ok")
    assert memory.name == "English"


# ── persistence ──────────────────────────────────────────────────────────────

def test_the_language_survives_a_restart(tmp_path):
    path = tmp_path / "language.json"
    first = LanguageMemory(path)
    for _ in range(SWITCH_AFTER):
        first.observe(GERMAN)
    assert first.name == "German"
    assert LanguageMemory(path).name == "German", "relearned from scratch"


def test_a_corrupt_file_is_survivable(tmp_path):
    path = tmp_path / "language.json"
    path.write_text("{ not json", encoding="utf-8")
    memory = LanguageMemory(path)
    assert memory.known is False
    assert memory.prompt_line() == ""


def test_an_unknown_code_on_disk_is_ignored(tmp_path):
    path = tmp_path / "language.json"
    path.write_text(json.dumps({"language": "klingon"}), encoding="utf-8")
    assert LanguageMemory(path).known is False


def test_forget_clears_it(memory):
    for _ in range(SWITCH_AFTER):
        memory.observe(FRENCH)
    memory.forget()
    assert memory.known is False


# ── what the prompt carries ──────────────────────────────────────────────────

def test_nothing_is_claimed_before_anything_is_known(memory):
    assert memory.prompt_line() == ""


def test_the_prompt_line_targets_what_orion_opens(memory):
    """The conversation already adapts on its own; the briefing does not."""
    for _ in range(SWITCH_AFTER):
        memory.observe(FRENCH)
    line = memory.prompt_line()
    assert "French" in line
    assert "briefing" in line
    assert "greeting" in line


def test_the_user_may_still_switch_mid_conversation(memory):
    """Remembering a language must not override what they just typed."""
    for _ in range(SWITCH_AFTER):
        memory.observe(FRENCH)
    line = memory.prompt_line().lower()
    assert "answer in that one" in line
    assert "do not comment on the change" in line


def test_the_capability_block_stays_silent_until_it_knows(tmp_path, monkeypatch):
    """Isolated from the machine's own store, deliberately.

    This read the real config/language.json and passed only until ORION had
    actually learned a language from being used — at which point it began
    failing on the developer's machine and nowhere else. A test that depends
    on accumulated local state is a test that reports the environment, not
    the code.
    """
    import orion_core.language_memory as language_memory
    from orion_core.self_knowledge import capability_block

    monkeypatch.setattr(language_memory, "LANGUAGE_PATH",
                        tmp_path / "language.json")
    assert "LANGUAGE:" not in capability_block()


def test_the_capability_block_says_the_language_once_it_knows(tmp_path, monkeypatch):
    """The other half, which the original could not express: it appears when
    there IS something to say."""
    import orion_core.language_memory as language_memory
    from orion_core.self_knowledge import capability_block

    path = tmp_path / "language.json"
    monkeypatch.setattr(language_memory, "LANGUAGE_PATH", path)
    memory = language_memory.LanguageMemory(path)
    for _ in range(language_memory.SWITCH_AFTER + 1):
        memory.observe("this is a sentence written in plain English")
    assert "LANGUAGE:" in capability_block()


def test_the_live_worker_observes_user_speech():
    """A detector nothing feeds is the same as no detector."""
    source = (Path(__file__).resolve().parents[1]
              / "orion_core" / "live_worker.py").read_text(encoding="utf-8")
    assert "_language.observe(user_text)" in source, (
        "nothing ever hands the language memory a transcript"
    )
