"""
"I've restarted": a restart remembered across the process that ends.

The owner, 26 Oct 2026: when ORION restarts he should remember, and say he
has restarted, instead of going through the normal greeting. The ending
process notes the restart; the new one takes the note once and says it is
back - and how long that took - in place of the greeting and the briefing
offer. A launch, or a note from a restart that never came back, greets as
before.

Fakes only: notes in tmp_path, a fake worker. Nothing starts ORION.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import restart_note  # noqa: E402
from orion_core.live_worker import GenAILiveWorker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOW = 1_800_000_000.0


# ── the note ────────────────────────────────────────────────────────────────

def test_a_restart_is_taken_once_by_the_next_process(tmp_path):
    note = tmp_path / "restart_note.json"
    assert restart_note.remember(now=NOW, path=note)
    restart = restart_note.take(now=NOW + 9.2, path=note)
    assert restart is not None and restart.took_phrase(NOW + 9.2) == "9 seconds"
    assert not note.exists()
    assert restart_note.take(now=NOW + 10, path=note) is None, "honoured twice"


def test_a_launch_with_no_note_is_a_launch(tmp_path):
    assert restart_note.take(now=NOW, path=tmp_path / "restart_note.json") is None


def test_a_note_from_a_restart_that_never_came_back_is_ignored_and_removed(tmp_path):
    note = tmp_path / "restart_note.json"
    restart_note.remember(now=NOW, path=note)
    assert restart_note.take(now=NOW + restart_note.MAX_AGE_S + 1, path=note) is None
    assert not note.exists()


def test_a_note_from_the_future_or_garbage_is_ignored(tmp_path):
    note = tmp_path / "restart_note.json"
    restart_note.remember(now=NOW + 3600, path=note)
    assert restart_note.take(now=NOW, path=note) is None
    note.write_text("{not json", encoding="utf-8")
    assert restart_note.take(now=NOW, path=note) is None and not note.exists()
    note.write_text(json.dumps({"at": "yesterday"}), encoding="utf-8")
    assert restart_note.take(now=NOW, path=note) is None


def test_how_long_it_took_is_said_only_when_it_is_sensible():
    restart = restart_note.Restart(NOW)
    assert restart.took_phrase(NOW + 1.2) == "1 second"
    assert restart.took_phrase(NOW + 14.6) == "15 seconds"
    assert restart.took_phrase(NOW + 0.2) == ""
    assert restart.took_phrase(NOW + 600) == ""


def test_the_offline_line_says_he_is_back_with_the_real_figure():
    first = lambda lines: lines[0]                      # noqa: E731
    timed = restart_note.spoken_line(restart_note.Restart(NOW), now=NOW + 9, choose=first)
    assert timed == "I'm back - that restart took 9 seconds."
    untimed = restart_note.spoken_line(restart_note.Restart(NOW), now=NOW + 900, choose=first)
    assert untimed == "I'm back - restart complete."


def test_the_live_instruction_replaces_the_greeting_and_the_briefing_offer():
    text = restart_note.instruction(restart_note.Restart(NOW), now=NOW + 12)
    assert "you've restarted and you're back" in text
    assert "12 seconds" in text and "exactly that figure" in text
    assert "Do NOT give the usual time-of-day greeting" in text
    assert "do NOT offer or mention the briefing" in text


# ── the startup turn ────────────────────────────────────────────────────────

class _Bus:
    def __init__(self):
        self.lines = []
        self.log = SimpleNamespace(emit=self.lines.append)


def _worker(restart, *, session):
    sent, said = [], []

    async def send(instruction):
        sent.append(instruction)

    async def no_greeting():
        raise AssertionError("a restart must not compose the launch greeting")

    worker = SimpleNamespace(
        connected=True, stop_event=SimpleNamespace(is_set=lambda: False),
        bus=_Bus(), session=session, awaiting_briefing=False, restart_note=restart,
        _refresh_wake_window=lambda: None, _compose_greeting=no_greeting,
        _emit_state=lambda state: None, _send_text_turn=send, _say=said.append,
        _greeted_at=0.0)
    worker._announce_restart = lambda r: GenAILiveWorker._announce_restart(worker, r)
    return worker, sent, said


def test_after_a_restart_the_live_model_says_he_is_back_instead_of_greeting():
    import time
    worker, sent, said = _worker(restart_note.Restart(time.time() - 8), session=object())
    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    assert len(sent) == 1 and "you've restarted and you're back" in sent[0]
    assert said == []
    assert worker.awaiting_briefing is False, "the briefing offer was made anyway"
    assert worker.restart_note is None, "the note would be announced again"
    assert any(line.startswith("SYS: back from a restart") for line in worker.bus.lines)


def test_after_a_restart_offline_he_says_it_in_his_own_voice():
    import time
    worker, sent, said = _worker(restart_note.Restart(time.time() - 8), session=None)
    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    assert sent == [] and len(said) == 1
    assert said[0].startswith(("I'm back", "Restart complete", "Back", "And we're back",
                               "Restarted"))
    assert worker.awaiting_briefing is False


# ── the wiring ──────────────────────────────────────────────────────────────

def test_the_restart_is_noted_and_the_next_start_reads_it_before_greeting():
    app = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    restart = app[app.index("    def request_restart() -> None:"):]
    restart = restart[:restart.index("app.aboutToQuit.connect")]
    assert "restart_note.remember()" in restart
    assert restart.index("restart_note.remember()") < restart.index("request_shutdown()")
    assert app.index("await asyncio.to_thread(_restart_note.take_return)") < app.index(
        "worker.offer_startup_briefing()")
    # Every clean exit leaves its note (Mark 39), restart or shut-down.
    assert "_exit_note.remember_exit(" in app


# ── Mark XXXIX: every return, and his own words ──────────────────────────────
# Owner, 27 Oct 2026: "when ORION restarts he must remember that he's just
# restarted and not come out with the same greetings".

def _notes(tmp_path):
    return tmp_path / "restart_note.json", tmp_path / "last_exit.json"


def test_a_quick_relaunch_after_a_shutdown_is_a_return_not_a_new_day(tmp_path):
    note, last_exit = _notes(tmp_path)
    restart_note.remember_exit("shutdown", now=NOW, path=last_exit, mark="Mark XXXIX")
    back = restart_note.take_return(now=NOW + 240, path=note, exit_path=last_exit,
                                    current_mark="Mark XXXIX")
    assert back is not None and back.kind == "relaunch" and not back.updated
    assert back.gap_phrase(NOW + 240) == "4 minutes" and not last_exit.exists()
    text = restart_note.instruction(back, now=NOW + 240, recent=["Back so soon? I'm here."])
    assert "4 minutes" in text and "do NOT mention the weather" in text
    assert "Back so soon? I'm here." in text and "say it differently" in text


def test_a_start_long_after_a_shutdown_is_an_ordinary_launch(tmp_path):
    note, last_exit = _notes(tmp_path)
    restart_note.remember_exit("shutdown", now=NOW, path=last_exit, mark="Mark XXXIX")
    gap = restart_note.QUICK_RELAUNCH_S + 60
    assert restart_note.take_return(now=NOW + gap, path=note, exit_path=last_exit,
                                    current_mark="Mark XXXIX") is None
    assert not last_exit.exists()


def test_the_first_start_on_a_new_mark_greets_and_names_it(tmp_path):
    note, last_exit = _notes(tmp_path)
    restart_note.remember_exit("shutdown", now=NOW, path=last_exit, mark="Mark XXXVIII")
    back = restart_note.take_return(now=NOW + 9 * 3600, path=note, exit_path=last_exit,
                                    current_mark="Mark XXXIX")
    assert back is not None and back.kind == "update" and back.updated
    line = restart_note.update_line(back)
    assert "Mark XXXIX" in line and "Mark XXXVIII" in line
    assert restart_note.update_spoken(back) == "I'm running Mark thirty-nine now."


def test_a_restart_into_a_new_mark_says_both(tmp_path):
    note, last_exit = _notes(tmp_path)
    restart_note.remember(now=NOW, path=note, mark="Mark XXXVIII")
    restart_note.remember_exit("restart", now=NOW + 1, path=last_exit, mark="Mark XXXVIII")
    back = restart_note.take_return(now=NOW + 7, path=note, exit_path=last_exit,
                                    current_mark="Mark XXXIX")
    assert back.kind == "restart" and back.updated
    assert "now running Mark XXXIX" in restart_note.instruction(back, now=NOW + 7)
    assert not note.exists() and not last_exit.exists()


def test_what_he_said_is_kept_newest_last_and_bounded(tmp_path):
    said = tmp_path / "greeting_memory.json"
    for i in range(restart_note.SAID_KEEP + 2):
        assert restart_note.note_said("greeting", f"Good morning number {i}.", path=said)
    restart_note.note_said("restart", "Back in a flash.", path=said)
    recent = restart_note.recent_said("greeting", path=said, limit=3)
    assert recent == [f"Good morning number {i}." for i in (5, 6, 7)]
    assert restart_note.recent_said("restart", path=said) == ["Back in a flash."]
    assert not restart_note.note_said("greeting", "   ", path=said)


def test_after_a_quick_relaunch_he_does_not_greet_or_offer_the_briefing():
    import time
    back = restart_note.Restart(time.time() - 300, "relaunch", "Mark XXXIX", "Mark XXXIX")
    worker, sent, said = _worker(back, session=object())
    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    assert len(sent) == 1 and "quick return" in sent[0]
    assert worker.awaiting_briefing is False and worker._opening_kind == "relaunch"
    assert any(line.startswith("SYS: back after a short break") for line in worker.bus.lines)


def test_an_update_is_known_even_without_an_exit_note(tmp_path):
    note, last_exit = _notes(tmp_path)
    marks, evidence = tmp_path / "last_mark.json", tmp_path / "identity.json"
    kw = dict(path=note, exit_path=last_exit, mark_path=marks, evidence_path=evidence)
    # A brand-new install: nothing ran before, so nothing to announce.
    assert restart_note.take_return(now=NOW, current_mark="Mark XXXIX", **kw) is None
    # Same Mark, no exit note (a crash): an ordinary launch.
    assert restart_note.take_return(now=NOW + 60, current_mark="Mark XXXIX", **kw) is None
    # A newer Mark after a run that left no note: an update.
    back = restart_note.take_return(now=NOW + 120, current_mark="Mark XXXX", **kw)
    assert back.kind == "update" and back.previous_mark == "Mark XXXIX"
    # The first start of a mark-keeping build on a PC an older ORION used.
    marks.unlink()
    evidence.write_text("{}", encoding="utf-8")
    back = restart_note.take_return(now=NOW + 180, current_mark="Mark XXXIX", **kw)
    assert back.kind == "update" and back.updated and back.previous_mark == ""
    assert "first start as Mark XXXIX (say it 'Mark thirty-nine'" in restart_note.update_line(back)


def test_the_offline_return_line_is_never_the_last_one_said():
    back = restart_note.Restart(NOW - 300, "relaunch", "Mark XXXX", "Mark XXXX")
    said = list(restart_note._RELAUNCH_LINES[:-1])
    assert restart_note.spoken_line(back, recent=said) == restart_note._RELAUNCH_LINES[-1]
    # Every line used: any of them, rather than nothing.
    assert restart_note.spoken_line(back, recent=list(restart_note._RELAUNCH_LINES)) \
        in restart_note._RELAUNCH_LINES


def test_the_mark_is_spoken_as_a_number():
    assert restart_note.spoken_mark("Mark XXXX") == "Mark forty"
    assert restart_note.spoken_mark("Mark XXXIX") == "Mark thirty-nine"
    assert restart_note.spoken_mark("Mark XIV") == "Mark fourteen"
    assert restart_note.spoken_mark("Mark IX") == "Mark nine"
    assert restart_note.spoken_mark("Version 2") == "Version 2"
    back = restart_note.Restart(NOW, "update", "Mark XXXIX", "Mark XXXX")
    assert restart_note.update_spoken(back) == "I'm running Mark forty now."


def test_the_same_mark_written_another_way_is_not_an_update():
    """28 Oct 2026: forty is written XXXX now, not XL. The first start after
    that must not announce an update to the mark he was already on."""
    assert restart_note.same_mark("Mark " + "XL", "Mark XXXX")
    assert not restart_note.same_mark("Mark XXXIX", "Mark XXXX")
    back = restart_note.Restart(NOW, "restart", "Mark " + "XL", "Mark XXXX")
    assert not back.updated
