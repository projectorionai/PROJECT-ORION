"""
Undo — taking back what ORION did to the filesystem.

ORION's ``file_controller`` could write, append, create and delete with no way
back at all. A voice assistant mishears, and when it does "sorry" is not a
remedy: the file is already overwritten. The alternative of confirming every
write is worse — each confirmation is a round trip to the model, and an
assistant that checks before creating a folder is one you stop talking to.

So the rule is act immediately, remember how to reverse it, and reserve
confirmation for what genuinely cannot be reversed. These tests pin the three
properties that make that trade honest rather than merely convenient:

  * it does not guess — where the previous state could not be read, nothing is
    registered, because an undo that restores a reconstruction is worse than
    none at all;
  * it does not hoard — oversized files are excluded and say so;
  * it does not delete the user's files to tidy up after ORION.

Offline: a temp directory, no Qt, no model.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from orion_core import undo
from orion_core.dispatch_files import FilesDispatchMixin


class _Files(FilesDispatchMixin):
    """The file tool with its path guards stubbed to a temp directory."""

    def _resolve_user_path(self, raw):
        return pathlib.Path(raw)

    def _ensure_write_safe(self, path):
        return None


@pytest.fixture
def files():
    undo.clear()
    yield _Files()
    undo.clear()


# ── the stack ────────────────────────────────────────────────────────────────

def test_an_empty_stack_says_so_without_pretending(files):
    ok, message = undo.undo_last()
    assert ok is False
    assert "nothing to undo" in message.lower()
    assert "changed myself" in message, "it should say what it does and does not track"


def test_the_stack_is_bounded(files):
    for i in range(undo.MAX_DEPTH + 5):
        undo.push_undo(f"thing {i}", lambda: "")
    assert undo.depth() == undo.MAX_DEPTH
    # The OLDEST are dropped: the recent past is what people ask to undo.
    assert undo.peek() == f"thing {undo.MAX_DEPTH + 4}"


def test_a_failing_undo_is_not_retried_forever(files):
    def explode() -> str:
        raise RuntimeError("the world moved on")

    undo.push_undo("doomed", explode)
    ok, message = undo.undo_last()
    assert ok is False and "doomed" in message
    assert undo.can_undo() is False, "a failing entry must still be popped"


def test_a_broken_registration_cannot_break_the_action(files):
    undo.push_undo("not callable", None)     # type: ignore[arg-type]
    assert undo.can_undo() is False


def test_history_is_most_recent_first(files):
    undo.push_undo("first", lambda: "")
    undo.push_undo("second", lambda: "")
    assert undo.history() == ["second", "first"]


# ── writes ───────────────────────────────────────────────────────────────────

def test_overwriting_a_file_can_be_taken_back(files, tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    files.file_controller({"action": "write_text", "path": str(target),
                           "text": "CLOBBERED"})
    assert target.read_text(encoding="utf-8") == "CLOBBERED"
    ok, _message = undo.undo_last()
    assert ok is True
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


def test_undoing_a_write_to_a_new_file_removes_it(files, tmp_path):
    target = tmp_path / "fresh.txt"
    files.file_controller({"action": "write_text", "path": str(target),
                           "text": "new"})
    assert target.exists()
    undo.undo_last()
    assert not target.exists(), "a file ORION created should go away again"


def test_appending_can_be_taken_back(files, tmp_path):
    target = tmp_path / "log.txt"
    target.write_text("line one\n", encoding="utf-8")
    files.file_controller({"action": "append_text", "path": str(target),
                           "text": "line two\n"})
    assert "line two" in target.read_text(encoding="utf-8")
    undo.undo_last()
    assert target.read_text(encoding="utf-8") == "line one\n"


def test_an_oversized_write_is_not_undoable_and_says_so(files, tmp_path, monkeypatch):
    """It does not hoard: holding a 200 MB log alive for the session to make an
    undo possible is a worse outcome than saying the undo is unavailable."""
    monkeypatch.setattr("orion_core.dispatch_files.UNDO_MAX_BYTES", 16)
    target = tmp_path / "big.txt"
    target.write_text("x" * 500, encoding="utf-8")
    result = files.file_controller({"action": "write_text", "path": str(target),
                                    "text": "replaced"})
    assert "cannot be undone" in result.text
    assert undo.can_undo() is False


# ── deletes ──────────────────────────────────────────────────────────────────

def test_deleting_a_file_can_be_taken_back(files, tmp_path):
    target = tmp_path / "gone.txt"
    target.write_text("PRECIOUS", encoding="utf-8")
    result = files.file_controller({"action": "delete", "path": str(target)})
    assert not target.exists()
    assert "undo" in result.text.lower()
    ok, _message = undo.undo_last()
    assert ok is True
    assert target.read_text(encoding="utf-8") == "PRECIOUS"


def test_an_oversized_delete_is_honest_about_being_final(files, tmp_path, monkeypatch):
    monkeypatch.setattr("orion_core.dispatch_files.UNDO_MAX_BYTES", 16)
    target = tmp_path / "huge.txt"
    target.write_text("y" * 500, encoding="utf-8")
    result = files.file_controller({"action": "delete", "path": str(target)})
    assert "final" in result.text
    assert undo.can_undo() is False, "nothing should be registered it cannot do"


def test_removing_an_empty_folder_can_be_taken_back(files, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    files.file_controller({"action": "delete", "path": str(folder)})
    assert not folder.exists()
    undo.undo_last()
    assert folder.is_dir()


# ── folders ──────────────────────────────────────────────────────────────────

def test_creating_a_folder_can_be_taken_back(files, tmp_path):
    folder = tmp_path / "made"
    files.file_controller({"action": "mkdir", "path": str(folder)})
    assert folder.is_dir()
    undo.undo_last()
    assert not folder.exists()


def test_undo_never_deletes_the_users_files_to_tidy_up(files, tmp_path):
    """THE safety property. The reverse of 'create a folder' is removing it
    only while it is still EMPTY. Once the user has put something inside, the
    folder stays — an undo that destroyed their work would be far worse than
    the misheard command it was correcting."""
    folder = tmp_path / "project"
    files.file_controller({"action": "mkdir", "path": str(folder)})
    mine = folder / "my_work.txt"
    mine.write_text("hours of effort", encoding="utf-8")

    ok, message = undo.undo_last()
    assert ok is True
    assert "no longer empty" in message
    assert folder.is_dir(), "the folder was removed with a file inside it"
    assert mine.read_text(encoding="utf-8") == "hours of effort"


def test_a_folder_that_already_existed_is_not_registered(files, tmp_path):
    """mkdir is exist_ok, so ORION may not have created it at all. Removing a
    folder he merely touched would be destroying something he never made."""
    folder = tmp_path / "already"
    folder.mkdir()
    files.file_controller({"action": "mkdir", "path": str(folder)})
    assert undo.can_undo() is False


# ── the tool ─────────────────────────────────────────────────────────────────

def test_the_undo_tool_is_declared_and_routed():
    """An undo the model cannot reach is dead code — which ORION has shipped
    before, in the shape of a viseme signal with no producer."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    declared = {tool["name"] for tool in TOOL_DECLARATIONS}
    assert "undo" in declared
    spec = next(t for t in TOOL_DECLARATIONS if t["name"] == "undo")
    # The model has to know to reach for it on the plain English word.
    assert "undo" in spec["description"].lower()
    assert "any language" in spec["description"]


def test_the_tool_lists_what_it_can_reverse(files, tmp_path):
    files.file_controller({"action": "mkdir", "path": str(tmp_path / "one")})
    files.file_controller({"action": "mkdir", "path": str(tmp_path / "two")})
    listing = files.undo_tool({"action": "list"}).text
    assert "two" in listing and "one" in listing
    assert listing.index("two") < listing.index("one"), "most recent first"


def test_listing_changes_nothing(files, tmp_path):
    folder = tmp_path / "kept"
    files.file_controller({"action": "mkdir", "path": str(folder)})
    files.undo_tool({"action": "list"})
    assert folder.is_dir(), "listing must not perform the undo"
    assert undo.depth() == 1


def test_the_tool_reverses_the_last_action(files, tmp_path):
    target = tmp_path / "t.txt"
    target.write_text("BEFORE", encoding="utf-8")
    files.file_controller({"action": "write_text", "path": str(target),
                           "text": "AFTER"})
    result = files.undo_tool({"action": "last"})
    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "BEFORE"


def test_the_tool_defaults_to_reversing(files, tmp_path):
    files.file_controller({"action": "mkdir", "path": str(tmp_path / "d")})
    assert files.undo_tool({}).ok is True


def test_the_tool_reports_an_empty_stack_without_failing_loudly(files):
    result = files.undo_tool({"action": "last"})
    assert result.ok is False
    assert "nothing" in result.text.lower()
