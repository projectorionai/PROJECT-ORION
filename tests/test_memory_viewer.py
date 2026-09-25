"""
Reading — and deleting — what ORION has stored about you.

The Command Deck already had a MEMORY panel. It showed counts:

    identity:2  personal:5  knowledge:1207

A number is not transparency. If an assistant keeps a file on someone, that
person has to be able to read it and cross things out, and neither was possible
anywhere in the interface — while ``records()`` and ``forget()`` had been sitting
in the matrix the whole time with nothing to reach them. A capability with no
way to reach it is not a capability.

The property worth guarding
---------------------------
``forget()`` takes a bare category and will remove twelve hundred rows. That is
correct for a tool call and very wrong for a button sitting beside a single
line, so the viewer's delete is scoped by category AND key and returns zero
rather than guessing when either is missing.

Rendered in a subprocess: a QApplication here breaks later Qt tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_PROBE = r'''
import json, os, sys, pathlib, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.memory import OrionMemoryMatrix
from orion_core.gui.memory_viewer import MemoryViewer


class _Sig:
    def emit(self, *a):
        pass


class _Bus:
    log = _Sig()

    def __getattr__(self, name):
        return _Sig()


tmp = pathlib.Path(tempfile.mkdtemp())
matrix = OrionMemoryMatrix(tmp / "core.db", tmp, _Bus())
SEEDS = [
    ("relationship", "sister", "Her sister is a paediatric nurse in Leeds."),
    ("personal", "coffee", "Takes coffee black, no sugar, before ten."),
    ("projects", "orion", "Building ORION, a personal AI operating system."),
    ("identity", "honorific", "Prefers to be addressed as sir."),
    ("knowledge", "fact_1", "A" * 400),
]
for category, key, value in SEEDS:
    matrix.save(category, key, value)

viewer = MemoryViewer(matrix)
viewer.resize(900, 460)
viewer.show()
app.processEvents()


def snapshot():
    return [[viewer.table.item(r, c).text() if viewer.table.item(r, c) else None
             for c in range(4)]
            for r in range(viewer.table.rowCount())]


out = {{
    "columns": [viewer.table.horizontalHeaderItem(i).text()
                for i in range(viewer.table.columnCount())],
    "rows": snapshot(),
    "has_button": viewer.table.cellWidget(0, 4) is not None,
    "long_value_truncated": any(
        row[2].endswith("…") for row in snapshot() if row[2]),
    "long_value_tooltip": max(
        (len(viewer.table.item(r, 2).toolTip())
         for r in range(viewer.table.rowCount())
         if viewer.table.item(r, 2)), default=0),
}}

before = viewer.table.rowCount()
out["removed_one"] = viewer.forget_entry("personal", "coffee")
viewer.refresh()
out["rows_after"] = viewer.table.rowCount()
out["rows_before"] = before
out["bare_category"] = viewer.forget_entry("knowledge", "")
out["bare_key"] = viewer.forget_entry("", "coffee")
viewer.refresh()
out["rows_after_bare"] = viewer.table.rowCount()

viewer.search.setText("sister")
viewer.refresh()
out["search_rows"] = viewer.table.rowCount()
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def viewer() -> dict:
    proc = subprocess.run([sys.executable, "-c", _PROBE.format(
        root=str(ROOT).replace("\\", "/"))], cwd=ROOT,
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced nothing:\n{proc.stdout[-1500:]}")


# ── reading ──────────────────────────────────────────────────────────────────

def test_it_shows_the_facts_not_a_count(viewer):
    """The whole complaint about the old panel."""
    assert viewer["rows_before"] == 5
    values = [row[2] for row in viewer["rows"]]
    assert any("paediatric nurse" in v for v in values), (
        "the stored sentence is not shown anywhere"
    )


def test_it_shows_when_each_thing_was_learned(viewer):
    assert "Learned" in viewer["columns"]
    for row in viewer["rows"]:
        assert row[3].strip(), "an entry with no date"
        assert "T" not in row[3], "raw ISO timestamp shown to a person"


def test_every_row_can_be_forgotten(viewer):
    assert viewer["has_button"] is True


def test_a_long_value_is_wrapped_but_not_hidden(viewer):
    """Truncated in the cell, complete in the tooltip — and complete again in
    the confirmation, so nothing is deleted unseen."""
    assert viewer["long_value_truncated"] is True
    assert viewer["long_value_tooltip"] > 200


def test_search_narrows_the_table(viewer):
    assert viewer["search_rows"] < viewer["rows_after"]
    assert viewer["search_rows"] >= 1


# ── deleting ─────────────────────────────────────────────────────────────────

def test_forgetting_removes_exactly_one_entry(viewer):
    assert viewer["removed_one"] == 1
    assert viewer["rows_after"] == viewer["rows_before"] - 1


def test_a_row_button_can_never_wipe_a_whole_category(viewer):
    """THE guard. forget() accepts a bare category and would take 1,200 rows
    with it — right for a tool call, catastrophic for a button beside one line.
    """
    assert viewer["bare_category"] == 0
    assert viewer["bare_key"] == 0
    assert viewer["rows_after_bare"] == viewer["rows_after"], (
        "a partial identifier deleted something"
    )


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_viewer_is_reachable():
    """A window nothing opens is no window."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert "def open_memory_viewer" in source
    assert "memory_viewer" in source and "forget_something" in source, (
        "ORION cannot show the user what he has stored"
    )


def test_deleting_asks_first():
    source = (ROOT / "orion_core" / "gui" / "memory_viewer.py").read_text(encoding="utf-8")
    start = source.index("def _forget(")
    body = source[start:source.index("def forget_entry")]
    assert "QMessageBox" in body, "an entry is deleted with no confirmation"
    assert "excerpt" in body, (
        "the confirmation does not show what is about to be deleted — "
        "'Delete 1 item?' is not informed consent"
    )
    assert "DestructiveRole" in body
    assert "box.buttons()[-1]" in body, "the default button is not the safe one"
