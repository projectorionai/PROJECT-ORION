"""Materialisation counts actual accepted writes, preserving journal history."""
from types import SimpleNamespace

import pytest

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.memory import OrionMemoryMatrix
from orion_core.sync import SyncJournal, materialise_into


def test_blank_payload_is_not_counted_as_a_materialised_memory(tmp_path):
    journal = SyncJournal(tmp_path / "journal.db", node_id="peer")
    matrix = OrionMemoryMatrix(tmp_path / "memory.db", tmp_path, OrionBus())
    try:
        journal.record("notes", "notes", "blank", "  ")
        assert materialise_into(journal, matrix) == 0
        assert matrix.conn.execute("SELECT COUNT(*) FROM intelligence").fetchone()[0] == 0
        assert len(journal.all_entries()) == 1
    finally:
        journal.close()
        matrix.conn.close()


@pytest.mark.parametrize("receipt", [False, ToolResult("synthetic refusal", ok=False)])
def test_failed_adapter_receipt_is_not_counted_as_a_write(tmp_path, receipt):
    journal = SyncJournal(tmp_path / "journal.db", node_id="peer")
    try:
        journal.record("notes", "notes", "key", "value")
        matrix = SimpleNamespace(save=lambda *_args, **_kwargs: receipt)
        assert materialise_into(journal, matrix) == 0
    finally:
        journal.close()
