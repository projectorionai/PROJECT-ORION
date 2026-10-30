"""Dossiers share the guarded publisher used by Word and legacy papers."""

from pathlib import Path
from types import SimpleNamespace

from orion_core import research_docx, research_worker
from orion_core.research_worker import Dossier, ResearchWorker


def test_dossier_transient_windows_publication_denial_preserves_complete_result(tmp_path, monkeypatch):
    original = research_worker.os.replace
    attempts, sleeps = [], []

    def replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) == 1:
            error = PermissionError("synthetic Windows denial")
            error.winerror = 5
            raise error
        return original(source, destination)

    monkeypatch.setattr(research_worker.os, "replace", replace)
    monkeypatch.setattr(research_docx, "_WINDOWS", True)
    monkeypatch.setattr(research_docx, "time", SimpleNamespace(sleep=sleeps.append))
    dossier = Dossier("Synthetic topic", 1.0, summary="Complete synthetic summary")
    path = ResearchWorker(root=tmp_path).file(dossier)
    assert path is not None, dossier.filing_error
    assert "Complete synthetic summary" in path.read_text(encoding="utf-8")
    assert len(attempts) == 2 and len(sleeps) == 1
    assert dossier.filing_error == ""


def test_dossier_changed_destination_is_preserved_after_denied_publication(tmp_path, monkeypatch):
    attempts = []

    def replace(_source, destination):
        attempts.append(destination)
        Path(destination).write_bytes(b"another owner's content")
        error = PermissionError("synthetic changed destination")
        error.winerror = 5
        raise error

    monkeypatch.setattr(research_worker.os, "replace", replace)
    monkeypatch.setattr(research_docx, "_WINDOWS", True)
    dossier = Dossier("Synthetic topic", 1.0, summary="New summary")
    path = ResearchWorker(root=tmp_path).file(dossier)
    assert path is None and dossier.filing_error
    assert len(attempts) == 1
    assert Path(attempts[0]).read_bytes() == b"another owner's content"
    assert list(tmp_path.iterdir()) == [Path(attempts[0])]
