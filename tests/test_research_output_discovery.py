"""Research saved by either runtime must stay discoverable after redirection."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

from orion_core import constants, file_search


ROOT = Path(__file__).resolve().parents[1]


def test_default_paper_export_uses_the_redirected_config(tmp_path):
    """A fresh process resolves its real configuration before the writer imports."""
    configured = tmp_path / "personal-data"
    script = textwrap.dedent("""\
        from pathlib import Path
        from types import SimpleNamespace
        from orion_core import constants, research_docx

        configured = Path(__import__('os').environ['ORION_CONFIG_DIR'])
        assert constants.CONFIG_DIR == configured
        assert research_docx.DEFAULT_DIR == configured / 'research'
        # Both Word and its Markdown fallback use the same destination. Force
        # the fallback so this path regression needs no optional Word library.
        research_docx.available = lambda: False
        paper = SimpleNamespace(topic='Solar research', sections=[], sources=[])
        path, note = research_docx.write_paper(paper)
        assert path is not None, note
        assert path.parent == configured / 'research'
        assert path.read_text(encoding='utf-8').startswith('# Solar research')
        print('saved under redirected configuration')
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        env=dict(os.environ, ORION_CONFIG_DIR=str(configured)),
        capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "saved under redirected configuration" in result.stdout


def test_source_lookup_finds_configured_legacy_and_installed_research(tmp_path, monkeypatch):
    source, configured = tmp_path / "source", tmp_path / "personal-data"
    monkeypatch.setattr(constants, "BASE_DIR", source)
    monkeypatch.setattr(constants, "CONFIG_DIR", configured)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    artifacts = [
        configured / "research" / "solar-dossier.md",
        source / "config" / "research" / "solar-old-paper.docx",
        source / "research" / "solar-source-notes.md",
        source / "dist" / "ORION" / "config" / "research" / "solar-installed-paper.docx",
        source / "dist" / "ORION" / "research" / "solar-saved-notes.md",
    ]
    for path in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic research artifact", encoding="utf-8")
    # Similar names in the installed runtime are not research outputs.
    runtime_file = source / "dist" / "ORION" / "_internal" / "solar-library.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("# runtime fixture", encoding="utf-8")
    output_dirs = file_search.orion_output_dirs()
    assert {path.parent for path in artifacts} <= set(output_dirs)
    assert source / "dist" / "ORION" not in output_dirs
    assert runtime_file.parent not in output_dirs
    assert {Path(hit.path) for hit in file_search.find_own_documents("solar")} == set(artifacts)


def test_same_config_directory_is_only_searched_once(tmp_path, monkeypatch):
    research = tmp_path / "config" / "research"
    research.mkdir(parents=True)
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path)
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert file_search.orion_output_dirs().count(research) == 1


def test_frozen_lookup_does_not_descend_into_another_nested_bundle(tmp_path, monkeypatch):
    research = tmp_path / "config" / "research"
    research.mkdir(parents=True)
    nested = tmp_path / "dist" / "ORION" / "research"
    nested.mkdir(parents=True)
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path)
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert file_search.orion_output_dirs() == [research]


def test_installed_research_is_searched_before_a_slow_source_report_tree(tmp_path, monkeypatch):
    source = tmp_path / "source"
    paper = source / "dist" / "ORION" / "research" / "solar-notes.md"
    paper.parent.mkdir(parents=True)
    paper.write_text("synthetic research", encoding="utf-8")
    reports = source / "reports"
    reports.mkdir()
    monkeypatch.setattr(constants, "BASE_DIR", source)
    monkeypatch.setattr(constants, "CONFIG_DIR", source / "config")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    real_walk = file_search.search_walk
    elapsed = [0.0]

    def walk(query, budget):
        if Path(query.folder) == reports:
            elapsed[0] += budget + 1.0
            return [], "synthetic slow report tree", True
        return real_walk(query, budget=budget)

    with monkeypatch.context() as timed:
        timed.setattr(file_search.time, "monotonic", lambda: elapsed[0])
        timed.setattr(file_search, "search_walk", walk)
        found = file_search.find_own_documents("solar")
    assert [Path(hit.path) for hit in found] == [paper]


def test_topic_folder_lookup_finds_legacy_word_names_without_reading_word(tmp_path, monkeypatch):
    source = tmp_path / "source"
    research = source / "dist" / "ORION" / "config" / "research"
    research.mkdir(parents=True)
    # The lookup should not need a valid Word payload to identify its topic.
    (research / "2026-10-09-university-of-essex-neural-engineering.docx").write_bytes(b"not opened")
    (research / "2026-10-09-unrelated-research.docx").write_bytes(b"also not opened")
    monkeypatch.setattr(constants, "BASE_DIR", source)
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "personal-data")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    def no_content_reads(*_args, **_kwargs):
        raise AssertionError("Word contents should not be read for topic lookup")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", no_content_reads)
        found = file_search.find_research_folders("university essex")
        unrelated = file_search.find_research_folders("solar physics")
    assert [Path(hit.path) for hit in found] == [research]
    assert unrelated == []
