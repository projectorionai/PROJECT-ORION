"""
ORION reads his own source and says what changed.

The ask: "I want ORION to be able to tell me in detail what's happened in its
changelogs, not just what files have been changed. I want it to look in its own
code and tell me."

So the bar is: a semantic diff (functions and classes, not files), explained in
the code's own words, and never inventing a purpose it cannot read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orion_core.code_changelog import CodeChangelog, CodeIndex

BEFORE = '''"""Audio playback."""

def render(chunk, device=None):
    """Write a PCM chunk to the output device."""
    return len(chunk)

class Speaker:
    """The output device."""

    def open(self, index):
        """Open the device."""
        return True

    def doomed(self):
        """A method about to be deleted."""
        return None
'''

AFTER = '''"""Audio playback."""

def render(chunk, device=None, latency=0.3):
    """Write a PCM chunk to the output device."""
    return len(chunk)

class Speaker:
    """The output device."""

    def open(self, index):
        """Open the device, verifying it accepts a stream."""
        probe = index
        return True
'''


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "orion_core"
    root.mkdir()
    (root / "audio.py").write_text(BEFORE, encoding="utf-8")
    return root, tmp_path / "index.json"


def _kinds(changes):
    return {(c.kind, c.symbol) for c in changes}


# ── the semantic diff ─────────────────────────────────────────────────────────

def test_a_new_parameter_is_a_signature_change(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    assert ("resigned", "render") in _kinds(changes)
    detail = next(c.detail for c in changes if c.symbol == "render")
    assert "latency=0.3" in detail, "it must say WHAT changed, not just that it did"


def test_a_changed_body_is_a_rewrite(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    assert ("rewritten", "Speaker.open") in _kinds(changes)


def test_a_deleted_method_is_reported_as_removed(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    assert ("removed", "Speaker.doomed") in _kinds(changes)


def test_a_new_module_is_reported_with_its_purpose(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "smoothing.py").write_text(
        '"""Caps over-long silences so speech keeps its rhythm."""\n\n'
        "def compress(pcm):\n    return pcm\n", encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    added = [c for c in changes if c.kind == "added" and "smoothing" in c.module]
    assert added
    assert "silences" in added[0].purpose


def test_a_class_is_not_rewritten_just_because_a_method_changed(project):
    """One edit must not produce two entries about the same change.

    Only the method BODY changes here — no method is added or removed, since
    either of those genuinely does change the class's own shape and should be
    reported as such.
    """
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    body_only_change = BEFORE.replace(
        '        """Open the device."""\n        return True',
        '        """Open the device."""\n        probe = index\n        return True')
    (root / "audio.py").write_text(body_only_change, encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    assert ("rewritten", "Speaker.open") in _kinds(changes)
    assert ("rewritten", "Speaker") not in _kinds(changes)


def test_removing_a_method_does_change_the_class_shape(project):
    """The counterpart: a class's own shape changing IS worth reporting."""
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    assert ("removed", "Speaker.doomed") in _kinds(changes)
    assert ("rewritten", "Speaker") in _kinds(changes)


def test_only_the_docstring_changing_is_not_a_behaviour_change(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(
        BEFORE.replace('"""Open the device."""',
                       '"""Open the device and prove it accepts a stream."""'),
        encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    assert ("redocumented", "Speaker.open") in _kinds(changes)
    assert ("rewritten", "Speaker.open") not in _kinds(changes)


def test_reformatting_alone_is_not_a_change(project):
    """A text diff would scream; a semantic one must stay quiet."""
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    noisy = BEFORE.replace("def render(chunk, device=None):",
                           "def render(\n    chunk,\n    device=None,\n):")
    noisy = "# a new comment nobody asked about\n" + noisy
    (root / "audio.py").write_text(noisy, encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    assert changes == [], f"reformatting is not a change: {changes}"


def test_nothing_changing_reports_nothing(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    changes, previous = changelog.changes_since_snapshot()
    assert changes == []
    assert "Nothing in my code has changed" in changelog.narrate(changes, previous)


# ── the narration ─────────────────────────────────────────────────────────────

def test_the_report_names_functions_not_files(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")

    text = changelog.report(update=False)
    assert "render" in text
    assert "Speaker.open" in text
    assert "Write a PCM chunk to the output device" in text, (
        "the purpose must come from the code's own docstring")


def test_no_double_full_stops(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")
    assert ".." not in changelog.report(update=False).replace("…", "")


def test_the_spoken_form_conjugates_the_verb(project):
    """Docstrings are imperative; spoken English needs the third person."""
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")

    changes, previous = changelog.changes_since_snapshot()
    spoken = changelog.narrate(changes, previous, spoken=True)
    assert "which writes" in spoken
    assert "which write " not in spoken


def test_a_noun_phrase_docstring_is_not_forced_through_which(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(
        BEFORE.replace("def doomed(self):", "def added_later(self):"), encoding="utf-8")

    changes, previous = changelog.changes_since_snapshot()
    spoken = changelog.narrate(changes, previous, spoken=True)
    assert "which a method" not in spoken


@pytest.mark.parametrize("verb,expected", [
    ("write", "writes"), ("open", "opens"), ("cap", "caps"),
    ("catch", "catches"), ("verify", "verifies"), ("returns", "returns"),
    ("go", "goes"), ("pass", "passes"),
])
def test_third_person_conjugation(verb, expected):
    assert CodeChangelog._third_person(verb) == expected


def test_the_spoken_form_is_bounded(project):
    """Read aloud, an unbounded list is a monologue."""
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    body = "\n".join(f'def f{i}():\n    """Return {i}."""\n    return {i}\n'
                     for i in range(40))
    # Appended to an EXISTING module: a brand-new module is deliberately
    # collapsed to a single "added the whole module" entry, so it would never
    # exceed the spoken cap however many functions it contained.
    (root / "audio.py").write_text(BEFORE + "\n" + body, encoding="utf-8")

    changes, previous = changelog.changes_since_snapshot()
    spoken = changelog.narrate(changes, previous, spoken=True)
    assert "more; the full list is on screen" in spoken


# ── honesty ───────────────────────────────────────────────────────────────────

def test_a_symbol_with_no_docstring_gets_no_invented_purpose(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(
        BEFORE + "\ndef undocumented(x):\n    return x\n", encoding="utf-8")

    changes, _ = changelog.changes_since_snapshot()
    added = next(c for c in changes if c.symbol == "undocumented")
    assert added.purpose == "", "a missing docstring must not be filled in"


def test_explain_says_when_a_symbol_has_no_docstring(project):
    root, index = project
    (root / "audio.py").write_text(
        BEFORE + "\ndef undocumented(x):\n    return x\n", encoding="utf-8")
    text = CodeChangelog(root, index).explain("undocumented")
    assert "does not say what it is for" in text


def test_explain_is_honest_about_a_name_it_does_not_have(project):
    root, index = project
    assert "nothing called" in CodeChangelog(root, index).explain("nonexistent_thing")


def test_the_first_ever_run_takes_a_baseline_and_says_so(project):
    root, index = project
    changelog = CodeChangelog(root, index)
    changes, previous = changelog.changes_since_snapshot()
    assert previous is None
    assert "no earlier snapshot" in changelog.narrate(changes, previous)


# ── robustness ────────────────────────────────────────────────────────────────

def test_an_unparseable_file_is_skipped_not_fatal(project):
    root, _index = project
    (root / "broken.py").write_text("def (:\n", encoding="utf-8")
    index = CodeIndex.build(root)
    assert "orion_core/audio.py" in index.modules
    assert not any("broken" in path for path in index.modules)


def test_pycache_is_not_indexed(project):
    root, _index = project
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("x = 1\n", encoding="utf-8")
    assert not any("__pycache__" in p for p in CodeIndex.build(root).modules)


def test_the_index_round_trips_through_disk(project):
    root, index_path = project
    built = CodeIndex.build(root)
    assert built.save(index_path)
    loaded = CodeIndex.load(index_path)
    assert loaded is not None
    assert built.diff(loaded) == []


def test_a_missing_index_file_loads_as_none(tmp_path):
    assert CodeIndex.load(tmp_path / "nope.json") is None


def test_a_corrupt_index_file_loads_as_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert CodeIndex.load(path) is None


def test_output_is_console_safe(project):
    """This text reaches the log, a cp1252 console and the voice channel."""
    root, index = project
    changelog = CodeChangelog(root, index)
    changelog.snapshot()
    (root / "audio.py").write_text(AFTER, encoding="utf-8")
    text = changelog.report(update=False)
    text.encode("cp1252")          # must not raise


# ── it indexes ORION himself ─────────────────────────────────────────────────

def test_it_can_index_orions_real_source():
    root = Path(__file__).resolve().parents[1] / "orion_core"
    index = CodeIndex.build(root)
    assert len(index.modules) > 100
    assert any("code_changelog.py" in path for path in index.modules)


def test_the_tool_is_registered():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    names = {d["name"] for d in TOOL_DECLARATIONS}
    assert "code_changes" in names


def test_the_tool_description_distinguishes_it_from_self_changes():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    entry = next(d for d in TOOL_DECLARATIONS if d["name"] == "code_changes")
    assert "self_changes gives files" in entry["description"], (
        "the model must know which of the three to reach for")
