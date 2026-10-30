"""
The suite must not be able to stop existing.

What happened
-------------
tools/prepare_public_source.py wrote a publication export to
release/validation-2026-09-19/ORION/, which contains a complete copy of tests/.
pytest then tried to import two different files as `test_memory.py`, hit the
import-file-mismatch error, and aborted:

    !!!!!!!! Interrupted: 626 errors during collection !!!!!!!!
    626 errors in 15.61s

Zero tests ran. Not one failure was reported, because nothing was collected to
fail — the safety net did not go red, it went away. Anything committed in that
window would have looked exactly as green as a clean run.

The export was correctly gitignored, which is why this was easy to miss:
.gitignore governs what git tracks and has no bearing whatever on what pytest
walks.

These tests are cheap and exist only so that a generated copy of the project
can never again silently delete the suite.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ignored() -> list[str]:
    import conftest

    return list(getattr(conftest, "collect_ignore_glob", []))


@pytest.mark.parametrize("directory", ["release", "dist", "build", ".venv", "reports"])
def test_generated_copies_of_the_project_are_not_collected(directory):
    """Each of these can hold a full tests/ tree with the same basenames."""
    assert f"{directory}/*" in _ignored(), (
        f"{directory}/ is not excluded from collection; a copy of the project "
        f"landing there will abort the whole run"
    )


def test_the_export_tool_writes_somewhere_that_is_excluded():
    """Pin the two together. If the export path moves and the ignore list does
    not follow, the suite dies again — and dies quietly."""
    source = (ROOT / "tools" / "prepare_public_source.py").read_text(encoding="utf-8")
    if "release" not in source:
        pytest.skip("the export tool no longer mentions a release directory")
    assert "release/*" in _ignored()


def test_a_real_test_tree_is_still_collected():
    """The exclusions must not be so broad that they take the suite with them."""
    assert (ROOT / "tests").is_dir()
    collected = list((ROOT / "tests").glob("test_*.py"))
    assert len(collected) > 100, f"only {len(collected)} test modules found"


def _stray_test_trees(root: Path, norecursedirs: list[str]) -> list[str]:
    """Inspect only directories pytest itself would walk.

    Pruning before descent avoids scanning dependency caches. A hidden virtual
    environment can contain thousands of third-party tests without introducing
    any duplicate modules into this suite.
    """
    excluded = {"release", "dist", "build", "reports", ".venv", ".git", "node_modules",
                # ORION writes whole projects of his own here, each with its
                # own tests; conftest excludes them from collection and so
                # must this.
                "projects",
                "android", "__pycache__"}
    duplicates = []
    for parent, directories, files in os.walk(root):
        directories[:] = [name for name in directories
                          if name not in excluded
                          and not any(fnmatch.fnmatch(name, pattern)
                                      for pattern in norecursedirs)]
        candidate = Path(parent)
        if candidate.name != "tests" or candidate == root / "tests":
            continue
        if any(fnmatch.fnmatch(name, "test_*.py") for name in files):
            duplicates.append(str(candidate.relative_to(root)))
    return duplicates


def test_no_stray_copy_of_the_suite_exists_right_now(pytestconfig):
    """A generated copy must not introduce import mismatches into collection."""
    duplicates = _stray_test_trees(ROOT, pytestconfig.getini("norecursedirs"))
    assert not duplicates, (
        "a second copy of the test suite is present and not excluded: "
        + ", ".join(duplicates[:3])
    )


@pytest.mark.parametrize("directory, detected", [(".tool-cache", False), ("reports", False), ("stray-copy", True)])
def test_duplicate_check_matches_pytest_directory_exclusions(tmp_path, pytestconfig,
                                                           directory, detected):
    copied = tmp_path / directory / "tests"
    copied.mkdir(parents=True)
    (copied / "test_memory.py").write_text("def test_memory(): pass\n", encoding="utf-8")
    assert bool(_stray_test_trees(tmp_path, pytestconfig.getini("norecursedirs"))) is detected


def _qt_lifetime_child(source: str) -> None:
    """Keep a broken native lifetime isolated from the parent test process."""
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)], cwd=ROOT,
        env=environment, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=20,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]


@pytest.mark.parametrize("preexisting", [False, True], ids=["fresh-session", "existing-instance"])
def test_session_qt_owner_preserves_discarded_local_application_and_live_widgets(preexisting):
    pytest.importorskip("PyQt6.QtWidgets")
    _qt_lifetime_child(f'''
        import gc
        import weakref
        import conftest
        from PyQt6 import sip
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication, QWidget
        preexisting = {preexisting!r}
        if preexisting:
            application = QApplication(["orion-existing-application"])
            existing_reference = weakref.ref(application)

        # Emulate pytest activating the session fixture after Qt collection.
        # The original conftest has no owner fixture, so it is a no-op control.
        fixture = getattr(conftest, "_keep_qt_application_alive", None)
        session = fixture.__wrapped__() if fixture is not None else iter((None,))
        next(session)
        if preexisting:
            assert QApplication.instance() is application
            del application

        def create_and_discard_application():
            application = QApplication.instance() or QApplication(["orion-test-lifetime"])
            owner = QWidget()
            timer = QTimer(owner)
            timer.setInterval(33)
            timer.start()
            return weakref.ref(application), owner, timer

        reference, owner, timer = create_and_discard_application()
        gc.collect()
        assert reference() is not None, "the discarded local Qt application lost its session owner"
        assert reference() is QApplication.instance()
        if preexisting:
            assert reference() is existing_reference(), "the fixture replaced the existing application"
        assert not sip.isdeleted(owner) and not sip.isdeleted(timer)
        assert timer.isActive()
        QApplication.processEvents()
        gc.collect()
        assert reference() is QApplication.instance()
        assert not sip.isdeleted(owner) and timer.isActive()
        sip.delete(owner)
        assert sip.isdeleted(timer)
        if fixture is not None:
            session.close()
    ''')


def test_session_qt_owner_does_not_import_or_create_optional_qt():
    _qt_lifetime_child('''
        import sys
        assert not any(name == "PyQt6" or name.startswith("PyQt6.") for name in sys.modules)
        import conftest
        fixture = getattr(conftest, "_keep_qt_application_alive", None)
        session = fixture.__wrapped__() if fixture is not None else iter((None,))
        assert next(session) is None
        assert not any(name == "PyQt6" or name.startswith("PyQt6.") for name in sys.modules), \
            "pure backend tests must not acquire a Qt dependency from the session owner"
        if fixture is not None:
            session.close()
    ''')


def test_session_qt_owner_leaves_core_only_application_without_widgets():
    pytest.importorskip("PyQt6.QtCore")
    _qt_lifetime_child('''
        import sys
        import conftest
        from PyQt6.QtCore import QCoreApplication
        application = QCoreApplication(["orion-core-only-test"])
        assert "PyQt6.QtWidgets" not in sys.modules
        fixture = getattr(conftest, "_keep_qt_application_alive", None)
        session = fixture.__wrapped__() if fixture is not None else iter((None,))
        assert next(session) is None
        assert "PyQt6.QtWidgets" not in sys.modules
        assert QCoreApplication.instance() is application
        if fixture is not None:
            session.close()
    ''')
