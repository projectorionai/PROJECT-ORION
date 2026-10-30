"""Learned shortcuts must follow the writable config store, independent of cwd."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_learning_persists_in_redirected_config_from_an_unrelated_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    chosen = tmp_path / "external writable data"
    cwd = tmp_path / "shortcut working directory"
    cwd.mkdir()
    # A file named config deliberately makes the historical relative path
    # unusable. The configured writable store must still work across restarts.
    (cwd / "config").write_text("not a directory", encoding="utf-8")
    script = """
from pathlib import Path
from orion_core.constants import CONFIG_DIR
from orion_core.reflex_learning import ReflexLearner
learner = ReflexLearner()
assert learner.path == CONFIG_DIR / 'reflex_learning.db'
learner.observe('find recent papers', 'web_search', {}, ok=True)
learner.close()
reopened = ReflexLearner()
assert reopened._db.execute('select hits from observations').fetchone()[0] == 1
reopened.close()
print('learning stored in configured directory')
"""
    environment = dict(os.environ, ORION_CONFIG_DIR=str(chosen), PYTHONPATH=str(root))
    result = subprocess.run([sys.executable, "-c", script], cwd=cwd, env=environment,
                            capture_output=True, text=True, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (chosen / "reflex_learning.db").is_file()
    assert (cwd / "config").is_file()
