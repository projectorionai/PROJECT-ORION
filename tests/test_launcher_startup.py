"""Exercise the actual desktop entry point without launching ORION services."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, ORION_BACKGROUND="0", ORION_RELAUNCHED="1",
               ORION_HEADLESS="0", QT_QPA_PLATFORM="offscreen",
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(code)], cwd=ROOT,
        env=env, capture_output=True, text=True, timeout=60,
    )


def test_desktop_launcher_keeps_expensive_packages_deferred():
    result = _run("""
        import json, sys
        import orion
        print(json.dumps({
            'desktop': 'orion_core.app' in sys.modules,
            'heavy': [name for name in ('google.genai', 'aiohttp', 'sounddevice')
                      if name in sys.modules],
            'main': callable(orion.main),
        }))
    """)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "desktop": True, "heavy": [], "main": True,
    }


def test_missing_dependencies_are_reported_together_before_loading_the_app():
    result = _run("""
        import importlib.util, sys
        original = importlib.util.find_spec
        def find_spec(name):
            if name in {'sounddevice', 'aiohttp'}:
                return None
            if name == 'google.genai':
                raise ModuleNotFoundError("No module named 'google'")
            return original(name)
        importlib.util.find_spec = find_spec
        try:
            import orion
        except SystemExit as exc:
            assert 'orion_core.app' not in sys.modules
            raise
    """)
    assert result.returncode == 1
    assert "Missing packages: aiohttp, sounddevice, google.genai" in result.stdout
    assert "pip install" in result.stdout
    assert "Traceback" not in result.stderr


def test_headless_launcher_does_not_check_desktop_dependencies():
    result = _run("""
        import importlib.util, json, sys, types
        sys.argv = ['orion.py', '--headless']
        checked = []
        def find_spec(name):
            checked.append(name)
            return object()
        importlib.util.find_spec = find_spec
        server = types.ModuleType('orion_core.server')
        server.main = lambda: None
        sys.modules['orion_core.server'] = server
        import orion
        print(json.dumps(checked))
        assert orion.main is server.main
        assert 'orion_core.app' not in sys.modules
    """)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ['aiohttp', 'qasync', 'PyQt6.QtCore']


def test_preflight_never_executes_a_runtime_dependency():
    result = _run("""
        import importlib.abc, importlib.machinery, importlib.util, json, sys, types
        checked = []
        class NeverExecute(importlib.abc.Loader):
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise AssertionError('preflight imported a runtime dependency')
        def find_spec(name):
            checked.append(name)
            return importlib.machinery.ModuleSpec(name, NeverExecute())
        importlib.util.find_spec = find_spec
        app = types.ModuleType('orion_core.app')
        app.main = lambda: None
        sys.modules['orion_core.app'] = app
        import orion
        print(json.dumps(checked))
    """)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "aiohttp", "mss", "psutil", "qasync", "sounddevice",
        "google.genai", "PIL.Image", "PyQt6.QtWidgets",
    ]


def test_desktop_entry_passes_its_launch_clock_to_the_startup_report():
    # runpy with run_name='__main__' is a real launch, so the single-instance
    # guard fires -- correctly. Give this one its own lock name so the test
    # measures the launch clock rather than whether the developer happens to
    # have ORION open.
    result = _run("""
        import os, runpy, sys, time, types
        import orion_core.single_instance as si
        si.MUTEX_NAME = si.MUTEX_NAME + '-test-%d' % os.getpid()
        si.SOCKET_NAME = si.SOCKET_NAME + '-test-%d' % os.getpid()
        si.LOCK_FILENAME = 'orion-test-%d.lock' % os.getpid()
        captured = {}
        app = types.ModuleType('orion_core.app')
        app.main = lambda **kwargs: captured.update(kwargs)
        sys.modules['orion_core.app'] = app
        before = time.perf_counter()
        runpy.run_path('orion.py', run_name='__main__')
        assert before <= captured['started_at'] <= time.perf_counter()
        print('ok')
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
