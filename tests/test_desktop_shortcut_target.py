"""ORION's shortcut has to point at ORION, on the Desktop the user can see.

Two separate faults made "it still opens as Python 3.13" survive a fix that
had already been written:

  * `_desktop_dir()` built the path from the ONEDRIVE environment variable.
    That is a guess, and on a machine signed into two OneDrive accounts it was
    the wrong one — ONEDRIVE named "OneDrive - Exeter College" while the
    redirected Desktop was "OneDrive - Example University". ORION
    refreshed a shortcut nobody looks at.
  * so the real Desktop kept a stale shortcut launching pythonw with orion.py,
    which makes the running process Python, whatever the icon says.

Offline: no shortcuts are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import desktop_app  # noqa: E402


def test_the_shortcut_launches_orion_not_an_interpreter():
    """The whole point of the standalone build is that ORION.exe IS the
    process. A shortcut to pythonw throws that away."""
    target, arguments, _workdir = desktop_app._launch_target()
    if not desktop_app.STANDALONE_EXE.exists():
        pytest.skip("no standalone build on this machine")
    assert target == str(desktop_app.STANDALONE_EXE)
    assert "python" not in Path(target).name.lower()
    assert arguments == "", "a real executable needs no script argument"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell folders")
def test_the_desktop_is_the_one_windows_reports():
    """Not the one ONEDRIVE happens to name."""
    import ctypes
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8)]

    folder = _GUID(0xB4BFCC3A, 0xDB2C, 0x424C,
                   (ctypes.c_byte * 8)(*bytes.fromhex("B0297FE99A87C641")))
    out = ctypes.c_wchar_p()
    ok = ctypes.WinDLL("shell32").SHGetKnownFolderPath(
        ctypes.byref(folder), 0, None, ctypes.byref(out))
    if ok != 0:
        pytest.skip("shell folder unavailable")
    expected = Path(out.value or "")
    ctypes.WinDLL("ole32").CoTaskMemFree(out)
    assert desktop_app._desktop_dir() == expected


def test_it_does_not_guess_from_onedrive_first():
    """The environment variable is a last resort now, not the first choice."""
    import inspect

    source = inspect.getsource(desktop_app._desktop_dir)
    body = source.split('"""')[-1]        # past the docstring, which names both
    assert body.index("SHGetKnownFolderPath") < body.index("ONEDRIVE")


def test_a_broken_shell_call_still_returns_somewhere():
    """Installing a shortcut must not fail because a Win32 call did."""
    assert desktop_app._desktop_dir().is_absolute()
