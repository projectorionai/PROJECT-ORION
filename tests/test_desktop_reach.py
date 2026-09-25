"""ORION has to be able to see, and open, everything on the desk.

Three separate faults, each of which made him truthfully report that
something was not there when it plainly was.

1. Store apps were invisible. A Microsoft Store app has no App Paths registry
   entry, nothing on PATH and no .lnk in the Start Menu folders — the three
   places the resolver looked. WhatsApp, TikTok, Sticky Notes and 65 others on
   this machine existed only in the shell's AppsFolder namespace, so the user
   had to name a path for something sitting on their taskbar.

2. Vision looked at one screen. `_monitor` defaulted to `monitors[1]` — the
   primary — so on a two-monitor desk half of what was in front of the user
   was outside ORION's view entirely.

3. OCR searched one screen, as a downscaled panorama. Measured here: one pass
   over 3840x1080 took 10.7 s, the primary alone 1.6 s, and scaling a
   two-monitor capture down to 1600 px halves small text until the recogniser
   stops reading it.

Offline: no app is launched and no screen is required.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core import app_resolver, vision  # noqa: E402


# ── 1. Store apps ────────────────────────────────────────────────────────────

def test_the_apps_folder_is_one_of_the_sources():
    """The three original sources cannot see a Store app at all."""
    source = inspect.getsource(app_resolver.AppCatalogue.build)
    assert "_from_apps_folder" in source


def test_a_store_app_target_is_launched_through_the_shell():
    """A Store app has no executable to run; Explorer starts it by its
    AppUserModelID."""
    source = inspect.getsource(app_resolver.AppCatalogue._from_apps_folder)
    assert "shell:AppsFolder" in source
    assert '"!" not in app_id' in source, (
        "ordinary shortcuts would be duplicated from the other sources")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell namespace")
def test_store_apps_are_actually_found_on_this_machine():
    found = app_resolver.AppCatalogue._from_apps_folder()
    if not found:
        pytest.skip("no AppsFolder entries available in this environment")
    assert all("shell:AppsFolder" in c.target for c in found)
    assert all(c.source == "apps_folder" for c in found)


def test_the_launcher_handles_a_shell_target():
    """os.startfile understands shell: URIs; Popen does not."""
    from orion_core.agents import DesktopAgent

    source = inspect.getsource(DesktopAgent._launch)
    exe_branch = source.index('endswith(".exe")')
    startfile = source.index("os.startfile")
    assert exe_branch < startfile, (
        "a shell: target must fall through to the shell, not to Popen")


# ── 2. every screen ──────────────────────────────────────────────────────────

class _FakeCapture:
    """Two 1920x1080 screens side by side, as mss reports them."""

    monitors = [
        {"left": 0, "top": 0, "width": 3840, "height": 1080},      # the union
        {"left": 0, "top": 0, "width": 1920, "height": 1080},      # primary
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},
    ]


def _grabber():
    cls = next(o for _n, o in inspect.getmembers(vision, inspect.isclass)
               if hasattr(o, "monitor_bounds"))
    return object.__new__(cls), cls


def test_capture_defaults_to_every_screen():
    """It defaulted to monitors[1]. Asked to find something on the second
    screen, ORION would answer — truthfully, as far as he could tell — that it
    was not there."""
    instance, cls = _grabber()
    chosen = cls._monitor(instance, _FakeCapture(), None)
    assert chosen["width"] == 3840, "only one screen is being captured"


def test_an_explicit_screen_is_still_honoured():
    instance, cls = _grabber()
    assert cls._monitor(instance, _FakeCapture(), 2)["left"] == 1920
    assert cls._monitor(instance, _FakeCapture(), 1)["left"] == 0


def test_an_out_of_range_screen_falls_back_to_everything():
    instance, cls = _grabber()
    assert cls._monitor(instance, _FakeCapture(), 9)["width"] == 3840


# ── 3. OCR across screens ────────────────────────────────────────────────────

def _agent():
    cls = next(o for _n, o in inspect.getmembers(vision, inspect.isclass)
               if hasattr(o, "_screens_to_search"))
    agent = object.__new__(cls)

    class _Grab:
        @staticmethod
        def _grabber():
            return _FakeCapture()

    agent.grabber = _Grab()
    return agent, cls


def test_every_screen_is_searched_primary_first():
    agent, cls = _agent()
    assert cls._screens_to_search(agent, None) == [1, 2]


def test_an_explicit_screen_narrows_the_search():
    agent, cls = _agent()
    assert cls._screens_to_search(agent, 2) == [2]


def test_a_hit_is_returned_in_virtual_desktop_coordinates():
    """A coordinate is only meaningful across two screens once the monitor's
    own offset is added back. Without it, a click for the right-hand screen
    lands on the left one."""
    agent, cls = _agent()

    class _Engine:
        available = True

        @staticmethod
        def find_text_box(_image, _query):
            return (10, 20, 100, 40)          # relative to whichever screen

    class _Grab(agent.grabber.__class__):
        @staticmethod
        def monitor_bounds(index=None):
            return {1: (0, 0, 1920, 1080), 2: (1920, 0, 1920, 1080)}[index]

        @staticmethod
        def capture_image(max_side=0, monitor=None):
            return object()

        @staticmethod
        def _grabber():
            return _FakeCapture()

    agent.grabber = _Grab()
    agent.ocr_engine = _Engine()

    hit = cls._find_element_via_ocr_sync(agent, "anything", 2)
    assert hit["center"] == (1920 + 10 + 50, 0 + 20 + 20)
    assert hit["monitor"] == 2


def test_a_screen_that_fails_does_not_end_the_search():
    """One bad capture must not hide a match on the other screen."""
    agent, cls = _agent()
    seen: list = []

    class _Engine:
        available = True

        @staticmethod
        def find_text_box(image, _query):
            seen.append(image)
            return (1, 2, 3, 4) if image == "screen-2" else None

    class _Grab:
        @staticmethod
        def monitor_bounds(index=None):
            if index == 1:
                raise OSError("screen 1 is unreadable")
            return (1920, 0, 1920, 1080)

        @staticmethod
        def capture_image(max_side=0, monitor=None):
            return f"screen-{monitor}"

        @staticmethod
        def _grabber():
            return _FakeCapture()

    agent.grabber = _Grab()
    agent.ocr_engine = _Engine()
    hit = cls._find_element_via_ocr_sync(agent, "anything", None)
    assert hit is not None and hit["monitor"] == 2


def test_ocr_reads_at_native_resolution():
    """1600 px was fine for one 1920-wide screen. Across two it scales the
    desktop to 42% and small labels stop being legible."""
    source = inspect.getsource(vision)
    assert "capture_image(max_side=1600)" not in source, (
        "a two-monitor capture is being downscaled before OCR")
