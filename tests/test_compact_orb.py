"""The compact orb: where it lands, which face it shows, and that it never
touches the main window.

Three faults, all found by using it rather than by reading it.

It appeared in the middle of the desktop instead of the bottom-right corner,
because it was placed relative to the PRIMARY screen rather than the screen
ORION was actually on.

Pressing it brought back an old software-rendered face — and, on the way
back, a maximised window painted black around a small face in its corner, or
"Not Responding". All of that came from one decision: the overlay re-flagged
the MAIN window. setWindowFlags destroys and recreates the native window, a
QWebEngineView cannot survive that, and a freshly rebuilt WebGL page inside a
translucent tool window often never started — so the health watcher fell back
to the software head for the rest of the session.

Mark XXXI makes the orb its own small window (gui/orb_overlay.py). The main
window is only hidden and shown again, so the face is never rebuilt at all.

No QApplication is constructed here: doing that inline hangs this suite. These
read the source; the behaviour is driven in test_core_window_facefirst.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WINDOW = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(
    encoding="utf-8")
FACE = (ROOT / "orion_core" / "gui" / "face3d.py").read_text(encoding="utf-8")
ORB = (ROOT / "orion_core" / "gui" / "orb_overlay.py").read_text(encoding="utf-8")


def _body(source: str, start: str, end: str = "\n    def ") -> str:
    body = source[source.index(start):]
    return body[:body.index(end, len(start))]


# ── it never changes the main window ─────────────────────────────────────────

def test_entering_the_orb_does_not_reflag_the_main_window():
    """setWindowFlags on the window that owns the WebGL face is the root of
    every compact-overlay fault. It must not come back."""
    for name in ("def enter_overlay_mode", "def exit_overlay_mode"):
        body = _body(WINDOW, name)
        assert "setWindowFlags" not in body, f"{name} re-flags the main window"
        assert "setWindowOpacity" not in body, (
            f"{name} makes the main window translucent (a layered window "
            "is where the WebGL page failed to start)")
        assert "_rebuild_face" not in body, f"{name} rebuilds the face"


def test_the_main_window_is_hidden_and_restored_as_it_was():
    enter = _body(WINDOW, "def enter_overlay_mode")
    assert "self.hide()" in enter
    assert "saveGeometry" in enter
    exit_body = _body(WINDOW, "def exit_overlay_mode")
    assert "restoreGeometry" in exit_body
    for how in ("showMaximized", "showFullScreen", "showNormal"):
        assert how in exit_body, f"a window left {how} does not come back so"


def test_the_orb_opens_before_the_window_hides():
    """If the orb failed to open and the window had already gone, there would
    be nothing on screen to get ORION back with."""
    enter = _body(WINDOW, "def enter_overlay_mode")
    assert enter.index("overlay.show()") < enter.index("self.hide()")
    assert "except Exception" in enter


# ── where it lands ───────────────────────────────────────────────────────────

def test_the_orb_is_placed_on_the_screen_orion_is_on():
    enter = _body(WINDOW, "def enter_overlay_mode")
    assert "self.screen()" in enter
    assert "QGuiApplication.primaryScreen()" in enter, "no fallback screen"
    assert enter.index("self.screen()") < enter.index("place_on")


def test_the_orb_still_goes_to_the_bottom_right():
    body = _body(ORB, "    def place_on")
    assert "availableGeometry()" in body
    assert "area.right()" in body and "area.bottom()" in body


def test_the_orb_window_paints_no_webengine_or_gl():
    """Nothing in it can be broken by a window change."""
    imports = [line for line in ORB.splitlines()
               if line.startswith(("import ", "from "))]
    assert not [line for line in imports
                if "WebEngine" in line or "OpenGL" in line], imports


def test_the_orb_stops_animating_while_hidden():
    assert "self.orb.timer.stop()" in _body(ORB, "    def hideEvent")


# ── which face it shows ──────────────────────────────────────────────────────

def test_the_page_says_whether_it_failed_or_is_merely_loading():
    """Three states, not two.

    Absence of a rendered frame was read as failure, so a page that only
    needed another second was condemned — and the fallback it triggered was
    permanent.
    """
    assert "__orionFailed" in FACE, (
        "the page does not record its failure, so the Qt side can only infer "
        "it from the absence of a frame")
    assert "def check_state" in FACE
    body = _body(FACE, "    def check_state", "\n    def check_alive")
    for state in ("alive", "failed", "loading"):
        assert f"'{state}'" in body or f'"{state}"' in body


def test_the_watcher_waits_rather_than_deciding_while_it_loads():
    body = _body(WINDOW, "def _watch_face_health", "\n    @staticmethod")
    assert "check_state" in body, "still using the two-state check"
    assert 'state != "failed"' in body, (
        "a page that is still loading is treated the same as one that failed")
    assert "attempts" in body, "there is no second look; one poll decides"


def test_the_fallback_still_happens_when_the_page_really_failed():
    """Patience must not become never giving up. A machine without WebGL has
    to end up with something it can draw — after one fresh retry, and as his
    ORB, never the old sculpted head the user called "an old avatar"."""
    body = _body(WINDOW, "def _watch_face_health", "\n    @staticmethod")
    assert "_face_retried" in body, "no retry before giving up"
    assert 'os.environ["ORION_FACE_FALLBACK"] = "orb"' in body
    assert "ORION_HOLO_HEAD" not in body, "the fallback is the old head again"
    assert "_swap_face()" in body


def test_a_watcher_does_not_condemn_the_face_that_replaced_its_own():
    """Every toggle starts another watcher, so several are in flight.

    Without this, one left over from a discarded face polls on, gives up, and
    falls back — replacing a face that was working perfectly well.
    """
    body = _body(WINDOW, "def _watch_face_health", "\n    @staticmethod")
    assert "watched" in body
    assert 'is not watched' in body, (
        "a stale watcher can still act on whichever face is current")


def test_polling_a_face_that_has_gone_cannot_raise():
    """The probe runs from a timer, where an exception has nothing to catch
    it, and the widget it asks may have been destroyed in between."""
    body = _body(WINDOW, "    def _probe_face", "\n    def ")
    assert "except Exception" in body
