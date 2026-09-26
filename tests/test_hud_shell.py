"""
The shell around ORION's face.

The Core Window was a header and a face. Everything else — the log, the
telemetry, the content, somewhere to type — lived on the Command Deck, a
second window, and a glance at the assistant told you nothing about what it
was doing.

The constraint these tests exist to hold is that the HUD is chrome and nothing
more. It must not build, wrap, restyle or reparent the face; it must not
become a second source of truth about ORION; and it must not cost paint time
on the loop that also carries audio. A meter redrawn at the same width is pure
cost, and a rail that spawns a probe to fill itself is how the last telemetry
regression happened.

Rendered in a subprocess because a QApplication built inline hangs the suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
app = QApplication([])
from orion_core.constants import C
from orion_core.gui.hud_shell import (
    COLUMN_WIDTH, LOG_LINES, RAIL_WIDTH, ActivityLog, CommandBar,
    ControlsPanel, DropZone, HudShell, Meter, StatusChips, SysMonitorRail,
    hud_stylesheet, mono)

out = {{}}

# ── the face is placed, never touched ────────────────────────────────────────
face = QLabel("FACE")
before = (face.styleSheet(), face.objectName())
shell = HudShell(face)
out["face_is_the_same_object"] = shell.centre is face
out["face_stylesheet_untouched"] = face.styleSheet() == before[0]
out["face_objectname_untouched"] = face.objectName() == before[1]

# ── a styled background is actually painted ──────────────────────────────────
from PyQt6.QtCore import Qt as _Qt
_styled = _Qt.WidgetAttribute.WA_StyledBackground
out["panels_paint_their_background"] = {{
    "log": shell.log.testAttribute(_styled),
    "command": shell.command.testAttribute(_styled),
}}
out["shell_paints_over_the_face"] = shell.testAttribute(_styled)

# ── meters ───────────────────────────────────────────────────────────────────
meter = Meter("CPU")
meter.resize(120, 30)
out["meter_starts_unknown"] = meter.value_label.text()
meter.set_value(42.0)
out["meter_shows_percent"] = meter.value_label.text()
meter.set_value(None)
out["meter_unknown_again"] = meter.value_label.text()

# Repainting at the same value is pure cost on the loop that carries audio.
meter.set_value(50.0)
width_a = meter.fill.width()
meter.set_value(50.2)
out["tiny_change_is_ignored"] = meter.fill.width() == width_a
meter.set_value(80.0)
out["real_change_moves_it"] = meter.fill.width() != width_a

# The bar earns its colour.
meter.set_value(10.0); out["colour_low"] = meter.fill.styleSheet()
meter.set_value(75.0); out["colour_busy"] = meter.fill.styleSheet()
meter.set_value(95.0); out["colour_full"] = meter.fill.styleSheet()
out["PRI"], out["GOOD"], out["WARN"], out["BAD"] = C.PRI, C.GOOD, C.WARN, C.BAD

# ── the rail ─────────────────────────────────────────────────────────────────
rail = SysMonitorRail()
rail.apply_sample({{"cpu": 24.0, "ram": 94.0, "net_bps": 1024.0,
                   "net_percent": 3.0}})
out["cpu"] = rail.meters["cpu"].value_label.text()
out["mem"] = rail.meters["ram"].value_label.text()
out["net"] = rail.meters["net"].value_label.text()
out["gpu_absent"] = rail.meters["gpu"].value_label.text()
out["tmp_absent"] = rail.meters["tmp"].value_label.text()
rail.apply_sample({{"cpu": 5.0, "gpu": 61.0, "temperature": 48.0}})
out["gpu_when_known"] = rail.meters["gpu"].value_label.text()
out["tmp_when_known"] = rail.meters["tmp"].value_label.text()
rail.apply_sample("not a sample")
rail.apply_sample(None)
out["rubbish_survived"] = True
rail.set_facts("07:14", 427, "WIN")
out["facts"] = rail.facts.text()
out["rail_width"] = rail.width()

# net rate formatting
rail.apply_sample({{"net_bps": 5_400_000.0, "net_percent": 40.0}})
out["net_fast"] = rail.meters["net"].value_label.text()
rail.apply_sample({{"net_bps": 12.0, "net_percent": 0.0}})
out["net_slow"] = rail.meters["net"].value_label.text()

# ── the log ──────────────────────────────────────────────────────────────────
log = ActivityLog()
out["log_starts_empty"] = log.line_count
for i in range(LOG_LINES + 50):
    log.append(f"line {{i}}")
out["log_capped"] = log.line_count <= LOG_LINES
log.clear()
log.append("")
log.append("   ")
log.append(None)
out["blank_lines_ignored"] = log.line_count

# ── the command bar ──────────────────────────────────────────────────────────
bar = CommandBar()
sent = []
bar.submitted.connect(sent.append)
bar.input.setText("  what is the weather  ")
bar._submit()
out["submitted"] = sent
out["input_cleared"] = bar.input.text()
bar.input.setText("   ")
bar._submit()
out["blank_not_submitted"] = len(sent)

out["mic_starts_on"] = bar.mic_on
toggles = []
bar.mic_toggled.connect(toggles.append)
bar._toggle_mic()
out["mic_after_toggle"] = bar.mic_on
out["mic_emitted"] = toggles
out["mic_label_muted"] = bar.mic_btn.text()
# set_mic must NOT re-emit — it is how the window follows bus.mic_enabled,
# and a loop there would fight the gate.
bar.set_mic(True)
out["set_mic_did_not_emit"] = len(toggles)
out["mic_label_active"] = bar.mic_btn.text()

interrupts = []
bar.interrupted.connect(lambda: interrupts.append(1))
bar.interrupt_btn.click()
out["interrupt_fires"] = len(interrupts)

# ── controls ─────────────────────────────────────────────────────────────────
controls = ControlsPanel()
fired = []
controls.add("autostart", "AUTO-START: OFF", lambda: fired.append("a"))
controls.add("memory", "MEMORY", lambda: fired.append("m"))
out["control_keys"] = controls.keys
controls.button("autostart").click()
out["control_fired"] = fired
controls.set_state("autostart", "AUTO-START: ON")
out["control_relabelled"] = controls.button("autostart").text()
controls.set_state("nonexistent", "nothing")
out["unknown_key_survived"] = True

# ── accessibility ────────────────────────────────────────────────────────────
missing = []
for widget in shell.findChildren(QWidget):
    from PyQt6.QtWidgets import QPushButton
    if isinstance(widget, QPushButton) and not widget.accessibleName():
        missing.append(widget.text() or widget.objectName())
out["buttons_without_a_name"] = missing

# ── styling ──────────────────────────────────────────────────────────────────
sheet = hud_stylesheet()
out["sheet_uses_crimson"] = C.PRI in sheet
out["sheet_has_no_cyan"] = "#39b6ff" not in sheet and "#00e5ff" not in sheet
themed = hud_stylesheet("#00aa55")
out["sheet_recolours"] = "#00aa55" in themed and C.PRI not in themed

# ── the content strip goes under the centre, not over it ─────────────────────
strip = QLabel("results")
shell.add_below_centre(strip)
out["strip_below_centre"] = shell.centre_stack.indexOf(strip) > \
    shell.centre_stack.indexOf(face)
out["column_width"] = shell.right_column.width()

print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def hud() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(root=str(ROOT).replace("\\", "/"))],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced nothing:\n{proc.stdout[-2000:]}")


# ── the face is not this module's business ────────────────────────────────────

def test_the_face_is_placed_not_wrapped(hud):
    """The load-bearing constraint.

    The face has been rebuilt enough times to have earned being left alone.
    The HUD takes a centre widget and puts it in the middle; what that widget
    is, and how it paints, is entirely the caller's business.
    """
    assert hud["face_is_the_same_object"] is True
    assert hud["face_stylesheet_untouched"] is True
    assert hud["face_objectname_untouched"] is True


# ── meters ────────────────────────────────────────────────────────────────────

def test_an_unknown_reading_says_so(hud):
    """A machine with no discrete GPU reporting "0%" reads as an idle GPU,
    which is a different and wrong claim."""
    assert hud["meter_starts_unknown"] == "N/A"
    assert hud["meter_unknown_again"] == "N/A"
    assert hud["gpu_absent"] == "N/A"
    assert hud["tmp_absent"] == "N/A"


def test_a_known_reading_is_shown(hud):
    assert hud["meter_shows_percent"] == "42%"
    assert hud["cpu"] == "24%"
    assert hud["mem"] == "94%"
    assert hud["gpu_when_known"] == "61%"
    assert "48" in hud["tmp_when_known"]


def test_a_bar_is_not_redrawn_at_the_same_width(hud):
    """The Qt thread is the asyncio event loop, so paint time here is stolen
    directly from audio."""
    assert hud["tiny_change_is_ignored"] is True
    assert hud["real_change_moves_it"] is True


def test_the_bar_earns_its_colour(hud):
    """Three distinguishable steps, escalating — and the accent reserved for
    the top one.

    A nominal reading used to be painted in the accent, so four crimson bars
    sat at 20% all day and taught the eye that crimson means "a number". The
    assertion that matters is not which token each step uses but that they
    differ and that only saturation gets the alarm colour.
    """
    low, busy, full = (hud["colour_low"], hud["colour_busy"], hud["colour_full"])
    assert len({low, busy, full}) == 3, "two loads look identical"
    assert hud["BAD"] in full, "a saturated resource must get the alarm colour"
    assert hud["BAD"] not in low and hud["BAD"] not in busy, (
        "the alarm colour is being spent on a healthy reading")
    assert hud["GOOD"] in low, "nominal should recede, not shout"


def test_a_rate_is_readable_at_a_glance(hud):
    assert hud["net"] == "1KB/s"
    assert hud["net_fast"] == "5.4MB/s"
    assert hud["net_slow"] == "12B/s"


def test_a_malformed_sample_does_not_raise(hud):
    """Telemetry arrives over a signal from another subsystem; a widget that
    dies on a bad payload takes the window with it."""
    assert hud["rubbish_survived"] is True


def test_the_machine_facts_are_shown(hud):
    assert "07:14" in hud["facts"] and "427" in hud["facts"]
    assert "WIN" in hud["facts"]


def test_the_rail_keeps_its_width(hud):
    assert hud["rail_width"] > 0


# ── the log ───────────────────────────────────────────────────────────────────

def test_the_log_starts_empty(hud):
    assert hud["log_starts_empty"] == 0


def test_the_log_is_capped(hud):
    """A view of the last few minutes. The journals on disk are the record,
    and an unbounded document in a widget that repaints on the event loop is
    how a long session starts to stutter."""
    assert hud["log_capped"] is True


def test_blank_lines_are_not_logged(hud):
    assert hud["blank_lines_ignored"] == 0


# ── the command bar ───────────────────────────────────────────────────────────

def test_typing_sends_what_was_typed(hud):
    assert hud["submitted"] == ["what is the weather"]
    assert hud["input_cleared"] == ""


def test_an_empty_line_is_not_sent(hud):
    assert hud["blank_not_submitted"] == 1


def test_the_interrupt_button_fires(hud):
    """It was a keyboard shortcut with no visible affordance at all."""
    assert hud["interrupt_fires"] == 1


def test_the_microphone_button_toggles_and_says_which(hud):
    assert hud["mic_starts_on"] is True
    assert hud["mic_after_toggle"] is False
    assert hud["mic_emitted"] == [False]
    assert "MUTED" in hud["mic_label_muted"]
    assert "ACTIVE" in hud["mic_label_active"]


def test_following_the_real_mic_state_does_not_re_emit(hud):
    """set_mic is how the window follows bus.mic_enabled. The mic is also
    turned off by voice and by the gate, and a button that re-emitted on
    every update would fight them."""
    assert hud["set_mic_did_not_emit"] == 1


# ── controls ──────────────────────────────────────────────────────────────────

def test_a_control_runs_what_it_says(hud):
    assert hud["control_keys"] == ["autostart", "memory"]
    assert hud["control_fired"] == ["a"]


def test_a_toggle_says_what_it_currently_is(hud):
    """"AUTO-START: OFF" rather than a checkbox whose meaning depends on
    which way round you read it."""
    assert "AUTO-START: ON" in hud["control_relabelled"]


def test_relabelling_something_that_is_not_there_is_harmless(hud):
    assert hud["unknown_key_survived"] is True


# ── accessibility ─────────────────────────────────────────────────────────────

def test_every_button_has_an_accessible_name(hud):
    """ORION has shipped twenty-one symbol-only buttons that a screen reader
    read out as punctuation."""
    assert hud["buttons_without_a_name"] == [], (
        f"unnamed: {hud['buttons_without_a_name']}")


# ── styling ───────────────────────────────────────────────────────────────────

def test_the_hud_is_crimson_not_cyan(hud):
    assert hud["sheet_uses_crimson"] is True
    assert hud["sheet_has_no_cyan"] is True


def test_a_panel_with_a_background_actually_paints_one(hud):
    """Qt honours a stylesheet ``background`` on a plain QWidget only when
    WA_StyledBackground is set; QFrame paints one either way.

    So the meters and the drop zone (QFrame) had surfaces while the activity
    log and the command bar (QWidget) sat directly on the void — the rule was
    written, applied, and silently ignored. Nothing raised, nothing logged;
    it was only visible by rendering the HUD and looking at it.
    """
    painted = hud["panels_paint_their_background"]
    unpainted = [name for name, ok in painted.items() if not ok]
    assert not unpainted, (
        f"these panels are styled with a background that Qt will not paint: "
        f"{unpainted}")


def test_the_shell_does_not_paint_over_the_face(hud):
    """The panels need WA_StyledBackground; the SHELL must not have it.

    HudShell wraps the centre widget, which is the face — a native surface the
    compositor draws itself. Giving the shell a styled background puts a
    Qt-painted fill across that child's area, and the symptom is an
    intermittent black frame: the Mark XXIII black flicker by another route.

    Nothing is lost: the shell's background is the window's background, the
    same C.BG, so the parent showing through is what the fill would have been.
    """
    assert hud["shell_paints_over_the_face"] is False


def test_the_hud_can_be_recoloured(hud):
    """Live theming recolours everything else by substitution; the HUD takes
    its colours as arguments so it can be recoloured the same way."""
    assert hud["sheet_recolours"] is True


# ── composition ───────────────────────────────────────────────────────────────

def test_results_go_under_the_face_not_over_it(hud):
    assert hud["strip_below_centre"] is True


def test_the_right_column_keeps_its_width(hud):
    assert hud["column_width"] > 0


# ── wiring ────────────────────────────────────────────────────────────────────

def _window_source() -> str:
    return (ROOT / "orion_core" / "gui" / "core_window.py").read_text(
        encoding="utf-8")


def test_the_hud_is_fed_by_signals_that_already_existed():
    """Not a new source of truth — a new view of one."""
    source = _window_source()
    for signal in ("telemetry_sample.connect(self.hud.rail.apply_sample)",
                   "log.connect(self.hud.log.append)",
                   "mic_enabled.connect(self.hud.command.set_mic)"):
        assert signal in source, f"not wired: {signal}"


def test_the_face_is_still_the_centre_widget():
    source = _window_source()
    assert "HudShell(self.content_splitter)" in source, (
        "the face's splitter should be passed through as the centre")


def test_the_overlay_orb_is_still_alone():
    """Compact overlay is the orb on its own. Mark XXXI: it is a separate
    window (gui/orb_overlay.py) and the HUD is hidden whole behind it, so no
    rail, chip or log column can be left showing around it."""
    source = _window_source()
    body = source[source.index("def enter_overlay_mode"):
                  source.index("def exit_overlay_mode")]
    assert "_ensure_orb_overlay()" in body
    assert "self.hide()" in body


def test_the_face_wears_orions_colours():
    """Every renderer shipped with its own default — the sculpted head's is a
    cyan — replaced only if the user opened Preferences and picked an accent.
    ORION's identity was crimson everywhere except the thing you look at."""
    source = _window_source()
    assert "_wear_orion_colours" in source
    assert source.count("_wear_orion_colours(") >= 5, (
        "every face construction path must wear them")


def test_that_failure_is_reported_rather_than_swallowed():
    """It was swallowed, and the face stayed cyan with nothing saying why."""
    source = _window_source()
    start = source.index("def _wear_orion_colours")
    body = source[start:start + 2000]
    assert "say(" in body
    assert "except Exception:\n            pass" not in body


def test_the_controls_have_somewhere_to_be_opened_from():
    source = _window_source()
    assert "def open_controls" in source
    assert "self.controls_btn" in source
