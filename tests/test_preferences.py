"""
Preferences — the settings that existed but could not be reached.

ORION's audio layer could already list the short, deduplicated set of real
endpoints and MEASURE whether each one actually carries sound. None of it was
exposed anywhere, so "ORION can't hear me" still had no answer in the
interface. A capability with no way to reach it is not a capability.

What these tests hold onto:

  * the pickers store a device by NAME. An index moves the moment anything is
    plugged in, so a saved index silently points at different hardware later.
  * nothing slow runs on the Qt thread. Under qasync that thread IS the event
    loop, and the transport probe deliberately takes about a second because it
    is timing real audio.
  * the dialog only collects; the caller decides what to persist. That is what
    keeps it testable without a running assistant.

The dialog itself is exercised in a subprocess — a QApplication inside this
suite has been observed to break later Qt tests.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_PROBE = r'''
import json, os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.gui.preferences import PreferencesDialog, SYSTEM_DEFAULT

dialog = PreferencesDialog(assistant_name="Orion", user_name="")
dialog.resize(600, 520)
dialog.show()
for _ in range(50):
    app.processEvents()
    time.sleep(0.04)

def options(row):
    return [row.combo.itemText(i) for i in range(row.combo.count())]

out = {{
    "mic_options": options(dialog.microphone),
    "spk_options": options(dialog.speakers),
    "mic_first": dialog.microphone.combo.itemText(0),
    "mic_data_first": dialog.microphone.combo.itemData(0),
    "values": dialog.values(),
    "default_label": SYSTEM_DEFAULT,
    "mic_accessible": dialog.microphone.combo.accessibleName(),
    "test_accessible": dialog.microphone.test_button.accessibleName(),
    "swatch_accessible": dialog.swatch.accessibleName(),
}}
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def dialog() -> dict:
    proc = subprocess.run([sys.executable, "-c", _PROBE.format(
        root=str(ROOT).replace("\\", "/"))], cwd=ROOT,
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced nothing:\n{proc.stdout[-1500:]}")


# ── the pickers ──────────────────────────────────────────────────────────────

def test_system_default_is_always_offered_first(dialog):
    """An untouched install and a deliberate default must be the same thing."""
    assert dialog["mic_first"] == dialog["default_label"]
    assert dialog["mic_data_first"] == ""


def test_both_directions_are_offered(dialog):
    assert len(dialog["mic_options"]) >= 1
    assert len(dialog["spk_options"]) >= 1


def test_device_names_are_fit_to_read(dialog):
    """Windows hands back unresolved driver resource strings for some
    endpoints, chiefly Bluetooth. An ordinary headset arrives with a driver
    path, a resource index and a literal carriage return embedded in its
    name, and seven of the thirty-nine names on the development machine
    looked like that.

    A list of driver paths is not a list anyone can choose from, which is
    the very problem the deduplication exists to solve.
    """
    for label in dialog["mic_options"] + dialog["spk_options"]:
        assert "@System32" not in label, f"raw driver string: {label}"
        assert ".sys" not in label, f"raw driver string: {label}"
        assert chr(13) not in label, f"carriage return in entry: {label!r}"
        assert chr(10) not in label, f"newline in entry: {label!r}"


def test_a_bidirectional_device_may_appear_in_both_lists(dialog):
    """Deliberately NOT asserted as a conflict. A Bluetooth hands-free headset
    genuinely is both a microphone and a pair of speakers, so the same name
    appearing in both directions is correct — an earlier version of this test
    called that a bug and was simply wrong about the hardware."""
    mics = set(dialog["mic_options"]) - {dialog["default_label"]}
    speakers = set(dialog["spk_options"]) - {dialog["default_label"]}
    assert isinstance(mics & speakers, set)


def test_devices_are_identified_by_name(dialog):
    """Stored by name, never by index — indices shift whenever hardware moves."""
    values = dialog["values"]
    assert isinstance(values["input_device"], str)
    assert isinstance(values["output_device"], str)


def test_the_dialog_only_collects(dialog):
    """It returns values; persisting is the caller's decision. That is what
    lets it be tested without an assistant behind it."""
    assert set(dialog["values"]) == {
        "input_device", "output_device", "assistant_name", "user_name", "accent"}


def test_the_accent_defaults_to_orions_own_crimson(dialog):
    from orion_core.gui.style import C

    assert dialog["values"]["accent"].lower() == C.PRI.lower()


# ── accessibility ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", ["mic_accessible", "test_accessible",
                                 "swatch_accessible"])
def test_controls_carry_an_accessible_name(dialog, key):
    """An audit here once found 21 symbol-only buttons with no accessible name
    at all; a screen reader announced the glyph. New controls do not get to
    reintroduce that."""
    assert dialog[key].strip(), f"{key} has no accessible name"


# ── threading and wiring, read from the source ───────────────────────────────

def _preferences_source() -> str:
    return (ROOT / "orion_core" / "gui" / "preferences.py").read_text(encoding="utf-8")


def test_enumeration_and_probing_leave_the_qt_thread():
    """query_devices() is hundreds of milliseconds and the transport probe is
    about a second. On the qasync thread either would stall the audio callback
    and the Live socket, not merely the repaint."""
    source = _preferences_source()
    assert source.count("threading.Thread(") >= 2, (
        "device work appears to run on the GUI thread"
    )
    for call in ("usable_devices(", "transport_works("):
        for match in re.finditer(re.escape(call), source):
            window = source[max(0, match.start() - 700):match.start()]
            assert "def work()" in window, (
                f"{call} is not inside an off-thread worker"
            )


def test_results_come_back_over_signals():
    source = _preferences_source()
    assert "pyqtSignal" in source
    assert ".emit(" in source


def test_the_test_button_measures_rather_than_opens():
    """Opening a stream proves nothing: DirectSound output opens, accepts every
    write instantly, and plays silence."""
    source = _preferences_source()
    assert "transport_works" in source
    assert "use_cache=False" in source, (
        "a cached verdict would make the Test button lie after a device change"
    )


def test_the_dialog_is_reachable_from_the_window():
    """A dialog nothing opens is the same as no dialog."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert "def open_preferences" in source
    assert '"preferences", "settings", "options"' in source, (
        "ORION cannot open his own settings by voice"
    )


def test_applying_goes_through_the_existing_verify_path():
    """set_device runs the verify-and-narrate ladder; writing the config
    directly would skip it and lose the fallback message."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert "audio_devices.set_device" in source
