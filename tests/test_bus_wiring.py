"""
No signal may be listened for and never sent.

Why this exists
---------------
``bus.viseme`` was declared in bus.py, subscribed to by the Core Window, and
backed by a complete scheduler in orion_core/viseme.py — and nothing in ORION
ever emitted one. Every face in the app ran on a loudness meter for an entire
release while a finished lip-sync module sat inert beside it.

That failure is invisible from every angle that normally catches things. The
module's own unit tests passed, because the module worked. The GUI worked,
because a signal with no sender simply never fires. Nothing logged, nothing
raised, nothing went red. The feature just quietly did not exist.

So the direction that matters is asserted here: a signal with listeners must
have a producer somewhere. The opposite case — produced with nobody listening —
is wasteful but harmless, so it is reported as a known list rather than a
failure, and a NEW one has to be added deliberately.

Reading the code rather than running it
---------------------------------------
This is static analysis, and ORION wires several signals dynamically:

    self.scan_requested.connect(request_signal.emit)     # emit as a callback
    signal = getattr(bus, "electronics_scan_result")     # looked up by name

An earlier version of this audit missed both forms and produced four confident
false positives, including one for a signal that was wired perfectly. So the
detection deliberately over-approximates: a signal counts as wired if its name
appears anywhere as a quoted string. Better to miss a dead signal than to send
somebody hunting a bug that is not there.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".venv", "dist", "build", "release", "android", ".git",
              "__pycache__", "node_modules"}

#: Signals that are emitted with no subscriber anywhere. Wasteful, not broken.
#: Each is here because it was checked, not because it was convenient.
#:
#:   browser_step        the co-pilot narrates to bus.log, which IS consumed;
#:                       this is the structured twin, kept as the seam a GUI or
#:                       phone mirror would attach to.
#:   sentiment_changed   a simplified (str, float) twin of sentiment_payload,
#:                       which is consumed. Redundant rather than missing.
#:   speaker_identified  emitted by the GENDER tracker (voice_gender), which
#:                       has no subscriber: the male/female guess is read on
#:                       demand through the speaker_id tool rather than pushed.
#:                       The live-path gap this note used to describe — the
#:                       trained voiceprint never being compared during a Live
#:                       session — is closed; see orion_core/live_presence.py.
KNOWN_UNCONSUMED = {"browser_step", "sentiment_changed", "speaker_identified"}


def _sources() -> dict[Path, str]:
    out = {}
    for directory, subdirs, files in os.walk(ROOT):
        # Prune relative to the checkout: an ancestor named "release" or
        # "tests" must not hide all production code in an exported checkout.
        subdirs[:] = [name for name in subdirs if name not in SKIP_PARTS]
        for name in files:
            if not name.endswith(".py") or name == "bus.py":
                continue
            path = Path(directory) / name
            try:
                out[path.relative_to(ROOT)] = path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
    return out


@pytest.fixture(scope="module")
def wiring():
    bus_src = (ROOT / "orion_core" / "bus.py").read_text(encoding="utf-8")
    signals = sorted(set(re.findall(r"^\s*(\w+)\s*=\s*pyqtSignal", bus_src, re.M)))
    assert signals, "no signals found — has bus.py moved?"

    sources = _sources()
    production = {p: t for p, t in sources.items() if "tests" not in p.parts}
    everything = "\n".join(sources.values())
    produced_text = "\n".join(production.values())

    report = {}
    for name in signals:
        # Emitted directly, or handed round as a bound callback with no call.
        emits = len(re.findall(rf"\.{name}\.emit\s*\(", produced_text))
        refs = len(re.findall(rf"\.{name}\.emit\b(?!\s*\()", produced_text))
        # Reached by name: getattr(bus, "x") / emit("x", ...) / a name table.
        dynamic = len(re.findall(rf"[\"']{name}[\"']", produced_text))
        listens = len(re.findall(rf"\.{name}\.connect\s*\(", everything))
        # A quoted name counts as a SUBSCRIPTION only in a file that also
        # connects something. Counting every quoted mention made this test
        # circular: the KNOWN_UNCONSUMED list below names all three, which read
        # back as proof that they were consumed.
        listens += sum(
            len(re.findall(rf"[\"']{name}[\"']", text))
            for path, text in sources.items()
            if ".connect(" in text and "tests" not in path.parts
        )
        report[name] = {"produced": emits + refs + dynamic, "consumed": listens}
    return report


def test_the_bus_still_declares_signals(wiring):
    assert len(wiring) > 20, f"only {len(wiring)} signals found"


def test_no_signal_is_listened_for_but_never_sent(wiring):
    """THE regression. This is the shape bus.viseme had for a whole release."""
    dead = sorted(name for name, counts in wiring.items()
                  if counts["consumed"] > 0 and counts["produced"] == 0)
    assert not dead, (
        "these signals have subscribers but nothing ever emits them, so the "
        "features behind them silently do not work: " + ", ".join(dead)
    )


def test_unconsumed_signals_are_a_known_and_deliberate_list(wiring):
    """Emitting into the void is harmless, but it should be a decision.

    If this fails with a NEW name, either wire it up or add it above with a
    sentence saying why it is emitted for nobody.
    """
    unconsumed = {name for name, counts in wiring.items()
                  if counts["produced"] > 0 and counts["consumed"] == 0}
    surprises = unconsumed - KNOWN_UNCONSUMED
    assert not surprises, (
        "new signals are emitted with no subscriber: " + ", ".join(sorted(surprises))
    )


def test_the_known_list_does_not_rot(wiring):
    """If one of these gets wired up, remove it from the list rather than
    leaving a stale excuse behind."""
    stale = {name for name in KNOWN_UNCONSUMED
             if name in wiring and wiring[name]["consumed"] > 0}
    assert not stale, (
        "these are now consumed and should leave KNOWN_UNCONSUMED: "
        + ", ".join(sorted(stale))
    )


def test_the_lip_sync_signal_specifically_has_a_producer(wiring):
    """Named on its own because it is the one that was actually broken."""
    assert wiring["viseme"]["produced"] > 0, "bus.viseme has lost its producer again"
    assert wiring["viseme"]["consumed"] > 0
