"""Double-clap wake detection is opt-in and does not open another microphone."""

from array import array

from orion_core.clap_wake import DoubleClapDetector


QUIET = bytes(512 * 2)
CLAP = array("h", [25000] * 10 + [0] * 502).tobytes()
SPEECH_LIKE = array("h", [5000] * 512).tobytes()


def test_two_short_peaks_separated_by_quiet_wake_once():
    detector = DoubleClapDetector()
    assert not detector.feed(QUIET, 0.0)
    assert not detector.feed(QUIET, 0.08)
    assert not detector.feed(CLAP, 0.10)
    assert not detector.feed(QUIET, 0.13)
    assert not detector.feed(QUIET, 0.22)
    assert detector.feed(CLAP, 0.36)
    assert not detector.feed(CLAP, 0.40)


def test_sustained_sound_and_single_clap_do_not_wake():
    detector = DoubleClapDetector()
    detector.feed(QUIET, 0.0)
    detector.feed(QUIET, 0.10)
    assert not detector.feed(SPEECH_LIKE, 0.12)
    detector.feed(QUIET, 0.20)
    detector.feed(QUIET, 0.30)
    assert not detector.feed(CLAP, 0.32)
    assert not detector.feed(QUIET, 0.36)


def test_reset_drops_half_finished_wake():
    detector = DoubleClapDetector()
    detector.feed(QUIET, 0.0)
    detector.feed(QUIET, 0.10)
    detector.feed(CLAP, 0.12)
    detector.reset()
    detector.feed(QUIET, 0.25)
    detector.feed(QUIET, 0.33)
    assert not detector.feed(CLAP, 0.36)
