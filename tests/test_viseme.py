"""
Viseme pipeline (Mark XXVI, Phase 3) — pure mouth-shape mapping and scheduling.

No Qt, no audio: the maps, the grapheme heuristic, the SAPI id translation and the
timeline scheduler are all deterministic data + arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.viseme import (  # noqa: E402
    PHONEME_TO_VISEME,
    SAPI_VISEME_MAP,
    VISEMES,
    VisemeScheduler,
    sapi_viseme_to_id,
    text_to_visemes,
)


# ── the viseme set ────────────────────────────────────────────────────────────

def test_every_viseme_posture_is_normalised():
    for name, m in VISEMES.items():
        assert 0.0 <= m.open <= 1.0, name
        assert 0.0 <= m.width <= 1.0, name
        assert 0.0 <= m.round <= 1.0, name


def test_the_extremes_are_where_they_should_be():
    assert VISEMES["PP"].open == 0.0                 # p/b/m — closed
    assert VISEMES["aa"].open == max(v.open for v in VISEMES.values())  # widest open
    assert VISEMES["ou"].round == max(v.round for v in VISEMES.values())  # most pursed
    assert VISEMES["ih"].width == max(v.width for v in VISEMES.values())  # most spread


# ── SAPI bridge ───────────────────────────────────────────────────────────────

def test_sapi_ids_map_onto_our_set():
    assert sapi_viseme_to_id(0) == "sil"
    assert sapi_viseme_to_id(21) == "PP"      # p/b/m
    assert sapi_viseme_to_id(1) == "aa"
    assert all(v in VISEMES for v in SAPI_VISEME_MAP.values())


def test_an_unknown_sapi_id_rests_the_mouth():
    assert sapi_viseme_to_id(99) == "sil"
    assert sapi_viseme_to_id(-1) == "sil"


# ── phoneme map ───────────────────────────────────────────────────────────────

def test_phonemes_map_to_real_visemes():
    for phoneme, vis in PHONEME_TO_VISEME.items():
        assert vis in VISEMES, phoneme
    assert PHONEME_TO_VISEME["M"] == "PP"
    assert PHONEME_TO_VISEME["UW"] == "ou"


# ── grapheme heuristic ────────────────────────────────────────────────────────

def test_text_to_visemes_is_deterministic_and_plausible():
    seq = [v for v, _d in text_to_visemes("mama")]
    assert seq == ["PP", "aa", "PP", "aa", "sil"]


def test_a_run_of_the_same_posture_collapses():
    # "hello" -> h e l l o : the double-l is one DD posture, not two.
    seq = [v for v, _d in text_to_visemes("hello")]
    assert seq == ["aa", "E", "DD", "oh", "sil"]


def test_digraphs_resolve_before_single_letters():
    assert [v for v, _d in text_to_visemes("the")] == ["TH", "E", "sil"]
    assert [v for v, _d in text_to_visemes("she")] == ["CH", "E", "sil"]


def test_vowels_are_held_longer_than_consonants():
    durs = dict((v, d) for v, d in text_to_visemes("ba"))
    assert durs["aa"] > durs["PP"]


def test_empty_text_is_an_empty_timeline():
    assert text_to_visemes("") == []
    assert text_to_visemes("   ...  ") == []


# ── scheduler ─────────────────────────────────────────────────────────────────

def test_scheduler_walks_the_timeline_against_the_clock():
    sched = VisemeScheduler([("PP", 0.1), ("aa", 0.2), ("sil", 0.1)])
    assert sched.at(0.05) == "PP"
    assert sched.at(0.15) == "aa"
    assert sched.at(0.35) == "sil"
    assert sched.total == pytest.approx(0.4)


def test_scheduler_past_the_end_rests_and_reports_done():
    sched = VisemeScheduler([("aa", 0.1)])
    assert sched.at(5.0) == "sil"
    assert sched.done(5.0) is True
    assert sched.done(0.05) is False


def test_an_empty_schedule_is_silent():
    sched = VisemeScheduler([])
    assert sched.at(0.0) == "sil"
    assert sched.done(0.0) is True


# ── the 3-D face bridge (viseme → existing spectral mouth) ────────────────────

def test_viseme_to_spectral_maps_openness_to_amplitude():
    from orion_core.viseme import viseme_to_spectral
    amp_open, *_ = viseme_to_spectral("aa", 1.0)      # widest open
    amp_shut, *_ = viseme_to_spectral("PP", 1.0)      # lips closed
    assert amp_open > 0.9
    assert amp_shut < 0.1


def test_viseme_to_spectral_maps_spread_and_round_to_bands():
    from orion_core.viseme import viseme_to_spectral
    _a, low_i, _m, high_i = viseme_to_spectral("ih", 1.0)   # very spread
    _a, low_o, _m, high_o = viseme_to_spectral("ou", 1.0)   # very round
    assert high_i > low_i, "a spread vowel should read as brighter (high band)"
    assert low_o > high_o, "a round vowel should read as low band"


def test_viseme_to_spectral_scales_with_weight():
    from orion_core.viseme import viseme_to_spectral
    full, *_ = viseme_to_spectral("aa", 1.0)
    half, *_ = viseme_to_spectral("aa", 0.5)
    assert half == pytest.approx(full * 0.5, abs=0.05)


def test_the_3d_face_accepts_visemes_via_the_spectral_bridge():
    # Source-checked (importing face3d pulls in QtWebEngine and can corrupt Qt
    # for later tests): the 3-D face must expose set_viseme and route it through
    # the existing, vetted amplitude/spectrum bridge — no renderer edit.
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "gui"
           / "face3d.py").read_text(encoding="utf-8")
    assert "def set_viseme" in src
    assert "viseme_to_spectral" in src
    assert "self.set_amplitude" in src and "self.set_spectrum" in src
