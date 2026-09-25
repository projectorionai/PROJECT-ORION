"""
Tests for the consolidated design system tokens (Mark XX design-spec §9,
High Impact): a single status-colour triad, spacing scale, type scale, and
motion vocabulary, replacing what the audit found fragmented across nine
files — three independently-defined OK/DEGRADED/DOWN triads, 83 hand-picked
setContentsMargins/setSpacing literals, no named type scale, and two
different "typewriter" speeds (24ms/22ms) tuning the same visual effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.constants import C, HEALTH_COLOURS, HEALTH_GLYPHS, MOTION, SPACE, TYPE


def test_health_colours_cover_all_three_statuses_and_reuse_c():
    assert set(HEALTH_COLOURS) == {"OK", "DEGRADED", "DOWN"}
    assert HEALTH_COLOURS["OK"] == C.GOOD
    assert HEALTH_COLOURS["DEGRADED"] == C.WARN
    assert HEALTH_COLOURS["DOWN"] == C.BAD


def test_health_glyphs_cover_all_three_statuses():
    assert set(HEALTH_GLYPHS) == {"OK", "DEGRADED", "DOWN"}
    assert all(isinstance(v, str) and v for v in HEALTH_GLYPHS.values())


def test_spacing_scale_is_a_strictly_increasing_four_step_scale():
    values = [SPACE.XS, SPACE.SM, SPACE.MD, SPACE.LG]
    assert values == sorted(values)
    assert len(set(values)) == 4
    assert values == [4, 8, 16, 24]


def test_type_scale_defines_all_four_roles():
    assert TYPE.TITLE_PX > TYPE.HEADING_PX >= TYPE.BODY_PX
    assert "Cascadia" in TYPE.MONO_FAMILY


def test_motion_stream_unifies_the_two_former_typewriter_speeds():
    # Was 24ms in ops_deck.py and 22ms in command_centre.py for the same
    # thought-typewriter effect — one named value now.
    assert MOTION.STREAM_MS == 24


def test_motion_refresh_tiers_are_staggered_by_volatility():
    assert MOTION.REFRESH_FAST_MS < MOTION.REFRESH_MED_MS < MOTION.REFRESH_SLOW_MS
    assert MOTION.REFRESH_FAST_MS == 2_000    # mission_deck.py's existing cadence


def test_motion_page_matches_the_existing_deck_crossfade_duration():
    assert MOTION.PAGE_MS == 220   # unified_dashboard.py's existing cross-fade
