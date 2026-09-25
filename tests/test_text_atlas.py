"""
Tests for the GPU text system (Mark XXIII).

Everything that decides WHERE glyphs go is pure and tested here. Only
rasterisation touches Qt, and its output is checked for structure rather than
appearance — glyph rendering under the offscreen QPA platform is not
representative of a real display (see test_the_distance_field_has_an_inside
for the specific limitation).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.swarm.render.text_atlas import (  # noqa: E402
    DEFAULT_CHARSET,
    GLYPH_FLOATS,
    grid_for,
    layout_many,
    layout_string,
    measure,
    rasterise_atlas,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def atlas(_app):
    return rasterise_atlas(font_size=32)


# ── atlas packing (pure) ─────────────────────────────────────────────────────

def test_the_grid_is_power_of_two():
    """Some drivers still handle NPOT textures poorly with mipmaps."""
    width, height, _columns = grid_for(93, 96)
    assert width & (width - 1) == 0
    assert height & (height - 1) == 0


def test_the_grid_fits_every_cell():
    count, cell = 93, 96
    width, height, columns = grid_for(count, cell)
    rows = -(-count // columns)
    assert columns * cell <= width
    assert rows * cell <= height


def test_an_empty_charset_is_handled():
    assert grid_for(0, 96) == (0, 0, 1)


# ── rasterisation (Qt) ───────────────────────────────────────────────────────

def test_every_requested_glyph_is_in_the_atlas(atlas):
    for char in DEFAULT_CHARSET:
        assert atlas.glyph(char) is not None, repr(char)


def test_glyph_uvs_are_normalised_and_ordered(atlas):
    for char in "AWgz0@":
        glyph = atlas.glyph(char)
        assert 0.0 <= glyph.u0 < glyph.u1 <= 1.0
        assert 0.0 <= glyph.v0 < glyph.v1 <= 1.0


def test_glyphs_do_not_share_a_cell(atlas):
    corners = {(g.u0, g.v0) for g in atlas.glyphs.values()}
    assert len(corners) == len(atlas.glyphs)


def test_the_atlas_is_a_single_channel_float_field(atlas):
    assert atlas.image.dtype == np.float32
    assert atlas.image.ndim == 2
    assert 0.0 <= atlas.image.min() and atlas.image.max() <= 1.0


def test_the_distance_field_has_an_inside_and_an_outside(atlas):
    """The property a shader needs: values either side of the 0.5 edge.

    Deliberately NOT asserting how far above 0.5 the interior reaches. Under
    the offscreen QPA platform glyphs rasterise hairline (measured: no texel
    more than one unit from an edge at any font size), so interior contrast
    here reflects the headless font engine, not the renderer. Structure is
    testable headlessly; contrast is not.
    """
    assert (atlas.image > 0.5).any()
    assert (atlas.image < 0.5).any()


def test_advances_are_positive(atlas):
    for char in "AWgz0":
        assert atlas.glyph(char).advance > 0


# ── string layout (pure) ─────────────────────────────────────────────────────

def test_layout_produces_one_quad_per_visible_glyph(atlas):
    quads = layout_string(atlas, "ABC", 0.0, 0.0)
    assert len(quads) == 3
    assert quads.shape[1] == GLYPH_FLOATS


def test_spaces_take_width_but_draw_nothing(atlas):
    """A space quad would be an invisible instance costing a draw slot."""
    assert len(layout_string(atlas, "A B", 0.0, 0.0)) == 2
    assert measure(atlas, "A B")[0] > measure(atlas, "AB")[0]


def test_glyphs_advance_left_to_right(atlas):
    quads = layout_string(atlas, "ABC", 100.0, 50.0)
    assert quads[0][0] < quads[1][0] < quads[2][0]


def test_scale_changes_both_size_and_advance(atlas):
    small = layout_string(atlas, "ABC", 0.0, 0.0, scale=0.5)
    large = layout_string(atlas, "ABC", 0.0, 0.0, scale=1.0)
    assert large[0][2] > small[0][2]
    assert large[2][0] > small[2][0]


def test_right_alignment_ends_at_the_anchor(atlas):
    """Callouts on the left of the ring right-align against their leader
    line; getting this inconsistent detaches labels from their lines."""
    width, _ = measure(atlas, "ABCDEF")
    quads = layout_string(atlas, "ABCDEF", 500.0, 0.0, align="right")
    assert quads[0][0] < 500.0 - width * 0.5


def test_centre_alignment_straddles_the_anchor(atlas):
    quads = layout_string(atlas, "ABCDEF", 500.0, 0.0, align="centre")
    assert quads[0][0] < 500.0 < quads[-1][0] + quads[-1][2]


def test_colour_and_alpha_reach_every_quad(atlas):
    quads = layout_string(atlas, "ABC", 0.0, 0.0,
                          colour=(0.2, 0.6, 0.9), alpha=0.4)
    for quad in quads:
        assert tuple(round(float(v), 3) for v in quad[8:11]) == (0.2, 0.6, 0.9)
        assert round(float(quad[11]), 3) == 0.4


def test_empty_text_produces_no_quads(atlas):
    assert len(layout_string(atlas, "", 0.0, 0.0)) == 0


def test_unknown_characters_are_skipped_not_crashed(atlas):
    quads = layout_string(atlas, "A中B", 0.0, 0.0)
    assert len(quads) == 2


def test_measure_matches_the_laid_out_extent(atlas):
    text = "REASONING"
    width, _ = measure(atlas, text)
    quads = layout_string(atlas, text, 0.0, 0.0)
    assert quads[-1][0] < width + 1.0


# ── batching: the whole point of an atlas ────────────────────────────────────

def test_many_strings_batch_into_one_buffer(atlas):
    batch = layout_many(atlas, [
        (atlas, "ALPHA", 0.0, 0.0) if False else
        {"text": "ALPHA", "x": 0.0, "y": 0.0},
        {"text": "BETA", "x": 200.0, "y": 40.0},
        {"text": "GAMMA", "x": 400.0, "y": 80.0},
    ])
    assert len(batch) == len("ALPHABETAGAMMA")
    assert batch.dtype == np.float32
    assert batch.flags["C_CONTIGUOUS"]


def test_batching_nothing_is_safe(atlas):
    assert len(layout_many(atlas, [])) == 0


def test_a_batch_of_empty_strings_is_safe(atlas):
    assert len(layout_many(atlas, [{"text": "", "x": 0.0, "y": 0.0}])) == 0
