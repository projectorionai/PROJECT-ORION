"""
The screen-space holographic layer (Mark XXIII).

Callout text, leader lines, the unfolding hover readout and packets in flight
are all built here into one glyph buffer and one line buffer. These tests pin
the things a screenshot would otherwise be the only way to catch: text that
runs off the viewport, a readout that fades instead of unfolding, and a
travelling packet drawn as though it were standing still.

The atlas is rasterised once per session — it needs a real font engine, which
is the one Qt-dependent step in the whole text system.
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

from orion_core.swarm.hover import HoverController  # noqa: E402
from orion_core.swarm.model import (  # noqa: E402
    Activity,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmNode,
)
from orion_core.swarm.render.labels import Callout  # noqa: E402
from orion_core.swarm.render.overlay import (  # noqa: E402
    EMPTY,
    LINE_FLOATS,
    PANEL_MARGIN,
    BridgeMark,
    build_bridges,
    build_callouts,
    build_hover,
    compose_overlay,
    merge,
    panel_size,
    place_panel,
)
from orion_core.swarm.render.text_atlas import GLYPH_FLOATS, rasterise_atlas  # noqa: E402

VIEWPORT = (1280.0, 800.0)


@pytest.fixture(scope="module")
def _app():
    # Rasterisation needs a real font engine, and QPainter without a
    # QApplication takes the process down natively rather than raising.
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def atlas(_app):
    return rasterise_atlas(font_size=32)


def _callout(text="INTELLIGENCE", side="right", x=1000.0, y=300.0, alpha=1.0):
    return Callout(node_id="cluster:INTELLIGENCE", text=text,
                   anchor=(640.0, 400.0), position=(x, y), elbow=(900.0, 320.0),
                   side=side, alpha=alpha, perimeter_t=0.4)


def _panel(node=None, dwell=1.0):
    hover = HoverController()
    node = node or SwarmNode(
        id="module:vision", kind=NodeKind.SUBSYSTEM, label="vision",
        cluster="SYSTEM", health=Health.OK, activity=Activity.ACTIVE,
        telemetry=NodeTelemetry(calls=120, failures=2, latency_p95_ms=88.0))
    now = 1000.0
    hover.point_at(node.id, now)
    return hover.panel(node, None, now + dwell), node


# ── shape of what the GPU receives ───────────────────────────────────────────

def test_geometry_matches_the_shader_layouts(atlas):
    geometry = build_callouts(atlas, [_callout()])
    assert geometry.glyphs.shape[1] == GLYPH_FLOATS
    assert geometry.lines.shape[1] == LINE_FLOATS
    assert geometry.glyphs.dtype == np.float32
    assert geometry.lines.dtype == np.float32


def test_leader_lines_come_out_as_vertex_pairs(atlas):
    """GL_LINES, not a strip — a strip would need one draw call per leader."""
    geometry = build_callouts(atlas, [_callout()])
    assert geometry.line_vertex_count % 2 == 0
    # anchor -> elbow -> label is two segments, so four vertices.
    assert geometry.line_vertex_count == 4


def test_everything_batches_into_one_buffer(atlas):
    """The whole point of the atlas: one draw call however many labels."""
    many = build_callouts(atlas, [_callout(x=200.0 + i * 40) for i in range(12)])
    one = build_callouts(atlas, [_callout()])
    assert many.glyph_count == one.glyph_count * 12


def test_no_callouts_produces_nothing_to_draw(atlas):
    assert build_callouts(atlas, []).is_empty


def test_an_absent_atlas_is_survivable():
    """The first frames of startup legitimately have no atlas; a renderer that
    crashed there would take the whole scene down."""
    assert compose_overlay(None, [_callout()], None, None, VIEWPORT) is EMPTY


def test_merging_parts_concatenates_both_buffers(atlas):
    part = build_callouts(atlas, [_callout()])
    merged = merge(part, part)
    assert merged.glyph_count == part.glyph_count * 2
    assert merged.line_vertex_count == part.line_vertex_count * 2


def test_merging_nothing_is_empty():
    assert merge().is_empty


# ── callout text reads outward ───────────────────────────────────────────────

def test_text_on_the_right_of_the_ring_runs_outward(atlas):
    """Left-aligned on the right side, so the glyphs move away from the scene
    instead of back over it."""
    right = build_callouts(atlas, [_callout(side="right", x=1000.0)])
    left = build_callouts(atlas, [_callout(side="left", x=280.0)])
    # First glyph of a left-side (right-aligned) label starts well before its
    # anchor; a right-side one starts after it.
    assert right.glyphs[0, 0] > 1000.0
    assert left.glyphs[0, 0] < 280.0


def test_a_faded_callout_carries_its_alpha_into_the_glyphs(atlas):
    faint = build_callouts(atlas, [_callout(alpha=0.4)])
    assert faint.glyphs[:, 11].max() == pytest.approx(0.4, abs=1e-5)


# ── nothing runs off the edge ────────────────────────────────────────────────

def _ink(geometry, atlas, text, side, index=0):
    """The span of pixels the label's INK covers.

    Not the quads' extent: an atlas cell is far wider than the glyph inside it
    (deliberately — the padding is what keeps neighbouring distance fields
    from bleeding together), so the surplus is empty field the shader
    discards. Measuring quads would report ~16px of clipping that nobody can
    see. The label's origin is the last vertex of its own leader line, and its
    width is what `measure` reports.
    """
    from orion_core.swarm.render.overlay import CALLOUT_SCALE, text_width
    origin = float(geometry.lines[index * 4 + 3, 0])
    span = text_width(atlas, text, CALLOUT_SCALE)
    return (origin, origin + span) if side == "right" else (origin - span, origin)


def test_a_long_label_on_the_right_is_clamped_inside_the_viewport(atlas):
    """A real frame rendered INTELLIGENCE as "LLIGENCE": the anchor was
    comfortably inside the edge, and the string was not."""
    geometry = build_callouts(atlas, [_callout(text="COMMUNICATION",
                                               side="right", x=1180.0)],
                              VIEWPORT)
    assert _ink(geometry, atlas, "COMMUNICATION", "right")[1] <= VIEWPORT[0]


def test_a_long_label_on_the_left_is_clamped_inside_the_viewport(atlas):
    geometry = build_callouts(atlas, [_callout(text="INTELLIGENCE",
                                               side="left", x=90.0)],
                              VIEWPORT)
    assert _ink(geometry, atlas, "INTELLIGENCE", "left")[0] >= 0.0


def test_every_callout_around_a_full_ring_stays_on_screen(atlas):
    """The ring places labels all the way round; none of them may clip."""
    ring = [_callout(text="COMMUNICATION", side="right" if i % 2 else "left",
                     x=40.0 + i * 130.0, y=60.0 + i * 70.0) for i in range(10)]
    geometry = build_callouts(atlas, ring, VIEWPORT)
    for index, callout in enumerate(ring):
        left, right = _ink(geometry, atlas, callout.text, callout.side, index)
        assert left >= 0.0 and right <= VIEWPORT[0]


def test_clamping_a_label_drags_its_leader_line_with_it(atlas):
    """Otherwise the tether points at where the text used to be."""
    callout = _callout(text="COMMUNICATION", side="right", x=1180.0)
    geometry = build_callouts(atlas, [callout], VIEWPORT)
    assert geometry.lines[-1, 0] < callout.position[0]


def test_without_a_viewport_the_text_is_placed_unclamped(atlas):
    """The clamp is opt-in so a caller that does not care can skip measuring;
    the renderer always passes one."""
    far = build_callouts(atlas, [_callout(text="COMMUNICATION", x=1180.0)])
    assert _ink(far, atlas, "COMMUNICATION", "right")[1] > VIEWPORT[0]


# ── the readout stays on screen ──────────────────────────────────────────────

def test_the_panel_flips_to_the_left_rather_than_running_off_the_edge():
    size = (236.0, 120.0)
    x, _y, side = place_panel((1240.0, 400.0), size, VIEWPORT)
    assert side == "left"
    assert x + size[0] <= VIEWPORT[0]


def test_the_panel_sits_to_the_right_when_there_is_room():
    _x, _y, side = place_panel((300.0, 400.0), (236.0, 120.0), VIEWPORT)
    assert side == "right"


def test_the_panel_never_hangs_off_the_bottom():
    _x, y, _side = place_panel((300.0, 795.0), (236.0, 200.0), VIEWPORT)
    assert y + 200.0 <= VIEWPORT[1]


def test_a_viewport_too_small_for_the_panel_clamps_rather_than_clipping():
    x, y, _side = place_panel((40.0, 40.0), (236.0, 200.0), (200.0, 150.0))
    assert x >= 0.0 and y >= 0.0
    assert (x, y) == (PANEL_MARGIN, PANEL_MARGIN)


def test_the_panel_grows_with_the_rows_that_have_arrived():
    """It unfolds — an empty box appearing and then filling would be a fade."""
    early, node = _panel(dwell=0.19)
    late, _ = _panel(node=node, dwell=1.0)
    assert panel_size(early)[1] < panel_size(late)[1]


# ── the readout is attached to its node ──────────────────────────────────────

def test_the_readout_is_tethered_to_the_node_it_describes(atlas):
    panel, _ = _panel()
    anchor = (400.0, 300.0)
    geometry = build_hover(atlas, panel, anchor, VIEWPORT)
    starts = geometry.lines[:, :2]
    assert np.isclose(starts, np.array(anchor)).all(axis=1).any()


def test_a_closed_readout_draws_nothing(atlas):
    hover = HoverController()
    hover.point_at("module:vision", 1000.0)
    panel = hover.panel(SwarmNode(id="module:vision", kind=NodeKind.SUBSYSTEM,
                                  label="vision", cluster="SYSTEM"),
                        None, 1000.0)
    assert build_hover(atlas, panel, (100.0, 100.0), VIEWPORT).is_empty


def test_a_failing_node_tints_the_readout_red(atlas):
    from orion_core.swarm.render.overlay import INK_ALARM
    node = SwarmNode(id="module:forge", kind=NodeKind.SUBSYSTEM, label="forge",
                     cluster="SYSTEM", health=Health.DOWN,
                     activity=Activity.ERROR)
    panel, _ = _panel(node=node)
    geometry = build_hover(atlas, panel, (400.0, 300.0), VIEWPORT)
    reds = geometry.glyphs[:, 8:11]
    assert np.isclose(reds, np.array(INK_ALARM)).all(axis=1).any()


def test_the_title_and_subtitle_do_not_print_through_each_other(atlas):
    """A real frame had "Subsystem · Idle" overlapping "Memory".

    Baseline separation must clear the type's LINE height — ascent, descent
    and leading. Checking cap height instead is what let the first fix pass
    review and still collide on screen: "Memory" has a descender."""
    from orion_core.swarm.render.overlay import (
        HOVER_TITLE_SCALE, SUBTITLE_BASELINE, TITLE_BASELINE)
    assert (SUBTITLE_BASELINE - TITLE_BASELINE
            >= atlas.line_height * HOVER_TITLE_SCALE)


def test_the_rule_under_the_title_clears_the_subtitle(atlas):
    from orion_core.swarm.render.overlay import (
        HOVER_ROW_SCALE, RULE_OFFSET, SUBTITLE_BASELINE, TITLE_HEIGHT)
    rule_y = TITLE_HEIGHT - RULE_OFFSET
    descent = atlas.line_height * HOVER_ROW_SCALE * 0.25
    assert rule_y >= SUBTITLE_BASELINE + descent


def test_the_first_row_clears_the_rule(atlas):
    from orion_core.swarm.render.overlay import (
        HOVER_ROW_SCALE, ROW_HEIGHT, RULE_OFFSET, TITLE_HEIGHT)
    rule_y = TITLE_HEIGHT - RULE_OFFSET
    first_row_top = (TITLE_HEIGHT + ROW_HEIGHT * 0.7
                     - atlas.font_size * HOVER_ROW_SCALE)
    assert first_row_top >= rule_y


def test_rows_never_overlap_each_other(atlas):
    from orion_core.swarm.render.overlay import HOVER_ROW_SCALE, ROW_HEIGHT
    assert ROW_HEIGHT >= atlas.line_height * HOVER_ROW_SCALE


def test_the_frame_extends_as_the_panel_opens(atlas):
    early, node = _panel(dwell=0.20)
    late, _ = _panel(node=node, dwell=1.0)
    a = build_hover(atlas, early, (400.0, 300.0), VIEWPORT)
    b = build_hover(atlas, late, (400.0, 300.0), VIEWPORT)
    assert a.lines[:, 1].max() < b.lines[:, 1].max()


def test_an_absent_measurement_actually_draws_something(atlas):
    """inspect.py's whole absent-versus-zero rule renders as "—". The em dash
    was missing from the charset, so on a real display "Failures: —" drew as
    "Failures:" and never-called became indistinguishable from never-failed —
    the exact confusion the rule exists to prevent, reintroduced by the font.
    """
    from orion_core.swarm.hover import HoverController
    from orion_core.swarm.inspect import ABSENT

    assert atlas.glyph(ABSENT) is not None

    node = SwarmNode(id="module:quiet", kind=NodeKind.SUBSYSTEM, label="quiet",
                     cluster="SYSTEM")           # never called
    hover = HoverController()
    hover.point_at(node.id, 1000.0)
    panel = hover.panel(node, None, 1001.0)
    with_dash = build_hover(atlas, panel, (400.0, 300.0), VIEWPORT)

    stripped = build_hover(
        atlas,
        panel.__class__(node_id=panel.node_id, title=panel.title,
                        subtitle=panel.subtitle,
                        rows=tuple(r.__class__(r.label, "", r.severity, r.reveal)
                                   for r in panel.rows),
                        unfold=panel.unfold, alarm=panel.alarm),
        (400.0, 300.0), VIEWPORT)
    assert with_dash.glyph_count > stripped.glyph_count


def test_the_separator_between_a_name_and_its_kind_renders(atlas):
    """The readout subtitle is "Subsystem · Idle"; without the middle dot it
    reads as two words with an unexplained gap."""
    assert atlas.glyph("·") is not None


# ── packets in flight ────────────────────────────────────────────────────────

def _mark(**kw):
    fields = dict(head=(600.0, 400.0), tail=(560.0, 380.0), intensity=1.0,
                  holding=False, failed=False, label="research")
    fields.update(kw)
    return BridgeMark(**fields)


def test_a_travelling_packet_is_a_comet_that_fades_towards_its_tail(atlas):
    """The taper is per-vertex alpha interpolated by the line shader — two
    vertices, no extra geometry."""
    geometry = build_bridges(atlas, [_mark()])
    assert geometry.line_vertex_count == 2
    assert geometry.lines[0, 5] == 0.0      # tail, transparent
    assert geometry.lines[1, 5] == 1.0      # head, full


def test_a_travelling_packet_carries_no_label(atlas):
    """Text on something moving that fast is unreadable clutter."""
    assert build_bridges(atlas, [_mark()]).glyph_count == 0


def test_a_waiting_packet_becomes_a_marker_and_names_what_it_is_waiting_on(atlas):
    geometry = build_bridges(atlas, [_mark(holding=True, label="github")])
    assert geometry.line_vertex_count == 8       # a closed diamond
    assert geometry.glyph_count > 0


def test_a_failed_packet_is_drawn_in_the_alarm_colour(atlas):
    from orion_core.swarm.render.overlay import INK_ALARM
    geometry = build_bridges(atlas, [_mark(failed=True, label="")])
    assert np.allclose(geometry.lines[0, 2:5], np.array(INK_ALARM))


def test_no_packets_draws_nothing(atlas):
    assert build_bridges(atlas, []).is_empty


# ── it all composes ──────────────────────────────────────────────────────────

def test_the_whole_layer_composes_into_two_buffers(atlas):
    panel, _ = _panel()
    geometry = compose_overlay(atlas, [_callout()], panel, (400.0, 300.0),
                               VIEWPORT, [_mark()])
    assert geometry.glyph_count > 0
    assert geometry.line_vertex_count > 0
    assert geometry.glyphs.shape[1] == GLYPH_FLOATS
    assert geometry.lines.shape[1] == LINE_FLOATS


def test_composing_with_nothing_on_screen_is_empty(atlas):
    assert compose_overlay(atlas, [], None, None, VIEWPORT).is_empty
