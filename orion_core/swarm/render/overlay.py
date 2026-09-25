"""
overlay.py — the screen-space holographic layer (Mark XXIII).

Everything drawn in pixels rather than in the scene: the callout ring around
the viewport, the leader lines tethering each callout to its subsystem, and
the diagnostic readout that unfolds under the pointer.

These are built together, in one module, producing ONE glyph buffer and ONE
line buffer, because they are one visual language and because the whole
overlay then costs two draw calls no matter how much is on screen. Building
them separately is how an interface ends up with a callout ring in one
typeface and a tooltip in another.

Nothing here touches Qt or GL. It consumes an already-rasterised Atlas and
already-placed Callouts and emits float32 arrays the renderer uploads
verbatim, which is what lets the entire overlay — placement, wrapping,
edge-flipping, unfold staging — be verified with no display at all.

Coordinates are pixels, origin top-left, matching Qt and the Callout
placement engine. The shader converts to NDC; doing it here would make the
geometry untestable against anything a human can reason about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hover import HoverPanel
from .labels import Callout
from .text_atlas import GLYPH_FLOATS, Atlas, layout_string, measure

# Per-line-vertex floats: x, y (pixels), r, g, b, a.
LINE_FLOATS = 6

# Holographic ink. Slightly blue-white rather than pure white so text sits in
# the same colour world as the field instead of on top of it.
INK: tuple[float, float, float] = (0.82, 0.92, 1.00)
INK_DIM: tuple[float, float, float] = (0.55, 0.68, 0.84)
INK_ALARM: tuple[float, float, float] = (1.00, 0.45, 0.42)
LEADER: tuple[float, float, float] = (0.36, 0.62, 0.86)

# Callout text size relative to the atlas's own rasterised size. The atlas is
# rendered large and reduced by the distance field, so this is a straight
# scale factor rather than a re-rasterisation.
CALLOUT_SCALE = 0.30
HOVER_TITLE_SCALE = 0.32
HOVER_ROW_SCALE = 0.26

# Keep callout text this far inside the viewport. The placement ring sits at
# 0.86/0.80 of the half-extent, which positions the label's ANCHOR inside the
# edge but says nothing about where the string ends — so long names ran off
# both sides and rendered as "LLIGENCE" and "COMMUNICA". Measured clamping is
# the fix; shrinking the ring would only move the cliff.
CALLOUT_MARGIN = 8.0

# Hover panel metrics, pixels.
PANEL_WIDTH = 236.0
PANEL_PAD = 11.0
ROW_HEIGHT = 18.0
# Sized from the type's own line height, not guessed. The atlas reports 64px
# of line height at its 48px rasterisation, so a title at 0.32 needs ~20.5px
# of baseline separation and a row at 0.26 needs ~16.6px. The first attempt
# used the CAP height instead and the subtitle printed straight through the
# title's descenders on a real display — "Memory" and "Subsystem · Idle"
# occupying the same pixels.
TITLE_HEIGHT = 54.0
TITLE_BASELINE = 19.0
SUBTITLE_BASELINE = 41.0
RULE_OFFSET = 7.0
# Gap between the node and the panel's near edge, so the readout never sits
# on top of the thing it describes.
PANEL_GAP = 24.0
# Keep the panel this far inside the viewport.
PANEL_MARGIN = 12.0
# How far each row slides in as it arrives. Small: this is an unfold, not a
# slide transition.
ROW_SLIDE = 9.0


@dataclass(frozen=True, slots=True)
class OverlayGeometry:
    """One frame of overlay, ready to upload."""

    glyphs: np.ndarray        # (N, GLYPH_FLOATS)
    lines: np.ndarray         # (M, LINE_FLOATS), consecutive pairs = segments

    @property
    def glyph_count(self) -> int:
        return int(self.glyphs.shape[0])

    @property
    def line_vertex_count(self) -> int:
        return int(self.lines.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.glyph_count == 0 and self.line_vertex_count == 0


def _empty_glyphs() -> np.ndarray:
    return np.zeros((0, GLYPH_FLOATS), dtype=np.float32)


def _empty_lines() -> np.ndarray:
    return np.zeros((0, LINE_FLOATS), dtype=np.float32)


EMPTY = OverlayGeometry(glyphs=_empty_glyphs(), lines=_empty_lines())


def _polyline(points, colour, alpha: float) -> list[tuple[float, ...]]:
    """A polyline as GL_LINES vertex pairs.

    GL_LINE_STRIP would need one draw call per polyline; emitting pairs lets
    every leader line in the scene share a single draw."""
    out: list[tuple[float, ...]] = []
    for start, end in zip(points, points[1:]):
        out.append((float(start[0]), float(start[1]), *colour, float(alpha)))
        out.append((float(end[0]), float(end[1]), *colour, float(alpha)))
    return out


# ── the callout ring ─────────────────────────────────────────────────────────

def build_callouts(atlas: Atlas, callouts: list[Callout],
                   viewport: tuple[float, float] | None = None,
                   scale: float = CALLOUT_SCALE) -> OverlayGeometry:
    """Text and leader lines for every placed callout.

    *viewport* enables edge clamping. It is optional only so a caller that
    genuinely does not care about clipping can omit it; the renderer always
    passes it.
    """
    if not callouts:
        return EMPTY

    strings: list[tuple] = []
    lines: list[tuple[float, ...]] = []
    for callout in callouts:
        x, y = callout.position
        # Text is aligned AWAY from the ring, so it reads outward from the
        # scene rather than overlapping it — and it means the leader's final
        # horizontal run always terminates at the text's near edge.
        align = "left" if callout.side == "right" else "right"
        # Nudge off the ring so the glyphs do not sit on the leader's elbow.
        text_x = x + (6.0 if callout.side == "right" else -6.0)

        if viewport is not None:
            # Clamp by MEASURED width. A label is placed by its anchor, and a
            # long name overruns the edge from an anchor that is comfortably
            # inside it — which is exactly how INTELLIGENCE rendered as
            # "LLIGENCE" on a real display.
            span = text_width(atlas, callout.text, scale)
            limit = viewport[0] - CALLOUT_MARGIN
            if align == "left":
                text_x = min(text_x, limit - span)
                text_x = max(text_x, CALLOUT_MARGIN)
            else:
                text_x = max(text_x, CALLOUT_MARGIN + span)
                text_x = min(text_x, limit)

        strings.append((callout.text, text_x, y, scale, INK, callout.alpha, align))
        # The leader ends where the text actually starts, so clamping a label
        # inward can never leave its tether pointing into empty space.
        anchor, elbow, _ = callout.leader
        lines.extend(_polyline([anchor, elbow, (text_x, y)],
                               LEADER, callout.alpha * 0.75))

    blocks = [b for b in (layout_string(atlas, *s) for s in strings) if len(b)]
    glyphs = np.concatenate(blocks) if blocks else _empty_glyphs()
    return OverlayGeometry(
        glyphs=np.ascontiguousarray(glyphs, dtype=np.float32),
        lines=np.ascontiguousarray(np.array(lines, dtype=np.float32))
        if lines else _empty_lines(),
    )


# ── the hover readout ────────────────────────────────────────────────────────

def panel_size(panel: HoverPanel) -> tuple[float, float]:
    """How large the readout is right now.

    Sized from the rows that have ARRIVED, not from the eventual total, so the
    frame grows with its contents instead of an empty box appearing and then
    filling — which is the difference between unfolding and fading in."""
    rows = len(panel.visible_rows)
    return (PANEL_WIDTH, TITLE_HEIGHT + rows * ROW_HEIGHT + PANEL_PAD)


def place_panel(anchor: tuple[float, float], size: tuple[float, float],
                viewport: tuple[float, float]) -> tuple[float, float, str]:
    """Where the readout goes: (x, y, side).

    Prefers the right of the node and flips left when it would run off the
    edge. A panel clipped by the viewport is worse than one on the unusual
    side, because half a diagnostic is not a diagnostic.
    """
    width, height = viewport
    panel_w, panel_h = size
    ax, ay = anchor

    x = ax + PANEL_GAP
    side = "right"
    if x + panel_w > width - PANEL_MARGIN:
        x = ax - PANEL_GAP - panel_w
        side = "left"
    # Both sides blocked (a narrow viewport): clamp rather than clip.
    x = min(max(PANEL_MARGIN, x), max(PANEL_MARGIN, width - panel_w - PANEL_MARGIN))

    y = ay - TITLE_HEIGHT * 0.5
    y = min(max(PANEL_MARGIN, y), max(PANEL_MARGIN, height - panel_h - PANEL_MARGIN))
    return (float(x), float(y), side)


def build_hover(atlas: Atlas, panel: HoverPanel | None,
                anchor: tuple[float, float],
                viewport: tuple[float, float]) -> OverlayGeometry:
    """The unfolding diagnostic readout."""
    if panel is None or not panel.is_open:
        return EMPTY

    size = panel_size(panel)
    x, y, side = place_panel(anchor, size, viewport)
    panel_w, panel_h = size
    unfold = panel.unfold
    accent = INK_ALARM if panel.alarm else INK

    strings: list[tuple] = []
    lines: list[tuple[float, ...]] = []

    # The frame is a bracket, not a box: a spine down the panel's near edge
    # with a rule under the title. A full rectangle would read as a dialog;
    # a bracket reads as an instrument annotation attached to the node.
    spine_x = x if side == "right" else x + panel_w
    spine_end = y + panel_h * unfold
    lines.extend(_polyline([(spine_x, y), (spine_x, spine_end)], accent, unfold))
    rule_w = panel_w * unfold
    rule_x0 = spine_x if side == "right" else spine_x - rule_w
    rule_y = y + TITLE_HEIGHT - RULE_OFFSET
    lines.extend(_polyline([(rule_x0, rule_y), (rule_x0 + rule_w, rule_y)],
                           accent, unfold * 0.7))
    # The tether: a short run from the node to the spine, so the readout is
    # visibly ABOUT the thing under the pointer rather than floating near it.
    lines.extend(_polyline([anchor, (spine_x, y + TITLE_HEIGHT * 0.5)],
                           LEADER, unfold * 0.8))

    text_x = x + PANEL_PAD if side == "right" else x + panel_w - PANEL_PAD
    align = "left" if side == "right" else "right"
    strings.append((panel.title, text_x, y + TITLE_BASELINE, HOVER_TITLE_SCALE,
                    accent, unfold, align))
    strings.append((panel.subtitle, text_x, y + SUBTITLE_BASELINE,
                    HOVER_ROW_SCALE, INK_DIM, unfold * 0.85, align))

    # Values are right-aligned against the panel's far edge so digits line up
    # in a column — a readout whose numbers wander is unreadable at a glance.
    value_x = x + panel_w - PANEL_PAD if side == "right" else x + PANEL_PAD
    value_align = "right" if side == "right" else "left"

    baseline = y + TITLE_HEIGHT + ROW_HEIGHT * 0.7
    for index, row in enumerate(panel.rows):
        if row.reveal <= 0.0:
            continue
        row_y = baseline + index * ROW_HEIGHT
        # Arriving rows slide in from the spine.
        slide = (1.0 - row.reveal) * ROW_SLIDE * (1.0 if side == "right" else -1.0)
        colour = INK_ALARM if row.severity == "alarm" else INK_DIM
        strings.append((row.label, text_x + slide, row_y, HOVER_ROW_SCALE,
                        colour, row.reveal * 0.8, align))
        strings.append((row.value, value_x + slide, row_y, HOVER_ROW_SCALE,
                        INK if row.severity != "alarm" else INK_ALARM,
                        row.reveal, value_align))

    blocks = [b for b in (layout_string(atlas, *s) for s in strings) if len(b)]
    glyphs = np.concatenate(blocks) if blocks else _empty_glyphs()
    return OverlayGeometry(
        glyphs=np.ascontiguousarray(glyphs, dtype=np.float32),
        lines=np.ascontiguousarray(np.array(lines, dtype=np.float32)),
    )


# ── commands in flight ───────────────────────────────────────────────────────

# The packet's own colour: brighter and cooler than the leader lines, so a
# command crossing an edge is unmistakably different from the edge itself.
BRIDGE_INK: tuple[float, float, float] = (0.62, 0.94, 1.00)

# How long the comet tail is, in pixels, at full speed.
BRIDGE_TAIL = 34.0
# Half-diagonal of the diamond drawn on a packet that has arrived and is
# waiting for its work to finish.
BRIDGE_MARK = 7.0
BRIDGE_LABEL_SCALE = 0.22


@dataclass(frozen=True, slots=True)
class BridgeMark:
    """One packet, already projected to the screen by the renderer."""

    head: tuple[float, float]
    tail: tuple[float, float]
    intensity: float
    holding: bool
    failed: bool
    label: str = ""


def build_bridges(atlas: Atlas, marks: list[BridgeMark]) -> OverlayGeometry:
    """Comet streaks for travelling packets, markers for waiting ones.

    A streak rather than a dot, and the taper is free: the line shader
    interpolates per-vertex colour along the segment, so giving the tail alpha
    0 and the head full alpha produces a real comet with two vertices and no
    extra geometry.
    """
    if not marks:
        return EMPTY

    lines: list[tuple[float, ...]] = []
    strings: list[tuple] = []
    for mark in marks:
        colour = INK_ALARM if mark.failed else BRIDGE_INK
        intensity = max(0.0, min(1.0, mark.intensity))
        hx, hy = mark.head
        if mark.holding or mark.failed:
            # Parked. A diamond rather than a streak, because a streak with no
            # direction of travel would imply movement that is not happening —
            # and this packet standing still IS the information.
            size = BRIDGE_MARK * (0.75 + 0.45 * intensity)
            points = [(hx, hy - size), (hx + size, hy), (hx, hy + size),
                      (hx - size, hy), (hx, hy - size)]
            lines.extend(_polyline(points, colour, intensity))
            # Only a waiting packet gets a label. On one still travelling the
            # text would be unreadable; on one that has stopped it names what
            # ORION is actually waiting on.
            if mark.label:
                strings.append((mark.label, hx + size + 5.0, hy + 3.0,
                                BRIDGE_LABEL_SCALE, colour, intensity * 0.9,
                                "left"))
        else:
            tx, ty = mark.tail
            lines.append((float(tx), float(ty), *colour, 0.0))
            lines.append((float(hx), float(hy), *colour, intensity))

    blocks = [b for b in (layout_string(atlas, *s) for s in strings) if len(b)]
    return OverlayGeometry(
        glyphs=np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)
        if blocks else _empty_glyphs(),
        lines=np.ascontiguousarray(np.array(lines, dtype=np.float32)),
    )


# ── composition ──────────────────────────────────────────────────────────────

def merge(*parts: OverlayGeometry) -> OverlayGeometry:
    """Concatenate overlay parts into one uploadable pair of buffers."""
    glyph_blocks = [p.glyphs for p in parts if p.glyph_count]
    line_blocks = [p.lines for p in parts if p.line_vertex_count]
    return OverlayGeometry(
        glyphs=np.ascontiguousarray(np.concatenate(glyph_blocks), dtype=np.float32)
        if glyph_blocks else _empty_glyphs(),
        lines=np.ascontiguousarray(np.concatenate(line_blocks), dtype=np.float32)
        if line_blocks else _empty_lines(),
    )


def compose_overlay(atlas: Atlas | None,
                    callouts: list[Callout] | None,
                    panel: HoverPanel | None,
                    hover_anchor: tuple[float, float] | None,
                    viewport: tuple[float, float],
                    bridges: list[BridgeMark] | None = None) -> OverlayGeometry:
    """The whole 2D layer for this frame.

    Returns empty geometry rather than raising when the atlas has not been
    rasterised yet — the first frames of startup legitimately have no text,
    and a renderer that crashed there would take the entire scene with it.
    """
    if atlas is None:
        return EMPTY
    parts = [build_callouts(atlas, callouts or [], viewport)]
    if bridges:
        parts.append(build_bridges(atlas, bridges))
    if panel is not None and hover_anchor is not None:
        parts.append(build_hover(atlas, panel, hover_anchor, viewport))
    return merge(*parts)


def text_width(atlas: Atlas, text: str, scale: float) -> float:
    """Rendered width, for anything that needs to align against text."""
    return measure(atlas, text, scale)[0]
