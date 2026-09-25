"""
text_atlas.py — GPU holographic typography (Mark XXIII).

Every label in ORION is drawn from one texture atlas by one instanced draw
call. This is the foundation for all holographic text in the interface, so it
is built as a general text system rather than something that only knows how
to draw callouts.

WHY SDF. A plain rasterised atlas is sharp at exactly one size and turns to
mush at any other — fatal here, because callouts scale with camera distance
and fade continuously. A signed distance field stores, per texel, the
distance to the nearest glyph edge, so the shader can reconstruct a crisp
edge at ANY scale with a single smoothstep, and can additionally derive
outlines and glow from the same texture for free. That last part matters:
holographic text needs a bloom-compatible soft edge, and with an SDF that is
another threshold on data already present rather than a second render pass.

STRUCTURE. Everything that decides WHERE things go — atlas packing, glyph
metrics, string layout into quads — is pure and testable. Only
`rasterise_atlas` touches Qt, because glyph rasterisation needs a real font
engine. That split is the same one used throughout this renderer, and it is
what lets the text system be verified without a display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Per-glyph-instance floats, as the text shader reads them:
#   0..1  screen position of the quad's top-left, pixels
#   2..3  quad size, pixels
#   4..7  atlas UV rect (u0, v0, u1, v1)
#   8..10 colour rgb
#   11    alpha
GLYPH_FLOATS = 12

# The characters ORION's interface actually uses. Restricted on purpose: a
# full Unicode atlas would be enormous and almost entirely unused, and the
# label text is subsystem names, numbers and light punctuation.
#
# The em dash and middle dot are NOT decoration. inspect.py's entire
# absent-versus-zero rule renders an unmeasured value as "—", and the readouts
# separate a name from its kind with "·". Both were missing from the first
# charset, so on a real display "Failures: —" drew as "Failures:" and "never
# called" became indistinguishable from "called and never failed" — the exact
# confusion that rule exists to prevent, reintroduced by the font.
DEFAULT_CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,:;!?/\\|-_+=()[]<>@#%&*'\"`~^$"
    "—·°"
)

# Padding around each glyph in the atlas, in texels. Must exceed the SDF
# spread or neighbouring glyphs bleed into each other's distance fields.
GLYPH_PADDING = 8
# How far, in texels, the distance field extends from an edge.
#
# Tuned against a real rasterisation rather than guessed: at a spread of 6 the
# field only reached 0.58 at glyph centres — a mere 0.08 above the 0.5 edge
# threshold — because typical stroke half-widths are small relative to that
# spread, leaving the shader almost no precision to reconstruct an edge from.
# 3.5 puts stroke centres near 0.8, giving smoothstep real range to work with
# while still leaving room for outline and glow thresholds.
SDF_SPREAD = 3.5


@dataclass(frozen=True, slots=True)
class Glyph:
    """One character's place in the atlas and how to lay it out."""

    char: str
    u0: float
    v0: float
    u1: float
    v1: float
    width: float        # rendered size in pixels at the atlas's own font size
    height: float
    bearing_x: float    # offset from the pen position to the glyph's left edge
    bearing_y: float    # offset from the baseline UP to the glyph's top edge
    advance: float      # how far the pen moves after drawing this glyph


@dataclass(frozen=True, slots=True)
class Atlas:
    """A rasterised glyph atlas plus its metrics."""

    image: np.ndarray               # (h, w) float32 SDF in 0..1, or coverage
    glyphs: dict[str, Glyph]
    font_size: float
    line_height: float
    is_sdf: bool

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.image.shape[1]), int(self.image.shape[0]))

    def glyph(self, char: str) -> Glyph | None:
        return self.glyphs.get(char)


def grid_for(count: int, cell: int) -> tuple[int, int, int]:
    """Atlas dimensions for *count* cells of *cell* texels.

    Square-ish and power-of-two: some drivers still handle NPOT textures
    poorly with mipmaps, and a square atlas wastes the least space for an
    unknown glyph count."""
    if count <= 0:
        return (0, 0, 1)
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / columns))
    width = 1 << max(0, (columns * cell - 1)).bit_length()
    height = 1 << max(0, (rows * cell - 1)).bit_length()
    return (int(width), int(height), int(columns))


def _distance_field(coverage: np.ndarray, spread: float) -> np.ndarray:
    """Signed distance field from a binary coverage mask, normalised to 0..1.

    Inside the glyph the value rises above 0.5, outside it falls below, so a
    shader recovers the edge with smoothstep around 0.5 at any scale.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:       # pragma: no cover — scipy absent
        return coverage.astype(np.float32)

    # Threshold BELOW the midpoint. Glyph edges are antialiased, so most of a
    # thin stroke's texels carry partial coverage; cutting at 0.5 keeps only
    # the stroke's core and leaves it 1-2 texels wide, which caps the distance
    # field at ~0.57 and leaves no headroom for outline or glow. 0.35 captures
    # the stroke as drawn.
    inside = coverage > 0.35
    if not inside.any():
        return np.zeros_like(coverage, dtype=np.float32)
    # Distance to the nearest edge, measured separately on each side.
    outside_distance = distance_transform_edt(~inside)
    inside_distance = distance_transform_edt(inside)
    # The half-texel offset is essential. distance_transform_edt measures from
    # pixel CENTRES, so the nearest inside pixel reports 1 and the nearest
    # outside pixel reports 1 — a naive difference jumps from -1 straight to
    # +1 and produces a field with NO values at the edge itself, leaving the
    # shader's smoothstep nothing to interpolate across and the text aliased.
    # Subtracting half a texel on each side places the true edge between them,
    # at exactly 0.
    signed = np.where(inside, inside_distance - 0.5, -(outside_distance - 0.5))
    return np.clip(0.5 + signed / (2.0 * max(1e-6, spread)), 0.0, 1.0).astype(np.float32)


def rasterise_atlas(font_family: str = "Consolas", font_size: int = 48,
                    charset: str = DEFAULT_CHARSET,
                    sdf: bool = True) -> Atlas:
    """Render *charset* into an SDF atlas. The only Qt-dependent step.

    Rasterised well above display size and then reduced to a distance field,
    which is what allows one atlas to serve every label size crisply.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter

    font = QFont(font_family, font_size)
    # Bold: holographic labels want presence, and a heavier stroke also gives
    # the distance field more interior to measure, which is what outline and
    # glow thresholds are derived from.
    font.setWeight(QFont.Weight.DemiBold)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    metrics = QFontMetricsF(font)

    cell = int(font_size * 2.0) + GLYPH_PADDING * 2
    width, height, columns = grid_for(len(charset), cell)

    image = QImage(width, height, QImage.Format.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

    glyphs: dict[str, Glyph] = {}
    ascent = metrics.ascent()
    for index, char in enumerate(charset):
        column, row = index % columns, index // columns
        x0, y0 = column * cell, row * cell
        pen_x = x0 + GLYPH_PADDING
        pen_y = y0 + GLYPH_PADDING + ascent
        painter.drawText(int(pen_x), int(pen_y), char)

        rect = metrics.boundingRect(char)
        glyphs[char] = Glyph(
            char=char,
            u0=x0 / width, v0=y0 / height,
            u1=(x0 + cell) / width, v1=(y0 + cell) / height,
            width=float(cell), height=float(cell),
            bearing_x=float(rect.x()),
            bearing_y=float(-rect.y()),
            advance=float(metrics.horizontalAdvance(char)),
        )
    painter.end()

    # QImage -> numpy. bytesPerLine can exceed width (row padding), so the
    # buffer is reshaped by stride and then trimmed; assuming width would
    # shear the atlas diagonally on some sizes.
    buffer = np.frombuffer(image.constBits().asstring(
        image.sizeInBytes()), dtype=np.uint8)
    coverage = buffer.reshape(height, image.bytesPerLine())[:, :width]
    coverage = coverage.astype(np.float32) / 255.0

    field = _distance_field(coverage, SDF_SPREAD) if sdf else coverage
    return Atlas(image=field, glyphs=glyphs, font_size=float(font_size),
                 line_height=float(metrics.height()), is_sdf=bool(sdf))


def measure(atlas: Atlas, text: str, scale: float = 1.0) -> tuple[float, float]:
    """Rendered width and height of *text*, for alignment and hit-testing."""
    width = sum(g.advance for ch in text if (g := atlas.glyph(ch)) is not None)
    return (width * scale, atlas.line_height * scale)


def layout_string(atlas: Atlas, text: str, x: float, y: float,
                  scale: float = 1.0,
                  colour: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  alpha: float = 1.0,
                  align: str = "left") -> np.ndarray:
    """Lay *text* out into glyph quads ready for instanced drawing.

    (x, y) is the BASELINE origin. Alignment is applied here rather than by
    the caller so every surface positions text identically — a callout on the
    left of the ring right-aligns against its leader line, and getting that
    inconsistent between call sites is how labels end up detached from their
    lines.
    """
    if not text:
        return np.zeros((0, GLYPH_FLOATS), dtype=np.float32)

    total_width, _ = measure(atlas, text, scale)
    if align == "right":
        x -= total_width
    elif align == "centre":
        x -= total_width * 0.5

    rows: list[tuple[float, ...]] = []
    pen = x
    for char in text:
        glyph = atlas.glyph(char)
        if glyph is None:
            continue
        if char != " ":
            # The cell is drawn as one quad: the padding and bearing are
            # baked into the atlas cell, so position is simply the pen minus
            # the same padding the rasteriser applied.
            quad_x = pen - GLYPH_PADDING * scale
            quad_y = y - (atlas.font_size + GLYPH_PADDING) * scale
            rows.append((
                quad_x, quad_y,
                glyph.width * scale, glyph.height * scale,
                glyph.u0, glyph.v0, glyph.u1, glyph.v1,
                colour[0], colour[1], colour[2], alpha,
            ))
        pen += glyph.advance * scale

    if not rows:
        return np.zeros((0, GLYPH_FLOATS), dtype=np.float32)
    return np.ascontiguousarray(np.array(rows, dtype=np.float32))


def layout_many(atlas: Atlas, items) -> np.ndarray:
    """Batch several strings into ONE instance buffer.

    The whole point of the atlas: every label in the scene becomes a single
    draw call, however many there are.
    """
    blocks = [layout_string(atlas, *args) if isinstance(args, tuple)
              else layout_string(atlas, **args) for args in items]
    blocks = [b for b in blocks if len(b)]
    if not blocks:
        return np.zeros((0, GLYPH_FLOATS), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)
