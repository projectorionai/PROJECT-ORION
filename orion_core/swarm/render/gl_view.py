"""
gl_view.py — the QOpenGLWidget shell (Mark XXII, Phase 1).

The ONLY file in the renderer that touches OpenGL or Qt. Everything with
logic — layout, palette, camera, instance buffer, picking, mesh — lives in
pure numpy modules beside this one and is fully unit-tested without a
graphics context. This file owns GL handles and event plumbing and delegates
every decision.

That split is not stylistic. This session lost an entire suite run to a native
STATUS_STACK_BUFFER_OVERRUN caused by a test that constructed a real Chromium
context, and the fix was to stop importing the heavy module at collection
time. The same hazard applies in full to a GL context, so: nothing here is
imported at orion_core import time (SwarmGLView is reached lazily), and no
test constructs this class.

Rendering is a single glDrawArraysInstanced over one shared icosphere, with
per-instance position/colour/emissive/pulse/radius/selected supplied through
a divisor-1 vertex attribute. Uploads are partial — only the dirty slot span
is written, via glBufferSubData — so a steady-state frame transfers nothing.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..hover import HoverController
from ..model import NodeKind, SwarmSnapshot
from .camera import OrbitCamera
from .edges import EDGE_FLOATS
from .instances import INSTANCE_FLOATS
from .labels import layout_callouts, project_to_screen
from .mesh import build_sphere
from .face_expression import FaceAnimator
from .face_geometry import FACE_FLOATS, HEAD_RADIUS as FACE_HEAD_RADIUS, build_face
from .overlay import LINE_FLOATS, compose_overlay
from .particles import PARTICLE_FLOATS, build_field
from .pick import pick_slot, ray_from_screen
from .scene import SwarmScene
from .text_atlas import GLYPH_FLOATS

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    GL_AVAILABLE = True
except Exception:  # pragma: no cover — PyQt6 without the OpenGL module
    QOpenGLWidget = object  # type: ignore[assignment,misc]
    GL_AVAILABLE = False


_VERTEX_SHADER = """
#version 410 core
layout(location = 0) in vec3 mesh_position;   // unit sphere; also its normal
layout(location = 1) in vec3 inst_position;
layout(location = 2) in vec3 inst_colour;
layout(location = 3) in float inst_emissive;
layout(location = 4) in float inst_pulse;
layout(location = 5) in float inst_radius;
layout(location = 6) in float inst_selected;

uniform mat4 u_view_projection;
uniform float u_time;
uniform vec3 u_eye;
// Distance inside which a node gets out of the way, or 0 when the whole field
// is on show. Used by the compact orb: at portrait range the nodes that
// happen to orbit between the camera and ORION fill a third of the frame,
// and the orb is supposed to be his FACE.
uniform float u_portrait_cull;

out vec3 v_normal;
out vec3 v_colour;
out float v_emissive;
out float v_selected;

void main() {
    // Pulse scales the node gently rather than flashing its colour, so an
    // active node is legible in peripheral vision without the graph
    // strobing when many nodes are busy at once.
    float pulse = inst_pulse > 0.0
        ? 1.0 + 0.11 * sin(u_time * inst_pulse * 6.2831853)
        : 1.0;
    float radius = inst_radius * pulse * (inst_selected > 0.5 ? 1.22 : 1.0);

    // Shrink rather than fade: the node pass draws opaque, so fading would
    // mean giving the whole graph a blend mode purely for the orb. A node
    // scaled to zero radius draws nothing at all, and scales back smoothly.
    if (u_portrait_cull > 0.0) {
        radius *= smoothstep(u_portrait_cull * 0.62, u_portrait_cull,
                             distance(u_eye, inst_position));
    }

    vec3 world = inst_position + mesh_position * radius;
    gl_Position = u_view_projection * vec4(world, 1.0);
    v_normal = mesh_position;          // unit sphere => position is the normal
    v_colour = inst_colour;
    v_emissive = inst_emissive;
    v_selected = inst_selected;
}
"""

_FRAGMENT_SHADER = """
#version 410 core
in vec3 v_normal;
in vec3 v_colour;
in float v_emissive;
in float v_selected;
out vec4 frag_colour;

const vec3 KEY_LIGHT = normalize(vec3(0.4, 0.8, 0.6));
const vec3 FILL_LIGHT = normalize(vec3(-0.6, -0.2, 0.4));
// ORION's own crimson, used to tint shadow rather than leaving it neutral
// grey — unlit surfaces then read as sitting inside his shell instead of
// floating in a colourless void.
const vec3 SHADOW_TINT = vec3(0.16, 0.05, 0.07);

void main() {
    vec3 normal = normalize(v_normal);
    float lambert = max(dot(normal, KEY_LIGHT), 0.0);
    // A dim opposing fill keeps the dark side of a sphere from going flat
    // black, so nodes read as volumes rather than as discs.
    float fill = max(dot(normal, FILL_LIGHT), 0.0) * 0.22;

    // Fresnel rim, sharpened from the old pow(...,2). The tighter falloff
    // concentrates light at the silhouette, which is what makes a sphere read
    // as a holographic shell rather than a billiard ball.
    float fresnel = pow(1.0 - abs(normal.z), 3.0);

    vec3 lit = v_colour * (0.20 + 0.62 * lambert + fill) + SHADOW_TINT;
    // Rim and emissive are added rather than mixed so bright nodes bloom past
    // their own base colour instead of just becoming paler.
    vec3 glow = v_colour * (v_emissive * 0.75) + v_colour * fresnel * 1.15;
    vec3 result = lit + glow;

    if (v_selected > 0.5) {
        // Selection reads as a hot outline, not a wash: almost all of the
        // added light sits on the rim so the node's own colour survives.
        result += vec3(1.00, 0.85, 0.80) * fresnel * 0.9 + vec3(0.06);
    }
    frag_colour = vec4(result, 1.0);
}
"""

_FACE_VERTEX_SHADER = """
#version 410 core
layout(location = 0) in vec3  f_rest;      // rest position, head space
layout(location = 1) in vec3  f_colour;
layout(location = 2) in float f_size;
layout(location = 3) in float f_feature;   // FEATURE_* tag
layout(location = 4) in float f_phase;
layout(location = 5) in vec3  f_normal;    // surface normal of the sculpt

uniform mat4  u_view_projection;
uniform float u_time;
uniform float u_viewport_h;
uniform vec3  u_eye;                       // camera position, world space

// The entire performance. Geometry is never regenerated — the avatar is
// animated purely by these, which is what makes the face affordable inside
// the shared particle pipeline.
uniform float u_blink, u_mouth, u_jaw, u_brow;
uniform float u_gaze_x, u_gaze_y, u_breath;
uniform float u_drift_x, u_drift_y, u_energy, u_focus;

out vec3  v_colour;
out float v_alpha;

const float FEATURE_SKIN  = 0.0;
const float FEATURE_EYE_L = 1.0;
const float FEATURE_EYE_R = 2.0;
const float FEATURE_MOUTH = 3.0;
const float FEATURE_BROW  = 4.0;
const float FEATURE_HALO  = 5.0;

bool is(float tag, float want) { return abs(tag - want) < 0.5; }

void main() {
    vec3 p = f_rest;
    float glow = 1.0;
    // How much this point ignores the key/fill lighting and simply shines.
    float emissive = 0.0;

    // Eyes: the lid closes by collapsing points toward the eye's own centre
    // line rather than scaling the whole head, so a blink stays local.
    if (is(f_feature, FEATURE_EYE_L) || is(f_feature, FEATURE_EYE_R)) {
        float side = is(f_feature, FEATURE_EYE_R) ? 1.0 : -1.0;
        vec3 centre = vec3(side * 1.86, 0.99, 4.59);   // EYE_X/Y/Z * HEAD_RADIUS
        // Squint from concentration narrows the eye a little; blink closes it.
        float close = clamp(u_blink + u_focus * 0.22, 0.0, 1.0);
        p.y = mix(p.y, centre.y, close);
        // Gaze slides the eye points across the socket. Small travel — eyes
        // that swing to the extremes look anxious rather than thoughtful.
        p.x += u_gaze_x * 0.55;
        p.y += u_gaze_y * 0.38;
        glow = 1.05;
        // Eyes are ORION's own light, not a lit surface. Shading them like
        // skin put them at the bottom of a recess the key light never reaches,
        // and both eyes rendered as empty black sockets.
        emissive = 1.0;
    }
    else if (is(f_feature, FEATURE_MOUTH)) {
        // Mouth opens downward and the jaw carries the lower lip further,
        // so the aperture is asymmetric like a real mouth rather than an
        // ellipse expanding from its centre.
        float below = step(p.y, -2.1);
        p.y -= (u_mouth * 0.95 + u_jaw * 0.75) * below;
        p.y += u_mouth * 0.10 * (1.0 - below);
        p.z += u_mouth * 0.18;
        glow = 0.92 + u_mouth * 0.45;
    }
    else if (is(f_feature, FEATURE_BROW)) {
        p.y += u_brow * 0.62;
        // Furrowing pulls the inner ends down and together.
        p.x -= sign(p.x) * max(0.0, -u_brow) * 0.30;
        glow = 1.05;
    }
    else if (is(f_feature, FEATURE_HALO)) {
        // The halo breathes harder than the head and shimmers, so ORION's
        // outline reads as energy rather than a shell.
        float shimmer = 0.5 + 0.5 * sin(u_time * 0.9 + f_phase);
        p *= 1.0 + u_breath * 2.2 + shimmer * 0.02;
        glow = 0.55 + shimmer * 0.5;
    }

    // Whole-head motion, applied after features so it carries them along.
    p *= 1.0 + u_breath;
    p.x += u_drift_x * 2.2;
    p.y += u_drift_y * 2.2;

    vec4 clip = u_view_projection * vec4(p, 1.0);
    gl_Position = clip;
    float w = max(clip.w, 0.001);

    // ── lighting ────────────────────────────────────────────────────────────
    // The first version shaded by camera distance alone, on the theory that
    // "nearer reads brighter" would supply form. It cannot: distance fade is
    // one global gradient across the whole head, identical for the brow and
    // the cheek beside it, and the back of the skull draws straight through
    // the front. That is why the avatar was a ball of fog.
    vec3 N = normalize(f_normal);
    vec3 V = normalize(u_eye - p);
    float facing = dot(N, V);

    // The key is VIEW-RELATIVE: three-quarters from the viewer's upper left,
    // wherever the viewer happens to be.
    //
    // It was fixed in world space, which is correct for a subject you look at
    // from one bearing — the portrait camera. The swarm's camera orbits, and
    // from its default three-quarter bearing that fixed key sat behind the
    // subject: ORION's brain rendered as a dark blob with all of its folds on
    // the unlit side. Keeping the light with the camera means the modelling
    // faces you from every angle, which is what a turntable needs.
    vec3 world_up = vec3(0.0, 1.0, 0.0);
    vec3 side = normalize(cross(V, world_up) + vec3(1e-5));
    vec3 KEY  = normalize(V * 0.40 - side * 0.78 + world_up * 0.55);
    vec3 FILL = normalize(V * 0.30 + side * 0.80 - world_up * 0.25);
    float key  = max(0.0, dot(N, KEY));
    float fill = max(0.0, dot(N, FILL)) * 0.26;
    // Rim: brightest where the surface turns away, which draws the silhouette
    // and is what makes a point cloud read as a volume rather than a patch.
    float rim = pow(clamp(1.0 - facing, 0.0, 1.0), 3.0) * 0.75;

    // Low ambient on purpose. Lifting the shadows is what flattens a face.
    float shade = 0.09 + key * 1.15 + fill + rim;
    // Emissive points keep their own value, with just enough of the rim so
    // they still sit inside the head's volume rather than floating on it.
    shade = mix(shade, 1.0 + rim * 0.45, emissive);

    // Points on the far side of the head must not print through the front.
    // Suppressing them by facing is the point-cloud equivalent of back-face
    // culling, and without it every feature is veiled by the skull behind it.
    float back = smoothstep(-0.05, 0.30, facing);
    if (is(f_feature, FEATURE_HALO)) {
        // The halo is a shell of presence, not a surface — it should be
        // visible all the way round, only dimmer behind him.
        back = mix(0.45, 1.0, back);
        shade = 0.55 + rim * 0.8;
    }

    gl_PointSize = clamp(f_size * u_viewport_h / (w * 3.2), 1.0, 26.0)
                   * (0.55 + 0.45 * back);

    // Aerial perspective, kept but demoted to what it actually is: a depth
    // cue on top of the shading, not a substitute for it.
    float depth_fade = clamp(1.0 - (w - 20.0) / 320.0, 0.45, 1.0);
    // Roll highlights off BELOW 1.0, preserving hue.
    //
    // Not a stylistic choice. Measured on real hardware: a channel that
    // reaches 1.0 came back as ~0 through this path, and ORION's skin blue
    // was exactly 1.00 — so the brightest lit points along his key side lost
    // their blue entirely and rendered as yellow-green speckles across his
    // temple. Per-channel clamping would not have helped, since clamp() still
    // returns exactly 1.0; scaling the whole colour keeps the hue intact
    // instead of letting the widest channel clip first and shift it.
    vec3 lit = f_colour * glow * u_energy * shade;
    float peak = max(max(lit.r, lit.g), lit.b);
    v_colour = lit * (0.985 / max(peak, 0.985));
    v_alpha = depth_fade * back;
}
"""

_FACE_FRAGMENT_SHADER = """
#version 410 core
in vec3  v_colour;
in float v_alpha;
out vec4 frag_colour;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    // An opaque core with a soft rim, NOT a squared falloff. The squared
    // version left each point visibly transparent over most of its disc, so
    // even at just-touching spacing the skin never closed into a surface —
    // you saw black background between the points and the lighting had
    // nothing continuous to model. The nose was fully shaded in the geometry
    // and completely invisible on screen for exactly this reason.
    float core = smoothstep(0.25, 0.045, r2);
    frag_colour = vec4(v_colour * (0.72 + 0.28 * core), v_alpha * core);
}
"""

_FA_F = 4
_FACE_STRIDE = FACE_FLOATS * _FA_F
_FACE_ATTRIBS = (
    (0, 3, 0 * _FA_F),   # rest position
    (1, 3, 3 * _FA_F),   # colour
    (2, 1, 6 * _FA_F),   # size
    (3, 1, 7 * _FA_F),   # feature tag
    (4, 1, 8 * _FA_F),   # phase
    (5, 3, 9 * _FA_F),   # surface normal
)


_PARTICLE_VERTEX_SHADER = """
#version 410 core
layout(location = 0) in vec3  p_centre;
layout(location = 1) in float p_radius;
layout(location = 2) in float p_speed;
layout(location = 3) in float p_phase;
layout(location = 4) in vec3  p_axis;
layout(location = 5) in vec3  p_colour;
layout(location = 6) in float p_size;
layout(location = 7) in float p_twinkle;

uniform mat4  u_view_projection;
uniform float u_time;
uniform float u_viewport_h;
uniform vec3  u_eye;
uniform float u_portrait_cull;

out vec3  f_colour;
out float f_alpha;

// Rodrigues rotation: spin a vector about an arbitrary axis. This is the
// whole motion system — every particle's position is DERIVED here from its
// emitted orbit parameters plus u_time, so nothing is recomputed on the CPU
// and nothing is re-uploaded per frame.
vec3 rotate_about(vec3 v, vec3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

void main() {
    vec3 axis = normalize(p_axis);
    // Any vector not parallel to the axis gives a valid orbital start point;
    // cross with the axis twice to get one lying in the orbit plane.
    vec3 seed = abs(axis.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 radial = normalize(cross(axis, seed)) * p_radius;
    vec3 offset = rotate_about(radial, axis, u_time * p_speed + p_phase);
    vec4 clip = u_view_projection * vec4(p_centre + offset, 1.0);
    gl_Position = clip;

    // Perspective-correct point size: particles shrink with distance like
    // real geometry instead of staying a fixed pixel blob, which is what
    // gives the field genuine depth rather than a flat sprinkle.
    float w = max(clip.w, 0.001);
    // Divisor tuned against a real 1600x900 render at the default zoom: at
    // /12.0 every particle collapsed to under two pixels and the field was
    // invisible. /3.2 puts micro particles at ~4px and halos at ~8px, which
    // is where the cloud actually reads as substance.
    gl_PointSize = clamp(p_size * u_viewport_h / (w * 3.2), 1.0, 22.0);

    float twinkle = p_twinkle > 0.0
        ? 0.55 + 0.45 * sin(u_time * p_twinkle * 6.2831853 + p_phase)
        : 1.0;
    // Fade with distance so the far side of the field recedes instead of
    // reading as noise laid over the near side.
    float depth_fade = clamp(1.0 - (w - 20.0) / 190.0, 0.18, 1.0);
    // In the orb, particles drifting between the viewer and ORION read as
    // dirt on the lens. Faded by the same rule as the nodes they belong to,
    // so the whole field recedes together rather than in layers.
    float near = u_portrait_cull > 0.0
        ? smoothstep(u_portrait_cull * 0.62, u_portrait_cull,
                     distance(u_eye, p_centre + offset))
        : 1.0;

    f_colour = p_colour;
    f_alpha = twinkle * depth_fade * near;
}
"""

_PARTICLE_FRAGMENT_SHADER = """
#version 410 core
in vec3  f_colour;
in float f_alpha;
out vec4 frag_colour;

void main() {
    // Round, soft-edged points. gl_PointCoord is the unit square of the
    // sprite; discarding outside the circle avoids square particles, and the
    // squared falloff gives each one its own small bloom.
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float falloff = 1.0 - (r2 * 4.0);
    float glow = falloff * falloff;
    frag_colour = vec4(f_colour * (0.55 + 0.85 * glow), f_alpha * glow);
}
"""

_P_F = 4
_PARTICLE_STRIDE = PARTICLE_FLOATS * _P_F
_PARTICLE_ATTRIBS = (
    (0, 3, 0 * _P_F),    # centre
    (1, 1, 3 * _P_F),    # radius
    (2, 1, 4 * _P_F),    # speed
    (3, 1, 5 * _P_F),    # phase
    (4, 3, 6 * _P_F),    # axis
    (5, 3, 9 * _P_F),    # colour
    (6, 1, 12 * _P_F),   # size
    (7, 1, 13 * _P_F),   # twinkle
)


_EDGE_VERTEX_SHADER = """
#version 410 core
layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_colour;
layout(location = 2) in float v_t;         // 0.0 at source, 1.0 at target
layout(location = 3) in float v_traffic;   // 0..1 measured activity
layout(location = 4) in float v_seed;      // decorrelates packet phase

uniform mat4 u_view_projection;
uniform float u_time;
uniform vec3 u_eye;
uniform float u_portrait_cull;

out vec3 f_colour;
out float f_t;
out float f_traffic;
out float f_seed;
out float f_near;

void main() {
    gl_Position = u_view_projection * vec4(v_position, 1.0);
    f_colour = v_colour;
    f_t = v_t;
    f_traffic = v_traffic;
    f_seed = v_seed;
    // Edges fade rather than shrink — they are already translucent, and a
    // line drawn straight across ORION's face in the orb is exactly as
    // unwelcome as the node on the end of it.
    f_near = u_portrait_cull > 0.0
        ? smoothstep(u_portrait_cull * 0.62, u_portrait_cull,
                     distance(u_eye, v_position))
        : 1.0;
}
"""

_EDGE_FRAGMENT_SHADER = """
#version 410 core
in vec3 f_colour;
in float f_t;
in float f_traffic;
in float f_seed;
in float f_near;
out vec4 frag_colour;

// Declared in BOTH stages: GLSL scopes uniforms per shader object, so the
// vertex shader declaring it does not make it visible here. Omitting it
// failed the link with "undefined variable u_time" and silently dropped the
// whole edge pass — nodes still drew, so the graph looked merely edgeless
// rather than broken.
uniform float u_time;

void main() {
    // The packet: a bright bead running source -> target. Its position comes
    // entirely from u_time, the per-edge seed and the interpolated t, so the
    // animation costs no CPU work and no re-upload — a busy edge and an idle
    // one differ by a single float.
    float speed = 0.35 + f_traffic * 0.9;
    float head = fract(f_seed + u_time * speed);
    float distance_to_head = abs(f_t - head);
    // wrap, so a packet leaving the far end re-enters at the near one
    distance_to_head = min(distance_to_head, 1.0 - distance_to_head);
    float packet = exp(-distance_to_head * 42.0) * f_traffic;

    // Resting lines stay faint; they are context, not content.
    float base_alpha = 0.16 + f_traffic * 0.42;
    vec3 colour = f_colour * (0.7 + f_traffic * 0.6) + vec3(packet * 0.9);
    frag_colour = vec4(colour, (base_alpha + packet * 0.85) * f_near);
}
"""

_EDGE_F = 4
_EDGE_STRIDE = EDGE_FLOATS * _EDGE_F
_EDGE_ATTRIBS = (
    (0, 3, 0 * _EDGE_F),   # position
    (1, 3, 3 * _EDGE_F),   # colour
    (2, 1, 6 * _EDGE_F),   # t
    (3, 1, 7 * _EDGE_F),   # traffic
    (4, 1, 8 * _EDGE_F),   # seed
)

_TEXT_VERTEX_SHADER = """
#version 410 core
layout(location = 0) in vec2 v_corner;      // unit quad, 0..1
layout(location = 1) in vec2 i_position;    // quad top-left, PIXELS
layout(location = 2) in vec2 i_size;        // quad size, pixels
layout(location = 3) in vec4 i_uv;          // atlas rect (u0, v0, u1, v1)
layout(location = 4) in vec3 i_colour;
layout(location = 5) in float i_alpha;

uniform vec2 u_viewport;

out vec2 f_uv;
out vec3 f_colour;
out float f_alpha;

void main() {
    // Pixels -> NDC here rather than in the geometry builder, so every
    // placement decision upstream stays in coordinates a human can check
    // against a screenshot.
    vec2 px = i_position + v_corner * i_size;
    gl_Position = vec4((px.x / u_viewport.x) * 2.0 - 1.0,
                       1.0 - (px.y / u_viewport.y) * 2.0,
                       0.0, 1.0);
    f_uv = mix(i_uv.xy, i_uv.zw, v_corner);
    f_colour = i_colour;
    f_alpha = i_alpha;
}
"""

_TEXT_FRAGMENT_SHADER = """
#version 410 core
in vec2 f_uv;
in vec3 f_colour;
in float f_alpha;
out vec4 frag_colour;

uniform sampler2D u_atlas;

void main() {
    float d = texture(u_atlas, f_uv).r;
    // fwidth gives the distance field's rate of change across THIS pixel, so
    // the edge is reconstructed at exactly the right softness for whatever
    // size the glyph happens to be drawn at. This is the entire reason the
    // atlas is a distance field rather than a bitmap: one texture, crisp at
    // every scale, with no re-rasterisation.
    float w = max(fwidth(d), 0.0008);
    float edge = smoothstep(0.5 - w, 0.5 + w, d);
    // The halo comes from the same texel — holographic text needs a bloom to
    // sit in a glowing field, and with an SDF that is another threshold on
    // data already fetched rather than a second render pass.
    float glow = smoothstep(0.28, 0.5, d) * 0.42;
    float alpha = max(edge, glow) * f_alpha;
    if (alpha <= 0.004) discard;
    frag_colour = vec4(mix(f_colour * 0.55, f_colour, edge), alpha);
}
"""

_LINE_VERTEX_SHADER = """
#version 410 core
layout(location = 0) in vec2 v_position;    // pixels
layout(location = 1) in vec4 v_colour;

uniform vec2 u_viewport;

out vec4 f_colour;

void main() {
    gl_Position = vec4((v_position.x / u_viewport.x) * 2.0 - 1.0,
                       1.0 - (v_position.y / u_viewport.y) * 2.0,
                       0.0, 1.0);
    f_colour = v_colour;
}
"""

_LINE_FRAGMENT_SHADER = """
#version 410 core
in vec4 f_colour;
out vec4 frag_colour;
void main() { frag_colour = f_colour; }
"""

_T_F = 4
_GLYPH_STRIDE = GLYPH_FLOATS * _T_F
_GLYPH_ATTRIBS = (
    (1, 2, 0 * _T_F),    # position
    (2, 2, 2 * _T_F),    # size
    (3, 4, 4 * _T_F),    # uv rect
    (4, 3, 8 * _T_F),    # colour
    (5, 1, 11 * _T_F),   # alpha
)
_LINE_STRIDE = LINE_FLOATS * _T_F
_LINE_ATTRIBS = (
    (0, 2, 0 * _T_F),    # position
    (1, 4, 2 * _T_F),    # colour
)

# One unit quad, shared by every glyph in the atlas.
_TEXT_QUAD = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
    [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],
], dtype=np.float32)

# Byte offsets of each per-instance attribute within one INSTANCE_FLOATS row.
_F = 4  # sizeof(float32)
_INSTANCE_STRIDE = INSTANCE_FLOATS * _F
_ATTRIBS = (
    # (location, components, byte offset)
    (1, 3, 0 * _F),    # position
    (2, 3, 3 * _F),    # colour
    (3, 1, 6 * _F),    # emissive
    (4, 1, 7 * _F),    # pulse
    (5, 1, 8 * _F),    # radius
    (6, 1, 9 * _F),    # selected
)

_TARGET_FPS = 60

# How far back the camera sits for the compact orb's portrait of ORION.
# Named once and shared by the framing and the near-cull, so the two can never
# drift out of agreement — the cull exists precisely to clear this shot.
PORTRAIT_DISTANCE = FACE_HEAD_RADIUS * 3.15


class _SwarmGLWindow(QOpenGLWidget):  # type: ignore[misc]
    """The renderer proper.

    This was briefly a QOpenGLWindow embedded via createWindowContainer,
    because a QOpenGLWidget and a QWebEngineView could not co-render in one
    window — with the swarm present, ORION's WebEngine face went black.
    Giving the GL side its own native surface fixed that, at the cost of a
    surface that Qt cannot composite widgets onto: no QPainter overlay, and
    QWidget.grab() could not see it.

    That trade is now unnecessary. ORION's face is drawn by this renderer as
    point-cloud geometry, so there is no second GL compositor in the window
    to conflict with, and the surface returns to a plain QOpenGLWidget — one
    renderer, normal Qt compositing, overlays possible again.
    """

    node_selected = pyqtSignal(str)
    node_activated = pyqtSignal(str)

    def __init__(self, scene: SwarmScene | None = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene if scene is not None else SwarmScene()
        self.camera = OrbitCamera()
        self.camera.snap_to_goal()

        self._gl: Any = None
        self._program: Any = None
        self._vao: Any = None
        self._mesh_vbo: Any = None
        self._instance_vbo: Any = None
        self._instance_capacity = 0
        self._mesh_vertices = 0
        self._ready = False
        self._edge_program: Any = None
        self._edge_vao: Any = None
        self._edge_vbo: Any = None
        self._edge_capacity = 0
        self._particle_program: Any = None
        self._particle_vao: Any = None
        self._particle_vbo: Any = None
        self._particle_count = 0
        self._particle_capacity = 0
        self._particle_dirty = False
        self._field_signature: tuple | None = None
        # Adaptive quality: thinned automatically if frames run long, so a
        # slower GPU loses density rather than frame rate.
        self._density = 1.0
        self._slow_frames = 0
        self._face_program: Any = None
        self._face_vao: Any = None
        self._face_vbo: Any = None
        self._face_count = 0
        # ORION himself, inside the scene rather than layered over it.
        self.face = FaceAnimator()
        self._speech_amplitude = 0.0

        # ── the screen-space holographic layer ──────────────────────────────
        # Callout ring, leader lines and the hover readout all draw from one
        # SDF atlas through one text program and one line program, so the
        # entire overlay costs two draw calls however much is on it.
        self._text_program: Any = None
        self._text_vao: Any = None
        self._text_quad_vbo: Any = None
        self._glyph_vbo: Any = None
        self._glyph_capacity = 0
        self._glyph_count = 0
        self._line_program: Any = None
        self._line_vao: Any = None
        self._line_vbo: Any = None
        self._line_capacity = 0
        self._line_count = 0
        self._atlas: Any = None
        self._atlas_texture: Any = None
        self.hover = HoverController()
        self._hover_anchor: tuple[float, float] | None = None
        self._pointer: tuple[float, float] | None = None
        self.labels_enabled = True
        # Portrait: the camera held on ORION's face, which is what the compact
        # overlay orb shows. A camera state, never a second avatar.
        self.portrait_mode = False
        # What sits at the centre of the scene. His mind by default — his face
        # is in the Core Window now. Portrait framing swaps to the face, since
        # a close-up of a brain is not a portrait of anyone.
        self.core_model = "brain"
        self._core_dirty = False
        # Bridges: commands in flight, drawn as packets crossing real edges.
        # Attached from outside so this widget never reaches into ORION.
        self.bridges: Any = None

        self._last_frame = time.perf_counter()
        self._start = self._last_frame
        self._last_paint = self._last_frame
        self._drag_button: Any = None
        self._drag_pos: Any = None
        self._on_status: Callable[[str], None] | None = None

        # A repaint timer rather than continuous redraw: the surface only
        # repaints while something is actually moving (see _tick), so a
        # settled graph costs no GPU at all.
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, 1000 // _TARGET_FPS))
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── public API ───────────────────────────────────────────────────────────

    def attach_status_sink(self, sink: Callable[[str], None]) -> None:
        self._on_status = sink

    def apply_snapshot(self, snapshot: SwarmSnapshot) -> None:
        """Feed a new observation. Cost is proportional to what changed."""
        delta = self.scene.apply(snapshot)
        # The field is re-emitted only when the graph's SHAPE or a node's
        # appearance changed — never for telemetry churn, which is the common
        # case and would otherwise rebuild several thousand particles twice a
        # second for no visible difference.
        if delta.needs_gpu_upload:
            self._particle_dirty = True
        if not delta.is_empty:
            self.update()

    def set_speech_amplitude(self, amplitude: float) -> None:
        """Drive the mouth from the real speech pipeline.

        Stored rather than applied here: the envelope is smoothed per drawn
        frame in FaceAnimator, so bursts of amplitude updates between frames
        cannot make the jaw chatter."""
        try:
            self._speech_amplitude = max(0.0, min(1.0, float(amplitude)))
        except (TypeError, ValueError):
            self._speech_amplitude = 0.0

    def set_speech_spectrum(self, bands: Any) -> None:
        """Feed the voice spectral profile to the mouth so it ARTICULATES —
        wide on vowels, tight on sibilants — instead of just pulsing to
        loudness. Stored on the animator and read on the next drawn frame."""
        try:
            self.face.set_spectrum(bands)
        except Exception:
            pass

    def set_face_mode(self, mode: Any) -> None:
        """Point the avatar's expression at ORION's real cognition mode."""
        self.face.set_mode(mode)

    def look_at(self, x: float, y: float) -> None:
        self.face.look_at(x, y)

    def select(self, node_id: str | None) -> None:
        self.scene.select(node_id)
        self.update()

    def focus_selected(self) -> None:
        node_id = self.scene.selected
        if node_id is None:
            return
        position = self.scene.position_of(node_id)
        if position is not None:
            self.camera.focus_on(position, distance=16.0)

    def reset_camera(self) -> None:
        self.camera.reset()
        self.camera.drift_enabled = False
        self.portrait_mode = False
        self.labels_enabled = True

    def set_portrait_mode(self, enabled: bool) -> None:
        """Frame ORION head-on and fill the view with him.

        This is what the compact overlay orb shows. It is a CAMERA state, not
        a second avatar: the same renderer, the same scene, the same ORION —
        just close enough to see his face. Building a separate face widget for
        the orb is what left it showing nothing at all, because the widget it
        had was the one the unified shell hides.
        """
        enabled = bool(enabled)
        self.portrait_mode = enabled
        if enabled:
            self.camera.face_on(PORTRAIT_DISTANCE)
            # Drift off: a slow orbit would rotate his portrait to the back of
            # his head while the user is looking at it.
            self.camera.drift_enabled = False
            # The callout ring is a legend for the whole field; at portrait
            # distance it would be a wall of text over his face.
            self.labels_enabled = False
            # Any readout already open belongs to the field view behind him.
            self.hover.reset()
            self.set_core_model("face")
        else:
            self.camera.drift_enabled = False
            self.labels_enabled = True
            self.set_core_model("brain")
            self.camera.reset()
        self.update()

    def stats(self) -> dict[str, Any]:
        return self.scene.stats()

    # ── frame loop ───────────────────────────────────────────────────────────

    def _tick(self) -> None:
        # Two guards, both load-bearing: a hidden widget must never drive the
        # GPU (the visibility discipline used across this codebase's decks),
        # and a settled camera needs no repaint at all.
        if not self.isVisible():
            return
        now = time.perf_counter()
        dt = min(0.25, now - self._last_frame)
        self._last_frame = now
        if self.camera.drift_enabled:
            self.camera.cinematic_drift(now - self._start)
        moved = self.camera.update(dt)
        # Pulsing nodes animate in the shader against u_time, so they need a
        # repaint even when nothing else changed.
        if (moved or self.scene.buffer.is_dirty or self.scene.edges.is_dirty
                or self._has_animation()):
            self.update()

    def _has_animation(self) -> bool:
        """True while anything on screen is genuinely moving.

        Particle fields provide static depth while ORION is idle. Motion is
        reserved for an active inspection or real cognition traffic, which
        keeps the neural map calm and lets the GPU rest between events.

        Covers both pulsing nodes and edges carrying traffic, since both
        animate against u_time in their shaders and so need a repaint even
        when no buffer changed."""
        # A readout mid-unfold and a command mid-flight both animate on the
        # CPU, so neither is covered by the shader-driven cases below.
        #
        # A PENDING hover counts too, not just an open one. During the dwell
        # delay the unfold is still exactly 0, so testing only for openness
        # let the repaint loop stop in the gap — and a panel that opens on the
        # next frame never gets a next frame. It would have looked like hover
        # diagnostics simply not working.
        if self.hover.target is not None or self.hover.unfold() > 0.0:
            return True
        if self.bridges is not None and self.bridges.live():
            return True
        buffer = self.scene.buffer
        if buffer.draw_count and bool((buffer.view()[:, 7] > 0.0).any()):
            return True
        edges = self.scene.edges
        if edges.vertex_count and bool((edges.view()[:, 7] > 0.0).any()):
            return True
        return False

    # ── GL lifecycle ─────────────────────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802  (Qt naming)
        from PyQt6.QtGui import QOpenGLContext, QSurfaceFormat
        from PyQt6.QtOpenGL import (
            QOpenGLBuffer,
            QOpenGLShader,
            QOpenGLShaderProgram,
            QOpenGLVersionFunctionsFactory,
            QOpenGLVersionProfile,
            QOpenGLVertexArrayObject,
        )

        context = QOpenGLContext.currentContext()
        profile = QOpenGLVersionProfile()
        profile.setVersion(4, 1)
        profile.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        try:
            self._gl = QOpenGLVersionFunctionsFactory.get(profile, context)
        except Exception:
            self._gl = None
        if self._gl is None:
            # Not fatal to ORION: SwarmDeckView keeps its non-GL fallback, so
            # an old driver or a remote-desktop session degrades to the list
            # view instead of taking the app down.
            self._fail("OpenGL 4.1 core profile unavailable — using fallback view")
            return

        self._gl.initializeOpenGLFunctions()

        program = QOpenGLShaderProgram(self)
        ok = program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX_SHADER)
        ok = ok and program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT_SHADER)
        ok = ok and program.link()
        if not ok:
            self._fail(f"shader compilation failed: {program.log()}")
            return
        self._program = program

        mesh = build_sphere()
        self._mesh_vertices = int(mesh.shape[0])

        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vao.bind()

        self._mesh_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._mesh_vbo.create()
        self._mesh_vbo.bind()
        self._mesh_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        mesh_bytes = np.ascontiguousarray(mesh, dtype=np.float32).tobytes()
        self._mesh_vbo.allocate(mesh_bytes, len(mesh_bytes))
        self._gl.glEnableVertexAttribArray(0)
        # Offset 0 as a plain int, NOT None. PyQt6's binding for the pointer
        # argument dereferences whatever it is given: None does not map to
        # nullptr, it takes the process down with STATUS_STACK_BUFFER_OVERRUN
        # and no Python exception to catch. Verified directly against a live
        # 4.1 core context — int works, None crashes, and sip.voidptr raises
        # "object has an unknown size".
        self._gl.glVertexAttribPointer(0, 3, 0x1406, False, 3 * _F, 0)  # GL_FLOAT

        self._instance_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._instance_vbo.create()
        self._instance_vbo.bind()
        # DynamicDraw: rewritten in place by glBufferSubData, never reallocated
        # per frame — the whole point of the instance buffer's dirty tracking.
        self._instance_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
        self._reserve_instances(max(256, self.scene.buffer.capacity))

        self._vao.release()

        # ── edges: their own program, VAO and buffer ─────────────────────────
        edge_program = QOpenGLShaderProgram(self)
        ok = edge_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _EDGE_VERTEX_SHADER)
        ok = ok and edge_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _EDGE_FRAGMENT_SHADER)
        ok = ok and edge_program.link()
        if ok:
            self._edge_program = edge_program
            self._edge_vao = QOpenGLVertexArrayObject(self)
            self._edge_vao.create()
            self._edge_vao.bind()
            self._edge_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._edge_vbo.create()
            self._edge_vbo.bind()
            self._edge_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
            self._reserve_edges(max(512, self.scene.edges.capacity))
            self._edge_vao.release()
        else:
            # Nodes without edges is a degraded but perfectly usable graph;
            # losing the whole page over the connective tissue would not be.
            self._fail(f"edge shader failed, drawing nodes only: {edge_program.log()}")

        # ── the neural field ────────────────────────────────────────────────
        particle_program = QOpenGLShaderProgram(self)
        ok = particle_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _PARTICLE_VERTEX_SHADER)
        ok = ok and particle_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _PARTICLE_FRAGMENT_SHADER)
        ok = ok and particle_program.link()
        if ok:
            self._particle_program = particle_program
            self._particle_vao = QOpenGLVertexArrayObject(self)
            self._particle_vao.create()
            self._particle_vao.bind()
            self._particle_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._particle_vbo.create()
            self._particle_vbo.bind()
            self._particle_vbo.setUsagePattern(
                QOpenGLBuffer.UsagePattern.StaticDraw)
            self._particle_vao.release()
            self._particle_dirty = True
        else:
            self._fail(f"particle shader failed, field disabled: "
                       f"{particle_program.log()}")

        # ── ORION's face, as scene geometry ─────────────────────────────────
        face_program = QOpenGLShaderProgram(self)
        ok = face_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _FACE_VERTEX_SHADER)
        ok = ok and face_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _FACE_FRAGMENT_SHADER)
        ok = ok and face_program.link()
        if ok:
            self._face_program = face_program
            self._face_vao = QOpenGLVertexArrayObject(self)
            self._face_vao.create()
            self._face_vao.bind()
            self._face_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._face_vbo.create()
            self._face_vbo.bind()
            # StaticDraw: the cloud is uploaded ONCE and animated entirely by
            # uniforms — the whole reason the avatar is affordable here.
            self._face_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
            self._upload_core_cloud()
            for location, components, offset in _FACE_ATTRIBS:
                self._gl.glEnableVertexAttribArray(location)
                self._gl.glVertexAttribPointer(
                    location, components, 0x1406, False, _FACE_STRIDE, int(offset))
            self._face_vao.release()
        else:
            # A graph without ORION at the centre is still usable; losing the
            # whole page over the avatar would not be.
            self._fail(f"face shader failed, avatar disabled: {face_program.log()}")

        # ── the overlay: holographic text and leader lines ──────────────────
        self._init_overlay(QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram,
                           QOpenGLVertexArrayObject)

        self._gl.glEnable(0x0B71)        # GL_DEPTH_TEST
        self._gl.glEnable(0x0BE2)        # GL_BLEND — edges are translucent
        self._gl.glBlendFunc(0x0302, 0x0303)   # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
        self._gl.glEnable(0x8642)        # GL_PROGRAM_POINT_SIZE

        # GL_POINT_SPRITE, on a compatibility context only.
        #
        # ORION and his whole particle field are GL_POINTS, and both fragment
        # shaders shape the point with gl_PointCoord. In a CORE profile point
        # sprites are always on and gl_PointCoord is always defined. In a
        # COMPATIBILITY profile it is undefined until this is enabled — so
        # every fragment failed the discard test and both passes rendered
        # nothing at all, while the triangle and line passes were unaffected.
        #
        # That is exactly what the running app did: nodes, edges and labels
        # present, ORION and the field simply absent. It never showed up in
        # any harness here because every harness requested 4.1 core, and
        # app.py requests no format at all — so Qt handed it 4.6 compatibility.
        # Enabling it unconditionally is not an option: in a core profile the
        # same call raises GL_INVALID_ENUM.
        if self._is_compatibility_profile(context):
            self._gl.glEnable(0x8861)    # GL_POINT_SPRITE
        self._ready = True
        # The driver's copy is empty; force a full upload of whatever the
        # scene already holds.
        self.scene.buffer.mark_all_dirty()
        self.scene.edges.mark_all_dirty()

    # ── the overlay layer ────────────────────────────────────────────────────

    def _init_overlay(self, QOpenGLBuffer: Any, QOpenGLShader: Any,
                      QOpenGLShaderProgram: Any,
                      QOpenGLVertexArrayObject: Any) -> None:
        """Text and line programs, plus the SDF atlas texture.

        Every failure here is soft. Losing labels leaves a perfectly readable
        graph; taking the page down over typography would not be a trade
        anyone would choose."""
        gl = self._gl
        if gl is None:
            return

        text_program = QOpenGLShaderProgram(self)
        ok = text_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _TEXT_VERTEX_SHADER)
        ok = ok and text_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _TEXT_FRAGMENT_SHADER)
        ok = ok and text_program.link()
        if not ok:
            self._fail(f"text shader failed, labels disabled: {text_program.log()}")
        else:
            self._text_program = text_program
            self._text_vao = QOpenGLVertexArrayObject(self)
            self._text_vao.create()
            self._text_vao.bind()

            self._text_quad_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._text_quad_vbo.create()
            self._text_quad_vbo.bind()
            self._text_quad_vbo.setUsagePattern(
                QOpenGLBuffer.UsagePattern.StaticDraw)
            quad = np.ascontiguousarray(_TEXT_QUAD, dtype=np.float32).tobytes()
            self._text_quad_vbo.allocate(quad, len(quad))
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 2, 0x1406, False, 2 * _T_F, 0)

            self._glyph_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._glyph_vbo.create()
            self._glyph_vbo.bind()
            self._glyph_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
            self._reserve_glyphs(1024)
            self._text_vao.release()

        line_program = QOpenGLShaderProgram(self)
        ok = line_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _LINE_VERTEX_SHADER)
        ok = ok and line_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _LINE_FRAGMENT_SHADER)
        ok = ok and line_program.link()
        if not ok:
            self._fail(f"leader-line shader failed: {line_program.log()}")
        else:
            self._line_program = line_program
            self._line_vao = QOpenGLVertexArrayObject(self)
            self._line_vao.create()
            self._line_vao.bind()
            self._line_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._line_vbo.create()
            self._line_vbo.bind()
            self._line_vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
            self._reserve_lines(512)
            self._line_vao.release()

        self._build_atlas_texture()

    def _reserve_glyphs(self, capacity: int) -> None:
        if self._glyph_vbo is None or self._gl is None:
            return
        self._glyph_vbo.bind()
        self._glyph_vbo.allocate(int(capacity) * _GLYPH_STRIDE)
        for location, components, offset in _GLYPH_ATTRIBS:
            self._gl.glEnableVertexAttribArray(location)
            self._gl.glVertexAttribPointer(
                location, components, 0x1406, False, _GLYPH_STRIDE, int(offset))
            # One quad per glyph, advanced per INSTANCE — which is what makes
            # every label in the scene a single draw call.
            self._gl.glVertexAttribDivisor(location, 1)
        self._glyph_capacity = int(capacity)

    def _reserve_lines(self, capacity: int) -> None:
        if self._line_vbo is None or self._gl is None:
            return
        self._line_vbo.bind()
        self._line_vbo.allocate(int(capacity) * _LINE_STRIDE)
        for location, components, offset in _LINE_ATTRIBS:
            self._gl.glEnableVertexAttribArray(location)
            self._gl.glVertexAttribPointer(
                location, components, 0x1406, False, _LINE_STRIDE, int(offset))
        self._line_capacity = int(capacity)

    def _build_atlas_texture(self) -> None:
        """Rasterise the glyph atlas once and upload it.

        Uploaded through QOpenGLTexture from a QImage rather than a raw
        glTexImage2D call: PyQt6's binding for that pointer argument is the
        same one that took the process down with a null offset elsewhere in
        this file, and there is no reason to go near it when Qt will do the
        allocation itself. The distance field is quantised to 8 bits on the
        way, which is far more precision than a 3.5-texel spread needs.
        """
        try:
            from PyQt6.QtGui import QImage
            from PyQt6.QtOpenGL import QOpenGLTexture

            from .text_atlas import rasterise_atlas

            atlas = rasterise_atlas()
            field = np.ascontiguousarray(
                np.clip(atlas.image, 0.0, 1.0) * 255.0).astype(np.uint8)
            height, width = field.shape
            payload = field.tobytes()
            image = QImage(payload, width, height, width,
                           QImage.Format.Format_Grayscale8).copy()

            # setData(QImage) does the whole setup itself — create, format,
            # size, mip levels, allocate. Doing any of that by hand first
            # makes it complain on every launch that it "cannot change format
            # once storage has been allocated" and cannot resize or set mip
            # levels either. The image is the single source of all four.
            texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            texture.setData(image, QOpenGLTexture.MipMapGeneration.DontGenerateMipMaps)
            # Linear, no mipmaps: an SDF is meant to be filtered linearly and
            # then thresholded; mipmapping it averages distances across glyph
            # boundaries and softens small text into mush.
            texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
            self._atlas = atlas
            self._atlas_texture = texture
        except Exception as error:      # pragma: no cover — driver dependent
            self._atlas = None
            self._atlas_texture = None
            self._fail(f"glyph atlas unavailable, labels disabled: {error}")

    def _callout_entries(self) -> list[tuple[str, str, tuple[float, float, float]]]:
        """Which nodes are worth labelling.

        The twelve cluster anchors, plus whatever is selected or under the
        pointer. Labelling all seventeen hundred nodes would be a wall of
        text; labelling the anchors turns the ring into a legend for the whole
        field, and the selected node is the one the user is asking about."""
        snapshot = self.scene.snapshot
        if snapshot is None:
            return []
        wanted = {self.scene.selected, self.hover.target}
        entries = []
        for node in snapshot.nodes:
            if not (node.is_cluster or node.kind is NodeKind.CORE
                    or node.id in wanted):
                continue
            position = self.scene.position_of(node.id)
            if position is not None:
                entries.append((node.id, node.label.upper(), position))
        return entries

    def _hover_anchor_for(self, node_id: str | None,
                          aspect: float) -> tuple[float, float] | None:
        """Where on screen the hovered node currently is.

        Recomputed every frame from the live position rather than cached from
        the mouse event, so the readout stays tethered to its node as the
        rings turn underneath it."""
        if not node_id:
            return None
        position = self.scene.position_of(node_id)
        if position is None:
            return None
        screen, depth = project_to_screen(
            np.array([position], dtype=np.float64),
            self.camera.view_projection(aspect), self.width(), self.height())
        if depth[0] <= 0.0:
            return None
        return (float(screen[0, 0]), float(screen[0, 1]))

    def portrait_cull_distance(self) -> float:
        """How close to the camera a thing may be before it gets out of the way.

        0.0 whenever the whole field is on show — the cull exists only for the
        compact orb, where the camera sits close enough that a cluster orbiting
        between it and ORION fills a third of the frame. Derived from the
        camera's own distance and the size of his head, so it stays correct if
        either is retuned rather than being a magic number that silently stops
        matching the framing.
        """
        if not self.portrait_mode:
            return 0.0
        # Bounded by the settled portrait framing, not by wherever the camera
        # happens to be mid-flight. Entering the orb eases in from the whole
        # field's distance, and a cull scaled to THAT would briefly exceed the
        # radius of the entire graph — every node would dissolve on the way in
        # and swim back as the camera arrived.
        distance = min(float(self.camera.distance), PORTRAIT_DISTANCE)
        # Clears everything level with him or in front of him, leaving only
        # what is genuinely BEHIND his head. A tighter band (his front surface)
        # still let a cluster sit beside his cheek and fill a corner of the
        # frame, and this window is meant to be his face — the field reading as
        # depth behind him, not as company beside him.
        return max(1.0, distance - FACE_HEAD_RADIUS * 0.35)

    def _set_portrait_uniforms(self, program: Any, eye: Any) -> None:
        """Feed the cull to whichever pass is binding. Passes that do not
        declare the uniforms get location -1, which GL ignores."""
        program.setUniformValue(program.uniformLocation("u_eye"),
                                float(eye[0]), float(eye[1]), float(eye[2]))
        program.setUniformValue(program.uniformLocation("u_portrait_cull"),
                                float(self.portrait_cull_distance()))

    def _bridge_marks(self, aspect: float) -> list[Any]:
        """Every command in flight, projected to the screen.

        The packet's world position is interpolated between the two nodes it
        is crossing, resolved fresh each frame — so a command keeps tracking
        its route even as the cognition rings turn beneath it, which a
        precomputed screen path could not do."""
        if self.bridges is None:
            return []
        pulses = self.bridges.pulses()
        if not pulses:
            return []

        from .overlay import BRIDGE_TAIL, BridgeMark

        world: list[tuple[float, float, float]] = []
        keep: list[Any] = []
        for pulse in pulses:
            source = self.scene.position_of(pulse.source)
            target = self.scene.position_of(pulse.target)
            if source is None or target is None:
                # A route can name a node that is not in the current snapshot
                # (a workflow removed mid-flight). Dropping the packet is
                # correct: drawing it at the origin would invent a location.
                continue
            t = pulse.t
            world.append((source[0] + (target[0] - source[0]) * t,
                          source[1] + (target[1] - source[1]) * t,
                          source[2] + (target[2] - source[2]) * t))
            keep.append(pulse)
        if not keep:
            return []

        screen, depth = project_to_screen(
            np.array(world, dtype=np.float64),
            self.camera.view_projection(aspect), self.width(), self.height())

        marks = []
        for index, pulse in enumerate(keep):
            if depth[index] <= 0.0:
                continue            # behind the camera
            head = (float(screen[index, 0]), float(screen[index, 1]))
            holding = pulse.source == pulse.target
            failed = pulse.phase.value == "failed"
            tail = head
            if not holding:
                # The tail trails back along the segment the packet is
                # crossing, in screen space, so the streak always points the
                # way the command is actually going.
                back = self.scene.position_of(pulse.source)
                back_screen, _ = project_to_screen(
                    np.array([back], dtype=np.float64),
                    self.camera.view_projection(aspect),
                    self.width(), self.height())
                dx = head[0] - float(back_screen[0, 0])
                dy = head[1] - float(back_screen[0, 1])
                length = math.hypot(dx, dy) or 1.0
                reach = min(BRIDGE_TAIL, length)
                tail = (head[0] - dx / length * reach,
                        head[1] - dy / length * reach)
            marks.append(BridgeMark(head=head, tail=tail,
                                    intensity=pulse.intensity,
                                    holding=holding and not failed,
                                    failed=failed, label=pulse.label))
        return marks

    def _build_overlay(self, aspect: float) -> None:
        """Compose this frame's text and lines, and upload them."""
        if self._atlas is None or self._gl is None:
            self._glyph_count = 0
            self._line_count = 0
            return

        width, height = float(self.width()), float(self.height())
        callouts = []
        if self.labels_enabled:
            callouts = layout_callouts(
                self._callout_entries(), self.camera.view_projection(aspect),
                width, height)

        open_node = self.hover.open_node()
        node = self.scene.node(open_node) if open_node else None
        anchor = self._hover_anchor_for(open_node, aspect)
        panel = self.hover.panel(node) if node is not None else None
        # Keep the last known anchor while a panel folds away, so it collapses
        # towards its node instead of jumping to the origin.
        if anchor is not None:
            self._hover_anchor = anchor
        anchor = anchor or self._hover_anchor

        geometry = compose_overlay(self._atlas, callouts, panel, anchor,
                                   (width, height), self._bridge_marks(aspect))
        self._upload_overlay(geometry)

    def _upload_overlay(self, geometry: Any) -> None:
        gl = self._gl
        self._glyph_count = geometry.glyph_count
        self._line_count = geometry.line_vertex_count

        if self._glyph_vbo is not None and self._glyph_count:
            if self._glyph_count > self._glyph_capacity:
                self._text_vao.bind()
                self._reserve_glyphs(max(self._glyph_count * 2, 1024))
                self._text_vao.release()
            payload = np.ascontiguousarray(
                geometry.glyphs, dtype=np.float32).tobytes()
            self._glyph_vbo.bind()
            self._glyph_vbo.write(0, payload, len(payload))

        if self._line_vbo is not None and self._line_count:
            if self._line_count > self._line_capacity:
                self._line_vao.bind()
                self._reserve_lines(max(self._line_count * 2, 512))
                self._line_vao.release()
            payload = np.ascontiguousarray(
                geometry.lines, dtype=np.float32).tobytes()
            self._line_vbo.bind()
            self._line_vbo.write(0, payload, len(payload))
        del gl

    def _draw_overlay(self) -> None:
        """The last pass: screen-space, depth-free, additive-free.

        Straight alpha rather than additive blending — additive text over a
        bright cluster washes out to white and becomes unreadable exactly
        where the field is busiest, which is where labels matter most."""
        gl = self._gl
        if gl is None:
            return
        viewport = (float(max(1, self.width())), float(max(1, self.height())))
        gl.glDisable(0x0B71)          # GL_DEPTH_TEST — the overlay is on top
        gl.glBlendFunc(0x0302, 0x0303)

        if self._line_program is not None and self._line_count:
            self._line_vao.bind()
            self._line_program.bind()
            self._line_program.setUniformValue(
                self._line_program.uniformLocation("u_viewport"), *viewport)
            gl.glDrawArrays(0x0001, 0, self._line_count)   # GL_LINES
            self._line_program.release()
            self._line_vao.release()

        if (self._text_program is not None and self._glyph_count
                and self._atlas_texture is not None):
            self._text_vao.bind()
            self._text_program.bind()
            self._atlas_texture.bind(0)
            self._text_program.setUniformValue(
                self._text_program.uniformLocation("u_viewport"), *viewport)
            self._text_program.setUniformValue(
                self._text_program.uniformLocation("u_atlas"), 0)
            gl.glDrawArraysInstanced(0x0004, 0, 6, self._glyph_count)
            self._atlas_texture.release(0)
            self._text_program.release()
            self._text_vao.release()

        gl.glEnable(0x0B71)

    def core_cloud(self) -> Any:
        """The point cloud at the centre of the scene.

        A BRAIN by default. ORION's face lives in the Core Window now, drawn
        by face3d, which freed the middle of the swarm to be the thing the
        graph is actually about — his mind, with every subsystem radiating out
        of it rather than out of a portrait.

        Both models emit the same vertex format, so switching between them is
        one buffer upload and no shader change whatsoever.
        """
        if self.core_model == "face":
            return build_face()
        from .brain_geometry import build_brain
        return build_brain()

    def _upload_core_cloud(self) -> None:
        """(Re)upload the core model. Static: written once per model change,
        never per frame — the whole reason it is affordable in this pass."""
        if self._face_vbo is None:
            return
        cloud = self.core_cloud()
        self._face_count = int(cloud.shape[0])
        payload = np.ascontiguousarray(cloud, dtype=np.float32).tobytes()
        self._face_vbo.bind()
        self._face_vbo.allocate(payload, len(payload))
        self._core_dirty = False

    def set_core_model(self, model: str) -> None:
        """Swap what sits at the centre — "brain" or "face"."""
        model = "face" if str(model).strip().lower() == "face" else "brain"
        if model == self.core_model:
            return
        self.core_model = model
        # Re-uploaded on the next paint, where a GL context is current. Doing
        # it here would touch the buffer from whatever thread or call stack
        # asked for the swap.
        self._core_dirty = True
        self.update()

    @staticmethod
    def _is_compatibility_profile(context: Any) -> bool:
        """Whether Qt handed us a compatibility rather than a core context.

        Qt gives a QOpenGLWidget whatever the default QSurfaceFormat asks for,
        and app.py asks for nothing — so the profile is the driver's choice
        and cannot be assumed. Reported defensively: a binding that does not
        expose the profile must not take the renderer down."""
        try:
            from PyQt6.QtGui import QSurfaceFormat
            profile = context.format().profile()
            return profile == QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile
        except Exception:      # pragma: no cover — binding dependent
            return False

    def _reserve_edges(self, edge_capacity: int) -> None:
        if self._edge_vbo is None or self._gl is None:
            return
        self._edge_vbo.bind()
        self._edge_vbo.allocate(edge_capacity * 2 * _EDGE_STRIDE)
        for location, components, offset in _EDGE_ATTRIBS:
            self._gl.glEnableVertexAttribArray(location)
            self._gl.glVertexAttribPointer(
                location, components, 0x1406, False, _EDGE_STRIDE,
                int(offset))
        self._edge_capacity = edge_capacity
        self.scene.edges.mark_all_dirty()

    def _rebuild_field(self) -> None:
        """Re-emit the particle field from the CURRENT graph.

        Only ever called when the graph's SHAPE changed (or density was
        adapted) — never per frame. All motion afterwards happens in the
        vertex shader, so this buffer is written once and then simply drawn.
        """
        if self._particle_vbo is None:
            return
        scene = self.scene
        snapshot = scene.snapshot
        if snapshot is None:
            return
        positions = {n.id: scene.position_of(n.id) or (0.0, 0.0, 0.0)
                     for n in snapshot.nodes}
        # Reuse the SAME colours the spheres are drawn with, so a node's cloud
        # is unmistakably its own rather than a generic haze.
        from .palette import appearance_for
        colours = {n.id: appearance_for(n).colour for n in snapshot.nodes}

        field = build_field(list(snapshot.nodes), positions, colours,
                            density=self._density)
        data = np.ascontiguousarray(field.data, dtype=np.float32)
        self._particle_count = int(data.shape[0])

        self._particle_vao.bind()
        self._particle_vbo.bind()
        payload = data.tobytes()
        if self._particle_count > self._particle_capacity or not payload:
            self._particle_vbo.allocate(payload, len(payload))
            self._particle_capacity = self._particle_count
            for location, components, offset in _PARTICLE_ATTRIBS:
                self._gl.glEnableVertexAttribArray(location)
                self._gl.glVertexAttribPointer(
                    location, components, 0x1406, False, _PARTICLE_STRIDE,
                    int(offset))
        else:
            self._particle_vbo.write(0, payload, len(payload))
        self._particle_vao.release()
        self._particle_dirty = False

    def _upload_dirty_edges(self) -> None:
        edges = self.scene.edges
        if not edges.is_dirty or self._edge_vbo is None:
            return
        if edges.capacity > self._edge_capacity:
            self._reserve_edges(edges.capacity)
        span = edges.dirty_vertex_range()
        if span is None:
            return
        start, end = span
        chunk = np.ascontiguousarray(edges.view()[start:end], dtype=np.float32)
        self._edge_vbo.bind()
        self._edge_vbo.write(start * _EDGE_STRIDE, chunk.tobytes(), chunk.nbytes)
        edges.clear_dirty()

    def _reserve_instances(self, capacity: int) -> None:
        """(Re)allocate the instance VBO and rebind its attribute pointers."""
        if self._instance_vbo is None or self._gl is None:
            return
        self._instance_vbo.bind()
        self._instance_vbo.allocate(capacity * _INSTANCE_STRIDE)
        for location, components, offset in _ATTRIBS:
            self._gl.glEnableVertexAttribArray(location)
            self._gl.glVertexAttribPointer(
                location, components, 0x1406, False, _INSTANCE_STRIDE,
                int(offset))
            # divisor 1 => advance once per INSTANCE, not per vertex
            self._gl.glVertexAttribDivisor(location, 1)
        self._instance_capacity = capacity
        self.scene.buffer.mark_all_dirty()

    def _upload_dirty(self) -> None:
        buffer = self.scene.buffer
        if not buffer.is_dirty or self._instance_vbo is None:
            return
        if buffer.capacity > self._instance_capacity:
            self._reserve_instances(buffer.capacity)
        span = buffer.dirty_range()
        if span is None:
            return
        start, end = span
        chunk = np.ascontiguousarray(buffer.view()[start:end], dtype=np.float32)
        self._instance_vbo.bind()
        self._instance_vbo.write(start * _INSTANCE_STRIDE, chunk.tobytes(),
                                 chunk.nbytes)
        buffer.clear_dirty()

    def paintGL(self) -> None:  # noqa: N802
        if not self._ready or self._gl is None or self._program is None:
            return
        gl = self._gl
        # Deep space with the faintest violet cast — black enough that the
        # field's own glow carries the image, not a lit background.
        gl.glClearColor(0.012, 0.010, 0.026, 1.0)
        gl.glClear(0x00004000 | 0x00000100)   # COLOR_BUFFER_BIT | DEPTH_BUFFER_BIT

        now = time.perf_counter()
        elapsed = float(now - self._start)
        # The face is advanced from the PAINT delta, not the tick delta: it
        # animates per drawn frame, and driving it from a timer that can fire
        # without a repaint would make the expression jump.
        dt_for_face = min(0.25, max(0.0, now - self._last_paint))
        self._last_paint = now
        aspect_now = max(1e-4, self.width() / max(1, self.height()))
        eye_now = self.camera.eye()

        # Field first: it is the deepest layer, and drawing it with depth
        # WRITES disabled lets the nodes and edges sit correctly in front of
        # it without the particles punching holes in each other.
        if self._particle_program is not None:
            if self._particle_dirty:
                self._rebuild_field()
            if self._particle_count:
                gl.glDepthMask(False)
                gl.glBlendFunc(0x0302, 0x0001)      # SRC_ALPHA, ONE — additive
                self._particle_vao.bind()
                self._particle_program.bind()
                self._particle_program.setUniformValue(
                    self._particle_program.uniformLocation("u_view_projection"),
                    self._qmatrix(aspect_now))
                self._particle_program.setUniformValue(
                    self._particle_program.uniformLocation("u_time"), elapsed)
                self._particle_program.setUniformValue(
                    self._particle_program.uniformLocation("u_viewport_h"),
                    float(max(1, self.height())))
                self._set_portrait_uniforms(self._particle_program, eye_now)
                gl.glDrawArrays(0x0000, 0, self._particle_count)  # GL_POINTS
                self._particle_program.release()
                self._particle_vao.release()
                gl.glBlendFunc(0x0302, 0x0303)      # back to straight alpha
                gl.glDepthMask(True)

        # Edges first: they are translucent context behind the nodes, and
        # drawing them after would blend them over the spheres.
        if self._edge_program is not None and self.scene.edges.vertex_count:
            self._edge_vao.bind()
            self._upload_dirty_edges()
            self._edge_program.bind()
            self._edge_program.setUniformValue(
                self._edge_program.uniformLocation("u_view_projection"),
                self._qmatrix(aspect_now))
            self._edge_program.setUniformValue(
                self._edge_program.uniformLocation("u_time"), elapsed)
            self._set_portrait_uniforms(self._edge_program, eye_now)
            gl.glDrawArrays(0x0001, 0, self.scene.edges.vertex_count)  # GL_LINES
            self._edge_program.release()
            self._edge_vao.release()

        # ORION himself. Drawn with depth WRITES on, before the node spheres,
        # so the network genuinely occludes and is occluded by him — the
        # thing that was impossible while the face was a layer on top.
        if self._face_program is not None and self._face_count:
            expression = self.face.update(dt_for_face, self._speech_amplitude)
            self._face_vao.bind()
            if self._core_dirty:
                self._upload_core_cloud()
            self._face_program.bind()
            self._face_program.setUniformValue(
                self._face_program.uniformLocation("u_view_projection"),
                self._qmatrix(aspect_now))
            self._face_program.setUniformValue(
                self._face_program.uniformLocation("u_time"), elapsed)
            self._face_program.setUniformValue(
                self._face_program.uniformLocation("u_viewport_h"),
                float(max(1, self.height())))
            # The lighting needs to know where the viewer is: without it there
            # is no way to tell the front of the head from the back, and every
            # far-side point prints through the face. (ORION himself is never
            # culled by the portrait rule — he is what it is protecting.)
            self._set_portrait_uniforms(self._face_program, eye_now)
            for name, value in expression.as_uniforms().items():
                self._face_program.setUniformValue(
                    self._face_program.uniformLocation(name), float(value))
            gl.glDrawArrays(0x0000, 0, self._face_count)   # GL_POINTS
            self._face_program.release()
            self._face_vao.release()

        draw_count = self.scene.buffer.draw_count
        if draw_count <= 0:
            self._build_overlay(aspect_now)
            self._draw_overlay()
            return

        self._vao.bind()
        self._upload_dirty()
        self._program.bind()
        # QMatrix4x4 via setUniformValue rather than a raw glUniformMatrix4fv
        # with a bytes payload: PyQt6's binding for the latter varies in what
        # pointer types it accepts, and getting it subtly wrong yields a
        # silently blank scene rather than an error.
        self._program.setUniformValue(
            self._program.uniformLocation("u_view_projection"),
            self._qmatrix(aspect_now))
        self._program.setUniformValue(
            self._program.uniformLocation("u_time"), elapsed)
        self._set_portrait_uniforms(self._program, eye_now)

        # ONE call for the entire graph, regardless of node count.
        gl.glDrawArraysInstanced(0x0004, 0, self._mesh_vertices, draw_count)

        self._program.release()
        self._vao.release()

        # Last, and on top of everything: the holographic layer.
        self._build_overlay(aspect_now)
        self._draw_overlay()

    def _qmatrix(self, aspect: float) -> Any:
        """The camera's view-projection as a QMatrix4x4.

        QMatrix4x4's constructor takes values in ROW-major order, which is the
        orientation camera.view_projection() already produces — so this passes
        the un-transposed matrix, unlike gl_view_projection() which flattens
        column-major for the raw GL path."""
        from PyQt6.QtGui import QMatrix4x4
        values = self.camera.view_projection(aspect).astype(np.float32).ravel()
        return QMatrix4x4(*[float(v) for v in values])

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        if self._gl is not None:
            self._gl.glViewport(0, 0, max(1, w), max(1, h))

    def _fail(self, message: str) -> None:
        self._ready = False
        if self._on_status is not None:
            try:
                self._on_status(f"SWARM: {message}")
            except Exception:
                pass

    # ── input ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        self._drag_button = event.button()
        self._drag_pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            node_id = self._pick(event.position().x(), event.position().y())
            self.scene.select(node_id)
            if node_id:
                self.node_selected.emit(node_id)
            self.update()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        position = event.position()
        if self._drag_pos is None:
            # Not dragging: this is a hover. Picking on every move is
            # affordable because it is a vectorised pass over the instance
            # buffer (see pick.py) — microseconds, no GPU round trip.
            self._pointer = (position.x(), position.y())
            if self.portrait_mode:
                # Nodes near the camera are culled to nothing in the orb, but
                # picking still sees them — a readout would unfold for
                # something the user cannot see, anchored to empty space.
                if self.hover.clear():
                    self.update()
                return
            if self.hover.point_at(self._pick(position.x(), position.y())):
                self.update()
            return
        dx = position.x() - self._drag_pos.x()
        dy = position.y() - self._drag_pos.y()
        self._drag_pos = position
        if self._drag_button == Qt.MouseButton.LeftButton:
            self.camera.orbit(dx * 0.008, -dy * 0.008)
        elif self._drag_button in (Qt.MouseButton.MiddleButton,
                                   Qt.MouseButton.RightButton):
            self.camera.pan(dx, dy)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._pointer = None
        if self.hover.clear():
            self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        self._drag_button = None
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        node_id = self._pick(event.position().x(), event.position().y())
        if node_id:
            self.scene.select(node_id)
            self.focus_selected()
            self.node_activated.emit(node_id)

    def wheelEvent(self, event: Any) -> None:
        self.camera.zoom_by_steps(event.angleDelta().y() / 120.0)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() in {Qt.Key.Key_R, Qt.Key.Key_Home}:
            self.reset_camera()
        elif event.key() == Qt.Key.Key_F:
            self.focus_selected()
        else:
            super().keyPressEvent(event)

    def _pick(self, x: float, y: float) -> str | None:
        buffer = self.scene.buffer
        if buffer.draw_count == 0:
            return None
        aspect = max(1e-4, self.width() / max(1, self.height()))
        origin, direction = ray_from_screen(
            x, y, self.width(), self.height(),
            self.camera.view_projection(aspect), self.camera.eye())
        slot = pick_slot(buffer.view(), origin, direction, buffer.draw_count)
        return None if slot is None else buffer.id_at(slot)


class SwarmGLView(QWidget):
    """Widget wrapper around the native GL window.

    Everything outside this file talks to SwarmGLView and never touches the
    QOpenGLWindow, so the surface-vs-widget distinction (see _SwarmGLWindow's
    docstring — it is what stops the swarm blanking ORION's face) stays an
    implementation detail. The API is proxied rather than inherited because a
    QWidget and a QWindow share no useful base.
    """

    node_selected = pyqtSignal(str)
    node_activated = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None,
                 scene: SwarmScene | None = None) -> None:
        super().__init__(parent)
        # A plain child widget now, not a native window container — so Qt can
        # composite over it again (which is what the callout text layer
        # needs) and QWidget.grab() can capture it.
        self._window = _SwarmGLWindow(scene, self)
        self._window.setMinimumSize(320, 240)
        self._window.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._window.setMouseTracking(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._window)
        self.setMinimumSize(320, 240)
        self._window.node_selected.connect(self.node_selected)
        self._window.node_activated.connect(self.node_activated)

    # ── proxied API ──────────────────────────────────────────────────────────

    @property
    def scene(self) -> SwarmScene:
        return self._window.scene

    @property
    def camera(self) -> OrbitCamera:
        return self._window.camera

    def attach_status_sink(self, sink: Callable[[str], None]) -> None:
        self._window.attach_status_sink(sink)

    # ── ORION's face, now inside the scene ───────────────────────────────────

    @property
    def face(self) -> Any:
        return self._window.face

    def set_speech_amplitude(self, amplitude: float) -> None:
        self._window.set_speech_amplitude(amplitude)

    def set_speech_spectrum(self, bands: Any) -> None:
        self._window.set_speech_spectrum(bands)

    def set_face_mode(self, mode: Any) -> None:
        self._window.set_face_mode(mode)

    def look_at(self, x: float, y: float) -> None:
        self._window.look_at(x, y)

    def apply_snapshot(self, snapshot: SwarmSnapshot) -> None:
        self._window.apply_snapshot(snapshot)

    def select(self, node_id: str | None) -> None:
        self._window.select(node_id)

    def focus_selected(self) -> None:
        self._window.focus_selected()

    def reset_camera(self) -> None:
        self._window.reset_camera()

    def stats(self) -> dict[str, Any]:
        return self._window.stats()

    # ── the holographic overlay ──────────────────────────────────────────────

    @property
    def hover(self) -> Any:
        """The hover-diagnostics controller, so a host can drive or read it."""
        return self._window.hover

    def attach_bridges(self, ledger: Any) -> None:
        """Show commands in flight from *ledger*.

        Injected rather than constructed here: the renderer must not know how
        ORION dispatches anything, and a headless test can drive the same
        ledger with no GL context at all."""
        self._window.bridges = ledger

    @property
    def bridges(self) -> Any:
        return self._window.bridges

    def set_labels_enabled(self, enabled: bool) -> None:
        self._window.labels_enabled = bool(enabled)
        self._window.update()

    def set_portrait_mode(self, enabled: bool) -> None:
        """Fill the view with ORION's face — what the compact orb shows."""
        self._window.set_portrait_mode(enabled)

    @property
    def portrait_mode(self) -> bool:
        return bool(self._window.portrait_mode)


