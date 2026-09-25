"""
Static checks on the renderer's GLSL (Mark XXIII).

These exist because this suite has twice shipped a shader that only failed on
a real GPU. The edge pass linked fine in review and died at runtime with
"undefined variable u_time", because GLSL scopes uniforms PER SHADER OBJECT
and the fragment stage never declared what it used — the graph simply drew
without edges, which looks like a design choice rather than a fault. The same
class of mistake in a varying name or an attribute's component count produces
a blank pass or garbled geometry with no exception anywhere.

A GL context cannot be created here, so this parses the shader sources
instead and checks the invariants a linker would: declared-before-used
uniforms, matching varyings across stages, and attribute layouts that agree
with the Python side that feeds them.

This is not a substitute for looking at the screen. It is a floor.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from orion_core.swarm.render import gl_view as g  # noqa: E402

# (name, vertex source, fragment source, python attribute table, extra
# locations fed from a separate non-instanced buffer)
PROGRAMS = [
    ("nodes", g._VERTEX_SHADER, g._FRAGMENT_SHADER, g._ATTRIBS, {0: 3}),
    ("edges", g._EDGE_VERTEX_SHADER, g._EDGE_FRAGMENT_SHADER, g._EDGE_ATTRIBS, {}),
    ("particles", g._PARTICLE_VERTEX_SHADER, g._PARTICLE_FRAGMENT_SHADER,
     g._PARTICLE_ATTRIBS, {}),
    ("face", g._FACE_VERTEX_SHADER, g._FACE_FRAGMENT_SHADER, g._FACE_ATTRIBS, {}),
    ("text", g._TEXT_VERTEX_SHADER, g._TEXT_FRAGMENT_SHADER, g._GLYPH_ATTRIBS,
     {0: 2}),
    ("lines", g._LINE_VERTEX_SHADER, g._LINE_FRAGMENT_SHADER, g._LINE_ATTRIBS, {}),
]

_COMPONENTS = {"float": 1, "vec2": 2, "vec3": 3, "vec4": 4}

# The comma form (`uniform float u_blink, u_mouth;`) is used heavily by the
# face shader, so a one-name-per-declaration pattern would report every one of
# those as undeclared.
_UNIFORM = re.compile(r"^\s*uniform\s+(\w+)\s+([\w\s,]+?)\s*;", re.MULTILINE)
_LAYOUT = re.compile(
    r"^\s*layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*in\s+(\w+)\s+(\w+)\s*;",
    re.MULTILINE)
_OUT = re.compile(r"^\s*out\s+(\w+)\s+(\w+)\s*;", re.MULTILINE)
_IN = re.compile(r"^\s*in\s+(\w+)\s+(\w+)\s*;", re.MULTILINE)


def _strip_declarations(source: str) -> str:
    """The body, with declarations removed, so a name found here is a USE."""
    for pattern in (_UNIFORM, _LAYOUT, _OUT, _IN):
        source = pattern.sub("", source)
    return source


def _uses(source: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", _strip_declarations(source)) is not None


@pytest.mark.parametrize("name,vertex,fragment,_attribs,_extra", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_every_uniform_used_is_declared_in_that_stage(name, vertex, fragment,
                                                      _attribs, _extra):
    """The bug that silently deleted the entire edge pass.

    GLSL scopes uniforms per shader object: the vertex shader declaring
    u_time does NOT make it visible to the fragment shader."""
    for stage, source in (("vertex", vertex), ("fragment", fragment)):
        declared = {part.strip()
                    for m in _UNIFORM.finditer(source)
                    for part in m.group(2).split(",")}
        used = set(re.findall(r"\bu_\w+\b", _strip_declarations(source)))
        missing = used - declared
        assert not missing, f"{name}/{stage} uses undeclared {sorted(missing)}"


@pytest.mark.parametrize("name,vertex,fragment,_attribs,_extra", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_varyings_match_across_the_two_stages(name, vertex, fragment,
                                              _attribs, _extra):
    """A name or type that differs between stages fails the link, and the
    whole pass vanishes rather than erroring anywhere visible."""
    produced = {m.group(2): m.group(1) for m in _OUT.finditer(vertex)}
    consumed = {m.group(2): m.group(1)
                for m in _IN.finditer(fragment)
                if m.group(2) != "frag_colour"}
    for varying, kind in consumed.items():
        assert varying in produced, f"{name}: fragment reads unproduced {varying}"
        assert produced[varying] == kind, (
            f"{name}: {varying} is {produced[varying]} out, {kind} in")


@pytest.mark.parametrize("name,vertex,fragment,_attribs,_extra", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_the_fragment_stage_writes_exactly_one_output(name, vertex, fragment,
                                                      _attribs, _extra):
    outputs = [m.group(2) for m in _OUT.finditer(fragment)]
    assert outputs == ["frag_colour"], f"{name}: fragment outputs {outputs}"


@pytest.mark.parametrize("name,vertex,fragment,attribs,extra", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_attribute_layouts_agree_with_the_python_that_feeds_them(
        name, vertex, fragment, attribs, extra):
    """A component-count mismatch between the shader and the attribute table
    is silent: the geometry simply comes out wrong."""
    declared = {int(m.group(1)): _COMPONENTS[m.group(2)]
                for m in _LAYOUT.finditer(vertex)
                if m.group(2) in _COMPONENTS}
    expected = {int(location): int(components)
                for location, components, _offset in attribs}
    expected.update(extra)
    assert declared == expected, f"{name}: shader {declared} vs python {expected}"


@pytest.mark.parametrize("name,vertex,fragment,attribs,_extra", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_attribute_locations_are_unique(name, vertex, fragment, attribs, _extra):
    locations = [int(m.group(1)) for m in _LAYOUT.finditer(vertex)]
    assert len(locations) == len(set(locations)), f"{name}: duplicate locations"


@pytest.mark.parametrize("name,vertex,fragment,_a,_e", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_every_stage_declares_the_core_profile(name, vertex, fragment, _a, _e):
    """A missing #version defaults to GLSL 1.10, where none of this compiles."""
    for source in (vertex, fragment):
        assert source.lstrip().startswith("#version 410 core")


@pytest.mark.parametrize("name,vertex,fragment,_a,_e", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_every_stage_has_a_main(name, vertex, fragment, _a, _e):
    for source in (vertex, fragment):
        assert re.search(r"\bvoid\s+main\s*\(", source)


# ── the offsets the Python side writes must tile its own row ─────────────────

@pytest.mark.parametrize("name,_v,_f,attribs,_e", PROGRAMS,
                         ids=[p[0] for p in PROGRAMS])
def test_attribute_offsets_tile_the_row_without_gaps_or_overlap(
        name, _v, _f, attribs, _e):
    """Every float in the row is claimed by exactly one attribute — an
    off-by-one here shifts every field after it."""
    cursor = 0
    for _location, components, offset in attribs:
        assert offset == cursor * 4, f"{name}: attribute at {offset}, expected {cursor * 4}"
        cursor += components


def test_the_text_program_reads_the_atlas():
    """The whole SDF system is pointless if the sampler is never fetched."""
    assert _uses(g._TEXT_FRAGMENT_SHADER, "u_atlas")
    assert "fwidth" in g._TEXT_FRAGMENT_SHADER, (
        "an SDF without a screen-space derivative is just a blurry bitmap")


def test_the_overlay_converts_pixels_to_ndc_in_the_shader():
    """Placement stays in pixels everywhere upstream so it can be checked
    against a screenshot; only the shader leaves that space."""
    for source in (g._TEXT_VERTEX_SHADER, g._LINE_VERTEX_SHADER):
        assert _uses(source, "u_viewport")
