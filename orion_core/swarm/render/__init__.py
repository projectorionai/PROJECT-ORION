"""
orion_core.swarm.render — the native GPU renderer (Mark XXII, Phase 1).

Replaces the Three.js-in-QWebEngineView swarm. That approach worked, but it
carried three defects no amount of optimisation could remove: it loaded
Three.js from cdn.jsdelivr.net (so the swarm was blank without internet), it
ran in a separate Chromium process (so "one renderer, one event loop, one
memory space" was structurally unreachable), and every update crossed a JSON
bridge (170 KB per tick at the 1,000-agent target).

Feasibility was verified before committing: PyQt6 exposes
QOpenGLFunctions_4_1_Core with glDrawArraysInstanced and glVertexAttribDivisor
in BOTH runtimes ORION uses — the .venv and the WindowsApps interpreter that
actually launches orion.py — so GPU instancing needs no new dependency in
either.

The layering exists because of a lesson this codebase learned painfully: a
test that constructs a real GPU/browser context is a test that hangs or
crashes the suite. So everything with logic lives in pure, numpy-only modules
that never touch Qt or GL, and the QOpenGLWidget is a thin shell that owns
nothing but the GL handles:

    palette.py    node appearance from kind/health/activity   pure
    layout.py     deterministic cluster-orbit positions       pure
    camera.py     orbit camera, easing, view/projection       pure
    instances.py  the instance buffer + slot pool + dirty     pure
    gl_view.py    QOpenGLWidget — GL calls only               thin

Everything above gl_view is fully unit-tested without a display.
"""

from __future__ import annotations

from .camera import OrbitCamera
from .instances import INSTANCE_FLOATS, InstanceBuffer
from .layout import LayoutSolver, Vec3
from .palette import appearance_for, cluster_colour

__all__ = [
    "INSTANCE_FLOATS",
    "InstanceBuffer",
    "LayoutSolver",
    "OrbitCamera",
    "Vec3",
    "appearance_for",
    "cluster_colour",
]
