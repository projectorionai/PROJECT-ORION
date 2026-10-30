"""Teaching pace follows the persona across ordinary, lean and fallback paths.

Construct renderers without initialisers so these tests do not read or write the
owner's preferences, connect to providers, or start background services.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.cyber_knowledge import SECURITY_TEACHING_DIRECTIVE
from orion_core.identity import CORE_IDENTITY, IdentityManager
from orion_core.learning_style import LEARNING_STYLE_DIRECTIVE
from orion_core.providers import ProviderRouter


def _identity() -> IdentityManager:
    manager = IdentityManager.__new__(IdentityManager)
    manager.core = CORE_IDENTITY
    manager.preferences = {"honorific": "sir", "honorific_frequency": "occasional"}
    return manager


def _router(identity=None) -> ProviderRouter:
    router = ProviderRouter.__new__(ProviderRouter)
    router._identity = identity
    router.memory = SimpleNamespace(prompt_context=lambda **kwargs: "")
    router._with_self_knowledge = lambda persona: persona
    for name in ("_temporal_line", "_environment_line", "_emotion_line", "_listener_line"):
        setattr(router, name, lambda: "")
    return router


@pytest.mark.parametrize("render", ["persona_text", "style_capsule"])
@pytest.mark.parametrize("context", ["", "coding"])
def test_identity_keeps_learning_pace_and_security_boundaries(render, context):
    text = getattr(_identity(), render)(context=context)
    assert LEARNING_STYLE_DIRECTIVE in text
    assert SECURITY_TEACHING_DIRECTIVE in text
    assert text.index(SECURITY_TEACHING_DIRECTIVE) < text.index(LEARNING_STYLE_DIRECTIVE)


@pytest.mark.parametrize("render", ["system_instruction", "system_instruction_lean"])
@pytest.mark.parametrize("has_identity", [False, True])
def test_provider_paths_keep_learner_control_without_provider_initialisation(render, has_identity):
    router = _router(_identity() if has_identity else None)
    text = getattr(router, render)(context="coding")
    assert LEARNING_STYLE_DIRECTIVE in text
    assert SECURITY_TEACHING_DIRECTIVE in text
    assert text.count(LEARNING_STYLE_DIRECTIVE) == 1


def test_broken_identity_renderer_falls_back_with_the_same_teaching_pace():
    def unavailable(**kwargs):
        raise RuntimeError("identity temporarily unavailable")

    router = _router(SimpleNamespace(persona_text=unavailable))
    text = router.system_instruction_lean()
    assert LEARNING_STYLE_DIRECTIVE in text
    assert SECURITY_TEACHING_DIRECTIVE in text


def test_directive_preserves_explicit_detail_requests_and_non_medical_scope():
    text = LEARNING_STYLE_DIRECTIVE
    for requirement in (
        "one small concept per turn",
        "learner's explicit next",
        "Honour explicit requests for full detail",
        "at most one optional check or question",
        "brief recap",
        "Do not assume or record a medical diagnosis or health label",
        "Never shame disengagement",
        "timers or reminders",
        "show its code as text",
    ):
        assert requirement in text


def test_directive_uses_persistent_coaching_actions_and_retains_full_lessons():
    for action in (
        "coach_start", "coach_next", "coach_answer", "coach_simpler",
        "coach_repeat", "coach_pause", "coach_resume", "coach_status",
    ):
        assert action in LEARNING_STYLE_DIRECTIVE
    assert "Full teach/code actions remain available" in LEARNING_STYLE_DIRECTIVE
