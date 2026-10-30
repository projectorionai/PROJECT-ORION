"""Typed control refusals survive verification without replaying an action."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.execution_policy import background_scope, require_foreground
from orion_core.verification import VerificationResult, VisualVerificationEngine


def _engine():
    engine = object.__new__(VisualVerificationEngine)
    captures = []
    async def capture(region):
        captures.append(region)
        return object()
    engine._capture = capture
    engine._change_ratio = lambda *_args: pytest.fail("Refused action was visually verified")
    engine.telemetry = None
    engine.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_args: None))
    engine.trace = SimpleNamespace(record=lambda **_kwargs: None)
    engine.display = SimpleNamespace(clamp_to_desktop=lambda x, y: (x, y))
    return engine, captures


def _assert_deferral(result):
    assert not result.ok
    assert result.receipt["status"] == "requires_foreground"
    assert result.receipt["desktop_mutated"] is False
    assert result.receipt["origin"] == "test-verification"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("repeatable", [False, True])
async def test_generic_verification_preserves_deferral_without_retry_or_after_capture(asynchronous, repeatable):
    engine, captures = _engine()
    calls = []
    def synchronous_action():
        calls.append(True)
        return require_foreground("synthetic_action")
    async def async_action():
        return synchronous_action()
    with background_scope("test-verification"):
        verified = await engine.verify_action(
            async_action if asynchronous else synchronous_action,
            attempts=5, repeatable=repeatable,
        )
    result = verified.to_tool_result()
    _assert_deferral(result)
    assert verified.attempts == 1 and verified.change_ratio == 0.0
    assert calls == [True]
    assert captures == [None]


@pytest.mark.parametrize("region,repeatable", [(None, True), ((1, 2, 30, 40), False)])
async def test_dispatcher_verification_bridge_retains_typed_deferral(region, repeatable):
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    engine, captures = _engine()
    dispatcher = DesktopDispatchMixin()
    dispatcher.verifier = engine
    calls = []
    def action():
        calls.append(True)
        return require_foreground("synthetic_action")
    with background_scope("test-verification"):
        result = await dispatcher._maybe_verify(action, True, region=region, repeatable=repeatable)
    _assert_deferral(result)
    assert calls == [True]
    assert captures == [region]


@pytest.mark.parametrize("method,selector", [("click_text", ""), ("click_text", "observed-id"),
                                             ("click_element", "observed-id")])
async def test_named_click_keeps_deferral_and_never_starts_scroll_or_ocr_fallback(method, selector):
    engine, captures = _engine()
    clicks = []
    queries = []
    element = {"center": (50, 60), "rect": (40, 50, 20, 20),
               "name": "Synthetic target", "role": "button"}
    async def resolve(query, kinds, automation_id):
        queries.append((query, kinds, automation_id))
        return {"status": "found", "element": element}
    def click(*args, **kwargs):
        clicks.append((args, kwargs))
        return require_foreground("synthetic_click")
    engine.vision = SimpleNamespace(resolve_element=resolve,
        find_element_via_ocr=lambda *_a, **_k: pytest.fail("Refusal fell back to OCR"))
    engine.control = SimpleNamespace(click=click,
        scroll=lambda *_a: pytest.fail("Refusal started a scroll search"))
    with background_scope("test-verification"):
        result = await getattr(engine, method)("Synthetic target", attempts=5, automation_id=selector)
    _assert_deferral(result)
    assert result.target_found is True
    assert len(clicks) == len(captures) == len(queries) == 1


def test_verification_result_receipt_is_snapshotted_before_and_after_conversion():
    original = ToolResult("Deferred", ok=False, receipt={
        "status": "requires_foreground", "desktop_mutated": False,
        "detail": {"effect": "synthetic"},
    })
    verified = VerificationResult(False, 0.0, 1, original.text, receipt=original.receipt)
    original.receipt["detail"]["effect"] = "changed"
    result = verified.to_tool_result()
    assert result.receipt["detail"]["effect"] == "synthetic"
    verified.receipt["detail"]["effect"] = "changed-again"
    assert result.receipt["detail"]["effect"] == "synthetic"


def test_legacy_verification_result_constructor_keeps_existing_contract():
    result = VerificationResult(True, 0.2, 1, "Verified").to_tool_result()
    assert result.ok and result.text == "Verified" and result.receipt is None
