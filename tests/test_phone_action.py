"""
phone_action tool — ORION hands a native action to the paired phone.

Verifies the tool validates input and emits a well-formed directive on
bus.phone_action (which the remote gateway fans out to connected phones).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatcher import OrionDispatcher


class _Rec:
    def __init__(self) -> None:
        self.emitted: list = []

    def emit(self, *payload) -> None:
        self.emitted.append(payload[0] if len(payload) == 1 else payload)


class _Bus:
    def __init__(self) -> None:
        self.phone_action = _Rec()
        self.log = _Rec()

    def __getattr__(self, name):
        rec = _Rec()
        object.__setattr__(self, name, rec)
        return rec


def _dispatcher():
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _Bus()
    return d


def test_phone_action_emits_call_directive():
    d = _dispatcher()
    result = d.phone_action({"kind": "call", "number": "+447700900123"})
    assert result.ok
    assert d.bus.phone_action.emitted == [{"kind": "call", "number": "+447700900123"}]


def test_phone_action_text_alias_maps_to_sms():
    d = _dispatcher()
    result = d.phone_action({"kind": "text", "number": "123", "body": "on my way"})
    assert result.ok
    payload = d.bus.phone_action.emitted[0]
    assert payload["kind"] == "sms" and payload["body"] == "on my way"


def test_phone_action_requires_number_for_call():
    d = _dispatcher()
    result = d.phone_action({"kind": "call"})
    assert not result.ok
    assert d.bus.phone_action.emitted == []


def test_phone_action_rejects_unknown_kind():
    d = _dispatcher()
    result = d.phone_action({"kind": "telepathy"})
    assert not result.ok
    assert d.bus.phone_action.emitted == []


def test_phone_action_navigate_needs_query():
    d = _dispatcher()
    assert not d.phone_action({"kind": "navigate"}).ok
    ok = d.phone_action({"kind": "navigate", "query": "Birmingham New Street"})
    assert ok.ok
    assert d.bus.phone_action.emitted[0] == {
        "kind": "navigate", "query": "Birmingham New Street"}
