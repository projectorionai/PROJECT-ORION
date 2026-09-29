"""tools/setup_twilio.py: what it verifies and what it writes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import setup_twilio  # noqa: E402

pytest.importorskip("httpx")

SID = "AC" + "0" * 32


class _Reply:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code, self._body = status, body

    def json(self) -> dict:
        return self._body


def _twilio(monkeypatch, *, numbers_status=200, numbers=("+447700900123",)):
    import httpx

    def get(url, **_kwargs):
        if url.endswith(f"/Accounts/{SID}.json"):
            return _Reply(200, {"friendly_name": "Test account"})
        if "IncomingPhoneNumbers" in url:
            if numbers_status != 200:
                return _Reply(numbers_status, {"code": 20003, "message": "Forbidden"})
            return _Reply(200, {"incoming_phone_numbers": [
                {"phone_number": n} for n in numbers]})
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "get", get)
    saved = []
    monkeypatch.setattr(setup_twilio.direct, "save_credentials",
                        lambda sid, token, number: saved.append(number) or Path("x"))
    return saved


def test_a_number_typed_with_spaces_matches_and_is_saved_dialable(monkeypatch):
    saved = _twilio(monkeypatch)
    assert setup_twilio.setup(SID, "token-value", "+44 7700 900-123") == 0
    assert saved == ["+447700900123"]


def test_a_number_not_in_international_form_writes_nothing(monkeypatch):
    saved = _twilio(monkeypatch)
    assert setup_twilio.setup(SID, "token-value", "07700 900123") == 1
    assert saved == []


def test_a_refused_number_listing_is_not_read_as_no_numbers(monkeypatch, capsys):
    """An error reply has no number list. It used to read as 'this account has
    no phone numbers yet' and block credentials that had just verified."""
    saved = _twilio(monkeypatch, numbers_status=403)
    assert setup_twilio.setup(SID, "token-value", "+447700900123") == 0
    assert saved == ["+447700900123"]


def test_a_number_that_is_not_on_the_account_writes_nothing(monkeypatch, capsys):
    saved = _twilio(monkeypatch, numbers=("+447700900999",))
    assert setup_twilio.setup(SID, "token-value", "+447700900123") == 1
    assert saved == []
    assert "not on this account" in capsys.readouterr().out
