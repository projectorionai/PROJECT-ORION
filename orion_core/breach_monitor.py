"""
Credential-breach awareness (Security Sentinel Expansion).

``BreachMonitor`` lets ORION answer "has this password appeared in a known data
breach?" and "has this email been in a breach?" using Have I Been Pwned — a
defensive, privacy-first capability.

Privacy invariants (enforced by design, covered by tests):

    • Password checks use HIBP's **k-anonymity** range API.  The password is
      hashed with SHA-1 *locally*; only the first FIVE hex characters of that
      hash ever leave the machine.  The plaintext password and the full hash are
      never transmitted, never written to a log, never persisted.  The response
      is a list of hash *suffixes* + counts that we match locally.
    • The email check hits the authenticated HIBP breachedaccount endpoint and
      therefore needs an API key (``config/api_keys.json → integrations.hibp`` or
      ``ORION_HIBP_API_KEY``).  With no key it degrades gracefully to an
      actionable hint instead of failing.
    • This module makes network calls **only on explicit request**.  The passive
      ``SecuritySentinel`` loop never calls it, so that watcher keeps its
      "never touches the network" guarantee.

Nothing here can be used to attack an account — it only tells the *owner* whether
their own credential is already exposed, which is standard security hygiene.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")

from .bus import OrionBus
from .data import ToolResult

_PWNED_RANGE_URL = "https://api.pwnedpasswords.com/range/{prefix}"
_HIBP_ACCOUNT_URL = "https://haveibeenpwned.com/api/v3/breachedaccount/{account}"
_USER_AGENT = "ORION-SecuritySentinel"


def severity_for(count: int) -> str:
    """Map a breach appearance count to a plain-language severity band."""
    if count <= 0:
        return "none"
    if count < 100:
        return "low"
    if count < 10_000:
        return "moderate"
    if count < 1_000_000:
        return "high"
    return "critical"


def parse_pwned_range(body: str, suffix: str) -> int:
    """
    Parse a HIBP range response (``SUFFIX:COUNT`` lines) and return the breach
    count for our hash suffix, or 0 if absent.  Case-insensitive, tolerant of
    blank lines and CRLF.  Pure function — the unit tests hit this directly.
    """
    target = suffix.strip().upper()
    for line in body.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        got, _, count = line.partition(":")
        if got.strip().upper() == target:
            try:
                return int(count.strip().replace(",", ""))
            except ValueError:
                return 0
    return 0


class BreachMonitor:
    """On-demand credential-breach checks over Have I Been Pwned."""

    def __init__(self, bus: OrionBus, config: Any | None = None,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.config = config          # ProviderConfig-like; optional
        self.telemetry = telemetry

    # ── key resolution ────────────────────────────────────────────────────────

    def _hibp_api_key(self) -> str:
        env = os.getenv("ORION_HIBP_API_KEY", "").strip()
        if env:
            return env
        try:
            integrations = (self.config.integrations if self.config is not None else {}) or {}
            hibp = integrations.get("hibp") or integrations.get("haveibeenpwned") or {}
            if isinstance(hibp, str):
                return hibp.strip()
            return str(hibp.get("api_key") or hibp.get("token") or "").strip()
        except Exception:
            return ""

    # ── password breach (k-anonymity, offline-safe) ───────────────────────────

    async def check_password(self, password: str) -> ToolResult:
        """
        Report how many times a password has appeared in known breaches, without
        ever sending the password or its full hash.  ``password`` is consumed
        immediately and never stored.
        """
        if not password:
            return ToolResult("Give me a password to check and I'll test it "
                              "privately — only a partial hash ever leaves this machine.", ok=False)
        digest = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
        prefix, suffix = digest[:5], digest[5:]
        # Do not keep the plaintext around a moment longer than needed.
        del password, digest
        body = await self._get_text(_PWNED_RANGE_URL.format(prefix=prefix),
                                    headers={"Add-Padding": "true"})
        if body is None:
            return ToolResult("I couldn't reach the breach database just now — "
                              "check the connection and try again. Your password was "
                              "not sent anywhere.", ok=False)
        count = parse_pwned_range(body, suffix)
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("security.breach.password_check")
            except Exception:
                pass
        if count <= 0:
            return ToolResult("Good news: that password does not appear in any "
                              "known breach corpus. Still, never reuse it across sites.")
        sev = severity_for(count)
        self.bus.banner.emit(f"🛡 Password found in breaches ({count:,}×, {sev})", 5)
        return ToolResult(
            f"That password has appeared in known breaches {count:,} time(s) — "
            f"severity {sev}. Treat it as compromised: change it anywhere you use it "
            f"and switch to a unique passphrase. (Only a 5-character hash prefix was "
            f"ever transmitted.)")

    # ── account breach (needs API key) ─────────────────────────────────────────

    async def check_account(self, account: str) -> ToolResult:
        """List the breaches a given email/username appears in (HIBP v3)."""
        account = (account or "").strip()
        if not account:
            return ToolResult("Give me an email address or username to check.", ok=False)
        key = self._hibp_api_key()
        if not key:
            return ToolResult(
                "Account breach lookup needs a Have I Been Pwned API key. Add it as "
                "integrations.hibp.api_key in config/api_keys.json (or set "
                "ORION_HIBP_API_KEY), then ask again. The password check needs no key.",
                ok=False)
        url = _HIBP_ACCOUNT_URL.format(account=account)
        status, data = await self._get_json_with_status(
            url + "?truncateResponse=false",
            headers={"hibp-api-key": key})
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("security.breach.account_check")
            except Exception:
                pass
        if status == 404:
            return ToolResult(f"Clean: {account} does not appear in any breach "
                              "tracked by Have I Been Pwned.")
        if status == 401:
            return ToolResult("The Have I Been Pwned API key was rejected (401). "
                              "Check the key value.", ok=False)
        if status == 429:
            return ToolResult("Have I Been Pwned rate-limited the request (429). "
                              "Give it a moment and try again.", ok=False)
        if status != 200 or not isinstance(data, list):
            return ToolResult("I couldn't complete the account breach lookup just now.",
                              ok=False)
        names = [str(b.get("Name") or b.get("Title") or "?") for b in data]
        self.bus.banner.emit(f"🛡 {account} in {len(names)} breach(es)", 5)
        head = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        return ToolResult(
            f"{account} appears in {len(names)} known breach(es): {head}. "
            f"Change the password anywhere you reused it and enable MFA.")

    # ── HTTP helpers (match geo.py hygiene) ────────────────────────────────────

    async def _get_text(self, url: str, *, headers: dict[str, str] | None = None,
                        timeout: float = 12.0) -> str | None:
        try:
            hdrs = {"User-Agent": _USER_AGENT}
            hdrs.update(headers or {})
            client_timeout = ClientTimeout(total=timeout, connect=5.0)
            async with ClientSession(timeout=client_timeout) as session:
                async with session.get(url, headers=hdrs) as response:
                    if response.status != 200:
                        return None
                    return await response.text()
        except Exception as exc:
            self.bus.log.emit(f"BREACH: request failed - {str(exc).splitlines()[0][:100]}")
            return None

    async def _get_json_with_status(self, url: str, *, headers: dict[str, str] | None = None,
                                    timeout: float = 12.0) -> tuple[int, Any]:
        try:
            hdrs = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
            hdrs.update(headers or {})
            client_timeout = ClientTimeout(total=timeout, connect=5.0)
            async with ClientSession(timeout=client_timeout) as session:
                async with session.get(url, headers=hdrs) as response:
                    if response.status != 200:
                        return response.status, None
                    return 200, await response.json()
        except Exception as exc:
            self.bus.log.emit(f"BREACH: request failed - {str(exc).splitlines()[0][:100]}")
            return 0, None
