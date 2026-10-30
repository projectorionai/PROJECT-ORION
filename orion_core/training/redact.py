"""
Redaction for training data.

Everything that goes into a training set passes through ``Redactor`` first.
It replaces personal strings with fixed placeholders - the same placeholder
every time, so redaction is deterministic, idempotent, and two rows that
differed only by a phone number become the same row and de-duplicate:

    <EMAIL>   e-mail addresses
    <PHONE>   UK and international telephone numbers
    <SECRET>  API keys, tokens, JWTs, passwords given as key=value, private
              keys, long opaque strings
    <CARD>    payment card numbers (Luhn-valid)
    <IBAN>    bank account numbers in IBAN form
    <POSTCODE> UK postcodes
    <IP>      IPv4 addresses (loopback is kept)
    <HOME>    the user's profile folder in a path (``C:\\Users\\name`` ...)
    <USER>    the operating-system user name and any extra terms given

It also offers ``sensitive_topic`` - a coarser test for text that should not
be trained on at all, redacted or not (someone dictating a password, a PIN, a
sort code). Those rows are dropped, not scrubbed: a placeholder in the middle
of "my PIN is ..." still teaches the model that it is a thing to talk about.

This module reads no files and makes no network calls. It does read the
operating-system user name (``getpass``) so that the owner's own account name
never reaches a dataset.
"""

from __future__ import annotations

import getpass
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

EMAIL = "<EMAIL>"
PHONE = "<PHONE>"
SECRET = "<SECRET>"
CARD = "<CARD>"
IBAN = "<IBAN>"
POSTCODE = "<POSTCODE>"
IP = "<IP>"
HOME = "<HOME>"
USER = "<USER>"

PLACEHOLDERS: frozenset[str] = frozenset(
    {EMAIL, PHONE, SECRET, CARD, IBAN, POSTCODE, IP, HOME, USER})

#: User names that are ordinary words or system accounts: replacing them would
#: damage plain text without protecting anyone.
_COMMON_NAMES = frozenset({
    "user", "users", "admin", "administrator", "owner", "orion", "public", "default",
    "guest", "test", "root", "system", "home", "me", "pc",
})

# ── patterns ─────────────────────────────────────────────────────────────────

_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b")
_PREFIXED = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-[A-Za-z0-9_-]{16,}"
    r"|sk_(?:live|test)_[A-Za-z0-9]{12,}"
    r"|pk_(?:live|test)_[A-Za-z0-9]{12,}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|AQ\.[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{30,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|ya29\.[A-Za-z0-9_-]{20,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|AC[0-9a-f]{32}"
    r")(?![A-Za-z0-9_-])")
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{16,}")
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")
_URL_SECRET_PARAM = re.compile(
    r"(?i)([?&](?:[a-z_]*(?:token|key|secret|sig|signature|password|passwd|auth)[a-z_]*|code)=)"
    r"[^&\s#\"'<>]+")
_KEY_VALUE = re.compile(
    r"(?i)\b((?:api[_ -]?key|secret(?:[_ -]?key)?|client[_ -]?secret|access[_ -]?(?:key|token)|"
    r"auth(?:orization)?(?:[_ -]?token)?|token|password|passwd|pwd|passphrase)\s*[:=]\s*"
    r"|(?:password|passwd|pwd|passphrase|passcode|secret|api[_ -]?key)\s+is\s+)"
    r"([\"']?)([^\s\"',;<>]{4,})\2")
_EMAIL_RE = re.compile(r"(?i)(?<![A-Za-z0-9._%+-])[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
_PHONE_INTL = re.compile(
    r"(?<![\w.])\+\d{1,3}[\s.-]?(?:\(0\)[\s.-]?)?\d{1,4}(?:[\s.-]?\d{2,4}){2,4}(?![\w])")
_PHONE_UK = re.compile(
    r"(?<![\w.:/+-])(?:0\d{4}[ -]?\d{6}|0\d{3}[ -]?\d{3}[ -]?\d{4}|0\d{2}[ -]?\d{4}[ -]?\d{4}"
    r"|07\d{3}[ -]?\d{3}[ -]?\d{3}|07\d{9}|0\d{10})(?![\w])")
_CARD_RE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?[ ]?\d[A-Z]{2}\b")
_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_OPAQUE = re.compile(r"(?<![A-Za-z0-9_\\/.:-])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])")
_HEX_LONG = re.compile(r"(?<![A-Za-z0-9_\\/.:-])[0-9a-fA-F]{32,}(?![A-Za-z0-9_-])")
#: ``C:\Users\name`` / ``C:/Users/name`` / ``/c/Users/name`` / ``/home/name`` /
#: ``/Users/name`` - the profile folder, not what is inside it.
_PROFILE_PATH = re.compile(
    r"(?i)(?:\b[a-z]:[\\/]+users[\\/]+|(?<![\w])/(?:c/)?users/|(?<![\w])/home/)"
    r"(?!<)(?:[^\\/\s\"'<>|:*?]+)")
_ONEDRIVE = re.compile(r"(?i)\bOneDrive - [^\\/\r\n\"'<>|]+")

#: Text that should not be trained on at all.
_SENSITIVE = re.compile(
    r"(?i)\b(?:"
    r"pass(?:word|code|phrase)s?"
    r"|my pin(?: number| code)?\b"
    r"|pin (?:number|code)"
    r"|sort code|account number|card number|cvv|cvc|security code"
    r"|national insurance|ni number|social security|\bssn\b|passport number"
    r"|iban|bank details|bank account|routing number"
    r"|one[- ]time (?:code|password)|\botp\b|2fa code|recovery code|backup code|verification code"
    r"|seed phrase|recovery phrase|private key|secret key|api key"
    r")\b")


#: The UK postcode areas, so "MP3 4GB" is not a postcode.
_POSTCODE_AREAS = frozenset(
    "AB AL B BA BB BD BH BL BN BR BS BT CA CB CF CH CM CO CR CT CV CW DA DD DE DG DH DL DN DT DY "
    "E EC EH EN EX FK FY G GL GU GY HA HD HG HP HR HS HU HX IG IM IP IV JE KA KT KW KY L LA LD LE "
    "LL LN LS LU M ME MK ML N NE NG NN NP NR NW OL OX PA PE PH PL PO PR RG RH RM S SA SE SG SK SL "
    "SM SN SO SP SR SS ST SW SY TA TD TF TN TQ TR TS TW UB W WA WC WD WF WN WR WS WV YO ZE".split())


def _postcode_ok(text: str) -> bool:
    area = re.match(r"[A-Z]{1,2}", text)
    return bool(area) and area.group(0) in _POSTCODE_AREAS


def _looks_like_name(token: str) -> bool:
    """File and project names ('Annual_Report_2026_Final_Draft') are long and
    mixed but built from words; a key is not."""
    parts = [p for p in re.split(r"[_-]", token) if p]
    if len(parts) < 3:
        return False
    wordy = sum(1 for p in parts if re.fullmatch(r"[A-Za-z]{2,}", p) or re.fullmatch(r"\d{1,4}", p))
    return wordy / len(parts) >= 0.7


def luhn_valid(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total, flip = 0, False
    for n in reversed(nums):
        if flip:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        flip = not flip
    return total % 10 == 0


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    size = len(text)
    return -sum(c / size * math.log2(c / size) for c in counts.values())


def sensitive_topic(text: str) -> str:
    """The first sensitive phrase in *text*, or '' (the phrase is a topic
    word, not the owner's data)."""
    found = _SENSITIVE.search(str(text or ""))
    return found.group(0).lower() if found else ""


def _dedupe_terms(terms: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for raw in terms:
        term = str(raw or "").strip()
        if len(term) >= 3 and term.casefold() not in _COMMON_NAMES:
            seen.setdefault(term.casefold(), term)
    return sorted(seen.values(), key=lambda t: (-len(t), t.casefold()))


def environment_terms() -> list[str]:
    """The operating-system account names that identify the owner."""
    names: list[str] = []
    for getter in (getpass.getuser, lambda: Path.home().name,
                   lambda: os.environ.get("USERNAME", ""),
                   lambda: os.environ.get("USER", "")):
        try:
            names.append(str(getter() or ""))
        except Exception:
            continue
    return _dedupe_terms(names)


class Redactor:
    """Deterministic, idempotent redaction with per-category counts.

    ``terms`` are replaced (whole words, any case) with ``<USER>``: the OS user
    name and anything else the owner wants kept out - a partner's name, an
    employer. The counts hold categories and numbers only, never the text.
    """

    def __init__(self, terms: Iterable[str] = ()) -> None:
        self.terms = _dedupe_terms(terms)
        self.counts: Counter[str] = Counter()
        self._term_re = (re.compile(
            r"(?i)(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(t) for t in self.terms)
            + r")(?![A-Za-z0-9_])") if self.terms else None)

    @classmethod
    def from_environment(cls, extra_terms: Iterable[str] = ()) -> "Redactor":
        return cls([*environment_terms(), *extra_terms])

    # ── one string ───────────────────────────────────────────────────────────

    def _sub(self, label: str, pattern: re.Pattern[str], repl: Any, text: str) -> str:
        def hit(match: re.Match[str]) -> str:
            out = repl(match) if callable(repl) else repl
            if out != match.group(0):
                self.counts[label] += 1
            return out
        return pattern.sub(hit, text)

    def redact(self, value: str) -> str:
        text = str(value)
        if not text:
            return text
        text = self._sub("secret", _PRIVATE_KEY, SECRET, text)
        text = self._sub("secret", _JWT, SECRET, text)
        text = self._sub("secret", _PREFIXED, SECRET, text)
        text = self._sub("secret", _BEARER, lambda m: f"{m.group(1)} {SECRET}", text)
        text = self._sub("secret", _URL_CREDENTIALS, lambda m: f"{m.group(1)}{SECRET}@", text)
        text = self._sub("secret", _URL_SECRET_PARAM, lambda m: f"{m.group(1)}{SECRET}", text)
        text = self._sub(
            "secret", _KEY_VALUE,
            lambda m: m.group(0) if (m.group(3) in PLACEHOLDERS or m.group(3).lower() in {"bearer", "basic"}) else
            f"{m.group(1)}{m.group(2)}{SECRET}{m.group(2)}", text)
        text = self._sub("email", _EMAIL_RE, EMAIL, text)
        text = self._sub("iban", _IBAN_RE,
                         lambda m: IBAN if sum(c.isdigit() for c in m.group(0)) >= 6 else m.group(0),
                         text)
        text = self._sub("card", _CARD_RE,
                         lambda m: CARD if luhn_valid(m.group(0)) else m.group(0), text)
        text = self._sub("phone", _PHONE_INTL, PHONE, text)
        text = self._sub("phone", _PHONE_UK, PHONE, text)
        text = self._sub("postcode", _POSTCODE_RE,
                         lambda m: POSTCODE if _postcode_ok(m.group(0)) else m.group(0), text)
        text = self._sub("ip", _IPV4,
                         lambda m: m.group(0) if m.group(0).startswith(("127.", "0.0.0.0"))
                         else IP, text)
        text = self._sub("secret", _HEX_LONG, SECRET, text)
        text = self._sub(
            "secret", _OPAQUE,
            lambda m: SECRET if (re.search(r"\d", m.group(0)) and re.search(r"[A-Za-z]", m.group(0))
                                 and _entropy(m.group(0)) >= 4.0
                                 and not _looks_like_name(m.group(0))) else m.group(0), text)
        text = self._sub("home", _PROFILE_PATH, HOME, text)      # C:\Users\name -> <HOME>
        text = self._sub("home", _ONEDRIVE, "OneDrive", text)
        if self._term_re is not None:
            text = self._sub("user", self._term_re, USER, text)
        return text

    # ── structures ───────────────────────────────────────────────────────────

    def redact_obj(self, value: Any) -> Any:
        """Strings inside dicts, lists and tuples; keys and other types as they are."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {k: self.redact_obj(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact_obj(v) for v in value]
        return value


def contains_unredacted(text: str, redactor: Redactor | None = None) -> bool:
    """Whether redacting *text* would still change it - used by the dataset
    checks (and tests) to assert that a written row is already clean."""
    probe = Redactor(redactor.terms if redactor else ())
    return probe.redact(text) != text


__all__ = ["CARD", "EMAIL", "HOME", "IBAN", "IP", "PHONE", "PLACEHOLDERS", "POSTCODE", "Redactor",
           "SECRET", "USER", "contains_unredacted", "environment_terms", "luhn_valid",
           "sensitive_topic"]
