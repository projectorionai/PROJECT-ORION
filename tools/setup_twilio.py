"""
Give ORION a phone line, in one step.

What was asked for was calling "without it being set up". That is not
something code can deliver: Twilio bills a real account for a real call and
will not place one for an account that does not exist. Three values are
needed, and nothing can invent them.

What this does instead is make that the ONLY step. No Node, no npx, no MCP
server, no editing JSON by hand, no restart — paste three values and ORION can
dial. It then verifies them against Twilio before saying so, because a setup
tool that reports success on credentials that do not work is worse than no
setup tool at all.

Where the three values are
--------------------------
All on the Twilio console home page at https://console.twilio.com

  Account SID    starts with AC, shown under "Account Info"
  Auth Token     on the same panel, behind a "show" toggle
  Phone number   Phone Numbers -> Manage -> Active numbers. This is the number
                 calls come FROM. A trial account has one, and can only call
                 numbers you have verified.

Usage
-----
    python tools/setup_twilio.py                 # prompts for each value
    python tools/setup_twilio.py --check         # what is set, and does it work
    python tools/setup_twilio.py --sid AC... --token ... --from +44...

The token is a bearer credential for an account that can be charged. It is
written to config/telephony.json and is never printed back, not even in part.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import telephony_direct as direct   # noqa: E402


def _verify(credentials: direct.Credentials) -> tuple[bool, str]:
    """Ask Twilio whether these credentials actually work.

    Fetching the account is a free, read-only GET. It distinguishes the three
    things that otherwise all look like "it didn't work": wrong SID, wrong
    token, and a number that is not on this account.
    """
    import httpx

    url = f"{direct.API_ROOT}/Accounts/{credentials.account_sid}.json"
    try:
        response = httpx.get(url, timeout=direct.TIMEOUT_SECONDS,
                             auth=(credentials.username, credentials.password))
    except Exception as exc:
        return False, f"could not reach Twilio ({type(exc).__name__})"
    if response.status_code == 401:
        return False, "Twilio rejected those credentials (401)"
    if response.status_code == 404:
        return False, "no account with that SID (404) — check it starts with AC"
    if response.status_code != 200:
        return False, f"Twilio answered {response.status_code}"
    try:
        name = response.json().get("friendly_name") or "your account"
    except Exception:
        name = "your account"
    return True, str(name)


def _owns_the_number(credentials: direct.Credentials) -> tuple[bool, str]:
    """Whether the FROM number is really on this account.

    A number that is not is a call that fails at dial time with a message
    nobody reads, long after this tool said everything was fine.
    """
    import httpx

    url = (f"{direct.API_ROOT}/Accounts/{credentials.account_sid}"
           f"/IncomingPhoneNumbers.json?PageSize=100")
    try:
        response = httpx.get(url, timeout=direct.TIMEOUT_SECONDS,
                             auth=(credentials.username, credentials.password))
        numbers = [str(n.get("phone_number", ""))
                   for n in response.json().get("incoming_phone_numbers", [])]
    except Exception as exc:
        # Not fatal: the credentials verified, so this is a nicety.
        return True, f"could not list your numbers ({type(exc).__name__})"
    if not numbers:
        return False, "this account has no phone numbers yet"
    if credentials.from_number not in numbers:
        return False, (f"{credentials.from_number} is not on this account. "
                       f"It has: {', '.join(numbers)}")
    return True, ""


def check() -> int:
    credentials = direct.read_credentials()
    print(f"  credentials found in : {credentials.source}")
    print(f"  Account SID          : {credentials.account_sid or '(not set)'}")
    print(f"  secret               : {'set' if credentials.password else '(not set)'}")
    print(f"  calling FROM         : {credentials.from_number or '(not set)'}")

    gaps = direct.missing(credentials)
    if gaps:
        print("\n  Not ready. Still needed:")
        for gap in gaps:
            print(f"    - {gap}")
        print("\n  Run this without --check to set them.")
        return 1

    print("\n  verifying against Twilio…")
    ok, detail = _verify(credentials)
    if not ok:
        print(f"  FAILED: {detail}")
        return 1
    print(f"  credentials work — account: {detail}")

    owns, why = _owns_the_number(credentials)
    if not owns:
        print(f"  BUT: {why}")
        return 1
    if why:
        print(f"  note: {why}")

    print("\n  ORION can place calls. He still asks for the on-screen "
          "approval every time,\n  because a call costs money and connects "
          "immediately.")
    return 0


def setup(sid: str, token: str, number: str) -> int:
    if not sid:
        sid = input("  Account SID (starts with AC): ").strip()
    if not token:
        token = getpass.getpass("  Auth Token (hidden): ").strip()
    if not number:
        number = input("  Your Twilio number to call FROM (+44...): ").strip()

    if not (sid and token and number):
        print("\n  All three are needed. Nothing was written.")
        return 1
    if not sid.startswith("AC"):
        print(f"\n  {sid[:4]}... does not look like an Account SID — those "
              "start with AC.\n  Nothing was written.")
        return 1
    if not number.startswith("+"):
        print(f"\n  {number} needs to be in international form, like "
              "+441234567890.\n  Nothing was written.")
        return 1

    candidate = direct.Credentials(account_sid=sid, username=sid,
                                   password=token, from_number=number)
    print("\n  verifying against Twilio before writing anything…")
    ok, detail = _verify(candidate)
    if not ok:
        print(f"  FAILED: {detail}")
        print("  Nothing was written.")
        return 1
    print(f"  credentials work — account: {detail}")

    owns, why = _owns_the_number(candidate)
    if not owns:
        print(f"  FAILED: {why}")
        print("  Nothing was written.")
        return 1

    path = direct.save_credentials(sid, token, number)
    print(f"\n  written to {path}")
    print("  ORION can place calls now — no restart needed, and no Node or "
          "MCP server.")
    print("  He still asks for on-screen approval - every call costs "
          "money.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report what is set and whether it works")
    parser.add_argument("--sid", default="", help="Twilio Account SID")
    parser.add_argument("--token", default="", help="Twilio Auth Token")
    parser.add_argument("--from", dest="number", default="",
                        help="the Twilio number to call FROM")
    args = parser.parse_args(argv)

    if args.check:
        return check()
    return setup(args.sid, args.token, args.number)


if __name__ == "__main__":
    raise SystemExit(main())
