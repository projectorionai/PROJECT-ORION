"""Read the owner's live Meta and TikTok ad results (Mark XXXX.3).

ad_campaigns.py can judge a campaign from numbers the owner reads out. This
fetches those numbers from the ad accounts themselves - the last N days, per
campaign - and runs each campaign through the same scale / hold / fix / kill
analysis.

READ-ONLY. Nothing here changes a budget, pauses a campaign or spends: the
only calls are Meta's Insights read and TikTok's reporting read.

Configuration (config/api_keys.json -> "integrations"; the tokens are sealed
to the Windows account by key_seal.py):

    "meta_ads":   {"access_token": "...", "ad_account_id": "act_1234567890"}
    "tiktok_ads": {"access_token": "...", "advertiser_id": "7000000000000000000"}

A Meta token needs the ads_read permission (a System User token from Business
Settings lasts); a TikTok token comes from a TikTok for Business developer app
with Reporting access. TikTok's metric names differ by report type, so the
list read is "metrics" in the section when the defaults are refused.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from . import ad_campaigns

META_GRAPH = "https://graph.facebook.com/v21.0"
TIKTOK_API = "https://business-api.tiktok.com/open_api/v1.3"
TIMEOUT_S = 20.0
MAX_PAGES = 5
MAX_CAMPAIGNS = 8
#: Meta's purchase action types, most complete first.
META_PURCHASES = ("omni_purchase", "purchase", "offsite_conversion.fb_pixel_purchase",
                  "onsite_web_purchase")
TIKTOK_METRICS = ("campaign_name", "spend", "impressions", "clicks", "conversion",
                  "complete_payment", "complete_payment_roas")

Get = Callable[[str, dict[str, str]], dict[str, Any]]


class AdAccountError(RuntimeError):
    """A sentence the owner can act on; never contains a token."""


@dataclass
class CampaignRow:
    platform: str
    name: str
    spend: float
    impressions: float = 0.0
    clicks: float = 0.0
    purchases: float = 0.0
    revenue: float = 0.0
    reach: float = 0.0


def _section(name: str) -> dict[str, Any]:
    from .constants import API_CONFIG_PATH
    from .key_seal import load_json

    integrations = load_json(API_CONFIG_PATH).get("integrations") or {}
    section = integrations.get(name) if isinstance(integrations, dict) else None
    return section if isinstance(section, dict) else {}


def configured() -> dict[str, bool]:
    meta, tiktok = _section("meta_ads"), _section("tiktok_ads")
    return {"meta": bool(meta.get("access_token") and meta.get("ad_account_id")),
            "tiktok": bool(tiktok.get("access_token") and tiktok.get("advertiser_id"))}


def _get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 (fixed hosts)
            body = response.read(4_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read(200_000).decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AdAccountError(f"couldn't reach the ad platform ({type(exc).__name__})") from None
    try:
        data = json.loads(body or "{}")
    except ValueError:
        raise AdAccountError("the ad platform sent back something that isn't JSON") from None
    return data if isinstance(data, dict) else {}


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ── Meta ─────────────────────────────────────────────────────────────────────

def _meta_action(items: Any) -> float:
    if not isinstance(items, list):
        return 0.0
    found = {str(i.get("action_type")): _number(i.get("value")) for i in items if isinstance(i, dict)}
    for kind in META_PURCHASES:
        if kind in found:
            return found[kind]
    return 0.0


def fetch_meta(days: int = 7, *, get: Get = _get, section: dict[str, Any] | None = None
               ) -> list[CampaignRow]:
    config = section if section is not None else _section("meta_ads")
    token = str(config.get("access_token") or "").strip()
    account = str(config.get("ad_account_id") or "").strip()
    if not token or not account:
        raise AdAccountError("Meta isn't connected: add integrations.meta_ads.access_token and "
                             "ad_account_id to config/api_keys.json")
    if not account.startswith("act_"):
        account = "act_" + account
    until = date.today()
    since = until - timedelta(days=max(1, int(days)) - 1)
    query = {"level": "campaign",
             "fields": "campaign_name,spend,impressions,clicks,reach,actions,action_values",
             "time_range": json.dumps({"since": since.isoformat(), "until": until.isoformat()}),
             "limit": "100"}
    url = f"{META_GRAPH}/{urllib.parse.quote(account)}/insights?" + urllib.parse.urlencode(query)
    rows: list[CampaignRow] = []
    for _page in range(MAX_PAGES):
        data = get(url, {"Authorization": f"Bearer {token}"})
        if isinstance(data.get("error"), dict):
            message = str(data["error"].get("message") or "refused")
            raise AdAccountError(f"Meta refused the read: {message[:200]}")
        for item in data.get("data") or []:
            if not isinstance(item, dict):
                continue
            rows.append(CampaignRow(
                "meta", str(item.get("campaign_name") or "Unnamed campaign"),
                _number(item.get("spend")), _number(item.get("impressions")),
                _number(item.get("clicks")), _meta_action(item.get("actions")),
                _meta_action(item.get("action_values")), _number(item.get("reach"))))
        following = (data.get("paging") or {}).get("next") if isinstance(data.get("paging"), dict) else None
        if not following or not str(following).startswith(META_GRAPH):
            break
        url = str(following)
    return rows


# ── TikTok ───────────────────────────────────────────────────────────────────

def fetch_tiktok(days: int = 7, *, get: Get = _get, section: dict[str, Any] | None = None
                 ) -> list[CampaignRow]:
    config = section if section is not None else _section("tiktok_ads")
    token = str(config.get("access_token") or "").strip()
    advertiser = str(config.get("advertiser_id") or "").strip()
    if not token or not advertiser:
        raise AdAccountError("TikTok isn't connected: add integrations.tiktok_ads.access_token and "
                             "advertiser_id to config/api_keys.json")
    metrics = [str(m) for m in (config.get("metrics") or TIKTOK_METRICS)]
    until = date.today()
    since = until - timedelta(days=max(1, int(days)) - 1)
    rows: list[CampaignRow] = []
    for page in range(1, MAX_PAGES + 1):
        query = {"advertiser_id": advertiser, "report_type": "BASIC",
                 "data_level": "AUCTION_CAMPAIGN", "dimensions": json.dumps(["campaign_id"]),
                 "metrics": json.dumps(metrics), "start_date": since.isoformat(),
                 "end_date": until.isoformat(), "page": str(page), "page_size": "100"}
        data = get(f"{TIKTOK_API}/report/integrated/get/?" + urllib.parse.urlencode(query),
                   {"Access-Token": token})
        if data.get("code") not in (0, "0"):
            raise AdAccountError(f"TikTok refused the read: {str(data.get('message') or 'refused')[:200]}")
        body = data.get("data") if isinstance(data.get("data"), dict) else {}
        for item in body.get("list") or []:
            m = item.get("metrics") if isinstance(item, dict) and isinstance(item.get("metrics"), dict) else {}
            spend = _number(m.get("spend"))
            purchases = _number(m.get("complete_payment")) or _number(m.get("conversion"))
            rows.append(CampaignRow(
                "tiktok", str(m.get("campaign_name") or "Unnamed campaign"), spend,
                _number(m.get("impressions")), _number(m.get("clicks")), purchases,
                _number(m.get("complete_payment_roas")) * spend))
        info = body.get("page_info") if isinstance(body.get("page_info"), dict) else {}
        if page >= int(_number(info.get("total_page")) or 1):
            break
    return rows


# ── the report ───────────────────────────────────────────────────────────────

def summarise(rows: list[CampaignRow], *, platform: str, days: int, margin: float | None) -> str:
    label = "TikTok Shop" if platform == "tiktok" else "Meta"
    spent = [r for r in rows if r.spend > 0]
    if not spent:
        return f"{label}: no spend in the last {days} day(s)."
    total = CampaignRow(platform, "all", sum(r.spend for r in spent),
                        sum(r.impressions for r in spent), sum(r.clicks for r in spent),
                        sum(r.purchases for r in spent), sum(r.revenue for r in spent),
                        sum(r.reach for r in spent))
    overall = ad_campaigns.analyse(spend=total.spend, impressions=total.impressions,
                                   clicks=total.clicks, purchases=total.purchases,
                                   revenue=total.revenue, margin=margin, platform=platform)
    lines = [f"{label}, last {days} day(s): £{total.spend:,.2f} spent on {len(spent)} campaign(s), "
             f"{total.purchases:,.0f} purchase(s), £{total.revenue:,.2f} in sales.",
             overall.text()]
    for row in sorted(spent, key=lambda r: -r.spend)[:MAX_CAMPAIGNS]:
        one = ad_campaigns.analyse(spend=row.spend, impressions=row.impressions, clicks=row.clicks,
                                   purchases=row.purchases, revenue=row.revenue, margin=margin,
                                   reach=row.reach, platform=platform)
        roas = one.numbers.get("roas")
        reason = next((r for r in one.reasons if not r.startswith("Give me your margin")), "")
        lines.append(f"- {row.name}: {one.verdict.upper()} - £{row.spend:,.2f} spent"
                     + (f", ROAS {roas:.2f}x" if roas is not None else "")
                     + (f". {reason}" if reason else "."))
    if len(spent) > MAX_CAMPAIGNS:
        lines.append(f"(and {len(spent) - MAX_CAMPAIGNS} smaller campaign(s))")
    return "\n".join(lines)


def report(platform: str = "", days: int = 7, margin: float | None = None, *,
           get: Get = _get) -> str:
    """The live results of every connected account (blocking: run off the loop)."""
    days = max(1, min(90, int(days or 7)))
    wanted = ad_campaigns._platform(platform) if platform else ""
    linked = configured()
    targets = [p for p in ("meta", "tiktok") if (not wanted or p == wanted)]
    if not any(linked[p] for p in targets):
        return ("No ad account is connected yet. Add integrations.meta_ads (access_token, "
                "ad_account_id) or integrations.tiktok_ads (access_token, advertiser_id) to "
                "config/api_keys.json - I only read results, never change or spend.")
    parts: list[str] = []
    started = time.monotonic()
    for name in targets:
        if not linked[name]:
            continue
        try:
            rows = (fetch_meta if name == "meta" else fetch_tiktok)(days, get=get)
            parts.append(summarise(rows, platform=name, days=days, margin=margin))
        except AdAccountError as exc:
            parts.append(str(exc) + ".")
    if time.monotonic() - started > TIMEOUT_S * 2:
        parts.append("(The ad platform was slow to answer.)")
    return "\n\n".join(parts)


__all__ = ["AdAccountError", "CampaignRow", "configured", "fetch_meta", "fetch_tiktok",
           "report", "summarise"]
