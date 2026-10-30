"""Mark XXXX.2: paid campaigns on Meta and TikTok Shop - exact arithmetic, a
plan built from price and margin, and copy that ad review would reject."""
from __future__ import annotations

import asyncio
import types

import pytest

from orion_core import ad_campaigns as ac


def test_the_numbers_are_exact_and_the_verdict_follows_the_margin():
    result = ac.analyse(spend=500, impressions=60000, clicks=720, purchases=21, revenue=1470,
                        margin=0.45, reach=24000)
    n = result.numbers
    assert n["ctr"] == pytest.approx(1.2) and n["cpm"] == pytest.approx(8.3333, rel=1e-3)
    assert n["cpa"] == pytest.approx(23.81, rel=1e-3) and n["roas"] == pytest.approx(2.94)
    assert n["break_even_roas"] == pytest.approx(2.2222, rel=1e-3)
    assert n["profit"] == pytest.approx(161.5) and n["frequency"] == pytest.approx(2.5)
    assert result.verdict == "scale"


@pytest.mark.parametrize("revenue,verdict", [(1000, "hold"), (800, "fix"), (300, "kill")])
def test_under_or_near_break_even_is_held_fixed_or_killed(revenue, verdict):
    assert ac.analyse(spend=500, clicks=500, impressions=40000, purchases=10, revenue=revenue,
                      margin=0.5).verdict == verdict


def test_no_sales_says_whether_the_creative_or_the_page_is_the_problem():
    clicks_no_buys = ac.analyse(spend=100, impressions=10000, clicks=200, purchases=0)
    assert clicks_no_buys.verdict == "fix" and "product page" in clicks_no_buys.text()
    nobody_clicks = ac.analyse(spend=100, impressions=10000, clicks=20, purchases=0)
    assert nobody_clicks.verdict == "kill" and "creative" in nobody_clicks.text()
    assert "margin" in clicks_no_buys.text()          # asks for what it needs


def test_a_plan_is_worked_out_from_price_and_margin():
    text = ac.plan(platform="tiktok", product="desk lamp", price=29.99, margin=50, daily_budget=60)
    assert "break-even CPA £14.99" in text and "break-even ROAS 2.00x" in text
    assert "GMV Max" in text and "Spark Ads" in text
    assert "under that" in text                       # £60/day is below the learning need
    meta = ac.plan(platform="meta", product="course", price=100, margin=0.8, daily_budget=500)
    assert "Advantage+" in meta and "Conversions API" in meta and "supports" in meta


def test_copy_that_ad_review_rejects_is_flagged():
    issues = ac.copy_issues("Guaranteed to lose 10kg in a month!! Are you overweight?")
    assert len(issues) == 4
    assert ac.copy_issues("A warm desk lamp for late study sessions.") == []
    assert ac.copy_issues("Free delivery - terms apply") == []


def test_brand_growth_offers_the_ads_actions_without_an_engine():
    from orion_core.dispatch_commerce import CommerceDispatchMixin
    host = types.SimpleNamespace(commerce=None)
    plan = asyncio.run(CommerceDispatchMixin.brand_growth_tool(
        host, {"action": "meta_plan", "product": "course", "price": "£100", "margin": "80%"}))
    assert plan.ok and "Meta plan for course" in plan.text
    metrics = asyncio.run(CommerceDispatchMixin.brand_growth_tool(
        host, {"action": "ad_metrics", "spend": "1,000", "revenue": 3000, "margin": 0.5,
               "purchases": 30, "clicks": 900, "impressions": 70000}))
    assert metrics.ok and "SCALE" in metrics.text
    assert not asyncio.run(CommerceDispatchMixin.brand_growth_tool(
        host, {"action": "ad_metrics"})).ok

# ── folded in from test_ad_accounts.py (Mark XXXX.3: read the owner's live Meta and TikTok ad results, read-only.) ────

import pytest

from orion_core import ad_accounts
from orion_core.ad_accounts import AdAccountError, fetch_meta, fetch_tiktok, summarise

META = {"access_token": "EAAtoken", "ad_account_id": "123"}
TIKTOK = {"access_token": "tttoken", "advertiser_id": "700"}


def test_meta_reads_purchases_and_sales_per_campaign():
    seen = []

    def get(url, headers):
        seen.append((url, headers))
        return {"data": [{"campaign_name": "Hoodies", "spend": "200", "impressions": "40000",
                          "clicks": "800", "reach": "20000",
                          "actions": [{"action_type": "link_click", "value": "800"},
                                      {"action_type": "omni_purchase", "value": "16"}],
                          "action_values": [{"action_type": "omni_purchase", "value": "720"}]}]}
    rows = fetch_meta(7, get=get, section=META)
    assert rows[0].name == "Hoodies" and rows[0].purchases == 16 and rows[0].revenue == 720
    url, headers = seen[0]
    assert "/act_123/insights?" in url and "EAAtoken" not in url
    assert headers == {"Authorization": "Bearer EAAtoken"}


def test_meta_follows_its_own_pages_only():
    pages = iter([
        {"data": [{"campaign_name": "A", "spend": "1"}],
         "paging": {"next": "https://graph.facebook.com/v21.0/next"}},
        {"data": [{"campaign_name": "B", "spend": "2"}],
         "paging": {"next": "https://evil.example/steal"}},
    ])
    rows = fetch_meta(7, get=lambda url, headers: next(pages), section=META)
    assert [r.name for r in rows] == ["A", "B"]


def test_a_refusal_is_reported_without_the_token():
    with pytest.raises(AdAccountError) as caught:
        fetch_meta(7, get=lambda u, h: {"error": {"message": "Invalid OAuth access token."}},
                   section=META)
    assert "Invalid OAuth" in str(caught.value) and "EAAtoken" not in str(caught.value)
    with pytest.raises(AdAccountError):
        fetch_meta(7, section={})


def test_tiktok_reads_roas_into_sales():
    seen = []

    def get(url, headers):
        seen.append(headers)
        return {"code": 0, "data": {"list": [{"metrics": {
            "campaign_name": "GMV Max", "spend": "100", "impressions": "50000", "clicks": "600",
            "complete_payment": "10", "complete_payment_roas": "3.5"}}],
            "page_info": {"total_page": 1}}}
    rows = fetch_tiktok(7, get=get, section=TIKTOK)
    assert rows[0].revenue == pytest.approx(350) and rows[0].purchases == 10
    assert seen == [{"Access-Token": "tttoken"}]
    with pytest.raises(AdAccountError):
        fetch_tiktok(7, get=lambda u, h: {"code": 40002, "message": "invalid metric"}, section=TIKTOK)


def test_the_summary_gives_each_campaign_a_verdict():
    rows = [ad_accounts.CampaignRow("meta", "Winner", 100, 20000, 500, 20, 600),
            ad_accounts.CampaignRow("meta", "Dud", 80, 30000, 90, 0, 0),
            ad_accounts.CampaignRow("meta", "Idle", 0)]
    text = summarise(rows, platform="meta", days=7, margin=0.5)
    assert "£180.00 spent on 2 campaign(s)" in text
    assert "- Winner: SCALE" in text and "- Dud: KILL" in text and "Idle" not in text


def test_nothing_connected_says_how(monkeypatch):
    monkeypatch.setattr(ad_accounts, "configured", lambda: {"meta": False, "tiktok": False})
    assert "No ad account is connected" in ad_accounts.report()
