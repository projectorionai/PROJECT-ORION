"""Paid campaigns on Meta (Facebook and Instagram) and TikTok Shop: the
arithmetic, the plan and the policy check, done exactly rather than guessed.

  analyse(...)    a campaign's numbers -> CTR, CPC, CPM, conversion rate,
                  CPA, ROAS, AOV, frequency, break-even ROAS and CPA, profit,
                  and a verdict (scale / hold / fix / kill) with the reason
  plan(...)       a launch plan for one platform: structure, budget split,
                  the daily budget the learning phase needs, a creative test
                  grid, KPIs and the scale and kill rules
  copy_issues(..) wording the platforms' ad policies tend to reject

The thresholds below are starting rules of thumb, stated as such - they
differ by niche, country and season, and the owner's own results replace
them as soon as there are any. Nothing here touches an ad account.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

PLATFORMS = ("meta", "tiktok")

#: Starting thresholds (not market data): below them, look at the creative or
#: the page before spending more.
RULES = {
    "meta": {"ctr_low": 0.8, "cvr_low": 1.5, "frequency_high": 3.0, "events_per_week": 50},
    "tiktok": {"ctr_low": 0.7, "cvr_low": 1.0, "frequency_high": 4.0, "events_per_week": 50},
}
#: ROAS this far above break-even earns more budget.
SCALE_MARGIN = 1.3
#: Most a working ad set's budget is raised at once (keeps delivery stable).
SCALE_STEP = 0.2


def _platform(name: Any) -> str:
    text = str(name or "").lower()
    return "tiktok" if "tik" in text else "meta"


@dataclass
class Analysis:
    platform: str
    numbers: dict[str, float]
    verdict: str
    reasons: list[str]

    def text(self) -> str:
        n = self.numbers

        def money(key: str) -> str:
            return f"{n[key]:,.2f}" if key in n else "-"
        lines = [f"{'TikTok Shop' if self.platform == 'tiktok' else 'Meta'} campaign: "
                 f"{self.verdict.upper()}."]
        parts = []
        for key, label, kind in (("ctr", "CTR", "%"), ("cpc", "CPC", "£"), ("cpm", "CPM", "£"),
                                 ("cvr", "conversion", "%"), ("cpa", "CPA", "£"),
                                 ("roas", "ROAS", "x"), ("break_even_roas", "break-even ROAS", "x"),
                                 ("aov", "order value", "£"), ("frequency", "frequency", "x"),
                                 ("profit", "profit after ads", "£")):
            if key in n:
                value = n[key]
                parts.append(f"{label} {value:.2f}%" if kind == "%" else
                             f"{label} {value:.2f}x" if kind == "x" else f"{label} £{money(key)}")
        lines.append(", ".join(parts) + ".")
        lines += [f"- {reason}" for reason in self.reasons]
        return "\n".join(lines)


def analyse(*, spend: float, impressions: float = 0, clicks: float = 0, purchases: float = 0,
            revenue: float = 0, margin: float | None = None, reach: float = 0,
            platform: str = "meta") -> Analysis:
    """Every ratio that matters, and what to do next."""
    platform = _platform(platform)
    rules = RULES[platform]
    spend = max(0.0, float(spend))
    n: dict[str, float] = {"spend": spend}
    if impressions:
        n["cpm"] = spend / impressions * 1000
    if impressions and clicks:
        n["ctr"] = clicks / impressions * 100
    if clicks:
        n["cpc"] = spend / clicks
        n["cvr"] = purchases / clicks * 100
    if purchases:
        n["cpa"] = spend / purchases
        n["aov"] = revenue / purchases
    if spend:
        n["roas"] = revenue / spend
    if reach and impressions:
        n["frequency"] = impressions / reach
    if margin:
        margin = float(margin) / 100 if float(margin) > 1 else float(margin)
        n["break_even_roas"] = 1 / margin
        n["profit"] = revenue * margin - spend
        if purchases:
            n["break_even_cpa"] = revenue / purchases * margin
    reasons: list[str] = []
    verdict = "hold"
    roas, even = n.get("roas"), n.get("break_even_roas")
    if spend and not purchases:
        verdict = "fix" if n.get("ctr", 0) >= rules["ctr_low"] else "kill"
        reasons.append("No purchases yet: " + (
            "people click but don't buy - check the offer, price and product page."
            if verdict == "fix" else "few people even click - replace the creative."))
    elif roas is not None and even is not None:
        if roas >= even * SCALE_MARGIN:
            verdict = "scale"
            reasons.append(f"ROAS {roas:.2f}x is well over break-even {even:.2f}x: raise the "
                           f"budget by up to {SCALE_STEP:.0%} every 48 hours while it holds.")
        elif roas >= even:
            reasons.append(f"Profitable but thin ({roas:.2f}x against {even:.2f}x): hold the "
                           "budget and test new hooks before scaling.")
        else:
            verdict = "fix" if roas >= even * 0.7 else "kill"
            reasons.append(f"Losing money: ROAS {roas:.2f}x is under break-even {even:.2f}x.")
    if "ctr" in n and n["ctr"] < rules["ctr_low"]:
        reasons.append(f"CTR {n['ctr']:.2f}% is low: the first two seconds aren't stopping "
                       "the scroll - test new hooks and thumbnails.")
    if "cvr" in n and purchases and n["cvr"] < rules["cvr_low"]:
        reasons.append(f"Only {n['cvr']:.2f}% of clicks buy: fix the page, the offer or the "
                       "price before buying more traffic.")
    if n.get("frequency", 0) > rules["frequency_high"]:
        reasons.append(f"Frequency {n['frequency']:.1f}: the same people keep seeing it - "
                       "refresh the creative or widen the audience.")
    if margin is None:
        reasons.append("Give me your margin and I'll tell you whether this is actually profitable.")
    return Analysis(platform, n, verdict, reasons or ["Within the starting thresholds."])


def plan(*, platform: str = "meta", product: str = "", price: float = 0, margin: float = 0.4,
         daily_budget: float = 0, goal: str = "sales", audience: str = "") -> str:
    """A launch plan. Numbers are worked out from price and margin."""
    platform = _platform(platform)
    rules = RULES[platform]
    margin = float(margin) / 100 if float(margin) > 1 else float(margin or 0.4)
    price = float(price or 0)
    break_even_cpa = price * margin if price else 0.0
    target_cpa = break_even_cpa * 0.75
    even = 1 / margin if margin else 0.0
    learning = rules["events_per_week"] * target_cpa / 7 if target_cpa else 0.0
    thing = product or "the product"
    lines = [f"{'TikTok Shop' if platform == 'tiktok' else 'Meta'} plan for {thing}"
             + (f" (£{price:.2f}, {margin:.0%} margin)" if price else "") + "."]
    if price:
        lines.append(f"Numbers: break-even CPA £{break_even_cpa:.2f}, target CPA "
                     f"£{target_cpa:.2f}, break-even ROAS {even:.2f}x, target ROAS "
                     f"{even * SCALE_MARGIN:.2f}x.")
        fits = ""
        if daily_budget and learning:
            sets = int(daily_budget // learning)
            fits = (f" - your £{daily_budget:,.0f} a day supports {sets} ad set"
                    f"{'s' if sets != 1 else ''}." if sets else
                    f" - your £{daily_budget:,.0f} a day is under that: run one ad set, expect a "
                    f"longer learning phase, or raise it to £{learning:,.0f}.")
        lines.append(f"Learning phase: about {rules['events_per_week']} purchases a week per "
                     f"ad set means roughly £{learning:,.0f} a day per ad set at the target CPA"
                     + (fits or "."))
    if platform == "meta":
        lines += [
            "Structure: one Sales campaign with Advantage+ placements. Start with Advantage+ "
            "audience (broad); alongside it, one interest-led ad set"
            + (f" around {audience}" if audience else "") + " and, once traffic exists, a "
            "retargeting ad set (site visitors and engagers, last 30 days).",
            "Budget split: 70% prospecting broad, 20% interest test, 10% retargeting.",
            "Creative grid: 3 hooks x 2 formats (9:16 video, 1:1 image or carousel) = 6 ads; "
            "UGC-style video first, a clear offer in the first line of text.",
            "Tracking: Pixel plus the Conversions API, purchase as the optimisation event.",
        ]
    else:
        lines += [
            "Structure: list the product on TikTok Shop with a clean title (keyword first), "
            "five or more images and a short demo video; run GMV Max (or Product Shopping "
            "Ads) for the shop, and Spark Ads to boost the best organic and creator videos.",
            "Creators: send samples to 20-30 affiliates in the niche at a 10-20% commission; "
            "boost the top performers as Spark Ads.",
            "Content: 1-3 shoppable videos a day; a hook in the first 2 seconds, the product in "
            "use by second 3, the price or offer on screen; one LIVE session a week to start.",
            "Budget split: 60% GMV Max / shopping ads, 30% Spark Ads on winning videos, 10% tests.",
        ]
    lines += [
        f"KPIs to watch daily: CTR over {rules['ctr_low']}%, conversion over "
        f"{rules['cvr_low']}% of clicks, CPA under target, frequency under "
        f"{rules['frequency_high']:.0f}.",
        "Kill rule: an ad that has spent twice the target CPA with no purchase, or whose CTR is "
        "under the threshold after 1,000 impressions.",
        f"Scale rule: ROAS above {SCALE_MARGIN}x break-even for 3 days - raise the budget "
        f"{SCALE_STEP:.0%} every 48 hours; duplicate rather than jump.",
        "First week: learn, don't judge - change nothing for 3 days unless the kill rule fires.",
    ]
    return "\n".join(lines)


#: Wording ad reviewers tend to reject, with why. Not legal advice.
_POLICY = (
    (r"\bguarantee(?:d|s)?\b", "guaranteed results are a claim reviewers reject without proof"),
    (r"\b(?:cure|cures|heal|heals|treats?)\b", "medical claims need evidence and are restricted"),
    (r"\blose \d+\s*(?:kg|lbs?|pounds|stone)\b", "specific weight-loss claims are restricted"),
    (r"\bbefore and after\b", "before-and-after framing is restricted for health and beauty"),
    (r"\b(?:get rich|make \$?£?\d+[k]? (?:a|per) (?:day|week))\b", "income claims are restricted"),
    (r"\bare you (?:depressed|overweight|in debt|diabetic|single|gay|anxious)\b",
     "speaking to a personal attribute ('are you...') breaks Meta's personal attributes rule"),
    (r"\b100%\s*(?:free|safe|effective|natural)\b", "absolute claims need proof"),
    (r"\bfree\b(?!.*\b(?:terms|t&cs|conditions)\b)", "'free' needs its conditions stated"),
    (r"!{2,}|\b[A-Z]{5,}\b", "shouting (repeated ! or long capitals) reads as spam to reviewers"),
)


def copy_issues(text: str, platform: str = "meta") -> list[str]:
    """Phrases in ad copy that ad review tends to reject, with the reason."""
    found = []
    for pattern, why in _POLICY:
        match = re.search(pattern, str(text or ""), re.I if not pattern.startswith("!") else 0)
        if match:
            found.append(f"'{match.group(0)}': {why}.")
    return found


__all__ = ["Analysis", "PLATFORMS", "RULES", "analyse", "copy_issues", "plan"]
