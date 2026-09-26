"""
Dispatch domain — Commerce research and entrepreneurial advisory tools.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line


class CommerceDispatchMixin:
    """Commerce research and entrepreneurial advisory tools."""

    async def product_research(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Commerce intelligence unavailable.", ok=False)
        action = str(args.get("action") or "score").lower().strip()
        name = str(args.get("name") or args.get("product") or "")
        desc = str(args.get("description") or args.get("notes") or "")
        if action in {"score", "evaluate"}:
            return await self.commerce.dropship.score_product(name, desc, args.get("metrics"))
        if action in {"discover", "discovery", "find"}:
            return await self.commerce.product.discover(
                str(args.get("niche") or name or "home products"),
                int(args.get("count") or 5))
        if action in {"validate", "validation"}:
            return await self.commerce.dropship.validate(name, desc)
        if action in {"competition", "saturation"}:
            return await self.commerce.dropship.analyse_competition(
                str(args.get("niche") or name))
        if action in {"log", "research_log"}:
            rows = self.commerce.dropship.research_log()
            return ToolResult("Product research log:\n" + "\n".join(
                f"- {r.get('key_ref')}: {r.get('value', '')[:120]}" for r in rows) or "empty")
        return ToolResult(f"Unsupported product_research action: {action}.", ok=False)

    async def tiktok_intel(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Commerce intelligence unavailable.", ok=False)
        action = str(args.get("action") or "trend").lower().strip()
        if action in {"trend", "trends", "report"}:
            return await self.commerce.tiktok.trend_report(str(args.get("niche") or ""))
        if action in {"product"}:
            return await self.commerce.tiktok.product_report(str(args.get("product") or ""))
        if action in {"virality", "score"}:
            return self.commerce.tiktok.score_virality(args.get("signals") or {})
        return ToolResult(f"Unsupported tiktok_intel action: {action}.", ok=False)

    async def instagram_intel(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Commerce intelligence unavailable.", ok=False)
        action = str(args.get("action") or "discover").lower().strip()
        if action in {"discover", "discovery"}:
            return await self.commerce.instagram.discover(str(args.get("niche") or "home products"))
        if action in {"influencer", "influencers"}:
            return await self.commerce.instagram.influencer_strategy(str(args.get("brand") or ""))
        if action in {"weekly", "report"}:
            return await self.commerce.instagram.weekly_report(str(args.get("niche") or "home products"))
        return ToolResult(f"Unsupported instagram_intel action: {action}.", ok=False)

    async def founder_knowledge(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Founder knowledge unavailable.", ok=False)
        action = str(args.get("action") or "profile").lower().strip()
        if action in {"list"}:
            return ToolResult("Founders in the knowledge base: "
                              + ", ".join(self.commerce.founder.list_founders()))
        if action in {"profile"}:
            return self.commerce.founder.profile(str(args.get("name") or ""))
        if action in {"learn", "apply"}:
            return await self.commerce.founder.learn_from(
                str(args.get("name") or ""), str(args.get("question") or ""))
        return ToolResult(f"Unsupported founder_knowledge action: {action}.", ok=False)

    async def business_advisor(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None:
            return ToolResult("Business advisor unavailable.", ok=False)
        action = str(args.get("action") or "advise").lower().strip()
        advisor = self.commerce.advisor
        if action in {"advise", "advice"}:
            return await advisor.advise(str(args.get("topic") or args.get("question") or ""))
        if action in {"brand", "brand_strategy"}:
            return await advisor.brand_strategy()
        if action in {"growth", "growth_plan"}:
            return await advisor.growth_plan(str(args.get("horizon") or "next 90 days"))
        if action in {"store", "optimise", "optimize", "cro"}:
            return await advisor.store_optimisation()
        return ToolResult(f"Unsupported business_advisor action: {action}.", ok=False)

    def commerce_hub(self, args: dict[str, Any]) -> ToolResult:
        if self.hub is None:
            return ToolResult("E-commerce hub unavailable.", ok=False)
        return self.hub.report()

    def community_share(self, args: dict[str, Any]) -> ToolResult:
        if self.community is None:
            return ToolResult("Community layer unavailable.", ok=False)
        action = str(args.get("action") or "list").lower().strip()
        if action in {"export_pack"}:
            return self.community.export_pack(str(args.get("id") or args.get("pack_id") or ""),
                                              str(args.get("privacy") or "shared"))
        if action in {"import_pack"}:
            return self.community.import_pack(str(args.get("path") or ""))
        if action in {"export_research"}:
            return self.community.export_research(str(args.get("privacy") or "shared"),
                                                  bool(args.get("anonymise", True)))
        if action in {"import_research"}:
            return self.community.import_research(str(args.get("path") or ""))
        if action in {"list", "bundles"}:
            return ToolResult("Community bundles:\n" + "\n".join(self.community.list_bundles()) or "none")
        return ToolResult(f"Unsupported community_share action: {action}.", ok=False)

    async def competitor_intel_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None or getattr(self.commerce, "competitor", None) is None:
            return ToolResult("Competitor intelligence is not available.", ok=False)
        action = str(args.get("action") or "store").lower().strip()
        target = str(args.get("target") or args.get("store") or args.get("competitor") or "")
        if not target:
            return ToolResult("A competitor/store name is required.", ok=False)
        if action in {"store", "store_analysis"}:
            return await self.commerce.competitor.store_analysis(target)
        if action in {"offer", "offer_analysis"}:
            return await self.commerce.competitor.offer_analysis(
                target, str(args.get("offer") or ""))
        if action in {"funnel", "funnel_analysis"}:
            return await self.commerce.competitor.funnel_analysis(target)
        return ToolResult(
            f"Unsupported competitor_intel action: {action}. Use store, offer, or funnel.",
            ok=False,
        )

    async def brand_growth_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.commerce is None or getattr(self.commerce, "growth", None) is None:
            return ToolResult("The brand growth agent is not available.", ok=False)
        action = str(args.get("action") or "strategy").lower().strip()
        if action in {"strategy"}:
            return await self.commerce.growth.strategy(str(args.get("focus") or ""))
        if action in {"conversion", "cro", "conversion_optimisation"}:
            return await self.commerce.growth.conversion_optimisation(
                str(args.get("page") or "store"))
        if action in {"positioning", "position"}:
            return await self.commerce.growth.positioning(
                str(args.get("product") or ""))
        if action in {"retention", "retention_systems"}:
            return await self.commerce.growth.retention_systems()
        return ToolResult(
            f"Unsupported brand_growth action: {action}. Use strategy, conversion, "
            "positioning, or retention.",
            ok=False,
        )
