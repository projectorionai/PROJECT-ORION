"""Read-only scholarly metadata discovery through Crossref and Europe PMC.

Fixed HTTPS endpoints, no credentials, full-text downloads or persistent store.
Metadata is a discovery receipt, not evidence that a paper supports a claim.
"""
from __future__ import annotations

import asyncio
import html
import json
import re
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import quote

from .data import ToolResult
from .lazy_import import LazyModule

aiohttp = LazyModule("aiohttp")

ENDPOINTS = {
    "crossref": "https://api.crossref.org/works",
    "europepmc": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
}
MAX_BYTES = 1_048_576
TIMEOUT_SECONDS = 8
_DOI = re.compile(r"10\.\d{4,9}/[^\s<>]+", re.IGNORECASE)


def _text(value: Any, maximum: int = 600) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", "", value))).strip()[:maximum]


def _first(value: Any) -> str:
    return _text(value[0]) if isinstance(value, list) and value else _text(value)


def _doi(value: Any) -> str:
    return value if isinstance(value, str) and len(value) <= 256 and _DOI.fullmatch(value) else ""


def _date(row: dict) -> str:
    for key in ("published", "published-print", "published-online", "issued"):
        block = row.get(key)
        parts = block.get("date-parts") if isinstance(block, dict) else None
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            continue
        values = parts[0]
        if not 1 <= len(values) <= 3 or any(type(v) is not int for v in values):
            continue
        try:
            date(values[0], values[1] if len(values) > 1 else 1,
                 values[2] if len(values) > 2 else 1)
        except (OverflowError, ValueError):
            continue
        return "-".join(str(v) if i == 0 else f"{v:02d}" for i, v in enumerate(values))
    return ""


def _normalise(source: str, payload: Any, limit: int) -> list[dict]:
    if not isinstance(payload, dict):
        raise TypeError("Invalid metadata envelope")
    if source == "crossref":
        block = payload.get("message")
        if payload.get("status") != "ok" or not isinstance(block, dict):
            raise ValueError("Invalid Crossref envelope")
        rows = block.get("items")
    else:
        block = payload.get("resultList")
        rows = block.get("result") if isinstance(block, dict) else None
    if not isinstance(rows, list):
        raise TypeError("Invalid metadata records")
    records = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            raise TypeError("Invalid metadata record")
        title = _first(row.get("title"))
        if not title:
            continue
        doi = _doi(row.get("DOI") if source == "crossref" else row.get("doi"))
        if source == "crossref":
            authors = row.get("author")
            author_names = [" ".join(filter(None, [_text(a.get("given"), 80), _text(a.get("family"), 80)]))
                            for a in authors[:10] if isinstance(a, dict)] if isinstance(authors, list) else []
            journal = _first(row.get("container-title"))
            date = _date(row)
            count = row.get("is-referenced-by-count")
            identifier = doi
            url = "https://doi.org/" + quote(doi, safe="/") if doi else ""
        else:
            author_names = [_text(row.get("authorString"))] if row.get("authorString") else []
            journal = _text(row.get("journalTitle"))
            date = _text(row.get("firstPublicationDate") or row.get("pubYear"), 32)
            count = row.get("citedByCount")
            identifier = _text(row.get("id"), 128)
            index = _text(row.get("source"), 16)
            url = "https://doi.org/" + quote(doi, safe="/") if doi else (
                f"https://europepmc.org/article/{quote(index, safe='')}/{quote(identifier, safe='')}"
                if identifier and re.fullmatch(r"[A-Z]{2,10}", index) else "")
        records.append({"title": title, "doi": doi, "identifier": identifier,
                        "url": url, "authors": author_names, "journal": journal,
                        "publication_date": date,
                        "citation_count": count if type(count) is int and count >= 0 else None,
                        "sources": [source], "full_text_read": False, "claim_verified": False})
    return records


async def _request(source: str, query: str, limit: int) -> Any:
    params = ({"query.bibliographic": query, "rows": str(limit)} if source == "crossref" else
              {"query": query, "format": "json", "resultType": "lite", "pageSize": str(limit)})
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
    async with (
        aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "ORION-Scholarly/1.0",
                                                      "Accept": "application/json"}) as session,
        session.get(ENDPOINTS[source], params=params, allow_redirects=False) as response,
    ):
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        if response.content_length is not None and response.content_length > MAX_BYTES:
            raise ValueError("Metadata response exceeds size limit")
        body = bytearray()
        async for chunk in response.content.iter_chunked(65536):
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise ValueError("Metadata response exceeds size limit")
        return json.loads(body)


async def search(args: dict[str, Any]) -> ToolResult:
    """Bounded, attributed discovery; failures and empty results stay distinct."""
    query = args.get("query")
    source = args.get("source", "all")
    limit = args.get("limit", 5)
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        return ToolResult("Scholarly search requires a query of 1–500 characters.", ok=False)
    if not isinstance(source, str) or source not in (*ENDPOINTS, "all"):
        return ToolResult("Scholarly source must be crossref, europepmc or all.", ok=False)
    if type(limit) is not int or not 1 <= limit <= 10:
        return ToolResult("Scholarly limit must be an integer from 1 to 10.", ok=False)
    selected = list(ENDPOINTS) if source == "all" else [source]

    async def one(name):
        try:
            records = _normalise(name, await _request(name, query.strip(), limit), limit)
            return {"source": name, "ok": True, "records": records}
        except (aiohttp.ClientError, asyncio.TimeoutError, TypeError, ValueError, UnicodeError) as error:
            detail = str(error) if isinstance(error, ValueError) and str(error).startswith("HTTP ") else type(error).__name__
            return {"source": name, "ok": False, "error": detail, "records": []}

    results = await asyncio.gather(*(one(name) for name in selected))
    records, seen = [], {}
    # Interleave indexes before limiting the union. Keep each index's fields:
    # a shared DOI does not make its title/date/citation counts agree.
    for index in range(limit):
        for result in results:
            if index >= len(result["records"]):
                continue
            record = result["records"][index]
            key = record["doi"].lower() if record["doi"] else (result["source"], record["identifier"], record["title"])
            observation = {k: v for k, v in record.items() if k != "sources"}
            if key in seen:
                seen[key]["sources"].extend(s for s in record["sources"] if s not in seen[key]["sources"])
                seen[key]["source_records"][result["source"]] = observation
            else:
                record["source_records"] = {result["source"]: observation}
                seen[key] = record
                records.append(record)
    report = {"query": query.strip(), "observed_at": datetime.now(timezone.utc).isoformat(),
              "metadata_only": True, "notice": "Discovery metadata; full texts were not read and claims were not verified.",
              "selection": "Interleaved index order; source_records retains each index's metadata.",
              "partial": any(not r["ok"] for r in results) and any(r["ok"] for r in results),
              "sources": [{k: v for k, v in r.items() if k != "records"} for r in results],
              "records": records[:limit]}
    return ToolResult(json.dumps(report, ensure_ascii=False), ok=any(r["ok"] for r in results))
