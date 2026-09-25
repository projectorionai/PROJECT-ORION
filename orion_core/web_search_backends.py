"""
Finding pages to read, by whichever route is actually working.

Built after watching a single-backend version fail in exactly the way that
matters: DuckDuckGo's HTML endpoint answered the first query with forty
results and every query after that with ``202 Accepted`` and an empty page.
Not an error, not a 429 — a 200-shaped response containing nothing. A
research engine sitting on that alone reports "I could not find anything
about X" and sounds like it looked.

So there are several backends and they are tried in order, and when they all
decline the caller is told **that searching failed**, which is a different
statement from "there is nothing to find" and leads somewhere different.

    Brave        needs a free API key. Reliable, ranked, 2,000/month.
    DuckDuckGo   needs nothing. Rate-limits aggressively.
    Wikipedia    needs nothing. Encyclopaedic only, but it always answers.

Wikipedia last on purpose: it is a poor general search engine and an excellent
last resort, because an investigation that returns one solid encyclopaedia
article is worth more than one that returns nothing at all.

On being throttled
------------------
The back-off is per-process and deliberate. Once DuckDuckGo starts answering
202 it keeps doing so for a while, and hammering it makes that window longer.
So it is stood down for a few minutes and the next backend is used, rather
than retried into the ground.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote_plus, unquote, urlparse

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"

TIMEOUT_S = 15.0

#: How long a throttled backend is left alone. Long enough for DuckDuckGo's
#: window to pass, short enough that a later run in the same session gets it
#: back.
COOLDOWN_S = 300.0

_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

#: Wikimedia's robot policy REFUSES a generic agent — "ORION-Research/1.0"
#: earned a 403 and a link to the policy, which is a polite way of saying
#: identify yourself. It wants a name and somewhere to complain to. The
#: project URL, never the user's own address: their email is theirs, and a
#: research tool has no business putting it in an outbound header.
try:
    from . import __version__ as _VERSION
except Exception:                      # imported outside the package
    _VERSION = "31"
_WIKI_AGENT = (f"ORION/{_VERSION} (https://github.com/projectorionai/PROJECT-ORION) "
               "python-aiohttp")

_RESULT = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL)
_SNIPPET = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")

#: When a backend was last seen to be throttling. Module level, because the
#: throttle is per-IP and every researcher in this process shares one.
_STOOD_DOWN: dict[str, float] = {}


@dataclass
class Hit:
    """One search result, before anybody has read the page."""

    title: str
    url: str
    snippet: str = ""
    backend: str = ""


class SearchUnavailable(Exception):
    """Every backend declined.

    Deliberately not an empty list: "nothing was found" and "nothing could be
    asked" look identical to a caller and mean opposite things. One says the
    topic is obscure; the other says the engine is blocked.
    """


def _text(raw: str) -> str:
    # Tags become spaces, so "A <b>paper</b>" would read "A  paper" without
    # the collapse.
    text = re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", raw or "")))
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def _available(name: str) -> bool:
    until = _STOOD_DOWN.get(name, 0.0)
    return time.monotonic() >= until


def _stand_down(name: str, seconds: float = COOLDOWN_S) -> None:
    _STOOD_DOWN[name] = time.monotonic() + max(1.0, float(seconds))


def brave_key() -> str:
    """The Brave API key, from the environment or the MCP server config.

    Read from the MCP config too, because someone who has already filled that
    in to enable the brave_search server has said what their key is and
    should not have to say it twice.
    """
    key = os.getenv("BRAVE_API_KEY", "").strip()
    if key:
        return key
    try:
        from .constants import CONFIG_DIR

        data = json.loads((CONFIG_DIR / "mcp_servers.json")
                          .read_text(encoding="utf-8"))
        server = (data.get("servers") or {}).get("brave_search") or {}
        return str((server.get("env") or {}).get("BRAVE_API_KEY") or "").strip()
    except Exception:
        return ""


async def _get(url: str, headers: dict, params: dict | None = None) -> tuple[int, str]:
    try:
        from aiohttp import ClientSession, ClientTimeout
    except Exception:
        return 0, ""
    try:
        async with ClientSession(timeout=ClientTimeout(total=TIMEOUT_S)) as session:
            async with session.get(url, headers=headers, params=params,
                                   allow_redirects=True) as response:
                return response.status, await response.text(errors="replace")
    except Exception:
        return 0, ""


# ── the backends ─────────────────────────────────────────────────────────────

async def search_brave(query: str, limit: int) -> list[Hit]:
    key = brave_key()
    if not key:
        return []
    status, body = await _get(
        BRAVE_ENDPOINT,
        {"Accept": "application/json", "X-Subscription-Token": key},
        {"q": query, "count": str(max(1, min(20, limit)))})
    if status == 429:
        _stand_down("brave")
        return []
    if status != 200:
        return []
    try:
        results = (json.loads(body).get("web") or {}).get("results") or []
    except Exception:
        return []
    return [Hit(title=str(r.get("title") or ""), url=str(r.get("url") or ""),
                snippet=_text(str(r.get("description") or "")), backend="brave")
            for r in results if r.get("url")][:limit]


async def search_duckduckgo(query: str, limit: int) -> list[Hit]:
    """No key, and no guarantees.

    A 202 here is the throttle: the page comes back the right shape with no
    results in it, so an empty list alone would read as "nothing found".
    """
    status, body = await _get(DDG_ENDPOINT + "?q=" + quote_plus(query),
                              {"User-Agent": _USER_AGENT})
    if status != 200 or "result__a" not in body:
        _stand_down("duckduckgo")
        return []

    snippets = [_text(s) for s in _SNIPPET.findall(body)]
    hits: list[Hit] = []
    for index, (href, title) in enumerate(_RESULT.findall(body)):
        link = html.unescape(href)
        match = re.search(r"uddg=([^&]+)", link)
        if match:
            link = unquote(match.group(1))
        if not link.startswith("http"):
            continue
        hits.append(Hit(title=_text(title)[:200], url=link,
                        snippet=snippets[index][:400] if index < len(snippets) else "",
                        backend="duckduckgo"))
        if len(hits) >= limit:
            break
    if not hits:
        _stand_down("duckduckgo")
    return hits


async def search_wikipedia(query: str, limit: int) -> list[Hit]:
    """The last resort, and a genuinely useful one.

    A poor general search engine — it only knows what has an article — but it
    never throttles and never returns a page that is really a cookie wall.
    """
    status, body = await _get(
        WIKI_SEARCH, {"User-Agent": _WIKI_AGENT},
        {"action": "query", "list": "search", "srsearch": query,
         "srlimit": str(max(1, min(20, limit))), "format": "json"})
    if status != 200:
        return []
    try:
        results = (json.loads(body).get("query") or {}).get("search") or []
    except Exception:
        return []
    hits = []
    for item in results[:limit]:
        title = str(item.get("title") or "")
        hits.append(Hit(
            title=f"Wikipedia: {title}",
            url="https://en.wikipedia.org/wiki/" + quote_plus(title.replace(" ", "_")),
            snippet=_text(str(item.get("snippet") or "")), backend="wikipedia"))
    return hits


GEMINI_SEARCH_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/"
                          "models/{model}:generateContent")
#: Tried in turn when the search model's quota refuses (each has its own).
SEARCH_FALLBACK_MODELS = ("gemini-2.5-flash-lite",)


def _retry_delay(raw: str) -> float:
    """Seconds Google asked us to wait ("retryDelay": "37s"), or 0."""
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', raw or "")
    return min(float(match.group(1)), COOLDOWN_S) if match else 0.0


def gemini_key() -> str:
    """The Gemini key ORION already holds for his voice, or ""."""
    key = os.getenv("ORION_GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        from .constants import API_CONFIG_PATH

        providers = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8")).get("providers") or {}
        for name in ("gemini_text", "gemini"):
            entry = providers.get(name) or {}
            value = str(entry.get("api_key") or "").strip()
            if value and entry.get("enabled", True):
                return value
    except Exception:
        pass
    return ""


async def _resolve_redirect(url: str) -> str:
    """Follow one redirect hop without downloading the page."""
    try:
        from aiohttp import ClientSession, ClientTimeout
    except Exception:
        return url
    try:
        async with ClientSession(timeout=ClientTimeout(total=8.0)) as session:
            async with session.get(url, allow_redirects=False,
                                   headers={"User-Agent": _USER_AGENT}) as response:
                location = response.headers.get("Location") or ""
                return location if location.startswith("http") else url
    except Exception:
        return url


async def search_google(query: str, limit: int) -> list[Hit]:
    """Google Search, through Gemini's search grounding.

    The keyless engines are the weak link in researching anything: DuckDuckGo
    throttles after one query and Wikipedia only knows encyclopaedia topics.
    ORION already holds a Gemini key for his voice, and Gemini can run a real
    Google search and say which pages it used — ranked, current, and not a
    scraper that breaks when a results page changes shape.

    The model only chooses queries and lists sources here; ORION still opens
    and reads every page himself.
    """
    key = gemini_key()
    if not key:
        return []
    try:
        from aiohttp import ClientSession, ClientTimeout
    except Exception:
        return []
    body = {
        "contents": [{"parts": [{"text":
            f"Search the web for: {query}\n\nFind the most authoritative, "
            f"substantive pages on this (primary sources, journals, official "
            f"bodies, reputable outlets). Briefly say what each covers."}]}],
        "tools": [{"google_search": {}}],
    }
    # Quotas are per model, and the per-minute one is shared with everything
    # else ORION says through the same key — a research run's own notes used
    # it up and every search after them came back 429. Flash-Lite has an
    # allowance of its own, so a refusal moves down the list before standing
    # the backend down.
    models = list(dict.fromkeys([os.getenv("ORION_SEARCH_MODEL", "gemini-2.5-flash"),
                                 *SEARCH_FALLBACK_MODELS]))
    status, raw, retry_after = 0, "", 0.0
    all_daily = True
    try:
        async with ClientSession(timeout=ClientTimeout(total=45.0)) as session:
            for model in models:
                async with session.post(GEMINI_SEARCH_ENDPOINT.format(model=model),
                                        json=body, headers={
                        "x-goog-api-key": key,
                        "Content-Type": "application/json"}) as response:
                    status, raw = response.status, await response.text(errors="replace")
                if status != 429:
                    break
                retry_after = max(retry_after, _retry_delay(raw))
                all_daily = all_daily and bool(re.search(r"(?i)per.?day", raw))
    except Exception:
        return []
    if status == 429:
        # A per-minute limit clears in a minute; standing down for the full
        # five would hand research to the weaker engines for no reason. A
        # per-DAY limit on every model clears at Google's reset (midnight US
        # Pacific) — asking again every minute until then only burns time.
        if all_daily:
            from .providers import ProviderRouter
            _stand_down("google", ProviderRouter.daily_reset_s())
        else:
            _stand_down("google", retry_after or 60.0)
        return []
    if status != 200:
        return []
    try:
        candidate = (json.loads(raw).get("candidates") or [{}])[0]
    except Exception:
        return []
    meta = candidate.get("groundingMetadata") or {}
    chunks = meta.get("groundingChunks") or []
    snippets: dict[int, list[str]] = {}
    for support in meta.get("groundingSupports") or []:
        text = str((support.get("segment") or {}).get("text") or "").strip()
        for index in support.get("groundingChunkIndices") or []:
            if text:
                snippets.setdefault(int(index), []).append(text)
    rows = []
    for index, chunk in enumerate(chunks):
        web = chunk.get("web") or {}
        link = str(web.get("uri") or "")
        if link.startswith("http"):
            rows.append((str(web.get("title") or ""), link,
                         " ".join(snippets.get(index, []))[:400]))
    rows = rows[:limit]
    # Grounding hands back redirect links; resolve them so a citation names
    # the page, not a redirector, and so the same page is not read twice.
    resolved = await asyncio.gather(*(_resolve_redirect(link) for _t, link, _s in rows))
    hits, seen = [], set()
    for (title, _link, snippet), real in zip(rows, resolved):
        if real in seen:
            continue
        seen.add(real)
        host = urlparse(real).hostname or title
        hits.append(Hit(title=title or host, url=real, snippet=snippet, backend="google"))
    return hits


#: A search that runs through an MCP server (set by app.py once the
#: DuckDuckGo MCP server is live): async (query, limit) -> text.
_MCP_SEARCH: Any = None


def attach_mcp_search(call: Any) -> None:
    """Let research search through the DuckDuckGo MCP server as well."""
    global _MCP_SEARCH
    _MCP_SEARCH = call


_MCP_RESULT = re.compile(r"^\s*\d+\.\s*(.+?)\s*$\s*URL:\s*(\S+)\s*$(?:\s*Summary:\s*(.+?)\s*$)?",
                         re.MULTILINE)


async def search_mcp_duckduckgo(query: str, limit: int) -> list[Hit]:
    """DuckDuckGo through its MCP server, which rate-limits itself politely
    rather than being refused like the raw HTML endpoint."""
    if _MCP_SEARCH is None:
        return []
    try:
        text = str(await _MCP_SEARCH(query, limit) or "")
    except Exception:
        return []
    hits = []
    for title, url, summary in _MCP_RESULT.findall(text):
        if url.startswith("http"):
            hits.append(Hit(title=_text(title)[:200], url=url,
                            snippet=_text(summary or "")[:400], backend="duckduckgo-mcp"))
        if len(hits) >= limit:
            break
    if not hits and "rate" in text.lower():
        _stand_down("duckduckgo-mcp")
    return hits


#: Tried in this order. Google (through the Gemini key ORION already has)
#: first because it is ranked, current and not scraped; Brave when a key
#: exists; DuckDuckGo keyless (through its MCP server first); Wikipedia last
#: because it always answers and rarely answers well.
#:
#: Bing was removed in Mark XXXI. Its HTML page had started answering
#: scripted requests with whatever it liked — "microplastics UK tap water"
#: came back as a list of adult sites, a long question as self-care blogs —
#: with a 200 and a normal-looking page, so nothing short of reading the
#: results could tell.
BACKENDS: tuple[tuple[str, Callable], ...] = (
    ("google", search_google),
    ("brave", search_brave),
    ("duckduckgo-mcp", search_mcp_duckduckgo),
    ("duckduckgo", search_duckduckgo),
    ("wikipedia", search_wikipedia),
)

#: Hosts never handed on for reading, whatever a backend returns.
_UNSAFE_HOST = re.compile(
    r"porn|xvideo|xnxx|xhamster|redtube|youporn|onlyfans|chaturbate|brazzers|"
    r"spankbang|livejasmin|stripchat|bongacams|camsoda|hentai|rule34|nsfw",
    re.IGNORECASE)

#: Backends whose results are checked against the question before use. Google
#: and Brave rank for the query they were given; the keyless ones are scraped
#: or encyclopaedic and can answer with pages that share no word with it.
_GATED = frozenset({"duckduckgo-mcp", "duckduckgo", "wikipedia"})

#: Words that say nothing about what a page must be about.
_STOPWORDS = frozenset("""
    what which when where whom whose does have been being were with from that
    this these those into onto about than then there their they them your
    yours some many much more most other such only also very could would
    should shall might must between among across over under after before
    during within without upon anything something everything nothing
    please find search show tell give information
""".split())


def _content_words(query: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(query).lower())
            if len(w) >= 4 and w not in _STOPWORDS]


def _relevant(hit: Hit, words: list[str]) -> bool:
    """True when the result shares enough of the question's words.

    Compared with the spaces taken out, because one backend strips the bold
    from matched terms without putting the spaces back ("energydensityin").
    Six-letter prefixes let "concentrations" meet "concentration".
    """
    if not words:
        return True
    blob = re.sub(r"[^a-z0-9]", "", f"{hit.title} {hit.snippet} {unquote(hit.url)}".lower())
    matched = sum(1 for word in words if word[:6] in blob)
    return matched >= (1 if len(words) <= 3 else 2)


def _usable(hits: list[Hit], name: str, query: str) -> list[Hit]:
    safe = [h for h in hits if not _UNSAFE_HOST.search(urlparse(h.url).hostname or "")]
    if name not in _GATED:
        return safe
    words = _content_words(query)
    return [h for h in safe if _relevant(h, words)]


async def search(query: str, limit: int = 8,
                 say: Callable[[str, str], None] | None = None) -> list[Hit]:
    """Find pages, trying each backend until one answers.

    Raises :class:`SearchUnavailable` when every backend declines, rather than
    returning an empty list — the caller needs to tell the difference between
    an obscure topic and a blocked engine, and so does whoever reads the
    report afterwards.
    """
    query = str(query or "").strip()
    if not query:
        return []

    tried, skipped = [], []
    for name, backend in BACKENDS:
        if not _available(name):
            skipped.append(name)
            continue
        tried.append(name)
        try:
            raw = await backend(query, limit)
        except Exception:
            raw = []
        hits = _usable(raw, name, query)
        if raw and not hits and say is not None:
            say("search_backend", f"{name} answered with pages unrelated to the "
                                  "question — trying the next engine")
        if hits:
            if say is not None:
                keyless = name in {"duckduckgo-mcp", "duckduckgo", "wikipedia"}
                say("search_backend",
                    f"searched with {name} — {len(hits)} result(s)"
                    + (" (a Gemini or Brave API key makes searching reliable)"
                       if keyless and not gemini_key() and not brave_key() else ""))
            return hits

    if say is not None:
        detail = ", ".join(tried) or "none"
        if skipped:
            detail += f" (standing down: {', '.join(skipped)})"
        say("search_unavailable",
            f"no search backend would answer — tried {detail}."
            + ("" if gemini_key() else
               " A Gemini key (ORION's voice key) makes Google search available."))
    raise SearchUnavailable(
        f"no search backend answered for {query!r}; tried {tried or 'none'}")


def status() -> dict[str, Any]:
    """Which backends are usable right now, for diagnostics."""
    return {
        "google": bool(gemini_key()),
        "brave_key": bool(brave_key()),
        "standing_down": {name: round(max(0.0, until - time.monotonic()), 1)
                          for name, until in _STOOD_DOWN.items()
                          if until > time.monotonic()},
        "order": [name for name, _ in BACKENDS],
    }


def reset() -> None:
    """Forget the cool-downs. For tests, and for 'try again now'."""
    _STOOD_DOWN.clear()


__all__ = ["BACKENDS", "COOLDOWN_S", "Hit", "SearchUnavailable",
           "brave_key", "gemini_key", "reset", "search",
           "search_brave", "search_duckduckgo", "search_google",
           "search_wikipedia", "status"]
