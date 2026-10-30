"""
mcp_host.py — ORION as a Model Context Protocol (MCP) host.

The user asked for ORION to "use all relevant MCPs" and to link his calendar to
Gmail through one.  MCP (modelcontextprotocol.io) is the open standard for
exposing tools/data to an assistant; a *host* connects to MCP *servers* (small
programs — Gmail, Google Calendar, filesystem, web search, …) and makes their
tools callable.  This module makes ORION that host.

It is a self-contained MCP client — a minimal JSON-RPC 2.0 implementation over
the stdio transport (newline-delimited messages), so it adds NO new Python
dependency and cannot break the rest of ORION if the ``mcp`` SDK is absent.
Servers are declared in ``config/mcp_servers.json`` (a template is written on
first run); each is a command ORION launches as a subprocess.  Nothing here
ever raises to the caller — a missing command, a bad handshake or a server
crash is logged and that server is simply skipped, so the app always starts.

Model integration (widened in the Mark XXI pass, Track E1): each connected
server's tools are ALSO registered as first-class, namespaced dispatcher
tools ("mcp__<server>__<tool>", see OrionDispatcher.register_mcp_tools) —
callable directly, the same as any native ORION tool, not only through the
generic ``mcp`` list/call indirection (which still exists, for browsing).
The generic tool remains useful for discovery ("what can this server do")
and as a fallback if a namespaced call is ever mistyped.

Reconnection (Track E2): a dropped or crashed server no longer stays gone
until the next full app restart — MCPHost.supervise() watches every
enabled server and reconnects with bounded backoff, re-registering its
tools the moment it's back; per-server health is reported through the
attached Telemetry, when one is given.
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .constants import BASE_DIR, CONFIG_DIR
from .atomic_io import atomic_write_text
from .mcp_policy import _SECRET as _SECRET_RE

MCP_CONFIG_PATH = CONFIG_DIR / "mcp_servers.json"
#: Each server's tool list from its last successful start, so the next boot
#: can offer the tools without launching the process (see _cached_tools).
MCP_CATALOGUE_PATH = CONFIG_DIR / "mcp_catalogue.json"
_PROTOCOL_VERSION = "2024-11-05"
_MODERN_PROTOCOL_VERSION = "2026-07-28"
def _bundled_spec(name: str) -> tuple[str, list[str]]:
    """How to start one of ORION's own MCP servers (mcp_bundled.launch_spec)."""
    from .mcp_bundled import launch_spec
    return launch_spec(name)


_SUPPORTED_LEGACY_VERSIONS = frozenset({
    "2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25",
})
# A uvx server can spend several seconds importing its document libraries.
# Treat that as startup latency rather than evidence of a legacy protocol.
_DISCOVER_TIMEOUT = 30.0
_CALL_TIMEOUT = 45.0
#: Long enough for the FIRST launch, when npx/uvx download the server package
#: before it can answer. 25 s failed every first run on a home connection and
#: the server was then reported as broken rather than as still installing.
_HANDSHAKE_TIMEOUT = 120.0
#: A server nobody has called for this long is paused (its process stopped,
#: its tool list kept) and restarts on its next call. Ten servers held about
#: a gigabyte between them while doing nothing. ORION_MCP_IDLE_PARK_S=0 keeps
#: every server running.
_IDLE_PARK_S = 1800.0
# Discovery is atomic and bounded even if a server emits endless cursors.
_CATALOGUE_MAX_PAGES = 128
_CATALOGUE_MAX_TOOLS = 10_000

#: Bumped when migrate_config learns a new fix; recorded in the config so a
#: server the user later switches off is not switched back on.
CONFIG_REVISION = "mark33-desktop-targets"

#: Env names that hold a credential. An enabled server with one of these empty
#: cannot work, so it is reported as waiting for that key — not launched,
#: failed, retried and reported as broken every thirty seconds.
# Per-tool confirmation lists for the shipped entries (see requires_confirmation).
# Exact tool names, so a tool a server adds later is judged when it is added
# rather than swept in or out by a pattern.
#: mcp-server-git's tools that only read. The server is "confirm": true with
#: these excepted rather than listing the writes: its tool set is small and
#: stable, a tool it adds later asks until judged, and an ORION build that
#: predates confirm_except still gates every git tool instead of none.
_GIT_READ_TOOLS = ("git_status", "git_diff_unstaged", "git_diff_staged", "git_diff",
                   "git_log", "git_show", "git_branch")
#: github-mcp-server: landing, deleting or publishing code, and running
#: workflows. Issues, comments and pull-request text can be edited afterwards.
_GITHUB_CONFIRM_TOOLS = ("merge_pull_request", "push_files", "create_or_update_file",
                         "delete_file", "create_repository", "fork_repository",
                         "update_pull_request_branch", "actions_run_trigger")
#: @playwright/mcp: the RCE-equivalent code runner, and the two tools that can
#: hand a local file to a web page (the way a prompt-injected page would
#: exfiltrate one).
_PLAYWRIGHT_CONFIRM_TOOLS = ("browser_run_code_unsafe", "browser_file_upload",
                             "browser_drop")
#: Home Assistant's Assist API tools that only read. Everything else it
#: exposes — every intent, every script — acts on the house, so it asks.
#: Filesystem tools that change a file. write_file replaces a whole file with
#: no undo, and the allowed folders can include ORION's own source.
_FILESYSTEM_CONFIRM_TOOLS = ("write_file", "edit_file", "move_file")
_HOME_ASSISTANT_READ_TOOLS = ("GetLiveContext", "GetDateTime", "HassTimerStatus",
                              "todo_get_items", "calendar_get_events")

# The shipped Gmail and Calendar servers can send messages and change the
# user's account. Older local configs have no confirm rules, so enforce these
# read-only exceptions in code as well. Unknown tools ask by default: a server
# update must not silently gain permission to act on the user's behalf.
_ACCOUNT_READ_TOOLS = {
    "gmail": frozenset({"read-email", "search-emails", "list-email-labels"}),
    "google_calendar": frozenset({
        "list-calendars", "list-events", "search-events", "get-event",
        "list-colors", "get-freebusy", "get-current-time",
    }),
}


def _default_config() -> dict[str, Any]:
    """The starting configuration. The recommended servers (fetch, DuckDuckGo,
    Notion — see _RECOMMENDED) ship ON because they need nothing ORION does
    not already have; everything needing an account ships disabled with
    empty credentials. Gmail/Calendar need Google OAuth the user sets up
    themselves — see docs/MCP_SETUP.md."""
    template = _default_template()
    template["servers"].update(json.loads(json.dumps(_RECOMMENDED)))
    template["_revision"] = CONFIG_REVISION
    return template


def _default_template() -> dict[str, Any]:
    return {
        "_comment": (
            "ORION MCP servers. Set 'enabled': true and fill credentials to "
            "activate one. Node servers run through npx and Python ones through "
            "uvx (ORION finds both even when they are not on PATH). A server "
            "with an empty credential is shown as waiting for it rather than "
            "launched. See docs/MCP_SETUP.md for Gmail/Calendar OAuth."
        ),
        "servers": {
            "orion_telemetry": {
                "enabled": False,
                "command": _bundled_spec("orion_telemetry")[0],
                "args": _bundled_spec("orion_telemetry")[1],
                "env": {},
                "description": "Local read-only CPU, RAM and process-memory telemetry; no account or API fee.",
            },
            "gmail": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
                "env": {},
                "description": "Gmail: read, search, send and manage email.",
            },
            "google_calendar": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@cocal/google-calendar-mcp"],
                "env": {"GOOGLE_OAUTH_CREDENTIALS": ""},
                "description": "Google Calendar: list, create and update events; links to your Gmail account.",
            },
            # Named folders, never the whole profile: the home directory holds
            # ORION's own config (and its secrets) and every app's data.
            "filesystem": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem",
                         *(os.path.join(os.path.expanduser("~"), folder)
                           for folder in ("Desktop", "Documents", "Downloads"))],
                "env": {},
                "confirm_tools": list(_FILESYSTEM_CONFIRM_TOOLS),
                "confirm_reason": "This changes or overwrites a file on your computer.",
                "description": ("Local filesystem: read, write and search files "
                                "in the listed folders only."),
            },
            # ── SEARCH: a web index ORION can query without scraping ────────
            # Brave's Search API has a free tier (2,000 queries a month at the
            # time of writing) and an official MCP server. It answers the
            # "look this up" half of research without ORION driving a browser,
            # which is slower and more fragile.
            # Setup: an API key from brave.com/search/api, then enabled:true.
            "brave_search": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@brave/brave-search-mcp-server"],
                "env": {"BRAVE_API_KEY": ""},
                "description": ("Web search: query Brave's index directly, "
                                "with a free tier, instead of driving a "
                                "browser."),
            },
            # ── GITHUB: repositories, issues, pull requests ──────────────────
            # GitHub's own server (github/github-mcp-server); the old npm
            # @modelcontextprotocol/server-github is deprecated. It ships as a
            # release binary, not a package. A fine-grained personal access
            # token scoped to the repositories you want is the right
            # credential — a classic token with full scopes hands ORION every
            # repository you can see. Changes that land code, delete it or
            # run workflows need an on-screen yes (confirm_tools).
            "github": {
                "enabled": False,
                "command": os.path.join(os.path.expanduser("~"), "ORION", "mcp",
                                        "github-mcp-server", "github-mcp-server.exe"),
                "args": ["stdio"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "",
                        "GITHUB_TOOLSETS": "context,repos,issues,pull_requests,actions"},
                "confirm_tools": list(_GITHUB_CONFIRM_TOOLS),
                "confirm_reason": ("changes code, branches or workflows on GitHub, "
                                   "under your name"),
                "description": ("GitHub: repositories, commits, issues, pull "
                                "requests and Actions. Binary from "
                                "github.com/github/github-mcp-server/releases."),
            },
            # ── HOME ASSISTANT: the house ────────────────────────────────────
            # Home Assistant's OWN MCP server (the "Model Context Protocol
            # Server" integration, at /api/mcp) — not a third-party package.
            # It speaks Streamable HTTP, so mcp-proxy bridges it to stdio, as
            # Home Assistant's docs recommend. mcp-proxy is held below mcp 2
            # (0.12.0 does not import against it) and kept at WARNING: at INFO
            # it logs every request's headers, bearer token included, into the
            # stderr tail a failed handshake quotes.
            #
            # API_ACCESS_TOKEN is a long-lived access token from your own
            # profile page, and it can do anything you can — including
            # unlocking whatever your Home Assistant unlocks. So every tool
            # asks first except the ones that only read state.
            "home_assistant": {
                "enabled": False,
                "confirm": True,
                "confirm_except": list(_HOME_ASSISTANT_READ_TOOLS),
                "confirm_reason": "controls a real device or service in your home",
                "command": "uvx",
                "args": ["--with", "mcp>=1.17,<2", "mcp-proxy@0.12.0",
                         "--log-level", "WARNING", "--transport=streamablehttp",
                         "--stateless", "http://homeassistant.local:8123/api/mcp"],
                "env": {"API_ACCESS_TOKEN": ""},
                "description": ("Home Assistant: read and control lights, "
                                "climate, media, timers and scripts. Needs the "
                                "MCP Server integration enabled in Home "
                                "Assistant; edit the URL in args."),
            },
            # ── SEARCH, without an account ───────────────────────────────────
            # DuckDuckGo needs no key at all, which makes it the one to try
            # first: an integration that requires signing up is an integration
            # that stays off. Brave above is better ranked and rate-limited by
            # a key; this is the zero-friction fallback.
            "duckduckgo": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "duckduckgo-mcp-server"],
                "env": {},
                "description": ("Web search with no API key and no account, "
                                "for when Brave's free tier runs out."),
            },
            # ── OBSIDIAN: the notes vault ────────────────────────────────────
            # Reads and writes the markdown files directly, so it works
            # whether or not Obsidian itself is running. Point it at the vault
            # FOLDER — and be aware this gives ORION read/write access to
            # everything in it, which for most people is years of private
            # notes. A vault ORION should not read is one to leave out.
            "obsidian": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "mcp-obsidian", "/path/to/your/vault"],
                "env": {},
                "description": ("Obsidian vault: search, read and write notes "
                                "as plain markdown files."),
            },
            # ── DATABASES: ask questions of your own data ────────────────────
            # Google's MCP Toolbox for Databases serves both (the reference
            # SQLite and Postgres servers are archived, and the Postgres one's
            # read-only guard could be escaped). SQLite gets a database of its
            # OWN — not one of ORION's stores, which a stray UPDATE would
            # corrupt while ORION holds them open.
            "sqlite": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@toolbox-sdk/server", "--prebuilt=sqlite",
                         "--stdio", "--disable-version-check"],
                "env": {"SQLITE_DATABASE": str(CONFIG_DIR / "mcp_sqlite.db")},
                "description": ("SQLite: a scratch database of ORION's own for "
                                "tables he builds — list tables, run SQL."),
            },
            # Use a READ-ONLY role: a model composing SQL against a writable
            # production database is a bad afternoon waiting to happen, and
            # execute_sql cannot tell a SELECT from a DROP; the role can.
            "postgres": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@toolbox-sdk/server", "--prebuilt=postgres/data",
                         "--stdio", "--disable-version-check",
                         "--defer-source-connect"],
                "env": {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
                        "POSTGRES_DATABASE": "", "POSTGRES_USER": "",
                        "POSTGRES_PASSWORD": ""},
                "description": ("PostgreSQL: list schemas and tables and run "
                                "SQL. Point it at a READ-ONLY role."),
            },
            # ── GIT: this repository, read freely, changed only on a yes ─────
            "git": {
                "enabled": False,
                "command": "uvx",
                "args": ["mcp-server-git", "--repository", str(BASE_DIR)],
                "env": {},
                "confirm": True,
                "confirm_except": list(_GIT_READ_TOOLS),
                "confirm_reason": "changes the repository (index, commits or branches)",
                "description": ("Git: status, diffs, log and history of the "
                                "ORION repository."),
            },
            # ── BROWSER AUTOMATION: operate websites ─────────────────────────
            # Microsoft's server. Headless, because the browser co-pilot is
            # ORION's visible browser; isolated, so it carries none of your
            # logins. The RCE-equivalent code runner and the two tools that
            # can hand a local file to a web page ask first.
            "playwright": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@playwright/mcp", "--headless", "--isolated",
                         "--browser", "chrome", "--image-responses", "omit",
                         "--idle-timeout", "300000", "--file-paths", "absolute",
                         "--output-dir", os.path.join(os.path.expanduser("~"), "ORION",
                                                      "mcp", "playwright-output"),
                         "--output-max-size", "52428800"],
                "env": {},
                "confirm_tools": list(_PLAYWRIGHT_CONFIRM_TOOLS),
                "confirm_reason": ("runs arbitrary code or hands a local file "
                                   "to a web page"),
                "description": ("Operate websites in a headless browser: "
                                "navigate, read the page, click, type, fill forms."),
            },
            # ── MEMORY: a knowledge graph of entities and relations ──────────
            "memory": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
                "env": {"MEMORY_FILE_PATH": str(CONFIG_DIR / "mcp_memory.jsonl")},
                "description": ("Knowledge graph: entities, relations and "
                                "observations kept across sessions."),
            },
            # ── GOOGLE DRIVE ─────────────────────────────────────────────────
            # Google's own Drive MCP server is remote-only and needs the
            # Workspace Developer Preview; workspace-mcp runs locally and works
            # with a personal account. Pinned, read-only, Drive tools only.
            "google_drive": {
                "enabled": False,
                "command": "uvx",
                "args": ["workspace-mcp@1.29.0", "--single-user", "--tools",
                         "drive", "--read-only"],
                "env": {"GOOGLE_OAUTH_CLIENT_ID": "", "GOOGLE_OAUTH_CLIENT_SECRET": "",
                        "OAUTHLIB_INSECURE_TRANSPORT": "1",
                        "WORKSPACE_MCP_CREDENTIALS_DIR": str(CONFIG_DIR / "mcp_google_drive")},
                "description": ("Google Drive: search, list and read your files "
                                "(read-only). Needs a Google OAuth desktop client."),
            },
            # ── DOCUMENTATION + CONVERSION ───────────────────────────────────
            "context7": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp"],
                "env": {},
                "description": ("Current, version-specific documentation and "
                                "code examples for libraries and frameworks."),
            },
            "markitdown": {
                "enabled": False,
                "command": "uvx",
                "args": ["markitdown-mcp"],
                "env": {},
                "description": ("Convert PDF, Word, PowerPoint, Excel, EPUB, "
                                "HTML and more to Markdown from a file or URL."),
            },
            # ── MATRIX: messaging on a server you choose ─────────────────────
            # Free homeservers exist and you can run your own, which is the
            # difference from the other chat integrations. The access token is
            # a full login — it can read every room the account is in.
            "matrix": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "matrix-mcp-server"],
                "env": {"MATRIX_HOMESERVER": "https://matrix.org",
                        "MATRIX_USER_ID": "@you:matrix.org",
                        "MATRIX_ACCESS_TOKEN": ""},
                "description": ("Matrix: read and send messages in rooms, on "
                                "any homeserver including your own."),
            },
            # ── CAL.COM: scheduling ──────────────────────────────────────────
            # Open-source scheduling with a free tier. Useful mostly if people
            # book time with you; if you only keep your own calendar, the
            # google_calendar entry above is the simpler thing.
            "calcom": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@calcom/mcp"],
                "env": {"CALCOM_API_KEY": ""},
                "description": ("Cal.com: list and manage bookings other "
                                "people have made with you."),
            },
            # ── TELEPHONY: let ORION place REAL outbound phone calls + SMS ──────
            # Twilio's MCP server. Unlike phone_action (which pre-fills the dialer
            # on your paired Android phone for you to tap), this DIALS DIRECTLY via
            # your Twilio account — real calls cost money and connect immediately,
            # so treat every call/SMS tool here as a CONFIRMED action.
            # Setup: create a Twilio account + an API key, then fill the three env
            # values and set enabled:true. Exact args can vary by package version —
            # see docs/MCP_SETUP.md (Telephony). Alternatives: any telephony MCP
            # that speaks stdio works here in the same shape.
            # NOTE: launching this needs npx, i.e. Node. A Windows machine
            # without Node has none, and then this server can never start
            # whatever the credentials say — which is exactly how outbound
            # calling appeared broken for months. ORION no longer depends on
            # it: telephony_direct.py reaches Twilio's REST API with nothing
            # but the credentials, and `python tools/setup_twilio.py` sets
            # those in one step. This server is still preferred when it IS
            # connected, because it does more than dial.
            "twilio": {
                "enabled": False,
                # Enforced, not merely described: mcp 'call' refuses tools on
                # a server marked this way unless confirm=true is passed.
                "confirm": True,
                "command": "npx",
                "args": ["-y", "@twilio-alpha/mcp"],
                "env": {
                    "TWILIO_ACCOUNT_SID": "",
                    "TWILIO_API_KEY": "",
                    "TWILIO_API_SECRET": "",
                    "TWILIO_FROM_NUMBER": "",
                },
                "description": ("Telephony: place REAL outbound phone calls and send "
                                "SMS to any number via your Twilio account. Calls dial "
                                "immediately and cost money — confirm before calling."),
            },
        },
    }


def _tool_dirs() -> list[str]:
    """Where Node, npx and uvx land on Windows, whether or not PATH knows yet.

    A tool installed while ORION is running is not on THIS process's PATH
    until he restarts, and a winget/MSI install may not have updated PATH at
    all for a process started earlier. Looking in the standard places means
    the servers start the moment the tools exist.
    """
    env = os.environ
    candidates = [
        os.path.join(env.get("ProgramFiles", r"C:\Program Files"), "nodejs"),
        os.path.join(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "nodejs"),
        os.path.join(env.get("APPDATA", ""), "npm"),
        os.path.join(env.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links"),
        os.path.join(os.path.expanduser("~"), ".local", "bin"),
        os.path.join(os.path.expanduser("~"), ".cargo", "bin"),
        os.path.join(env.get("LOCALAPPDATA", ""), "Programs", "nodejs"),
    ]
    return [path for path in candidates if path and os.path.isdir(path)]


def resolve_command(command: str) -> str:
    """The full path of *command*, or "" if it cannot be found.

    On Windows ``npx`` is ``npx.cmd``. asyncio's subprocess launcher calls
    CreateProcess, which appends only ``.exe`` to a bare name — so "npx" was
    never found even on a machine with Node installed, and every Node server
    failed to launch. The resolved path carries the right extension.
    """
    command = str(command or "").strip()
    if not command:
        return ""
    if os.path.isabs(command):
        return command if os.path.exists(command) else ""
    found = shutil.which(command)
    if found:
        return found
    search = os.pathsep.join(_tool_dirs())
    return shutil.which(command, path=search) or ""


def missing_credentials(spec: dict[str, Any]) -> list[str]:
    """The credential env vars an enabled server still needs filled in."""
    env = spec.get("env") or {}
    if "TWILIO_ACCOUNT_SID" in env:
        missing = [] if str(env.get("TWILIO_ACCOUNT_SID") or "").strip() else ["TWILIO_ACCOUNT_SID"]
        token = str(env.get("TWILIO_AUTH_TOKEN") or "").strip()
        key = str(env.get("TWILIO_API_KEY") or "").strip()
        secret = str(env.get("TWILIO_API_SECRET") or "").strip()
        if not token and not (key and secret):
            missing.append("TWILIO_AUTH_TOKEN or TWILIO_API_KEY + TWILIO_API_SECRET")
        return missing
    return [str(k) for k, v in env.items()
            if _SECRET_RE.search(str(k)) and not str(v or "").strip()]


def _tool_listed(tool: str, patterns: Any) -> bool:
    """Whether *tool* is named in *patterns* — exact names or shell wildcards
    ("git_*"), ignoring case. A lone string counts as a list of one."""
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, (list, tuple)):
        return False
    tool = str(tool or "").strip().lower()
    return any(fnmatch.fnmatchcase(tool, str(p).strip().lower())
               for p in patterns if str(p or "").strip())


def configured_confirmation(server: str, tool: str, spec: dict[str, Any]) -> str:
    """Why a server's OWN config says *tool* must be confirmed, or "".

    * ``"confirm": true`` — every tool, except any named in ``"confirm_except"``
      (for a server where nearly everything acts: Home Assistant, whose
      state-reading tools need not ask).
    * ``"confirm_tools": [...]`` — only the named tools (for a server that is
      mostly reading: git, whose commit and checkout should ask and whose
      status and log should not).
    * ``"confirm_reason"`` — what the person is approving, in words; shown in
      the prompt instead of the bare "marked as requiring confirmation".
    """
    if not isinstance(spec, dict):
        return ""
    why = str(spec.get("confirm_reason") or "").strip()
    if spec.get("confirm"):
        if _tool_listed(tool, spec.get("confirm_except")):
            return ""
        return (f"{server}.{tool} {why}" if why
                else f"'{server}' is marked as requiring confirmation")
    if _tool_listed(tool, spec.get("confirm_tools")):
        return (f"{server}.{tool} {why}" if why
                else f"'{server}.{tool}' is marked as requiring confirmation")
    return ""


def autofill_credentials(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Fill credentials ORION already holds, so they are not asked for twice.

    The Notion token is in api_keys.json's integrations (ORION's own Notion
    tool uses it); Brave/Tavily/GitHub keys may be in the environment. Values
    the user has typed into the server's own env always win.
    """
    spec = dict(spec)
    env = dict(spec.get("env") or {})
    known: dict[str, str] = {}
    try:
        from .providers import read_provider_settings

        notion = read_provider_settings().integration("notion")
        if str(notion.get("token") or "").strip():
            known["NOTION_TOKEN"] = str(notion["token"]).strip()
    except Exception:
        pass
    for var in ("BRAVE_API_KEY", "TAVILY_API_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "GOOGLE_MAPS_API_KEY", "SLACK_BOT_TOKEN"):
        if os.getenv(var, "").strip():
            known[var] = os.environ[var].strip()
    if os.getenv("GITHUB_TOKEN", "").strip() and "GITHUB_PERSONAL_ACCESS_TOKEN" not in known:
        known["GITHUB_PERSONAL_ACCESS_TOKEN"] = os.environ["GITHUB_TOKEN"].strip()
    for key, value in list(env.items()):
        if not str(value or "").strip() and key in known:
            env[key] = known[key]
    spec["env"] = env
    return spec


#: Servers worth having on out of the box: useful, and needing nothing ORION
#: does not already have. Anything else stays off until the user turns it on.
_RECOMMENDED: dict[str, dict[str, Any]] = {
    "orion_desktop": {
        "enabled": sys.platform == "win32",
        "command": _bundled_spec("orion_desktop")[0],
        "args": _bundled_spec("orion_desktop")[1],
        "env": {},
        "description": ("On-demand foreground control labels, automation IDs and physical monitor geometry. "
                        "Read-only targeting; native desktop_control owns clicks and their verification."),
    },
    "orion_world": {
        "enabled": True,
        "command": _bundled_spec("orion_world")[0],
        "args": _bundled_spec("orion_world")[1],
        "env": {},
        "description": ("Official read-only CelesTrak satellite predictions, USGS earthquakes, "
                        "NOAA tsunami bulletins and Met Office UK weather warnings; "
                        "shared cache, attributed sources, no account or API key."),
    },
    "orion_network": {
        "enabled": True,
        "command": _bundled_spec("orion_network")[0],
        "args": _bundled_spec("orion_network")[1],
        "env": {},
        "description": ("Read-only checks of this PC's Tailscale name and "
                        "ORION phone gateway listener; no account or API key."),
    },
    "fetch": {
        "enabled": True,
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": {},
        "description": ("Fetch any web page as clean text or Markdown — ORION "
                        "reads sources with it when researching."),
    },
    "duckduckgo": {
        "enabled": True,
        "command": "uvx",
        "args": ["duckduckgo-mcp-server"],
        "env": {},
        "description": ("Web search with no API key and no account, plus page "
                        "fetching — a second search engine for research."),
    },
    "notion": {
        "enabled": True,
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "env": {"NOTION_TOKEN": ""},
        "description": ("Notion: search, read and update any page or database "
                        "the integration can see (token taken from ORION's "
                        "Notion settings)."),
    },
}


def migrate_config(data: dict[str, Any]) -> bool:
    """Repair known-broken entries and add the recommended servers. Returns
    whether anything changed. Preserve explicit enabled choices across
    revisions so a server the user switched off stays off."""
    servers = data.setdefault("servers", {})
    if not isinstance(servers, dict):
        return False
    changed = False
    # Always-safe repairs: these could never have started as written.
    fetch = servers.get("fetch")
    if isinstance(fetch, dict) and fetch.get("command") == "npx" and \
            "mcp-server-fetch" in " ".join(map(str, fetch.get("args") or [])):
        # mcp-server-fetch is a PYTHON package; there is no npm one.
        fetch["command"], fetch["args"] = "uvx", ["mcp-server-fetch"]
        changed = True
    brave = servers.get("brave_search")
    if isinstance(brave, dict) and "@modelcontextprotocol/server-brave-search" in \
            " ".join(map(str, brave.get("args") or [])):
        brave["args"] = ["-y", "@brave/brave-search-mcp-server"]   # the old one is deprecated
        changed = True
    ddg = servers.get("duckduckgo")
    if isinstance(ddg, dict) and ddg.get("command") == "npx":
        # The npm package of that name is a stale stub; the maintained server
        # is the Python one.
        ddg["command"], ddg["args"] = "uvx", ["duckduckgo-mcp-server"]
        changed = True
    if data.get("_revision") != CONFIG_REVISION:
        for name, spec in _RECOMMENDED.items():
            existing = servers.get(name)
            if not isinstance(existing, dict):
                servers[name] = json.loads(json.dumps(spec))
            else:
                existing.setdefault("enabled", spec["enabled"])
                existing.setdefault("description", spec["description"])
        data["_revision"] = CONFIG_REVISION
        changed = True
    if "orion_telemetry" not in servers:
        servers["orion_telemetry"] = _default_template()["servers"]["orion_telemetry"]
        changed = True
    return changed


def _host_runtime_servers(data: dict[str, Any]) -> dict[str, Any]:
    """Bundled Python servers use the same working runtime as the host.

    Frozen, that is the executable itself (`ORION.exe --mcp-server NAME`, see
    mcp_bundled): a saved config still pointing `python` at the loose
    _internal copy of the module is rewritten here, because that copy cannot
    import the rest of ORION - orion_world died on `from orion_core.world_feeds`."""
    from .mcp_bundled import FILES, FLAG
    frozen = getattr(sys, "frozen", False)
    for name, filename in FILES.items():
        spec = data.get("servers", {}).get(name)
        if not isinstance(spec, dict):
            continue
        args = spec.get("args") or []
        expected = Path(__file__).with_name(filename)
        if frozen:
            ours = (args == [FLAG, name]
                    or (len(args) == 1 and Path(str(args[0])).name == filename
                        and Path(str(args[0])).parent.name == "orion_core"))
            if ours:
                spec["command"], spec["args"] = _bundled_spec(name)
        elif args == [str(expected)]:
            spec["command"] = sys.executable
    return data


class _UnavailableConfig(dict[str, Any]):
    """Keep absent policy distinct from an intentionally empty server list."""

    error = ("MCP configuration is unavailable. Restore or repair mcp_servers.json "
             "before starting servers or changing their settings; the file has been preserved.")

    def __init__(self) -> None:
        super().__init__(servers={})


def load_config() -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate configuration field")
            result[key] = value
        return result

    try:
        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object)
        if (not isinstance(data, dict) or not isinstance(data.get("servers"), dict)
                or any(not isinstance(spec, dict)
                       or "enabled" in spec and type(spec["enabled"]) is not bool
                       for spec in data["servers"].values())):
            return _UnavailableConfig()
        if migrate_config(data):
            try:
                atomic_write_text(MCP_CONFIG_PATH, json.dumps(data, indent=2),
                                  encoding="utf-8")
            except OSError:
                pass
        return _host_runtime_servers(data)
    except FileNotFoundError:
        # First run — drop a template so the user can see the shape and enable one.
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(MCP_CONFIG_PATH, json.dumps(_default_config(), indent=2), encoding="utf-8")
        except OSError:
            pass
    except (OSError, ValueError, TypeError, RecursionError):
        # A retained unreadable policy is not a new installation. Do not
        # enable defaults, import independent settings or overwrite its bytes.
        return _UnavailableConfig()
    return _host_runtime_servers(_default_config())


#: How long a server gets to exit on its own, and again after being forced.
_EXIT_GRACE_S = 2.0


class MCPProtocolError(RuntimeError):
    """A JSON-RPC error reply: the server refused the request itself (unknown
    method, invalid params), as opposed to a tool that ran and reported an
    error in its result."""

    def __init__(self, error: Any) -> None:
        err = error if isinstance(error, dict) else {"message": str(error)}
        self.code = err.get("code")
        self.message = str(err.get("message") or "request rejected")
        self.data = err.get("data")
        detail = self.message if self.code is None else f"{self.message} (JSON-RPC {self.code})"
        if self.data:
            detail += f": {str(self.data)[:300]}"
        super().__init__(detail)


class _MalformedReply(MCPProtocolError):
    """A correlated response violated JSON-RPC framing or MCP's result shape."""


class _LegacyProbeClosed(RuntimeError):
    """The server exited when probed before the legacy initialize request."""


class MCPReply(str):
    """What ``MCPHost.call`` returns: the reply text, plus how it failed.

    A ``str`` so every caller that treats replies as text keeps working, but
    one that carries its outcome, so failure is not guessed from wording (a
    successful read of a file that happens to begin "(tool error)" is not a
    failure). ``kind`` is "" on success, else one of ``ERROR_KINDS``.
    """

    ERROR_KINDS = ("tool", "protocol", "timeout", "unavailable")
    kind: str
    content: list[dict[str, Any]]
    structured_content: dict[str, Any] | None
    meta: dict[str, Any]
    provenance: dict[str, Any]

    def __new__(cls, text: str, kind: str = "", *, content: Any = None,
                structured_content: Any = None, meta: Any = None,
                provenance: Any = None) -> "MCPReply":
        reply = super().__new__(cls, text)
        reply.kind = kind
        # Keep the wire result separate from its backwards-compatible spoken
        # summary. Callers can retain image/resource/structured evidence without
        # guessing it from bracketed text or fetching an untrusted resource URI.
        reply.content = copy.deepcopy(content or [])
        reply.structured_content = copy.deepcopy(structured_content)
        reply.meta = copy.deepcopy(meta or {})
        reply.provenance = copy.deepcopy(provenance or {})
        return reply

    @property
    def is_error(self) -> bool:
        return bool(self.kind)

    def as_tool_receipt(self) -> dict[str, Any]:
        """Snapshot protocol content separately from the spoken summary."""
        return copy.deepcopy({
            "type": "mcp", "content": self.content,
            "structuredContent": self.structured_content, "_meta": self.meta,
            "provenance": self.provenance, "isError": self.is_error,
            "errorKind": self.kind, "untrusted": True,
        })


_BACKGROUND_BUNDLED_TOOLS = {
    "orion_desktop": frozenset({"desktop_target_status", "desktop_monitors",
                                 "desktop_controls", "desktop_find_target"}),
    "orion_telemetry": frozenset({"system_snapshot", "process_memory"}),
    "orion_network": frozenset({"tailnet_status", "phone_gateway_status"}),
    "orion_world": frozenset({"world_satellites", "world_earthquakes", "world_tsunami",
                               "world_weather", "world_events", "world_feed_status"}),
}


def _background_mcp_refusal(server: str, tool: str, spec: Any) -> MCPReply | None:
    """Enforce the host policy before crossing a native process boundary.

    ContextVars are not a subprocess sandbox. Only exact tools of reviewed,
    bundled read-only modules can be called from a restricted task; neither
    an MCP annotation nor a familiar server name can confer that permission.
    Custom extensions need a separately reviewed, isolated execution adapter.
    """
    from .execution_policy import is_background, require_foreground
    effect = f"mcp:{server}.{tool}"
    if is_background():
        allowed = tool in _BACKGROUND_BUNDLED_TOOLS.get(server, ())
        if allowed and isinstance(spec, dict):
            command, args = _bundled_spec(server)
            allowed = (spec.get("command") == command and spec.get("args") == args
                       and not spec.get("env") and not spec.get("inherit_env"))
        else:
            allowed = False
        if allowed:
            return None
        blocked = require_foreground(effect)
    else:
        # A copied foreground worker may outlive the host's explicit grant.
        blocked = require_foreground(effect)
    if blocked is None:
        return None
    return MCPReply(blocked.text, "tool", structured_content=blocked.receipt,
                    provenance={"server": server, "tool": tool, "policy": "host_background"})


async def _kill_tree(proc: Any) -> None:
    """Force a server down together with anything it started.

    On Windows the process we hold is often a launcher (uvx.exe, or cmd.exe
    running npx.cmd) and the server proper is its child; terminating only the
    launcher orphans the child. taskkill /T takes the whole tree.
    """
    if os.name == "nt" and getattr(proc, "pid", None):
        import subprocess

        def _taskkill() -> int:
            return subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True, creationflags=0x08000000,   # CREATE_NO_WINDOW
                timeout=5).returncode

        try:
            if await asyncio.to_thread(_taskkill) == 0:
                return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


class MCPServerConn:
    """One MCP server subprocess and the JSON-RPC conversation with it."""

    def __init__(self, name: str, spec: dict[str, Any], bus: Any) -> None:
        self.name = name
        self.command = str(spec.get("command") or "")
        self.args = [str(a) for a in (spec.get("args") or [])]
        self.env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
        self.spec = dict(spec)
        self._credential_values = sorted({
            value for key, value in self.env.items()
            if _SECRET_RE.search(key) and value.strip()
            and value.strip().lower() not in {"0", "1", "true", "false"}
        }, key=len, reverse=True)
        self.description = str(spec.get("description") or "")
        self.bus = bus
        self.proc: asyncio.subprocess.Process | None = None
        self.tools: list[dict[str, Any]] = []
        self._id = 0
        self._lock = asyncio.Lock()          # one in-flight request at a time
        self._active_calls = 0               # includes off-thread validation / queued requests
        self.ready = False
        self.last_error = ""
        self._stderr_tail: list[str] = []
        self._stderr_task: Any = None
        self._receive_task: asyncio.Task[Any] | None = None
        self._receive_proc: Any = None
        self._receive_error: Exception | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._refresh_task: asyncio.Task[Any] | None = None
        self._catalogue_dirty = False
        self._catalogue_refresh_failed = False
        self._on_catalogue_changed: Any = None
        self._stopping = False
        self.capabilities: dict[str, Any] = {}
        self.protocol_version: str | None = None
        self._force_legacy = False
        #: When a tool was last called (monotonic), for idle parking.
        self.last_used = time.monotonic()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        executable = resolve_command(self.command)
        if not executable:
            hint = (" — install Node.js (winget install OpenJS.NodeJS.LTS)"
                    if self.command in {"npx", "node", "npm"} else
                    " — install uv (winget install astral-sh.uv)"
                    if self.command in {"uvx", "uv"} else "")
            self.last_error = f"'{self.command}' is not installed{hint}"
            self._log(f"{self.last_error}; skipped.")
            return False
        # The server's own directory on PATH too, so npx can find node.
        from .mcp_policy import child_environment
        try:
            env = child_environment(self.spec, dict(os.environ))
        except ValueError as exc:
            self.last_error = str(exc)
            return False
        # Redact what this process actually receives, including credentials
        # explicitly inherited from the parent rather than stored in env.
        self._credential_values = sorted({
            value for key, value in env.items()
            if _SECRET_RE.search(key) and value.strip()
            and value.strip().lower() not in {"0", "1", "true", "false"}
        }, key=len, reverse=True)
        extra = os.pathsep.join([os.path.dirname(executable), *_tool_dirs()])
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
        flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
        try:
            self._stopping = False
            self.proc = await asyncio.create_subprocess_exec(
                executable, *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                creationflags=flags,
                # A chatty server's stderr (npx download progress) must never
                # fill the pipe and stall it mid-handshake.
                limit=4 * 1024 * 1024,
            )
        except Exception as exc:
            self.last_error = self._redact(f"failed to launch - {exc}")
            self._log(self.last_error)
            return False
        self._drain_stderr()
        try:
            await asyncio.wait_for(self._handshake(), timeout=_HANDSHAKE_TIMEOUT)
            # The sole reader can resolve the final discovery response and
            # then observe EOF/invalid output before this coroutine resumes.
            # Never overwrite that terminal failure with a successful start.
            if self._receive_error is not None:
                raise self._receive_error
            if self.proc is None or self.proc.returncode is not None:
                raise RuntimeError("server closed the connection")
        except _LegacyProbeClosed:
            # Some handshake-era servers exit on any request before initialize.
            # Retry once with a fresh process rather than declaring them broken.
            await self.stop()
            self._force_legacy = True
            return await self.start()
        except asyncio.CancelledError:
            await self.stop(force=True)
            raise
        except Exception as exc:
            tail = " | ".join(self._stderr_tail[-3:])
            self.last_error = self._redact(f"handshake failed - {exc or type(exc).__name__}" + (
                f" ({tail[:200]})" if tail else "")
            )
            self._log(self.last_error)
            await self.stop()
            return False
        self.ready = True
        self.last_error = ""
        self.last_used = time.monotonic()
        self._log(f"connected — {len(self.tools)} tool(s) available.")
        self._schedule_refresh()
        return True

    def _drain_stderr(self) -> None:
        """Read the server's stderr continuously, keeping the last few lines.

        Nothing read it before, so a server that logged enough (npx prints
        download progress there) filled the OS pipe buffer and blocked — and
        when a handshake failed there was no clue why. The tail is what the
        failure message quotes.
        """
        proc = self.proc
        if proc is None or proc.stderr is None:
            return

        async def _pump() -> None:
            try:
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        return
                    text = self._redact(line.decode("utf-8", "replace").strip())
                    if text:
                        self._stderr_tail.append(text[:240])
                        del self._stderr_tail[:-12]
            except Exception:
                return

        try:
            self._stderr_task = asyncio.get_running_loop().create_task(_pump())
        except RuntimeError:
            pass

    async def _handshake(self) -> None:
        if not self._force_legacy and self.spec.get("protocol_mode") != "legacy":
            try:
                discovered = await asyncio.wait_for(
                    self._request("server/discover", self._modern_meta()),
                    timeout=_DISCOVER_TIMEOUT,
                )
            except _MalformedReply:
                raise
            except MCPProtocolError as exc:
                # -32022 is a modern server explicitly rejecting this version.
                # Unknown-method and other errors can be a legacy server.
                if exc.code == -32022:
                    raise RuntimeError(f"modern protocol version unsupported: {exc}") from exc
            except asyncio.TimeoutError:
                pass
            except RuntimeError as exc:
                if "server closed the connection" in str(exc):
                    raise _LegacyProbeClosed() from exc
                raise
            else:
                versions = discovered.get("supportedVersions") if isinstance(discovered, dict) else None
                if (not isinstance(discovered, dict)
                        or discovered.get("resultType") != "complete"
                        or not isinstance(versions, list)
                        or _MODERN_PROTOCOL_VERSION not in versions
                        or not isinstance(discovered.get("capabilities"), dict)):
                    raise _MalformedReply("invalid or unsupported server/discover result")
                self.protocol_version = _MODERN_PROTOCOL_VERSION
                self.capabilities = discovered["capabilities"]
                await self._list_tools()
                return

        try:
            result = await self._request("initialize", {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "O.R.I.O.N.", "version": "1.0"},
            })
        except MCPProtocolError as exc:
            supported = exc.data.get("supported") if isinstance(exc.data, dict) else None
            if (exc.code != -32022 or self._force_legacy or self.spec.get("protocol_mode") == "legacy"
                    or not isinstance(supported, list) or _MODERN_PROTOCOL_VERSION not in supported):
                raise
            # A slow discover may have selected the modern era at the server
            # after our deadline. Re-probe once on that connection, as the
            # official SDK does, instead of sending an incompatible handshake.
            discovered = await asyncio.wait_for(
                self._request("server/discover", self._modern_meta()), timeout=_DISCOVER_TIMEOUT)
            if (not isinstance(discovered, dict) or discovered.get("resultType") != "complete"
                    or not isinstance(discovered.get("supportedVersions"), list)
                    or _MODERN_PROTOCOL_VERSION not in discovered["supportedVersions"]
                    or not isinstance(discovered.get("capabilities"), dict)):
                raise _MalformedReply("invalid or unsupported server/discover result")
            self.protocol_version = _MODERN_PROTOCOL_VERSION
            self.capabilities = discovered["capabilities"]
            await self._list_tools()
            return
        if not isinstance(result, dict):
            raise _MalformedReply("no initialize result")
        version = result.get("protocolVersion")
        if version not in _SUPPORTED_LEGACY_VERSIONS:
            raise _MalformedReply(f"unsupported initialize protocolVersion: {version!r}")
        if not isinstance(result.get("capabilities"), dict):
            raise _MalformedReply("initialize result has no capabilities object")
        self.protocol_version = version
        self.capabilities = result["capabilities"]
        await self._notify("notifications/initialized")
        await self._list_tools()

    @staticmethod
    def _modern_meta() -> dict[str, Any]:
        return {"_meta": {
            "io.modelcontextprotocol/protocolVersion": _MODERN_PROTOCOL_VERSION,
            "io.modelcontextprotocol/clientInfo": {"name": "O.R.I.O.N.", "version": "1.0"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }}

    async def _list_tools(self, *, request: Any = None) -> None:
        request = request or self._request
        tools: list[dict[str, Any]] = []
        names: set[str] = set()
        cursors: set[str] = set()
        params: dict[str, Any] = {}
        for _page in range(_CATALOGUE_MAX_PAGES):
            listed = await request("tools/list", params)
            page = listed.get("tools") if isinstance(listed, dict) else None
            page = _validate_catalogue(page)
            if (self.protocol_version == _MODERN_PROTOCOL_VERSION
                    and listed.get("resultType") != "complete"):
                raise _MalformedReply("tools/list did not return a complete result")
            for tool in page:
                if tool["name"] in names:
                    raise _MalformedReply("tools/list returned duplicate tool names")
                names.add(tool["name"])
                tools.append(tool)
            if len(tools) > _CATALOGUE_MAX_TOOLS:
                raise _MalformedReply("tools/list exceeded the tool limit")
            cursor = listed.get("nextCursor")
            if cursor is None:
                # Never expose the first page of a failed or incomplete catalogue.
                self.tools = tools
                return
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise _MalformedReply("tools/list returned an invalid or repeated cursor")
            cursors.add(cursor)
            params = {"cursor": cursor}
        raise _MalformedReply("tools/list exceeded the page limit")

    async def stop(self, *, force: bool = False) -> None:
        self._stopping = True
        proc = self.proc
        self.proc = None
        self.ready = False
        self.protocol_version = None
        task = self._stderr_task
        self._stderr_task = None
        owned = [t for t in (self._receive_task, self._refresh_task)
                 if t is not None and t is not asyncio.current_task()]
        self._receive_task = None
        self._refresh_task = None
        self._catalogue_dirty = False
        for child in owned:
            child.cancel()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("server closed the connection"))
        # The MCP stdio shutdown order: close the server's stdin and let it
        # exit, and only then force it. The order matters on Windows: uvx and
        # npx run the real server as a CHILD that shares the pipes, so
        # terminating the launcher left wait() blocked on pipes the child
        # still held — the full timeout, per server, at every shutdown
        # (measured: 12.0 s of a 14.3 s exit for three servers).
        try:
            # A cancelled handshake has no established session to drain.
            # Kill its owned tree before stderr retirement or a graceful wait;
            # otherwise cancellation consumes the entire shutdown grace budget.
            if force and proc is not None:
                await _kill_tree(proc)
            if owned:
                await asyncio.gather(*owned, return_exceptions=True)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if proc is None:
                return
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=_EXIT_GRACE_S)
            except Exception:
                await _kill_tree(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_EXIT_GRACE_S)
                except Exception:
                    pass
        except asyncio.CancelledError:
            # A shutdown budget may cancel a reconnect while it is retiring
            # the old connection. Its process is still ours: force it down
            # and close the pipes before propagating that cancellation.
            if proc is not None:
                await _kill_tree(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_EXIT_GRACE_S)
                except Exception:
                    pass
            raise
        finally:
            for child in owned:
                if not child.done():
                    child.cancel()
            if owned:
                await asyncio.gather(*owned, return_exceptions=True)
            # Close the pipes on every exit, including cancellation during
            # stderr retirement; garbage collection is too late for cleanup.
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

    def is_alive(self) -> bool:
        """Mark XXI, Track E2: True only while the subprocess is genuinely
        still running — a crashed server (non-None returncode) or one that
        was never started is not, regardless of the last-known `ready`
        flag, so a supervisor can tell the difference between "healthy"
        and "handshake succeeded once but has since died"."""
        return self.ready and self.proc is not None and self.proc.returncode is None

    # ── JSON-RPC over stdio (newline-delimited) ────────────────────────────────

    async def _send(self, message: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("server not running")
        self.proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        async with self._lock:
            return await self._request_locked(method, params)

    async def _request_locked(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send one request while the caller owns the connection's wire lock."""
        if self._stopping:
            raise RuntimeError("server closed the connection")
        self._id += 1
        req_id = self._id
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None or self.protocol_version == _MODERN_PROTOCOL_VERSION:
            request_params = dict(params or {})
            if self.protocol_version == _MODERN_PROTOCOL_VERSION:
                request_params.update(self._modern_meta())
            msg["params"] = request_params
        future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            self._ensure_receiver()
            await self._send(msg)
            return await future
        finally:
            self._pending.pop(req_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                # A pipe failure can race a failed stdin write. Retrieve the
                # future's exception even when the write raised first.
                future.exception()

    def _ensure_receiver(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("server not running")
        if self._receive_proc is proc and self._receive_task is not None:
            if self._receive_task.done() and self._receive_error is not None:
                raise self._receive_error
            return
        self._receive_proc = proc
        self._receive_error = None
        self._receive_task = asyncio.create_task(
            self._receive(proc), name=f"orion-mcp-receive-{self.name}")

    async def _receive(self, proc: Any) -> None:
        """The sole stdout reader, including while no request is active."""
        try:
            malformed_lines = 0
            while self.proc is proc and not self._stopping:
                line = await proc.stdout.readline()
                if not line:
                    if malformed_lines:
                        raise _MalformedReply("server sent invalid JSON-RPC output")
                    raise RuntimeError("server closed the connection")
                try:
                    obj = json.loads(line.decode("utf-8").strip())
                except (ValueError, UnicodeError):
                    malformed_lines += 1
                    if malformed_lines >= 8:
                        self.ready = False
                        raise _MalformedReply("server sent repeated invalid JSON-RPC output")
                    continue
                if not isinstance(obj, dict):
                    malformed_lines += 1
                    if malformed_lines >= 8:
                        self.ready = False
                        raise _MalformedReply("server sent repeated invalid JSON-RPC output")
                    continue
                if "method" in obj:
                    if obj.get("jsonrpc") != "2.0":
                        continue
                    if ("id" not in obj
                            and obj["method"] == "notifications/tools/list_changed"
                            and ("params" not in obj or isinstance(obj["params"], dict))):
                        self._catalogue_dirty = True
                        self._schedule_refresh()
                    elif type(obj.get("id")) in (str, int):
                        await self._send({"jsonrpc": "2.0", "id": obj["id"],
                                          "error": {"code": -32601, "message": "Client method unsupported"}})
                    continue
                future = (self._pending.get(obj["id"])
                          if type(obj.get("id")) is int else None)
                if future is not None and not future.done():
                    if (obj.get("jsonrpc") != "2.0"
                            or ("result" in obj) == ("error" in obj)):
                        self.ready = False
                        raise _MalformedReply("server sent an invalid JSON-RPC response")
                    if "error" in obj:
                        if (not isinstance(obj["error"], dict)
                                or type(obj["error"].get("code")) is not int
                                or not isinstance(obj["error"].get("message"), str)):
                            self.ready = False
                            raise _MalformedReply("server sent an invalid JSON-RPC error")
                        future.set_exception(MCPProtocolError(obj["error"]))
                    else:
                        future.set_result(obj["result"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.proc is proc and not self._stopping:
                self.ready = False
                self._receive_error = exc
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(exc)
                if self._on_catalogue_changed is not None:
                    self._on_catalogue_changed(self)

    def _schedule_refresh(self) -> None:
        if (not self._catalogue_dirty or not self.ready or self._stopping
                or self.proc is None
                or (self._refresh_task is not None and not self._refresh_task.done())):
            return
        self._refresh_task = asyncio.create_task(
            self._refresh_catalogue(), name=f"orion-mcp-refresh-{self.name}")

    async def _refresh_catalogue(self) -> None:
        proc = self.proc
        self._active_calls += 1
        try:
            while self._catalogue_dirty and self.ready and self.proc is proc:
                self._catalogue_dirty = False
                try:
                    # An admitted tool can legitimately use its full call
                    # budget. Discovery's shorter budget starts only after
                    # that tool releases the wire, and covers all pages.
                    async with self._lock:
                        if self._stopping or self.proc is not proc or not self.ready:
                            return
                        await asyncio.wait_for(
                            self._list_tools(request=self._request_locked),
                            timeout=_DISCOVER_TIMEOUT)
                    self._catalogue_refresh_failed = False
                except Exception as exc:
                    self.ready = False
                    self._catalogue_refresh_failed = True
                    self.last_error = self._redact(f"tool catalogue refresh failed - {exc}")
                    self._log(self.last_error)
                if self._stopping or self.proc is not proc:
                    return
                if self._on_catalogue_changed is not None:
                    self._on_catalogue_changed(self)
        finally:
            self._active_calls -= 1

    def busy(self) -> bool:
        """A request is in flight, so the process must not be stopped."""
        return (self._active_calls > 0 or self._lock.locked()
                or (self._refresh_task is not None and not self._refresh_task.done()))

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> "MCPReply":
        blocked = _background_mcp_refusal(self.name, tool, self.spec)
        if blocked is not None:
            return blocked
        from .mcp_policy import validate_arguments
        # Argument validation may resolve filesystem paths in a worker thread.
        # Mark the connection in use before yielding, so idle parking cannot
        # stop the server between the host's parked check and this request.
        self._active_calls += 1
        self.last_used = time.monotonic()
        try:
            arguments = await asyncio.to_thread(validate_arguments, self.spec, arguments)
            result = await asyncio.wait_for(
                self._request("tools/call", {"name": tool, "arguments": arguments or {}}),
                timeout=_CALL_TIMEOUT,
            )
            if (self.protocol_version == _MODERN_PROTOCOL_VERSION
                    and (not isinstance(result, dict) or "resultType" not in result)):
                self.ready = False
                return MCPReply("MCP server returned a modern result without resultType.", "protocol")
            reply = self._render(result)
            reply.provenance = {"server": self.name, "tool": tool,
                                "protocol_version": self.protocol_version}
            if not isinstance(result, dict) or not isinstance(result.get("content"), list):
                self.ready = False
            return self._redact(reply)
        finally:
            self._active_calls -= 1
            self.last_used = time.monotonic()

    @staticmethod
    def _render(result: Any) -> "MCPReply":
        """Keep typed evidence and derive only a separate readable summary."""
        if not isinstance(result, dict):
            return MCPReply("MCP server returned an invalid tool result.", "protocol")
        if result.get("resultType", "complete") != "complete":
            return MCPReply("MCP tool requires an unsupported follow-up request.", "protocol")
        if not isinstance(result.get("content"), list):
            return MCPReply("MCP server returned an invalid tool result.", "protocol")
        if ("structuredContent" in result and not isinstance(result["structuredContent"], dict)
                or "isError" in result and type(result["isError"]) is not bool
                or "_meta" in result and not isinstance(result["_meta"], dict)):
            return MCPReply("MCP server returned invalid structured tool content.", "protocol")
        retained = {"content": result["content"],
                    "structured_content": result.get("structuredContent"),
                    "meta": result.get("_meta", {})}
        parts: list[str] = []
        for block in result["content"]:
            if not _valid_content(block):
                return MCPReply("MCP server returned malformed or unsupported content.",
                                "protocol", **retained)
            kind = block["type"]
            if kind == "text":
                parts.append(block["text"])
            elif kind in {"image", "audio"}:
                parts.append(f"[{kind}: {block['mimeType']}]")
            elif kind == "resource_link":
                parts.append(f"[resource link: {block['name']}] {block['uri']} (not fetched)")
            elif kind == "resource":
                resource = block["resource"]
                parts.append(f"[resource: {resource['uri']}] " + resource.get("text", "[binary content]"))
        text = "\n".join(p for p in parts if p).strip()
        if not text and result.get("structuredContent") is not None:
            text = json.dumps(result["structuredContent"], ensure_ascii=False)
        if result.get("isError"):
            # The tool ran and failed: its own message is what lets the model
            # correct the arguments, so it is kept whole.
            return MCPReply(f"(tool error) {text or 'the MCP server reported an error'}", "tool", **retained)
        return MCPReply(text or "(the tool returned no content)", **retained)

    def _log(self, message: str) -> None:
        try:
            self.bus.log.emit(f"MCP[{self.name}]: {self._redact(message)}")
        except Exception:
            pass

    def _redact(self, message: str) -> str:
        """Keep configured credentials out of server errors and log tails."""
        reply = message if isinstance(message, MCPReply) else None
        message = str(message)
        for value in self._credential_values:
            message = message.replace(value, "[redacted]")
        if reply is None:
            return message

        def redact_data(data: Any) -> Any:
            if isinstance(data, str):
                return self._redact(data)
            if isinstance(data, list):
                return [redact_data(item) for item in data]
            if isinstance(data, dict):
                return {self._redact(key): redact_data(value) for key, value in data.items()}
            return data

        return MCPReply(message, reply.kind, content=redact_data(reply.content),
                        structured_content=redact_data(reply.structured_content),
                        meta=redact_data(reply.meta), provenance=redact_data(reply.provenance))


def _spec_fingerprint(spec: dict[str, Any]) -> str:
    """A digest of what launches a server; a changed command, argument or
    credential means its cached tool list can no longer be trusted. Only the
    digest is stored, never the credential itself."""
    import hashlib
    material = json.dumps({"command": spec.get("command"), "args": spec.get("args"),
                           "env": spec.get("env"), "inherit_env": spec.get("inherit_env"),
                           "path_policy": spec.get("path_policy"), "allowed_roots": spec.get("allowed_roots")},
                          sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _valid_schema(schema: Any, *, depth: int = 0, budget: list[int] | None = None) -> bool:
    """Bounded structural checks without downloading/resolving any schema URI.

    Unknown annotation/extension keywords remain available to the downstream
    schema translator. This checks the containers it consumes, rather than
    pretending to validate arguments or resolve every JSON Schema dialect.
    """
    if budget is None:
        budget = [10_000]
    budget[0] -= 1
    if depth > 32 or budget[0] < 0:
        return False
    if type(schema) is bool:
        return True
    if not isinstance(schema, dict):
        return False
    allowed_types = {"null", "boolean", "object", "array", "number", "string", "integer"}
    raw_type = schema.get("type")
    if raw_type is not None:
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        if (not types or any(not isinstance(t, str) or t not in allowed_types for t in types)
                or len(types) != len(set(types))):
            return False
    if "required" in schema:
        required = schema["required"]
        if (not isinstance(required, list) or any(not isinstance(k, str) for k in required)
                or len(required) != len(set(required))):
            return False
    for keyword in ("$ref", "$schema", "$id", "description", "title"):
        if keyword in schema and not isinstance(schema[keyword], str):
            return False
    for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        if keyword in schema:
            mapping = schema[keyword]
            if (not isinstance(mapping, dict)
                    or any(not isinstance(key, str)
                           or not _valid_schema(value, depth=depth + 1, budget=budget)
                           for key, value in mapping.items())):
                return False
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if keyword in schema:
            children = schema[keyword]
            if (not isinstance(children, list)
                    or (not children and keyword != "prefixItems")
                    or any(not _valid_schema(child, depth=depth + 1, budget=budget) for child in children)):
                return False
    for keyword in ("items", "additionalItems", "additionalProperties", "unevaluatedProperties",
                    "unevaluatedItems", "contains", "not", "if", "then", "else", "propertyNames"):
        if keyword in schema:
            children = schema[keyword]
            children = children if keyword == "items" and isinstance(children, list) else [children]
            if any(not _valid_schema(child, depth=depth + 1, budget=budget) for child in children):
                return False
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        return False
    return True


def _validate_catalogue(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list) or len(tools) > _CATALOGUE_MAX_TOOLS:
        raise _MalformedReply("tools/list returned an invalid tool catalogue")
    names: set[str] = set()
    for tool in tools:
        if (not isinstance(tool, dict) or not isinstance(tool.get("name"), str)
                or not tool["name"].strip() or tool["name"] != tool["name"].strip()
                or tool["name"] in names
                or ("description" in tool and not isinstance(tool["description"], str))
                or any(key in tool and (not isinstance(tool[key], dict) or not _valid_schema(tool[key]))
                       for key in ("inputSchema", "outputSchema"))):
            raise _MalformedReply("tools/list returned an invalid tool catalogue")
        names.add(tool["name"])
    return copy.deepcopy(tools)


def _valid_content(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if any(key in block and not isinstance(block[key], dict) for key in ("annotations", "_meta")):
        return False
    kind = block.get("type")
    if not isinstance(kind, str):
        return False
    if kind == "text":
        return isinstance(block.get("text"), str)
    if kind in {"image", "audio"}:
        return (isinstance(block.get("data"), str)
                and isinstance(block.get("mimeType"), str) and bool(block["mimeType"]))
    if kind == "resource_link":
        return (isinstance(block.get("uri"), str) and bool(block["uri"])
                and isinstance(block.get("name"), str))
    if kind == "resource":
        resource = block.get("resource")
        return (isinstance(resource, dict) and isinstance(resource.get("uri"), str)
                and bool(resource["uri"])
                and ("text" in resource) != ("blob" in resource)
                and isinstance(resource.get("text", resource.get("blob")), str))
    return False


def _load_catalogue() -> dict[str, Any]:
    try:
        data = json.loads(MCP_CATALOGUE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _lazy_start_enabled(spec: dict[str, Any]) -> bool:
    """Servers start on first use unless ORION_MCP_LAZY=0 or the server's
    own config says "eager": true."""
    if os.getenv("ORION_MCP_LAZY", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return not spec.get("eager")


class MCPHost:
    """Connects to the configured MCP servers and routes tool calls to them."""

    # Mark XXI, Track E2: bounded backoff for reconnect attempts — a server
    # that's genuinely down (bad credentials, package removed) must not be
    # hammered every supervision tick forever.
    _RECONNECT_BACKOFF_S = (5.0, 15.0, 60.0, 300.0)

    def __init__(self, bus: Any, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.servers: dict[str, MCPServerConn] = {}
        self._specs: dict[str, dict[str, Any]] = {}    # last-known enabled specs, for reconnect
        self._reconnect_attempts: dict[str, int] = {}
        self._next_reconnect_at: dict[str, float] = {}
        self._supervising = False
        #: Why each enabled server is not live, in words ("waiting for
        #: TWILIO_AUTH_TOKEN", "'npx' is not installed", a handshake error).
        self.status: dict[str, str] = {}
        #: Servers stopped while idle. They stay in ``servers`` so their tools
        #: are still listed, and ``call`` restarts one before using it.
        self.parked: set[str] = set()
        self._wake_locks: dict[str, asyncio.Lock] = {}
        self._connection_locks: dict[str, asyncio.Lock] = {}
        self._lifecycle_tasks: set[asyncio.Task[Any]] = set()
        self._closing = False
        self._close_lock = asyncio.Lock()
        self._on_reconnect: Any = None
        self._pressure_park_s: float | None = None
        # App startup connects optional servers in the background so the GUI
        # paints immediately.  A first user request can therefore arrive in
        # the small gap before ``connect_all`` publishes a live connection.
        # The call path waits briefly for that owned task before reporting a
        # missing server; disabled or already-finished hosts remain instant.
        self._startup_task: asyncio.Task | None = None
        try:
            self.idle_park_s = float(os.getenv("ORION_MCP_IDLE_PARK_S", _IDLE_PARK_S))
        except ValueError:
            self.idle_park_s = _IDLE_PARK_S

    def _prepare(self, name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
        """The spec to launch, or None when it cannot work yet (and why)."""
        if spec.get("transport", "stdio") != "stdio":
            self.status[name] = "unsupported transport; this host supports stdio tools only"
            self._beat(name, "DOWN", self.status[name])
            return None
        spec = autofill_credentials(name, spec)
        missing = missing_credentials(spec)
        if missing:
            self.status[name] = "waiting for " + ", ".join(missing)
            self._beat(name, "NEEDS_KEY", self.status[name])
            return None
        return spec

    async def connect_all(self) -> None:
        if self._closing:
            return
        config = load_config()
        if isinstance(config, _UnavailableConfig):
            self._log_host(config.error)
            return
        specs = config.get("servers") or {}
        enabled = {n: s for n, s in specs.items() if isinstance(s, dict) and s.get("enabled")}
        self._specs = enabled
        if not enabled:
            try:
                self.bus.log.emit(
                    "MCP: host ready — no servers enabled yet "
                    "(edit config/mcp_servers.json; see docs/MCP_SETUP.md)."
                )
            except Exception:
                pass
            return

        catalogue = _load_catalogue()

        async def _start_one(name: str, spec: dict[str, Any]) -> None:
            old = self.servers.get(name)
            if old is not None and (old.is_alive() or name in self.parked):
                return
            if old is not None:
                # A protocol failure can clear readiness while the process
                # keeps running. Reuse retirement before replacing it, while
                # _one owns the same per-server lock as reconnect().
                await self._reconnect(name)
                return
            ready = self._prepare(name, spec)
            if ready is None:
                self._log_host(f"'{name}' is enabled but {self.status[name]} — "
                               f"not started (add it in config/mcp_servers.json).")
                return
            conn = MCPServerConn(name, ready, self.bus)
            self._bind_connection(name, conn)
            cached = catalogue.get(name)
            cached = cached if isinstance(cached, dict) else {}
            try:
                cached_tools = _validate_catalogue(cached.get("tools", []))
            except MCPProtocolError:
                cached_tools = []
            if (_lazy_start_enabled(spec) and cached.get("tools")
                    and cached_tools and cached.get("fingerprint") == _spec_fingerprint(ready)):
                # Ten servers held about a gigabyte from the moment ORION
                # started, on a machine already near its limit — which is when
                # the face flashed black. Their tool lists are known from the
                # last run, so offer those now and start each process only
                # when one of its tools is actually called (call -> _wake).
                conn.tools = cached_tools
                self.servers[name] = conn
                self.parked.add(name)
                self.status[name] = "idle - starts when first used"
                self._beat(name, "DEGRADED", f"{len(conn.tools)} tool(s), starts on first use")
                return
            try:
                ok = await conn.start()
            except asyncio.CancelledError:
                await conn.stop(force=True)
                raise
            except BaseException:
                await conn.stop()
                raise
            if self._closing:
                await conn.stop()
                return
            if ok:
                self.servers[name] = conn
                self._reconnect_attempts[name] = 0
                self.status[name] = "live"
                self._remember_tools(name, ready, conn)
                self._beat(name, "OK", f"{len(conn.tools)} tool(s)")
            else:
                self.status[name] = conn.last_error or "did not start"
                self._beat(name, "DOWN", self.status[name])

        async def _one(name: str, spec: dict[str, Any]) -> None:
            task = asyncio.current_task()
            if task is not None:
                self._lifecycle_tasks.add(task)
            try:
                lock = self._connection_locks.setdefault(name, asyncio.Lock())
                async with lock:
                    if not self._closing:
                        await _start_one(name, spec)
            finally:
                if task is not None:
                    self._lifecycle_tasks.discard(task)

        # Concurrently: a first launch downloads each server's package, and
        # doing that one after another made the last server wait minutes.
        await asyncio.gather(*(_one(n, s) for n, s in enabled.items()),
                             return_exceptions=True)
        live = [n for n in enabled if self.status.get(n) == "live"]
        waiting = [n for n in enabled if n in self.parked]
        self._log_host(f"{len(live)} of {len(enabled)} enabled server(s) live"
                       + (f": {', '.join(live)}" if live else "")
                       + (f"; {len(waiting)} start on first use: {', '.join(waiting)}"
                          if waiting else "") + ".")

    def _remember_tools(self, name: str, spec: dict[str, Any], conn: "MCPServerConn") -> None:
        """Save a server's tool list for the next boot's lazy start."""
        try:
            catalogue = _load_catalogue()
            catalogue[name] = {"fingerprint": _spec_fingerprint(spec),
                               "description": conn.description,
                               "tools": conn.tools}
            atomic_write_text(MCP_CATALOGUE_PATH, json.dumps(catalogue, indent=1),
                              encoding="utf-8")
        except Exception:
            pass

    def _forget_tools(self, name: str) -> None:
        """Retire a known-invalid catalogue without disturbing other servers."""
        try:
            catalogue = _load_catalogue()
            if name not in catalogue:
                return
            del catalogue[name]
            atomic_write_text(MCP_CATALOGUE_PATH, json.dumps(catalogue, indent=1),
                              encoding="utf-8")
        except Exception as exc:
            self._log_host(f"could not retire cached tools for '{name}' - {exc}")

    # ── health (Track E2) ────────────────────────────────────────────────────

    def _beat(self, name: str, status: str, detail: str = "") -> None:
        if self.telemetry is None:
            return
        component = f"mcp.{name}"
        try:
            self.telemetry.health.register(component)
            self.telemetry.health.beat(component, status, detail)
        except Exception:
            pass

    def health_snapshot(self) -> dict[str, str]:
        """Current OK, IDLE, NEEDS_KEY or DOWN state for enabled servers.

        A disconnected server remains visible while awaiting reconnect.
        """
        out: dict[str, str] = {}
        for name in self._specs:
            conn = self.servers.get(name)
            if name in self.parked:
                out[name] = "IDLE"        # paused to save memory, not broken
            elif conn is not None and conn.is_alive():
                out[name] = "OK"
            elif str(self.status.get(name, "")).startswith("waiting for"):
                out[name] = "NEEDS_KEY"   # not broken — unconfigured
            else:
                out[name] = "DOWN"
        return out

    # ── reconnect + supervision (Track E2) ──────────────────────────────────

    def _publish_tools(self, name: str, tools: Any) -> None:
        """Refresh dispatcher routes when a live catalogue changes or vanishes."""
        if self._on_reconnect is not None:
            try:
                self._on_reconnect(name, tools)
            except Exception as exc:
                self._log_host(f"tool registration for '{name}' failed - {exc}")

    def _bind_connection(self, name: str, conn: MCPServerConn) -> None:
        conn._on_catalogue_changed = lambda changed: self._catalogue_changed(name, changed)

    def _catalogue_changed(self, name: str, conn: MCPServerConn) -> None:
        # A late callback from an old receiver cannot revive a disabled server
        # or overwrite its replacement's catalogue during reconnect.
        if (self._closing or self.servers.get(name) is not conn
                or name not in self._specs or name in self.parked):
            return
        if conn.ready:
            self._remember_tools(name, conn.spec, conn)
            self._beat(name, "OK", f"{len(conn.tools)} tool(s), catalogue refreshed")
            self._publish_tools(name, conn)
        else:
            if conn._catalogue_refresh_failed:
                self._forget_tools(name)
            self.status[name] = conn.last_error or "connection dropped - server closed the connection"
            self._beat(name, "DOWN", self.status[name])
            self._publish_tools(name, [])

    async def reconnect(self, name: str) -> bool:
        """Tear down and re-establish one server's connection from its
        last-known enabled spec — re-reading config/mcp_servers.json first,
        so a credential/command fix the user just made takes effect without
        a full app restart. Returns whether it's alive afterwards."""
        if self._closing:
            return False
        task = asyncio.current_task()
        if task is not None:
            self._lifecycle_tasks.add(task)
        try:
            lock = self._connection_locks.setdefault(name, asyncio.Lock())
            async with lock:
                if self._closing:
                    return False
                return await self._reconnect(name)
        finally:
            if task is not None:
                self._lifecycle_tasks.discard(task)

    async def _reconnect(self, name: str) -> bool:
        # A manual reconnect also wakes a parked server. Clear its parked
        # state before either the live or failed outcome is reported; otherwise
        # health_snapshot says IDLE and the next call starts it a second time.
        self.parked.discard(name)
        config = load_config()
        if isinstance(config, _UnavailableConfig):
            self.status[name] = config.error
            self._beat(name, "DOWN", config.error)
            self._log_host(config.error)
        spec = (config.get("servers") or {}).get(name)
        if not isinstance(spec, dict) or not spec.get("enabled"):
            # The user disabled or removed it — stop tracking it as a live
            # target rather than retrying a server that's meant to be off.
            self._specs.pop(name, None)
            old = self.servers.pop(name, None)
            if old is not None:
                await old.stop()
            self._publish_tools(name, [])
            return False
        self._specs[name] = spec
        old = self.servers.pop(name, None)
        if old is not None:
            await old.stop()
        ready = self._prepare(name, spec)
        if ready is None:
            self._publish_tools(name, [])
            return False
        conn = MCPServerConn(name, ready, self.bus)
        self._bind_connection(name, conn)
        try:
            ok = await conn.start()
        except asyncio.CancelledError:
            await conn.stop(force=True)
            raise
        except BaseException:
            await conn.stop()
            raise
        if self._closing:
            await conn.stop()
            return False
        if ok:
            self.servers[name] = conn
            self._reconnect_attempts[name] = 0
            self.status[name] = "live"
            self._remember_tools(name, ready, conn)
            self._beat(name, "OK", f"reconnected — {len(conn.tools)} tool(s)")
            self._publish_tools(name, conn)
        else:
            self.status[name] = conn.last_error or "reconnect attempt failed"
            self._beat(name, "DOWN", "reconnect attempt failed")
            self._publish_tools(name, [])
        return ok

    def _due_for_reconnect(self, name: str, now: float) -> bool:
        return now >= self._next_reconnect_at.get(name, 0.0)

    def _schedule_next_reconnect(self, name: str, now: float) -> None:
        attempt = self._reconnect_attempts.get(name, 0)
        delay = self._RECONNECT_BACKOFF_S[min(attempt, len(self._RECONNECT_BACKOFF_S) - 1)]
        self._reconnect_attempts[name] = attempt + 1
        self._next_reconnect_at[name] = now + delay

    async def supervise(self, interval_s: float = 30.0,
                        on_reconnect: Any | None = None) -> None:
        """Run forever (until cancelled): every *interval_s*, check each
        enabled server that isn't currently alive and reconnect it with
        bounded backoff. *on_reconnect(name, conn)* — a plain sync
        callable, e.g. re-registering the dispatcher's first-class MCP
        tools (Track E1) — fires after each successful reconnect. Never
        raises out of the loop; one server's trouble never stops the
        supervisor watching the others."""
        if self._closing:
            return
        self._supervising = True
        self._on_reconnect = on_reconnect
        try:
            while self._supervising:
                await asyncio.sleep(interval_s)
                if not self._supervising or self._closing:
                    break
                try:
                    await self.park_idle()
                except Exception as exc:
                    self._log_host(f"idle parking skipped - {exc}")
                now = asyncio.get_running_loop().time()
                for name in list(self._specs):
                    if name in self.parked:
                        continue              # asleep on purpose; woken by a call
                    conn = self.servers.get(name)
                    if conn is not None and conn.is_alive():
                        continue
                    if not self._due_for_reconnect(name, now):
                        continue
                    if str(self.status.get(name, "")).startswith("waiting for") and \
                            missing_credentials(autofill_credentials(name, self._specs[name])):
                        # Still no key: retrying cannot help, and every retry
                        # used to be logged as a fresh failure.
                        self._schedule_next_reconnect(name, now)
                        continue
                    self._schedule_next_reconnect(name, now)
                    try:
                        await self.reconnect(name)
                    except Exception as exc:
                        self._log_host(f"supervised reconnect of '{name}' raised - {exc}")
                        continue
        except asyncio.CancelledError:
            pass

    def stop_supervising(self) -> None:
        self._supervising = False

    # ── idle parking ────────────────────────────────────────────────────────

    def set_pressure_parking(self, seconds: float | None) -> None:
        """Park sooner while memory is short (the resource governor's lever);
        None goes back to the everyday limit."""
        self._pressure_park_s = seconds

    def _park_limit(self) -> float | None:
        limits = [v for v in (self.idle_park_s, self._pressure_park_s)
                  if v is not None and v > 0]
        return min(limits) if limits else None

    async def park_idle(self, now: float | None = None) -> list[str]:
        """Stop every live server unused for longer than the limit.

        Its tool list is kept, so the model and the dispatcher still see the
        tools; the process (and its memory) is what goes.
        """
        limit = self._park_limit()
        if limit is None:
            return []
        now = time.monotonic() if now is None else now
        parked: list[str] = []
        for name, conn in list(self.servers.items()):
            if name in self.parked or not conn.is_alive() or conn.busy():
                continue
            if now - conn.last_used < limit:
                continue
            self.parked.add(name)
            self.status[name] = "paused while idle (restarts on its next use)"
            await conn.stop()
            self._beat(name, "DEGRADED", self.status[name])
            parked.append(name)
        if parked:
            self._log_host(f"paused idle server(s) to free memory: "
                           f"{', '.join(parked)} - each restarts when next used.")
        return parked

    async def _wake(self, name: str) -> bool:
        lock = self._wake_locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name not in self.parked:
                conn = self.servers.get(name)
                return conn is not None and conn.is_alive()
            ok = await self.reconnect(name)
            self.parked.discard(name)       # failed: the supervisor takes over
            return ok

    def _log_host(self, message: str) -> None:
        try:
            self.bus.log.emit(f"MCP: {message}")
        except Exception:
            pass

    def catalogue(self) -> dict[str, Any]:
        """Every connected server and its tools — for the model to browse."""
        out: dict[str, Any] = {}
        for name, conn in self.servers.items():
            out[name] = {
                "description": conn.description,
                "tools": [
                    {"name": t.get("name"), "description": t.get("description", "")}
                    for t in conn.tools
                ],
            }
        return out

    def describe(self) -> str:
        cat = self.catalogue()
        if not cat:
            return ("No MCP servers are connected. Enable one in "
                    "config/mcp_servers.json (see docs/MCP_SETUP.md).")
        lines: list[str] = []
        for server, info in cat.items():
            lines.append(f"• {server} — {info['description']}")
            for tool in info["tools"]:
                lines.append(f"    - {tool['name']}: {tool['description'][:100]}")
        lines.append("MCP support: stdio tool calls; resource links are retained without fetching. "
                     "Resource requests, prompts, and other transports are unsupported.")
        return "\n".join(lines)

    #: Servers whose tools act in the world at the user's expense, even if
    #: their spec predates the "confirm" flag. Twilio dials immediately and
    #: bills immediately; there is no draft state to inspect afterwards.
    CONFIRM_SERVERS = {"twilio"}

    #: Tool names that spend money or leave the machine, on any server. Matched
    #: as substrings, because servers name these differently — create_call,
    #: send_sms, make_outbound_call — and the cost is the same whichever.
    CONFIRM_WORDS = ("call", "sms", "dial", "fax", "purchase", "buy",
                     "payment", "charge", "transfer")

    def requires_confirmation(self, server: str, tool: str) -> str:
        """Why this call should be confirmed first, or "" if it need not be.

        A reason rather than a boolean: the person is being asked to approve
        something, and "this needs confirmation" does not tell them what they
        are approving.
        """
        server = str(server or "").strip().lower()
        shown = str(tool or "").strip()
        tool = shown.lower()
        if (server in _ACCOUNT_READ_TOOLS
                and tool.replace("_", "-") not in _ACCOUNT_READ_TOOLS[server]):
            action = ("send email or change the mailbox" if server == "gmail"
                      else "change the calendar or its connected accounts")
            return f"{server}.{shown} may {action}"
        config = load_config()
        if isinstance(config, _UnavailableConfig):
            return config.error
        spec = (config.get("servers") or {}).get(server)
        reason = configured_confirmation(server, shown, spec)
        if reason:
            return reason
        if server in self.CONFIRM_SERVERS:
            hit = next((w for w in self.CONFIRM_WORDS if w in tool), "")
            if hit in ("call", "dial"):
                return (f"{server}.{tool} places a real phone call that "
                        f"connects immediately and costs money")
            if hit in ("sms", "fax"):
                return (f"{server}.{tool} sends a real {hit.upper()} and "
                        f"costs money")
            if hit:
                return f"{server}.{tool} spends real money"
            return f"'{server}' acts at your expense"
        return ""

    async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> MCPReply:
        if self._closing:
            return MCPReply("MCP host is closed; no new tool calls are accepted.", "unavailable")
        config = load_config()
        if isinstance(config, _UnavailableConfig):
            # A previously connected custom server may have depended on a
            # confirmation policy in that file. Never execute under guesses.
            return MCPReply(config.error, "unavailable")
        blocked = _background_mcp_refusal(server, tool, (config.get("servers") or {}).get(server))
        if blocked is not None:
            return blocked
        startup = self._startup_task
        if (server not in self.servers and server not in self.parked
                and startup is not None and not startup.done()):
            try:
                # Two seconds covers the normal local handshake without
                # turning a slow first package download into a voice stall.
                await asyncio.wait_for(asyncio.shield(startup), timeout=2.0)
            except asyncio.CancelledError:
                # The startup task being cancelled is fine; THIS call being
                # cancelled (a cancelled turn, shutdown) must not be swallowed.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            except Exception:
                pass
        if self._closing:
            return MCPReply("MCP host is closed; no new tool calls are accepted.", "unavailable")
        if server in self.parked and not await self._wake(server):
            return MCPReply(
                f"MCP server '{server}' was paused to save memory and could "
                f"not restart: {self.status.get(server) or 'unknown error'}.", "unavailable")
        conn = self.servers.get(server)
        if conn is None:
            avail = ", ".join(self.servers) or "none"
            return MCPReply(
                f"No connected MCP server named '{server}'. Available: {avail}.", "unavailable")
        blocked = _background_mcp_refusal(server, tool, getattr(conn, "spec", None))
        if blocked is not None:
            return blocked
        if conn.proc is not None and not conn.is_alive():
            return MCPReply(f"MCP server '{server}' is disconnected; awaiting reconnect.",
                            "unavailable")
        if not any(t.get("name") == tool for t in conn.tools):
            names = ", ".join(t.get("name", "?") for t in conn.tools) or "none"
            return MCPReply(
                f"Server '{server}' has no tool '{tool}'. Its tools: {names}.", "protocol")
        try:
            return await conn.call_tool(tool, arguments)
        except asyncio.TimeoutError:
            return MCPReply(f"MCP tool '{server}.{tool}' timed out.", "timeout")
        except _MalformedReply as exc:
            return MCPReply(f"MCP tool '{server}.{tool}' received an invalid protocol reply: "
                            f"{conn._redact(str(exc))}.", "protocol")
        except MCPProtocolError as exc:
            # The server refused the call itself — almost always arguments
            # that do not match the tool's input schema. Say so, so the model
            # fixes the arguments rather than retrying the same call.
            return MCPReply(
                f"MCP tool '{server}.{tool}' failed: the server rejected the request "
                f"- {conn._redact(str(exc))}. Check the arguments against the tool's input schema.", "protocol")
        except ValueError as exc:
            return MCPReply(f"MCP tool '{server}.{tool}' rejected by local input policy: {exc}", "protocol")
        except Exception as exc:
            return MCPReply(f"MCP tool '{server}.{tool}' failed: {conn._redact(str(exc))}",
                            "unavailable")

    # ── runtime server management ───────────────────────────────────────────
    # Enabling a server used to mean editing config/mcp_servers.json by hand
    # and restarting ORION.  These make the whole MCP surface manageable while
    # he is running — the config file stays the single source of truth, it is
    # just written for the user and applied live.

    def list_servers(self) -> str:
        """Every DECLARED server (not merely the connected ones) with its
        enabled and live state, so a server that is off — or enabled but
        failing to start — is visible instead of simply absent."""
        config = load_config()
        if isinstance(config, _UnavailableConfig):
            return config.error
        specs = config.get("servers") or {}
        if not specs:
            return "No MCP servers are declared in config/mcp_servers.json."
        lines: list[str] = []
        for name in sorted(specs):
            spec = specs[name] if isinstance(specs[name], dict) else {}
            enabled = bool(spec.get("enabled"))
            conn = self.servers.get(name)
            live = conn is not None and conn.is_alive()
            reason = self.status.get(name, "")
            if reason == "live" and not live:
                reason = "connection dropped; reconnecting"
            if name in self.parked:
                reason = (f"idle, paused to save memory ({len(conn.tools) if conn else 0}"
                          f" tool(s); restarts on its next use)")
            if not reason and enabled and not live:
                missing = missing_credentials(autofill_credentials(name, spec))
                if missing:
                    reason = "waiting for " + ", ".join(missing)
                elif not resolve_command(str(spec.get("command") or "")):
                    reason = f"'{spec.get('command')}' is not installed"
            state = "live" if live else (
                f"enabled — {reason}" if enabled and reason else
                "enabled (not connected)" if enabled else "disabled")
            tools = f", {len(conn.tools)} tool(s)" if live and conn else ""
            lines.append(f"• {name} — {state}{tools}")
            description = str(spec.get("description") or "").strip()
            if description:
                lines.append(f"    {description[:120]}")
        lines.append("")
        lines.append("Use mcp enable/disable <server> to change this, then "
                     "mcp reconnect <server> if needed.")
        return "\n".join(lines)

    def _write_config(self, config: dict[str, Any]) -> bool:
        if isinstance(config, _UnavailableConfig):
            return False
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(MCP_CONFIG_PATH, json.dumps(config, indent=2), encoding="utf-8")
            return True
        except OSError:
            return False

    async def set_enabled(self, name: str, enabled: bool) -> str:
        """Enable or disable one declared server and apply it immediately."""
        if self._closing:
            return "MCP host is closed."
        server = str(name or "").strip()
        if not server:
            return "Which MCP server, sir?"
        config = load_config()
        if isinstance(config, _UnavailableConfig):
            return config.error
        specs = config.get("servers") or {}
        if server not in specs or not isinstance(specs[server], dict):
            known = ", ".join(sorted(specs)) or "none"
            return f"No MCP server named '{server}'. Declared: {known}."
        specs[server]["enabled"] = bool(enabled)
        config["servers"] = specs
        if not self._write_config(config):
            return f"Could not write {MCP_CONFIG_PATH.name}."
        if enabled:
            ok = await self.reconnect(server)
            return (f"MCP server '{server}' enabled and connected."
                    if ok else
                    f"MCP server '{server}' enabled, but it did not start — "
                    "check its command and credentials in "
                    f"{MCP_CONFIG_PATH.name}.")
        lock = self._connection_locks.setdefault(server, asyncio.Lock())
        async with lock:
            conn = self.servers.pop(server, None)
            if conn is not None:
                try:
                    await conn.stop()
                except Exception:
                    pass
            self._specs.pop(server, None)
            self.parked.discard(server)
            self._beat(server, "DOWN", "disabled by the user")
            self._publish_tools(server, [])
        return f"MCP server '{server}' disabled and disconnected."

    def add_server(self, name: str, command: str, args: list[str] | None = None,
                   env: dict[str, str] | None = None,
                   description: str = "") -> str:
        """Declare a NEW MCP server. It is written disabled, so nothing is
        launched until the user has filled in credentials and enabled it."""
        if self._closing:
            return "MCP host is closed."
        server = str(name or "").strip()
        if not server:
            return "Give the server a name, sir."
        if not str(command or "").strip():
            return "Give the server a command to run (e.g. npx)."
        config = load_config()
        if isinstance(config, _UnavailableConfig):
            return config.error
        specs = config.get("servers") or {}
        if server in specs:
            return (f"An MCP server called '{server}' is already declared. "
                    "Edit it in config/mcp_servers.json, or pick another name.")
        specs[server] = {
            "enabled": False,
            "command": str(command).strip(),
            "args": [str(a) for a in (args or [])],
            "env": {str(k): str(v) for k, v in (env or {}).items()},
            "description": str(description or "").strip(),
        }
        config["servers"] = specs
        if not self._write_config(config):
            return f"Could not write {MCP_CONFIG_PATH.name}."
        return (f"Declared MCP server '{server}' (disabled). Fill in any "
                f"credentials in {MCP_CONFIG_PATH.name}, then: mcp enable {server}.")

    async def close(self) -> None:
        self._closing = True
        self.stop_supervising()
        async with self._close_lock:
            current = asyncio.current_task()
            pending = {task for task in self._lifecycle_tasks
                       if task is not current and not task.done()}
            if (self._startup_task is not None and self._startup_task is not current
                    and not self._startup_task.done()):
                # Cancelling both gather's parent and its children can inject
                # a second cancellation while a child is reaping its process.
                # Cancel owned children first, then await startup as well.
                if not pending:
                    self._startup_task.cancel()
            for task in pending:
                task.cancel()
            if (self._startup_task is not None and self._startup_task is not current
                    and not self._startup_task.done()):
                pending.add(self._startup_task)
            if pending:
                # start() reaps its process on cancellation; the host also
                # cleans up injectable transports before those tasks retire.
                await asyncio.gather(*pending, return_exceptions=True)
            names = list(self.servers)
            # Together, not in turn: each stop can wait out its grace period.
            await asyncio.gather(*(conn.stop() for conn in list(self.servers.values())),
                                 return_exceptions=True)
            self.servers.clear()
            self.parked.clear()
            self._specs.clear()
            for name in names:
                self._publish_tools(name, [])
