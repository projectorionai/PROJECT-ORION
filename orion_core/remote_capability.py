"""
Remote capability policy + confirmation gate (#12 — full-parity remote).

When a request originates from a *paired* remote device (the phone), ORION
should be able to do on the move almost everything he does at the desk — but
WITHOUT turning the uplink into a remote-code-execution surface, and without
letting an irreversible or outward-facing action fire while the user is away
unless they explicitly tap to approve it.

Every dispatcher tool falls into one of three tiers:

  ALLOW    read/benign — runs immediately.
  CONFIRM  irreversible or outward-facing — queued as a pending confirmation
           pushed to the phone; runs only after the user approves ON the phone.
           This is the desktop's on-screen approval gate, relocated to where
           the user actually is.
  FORBID   code execution / self-modification / security toggles — refused
           outright, regardless of anything, because exposing them to a
           network origin is a remote-code-execution surface. Desktop-only.

Design notes:
  • Classification is action-aware: a CONFIRM tool called with a *read* action
    (read the inbox, list tasks, read a page) downgrades to ALLOW, so parity is
    real — you read your mail instantly, but *sending* prompts a tap.
  • The default for an unknown/unclassified tool is CONFIRM — fail safe: nothing
    new ever runs remotely without the user seeing it first.
  • This module has ZERO heavy dependencies (stdlib only) so it unit-tests fast
    and can be reasoned about in isolation from the aiohttp gateway.
"""

from __future__ import annotations

import json
import secrets
import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from . import phone_control
from .data import ToolResult
from .dispatch_schema import PHONE_DEVICE_DECLARATION
from .remote_mobile_ws import _valid_command_params
from .phone_contracts import translate as native_phone_command


_PHONE_COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "capture_dom": ("viewport.capture_dom", ()),
    "click": ("viewport.click", ("snapshotId", "nodeId")),
    "scroll": ("viewport.scroll", ("snapshotId", "nodeId", "direction")),
    "set_text": ("viewport.set_text", ("snapshotId", "nodeId", "text")),
    "open_app": ("device.open_app", ("package",)),
    "make_call": ("device.make_call", ("number",)),
    "send_sms": ("device.send_sms", ("number", "body")),
}


def _phone_command(args: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Translate exactly one phone tool action to the bridge's fixed RPC schema."""
    action = args.get("action")
    if action == "latest_frame":
        return ("latest_frame", {}) if set(args) == {"action"} else None
    native = native_phone_command(action, args)
    if native is not None:
        return native
    if action == "open_app" and set(args) == {"action", "name"}:
        params = {"name": args["name"]}
        return ("device.open_app", params) if _valid_command_params("device.open_app", params) else None
    spec = _PHONE_COMMANDS.get(action) if isinstance(action, str) else None
    if spec is None:
        return None
    method, keys = spec
    if set(args) != {"action", *keys}:
        return None
    params = {key: args[key] for key in keys}
    if not _valid_command_params(method, params):
        return None
    return method, params


class Tier(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    FORBID = "forbid"


# Runtime overrides for dynamically-loaded plugins (plugin_manifest.py): each
# <name>.plugin.json declares its own tier, registered here the moment the
# plugin loads successfully — checked before the read-action downgrade table
# so a loaded plugin's declared tier is authoritative. The manifest author is
# the same user who already has full filesystem access to this machine (they
# dropped the file into config/custom_tools/ themselves), so trusting the
# declared tier grants nothing they couldn't already grant by editing
# ALLOW_TOOLS directly — the value here is a safe, auditable, per-file
# default (CONFIRM) rather than requiring a code change for every plugin.
_PLUGIN_TIERS: dict[str, "Tier"] = {}


def register_plugin_tier(tool_name: str, tier: str) -> None:
    """Feed a loaded plugin's manifest-declared tier into the remote gate.
    An unrecognised tier string is silently ignored — the tool then falls
    back to the normal fail-safe (CONFIRM for anything unclassified)."""
    name = str(tool_name or "").strip()
    if not name:
        return
    try:
        _PLUGIN_TIERS[name] = Tier(str(tier or "").strip().lower())
    except ValueError:
        pass


# ── tier membership ───────────────────────────────────────────────────────────

# Read/benign: safe to run the instant a paired device asks.
ALLOW_TOOLS: frozenset[str] = frozenset({
    "project_rehearsal",
    "query_intelligence", "recall_conversation", "resource_status",
    "token_usage", "diagnostics", "patch_notes", "web_search",
    "vision_analyse", "find_files", "geo", "briefing", "open_news",
    "audio_devices",
    # Device commands still require Android's local per-command slide and a
    # correlated authorised result. A web confirmation would steal focus from
    # the target window before an Accessibility gesture can run.
    "phone_device",
    "display", "ui_theme", "ui_reactor", "ui_orbit", "ui_chrome", "ui_effect", "ui_screen", "ui_reset",
})

# Code execution / self-modification / security toggles: NEVER from a remote
# origin, no matter how well authenticated. These stay desktop-only forever.
FORBID_TOOLS: frozenset[str] = frozenset({
    "forge",            # LLM code generation + dynamic loading (arbitrary code)
    "dev_workbench",    # runs/edits project code
    "execute_plan",     # multi-step tool execution planner
    "self_repair",      # rewrites ORION's own source
    "process_governor", # can kill critical/system processes
    "security_watch",   # could disable the security sentinel
    "interface_control",# reconfigures ORION's own desktop UI (pointless remote)
    "software_install", # runs an installer on this PC: only ever on the owner's click here
})

# Irreversible or outward-facing: run only after an explicit on-phone tap.
# (This is also the fail-safe default for any tool not named anywhere here.)
CONFIRM_TOOLS: frozenset[str] = frozenset({
    "context_capsule", "experiment_notebook",
    "desktop_control", "file_controller", "outlook_mail", "notion_workspace",
    "web_control", "web_automation", "messaging", "social_media", "peripherals",
    "shutdown_orion", "restart_orion", "clipboard_operate", "organise_files",
    "backup", "gaming", "entertainment", "audio_studio", "mcp", "commerce",
    "security_recon",
    # Backgrounding detaches work from the turn, so from the phone it needs a
    # tap — but asking WHAT is running is a read (see the downgrades below).
    "job",
})

# CONFIRM tool + one of these *read* actions → downgrade to ALLOW. Only verified,
# genuinely read-only actions appear here; an action that is not listed simply
# stays CONFIRM (safe). Unknown action names are harmless — they never match.
READ_ACTION_DOWNGRADES: dict[str, frozenset[str]] = {
    "context_capsule": frozenset({"list", "show"}),
    "experiment_notebook": frozenset({"list", "show", "analyse"}),
    "outlook_mail": frozenset({
        "read_inbox", "read", "inbox", "check", "priority", "priority_emails",
        "important", "read_email", "open_email", "full_body", "pending_drafts",
        "drafts",
    }),
    "notion_workspace": frozenset({
        "list_tasks", "tasks", "todo", "upcoming_events", "calendar",
        "schedule", "agenda", "projects", "project_overview",
    }),
    "web_automation": frozenset({
        "launch", "go_to", "goto", "open", "navigate", "read", "read_page",
        "links", "scroll", "highlight", "screenshot", "capture",
    }),
    "web_control": frozenset({
        "open", "go_to", "navigate", "new_tab", "read_page", "read", "browse",
        "narrated_scroll", "read_down", "explore", "summarise", "summarize",
        "evaluate", "review",
    }),
    "desktop_control": frozenset({"list_windows", "windows", "read"}),
    # Reading Instagram DMs from the phone is instant; posting to TikTok or
    # actually sending a reply still needs an on-phone tap (outward-facing,
    # real-account actions).
    "social_media": frozenset({"instagram_check_dms", "check_dms"}),
    # Reading the authorized-target list or a public CVE database is a pure
    # read; actually scanning/sniffing/pinging a host still needs a tap.
    "security_recon": frozenset({"list_authorized", "list", "cve_lookup", "lan_devices"}),
    # Checking on background work from the phone is exactly the case the
    # feature exists for; only starting or cancelling one needs a tap.
    "job": frozenset({"list", "status", "jobs", "result", "collect", "output"}),
    "clipboard_operate": frozenset({"read"}),
    "peripherals": frozenset({"volume", "mute", "unmute", "toggle_mute", "volume_up",
                              "volume_down", "volume_status", "brightness_status"}),
    # file_controller read actions (conservative superset; non-existent names
    # simply never match and stay CONFIRM).
    "file_controller": frozenset({
        "read", "view", "search", "find", "list", "stat", "info", "preview",
        "open", "summarise", "summarize",
    }),
}

def _check_tiers() -> None:
    """The three tiers must be mutually exclusive.

    A real check rather than an assert. `python -O` strips assertions, and
    PyInstaller can be told to build that way for size — at which point this
    security invariant would stop being checked and a tool sitting in both
    ALLOW and FORBID would ship silently permitted. A control that disappears
    under an optimisation flag is not a control.
    """
    for left, right, label in ((ALLOW_TOOLS, FORBID_TOOLS, "ALLOW/FORBID"),
                               (ALLOW_TOOLS, CONFIRM_TOOLS, "ALLOW/CONFIRM"),
                               (CONFIRM_TOOLS, FORBID_TOOLS, "CONFIRM/FORBID")):
        overlap = left & right
        if overlap:
            raise RuntimeError(
                f"remote capability tiers overlap ({label}): "
                f"{', '.join(sorted(overlap))}. A tool cannot be in two tiers; "
                f"the stricter one would be silently ignored.")


_check_tiers()


def classify(tool: str, args: dict[str, Any] | None = None) -> Tier:
    """Return the remote-execution tier for *tool* called with *args*."""
    name = str(tool or "").strip()
    if name in FORBID_TOOLS:
        return Tier.FORBID
    if name in ALLOW_TOOLS:
        return Tier.ALLOW
    # A loaded plugin's own manifest-declared tier — checked after the static
    # tables (a core tool name always wins) but before the read-action
    # downgrades, since a plugin's declared tier is authoritative for it.
    if name in _PLUGIN_TIERS:
        return _PLUGIN_TIERS[name]
    action = str((args or {}).get("action") or "").lower().strip()
    if action and action in READ_ACTION_DOWNGRADES.get(name, frozenset()):
        return Tier.ALLOW
    # CONFIRM tools and everything unknown → CONFIRM (fail safe).
    return Tier.CONFIRM


def _gateway_target(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Expose the tool behind use_tool before applying the remote policy.

    The desktop gateway dispatches its target through the desktop dispatcher.
    Letting a remote approval execute that wrapper would bypass this gate.
    """
    if name != "use_tool":
        return name, args
    target = str(args.get("name") or args.get("tool") or "").strip()
    raw = args.get("arguments", args.get("args", {}))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw.strip() or "{}")
        except ValueError as exc:
            raise ValueError("use_tool arguments must be a JSON object") from exc
    if not target or target in {"find_tool", "use_tool"} or not isinstance(raw, dict):
        raise ValueError("use_tool requires a non-gateway tool and object arguments")
    return target, raw


def describe_action(tool: str, args: dict[str, Any] | None = None) -> str:
    """A short human sentence for the on-phone confirmation prompt."""
    name = str(tool or "").strip() or "an action"
    args = args or {}
    action = str(args.get("action") or "").lower().strip()
    if name == "phone_device":
        if action == "click":
            return "tap a control on your phone"
        if action == "scroll":
            return "scroll your phone screen"
        if action == "open_app":
            return f"open {str(args.get('package') or 'an app')[:150]} on your phone"
        if action == "make_call":
            return f"call {str(args.get('number') or 'a number')[:40]} from your phone"
        if action == "send_sms":
            return f"prepare an SMS to {str(args.get('number') or 'a number')[:40]}"
        if action in {"compose_email", "send_email"}:
            return f"{'prepare an email' if action == 'compose_email' else 'send an email'} to {str(args.get('to') or 'a recipient')[:80]} on your phone"
        if action in {"open_url", "map", "navigate", "share", "open_settings", "set_alarm", "set_timer", "create_contact", "create_calendar_event", "show_trending_videos", "set_text", "back", "home"}:
            return f"{action.replace('_', ' ')} on your phone"
        return "read your phone screen"
    pretty = {
        "outlook_mail": "email", "notion_workspace": "your Notion workspace",
        "messaging": "a message", "social_media": "your social media accounts",
        "security_recon": "a security scan/capture",
        "desktop_control": "your desktop",
        "file_controller": "your files", "web_automation": "the browser",
        "web_control": "the browser", "peripherals": "your PC's hardware",
        "shutdown_orion": "shut ORION down", "restart_orion": "restart ORION",
        "clipboard_operate": "your clipboard", "organise_files": "your files",
        "backup": "a backup", "gaming": "a game", "entertainment": "media",
        "audio_studio": "audio", "mcp": "an external service", "commerce": "a purchase/commerce action",
    }.get(name, name.replace("_", " "))
    if name in ("shutdown_orion", "restart_orion"):
        return pretty  # already a full phrase
    if name == "messaging" and action == "send":
        to = args.get("contact") or args.get("platform") or ""
        return f"send a message{f' to {to}' if to else ''}"
    if name == "outlook_mail" and action in ("send", "send_draft"):
        return "send an email"
    if name == "social_media" and action == "instagram_reply":
        to = args.get("contact") or ""
        return f"send an Instagram reply{f' to {to}' if to else ''}"
    if name == "social_media" and action == "tiktok_upload":
        return "post a video to TikTok"
    if action:
        return f"{action.replace('_', ' ')} · {pretty}"
    return f"use {pretty}"


def review_parameters(args: dict[str, Any]) -> str:
    """Render complete, bounded action parameters for the phone approval deck."""
    secret_words = ("password", "secret", "token", "credential", "api_key", "cookie")

    def visible(value: Any, key: str = "") -> Any:
        if any(word in key.lower() for word in secret_words):
            return "[redacted credential]"
        if isinstance(value, dict):
            return {str(k): visible(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [visible(item) for item in value]
        return value

    return json.dumps(visible(args), ensure_ascii=False, indent=2, default=str)


# ── pending-confirmation registry ─────────────────────────────────────────────

@dataclass
class PendingConfirmation:
    id: str
    tool: str
    args: dict[str, Any]
    device_id: str
    summary: str
    token: str
    created_at: float
    status: str = "pending"          # pending | approved | denied | expired | consumed

    def public(self) -> dict[str, Any]:
        """The view sent to the phone — never leaks the approval token."""
        return {
            "id": self.id, "tool": self.tool, "summary": self.summary,
            "parameterDiff": review_parameters(self.args),
            "status": self.status, "created_at": self.created_at,
        }


class RemoteConfirmationRegistry:
    """Holds CONFIRM-tier actions awaiting an on-phone approval tap.

    A confirmation carries a single-use ``token`` the phone must echo back, so a
    stray or replayed approval can't fire the action. Confirmations expire after
    ``ttl_s`` seconds; an expired one can never be approved.
    """

    def __init__(self, ttl_s: float = 180.0, capacity: int = 64) -> None:
        self.ttl_s = float(ttl_s)
        self.capacity = int(capacity)
        self._items: dict[str, PendingConfirmation] = {}

    def create(self, tool: str, args: dict[str, Any] | None,
               device_id: str, summary: str) -> PendingConfirmation:
        self.purge_expired()
        if len(self._items) >= self.capacity:
            # Drop the oldest to bound memory (a device that never taps).
            oldest = min(self._items.values(), key=lambda c: c.created_at)
            self._items.pop(oldest.id, None)
        conf = PendingConfirmation(
            id=secrets.token_urlsafe(9),
            tool=str(tool or ""),
            args=deepcopy(dict(args or {})),
            device_id=str(device_id or ""),
            summary=str(summary or ""),
            token=secrets.token_urlsafe(16),
            created_at=time.time(),
        )
        self._items[conf.id] = conf
        return conf

    def get(self, conf_id: str) -> PendingConfirmation | None:
        conf = self._items.get(str(conf_id or ""))
        if conf is None:
            return None
        if self._expired(conf):
            conf.status = "expired"
        return conf

    def resolve(self, conf_id: str, token: str, approve: bool, *, device_id: str
                ) -> PendingConfirmation | None:
        """Approve/deny a pending confirmation. Returns the confirmation on a
        valid, in-time, correct-token, still-pending resolution; otherwise None
        (wrong id, wrong token, expired, or already resolved)."""
        conf = self._items.get(str(conf_id or ""))
        if conf is None:
            return None
        if conf.status != "pending":
            return None
        if (not isinstance(device_id, str) or not device_id or not device_id.isascii()
                or not secrets.compare_digest(conf.device_id, device_id)):
            return None
        if self._expired(conf):
            conf.status = "expired"
            return None
        supplied = str(token or "")
        if not supplied.isascii() or not secrets.compare_digest(conf.token, supplied):
            return None
        conf.status = "approved" if approve else "denied"
        return conf

    def consume_approved(self, conf: PendingConfirmation) -> bool:
        """Claim one registered approval before any awaited execution starts.

        Runs on the gateway's event loop. Removing the entry synchronously
        prevents two callers from executing the same authorisation.
        """
        if self.get(conf.id) is not conf or conf.status != "approved":
            return False
        self._items.pop(conf.id)
        conf.status = "consumed"
        return True

    def pending_for(self, device_id: str) -> list[PendingConfirmation]:
        self.purge_expired()
        did = str(device_id or "")
        return [c for c in self._items.values()
                if c.status == "pending" and (not did or c.device_id == did)]

    def purge_expired(self) -> None:
        for conf in list(self._items.values()):
            if conf.status in ("approved", "denied") or self._expired(conf):
                self._items.pop(conf.id, None)

    def _expired(self, conf: PendingConfirmation) -> bool:
        return (time.time() - conf.created_at) > self.ttl_s


# ── the gate: remote tool execution through the policy ────────────────────────

class RemoteToolGate:
    """Execute dispatcher tools for a *remote* origin through the capability
    policy. FORBID is refused; CONFIRM is parked as a pending confirmation and
    pushed to the phone (nothing runs until the user taps); ALLOW runs now.

    Exposes a ``dispatch(name, args)`` method shape-compatible with the desktop
    dispatcher, so a gated LocalBrain can be built simply by handing it this
    object in place of the real dispatcher.
    """

    def __init__(self, dispatcher: Any, confirmations: RemoteConfirmationRegistry,
                 *, notify: Any = None, log: Any = None) -> None:
        self.dispatcher = dispatcher
        self.confirmations = confirmations
        # RemoteGateway binds its MobileViewportBridge after route registration.
        self.mobile_viewport: Any = None
        self._notify = notify                     # async callable(device_id, conf)
        self._log = log or (lambda _m: None)
        # The device the current turn belongs to (set per-request by the
        # gateway so a gated LocalBrain's tool calls attribute correctly).
        self.current_device: str = ""
        self._request_device: ContextVar[str | None] = ContextVar(
            "remote_request_device", default=None)

    @property
    def TOOL_DECLARATIONS(self) -> list[dict[str, Any]]:
        """Expose the Android tool to remote callers, never desktop callers."""
        desktop = getattr(self.dispatcher, "TOOL_DECLARATIONS", ()) or ()
        return [*desktop, PHONE_DEVICE_DECLARATION]

    def request_device(self) -> str | None:
        """The device the current request belongs to, if any."""
        return self._request_device.get()

    @contextmanager
    def for_device(self, device_id: str):
        """Scope tool ownership to one async request across awaited calls."""
        token = self._request_device.set(device_id)
        try:
            yield
        finally:
            self._request_device.reset(token)

    async def dispatch(self, name: str, args: dict[str, Any] | None = None,
                       device_id: str | None = None) -> ToolResult:
        args = dict(args or {})
        try:
            name, args = _gateway_target(name, args)
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        scoped = self._request_device.get()
        device = device_id if device_id is not None else (
            scoped if scoped is not None else self.current_device)
        if not phone_control.enabled():
            phone_control.record(name, "-", "off", device=device or "")
            self._log(f"REMOTE: phone control is off - '{name}' not run")
            return ToolResult(phone_control.OFF_REPLY, ok=False)
        if name == "phone_device":
            # The caller cannot select a device id. A phone turn must have
            # entered for_device() after gateway authentication.
            if not scoped or (device_id is not None and device_id != scoped):
                return ToolResult("A paired phone request is required.", ok=False)
            if self.mobile_viewport is None:
                return ToolResult("The phone viewport is unavailable.", ok=False)
            if _phone_command(args) is None:
                return ToolResult("Invalid phone action or parameters.", ok=False)
        tier = classify(name, args)
        if tier is Tier.FORBID:
            phone_control.record(name, tier.value, "refused", device=device or "")
            self._log(f"REMOTE: refused desktop-only tool '{name}'")
            return ToolResult(
                "That's a desktop-only capability — for safety I won't run "
                "it from a remote session.", ok=False)
        if tier is Tier.CONFIRM:
            if len(review_parameters(args)) > 8_000:
                return ToolResult("Action parameters are too large for phone review.", ok=False)
            summary = describe_action(name, args)
            conf = self.confirmations.create(name, args, device or "", summary)
            if self._notify is not None:
                try:
                    await self._notify(device or "", conf)
                except Exception:
                    pass
            phone_control.record(name, tier.value, "asked", device=device or "")
            self._log(f"REMOTE: confirm-gated '{name}' → {conf.id}")
            return ToolResult(
                f"That would {summary}. I've sent it to your phone to "
                "approve — tap to confirm and I'll carry on.",
                ok=True, media={"confirm": conf.public()})
        if name == "phone_device":
            result = await self._dispatch_phone(scoped or "", args)
        elif self.dispatcher is None:
            return ToolResult("No dispatcher is wired on this node.", ok=False)
        else:
            result = await self.dispatcher.dispatch(name, args)
        phone_control.record(name, tier.value, "ran" if getattr(result, "ok", True) else "failed",
                             device=device or "")
        return result

    async def _dispatch_phone(self, device_id: str, args: dict[str, Any]) -> ToolResult:
        bridge = self.mobile_viewport
        if bridge is None or not device_id:
            return ToolResult("The paired phone viewport is unavailable.", ok=False)
        command = _phone_command(args)
        if command is None:
            return ToolResult("Invalid phone action or parameters.", ok=False)
        method, params = command
        try:
            if method == "latest_frame":
                frame = bridge.latest_frame(device_id)
                if frame is None:
                    return ToolResult("No fresh phone screen frame is available.", ok=False)
                return ToolResult(
                    "Current paired phone screen frame.",
                    media={"data": frame.jpeg, "mime_type": "image/jpeg"},
                )
            result = await bridge.request_device_action(device_id, method, params)
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolResult(f"Phone action failed: {str(exc).splitlines()[0][:200]}", ok=False)
        if method == "viewport.capture_dom":
            return ToolResult(json.dumps(result["snapshot"], ensure_ascii=False))
        lines = str(result.get("detail") or "").splitlines()
        detail = lines[0][:300] if lines else ""
        return ToolResult(detail or f"{args['action'].replace('_', ' ').capitalize()} completed on the phone.")

    async def run_approved(self, conf: PendingConfirmation) -> ToolResult:
        """Run a confirmation the user has approved on the phone."""
        if not self.confirmations.consume_approved(conf):
            return ToolResult("Action has not been approved, expired, or already ran.", ok=False)
        if not phone_control.enabled():
            phone_control.record(conf.tool, "-", "off", device=conf.device_id)
            return ToolResult(phone_control.OFF_REPLY, ok=False)
        phone_control.record(conf.tool, Tier.CONFIRM.value, "approved", device=conf.device_id)
        try:
            name, args = _gateway_target(conf.tool, dict(conf.args))
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if classify(name, args) is Tier.FORBID:
            return ToolResult("That capability is now desktop-only.", ok=False)
        if name == "remote_batch":
            return await self._run_plan(conf.args.get("steps"), conf.device_id)
        if name == "phone_device":
            self._log(f"REMOTE: running approved phone action ({conf.id})")
            return await self._dispatch_phone(conf.device_id, args)
        if self.dispatcher is None:
            return ToolResult("No dispatcher is wired on this node.", ok=False)
        self._log(f"REMOTE: running approved '{name}' ({conf.id})")
        return await self.dispatcher.dispatch(name, args)

    def _validate_plan(self, steps: Any) -> str:
        if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
            return "A task must contain 1–6 tool steps."
        known = {item.get("name") for item in self.TOOL_DECLARATIONS}
        from .plan_executor import ToolCatalogue
        catalogue = ToolCatalogue(self.TOOL_DECLARATIONS)
        for step in steps:
            if (not isinstance(step, dict) or step.get("tool") not in known
                    or not isinstance(step.get("args"), dict)):
                return "The plan contains an unknown tool or invalid parameters. Nothing has run."
            try:
                name, args = _gateway_target(step["tool"], step["args"])
            except ValueError as exc:
                return f"{exc}. Nothing has run."
            if name not in known:
                return "The plan contains an unknown tool or invalid parameters. Nothing has run."
            reason = catalogue.check_args(name, args)
            if reason:
                return f"{name}: {reason}. Nothing has run."
            if name == "phone_device" and _phone_command(args) is None:
                return "Invalid phone action or parameters. Nothing has run."
            if classify(name, args) is Tier.FORBID:
                return f"{name} is a desktop-only capability. Nothing has run."
        try:
            review_limit = 32_768 if all(step["tool"] == "phone_device" for step in steps) else 8000
            if len(review_parameters({"steps": steps})) > review_limit:
                return "The task is too large for phone review. Nothing has run."
        except (TypeError, ValueError):
            return "Invalid task parameters. Nothing has run."
        return ""

    async def dispatch_plan(self, steps: list[dict[str, Any]]) -> ToolResult:
        """Review a whole task once so approval preserves its remaining steps."""
        error = self._validate_plan(steps)
        device_id = self.request_device()
        if error or not device_id:
            return ToolResult(error or "A paired phone request is required.", ok=False)
        if all(classify(*_gateway_target(step["tool"], step["args"])) is Tier.ALLOW
               for step in steps):
            return await self._run_plan(steps, device_id)
        summary = "; then ".join(
            describe_action(*_gateway_target(step["tool"], step["args"]))
            for step in steps
        )
        conf = self.confirmations.create("remote_batch", {"steps": steps}, device_id, summary)
        if self._notify is not None:
            await self._notify(device_id, conf)
        return ToolResult(
            "The task is ready for your approval in the app. Nothing has run yet. "
            "Review the steps and slide to approve.", media={"confirm": conf.public()})

    async def _run_plan(self, steps: Any, device_id: str) -> ToolResult:
        error = self._validate_plan(steps)
        if error:
            return ToolResult(error, ok=False)
        receipts = []
        evidence = []
        for index, step in enumerate(steps, 1):
            name = str(step.get("tool") or "")
            try:
                name, args = _gateway_target(step["tool"], step["args"])
                if name == "phone_device":
                    result = await self._dispatch_phone(device_id, args)
                elif self.dispatcher is None:
                    result = ToolResult("No dispatcher is available.", ok=False)
                else:
                    with self.for_device(device_id):
                        result = await self.dispatcher.dispatch(name, dict(args))
            except Exception:
                result = ToolResult("The tool failed. Check the PC diagnostics log.", ok=False)
            receipts.append(f"{index}. {name}: {result.text}")
            evidence.extend(getattr(result, "evidence", None) or [])
            if not result.ok:
                return ToolResult("\n".join(receipts) + "\nStopped; later steps have not run.", ok=False, evidence=evidence)
        return ToolResult("\n".join(receipts), evidence=evidence + [{"steps_completed": len(receipts)}])
