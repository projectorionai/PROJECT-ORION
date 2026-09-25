"""
SystemActionGuard — deterministic intent separation and confirmation policy
for destructive system commands (Section 10).

The problem this solves: an input such as "shut down" must never immediately
power off the whole computer.  "Shut down" is genuinely ambiguous — it could
mean stop speaking, stop the current task, close ORION, or power off the host —
so it is classified and, when it resolves to a destructive OS action, gated
behind an explicit, short-lived, token-bound confirmation that is approved by a
human through the UI, not by model text.

This module is pure and deterministic:

* No LLM, no Qt, no OS calls — it only classifies text and manages pending
  confirmations.  The caller performs the resolved action.
* A destructive action is never "executed" here; ``confirm()`` merely returns
  the approved intent so the caller's deterministic executor can run it.
* Confirmation tokens are random, single-use, bound to one specific action, and
  expire.  A bare "yes" is not a confirmation unless it is securely bound to the
  single active pending action within its lifetime.
"""

from __future__ import annotations

import re
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

try:
    from .constants import SYSTEM_CONFIRM_TTL_SECONDS
except Exception:  # pragma: no cover - constants always import in-tree
    SYSTEM_CONFIRM_TTL_SECONDS = 60.0


class ActionIntent(str, Enum):
    # ── non-destructive: handled locally by the app, no OS/host effect ──────
    STOP_SPEAKING = "stop_speaking"          # silence current speech
    STOP_RESPONSE = "stop_response"          # abort the reply being generated
    STOP_TASK = "stop_task"                  # cancel the running task
    STOP_ORION = "stop_orion"                # stop the assistant runtime
    CLOSE_APP = "close_app"                  # close the ORION application
    STOP_SERVICE = "stop_service"            # stop an internal service
    # ── destructive: affect the whole machine, require confirmation ─────────
    PROCESS_KILL = "process_kill"            # terminate another program
    OS_SHUTDOWN = "os_shutdown"
    OS_RESTART = "os_restart"
    OS_LOGOUT = "os_logout"
    OS_SLEEP = "os_sleep"
    OS_HIBERNATE = "os_hibernate"
    # ── spends the user's money, immediately and irreversibly ───────────────
    PLACE_CALL = "place_call"                # dial a real telephone number
    SEND_MESSAGE = "send_message"            # a billable SMS or equivalent
    # ── could not resolve confidently ───────────────────────────────────────
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


DESTRUCTIVE_INTENTS = frozenset({
    # Killing a process cannot be undone and takes unsaved work with it, so it
    # belongs behind the same human-issued token as a power action rather than
    # behind a parameter the model fills in for itself.
    ActionIntent.PROCESS_KILL,
    ActionIntent.OS_SHUTDOWN,
    ActionIntent.OS_RESTART,
    ActionIntent.OS_LOGOUT,
    ActionIntent.OS_SLEEP,
    ActionIntent.OS_HIBERNATE,
    # A phone call connects immediately, costs money, and cannot be un-made.
    # It belongs here for the same reason PROCESS_KILL does: a confirm flag
    # the model fills in for itself is a convention, not a gate.
    ActionIntent.PLACE_CALL,
    ActionIntent.SEND_MESSAGE,
})

# Human-readable descriptions used in the confirmation prompt.
_INTENT_PHRASE = {
    ActionIntent.PLACE_CALL: "place a real phone call from",
    ActionIntent.SEND_MESSAGE: "send a real, billable message from",
    ActionIntent.PROCESS_KILL: "terminate a running program on",
    ActionIntent.OS_SHUTDOWN: "power off (shut down)",
    ActionIntent.OS_RESTART: "restart (reboot)",
    ActionIntent.OS_LOGOUT: "log out of",
    ActionIntent.OS_SLEEP: "put to sleep",
    ActionIntent.OS_HIBERNATE: "hibernate",
}


class DecisionKind(str, Enum):
    EXECUTE = "execute"          # non-destructive: caller may act now
    CONFIRM = "confirm"          # destructive: awaiting bound confirmation
    CLARIFY = "clarify"          # ambiguous: ask the user what they meant
    REJECT = "reject"            # confirmation invalid/expired/replayed
    UNKNOWN = "unknown"


@dataclass
class PendingAction:
    token: str
    intent: ActionIntent
    machine: str
    created_at: float
    ttl: float
    origin: str = ""
    #: What the action applies to — a PID, a window label. Bound to the token
    #: at arming time and returned unchanged on confirmation, so the thing the
    #: user approved is necessarily the thing that runs. Without this the
    #: target would have to be supplied again at release, and whatever supplied
    #: it could change it after approval.
    payload: dict = field(default_factory=dict)

    def expired(self, now: float) -> bool:
        return now >= self.created_at + self.ttl


@dataclass
class GuardDecision:
    kind: DecisionKind
    intent: ActionIntent
    message: str
    token: str | None = None
    machine: str = ""
    warnings: list[str] = field(default_factory=list)
    #: Carried back from the pending action on EXECUTE — see PendingAction.
    payload: dict = field(default_factory=dict)

    @property
    def requires_confirmation(self) -> bool:
        return self.kind is DecisionKind.CONFIRM


# ── deterministic intent classification ─────────────────────────────────────

def _has(text: str, *phrases: str) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text) for p in phrases)


# Words that may legitimately accompany a BARE power command ("shut down please
# now, Orion") without turning it into a device command.
_POWER_FILLERS = frozenset({
    "please", "now", "just", "can", "you", "could", "would", "will", "kindly",
    "orion", "for", "me", "the", "app", "application", "assistant", "yourself",
    "right", "away", "immediately", "everything", "all", "and", "then", "let",
    "us", "lets", "okay", "ok", "go", "ahead", "it", "down", "off", "please",
    "completely", "fully", "yourself", "self",
})


def _is_bare_power(text: str, *triggers: str) -> bool:
    """True when *text* is a BARE power command — the trigger verb plus only
    filler/politeness words, with no concrete object.

    'shut down', 'shut it down', 'turn off now please' are bare (a power command
    addressed to ORION); 'turn off the lights', 'shut down the download', 'turn
    off notifications' carry a real object and are NOT power commands — they must
    fall through to the model as ordinary device/app requests."""
    remainder = " " + text.strip() + " "
    for trig in triggers:
        remainder = re.sub(rf"(?<!\w){re.escape(trig)}(?!\w)", " ", remainder)
    leftover = [w for w in re.findall(r"[a-z]+", remainder) if w not in _POWER_FILLERS]
    return not leftover


def classify_intent(text: str) -> ActionIntent:
    """Map a natural-language command to a single intent, deterministically.

    Ordering matters: more specific phrases are tested before the bare,
    ambiguous "shut down"/"stop" which fall through to AMBIGUOUS."""
    t = " " + re.sub(r"\s+", " ", str(text or "").strip().lower()) + " "
    if not t.strip():
        return ActionIntent.UNKNOWN

    # An explicit host/computer object marks an OS-level power intent.
    host = _has(t, "computer", "pc", "laptop", "machine", "system", "desktop",
                "windows", "whole thing", "everything", "my computer", "the computer")

    # Hibernate / sleep / logout are unambiguous verbs.
    if _has(t, "hibernate"):
        return ActionIntent.OS_HIBERNATE
    if _has(t, "log out", "logout", "log off", "logoff", "sign out", "signout"):
        return ActionIntent.OS_LOGOUT
    if _has(t, "hibernate the computer") or (_has(t, "sleep") and host) or \
            _has(t, "go to sleep", "put to sleep", "suspend"):
        # "go to sleep" said TO Orion (standby) is handled upstream as a pause;
        # here we only reach OS sleep when a host object is present or "suspend".
        if _has(t, "suspend") or host or _has(t, "put the computer to sleep",
                                              "put the pc to sleep", "sleep the computer",
                                              "sleep the pc"):
            return ActionIntent.OS_SLEEP

    if _has(t, "reboot", "restart"):
        # "restart orion/yourself/the app" is an app restart, not an OS reboot.
        if _has(t, "orion", "yourself", "the app", "the application", "the assistant"):
            return ActionIntent.STOP_ORION
        if host or _has(t, "reboot", "restart the computer", "restart the pc",
                        "restart the machine", "restart the system"):
            return ActionIntent.OS_RESTART
        # Bare "restart" with no object → ambiguous (means ORION), ask.  But
        # "restart the download" carries an object → not a power command.
        if _is_bare_power(t, "restart", "reboot"):
            return ActionIntent.AMBIGUOUS
        return ActionIntent.UNKNOWN

    # Self-directed shutdown synonyms addressed to ORION that never name a host
    # ("turn yourself off", "go offline", "go dark") → close ORION.
    if _has(t, "turn yourself off", "turn off yourself", "go offline",
            "go dark", "power yourself down", "power yourself off"):
        return ActionIntent.CLOSE_APP

    # Shutdown / power off / power down / turn off.
    if _has(t, "power off", "poweroff", "power down", "turn off", "switch off",
            "shut down", "shutdown", "shut it down", "shut down the computer"):
        if _has(t, "orion", "yourself", "the app", "the application", "the assistant"):
            return ActionIntent.CLOSE_APP
        if host or _has(t, "power off", "poweroff", "turn off the computer",
                        "turn off the pc", "shut down the computer",
                        "shut down the pc", "shut down the machine"):
            return ActionIntent.OS_SHUTDOWN
        # Bare "shut down" / "turn off" with no object → genuinely ambiguous
        # (in conversation the subject is ORION).  "turn off the lights" /
        # "shut down the download" carry an object → device command, not power.
        if _is_bare_power(t, "power off", "poweroff", "power down", "turn off",
                          "switch off", "shut down", "shutdown", "shut it down"):
            return ActionIntent.AMBIGUOUS
        return ActionIntent.UNKNOWN

    # Non-destructive stop family.
    if _has(t, "stop speaking", "stop talking", "be quiet", "quiet please",
            "hush", "stop the voice", "mute yourself"):
        return ActionIntent.STOP_SPEAKING
    if _has(t, "stop the response", "stop responding", "stop generating",
            "cancel the response", "stop the answer"):
        return ActionIntent.STOP_RESPONSE
    if _has(t, "stop the task", "cancel the task", "abort the task",
            "stop the current task", "cancel that", "abort"):
        return ActionIntent.STOP_TASK
    if _has(t, "close orion", "close the app", "close the application",
            "quit orion", "exit orion", "quit the app", "close yourself"):
        return ActionIntent.CLOSE_APP
    if _has(t, "stop orion", "shut yourself down", "stop the assistant",
            "shut down orion", "power down orion"):
        return ActionIntent.STOP_ORION
    if _has(t, "stop the service", "shut down the service", "stop service",
            "restart the service", "stop the server"):
        return ActionIntent.STOP_SERVICE
    if _has(t, "stop") and not host:
        # A bare "stop" is a soft stop of the current activity, never the OS.
        return ActionIntent.STOP_TASK

    return ActionIntent.UNKNOWN


class SystemActionGuard:
    """Policy layer that gates destructive OS actions behind bound confirmation.

    Not a tool the model can call to execute anything — it only classifies and
    tracks pending confirmations.  The dispatcher/GUI perform resolved actions.
    """

    def __init__(self, machine: str = "this computer", *,
                 ttl_s: float | None = None,
                 clock: Callable[[], float] | None = None,
                 logger: Callable[[str], None] | None = None) -> None:
        self.machine = machine
        self.ttl_s = float(ttl_s if ttl_s is not None else SYSTEM_CONFIRM_TTL_SECONDS)
        self._clock = clock or time.monotonic
        self._log = logger or (lambda _m: None)
        self._pending: dict[str, PendingAction] = {}

    # ── evaluation ──────────────────────────────────────────────────────────

    def evaluate(self, text: str, origin: str = "") -> GuardDecision:
        """Classify a command and decide how to handle it.

        Destructive OS intents produce a CONFIRM decision carrying a fresh,
        single-use token bound to that exact action; the token is delivered to
        the confirmation UI, NOT surfaced to the model."""
        self._sweep()
        intent = classify_intent(text)
        if intent in DESTRUCTIVE_INTENTS:
            pending = self._arm(intent, origin)
            phrase = _INTENT_PHRASE.get(intent, intent.value)
            self._log(f"GUARD: destructive intent '{intent.value}' armed (token issued), awaiting confirmation.")
            return GuardDecision(
                kind=DecisionKind.CONFIRM,
                intent=intent,
                message=(f"This will {phrase} {self.machine}. This is a destructive "
                         "action — please confirm the on-screen prompt to proceed, "
                         "or cancel to abort."),
                token=pending.token,
                machine=self.machine,
            )
        if intent is ActionIntent.AMBIGUOUS:
            self._log("GUARD: ambiguous stop/shutdown command — requesting clarification.")
            return GuardDecision(
                kind=DecisionKind.CLARIFY,
                intent=intent,
                message=("Did you mean stop speaking, stop the current task, close "
                         "ORION, or shut down the computer? I won't power anything "
                         "off without a clear instruction."),
            )
        if intent is ActionIntent.UNKNOWN:
            return GuardDecision(DecisionKind.UNKNOWN, intent,
                                 "That isn't a recognised system command.")
        # Non-destructive: the caller may act immediately.
        return GuardDecision(DecisionKind.EXECUTE, intent,
                             f"Proceeding: {intent.value.replace('_', ' ')}.")

    # ── confirmation ─────────────────────────────────────────────────────────

    def _arm(self, intent: ActionIntent, origin: str,
             payload: dict | None = None) -> PendingAction:
        token = secrets.token_urlsafe(16)
        pending = PendingAction(
            token=token, intent=intent, machine=self.machine,
            created_at=self._clock(), ttl=self.ttl_s, origin=origin,
            payload=deepcopy(payload or {}),
        )
        self._pending[token] = pending
        return pending

    def request_confirmation(self, intent: ActionIntent, origin: str = "",
                             payload: dict | None = None) -> GuardDecision:
        """Directly arm a destructive intent chosen by a deterministic caller
        (e.g. the peripherals tool asked to 'shutdown').  Non-destructive
        intents are returned as EXECUTE.

        ``payload`` describes WHAT the action applies to and is bound to the
        token, so the target the user sees on the prompt is necessarily the
        target that runs."""
        if intent not in DESTRUCTIVE_INTENTS:
            return GuardDecision(DecisionKind.EXECUTE, intent,
                                 f"Proceeding: {intent.value.replace('_', ' ')}.",
                                 payload=deepcopy(payload or {}))
        pending = self._arm(intent, origin, payload)
        phrase = _INTENT_PHRASE.get(intent, intent.value)
        self._log(f"GUARD: destructive intent '{intent.value}' armed for confirmation.")
        return GuardDecision(
            kind=DecisionKind.CONFIRM, intent=intent,
            message=(f"This will {phrase} {self.machine}. Confirm the on-screen "
                     "prompt to proceed, or cancel to abort."),
            token=pending.token, machine=self.machine,
            payload=deepcopy(pending.payload),
        )

    def confirm(self, token: str) -> GuardDecision:
        """Approve a pending action by its bound token.  Single-use: the token
        is consumed whether it succeeds or fails, defeating replay."""
        self._sweep()
        pending = self._pending.pop(token, None)
        if pending is None:
            self._log("GUARD: confirmation rejected — unknown or already-used token.")
            return GuardDecision(DecisionKind.REJECT, ActionIntent.UNKNOWN,
                                 "No matching pending action — it may have expired, "
                                 "been cancelled, or already been used.")
        if pending.expired(self._clock()):
            self._log("GUARD: confirmation rejected — token expired.")
            return GuardDecision(DecisionKind.REJECT, pending.intent,
                                 "That confirmation has expired. Please request the "
                                 "action again.")
        self._log(f"GUARD: destructive action '{pending.intent.value}' confirmed and released.")
        return GuardDecision(DecisionKind.EXECUTE, pending.intent,
                             f"Confirmed: {_INTENT_PHRASE.get(pending.intent, pending.intent.value)} "
                             f"{pending.machine}.", token=token, machine=pending.machine,
                             payload=deepcopy(pending.payload))

    def confirm_phrase(self, phrase: str) -> GuardDecision:
        """Bind an affirmative reply ("yes", "go ahead") to the single active
        pending action, ONLY when exactly one is pending and unexpired.  Any
        ambiguity (zero or many pending) is rejected — a stray "yes" can never
        trigger a destructive action."""
        self._sweep()
        affirmative = _has(" " + str(phrase or "").strip().lower() + " ",
                           "yes", "yeah", "yep", "confirm", "confirmed", "do it",
                           "go ahead", "proceed", "affirmative", "correct", "ok", "okay")
        if not affirmative:
            return GuardDecision(DecisionKind.REJECT, ActionIntent.UNKNOWN,
                                 "That was not read as a confirmation.")
        if len(self._pending) != 1:
            self._log("GUARD: affirmative reply ignored — no single pending action to bind it to.")
            return GuardDecision(DecisionKind.REJECT, ActionIntent.UNKNOWN,
                                 "There isn't a single pending action to confirm; "
                                 "please use the on-screen prompt.")
        token = next(iter(self._pending))
        return self.confirm(token)

    def cancel(self, token: str | None = None) -> GuardDecision:
        """Cancel a specific pending action, or all of them when token is None."""
        if token is None:
            n = len(self._pending)
            self._pending.clear()
            self._log(f"GUARD: cancelled {n} pending action(s).")
            return GuardDecision(DecisionKind.REJECT, ActionIntent.UNKNOWN,
                                 "Cancelled — nothing was executed.")
        self._pending.pop(token, None)
        self._log("GUARD: pending action cancelled.")
        return GuardDecision(DecisionKind.REJECT, ActionIntent.UNKNOWN,
                             "Cancelled — nothing was executed.")

    # ── introspection / housekeeping ──────────────────────────────────────────

    def pending_count(self) -> int:
        self._sweep()
        return len(self._pending)

    def _sweep(self) -> None:
        now = self._clock()
        stale = [tok for tok, p in self._pending.items() if p.expired(now)]
        for tok in stale:
            self._pending.pop(tok, None)
        if stale:
            self._log(f"GUARD: {len(stale)} pending confirmation(s) expired.")
