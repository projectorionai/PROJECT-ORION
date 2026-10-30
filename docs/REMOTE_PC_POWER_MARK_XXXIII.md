# Remote PC shutdown

ORION 3.13 adds **Options → Connections → PC controls** to the independent phone console. It uses the PC pairing and current saved HTTPS address; the phone AI and its own key continue to work independently.

## Use

1. Keep ORION running on the Windows PC and expose its local gateway through the configured HTTPS route.
2. Open **PC controls**. If needed, choose **Pair or change PC**, enter the current HTTPS origin and a fresh code from the PC's **SECURITY → PAIR PHONE** control.
3. Choose **Shut down PC**, save open work and confirm the explicit native dialogue.
4. ORION acknowledges a 60-second countdown. **Cancel shutdown** is available on that same paired phone until the countdown ends.

Closing the phone controls does not cancel an acknowledged countdown. Closing or restarting ORION on the PC does cancel it. An unavailable PC, expired pairing or stale temporary tunnel address produces a connection failure; it does not fall back to an unauthenticated network command. A switched-off PC cannot be woken by this feature.

## Windows behaviour

The countdown is managed by ORION. At expiry the backend runs a fixed `shutdown.exe /s /t 0 /d p:0:0` command without a shell, `/f` or an arbitrary remote target. Windows can block shutdown when an application has unsaved work. The app reports **Windows accepted shutdown** when the command succeeds; that acknowledgement does not prove the PC has physically switched off.

Microsoft documents that a positive Windows `/t` implicitly enables forced application closure. ORION's own countdown avoids that behaviour. [Microsoft shutdown command documentation](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/shutdown).

## Endpoint contract

All endpoints require the existing paired **Authorization: Bearer** access token. They set `Cache-Control: no-store` and reject unknown JSON fields and bodies exceeding 2,048 bytes.

| Method | Path | Request |
| --- | --- | --- |
| GET | `/api/pc/power` | No body; own latest shutdown status |
| POST | `/api/pc/power/prepare` | `{}`; returns device-bound confirmation valid for 90 seconds |
| POST | `/api/pc/power/shutdown` | `request_id` (canonical UUID v4), `confirmation` |
| POST | `/api/pc/power/cancel` | New `request_id`, original `shutdown_id` |

Shutdown requires a fresh confirmation for each new request. Cancellation is restricted to the device that scheduled it. Simultaneous requests admit one countdown. Pairing is checked again at execution, including expiry and revocation. Native HTTP mutations are sent once to one authenticated HTTPS origin; redirect, automatic transport replay and mutation failover are disabled. A lost response requires a status refresh before a fresh confirmation.

The atomic ledger `config/remote_power_requests.json` retains at most 128 request records for one day, together with active records. Duplicate IDs return the original outcome. Consumed confirmations cannot be reused after older ledger entries expire. Corrupt or unwritable persistence fails closed. On restart a saved countdown is cancelled; an interrupted Windows command is marked uncertain and is never replayed.

## Verification limits

Tests inject the OS runner and cover confirmation expiry, ownership, revocation, concurrent admission, cancellation, lifecycle close, restart, duplicate IDs, corrupt or unwritable persistence, rejected bodies, fixed command arguments and honest Windows failure states. The actual PC is never shut down during those tests. A real power-off remains a deliberate user action after saving work.

The final power suite passed **29 tests**. The related gateway regression run passed **141 tests**, covering existing paired chat, confirmation, body limits, audio, reconnect and reverse proxy behaviour alongside the power controls. Review also caught and fixed an unbounded process reap and a countdown persistence failure that could otherwise leave a false “executing” state. Process admission, waiting and reaping each have a finite limit, and gateway exit cancels an unfinished countdown before closing streams.
