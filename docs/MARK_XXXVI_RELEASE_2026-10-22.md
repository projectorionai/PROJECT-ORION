# ORION Mark XXXVI — Background Operations

Package identity: 36.0.0. Build identity: 2026.10.22-r1-xxxvi.

## Connect messaging accounts

Open Preferences and use the Background messaging controls to connect WhatsApp
or Discord. Each Connect button opens ORION's separate account window. Complete
WhatsApp pairing or Discord sign-in there, then choose Check connection.
ORION closes that setup window after confirming the connection and uses its
separate hidden profile for later messages. The normal browser profile is not
shared. No message is sent during connection checks.

An existing login in the Windows WhatsApp/Discord app does not sign in this
separate profile. A missing login returns a setup result instead of opening a
window during a task. Status reports configuration presence; Check connection
verifies the account page. Dependency presence alone does not mean connected.

Personal Discord delivery requires a unique exact existing direct-message
entry and a verifiable signed-in sender. WhatsApp self-chat uses the app's
explicit self marker. Duplicate recipient names, an existing draft, an
unsupported page layout or missing verification structure stop submission.
Message text retains its original spaces, line breaks and Unicode.

Configured bots and webhooks use their own sender identities; their receipts
identify that route. They do not become a personal account. Telegram uses a
configured bot. Slack and Teams do not gain a new hidden personal-account route
in this release.

## Work while ORION runs

Ordinary tool calls, scheduled jobs, workflows and plugin events enter a
background execution scope. Reviewed API, HTTP, file and owned-browser routes
can operate there. Shared mouse/keyboard, clipboard writes, visible browser
control, shell-open actions and modal requests defer before changing the
desktop. Model arguments cannot grant foreground access. The Preferences
Connect button grants only its explicit user-requested setup operation.

Unrestricted host code is not isolated merely because its console is hidden.
Background development execution requires an explicitly isolated runtime.
Reviewed isolation support still depends on that runtime being available.
Trusted Python plugins are not OS sandboxes; static effect detection and
cooperative guards do not establish containment of concealed native calls.

## Delivery outcomes and extension reliability

A submission acknowledgement is separate from delivery/read confirmation.
A connection lost after submission returns an uncertain outcome. Check the
conversation before trying again. ORION does not automatically retry an
uncertain send. Ownership remains with a timed-out physical worker until it
actually exits, preventing another invocation from overlapping that operation.

Registered plugin calls share validation, enabled-state checks, confirmation
policy, a total invocation deadline and declared task-scoped credentials.
Source fingerprints bind detected effects to the activated handler. Result
contracts require actual booleans and bounded text; selected HTTP integrations
also enforce response byte limits. Camera failures are typed failures.

Shipped plugin updates retain user edits and deliberate removals. Missing
modules, account credentials and unavailable optional services remain explicit
readiness states. Source regression tests and recovery evidence are retained
for maintenance; obsolete generated artifacts are cleaned separately.
