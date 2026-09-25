# Security and private data

ORION is a personal desktop application. A running installation accumulates
private information. Never publish the complete installation directory.

## Data that stays local

Credentials, identity, browser profiles, remote pairing data, contacts, memories,
conversations, recordings, location information, research and generated reports
are runtime data. Most settings/databases are under `config/`; other private
directories include `conversations/`, `research/`, `exports/`, `reports/` and
audio-studio output. An installed `dist/ORION` can contain its own private state.
Local tool folders, virtual environments and IDE files are excluded from source.

`.gitignore` prevents ordinary staging of new matching files. It does not remove
tracked files or clean old commits. Deleting a secret from source does not
revoke it or remove older copies.

## Before publication

Use the [source export procedure](docs/PUBLISHING.md). Review source and Git
history separately. Revoke/rotate credentials that have entered commits or
shared builds through their providers. Do not post values in issues or logs.

The publication helper checks source boundaries, recognised key formats and
optionally known local credentials/private terms. It cannot recognise every
secret or determine whether arbitrary prose is personal. A clean result is
one check, not a security certification.

## Operational boundaries

Provider-backed features can transmit prompts, images and selected context to
the configured provider. Offline availability varies by feature and models.
Code plugins and computer-control tools run with local user permissions; review
them before enabling. Remote access needs appropriate authentication and network
configuration.

## Reporting a problem

Use synthetic reproduction data in public reports, without credentials, private
databases or conversations. If the repository enables private vulnerability
reporting, use it for sensitive findings; otherwise contact the maintainer
privately before disclosing sensitive details.
