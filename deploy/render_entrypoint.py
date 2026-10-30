"""Validate a Render container's network settings before importing ORION.

Render terminates HTTPS outside the container. Authentication remains the
normal ORION one-time pairing and short-lived bearer-token exchange. No
forwarded header is treated as proof of identity or a secure connection.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import MutableMapping


def configure_environment(environ: MutableMapping[str, str]) -> None:
    """Apply validated platform settings without reading credentials or stores."""
    if environ.get("RENDER", "").strip().lower() != "true":
        raise ValueError("This launcher requires a Render web service.")
    if environ.get("ORION_RENDER_EPHEMERAL_ACK", "") != "1":
        raise ValueError(
            "Acknowledge Render Free's disposable memory and pairing state with "
            "ORION_RENDER_EPHEMERAL_ACK=1; see deploy/README_RENDER_FREE.md.")

    port = environ.get("PORT", "10000").strip()
    if not re.fullmatch(r"[0-9]{1,5}", port) or not 1 <= int(port) <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535.")
    if environ.get("ORION_REMOTE_TLS", "").strip().lower() not in {"", "0", "false", "no", "off"}:
        raise ValueError("ORION_REMOTE_TLS is not supported by the Render Free trial.")

    # This helper is stdlib-only and does not resolve CONFIG_DIR. Importing the
    # application itself before setting its store would select the wrong data.
    from orion_core.remote_endpoints import public_https_endpoint

    origin = environ.get("RENDER_EXTERNAL_URL", "").strip()
    endpoint = public_https_endpoint(origin)
    if endpoint is None or not endpoint["host"].endswith(".onrender.com"):
        raise ValueError("RENDER_EXTERNAL_URL must be Render's assigned HTTPS origin.")

    config_dir = environ.get("ORION_CONFIG_DIR", "/app/data/config").strip()
    if not config_dir or not Path(config_dir).is_absolute():
        raise ValueError("ORION_CONFIG_DIR must be an absolute writable directory.")

    # Retain only operator-owned secrets, never place them in command arguments
    # or print the environment. The state directory is deliberately disposable.
    environ.update({
        "ORION_HEADLESS": "1",
        "ORION_MODE": "cloud",
        "ORION_REMOTE_ACCESS": "1",
        "ORION_REMOTE_HOST": "0.0.0.0",
        "ORION_REMOTE_PORT": str(int(port)),
        "ORION_REMOTE_EXPOSURE": "platform-proxy",
        "ORION_REMOTE_PUBLIC_URL": endpoint["url"],
        "ORION_REMOTE_TLS": "0",
        "ORION_CONFIG_DIR": config_dir,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONUNBUFFERED": "1",
    })


def main() -> int:
    """Replace this process so SIGTERM reaches ORION's graceful shutdown."""
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    environ = dict(os.environ)
    try:
        configure_environment(environ)
    except ValueError as exc:
        print(f"ORION cloud configuration failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print("Render Free trial: memory and pairing are lost after sleep or restart. "
          "Pair using the fresh one-time code in this private service log.", flush=True)
    os.execve(sys.executable, [sys.executable, str(project / "orion.py"), "--headless"], environ)
    return 0  # pragma: no cover - execve replaces this process or raises


if __name__ == "__main__":
    raise SystemExit(main())
