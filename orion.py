"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  O.R.I.O.N.  Mark VIII  —  Open Resolution Intelligence Overt Network        ║
║  Launcher  |  python orion.py  |  Windows / Linux                            ║
║                                                                              ║
║  The Mark VIII architecture lives in the orion_core/ package:                ║
║  modular services and managers (audio, vision, agents, memory, providers,   ║
║  Outlook, Notion, briefing, dual-window GUI).  This file is intentionally   ║
║  a thin shim so `python orion.py` keeps working exactly as it always has.   ║
║                                                                              ║
║  The Mark VII single-file build is archived at legacy/orion_mark7_monolith.py║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys


def _headless_requested() -> bool:
    """Headless (cloud/server) mode: no GUI, brain + remote uplink only."""
    if os.getenv("ORION_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return any(arg in {"--headless", "--server", "--cloud"} for arg in sys.argv[1:])


if _headless_requested():
    # Lighter gate: the cloud node needs no GUI, audio or screen-grab stack.
    try:
        import aiohttp          # noqa: F401
        import qasync           # noqa: F401
        from PyQt6 import QtCore  # noqa: F401
    except Exception as import_error:  # pragma: no cover - env dependent
        print("O.R.I.O.N. headless node cannot initialise. Missing dependency:")
        print(f"  {import_error}")
        print("Install: pip install -r deploy/requirements-server.txt")
        raise SystemExit(1)
    from orion_core.server import main
else:
    # Full desktop gate: fail with an actionable message instead of a stack trace.
    try:
        import aiohttp          # noqa: F401
        import mss              # noqa: F401
        import psutil           # noqa: F401
        import qasync           # noqa: F401
        import sounddevice      # noqa: F401
        from google import genai            # noqa: F401
        from PIL import Image               # noqa: F401
        from PyQt6 import QtWidgets         # noqa: F401
    except Exception as import_error:
        print("O.R.I.O.N. cannot initialise. Missing runtime dependency:")
        print(f"  {import_error}")
        print(
            "Install: pip install PyQt6 qasync aiohttp sounddevice "
            "google-genai pillow mss psutil\n"
            "Optional (recommended): pip install pyttsx3 pywin32 pytesseract vosk pypdf\n"
            "Cloud/headless node instead? Run: python orion.py --headless"
        )
        raise SystemExit(1)
    from orion_core.app import main

if __name__ == "__main__":
    main()
