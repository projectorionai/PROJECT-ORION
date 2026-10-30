"""Resolve an already installed Vosk model without its automatic downloader."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def cached_vosk_model(vosk: Any, language: str = "en-us") -> str:
    """An explicit model or a complete cached language model; never download.

    Vosk's Model(lang=...) looks up a remote catalogue and downloads a model
    when its cache is empty. Passing a verified local path avoids that path.
    """
    def valid(path: Path) -> bool:
        return (path / "am" / "final.mdl").is_file() and (path / "conf" / "mfcc.conf").is_file()

    configured = os.getenv("ORION_VOSK_MODEL", "").strip().strip('"')
    if configured:
        path = Path(os.path.expandvars(os.path.expanduser(configured)))
        if valid(path):
            return str(path)
        raise FileNotFoundError("ORION_VOSK_MODEL does not point to a complete local Vosk model")
    pattern = re.compile(r"^vosk-model(?:-small)?-" + re.escape(language) + r"(?:-|$)", re.IGNORECASE)
    for value in getattr(vosk, "MODEL_DIRS", ()):
        if not value:
            continue
        try:
            root = Path(value)
            for path in sorted(root.iterdir(), key=lambda item: ("-small-" not in item.name, item.name)):
                if pattern.match(path.name) and valid(path):
                    return str(path)
        except OSError:
            continue
    raise FileNotFoundError("No cached Vosk English model; install a model and set ORION_VOSK_MODEL")
