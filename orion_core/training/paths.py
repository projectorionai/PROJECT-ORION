"""
Where training data lives, and the rule that it never lives somewhere public.

A training set is a copy of the owner's conversations with ORION, redacted but
still his. So:

* it is never written inside the project folder (a source checkout is a git
  repository whose history is published), and
* never inside a cloud-synced folder (OneDrive uploads whatever lands in it).

The default is ``<config dir>/training``. In the installed app the config
directory is ``~/ORION/config``, which is neither. A source checkout keeps its
config inside the project folder - which on the owner's PC is inside OneDrive -
so there the default falls back to ``~/ORION/training`` instead of silently
putting a dataset in the cloud. ``ORION_TRAINING_DIR`` overrides both; it is
checked the same way.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..constants import BASE_DIR, CONFIG_DIR

ENV_TRAINING_DIR = "ORION_TRAINING_DIR"


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _synced_folder(path: Path) -> str:
    """The name of the cloud-synced folder *path* sits in, or ''."""
    parts = [p.casefold() for p in path.resolve().parts]
    for part in parts:
        if part.startswith("onedrive"):
            return "OneDrive"
        if part in {"dropbox", "google drive", "googledrive", "icloud drive", "iclouddrive"}:
            return part.title()
    for key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        root = os.environ.get(key, "").strip()
        if root and _is_inside(path, Path(root)):
            return "OneDrive"
    return ""


def location_problem(path: Path | str, *, project_dir: Path | None = None) -> str:
    """Why *path* is not an acceptable place for private training data, or ''."""
    target = Path(path)
    project = Path(project_dir) if project_dir is not None else Path(BASE_DIR)
    if _is_inside(target, project):
        return f"it is inside the project folder ({project})"
    synced = _synced_folder(target)
    if synced:
        return f"it is inside a {synced} folder, which uploads what is written there"
    return ""


def training_root(config_dir: Path | str | None = None, *, environ: dict | None = None,
                  home: Path | None = None) -> Path:
    """The directory datasets, adapters and receipts go under."""
    env = os.environ if environ is None else environ
    override = str(env.get(ENV_TRAINING_DIR, "") or "").strip()
    if override:
        return Path(override).expanduser()
    config = Path(config_dir) if config_dir is not None else Path(CONFIG_DIR)
    preferred = config / "training"
    if not location_problem(preferred):
        return preferred
    return (home or Path.home()) / "ORION" / "training"


def require_private_location(path: Path | str, *, allow: bool = False) -> Path:
    """*path* itself, or ``ValueError`` saying why it must not hold private data."""
    problem = location_problem(path)
    if problem and not allow:
        raise ValueError(
            f"refusing to write private training data to {path}: {problem}. Choose a folder "
            f"outside the project and outside any synced folder (for example "
            f"{Path.home() / 'ORION' / 'training'}) with --out-dir or {ENV_TRAINING_DIR}.")
    return Path(path)


__all__ = ["ENV_TRAINING_DIR", "location_problem", "require_private_location", "training_root"]
