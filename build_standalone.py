"""
Build ORION as a TRUE standalone application — ORION.exe IS the process.

    "ORION still runs as a python 3.13 — make him solely his own application."

Unlike build_exe.py (a tiny launcher that spawns pythonw, so the running process
— and thus the taskbar icon — is still Python), this freezes the whole app with
PyInstaller so ``ORION.exe`` runs it IN-PROCESS. Windows then shows ORION's own
process and icon: no python.exe, no pythonw.exe, no console.

Output: ``dist/ORION/ORION.exe`` (a folder build — ``--onedir``, which is far
more reliable than a single-file bundle for Qt/QtWebEngine and large native
wheels; the folder is the "installed" app).

Deliberate choices for a build that actually WORKS:

  * ONEDIR, not onefile — onefile unpacks to a temp dir on every launch, which
    breaks QtWebEngine's resource lookup and doubles start-up time.
  * The two HEAVIEST optional libraries — torch and mediapipe (emotion/speaker
    embeddings and hand-gesture control) — are EXCLUDED. They add ~2 GB and are
    the usual cause of a failed freeze, and every code path that touches them is
    already guarded (they degrade gracefully). The result is a lean, working
    standalone with voice, the GUI, agents, research, chess, vision-OCR and the
    rest; the CV extras can be folded in later once the base is proven.
  * ``--collect-all`` for the packages whose data files PyInstaller's static
    analysis misses (google-genai protos, cv2, vosk, the PyQt6 WebEngine
    payload), so the frozen app finds them at runtime.

Freezing a GUI app is iterative: the first run on the target machine usually
surfaces one or two more hidden imports or data files. Run ORION.exe, and if it
reports a missing module, add it to HIDDEN_IMPORTS and rebuild.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.prepare_public_source import PUBLIC_CONFIG_FILES

BASE = Path(__file__).resolve().parent
ENTRY = BASE / "orion.py"
ICON = BASE / "assets" / "orion.ico"

# Packages whose data/submodules PyInstaller's analysis misses — pull them whole.
COLLECT_ALL = ["google", "cv2", "vosk", "sounddevice", "chess", "aiohttp",
               # rapidocr keeps its recognisers as 13 MB of .onnx plus a
               # handful of config.yaml files INSIDE the package. Static
               # analysis sees the import and bundles the code, but data
               # files are invisible to it, so the built ORION had
               # onnxruntime and no models to feed it — every screen read
               # fell over in the shipped app while working perfectly from
               # source. It is also the only reader that sees light text on
               # a dark background, which is most of a desktop.
               "rapidocr_onnxruntime",
               # Mark XXXI sight. Windows OCR is reached through winrt
               # projection modules imported BY NAME (ocr_engine tries winrt,
               # then winsdk) — invisible to the analysis, and each is a .pyd.
               # uiautomation ships its UIAutomationClient DLLs in a bin/
               # folder; dxcam and cv2_enumerate_cameras are small but loaded
               # lazily, only when a game is captured or a camera listed.
               "winrt", "uiautomation", "dxcam", "cv2_enumerate_cameras",
               # Imported inside the page reader, only when a site refuses a
               # plain request; it carries its own libcurl build.
               "curl_cffi",
               # The shipped json_validator plugin imports jsonschema, which
               # nothing in orion_core does, so the analysis never saw it: the
               # exe quarantined the plugin as soon as it was delivered. Its
               # metaschemas are data files in jsonschema_specifications.
               "jsonschema", "jsonschema_specifications",
               # The sentence encoder's tokenizer (semantic.py) — a Rust
               # extension imported inside a function. The model itself is
               # downloaded on first use into the app's config, not bundled.
               "tokenizers"]
# Modules imported lazily/by string that the analysis cannot see.
HIDDEN_IMPORTS = [
    "orion_core", "win32com", "win32com.client", "win32timezone",
    # Public plugins are loaded from files after startup, so the freezer cannot
    # discover their core helpers or optional classifier from the entry point.
    "orion_core.local_tool_outcomes", "sklearn.ensemble",
    "sklearn.model_selection", "sklearn.metrics",
    "pyttsx3.drivers", "pyttsx3.drivers.sapi5", "PIL", "numpy",
    # Both are imported inside functions rather than at module top level:
    # pytesseract only when OCR runs, httpx only when a call is placed.
    "pytesseract", "httpx",
    # uiautomation drives UI Automation through comtypes, which builds its
    # COM wrappers at runtime. (orion_core's own lazily imported modules are
    # covered by --collect-submodules below.)
    "comtypes", "comtypes.client", "comtypes.stream",
    # GPU meters: imported inside gpu_stats._ensure_init only.
    "pynvml",
]
# The heavy optionals we deliberately leave out to keep the build lean + working.
EXCLUDES = ["torch", "torchvision", "torchaudio", "mediapipe", "resemblyzer"]
# Data folders the app reads at runtime, as (src, dest-in-bundle).
DATAS = [
    ("assets", "assets"),
    # The Forge's isolated conformance child cannot import the contract module
    # from PyInstaller's PYZ archive. Keep the exact source beside the frozen
    # module so sandbox.probe_source() can ship it into the child process.
    ("orion_core/forge_contract.py", "orion_core"),
]


def _ensure_pyinstaller() -> bool:
    try:
        import PyInstaller  # noqa: F401
        return True
    except Exception:
        print("PyInstaller not found — installing it…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("Could not install PyInstaller:\n" + r.stderr[-1200:])
            return False
        return True


def _seed_config(app_dir: Path) -> None:
    """Seed public defaults only; credentials and personal state stay local."""
    import shutil
    seeded = 0
    for name in sorted(PUBLIC_CONFIG_FILES):
        source = BASE / name
        if not source.is_file():
            continue
        # Reject redirected source paths instead of following a local symlink.
        if source.resolve() != BASE.resolve() / name:
            raise ValueError(f"Public default is a redirected path: {name}")
        target = app_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        seeded += 1
    print(f"  seeded {seeded} public defaults; no personal configuration copied.")


def _has_private_config(app_dir: Path) -> bool:
    """Never let --noconfirm erase or repackage a personalised installation."""
    config = app_dir / "config"
    if app_dir.is_symlink() or config.is_symlink() or config.resolve() != app_dir.resolve() / "config":
        return True
    if not config.exists():
        return False
    return any(p.is_symlink() or (p.is_file() and
               p.relative_to(app_dir).as_posix() not in PUBLIC_CONFIG_FILES)
               for p in config.rglob("*"))


def _ensure_icon() -> None:
    if ICON.exists():
        return
    try:
        from orion_core import desktop_app
        desktop_app.ensure_icon()
    except Exception:
        pass


def _version_tuple() -> tuple[int, int, int, int]:
    """(major, minor, patch, 0) read from orion_core.__version__ without importing
    the whole package — the build must not drift from the codename bump."""
    import ast
    try:
        src = (BASE / "orion_core" / "__init__.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", "") == "__version__" for t in node.targets)
                    and isinstance(node.value, ast.Constant)):
                parts = [int(p) for p in str(node.value.value).split(".")[:3]]
                while len(parts) < 3:
                    parts.append(0)
                return (parts[0], parts[1], parts[2], 0)
    except Exception:
        pass
    return (1, 0, 0, 0)


def _write_version_file() -> "Path | None":
    """Emit a Windows VERSIONINFO resource so the frozen ORION.exe reports itself
    as O.R.I.O.N. — not the blank/Python identity that shows in Task Manager and
    file properties. Returns the path to pass to --version-file, or None."""
    v = _version_tuple()
    vs = f"{v[0]}.{v[1]}.{v[2]}"
    text = f"""# Auto-generated by build_standalone.py — do not edit by hand.
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={v}, prodvers={v},
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Project ORION'),
      StringStruct('FileDescription', 'O.R.I.O.N. — your local AI assistant'),
      StringStruct('FileVersion', '{vs}'),
      StringStruct('InternalName', 'ORION'),
      StringStruct('LegalCopyright', 'O.R.I.O.N.'),
      StringStruct('OriginalFilename', 'ORION.exe'),
      StringStruct('ProductName', 'O.R.I.O.N.'),
      StringStruct('ProductVersion', '{vs}'),
    ])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])]),
  ],
)
"""
    out = BASE / "build" / "_orion_version.txt"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return out
    except Exception:
        return None


def build() -> int:
    if _has_private_config(BASE / "dist" / "ORION"):
        print("Build stopped: dist/ORION contains personal configuration. "
              "Build from a clean source export in a separate directory to "
              "preserve the installed app and keep private data out of releases.")
        return 1
    if not ENTRY.is_file():
        print(f"Entry point missing: {ENTRY}")
        return 1
    if not _ensure_pyinstaller():
        return 1
    _ensure_icon()

    sep = ";" if sys.platform == "win32" else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", "ORION",
        "--onedir",
        "--windowed",                       # no console window
        # Dynamic core helpers and optional fallback modules must remain
        # available even when the entry-point import graph cannot see them.
        "--collect-submodules", "orion_core",
        "--distpath", str(BASE / "dist"),
        "--workpath", str(BASE / "build" / "_standalone_work"),
        "--specpath", str(BASE / "build" / "_standalone_spec"),
    ]
    if ICON.exists():
        cmd += ["--icon", str(ICON)]
    version_file = _write_version_file()
    if version_file is not None:
        cmd += ["--version-file", str(version_file)]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for mod in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", mod]
    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    for src, dest in DATAS:
        if (BASE / src).exists():
            cmd += ["--add-data", f"{BASE / src}{sep}{dest}"]
    # The reviewed plugins also go INSIDE the bundle: the app's data folder is
    # reached through a pointer, so copies beside the exe were never seen.
    # plugin_registry.seed_shipped delivers them from here at startup.
    for name in sorted(PUBLIC_CONFIG_FILES):
        if name.startswith("config/custom_tools/") and not name.endswith("__init__.py") \
                and (BASE / name).is_file():
            cmd += ["--add-data", f"{BASE / name}{sep}shipped_plugins"]
    cmd.append(str(ENTRY))

    print("Building the standalone app (this takes several minutes and produces "
          "a large folder)…\n" + " ".join(cmd) + "\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\nPyInstaller failed (exit {result.returncode}). The most common "
              "cause is a missing hidden import — add it to HIDDEN_IMPORTS and "
              "rebuild.")
        return result.returncode
    exe = BASE / "dist" / "ORION" / "ORION.exe"
    if exe.exists():
        _seed_config(exe.parent)
        print(f"\nDone. Your standalone app is:\n  {exe}\n"
              "Run it directly, or re-run installer.install() so the Desktop / "
              "Start-menu shortcuts point at this instead of the launcher.")
        return 0
    print("\nBuild finished but ORION.exe was not found — check the output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(build())
