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
  * Torch remains excluded from the default build. MediaPipe is included when
    installed so the orb's two-hand camera control also works in ORION.exe.
    A core-only install without MediaPipe keeps the mouse controls available.
  * ``--collect-all`` for the packages whose data files PyInstaller's static
    analysis misses (google-genai protos, cv2, vosk, the PyQt6 WebEngine
    payload), so the frozen app finds them at runtime.

Freezing a GUI app is iterative: the first run on the target machine usually
surfaces one or two more hidden imports or data files. Run ORION.exe, and if it
reports a missing module, add it to HIDDEN_IMPORTS and rebuild.
"""

from __future__ import annotations

import json
import ast
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.prepare_public_source import PUBLIC_CONFIG_FILES, RETIRED_PLUGINS

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
               "tokenizers",
               # The IANA zone files are package DATA zoneinfo reads at run
               # time (Windows has no tz database): world time + Europe/London.
               "tzdata",
               # Offline Whisper (phone voice input, video analysis). Its code
               # reached the PYZ but its assets/silero_vad_v6.onnx did not, and
               # every transcription runs with vad_filter=True: the 8 Oct exe
               # had no _internal/faster_whisper at all, so each one failed.
               "faster_whisper",
               # Whisper on the NVIDIA GPU (accelerator.CUDA_POLICY): the
               # CTranslate2 wheel ships cudnn64_9.dll, loaded at run time,
               # so the analysis left it out and the 40.5 exe could only run
               # on the CPU. cuBLAS comes from %USERPROFILE%\ORION\gpu-runtime.
               "ctranslate2",
               # Optional speech-group WebRTC canceller uses ctypes to load
               # its files/webrtc DLL; static import analysis omits that data.
               "aec_audio_processing",
               # ORION's offline voice (voice_kokoro). kokoro_onnx reads its
               # config.json vocabulary on import; espeakng_loader carries
               # espeak-ng.dll and its 18 MB espeak-ng-data folder, found
               # beside its own __init__ and loaded through ctypes; phonemizer
               # drives espeak-ng. All three are data the analysis cannot see.
               "kokoro_onnx", "espeakng_loader", "phonemizer",
               # The Forge verifies new tools with this bundle (--forge-verify,
               # forge_runtime). A generated test with test_ functions runs
               # under pytest, which the exe did not carry.
               "pytest", "_pytest", "pluggy", "iniconfig"]
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
    "pytesseract", "httpx", "playwright.sync_api",
    # uiautomation drives UI Automation through comtypes, which builds its
    # COM wrappers at runtime. (orion_core's own lazily imported modules are
    # covered by --collect-submodules below.)
    "comtypes", "comtypes.client", "comtypes.stream",
    # A robot on USB (robot_link "serial:" addresses), imported on connect.
    "serial", "serial.tools.list_ports",
    # The shell command channel (windows_shell: QLocalServer/QLocalSocket) is
    # the only user of QtNetwork, imported inside a function so the refused
    # second launch parses its argv without Qt. Its jump list declares its
    # COM interfaces by hand (no comtypes.gen), so comtypes above suffices.
    "PyQt6.QtNetwork",
    # GPU meters: imported inside gpu_stats._ensure_init only.
    "pynvml",
    # Mark XXXVII.1 Windows presence. Toasts with buttons are shown through
    # the WinRT projection, imported by name inside WinrtToastBackend on a
    # worker thread (the same way media_session reaches winrt.windows.media
    # .control), so the analysis never sees them. Each is a package of .pyd
    # files; the dom/foundation ones are what the notifications module pulls
    # in at import. COLLECT_ALL carries "winrt" as a whole; these keep the
    # notification path correct even if that collection is ever narrowed.
    "winrt.windows.ui.notifications", "winrt.windows.data.xml.dom",
    "winrt.windows.foundation", "winrt.windows.foundation.collections",
    "winrt.windows.media.control",
]
# Keep optional training libraries out of the normal app bundle. Upstream
# optional imports (for example CTranslate2's model converters) otherwise pull
# the trainer's Transformers/datasets stack into the freeze when the build
# environment also has it installed. ORION's planner uses Ollama and its
# separate trainer process; speech, OCR, semantic recall and voiceprints run
# on CTranslate2/ONNX, so none needs these training packages in the executable.
# Keep matplotlib: MediaPipe imports its drawing helpers at package startup.
EXCLUDES = ["torch", "torchvision", "torchaudio", "resemblyzer",
            "transformers", "datasets", "pyarrow", "peft", "trl",
            "accelerate", "bitsandbytes"]
try:
    import importlib.util
    if importlib.util.find_spec("mediapipe") is not None:
        COLLECT_ALL.append("mediapipe")
except (ImportError, ValueError):
    pass
try:
    import importlib.util
    # The adapter is prepared without downloading an engine or voice. If an
    # owner supplies piper-tts, keep its phoneme data and native libraries.
    if importlib.util.find_spec("piper") is not None:
        COLLECT_ALL.append("piper")
except (ImportError, ValueError):
    pass
# Explicit training build: package the installed CUDA-enabled Torch wheel.
# The default lean build retains all chess playing/inference capabilities.
if os.getenv("ORION_BUILD_CHESS_TRAINING") == "1":
    EXCLUDES.remove("torch")
    COLLECT_ALL.append("torch")
HIDDEN_IMPORTS.append("berserk")
# Packages that read their OWN installed metadata (importlib.metadata) when
# imported. PyInstaller bundles their code but not their dist-info unless
# asked, so they fail frozen with "No package metadata was found": berserk
# (the Lichess client) did exactly that in the 8 Oct exe, so the Lichess bot
# could not load in the installed app (installed-app parity check, 11 Oct).
COPY_METADATA = ["berserk"]
# Data folders the app reads at runtime, as (src, dest-in-bundle).
DATAS = [
    ("assets", "assets"),
    # The Forge's isolated conformance child cannot import the contract module
    # from PyInstaller's PYZ archive. Keep the exact source beside the frozen
    # module so sandbox.probe_source() can ship it into the child process.
    ("orion_core/forge_contract.py", "orion_core"),
    ("orion_core/mcp_telemetry.py", "orion_core"),
    ("orion_core/mcp_network.py", "orion_core"),
    ("orion_core/mcp_world.py", "orion_core"),
    ("orion_core/mcp_desktop.py", "orion_core"),
]
#: Where the offline voice's model goes in the bundle: voice_kokoro looks in
#: RESOURCE_DIR/assets/models/kokoro-v1.0 (_internal in the frozen app).
KOKORO_DEST = "assets/models/kokoro-v1.0"


def _kokoro_candidates(explicit: "str | None") -> "list[Path]":
    """Where the Kokoro model (354 MB, never in the repository) may be: the
    --kokoro-dir folder alone when given; else this checkout's config, then
    the config the installed app points at (dist/ORION/config_location.txt)."""
    if explicit:
        return [Path(explicit)]
    candidates = []
    try:
        from orion_core.constants import CONFIG_DIR
        candidates.append(Path(CONFIG_DIR) / "models" / "kokoro-v1.0")
    except Exception:
        pass
    pointer = BASE / "dist" / "ORION" / "config_location.txt"
    try:
        if pointer.is_file():
            candidates.append(Path(pointer.read_text(encoding="utf-8").strip())
                              / "models" / "kokoro-v1.0")
    except (OSError, ValueError):
        pass
    return candidates


def _kokoro_datas(explicit: "str | None" = None) -> "tuple[list[tuple[Path, str]], str]":
    """(model files to add, what was decided). A folder is used only when
    every file matches voice_kokoro.MODEL's pinned size and SHA-256."""
    from orion_core.voice_kokoro import MODEL, verify
    rejected = []
    for folder in _kokoro_candidates(explicit):
        if not folder.is_dir():
            rejected.append(f"{folder}: not a folder")
            continue
        problems = verify(folder, full=True)
        if problems:
            rejected.append(f"{folder}: {'; '.join(problems)}")
            continue
        return ([(folder / name, KOKORO_DEST) for name in MODEL.files],
                f"bundling the Kokoro offline voice from {folder} (sizes and SHA-256 checked)")
    return [], ("the Kokoro offline voice model is NOT bundled, so the installed app "
                "would speak offline in the Windows voice; pass --kokoro-dir <folder with "
                "kokoro-v1.0.onnx and voices-v1.0.bin>"
                + (f" ({'; '.join(rejected)})" if rejected else ""))


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


def _write_build_info() -> "Path | None":
    """Record when this build was made and the newest source it contains, so
    the frozen app can tell when its project has moved on without a rebuild
    (orion_core/build_freshness.py). Returns the path to bundle, or None."""
    newest, count = 0.0, 0
    for path in (BASE / "orion_core").rglob("*.py"):
        try:
            newest = max(newest, path.stat().st_mtime)
            count += 1
        except OSError:
            continue
    version = ".".join(str(part) for part in _version_tuple()[:3])
    out = BASE / "build" / "build_info.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"built_at": time.time(), "source_newest": newest,
                                   "source_files": count, "version": version}),
                       encoding="utf-8")
        return out
    except Exception:
        return None


#: Web-engine locale packs ORION's English UI uses; Chromium falls back to en-US.
KEEP_WEBENGINE_LOCALES = {"en-GB.pak", "en-US.pak"}


def _prune_unused_qt_data(internal: Path) -> None:
    """Drop Qt data an English-only ORION never loads (measured on the 25 Sep
    build: 43.6 MB of web-engine locale packs for 53 languages, 9.3 MB of Qt
    .qm translations - ORION installs no QTranslator). Only data files, never
    a library, so nothing that could be imported goes missing."""
    translations = internal / "PyQt6" / "Qt6" / "translations"
    if not translations.is_dir():
        return
    freed = 0
    for path in list(translations.rglob("*")):
        if not path.is_file():
            continue
        is_locale = path.parent.name == "qtwebengine_locales" and path.suffix == ".pak"
        if (is_locale and path.name not in KEEP_WEBENGINE_LOCALES) or path.suffix == ".qm":
            freed += path.stat().st_size
            path.unlink()
    print(f"Pruned unused Qt translations and locale packs: {freed / 2**20:.1f} MB")


def _build_environment() -> dict[str, str]:
    """Resolve Windows system DLLs before unrelated toolchain copies.

    For example, Poppler's ICU exposes versioned symbols, while Qt expects
    Windows' unversioned ICU API. Collecting Poppler's copy at the bundle root
    prevents QtCore from loading even though PyQt works in the source runtime.
    """
    environment = dict(os.environ)
    if sys.platform == "win32":
        system_root = (environment.get("SYSTEMROOT") or environment.get("SystemRoot")
                       or environment.get("WINDIR"))
        if system_root:
            system32 = Path(system_root) / "System32"
            if system32.is_dir():
                import _ssl
                python_dlls = Path(_ssl.__file__).resolve().parent
                environment["PATH"] = os.pathsep.join(filter(None, (
                    str(system32), str(python_dlls), environment.get("PATH", ""))))
    return environment


def _ssl_runtime_inputs(*, ssl_file: Path | None = None,
                        extension_file: Path | None = None) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Carry the active interpreter's native SSL pair and adjacent trust data.

    A toolchain on PATH can expose identically named, incompatible OpenSSL
    DLLs. Explicit Analysis inputs take precedence over discovered dependencies.
    Some Python runtimes also require a PEM referenced by their ssl.py; retain
    that runtime trust policy rather than replacing it with another SSL module.
    """
    import ssl
    import _ssl
    module_path = Path(ssl_file or ssl.__file__).resolve()
    extension = Path(extension_file or _ssl.__file__).resolve()
    binaries: list[tuple[Path, str]] = []
    if sys.platform == "win32":
        for pattern in ("libssl*.dll", "libcrypto*.dll"):
            choices = sorted(path for path in extension.parent.glob(pattern) if path.is_file())
            if len(choices) != 1:
                raise RuntimeError(f"Expected one interpreter SSL dependency {pattern} beside {extension}; found {len(choices)}")
            binaries.append((choices[0], "."))
    datas: list[tuple[Path, str]] = []
    # Collect only adjacent PEM literals used by a join(dirname(__file__), name)
    # expression, not unrelated strings, docs or an entire interpreter folder.
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "join" or len(node.args) != 2):
            continue
        parent, name = node.args
        if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "dirname" and len(parent.args) == 1
                and isinstance(parent.args[0], ast.Name) and parent.args[0].id == "__file__"
                and isinstance(name, ast.Constant) and isinstance(name.value, str)
                and name.value.endswith(".pem") and Path(name.value).name == name.value):
            names.add(name.value)
    for name in sorted(names):
        path = module_path.parent / name
        if not path.is_file():
            raise RuntimeError(f"The build interpreter's ssl.py requires missing adjacent data: {path}")
        datas.append((path, "."))
    return binaries, datas


def _verify_ssl_runtime_provenance(internal: Path, binaries: list[tuple[Path, str]],
                                   datas: list[tuple[Path, str]]) -> dict:
    """Reject a build which resolved pinned SSL names from another toolchain."""
    if not binaries and not datas:
        return {"ok": True, "files": []}
    toc_path = BASE / "build" / "_standalone_work" / "ORION" / "Analysis-00.toc"
    toc = ast.literal_eval(toc_path.read_text(encoding="utf-8"))
    entries: list[tuple[str, str, str]] = []
    def visit(value):
        if (isinstance(value, (list, tuple)) and len(value) == 3
                and all(isinstance(item, str) for item in value)
                and value[2] in {"BINARY", "EXTENSION", "DATA"}):
            entries.append(tuple(value))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(toc)
    files = []
    for source, destination in [*binaries, *datas]:
        relative = Path(destination) / source.name
        matches = [Path(origin).resolve() for target, origin, _kind in entries
                   if Path(target) == relative]
        if not matches or any(origin != source.resolve() for origin in matches):
            raise RuntimeError(f"SSL runtime provenance mismatch for {relative}: expected {source}")
        output = internal / relative
        if not output.is_file():
            raise RuntimeError(f"SSL runtime payload is absent: {output}")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
        # PyInstaller may rewrite PE headers; trust-data bytes must be exact.
        if (source, destination) in datas and source_hash != output_hash:
            raise RuntimeError(f"SSL runtime trust data changed: {relative}")
        files.append({"path": str(relative), "source": str(source),
                      "source_sha256": source_hash, "output_sha256": output_hash})
    receipt = {"ok": True, "analysis": str(toc_path), "files": files}
    (BASE / "build" / "ssl-runtime-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def _check_installed_parity(exe: Path) -> int:
    """The final gate uses the build's interpreter and bounds its entire tree."""
    from tools.check_installed_build import run_bounded

    command = [sys.executable, str(BASE / "tools" / "check_installed_build.py"),
               str(exe), "--source", str(BASE),
               "--json", str(exe.parent / "parity_report.json")]
    print("\nChecking installed-app parity before marking the build complete…")
    result = run_bounded(command, cwd=BASE, env=_build_environment(), timeout=600.0)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.error or result.timed_out or result.returncode != 0:
        reason = result.error or ("timed out after 600s (process tree killed)"
                                  if result.timed_out else f"exit code {result.returncode}")
        print(f"\nInstalled-app parity failed: {reason}. The build is not ready to install.")
        return result.returncode if result.returncode and result.returncode > 0 else 1
    return 0


def _plugin_of(path: str) -> str:
    """The plugin a shipped file belongs to: config/custom_tools/<name>_tool.py
    or <name>.plugin.json -> <name>."""
    file = path.rsplit("/", 1)[-1]
    for suffix in ("_tool.py", ".plugin.json"):
        if file.endswith(suffix):
            return file[: -len(suffix)]
    return file


def build(argv: "list[str] | None" = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build dist/ORION/ORION.exe.")
    parser.add_argument("--skip-parity-check", action="store_true",
                        help="do not run tools/check_installed_build.py on the new build")
    parser.add_argument("--kokoro-dir", default=None,
                        help="folder holding the Kokoro offline voice model to bundle "
                             "(default: CONFIG_DIR/models/kokoro-v1.0)")
    options = parser.parse_args([] if argv is None else list(argv))
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
    try:
        ssl_binaries, ssl_datas = _ssl_runtime_inputs()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Build stopped: SSL runtime inputs could not be verified: {exc}")
        return 1
    _ensure_icon()

    sep = ";" if sys.platform == "win32" else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", "ORION",
        "--onedir",
        # PyInstaller packs every DLL with UPX whenever upx is on PATH. A
        # packed DLL is unpacked into private memory on every load (no shared
        # pages, slower cold start) and Qt/onnxruntime DLLs break under it.
        # Not on PATH here (29 Oct 2026), so this changes nothing today; it
        # keeps a later install of UPX from silently changing the build.
        "--noupx",
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
    for path, destination in ssl_binaries:
        cmd += ["--add-binary", f"{path}{sep}{destination}"]
    for path, destination in ssl_datas:
        cmd += ["--add-data", f"{path}{sep}{destination}"]
    version_file = _write_version_file()
    if version_file is not None:
        cmd += ["--version-file", str(version_file)]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for mod in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", mod]
    for dist in COPY_METADATA:
        cmd += ["--copy-metadata", dist]
    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    for src, dest in DATAS:
        if (BASE / src).exists():
            cmd += ["--add-data", f"{BASE / src}{sep}{dest}"]
    kokoro_files, kokoro_note = _kokoro_datas(options.kokoro_dir)
    print(("" if kokoro_files else "WARNING: ") + kokoro_note)
    for path, dest in kokoro_files:
        cmd += ["--add-data", f"{path}{sep}{dest}"]
    build_info = _write_build_info()
    if build_info is not None:
        cmd += ["--add-data", f"{build_info}{sep}."]
    # The reviewed plugins also go INSIDE the bundle: the app's data folder is
    # reached through a pointer, so copies beside the exe were never seen.
    # plugin_registry.seed_shipped delivers them from here at startup.
    for name in sorted(PUBLIC_CONFIG_FILES):
        if name.startswith("config/custom_tools/") and not name.endswith("__init__.py") \
                and _plugin_of(name) not in RETIRED_PLUGINS and (BASE / name).is_file():
            cmd += ["--add-data", f"{BASE / name}{sep}shipped_plugins"]
    cmd.append(str(ENTRY))

    print("Building the standalone app (this takes several minutes and produces "
          "a large folder)…\n" + " ".join(cmd) + "\n")
    result = subprocess.run(cmd, text=True, env=_build_environment())
    if result.returncode != 0:
        print(f"\nPyInstaller failed (exit {result.returncode}). The most common "
              "cause is a missing hidden import — add it to HIDDEN_IMPORTS and "
              "rebuild.")
        return result.returncode
    exe = BASE / "dist" / "ORION" / "ORION.exe"
    if exe.exists():
        try:
            _verify_ssl_runtime_provenance(exe.parent / "_internal", ssl_binaries, ssl_datas)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Build stopped: SSL runtime provenance verification failed: {exc}")
            return 1
        _prune_unused_qt_data(exe.parent / "_internal")
        _seed_config(exe.parent)
        if options.skip_parity_check:
            print("\nInstalled-app parity check SKIPPED (--skip-parity-check): "
                  "this build is unverified.")
        else:
            parity_code = _check_installed_parity(exe)
            if parity_code:
                return parity_code
        print(f"\nDone. Your standalone app is:\n  {exe}\n"
              "Run it directly, or re-run installer.install() so the Desktop / "
              "Start-menu shortcuts point at this instead of the launcher.")
        return 0
    print("\nBuild finished but ORION.exe was not found — check the output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(build(sys.argv[1:]))
