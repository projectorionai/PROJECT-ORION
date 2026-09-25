"""
Making ORION a real desktop application, not a script you run in a terminal.

  "I want ORION to be able to be in the background apps in the taskbar, also can
   we make ORION a full on application that I can put in my taskbar and the
   desktop since he's that good now"

Three things turn a Python script into an application Windows treats as its own:

  1. **An identity.** Without an explicit AppUserModelID every Python process
     shares one taskbar identity — "Python", generic snake icon, all grouped
     together. Setting ORION's own ID makes the taskbar show HIM: his icon, his
     name, his own group, and — crucially — his pinned shortcut and his running
     window become the same taskbar button instead of two.
  3. **A face.** An icon, set on the process and the windows, so he is
     recognisable in the taskbar, the Alt-Tab list and the title bar.
  2. **Somewhere to launch from.** Shortcuts on the Desktop and in the Start
     menu, launched windowless (pythonw), which is what makes him pinnable to
     the taskbar and openable like any other app.

Everything here degrades to a no-op off Windows or when a dependency is
missing — a desktop-integration nicety must never be able to stop ORION
starting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .constants import BASE_DIR

#: A STABLE identity string. It must never change once shortcuts exist, or
#: Windows treats a new version as a different app and the taskbar pin goes
#: dead. Versionless on purpose — "Mark XXI" belongs in the title, not here.
APP_USER_MODEL_ID = "ProjectORION.Assistant"

#: Where the generated icon lives.
ICON_PATH = BASE_DIR / "assets" / "orion.ico"

#: The name shown under the shortcut.
SHORTCUT_NAME = "ORION"


# ── the icon ─────────────────────────────────────────────────────────────────

def ensure_icon() -> Path | None:
    """Generate ORION's app icon once, at several sizes. Returns the .ico path.

    Drawn rather than shipped: a committed binary is one more thing to license
    and to go missing, and ORION's look is well-defined enough to render — a
    crimson orb with a bright core on near-black, the same visual language as
    his overlay. Multi-size because Windows picks 16px for the taskbar and
    256px for large icons, and a single size scaled to both looks wrong at one.
    """
    if ICON_PATH.exists() and ICON_PATH.stat().st_size > 0:
        return ICON_PATH
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    try:
        ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
        layers = []
        for size in (256, 128, 64, 48, 32, 16):
            layers.append(_render_icon(Image, ImageDraw, size))
        layers[0].save(ICON_PATH, format="ICO",
                       sizes=[(im.width, im.height) for im in layers])
        return ICON_PATH
    except Exception:
        return None


def _render_icon(Image: "type", ImageDraw: "type", size: int):
    """One square RGBA icon: a crimson orb with a hot core on near-black."""
    scale = 4 if size >= 48 else 2            # supersample, then downscale
    big = size * scale
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = cy = big / 2

    # Rounded near-black tile so it reads as an app icon, not a floating dot.
    pad = big * 0.06
    radius = big * 0.22
    draw.rounded_rectangle([pad, pad, big - pad, big - pad],
                           radius=radius, fill=(14, 15, 18, 255))

    # The orb: concentric crimson rings, richest in the middle and fading at
    # the rim. Painting outer-to-inner (opaque outer, brighter inner) gives a
    # solid crimson disc with a glow rather than the washed-out pink a purely
    # additive fade produces.
    orb_r = big * 0.36
    rings = max(16, big // 6)
    for i in range(rings, 0, -1):
        t = i / rings                       # 1 at rim, →0 at centre
        r = orb_r * t
        # Deep crimson at the rim, warming toward the core.
        red = int(150 + 90 * (1.0 - t))
        grn = int(10 + 40 * (1.0 - t) ** 2)
        blu = int(26 + 40 * (1.0 - t) ** 2)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(min(255, red), grn, blu, 255))

    # A tight, hot core.
    core = orb_r * 0.28
    draw.ellipse([cx - core, cy - core, cx + core, cy + core],
                 fill=(255, 224, 232, 255))

    if scale != 1:
        img = img.resize((size, size), Image.LANCZOS)
    return img


# ── taskbar identity ─────────────────────────────────────────────────────────

def set_app_user_model_id() -> bool:
    """Give ORION his own taskbar identity. Must run BEFORE any window shows.

    Without this, ORION shares the generic "Python" taskbar group and icon,
    and a pinned shortcut never merges with the running window.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID)
        return True
    except Exception:
        return False


def apply_window_icon(app_or_window: "object") -> bool:
    """Set ORION's icon on the QApplication (and thus every window)."""
    icon_path = ensure_icon()
    if icon_path is None:
        return False
    try:
        from PyQt6.QtGui import QIcon
        icon = QIcon(str(icon_path))
        if icon.isNull():
            return False
        app_or_window.setWindowIcon(icon)
        return True
    except Exception:
        return False


# ── shortcuts ────────────────────────────────────────────────────────────────

#: The TRUE standalone build (build_standalone.py) — ORION.exe IS the process,
#: no python at all. Preferred over everything else when present.
STANDALONE_EXE = BASE_DIR / "dist" / "ORION" / "ORION.exe"
#: The built launcher executable (build_exe.py) — a real .exe, but it spawns
#: pythonw, so the running process is still Python. Fallback only.
EXE_PATH = BASE_DIR / "ORION.exe"


def _launch_target() -> tuple[str, str, str]:
    """(target, arguments, working-dir) for the shortcut, windowless.

    Preference order, best first:
      1. the STANDALONE build — its own process and icon, no python anywhere;
      2. the launcher ORION.exe (build_exe.py) — a real .exe, but spawns pythonw;
      3. orion.py under the windowless interpreter.
    A real executable is also what Windows pins to the taskbar cleanly (pinning
    a pythonw shortcut is the flaky path that produced "I can't pin the app").
    """
    if STANDALONE_EXE.exists():
        return str(STANDALONE_EXE), "", str(STANDALONE_EXE.parent)
    if EXE_PATH.exists():
        return str(EXE_PATH), "", str(BASE_DIR)
    interpreter = Path(sys.executable)
    windowed = interpreter.with_name(interpreter.name.replace("python", "pythonw"))
    exe = str(windowed if ("python" in interpreter.name.lower() and windowed.exists())
              else interpreter)
    return exe, f'"{BASE_DIR / "orion.py"}"', str(BASE_DIR)


def _make_shortcut(path: Path) -> bool:
    """Write one .lnk at *path* that carries ORION's taskbar IDENTITY.

    The critical part is the AppUserModelID property. A plain shortcut launches
    ORION.exe, which spawns the windowless interpreter; without a matching
    identity on the shortcut, Windows cannot connect the running window back to
    the pinned shortcut, so the taskbar button falls back to the interpreter's
    generic "Python" icon — exactly the "it still shows python" complaint. The
    running process sets ``SetCurrentProcessExplicitAppUserModelID`` to the SAME
    id (see set_app_user_model_id); stamping it on the shortcut too makes Windows
    treat the pin and the live window as one app, with ORION's icon.

    WScript.Shell cannot set that property, so this drives IShellLink +
    IPropertyStore directly, and falls back to the simple shortcut if the COM
    property store is unavailable.
    """
    target, arguments, workdir = _launch_target()
    icon = ensure_icon()
    try:
        import pythoncom
        from win32com.propsys import propsys, pscon
        from win32com.shell import shell

        link = pythoncom.CoCreateInstance(
            shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_IShellLink)
        link.SetPath(target)
        if arguments:
            link.SetArguments(arguments)
        link.SetWorkingDirectory(workdir)
        link.SetDescription("O.R.I.O.N. — your local AI assistant")
        if icon is not None:
            link.SetIconLocation(str(icon), 0)
        # THE fix: stamp ORION's stable identity onto the shortcut.
        store = link.QueryInterface(propsys.IID_IPropertyStore)
        store.SetValue(pscon.PKEY_AppUserModel_ID,
                       propsys.PROPVARIANTType(APP_USER_MODEL_ID, pythoncom.VT_LPWSTR))
        store.Commit()
        link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(path), True)
        return path.exists()
    except Exception:
        return _make_shortcut_simple(path, target, arguments, workdir, icon)


def _make_shortcut_simple(path: Path, target: str, arguments: str, workdir: str,
                          icon: "Path | None") -> bool:
    """Fallback shortcut via WScript.Shell — no identity property, but a working
    launcher with ORION's icon."""
    try:
        import win32com.client
    except Exception:
        return False
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        link = shell.CreateShortcut(str(path))
        link.TargetPath = target
        link.Arguments = arguments
        link.WorkingDirectory = workdir
        link.Description = "O.R.I.O.N. — your local AI assistant"
        if icon is not None:
            link.IconLocation = str(icon)
        link.Save()
        return path.exists()
    except Exception:
        return False


def install_shortcuts(desktop: bool = True, start_menu: bool = True) -> dict[str, bool]:
    """Create Desktop and Start-menu shortcuts. Returns what was created.

    Idempotent: re-running refreshes them (so a moved project or rebuilt venv
    fixes a dead shortcut). The Start-menu entry is what lets Windows pin ORION
    to the taskbar.
    """
    results: dict[str, bool] = {}
    if sys.platform != "win32":
        return {"supported": False}
    ensure_icon()
    if desktop:
        target = _desktop_dir() / f"{SHORTCUT_NAME}.lnk"
        results["desktop"] = _make_shortcut(target)
    if start_menu:
        programs = _start_menu_dir()
        if programs is not None:
            programs.mkdir(parents=True, exist_ok=True)
            results["start_menu"] = _make_shortcut(programs / f"{SHORTCUT_NAME}.lnk")
    return results


def _desktop_dir() -> Path:
    """Where Windows actually puts this user's Desktop.

    This used to build the path from the ONEDRIVE environment variable. That
    is only ever a guess, and on a machine signed into TWO OneDrive accounts it
    is the wrong one: ONEDRIVE named "OneDrive - Exeter College" while the
    redirected Desktop was "OneDrive - Example University". ORION
    dutifully refreshed a shortcut on a Desktop nobody looks at, and the stale
    one — still launching pythonw, which is why ORION "opens as Python 3.13" —
    sat untouched on the real Desktop.

    SHGetKnownFolderPath is the authoritative answer and follows redirection,
    so there is nothing left to guess. The old behaviour stays as the last
    fallback rather than the first choice.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            # FOLDERID_Desktop {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
            class _GUID(ctypes.Structure):
                _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                            ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8)]

            folder = _GUID(0xB4BFCC3A, 0xDB2C, 0x424C,
                           (ctypes.c_byte * 8)(*bytes.fromhex("B0297FE99A87C641")))
            out = ctypes.c_wchar_p()
            shell32 = ctypes.WinDLL("shell32")
            if shell32.SHGetKnownFolderPath(
                    ctypes.byref(folder), 0, None, ctypes.byref(out)) == 0:
                try:
                    path = Path(out.value or "")
                finally:
                    ctypes.WinDLL("ole32").CoTaskMemFree(out)
                if path.parts and path.is_dir():
                    return path
        except Exception:
            pass
    for env in ("ONEDRIVE",):
        base = os.environ.get(env)
        if base and (Path(base) / "Desktop").is_dir():
            return Path(base) / "Desktop"
    return Path.home() / "Desktop"


def _start_menu_dir() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "ORION"


def status() -> dict[str, object]:
    """Whether ORION is installed as a desktop app."""
    desktop = _desktop_dir() / f"{SHORTCUT_NAME}.lnk"
    start = _start_menu_dir()
    return {
        "supported": sys.platform == "win32",
        "icon": ICON_PATH.exists(),
        "desktop_shortcut": desktop.exists(),
        "start_menu_shortcut": bool(start and (start / f"{SHORTCUT_NAME}.lnk").exists()),
    }


def describe_status() -> str:
    st = status()
    if not st["supported"]:
        return "Desktop-app installation is a Windows feature."
    have = [name for name, key in (("Desktop", "desktop_shortcut"),
                                   ("Start menu", "start_menu_shortcut"))
            if st[key]]
    if have:
        return ("I'm installed as a desktop app (" + " and ".join(have) +
                "). You can pin me to the taskbar from the Start menu, and I "
                "start without a console window.")
    return ("I'm not installed as a desktop app yet — say 'install yourself as "
            "an app' and I'll add Desktop and Start-menu shortcuts you can pin.")


__all__ = [
    "APP_USER_MODEL_ID", "ICON_PATH", "SHORTCUT_NAME", "apply_window_icon",
    "describe_status", "ensure_icon", "install_shortcuts",
    "set_app_user_model_id", "status",
]
