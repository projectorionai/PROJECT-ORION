"""
Capability health matrix (Mark XXVI) — "why can't ORION do X?", answerable.

The OCR incident showed the real failure mode of a large system: a capability
that is PRESENT, REGISTERED and reports ``available: True`` can still be broken at
runtime, and nothing surfaces it. This module probes each major capability's
actual state and renders a matrix:

    Capability   present  registered  available  offline  detail

  * present     — the implementing module/engine imports.
  * registered  — the tool that exposes it is in the schema (None if not a tool).
  * available   — it is actually usable now (dependencies present, engine ready,
                  and — with deep=True — a real functional probe succeeded).
  * offline     — it works with no internet.

Probes never raise; a probe that itself fails is reported as such rather than
taking the matrix down. Deep probes (OCR functional read) are opt-in because they
cost a second or two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .utils import first_line


@dataclass
class CapabilityHealth:
    name: str
    present: bool
    available: bool
    offline: bool
    detail: str = ""
    registered: bool | None = None      # None = not exposed as a single tool

    def status(self) -> str:
        if not self.present:
            return "MISSING"
        if not self.available:
            return "DEGRADED"
        return "OK"

    def line(self) -> str:
        icon = {"OK": "✓", "DEGRADED": "▲", "MISSING": "✗"}[self.status()]
        reg = ("—" if self.registered is None else ("reg" if self.registered else "UNREG"))
        off = "offline" if self.offline else "cloud"
        return f"  {icon} {self.name:<16} [{reg:<5} {off:<7}] {self.detail}"


# ── helpers ───────────────────────────────────────────────────────────────────

class _NullBus:
    def __getattr__(self, _name):
        class _Sig:
            def emit(self, *a):
                pass

            def connect(self, *a):
                pass
        return _Sig()


#: Whether a module is installed, cached for the session.
#:
#: ``find_spec`` walks sys.path and stats the filesystem — profiling the matrix
#: showed **84 nt.stat calls per poll**, 99% of its cost, all of it disk I/O on
#: the qasync loop. Whether cv2 or pyttsx3 is installed cannot change without a
#: pip install, and reset_probe_cache() already exists for exactly that moment.
_IMPORTABLE: dict[str, bool] = {}


def _importable(module: str) -> bool:
    cached = _IMPORTABLE.get(module)
    if cached is not None:
        return cached
    import importlib.util
    try:
        found = importlib.util.find_spec(module) is not None
    except Exception:
        found = False
    _IMPORTABLE[module] = found
    return found


def _registered_tools() -> set[str]:
    try:
        from .dispatch_schema import TOOL_DECLARATIONS
        return {t["name"] for t in TOOL_DECLARATIONS}
    except Exception:
        return set()


def _val(x: Any) -> Any:
    return x() if callable(x) else x


# ── individual probes (each returns a CapabilityHealth, never raises) ─────────

#: Constructing an OcrEngine re-probes every backend (importing rapidocr alone
#: costs ~170 ms), and the matrix is polled by diagnostics and the proactive
#: loop. Which backends exist cannot change without a pip install, so the probe
#: engine is built once per session. reset_probe_cache() exists for after one.
_OCR_PROBE: Any = None


def reset_probe_cache() -> None:
    """Forget every cached probe — call after installing a backend.

    Clears the OCR probe engine, the plugin scan and the module-availability
    cache together: after a pip install all three are potentially stale, and
    forgetting only some of them is the kind of half-measure that makes a
    diagnostic lie.
    """
    global _OCR_PROBE, _PLUGIN_PROBE, _PLUGIN_PROBE_AT
    _OCR_PROBE = None
    _PLUGIN_PROBE = None
    _PLUGIN_PROBE_AT = 0.0
    _IMPORTABLE.clear()


def _probe_ocr(tools: set[str], deep: bool) -> CapabilityHealth:
    global _OCR_PROBE
    registered = "vision_analyse" in tools          # action='ocr'
    try:
        from .ocr_engine import OcrEngine
    except Exception as exc:
        return CapabilityHealth("OCR", False, False, True,
                                f"import failed: {first_line(exc, 80)}", registered)
    try:
        if _OCR_PROBE is None:
            _OCR_PROBE = OcrEngine(_NullBus())
        eng = _OCR_PROBE
        avail = bool(_val(eng.available))
        name = _val(eng.engine_name) or "none"
        if not avail:
            return CapabilityHealth("OCR", True, False, True,
                                    "no backend (pip install rapidocr-onnxruntime)", registered)
        detail = f"engine: {name}"
        if deep:
            ok, note = _functional_ocr(eng)
            if not ok:
                return CapabilityHealth("OCR", True, False, True,
                                        f"{name} present but read FAILED: {note}", registered)
            detail = f"engine: {name}; read OK"
        return CapabilityHealth("OCR", True, True, True, detail, registered)
    except Exception as exc:
        return CapabilityHealth("OCR", True, False, True,
                                f"probe error: {first_line(exc, 80)}", registered)


def _functional_ocr(engine: Any) -> tuple[bool, str]:
    """Render known text and confirm the engine reads it — the check that would
    have caught the numpy '.convert' fault."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (300, 80), (255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 34)
        except Exception:
            font = ImageFont.load_default()
        d.text((10, 22), "ORION", fill=(0, 0, 0), font=font)
        text = engine.image_to_text(img)
        return ("orion" in (text or "").lower()), (repr(text)[:40] or "empty")
    except Exception as exc:
        return False, first_line(exc, 60)


_PLUGIN_PROBE: Any = None
_PLUGIN_PROBE_AT: float = 0.0
#: The plugin scan reads and AST-parses every installed plugin. Plugins change
#: when the user installs one, not between two polls a second apart.
_PLUGIN_TTL_S = 30.0


def _probe_plugins(tools: set[str], deep: bool) -> CapabilityHealth:
    global _PLUGIN_PROBE, _PLUGIN_PROBE_AT
    import time as _time
    if (_PLUGIN_PROBE is not None and not deep
            and _time.monotonic() - _PLUGIN_PROBE_AT < _PLUGIN_TTL_S):
        return _PLUGIN_PROBE
    result = _probe_plugins_uncached(tools, deep)
    _PLUGIN_PROBE, _PLUGIN_PROBE_AT = result, _time.monotonic()
    return result


def _probe_plugins_uncached(tools: set[str], deep: bool) -> CapabilityHealth:
    """The plugin ecosystem as one row: how many are installed, how many are
    healthy, and whether any are disabled or broken.

    Plugins ADD tools at runtime, so a capability the user swears ORION had can
    be a disabled or dependency-broken plugin. Static scan only — the registry
    never imports a plugin to inspect it, so this stays safe and fast.
    """
    registered = "plugin" in tools
    try:
        from .plugin_registry import PluginRegistry
    except Exception as exc:
        return CapabilityHealth("Plugins", False, False, True,
                                f"import failed: {first_line(exc, 70)}", registered)
    try:
        records = PluginRegistry(bus=None).scan()
    except Exception as exc:
        return CapabilityHealth("Plugins", True, False, True,
                                f"scan failed: {first_line(exc, 70)}", registered)
    if not records:
        return CapabilityHealth("Plugins", True, True, True,
                                "none installed", registered)
    broken = [r.name for r in records if r.enabled and not r.healthy]
    disabled = [r.name for r in records if not r.enabled]
    undeclared = [r.name for r in records if r.undeclared]
    detail = f"{len(records)} installed"
    if disabled:
        detail += f", {len(disabled)} disabled"
    if undeclared:
        detail += f", {len(undeclared)} with undeclared capabilities"
    if broken:
        detail += f" — BROKEN: {', '.join(broken[:3])}"
        return CapabilityHealth("Plugins", True, False, True, detail, registered)
    return CapabilityHealth("Plugins", True, True, True, detail, registered)


def _simple(name: str, module: str, *, tool: str | None, offline: bool,
            tools: set[str], extra: str = "") -> CapabilityHealth:
    present = _importable(module)
    registered = (tool in tools) if tool else None
    detail = extra if present else f"missing module '{module}'"
    return CapabilityHealth(name, present, present, offline, detail, registered)


def _probe_engine(name: str, module: str, cls: str, *, tool: str | None,
                  offline: bool, tools: set[str], deep: bool = False) -> CapabilityHealth:
    registered = (tool in tools) if tool else None
    full = f"orion_core.{module}"
    # Shallow probe: confirm the module is IMPORTABLE without executing it —
    # `__import__` here cost ~1s across the matrix (forge alone pulls the LLM
    # brain + providers). find_spec is effectively free. The class is only
    # verified on a deep probe.
    if not deep:
        if not _importable(full):
            return CapabilityHealth(name, False, False, offline,
                                    f"missing module '{module}'", registered)
        return CapabilityHealth(name, True, True, offline, f"{module} present", registered)
    try:
        mod = __import__(full, fromlist=[cls])
        getattr(mod, cls)
    except Exception as exc:
        return CapabilityHealth(name, False, False, offline,
                                f"import failed: {first_line(exc, 70)}", registered)
    return CapabilityHealth(name, True, True, offline, f"{cls} present", registered)


# ── the matrix ────────────────────────────────────────────────────────────────

def capability_matrix(deep: bool = False) -> list[CapabilityHealth]:
    """Probe every major capability. deep=True adds functional probes (OCR read)."""
    tools = _registered_tools()
    rows: list[CapabilityHealth] = [
        _probe_ocr(tools, deep),
        _simple("Vision/screen", "cv2", tool="vision_analyse", offline=True,
                tools=tools, extra="cv2 + screen capture"),
        _simple("Voice (cloud)", "google.genai", tool=None, offline=False,
                tools=tools, extra="Gemini Live"),
        _simple("Voice (offline)", "pyttsx3", tool=None, offline=True,
                tools=tools, extra="local TTS fallback"),
        _simple("Speech-to-text", "vosk", tool=None, offline=True,
                tools=tools, extra="Vosk offline STT"),
        _simple("Desktop control", "pyautogui", tool="desktop_control",
                offline=True, tools=tools),
        _simple("UI inspection", "pywinauto", tool="vision_verify",
                offline=True, tools=tools, extra="accessibility tree"),
        _simple("Gesture control", "mediapipe", tool="gesture_control",
                offline=True, tools=tools),
        _probe_engine("Study", "study", "StudyEngine", tool="study",
                      offline=True, tools=tools, deep=deep),
        _probe_engine("Focus", "focus", "FocusEngine", tool="focus",
                      offline=True, tools=tools, deep=deep),
        _probe_engine("Finance", "finance", "FinanceEngine", tool="finance",
                      offline=True, tools=tools, deep=deep),
        _probe_engine("Wellbeing", "wellbeing", "WellbeingEngine", tool="wellbeing",
                      offline=True, tools=tools, deep=deep),
        _probe_engine("Forge", "forge", "ForgeOrchestrationManager", tool="forge",
                      offline=False, tools=tools, deep=deep),
        _probe_plugins(tools, deep),
        _simple("Chess", "chess", tool="chess", offline=True, tools=tools,
                extra="own engine; Stockfish optional"),
        _simple("Security recon", "scapy", tool="security_recon", offline=True,
                tools=tools, extra="nmap/scapy (authorized targets only)"),
    ]
    return rows


def render_matrix(rows: list[CapabilityHealth]) -> str:
    missing = [r for r in rows if r.status() == "MISSING"]
    degraded = [r for r in rows if r.status() == "DEGRADED"]
    ok = [r for r in rows if r.status() == "OK"]
    header = (f"Capability health — {len(ok)} OK, {len(degraded)} degraded, "
              f"{len(missing)} missing:")
    return header + "\n" + "\n".join(r.line() for r in rows)


def unavailable(rows: list[CapabilityHealth] | None = None) -> list[CapabilityHealth]:
    """The capabilities that would answer 'why can't ORION do X' — not OK."""
    rows = rows if rows is not None else capability_matrix()
    return [r for r in rows if r.status() != "OK"]


__all__ = [
    "CapabilityHealth", "capability_matrix", "render_matrix", "unavailable",
    "reset_probe_cache",
]
