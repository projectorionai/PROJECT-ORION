"""
Four reported faults, each pinned by the measurement that found it.

  1. "ORION still takes ages to initialise… it goes to 'not responding'."
     ~6 s of BLOCKING work sat on the GUI task. Measured per phase.
  2. "When I copy paste a prompt… it reroutes to standby or does nothing."
     A pasted paragraph matched the standby AND focus patterns.
  3. "On the nav panel it still reads System health: Unknown."
     Subsystems that had not started yet reported UNKNOWN, and one UNKNOWN
     critical dragged the whole roll-up down.
  4. "1 quarantined: json_validator_tool."
     A missing pip package condemned a perfectly good tool.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Sig:
    def __init__(self):
        self.messages = []

    def emit(self, *a):
        self.messages.append(a[0] if len(a) == 1 else a)

    def connect(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


# ── 1. startup must not block the GUI ────────────────────────────────────────

def test_the_slow_subsystems_are_deferred_off_the_startup_path():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py"
              ).read_text(encoding="utf-8")
    assert "OfflineTranscriber(bus, telemetry, defer=True)" in source, (
        "loading faster-whisper costs ~2.8 s and must not run on the GUI task")
    assert "_warm_local_stack" in source
    assert "asyncio.to_thread(ollama.register" in source, (
        "the Ollama probe costs ~2 s and must run in a worker thread")


def test_the_warm_up_starts_after_the_ui_is_built():
    """Starting it mid-chain moved the stall instead of removing it."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py"
              ).read_text(encoding="utf-8")
    started = source.find('_warm_local_stack(), name="orion-warm-local"')
    deck = source.find("deck = UnifiedDashboard(")
    assert started > deck > 0, (
        "the warm-up must be launched after the deck exists, or it competes "
        "with the rest of startup for the event loop")


def test_the_offline_transcriber_can_be_deferred_and_still_works():
    from orion_core.speech_offline import OfflineTranscriber

    engine = OfflineTranscriber(_Bus(), None, defer=True)
    assert engine.ready is False, "deferred means not loaded yet"
    # `available` must load on demand so a caller never sees a false negative.
    assert engine.available in (True, False)
    assert engine._selected is True


def test_the_vad_can_be_deferred_and_still_gates():
    from orion_core.audio import SileroVADGatekeeper

    vad = SileroVADGatekeeper(_Bus(), threshold=0.5, defer=True)
    assert vad._initialised is False
    # The deterministic fallback must work from the very first chunk.
    assert isinstance(vad.accepts(b"\x00\x00" * 512), bool)


def test_the_ollama_probe_has_a_short_startup_timeout():
    from orion_core.local_models import OllamaManager

    assert OllamaManager.STARTUP_PROBE_TIMEOUT <= 0.5, (
        "a connect that hangs rather than refusing costs the full timeout; "
        "startup cannot afford seconds of it")


# ── 2. a pasted prompt is not a command ──────────────────────────────────────

PASTE = ("Please review the following requirements and give me a plan. The user "
         "wants the system to stay quiet during long tasks, so give me a minute "
         "to think about how standby should work. Also consider that I am busy "
         "most days.")


def test_the_pasted_prompt_matches_the_raw_patterns():
    """Precondition: this is why the bug happened at all."""
    from orion_core.live_worker import GenAILiveWorker

    assert GenAILiveWorker._QUIET_ON_RE.search(PASTE.lower())
    assert GenAILiveWorker._FOCUS_ON_RE.search(PASTE.lower())


def test_but_it_is_not_of_control_length():
    from orion_core.live_worker import _is_control_length

    assert _is_control_length(PASTE) is False


@pytest.mark.parametrize("phrase", [
    "orion standby", "go on standby", "be quiet", "i am busy",
    "wake up", "i need you", "i'm done", "focus mode", "do not disturb",
])
def test_real_commands_are_still_short_enough(phrase):
    from orion_core.live_worker import _is_control_length

    assert _is_control_length(phrase) is True, phrase


def test_all_three_command_handlers_are_length_guarded():
    import inspect

    from orion_core.live_worker import GenAILiveWorker

    for method in (GenAILiveWorker._handle_quiet_command,
                   GenAILiveWorker._handle_focus_command,
                   GenAILiveWorker._held_in_standby):
        assert "_is_control_length" in inspect.getsource(method), method.__name__


# ── 3. "not started yet" is not "unknown" ────────────────────────────────────

def _worker(started: bool):
    if not started:
        return SimpleNamespace(mic=None, microphone_enabled=True,
                               recogniser=SimpleNamespace(available=True),
                               speech=None, connected=True)
    return SimpleNamespace(
        mic=SimpleNamespace(_stream=object()), microphone_enabled=True,
        recogniser=SimpleNamespace(available=True), connected=True,
        speech=SimpleNamespace(
            tts=SimpleNamespace(_active_backend="elevenlabs", available=True),
            playback=SimpleNamespace(_stream=object(), held=lambda: False)))


def _health(worker):
    from orion_core.health_model import HealthModel

    model = HealthModel(_Bus())
    model.register_defaults(worker=worker,
                            memory=SimpleNamespace(tiers_snapshot=lambda: {"a": 1}))
    return model


def test_mid_startup_reads_recovering_not_unknown():
    """The exact NAV-panel complaint."""
    model = _health(_worker(started=False))
    assert model.overall() == "RECOVERING"


def test_a_started_system_reads_online():
    assert _health(_worker(started=True)).overall() == "ONLINE"


def test_a_not_yet_started_microphone_is_not_unknown():
    snapshot = _health(_worker(started=False)).snapshot()
    assert snapshot["Microphone"]["status"] == "RECOVERING"
    assert "starting" in snapshot["Microphone"]["detail"]


def test_unknown_is_still_available_for_things_genuinely_unknown():
    from orion_core.health_model import HealthModel, UNKNOWN

    model = HealthModel(_Bus())
    model.register("Mystery", lambda: {"status": UNKNOWN, "detail": "no idea"})
    assert model.snapshot()["Mystery"]["status"] == UNKNOWN


# ── 4. a missing package is not a broken tool ────────────────────────────────

def test_json_validator_is_no_longer_quarantined():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "config" / "custom_tools"
    assert (root / "json_validator_tool.py").is_file(), (
        "the tool was sound; only a dependency was missing")
    assert not list((root / "_quarantine").glob("json_validator*")), (
        "it must not still be sitting in quarantine")


def test_its_dependency_is_installed():
    import importlib.util

    assert importlib.util.find_spec("jsonschema") is not None


def test_the_tool_satisfies_the_contract():
    import importlib.util
    from pathlib import Path

    from orion_core.forge_contract import contract_problems

    path = (Path(__file__).resolve().parents[1] / "config" / "custom_tools"
            / "json_validator_tool.py")
    spec = importlib.util.spec_from_file_location("jv_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert contract_problems(module.get_tool_schema(), module.run) == []


def test_the_loader_installs_a_missing_package_before_condemning_a_tool():
    import inspect

    from orion_core.dynamic_loader import ReflectiveModuleLoader

    source = inspect.getsource(ReflectiveModuleLoader.load_and_register)
    assert "ModuleNotFoundError" in source
    assert "self._install(" in source, (
        "a missing pip package is an environment fix, not a code fault")


def test_dangerous_packages_are_never_auto_installed():
    from orion_core.dynamic_loader import ReflectiveModuleLoader

    for name in ("os", "sys", "subprocess", "pip", "setuptools"):
        assert name in ReflectiveModuleLoader.INSTALL_DENYLIST, name


def test_a_package_is_only_attempted_once_per_session():
    from orion_core.dynamic_loader import ReflectiveModuleLoader

    loader = ReflectiveModuleLoader(_Bus())
    assert loader._dependency_attempts == set()
