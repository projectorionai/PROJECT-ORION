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


def _resident_stt():
    from orion_core.speech_offline import OfflineTranscriber

    stt = OfflineTranscriber(defer=True)
    stt.engine = "faster-whisper"
    stt._selected = True
    stt._model = object()
    stt._last_used = 0.0
    return stt


def test_pressure_cleanup_retries_after_the_initial_shed_was_too_early(monkeypatch):
    from orion_core import resource_governor as rg

    monkeypatch.setenv("ORION_STT_IDLE_S", "600")
    monkeypatch.setenv("ORION_GOVERNOR", "1")
    governor = rg.ResourceGovernor()
    governor.thresholds = (85.0, 92.0, 96.0)
    monkeypatch.setattr(rg, "GOVERNOR", governor)
    stt = _resident_stt()
    attempts = []
    governor.register("idle STT", rg.Pressure.CONSTRAINED,
                      lambda: attempts.append(stt.release_if_idle(now=30, limit_s=60)))
    governor.observe(90)
    governor.observe(90)
    assert attempts == [False] and stt._model is not None
    governor.observe(90)
    assert attempts == [False], "governor actions deliberately run once per transition"
    # This is the periodic reaper's ordinary call after the model becomes idle.
    assert stt.release_if_idle(now=90), "pressure cleanup reverted to the ten-minute delay"
    assert stt._model is None and not stt.ready
    loaded = []

    def reload():
        stt._model = object()
        stt.engine = "faster-whisper"
        loaded.append(stt.model_size)

    monkeypatch.setattr(stt, "_select_engine", reload)
    assert stt.ensure_ready()
    assert loaded == [stt.model_size], "unloading must retain on-demand hearing"


@pytest.mark.parametrize("case,limit,idle,expected", [
    ("normal", "600", 90, False),
    ("pressure", "600", 59, False),
    ("pressure", "600", 60, True),
    ("disabled", "600", 90, False),
    ("pressure", "0", 1000, False),
    ("pressure", "20", 25, True),
    ("normal", "600", 600, True),
])
def test_idle_model_release_preserves_limits_and_governor_opt_outs(
        monkeypatch, case, limit, idle, expected):
    from orion_core import resource_governor as rg

    monkeypatch.setenv("ORION_STT_IDLE_S", limit)
    governor = rg.ResourceGovernor()
    governor.acting = case != "disabled"
    governor.level = rg.Pressure.NORMAL if case == "normal" else rg.Pressure.CONSTRAINED
    monkeypatch.setattr(rg, "GOVERNOR", governor)
    stt = _resident_stt()
    assert stt.release_if_idle(now=idle) is expected
    assert (stt._model is None) is expected


def test_memory_pressure_does_not_unload_a_transcription_in_flight(monkeypatch):
    import threading
    from orion_core import resource_governor as rg

    monkeypatch.setenv("ORION_STT_IDLE_S", "600")
    governor = rg.ResourceGovernor()
    governor.acting = True
    governor.level = rg.Pressure.CRITICAL
    monkeypatch.setattr(rg, "GOVERNOR", governor)
    stt = _resident_stt()
    entered, finish = threading.Event(), threading.Event()

    def transcribing():
        with stt._lock:
            entered.set()
            assert finish.wait(2)

    thread = threading.Thread(target=transcribing)
    thread.start()
    try:
        assert entered.wait(1)
        assert not stt.release_if_idle(now=90)
        assert stt._model is not None
    finally:
        finish.set()
        thread.join(1)
    assert not thread.is_alive()
    assert stt.release_if_idle(now=90), "a previously busy model never got another cleanup chance"


async def test_production_stt_pressure_release_runs_off_the_gui_thread(monkeypatch):
    import ast
    import asyncio
    import threading
    import time
    from pathlib import Path
    from orion_core import resource_governor as rg
    from orion_core import speech_offline

    # Execute the actual registered callback without booting an app/model.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "orion_core/app.py").read_bytes())
    registration = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "register" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "the idle speech-to-text model")
    released_on = []
    collected_on = []
    monkeypatch.setattr(speech_offline.gc, "collect",
                        lambda: collected_on.append(threading.get_ident()) or 0)

    class Model:
        def __del__(self):
            released_on.append(threading.get_ident())

    stt = _resident_stt()
    stt._model = Model()
    stt._last_used = time.monotonic() - 120
    governor = rg.ResourceGovernor()
    governor.acting = True
    governor.level = rg.Pressure.CONSTRAINED
    monkeypatch.setattr(rg, "GOVERNOR", governor)
    monkeypatch.setenv("ORION_STT_IDLE_S", "600")
    callback = eval(compile(ast.Expression(registration.args[2]), "STT registration", "eval"),
                    {"asyncio": asyncio, "offline_stt": stt})
    gui_thread = threading.get_ident()
    result = callback()
    assert asyncio.iscoroutine(result), "model destruction still ran synchronously on the GUI thread"
    assert await result
    assert released_on and released_on[0] != gui_thread
    assert collected_on == [gui_thread], "global collection may destroy unrelated Qt objects"


async def test_idle_reaper_collects_on_the_loop_thread_after_worker_model_release(monkeypatch):
    import asyncio
    import threading
    import time
    from orion_core import speech_offline

    monkeypatch.setenv("ORION_STT_IDLE_S", "1")
    stt = _resident_stt()
    stt._last_used = time.monotonic() - 2
    released_on, collected_on = [], []
    done = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    class Model:
        def __del__(self):
            released_on.append(threading.get_ident())

    def collect():
        collected_on.append(threading.get_ident())
        loop.call_soon_threadsafe(done.set)
        return 0

    stt._model = Model()
    monkeypatch.setattr(speech_offline.gc, "collect", collect)
    reaper = asyncio.create_task(stt.idle_reaper(interval=0.01))
    try:
        await asyncio.wait_for(done.wait(), timeout=2)
    finally:
        reaper.cancel()
        await asyncio.gather(reaper, return_exceptions=True)
    assert released_on and released_on[0] != loop_thread
    assert collected_on == [loop_thread], "the reaper collected GUI objects on its worker"


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


def test_offline_whisper_uses_exact_cached_english_checkpoint_before_multilingual(tmp_path, monkeypatch):
    import sys
    from orion_core.speech_offline import OfflineTranscriber

    cache = tmp_path / "whisper"
    cache.mkdir()
    english = cache / "base.en.pt"
    english.write_bytes(b"synthetic checkpoint; never loaded")
    (cache / "base.pt").write_bytes(b"synthetic fallback")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    loaded, model = [], object()

    def unavailable(*_a, **_k):
        raise RuntimeError("synthetic faster-whisper unavailable")

    def load_model(path):
        loaded.append(path)
        return model

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=unavailable))
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=load_model))
    transcriber = OfflineTranscriber(model_size="base.en", defer=True)
    assert transcriber.ensure_ready() and transcriber.engine == "whisper"
    assert loaded == [str(english)] and transcriber._model is model


def test_offline_whisper_retains_installed_multilingual_fallback(tmp_path, monkeypatch):
    from orion_core import speech_offline

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    names = speech_offline._whisper_checkpoints("base.en")
    assert names == [tmp_path / "whisper" / "base.en.pt", tmp_path / "whisper" / "base.pt"]
    explicit = tmp_path / "private.pt"
    assert speech_offline._whisper_checkpoints(str(explicit)) == [explicit]


@pytest.mark.parametrize("complete,has_ref,expected", [
    (True, True, True), (False, True, False), (True, False, False),
])
def test_cached_offline_hearing_checks_complete_local_snapshot_without_loading(
        tmp_path, monkeypatch, complete, has_ref, expected):
    from orion_core import speech_offline

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    root = tmp_path / "models--Systran--faster-whisper-base.en"
    snapshot = root / "snapshots" / "synthetic-revision"
    snapshot.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        if complete or name != "tokenizer.json":
            (snapshot / name).write_bytes(b"synthetic local metadata")
    if has_ref:
        (root / "refs").mkdir()
        (root / "refs" / "main").write_text("synthetic-revision", encoding="utf-8")
    monkeypatch.setattr(speech_offline, "_package_present", lambda name: name == "faster_whisper")
    transcriber = speech_offline.OfflineTranscriber(model_size="base.en", defer=True)
    monkeypatch.setattr(transcriber, "_select_engine", lambda: pytest.fail("status loaded an ASR model"))
    assert transcriber.cached_available is expected
    assert not transcriber.ready and not transcriber._selected
    status = transcriber.status()
    assert status["cached_available"] is expected and not status["loaded"]


def test_cached_offline_hearing_recognises_english_checkpoint_without_importing_engine(tmp_path, monkeypatch):
    from orion_core import speech_offline

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    checkpoint = tmp_path / "whisper" / "base.en.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic checkpoint")
    monkeypatch.setattr(speech_offline, "_package_present", lambda name: name == "whisper")
    transcriber = speech_offline.OfflineTranscriber(model_size="base.en", defer=True)
    monkeypatch.setattr(transcriber, "_select_engine", lambda: pytest.fail("readiness loaded a model"))
    assert transcriber.cached_available and not transcriber.ready
    transcriber._selected = True  # an actual load failure wins over cached metadata
    assert not transcriber.cached_available


def test_optional_metrics_failure_preserves_successful_offline_transcript():
    from orion_core.speech_offline import OfflineTranscriber

    def broken_metric(*_):
        raise RuntimeError("synthetic optional metrics failure")

    transcriber = OfflineTranscriber(
        telemetry=SimpleNamespace(metrics=SimpleNamespace(incr=broken_metric)), defer=True)
    transcriber.engine = "faster-whisper"
    transcriber._selected = True
    transcriber._model = SimpleNamespace(
        transcribe=lambda *_a, **_k: ([SimpleNamespace(text="I can hear offline.")], None))
    assert transcriber.transcribe_pcm(b"\x01\x00" * 160, identity_checked=True) == "I can hear offline."


@pytest.mark.parametrize("alias,repo", [
    ("large", "Systran/faster-whisper-large-v3"),
    ("distil-large-v2", "Systran/faster-distil-whisper-large-v2"),
    ("distil-medium.en", "Systran/faster-distil-whisper-medium.en"),
    ("distil-small.en", "Systran/faster-distil-whisper-small.en"),
    ("distil-large-v3", "Systran/faster-distil-whisper-large-v3"),
    ("distil-large-v3.5", "distil-whisper/distil-large-v3.5-ct2"),
    ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    ("turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
])
def test_cached_offline_hearing_resolves_installed_faster_whisper_aliases(tmp_path, monkeypatch, alias, repo):
    from orion_core import speech_offline

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    root = tmp_path / ("models--" + repo.replace("/", "--"))
    snapshot = root / "snapshots" / "synthetic-revision"
    snapshot.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        (snapshot / name).write_bytes(b"synthetic metadata; never a real model")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("synthetic-revision", encoding="utf-8")
    monkeypatch.setattr(speech_offline, "_package_present", lambda name: name == "faster_whisper")
    transcriber = speech_offline.OfflineTranscriber(model_size=alias, defer=True)
    monkeypatch.setattr(transcriber, "_select_engine", lambda: pytest.fail("alias status loaded a model"))
    assert transcriber.cached_available and not transcriber.ready
