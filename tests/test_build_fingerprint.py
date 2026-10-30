from tools.inspect_installed_build import fingerprint


def test_packaged_comparison_ignores_install_path_and_line_numbers():
    before = compile("def value():\n    return 12\n", "a/source.py", "exec")
    after = compile("\n\ndef value():\n    return 12\n", "b/installed.py", "exec")
    assert fingerprint(before) == fingerprint(after)
    different = compile("def value():\n    return 13\n", "a/source.py", "exec")
    assert fingerprint(before) != fingerprint(different)


def test_inspection_does_not_inherit_its_own_future_flags(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    from tools.inspect_installed_build import inspect_build
    source = b'"A package without future annotations."\nvalue = 1\n'
    (tmp_path / "orion_core").mkdir()
    (tmp_path / "orion_core/__init__.py").write_bytes(source)
    executable = tmp_path / "test.exe"
    executable.write_bytes(b"fixture")
    archive = SimpleNamespace(toc={"orion_core": None},
        extract=lambda _: compile(source, "original.py", "exec", dont_inherit=True))
    readers = SimpleNamespace(CArchiveReader=lambda _: SimpleNamespace(
        toc={"PYZ.pyz": None}, open_embedded_archive=lambda _: archive))
    monkeypatch.setitem(sys.modules, "PyInstaller.archive.readers", readers)
    assert inspect_build(executable, tmp_path)["source_matches"]


def test_comparison_ignores_constant_interning_and_object_sharing():
    shared = "a reasonably long constant"
    independent = (shared + "x")[:-1]
    assert shared == independent and shared is not independent
    code = compile("x = 'value'", "source.py", "exec")
    before = code.replace(co_consts=(shared, shared))
    after = code.replace(co_consts=(shared, independent))
    assert fingerprint(before) == fingerprint(after)


# ── tools/check_installed_build.py: the installed-app parity harness ─────────
# None of these needs a built ORION.exe or touches the owner's config: the
# derivations run on small source trees in tmp_path (or read the real source),
# and the launch-based checks run against tiny fake children.

import ast  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from tools import check_installed_build as parity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORLD_TRACEBACK = ('Traceback (most recent call last):\n'
                   '  File "orion_core\\mcp_world.py", line 15, in <module>\n'
                   '    from orion_core.world_feeds import LAYER_NAMES\n'
                   "ModuleNotFoundError: No module named 'orion_core.world_feeds'\n")


def _scan(source: str, *, package: str = "orion_core.gui", everything: bool = False) -> dict:
    needs: dict = {}
    parity.scan_imports(ast.parse(source), site="x.py", package=package, needs=needs,
                        everything=everything)
    return needs


def test_scan_records_lazy_guarded_and_named_imports_only():
    needs = _scan(
        "import os\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import never_at_runtime\n"
        "try:\n    import optional_thing\nexcept ImportError:\n    optional_thing = None\n"
        "def f():\n    import pycaw.pycaw\n    from faster_whisper import WhisperModel\n"
        "    from .face3d import local_three\n    from .. import constants\n"
        "def g():\n    importlib.import_module('winrt.windows.media.control')\n"
        "    return lazy_attr('aiohttp', 'ClientSession')\n"
        "sd = LazyModule('sounddevice')\n"
        "probe = _simple('Vision', 'cv2', tool=None, offline=True)\n")
    assert "os" not in needs and "typing" not in needs          # plain module level: in the graph
    assert "never_at_runtime" not in needs                       # TYPE_CHECKING never runs
    assert needs["optional_thing"].kinds == {"guarded"}
    assert needs["pycaw.pycaw"].kinds == {"function"}
    assert needs["faster_whisper"].names == {"WhisperModel"}
    assert needs["orion_core.gui.face3d"].names == {"local_three"}  # relative, resolved
    assert needs["orion_core"].names == {"constants"}
    assert needs["winrt.windows.media.control"].kinds == {"dynamic"}
    assert needs["aiohttp"].names == {"ClientSession"}
    assert "sounddevice" in needs and "cv2" in needs             # LazyModule / _simple
    assert needs["pycaw.pycaw"].sites == ["x.py:10"]


def test_a_plugin_has_every_import_recorded():
    """A shipped plugin is loaded from a file: none of its imports is in the frozen graph."""
    needs = _scan("import jsonschema\nfrom sklearn.ensemble import RandomForestClassifier\n",
                  package="", everything=True)
    assert needs["jsonschema"].kinds == {"plugin"}
    assert needs["sklearn.ensemble"].names == {"RandomForestClassifier"}


def test_the_real_source_scan_covers_every_optional_by_design_entry():
    """No stale allowlist entry: each module allowed to be absent is really imported."""
    needs = parity.lazy_imports(ROOT)
    tops = {module.split(".")[0] for module in needs}
    assert set(parity.OPTIONAL_BY_DESIGN) <= tops
    assert "orion_core.forge_runtime" in needs                   # orion.py's lazy imports too
    assert "orion_core.__main__" not in parity.orion_modules(ROOT)


def test_optional_reasons_cover_build_excludes_and_dotted_names():
    assert "EXCLUDES" in parity.optional_reason("torchvision.ops", ["torchvision"])
    assert parity.optional_reason("easyocr.utils")
    assert parity.optional_reason("pycaw.pycaw") == ""
    assert parity.platform_absent_stdlib("fcntl") == (sys.platform == "win32")
    assert not parity.platform_absent_stdlib("json")


def test_build_config_reads_lists_and_conditional_appends(tmp_path):
    (tmp_path / "build_standalone.py").write_text(
        'COLLECT_ALL = ["tzdata", "torch"]\nHIDDEN_IMPORTS = ["x"]\n'
        'EXCLUDES = ["torch"]\nDATAS = [("assets", "assets")]\n'
        'HIDDEN_IMPORTS.append("berserk")\n'
        'if maybe:\n    COLLECT_ALL.append("mediapipe")\n    COLLECT_ALL.append("absent")\n',
        encoding="utf-8")
    config = parity.build_config(tmp_path)
    assert config["hidden_imports"] == ["x", "berserk"]
    assert config["datas"] == [("assets", "assets")]
    assert config["conditional"]["collect_all"] == ["mediapipe", "absent"]
    wanted = parity.expected_collect_all(config, installed=lambda name: name != "absent")
    assert wanted == ["tzdata", "mediapipe"]                     # excluded and absent dropped


def test_the_real_build_carries_what_was_lost_before():
    config = parity.build_config(ROOT)
    assert {"faster_whisper", "tzdata", "pytest", "aec_audio_processing"} <= set(config["collect_all"])
    assert ("orion_core/forge_contract.py", "orion_core") in config["datas"]
    kinds = parity.page_kinds(ROOT)
    assert kinds["brain"] == ["iris", "lanes"] and kinds["face"] == ["main"]
    assert parity.mode_flags(ROOT)["page"] == "--page-host"


@pytest.mark.parametrize("builder_has_piper", [False, True])
def test_prepared_piper_import_is_optional_only_when_absent_from_the_builder(
        tmp_path, builder_has_piper):
    entries = [{"kind": "import", "module": "piper.voice", "names": ["PiperVoice"],
                "group": "dynamic", "sites": ["orion_core/voice_piper.py"]}]
    results = [{"ok": False, "error": "ModuleNotFoundError: No module named piper",
                "missing": "piper"}]
    groups = parity.classify_imports(
        entries, results, excludes=[], archive=parity.Archive(set(), set(), set()),
        built_at=None, source=tmp_path, installed=lambda name: builder_has_piper)
    group = groups["dynamic"]
    assert bool(group["optional"]) is not builder_has_piper
    assert bool(group["failures"]) is builder_has_piper
    if not builder_has_piper:
        assert "prepared" in group["optional"][0].lower()


@pytest.mark.parametrize("missing", [None, "piper", "onnxruntime", "torch"])
def test_installed_piper_broken_dependency_is_a_real_parity_failure(tmp_path, missing):
    entries = [{"kind": "import", "module": "piper.voice", "names": ["PiperVoice"],
                "group": "dynamic"}]
    results = [{"ok": False, "error": "ImportError: DLL load failed", "missing": missing}]
    groups = parity.classify_imports(
        entries, results, excludes=[], archive=parity.Archive(set(), set(), set()),
        built_at=None, source=tmp_path, installed=lambda name: True)
    assert groups["dynamic"]["failures"] and not groups["dynamic"]["optional"]


def test_installed_piper_is_collected_with_its_phoneme_and_native_data():
    config = parity.build_config(ROOT)
    assert "piper" in config["conditional"]["collect_all"]
    assert "piper" not in parity.expected_collect_all(config, installed=lambda name: False)
    assert "piper" in parity.expected_collect_all(config, installed=lambda name: name == "piper")


def test_data_checks_name_each_missing_file(tmp_path):
    source, internal = tmp_path / "src", tmp_path / "_internal"
    (source / "assets" / "voiceprint").mkdir(parents=True)
    (source / "assets" / "voiceprint" / "encoder.onnx").write_bytes(b"x" * 10)
    (source / "assets" / "three.js").write_text("three", encoding="utf-8")
    (internal / "assets").mkdir(parents=True)
    (internal / "assets" / "three.js").write_text("three", encoding="utf-8")
    check = parity.check_datas({"datas": [("assets", "assets")]}, source, internal)
    assert not check.ok and "assets/voiceprint/encoder.onnx" in check.detail
    listed = {"faster_whisper": (["faster_whisper/assets/silero_vad_v6.onnx"], ""),
              "sounddevice": ([], "a single module, no package data")}
    check = parity.check_collect_all(["faster_whisper", "sounddevice"], internal,
                                     lister=lambda name: listed[name])
    assert not check.ok and "silero_vad_v6.onnx" in check.detail
    check = parity.check_critical(internal, tmp_path)
    assert not check.ok and "faster_whisper/assets/*.onnx" in check.detail


def test_resource_requests_resolve_constant_and_starred_arguments(tmp_path):
    package = tmp_path / "orion_core"
    package.mkdir()
    (package / "m.py").write_text(
        'PARTS = ("assets", "voiceprint", "encoder.onnx")\nTHREE = "assets/three"\n'
        'def a():\n    return resource_path("assets", "yamnet")\n'
        'def b():\n    return resource_path(*PARTS)\n'
        'def c():\n    return resource_path(*THREE.split("/"))\n'
        'def d(name):\n    return resource_path("assets", name)\n', encoding="utf-8")
    found = [parts for parts, _site in parity.resource_requests(tmp_path)]
    assert found == [("assets", "yamnet"), ("assets", "voiceprint", "encoder.onnx"),
                     ("assets", "three")]                         # the variable one is skipped


def test_collect_all_detects_the_missing_native_echo_canceller_dll(tmp_path):
    name = "aec_audio_processing/files/webrtc-audio-processing-2-1.dll"
    internal = tmp_path / "_internal"
    lister = lambda _package: ([name], "")
    check = parity.check_collect_all(["aec_audio_processing"], internal, lister=lister)
    assert not check.ok and "webrtc-audio-processing" in check.detail
    dll = internal / name
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"synthetic DLL, never loaded")
    assert parity.check_collect_all(["aec_audio_processing"], internal, lister=lister).ok


def test_entry_modes_come_from_the_entry_scripts_constants():
    code = compile('import multiprocessing\nmultiprocessing.freeze_support()\n'
                   'if "--mcp-server" in sys.argv: pass\n'
                   'def later():\n    return "--forge-verify"\n', "orion.py", "exec")
    strings: set = set()
    names: set = set()
    parity.walk_code(code, strings, names)
    modes = parity.entry_modes(parity.Archive(strings, names, None),
                               ["--mcp-server", "--face-host", "--forge-verify"])
    assert modes == {"--mcp-server": True, "--face-host": False, "--forge-verify": True,
                     "spawn": True}


def test_fatal_exception_and_child_failure_read_the_headless_guard_output():
    lines = WORLD_TRACEBACK.splitlines()
    assert parity.fatal_exception(lines) == \
        "ModuleNotFoundError: No module named 'orion_core.world_feeds'"
    assert parity.fatal_exception(["ModuleNotFoundError: logged, no traceback"]) == ""
    assert parity.fatal_exception(lines + ["[parity] entry 7"]) == ""   # it carried on
    assert parity.child_failure(1, lines).startswith("exit 1: ModuleNotFoundError")
    assert parity.child_failure(None, []) == "still running"


def test_window_problems_and_process_ancestry():
    good = {"exists": True, "class": "Qt6112QWindowToolSaveBits", "pid": 7, "top_level": True}
    assert parity.window_problems(good, 7) == []
    assert parity.window_problems({"exists": False}, 7) == ["is not a window"]
    bad = dict(good, **{"class": "Chrome_WidgetWin_0", "pid": 8, "top_level": False})
    assert len(parity.window_problems(bad, 7)) == 3
    assert parity.window_problems({"exists": None}, 7) == []      # not Windows: cannot ask
    tree = {30: 20, 20: 10, 10: 1}
    assert parity.descends_from(30, 10, tree) and parity.descends_from(10, 10, tree)
    assert not parity.descends_from(10, 30, tree) and not parity.descends_from(99, 10, tree)


def test_classify_separates_faults_drift_and_optional(tmp_path):
    (tmp_path / "orion_core").mkdir()
    newer = tmp_path / "orion_core" / "new_thing.py"
    newer.write_text("", encoding="utf-8")
    entries = [{"kind": "import", "module": "orion_core.new_thing", "names": [], "group": "orion_core"},
               {"kind": "import", "module": "pycaw.pycaw", "names": [], "group": "dynamic",
                "sites": ["orion_core/audio_follow.py:160"]},
               {"kind": "import", "module": "torch", "names": [], "group": "dynamic"},
               {"kind": "data", "key": "whisper-vad", "group": "data"}]
    results = [{"ok": False, "error": "ModuleNotFoundError: x", "missing": "orion_core.new_thing"},
               {"ok": False, "error": "ModuleNotFoundError: No module named 'pycaw'",
                "missing": "pycaw"},
               {"ok": False, "error": "ModuleNotFoundError: torch", "missing": "torch"},
               {"ok": False, "error": "NoSuchFile: silero_vad_v6.onnx"}]
    groups = parity.classify_imports(entries, results, excludes=["torch"],
                                     archive=parity.Archive(set(), set(), set()),
                                     built_at=newer.stat().st_mtime - 60, source=tmp_path,
                                     installed=lambda name: False)
    assert groups["orion_core"]["drift"] and not groups["orion_core"]["failures"]
    assert "not installed in the build environment" in groups["dynamic"]["failures"][0]
    assert groups["dynamic"]["optional"] == ["torch: excluded from the lean build "
                                             "(build_standalone.EXCLUDES); chess training only"]
    checks = parity.import_checks(groups, mechanism="spawn", run_notes=[], skipped=[], seconds=1)
    assert [check.ok for check in checks] == [False, False, False]
    assert "only because the source is newer" in checks[1].detail


def test_the_probe_scripts_compile_and_name_every_data_probe():
    for text in (parity.IMPORT_PROBE, parity.SPEC_PROBE, parity.PIP_PROBE, parity.IMPORT_CHECK):
        ast.parse(text)
    import re
    assert set(re.findall(r'@data\("([^"]+)"\)', parity.IMPORT_PROBE)) == set(parity.DATA_PROBES)


def test_probe_wheel_is_a_valid_offline_wheel(tmp_path):
    wheel = parity.build_probe_wheel(tmp_path)
    assert wheel.name == "orion_parity_probe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        record = archive.read("orion_parity_probe-1.0.dist-info/RECORD").decode().splitlines()
        for line in record:
            path, digest, size = line.split(",")
            if digest:
                data = archive.read(path)
                expected = parity.base64.urlsafe_b64encode(
                    parity.hashlib.sha256(data).digest()).rstrip(b"=").decode()
                assert digest == f"sha256={expected}" and int(size) == len(data)
        assert parity.PIP_PROBE_VALUE in archive.read("orion_parity_probe/__init__.py").decode()


def test_summary_and_rendering():
    checks = [parity.Check("a", True, "fine"), parity.Check("b", False, "broken",
                                                            data={"failures": ["why"]})]
    assert parity.summary_line(checks) == "FAILED: 1"
    assert parity.summary_line(checks[:1]) == "ALL OK"
    lines = parity.render(checks, verbose=True)
    assert lines[1].startswith("FAIL  b") and lines[2].strip() == "- why"


# -- launch-based checks against tiny fake children (no exe) ------------------

FAKE_MCP = r'''
import json, sys, time
mode = sys.argv[1]
if mode in ("crash", "crash-hang"):
    sys.stderr.write(TRACEBACK)
    sys.stderr.flush()
    if mode == "crash":
        sys.exit(1)             # orion.py's _headless guard: traceback on fd 2, exit 1
    time.sleep(60)              # a windowed exe without the guard: a dialog that waits
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {
            "protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake"}}}),
            flush=True)
    elif message.get("method") == "tools/list":
        print("a log line on stdout", flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                          "result": {"tools": [{"name": "a", "inputSchema": {}}, {"name": "b"}]}}),
              flush=True)
if mode == "linger":
    time.sleep(60)
'''


def _fake(tmp_path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text("TRACEBACK = " + repr(WORLD_TRACEBACK) + "\n" + text, encoding="utf-8")
    return path


def _mcp(tmp_path, mode: str, **kwargs):
    script = _fake(tmp_path, "fake_mcp.py", FAKE_MCP)
    return parity.check_mcp("orion_world", [sys.executable, str(script), mode], "fake",
                            env=parity.child_env(tmp_path / "config"), protocol="2024-11-05",
                            supported=frozenset({"2024-11-05"}), timeout=20, **kwargs)


def test_mcp_handshake_counts_tools_from_a_working_server(tmp_path):
    check = _mcp(tmp_path, "ok")
    assert check.ok and check.detail.startswith("2 tool(s)"), check.detail


def test_mcp_failure_reports_the_childs_traceback_and_exit_code(tmp_path):
    check = _mcp(tmp_path, "crash")
    assert not check.ok
    assert "exit 1: ModuleNotFoundError: No module named 'orion_core.world_feeds'" in check.detail


def test_a_crashed_server_holding_a_dialog_fails_fast_and_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(parity, "CRASH_GRACE_S", 0.5)
    started = time.monotonic()
    check = _mcp(tmp_path, "crash-hang")
    assert time.monotonic() - started < 15
    assert not check.ok and "crashed during startup" in check.detail
    assert check.data["report"]["returncode"] is not None          # killed, not left waiting


def test_a_server_that_ignores_stdin_closing_fails_and_is_killed(tmp_path):
    check = _mcp(tmp_path, "linger", exit_timeout=1.0)
    assert not check.ok and "did not exit within 1s" in check.detail


FAKE_HOST = r'''
import json, os, sys, time
mode, pidfile = sys.argv[1], sys.argv[2]
open(pidfile, "w").write(str(os.getpid()))
if mode in ("crash", "crash-hang"):
    sys.stderr.write(TRACEBACK)
    sys.stderr.flush()
    if mode == "crash":
        sys.exit(1)
    time.sleep(60)
def say(message):
    sys.stdout.write("@@" + json.dumps(message) + "\n")
    sys.stdout.flush()
say({"t": "wid", "wid": 101, "pid": os.getpid(), "part": "iris"})
say({"t": "wid", "wid": 102, "pid": os.getpid(), "part": "lanes"})
if mode == "replace":
    time.sleep(0.3)
    say({"t": "wid", "wid": 103, "pid": os.getpid(), "part": "iris"})
if mode == "ignore-eof":
    time.sleep(60)
for line in sys.stdin:
    message = json.loads(line)
    if message.get("m") == "check_state" and mode.startswith("state-"):
        say({"t": "reply", "id": message.get("id"), "value": mode[len("state-"):]})
'''


def _host(tmp_path, mode: str, live: set, **kwargs):
    script = _fake(tmp_path, "fake_host.py", FAKE_HOST)
    pidfile = tmp_path / "host.pid"

    def inspect(wid: int) -> dict:
        if wid not in live:
            return {"exists": False}
        return {"exists": True, "class": "Qt6112QWindowToolSaveBits", "top_level": True,
                "pid": int(pidfile.read_text())}

    return parity.host_check("page-host:brain", [sys.executable, str(script), mode, str(pidfile)],
                             env=parity.child_env(tmp_path / "config"), start_limit_s=15,
                             parts=["iris", "lanes"], hold_s=0.8, hard_timeout=20,
                             inspect=inspect, **kwargs)


def test_host_windows_are_embeddable_and_the_host_quits_on_eof(tmp_path):
    check = _host(tmp_path, "ok", {101, 102})
    assert check.ok, check.detail
    assert "iris 0x65" in check.detail and "quit" in check.detail


def test_host_render_state_requires_an_explicit_alive_reply(tmp_path):
    check = _host(tmp_path, "state-alive", {101, 102}, state_query=True, page_wait_s=0.4)
    assert check.ok and "; page alive;" in check.detail, check.detail


@pytest.mark.parametrize("mode, reason", [
    ("state-loading", "last state: loading"),
    ("ok", "did not answer render-state queries within 0.4s"),
    ("state-failed", "failed to render (WebGL / three.js)"),
    ("state-unknown", "last state: unknown"),
])
def test_host_windows_cannot_pass_without_a_ready_render_state(tmp_path, mode, reason):
    check = _host(tmp_path, mode, {101, 102}, state_query=True, page_wait_s=0.4)
    assert not check.ok and reason in check.detail, check.detail


def test_a_replaced_window_must_be_announced_again(tmp_path):
    check = _host(tmp_path, "replace", {102, 103})       # 101 died, 103 announced: fine
    assert check.ok and "1 replaced and re-announced" in check.detail, check.detail
    check = _host(tmp_path, "ok", {102})                 # 101 died, nothing announced
    assert not check.ok and "no longer a window" in check.detail


def test_a_crashing_host_reports_its_traceback(tmp_path, monkeypatch):
    check = _host(tmp_path, "crash", set())
    assert not check.ok and "exit 1: ModuleNotFoundError" in check.detail
    monkeypatch.setattr(parity, "CRASH_GRACE_S", 0.5)
    started = time.monotonic()
    check = _host(tmp_path, "crash-hang", set())
    assert not check.ok and "crashed before reporting" in check.detail
    assert time.monotonic() - started < 15


def test_a_host_that_ignores_stdin_closing_fails(tmp_path):
    check = _host(tmp_path, "ignore-eof", {101, 102}, exit_timeout=1.0)
    assert not check.ok and "did not quit within 1s" in check.detail


def test_run_bounded_kills_on_timeout_and_on_a_silent_crash(tmp_path):
    run = parity.run_bounded([sys.executable, "-c", "import time; time.sleep(30)"],
                             cwd=tmp_path, timeout=1.0)
    assert run.timed_out and run.returncode is not None
    script = _fake(tmp_path, "crash.py",
                   "import sys, time\nsys.stderr.write(TRACEBACK)\nsys.stderr.flush()\n"
                   "time.sleep(30)\n")
    started = time.monotonic()
    run = parity.run_bounded([sys.executable, str(script)], cwd=tmp_path, timeout=30,
                             crash_grace_s=0.5)
    assert time.monotonic() - started < 15
    assert run.error.startswith("crashed: ModuleNotFoundError") and run.returncode is not None
    script = _fake(tmp_path, "exit1.py", "import sys\nsys.stderr.write(TRACEBACK)\nsys.exit(1)\n")
    run = parity.run_bounded([sys.executable, str(script)], cwd=tmp_path, timeout=30)
    assert parity._run_failure(run).startswith("exit 1: ModuleNotFoundError")


FAKE_ORION = r'''
import os, runpy, sys
sys.path.insert(0, os.environ["FAKE_ORION_PACKAGE"])
flag, script = sys.argv[1], sys.argv[2]
assert flag == "--forge-verify"
from orion_core import forge_runtime
forge_runtime.add_packages_to_path()
sys.argv = [script, *sys.argv[3:]]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
'''

FAKE_FORGE_RUNTIME = r'''
import os, sys
from pathlib import Path
def packages_dir():
    return Path(os.environ["ORION_CONFIG_DIR"]) / "forge_packages"
def external_python():
    return sys.executable
def add_packages_to_path():
    if packages_dir().is_dir() and str(packages_dir()) not in sys.path:
        sys.path.insert(0, str(packages_dir()))
def pip_install_command(packages):
    if os.environ.get("FAKE_TARGET") == "none":     # source mode: installs into this Python
        return [sys.executable, "-m", "pip", "install", *packages]
    return [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(packages_dir()),
            *packages]
'''


def _fake_runtime(tmp_path, monkeypatch, target: str):
    package = tmp_path / "fake_pkg" / "orion_core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "forge_runtime.py").write_text(FAKE_FORGE_RUNTIME, encoding="utf-8")
    launcher = tmp_path / "fake_orion.py"
    launcher.write_text(FAKE_ORION, encoding="utf-8")
    monkeypatch.setenv("FAKE_ORION_PACKAGE", str(package.parent))
    monkeypatch.setenv("FAKE_TARGET", target)
    flag = "--forge-verify"
    return parity.FrozenRunner([sys.executable, str(launcher)], {flag: True}, flag), flag


def test_pip_check_refuses_any_install_outside_the_temporary_config(tmp_path, monkeypatch):
    runner, flag = _fake_runtime(tmp_path, monkeypatch, target="none")
    check = parity.pip_check(runner, {flag: True}, flag, tmp_path / "pip")
    assert not check.ok and "refusing" in check.detail
    assert "returncode" not in check.data["result"]                  # pip never ran


def test_pip_check_installs_the_local_wheel_offline_and_imports_it_fresh(tmp_path, monkeypatch):
    runner, flag = _fake_runtime(tmp_path, monkeypatch, target="config")
    check = parity.pip_check(runner, {flag: True}, flag, tmp_path / "pip")
    assert check.ok, check.detail
    assert "--no-index" not in " ".join(check.data["result"]["command"])  # the Forge's own command
    landed = tmp_path / "pip" / "config" / "forge_packages" / "orion_parity_probe" / "__init__.py"
    assert landed.is_file()


def test_forge_checks_stage_and_run_like_the_sandbox(tmp_path, monkeypatch):
    """The conformance probe + pytest wrapper, through a launcher that handles
    --forge-verify the way orion.py does (run_script runs the script as __main__)."""
    runner, flag = _fake_runtime(tmp_path, monkeypatch, target="config")
    checks = parity.forge_checks(runner.launcher, {flag: True}, flag, ROOT, ROOT, tmp_path / "forge")
    assert [check.name for check in checks] == ["forge:conformance", "forge:pytest"]
    assert all(check.ok for check in checks), [check.detail for check in checks]


def test_modes_a_build_lacks_are_reported_and_never_launched(tmp_path):
    """The 8 Oct build has only --mcp-server: launching --forge-verify or a host
    mode on it would start the full app, so those checks must not launch."""
    archive = parity.Archive({"--mcp-server"}, {"freeze_support"}, set())
    checks, _info = parity.run_checks(tmp_path / "ORION.exe", ROOT, groups=("modes", "face", "forge"),
                                      work_root=tmp_path, archive=archive,
                                      launcher=[str(tmp_path / "never-launched.exe")])
    named = {check.name: check for check in checks}
    assert set(named) >= {"modes", "face-host", "page-host:web", "page-host:brain",
                          "forge:conformance", "forge:pytest"}
    for name in ("face-host", "page-host:web", "page-host:brain", "forge:conformance"):
        assert not named[name].ok and named[name].detail.startswith("not in this build"), name
    assert not any("could not start" in check.detail for check in checks)


def test_main_fails_cleanly_for_a_missing_executable(tmp_path, capsys):
    assert parity.main([str(tmp_path / "absent" / "ORION.exe")]) == 2
    assert capsys.readouterr().out.strip().endswith("FAILED: 1")


# ── build_standalone: the parity check is the build's last step ─────────────

@pytest.mark.parametrize("argv,gated", [([], True), (["--skip-parity-check"], False)])
def test_skip_parity_check_is_the_only_way_past_the_gate(monkeypatch, capsys, tmp_path, argv, gated):
    """The parity gate is the build's last step; only an explicit flag skips it
    (and says the build is unverified). Nothing is frozen or launched here."""
    import types

    import build_standalone
    (tmp_path / "orion.py").write_text("# synthetic\n", encoding="utf-8")
    exe = tmp_path / "dist" / "ORION" / "ORION.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"synthetic")
    for name, value in {"BASE": tmp_path, "ENTRY": tmp_path / "orion.py",
                        "ICON": tmp_path / "absent.ico",
                        "_has_private_config": lambda _: False,
                        "_ensure_pyinstaller": lambda: True, "_ensure_icon": lambda: None,
                        "_ssl_runtime_inputs": lambda: ([], []),
                        "_write_version_file": lambda: None, "_write_build_info": lambda: None,
                        "_prune_unused_qt_data": lambda _: None,
                        "_seed_config": lambda _: None}.items():
        monkeypatch.setattr(build_standalone, name, value)
    monkeypatch.setattr(build_standalone.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0))
    gate = []
    monkeypatch.setattr(parity, "run_bounded", lambda command, **kwargs: (
        gate.append(command), parity.RunResult(1, "FAILED: 1\n", "", 0.01))[1])
    assert build_standalone.build(argv) == (1 if gated else 0)
    assert bool(gate) is gated
    if not gated:
        assert "SKIPPED (--skip-parity-check)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        build_standalone.build(["--no-such-flag"])
    assert "build(sys.argv[1:])" in (ROOT / "build_standalone.py").read_text(encoding="utf-8")


# ── the offline voice (Kokoro) and the GPU runtime in the build, 18 Oct ─────

def _tiny_kokoro(folder: Path, monkeypatch) -> dict:
    """Kokoro's two model files as a tiny pinned spec (the real ones are 354 MB)."""
    import hashlib
    from orion_core import voice_kokoro
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, body in ((voice_kokoro.MODEL_FILE, b"model"), (voice_kokoro.VOICES_FILE, b"voices")):
        (folder / name).write_bytes(body)
        files[name] = voice_kokoro.ModelFile(len(body), hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(voice_kokoro, "MODEL",
                        voice_kokoro.ModelSpec("kokoro-v1.0", "https://example.invalid", files))
    return files


def test_the_build_collects_the_offline_voice_and_the_check_proves_it():
    config = parity.build_config(ROOT)
    assert {"kokoro_onnx", "espeakng_loader", "phonemizer"} <= set(config["collect_all"])
    patterns = [pattern for pattern, _why in parity.CRITICAL_DATA]
    for wanted in ("assets/models/kokoro-v1.0/kokoro-v1.0.onnx",
                   "assets/models/kokoro-v1.0/voices-v1.0.bin",
                   "espeakng_loader/espeak-ng-data/phontab", "onnxruntime/capi/DirectML.dll"):
        assert wanted in patterns
    assert {"kokoro-voice", "onnxruntime-directml"} <= set(parity.DATA_PROBES)
    assert "self_check(require_bundled=True)" in parity.IMPORT_PROBE
    assert '"DmlExecutionProvider" not in providers' in parity.IMPORT_PROBE
    # The probe's imports are recorded as lazy imports the frozen app must resolve.
    needs = parity.lazy_imports(ROOT)
    assert {"kokoro_onnx", "kokoro_onnx.tokenizer", "espeakng_loader"} <= set(needs)


def test_a_bundle_without_the_offline_voice_fails_by_name(tmp_path):
    internal = tmp_path / "_internal"

    def place(skip: str = "") -> None:
        for pattern, _why in parity.CRITICAL_DATA:
            if skip and skip in pattern:
                continue
            path = internal / pattern.replace("*", "x")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    place(skip="kokoro")
    check = parity.check_critical(internal, tmp_path)
    assert not check.ok and "kokoro-v1.0.onnx" in check.detail and "voices-v1.0.bin" in check.detail
    place()
    assert parity.check_critical(internal, tmp_path).ok


def test_the_offline_voice_model_is_bundled_only_when_its_pinned_hashes_match(
        tmp_path, monkeypatch):
    import build_standalone
    from orion_core import voice_kokoro
    folder = tmp_path / "kokoro-v1.0"
    files = _tiny_kokoro(folder, monkeypatch)
    datas, note = build_standalone._kokoro_datas(str(folder))
    assert datas == [(folder / name, "assets/models/kokoro-v1.0") for name in files]
    assert "SHA-256 checked" in note
    (folder / voice_kokoro.VOICES_FILE).write_bytes(b"VOICES")      # same size, other bytes
    datas, note = build_standalone._kokoro_datas(str(folder))
    assert datas == [] and "SHA-256" in note and "NOT bundled" in note
    datas, note = build_standalone._kokoro_datas(str(tmp_path / "absent"))
    assert datas == [] and "--kokoro-dir" in note
    # Where the build puts it is where the frozen app looks (RESOURCE_DIR/assets/models).
    assert build_standalone.KOKORO_DEST == f"assets/models/{voice_kokoro.MODEL.name}"


def test_the_offline_voice_self_check_needs_espeak_and_the_bundled_files(tmp_path, monkeypatch):
    """What the frozen probe runs (voice_kokoro.self_check), here from source:
    espeak-ng really makes phonemes, and only the bundled copy counts."""
    pytest.importorskip("kokoro_onnx")
    pytest.importorskip("espeakng_loader")
    from orion_core import voice_kokoro
    bundle = tmp_path / "bundle"
    folder = bundle / "assets" / "models" / "kokoro-v1.0"
    _tiny_kokoro(folder, monkeypatch)
    monkeypatch.setattr("orion_core.constants.RESOURCE_DIR", bundle)
    note = voice_kokoro.self_check(require_bundled=True)
    assert str(folder) in note and "phonemes" in note
    (folder / voice_kokoro.MODEL_FILE).unlink()
    with pytest.raises(RuntimeError, match="kokoro-v1.0.onnx is missing"):
        voice_kokoro.self_check(require_bundled=True)
