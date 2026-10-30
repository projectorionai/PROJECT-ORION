"""
Test fixtures a forged tool's verification can rely on (Forge Mark III, W4.2).

Why this exists
---------------
The lesson corpus (config/self_improvement/forge_lessons.jsonl, 16 Oct 2026)
shows what generated tests kept doing wrong when they had nothing to lean on:

    filesystem  — 18 hits: FileNotFoundError / PermissionError on real paths
                  ('C:\\Users\\...', the ORION project folder, a JSON file the
                  test assumed existed)
    network     — tests calling the live internet from an offline verifier
    clock       — exact-string assertions on output that embedded "now"

A model asked to write "an offline, deterministic test" with no tools for it
writes one that touches the real disk and the real network, because that is
the only way it knows to exercise the code. So the verifier now hands it the
tools: a private working folder, a fake HTTP layer, a time freezer and a fake
ORION bus, staged beside every verification as ``orion_forge_fixtures.py``.

The same module also carries ``coerce_arguments`` and ``run_examples``, which
check a tool against the three example calls its plan promised (W4.3), so the
in-process Forge and the sandboxed child apply ONE implementation.

Why the source is a string
--------------------------
The sandboxed child cannot import ``orion_core`` (generated code must never
reach ORION's own package), so this code travels into the staging folder as
text. ``forge_contract.py`` does the same and needed a DATAS entry in
build_standalone.py, because a PyInstaller build keeps only bytecode and
``inspect.getsource`` then fails. Holding the source as a constant and
executing it here makes the text available in every build with no packaging
rule to forget; this module's own functions are that same text, compiled once.
Standard library only, no relative imports — it must run standalone.
"""

from __future__ import annotations

FIXTURE_MODULE_NAME = "orion_forge_fixtures"

_SOURCE = r'''
"""ORION Forge verification fixtures (staged as orion_forge_fixtures.py).

    from orion_forge_fixtures import tmp_workdir, make_file, fake_http, freeze_time, FakeBus

    with tmp_workdir({"notes.txt": "hello"}) as folder:   # private folder, cwd inside it
        make_file(folder / "big.bin", size_bytes=5_000_000)
        with fake_http({"api.example.com/weather": {"temp": 12}}):
            out = mytool.run(path=str(folder))
    with freeze_time("2026-01-15T12:00:00"):
        ...
"""

import contextlib
import datetime as _datetime_module
import io
import json as _json
import os
import pathlib
import re
import shutil
import socket
import sys
import tempfile
import time as _time_module
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["NetworkBlocked", "tmp_workdir", "make_file", "fake_http", "block_network",
           "freeze_time", "FakeBus", "FakeResponse", "coerce_arguments", "check_result",
           "run_examples", "validate_plan", "plan_fingerprint", "requirement_identity"]


class NetworkBlocked(ConnectionError):
    """Raised for any real network access during verification."""


# ── a private working folder ─────────────────────────────────────────────────

@contextlib.contextmanager
def tmp_workdir(files=None):
    """A temporary folder inside the verification directory, made current.

    *files* maps relative paths to str (written as UTF-8) or bytes. The folder
    and everything in it is removed afterwards, and the previous working
    directory restored, however the block exits.
    """
    try:
        base = tempfile.mkdtemp(prefix="fx_", dir=os.getcwd())
    except OSError:
        # The container verifier mounts the verification folder read-only;
        # its writable place is the system temporary folder.
        base = tempfile.mkdtemp(prefix="fx_")
    previous = os.getcwd()
    try:
        for relative, content in (files or {}).items():
            relative = str(relative)
            # Validate before mkdir/open: an absolute key ignores base in
            # os.path.join(), and '..' could overwrite the verifier's files.
            windows_path = pathlib.PureWindowsPath(relative)
            if pathlib.Path(relative).is_absolute() or windows_path.drive or windows_path.root:
                raise ValueError("fixture file paths must stay inside the temporary folder")
            target = (pathlib.Path(base) / relative).resolve()
            try:
                target.relative_to(pathlib.Path(base).resolve())
            except ValueError:
                raise ValueError("fixture file paths must stay inside the temporary folder") from None
            os.makedirs(os.path.dirname(target) or base, exist_ok=True)
            if isinstance(content, bytes):
                with open(target, "wb") as handle:
                    handle.write(content)
            else:
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(str(content))
        os.chdir(base)
        yield pathlib.Path(base)
    finally:
        os.chdir(previous)
        shutil.rmtree(base, ignore_errors=True)


def make_file(path, size_bytes=0, text=None):
    """Create *path* (parents included) holding *text*, or *size_bytes* bytes.

    Sized files are sparse where the filesystem allows, so a disk-usage tool
    can be tested with a "5 GB" file that costs nothing.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        target.write_text(str(text), encoding="utf-8")
        return target
    with open(target, "wb") as handle:
        if size_bytes:
            handle.seek(int(size_bytes) - 1)
            handle.write(b"\0")
    return target


# ── fake HTTP ────────────────────────────────────────────────────────────────

class FakeResponse:
    """Quacks like requests.Response, httpx.Response and urlopen's result."""

    def __init__(self, body="", status=200, headers=None, url=""):
        if isinstance(body, (dict, list)):
            body = _json.dumps(body)
            headers = dict(headers or {}, **{"Content-Type": "application/json"})
        self.content = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.text = self.content.decode("utf-8", errors="replace")
        self.status_code = self.status = int(status)
        self.headers = dict(headers or {})
        self.url = url
        self.ok = 200 <= self.status_code < 400
        self.reason = "OK" if self.ok else "Error"
        self.encoding = "utf-8"
        self._stream = io.BytesIO(self.content)

    def json(self, **_kwargs):
        return _json.loads(self.text)

    def read(self, *args):
        return self._stream.read(*args)

    def getcode(self):
        return self.status_code

    def info(self):
        return self.headers

    def iter_content(self, chunk_size=8192, **_kwargs):
        for start in range(0, len(self.content), chunk_size or 8192):
            yield self.content[start:start + (chunk_size or 8192)]

    def raise_for_status(self):
        if not self.ok:
            raise urllib.error.HTTPError(self.url, self.status_code, self.reason, self.headers, None)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Recorder:
    def __init__(self, responses, default):
        self.responses = list((responses or {}).items())
        self.default = default
        self.calls = []

    def answer(self, method, url):
        url = str(url)
        self.calls.append((str(method).upper(), url))
        for pattern, body in self.responses:
            text = str(pattern)
            hit = text in url
            if not hit:
                try:
                    hit = re.search(text, url) is not None
                except re.error:
                    hit = False
            if hit:
                return self._response(body, url)
        if self.default is not None:
            return self._response(self.default, url)
        raise NetworkBlocked(f"no real network during verification: {method} {url} "
                             "(give fake_http a response for it)")

    @staticmethod
    def _response(body, url):
        if isinstance(body, FakeResponse):
            body.url = body.url or url
            return body
        if isinstance(body, Exception):
            raise body
        if isinstance(body, tuple) and len(body) == 2 and isinstance(body[0], int):
            return FakeResponse(body[1], status=body[0], url=url)
        return FakeResponse(body, url=url)


def _blocked(*_args, **_kwargs):
    raise NetworkBlocked("no real network during verification")


@contextlib.contextmanager
def block_network():
    """Refuse every socket connection inside the block."""
    saved = (socket.create_connection, socket.socket.connect, socket.socket.connect_ex)
    socket.create_connection = _blocked
    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked
    try:
        yield
    finally:
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex = saved


@contextlib.contextmanager
def fake_http(responses=None, default=None):
    """Serve canned HTTP responses; any other request raises NetworkBlocked.

    *responses* maps a URL substring (or regex) to a body: str, bytes, a dict
    or list (sent as JSON), ``(status, body)``, a FakeResponse, or an
    exception instance to raise. Patches urllib, and requests/httpx when they
    are importable. Yields a recorder whose ``.calls`` lists (method, url).
    """
    recorder = _Recorder(responses, default)
    restore = []

    def patch(owner, name, value):
        restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def urlopen(request, *args, **kwargs):
        url = getattr(request, "full_url", request)
        method = getattr(request, "get_method", lambda: "GET")()
        return recorder.answer(method, url)

    patch(urllib.request, "urlopen", urlopen)
    try:
        import requests
        def _verb(method):
            return lambda url, *a, **k: recorder.answer(method, url)
        for verb in ("get", "post", "put", "patch", "delete", "head", "options"):
            patch(requests, verb, _verb(verb))
        patch(requests, "request", lambda method, url, *a, **k: recorder.answer(method, url))
        patch(requests.Session, "request", lambda self, method, url, *a, **k: recorder.answer(method, url))
    except ImportError:
        pass
    try:
        import httpx
        for verb in ("get", "post", "put", "patch", "delete", "head", "options"):
            patch(httpx, verb, (lambda method: lambda url, *a, **k: recorder.answer(method, url))(verb))
        patch(httpx, "request", lambda method, url, *a, **k: recorder.answer(method, url))
        patch(httpx.Client, "request", lambda self, method, url, *a, **k: recorder.answer(method, url))
    except ImportError:
        pass
    try:
        with block_network():
            yield recorder
    finally:
        for owner, name, value in reversed(restore):
            setattr(owner, name, value)


# ── a frozen clock ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def freeze_time(when="2026-01-15T12:00:00"):
    """Pin time.time() and datetime.now()/today()/utcnow() to *when*.

    Accepts an ISO string, a datetime or epoch seconds. Modules that did
    ``from datetime import datetime`` are patched too, so a tool imported
    before the block sees the frozen clock.
    """
    real_datetime = _datetime_module.datetime
    real_date = _datetime_module.date
    if isinstance(when, (int, float)):
        moment = real_datetime.fromtimestamp(float(when))
    elif isinstance(when, real_datetime):
        moment = when
    else:
        moment = real_datetime.fromisoformat(str(when))
    stamp = moment.timestamp()

    class FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            value = moment if tz is None else moment.astimezone(tz)
            return cls.fromtimestamp(value.timestamp(), tz) if tz else cls(*value.timetuple()[:6], value.microsecond)

        @classmethod
        def today(cls):
            return cls.now()

        @classmethod
        def utcnow(cls):
            return cls(*_time_module.gmtime(stamp)[:6])

    class FrozenDate(real_date):
        @classmethod
        def today(cls):
            return cls(moment.year, moment.month, moment.day)

    saved_time = (_time_module.time, getattr(_time_module, "time_ns", None))
    swapped = []
    _time_module.time = lambda: stamp
    if saved_time[1] is not None:
        _time_module.time_ns = lambda: int(stamp * 1_000_000_000)
    _datetime_module.datetime = FrozenDateTime
    _datetime_module.date = FrozenDate
    # The tool and its test sit beside this file in the staging folder; the
    # current directory may be a tmp_workdir below it.
    here = os.path.dirname(os.path.abspath(globals().get("__file__") or os.path.join(os.getcwd(), "x")))
    for module in list(sys.modules.values()):
        source = str(getattr(module, "__file__", "") or "")
        if not source or not os.path.abspath(source).startswith(here):
            continue          # only the tool and its test, never the standard library
        for name, value in list(vars(module).items()):
            if value is real_datetime:
                swapped.append((module, name, value))
                setattr(module, name, FrozenDateTime)
            elif value is real_date:
                swapped.append((module, name, value))
                setattr(module, name, FrozenDate)
    try:
        yield moment
    finally:
        _time_module.time = saved_time[0]
        if saved_time[1] is not None:
            _time_module.time_ns = saved_time[1]
        _datetime_module.datetime = real_datetime
        _datetime_module.date = real_date
        for module, name, value in swapped:
            setattr(module, name, value)


# ── a fake ORION bus ─────────────────────────────────────────────────────────

class FakeSignal:
    def __init__(self):
        self.emitted = []
        self._slots = []

    def emit(self, *payload):
        self.emitted.append(payload[0] if len(payload) == 1 else payload)
        for slot in list(self._slots):
            slot(*payload)

    def connect(self, slot, *_args, **_kwargs):
        self._slots.append(slot)


class FakeBus:
    """Any attribute is a signal: bus.log.emit("x"); bus.log.emitted == ["x"]."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        signal = FakeSignal()
        object.__setattr__(self, name, signal)
        return signal


# ── arguments as the dispatcher will deliver them ────────────────────────────

_TRUE = {"true", "yes", "y", "1", "on"}
_FALSE = {"false", "no", "n", "0", "off", ""}


def _coerce(value, spec):
    kind = str((spec or {}).get("type") or "").lower()
    if value is None or not kind:
        return value
    if kind == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+(?:[.]0*)?", value.strip()):
            # Do not pass IDs through float: integers above 2**53 lose digits.
            return int(value.strip().split(".", 1)[0])
        return value
    if kind == "number":
        if isinstance(value, str) and re.fullmatch(r"\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*", value):
            return float(value)
        return value
    if kind == "boolean":
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        return value
    if kind == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return value
    if kind == "array":
        items = (spec or {}).get("items") or {}
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    value = _json.loads(text)
                except ValueError:
                    return value
            else:
                value = [part.strip() for part in text.split(",") if part.strip()]
        elif isinstance(value, tuple):
            value = list(value)
        elif not isinstance(value, list):
            value = [value]
        return [_coerce(item, items) for item in value]
    if kind == "object" and isinstance(value, str) and value.strip().startswith("{"):
        try:
            return _json.loads(value)
        except ValueError:
            return value
    return value


def coerce_arguments(schema, arguments):
    """Arguments converted to the types *schema* declares, plus what changed.

    The voice channel delivers every number as a float (protobuf Struct) and
    models pass "10" or "true" as strings; ``range(10.0)`` then raises inside
    a tool that is otherwise correct. Unknown keys are kept. Never raises.
    """
    try:
        properties = ((schema or {}).get("parameters") or {}).get("properties") or {}
    except AttributeError:
        properties = {}
    out, notes = {}, []
    for key, value in dict(arguments or {}).items():
        spec = properties.get(key) if isinstance(properties, dict) else None
        try:
            converted = _coerce(value, spec if isinstance(spec, dict) else {})
        except Exception:
            converted = value
        if converted is not value and converted != value or type(converted) is not type(value):
            notes.append(f"{key}: {type(value).__name__} -> {type(converted).__name__}")
        out[key] = converted
    return out, notes


# ── acceptance against the plan's example calls ──────────────────────────────

_FAILED_PREFIXES = ("error:", "error in ", "an error occurred:", "failed to ", "failed:", "could not ")


def requirement_identity(value):
    """Canonical package name and exact optional pin; no extras/URL guessing."""
    if not isinstance(value, str):
        raise ValueError("Forge dependencies must be package requirement strings")
    match = re.fullmatch(
        r"([A-Za-z][A-Za-z0-9_.-]{0,99})(?:[ \t]*(===|~=|==|!=|<=|>=|<|>)[ \t]*([A-Za-z0-9.*+_-]{1,80}))?",
        value.strip())
    if match is None:
        raise ValueError("Unsupported Forge dependency requirement; use a package name and one explicit version constraint")
    name, operator, version = match.groups()
    return re.sub(r"[-_.]+", "-", name).lower(), (operator or "") + (version or "")


def validate_plan(plan):
    """Detach and bound the planner's JSON-only inputs/output/examples contract.

    This does not execute a tool or prove an answer; run_examples does that in
    the verifier. An invalid plan is refused before code generation.
    """
    try:
        encoded = _json.dumps(plan, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Forge plan must be finite JSON data") from exc
    if len(encoded.encode("utf-8")) > 32768:
        raise ValueError("Forge plan exceeds the 32 KiB limit")
    plan = _json.loads(encoded)
    if not isinstance(plan, dict):
        raise ValueError("Forge plan must be an object")
    inputs = plan.get("inputs")
    kinds = {"string", "integer", "number", "boolean", "array", "object"}
    if not isinstance(inputs, dict) or len(inputs) > 24:
        raise ValueError("Forge plan needs an inputs object with at most 24 parameters")
    for name, spec in inputs.items():
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
                or not isinstance(spec, dict) or spec.get("type") not in kinds):
            raise ValueError("Forge plan inputs need names and JSON types")
    required = plan.setdefault("required", [])
    if (not isinstance(required, list) or any(not isinstance(name, str) or name not in inputs
                                             for name in required)):
        raise ValueError("Forge plan required parameters must be declared inputs")
    output = plan.get("output")
    if not isinstance(output, str) or not output.strip() or len(output) > 2000:
        raise ValueError("Forge plan needs a bounded string output description")
    dependencies = plan.get("dependencies")
    if not isinstance(dependencies, list) or len(dependencies) > 12:
        raise ValueError("Forge plan dependencies must be a bounded list of package requirements")
    identities = [requirement_identity(item) for item in dependencies]
    if len({name for name, _constraint in identities}) != len(identities):
        raise ValueError("Forge plan must name each dependency once with its exact version constraint")
    plan["dependencies"] = [item.strip() for item in dependencies]
    examples = plan.get("examples")
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("Forge plan needs exactly three example calls")
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Forge plan examples must be objects")
        arguments = example.get("arguments")
        if (not isinstance(arguments, dict) or set(arguments) - set(inputs)
                or set(required) - set(arguments)):
            raise ValueError("Forge plan example arguments must match the declared inputs")
        pending = list(arguments.values())
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str):
                if "://" in value:
                    url = urllib.parse.urlsplit(value)
                    if url.scheme.lower() in {"http", "https"} and url.netloc:
                        continue
                    raise ValueError("Forge plan examples permit only faked HTTP(S) URLs")
                windows = pathlib.PureWindowsPath(value)
                if (pathlib.PurePosixPath(value).is_absolute() or windows.drive or windows.root
                        or ".." in pathlib.PurePosixPath(value.replace("\\", "/")).parts
                        or value.startswith("~") or re.search(r"%[A-Za-z_][A-Za-z0-9_]*%|\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})", value)):
                    raise ValueError("Forge plan examples must use private {workdir} paths")
        expect = example.get("expect")
        if not isinstance(expect, str) or not expect.strip() or len(expect) > 1000:
            raise ValueError("Every Forge plan example needs an expected answer substring")
        if type(example.get("expect_error", False)) is not bool:
            raise ValueError("Forge plan expect_error must be a boolean")
        for key in ("files", "http"):
            if not isinstance(example.get(key, {}), dict):
                raise ValueError("Forge plan files and http fixtures must be objects")
        for path, content in example.get("files", {}).items():
            windows = pathlib.PureWindowsPath(path)
            if (not path or pathlib.PurePosixPath(path).is_absolute() or windows.drive or windows.root
                    or ".." in pathlib.PurePosixPath(path.replace("\\", "/")).parts
                    or not isinstance(content, str)):
                raise ValueError("Forge plan files must be relative text fixtures inside the work directory")
    if all(example.get("expect_error", False) for example in examples):
        raise ValueError("Forge plan must include a successful example")
    return plan


def plan_fingerprint(plan):
    import hashlib
    text = _json.dumps(validate_plan(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_result(value, *, allow_error=False):
    """A tool result as the dispatcher will judge it: a non-empty string that
    is not an error message (unless *allow_error*)."""
    assert isinstance(value, str), f"run() must return a str, got {type(value).__name__}"
    assert value.strip(), "run() returned an empty string"
    failure = value.lstrip().lower().startswith(_FAILED_PREFIXES)
    if allow_error:
        assert failure, f"run() succeeded for an example that requires a failure: {value[:200]}"
    else:
        assert not failure, (
            f"run() reported failure for a valid example: {value[:200]}")
    return value


def run_examples(module, examples, *, plan=None):
    """Call ``module.run`` with each example the plan promised.

    Each example is {"arguments": {...}, "expect": "<substring, optional>",
    "expect_error": bool, "files": {relpath: text}, "http": {url: body}}.
    Runs inside a private folder with the network blocked (or faked). Raises
    AssertionError naming the example that failed.
    """
    schema = module.get_tool_schema()
    if plan is not None:
        plan = validate_plan(plan)
        parameters = schema.get("parameters") or {}
        actual = parameters.get("properties") or {}
        assert set(actual) == set(plan["inputs"]), "planned inputs differ from the generated schema"
        assert set(parameters.get("required") or []) == set(plan["required"]), (
            "planned required inputs differ from the generated schema")
        for name, spec in plan["inputs"].items():
            assert str(actual[name].get("type", "")).lower() == spec["type"], (
                f"planned type for {name} differs from the generated schema")
        examples = plan["examples"]

    def substitute(value, folder):
        if isinstance(value, str):
            return value.replace("{workdir}", str(folder))
        if isinstance(value, list):
            return [substitute(item, folder) for item in value]
        if isinstance(value, dict):
            return {key: substitute(item, folder) for key, item in value.items()}
        return value

    for index, example in enumerate(examples or [], 1):
        with tmp_workdir(example.get("files") or {}) as folder:
            arguments, _notes = coerce_arguments(schema, substitute(example.get("arguments") or {}, folder))
            label = f"example {index} {arguments!r}"
            with fake_http(example.get("http") or {}):
                try:
                    result = module.run(**arguments)
                except BaseException as exc:
                    raise AssertionError(f"{label} raised {type(exc).__name__}: {exc}") from exc
        try:
            check_result(result, allow_error=bool(example.get("expect_error")))
        except AssertionError as exc:
            raise AssertionError(f"{label}: {exc}") from None
        expected = str(example.get("expect") or "").strip()
        if expected and expected.lower() not in result.lower():
            raise AssertionError(f"{label}: expected the result to mention {expected!r}; got {result[:200]!r}")
        print(f"{label} OK")
'''


def fixture_source() -> str:
    """The text staged into every verification as ``orion_forge_fixtures.py``."""
    return _SOURCE.lstrip("\n")


# One implementation for both sides: compile the shipped text here too.
_namespace: dict = {"__name__": FIXTURE_MODULE_NAME}
exec(compile(_SOURCE, f"<{FIXTURE_MODULE_NAME}>", "exec"), _namespace)  # noqa: S102 - our own constant

NetworkBlocked = _namespace["NetworkBlocked"]
tmp_workdir = _namespace["tmp_workdir"]
make_file = _namespace["make_file"]
fake_http = _namespace["fake_http"]
block_network = _namespace["block_network"]
freeze_time = _namespace["freeze_time"]
FakeBus = _namespace["FakeBus"]
FakeResponse = _namespace["FakeResponse"]
coerce_arguments = _namespace["coerce_arguments"]
check_result = _namespace["check_result"]
run_examples = _namespace["run_examples"]
validate_plan = _namespace["validate_plan"]
plan_fingerprint = _namespace["plan_fingerprint"]
requirement_identity = _namespace["requirement_identity"]

#: What the generation prompt tells the model it may use.
FIXTURE_GUIDE = (
    "TEST FIXTURES (already staged beside the test — import them, never re-implement):\n"
    "  from orion_forge_fixtures import tmp_workdir, make_file, fake_http, freeze_time, FakeBus\n"
    "  • with tmp_workdir({'a.txt': 'hi'}) as folder:  — a private temp folder (cwd inside it);\n"
    "    create every file the test needs here, e.g. make_file(folder / 'big.bin', size_bytes=5_000_000).\n"
    "  • with fake_http({'example.com/api': {'temp': 12}}):  — canned HTTP for requests/httpx/urllib;\n"
    "    any other request raises NetworkBlocked. The verifier has NO real network.\n"
    "  • with freeze_time('2026-01-15T12:00:00'):  — fixed time.time()/datetime.now().\n"
    "  • FakeBus() — a stand-in for ORION's bus (bus.log.emit(...) is recorded).\n"
    "  The test must never use a real absolute path, the user's home folder, the real network,\n"
    "  sleep(), or input()."
)

__all__ = ["FIXTURE_MODULE_NAME", "FIXTURE_GUIDE", "fixture_source", "NetworkBlocked",
           "tmp_workdir", "make_file", "fake_http", "block_network", "freeze_time", "FakeBus",
           "FakeResponse", "coerce_arguments", "check_result", "run_examples", "validate_plan",
           "plan_fingerprint", "requirement_identity"]
