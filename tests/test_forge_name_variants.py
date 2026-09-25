"""
Regression tests for the 2026-07-17 forge failure: a tool named in CamelCase
('EnhancedResearchModule') whose generated test imports it in snake_case
('enhanced_research_module') must NOT be misread as a missing pip package.

Root cause and fix:
* The sandbox only staged the module under its verbatim name, so
  `import enhanced_research_module` raised ModuleNotFoundError.  It now stages
  every name variant (utils.module_name_variants), so the import resolves.
* The forge's 'is this the tool's own name?' guard compared raw strings, so
  the snake_case form slipped through to pip.  It now compares
  case/separator-insensitive canonical ids (utils.canonical_module_id).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.forge import ForgeOrchestrationManager
from orion_core.forge import _max_attempts as forge_attempts
from orion_core.sandbox import SandboxVerificationHarness
from orion_core.utils import canonical_module_id, module_name_variants


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, *_a, **_k):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


# ── the pure helpers ────────────────────────────────────────────────────────


def test_canonical_module_id_is_case_and_separator_insensitive():
    assert canonical_module_id("EnhancedResearchModule") == "enhancedresearchmodule"
    assert canonical_module_id("enhanced_research_module") == "enhancedresearchmodule"
    assert canonical_module_id("enhanced-research-module") == "enhancedresearchmodule"
    assert canonical_module_id("") == ""


def test_module_name_variants_covers_camel_and_snake():
    variants = module_name_variants("EnhancedResearchModule")
    assert "EnhancedResearchModule" in variants
    assert "enhanced_research_module" in variants           # the failing import
    assert "EnhancedResearchModule_tool" in variants
    assert "enhanced_research_module_tool" in variants


# ── the sandbox stages every alias ──────────────────────────────────────────


def test_sandbox_stages_snake_case_alias(tmp_path):
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    harness._write_artefacts(
        "EnhancedResearchModule",
        "def run(**kw):\n    return 'ok'\n",
        "print('test')\n",
    )
    staged = {p.name for p in tmp_path.glob("*.py")}
    # The snake_case import the generated test actually used now resolves.
    assert "enhanced_research_module.py" in staged
    assert "EnhancedResearchModule.py" in staged
    assert "EnhancedResearchModule_tool.py" in staged
    assert "EnhancedResearchModule_test.py" in staged


# ── the forge never sends a self-import to pip ──────────────────────────────


class _Outcome:
    def __init__(self, passed, errors=()):
        self.passed = passed
        self.error_log = list(errors)


class _DepOutcome:
    def __init__(self, ok=True):
        self.succeeded = ok
        self.installed_packages = []
        self.error_log = []


def _forge_with_recording_resolver(sandbox_results, fixer_calls):
    forge = ForgeOrchestrationManager(
        _StubBus(),
        code_generator=lambda name, plan: ("code", "test", []),
        llm_fixer=lambda code, errors: fixer_calls.append(errors) or "fixed code",
    )
    results = list(sandbox_results)

    class _Sandbox:
        async def verify_tool(self, code, test, name):
            return results.pop(0)

    class _Resolver:
        def __init__(self):
            self.installed = []

        async def resolve_and_install(self, requirements):
            self.installed.append(list(requirements))
            return _DepOutcome(True)

    forge.sandbox = _Sandbox()
    forge.resolver = _Resolver()
    return forge


def test_self_import_snake_case_routes_to_fixer_not_pip():
    # The CamelCase tool's test imports the snake_case form and fails.  This is
    # the tool itself — it must reach the LLM fixer, and pip must never be asked
    # to install 'enhanced_research_module'.
    fixer_calls: list = []
    # One outcome per permitted code attempt, so the session fails by
    # exhausting its repair budget rather than by running the stub dry.
    forge = _forge_with_recording_resolver(
        [
            _Outcome(False, ["ModuleNotFoundError: No module named 'enhanced_research_module'"])
            for _ in range(forge_attempts())
        ],
        fixer_calls,
    )
    result = asyncio.run(forge.forge_tool("EnhancedResearchModule", "research helper"))
    assert not result.ok                                   # genuinely broken module
    # pip was never asked for the tool's own name in any form.
    flat = [pkg for call in forge.resolver.installed for pkg in call]
    assert not any(canonical_module_id(p) == canonical_module_id("EnhancedResearchModule")
                   for p in flat), flat
    # The LLM self-heal was attempted instead.
    assert fixer_calls, "expected the self-import failure to reach the LLM fixer"
