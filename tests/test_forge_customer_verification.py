"""Customer candidates cannot be installed or imported into the host."""
import json
from types import SimpleNamespace

import pytest

from orion_core.forge import ForgeOrchestrationManager, ForgeSession
from orion_core.forge_isolation import customer_verification_mode, save_review_candidate
from orion_core.sandbox import VerificationOutcome


SOURCE = """def get_tool_schema():
    return {'name': 'demo', 'description': 'demo', 'parameters': {'type': 'object', 'properties': {}}}
def run():
    return 'ok'
"""
BUS = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))


@pytest.fixture
def customer(tmp_path, monkeypatch):
    from orion_core import forge
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "1")
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "container")
    monkeypatch.setattr(forge, "_custom_tools_dir", lambda: tmp_path / "custom_tools")
    manager = ForgeOrchestrationManager(BUS, code_generator=lambda *_: (SOURCE, "assert True", []))
    manager.sandbox.staging_dir = tmp_path / "staging"
    manager.sandbox.staging_dir.mkdir()

    async def forbidden(*_):
        pytest.fail("customer verification must not install, import or activate generated code")
    manager.resolver.resolve_and_install = forbidden
    manager.loader.load_and_register = forbidden
    return manager


async def test_customer_mode_requires_container_before_generation(customer, monkeypatch):
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    customer.code_generator = lambda *_: pytest.fail("unsafe policy must fail before spending model tokens")
    result = await customer.forge_tool("demo", "demo")
    assert not result.ok and "container" in result.text


async def test_customer_mode_rejects_generated_requirements_before_execution(customer):
    customer.code_generator = lambda *_: (SOURCE, "assert True", ["requests"])
    customer.sandbox.verify_tool = lambda *_: pytest.fail("requirements are not installed for customer candidates")
    result = await customer.forge_tool("demo", "demo")
    assert not result.ok and "generated dependencies" in result.text


async def test_verified_customer_candidate_is_inert_and_not_live(customer, tmp_path):
    async def verified(*_):
        return VerificationOutcome(True, "demo", stage="test", isolation="container", typecheck="passed")
    customer.sandbox.verify_tool = verified
    result = await customer.forge_tool("demo", "demo")
    assert result.ok and "not live" in result.text
    assert result.evidence[0]["activated"] is False
    candidates = list((tmp_path / "forge_candidates").glob("*.json"))
    assert len(candidates) == 1 and not (tmp_path / "custom_tools").exists()
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    assert data["source"] == SOURCE and data["tests"] == "assert True"
    assert data["activated"] is False and data["dependencies_installed"] is False
    assert data["verification"]["isolation"] == "container"
    assert len(data["source_sha256"]) == 64
    assert next(iter(customer.sessions.values())).activation_ok is False


@pytest.mark.parametrize("isolation,typecheck", [("process", "passed"), ("container", "disabled"), ("container", "unavailable")])
async def test_customer_candidate_needs_actual_required_checks(customer, isolation, typecheck, tmp_path):
    async def false_claim(*_):
        return VerificationOutcome(True, "demo", isolation=isolation, typecheck=typecheck)
    customer.sandbox.verify_tool = false_claim
    result = await customer.forge_tool("demo", "demo")
    assert not result.ok
    assert not (tmp_path / "custom_tools").exists() and not (tmp_path / "forge_candidates").exists()


async def test_customer_missing_dependency_never_triggers_host_install(customer):
    async def missing(*_):
        return VerificationOutcome(False, "demo", error_log=["ModuleNotFoundError: No module named 'requests'"])
    customer.sandbox.verify_tool = missing
    result = await customer.forge_tool("demo", "demo")
    assert not result.ok and "generated dependencies on the host" in result.text


async def test_direct_activation_refused_before_source_publication(customer, tmp_path):
    target = tmp_path / "demo_tool.py"
    with pytest.raises(ValueError, match="does not import"):
        await customer._activate_verified(ForgeSession("one", "demo", tool_code=SOURCE), target)
    assert not target.exists()


async def test_customer_startup_never_reloads_or_rehabilitates_generated_tools(customer, tmp_path):
    customer.dispatcher = SimpleNamespace()
    folder = tmp_path / "custom_tools"
    folder.mkdir()
    path = folder / "demo_tool.py"
    path.write_text("raise AssertionError('must not execute')", encoding="utf-8")
    async def forbidden(*_):
        pytest.fail("customer startup must not restore and import generated modules")
    customer.loader.rehabilitate_quarantined = forbidden
    assert await customer.reload_persisted_tools() == 0
    assert path.read_text(encoding="utf-8").startswith("raise AssertionError")
    assert "verification-only" in customer.health().text


@pytest.mark.parametrize("value", ["yes", "false", "", "2", "on"])
def test_malformed_customer_policy_fails_closed(value, monkeypatch):
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", value)
    with pytest.raises(ValueError):
        customer_verification_mode()


def test_review_receipt_names_cannot_escape_directory(tmp_path):
    with pytest.raises(ValueError):
        save_review_candidate(tmp_path, "../../escape", SOURCE, "pass", {})


async def test_policy_enabling_during_verification_prevents_host_activation(customer, monkeypatch, tmp_path):
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    async def changed(*_):
        monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "1")
        return VerificationOutcome(True, "demo", isolation="process", typecheck="passed")
    customer.sandbox.verify_tool = changed
    result = await customer.forge_tool("demo", "demo")
    assert not result.ok and not (tmp_path / "custom_tools").exists()
