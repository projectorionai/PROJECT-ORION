"""Every shipped plugin must satisfy the actual runtime loading contract."""
import asyncio
from pathlib import Path

from orion_core.bus import OrionBus
from orion_core.dynamic_loader import ReflectiveModuleLoader
from tools.prepare_public_source import PUBLIC_CONFIG_FILES


def test_all_public_plugins_load_without_installing_dependencies(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    loader = ReflectiveModuleLoader(OrionBus())

    async def no_install(package):
        raise AssertionError(f"Loading a shipped plugin must not install {package}")

    monkeypatch.setattr(loader, "_install", no_install)

    async def check():
        names = sorted(p for p in PUBLIC_CONFIG_FILES if p.endswith("_tool.py"))
        assert names
        for name in names:
            result = await loader.load_and_register(root / name)
            assert result.succeeded, (name, result.error_log)
            assert callable(result.handler_fn)

    asyncio.run(check())
