"""Local planning tools, loaded on first use and run away from the UI loop."""

from __future__ import annotations

import asyncio
from typing import Any

from .data import ToolResult


class PlanningDispatchMixin:
    async def project_rehearsal_tool(self, args: dict[str, Any]) -> ToolResult:
        from .project_rehearsal import run_rehearsal

        return await asyncio.to_thread(run_rehearsal, args)

    async def context_capsule_tool(self, args: dict[str, Any]) -> ToolResult:
        def run() -> ToolResult:
            from .context_capsules import ContextCapsuleStore

            store = getattr(self, "_context_capsules", None)
            if store is None:
                store = ContextCapsuleStore()
                self._context_capsules = store
            return store.handle(args)

        return await asyncio.to_thread(run)

    async def experiment_notebook_tool(self, args: dict[str, Any]) -> ToolResult:
        def run() -> ToolResult:
            from .experiment_notebook import ExperimentNotebook

            store = getattr(self, "_experiment_notebook", None)
            if store is None:
                store = ExperimentNotebook()
                self._experiment_notebook = store
            return store.handle(args)

        return await asyncio.to_thread(run)
