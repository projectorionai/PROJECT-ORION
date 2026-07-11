"""
Reflective Module Loader.

Dynamically loads newly written Python modules from config/custom_tools/
without restarting ORION. Uses importlib to hook modules into memory and
updates the tool declarations array inside OrionDispatcher instantly.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .bus import OrionBus
from .data import ToolResult


@dataclass
class ModuleLoadOutcome:
    """Structured result of reflective module loading."""

    succeeded: bool
    module_name: str
    module_path: Path | None = None
    tool_schema: dict[str, Any] | None = None
    error_log: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    handler_fn: Callable[..., Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "module_name": self.module_name,
            "module_path": str(self.module_path) if self.module_path else None,
            "tool_schema": self.tool_schema,
            "error_log": self.error_log,
            "duration_ms": self.duration_ms,
        }


class ReflectiveModuleLoader:
    """Dynamically loads and registers custom tools without restart."""

    def __init__(self, bus: OrionBus) -> None:
        """
        Initialise the reflective module loader.

        Args:
            bus: OrionBus for logging load states.
        """
        self.bus = bus
        self.custom_tools_dir = Path(__file__).parent.parent / "config" / "custom_tools"
        self.custom_tools_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_modules: dict[str, Any] = {}

    async def load_and_register(
        self,
        module_path: Path,
    ) -> ModuleLoadOutcome:
        """
        Load a Python module and register its tool into the dispatcher.

        Uses importlib.util.spec_from_file_location to dynamically import
        the module. Expects the module to export:
          - get_tool_schema() → dict with OpenAI function schema
          - run(**kwargs) → tool execution endpoint

        Args:
            module_path: Full path to the .py module file

        Returns:
            ModuleLoadOutcome with success status and handler function
        """
        self.bus.log.emit(
            f"[FORGE] Loader: starting load of '{module_path.name}'"
        )
        outcome = ModuleLoadOutcome(
            succeeded=False,
            module_name=module_path.stem,
            module_path=module_path,
        )
        start_time = asyncio.get_event_loop().time()

        try:
            # Load module via thread (I/O operation)
            module = await asyncio.to_thread(
                self._load_module_file,
                module_path,
            )

            # Extract schema and handler
            schema = self._extract_schema(module)
            handler_fn = self._extract_handler(module)

            if schema is None or handler_fn is None:
                outcome.error_log.append(
                    "Module must export get_tool_schema() and run() function"
                )
                self.bus.log.emit(
                    f"[FORGE] Loader: ✗ missing required exports in {module_path.name}"
                )
            else:
                outcome.succeeded = True
                outcome.tool_schema = schema
                outcome.handler_fn = handler_fn
                self.loaded_modules[outcome.module_name] = {
                    "module": module,
                    "schema": schema,
                    "handler": handler_fn,
                }
                self.bus.log.emit(
                    f"[FORGE] Loader: ✓ loaded '{outcome.module_name}' "
                    f"({schema.get('name', 'unnamed')})"
                )

        except SyntaxError as e:
            outcome.error_log = [f"Syntax error: {e}", traceback.format_exc()]
            self.bus.log.emit(
                f"[FORGE] Loader: ✗ syntax error in {module_path.name}: {e}"
            )

        except Exception as e:
            outcome.error_log = [
                f"{type(e).__name__}: {str(e)}",
                traceback.format_exc(),
            ]
            self.bus.log.emit(
                f"[FORGE] Loader: ✗ exception loading {module_path.name}: {e}"
            )

        finally:
            end_time = asyncio.get_event_loop().time()
            outcome.duration_ms = (end_time - start_time) * 1000.0

        return outcome

    def _load_module_file(self, module_path: Path) -> Any:
        """
        Load a Python file as a module using importlib.

        Args:
            module_path: Full path to the .py file

        Returns:
            Loaded module object

        Raises:
            FileNotFoundError: If module_path does not exist
            SyntaxError: If the file contains syntax errors
        """
        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")

        module_name = module_path.stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create module spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except SyntaxError as e:
            del sys.modules[module_name]
            raise e

        return module

    def _extract_schema(self, module: Any) -> dict[str, Any] | None:
        """
        Extract the OpenAI-compatible function schema from a module.

        Calls get_tool_schema() if available.

        Args:
            module: Loaded module object

        Returns:
            Schema dict, or None if not available
        """
        if not hasattr(module, "get_tool_schema"):
            return None

        try:
            schema = module.get_tool_schema()
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass

        return None

    def _extract_handler(self, module: Any) -> Callable[..., Any] | None:
        """
        Extract the runtime execution handler (run function) from a module.

        Args:
            module: Loaded module object

        Returns:
            Callable, or None if not available
        """
        if not hasattr(module, "run"):
            return None

        handler = getattr(module, "run")
        if callable(handler):
            return handler

        return None

    async def activate_tool(
        self,
        outcome: ModuleLoadOutcome,
        dispatcher: Any,
    ) -> ToolResult:
        """
        Register a successfully loaded tool into the dispatcher.

        Updates dispatcher's tool routing dictionary and TOOL_DECLARATIONS.
        Optionally registers the tool description into FTS5 KNOWLEDGE memory.

        Args:
            outcome: ModuleLoadOutcome from load_and_register()
            dispatcher: OrionDispatcher instance to register tool

        Returns:
            ToolResult indicating success or failure
        """
        if not outcome.succeeded or outcome.schema is None or outcome.handler_fn is None:
            return ToolResult(
                f"Cannot activate failed load: {', '.join(outcome.error_log)}",
                ok=False,
            )

        try:
            tool_name = outcome.schema.get("name", outcome.module_name)

            # Register handler in dispatcher's routing dictionary
            await asyncio.to_thread(
                self._register_handler,
                dispatcher,
                tool_name,
                outcome.handler_fn,
            )

            # Register tool declaration in TOOL_DECLARATIONS if present
            await asyncio.to_thread(
                self._register_declaration,
                dispatcher,
                tool_name,
                outcome.schema,
            )

            # Optionally register in memory tier (if memory available)
            if hasattr(dispatcher, "memory") and dispatcher.memory is not None:
                await asyncio.to_thread(
                    self._register_in_knowledge,
                    dispatcher.memory,
                    tool_name,
                    outcome.schema,
                )

            self.bus.log.emit(
                f"[FORGE] Loader: ✓ activated tool '{tool_name}' in dispatcher"
            )

            return ToolResult(
                f"✓ Tool '{tool_name}' activated and registered. "
                f"Schema: {outcome.schema.get('description', 'no description')}",
                ok=True,
            )

        except Exception as e:
            self.bus.log.emit(f"[FORGE] Loader: ✗ activation failed: {e}")
            return ToolResult(
                f"✗ Failed to activate tool: {type(e).__name__}: {e}",
                ok=False,
            )

    def _register_handler(
        self,
        dispatcher: Any,
        tool_name: str,
        handler_fn: Callable[..., Any],
    ) -> None:
        """
        Register a handler function in the dispatcher's routing dictionary.

        Args:
            dispatcher: OrionDispatcher instance
            tool_name: Name of the tool
            handler_fn: Callable to execute the tool
        """
        if not hasattr(dispatcher, "_tool_handlers"):
            dispatcher._tool_handlers = {}
        dispatcher._tool_handlers[tool_name] = handler_fn

    def _register_declaration(
        self,
        dispatcher: Any,
        tool_name: str,
        schema: dict[str, Any],
    ) -> None:
        """
        Register a tool declaration in TOOL_DECLARATIONS.

        Args:
            dispatcher: OrionDispatcher instance
            tool_name: Name of the tool
            schema: OpenAI-compatible schema dict
        """
        if not hasattr(dispatcher, "TOOL_DECLARATIONS"):
            return

        # Add to declarations list if not already present
        existing_tool = None
        for tool in dispatcher.TOOL_DECLARATIONS:
            if tool.get("name") == tool_name:
                existing_tool = tool
                break

        if existing_tool is None:
            dispatcher.TOOL_DECLARATIONS.append(schema)

    def _register_in_knowledge(
        self,
        memory_agent: Any,
        tool_name: str,
        schema: dict[str, Any],
    ) -> None:
        """
        Register tool description in FTS5 KNOWLEDGE memory tier.

        Args:
            memory_agent: MemoryAgent instance
            tool_name: Name of the tool
            schema: OpenAI-compatible schema dict
        """
        if not hasattr(memory_agent, "save_intelligence"):
            return

        description = schema.get("description", "Custom forge-generated tool")
        key_ref = f"forge_tool_{tool_name}"

        try:
            memory_agent.save_intelligence(
                category="forge_tools",
                key_ref=key_ref,
                value=description,
            )
        except Exception:
            # Silently fail; knowledge registration is optional
            pass

    def tool_result(self, outcome: ModuleLoadOutcome) -> ToolResult:
        """
        Convert ModuleLoadOutcome to ToolResult for dispatcher.

        Args:
            outcome: Module load outcome

        Returns:
            ToolResult with load summary
        """
        if outcome.succeeded:
            schema_name = outcome.tool_schema.get("name", "unknown") if outcome.tool_schema else "unknown"
            msg = (
                f"✓ Module '{outcome.module_name}' loaded successfully "
                f"({outcome.duration_ms:.1f}ms).\n"
                f"Tool: {schema_name}"
            )
            return ToolResult(msg, ok=True)
        else:
            msg = (
                f"✗ Failed to load module '{outcome.module_name}'. Errors:\n"
                + "\n".join(outcome.error_log[-10:])
            )
            return ToolResult(msg, ok=False)
