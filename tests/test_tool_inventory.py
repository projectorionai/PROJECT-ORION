"""
Tool-inventory drift guard (Mark X.12 §2.1).

The dispatcher exposes its capabilities two ways that must stay in step:

  * ``TOOL_DECLARATIONS`` — the function declarations the model sees.
  * ``handler_table()``   — the tool-name → bound-method routing map.

The common failure mode is drift: a tool gets added to one side but not the
other, or a declaration goes missing its description/parameters, and nobody
notices until the model tries to call something that isn't wired (or a wired
tool is invisible to the model). This also once bit the docs — ARCHITECTURE.md
claimed 33 tools when the dispatcher actually exposed 103.

These tests lock the invariants:

  * every declared tool has a handler;
  * every handler is declared, except a small, explicit allow-list of
    intentional internal aliases (documented below);
  * every declaration is structurally complete and callable;
  * no tool is declared twice.

When a genuinely new internal-only alias is added, extend ``_ALIAS_HANDLERS``
with a one-line justification — that keeps the allow-list itself honest.
"""

from __future__ import annotations

from orion_core.bus import OrionBus
from orion_core.dispatcher import OrionDispatcher, TOOL_DECLARATIONS

# Handlers that are intentionally routable but deliberately NOT declared to the
# model (back-compat aliases for a declared tool, internal chaining targets,
# etc.). Each entry must say what it aliases and why it stays undeclared.
_ALIAS_HANDLERS = {
    # Back-compat alias for the declared ``process_file`` tool — both route to
    # ``self.process_file``. Kept so an older caller (or a learned model habit)
    # using "image_processor" still resolves, without duplicating the schema.
    "image_processor",
}


def _bare_dispatcher() -> OrionDispatcher:
    """A dispatcher with no services wired — enough to read the static handler
    table and the declarations (neither depends on live subsystems)."""
    return OrionDispatcher(
        bus=OrionBus(), memory=None, grabber=None, file_intel=None,
        desktop=None, vision=None, outlook=None, notion=None,
        agent_manager=None, briefing=None,
    )


def test_every_declaration_is_structurally_complete():
    for tool in TOOL_DECLARATIONS:
        name = tool.get("name")
        assert name, f"a declaration is missing its name: {tool!r}"
        assert tool.get("description"), f"{name}: missing/empty description"
        assert "parameters" in tool, f"{name}: missing parameters block"


def test_no_duplicate_tool_declarations():
    names = [t["name"] for t in TOOL_DECLARATIONS]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"tools declared more than once: {duplicates}"


def test_every_declared_tool_has_a_handler():
    handlers = _bare_dispatcher().handler_table()
    declared = {t["name"] for t in TOOL_DECLARATIONS}
    missing = sorted(declared - set(handlers))
    assert not missing, f"declared tools with no handler: {missing}"


def test_every_handler_is_declared_or_an_allowed_alias():
    handlers = _bare_dispatcher().handler_table()
    declared = {t["name"] for t in TOOL_DECLARATIONS}
    undeclared = set(handlers) - declared - _ALIAS_HANDLERS
    assert not undeclared, (
        "handlers routable but neither declared nor an allowed alias: "
        f"{sorted(undeclared)} — declare them, or add to _ALIAS_HANDLERS with a "
        "justification if they are intentionally internal-only."
    )


def test_every_handler_is_callable():
    for name, handler in _bare_dispatcher().handler_table().items():
        assert callable(handler), f"handler for '{name}' is not callable"


def test_alias_handlers_are_actually_routable():
    # Guard the guard: if an alias is removed from the dispatcher, this list
    # must be trimmed too, so the allow-list can't rot into a lie.
    handlers = _bare_dispatcher().handler_table()
    stale = sorted(a for a in _ALIAS_HANDLERS if a not in handlers)
    assert not stale, f"_ALIAS_HANDLERS lists handlers that no longer exist: {stale}"
