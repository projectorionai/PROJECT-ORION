"""Repeated batch reads must not exhaust the agent tool budget."""


async def test_agent_duplicate_reads_spend_one_budget_slot_per_batch():
    from orion_core.agents import AgentToolbelt
    calls = []
    async def read(args):
        calls.append(args)
        return "reading"
    belt = AgentToolbelt(extras={"read": ("read-only probe", read)}, budget=3)
    observations = await belt.run_many([("read", {"a": 1, "b": 2}),
                                        ("read", {"b": 2, "a": 1}), ("read", {"a": 3})])
    assert len(observations) == len(calls) == belt.spent == 2
    await belt.run_many([("read", {"a": 1, "b": 2})])
    assert len(calls) == 3  # a later batch observes fresh state
