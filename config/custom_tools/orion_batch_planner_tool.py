"""Original ORION dependency planning: proposals without action execution."""
from orion_core.plugins_analysis import batch_planner, schema


def get_tool_schema() -> dict:
    return schema("orion_batch_planner")


def run(tasks, max_concurrency=3):
    return batch_planner(tasks, max_concurrency)
