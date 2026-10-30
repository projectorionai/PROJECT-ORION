"""Original ORION timing analysis: no I/O or model calls at import or run."""
from orion_core.plugins_analysis import latency_report, schema


def get_tool_schema() -> dict:
    return schema("orion_latency_report")


def run(samples):
    return latency_report(samples)
