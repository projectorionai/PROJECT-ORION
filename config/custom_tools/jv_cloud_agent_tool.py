"""Reviewed Mark XXXIII adapter; importing never starts a service."""

from orion_core.jv_plugins.catalogue import schema
from orion_core.jv_plugins.runner import execute


def get_tool_schema() -> dict:
    return schema("jv_cloud_agent")


async def run(**kwargs):
    return await execute("jv_cloud_agent", kwargs)
