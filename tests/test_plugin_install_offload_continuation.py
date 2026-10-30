"""A plugin install must leave the async/GUI event loop available."""
import threading
from types import SimpleNamespace

from orion_core.dispatch_productivity import ProductivityDispatchMixin
from orion_core.data import ToolResult


async def test_plugin_install_runs_outside_the_event_loop_thread():
    owner = threading.current_thread()
    observed = []
    def install(source):
        observed.append(threading.current_thread())
        assert source == "synthetic.orionplugin"
        return ToolResult("installed")
    context = SimpleNamespace(plugins=SimpleNamespace(install=install))
    result = await ProductivityDispatchMixin.plugin_tool(
        context, {"action": "install", "source": "synthetic.orionplugin"})
    assert result.ok
    assert observed and observed[0] is not owner
