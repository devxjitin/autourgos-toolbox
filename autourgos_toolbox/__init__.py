"""
autourgos-toolbox — Dynamic lazy-loading toolbox middleware for Autourgos agents.

Keeps context windows clean by only showing toolbox names upfront.
The agent calls `expose_toolbox(name)` at runtime to load the tools it needs.

Quick start::

    from autourgos_toolbox import Toolbox, ToolboxMiddleware

    web_box = Toolbox("web", "Web search tools.", [search, scrape])
    db_box  = Toolbox("database", "SQL tools.", [run_query, list_tables])

    middleware = ToolboxMiddleware(toolboxes=[web_box, db_box])
    agent = Agent(llm=my_llm, middleware=[middleware])
"""

from .base import CallbackHandler, StructuredTool, build_tool_list, register_tool
from .middleware import Toolbox, ToolboxMiddleware

try:
    from importlib.metadata import version as _meta_version
    __version__ = _meta_version("autourgos-toolbox")
except Exception:
    __version__ = "3.1.1"

__all__ = [
    "Toolbox",
    "ToolboxMiddleware",
    "StructuredTool",
    "CallbackHandler",
    "build_tool_list",
    "register_tool",
]
