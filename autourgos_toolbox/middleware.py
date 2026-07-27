"""
middleware.py — Toolbox and ToolboxMiddleware.

Keeps the agent's context window clean by only presenting toolbox names
and descriptions upfront. The agent calls `expose_toolbox(name)` at runtime
to lazy-load the actual tools it needs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Union

from .base import CallbackHandler, StructuredTool, build_tool_list, register_tool


def _to_agent_tool_dict(tool: Any) -> Any:
    """
    Convert a StructuredTool / plain callable / already-shaped dict into the
    real ``ReactAgent`` tool dict shape (``{"name", "description",
    "parameters", "func"}``) expected by ``agent.tools`` / ``agent.add_tools``.

    react-agent's own tool list is a list of plain dicts (see its "Tool Dict
    Reference"), not StructuredTool instances or raw callables — passing
    those straight through used to silently corrupt agent.tools and break
    the loop's ``t["name"]`` lookups the first time the LLM actually tried to
    call the tool.
    """
    if isinstance(tool, dict) and "name" in tool:
        return tool
    if isinstance(tool, StructuredTool):
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema,
            "func": tool.func,
        }
    if callable(tool):
        registry: Dict[str, Dict[str, Any]] = {}
        register_tool(registry, tool)
        (name, info), = registry.items()
        return {"name": name, "description": info["description"],
                "parameters": info["parameters"], "func": info["func"]}
    return tool


class Toolbox:
    """
    A named group of tools that can be lazy-loaded into an agent at runtime.

    Parameters
    ----------
    name : str
        Unique identifier for this toolbox (e.g. ``"github"``).
    description : str
        One-sentence description shown to the agent so it knows when to load
        this toolbox.
    tools : list
        List of tools — ``StructuredTool`` instances or plain callables.

    Example
    -------
    ::

        from autourgos_toolbox import Toolbox

        github_box = Toolbox(
            name="github",
            description="Tools for reading and writing GitHub issues and PRs.",
            tools=[search_issues, create_pr, list_repos],
        )
    """

    def __init__(self, name: str, description: str, tools: List[Any]) -> None:
        self.name        = name.strip()
        self.description = description.strip()
        self.tools       = list(tools)


class ToolboxMiddleware(CallbackHandler):
    """
    Middleware for dynamic lazy-loading and sandboxing of toolboxes.

    Keeps the agent's context window clean by only presenting names and
    descriptions of available toolboxes upfront. The agent must call
    ``expose_toolbox(toolbox_name)`` to load the actual tools at runtime.

    How it works
    ------------
    1. On ``on_agent_start`` the middleware injects two meta-tools —
       ``expose_toolbox`` and ``expose_tool`` — plus a catalog of available
       toolboxes into the agent's system prompt.
    2. When the agent calls ``expose_toolbox("github")``, all tools inside
       that toolbox are registered on the agent and their schemas are added
       to the prompt.
    3. On ``on_agent_end`` / ``on_agent_error`` the agent is fully restored
       to its original state (tools + prompt), ready for the next run.

    Parameters
    ----------
    toolboxes : list of Toolbox or dict, optional
        Toolboxes to register. Accepts either a list of :class:`Toolbox`
        objects or a dict in the form::

            {
                "github": {
                    "description": "GitHub tools",
                    "tools": [search_issues, create_pr],
                }
            }

    Example
    -------
    ::

        from autourgos_toolbox import Toolbox, ToolboxMiddleware

        web_box = Toolbox("web", "Web search and scraping tools.", [web_search, scrape_url])
        db_box  = Toolbox("database", "SQL query tools.", [run_query, list_tables])

        middleware = ToolboxMiddleware(toolboxes=[web_box, db_box])
        agent = ReactAgent(llm=my_llm, middleware=[middleware])
        agent.invoke("Find the latest Python release and save it to the DB")
    """

    def __init__(
        self,
        toolboxes: Optional[Union[List[Toolbox], Dict[str, Dict[str, Any]]]] = None,
    ) -> None:
        self.logger    = logging.getLogger(__name__)
        self.toolboxes: Dict[str, Toolbox] = {}
        self._exposed:  Set[str]           = set()
        self._exposed_tools: Set[str]      = set()

        self._agent:                   Optional[Any] = None
        self._initial_tools:           Optional[List[Any]] = None
        self._initial_system_prompt:   Optional[str] = None
        self._initial_prompt_template: Optional[str] = None

        if toolboxes:
            if isinstance(toolboxes, list):
                for tb in toolboxes:
                    if isinstance(tb, Toolbox):
                        self.toolboxes[tb.name] = tb
            elif isinstance(toolboxes, dict):
                for name, info in toolboxes.items():
                    self.add_toolbox(name, info.get("description", ""), info.get("tools", []))

    # ── public API ─────────────────────────────────────────────────────────────

    def add_toolbox(self, name: str, description: str, tools: List[Any]) -> None:
        """Register a new toolbox. Can be called before or after agent start."""
        self.toolboxes[name.strip()] = Toolbox(name, description, tools)

    # ── lifecycle hooks ────────────────────────────────────────────────────────

    def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None:
        if agent is None:
            agent = kwargs.get("agent")
        if agent is None:
            return

        self._agent  = agent
        self._exposed = set()
        self._exposed_tools = set()

        # snapshot original state for restore
        self._initial_tools           = list(agent.tools) if hasattr(agent, "tools") else []
        self._initial_system_prompt   = getattr(agent, "system_prompt", None)
        self._initial_prompt_template = getattr(agent, "prompt_template", None)

        # inject meta-tools
        def expose_toolbox(toolbox_name: str) -> str:
            """Expose all tools in the specified toolbox, making them available for you to use.

            Args:
                toolbox_name: The exact name of the toolbox to expose (e.g. 'github').
            """
            return self._expose_toolbox_action(toolbox_name)

        def expose_tool(tool_name: str) -> str:
            """Expose a single tool by name, searching across all registered toolboxes.

            Args:
                tool_name: The exact name of the tool to expose (e.g. 'create_pr'). This
                    does not require knowing which toolbox the tool lives in.
            """
            return self._expose_tool_action(tool_name)

        expose_toolbox_tool = StructuredTool.from_function(
            func=expose_toolbox,
            name="expose_toolbox",
            description="Expose all tools in the specified toolbox, making them available for you to use.",
        )
        expose_tool_tool = StructuredTool.from_function(
            func=expose_tool,
            name="expose_tool",
            description=(
                "Expose a single tool by name (searched across all registered toolboxes), "
                "making just that tool available for you to use without loading the rest "
                "of its toolbox."
            ),
        )
        agent.add_tools(
            _to_agent_tool_dict(expose_toolbox_tool),
            _to_agent_tool_dict(expose_tool_tool),
        )

        # inject toolbox catalog into system prompt
        if self.toolboxes:
            catalog = "\n".join(
                f"- **{name}**: {tb.description}"
                for name, tb in self.toolboxes.items()
            )
            instruction = (
                "## Dynamic Toolboxes\n"
                "You have access to specialized toolboxes that are NOT loaded by default to keep "
                "the context window clean. If you need tools from any toolbox below, you MUST call "
                "`expose_toolbox(toolbox_name)` or `expose_tool(toolbox_name)` with the exact name "
                "of the toolbox first. Once called, all tools inside that toolbox will be loaded "
                "and available.\n\n"
                f"Available Toolboxes:\n{catalog}\n\n"
                "Do NOT attempt to use any toolbox tools until you have called `expose_toolbox` "
                "and received confirmation."
            )
            self._inject_prompt(agent, instruction)

    def on_agent_end(self, response: str, agent: Any = None, **kwargs: Any) -> None:
        self._restore_agent()

    def on_agent_error(self, error: Exception, agent: Any = None, **kwargs: Any) -> None:
        self._restore_agent()

    # ── internal ───────────────────────────────────────────────────────────────

    def _expose_toolbox_action(self, toolbox_name: str) -> str:
        if not self._agent:
            return "Error: No active agent reference found."

        toolbox_name = toolbox_name.strip()
        if toolbox_name not in self.toolboxes:
            available = ", ".join(self.toolboxes.keys())
            return f"Error: Toolbox '{toolbox_name}' not found. Available: {available}"

        if toolbox_name in self._exposed:
            return f"Toolbox '{toolbox_name}' is already exposed and its tools are available."

        tb = self.toolboxes[toolbox_name]
        self._agent.add_tools(*[_to_agent_tool_dict(t) for t in tb.tools])

        new_registry: Dict[str, Dict[str, Any]] = {}
        for tool in tb.tools:
            register_tool(new_registry, tool)
        schemas = build_tool_list(new_registry)

        exposure_note = (
            f"\n\n### Exposed Toolbox: '{toolbox_name}'\n"
            f"The following tools are now active and ready to use:\n{schemas}\n"
        )
        self._inject_prompt(self._agent, exposure_note)

        self._exposed.add(toolbox_name)
        self.logger.info(f"Exposed toolbox '{toolbox_name}' to agent.")
        logger = getattr(self._agent, "logger", None)
        if logger:
            logger.middleware("Toolbox", f"Exposed toolbox '{toolbox_name}' to agent.")
        return f"Success: Exposed all tools in '{toolbox_name}' toolbox. You can now call them."

    @staticmethod
    def _get_tool_name(tool: Any) -> Optional[str]:
        """Best-effort extraction of a tool's name across supported tool shapes."""
        if isinstance(tool, StructuredTool):
            return tool.name
        if isinstance(tool, dict):
            return tool.get("name")
        if callable(tool):
            return getattr(tool, "__name__", None)
        return None

    def _find_tool(self, tool_name: str) -> Optional[Any]:
        """Search every registered toolbox for a tool matching ``tool_name``."""
        for tb in self.toolboxes.values():
            for tool in tb.tools:
                if self._get_tool_name(tool) == tool_name:
                    return tool
        return None

    def _expose_tool_action(self, tool_name: str) -> str:
        if not self._agent:
            return "Error: No active agent reference found."

        tool_name = tool_name.strip()

        if tool_name in self._exposed_tools:
            return f"Tool '{tool_name}' is already exposed and available."

        tool = self._find_tool(tool_name)
        if tool is None:
            return f"Error: Tool '{tool_name}' not found in any registered toolbox."

        self._agent.add_tools(_to_agent_tool_dict(tool))

        new_registry: Dict[str, Dict[str, Any]] = {}
        register_tool(new_registry, tool)
        schemas = build_tool_list(new_registry)

        exposure_note = (
            f"\n\n### Exposed Tool: '{tool_name}'\n"
            f"The following tool is now active and ready to use:\n{schemas}\n"
        )
        self._inject_prompt(self._agent, exposure_note)

        self._exposed_tools.add(tool_name)
        self.logger.info(f"Exposed tool '{tool_name}' to agent.")
        logger = getattr(self._agent, "logger", None)
        if logger:
            logger.middleware("Toolbox", f"Exposed tool '{tool_name}' to agent.")
        return f"Success: Exposed tool '{tool_name}'. You can now call it."

    @staticmethod
    def _inject_prompt(agent: Any, text: str) -> None:
        if hasattr(agent, "system_prompt"):
            agent.system_prompt = (
                f"{text}\n\n{agent.system_prompt}" if agent.system_prompt else text
            )
        elif hasattr(agent, "prompt_template"):
            agent.prompt_template = (
                f"{text}\n\n{agent.prompt_template}" if agent.prompt_template else text
            )

    def _restore_agent(self) -> None:
        if not self._agent:
            return
        if self._initial_tools is not None and hasattr(self._agent, "tools"):
            self._agent.tools = list(self._initial_tools)
        if hasattr(self._agent, "system_prompt"):
            self._agent.system_prompt = self._initial_system_prompt
        if hasattr(self._agent, "prompt_template"):
            self._agent.prompt_template = self._initial_prompt_template
        self._agent                   = None
        self._initial_tools           = None
        self._initial_system_prompt   = None
        self._initial_prompt_template = None
        self._exposed.clear()
        self._exposed_tools.clear()
