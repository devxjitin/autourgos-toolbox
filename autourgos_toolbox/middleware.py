"""
middleware.py — Toolbox and ToolboxMiddleware.

Keeps the agent's context window clean by only presenting toolbox names
and descriptions upfront. The agent calls `expose_toolbox(name)` at runtime
to lazy-load the actual tools it needs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from autourgos_agent import inject_prompt_block, remove_prompt_block
from autourgos_agent.base import _tool_name
from autourgos_core import PerAgentRegistry

from .base import CallbackHandler, StructuredTool, build_tool_list, register_tool


def _to_agent_tool_dict(tool: Any) -> Any:
    """
    Convert a StructuredTool / plain callable / already-shaped dict into the
    real ``Agent`` tool dict shape (``{"name", "description",
    "parameters", "func"}``) expected by ``agent.tools`` / ``agent.add_tools``.

    autourgos-agent's own tool list is a list of plain dicts (see its "Tool Dict
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
        agent = Agent(llm=my_llm, middleware=[middleware])
        agent.invoke("Find the latest Python release and save it to the DB")
    """

    def __init__(
        self,
        toolboxes: Optional[Union[List[Toolbox], Dict[str, Dict[str, Any]]]] = None,
    ) -> None:
        self.logger    = logging.getLogger(__name__)
        self.toolboxes: Dict[str, Toolbox] = {}

        # Per-agent run state, keyed by the actual agent object rather than
        # flat instance attributes. A single ToolboxMiddleware instance can
        # back multiple concurrent agent.invoke() calls (its own README
        # advertises "works with any Autourgos agent" with no single-run
        # caveat); flat self._agent/self._exposed/... attributes would let
        # one run's on_agent_start overwrite another's in-flight state and
        # restore the wrong agent's tools on on_agent_end. WeakKeyDictionary
        # also means a run's entry is freed automatically if its agent is
        # garbage-collected without on_agent_end ever firing.
        self._runs: "PerAgentRegistry[Dict[str, Any]]" = PerAgentRegistry()
        # Fallback target for the private _expose_toolbox_action/_expose_tool_action
        # methods when called directly without an explicit agent (e.g. tests
        # exercising a single run) -- the meta-tool closures built in
        # on_agent_start always pass their own agent explicitly and never
        # rely on this.
        self._last_agent: Optional[Any] = None

        if toolboxes:
            if isinstance(toolboxes, list):
                for tb in toolboxes:
                    if isinstance(tb, Toolbox):
                        self.add_toolbox(tb.name, tb.description, tb.tools)
            elif isinstance(toolboxes, dict):
                for name, info in toolboxes.items():
                    self.add_toolbox(name, info.get("description", ""), info.get("tools", []))

    # ── public API ─────────────────────────────────────────────────────────────

    def add_toolbox(self, name: str, description: str, tools: List[Any]) -> None:
        """Register a new toolbox. Can be called before or after agent start.

        Raises
        ------
        ValueError
            If any tool in ``tools`` shares a name with a tool already
            registered in a *different* toolbox. Without this check, two
            toolboxes both exposing e.g. a ``"search"`` tool would silently
            both land on ``agent.tools`` once both were exposed, leaving
            lookup order-dependent/undefined for whichever one the agent's
            LLM ends up calling.
        """
        name = name.strip()
        incoming_names = {n for n in (self._get_tool_name(t) for t in tools) if n}
        for other_name, other_tb in self.toolboxes.items():
            if other_name == name:
                continue  # re-registering/updating this same toolbox is fine
            existing_names = {n for n in (self._get_tool_name(t) for t in other_tb.tools) if n}
            collisions = incoming_names & existing_names
            if collisions:
                raise ValueError(
                    f"Toolbox '{name}' has tool name(s) {sorted(collisions)} that "
                    f"already exist in toolbox '{other_name}'. Tool names must be "
                    f"unique across all registered toolboxes."
                )
        self.toolboxes[name] = Toolbox(name, description, tools)

    # ── lifecycle hooks ────────────────────────────────────────────────────────

    def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None:
        if agent is None:
            agent = kwargs.get("agent")
        if agent is None:
            return

        self._last_agent = agent
        run_state: Dict[str, Any] = {
            # Neither agent.tools nor system_prompt/prompt_template is
            # snapshotted-and-restored-whole: with multiple middleware
            # (toolbox, hcix, skills) each independently adding to
            # agent.tools / injecting into the same shared prompt string,
            # whichever registered second would snapshot the first's
            # already-added tools/text as if they were "the original," and
            # restoring would never actually get back to the true original
            # -- see inject_prompt_block's module docstring in
            # autourgos-agent for the full explanation (same root cause,
            # same fix shape, applied here to agent.tools instead of a
            # prompt string). Instead this run tracks only the exact tool
            # objects / prompt blocks THIS instance added (below), each
            # independently removable regardless of order or what other
            # middleware does in between.
            "added_tools":     [],
            "injected_blocks": [],
            "exposed":         set(),
            "exposed_tools":   set(),
        }
        self._runs.set(agent, run_state)

        # inject meta-tools, bound to THIS run's agent explicitly so a
        # concurrent run's expose_toolbox/expose_tool calls can never act
        # on the wrong agent even if runs interleave.
        def expose_toolbox(toolbox_name: str) -> str:
            """Expose all tools in the specified toolbox, making them available for you to use.

            Args:
                toolbox_name: The exact name of the toolbox to expose (e.g. 'github').
            """
            return self._expose_toolbox_action(toolbox_name, agent=agent)

        def expose_tool(tool_name: str) -> str:
            """Expose a single tool by name, searching across all registered toolboxes.

            Args:
                tool_name: The exact name of the tool to expose (e.g. 'create_pr'). This
                    does not require knowing which toolbox the tool lives in.
            """
            return self._expose_tool_action(tool_name, agent=agent)

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
        added = [_to_agent_tool_dict(expose_toolbox_tool), _to_agent_tool_dict(expose_tool_tool)]
        agent.add_tools(*added)
        run_state["added_tools"].extend(added)

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
            run_state["injected_blocks"].append(inject_prompt_block(agent, instruction))

    def on_agent_end(self, response: str, agent: Any = None, **kwargs: Any) -> None:
        self._restore_agent(agent or self._last_agent)

    def on_agent_error(self, error: Exception, agent: Any = None, **kwargs: Any) -> None:
        self._restore_agent(agent or self._last_agent)

    # ── internal ───────────────────────────────────────────────────────────────

    def _expose_toolbox_action(self, toolbox_name: str, agent: Any = None) -> str:
        agent = agent or self._last_agent
        run_state = self._runs.peek(agent) if agent is not None else None
        if agent is None or run_state is None:
            return "Error: No active agent reference found."

        toolbox_name = toolbox_name.strip()
        if toolbox_name not in self.toolboxes:
            available = ", ".join(self.toolboxes.keys())
            return f"Error: Toolbox '{toolbox_name}' not found. Available: {available}"

        if toolbox_name in run_state["exposed"]:
            return f"Toolbox '{toolbox_name}' is already exposed and its tools are available."

        tb = self.toolboxes[toolbox_name]
        added = [_to_agent_tool_dict(t) for t in tb.tools]
        agent.add_tools(*added)
        run_state["added_tools"].extend(added)

        new_registry: Dict[str, Dict[str, Any]] = {}
        for tool in tb.tools:
            register_tool(new_registry, tool)
        schemas = build_tool_list(new_registry)

        exposure_note = (
            f"\n\n### Exposed Toolbox: '{toolbox_name}'\n"
            f"The following tools are now active and ready to use:\n{schemas}\n"
        )
        run_state["injected_blocks"].append(inject_prompt_block(agent, exposure_note))

        run_state["exposed"].add(toolbox_name)
        self.logger.info(f"Exposed toolbox '{toolbox_name}' to agent.")
        logger = getattr(agent, "logger", None)
        if logger:
            logger.middleware("Toolbox", f"Exposed toolbox '{toolbox_name}' to agent.")
        return f"Success: Exposed all tools in '{toolbox_name}' toolbox. You can now call them."

    @staticmethod
    def _get_tool_name(tool: Any) -> Optional[str]:
        """Best-effort extraction of a tool's name across supported tool
        shapes. Shared with autourgos-agent's _tool_name (dict / .name
        attribute / bare-callable __name__), which already covers
        StructuredTool via its .name attribute."""
        return _tool_name(tool)

    def _find_tool(self, tool_name: str) -> Optional[Any]:
        """Search every registered toolbox for a tool matching ``tool_name``."""
        for tb in self.toolboxes.values():
            for tool in tb.tools:
                if self._get_tool_name(tool) == tool_name:
                    return tool
        return None

    def _expose_tool_action(self, tool_name: str, agent: Any = None) -> str:
        agent = agent or self._last_agent
        run_state = self._runs.peek(agent) if agent is not None else None
        if agent is None or run_state is None:
            return "Error: No active agent reference found."

        tool_name = tool_name.strip()

        if tool_name in run_state["exposed_tools"]:
            return f"Tool '{tool_name}' is already exposed and available."

        tool = self._find_tool(tool_name)
        if tool is None:
            return f"Error: Tool '{tool_name}' not found in any registered toolbox."

        converted = _to_agent_tool_dict(tool)
        agent.add_tools(converted)
        run_state["added_tools"].append(converted)

        new_registry: Dict[str, Dict[str, Any]] = {}
        register_tool(new_registry, tool)
        schemas = build_tool_list(new_registry)

        exposure_note = (
            f"\n\n### Exposed Tool: '{tool_name}'\n"
            f"The following tool is now active and ready to use:\n{schemas}\n"
        )
        run_state["injected_blocks"].append(inject_prompt_block(agent, exposure_note))

        run_state["exposed_tools"].add(tool_name)
        self.logger.info(f"Exposed tool '{tool_name}' to agent.")
        logger = getattr(agent, "logger", None)
        if logger:
            logger.middleware("Toolbox", f"Exposed tool '{tool_name}' to agent.")
        return f"Success: Exposed tool '{tool_name}'. You can now call it."

    def _restore_agent(self, agent: Any = None) -> None:
        if agent is None:
            return
        run_state = self._runs.pop(agent, None)
        if run_state is None:
            return
        # Remove exactly the tool objects THIS run added, by identity --
        # not a whole-list snapshot/restore, which is order-dependent the
        # moment another middleware (hcix, skills) also adds tools: see
        # run_state's "added_tools" comment in on_agent_start. Identity
        # (not name) match, since add_tools() itself may have replaced an
        # earlier same-named tool when we added ours -- if something else
        # has since taken over that exact object's slot in agent.tools
        # (e.g. a later add_tools() call replacing it by name), there's
        # nothing of ours left to remove there, which is correct: that
        # slot is no longer ours to manage.
        if hasattr(agent, "tools") and run_state["added_tools"]:
            added_ids = {id(t) for t in run_state["added_tools"]}
            agent.tools = [t for t in agent.tools if id(t) not in added_ids]
        # Remove exactly the blocks THIS run injected -- order-independent
        # and correct regardless of what other middleware (hcix, skills)
        # injects/removes before, after, or in between. See
        # inject_prompt_block's module docstring in autourgos-agent.
        for block in run_state["injected_blocks"]:
            remove_prompt_block(agent, block)
        if self._last_agent is agent:
            self._last_agent = None
