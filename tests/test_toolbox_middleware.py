import json
import unittest
from unittest.mock import MagicMock

from autourgos_agent.testing import make_test_agent

from autourgos_toolbox import Toolbox, ToolboxMiddleware


def _tool_names(agent):
    """Extract tool names from agent.tools, a list of dicts / StructuredTool."""
    names = []
    for t in agent.tools:
        if isinstance(t, dict):
            names.append(t.get("name"))
        else:
            names.append(getattr(t, "name", None))
    return names


def search_issues(query: str) -> str:
    """Search GitHub issues.

    Args:
        query: search text.
    """
    return f"results for {query}"


def create_pr(title: str) -> str:
    """Create a GitHub PR.

    Args:
        title: PR title.
    """
    return f"created {title}"


def run_query(sql: str) -> str:
    """Run a SQL query.

    Args:
        sql: the SQL text.
    """
    return f"ran {sql}"


def _build_middleware():
    github_box = Toolbox(
        name="github",
        description="GitHub tools",
        tools=[search_issues, create_pr],
    )
    db_box = Toolbox(
        name="database",
        description="SQL tools",
        tools=[run_query],
    )
    return ToolboxMiddleware(toolboxes=[github_box, db_box])


class ToolboxMiddlewareTests(unittest.TestCase):
    """
    These tests run against a REAL Agent built by make_test_agent(),
    not a hand-rolled fake. agent.tools is a real list (not a dict), which
    is exactly the shape that used to break ToolboxMiddleware's
    dict(agent.tools) snapshot with a ValueError.
    """

    def test_expose_toolbox_actually_adds_tools_to_real_agent(self):
        middleware = _build_middleware()
        agent = make_test_agent(
            responses=['{"thought": null, "actions": [], "final_answer": "done"}'],
            middleware=[middleware],
        )

        # Simulate what react-agent's real run does: fire on_agent_start,
        # then invoke the meta-tool action directly (as the LLM would via
        # the expose_toolbox tool call).
        middleware.on_agent_start("query", agent=agent)
        result = middleware._expose_toolbox_action("github")

        self.assertIn("Success", result)
        names = _tool_names(agent)
        self.assertIn("search_issues", names)
        self.assertIn("create_pr", names)
        self.assertIn("Exposed Toolbox: 'github'", agent.system_prompt)

        # idempotent on second call
        second = middleware._expose_toolbox_action("github")
        self.assertIn("already exposed", second)

        middleware.on_agent_end("done", agent=agent)

    def test_expose_toolbox_not_found(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        middleware.on_agent_start("query", agent=agent)

        result = middleware._expose_toolbox_action("nonexistent")
        self.assertIn("not found", result)

    def test_expose_tool_exposes_single_tool_only_on_real_agent(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        middleware.on_agent_start("query", agent=agent)

        result = middleware._expose_tool_action("create_pr")

        self.assertIn("Success", result)
        names = _tool_names(agent)
        self.assertIn("create_pr", names)
        # the sibling tool in the same toolbox must NOT be exposed
        self.assertNotIn("search_issues", names)
        self.assertIn("Exposed Tool: 'create_pr'", agent.system_prompt)

        # idempotent on second call
        second = middleware._expose_tool_action("create_pr")
        self.assertIn("already exposed", second)

        middleware.on_agent_end("done", agent=agent)

    def test_expose_tool_searches_across_all_toolboxes(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        middleware.on_agent_start("query", agent=agent)

        result = middleware._expose_tool_action("run_query")

        self.assertIn("Success", result)
        self.assertIn("run_query", _tool_names(agent))

    def test_expose_tool_unknown_name_returns_clear_message(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        middleware.on_agent_start("query", agent=agent)

        result = middleware._expose_tool_action("does_not_exist")

        self.assertIn("not found", result)
        self.assertNotIn("does_not_exist", _tool_names(agent))

    def test_expose_tool_meta_tool_is_wired_to_new_implementation(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        middleware.on_agent_start("query", agent=agent)

        by_name = {}
        for t in agent.tools:
            name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
            by_name[name] = t

        expose_tool_meta = by_name["expose_tool"]
        expose_toolbox_meta = by_name["expose_toolbox"]

        # meta-tools land on agent.tools as real tool dicts (the shape a
        # real Agent expects), each with its own "func" callable.
        expose_tool_func = expose_tool_meta["func"] if isinstance(expose_tool_meta, dict) else expose_tool_meta.func
        expose_toolbox_func = expose_toolbox_meta["func"] if isinstance(expose_toolbox_meta, dict) else expose_toolbox_meta.func

        # the two meta-tools must no longer share the same underlying function
        self.assertIsNot(expose_tool_func, expose_toolbox_func)

        result = expose_tool_func("create_pr")
        self.assertIn("Success", result)
        names = _tool_names(agent)
        self.assertIn("create_pr", names)
        self.assertNotIn("search_issues", names)

    def test_expose_toolbox_narrates_via_agent_logger(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        agent.logger = MagicMock()
        middleware.on_agent_start("query", agent=agent)

        middleware._expose_toolbox_action("github")

        agent.logger.middleware.assert_called_once()
        args, _ = agent.logger.middleware.call_args
        self.assertEqual(args[0], "Toolbox")
        self.assertIn("github", args[1])

    def test_expose_tool_narrates_via_agent_logger(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        agent.logger = MagicMock()
        middleware.on_agent_start("query", agent=agent)

        middleware._expose_tool_action("create_pr")

        agent.logger.middleware.assert_called_once()
        args, _ = agent.logger.middleware.call_args
        self.assertEqual(args[0], "Toolbox")
        self.assertIn("create_pr", args[1])

    def test_snapshot_restore_does_not_raise_on_real_list_shaped_tools(self):
        """
        Regression test for the core bug: dict(agent.tools) used to raise
        ValueError because agent.tools is a list of tool dicts, not a dict.
        """
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        original_tool_names = set(_tool_names(agent))

        middleware.on_agent_start("query", agent=agent)
        middleware._expose_toolbox_action("github")
        middleware._expose_tool_action("run_query")

        # must not raise, and must restore original (pre-expose) tool list
        middleware.on_agent_end("done", agent=agent)
        self.assertEqual(set(_tool_names(agent)), original_tool_names)

    def test_full_run_through_real_agent_invoke_exposes_toolbox(self):
        """
        End-to-end: run make_test_agent().invoke() with the LLM scripted to
        call expose_toolbox, then verify the toolbox's tools actually landed
        on agent.tools while the run is in progress (advertised behavior).
        By design ToolboxMiddleware restores the agent's original tool list
        in on_agent_end, so we capture agent.tools mid-run via a second
        handler's on_iteration_start (fires after iteration 1's tool call,
        before the restore in on_agent_end).
        """
        from autourgos_agent import CallbackHandler

        middleware = _build_middleware()

        captured = {}

        class ToolsSpy(CallbackHandler):
            def on_iteration_start(self, iteration, agent=None, **kwargs):
                captured[iteration] = _tool_names(agent)

        responses = [
            json.dumps({
                "thought": "I need github tools",
                "actions": [{"action": "expose_toolbox", "action_input": {"toolbox_name": "github"}}],
                "final_answer": None,
            }),
            json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
        ]
        agent = make_test_agent(responses=responses, middleware=[middleware, ToolsSpy()])

        result = agent.invoke("expose the github toolbox")

        self.assertEqual(result, "done")
        # by iteration 2 (after iteration 1's expose_toolbox call ran),
        # the toolbox's tools must actually be present on the real agent.
        self.assertIn("search_issues", captured[2])
        self.assertIn("create_pr", captured[2])
        # after the run ends, the middleware restores the original tool set
        self.assertNotIn("search_issues", _tool_names(agent))

    def test_no_crash_when_agent_has_no_logger(self):
        middleware = _build_middleware()
        agent = make_test_agent(middleware=[middleware])
        if hasattr(agent, "logger"):
            del agent.logger
        middleware.on_agent_start("query", agent=agent)

        result = middleware._expose_toolbox_action("github")
        self.assertIn("Success", result)

        result2 = middleware._expose_tool_action("run_query")
        self.assertIn("Success", result2)


    def test_toolboxes_with_colliding_tool_names_are_rejected_at_registration(self):
        """Two toolboxes both exposing a tool named 'search_issues' used to
        both silently succeed; once both were exposed, agent.tools ended up
        with two entries sharing a name and lookup became order-dependent.
        This must now fail fast at registration instead."""
        box_a = Toolbox(name="a", description="d", tools=[search_issues])

        def search_issues_duplicate(query: str) -> str:
            """Also called search_issues.

            Args:
                query: search text.
            """
            return f"dup results for {query}"
        search_issues_duplicate.__name__ = "search_issues"

        box_b = Toolbox(name="b", description="d", tools=[search_issues_duplicate])

        with self.assertRaises(ValueError):
            ToolboxMiddleware(toolboxes=[box_a, box_b])

    def test_add_toolbox_rejects_collision_with_existing_toolbox(self):
        middleware = _build_middleware()  # has 'github' (search_issues, create_pr) and 'database'

        with self.assertRaises(ValueError):
            middleware.add_toolbox("github2", "d", [search_issues])

    def test_add_toolbox_reregistering_same_name_does_not_raise(self):
        middleware = _build_middleware()
        # updating the 'github' toolbox's own tools must not trip the
        # cross-toolbox collision check against itself
        middleware.add_toolbox("github", "updated", [search_issues, create_pr])
        self.assertIn("create_pr", {t.__name__ for t in middleware.toolboxes["github"].tools})


if __name__ == "__main__":
    unittest.main()
