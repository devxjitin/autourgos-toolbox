import json
import logging
import threading
import unittest
from unittest.mock import MagicMock

from autourgos_agent.testing import make_test_agent

from autourgos_toolbox import StructuredTool, Toolbox, ToolboxMiddleware


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

    def test_concurrent_runs_do_not_corrupt_each_others_state(self):
        """Regression: flat self._agent/self._exposed instance attributes used
        to be shared across every run of a middleware instance. If run A's
        on_agent_start fires, then run B's on_agent_start fires before A ends,
        B would overwrite A's snapshot/exposed-set, and A's later
        expose_toolbox/on_agent_end would silently act on B's agent instead
        of its own. Runs must be isolated per agent object."""
        middleware = _build_middleware()
        agent_a = make_test_agent(middleware=[middleware])
        agent_b = make_test_agent(middleware=[middleware])
        # Pre-start snapshot -- what on_agent_end restores back to.
        pre_start_a = set(_tool_names(agent_a))
        pre_start_b = set(_tool_names(agent_b))

        # Interleave: start A, start B (before A ends), expose github only on A.
        middleware.on_agent_start("query a", agent=agent_a)
        middleware.on_agent_start("query b", agent=agent_b)

        # Post-start snapshot (meta-tools already injected on both agents by
        # design) so the mid-run assertions below isolate exactly the
        # cross-run leakage this test targets.
        post_start_b = set(_tool_names(agent_b))

        result = middleware._expose_toolbox_action("github", agent=agent_a)
        self.assertIn("Success", result)
        self.assertIn("search_issues", _tool_names(agent_a))
        # B must be unaffected by A's exposure.
        self.assertEqual(set(_tool_names(agent_b)), post_start_b)

        # End B first -- must restore B to its own pre-start tools, not A's.
        middleware.on_agent_end("done b", agent=agent_b)
        self.assertEqual(set(_tool_names(agent_b)), pre_start_b)
        # A's exposed toolbox must still be intact -- B's end must not have
        # touched A's state.
        self.assertIn("search_issues", _tool_names(agent_a))

        # End A -- must restore A to its own pre-start tools.
        middleware.on_agent_end("done a", agent=agent_a)
        self.assertEqual(set(_tool_names(agent_a)), pre_start_a)

    def test_structured_tool_warns_on_unmapped_type_annotation(self):
        """Regression: an annotation present but not one of the primitive
        types _infer_schema maps (e.g. a custom class) silently fell back to
        "string" with no signal to the tool author. A genuinely absent
        annotation must NOT warn -- that's the normal, expected case."""

        class Widget:
            pass

        def uses_custom_type(thing: Widget) -> str:
            """Use a widget."""
            return "ok"

        def uses_no_annotation(thing) -> str:
            """Use anything."""
            return "ok"

        with self.assertLogs("autourgos_toolbox.base", level="WARNING") as cm:
            tool = StructuredTool.from_function(uses_custom_type)
        self.assertIn("unmapped type", cm.output[0])
        self.assertEqual(tool.args_schema["properties"]["thing"]["type"], "string")

        logger = logging.getLogger("autourgos_toolbox.base")
        handler = logging.Handler()
        records = []
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            StructuredTool.from_function(uses_no_annotation)
        finally:
            logger.removeHandler(handler)
        self.assertEqual(records, [])

    def test_param_description_parsed_with_args_header(self):
        """Args:-headed docstrings (the format this package's own tools/tests
        already use) still get their param descriptions populated."""

        def headed(city: str) -> str:
            """Get weather.

            Args:
                city: the city to check.
            """
            return "ok"

        tool = StructuredTool.from_function(headed)
        self.assertEqual(tool.args_schema["properties"]["city"]["description"], "the city to check.")

    def test_param_description_empty_without_args_header(self):
        """
        Behavior change (Sprint 3b): _infer_schema()'s inline per-line scan
        used to pick up a bare "param: description" line with no section
        header. Now unified with autourgos-agent's @tool decorator via
        autourgos_core.parse_param_descriptions(), which requires an
        Args:/Arguments:/Parameters: header -- a header-free docstring now
        gets an empty description, matching what @tool has always done.
        """

        def headerless(city: str) -> str:
            """Get weather.

            city: the city to check.
            """
            return "ok"

        tool = StructuredTool.from_function(headerless)
        self.assertEqual(tool.args_schema["properties"]["city"]["description"], "")

    def test_add_toolbox_reregistering_same_name_does_not_raise(self):
        middleware = _build_middleware()
        # updating the 'github' toolbox's own tools must not trip the
        # cross-toolbox collision check against itself
        middleware.add_toolbox("github", "updated", [search_issues, create_pr])
        self.assertIn("create_pr", {t.__name__ for t in middleware.toolboxes["github"].tools})

    def test_two_concurrent_agents_share_one_middleware_without_state_clash(self):
        """
        Sprint 5 regression: ToolboxMiddleware is commonly a single shared
        instance handed to multiple agents. Its per-run state (self._runs,
        now a PerAgentRegistry) must stay isolated by agent identity even
        when two agents' on_agent_start/expose/on_agent_end calls interleave
        on separate threads -- one agent's exposed toolbox must never leak
        into, or get clobbered by, the other's.
        """
        middleware = _build_middleware()
        agent_a = make_test_agent(middleware=[middleware])
        agent_b = make_test_agent(middleware=[middleware])

        errors = []
        barrier = threading.Barrier(2)

        def drive(agent, toolbox_name, other_toolbox_name):
            try:
                middleware.on_agent_start("query", agent=agent)
                barrier.wait(timeout=5)
                middleware._expose_toolbox_action(toolbox_name, agent=agent)
                for _ in range(20):
                    names = set(_tool_names(agent))
                    if other_toolbox_name == "database":
                        assert "run_query" not in names, "agent_a saw agent_b's tool"
                    else:
                        assert "search_issues" not in names, "agent_b saw agent_a's tool"
                middleware.on_agent_end("done", agent=agent)
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(exc)

        t_a = threading.Thread(target=drive, args=(agent_a, "github", "database"))
        t_b = threading.Thread(target=drive, args=(agent_b, "database", "github"))
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        self.assertEqual(errors, [])
        # both runs restored cleanly -- no leftover toolbox tools on either agent
        self.assertNotIn("search_issues", _tool_names(agent_a))
        self.assertNotIn("run_query", _tool_names(agent_b))


if __name__ == "__main__":
    unittest.main()
