# Changelog

## [3.2.0] - 2026-09-01

- Added: `add_toolbox()` now raises `ValueError` if a tool name collides
  with one already registered in a *different* toolbox. Previously two
  toolboxes could both silently register a tool with the same name (e.g.
  `"search"`); once both were exposed, `agent.tools` ended up with two
  entries sharing a name and lookup became order-dependent/undefined.
  Re-registering/updating the same toolbox name is unaffected.

## [3.1.0] - 2026-08-30

- BREAKING: dependency migrated from `autourgos-react-agent>=1.6.0` (the
  pre-fork legacy package) to `autourgos-agent>=2.0.2`. `autourgos-react-agent`
  still carries its original loop bugs (denied tool calls never firing
  `on_tool_end`, an async `approval_callback` silently always-approving under
  `invoke()`, no duck-typed tool support, an unbounded tool-call thread pool)
  that `autourgos-agent` 2.0.2 fixed — staying on the old dependency meant
  this middleware ran against an agent loop with unfixed bugs regardless of
  fixes made downstream. All `ReactAgent` references in code/docs/tests are
  now `Agent`, matching the current package name.

## [3.0.0] - 2026-07-27

- BREAKING: requires autourgos-react-agent>=1.6.0.
- Fixed: `expose_toolbox`/`expose_tool` tool-list snapshot/restore was
  silently broken against a real `ReactAgent` — `agent.tools` is a list of
  tool dicts, not a dict, so `dict(agent.tools)` raised `ValueError` and the
  snapshot/restore path never actually ran on a real agent. This release
  fixes the snapshot to use `list(agent.tools)` / `agent.tools = list(...)`
  so toolbox exposure and restore actually work end-to-end.
- Fixed: `expose_toolbox`/`expose_tool` (and the `expose_toolbox`/
  `expose_tool` meta-tools themselves) were adding raw `StructuredTool`
  instances / plain callables directly to `agent.tools`, instead of the
  plain `{"name", "description", "parameters", "func"}` dict shape a real
  `ReactAgent` expects (see react-agent's Tool Dict Reference). Against a
  real agent this corrupted `agent.tools` and broke the loop's tool lookup
  (`t["name"]`) the moment any tool call was attempted. Added
  `_to_agent_tool_dict()` and route every `agent.add_tools(...)` call
  through it.
- Tests rewritten to run against `make_test_agent()` (a real `ReactAgent`)
  instead of hand-rolled fake agents, and now assert that
  `expose_toolbox`/`expose_tool` actually add tools to `agent.tools`.

## [2.1.1] - 2026-07-27

- Fixed: standardized logger to logging.getLogger(__name__).

## [2.1.0] - 2026-07-27

- Added: narrates its own actions into the host ReactAgent's verbose trace via
  `agent.logger.middleware(...)` when available (see autourgos-react-agent's
  README for the pattern). Purely additive and defensive — no crash if the
  host agent has no `.logger`, no output when verbose=False, existing stdlib
  logging unaffected.

## [2.0.0] - 2026-07-27

- BREAKING: this package now depends on autourgos-react-agent>=1.1.0
  (previously zero-dependency). `CallbackHandler` is now re-exported from
  autourgos-react-agent instead of being duplicated locally, to eliminate
  interface drift risk. `StructuredTool`, `register_tool`, and
  `build_tool_list` remain defined locally (not duplicates of
  autourgos-react-agent's own `build_tool_list`, which has a different
  signature and behavior). No public API/behavior change for typical usage —
  `CallbackHandler`'s method signatures and semantics are unchanged.

## [1.1.0] - 2026-07-27

- Fixed `expose_tool` meta-tool: it was bound to the same underlying function as
  `expose_toolbox`, so calling `expose_tool("some_tool_name")` would fail unless
  the name happened to also be a toolbox name. `expose_tool` now genuinely
  searches every registered toolbox for a tool matching the given name and
  exposes only that single tool (not the rest of its toolbox), injecting the
  same kind of prompt note used for whole-toolbox exposure. Returns a clear
  "not found" message when no tool matches. `expose_toolbox` behavior is
  unchanged.
- Added `tests/test_toolbox_middleware.py` covering `expose_toolbox` regression,
  single-tool exposure via `expose_tool`, and the not-found case.

## [1.0.1] - 2026-06-17

- Update Documentation
