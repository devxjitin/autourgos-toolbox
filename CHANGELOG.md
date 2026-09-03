# Changelog

## [3.2.5] - 2026-09-03

- Added `features.md` documenting the module's feature set and a competitor comparison. No code changes.


## [3.2.4] - 2026-09-01

- Dependency: raised the `autourgos-agent` floor from `>=2.0.2` to
  `>=3.1.0`. `autourgos-agent` 3.1.0 added sync-hook thread offloading in
  `CallbackManager` under `ainvoke()` (a sync `on_iteration_start`/etc.
  handler now runs off the event-loop thread instead of inline) -- below
  that version, a blocking call inside this middleware's hooks would stall
  every other concurrent `ainvoke()` run sharing that thread. The old
  floor allowed resolving against a pre-3.1.0 install that lacks this fix.
  No code changes here.

## [3.2.3] - 2026-09-01

- Metadata: added `maintainers` (Sonia, Vishwanil Suman) to `pyproject.toml`,
  and linked the README's existing Sonia contributor badge to her GitHub
  profile (https://github.com/dahiyasonia). No code changes.

## [3.2.2] - 2026-09-01

- Fixed: `StructuredTool._infer_schema` silently defaulted a parameter's
  JSON schema type to `"string"` whenever its type annotation wasn't one
  of the mapped primitives (`str`/`int`/`float`/`bool`/`list`/`dict`) --
  e.g. `Optional[str]`, `List[int]`, or a custom class -- with no signal to
  the tool author that the inferred schema might be wrong. Now logs a
  warning naming the parameter, the function, and the unmapped annotation
  when this fallback is used. A genuinely absent annotation (no type hint
  at all) is unaffected -- that's the normal case and still defaults to
  `"string"` without warning.

## [3.2.1] - 2026-09-01

- Fixed: `ToolboxMiddleware` held run state (`_agent`, `_exposed`,
  `_exposed_tools`, `_initial_tools`, `_initial_system_prompt`,
  `_initial_prompt_template`) as flat instance attributes shared across
  every `agent.invoke()` call. One middleware instance backing two
  concurrent runs meant a later run's `on_agent_start` silently overwrote
  an earlier run's in-flight snapshot, and `on_agent_end`/`on_agent_error`
  ignored the `agent` argument entirely -- restoring whichever agent
  `self._agent` currently pointed to, not necessarily the one that
  actually ended. State is now keyed per-agent in a
  `weakref.WeakKeyDictionary`, and the `expose_toolbox`/`expose_tool`
  meta-tool closures built in `on_agent_start` bind to that run's specific
  agent object. Public API (`Toolbox`, `add_toolbox`, constructor, hook
  signatures) is unchanged; single-agent-at-a-time usage behaves
  identically.

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
