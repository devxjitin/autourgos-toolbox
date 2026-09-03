# autourgos-toolbox — Features

A dynamic lazy-loading **toolbox middleware** for Autourgos agents (depends on `autourgos-agent`). It groups tools into named `Toolbox` collections and only shows toolbox names/descriptions to the agent upfront; the agent calls `expose_toolbox(name)` to register a whole group's real tool schemas at runtime. It solves the same "too many tools blow the context window / confuse the model" problem that Anthropic's Agent Skills solves for instructions, and that OpenAI's `ToolSearchTool`/`defer_loading` solves for individual tool schemas — but here the unit of lazy-loading is a *named group of tools*, not a single skill or a single tool found via search.

## Full Feature List

### Core mechanism
- Lazy tool loading — agent sees only toolbox names/descriptions at start; real tool schemas register only after `expose_toolbox(name)` is called
- `on_agent_start` injects `expose_toolbox` and `expose_tool` meta-tools plus a toolbox catalog into the agent's system prompt — no real tools loaded yet at this point
- On `expose_toolbox`, all tools in that toolbox are registered on the agent and their schemas appended to the prompt
- `on_agent_end`/`on_agent_error` fully restore the agent to its original tools + prompt, so each run starts clean

### Defining toolboxes (three ways)
- A list of `Toolbox` objects (`name`, `description`, `tools`)
- A dict keyed by toolbox name, each value `{description, tools}`
- Dynamically via `middleware.add_toolbox(name, description, tools)`, which raises `ValueError` if a tool name collides with a *different* already-registered toolbox (re-registering the same toolbox name to update its tools is explicitly allowed)

### `StructuredTool`
- Auto-infers a JSON schema from a plain function's type annotations and docstring (`StructuredTool.from_function(fn)`) — tools can be plain callables or `StructuredTool` instances interchangeably

### Observability
- `Agent(verbose=True)` narrates the middleware's own actions into the trace (e.g. `[Toolbox] Exposed toolbox 'web' to agent.`)

### Requirements
- Works with any Autourgos agent exposing `add_tools()`, `agent.tools`, and `agent.system_prompt`/`agent.prompt_template`

---

## Competitor Comparison

Landscape research into 2026 approaches for managing large tool inventories in LLM agents without overloading context.

| Capability | **autourgos-toolbox** | [OpenAI Agents SDK — `ToolSearchTool`/`defer_loading`](https://callsphere.ai/blog/tool-search-deferred-loading-large-tool-sets-openai-agents-sdk) | [LangChain BigTool](https://cobusgreyling.medium.com/bigtool-from-langchain-9d802cf5b6df) | [LangChain `tool_selection` middleware](https://reference.langchain.com/python/langchain/agents/middleware/tool_selection) | [Microsoft Semantic Kernel plugins](https://devblogs.microsoft.com/semantic-kernel/guest-blog-orchestrating-ai-agents-with-semantic-kernel-plugins-a-technical-deep-dive/) |
|---|---|---|---|---|---|
| Unit of lazy-loading | A named group of tools (a `Toolbox`) | Individual tool schemas, found by search | Individual tools, selected by embedding similarity | Individual tools, filtered by an LLM pre-pass | Individual plugin functions |
| Selection mechanism | Agent explicitly calls `expose_toolbox(name)` after reading names/descriptions — no search step | `ToolSearchTool` meta-tool performs a search over tool descriptions before loading | `InMemoryStore` + embeddings match natural-language queries to tools semantically | A dedicated LLM call filters the full tool list down to the relevant subset before the main model runs | Kernel-mediated discovery; LLM picks from exposed plugin descriptions, typically all resident at once unless custom-gated |
| Extra LLM/model round-trip to select? | No — reading the catalog and calling `expose_toolbox` happens in the same reasoning turn the agent already makes | Effectively yes — the search step is itself a tool call before the real tool is usable | Yes — an embedding lookup (cheaper than an LLM call, but still a lookup step) before the tool is available | Yes — a full extra LLM call dedicated to filtering | Not inherently, but doesn't solve context bloat for very large plugin sets without additional gating |
| Explicit auto-schema inference from Python functions | Yes, `StructuredTool.from_function()` reads type hints + docstring | Standard OpenAI function-calling schema authoring (JSON schema, often via SDK helpers) | Uses LangChain's existing `Tool`/`StructuredTool` conventions | Same, built on LangChain's tool abstraction | Uses C#/other-language method signatures and semantic descriptions |
| Grouping semantics | First-class — `Toolbox` is the core abstraction (name + description + tool list) | No native grouping concept — tools are flat, found individually by search | No native grouping — tools are flat, matched individually by similarity | No native grouping — operates on the flat tool list | Plugins are somewhat group-like by convention (a plugin = a related set of functions), closer to autourgos-toolbox's model than the others |
| Collision/name-safety checks | Yes — `add_toolbox()` errors on a tool-name collision across *different* toolboxes | Not a documented concern at this layer | Not a documented concern at this layer | Not a documented concern at this layer | Not a documented concern at this layer |
| State cleanup between agent runs | Explicit, via `on_agent_end`/`on_agent_error` restoring original tools + prompt | Handled by the SDK's session/context management | Handled by the host application | Handled by the host application | Handled by the kernel/host app |
| Framework dependency | Autourgos agents only | OpenAI Agents SDK only | LangChain/LangGraph ecosystem | LangChain ecosystem | Semantic Kernel ecosystem |

### How to read this

- **vs. OpenAI Agents SDK tool search**: OpenAI's mechanism searches over individual tools by description and pays a real extra round-trip to do it, which scales well to very large, flat tool inventories (hundreds+) but adds latency per lookup. autourgos-toolbox avoids that extra hop by having the agent read a short toolbox catalog directly and call `expose_toolbox` in the same turn — cheaper per-call, but it front-loads the organizational work onto the developer (you must group tools into sensible toolboxes yourself; there's no semantic search to paper over a bad grouping).
- **vs. LangChain BigTool**: BigTool's embedding-based semantic matching is the more scalable answer once you're past what a human can sensibly name and group — it can find a relevant tool among thousands without anyone maintaining categories. autourgos-toolbox has no semantic layer at all; it depends entirely on the LLM reading toolbox names/descriptions, which works well for a handful of well-named toolboxes and degrades as the catalog grows.
- **vs. LangChain `tool_selection` middleware**: that approach uses a dedicated LLM call to filter tools per-query — more adaptive (can react to the specific user request) but costs a full extra model call every time. autourgos-toolbox's grouping is static per-session (chosen at toolbox-definition time, not re-evaluated per query) but costs nothing extra to select.
- **vs. Semantic Kernel plugins**: Semantic Kernel's plugin concept is the closest conceptual peer to `Toolbox` (a named bundle of related functions), but it's part of a much larger, multi-language enterprise orchestration framework with no built-in "hide the bundle until asked for" lazy-exposure primitive as central as autourgos-toolbox's `expose_toolbox` flow — you'd have to build that gating yourself on top of Semantic Kernel.

Sources:
- [Tool Search and Deferred Loading for Large Tool Sets | CallSphere Blog](https://callsphere.ai/blog/tool-search-deferred-loading-large-tool-sets-openai-agents-sdk)
- [Context management - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/context/)
- [Using tools | OpenAI API](https://developers.openai.com/api/docs/guides/tools)
- [BigTool From LangChain](https://cobusgreyling.medium.com/bigtool-from-langchain-9d802cf5b6df)
- [Tools - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/tools)
- [tool_selection | langchain | LangChain Reference](https://reference.langchain.com/python/langchain/agents/middleware/tool_selection)
- [LangChain - Changelog | Dynamic tool calling in LangGraph agents](https://changelog.langchain.com/announcements/dynamic-tool-calling-in-langgraph-agents)
- [Guest Blog: Orchestrating AI Agents with Semantic Kernel Plugins](https://devblogs.microsoft.com/semantic-kernel/guest-blog-orchestrating-ai-agents-with-semantic-kernel-plugins-a-technical-deep-dive/)
- [Feature: Hybrid Tool Pre-Selection (Semantic + Keyword) · Issue #13332](https://github.com/NousResearch/hermes-agent/issues/13332)
