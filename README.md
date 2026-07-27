# autourgos-toolbox

Dynamic lazy-loading toolbox middleware for [Autourgos](https://github.com/devxjitin) agents.

Keeps the agent's context window clean by only showing toolbox names and descriptions upfront. The agent loads the tools it actually needs at runtime by calling `expose_toolbox(name)`.

---

## Why use this?

Real-world agents often need dozens of tools — GitHub, databases, web search, file system, APIs. Dumping all of them into the prompt at once:

- Wastes tokens on tools the agent will never use for a given task
- Confuses the LLM with too many choices
- Can hit the context window limit before the task even starts

`ToolboxMiddleware` solves this by grouping tools into named **toolboxes** and only showing the catalog upfront. The agent picks what it needs and loads it on demand.

---

## Install

```bash
pip install autourgos-toolbox
```

Depends on `autourgos-react-agent`. Works with any Autourgos agent.

---

## Quick Start

```python
from autourgos_toolbox import Toolbox, ToolboxMiddleware
from autourgos_react_agent import ReactAgent

# Define toolboxes
web_box = Toolbox(
    name="web",
    description="Web search and page scraping tools.",
    tools=[web_search, scrape_url],
)
db_box = Toolbox(
    name="database",
    description="SQL query tools for the production database.",
    tools=[run_query, list_tables, describe_table],
)

# Attach middleware
middleware = ToolboxMiddleware(toolboxes=[web_box, db_box])
agent = ReactAgent(llm=my_llm, middleware=[middleware])

result = agent.invoke("Find the latest Python release and log it to the database")
print(result)
```

When used with a verbose `ReactAgent` (`ReactAgent(..., verbose=True)`), this
middleware narrates its own actions into the same trace as the agent's
Thought/Action/Observation output, for example:

```
[Toolbox] Exposed toolbox 'web' to agent.
```

The agent sees this in its prompt at the start:

```
## Dynamic Toolboxes
Available Toolboxes:
- **web**: Web search and page scraping tools.
- **database**: SQL query tools for the production database.
```

When it needs the web tools, it calls:
```json
{"action": "expose_toolbox", "action_input": {"toolbox_name": "web"}}
```

And immediately all web tools are registered and their schemas injected into the prompt.

---

## How it works

1. **on_agent_start** — middleware injects `expose_toolbox` and `expose_tool` meta-tools plus a toolbox catalog into the agent's system prompt. No actual tools are loaded yet.
2. **Agent calls expose_toolbox** — all tools in that toolbox are registered on the agent and their schemas are appended to the prompt.
3. **Agent uses the tools** — now fully loaded and callable.
4. **on_agent_end / on_agent_error** — agent is fully restored to its original state (tools + prompt) so the next run starts clean.

---

## Define toolboxes

### As a list of Toolbox objects

```python
from autourgos_toolbox import Toolbox, ToolboxMiddleware

github_box = Toolbox(
    name="github",
    description="Tools for reading and writing GitHub issues and PRs.",
    tools=[search_issues, create_pr, list_repos],
)

middleware = ToolboxMiddleware(toolboxes=[github_box])
```

### As a dict

```python
middleware = ToolboxMiddleware(toolboxes={
    "github": {
        "description": "Tools for reading and writing GitHub issues and PRs.",
        "tools": [search_issues, create_pr, list_repos],
    },
    "slack": {
        "description": "Tools for sending and reading Slack messages.",
        "tools": [send_message, read_channel],
    },
})
```

### Add toolboxes dynamically

```python
middleware = ToolboxMiddleware()
middleware.add_toolbox("github", "GitHub tools.", [search_issues, create_pr])
middleware.add_toolbox("slack", "Slack tools.", [send_message, read_channel])
```

---

## Using StructuredTool

Tools can be plain callables or `StructuredTool` instances. `StructuredTool` auto-infers the JSON schema from type annotations and docstrings:

```python
from autourgos_toolbox import StructuredTool

def search_issues(query: str, repo: str) -> str:
    """Search GitHub issues.

    Args:
        query: Search keywords.
        repo: Repository name in owner/repo format.
    """
    ...

tool = StructuredTool.from_function(search_issues)
```

---

## Combine with other middleware

```python
from autourgos_toolbox import ToolboxMiddleware, Toolbox
from autourgos_history import AgentHistoryMiddleware
from autourgos_summarizer import AutoSummarizeMiddleware

middleware = [
    ToolboxMiddleware(toolboxes=[web_box, db_box]),
    AutoSummarizeMiddleware(summarize_every=5),
    AgentHistoryMiddleware(),
]
agent = ReactAgent(llm=my_llm, middleware=middleware)
```

---

## Requirements

- Python 3.9+
- Any Autourgos agent with `add_tools()`, `agent.tools`, and `agent.system_prompt` / `agent.prompt_template`

---

## License

MIT — see [LICENSE](LICENSE)
