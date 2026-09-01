# autourgos-toolbox

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/autourgos-toolbox/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-toolbox/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)
[![Contributor](https://img.shields.io/badge/Contributor-Sonia-blueviolet.svg)]()
[![Contributor](https://img.shields.io/badge/Contributor-Vishwanil%20Suman-blueviolet.svg)]()

Dynamic lazy-loading toolbox middleware for [Autourgos](https://github.com/devxjitin) agents. Keeps the
agent's context window clean by only showing toolbox names and descriptions upfront — the agent loads the
tools it actually needs at runtime by calling `expose_toolbox(name)`.

```python
from autourgos_toolbox import Toolbox, ToolboxMiddleware
from autourgos_agent import Agent

web_box = Toolbox(name="web", description="Web search and page scraping tools.", tools=[web_search, scrape_url])
db_box  = Toolbox(name="database", description="SQL query tools.", tools=[run_query, list_tables])

agent = Agent(llm=my_llm, middleware=[ToolboxMiddleware(toolboxes=[web_box, db_box])])
result = agent.invoke("Find the latest Python release and log it to the database")
```

---

## Features

- **Lazy tool loading** — the agent sees only toolbox names/descriptions upfront, real tool schemas load on
  demand via `expose_toolbox(name)`
- **Define toolboxes three ways** — a list of `Toolbox` objects, a dict, or dynamically with `add_toolbox()`
- **`StructuredTool`** — auto-infers a JSON schema from type annotations and docstrings
- **Clean state per run** — `on_agent_end`/`on_agent_error` fully restore the agent's original tools + prompt
- Depends on `autourgos-agent`; works with any Autourgos agent

---

## Table of Contents

- [Why Use This?](#why-use-this)
- [Install](#install)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Define Toolboxes](#define-toolboxes)
- [Using StructuredTool](#using-structuredtool)
- [License](#license)

---

## Why Use This?

Real-world agents often need dozens of tools — GitHub, databases, web search, file system, APIs. Dumping all
of them into the prompt at once wastes tokens on tools the agent will never use for a given task, confuses
the LLM with too many choices, and can hit the context window limit before the task even starts.
`ToolboxMiddleware` groups tools into named **toolboxes** and only shows the catalog upfront.

---

## Install

```bash
pip install autourgos-toolbox
```

Depends on `autourgos-agent`.

---

## Quick Start

```python
from autourgos_toolbox import Toolbox, ToolboxMiddleware
from autourgos_agent import Agent

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

middleware = ToolboxMiddleware(toolboxes=[web_box, db_box])
agent = Agent(llm=my_llm, middleware=[middleware])

result = agent.invoke("Find the latest Python release and log it to the database")
print(result)
```

With `Agent(verbose=True)`, this middleware narrates its own actions into the trace:

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

## How It Works

1. **on_agent_start** — middleware injects `expose_toolbox` and `expose_tool` meta-tools plus a toolbox
   catalog into the agent's system prompt. No actual tools are loaded yet.
2. **Agent calls expose_toolbox** — all tools in that toolbox are registered on the agent and their schemas
   appended to the prompt.
3. **Agent uses the tools** — now fully loaded and callable.
4. **on_agent_end / on_agent_error** — agent is fully restored to its original state so the next run starts
   clean.

---

## Define Toolboxes

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
    "github": {"description": "Tools for reading and writing GitHub issues and PRs.", "tools": [search_issues, create_pr, list_repos]},
    "slack":  {"description": "Tools for sending and reading Slack messages.", "tools": [send_message, read_channel]},
})
```

### Add toolboxes dynamically

```python
middleware = ToolboxMiddleware()
middleware.add_toolbox("github", "GitHub tools.", [search_issues, create_pr])
middleware.add_toolbox("slack", "Slack tools.", [send_message, read_channel])
```

`add_toolbox()` raises `ValueError` if any tool name in the new toolbox already exists in a
*different*, previously registered toolbox — tool names must be unique across every toolbox on
one middleware instance. Re-registering the same toolbox name (to update its tools) is fine and
doesn't trip this check.

---

## Using StructuredTool

Tools can be plain callables or `StructuredTool` instances. `StructuredTool` auto-infers the JSON schema from
type annotations and docstrings:

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

## Requirements

- Python 3.9+
- Any Autourgos agent with `add_tools()`, `agent.tools`, and `agent.system_prompt` / `agent.prompt_template`

---

## License

Apache License 2.0, Copyright (c) 2026 Jitin Kumar Sengar
