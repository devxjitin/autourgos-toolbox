"""
base.py — Self-contained base classes for autourgos-toolbox.

Inlines CallbackHandler, StructuredTool, build_tool_list, and register_tool
so the package has zero dependency on autourgos-core.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional


# ── CallbackHandler ────────────────────────────────────────────────────────────

class CallbackHandler:
    """Base class for Autourgos agent middleware / event hooks."""

    def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None: pass
    def on_agent_end(self, response: str, agent: Any = None, **kwargs: Any) -> None: pass
    def on_agent_error(self, error: Exception, agent: Any = None, **kwargs: Any) -> None: pass
    def on_iteration_start(self, iteration: int, agent: Any = None, **kwargs: Any) -> None: pass
    def on_llm_end(self, response: str, agent: Any = None, **kwargs: Any) -> None: pass
    def on_tool_start(self, tool_name: str, tool_input: Dict, agent: Any = None, **kwargs: Any) -> None: pass
    def on_tool_end(self, tool_name: str, tool_output: Any, agent: Any = None, **kwargs: Any) -> None: pass
    def on_tool_error(self, tool_name: str, error: Exception, agent: Any = None, **kwargs: Any) -> None: pass
    def on_parse_error(self, iteration: int, raw_response: str, **kwargs: Any) -> None: pass


# ── StructuredTool ─────────────────────────────────────────────────────────────

class StructuredTool:
    """
    A callable tool with a name, description, and auto-inferred JSON schema
    built from the function's type annotations and docstring.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name        = name
        self.description = description
        self.func        = func
        self.args_schema = args_schema or self._infer_schema(func)

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "StructuredTool":
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "")
        return cls(name=tool_name, description=tool_desc, func=func)

    @staticmethod
    def _infer_schema(func: Callable) -> Dict[str, Any]:
        sig    = inspect.signature(func)
        hints  = {}
        try:
            hints = func.__annotations__
        except Exception:
            pass

        _PY_TO_JSON = {
            "str": "string", "int": "integer", "float": "number",
            "bool": "boolean", "list": "array", "dict": "object",
        }

        properties: Dict[str, Any] = {}
        required:   List[str]      = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "return"):
                continue
            hint = hints.get(param_name)
            type_name = "string"
            if hint is not None:
                raw = getattr(hint, "__name__", str(hint))
                type_name = _PY_TO_JSON.get(raw, "string")

            # parse inline description from docstring "param: description" lines
            doc   = inspect.getdoc(func) or ""
            pdesc = ""
            for line in doc.splitlines():
                line = line.strip()
                if line.startswith(f"{param_name}:") or line.startswith(f"{param_name} :"):
                    pdesc = line.split(":", 1)[-1].strip()
                    break

            properties[param_name] = {"type": type_name, "description": pdesc}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


# ── runtime helpers ────────────────────────────────────────────────────────────

def register_tool(registry: Dict[str, Dict[str, Any]], tool: Any) -> None:
    """Register a tool (StructuredTool or plain callable) into a dict registry."""
    if isinstance(tool, StructuredTool):
        registry[tool.name] = {
            "description": tool.description,
            "parameters":  tool.args_schema,
            "func":        tool.func,
        }
    elif callable(tool):
        name = getattr(tool, "__name__", str(tool))
        doc  = inspect.getdoc(tool) or ""
        registry[name] = {
            "description": doc,
            "parameters":  StructuredTool._infer_schema(tool),
            "func":        tool,
        }
    elif isinstance(tool, dict) and "name" in tool:
        registry[tool["name"]] = tool


def build_tool_list(tools: Dict[str, Dict[str, Any]]) -> str:
    """Format a tool registry into a prompt-ready string."""
    lines: List[str] = []
    for name, info in tools.items():
        desc   = info.get("description", "")
        params = info.get("parameters", {})
        lines.append(f"Tool: {name}\nDescription: {desc}\nParameters: {json.dumps(params, indent=2)}\n")
    return "\n".join(lines)
