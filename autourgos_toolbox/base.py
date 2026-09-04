"""
base.py — Base classes for autourgos-toolbox.

CallbackHandler is re-exported from autourgos-agent, the package that
owns this interface, to avoid divergent duplicate copies. StructuredTool,
register_tool, and build_tool_list are toolbox-specific (they operate on a
dict-based tool registry, unlike autourgos-agent's own build_tool_list which
operates on a plain list of tool dicts) and remain defined locally here.
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from autourgos_agent import CallbackHandler
from autourgos_core import parse_param_descriptions

__all__ = ["CallbackHandler", "StructuredTool", "register_tool", "build_tool_list"]

logger = logging.getLogger(__name__)


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

        # Requires an Args:/Arguments:/Parameters: header -- a docstring with
        # bare "param: description" lines and no header is not parsed. This
        # is the same stricter algorithm autourgos-agent's @tool decorator
        # uses (unified via autourgos-core so the two don't silently diverge
        # again); a header-free docstring that used to get its descriptions
        # picked up here now gets an empty description instead, matching
        # what @tool has always done.
        param_docs = parse_param_descriptions(inspect.getdoc(func))

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "return"):
                continue
            hint = hints.get(param_name)
            type_name = "string"
            if hint is not None:
                raw = getattr(hint, "__name__", str(hint))
                if raw in _PY_TO_JSON:
                    type_name = _PY_TO_JSON[raw]
                else:
                    # An annotation IS present but isn't one of the primitive
                    # types this inferrer maps (e.g. Optional[str], List[int],
                    # a custom class) -- falls back to "string" silently
                    # otherwise, which can produce a wrong schema with no
                    # signal to the tool author. A genuinely absent
                    # annotation (hint is None, handled above) is the normal
                    # case and does not warn.
                    logger.warning(
                        "StructuredTool: parameter %r of %r has unmapped type "
                        "annotation %r -- defaulting its JSON schema type to "
                        "'string'. Supported annotations: %s.",
                        param_name, getattr(func, "__name__", func), raw,
                        sorted(_PY_TO_JSON),
                    )

            properties[param_name] = {"type": type_name, "description": param_docs.get(param_name, "")}
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
