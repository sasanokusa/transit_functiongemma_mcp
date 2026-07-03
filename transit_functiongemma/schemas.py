from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transit_functiongemma.local_tools import (
    ASK_CLARIFICATION as CLARIFICATION_TOOL_NAME,
    ASK_CLARIFICATION_TOOL as CLARIFICATION_TOOL,
    RESOLVE_ROUTE_REQUEST_TOOL,
    is_local_tool as _is_local_tool,
)



def load_mcp_tools(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if "tools" in payload:
        return payload["tools"]
    if "result" in payload and "tools" in payload["result"]:
        return payload["result"]["tools"]
    raise ValueError(f"No tools array found in {path}")


def to_functiongemma_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tools/list objects to the HF function-schema representation."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object"}),
            },
        }
        for tool in mcp_tools
    ]


def compact_functiongemma_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep validation-critical schema while reducing 512-token context pressure."""
    compact: list[dict[str, Any]] = []
    for tool in mcp_tools:
        schema = tool.get("inputSchema", {})
        props: dict[str, Any] = {}
        for name, prop in schema.get("properties", {}).items():
            kept = {k: prop[k] for k in ("type", "enum", "pattern") if k in prop}
            if prop.get("type") == "array" and "items" in prop:
                kept["items"] = {
                    k: prop["items"][k]
                    for k in ("type", "enum")
                    if k in prop["items"]
                }
            props[name] = kept
        compact_schema: dict[str, Any] = {"type": "object", "properties": props}
        if schema.get("required"):
            compact_schema["required"] = schema["required"]
        compact.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", "").split(".")[0],
                    "parameters": compact_schema,
                },
            }
        )
    return compact


def tool_map(mcp_tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for tool in mcp_tools}


def tools_with_clarification(
    mcp_tools: list[dict[str, Any]], enabled: bool
) -> list[dict[str, Any]]:
    """Return MCP tools plus the safe local route-intent carrier.

    ``resolve_route_request`` is always available: FunctionGemma uses it to
    express semantics before deterministic MCP planning. The clarification
    tool remains opt-in. Neither local tool is ever sent to the MCP server.
    """
    result = [*mcp_tools, RESOLVE_ROUTE_REQUEST_TOOL]
    if enabled:
        result.append(CLARIFICATION_TOOL)
    return result


def is_local_tool(name: str) -> bool:
    return _is_local_tool(name)
