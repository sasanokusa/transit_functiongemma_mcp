from __future__ import annotations

from typing import Any, Iterable

from jsonschema import Draft202012Validator

from transit_functiongemma.toolcall import ToolCall


class ToolCallSchemaError(ValueError):
    pass


def validate_tool_name(name: str, tools: Iterable[dict[str, Any]]) -> None:
    names = {str(tool.get("name")) for tool in tools}
    if name not in names:
        raise ToolCallSchemaError(f"tool is not allow-listed: {name}")


def validate_tool_call(call: ToolCall, tools: Iterable[dict[str, Any]]) -> None:
    by_name = {str(tool.get("name")): tool for tool in tools}
    validate_tool_name(call.name, by_name.values())
    schema = by_name[call.name].get("inputSchema") or {"type": "object"}
    errors = sorted(Draft202012Validator(schema).iter_errors(call.arguments), key=str)
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ToolCallSchemaError(f"invalid arguments for {call.name}: {details}")


def validate_tool_calls(calls: Iterable[ToolCall], tools: Iterable[dict[str, Any]]) -> None:
    tool_list = list(tools)
    for call in calls:
        validate_tool_call(call, tool_list)


__all__ = [
    "ToolCallSchemaError",
    "validate_tool_call",
    "validate_tool_calls",
    "validate_tool_name",
]
