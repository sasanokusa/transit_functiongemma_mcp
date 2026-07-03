#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, MCP_ENDPOINT
from transit_functiongemma.mcp import MCPClient, format_mcp_result, save_mcp_artifact
from transit_functiongemma.schemas import load_mcp_tools
from transit_functiongemma.local_tools import execute_local_tool, is_local_tool
from transit_functiongemma.toolcall import parse_tool_calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and execute one allow-listed MCP tool call.")
    parser.add_argument("--endpoint", default=MCP_ENDPOINT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-output", help="Raw FunctionGemma output")
    source.add_argument("--call-json", help='JSON: {"name":"...","arguments":{...}}')
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts/mcp"))
    args = parser.parse_args()

    if args.model_output:
        calls = parse_tool_calls(args.model_output)
        if len(calls) != 1:
            raise SystemExit(f"refusing execution: expected exactly one parsed call, got {len(calls)}")
        name, arguments = calls[0].name, calls[0].arguments
    else:
        call = json.loads(args.call_json)
        name, arguments = call["name"], call.get("arguments", {})

    if is_local_tool(name):
        try:
            local_result = execute_local_tool(name, arguments)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if local_result["status"] == "clarification":
            print(local_result["question"])
        else:
            print(json.dumps(local_result, ensure_ascii=False, indent=2))
        print("\n[local tool; MCP was not called]")
        return

    tools = load_mcp_tools(args.schema)
    with MCPClient(args.endpoint) as client:
        envelope = client.call_tool(name, arguments, tools=tools)
    path = save_mcp_artifact(envelope, args.save_dir, name)
    print(format_mcp_result(envelope))
    print(f"\n[complete MCP result saved: {path}]")


if __name__ == "__main__":
    main()
