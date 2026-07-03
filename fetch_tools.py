#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, MCP_ENDPOINT
from transit_functiongemma.mcp import MCPClient, schema_hash
from transit_functiongemma.schemas import to_functiongemma_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MCP tools/list and save its schemas.")
    parser.add_argument("--endpoint", default=MCP_ENDPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--hf-output", type=Path, default=Path("data/tools_functiongemma.json"))
    parser.add_argument("--hash-output", type=Path, default=Path("data/tool_schema.sha256"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the live schema with the saved schema and do not overwrite files.",
    )
    args = parser.parse_args()

    with MCPClient(args.endpoint) as client:
        envelope = client.list_tools()
    tools = envelope["result"]["tools"]
    live_hash = schema_hash(tools)
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"saved schema not found: {args.output}")
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        saved_hash = schema_hash(saved.get("tools", saved))
        if live_hash != saved_hash:
            raise SystemExit(
                f"schema changed: saved={saved_hash} live={live_hash}; review before deploy"
            )
        print(f"schema unchanged: {live_hash}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"tools": tools}, ensure_ascii=False, indent=2), encoding="utf-8")
    args.hf_output.parent.mkdir(parents=True, exist_ok=True)
    args.hf_output.write_text(
        json.dumps(to_functiongemma_tools(tools), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.hash_output.parent.mkdir(parents=True, exist_ok=True)
    args.hash_output.write_text(live_hash + "\n", encoding="utf-8")
    print(f"saved {len(tools)} MCP tools to {args.output}")
    print(f"saved FunctionGemma schemas to {args.hf_output}")
    print(f"schema hash: {live_hash}")


if __name__ == "__main__":
    main()
