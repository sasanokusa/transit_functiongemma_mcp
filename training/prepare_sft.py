#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, DEVELOPER_PROMPT
from transit_functiongemma.japanese import normalize_japanese_prompt, normalize_user_messages
from transit_functiongemma.schemas import (
    CLARIFICATION_TOOL_NAME,
    compact_functiongemma_tools,
    load_mcp_tools,
    tools_with_clarification,
    to_functiongemma_tools,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def convert_record(
    row: dict[str, Any],
    tools: list[dict[str, Any]],
    schema_mode: str,
    clarification_tool: bool = False,
    normalize_ja: bool = False,
) -> dict[str, Any]:
    now = row.get("reference_datetime", "2026-06-28 12:00 Asia/Tokyo")
    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": DEVELOPER_PROMPT.format(now=now)}
    ]
    history = row.get("history", [])
    if normalize_ja:
        history = normalize_user_messages(history, now)
    messages.extend(history)
    if "user" in row:
        user_content = row["user"]
        if normalize_ja:
            user_content = normalize_japanese_prompt(user_content, now)
        messages.append({"role": "user", "content": user_content})

    target = row.get("assistant")
    if (target is None or target.get("no_tool_call")) and clarification_tool:
        clarification = (target or {}).get("clarification") or row.get("clarification") or {
            "missing": ["origin", "destination"],
            "question": "出発地と目的地を教えてください。",
        }
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": CLARIFICATION_TOOL_NAME,
                            "arguments": clarification,
                        },
                    }
                ],
            }
        )
    elif target is None or target.get("no_tool_call"):
        # An empty assistant turn teaches the end-of-turn/no-call action. Runtime
        # rejects all non-call text as well, so it can never execute prose.
        messages.append({"role": "assistant", "content": ""})
    else:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": target["tool_name"],
                            "arguments": target.get("arguments", {}),
                        },
                    }
                ],
            }
        )

    available_tools = tools_with_clarification(tools, clarification_tool)
    if schema_mode == "full":
        rendered_tools = to_functiongemma_tools(available_tools)
    elif schema_mode == "compact":
        rendered_tools = compact_functiongemma_tools(available_tools)
    else:
        rendered_tools = []
    return {
        "id": row["id"],
        "messages": messages,
        "tools": rendered_tools,
        "metadata": {
            "schema_mode": schema_mode,
            "expected_tool": (
                CLARIFICATION_TOOL_NAME
                if clarification_tool and (target is None or target.get("no_tool_call"))
                else None if target is None else target.get("tool_name")
            ),
            "local_tool_enabled": bool(
                clarification_tool
                or target and target.get("tool_name") == "resolve_route_request"
            ),
            "route_intent_tool_enabled": True,
            "clarification_tool_enabled": clarification_tool,
            "japanese_normalized": normalize_ja,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw JSONL into FunctionGemma SFT conversations.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/synthetic_template.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/sft.jsonl"))
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--schema-mode",
        choices=("baked", "compact", "full"),
        default="baked",
        help="baked is recommended at max_seq_length=512; no declarations are put in each prompt.",
    )
    parser.add_argument(
        "--clarification-tool",
        action="store_true",
        help="Convert no_tool_call targets into local-only ask_clarification calls.",
    )
    parser.add_argument("--normalize-ja", action="store_true")
    parser.add_argument(
        "--extra-input",
        type=Path,
        action="append",
        default=[],
        help="Append another raw JSONL file while keeping --input backward compatible.",
    )
    args = parser.parse_args()

    tools = load_mcp_tools(args.schema)
    tool_names = {
        tool["name"] for tool in tools_with_clarification(tools, args.clarification_tool)
    }
    converted = []
    rows = read_jsonl(args.input)
    for extra_path in args.extra_input:
        rows.extend(read_jsonl(extra_path))
    seen_ids: set[str] = set()
    for row in rows:
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate id: {row['id']}")
        seen_ids.add(row["id"])
        target = row.get("assistant")
        if target and not target.get("no_tool_call") and target["tool_name"] not in tool_names:
            raise ValueError(f"{row['id']}: unknown tool {target['tool_name']!r}")
        converted.append(
            convert_record(
                row,
                tools,
                args.schema_mode,
                args.clarification_tool,
                args.normalize_ja,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in converted:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"converted {len(converted)} records -> {args.output} ({args.schema_mode=})")


if __name__ == "__main__":
    main()
