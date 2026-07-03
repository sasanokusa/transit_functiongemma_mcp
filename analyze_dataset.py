#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from prepare_sft import convert_record, read_jsonl
from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, DEVELOPER_PROMPT, MODEL_ID
from transit_functiongemma.schemas import (
    compact_functiongemma_tools,
    load_mcp_tools,
    tools_with_clarification,
    to_functiongemma_tools,
)


def class_name(row: dict[str, Any], evaluation: bool) -> str:
    if evaluation:
        return row.get("expected_tool") or "no_tool_call"
    if "assistant" in row:
        target = row.get("assistant") or {}
        return "no_tool_call" if target.get("no_tool_call") else target.get("tool_name", "unknown")
    return row.get("metadata", {}).get("expected_tool") or "no_tool_call"


def has_expected_arguments(row: dict[str, Any], evaluation: bool) -> bool:
    if evaluation:
        return "expected_arguments" in row
    if "assistant" in row:
        target = row.get("assistant") or {}
        return "arguments" in target
    messages = row.get("messages", [])
    return bool(messages and messages[-1].get("tool_calls"))


def input_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in row.get("history", row.get("messages", [])):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            parts.append(message["content"].strip())
    if isinstance(row.get("user"), str):
        parts.append(row["user"].strip())
    return "\n".join(part for part in parts if part)


def duplicate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(input_text(row) for row in rows)
    duplicates = {text: count for text, count in counts.items() if text and count > 1}
    return {
        "duplicate_input_count": sum(count - 1 for count in duplicates.values()),
        "duplicate_groups": len(duplicates),
        "examples": dict(list(sorted(duplicates.items(), key=lambda item: -item[1]))[:50]),
    }


def dataset_summary(rows: list[dict[str, Any]], evaluation: bool) -> dict[str, Any]:
    distribution = Counter(class_name(row, evaluation) for row in rows)
    with_args = sum(has_expected_arguments(row, evaluation) for row in rows)
    return {
        "records": len(rows),
        "class_distribution": dict(sorted(distribution.items())),
        "no_call_count": distribution.get("no_tool_call", 0),
        "expected_arguments": {
            "present": with_args,
            "absent": len(rows) - with_args,
        },
        "input_duplicates": duplicate_summary(rows),
    }


def rendered_tools(
    mcp_tools: list[dict[str, Any]], schema_mode: str, clarification_tool: bool
) -> list[dict[str, Any]]:
    available = tools_with_clarification(mcp_tools, clarification_tool)
    if schema_mode == "full":
        return to_functiongemma_tools(available)
    if schema_mode == "compact":
        return compact_functiongemma_tools(available)
    return []


def token_lengths(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    model: str,
    mcp_tools: list[dict[str, Any]],
    schema_mode: str,
    clarification_tool: bool,
) -> dict[str, Any]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model)
    train_lengths: list[int] = []
    for row in train:
        converted = (
            row
            if "messages" in row
            else convert_record(row, mcp_tools, schema_mode, clarification_tool)
        )
        kwargs = {"tools": converted["tools"]} if converted.get("tools") else {}
        train_lengths.append(
            len(
                processor.apply_chat_template(
                    converted["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    **kwargs,
                )
            )
        )

    tools = rendered_tools(mcp_tools, schema_mode, clarification_tool)
    eval_lengths: list[int] = []
    for row in evaluation:
        now = row.get("reference_datetime", "2026-06-28 12:00 Asia/Tokyo")
        messages: list[dict[str, Any]] = [
            {"role": "developer", "content": DEVELOPER_PROMPT.format(now=now)}
        ]
        messages.extend(row.get("history", []))
        if "user" in row:
            messages.append({"role": "user", "content": row["user"]})
        kwargs = {"tools": tools} if tools else {}
        eval_lengths.append(
            len(
                processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **kwargs,
                )
            )
        )

    def stats(values: list[int]) -> dict[str, float | int | None]:
        return {
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "average": round(mean(values), 2) if values else None,
            "over_512": sum(value > 512 for value in values),
        }

    return {"model": model, "train": stats(train_lengths), "eval": stats(eval_lengths)}


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Dataset analysis", ""]
    for split in ("train", "eval"):
        summary = report[split]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Records: {summary['records']}",
                f"- No-call: {summary['no_call_count']}",
                f"- Expected arguments present: {summary['expected_arguments']['present']}",
                f"- Duplicate inputs: {summary['input_duplicates']['duplicate_input_count']}",
                "",
                "| Class | Count |",
                "|---|---:|",
            ]
        )
        lines.extend(
            f"| `{name}` | {count} |"
            for name, count in summary["class_distribution"].items()
        )
        lines.append("")
    overlap = report["overlap"]
    lines.extend(
        [
            "## Overlap",
            "",
            f"- Exact train/eval input overlap: {overlap['count']}",
            "",
        ]
    )
    if report.get("tokens"):
        lines.extend(["## Token length", "", "| Split | Min | Average | Max | >512 |", "|---|---:|---:|---:|---:|"])
        for split in ("train", "eval"):
            item = report["tokens"][split]
            lines.append(
                f"| {split} | {item['minimum']} | {item['average']} | {item['maximum']} | {item['over_512']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze transit tool-routing datasets.")
    parser.add_argument("--train", type=Path, default=Path("data/raw/synthetic_balanced.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval/eval_balanced.jsonl"))
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--schema-mode", choices=("baked", "compact", "full"), default="baked")
    parser.add_argument("--clarification-tool", action="store_true")
    parser.add_argument("--skip-token-length", action="store_true")
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/dataset_analysis.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("artifacts/dataset_analysis.md"))
    args = parser.parse_args()

    train = read_jsonl(args.train)
    evaluation = read_jsonl(args.eval)
    train_inputs = {input_text(row) for row in train if input_text(row)}
    eval_inputs = {input_text(row) for row in evaluation if input_text(row)}
    overlap = sorted(train_inputs & eval_inputs)
    report: dict[str, Any] = {
        "train": dataset_summary(train, False),
        "eval": dataset_summary(evaluation, True),
        "overlap": {"count": len(overlap), "inputs": overlap[:100]},
    }
    if not args.skip_token_length:
        report["tokens"] = token_lengths(
            train,
            evaluation,
            args.model,
            load_mcp_tools(args.schema),
            args.schema_mode,
            args.clarification_tool,
        )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(f"analysis JSON: {args.json_output}")
    print(f"analysis Markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
