#!/usr/bin/env python3
"""Convert operational_semantic_holdout_300.jsonl (intent-router SFT row shape,
already-structured resolve_route_request arguments) into the row shape
scripts/run_operational_samples.py expects (id/prompt/expected_* fields), so the
final-pipeline stage can run this holdout through the live Web API + real Transit MCP.

Unlike scripts/build_operational_intent_datasets.py, no legacy regex re-annotation is
needed here: the source rows already carry structured resolve_route_request arguments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_TIME_MODE_TO_MCP_TYPE = {
    "arrive_by": "arrival",
    "departure_at": "departure",
    "first_train": "first",
    "last_train": "last",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def mode_constraint(arguments: dict[str, Any]) -> str | None:
    if "tokyo_metro" in (arguments.get("allowed_operator_groups") or []):
        return "tokyo_metro"
    if "toei_subway" in (arguments.get("allowed_operator_groups") or []):
        return "toei_subway"
    if "subway" in (arguments.get("allowed_operator_groups") or []):
        return "subway"
    if "JR" in (arguments.get("avoid_operator_groups") or []):
        return "subway"
    if "bus" in (arguments.get("avoid_modes") or []):
        return "rail_only"
    return None


def convert(row: dict[str, Any]) -> dict[str, Any]:
    target = row["assistant"]
    arguments = target.get("arguments", {})
    out: dict[str, Any] = {
        "id": row["id"],
        "prompt": row["user"],
        "category": row.get("category"),
        "expected_tool": "plan_route_map" if arguments.get("graphical") else "plan_journey",
        "expected_model_intent": arguments,
    }
    time_mode = arguments.get("time_mode")
    if time_mode in _TIME_MODE_TO_MCP_TYPE:
        out["expected_mcp_type"] = _TIME_MODE_TO_MCP_TYPE[time_mode]
    if arguments.get("time"):
        out["expected_mcp_time"] = arguments["time"]
    if arguments.get("date"):
        out["expected_mcp_date"] = arguments["date"]
    if arguments.get("priority"):
        out["expected_priority"] = arguments["priority"]
    if arguments.get("avoid_station_texts"):
        out["expected_avoid_station"] = arguments["avoid_station_texts"][0]
    if arguments.get("via_station_texts"):
        out["expected_via_station"] = arguments["via_station_texts"][0]
    if arguments.get("avoid_line_texts"):
        out["expected_avoid_line"] = arguments["avoid_line_texts"][0]
    constraint = mode_constraint(arguments)
    if constraint:
        out["expected_mode_constraint"] = constraint
    return out


def execution_issues(row: dict[str, Any]) -> list[str]:
    arguments = (row.get("assistant") or {}).get("arguments") or {}
    origin = arguments.get("origin_text")
    destination = arguments.get("destination_text")
    via = set(arguments.get("via_station_texts") or [])
    avoid = set(arguments.get("avoid_station_texts") or [])
    issues: list[str] = []
    if origin == destination:
        issues.append("origin_equals_destination")
    if origin in via or destination in via:
        issues.append("via_is_endpoint")
    if origin in avoid or destination in avoid:
        issues.append("avoid_is_endpoint")
    if via & avoid:
        issues.append("via_avoid_conflict")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/eval/operational_semantic_holdout_300.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/operational_semantic_holdout_300_samples.jsonl"),
    )
    parser.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Keep structurally impossible rows in final-MCP evaluation.",
    )
    args = parser.parse_args()
    source_rows = read_jsonl(args.input)
    excluded = [
        {"id": row["id"], "issues": execution_issues(row)}
        for row in source_rows
        if execution_issues(row)
    ]
    eligible = source_rows if args.include_ineligible else [
        row for row in source_rows if not execution_issues(row)
    ]
    rows = [convert(row) for row in eligible]
    write_jsonl(args.output, rows)
    print(f"converted {len(rows)} rows -> {args.output}; excluded={len(excluded)}")
    for item in excluded:
        print(f"  {item['id']}: {','.join(item['issues'])}")


if __name__ == "__main__":
    main()
