#!/usr/bin/env python3
"""Audit intent-router split leakage and final-MCP eligibility."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def route_issues(row: dict[str, Any]) -> list[str]:
    target = row.get("assistant") or {}
    if target.get("tool_name") != "resolve_route_request":
        return []
    args = target.get("arguments") or {}
    origin, destination = args.get("origin_text"), args.get("destination_text")
    via = set(args.get("via_station_texts") or [])
    avoid = set(args.get("avoid_station_texts") or [])
    issues: list[str] = []
    if not origin or not destination:
        issues.append("missing_endpoint")
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
    parser.add_argument("--train", type=Path, default=Path("data/raw/intent_router_train_8000.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/eval/intent_router_dev_950.jsonl"))
    parser.add_argument("--stress", type=Path, default=Path("data/eval/intent_router_stress_600.jsonl"))
    parser.add_argument("--holdout", type=Path, default=Path("data/eval/operational_semantic_holdout_300.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/intent_asset_audit.json"))
    parser.add_argument("--strict-eval-independence", action="store_true")
    args = parser.parse_args()
    datasets = {name: read(getattr(args, name)) for name in ("train", "dev", "stress", "holdout")}
    overlaps: dict[str, dict[str, int]] = {}
    names = list(datasets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            left, right = datasets[first], datasets[second]
            overlaps[f"{first}:{second}"] = {
                "prompt_overlap": len({row["user"] for row in left} & {row["user"] for row in right}),
                "id_overlap": len({row["id"] for row in left} & {row["id"] for row in right}),
            }
    quality: dict[str, Any] = {}
    for name, rows in datasets.items():
        counts = Counter(issue for row in rows for issue in route_issues(row))
        quality[name] = {
            "rows": len(rows),
            "unique_ids": len({row["id"] for row in rows}),
            "unique_prompts": len({row["user"] for row in rows}),
            "route_issues": dict(counts),
        }
    report = {"overlaps": overlaps, "quality": quality}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failures = [
        key for key, value in overlaps.items()
        if key.startswith("train:") and value["prompt_overlap"]
    ]
    if args.strict_eval_independence:
        failures.extend(
            key for key, value in overlaps.items()
            if not key.startswith("train:") and (value["prompt_overlap"] or value["id_overlap"])
        )
    if failures:
        raise SystemExit(f"split leakage: {sorted(set(failures))}")


if __name__ == "__main__":
    main()
