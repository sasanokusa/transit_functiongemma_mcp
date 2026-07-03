#!/usr/bin/env python3
"""Convert intent_router SFT-shaped eval rows into eval_toolcall.py's expected_tool/
expected_arguments row shape (same shape scripts/build_operational_intent_datasets.py
already produces for operational_intent_raw_100.jsonl)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def convert(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("assistant")
    no_call = target is None or target.get("no_tool_call")
    eval_row: dict[str, Any] = {
        "id": row["id"],
        "user": row["user"],
        "reference_datetime": row.get("reference_datetime"),
        "expected_tool": None if no_call else target["tool_name"],
        "category": row.get("category"),
    }
    if not no_call:
        eval_row["expected_arguments"] = target.get("arguments", {})
    return eval_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [convert(row) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    print(f"converted {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
