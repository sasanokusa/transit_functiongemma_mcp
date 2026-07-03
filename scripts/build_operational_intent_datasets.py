#!/usr/bin/env python3
"""Build disjoint raw SFT and raw-model evaluation sets for route intent.

The legacy semantic parser is used only here as an offline annotation migration
helper. Runtime normalization never imports this script or its annotations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transit_functiongemma.route_intent import extract_route_intent


REFERENCE_DATETIME = "2026-07-01 09:00 Asia/Tokyo"
ROUTE_TOOL = "resolve_route_request"
NAME_ROTATION = {
    "羽田空港": "成田空港",
    "新宿三丁目": "東新宿",
    "御茶ノ水": "新大久保",
    "表参道": "溜池山王",
    "霞ケ関": "茅場町",
    "吉祥寺": "大井町",
    "秋葉原": "浜松町",
    "大手町": "四ツ谷",
    "日本橋": "代々木上原",
    "飯田橋": "豊洲",
    "六本木": "神保町",
    "浅草": "麻布十番",
    "新宿": "恵比寿",
    "品川": "赤羽",
    "池袋": "神田",
    "東京": "大崎",
    "上野": "北千住",
    "中野": "錦糸町",
    "渋谷": "市ケ谷",
    "押上": "赤坂見附",
    "荻窪": "西船橋",
    "有楽町": "虎ノ門",
    "目黒": "王子",
    "三田": "門前仲町",
    "銀座": "後楽園",
    "立川": "八王子",
    "横浜": "川崎",
    "高尾": "国分寺",
    "大宮": "浦和",
    "新橋": "御徒町",
    "永田町": "築地",
}
_NAME_PATTERN = re.compile(
    "|".join(re.escape(name) for name in sorted(NAME_ROTATION, key=len, reverse=True))
)


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


def rotate_text(value: str) -> str:
    return _NAME_PATTERN.sub(lambda match: NAME_ROTATION[match.group(0)], value)


def intent_arguments(row: dict[str, Any]) -> dict[str, Any]:
    annotated = extract_route_intent(row["prompt"], REFERENCE_DATETIME)
    origin = annotated.get("origin_text")
    destination = annotated.get("destination_text")
    if not origin or not destination:
        raise ValueError(f"{row['id']}: complete route has no annotated endpoints")
    arguments: dict[str, Any] = {
        "origin_text": origin,
        "destination_text": destination,
    }
    mappings = {
        "expected_avoid_station": "avoid_station_texts",
        "expected_via_station": "via_station_texts",
        "expected_avoid_line": "avoid_line_texts",
    }
    for source, target in mappings.items():
        if row.get(source):
            arguments[target] = [row[source]]
    if row.get("expected_priority"):
        arguments["priority"] = row["expected_priority"]
    time_mode = {
        "arrival": "arrive_by",
        "departure": "departure_at",
        "first": "first_train",
        "last": "last_train",
    }.get(row.get("expected_mcp_type"))
    if time_mode:
        arguments["time_mode"] = time_mode
    if row.get("expected_mcp_time"):
        arguments["time"] = row["expected_mcp_time"]
    if "明日" in row["prompt"]:
        arguments["date"] = "20260702"
    elif "今日" in row["prompt"]:
        arguments["date"] = "20260701"

    mode = row.get("expected_mode_constraint")
    if mode == "rail_only":
        arguments["avoid_modes"] = ["bus"]
    elif mode == "subway" and re.search(r"JR.*使わない", row["prompt"]):
        arguments["avoid_operator_groups"] = ["JR"]
    elif mode == "subway":
        arguments["allowed_operator_groups"] = ["subway"]
    return arguments


def rotate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"origin_text", "destination_text"}:
            output[key] = rotate_text(str(value))
        elif key in {"avoid_station_texts", "via_station_texts"}:
            output[key] = [rotate_text(str(item)) for item in value]
        else:
            output[key] = value
    return output


def build(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    evaluate: list[dict[str, Any]] = []
    for row in rows:
        no_call = bool(row.get("expected_no_call"))
        arguments = None if no_call else intent_arguments(row)
        eval_row: dict[str, Any] = {
            "id": f"intent-eval-{row['id']}",
            "user": row["prompt"],
            "reference_datetime": REFERENCE_DATETIME,
            "expected_tool": None if no_call else ROUTE_TOOL,
            "category": row.get("category"),
            "source_operational_id": row["id"],
        }
        if arguments is not None:
            eval_row["expected_arguments"] = arguments
        evaluate.append(eval_row)

        train_prompt = rotate_text(row["prompt"])
        if train_prompt == row["prompt"]:
            train_prompt = f"{train_prompt}、出発地と目的地はまだ未定"
        target: dict[str, Any]
        if no_call:
            target = {"no_tool_call": True}
        else:
            target = {
                "tool_name": ROUTE_TOOL,
                "arguments": rotate_arguments(arguments),
            }
        train.append(
            {
                "id": f"intent-train-{row['id']}",
                "user": train_prompt,
                "reference_datetime": REFERENCE_DATETIME,
                "assistant": target,
                "metadata": {
                    "category": row.get("category"),
                    "source_operational_id": row["id"],
                    "disjoint_by_station_rotation": train_prompt != row["prompt"],
                },
            }
        )
    return train, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/eval/operational_tokyo_routes_100.jsonl"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/raw/operational_intent_train.jsonl"),
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/eval/operational_intent_raw_100.jsonl"),
    )
    args = parser.parse_args()
    train, evaluate = build(read_jsonl(args.source))
    if len(train) != 100 or len(evaluate) != 100:
        raise ValueError("operational intent datasets must contain exactly 100 rows")
    if {row["user"] for row in train} & {row["user"] for row in evaluate}:
        raise ValueError("training and evaluation prompts overlap")
    write_jsonl(args.train_output, train)
    write_jsonl(args.eval_output, evaluate)
    print(f"train={len(train)} -> {args.train_output}")
    print(f"eval={len(evaluate)} -> {args.eval_output}")


if __name__ == "__main__":
    main()
