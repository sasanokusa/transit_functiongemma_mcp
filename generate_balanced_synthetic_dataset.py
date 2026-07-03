#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from generate_synthetic_dataset import (
    COORDS,
    PLACES,
    REF,
    STATIONS,
    make_history,
    station_response,
    tool_call,
    tool_resp,
)

CLASSES = [
    "suggest_stations",
    "suggest_places",
    "reverse_geocode",
    "station_departures",
    "get_station",
    "list_feeds",
    "plan_journey",
    "plan_route_map",
    "no_tool_call",
]

STYLE_SUFFIXES = [
    "",
    "お願いします",
    "候補を確認したいです",
    "検索してください",
    "音声入力です",
    "短く結果をください",
    "移動前の確認です",
    "旅行計画用です",
    "正確なIDが必要です",
    "候補だけ欲しいです",
    "今確認しています",
    "案内をお願いします",
]


def target(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool_name": tool_name, "arguments": arguments}


def raw_row(
    row_id: str,
    assistant: dict[str, Any],
    *,
    user: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": row_id,
        "reference_datetime": REF,
        "assistant": assistant,
    }
    if history is not None:
        row["history"] = history
    else:
        row["user"] = user
    return row


def eval_row(raw: dict[str, Any]) -> dict[str, Any]:
    assistant = raw["assistant"]
    row = {k: v for k, v in raw.items() if k not in {"assistant"}}
    if assistant.get("no_tool_call"):
        row.update(
            {
                "expected_tool": None,
                "missing_info": True,
                "expected_clarification": assistant.get("clarification"),
            }
        )
    else:
        row.update(
            {
                "expected_tool": assistant["tool_name"],
                "expected_arguments": assistant.get("arguments", {}),
            }
        )
        normalized = {
            key: value
            for key, value in assistant.get("arguments", {}).items()
            if key in {"date", "time"}
        }
        if normalized:
            row["expected_normalized"] = normalized
    return row


def input_key(row: dict[str, Any]) -> str:
    if "user" in row:
        return row["user"]
    return json.dumps(row.get("history", []), ensure_ascii=False, sort_keys=True)


def station_case(i: int, rng: random.Random, eval_mode: bool) -> dict[str, Any]:
    if not eval_mode and i == 0:
        return raw_row(
            "",
            target("suggest_stations", {"q": "東京駅", "limit": 5}),
            user="東京駅を検索して",
        )
    if not eval_mode and i in {1, 2}:
        user = "町田から東京まで"
        if i == 1:
            return raw_row(
                "", target("suggest_stations", {"q": "町田", "limit": 5}), user=user
            )
        history = [
            {"role": "user", "content": user},
            tool_call("suggest_stations", {"q": "町田", "limit": 5}),
            tool_resp("suggest_stations", station_response("町田", "demo-feed:machida")),
        ]
        return raw_row(
            "", target("suggest_stations", {"q": "東京", "limit": 5}), history=history
        )
    if not eval_mode and i in {3, 4, 5}:
        name = {3: "東京", 4: "三田", 5: "大手町"}[i]
        return raw_row(
            "",
            target("suggest_stations", {"q": name, "limit": 5}),
            user=f"{name}を駅名として検索して",
        )
    (origin, oid), (destination, _did) = rng.sample(STATIONS, 2)
    mode = i % 3
    if mode == 0:
        templates = (
            ["{name}を鉄道駅として候補検索してください", "駅名「{name}」のIDを探してください"]
            if eval_mode
            else [
                "{name}駅を検索して",
                "{name}の駅候補を出して",
                "乗車駅として{name}を解決して",
                "降車駅の{name}を検索",
                "{name}という駅のID候補を5件",
                "施設ではなく駅として{name}を探して",
            ]
        )
        text = templates[i % len(templates)].format(name=origin)
        query = origin if "駅" not in text.split(origin, 1)[1][:1] else origin + "駅"
        return raw_row("", target("suggest_stations", {"q": query, "limit": 5}), user=text)
    user_templates = (
        ["{fr}を出て{to}へ向かう経路を調べたい", "{fr}発、{to}着で検索"]
        if eval_mode
        else [
            "{fr}から{to}まで行きたい",
            "{fr}駅発で{to}駅へ",
            "{fr}→{to}の乗換を調べて",
            "明日、{fr}から{to}まで",
        ]
    )
    user = user_templates[i % len(user_templates)].format(fr=origin, to=destination)
    if mode == 1:
        return raw_row("", target("suggest_stations", {"q": origin, "limit": 5}), user=user)
    history = [
        {"role": "user", "content": user},
        tool_call("suggest_stations", {"q": origin, "limit": 5}),
        tool_resp("suggest_stations", station_response(origin, oid)),
    ]
    return raw_row(
        "", target("suggest_stations", {"q": destination, "limit": 5}), history=history
    )


def place_case(i: int, rng: random.Random, eval_mode: bool) -> dict[str, Any]:
    boundary_places = ["東京タワー", "東京スカイツリー", "東京都庁", "日本武道館", "六本木ヒルズ"]
    place = boundary_places[i] if not eval_mode and i < len(boundary_places) else rng.choice(PLACES)
    templates = (
        ["施設名「{name}」を乗降候補として解決して", "駅ではなく場所の{name}を検索して"]
        if eval_mode
        else [
            "{name}を場所として探して",
            "{name}の最寄り候補を検索",
            "目的地の施設は{name}",
            "住所・施設として{name}を解決して",
            "駅名ではなく観光地の{name}を探して",
            "{name}周辺の乗車地点候補",
            "{name}へ行くので場所IDを検索して",
        ]
    )
    text = templates[i % len(templates)].format(name=place)
    text += STYLE_SUFFIXES[(i // len(templates)) % len(STYLE_SUFFIXES)]
    return raw_row("", target("suggest_places", {"q": place, "limit": 10}), user=text)


def geocode_case(i: int, rng: random.Random, eval_mode: bool) -> dict[str, Any]:
    lat, lon, label = rng.choice(COORDS)
    radius = [50, 80, 120, 200, 300, 500][i % 6]
    templates = (
        [
            "座標 {lat}, {lon} の周囲{radius}mにある乗車地点",
            "緯度{lat} 経度{lon}から最寄り駅を逆引き",
        ]
        if eval_mode
        else [
            "緯度{lat}、経度{lon}の近くの駅を探して（半径{radius}m）",
            "現在地 {lat},{lon} から{radius}m以内の乗車地点",
            "{label}の座標{lat}/{lon}を逆ジオコード",
            "地図上の点 {lat}, {lon} に近い駅",
        ]
    )
    text = templates[i % len(templates)].format(
        lat=lat, lon=lon, radius=radius, label=label
    )
    text += STYLE_SUFFIXES[
        (i // (len(templates) * len(COORDS))) % len(STYLE_SUFFIXES)
    ]
    return raw_row(
        "",
        target(
            "reverse_geocode",
            {"lat": lat, "lon": lon, "limit": 3, "radiusMeters": radius},
        ),
        user=text,
    )


def departures_case(i: int, rng: random.Random, eval_mode: bool) -> dict[str, Any]:
    name, sid = rng.choice(STATIONS)
    templates = (
        ["駅ID {sid} の発車標を確認", "{sid}で明日9時以降に出る便"]
        if eval_mode
        else [
            "{sid} の発車案内を10件",
            "駅ID {sid} の次の出発を表示",
            "{name}のID {sid} から出る便",
            "{sid} の明日9時以降の発車時刻",
            "発車標を確認したい: {sid}",
        ]
    )
    text = templates[i % len(templates)].format(name=name, sid=sid)
    dated = "明日" in text
    args: dict[str, Any] = {"id": sid}
    explicit_limit = re.search(r"(\d+)件", text)
    if explicit_limit:
        args["limit"] = int(explicit_limit.group(1))
    if dated:
        args.update({"date": "20260629", "time": "9:00"})
    return raw_row("", target("station_departures", args), user=text)


def station_detail_case(i: int, rng: random.Random, eval_mode: bool) -> dict[str, Any]:
    name, sid = rng.choice(STATIONS)
    templates = (
        ["駅識別子 {sid} のホーム・路線情報", "{sid}が表す駅の詳細"]
        if eval_mode
        else [
            "{sid} の駅詳細を見たい",
            "駅ID {sid} のホーム情報",
            "{name}の識別子 {sid} を確認",
            "{sid} に乗り入れる路線と駅情報",
        ]
    )
    return raw_row(
        "", target("get_station", {"id": sid}), user=templates[i % len(templates)].format(name=name, sid=sid)
    )


def feeds_case(i: int, rng: random.Random, eval_mode: bool) -> dict[str, Any]:
    templates = (
        ["収録済み交通データの出典一覧を確認", "対応エリアとデータライセンスを一覧化"]
        if eval_mode
        else [
            "利用できる交通フィード一覧",
            "交通データのライセンスを確認",
            "どの事業者データを収録してる？",
            "対応範囲とデータ出典を見せて",
            "GTFSなどの更新状況を確認",
            "利用中の交通データソース一覧",
            "収録フィードの鮮度を知りたい",
            "対応交通機関と帰属表示を確認",
        ]
    )
    text = templates[i % len(templates)] + STYLE_SUFFIXES[
        (i // len(templates)) % len(STYLE_SUFFIXES)
    ]
    return raw_row("", target("list_feeds", {}), user=text)


def route_case(
    i: int, rng: random.Random, eval_mode: bool, map_mode: bool
) -> dict[str, Any]:
    (origin, oid), (destination, did) = rng.sample(STATIONS, 2)
    date_modes = [
        ("", {}),
        ("明日8時30分出発で", {"date": "20260629", "time": "8:30", "type": "departure"}),
        ("明日18時到着で", {"date": "20260629", "time": "18:00", "type": "arrival"}),
        ("今日の始発で", {"date": "20260628", "type": "first"}),
        ("今日の終電で", {"date": "20260628", "type": "last"}),
    ]
    date_text, date_args = date_modes[i % len(date_modes)]
    if map_mode:
        visual = ["地図で", "グラフィカルに", "地図表示で", "マップで"][i % 4]
        strategy_word, strategy = [
            ("", "balanced"),
            ("最速優先で", "fastest"),
            ("乗換少なめで", "fewestTransfers"),
            ("安い順で", "lowestFare"),
            ("徒歩短めで", "shortestWalk"),
        ][i % 5]
        text = f"{origin}から{destination}まで{date_text}{strategy_word}{visual}見せて"
        expected_tool = "plan_route_map"
    else:
        text = f"{origin}から{destination}まで{date_text}乗換経路を検索して"
        strategy = None
        expected_tool = "plan_journey"
    if eval_mode:
        text = (
            f"{origin}発{destination}着を{date_text}{'経路図として表示' if map_mode else '通常検索'}"
        )
    history = make_history(origin, oid, destination, did, text)
    args: dict[str, Any] = {
        "from": oid,
        "to": did,
        "fromLabel": origin,
        "toLabel": destination,
        **date_args,
    }
    if map_mode:
        args["strategy"] = strategy
    return raw_row("", target(expected_tool, args), history=history)


def no_call_case(i: int, rng: random.Random, eval_mode: bool) -> dict[str, Any]:
    if not eval_mode and i < 4:
        fixed = [
            ("東京駅まで行きたい", ["origin"], "出発地を教えてください。"),
            ("新宿から行きたい", ["destination"], "目的地を教えてください。"),
            ("明日の終電を調べて", ["origin", "destination"], "出発地と目的地を教えてください。"),
            ("安いやつで行きたい", ["origin", "destination"], "出発地と目的地を教えてください。"),
        ]
        text, missing, question = fixed[i]
        return raw_row(
            "",
            {
                "no_tool_call": True,
                "clarification": {"missing": missing, "question": question},
            },
            user=text,
        )
    name, _sid = rng.choice(STATIONS)
    cases = [
        (f"{name}駅まで行きたい", ["origin"], "出発地を教えてください。"),
        (f"{name}から行きたい", ["destination"], "目的地を教えてください。"),
        ("明日の終電を調べて", ["origin", "destination"], "出発地と目的地を教えてください。"),
        ("安いやつで行きたい", ["origin", "destination"], "出発地と目的地を教えてください。"),
        ("地図で経路を見せて", ["origin", "destination"], "出発地と目的地を教えてください。"),
        (f"{name}までの料金は？", ["origin"], "出発地を教えてください。"),
        (f"{name}発の所要時間は？", ["destination"], "目的地を教えてください。"),
        ("何時に出ればいい？", ["origin", "destination"], "出発地と目的地を教えてください。"),
    ]
    if eval_mode:
        cases = [
            (f"目的地は{name}だけど、どう行く？", ["origin"], "出発地を教えてください。"),
            (f"{name}を出る予定です", ["destination"], "目的地を教えてください。"),
            ("最終列車を検索してほしい", ["origin", "destination"], "出発地と目的地を教えてください。"),
        ]
    text, missing, question = cases[i % len(cases)]
    text += STYLE_SUFFIXES[(i // len(cases)) % len(STYLE_SUFFIXES)]
    return raw_row(
        "",
        {
            "no_tool_call": True,
            "clarification": {"missing": missing, "question": question},
        },
        user=text,
    )


BUILDERS: dict[str, Callable[[int, random.Random, bool], dict[str, Any]]] = {
    "suggest_stations": station_case,
    "suggest_places": place_case,
    "reverse_geocode": geocode_case,
    "station_departures": departures_case,
    "get_station": station_detail_case,
    "list_feeds": feeds_case,
    "plan_journey": lambda i, rng, eval_mode: route_case(i, rng, eval_mode, False),
    "plan_route_map": lambda i, rng, eval_mode: route_case(i, rng, eval_mode, True),
    "no_tool_call": no_call_case,
}


def build_class(
    class_name: str, count: int, seed: int, eval_mode: bool
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < count:
        candidate = BUILDERS[class_name](attempts, rng, eval_mode)
        key = input_key(candidate)
        attempts += 1
        if key in seen:
            if attempts > count * 200:
                raise RuntimeError(f"Unable to generate {count} unique {class_name} records")
            continue
        seen.add(key)
        candidate["id"] = (
            f"eval-bal-{class_name}-{len(rows) + 1:04d}"
            if eval_mode
            else f"bal-{class_name}-{len(rows) + 1:04d}"
        )
        rows.append(eval_row(candidate) if eval_mode else candidate)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate class-balanced transit tool-routing data.")
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--eval-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument(
        "--train-output", type=Path, default=Path("data/raw/synthetic_balanced.jsonl")
    )
    parser.add_argument(
        "--eval-output", type=Path, default=Path("data/eval/eval_balanced.jsonl")
    )
    args = parser.parse_args()
    if args.per_class < 1 or args.eval_per_class < 1:
        parser.error("--per-class and --eval-per-class must be positive")

    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for index, class_name in enumerate(CLASSES):
        train.extend(build_class(class_name, args.per_class, args.seed + index, False))
        evaluation.extend(
            build_class(class_name, args.eval_per_class, args.seed + 1000 + index, True)
        )
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(evaluation)
    write_jsonl(args.train_output, train)
    write_jsonl(args.eval_output, evaluation)
    print(
        f"wrote train={len(train)} ({args.per_class}/class) -> {args.train_output}"
    )
    print(
        f"wrote eval={len(evaluation)} ({args.eval_per_class}/class) -> {args.eval_output}"
    )


if __name__ == "__main__":
    main()
