#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

REF = "2026-06-28 09:00 Asia/Tokyo"

STATIONS = [
    ("東京", "demo-feed:tokyo"),
    ("新宿", "demo-feed:shinjuku"),
    ("渋谷", "demo-feed:shibuya"),
    ("池袋", "demo-feed:ikebukuro"),
    ("上野", "demo-feed:ueno"),
    ("品川", "demo-feed:shinagawa"),
    ("横浜", "demo-feed:yokohama"),
    ("大宮", "demo-feed:omiya"),
    ("秋葉原", "demo-feed:akihabara"),
    ("町田", "demo-feed:machida"),
    ("立川", "demo-feed:tachikawa"),
    ("川崎", "demo-feed:kawasaki"),
    ("大手町", "demo-feed:otemachi"),
    ("三田", "demo-feed:mita"),
    ("羽田空港", "demo-feed:haneda"),
    ("成田空港", "demo-feed:narita"),
    ("京都", "demo-feed:kyoto"),
    ("新大阪", "demo-feed:shin-osaka"),
    ("大阪", "demo-feed:osaka"),
    ("名古屋", "demo-feed:nagoya"),
    ("仙台", "demo-feed:sendai"),
]

PLACES = [
    "東京タワー",
    "東京スカイツリー",
    "渋谷区神南1丁目",
    "東京都庁",
    "日本武道館",
    "横浜赤レンガ倉庫",
    "東京ビッグサイト",
    "幕張メッセ",
    "浅草寺",
    "六本木ヒルズ",
]

COORDS = [
    (35.6812, 139.7671, "東京駅付近"),
    (35.7101, 139.8107, "スカイツリー付近"),
    (35.6586, 139.7454, "東京タワー付近"),
    (35.6896, 139.7006, "新宿付近"),
    (35.4437, 139.6380, "横浜付近"),
]

SEARCH_PATTERNS = [
    "{name}駅を検索して",
    "{name}の駅候補を出して",
    "{name}駅の候補を5件",
    "{name}って駅を探して",
    "乗車駅として{name}を検索",
    "降車駅として{name}を検索して",
]

PLACE_PATTERNS = [
    "{name}を場所として探して",
    "{name}の最寄り候補を探して",
    "{name}周辺の乗車地点を検索",
    "目的地が{name}なんだけど候補出して",
    "住所/施設として{name}を検索",
]

NEGATIVES = [
    "東京駅から行きたい",
    "大阪駅までの運賃を教えて",
    "明日の経路を調べて",
    "終電を調べて",
    "ここから帰りたい",
    "駅まで行きたい",
    "なるはやで行ける？",
    "安いルートある？",
    "地図で見せて",
    "何時に出ればいい？",
]


def row(id_: str, user: str, assistant: dict[str, Any], ref: str = REF, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": id_, "reference_datetime": ref}
    if history:
        out["history"] = history
    else:
        out["user"] = user
    out["assistant"] = assistant
    return out


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


def tool_resp(name: str, response: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "content": {"name": name, "response": response}}


def station_response(name: str, sid: str) -> dict[str, Any]:
    return {"stations": [{"id": sid, "name": name}]}


def route_user(fr: str, to: str, kind: str, dt_text: str = "", extra: str = "") -> str:
    templates = [
        "{fr}から{to}まで{dt}{extra}",
        "{fr}駅発で{to}駅まで{dt}{extra}",
        "{fr}→{to}の経路{dt}{extra}",
        "{fr}から{to}へ行きたい{dt}{extra}",
    ]
    t = random.choice(templates)
    dt = (dt_text + " ") if dt_text else ""
    ex = ("。" + extra) if extra else ""
    return t.format(fr=fr, to=to, dt=dt, extra=ex)


def make_history(fr: str, fid: str, to: str, tid: str, original_user: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": original_user},
        tool_call("suggest_stations", {"q": fr, "limit": 5}),
        tool_resp("suggest_stations", station_response(fr, fid)),
        tool_call("suggest_stations", {"q": to, "limit": 5}),
        tool_resp("suggest_stations", station_response(to, tid)),
    ]


def add_datetime(args: dict[str, Any], phrase: str) -> dict[str, Any]:
    if phrase == "now":
        return args
    if phrase == "tomorrow_0830_dep":
        args.update({"date": "20260629", "time": "8:30", "type": "departure"})
    elif phrase == "tomorrow_0900_arr":
        args.update({"date": "20260629", "time": "9:00", "type": "arrival"})
    elif phrase == "jul1_1800_arr":
        args.update({"date": "20260701", "time": "18:00", "type": "arrival"})
    elif phrase == "first":
        args.update({"date": "20260628", "type": "first"})
    elif phrase == "last":
        args.update({"date": "20260628", "type": "last"})
    return args


def build_records(seed: int, count_routes: int) -> list[dict[str, Any]]:
    random.seed(seed)
    records: list[dict[str, Any]] = []
    n = 1

    # Single-step resolution tasks.
    for name, _sid in STATIONS:
        for pat in SEARCH_PATTERNS[:4]:
            records.append(row(f"syn-{n:04d}", pat.format(name=name), {"tool_name": "suggest_stations", "arguments": {"q": name + ("駅" if "駅" in pat and not name.endswith("駅") else ""), "limit": 5}})); n += 1
    for name in PLACES:
        for pat in PLACE_PATTERNS:
            records.append(row(f"syn-{n:04d}", pat.format(name=name), {"tool_name": "suggest_places", "arguments": {"q": name, "limit": 10}})); n += 1
    for lat, lon, label in COORDS:
        records.append(row(f"syn-{n:04d}", f"緯度{lat}、経度{lon}の近くの駅を探して", {"tool_name": "reverse_geocode", "arguments": {"lat": lat, "lon": lon, "limit": 3, "radiusMeters": 80}})); n += 1
        records.append(row(f"syn-{n:04d}", f"現在地 {lat},{lon} から近くの乗車地点", {"tool_name": "reverse_geocode", "arguments": {"lat": lat, "lon": lon, "limit": 3, "radiusMeters": 80}})); n += 1
    for name, sid in STATIONS[:10]:
        records.append(row(f"syn-{n:04d}", f"{sid} の駅詳細を見たい", {"tool_name": "get_station", "arguments": {"id": sid}})); n += 1
        records.append(row(f"syn-{n:04d}", f"{sid} の発車案内を10件", {"tool_name": "station_departures", "arguments": {"id": sid, "limit": 10}})); n += 1
        records.append(row(f"syn-{n:04d}", f"{sid} の明日9時からの発車案内", {"tool_name": "station_departures", "arguments": {"id": sid, "date": "20260629", "time": "9:00", "limit": 20}})); n += 1
    for text in ["対応している交通データを見せて", "利用できる交通フィード一覧", "ライセンスとデータ出典を確認", "どの交通データを使ってる？"]:
        records.append(row(f"syn-{n:04d}", text, {"tool_name": "list_feeds", "arguments": {}})); n += 1

    # Negatives / no-call cases.
    for text in NEGATIVES:
        records.append(row(f"syn-{n:04d}", text, {"no_tool_call": True})); n += 1
    # Balance the safety class: a known origin without a destination (and vice
    # versa) must not trigger even a place-resolution call.
    for name, _sid in STATIONS:
        records.append(row(f"syn-{n:04d}", f"{name}から出発したい", {"no_tool_call": True})); n += 1
        records.append(row(f"syn-{n:04d}", f"{name}までの所要時間を教えて", {"no_tool_call": True})); n += 1

    # Route requests: first turn must resolve origin.
    for _ in range(count_routes):
        (fr, fid), (to, tid) = random.sample(STATIONS, 2)
        user = route_user(fr, to, "normal")
        records.append(row(f"syn-{n:04d}", user, {"tool_name": "suggest_stations", "arguments": {"q": fr, "limit": 5}})); n += 1

        # Second step: after origin is resolved, resolve destination.
        hist = [
            {"role": "user", "content": user},
            tool_call("suggest_stations", {"q": fr, "limit": 5}),
            tool_resp("suggest_stations", station_response(fr, fid)),
        ]
        records.append(row(f"syn-{n:04d}", "", {"tool_name": "suggest_stations", "arguments": {"q": to, "limit": 5}}, history=hist)); n += 1

        # Final step: both are resolved; choose journey or route map.
        mapish = random.random() < 0.25
        dt_key, dt_text = random.choice([
            ("now", ""),
            ("tomorrow_0830_dep", "明日8時30分出発で"),
            ("tomorrow_0900_arr", "明日9時着で"),
            ("jul1_1800_arr", "7月1日18時までに着きたい"),
            ("first", "始発で"),
            ("last", "終電で"),
        ])
        strategy_word, strategy = random.choice([
            *(("", None),) * 8,
            ("早いやつ", "fastest"),
            ("乗換少なめ", "fewestTransfers"),
            ("安いやつ", "lowestFare"),
            ("歩き少なめ", "shortestWalk"),
            ("バランスよく", "balanced"),
        ])
        visual_word = random.choice(["地図で", "マップで", "グラフィカルに", "経路図で"]) if mapish else ""
        original = route_user(fr, to, "normal", dt_text, " ".join(x for x in [visual_word, strategy_word] if x))
        hist2 = make_history(fr, fid, to, tid, original)
        args = {"from": fid, "to": tid, "fromLabel": fr, "toLabel": to}
        args = add_datetime(args, dt_key)
        # strategy exists only on plan_route_map. Do not approximate fastest,
        # cheapest, or shortest-walk using unrelated plan_journey arguments.
        if mapish or strategy is not None:
            if strategy:
                args["strategy"] = strategy
            records.append(row(f"syn-{n:04d}", "", {"tool_name": "plan_route_map", "arguments": args}, history=hist2)); n += 1
        else:
            records.append(row(f"syn-{n:04d}", "", {"tool_name": "plan_journey", "arguments": args}, history=hist2)); n += 1

    return records


def eval_rows() -> list[dict[str, Any]]:
    return [
        {"id":"eval-gen-001","reference_datetime":REF,"user":"秋葉原という駅のID候補をお願い","expected_tool":"suggest_stations","expected_arguments":{"q":"秋葉原","limit":5}},
        {"id":"eval-gen-002","reference_datetime":REF,"user":"施設名「東京スカイツリー」の候補検索","expected_tool":"suggest_places","expected_arguments":{"q":"東京スカイツリー","limit":10}},
        {"id":"eval-gen-003","reference_datetime":REF,"user":"緯度35.7101、経度139.8107の近くを探して","expected_tool":"reverse_geocode","expected_arguments":{"lat":35.7101,"lon":139.8107}},
        {"id":"eval-gen-004","reference_datetime":REF,"user":"収録済み交通データの出典一覧を確認したい","expected_tool":"list_feeds","expected_arguments":{}},
        {"id":"eval-gen-005","reference_datetime":REF,"user":"demo-feed:tokyo の明日9時からの発車を表示","expected_tool":"station_departures","expected_arguments":{"id":"demo-feed:tokyo","date":"20260629","time":"9:00"},"expected_normalized":{"date":"20260629","time":"9:00"}},
        {"id":"eval-gen-006","reference_datetime":REF,"user":"東京駅から行く経路","expected_tool":None,"missing_info":True},
        {"id":"eval-gen-007","reference_datetime":REF,"user":"名古屋駅までの料金","expected_tool":None,"missing_info":True},
        {"id":"eval-gen-008","reference_datetime":REF,"user":"明日の終電を調べて","expected_tool":None,"missing_info":True},
        {"id":"eval-gen-009","reference_datetime":REF,"user":"町田から東京まで安いやつを地図で","expected_tool":"suggest_stations","expected_arguments":{"q":"町田","limit":5}},
        {"id":"eval-gen-010","reference_datetime":REF,"user":"新宿から渋谷まで明日9時着で","expected_tool":"suggest_stations","expected_arguments":{"q":"新宿","limit":5}},
        {"id":"eval-gen-011","reference_datetime":REF,"user":"現在地 35.6812,139.7671 から近い駅","expected_tool":"reverse_geocode","expected_arguments":{"lat":35.6812,"lon":139.7671}},
        {"id":"eval-gen-012","reference_datetime":REF,"user":"東京タワーを目的地候補として解決して","expected_tool":"suggest_places","expected_arguments":{"q":"東京タワー","limit":10}},
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a larger deterministic synthetic dataset for transit MCP routing.")
    ap.add_argument("--train-output", type=Path, default=Path("data/raw/synthetic_generated.jsonl"))
    ap.add_argument("--eval-output", type=Path, default=Path("data/eval/eval_generated.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--route-groups", type=int, default=120, help="Each route group emits origin, destination, and final planning records.")
    args = ap.parse_args()
    train = build_records(args.seed, args.route_groups)
    random.Random(args.seed).shuffle(train)
    write_jsonl(args.train_output, train)
    write_jsonl(args.eval_output, eval_rows())
    print(f"wrote train={len(train)} -> {args.train_output}")
    print(f"wrote eval={len(eval_rows())} -> {args.eval_output}")


if __name__ == "__main__":
    main()
