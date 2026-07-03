#!/usr/bin/env python3
"""Create colloquial Japanese train/eval cases independent of the synthetic templates."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any


REFERENCE = "2026-06-29 09:00 Asia/Tokyo"
STATIONS = [
    ("横浜", "demo-feed:yokohama"),
    ("上野", "demo-feed:ueno"),
    ("東京", "demo-feed:tokyo"),
    ("渋谷", "demo-feed:shibuya"),
    ("新宿", "demo-feed:shinjuku"),
    ("品川", "demo-feed:shinagawa"),
    ("町田", "demo-feed:machida"),
    ("池袋", "demo-feed:ikebukuro"),
    ("秋葉原", "demo-feed:akihabara"),
    ("大宮", "demo-feed:omiya"),
]


def target(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool_name": tool_name, "arguments": arguments}


def train_row(row_id: str, user: str | None, assistant: dict[str, Any], history=None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": row_id,
        "reference_datetime": REFERENCE,
        "assistant": assistant,
    }
    if user is not None:
        row["user"] = user
    if history:
        row["history"] = history
    return row


def eval_row(row_id: str, user: str | None, expected_tool: str | None, arguments=None, history=None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": row_id,
        "reference_datetime": REFERENCE,
        "expected_tool": expected_tool,
    }
    if user is not None:
        row["user"] = user
    if arguments is not None:
        row["expected_arguments"] = arguments
    if history:
        row["history"] = history
    return row


def history_for(origin: tuple[str, str], destination: tuple[str, str], user: str) -> list[dict[str, Any]]:
    origin_name, origin_id = origin
    destination_name, destination_id = destination
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "suggest_stations", "arguments": {"q": origin_name, "limit": 5}}}],
        },
        {"role": "tool", "content": {"name": "suggest_stations", "response": {"stations": [{"id": origin_id, "name": origin_name}]}}},
        {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "suggest_stations", "arguments": {"q": destination_name, "limit": 5}}}],
        },
        {"role": "tool", "content": {"name": "suggest_stations", "response": {"stations": [{"id": destination_id, "name": destination_name}]}}},
    ]


def build_train() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    station_phrases = [
        ("東急の横浜駅ってどれだっけ", "横浜"),
        ("東京駅の候補を出して", "東京"),
        ("新宿って駅を探したい", "新宿"),
        ("渋谷駅、どれ選べばいい?", "渋谷"),
        ("町田の駅IDほしい", "町田"),
        ("池袋を鉄道駅として検索", "池袋"),
        ("品川駅を候補検索してくれる", "品川"),
        ("秋葉原の駅候補お願い", "秋葉原"),
        ("大宮駅って何件ある?", "大宮"),
        ("上野を駅として引いて", "上野"),
        ("JRの横浜駅を探して", "横浜"),
        ("駅名「東京」のID候補", "東京"),
    ]
    for index, (user, query) in enumerate(station_phrases):
        rows.append(train_row(f"real-train-station-{index:03d}", user, target("suggest_stations", {"q": query, "limit": 5})))

    place_phrases = [
        "東京タワーを場所として探して",
        "スカイツリーって施設を検索して",
        "東京都庁を駅じゃなく場所で引いて",
        "日本武道館の場所候補を出して",
        "六本木ヒルズを施設として探す",
        "幕張メッセって場所どこ",
        "横浜赤レンガ倉庫を場所検索",
        "東京ドームを乗降場所の候補にしたい",
        "浅草寺を施設名で検索",
        "羽田空港第3ターミナルを場所として候補検索",
        "国立競技場を場所として探してほしい",
        "東京ビッグサイトの施設候補お願い",
    ]
    place_names = ["東京タワー", "スカイツリー", "東京都庁", "日本武道館", "六本木ヒルズ", "幕張メッセ", "横浜赤レンガ倉庫", "東京ドーム", "浅草寺", "羽田空港第3ターミナル", "国立競技場", "東京ビッグサイト"]
    for index, (user, query) in enumerate(zip(place_phrases, place_names)):
        rows.append(train_row(f"real-train-place-{index:03d}", user, target("suggest_places", {"q": query, "limit": 10})))

    coords = [
        ("緯度35.6812、経度139.7671の最寄り駅", 35.6812, 139.7671),
        ("35.6586,139.7454 この座標の近くの乗り場", 35.6586, 139.7454),
        ("座標 35.6896, 139.7006 から逆引き", 35.6896, 139.7006),
        ("緯度35.4437 経度139.6380、周囲120mの駅", 35.4437, 139.638),
        ("35.7101,139.8107付近の乗車地点", 35.7101, 139.8107),
        ("この座標の駅: 35.7295, 139.7109", 35.7295, 139.7109),
        ("緯度35.5420 経度139.4455を駅に変換", 35.542, 139.4455),
        ("35.6285、139.7387の周りを探して", 35.6285, 139.7387),
    ]
    for index, (user, lat, lon) in enumerate(coords):
        rows.append(train_row(f"real-train-geo-{index:03d}", user, target("reverse_geocode", {"lat": lat, "lon": lon, "limit": 3, "radiusMeters": 200})))

    for index, (name, station_id) in enumerate(STATIONS[:8]):
        user = [
            f"{station_id} の発車標見せて",
            f"駅ID {station_id}、明日9時以降に出る便",
            f"{station_id} の発車案内を10件",
        ][index % 3]
        args: dict[str, Any] = {"id": station_id, "limit": 10}
        if "明日" in user:
            args.update({"date": "20260630", "time": "9:00"})
        rows.append(train_row(f"real-train-departures-{index:03d}", user, target("station_departures", args)))
        rows.append(train_row(f"real-train-detail-{index:03d}", f"{station_id}ってどの駅? 詳細を見たい", target("get_station", {"id": station_id})))

    feed_phrases = [
        "どの交通データに対応してる?",
        "収録してるフィード一覧",
        "データの出典を見せて",
        "対応エリアとライセンスを確認したい",
        "使える事業者の一覧ある?",
        "GTFSの収録状況を教えて",
        "利用可能なフィードを列挙して",
        "どの路線データが入ってるの",
    ]
    for index, user in enumerate(feed_phrases):
        rows.append(train_row(f"real-train-feeds-{index:03d}", user, target("list_feeds", {})))

    incomplete = [
        "東京駅まで行きたい",
        "横浜から行きたいんだけど",
        "明日の終電を調べて",
        "できれば安いやつで",
        "地図で経路を見たい",
        "上野まで。出発はまだ決めてない",
        "渋谷から、行き先はあとで",
        "始発って何時?",
        "乗換少ないやつお願い",
        "目的地だけ東京駅",
        "出発は新宿",
        "どっか行きたい",
    ]
    for index, user in enumerate(incomplete):
        rows.append(train_row(f"real-train-nocall-{index:03d}", user, {"no_tool_call": True}))

    routes = [
        ("横浜から上野までの経路を探させて", "横浜"),
        ("品川シーサイドから渋谷まで行きたい", "品川シーサイド"),
        ("町田→東京、できるだけ早く", "町田"),
        ("新宿から秋葉原へ。乗換少なめで", "新宿"),
        ("東京駅発で大宮駅着、終電", "東京"),
        ("池袋から横浜まで、渋谷は避けたい", "池袋"),
        ("上野から品川まで京浜東北線を使いたい", "上野"),
        ("渋谷から東京まで地図はいらない", "渋谷"),
        ("大宮から新宿へ明日9時着で", "大宮"),
        ("秋葉原⇒町田 マップで見せて", "秋葉原"),
        ("横浜発渋谷着、歩く距離少なめ", "横浜"),
        ("品川から上野まで普通に検索", "品川"),
    ]
    for index, (user, origin) in enumerate(routes):
        rows.append(train_row(f"real-train-route-{index:03d}", user, target("suggest_stations", {"q": origin, "limit": 5})))

    final_routes = [
        (STATIONS[0], STATIONS[1], "横浜から上野まで普通に案内", "plan_journey", {}),
        (STATIONS[2], STATIONS[3], "東京から渋谷まで地図で見たい", "plan_route_map", {"strategy": "balanced"}),
        (STATIONS[4], STATIONS[5], "新宿→品川、早いやつ", "plan_journey", {}),
        (STATIONS[6], STATIONS[7], "町田から池袋へマップ表示", "plan_route_map", {"strategy": "balanced"}),
        (STATIONS[8], STATIONS[9], "秋葉原から大宮まで終電", "plan_journey", {"date": "20260629", "type": "last"}),
        (STATIONS[1], STATIONS[0], "上野から横浜まで経路図で", "plan_route_map", {"strategy": "balanced"}),
    ]
    for index, (origin, destination, user, tool, extras) in enumerate(final_routes):
        args = {"from": origin[1], "to": destination[1], "fromLabel": origin[0], "toLabel": destination[0], **extras}
        rows.append(train_row(f"real-train-final-{index:03d}", None, target(tool, args), history_for(origin, destination, user)))
    return rows


def build_eval() -> list[dict[str, Any]]:
    rows = [
        eval_row("real-eval-station-01", "あれ、京急の横浜駅ってどれ?", "suggest_stations", {"q": "横浜", "limit": 5}),
        eval_row("real-eval-station-02", "立川駅の候補ほしい", "suggest_stations", {"q": "立川", "limit": 5}),
        eval_row("real-eval-place-01", "東京スカイツリーを駅じゃなく施設で探して", "suggest_places", {"q": "東京スカイツリー", "limit": 10}),
        eval_row("real-eval-place-02", "日本武道館って場所の候補ある?", "suggest_places", {"q": "日本武道館", "limit": 10}),
        eval_row("real-eval-geo-01", "北緯35.6812 東経139.7671の最寄り", "reverse_geocode", {"lat": 35.6812, "lon": 139.7671, "limit": 3, "radiusMeters": 200}),
        eval_row("real-eval-geo-02", "35.5075, 139.6175から駅を逆引き", "reverse_geocode", {"lat": 35.5075, "lon": 139.6175, "limit": 3, "radiusMeters": 200}),
        eval_row("real-eval-departures-01", "jp:rail:tokyo:001 の明日8時の出発便", "station_departures", {"id": "jp:rail:tokyo:001", "date": "20260630", "time": "8:00", "limit": 10}),
        eval_row("real-eval-detail-01", "jp:rail:yokohama:002は何駅なの", "get_station", {"id": "jp:rail:yokohama:002"}),
        eval_row("real-eval-feeds-01", "このAPI、どこの事業者に対応?", "list_feeds", {}),
        eval_row("real-eval-feeds-02", "交通データのライセンス一覧", "list_feeds", {}),
        eval_row("real-eval-nocall-01", "上野までお願い", None),
        eval_row("real-eval-nocall-02", "横浜から乗りたい", None),
        eval_row("real-eval-nocall-03", "終電で帰りたい", None),
        eval_row("real-eval-route-01", "品川シーサイドから新宿まで", "suggest_stations", {"q": "品川シーサイド", "limit": 5}),
        eval_row("real-eval-route-02", "池袋→町田、安めで", "suggest_stations", {"q": "池袋", "limit": 5}),
    ]
    origin, destination = STATIONS[5], STATIONS[1]
    user = "品川から上野まで、地図なしで普通に"
    rows.append(eval_row("real-eval-final-journey", None, "plan_journey", {"from": origin[1], "to": destination[1], "fromLabel": origin[0], "toLabel": destination[0]}, history_for(origin, destination, user)))
    origin, destination = STATIONS[3], STATIONS[0]
    user = "渋谷から横浜まで地図表示して"
    rows.append(eval_row("real-eval-final-map", None, "plan_route_map", {"from": origin[1], "to": destination[1], "fromLabel": origin[0], "toLabel": destination[0], "strategy": "balanced"}, history_for(origin, destination, user)))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, default=Path("data/raw/real_user_japanese.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/eval/eval_real_user_japanese.jsonl"))
    args = parser.parse_args()
    train_rows = build_train()
    eval_rows = build_eval()
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    print(f"train={len(train_rows)} -> {args.train_output}")
    print(f"eval={len(eval_rows)} -> {args.eval_output}")


if __name__ == "__main__":
    main()
