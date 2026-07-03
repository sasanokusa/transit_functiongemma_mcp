#!/usr/bin/env python3
"""Render normalized transit JSON as Japanese using templates only."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EMPTY_MESSAGE = (
    "条件に合う候補が見つかりませんでした。\n"
    "出発地・目的地・時刻を変えて再検索してください。"
)
ERROR_MESSAGE = "乗換情報を取得できませんでした。条件を確認して再検索してください。"


def _values(values: Iterable[Any] | None) -> list[str]:
    return [str(value) for value in (values or []) if value is not None and str(value)]


def _station_list(values: list[str]) -> str:
    return "、".join(values)


def _with_station_suffix(value: Any) -> str:
    text = str(value)
    return text if text.endswith("駅") else text + "駅"


def _summary(summary: dict[str, Any]) -> str:
    parts: list[str] = []
    if summary.get("duration_min") is not None:
        parts.append(f"所要時間{summary['duration_min']}分")
    if summary.get("fare_yen") is not None:
        parts.append(f"{summary['fare_yen']}円")
    if summary.get("transfers") is not None:
        parts.append(f"乗換{summary['transfers']}回")
    return " / ".join(parts)


def _single_summary_sentence(summary: dict[str, Any]) -> str | None:
    parts: list[str] = []
    if summary.get("duration_min") is not None:
        parts.append(f"所要時間は{summary['duration_min']}分")
    if summary.get("fare_yen") is not None:
        parts.append(f"運賃は{summary['fare_yen']}円")
    if summary.get("transfers") is not None:
        parts.append(f"乗換は{summary['transfers']}回")
    return "、".join(parts) + "です。" if parts else None


def _leg(leg: dict[str, Any]) -> str | None:
    origin = leg.get("from")
    destination = leg.get("to")
    if not origin and not destination:
        return None
    left = " ".join(str(v) for v in (leg.get("departure_time"), origin) if v is not None)
    right = " ".join(str(v) for v in (leg.get("arrival_time"), destination) if v is not None)
    text = f"{left} → {right}" if left and right else left or right
    if leg.get("line"):
        text += f"（{leg['line']}）"
    elif leg.get("type") in {"walk", "walking", "foot"}:
        text += "（徒歩）"
    return text


def _is_walk(leg: dict[str, Any]) -> bool:
    return str(leg.get("type") or "").casefold() in {"walk", "walking", "foot"}


def _merge_walk_legs(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for leg in legs:
        if not _is_walk(leg) or not merged or not _is_walk(merged[-1]):
            merged.append(dict(leg))
            continue
        previous = merged[-1]
        previous["to"] = leg.get("to") or previous.get("to")
        previous["to_id"] = leg.get("to_id") or previous.get("to_id")
        previous["arrival_time"] = leg.get("arrival_time") or previous.get("arrival_time")
        durations = [
            value
            for value in (previous.get("duration_min"), leg.get("duration_min"))
            if isinstance(value, (int, float))
        ]
        if durations:
            previous["duration_min"] = sum(durations)
    return merged


def _route_block(route: dict[str, Any], heading: str | None = None) -> str:
    lines: list[str] = []
    summary = _summary(route.get("summary") or {})
    if heading:
        lines.append(f"{heading}{summary}" if summary else heading.rstrip())
    elif summary:
        sentence = _single_summary_sentence(route.get("summary") or {})
        if sentence:
            lines.append(sentence)
    legs = [leg for leg in route.get("legs", []) or [] if isinstance(leg, dict)]
    for leg in _merge_walk_legs(legs):
        if isinstance(leg, dict):
            rendered = _leg(leg)
            if rendered:
                lines.append(rendered)
    return "\n".join(lines)


def _render_routes(data: dict[str, Any], max_routes: int | None) -> str:
    routes = [route for route in (data.get("routes") or []) if isinstance(route, dict)]
    if not routes:
        return EMPTY_MESSAGE

    query = data.get("query") or {}
    time_mode = query.get("time_mode")
    requested_time = query.get("time")
    avoid = _values(query.get("avoid_station_texts"))
    avoid_lines = _values(query.get("avoid_line_texts"))
    via = _values(query.get("via_station_texts"))
    preferred_lines = _values(query.get("preferred_line_texts"))
    allowed_operator_groups = _values(query.get("allowed_operator_groups"))
    avoid_operator_groups = _values(query.get("avoid_operator_groups"))
    avoid_modes = _values(query.get("avoid_modes"))
    if requested_time and time_mode in {"arrive_by", "departure_at", "depart_at"}:
        valid_time = [
            route
            for route in routes
            if (route.get("constraint_check") or {}).get("time_satisfied", False)
        ]
        if not valid_time:
            if time_mode == "arrive_by":
                return f"{requested_time}までに到着する候補は見つかりませんでした。条件を変えて再検索してください。"
            return f"{requested_time}以降に出発する候補は見つかりませんでした。条件を変えて再検索してください。"
        routes = valid_time

    valid_avoid = [
        route
        for route in routes
        if (route.get("constraint_check") or {}).get("avoid_satisfied", True)
    ]
    display_routes = routes
    intro: list[str] = []
    if avoid:
        names = _station_list(avoid)
        if valid_avoid:
            display_routes = valid_avoid
            if len(valid_avoid) == 1:
                intro.append(f"{names}を避ける候補です。")
            else:
                intro.append(f"{names}を避ける候補を{len(valid_avoid)}件見つけました。")
        else:
            intro.extend(
                [
                    f"{names}を完全に避ける候補は見つかりませんでした。",
                    f"以下は{names}を通る候補です。",
                ]
            )
    elif len(routes) > 1:
        intro.append(f"候補を{len(routes)}件見つけました。")
    else:
        intro.append("経路候補です。")

    if avoid_lines:
        valid_line_avoid = [
            route
            for route in display_routes
            if (route.get("constraint_check") or {}).get("avoid_line_satisfied", True)
        ]
        line_names = _station_list(avoid_lines)
        if valid_line_avoid:
            display_routes = valid_line_avoid
            intro.append(f"{line_names}を使わない候補だけを表示します。")
        else:
            intro.append(
                f"MCPが返した候補内では{line_names}を避ける経路を確認できませんでした。"
            )

    if preferred_lines:
        matching_lines = [
            route
            for route in display_routes
            if (route.get("constraint_check") or {}).get("line_satisfied", False)
        ]
        line_names = _station_list(preferred_lines)
        if matching_lines:
            display_routes = matching_lines
            intro.append(f"{line_names}を使う候補だけを表示します。")
        else:
            intro.append(
                f"MCPが返した候補内では{line_names}を使う経路を確認できませんでした。"
            )

    if allowed_operator_groups or avoid_operator_groups or avoid_modes:
        valid_operator = [
            route
            for route in display_routes
            if (route.get("operator_check") or {}).get("satisfied", False)
        ]
        if not valid_operator:
            if "tokyo_metro" in allowed_operator_groups:
                return "東京メトロだけの候補は見つかりませんでした。条件を変えて再検索してください。"
            if "toei_subway" in allowed_operator_groups:
                return "都営地下鉄だけの候補は見つかりませんでした。条件を変えて再検索してください。"
            if "subway" in allowed_operator_groups:
                return "地下鉄だけの候補は見つかりませんでした。条件を変えて再検索してください。"
            return "指定された交通機関の条件を満たす候補は見つかりませんでした。"
        display_routes = valid_operator
        if "tokyo_metro" in allowed_operator_groups:
            intro.append("東京メトロだけの候補を表示します。")
        elif "toei_subway" in allowed_operator_groups:
            intro.append("都営地下鉄だけの候補を表示します。")
        elif "subway" in allowed_operator_groups:
            intro.append("地下鉄だけの候補を表示します。")
        if "JR" in avoid_operator_groups:
            intro.append("JRを使わない候補だけを表示します。")
        if "subway" in avoid_operator_groups:
            intro.append("地下鉄を使わない候補だけを表示します。")
        elif "tokyo_metro" in avoid_operator_groups:
            intro.append("東京メトロを使わない候補だけを表示します。")
        elif "toei_subway" in avoid_operator_groups:
            intro.append("都営地下鉄を使わない候補だけを表示します。")
        if "bus" in avoid_modes:
            intro.append("バスを使わない候補だけを表示します。")

    ranking = data.get("ranking") or {}
    if ranking.get("message"):
        intro.append(str(ranking["message"]))

    if max_routes is not None:
        display_routes = display_routes[: max(0, max_routes)]

    blocks: list[str] = []
    multi = len(display_routes) > 1 or (avoid and not valid_avoid) or bool(preferred_lines)
    for index, route in enumerate(display_routes, 1):
        heading = f"【候補{index}】" if multi else None
        block = _route_block(route, heading)
        if block:
            blocks.append(block)

    closing: list[str] = []
    if avoid and valid_avoid:
        subject = "これらの候補では" if len(display_routes) > 1 else "この候補では"
        closing.append(
            f"{subject}{_station_list([_with_station_suffix(name) for name in avoid])}を通りません。"
        )
    if via:
        missing: list[str] = []
        for route in display_routes:
            missing.extend(_values((route.get("constraint_check") or {}).get("missing_via_station_texts")))
        if missing:
            closing.append(f"{_station_list(list(dict.fromkeys(missing)))}を経由することは確認できませんでした。")
        else:
            closing.append(f"{_station_list(via)}を経由する候補です。")
    if ranking.get("priority") == "less_walk" and any(
        any(_is_walk(leg) for leg in route.get("legs", []) if isinstance(leg, dict))
        for route in display_routes
    ):
        closing.append("※駅構内・駅前の徒歩を含みます。")
    closing.extend(_values(query.get("station_resolution_notes")))
    return "\n\n".join(part for part in ("\n".join(intro), *blocks, "\n".join(closing)) if part)


def _render_suggestions(data: dict[str, Any]) -> str:
    suggestions = [item for item in (data.get("suggestions") or []) if isinstance(item, dict)]
    if not suggestions:
        return EMPTY_MESSAGE
    station = data.get("suggestion_type") == "station"
    lines = ["駅候補を見つけました。" if station else "場所候補を見つけました。", ""]
    names = Counter(str(item.get("name")) for item in suggestions if item.get("name"))
    for index, item in enumerate(suggestions, 1):
        name = item.get("name")
        if not name:
            continue
        display = str(name)
        if station:
            display = _with_station_suffix(display)
        if names[str(name)] > 1 and item.get("source_label"):
            display += f"（{item['source_label']}）"
        lines.append(f"{index}. {display}")
    if len(lines) == 2:
        return EMPTY_MESSAGE
    lines.extend(["", "どの駅を使いますか？" if station else "どれを目的地にしますか？"])
    return "\n".join(lines)


def _render_departures(data: dict[str, Any]) -> str:
    departures = [item for item in (data.get("departures") or []) if isinstance(item, dict)]
    if not departures:
        return EMPTY_MESSAGE
    station = data.get("station") or {}
    title = (
        f"{_with_station_suffix(station['name'])}の発車情報です。"
        if station.get("name")
        else "発車情報です。"
    )
    lines = [title, ""]
    for departure in departures:
        fields = _values(
            (departure.get("time"), departure.get("line"), departure.get("direction"))
        )
        if fields:
            lines.append(" ".join(fields))
    return "\n".join(lines) if len(lines) > 2 else EMPTY_MESSAGE


def _render_station(data: dict[str, Any]) -> str:
    station = data.get("station") or {}
    if not station.get("name"):
        return EMPTY_MESSAGE
    return f"{_with_station_suffix(station['name'])}の情報です。"


def _render_feeds(data: dict[str, Any]) -> str:
    feeds = [item for item in (data.get("feeds") or []) if isinstance(item, dict)]
    if not feeds:
        return EMPTY_MESSAGE
    lines = ["利用できる交通データです。", ""]
    for index, feed in enumerate(feeds, 1):
        name = feed.get("name") or feed.get("title") or feed.get("id")
        if name:
            lines.append(f"{index}. {name}")
    return "\n".join(lines) if len(lines) > 2 else EMPTY_MESSAGE


def render_clarification(missing: Iterable[Any], question: str | None = None) -> str:
    if question:
        return question
    missing_set = {str(value) for value in missing}
    if missing_set == {"origin"}:
        return "出発地が不足しています。どこから出発しますか？"
    if missing_set == {"destination"}:
        return "目的地が不足しています。どこまで行きますか？"
    if "station_id" in missing_set:
        return "利用する駅を選んでください。"
    if {"origin", "destination"}.issubset(missing_set):
        return "出発地と目的地が不足しています。どこからどこまで行きますか？"
    return "検索に必要な情報が不足しています。条件を追加してください。"


def render_answer(data: dict[str, Any], max_routes: int | None = None) -> str:
    """Render only values present in normalized JSON; never call an LLM."""
    if data.get("status") == "clarification":
        return render_clarification(data.get("missing") or [], data.get("question"))
    if data.get("status") == "error":
        return ERROR_MESSAGE
    tool_name = data.get("raw_tool_name")
    if tool_name in {"suggest_stations", "suggest_places", "reverse_geocode"}:
        return _render_suggestions(data)
    if tool_name == "station_departures":
        return _render_departures(data)
    if tool_name == "get_station":
        return _render_station(data)
    if tool_name == "list_feeds":
        return _render_feeds(data)
    return _render_routes(data, max_routes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render normalized transit JSON in Japanese.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-routes", type=int)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(render_answer(data, args.max_routes))


if __name__ == "__main__":
    main()
