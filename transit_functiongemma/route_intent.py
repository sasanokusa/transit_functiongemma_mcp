"""Legacy rule-based semantic parser.

This module is excluded from the default runtime path. It remains for explicit
compatibility tests and offline annotation migration only.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any


_SPACE_RE = re.compile(r"[ \t\u3000]+")
_PLACE_ALIASES = {
    "ビッグサイト": "東京ビッグサイト",
    "赤レンガ倉庫": "横浜赤レンガ倉庫",
    "PayPayドーム": "福岡PayPayドーム",
    "羽田空港": "羽田空港第1・第2ターミナル",
}
_SPECIAL_TIME_PATTERNS = (
    (
        re.compile(
            r"(?:今日の)?(?:終電|最終電車|最終列車)(?:の電車)?"
            r"(?:で|に(?:間に合うように)?)?"
        ),
        "last_train",
    ),
    (re.compile(r"(?:明日の)?(?:始発|朝イチ)(?:の電車)?(?:で|に)?"), "first_train"),
)


def clean_japanese_text(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).strip())


def extract_special_time_mode(text: str) -> str | None:
    value = clean_japanese_text(text)
    for pattern, mode in _SPECIAL_TIME_PATTERNS:
        if pattern.search(value):
            return mode
    return None


def strip_special_time_phrases(text: str) -> str:
    value = clean_japanese_text(text)
    for pattern, _ in _SPECIAL_TIME_PATTERNS:
        value = pattern.sub("", value)
    return value.strip(" 、,")


def reference_datetime(value: str | None) -> datetime:
    if value:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2}))?", value)
        if match:
            return datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4) or 0),
                int(match.group(5) or 0),
            )
    return datetime.now()


def normalize_date(text: str, reference: str | None = None) -> str | None:
    value = clean_japanese_text(text)
    base = reference_datetime(reference)
    match = re.search(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日", value)
    if match:
        return f"{int(match.group(1) or base.year):04d}{int(match.group(2)):02d}{int(match.group(3)):02d}"
    match = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)", value)
    if match:
        return f"{base.year:04d}{int(match.group(1)):02d}{int(match.group(2)):02d}"
    if "明日" in value:
        return (base + timedelta(days=1)).strftime("%Y%m%d")
    if any(word in value for word in ("今日", "本日", "終電", "最終列車", "始発")):
        return base.strftime("%Y%m%d")
    return None


def normalize_time(text: str) -> str | None:
    value = clean_japanese_text(text)
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", value)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
    else:
        match = re.search(r"(?<!\d)(\d{1,2})時(?:(\d{1,2})分|半)?", value)
        if not match:
            return None
        hour = int(match.group(1))
        minute = 30 if "半" in match.group(0) else int(match.group(2) or 0)
    prefix = value[max(0, match.start() - 4) : match.start()]
    night_clock = prefix.endswith("夜") and not prefix.endswith("深夜")
    if ("午後" in prefix or night_clock) and hour < 12:
        hour += 12
    elif "午前" in prefix and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _clean_endpoint(value: str, *, destination: bool = False) -> str:
    value = value.strip(" 、,。.?？!")
    if not destination:
        value = re.sub(
            r"^(?:(?:今日|明日|本日|終電|最終列車|始発)[^、。]*?で|"
            r"(?:今日|明日|本日)?\s*\d{1,2}(?::\d{2}|時(?:\d{1,2}分|半)?)?\s*出発で)",
            "",
            value,
        )
    value = value.removesuffix("へ").removesuffix("駅").strip()
    if destination:
        value = _PLACE_ALIASES.get(value, value)
    return value


def extract_route_endpoints(text: str) -> tuple[str, str] | None:
    value = strip_special_time_phrases(text)
    departure = re.search(r"(.+?)(?:駅)?発で(.+?)(?:駅)?まで", value)
    if departure:
        return (
            _clean_endpoint(departure.group(1)),
            _clean_endpoint(departure.group(2), destination=True),
        )
    if "から" not in value:
        return None
    before, after = value.split("から", 1)
    origin = _clean_endpoint(before)
    destination = re.split(
        r"(?:まで|[、,。]|へ(?:行|向)|に向か|行きたい|行ける|\s+(?:早|安|乗換|歩))",
        after,
        maxsplit=1,
    )[0]
    destination = _clean_endpoint(destination, destination=True)
    if origin in {"ここ", "この辺", "そこ"} or destination in {
        "出たい",
        "帰りたい",
        "どこか",
        "検索して",
    }:
        return None
    if "わからない" in origin or "まだ決めてない" in value or "未定" in value:
        return None
    if not origin or not destination:
        return None
    return origin, destination


def _unique(matches: list[str]) -> list[str]:
    values: list[str] = []
    for value in matches:
        cleaned = value.strip(" 、,。.?？!をはだけ")
        cleaned = cleaned.removesuffix("嫌いだから")
        cleaned = cleaned.removesuffix("駅")
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values


def _captures(text: str, patterns: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(1) for match in re.finditer(pattern, text))
    return _unique(values)


def extract_route_intent(text: str, reference: str | None = None) -> dict[str, Any]:
    """Extract explicit route slots without predicting a route or transit facts."""
    value = clean_japanese_text(text)
    endpoints = extract_route_endpoints(value)
    avoid_stations = _captures(
        value,
        (
            r"(?:^|[、,。])\s*([^、,。\s]+?)(?:駅)?(?:だけ)?(?:は|を)?(?:避け(?:て|たい|る)|通りたくない|通らない(?:で|ルートで))",
            r"(?:^|[、,。])\s*([^、,。\s]+?)(?:駅)?(?:は)?嫌いだから避けて",
            r"(?:^|[、,。])\s*([^、,。\s]+?)(?:駅)?経由(?:は|が)?嫌",
        ),
    )
    via_stations = _captures(
        value,
        (
            r"(?:^|[、,。])\s*([^、,。\s]+?)(?:駅)?経由(?:で|して|する)?",
            r"(?:^|[、,。])\s*([^、,。\s]+?)(?:駅)?(?:に寄って|を通って)",
        ),
    )
    # Negative phrases such as 「秋葉原経由は嫌」 also match the surface
    # via pattern. Avoidance wins; never send the same station as a waypoint.
    via_stations = [station for station in via_stations if station not in avoid_stations]
    avoid_lines = _captures(
        value,
        (
            r"(?:^|[、,。])\s*([^、,。\s]+?線)(?:は|を)?(?:なし|使わない|避け|嫌)",
            r"(?:^|[、,。])\s*(JR)(?:は|を)?(?:なし|使わない|避け|嫌)",
        ),
    )
    if re.search(r"JR\s*(?:は|を)?(?:なし|使わない|避け|嫌)", value) and "JR" not in avoid_lines:
        avoid_lines.append("JR")
    priority = None
    if re.search(r"最速|早いやつ|早め|なるはや|なるべく早く|速いやつ", value):
        priority = "fast"
    elif re.search(r"最安|安いやつ|安め|安く", value):
        priority = "cheap"
    elif re.search(r"乗(?:り)?換(?:え)?(?:が)?少|乗換あんましない", value):
        priority = "few_transfers"
    elif re.search(r"歩き少な|徒歩少な|歩きたくない|歩くのだるい|歩く距離短", value):
        priority = "less_walk"

    time_mode = extract_special_time_mode(value)
    if time_mode:
        pass
    elif re.search(r"(?:着|到着|までに)", value):
        time_mode = "arrive_by"
    elif re.search(
        r"(?:出発|出たい|出る|(?<!始)(?<!終)(?:\d{1,2}(?::\d{2}|時(?:\d{1,2}分|半)?)?)\s*発)",
        value,
    ):
        time_mode = "departure_at"

    return {
        "origin_text": endpoints[0] if endpoints else None,
        "destination_text": endpoints[1] if endpoints else None,
        "avoid_station_texts": avoid_stations,
        "via_station_texts": via_stations,
        "avoid_line_texts": avoid_lines,
        "priority": priority,
        "time_mode": time_mode,
        "date": normalize_date(value, reference),
        "time": normalize_time(value),
        "graphical": bool(re.search(r"地図|マップ|経路図|グラフィカル", value)),
    }


def strategy_from_priority(priority: str | None) -> str:
    return {
        "fast": "fastest",
        "cheap": "lowestFare",
        "few_transfers": "fewestTransfers",
        "less_walk": "shortestWalk",
    }.get(priority, "balanced")


def mcp_time_type(time_mode: str | None) -> str | None:
    return {
        "last_train": "last",
        "first_train": "first",
        "arrive_by": "arrival",
        "departure_at": "departure",
    }.get(time_mode)


__all__ = [
    "clean_japanese_text",
    "extract_special_time_mode",
    "extract_route_endpoints",
    "extract_route_intent",
    "mcp_time_type",
    "normalize_date",
    "normalize_time",
    "strategy_from_priority",
    "strip_special_time_phrases",
]
