#!/usr/bin/env python3
"""Deterministic Tokyo line/operator classification for route filtering."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


TOKYO_METRO_LINES = (
    "銀座線",
    "丸ノ内線",
    "日比谷線",
    "東西線",
    "千代田線",
    "有楽町線",
    "半蔵門線",
    "南北線",
    "副都心線",
)
TOEI_SUBWAY_LINES = (
    "浅草線",
    "三田線",
    "新宿線",
    "大江戸線",
)
SUBWAY_LINES = TOKYO_METRO_LINES + TOEI_SUBWAY_LINES
JR_LINES = (
    "山手線",
    "中央線",
    "中央線快速",
    "中央・総武線",
    "総武線",
    "京浜東北線",
    "埼京線",
    "湘南新宿ライン",
    "常磐線",
    "横須賀線",
    "東海道線",
    "宇都宮線",
    "高崎線",
    "京葉線",
    "武蔵野線",
    "南武線",
    "横浜線",
    "青梅線",
    "五日市線",
)


def normalize_line_name(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def _contains_any(line: Any, names: tuple[str, ...]) -> bool:
    normalized = normalize_line_name(line)
    return any(normalize_line_name(name) in normalized for name in names)


def is_subway_line(line: Any) -> bool:
    return _contains_any(line, SUBWAY_LINES)


def is_tokyo_metro_line(line: Any) -> bool:
    return _contains_any(line, TOKYO_METRO_LINES)


def is_toei_subway_line(line: Any) -> bool:
    return _contains_any(line, TOEI_SUBWAY_LINES)


def is_jr_line(line: Any) -> bool:
    normalized = normalize_line_name(line)
    return normalized.startswith("jr") or _contains_any(line, JR_LINES)


def is_bus_leg(leg: dict[str, Any]) -> bool:
    mode = normalize_line_name(leg.get("type"))
    line = normalize_line_name(leg.get("line"))
    return "bus" in mode or "バス" in mode or "bus" in line or "バス" in line


def extract_operator_constraints(text: str) -> dict[str, list[str]]:
    """Legacy text extractor for offline migration; runtime consumes model slots."""
    allowed: list[str] = []
    avoided: list[str] = []
    avoid_modes: list[str] = []
    if re.search(r"(?:地下鉄|メトロ|東京メトロ|都営地下鉄)(?:だけ|のみ)(?:で|を)?", text):
        allowed.append("subway")
    if re.search(r"JR\s*(?:は|を)?(?:なし|使わない|避け)", text, re.IGNORECASE):
        avoided.append("JR")
    if re.search(r"バス\s*(?:は)?(?:なし|使わない|避け)", text):
        avoid_modes.append("bus")
    return {
        "allowed_operator_groups": allowed,
        "avoid_operator_groups": avoided,
        "avoid_modes": avoid_modes,
    }


def line_matches_constraint(line: Any, requested: str) -> bool:
    if requested.casefold() == "jr":
        return is_jr_line(line)
    return normalize_line_name(requested) in normalize_line_name(line)


_OPERATOR_GROUP_PREDICATES = {
    "subway": is_subway_line,
    "tokyo_metro": is_tokyo_metro_line,
    "toei_subway": is_toei_subway_line,
    "JR": is_jr_line,
}


def evaluate_operator_constraints(
    route: dict[str, Any],
    allowed_operator_groups: list[str] | None = None,
    avoid_operator_groups: list[str] | None = None,
    avoid_modes: list[str] | None = None,
) -> dict[str, Any]:
    legs = [leg for leg in route.get("legs", []) or [] if isinstance(leg, dict)]
    transit_legs = [
        leg
        for leg in legs
        if str(leg.get("type") or "").casefold() not in {"walk", "walking", "foot"}
        and leg.get("line")
    ]
    allowed = list(allowed_operator_groups or [])
    avoided = list(avoid_operator_groups or [])
    modes = list(avoid_modes or [])

    allowed_predicates = [
        _OPERATOR_GROUP_PREDICATES[group] for group in allowed if group in _OPERATOR_GROUP_PREDICATES
    ]
    allowed_satisfied = not allowed_predicates or (
        bool(transit_legs)
        and all(
            any(predicate(leg.get("line")) for predicate in allowed_predicates)
            for leg in transit_legs
        )
    )
    avoid_predicates = [
        _OPERATOR_GROUP_PREDICATES[group] for group in avoided if group in _OPERATOR_GROUP_PREDICATES
    ]
    avoid_satisfied = not avoid_predicates or not any(
        any(predicate(leg.get("line")) for predicate in avoid_predicates) for leg in transit_legs
    )
    bus_satisfied = "bus" not in modes or not any(is_bus_leg(leg) for leg in legs)
    return {
        "satisfied": allowed_satisfied and avoid_satisfied and bus_satisfied,
        "allowed_operator_groups_satisfied": allowed_satisfied,
        "avoid_operator_groups_satisfied": avoid_satisfied,
        "avoid_bus_satisfied": bus_satisfied,
        "allowed_operator_groups": allowed,
        "avoid_operator_groups": avoided,
        "avoid_modes": modes,
    }


__all__ = [
    "JR_LINES",
    "SUBWAY_LINES",
    "TOKYO_METRO_LINES",
    "TOEI_SUBWAY_LINES",
    "evaluate_operator_constraints",
    "extract_operator_constraints",
    "is_jr_line",
    "is_subway_line",
    "is_tokyo_metro_line",
    "is_toei_subway_line",
    "line_matches_constraint",
    "normalize_line_name",
]
