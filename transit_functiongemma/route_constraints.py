#!/usr/bin/env python3
"""Deterministic station avoid/via checks for normalized transit routes."""
from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any, Iterable

from transit_functiongemma.line_operator_rules import evaluate_operator_constraints, line_matches_constraint


_SPACE_RE = re.compile(r"\s+")
_STATION_COLLECTION_KEYS = {
    "stations",
    "stops",
    "stopovers",
    "intermediatestops",
    "passedstations",
    "transferstations",
    "transfers",
}
_ENDPOINT_KEYS = {"from", "to", "origin", "destination", "station", "stop"}
_CLOCK_RE = re.compile(r"^(?P<sign>-?)(?P<hour>\d{1,3}):(?P<minute>\d{2})")


def normalize_station_name(value: object) -> str:
    """Apply conservative comparison-only normalization without fuzzy matching."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = _SPACE_RE.sub("", normalized).casefold()
    if normalized.endswith("駅"):
        normalized = normalized[:-1]
    return normalized


def _normalize_line_name(value: object) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub("", unicodedata.normalize("NFKC", str(value))).casefold()


def _station_ref(value: Any) -> tuple[str | None, str | None] | None:
    if isinstance(value, str):
        return None, value
    if not isinstance(value, dict):
        return None
    station_id = value.get("id") or value.get("station_id") or value.get("stationId")
    name = (
        value.get("name")
        or value.get("label")
        or value.get("station_name")
        or value.get("stationName")
    )
    if station_id is None and name is None:
        return None
    return (
        str(station_id) if station_id is not None else None,
        str(name) if name is not None else None,
    )


def extract_route_stations(route: dict[str, Any]) -> list[dict[str, str]]:
    """Extract station references from route endpoints, legs, transfers, and stops."""
    refs: list[tuple[str | None, str | None]] = []

    def add(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                add(item)
            return
        ref = _station_ref(value)
        if ref:
            refs.append(ref)

    for key, value in route.items():
        lowered = key.replace("_", "").casefold()
        if lowered in _STATION_COLLECTION_KEYS or key.casefold() in _ENDPOINT_KEYS:
            add(value)

    for leg in route.get("legs", []) or []:
        if not isinstance(leg, dict):
            continue
        for endpoint in ("from", "to", "origin", "destination"):
            endpoint_id = leg.get(f"{endpoint}_id") or leg.get(f"{endpoint}Id")
            endpoint_name = leg.get(endpoint)
            if endpoint_id is not None or endpoint_name is not None:
                add({"id": endpoint_id, "name": endpoint_name})
        for key, value in leg.items():
            lowered = key.replace("_", "").casefold()
            if lowered in _STATION_COLLECTION_KEYS or key.casefold() in _ENDPOINT_KEYS:
                add(value)

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for station_id, name in refs:
        key = (station_id or "", normalize_station_name(name))
        if key in seen or not any(key):
            continue
        seen.add(key)
        item: dict[str, str] = {}
        if station_id:
            item["id"] = station_id
        if name:
            item["name"] = name
        unique.append(item)
    return unique


def _matches(
    station: dict[str, str], constraint_text: str | None, constraint_id: str | None
) -> bool:
    if constraint_id and station.get("id") == constraint_id:
        return True
    wanted = normalize_station_name(constraint_text)
    return bool(wanted and normalize_station_name(station.get("name")) == wanted)


def _as_strings(values: Iterable[Any] | None) -> list[str]:
    return [str(value) for value in (values or []) if value is not None and str(value)]


def _clock_minutes(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = _CLOCK_RE.match(value.strip())
    if not match:
        return None
    minute = int(match.group("minute"))
    if minute > 59:
        return None
    total = int(match.group("hour")) * 60 + minute
    return -total if match.group("sign") else total


def _time_constraint(
    route: dict[str, Any], time_mode: str | None, requested_time: str | None
) -> dict[str, Any]:
    requested_minutes = _clock_minutes(requested_time)
    summary = route.get("summary") or {}
    if time_mode == "arrive_by" and requested_minutes is not None:
        actual = summary.get("arrival_time")
        actual_minutes = _clock_minutes(actual)
        return {
            "time_satisfied": actual_minutes is not None
            and actual_minutes <= requested_minutes,
            "requested_time": requested_time,
            "actual_time": actual,
            "time_mode": time_mode,
        }
    if time_mode in {"departure_at", "depart_at"} and requested_minutes is not None:
        actual = summary.get("departure_time")
        actual_minutes = _clock_minutes(actual)
        return {
            "time_satisfied": actual_minutes is not None
            and actual_minutes >= requested_minutes,
            "requested_time": requested_time,
            "actual_time": actual,
            "time_mode": time_mode,
        }
    return {"time_satisfied": True}


def evaluate_route_constraints(
    route: dict[str, Any],
    avoid_station_texts: Iterable[Any] | None = None,
    via_station_texts: Iterable[Any] | None = None,
    avoid_station_ids: Iterable[Any] | None = None,
    via_station_ids: Iterable[Any] | None = None,
    preferred_line_texts: Iterable[Any] | None = None,
    avoid_line_texts: Iterable[Any] | None = None,
    time_mode: str | None = None,
    requested_time: str | None = None,
) -> dict[str, Any]:
    """Evaluate exact normalized-name/ID constraints against one route."""
    stations = extract_route_stations(route)
    avoid_texts = _as_strings(avoid_station_texts)
    via_texts = _as_strings(via_station_texts)
    avoid_ids = _as_strings(avoid_station_ids)
    via_ids = _as_strings(via_station_ids)
    preferred_lines = _as_strings(preferred_line_texts)
    avoid_lines = _as_strings(avoid_line_texts)

    violated_texts = [
        text for text in avoid_texts if any(_matches(s, text, None) for s in stations)
    ]
    violated_ids = [
        station_id
        for station_id in avoid_ids
        if any(_matches(s, None, station_id) for s in stations)
    ]
    missing_texts = [
        text for text in via_texts if not any(_matches(s, text, None) for s in stations)
    ]
    missing_ids = [
        station_id
        for station_id in via_ids
        if not any(_matches(s, None, station_id) for s in stations)
    ]
    route_lines = [
        str(leg.get("line"))
        for leg in route.get("legs", []) or []
        if isinstance(leg, dict) and leg.get("line")
    ]
    missing_lines = [
        requested
        for requested in preferred_lines
        if not any(_normalize_line_name(requested) in _normalize_line_name(line) for line in route_lines)
    ]
    violated_avoid_lines = [
        requested
        for requested in avoid_lines
        if any(line_matches_constraint(line, requested) for line in route_lines)
    ]
    return {
        "avoid_satisfied": not violated_texts and not violated_ids,
        "avoided_station_texts": avoid_texts,
        "violated_avoid_station_texts": violated_texts,
        "via_satisfied": not missing_texts and not missing_ids,
        "missing_via_station_texts": missing_texts,
        "line_satisfied": not missing_lines,
        "preferred_line_texts": preferred_lines,
        "missing_preferred_line_texts": missing_lines,
        "avoid_line_satisfied": not violated_avoid_lines,
        "avoided_line_texts": avoid_lines,
        "violated_avoid_line_texts": violated_avoid_lines,
        **_time_constraint(route, time_mode, requested_time),
        **({"avoided_station_ids": avoid_ids} if avoid_ids else {}),
        **({"violated_avoid_station_ids": violated_ids} if avoid_ids else {}),
        **({"via_station_ids": via_ids} if via_ids else {}),
        **({"missing_via_station_ids": missing_ids} if via_ids else {}),
    }


def apply_route_constraints(normalized: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with a constraint_check attached to every normalized route."""
    output = copy.deepcopy(normalized)
    query = output.get("query") or {}
    for route in output.get("routes", []) or []:
        if isinstance(route, dict):
            route["constraint_check"] = evaluate_route_constraints(
                route,
                query.get("avoid_station_texts"),
                query.get("via_station_texts"),
                query.get("avoid_station_ids"),
                query.get("via_station_ids"),
                query.get("preferred_line_texts"),
                query.get("avoid_line_texts"),
                query.get("time_mode"),
                query.get("time"),
            )
            route["operator_check"] = evaluate_operator_constraints(
                route,
                query.get("allowed_operator_groups"),
                query.get("avoid_operator_groups"),
                query.get("avoid_modes"),
            )
    return output


__all__ = [
    "apply_route_constraints",
    "evaluate_route_constraints",
    "extract_route_stations",
    "normalize_station_name",
]
