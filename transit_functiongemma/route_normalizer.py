#!/usr/bin/env python3
"""Normalize Transit MCP results into renderer-friendly, fact-only JSON."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _unwrap(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (MCP result, best structured/text payload)."""
    envelope = _dict(raw.get("envelope")) or raw
    result = _dict(envelope.get("result"))
    if not result and any(k in raw for k in ("content", "structuredContent", "_meta")):
        result = raw

    candidates = [
        raw.get("structuredContent"),
        result.get("structuredContent"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return result, candidate

    content = _list(result.get("content")) or _list(raw.get("content"))
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parsed = _json_object(item.get("text"))
            if parsed is not None:
                return result, parsed

    if result and not set(result).issubset({"content", "structuredContent", "_meta", "isError"}):
        return result, result
    return result, raw if not result else {}


def _name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    found = _first(value, "name", "label", "stationName", "stopName", "title")
    return str(found) if found is not None else None


def _station(value: Any) -> dict[str, str] | None:
    if isinstance(value, str):
        return {"name": value}
    if not isinstance(value, dict):
        return None
    item: dict[str, str] = {}
    station_id = _first(value, "id", "stationId", "stopId")
    name = _name(value)
    if station_id is not None:
        item["id"] = str(station_id)
    if name is not None:
        item["name"] = name
    return item or None


def _minutes(value: Any, *, seconds: bool = False) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if seconds:
        return int(math.ceil(number / 60.0))
    return int(number) if number.is_integer() else number


def _seconds_to_time(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        total = int(float(value))
    except (TypeError, ValueError):
        return None
    sign = "-" if total < 0 else ""
    total = abs(total)
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _clock(mapping: dict[str, Any], *string_keys: str, secs_keys: Iterable[str]) -> str | None:
    value = _first(mapping, *string_keys)
    if isinstance(value, str):
        if "T" in value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%H:%M")
            except ValueError:
                pass
        return value[:5] if len(value) >= 5 else value
    seconds = _first(mapping, *secs_keys)
    return _seconds_to_time(seconds)


def _fare(value: Any) -> int | float | None:
    if isinstance(value, dict):
        value = _first(value, "yen", "amount", "value", "price")
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _line_name(leg: dict[str, Any]) -> str | None:
    direct = _first(leg, "lineName", "routeName", "serviceName", "line", "route")
    if isinstance(direct, dict):
        return _name(direct)
    return str(direct) if direct is not None else None


def _normalize_leg(raw: dict[str, Any]) -> dict[str, Any]:
    departure_secs = _first(raw, "departureSecs", "departure_seconds")
    arrival_secs = _first(raw, "arrivalSecs", "arrival_seconds")
    duration = _minutes(_first(raw, "durationMin", "durationMinutes", "duration_min"))
    if duration is None:
        duration = _minutes(_first(raw, "durationSecs", "durationSeconds"), seconds=True)
    if duration is None and departure_secs is not None and arrival_secs is not None:
        try:
            duration = _minutes(float(arrival_secs) - float(departure_secs), seconds=True)
        except (TypeError, ValueError):
            pass

    leg: dict[str, Any] = {
        "type": _first(raw, "kind", "type", "mode"),
        "from": _name(_first(raw, "from", "origin")),
        "to": _name(_first(raw, "to", "destination")),
        "line": _line_name(raw),
        "departure_time": _clock(
            raw,
            "departureTime",
            "departure_time",
            "startTime",
            secs_keys=("departureSecs", "departure_seconds"),
        ),
        "arrival_time": _clock(
            raw,
            "arrivalTime",
            "arrival_time",
            "endTime",
            secs_keys=("arrivalSecs", "arrival_seconds"),
        ),
        "duration_min": duration,
        "distance_m": _first(raw, "distanceMeters", "distance_m", "distance"),
    }
    for endpoint in ("from", "to"):
        station = _station(_first(raw, endpoint, "origin" if endpoint == "from" else "destination"))
        if station and station.get("id"):
            leg[f"{endpoint}_id"] = station["id"]

    stops: list[dict[str, str]] = []
    for value in _list(_first(raw, "stops", "stopovers", "intermediateStops")):
        station = _station(value)
        if station:
            stops.append(station)
    if stops:
        leg["stops"] = stops
    return {key: value for key, value in leg.items() if value is not None}


def _priority_label(strategy: Any) -> str | None:
    return {
        "fastest": "早い候補",
        "fewestTransfers": "乗換が少ない候補",
        "lowestFare": "安い候補",
        "shortestWalk": "徒歩が短い候補",
        "balanced": "バランス候補",
        "fast": "早い候補",
    }.get(strategy)


def _normalize_route(
    raw: dict[str, Any],
    rank: int,
    priority: Any,
    origin: Any = None,
    destination: Any = None,
) -> dict[str, Any]:
    departure_secs = _first(raw, "departureSecs", "departure_seconds")
    arrival_secs = _first(raw, "arrivalSecs", "arrival_seconds")
    access_walk_secs = _first(raw, "accessWalkSecs", "access_walk_seconds") or 0
    egress_walk_secs = _first(raw, "egressWalkSecs", "egress_walk_seconds") or 0
    duration = _minutes(_first(raw, "durationMin", "durationMinutes", "duration_min"))
    if duration is None:
        duration_secs = _first(raw, "durationSecs", "durationSeconds")
        if duration_secs is not None:
            try:
                duration_secs = float(duration_secs) + float(access_walk_secs)
            except (TypeError, ValueError):
                pass
        duration = _minutes(duration_secs, seconds=True)
    if duration is None and departure_secs is not None and arrival_secs is not None:
        try:
            duration = _minutes(
                float(arrival_secs) - float(departure_secs) + float(access_walk_secs),
                seconds=True,
            )
        except (TypeError, ValueError):
            pass

    fare = _fare(_first(raw, "fareYen", "fare_yen", "totalFare", "fare", "price"))
    transfers = _first(raw, "transferCount", "transfers", "numberOfTransfers")
    if isinstance(transfers, list):
        transfers = len(transfers)
    summary_departure = _clock(
        raw,
        "departureTime",
        "departure_time",
        secs_keys=("departureSecs", "departure_seconds"),
    )
    if departure_secs is not None and access_walk_secs:
        try:
            summary_departure = _seconds_to_time(float(departure_secs) - float(access_walk_secs))
        except (TypeError, ValueError):
            pass
    summary = {
        "duration_min": duration,
        "fare_yen": fare,
        "transfers": transfers,
        "departure_time": summary_departure,
        "arrival_time": _clock(
            raw,
            "arrivalTime",
            "arrival_time",
            secs_keys=("arrivalSecs", "arrival_seconds"),
        ),
        "priority_label": _priority_label(priority),
    }
    raw_legs = [
        leg
        for leg in _list(_first(raw, "legs", "segments", "steps"))
        if isinstance(leg, dict)
    ]
    legs = [_normalize_leg(leg) for leg in raw_legs]
    if raw_legs and access_walk_secs:
        first_leg = raw_legs[0]
        first_departure = _first(first_leg, "departureSecs", "departure_seconds")
        origin_station = _station(origin)
        first_station = _station(_first(first_leg, "from", "origin"))
        try:
            access_departure = float(first_departure) - float(access_walk_secs)
        except (TypeError, ValueError):
            access_departure = None
        access_leg = {
            "type": "walk",
            "from": _name(origin),
            "to": _name(_first(first_leg, "from", "origin")),
            "departure_time": _seconds_to_time(access_departure),
            "arrival_time": _seconds_to_time(first_departure),
            "duration_min": _minutes(access_walk_secs, seconds=True),
            "from_id": origin_station.get("id") if origin_station else None,
            "to_id": first_station.get("id") if first_station else None,
        }
        legs.insert(0, {key: value for key, value in access_leg.items() if value is not None})
    if raw_legs and egress_walk_secs:
        last_leg = raw_legs[-1]
        last_arrival = _first(last_leg, "arrivalSecs", "arrival_seconds")
        last_station = _station(_first(last_leg, "to", "destination"))
        destination_station = _station(destination)
        egress_leg = {
            "type": "walk",
            "from": _name(_first(last_leg, "to", "destination")),
            "to": _name(destination),
            "departure_time": _seconds_to_time(last_arrival),
            "arrival_time": _seconds_to_time(arrival_secs),
            "duration_min": _minutes(egress_walk_secs, seconds=True),
            "from_id": last_station.get("id") if last_station else None,
            "to_id": destination_station.get("id") if destination_station else None,
        }
        legs.append({key: value for key, value in egress_leg.items() if value is not None})

    walk_legs = [
        leg
        for leg in legs
        if str(leg.get("type") or "").casefold() in {"walk", "walking", "foot"}
    ]
    if walk_legs:
        walk_durations = [
            float(leg["duration_min"])
            for leg in walk_legs
            if isinstance(leg.get("duration_min"), (int, float))
        ]
        walk_distances = [
            float(leg["distance_m"])
            for leg in walk_legs
            if isinstance(leg.get("distance_m"), (int, float))
        ]
        if walk_durations:
            summary["walk_duration_min"] = round(sum(walk_durations), 2)
        if walk_distances:
            summary["walk_distance_m"] = round(sum(walk_distances), 1)

    route: dict[str, Any] = {
        "rank": rank,
        "summary": {key: value for key, value in summary.items() if value is not None},
        "legs": legs,
    }
    for endpoint in ("from", "to"):
        station = _station(_first(raw, endpoint, "origin" if endpoint == "from" else "destination"))
        if station:
            route[endpoint] = station
    for key in ("stations", "stops", "transferStations", "stopovers"):
        stations = [_station(item) for item in _list(raw.get(key))]
        stations = [item for item in stations if item]
        if stations:
            route[key] = stations
    return route


def _query(
    payload: dict[str, Any], arguments: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    origin = _name(payload.get("from")) or arguments.get("fromLabel")
    destination = _name(payload.get("to")) or arguments.get("toLabel")
    date = _first(arguments, "date") or _first(payload, "date", "serviceDate")
    time = _first(arguments, "time")
    timezone = _first(payload, "timezone", "timeZone")
    dt: str | None = None
    if date:
        date_text = str(date)
        try:
            base = datetime.strptime(date_text, "%Y%m%d").strftime("%Y-%m-%d")
            if time:
                dt = f"{base}T{str(time)[:5]}:00"
                if timezone in ("Asia/Tokyo", "JST", "+09:00"):
                    dt += "+09:00"
            else:
                dt = base
        except ValueError:
            dt = date_text
    mode = _first(arguments, "type") or _first(payload, "type")
    time_mode = {
        "arrival": "arrive_by",
        "departure": "depart_at",
        "first": "first",
        "last": "last",
    }.get(mode, mode)
    strategy = arguments.get("strategy")
    priority = {
        "fastest": "fast",
        "fewestTransfers": "few_transfers",
        "lowestFare": "low_fare",
        "shortestWalk": "short_walk",
    }.get(strategy, strategy)
    query: dict[str, Any] = {
        "origin_text": origin,
        "destination_text": destination,
        "datetime": dt,
        "time_mode": time_mode,
        "priority": priority,
        "avoid_station_texts": [],
        "avoid_line_texts": [],
        "via_station_texts": list(arguments.get("viaLabel") or []),
        "preferred_line_texts": [],
        "graphical": False,
    }
    query.update({key: value for key, value in overrides.items() if value is not None})
    return query


def _suggestions(payload: dict[str, Any], tool_name: str | None) -> list[dict[str, Any]]:
    key = "stations" if tool_name == "suggest_stations" else "places"
    raw_items = _list(payload.get(key))
    if not raw_items:
        raw_items = _list(_first(payload, "suggestions", "results", "items"))
    output: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            if isinstance(raw, str):
                output.append({"name": raw})
            continue
        item = {
            "id": _first(raw, "id", "stationId", "stopId", "endpoint"),
            "endpoint": raw.get("endpoint"),
            "name": _name(raw),
            "kind": _first(raw, "kind", "type"),
            "description": raw.get("description"),
            "source_label": _first(raw, "feedName", "description", "source"),
            "source": raw.get("source"),
            "lat": raw.get("lat"),
            "lon": raw.get("lon"),
            "weight": raw.get("weight"),
        }
        output.append({k: v for k, v in item.items() if v is not None})
    return output


def _departures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in _list(_first(payload, "departures", "items", "results")):
        if not isinstance(raw, dict):
            continue
        item = {
            "time": _clock(
                raw,
                "departureTime",
                "departure_time",
                "time",
                secs_keys=("departureSecs", "departure_seconds"),
            ),
            "line": _line_name(raw),
            "direction": _first(raw, "direction", "headsign", "destination", "for"),
        }
        if isinstance(item["direction"], dict):
            item["direction"] = _name(item["direction"])
        output.append({k: v for k, v in item.items() if v is not None})
    return output


def normalize_mcp_result(
    raw: dict[str, Any],
    tool_name: str | None = None,
    tool_arguments: dict[str, Any] | None = None,
    query_overrides: dict[str, Any] | None = None,
    max_routes: int | None = None,
) -> dict[str, Any]:
    """Normalize raw/saved MCP JSON without inventing absent facts."""
    arguments = tool_arguments or {}
    result, payload = _unwrap(raw)
    inferred_tool = tool_name or raw.get("raw_tool_name")
    is_error = bool(result.get("isError")) or "error" in raw
    warnings = _list(_first(payload, "warnings", "messages"))
    normalized: dict[str, Any] = {
        "status": "error" if is_error else "ok",
        "query": _query(payload, arguments, query_overrides or {}),
        "routes": [],
        "warnings": warnings,
        "raw_tool_name": inferred_tool,
    }

    if inferred_tool in {"suggest_stations", "suggest_places", "reverse_geocode"}:
        normalized["suggestions"] = _suggestions(payload, inferred_tool)
        normalized["suggestion_type"] = (
            "station" if inferred_tool in {"suggest_stations", "reverse_geocode"} else "place"
        )
        if not normalized["suggestions"] and not is_error:
            normalized["status"] = "empty"
        return normalized

    if inferred_tool == "station_departures":
        normalized["station"] = _station(_first(payload, "station", "from"))
        normalized["departures"] = _departures(payload)
        if not normalized["departures"] and not is_error:
            normalized["status"] = "empty"
        return normalized

    if inferred_tool == "get_station":
        normalized["station"] = _station(_first(payload, "station", "result")) or _station(payload)
        normalized["station_detail"] = payload
        return normalized

    if inferred_tool == "list_feeds":
        normalized["feeds"] = _list(_first(payload, "feeds", "items", "results"))
        if not normalized["feeds"] and not is_error:
            normalized["status"] = "empty"
        return normalized

    raw_routes = _list(_first(payload, "journeys", "routes", "itineraries", "results"))
    if max_routes is not None:
        raw_routes = raw_routes[: max(0, max_routes)]
    strategy = arguments.get("strategy") or normalized["query"].get("priority")
    normalized["routes"] = [
        _normalize_route(route, rank, strategy, payload.get("from"), payload.get("to"))
        for rank, route in enumerate(raw_routes, 1)
        if isinstance(route, dict)
    ]
    if not normalized["routes"] and not is_error:
        normalized["status"] = "empty"
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a saved Transit MCP result.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tool-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-routes", type=int)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    normalized = normalize_mcp_result(raw, args.tool_name, max_routes=args.max_routes)
    text = json.dumps(normalized, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
