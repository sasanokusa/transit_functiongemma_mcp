#!/usr/bin/env python3
"""Resolve line-specific stop records into physical station locations."""
from __future__ import annotations

import math
from typing import Any

from transit_functiongemma.route_constraints import normalize_station_name


CLUSTER_RADIUS_METERS = 700.0
_NON_STATION_DESCRIPTIONS = {
    "artwork", "restaurant", "cafe", "bar", "shop", "hotel", "office", "parking"
}


def _is_station_candidate(item: dict[str, Any]) -> bool:
    description = str(item.get("description", "")).strip().casefold()
    if description in _NON_STATION_DESCRIPTIONS:
        return False
    if description.startswith("出入口") or "station entrance" in description:
        return False
    return True


def _is_physical_candidate(item: dict[str, Any]) -> bool:
    """Exclude feed-specific stop IDs from physical-station clustering.

    ``suggest_places`` may include both geo landmarks and line-specific stops.  A
    stop's coordinates are useful metadata, but treating it as another physical
    landmark can create a false ambiguity at large station complexes.
    """
    endpoint = item.get("endpoint")
    return isinstance(endpoint, str) and endpoint.startswith("geo:")


def _geo_endpoint(item: dict[str, Any]) -> str | None:
    endpoint = item.get("endpoint")
    if isinstance(endpoint, str) and endpoint.startswith("geo:"):
        return endpoint
    lat, lon = item.get("lat"), item.get("lon")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return f"geo:{lat:.6f},{lon:.6f}"
    return None


def _coordinates(item: dict[str, Any]) -> tuple[float, float] | None:
    lat, lon = item.get("lat"), item.get("lon")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return float(lat), float(lon)
    endpoint = _geo_endpoint(item)
    if endpoint:
        try:
            lat_text, lon_text = endpoint.removeprefix("geo:").split(",", 1)
            return float(lat_text), float(lon_text)
        except (ValueError, TypeError):
            return None
    return None


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(h))


def _candidate_score(item: dict[str, Any]) -> tuple[int, float]:
    item_id = str(item.get("id", ""))
    source = str(item.get("source", ""))
    preferred = 0
    if item_id.startswith("transit:query-landmark:"):
        preferred = 3
    elif "query-landmark:" in item_id:
        preferred = 2
    elif source == "transit":
        preferred = 1
    weight = item.get("weight")
    return preferred, float(weight) if isinstance(weight, (int, float)) else 0.0


def resolve_physical_station(
    query: str,
    suggestions: list[dict[str, Any]],
    cluster_radius_meters: float = CLUSTER_RADIUS_METERS,
    near: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an exact station name to one geo cluster, never to a line-specific ID."""
    wanted = normalize_station_name(query)
    usable = [
        item
        for item in suggestions
        if isinstance(item, dict)
        and normalize_station_name(item.get("name")) == wanted
        and str(item.get("kind", "")).casefold() == "station"
        and _is_physical_candidate(item)
        and _geo_endpoint(item)
        and _coordinates(item)
        and _is_station_candidate(item)
    ]
    if not usable:
        return {"status": "not_found", "query": query, "candidates": []}

    clusters: list[list[dict[str, Any]]] = []
    for item in usable:
        coordinates = _coordinates(item)
        assert coordinates is not None
        for cluster in clusters:
            anchor = _coordinates(cluster[0])
            if anchor and _distance_m(coordinates, anchor) <= cluster_radius_meters:
                cluster.append(item)
                break
        else:
            clusters.append([item])

    candidates: list[dict[str, Any]] = []
    for cluster in clusters:
        selected = max(cluster, key=_candidate_score)
        coordinates = _coordinates(selected)
        endpoint = _geo_endpoint(selected)
        cluster_coordinates = [value for item in cluster if (value := _coordinates(item))]
        span = max(
            (_distance_m(left, right) for left in cluster_coordinates for right in cluster_coordinates),
            default=0.0,
        )
        candidate: dict[str, Any] = {
            "name": selected.get("name") or query,
            "endpoint": endpoint,
            "lat": coordinates[0] if coordinates else None,
            "lon": coordinates[1] if coordinates else None,
            "description": selected.get("description"),
            "member_count": len(cluster),
            "source": selected.get("source"),
            "cluster_span_meters": round(span),
        }
        candidates.append({k: v for k, v in candidate.items() if v is not None})

    if len(candidates) == 1:
        resolution = "station_cluster" if candidates[0].get("cluster_span_meters", 0) > 300 else "exact"
        return {
            "status": "resolved",
            "query": query,
            "station": candidates[0],
            "candidates": candidates,
            "resolution": resolution,
        }
    near_coordinates = _coordinates(near or {})
    if near_coordinates:
        ranked = sorted(
            (
                (_distance_m(near_coordinates, coordinates), candidate)
                for candidate in candidates
                if (coordinates := _coordinates(candidate)) is not None
            ),
            key=lambda item: item[0],
        )
        if len(ranked) >= 2:
            nearest_distance, nearest = ranked[0]
            next_distance = ranked[1][0]
            if nearest_distance <= 100_000 and next_distance - nearest_distance >= 50_000:
                selected = dict(nearest)
                selected["context_distance_meters"] = round(nearest_distance)
                return {
                    "status": "resolved",
                    "query": query,
                    "station": selected,
                    "candidates": candidates,
                    "resolution": "nearest_to_route_context",
                }
    return {"status": "ambiguous", "query": query, "candidates": candidates}


def station_query_text(query: str) -> str:
    stripped = query.strip()
    return stripped if stripped.endswith("駅") else stripped + "駅"


__all__ = ["resolve_physical_station", "station_query_text"]
