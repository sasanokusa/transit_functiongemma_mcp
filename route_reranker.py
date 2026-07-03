#!/usr/bin/env python3
"""Deterministically rerank normalized routes using user-requested priorities."""
from __future__ import annotations

import copy
import math
from typing import Any


def _number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return math.inf


def rerank_routes(normalized: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(normalized)
    routes = [route for route in output.get("routes", []) if isinstance(route, dict)]
    priority = (output.get("query") or {}).get("priority")
    ranking: dict[str, Any] | None = None
    if priority == "few_transfers":
        routes.sort(
            key=lambda route: (
                _number((route.get("summary") or {}).get("transfers")),
                _number((route.get("summary") or {}).get("duration_min")),
            )
        )
        ranking = {"priority": priority, "applied": True, "message": "乗換が少ない順に並べました。"}
    elif priority == "less_walk":
        routes.sort(
            key=lambda route: (
                _number((route.get("summary") or {}).get("walk_duration_min")),
                _number((route.get("summary") or {}).get("walk_distance_m")),
                _number((route.get("summary") or {}).get("transfers")),
                _number((route.get("summary") or {}).get("duration_min")),
            )
        )
        ranking = {"priority": priority, "applied": True, "message": "徒歩が少ない順に並べました。"}
    elif priority == "cheap":
        has_fare = any((route.get("summary") or {}).get("fare_yen") is not None for route in routes)
        if has_fare:
            routes.sort(
                key=lambda route: (
                    _number((route.get("summary") or {}).get("fare_yen")),
                    _number((route.get("summary") or {}).get("duration_min")),
                )
            )
            ranking = {"priority": priority, "applied": True, "message": "運賃が安い順に並べました。"}
        else:
            ranking = {
                "priority": priority,
                "applied": False,
                "message": "運賃情報が取得できないため、安さ順は保証できません。",
            }
    elif priority == "fast":
        routes.sort(key=lambda route: _number((route.get("summary") or {}).get("duration_min")))
        ranking = {"priority": priority, "applied": True, "message": "所要時間が短い順に並べました。"}
    for rank, route in enumerate(routes, 1):
        route.setdefault("source_rank", route.get("rank"))
        route["rank"] = rank
    output["routes"] = routes
    if ranking:
        output["ranking"] = ranking
    return output


__all__ = ["rerank_routes"]
