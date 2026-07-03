#!/usr/bin/env python3
"""Local-only pseudo tools. Nothing in this module performs network access."""
from __future__ import annotations

import re
from typing import Any

from transit_functiongemma.route_intent import extract_route_intent
from line_operator_rules import extract_operator_constraints


ASK_CLARIFICATION = "ask_clarification"
RESOLVE_ROUTE_REQUEST = "resolve_route_request"

ASK_CLARIFICATION_TOOL: dict[str, Any] = {
    "name": ASK_CLARIFICATION,
    "description": "Ask the user for missing transit-search information. Local only.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "missing": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["origin", "destination", "date", "time", "station_id"],
                },
                "minItems": 1,
            },
            "question": {"type": "string", "minLength": 1},
        },
        "required": ["missing", "question"],
        "additionalProperties": False,
    },
    "_localOnly": True,
}

RESOLVE_ROUTE_REQUEST_TOOL: dict[str, Any] = {
    "name": RESOLVE_ROUTE_REQUEST,
    "description": "Normalize explicit route-request constraints. Local only; reserved for extension.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "origin_text": {"type": "string"},
            "destination_text": {"type": "string"},
            "avoid_station_texts": {"type": "array", "items": {"type": "string"}},
            "via_station_texts": {"type": "array", "items": {"type": "string"}},
            "avoid_line_texts": {"type": "array", "items": {"type": "string"}},
            "preferred_line_texts": {"type": "array", "items": {"type": "string"}},
            "allowed_operator_groups": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["subway", "tokyo_metro", "toei_subway"],
                },
            },
            "avoid_operator_groups": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["JR"],
                },
            },
            "avoid_modes": {
                "type": "array",
                "items": {"type": "string", "enum": ["bus"]},
            },
            "graphical": {"type": "boolean"},
            "priority": {
                "type": ["string", "null"],
                "enum": ["fast", "cheap", "few_transfers", "less_walk", None],
            },
            "time_mode": {
                "type": ["string", "null"],
                "enum": ["departure_at", "arrive_by", "first_train", "last_train", None],
            },
            "date": {"type": ["string", "null"], "pattern": "^\\d{8}$"},
            "time": {"type": ["string", "null"], "pattern": "^\\d{2}:\\d{2}$"},
        },
        "required": ["origin_text", "destination_text"],
        "additionalProperties": False,
    },
    "_localOnly": True,
}

LOCAL_TOOLS = [ASK_CLARIFICATION_TOOL, RESOLVE_ROUTE_REQUEST_TOOL]


def is_local_tool(name: str) -> bool:
    return name in {ASK_CLARIFICATION, RESOLVE_ROUTE_REQUEST}


def execute_local_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == ASK_CLARIFICATION:
        missing = arguments.get("missing") or []
        question = arguments.get("question")
        if not isinstance(missing, list) or not missing or not isinstance(question, str) or not question:
            raise ValueError("invalid local ask_clarification arguments")
        return {"status": "clarification", "missing": missing, "question": question}
    if name == RESOLVE_ROUTE_REQUEST:
        allowed = set(RESOLVE_ROUTE_REQUEST_TOOL["inputSchema"]["properties"])
        return {
            "status": "resolved_request",
            "query": {key: value for key, value in arguments.items() if key in allowed},
        }
    raise ValueError(f"unknown local tool: {name}")


def _near_keyword(text: str, keyword: str) -> list[str]:
    patterns = [
        rf"(?:^|[、,。\s])([^、,。\s]+?)(?:駅)?を{keyword}",
        rf"(?:^|[、,。\s])([^、,。\s]+?)(?:駅)?{keyword}",
    ]
    values: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1).strip()
            if value.endswith("を"):
                value = value[:-1]
            if value and value not in values:
                values.append(value)
    return values


def extract_route_hints(user_text: str) -> dict[str, Any]:
    """Legacy semantic fallback; never call this in the default runtime path.

    FunctionGemma owns route-slot extraction. This helper remains only for
    explicitly opted-in compatibility evaluation and offline dataset migration.
    """
    intent = extract_route_intent(user_text)
    preferred_lines: list[str] = []
    for match in re.finditer(r"([^\s、,。]+線)(?=で|を|経由|利用|使)", user_text):
        value = match.group(1)
        for separator in ("から", "まで", "なら", "は"):
            value = value.split(separator)[-1]
        if value and value not in preferred_lines:
            preferred_lines.append(value)
    return {
        "origin_text": intent["origin_text"],
        "destination_text": intent["destination_text"],
        "avoid_station_texts": intent["avoid_station_texts"],
        "via_station_texts": intent["via_station_texts"],
        "avoid_line_texts": intent["avoid_line_texts"],
        "preferred_line_texts": preferred_lines,
        "graphical": intent["graphical"],
        "priority": intent["priority"],
        "time_mode": intent["time_mode"],
        "date": intent["date"],
        "time": intent["time"],
        **extract_operator_constraints(user_text),
    }


def infer_missing_fields(user_text: str) -> list[str]:
    """Provide a conservative confirmation for an empty router turn."""
    has_origin = "から" in user_text
    has_destination = bool(
        re.search(r"(?:まで|へ(?:行|向|乗|$)|[^、。\s]+に(?:着き|到着))", user_text)
    )
    missing: list[str] = []
    if not has_origin:
        missing.append("origin")
    if not has_destination:
        missing.append("destination")
    return missing


__all__ = [
    "ASK_CLARIFICATION",
    "ASK_CLARIFICATION_TOOL",
    "LOCAL_TOOLS",
    "RESOLVE_ROUTE_REQUEST",
    "RESOLVE_ROUTE_REQUEST_TOOL",
    "execute_local_tool",
    "extract_route_hints",
    "infer_missing_fields",
    "is_local_tool",
]
