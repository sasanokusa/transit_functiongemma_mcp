from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "google/functiongemma-270m-it"
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "https://api.transit.ls8h.com/mcp")
DEVELOPER_PROMPT = (
    "You are a model that can do function calling with the following functions. "
    "Return only a function call. Never invent stations, routes, fares, durations, "
    "lines, station IDs, or coordinates. For a complete natural-language route "
    "request, call resolve_route_request and extract only what the user stated. "
    "The deterministic planner will resolve stations and query transit facts. "
    "If required user information is missing, "
    "return no function call. Current local datetime: {now}."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "data" / "tool_schema.json"
