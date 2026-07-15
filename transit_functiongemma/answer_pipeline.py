#!/usr/bin/env python3
"""FunctionGemma router -> Transit MCP -> deterministic Japanese renderer."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from transit_functiongemma.local_tools import (
    ASK_CLARIFICATION,
    RESOLVE_ROUTE_REQUEST,
    execute_local_tool,
    infer_missing_fields,
    is_local_tool,
)
from transit_functiongemma.route_constraints import apply_route_constraints, normalize_station_name
from transit_functiongemma.route_normalizer import normalize_mcp_result
from transit_functiongemma.route_renderer import render_answer, render_clarification
from transit_functiongemma.route_reranker import rerank_routes
from transit_functiongemma.station_resolver import resolve_physical_station, station_query_text
from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, MCP_ENDPOINT, MODEL_ID
from transit_functiongemma.japanese import repair_tool_call_values
from transit_functiongemma.mcp import MCPClient, save_mcp_artifact
from transit_functiongemma.schemas import load_mcp_tools
from transit_functiongemma.toolcall import ToolCall, ToolCallParseError, parse_tool_calls
from transit_functiongemma.validation import (
    ToolCallSchemaError,
    validate_tool_call,
    validate_tool_name,
)
from transit_functiongemma.schemas import tools_with_clarification


USER_ERROR = "乗換案内を処理できませんでした。入力内容を確認してもう一度お試しください。"


class StationSelectionRequired(RuntimeError):
    def __init__(self, query: str, candidates: list[dict[str, Any]]) -> None:
        super().__init__(f"station selection required: {query}")
        self.query = query
        self.candidates = candidates


class SuggestionSelectionRequired(RuntimeError):
    def __init__(
        self, tool_name: str, query: str, candidates: list[dict[str, Any]]
    ) -> None:
        super().__init__(f"suggestion selection required: {query}")
        self.tool_name = tool_name
        self.query = query
        self.candidates = candidates


_NON_CANDIDATE_DESCRIPTIONS = {
    "artwork", "restaurant", "cafe", "bar", "shop", "hotel", "office", "parking"
}


def _display_suggestions(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer exact names and remove clearly misclassified POIs from UI choices."""
    usable = [
        candidate
        for candidate in candidates
        if str(candidate.get("description", "")).strip().casefold()
        not in _NON_CANDIDATE_DESCRIPTIONS
    ]
    wanted = normalize_station_name(query)
    exact = [
        candidate
        for candidate in usable
        if normalize_station_name(candidate.get("name")) == wanted
    ]
    return exact or usable


def handle_no_call(user_text: str) -> str:
    """Render a conservative generic clarification without semantic parsing."""
    return render_clarification(["origin", "destination"])


def handle_local_call(name: str, arguments: dict[str, Any]) -> str:
    result = execute_local_tool(name, arguments)
    if result["status"] == "clarification":
        return render_answer(result)
    if result["status"] == "resolved_request":
        return json.dumps(result["query"], ensure_ascii=False, indent=2)
    return USER_ERROR


def _refine_clarification_call(call: ToolCall, user_text: str) -> ToolCall:
    """Remove only demonstrably present fields from a model clarification."""
    if call.name != ASK_CLARIFICATION:
        return call
    model_missing = call.arguments.get("missing")
    if not isinstance(model_missing, list) or not model_missing:
        return call
    runtime_missing = infer_missing_fields(user_text)
    model_set = set(model_missing)
    runtime_set = set(runtime_missing)
    if not runtime_set or not runtime_set < model_set:
        return call
    arguments = dict(call.arguments)
    arguments["missing"] = runtime_missing
    arguments["question"] = render_clarification(runtime_missing)
    return ToolCall(call.name, arguments)


def _save_normalized(data: dict[str, Any], destination: Path, tool_name: str) -> Path:
    if destination.suffix.lower() == ".json":
        path = destination
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        path = destination / f"{timestamp}_{tool_name}_normalized.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _resolution_message(resolution: dict[str, Any]) -> str:
    query = resolution.get("query", "")
    candidates = resolution.get("candidates") or []
    if resolution.get("status") == "not_found":
        return f"「{query}」に完全一致する物理駅を確認できませんでした。駅名を確認してください。"
    lines = [f"「{query}」に一致する物理駅が複数あります。", ""]
    for index, item in enumerate(candidates, 1):
        detail = item.get("description")
        coordinates = (
            f"{item['lat']:.5f}, {item['lon']:.5f}"
            if isinstance(item.get("lat"), (int, float)) and isinstance(item.get("lon"), (int, float))
            else None
        )
        suffix = detail or coordinates
        lines.append(f"{index}. {item.get('name', query)}" + (f"（{suffix}）" if suffix else ""))
    lines.extend(["", "どの駅を使いますか？"])
    return "\n".join(lines)


def _history_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def _history_result(name: str, response: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "content": {"name": name, "response": response}}


def _route_arguments(
    tool_name: str,
    model_arguments: dict[str, Any],
    resolved_stations: list[dict[str, Any]],
    route_intent: dict[str, Any] | None = None,
    num_itineraries: int | None = None,
) -> dict[str, Any]:
    """Build planner arguments from resolved state and the preserved route intent.

    When ``route_intent`` is provided it is authoritative for every semantic
    planner argument.  In particular, values from ``model_arguments`` are not
    allowed to leak into the final call: the small model extracts intent once,
    while endpoint binding and schema-level enum conversion belong here.
    """
    arguments = dict(model_arguments)
    if tool_name not in {"plan_journey", "plan_route_map"}:
        return arguments
    intent_owned = route_intent is not None
    if intent_owned:
        for key in (
            "from",
            "to",
            "fromLabel",
            "toLabel",
            "via",
            "viaLabel",
            "date",
            "time",
            "type",
            "strategy",
            "avoidModes",
            "numItineraries",
        ):
            arguments.pop(key, None)
    if len(resolved_stations) >= 2:
        origin, destination = resolved_stations[0], resolved_stations[-1]
        arguments.update(
            {
                "from": origin["endpoint"],
                "to": destination["endpoint"],
                "fromLabel": origin["name"],
                "toLabel": destination["name"],
            }
        )
        if len(resolved_stations) > 2:
            arguments["via"] = [station["endpoint"] for station in resolved_stations[1:-1]]
            arguments["viaLabel"] = [station["name"] for station in resolved_stations[1:-1]]
    intent = route_intent or {}
    if intent_owned:
        for key in ("date", "time"):
            if intent.get(key):
                arguments[key] = intent[key]
    explicit_type = {
        "last_train": "last",
        "first_train": "first",
        "arrive_by": "arrival",
        "departure_at": "departure",
        "depart_at": "departure",
    }.get(intent.get("time_mode"))
    if explicit_type:
        arguments["type"] = explicit_type
    else:
        arguments.pop("type", None)
    if isinstance(num_itineraries, int) and 1 <= num_itineraries <= 6:
        arguments["numItineraries"] = num_itineraries
    if intent_owned and intent.get("avoid_modes"):
        arguments["avoidModes"] = ",".join(intent["avoid_modes"])
    if tool_name == "plan_route_map":
        if intent_owned:
            arguments["strategy"] = {
                "fast": "fastest",
                "cheap": "lowestFare",
                "few_transfers": "fewestTransfers",
                "less_walk": "shortestWalk",
            }.get(intent.get("priority"), "balanced")
        elif arguments.get("strategy") not in {
            None,
            "balanced",
            "fastest",
            "fewestTransfers",
            "lowestFare",
            "shortestWalk",
        }:
            arguments.pop("strategy", None)
    return arguments


def _route_candidate_limits(
    route_intent: dict[str, Any], max_routes: int
) -> tuple[int, int]:
    constrained = any(
        route_intent.get(key)
        for key in (
            "avoid_station_texts",
            "via_station_texts",
            "avoid_line_texts",
            "allowed_operator_groups",
            "avoid_operator_groups",
            "avoid_modes",
            "priority",
        )
    )
    return (6, 3) if constrained else (1, max_routes)


def _planned_call_from_model_intent(
    route_intent: dict[str, Any],
    resolved_stations: list[dict[str, Any]],
    *,
    num_itineraries: int | None = None,
) -> ToolCall:
    """Turn a schema-valid model intent into deterministic planner calls."""
    via = [str(value) for value in route_intent.get("via_station_texts") or []]
    queries = [
        str(route_intent["origin_text"]),
        *via,
        str(route_intent["destination_text"]),
    ]
    if len(resolved_stations) < len(queries):
        return ToolCall(
            "suggest_stations",
            {"q": queries[len(resolved_stations)], "limit": 5},
        )

    tool_name = "plan_route_map" if route_intent.get("graphical") else "plan_journey"
    arguments = _route_arguments(
        tool_name,
        {},
        resolved_stations,
        route_intent,
        num_itineraries=num_itineraries,
    )
    return ToolCall(tool_name, arguments)


def _route_intent_with_defaults(
    route_intent: dict[str, Any], *, default_graphical: bool
) -> dict[str, Any]:
    """Apply presentation defaults without changing the model-owned intent record."""
    effective = dict(route_intent)
    if default_graphical:
        effective["graphical"] = True
    return effective


def run_pipeline(
    user_text: str,
    *,
    adapter: str | None = None,
    mcp_url: str = MCP_ENDPOINT,
    schema_mode: str = "baked",
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    save_raw: str | Path | None = Path("artifacts/mcp"),
    save_normalized: str | Path | None = None,
    max_routes: int = 1,
    max_tool_steps: int = 7,
    debug: bool = False,
    clarification_tool: bool = False,
    router_instance: Any | None = None,
    station_overrides: dict[str, dict[str, Any]] | None = None,
    interactive: bool = False,
    normalize_ja: bool = False,
    trace: dict[str, Any] | None = None,
    ui_payload: dict[str, Any] | None = None,
    default_graphical: bool = False,
) -> str:
    pipeline_started = time.monotonic()

    def add_timing(name: str, elapsed_ms: float) -> None:
        if trace is None:
            return
        timings = trace.setdefault("timings", {})
        timings[name] = round(float(timings.get(name, 0)) + elapsed_ms, 2)

    def finish_timing() -> None:
        if trace is not None:
            trace.setdefault("timings", {})["total_latency_ms"] = round(
                (time.monotonic() - pipeline_started) * 1000, 2
            )

    if router_instance is None:
        # Keep torch/Transformers imports out of offline and injected-router use.
        from transit_functiongemma.infer import ToolRouter

        router = ToolRouter(
            base_model=MODEL_ID,
            adapter=adapter,
            schema_path=schema_path,
            schema_mode=schema_mode,
            clarification_tool=clarification_tool,
            normalize_ja=normalize_ja,
        )
    else:
        router = router_instance
    station_overrides = station_overrides or {}
    if trace is not None:
        trace.update(
            {
                "request_id": trace.get("request_id") or uuid.uuid4().hex,
                "user_input": user_text,
                "router_outputs": [],
                "planner_steps": [],
                "tool_calls": [],
                "schema_validations": [],
                "mcp_calls": [],
                "timings": {},
            }
        )
    tools = load_mcp_tools(schema_path)
    route_request = False
    route_hints: dict[str, Any] = {}
    candidate_count, display_limit = 1, max_routes
    history: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    seen_calls: set[str] = set()
    resolved_stations: list[dict[str, Any]] = []
    resolution_notes: list[str] = []
    with MCPClient(mcp_url) as client:
        for step in range(max_tool_steps):
            calls = None
            if route_request:
                planned_call = _planned_call_from_model_intent(
                    route_hints,
                    resolved_stations,
                    num_itineraries=candidate_count,
                )
                calls = [planned_call]
                if trace is not None:
                    trace["planner_steps"].append(
                        {
                            "step": step + 1,
                            "route_stage": len(resolved_stations),
                            "call": planned_call.as_dict(),
                        }
                    )
                if debug:
                    print(
                        f"[deterministic planner {step + 1}] {planned_call.as_dict()}",
                        file=sys.stderr,
                    )
            if calls is None:
                model_started = time.monotonic()
                raw_output = router.generate(
                    user_text if step == 0 else None,
                    history=None if step == 0 else history,
                )
                add_timing("model_latency_ms", (time.monotonic() - model_started) * 1000)
                if trace is not None:
                    trace["router_outputs"].append(raw_output)
                if debug:
                    print(f"[router output {step + 1}] {raw_output}", file=sys.stderr)
                try:
                    calls = parse_tool_calls(raw_output)
                except ToolCallParseError as exc:
                    raise RuntimeError(f"tool callを解析できません: {exc}") from exc
                # Value-fidelity repair only: exact coordinates/IDs written by the
                # user and relative-date arithmetic are the runtime's job, not the
                # 270M model's. Tool choice and semantic slots stay model-owned.
                reference_now = datetime.now(ZoneInfo("Asia/Tokyo")).strftime(
                    "%Y-%m-%d %H:%M Asia/Tokyo"
                )
                calls = [
                    repair_tool_call_values(parsed_call, user_text, reference_now)
                    for parsed_call in calls
                ]
            try:
                for parsed_call in calls:
                    validate_tool_name(
                        parsed_call.name,
                        tools_with_clarification(tools, clarification_tool),
                    )
                    if trace is not None:
                        trace["schema_validations"].append(
                            {"tool": parsed_call.name, "valid": True, "phase": "name"}
                        )
            except ToolCallSchemaError as exc:
                raise RuntimeError(f"tool callを解析できません: {exc}") from exc
            if not calls:
                if debug:
                    print("[local] no call; MCP was not contacted", file=sys.stderr)
                answer = handle_no_call(user_text)
                if trace is not None:
                    trace["rendered_answer"] = answer
                    trace["no_call"] = True
                finish_timing()
                return answer
            if len(calls) != 1:
                raise RuntimeError(f"tool callは1件である必要があります: {len(calls)}件")

            call = calls[0]
            call = _refine_clarification_call(call, user_text)
            if trace is not None:
                trace["tool_calls"].append(call.as_dict())
            if is_local_tool(call.name):
                validate_tool_call(
                    call, tools_with_clarification(tools, clarification_tool)
                )
                if call.name == RESOLVE_ROUTE_REQUEST:
                    result = execute_local_tool(call.name, call.arguments)
                    model_route_intent = dict(result["query"])
                    route_hints = _route_intent_with_defaults(
                        model_route_intent,
                        default_graphical=default_graphical,
                    )
                    route_request = True
                    candidate_count, display_limit = _route_candidate_limits(
                        route_hints, max_routes
                    )
                    if trace is not None:
                        trace["model_route_intent"] = model_route_intent
                        if route_hints != model_route_intent:
                            trace["effective_route_intent"] = route_hints
                            trace["graphical_defaulted"] = True
                    if debug:
                        print(
                            f"[model route intent] {json.dumps(route_hints, ensure_ascii=False)}",
                            file=sys.stderr,
                        )
                    continue
                if debug:
                    print(f"[local] {call.name}; MCP was not contacted", file=sys.stderr)
                answer = handle_local_call(call.name, call.arguments)
                if trace is not None:
                    trace["rendered_answer"] = answer
                    if call.name == ASK_CLARIFICATION:
                        trace["no_call"] = True
                finish_timing()
                return answer
            # Repetition is suspicious only for model-generated MCP calls. The
            # deterministic route planner may legitimately resolve the same
            # station at different stages (same origin/destination or a repeated
            # via), and route_stage itself guarantees forward progress.
            if not route_request:
                signature = json.dumps(call.as_dict(), ensure_ascii=False, sort_keys=True)
                if signature in seen_calls:
                    raise RuntimeError("同じtool callが繰り返されたため停止しました")
                seen_calls.add(signature)

            # During route planning, resolve a named station to a physical geo cluster.
            # This avoids choosing a line/feed before the journey planner runs.
            if route_request and call.name in {"suggest_stations", "suggest_places"}:
                station_resolve_started = time.monotonic()
                validate_tool_call(call, tools)
                query = call.arguments.get("q")
                if not isinstance(query, str) or not query.strip():
                    raise RuntimeError("駅候補検索に駅名がありません")
                place_query = station_query_text(query) if call.name == "suggest_stations" else query
                place_arguments = {"q": place_query, "limit": 30}
                mcp_started = time.monotonic()
                envelope = client.call_tool("suggest_places", place_arguments, tools=tools)
                mcp_elapsed = round((time.monotonic() - mcp_started) * 1000, 2)
                raw_path = (
                    save_mcp_artifact(envelope, save_raw, "resolve_physical_station")
                    if save_raw is not None
                    else None
                )
                place_data = normalize_mcp_result(envelope, "suggest_places", place_arguments)
                if trace is not None:
                    trace["mcp_calls"].append(
                        {
                            "tool": "suggest_places",
                            "arguments": place_arguments,
                            "status": "ok",
                            "latency_ms": mcp_elapsed,
                            "attempts": client.last_attempts,
                            "raw_path": str(raw_path) if raw_path else None,
                        }
                    )
                resolution = resolve_physical_station(
                    query,
                    place_data.get("suggestions") or [],
                    near=resolved_stations[-1] if resolved_stations else None,
                )
                add_timing(
                    "station_resolve_latency_ms",
                    (time.monotonic() - station_resolve_started) * 1000,
                )
                override = station_overrides.get(normalize_station_name(query))
                if resolution.get("status") != "resolved" and override:
                    allowed_endpoints = {
                        candidate.get("endpoint") for candidate in resolution.get("candidates") or []
                    }
                    if override.get("endpoint") in allowed_endpoints:
                        resolution = {
                            "status": "resolved",
                            "query": query,
                            "station": override,
                            "candidates": resolution.get("candidates") or [],
                        }
                if debug:
                    print(f"[physical station] {json.dumps(resolution, ensure_ascii=False)}", file=sys.stderr)
                    if raw_path:
                        print(f"[raw] {raw_path}", file=sys.stderr)
                if resolution.get("status") != "resolved":
                    finish_timing()
                    if interactive and resolution.get("status") == "ambiguous":
                        raise StationSelectionRequired(query, resolution.get("candidates") or [])
                    answer = _resolution_message(resolution)
                    if trace is not None:
                        trace["rendered_answer"] = answer
                    return answer
                station = resolution["station"]
                resolved_stations.append(station)
                if resolution.get("resolution") == "station_cluster":
                    resolution_notes.append(
                        f"{query}駅候補の代表地点から検索しました。"
                    )
                response_key = "stations" if call.name == "suggest_stations" else "places"
                response = {
                    response_key: [
                        {
                            "id": station["endpoint"],
                            "name": station["name"],
                            "kind": "station",
                        }
                    ]
                }
                history.extend(
                    [_history_call(call.name, call.arguments), _history_result(call.name, response)]
                )
                continue

            # MCPClient enforces both the allow-list and the tool's JSON Schema.
            # A route-request planner call is already complete and authoritative.
            # Direct non-route tool calls retain the legacy value-safety binding.
            execution_arguments = (
                dict(call.arguments)
                if route_request
                else _route_arguments(
                    call.name,
                    call.arguments,
                    resolved_stations,
                    None,
                    num_itineraries=candidate_count,
                )
            )
            validate_tool_call(
                type(call)(call.name, execution_arguments),
                tools,
            )
            if trace is not None:
                trace["schema_validations"].append(
                    {"tool": call.name, "valid": True, "phase": "arguments"}
                )
            mcp_started = time.monotonic()
            envelope = client.call_tool(call.name, execution_arguments, tools=tools)
            mcp_elapsed = round((time.monotonic() - mcp_started) * 1000, 2)
            add_timing("mcp_plan_latency_ms", mcp_elapsed)
            if ui_payload is not None and call.name == "plan_route_map":
                # MCP Apps: hand the raw tool result to the caller so a UI host
                # can feed it to the ui://transit/route-map resource unchanged.
                result = envelope.get("result") or {}
                ui_payload.update(
                    {
                        "tool": call.name,
                        "content": result.get("content"),
                        "structuredContent": result.get("structuredContent"),
                        "isError": bool(result.get("isError")),
                    }
                )
            raw_path = (
                save_mcp_artifact(envelope, save_raw, call.name) if save_raw is not None else None
            )
            hints = dict(route_hints)
            if resolution_notes:
                hints["station_resolution_notes"] = resolution_notes
            normalize_started = time.monotonic()
            normalized = normalize_mcp_result(
                envelope,
                call.name,
                execution_arguments,
                query_overrides=hints,
                max_routes=None,
            )
            add_timing("normalize_latency_ms", (time.monotonic() - normalize_started) * 1000)
            rerank_started = time.monotonic()
            normalized = apply_route_constraints(normalized)
            normalized = rerank_routes(normalized)
            add_timing(
                "constraint_rerank_latency_ms", (time.monotonic() - rerank_started) * 1000
            )
            normalized_path = None
            if save_normalized is not None:
                normalized_path = _save_normalized(normalized, Path(save_normalized), call.name)
            if trace is not None:
                trace["mcp_calls"].append(
                    {
                        "tool": call.name,
                        "arguments": execution_arguments,
                        "status": "ok",
                        "latency_ms": mcp_elapsed,
                        "attempts": client.last_attempts,
                        "raw_path": str(raw_path) if raw_path else None,
                    }
                )
                trace["normalized_result"] = normalized
                trace["route_processing"] = {
                    "candidate_count": candidate_count,
                    "display_limit": display_limit,
                    "priority": route_hints.get("priority"),
                    "ranking": normalized.get("ranking"),
                    "constraints": {
                        key: route_hints.get(key)
                        for key in (
                            "avoid_station_texts",
                            "via_station_texts",
                            "avoid_line_texts",
                            "allowed_operator_groups",
                            "avoid_operator_groups",
                            "avoid_modes",
                        )
                        if route_hints.get(key)
                    },
                }
                trace["normalized_path"] = (
                    str(normalized_path) if normalized_path else None
                )
            if debug:
                print(f"[tool] {call.name} {json.dumps(execution_arguments, ensure_ascii=False)}", file=sys.stderr)
                if raw_path:
                    print(f"[raw] {raw_path}", file=sys.stderr)
                if normalized_path:
                    print(f"[normalized] {normalized_path}", file=sys.stderr)
                print(json.dumps(normalized, ensure_ascii=False, indent=2), file=sys.stderr)
            if (
                interactive
                and call.name in {"suggest_stations", "suggest_places", "reverse_geocode"}
                and normalized.get("suggestions")
            ):
                raise SuggestionSelectionRequired(
                    call.name,
                    str(call.arguments.get("q") or "候補"),
                    _display_suggestions(
                        str(call.arguments.get("q") or "候補"),
                        normalized["suggestions"],
                    ),
                )
            render_started = time.monotonic()
            answer = render_answer(normalized, max_routes=display_limit)
            add_timing("render_latency_ms", (time.monotonic() - render_started) * 1000)
            if trace is not None:
                trace["rendered_answer"] = answer
            finish_timing()
            return answer
    raise RuntimeError(f"{max_tool_steps}段階で経路tool callに到達しませんでした")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a tool call, execute Transit MCP, and render Japanese without a second LLM."
    )
    parser.add_argument("prompt")
    parser.add_argument("--adapter", default=os.getenv("FUNCTIONGEMMA_ADAPTER"))
    parser.add_argument("--mcp-url", default=MCP_ENDPOINT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--schema-mode", choices=("baked", "compact", "full"), default="baked")
    parser.add_argument("--save-raw", type=Path, default=Path("artifacts/mcp"))
    parser.add_argument("--save-normalized", type=Path)
    parser.add_argument("--max-routes", type=int, default=3)
    parser.add_argument("--max-tool-steps", type=int, default=7)
    parser.add_argument("--clarification-tool", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--normalize-ja", action="store_true")
    args = parser.parse_args()
    try:
        print(
            run_pipeline(
                args.prompt,
                adapter=args.adapter,
                mcp_url=args.mcp_url,
                schema_mode=args.schema_mode,
                schema_path=args.schema,
                save_raw=args.save_raw,
                save_normalized=args.save_normalized,
                max_routes=args.max_routes,
                max_tool_steps=args.max_tool_steps,
                debug=args.debug,
                clarification_tool=args.clarification_tool,
                normalize_ja=args.normalize_ja,
            )
        )
    except Exception as exc:  # CLI boundary: short Japanese message unless debugging.
        if args.debug:
            traceback.print_exc()
        else:
            print(f"{USER_ERROR}\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
