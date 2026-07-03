#!/usr/bin/env python3
"""Small persistent HTTP API for the deterministic transit answer pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from transit_functiongemma.answer_pipeline import (
    StationSelectionRequired,
    SuggestionSelectionRequired,
    run_pipeline,
)
from transit_functiongemma.route_constraints import normalize_station_name
from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, MCP_ENDPOINT, MODEL_ID
from transit_functiongemma.mcp import MCPClient, MCPError, MCPTimeoutError


MAX_BODY_BYTES = 16 * 1024
MAX_PROMPT_CHARS = 500
SESSION_TTL_SECONDS = 15 * 60
ROUTE_MAP_RESOURCE_URI = "ui://transit/route-map"
UI_RESOURCE_TTL_SECONDS = 3600.0
RATE_LIMIT_REQUESTS = int(os.getenv("TRANSIT_RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("TRANSIT_RATE_LIMIT_WINDOW", "60"))


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.lock = threading.Lock()
        self.events: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        threshold = now - self.window_seconds
        with self.lock:
            recent = [stamp for stamp in self.events.get(key, []) if stamp >= threshold]
            if len(recent) >= self.limit:
                self.events[key] = recent
                return False
            recent.append(now)
            self.events[key] = recent
            return True


class AnonymousAuditLogger:
    _COORDINATES = re.compile(
        r"(?<!\d)-?\d{1,2}(?:\.\d+)?\s*[,/、 ]\s*-?\d{2,3}(?:\.\d+)?(?!\d)"
    )

    def __init__(self) -> None:
        self.enabled = os.getenv("TRANSIT_AUDIT_LOG", "1") == "1"
        self.save_query = os.getenv("TRANSIT_LOG_USER_QUERY", "0") == "1"
        self.save_location = os.getenv("TRANSIT_LOG_RAW_LOCATION", "0") == "1"
        self.path = Path(os.getenv("TRANSIT_AUDIT_PATH", "artifacts/anonymous_requests.jsonl"))
        self.salt = os.getenv("TRANSIT_LOG_SALT", uuid.uuid4().hex)
        self.lock = threading.Lock()

    def _hash(self, value: str) -> str:
        return hashlib.sha256(f"{self.salt}:{value}".encode("utf-8")).hexdigest()[:16]

    def write(
        self,
        request_id: str,
        client: str,
        status: int,
        prompt: str | None,
        response_kind: str | None,
    ) -> None:
        if not self.enabled:
            return
        event: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "client_hash": self._hash(client),
            "status": status,
            "response_kind": response_kind,
        }
        if prompt:
            event["query_hash"] = self._hash(prompt)
            event["query_chars"] = len(prompt)
            if self.save_query:
                event["query"] = (
                    prompt
                    if self.save_location
                    else self._COORDINATES.sub("[location-redacted]", prompt)
                )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock, self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            # Logging must never turn a successful transit response into a 5xx.
            return


class BehaviorLogger:
    """Write privacy-filtered pipeline observations for short operational trials."""

    _COORDINATES = AnonymousAuditLogger._COORDINATES
    _GEO_ENDPOINT = re.compile(
        r"geo:\s*-?\d{1,2}(?:\.\d+)?\s*,\s*-?\d{2,3}(?:\.\d+)?",
        re.IGNORECASE,
    )
    _LOCATION_KEYS = {"lat", "lon", "latitude", "longitude", "coordinates"}

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        directory: str | Path | None = None,
        retention_days: int | None = None,
        save_query: bool | None = None,
        save_answer: bool | None = None,
    ) -> None:
        self.enabled = (
            os.getenv("TRANSIT_BEHAVIOR_LOG", "0") == "1" if enabled is None else enabled
        )
        self.directory = Path(
            directory
            or os.getenv("TRANSIT_BEHAVIOR_LOG_DIR", "artifacts/behavior_logs")
        )
        self.retention_days = max(
            1,
            retention_days
            if retention_days is not None
            else int(os.getenv("TRANSIT_BEHAVIOR_LOG_RETENTION_DAYS", "45")),
        )
        self.save_query = (
            os.getenv("TRANSIT_BEHAVIOR_LOG_USER_QUERY", "0") == "1"
            if save_query is None
            else save_query
        )
        self.save_answer = (
            os.getenv("TRANSIT_BEHAVIOR_LOG_ANSWER", "1") == "1"
            if save_answer is None
            else save_answer
        )
        self.lock = threading.Lock()
        self._last_cleanup_date = None
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._delete_expired()
            self._last_cleanup_date = datetime.now(ZoneInfo("Asia/Tokyo")).date()

    @classmethod
    def _redact_text(cls, value: str) -> str:
        value = cls._GEO_ENDPOINT.sub("[location-redacted]", value)
        return cls._COORDINATES.sub("[location-redacted]", value)

    @classmethod
    def _sanitize(cls, value: Any, key: str | None = None) -> Any:
        if key and key.casefold() in cls._LOCATION_KEYS:
            return "[location-redacted]"
        if isinstance(value, str):
            return cls._redact_text(value)
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, dict):
            return {
                str(item_key): cls._sanitize(item_value, str(item_key))
                for item_key, item_value in value.items()
            }
        return value

    def _delete_expired(self) -> None:
        cutoff = datetime.now(ZoneInfo("Asia/Tokyo")).date() - timedelta(
            days=self.retention_days
        )
        for path in self.directory.glob("????-??-??.jsonl"):
            try:
                log_date = datetime.strptime(path.stem, "%Y-%m-%d").date()
                if log_date < cutoff:
                    path.unlink()
            except (OSError, ValueError):
                continue

    def _trace_summary(self, trace: Any) -> dict[str, Any] | None:
        if not isinstance(trace, dict):
            return None
        summary: dict[str, Any] = {}
        for key in (
            "router_outputs",
            "planner_steps",
            "tool_calls",
            "schema_validations",
            "route_processing",
            "model_route_intent",
            "effective_route_intent",
            "timings",
        ):
            if trace.get(key):
                summary[key] = self._sanitize(trace[key])
        calls: list[dict[str, Any]] = []
        for call in trace.get("mcp_calls") or []:
            if not isinstance(call, dict):
                continue
            calls.append(
                self._sanitize(
                    {
                        key: call.get(key)
                        for key in ("tool", "arguments", "status", "latency_ms", "attempts")
                        if call.get(key) is not None
                    }
                )
            )
        if calls:
            summary["mcp_calls"] = calls
        if trace.get("no_call") is not None:
            summary["no_call"] = bool(trace["no_call"])
        if trace.get("graphical_defaulted") is not None:
            summary["graphical_defaulted"] = bool(trace["graphical_defaulted"])
        if self.save_answer and trace.get("rendered_answer"):
            summary["rendered_answer"] = self._redact_text(
                str(trace["rendered_answer"])
            )
        return summary or None

    def write(
        self,
        request_id: str,
        status: int,
        prompt: str | None,
        response: dict[str, Any],
        behavior: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = datetime.now(ZoneInfo("Asia/Tokyo"))
        behavior = behavior or {}
        effective_prompt = behavior.get("effective_prompt") or prompt
        input_prompt = behavior.get("input_prompt")
        event: dict[str, Any] = {
            "timestamp": now.isoformat(timespec="seconds"),
            "request_id": request_id,
            "status": status,
            "response": {
                key: response.get(key)
                for key in ("ok", "kind", "error", "message", "elapsed_ms")
                if response.get(key) is not None
            },
        }
        choices = response.get("choices")
        if isinstance(choices, list):
            event["response"]["choice_count"] = len(choices)
        if self.save_answer and response.get("answer"):
            event["response"]["answer"] = self._redact_text(str(response["answer"]))
        if effective_prompt:
            event["query_chars"] = len(str(effective_prompt))
            if self.save_query:
                event["query"] = self._redact_text(str(effective_prompt))
        if self.save_query and input_prompt and input_prompt != effective_prompt:
            event["input_query"] = self._redact_text(str(input_prompt))
        for key in ("conversation_id", "selection", "role"):
            if behavior.get(key) is not None:
                event[key] = behavior[key]
        trace = self._trace_summary(behavior.get("trace"))
        if trace:
            event["trace"] = trace
        path = self.directory / f"{now.date().isoformat()}.jsonl"
        try:
            with self.lock:
                if self._last_cleanup_date != now.date():
                    self._delete_expired()
                    self._last_cleanup_date = now.date()
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            # Observability must never make a transit request fail.
            return


def _station_label(candidate: dict[str, Any], fallback: str = "駅") -> str:
    name = str(candidate.get("name") or fallback).strip()
    return name if name.endswith("駅") else f"{name}駅"


def _station_override(candidate: dict[str, Any]) -> dict[str, Any]:
    """Convert a direct suggestion into the physical-station override shape."""
    override = dict(candidate)
    lat = candidate.get("lat")
    lon = candidate.get("lon")
    if not override.get("endpoint") and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        override["endpoint"] = f"geo:{lat},{lon}"
    override["name"] = _station_label(candidate)
    return override


def _compose_route_prompt(role: str, candidate: dict[str, Any], prompt: str) -> str:
    """Combine a confirmed endpoint with the user's missing-side input."""
    text = prompt.strip()
    label = _station_label(candidate)
    source = str(candidate.get("source_label") or "").strip()
    line_hint = f" {source}を利用" if source.endswith("線") and source not in text else ""
    if role == "origin":
        text = re.sub(r"^から\s*", "", text)
        if not re.search(r"(?:まで|へ)", text):
            text = f"{text}まで"
        return f"{label}から{text}{line_hint}"
    text = re.sub(r"(?:まで|へ)\s*$", "", text)
    if "から" in text:
        return f"{text}{label}まで{line_hint}"
    return f"{text}から{label}まで{line_hint}"


class TransitAPI:
    def __init__(
        self,
        adapter: str,
        schema_mode: str = "baked",
        normalize_ja: bool = False,
    ) -> None:
        from transit_functiongemma.infer import ToolRouter

        self.adapter = adapter
        self.schema_mode = schema_mode
        self.show_trace = os.getenv("TRANSIT_WEB_SHOW_TRACE", "0") == "1"
        self.behavior_logging = os.getenv("TRANSIT_BEHAVIOR_LOG", "0") == "1"
        self.default_graphical = (
            os.getenv("TRANSIT_DEFAULT_GRAPHICAL", "0") == "1"
        )
        self.lock = threading.Lock()
        self.router = ToolRouter(
            base_model=MODEL_ID,
            adapter=adapter,
            schema_path=DEFAULT_SCHEMA_PATH,
            schema_mode=schema_mode,
            clarification_tool=True,
            normalize_ja=normalize_ja,
        )
        self.sessions: dict[str, dict[str, Any]] = {}
        self.ui_lock = threading.Lock()
        self.ui_resource_cache: dict[str, tuple[float, str]] = {}

    def route_map_html(self) -> str | None:
        """Fetch and cache the MCP Apps route-map resource (text/html;profile=mcp-app)."""
        with self.ui_lock:
            cached = self.ui_resource_cache.get(ROUTE_MAP_RESOURCE_URI)
            if cached and cached[0] > time.monotonic():
                return cached[1]
        try:
            with MCPClient(MCP_ENDPOINT) as client:
                client.initialize()
                envelope = client.request(
                    "resources/read", {"uri": ROUTE_MAP_RESOURCE_URI}
                )
        except MCPError:
            # Serve a stale copy over an outage rather than dropping the map UI.
            return cached[1] if cached else None
        contents = (envelope.get("result") or {}).get("contents") or []
        html = next(
            (
                item.get("text")
                for item in contents
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ),
            None,
        )
        if not html:
            return cached[1] if cached else None
        with self.ui_lock:
            self.ui_resource_cache[ROUTE_MAP_RESOURCE_URI] = (
                time.monotonic() + UI_RESOURCE_TTL_SECONDS,
                html,
            )
        return html

    def _cleanup_sessions(self) -> None:
        threshold = time.monotonic() - SESSION_TTL_SECONDS
        self.sessions = {
            key: value for key, value in self.sessions.items() if value["updated_at"] >= threshold
        }

    def query(
        self,
        prompt: str | None,
        conversation_id: str | None = None,
        selection: int | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        session: dict[str, Any] | None = None

        def finish(
            response: dict[str, Any], trace: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            if getattr(self, "behavior_logging", False):
                response["_behavior"] = {
                    "conversation_id": conversation_id,
                    "selection": selection,
                    "role": role,
                    "input_prompt": prompt,
                    "effective_prompt": session.get("prompt") if session else prompt,
                    "trace": trace,
                }
            return response

        with self.lock:
            self._cleanup_sessions()
            if conversation_id:
                session = self.sessions.get(conversation_id)
                if session is None:
                    raise ValueError("conversation_not_found")
                if isinstance(selection, int):
                    candidates = session.get("candidates") or []
                    if not 0 <= selection < len(candidates):
                        raise ValueError("invalid_selection")
                    selected = candidates[selection]
                    if session.get("pending_kind") == "suggestion":
                        session["pending_kind"] = "selected_suggestion"
                        session["selected"] = selected
                        session["updated_at"] = time.monotonic()
                        source = selected.get("source_label") or selected.get("description")
                        display = str(selected.get("name") or session.get("pending_query") or "候補")
                        if source:
                            display += f"（{source}）"
                        return finish({
                            "ok": True,
                            "kind": "selected",
                            "conversation_id": conversation_id,
                            "answer": f"{display}を選択しました。",
                            "selected": selected,
                            "elapsed_ms": round((time.monotonic() - started) * 1000),
                        })
                    session["overrides"][normalize_station_name(session["pending_query"])] = selected
                    session["updated_at"] = time.monotonic()
                elif role is not None:
                    if role not in {"origin", "destination"}:
                        raise ValueError("invalid_role")
                    if session.get("pending_kind") != "selected_suggestion":
                        raise ValueError("invalid_conversation_state")
                    selected = session.get("selected") or {}
                    session["selected_role"] = role
                    session["pending_kind"] = "route_endpoint"
                    session["updated_at"] = time.monotonic()
                    label = _station_label(selected)
                    missing_label = "目的地" if role == "origin" else "出発地"
                    return finish({
                        "ok": True,
                        "kind": "awaiting_route",
                        "conversation_id": conversation_id,
                        "answer": f"{label}を{'出発地' if role == 'origin' else '目的地'}に設定しました。{missing_label}を入力してください。",
                        "placeholder": f"例：{'上野' if role == 'origin' else '横浜'}（時刻や利用路線も入力できます）",
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    })
                elif prompt and session.get("pending_kind") == "route_endpoint":
                    selected = session.get("selected") or {}
                    selected_role = session.get("selected_role")
                    session["prompt"] = _compose_route_prompt(selected_role, selected, prompt)
                    override = _station_override(selected)
                    session["overrides"][normalize_station_name(override["name"])] = override
                    session["pending_kind"] = "route"
                    session["updated_at"] = time.monotonic()
                else:
                    raise ValueError("invalid_conversation_state")
            else:
                if not prompt:
                    raise ValueError("prompt_required")
                conversation_id = uuid.uuid4().hex
                session = {
                    "prompt": prompt,
                    "overrides": {},
                    "updated_at": time.monotonic(),
                }
                self.sessions[conversation_id] = session

            try:
                show_trace = getattr(self, "show_trace", False)
                capture_trace = show_trace or getattr(self, "behavior_logging", False)
                trace: dict[str, Any] | None = {} if capture_trace else None
                ui_payload: dict[str, Any] = {}
                answer = run_pipeline(
                    session["prompt"],
                    adapter=self.adapter,
                    mcp_url=MCP_ENDPOINT,
                    schema_mode=self.schema_mode,
                    schema_path=DEFAULT_SCHEMA_PATH,
                    save_raw=Path("artifacts/web_mcp"),
                    save_normalized=Path("artifacts/web_normalized"),
                    max_routes=1,
                    max_tool_steps=7,
                    clarification_tool=True,
                    router_instance=self.router,
                    station_overrides=session["overrides"],
                    interactive=True,
                    trace=trace,
                    ui_payload=ui_payload,
                    default_graphical=getattr(self, "default_graphical", False),
                )
            except StationSelectionRequired as exc:
                session["pending_kind"] = "physical_station"
                session["pending_query"] = exc.query
                session["candidates"] = exc.candidates
                session["updated_at"] = time.monotonic()
                return finish({
                    "ok": True,
                    "kind": "selection",
                    "conversation_id": conversation_id,
                    "answer": f"「{exc.query}」に一致する物理駅が複数あります。",
                    "question": "どの駅を使いますか？",
                    "choices": [
                        {
                            "index": index,
                            "name": candidate.get("name") or exc.query,
                            "description": candidate.get("description"),
                            "lat": candidate.get("lat"),
                            "lon": candidate.get("lon"),
                        }
                        for index, candidate in enumerate(exc.candidates)
                    ],
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                }, trace)
            except SuggestionSelectionRequired as exc:
                session["pending_kind"] = "suggestion"
                session["pending_query"] = exc.query
                session["candidates"] = exc.candidates
                session["updated_at"] = time.monotonic()
                label = "駅" if exc.tool_name in {"suggest_stations", "reverse_geocode"} else "場所"
                return finish({
                    "ok": True,
                    "kind": "selection",
                    "conversation_id": conversation_id,
                    "answer": f"{label}候補を見つけました。",
                    "question": f"どの{label}を使いますか？",
                    "selection_label": label,
                    "choices": [
                        {
                            "index": index,
                            "name": candidate.get("name") or exc.query,
                            "description": candidate.get("description")
                            or candidate.get("source_label"),
                            "lat": candidate.get("lat"),
                            "lon": candidate.get("lon"),
                        }
                        for index, candidate in enumerate(exc.candidates)
                    ],
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                }, trace)

            self.sessions.pop(conversation_id, None)
            response = {
                "ok": True,
                "kind": "answer",
                "answer": answer,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
            if ui_payload.get("structuredContent") is not None or ui_payload.get("content"):
                response["kind"] = "map"
                # The MCP-authored text summary is the source-only answer for a
                # map result; the renderer does not parse the map options shape.
                map_text = next(
                    (
                        item.get("text")
                        for item in ui_payload.get("content") or []
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                        and item.get("text")
                    ),
                    None,
                )
                if map_text:
                    response["answer"] = map_text
                response["map_result"] = {
                    "content": ui_payload.get("content"),
                    "structuredContent": ui_payload.get("structuredContent"),
                    "isError": bool(ui_payload.get("isError")),
                }
            if show_trace:
                response["trace"] = trace
            return finish(response, trace)


class Handler(BaseHTTPRequestHandler):
    server_version = "TransitFunctionGemma/1.0"

    @property
    def api(self) -> TransitAPI:
        return self.server.api  # type: ignore[attr-defined]

    @property
    def limiter(self) -> SlidingWindowRateLimiter:
        return self.server.limiter  # type: ignore[attr-defined]

    @property
    def audit(self) -> AnonymousAuditLogger:
        return self.server.audit  # type: ignore[attr-defined]

    @property
    def behavior(self) -> BehaviorLogger:
        return self.server.behavior  # type: ignore[attr-defined]

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        request_id = getattr(self, "request_id", uuid.uuid4().hex)
        behavior = payload.pop("_behavior", None)
        payload.setdefault("request_id", request_id)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Request-Id", request_id)
        self.end_headers()
        self.wfile.write(body)
        if hasattr(self, "request_id"):
            self.audit.write(
                request_id,
                self.client_address[0],
                status,
                getattr(self, "audit_prompt", None),
                payload.get("kind"),
            )
            self.behavior.write(
                request_id,
                status,
                getattr(self, "audit_prompt", None),
                payload,
                behavior,
            )

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/health", "/api/health"}:
            self._json(200, {"ok": True, "model": MODEL_ID, "ready": True})
            return
        if self.path.rstrip("/") in {"/ui/route-map", "/api/ui/route-map"}:
            html = self.api.route_map_html()
            if html is None:
                self._json(502, {"ok": False, "error": "ui_resource_unavailable"})
                return
            # Served as text/plain: the frontend injects it into a sandboxed
            # iframe srcdoc, and third-party HTML must never execute on this
            # origin if the URL is opened directly.
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        supplied_request_id = self.headers.get("X-Request-Id", "")
        self.request_id = (
            supplied_request_id
            if re.fullmatch(r"[A-Za-z0-9._-]{8,80}", supplied_request_id)
            else uuid.uuid4().hex
        )
        if self.path.rstrip("/") not in {"/query", "/api/query"}:
            self._json(404, {"ok": False, "error": "not_found"})
            return
        if not self.limiter.allow(self.client_address[0]):
            self._json(429, {"ok": False, "error": "rate_limited"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"ok": False, "error": "invalid_content_length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(413, {"ok": False, "error": "request_too_large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"ok": False, "error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"ok": False, "error": "invalid_payload"})
            return
        conversation_id = payload.get("conversation_id")
        selection = payload.get("selection")
        role = payload.get("role")
        prompt = payload.get("prompt")
        self.audit_prompt = prompt if isinstance(prompt, str) else None
        if conversation_id is None:
            if not isinstance(prompt, str) or not prompt.strip():
                self._json(400, {"ok": False, "error": "prompt_required"})
                return
            prompt = prompt.strip()
            if len(prompt) > MAX_PROMPT_CHARS:
                self._json(400, {"ok": False, "error": "prompt_too_long"})
                return
        elif not isinstance(conversation_id, str):
            self._json(400, {"ok": False, "error": "invalid_conversation"})
            return
        elif selection is None and role is None and not isinstance(prompt, str):
            self._json(400, {"ok": False, "error": "invalid_conversation_action"})
            return
        elif selection is not None and not isinstance(selection, int):
            self._json(400, {"ok": False, "error": "invalid_selection"})
            return
        elif role is not None and not isinstance(role, str):
            self._json(400, {"ok": False, "error": "invalid_role"})
            return
        elif isinstance(prompt, str):
            prompt = prompt.strip()
            if not prompt:
                self._json(400, {"ok": False, "error": "prompt_required"})
                return
            if len(prompt) > MAX_PROMPT_CHARS:
                self._json(400, {"ok": False, "error": "prompt_too_long"})
                return
        try:
            self._json(200, self.api.query(prompt, conversation_id, selection, role))
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
        except MCPTimeoutError:
            self._json(
                504,
                {
                    "ok": False,
                    "error": "mcp_timeout",
                    "message": "交通データの応答が時間内に届きませんでした。少し待って再検索してください。",
                },
            )
        except MCPError:
            self._json(
                502,
                {
                    "ok": False,
                    "error": "mcp_unavailable",
                    "message": "交通データを取得できませんでした。少し待って再検索してください。",
                },
            )
        except Exception as exc:
            self.log_error("query failed: %s", exc)
            self._json(
                500,
                {
                    "ok": False,
                    "error": "pipeline_error",
                    "message": "乗換案内を処理できませんでした。条件を変えて再検索してください。",
                },
            )

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Transit FunctionGemma HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--adapter",
        default=os.getenv(
            "FUNCTIONGEMMA_ADAPTER",
            "outputs/functiongemma-transit-plus-r4",
        ),
    )
    parser.add_argument("--schema-mode", choices=("baked", "compact", "full"), default="baked")
    parser.add_argument(
        "--normalize-ja",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("FUNCTIONGEMMA_NORMALIZE_JA", "0") == "1",
    )
    args = parser.parse_args()

    api = TransitAPI(args.adapter, args.schema_mode, args.normalize_ja)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.api = api  # type: ignore[attr-defined]
    server.limiter = SlidingWindowRateLimiter(  # type: ignore[attr-defined]
        RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
    )
    server.audit = AnonymousAuditLogger()  # type: ignore[attr-defined]
    server.behavior = BehaviorLogger()  # type: ignore[attr-defined]
    print(f"Transit API ready on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
