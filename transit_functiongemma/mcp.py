from __future__ import annotations

import json
import hashlib
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator


class MCPError(RuntimeError):
    pass


class MCPTimeoutError(MCPError):
    pass


_CACHE_LOCK = threading.Lock()
_RESPONSE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class MCPClient:
    def __init__(
        self,
        endpoint: str,
        timeout: float | None = None,
        retries: int | None = None,
        backoff: float = 0.25,
    ):
        self.endpoint = endpoint
        self.timeout = timeout or float(os.getenv("TRANSIT_MCP_TIMEOUT", "15"))
        self.retries = retries if retries is not None else int(os.getenv("TRANSIT_MCP_RETRIES", "2"))
        self.backoff = backoff
        self.client = httpx.Client(
            timeout=self.timeout,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            follow_redirects=True,
        )
        self._request_id = 0
        self._initialized = False
        self.last_attempts = 0
        self.retry_success_count = 0

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            return response.json()
        events: list[dict[str, Any]] = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line.removeprefix("data:").strip()))
        if not events:
            raise MCPError("MCP SSE response contained no JSON data event")
        return events[-1]

    def _cache_key(self, method: str, params: dict[str, Any] | None) -> str:
        payload = json.dumps([self.endpoint, method, params], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        cache_ttl: float = 0,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(method, params)
        if cache_ttl > 0:
            with _CACHE_LOCK:
                cached = _RESPONSE_CACHE.get(cache_key)
                if cached and cached[0] > time.monotonic():
                    self.last_attempts = 0
                    return cached[1]
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        response = None
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self.last_attempts = attempt + 1
            try:
                response = self.client.post(self.endpoint, json=payload)
                if response.status_code not in {429, 502, 503, 504}:
                    break
                last_error = MCPError(f"retryable MCP HTTP status {response.status_code}")
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.backoff * (2**attempt))
        if response is None or response.status_code in {429, 502, 503, 504}:
            if isinstance(last_error, httpx.TimeoutException):
                raise MCPTimeoutError(
                    f"Transit MCP timed out after {self.last_attempts} attempt(s)"
                ) from last_error
            raise MCPError(
                f"Transit MCP request failed after {self.last_attempts} attempt(s): {last_error}"
            ) from last_error
        if self.last_attempts > 1:
            self.retry_success_count += 1
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self.client.headers["Mcp-Session-Id"] = session_id
        decoded = self._decode(response)
        if "error" in decoded:
            raise MCPError(json.dumps(decoded["error"], ensure_ascii=False))
        if cache_ttl > 0:
            with _CACHE_LOCK:
                _RESPONSE_CACHE[cache_key] = (time.monotonic() + cache_ttl, decoded)
        return decoded

    def initialize(self) -> dict[str, Any]:
        if self._initialized:
            return {}
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "transit-functiongemma", "version": "0.1.0"},
            },
        )
        # Notifications have no id and may not return a response. This endpoint is
        # stateless, so sending it is unnecessary for tools/list and tools/call.
        self._initialized = True
        return result

    def list_tools(self) -> dict[str, Any]:
        self.initialize()
        return self.request("tools/list", {}, cache_ttl=300)

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if tools is not None:
            by_name = {tool["name"]: tool for tool in tools}
            if name not in by_name:
                raise MCPError(f"Tool is not allow-listed: {name}")
            schema = by_name[name].get("inputSchema", {})
            errors = sorted(Draft202012Validator(schema).iter_errors(arguments), key=str)
            if errors:
                raise MCPError("Invalid tool arguments: " + "; ".join(e.message for e in errors))
        self.initialize()
        cache_ttl = (
            float(os.getenv("TRANSIT_MCP_CACHE_TTL", "30"))
            if name in {"suggest_stations", "suggest_places", "reverse_geocode", "get_station", "list_feeds"}
            else 0
        )
        return self.request(
            "tools/call", {"name": name, "arguments": arguments}, cache_ttl=cache_ttl
        )


def schema_hash(tools: list[dict[str, Any]]) -> str:
    canonical = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_mcp_artifact(envelope: dict[str, Any], directory: str | Path, stem: str) -> Path:
    """Preserve the complete result, including MCP Apps structured/resource/meta data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    path = directory / f"{timestamp}_{stem}.json"
    result = envelope.get("result", {})
    saved = {
        "savedAt": timestamp,
        "envelope": envelope,
        "structuredContent": result.get("structuredContent"),
        "content": result.get("content"),
        "resources": [
            item for item in result.get("content", []) if item.get("type") in {"resource", "resource_link"}
        ],
        "meta": result.get("_meta"),
    }
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_mcp_result(envelope: dict[str, Any]) -> str:
    """Initial deterministic formatter: return MCP text content, never model guesses."""
    chunks: list[str] = []
    for item in envelope.get("result", {}).get("content", []):
        if item.get("type") == "text":
            text = item.get("text", "")
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
            chunks.append(text)
    return "\n".join(chunks) or json.dumps(envelope.get("result", {}), ensure_ascii=False, indent=2)
