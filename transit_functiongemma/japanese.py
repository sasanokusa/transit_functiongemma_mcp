from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any

from transit_functiongemma.route_intent import (
    extract_route_endpoints,
    extract_route_intent,
    mcp_time_type,
    normalize_date,
    normalize_time,
)
from transit_functiongemma.toolcall import ToolCall


_SPACE_RE = re.compile(r"[ \t\u3000]+")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?::[A-Za-z0-9._:-]+)+")
_COORD_PATTERNS = (
    re.compile(r"北緯\s*(?P<lat>-?\d+(?:\.\d+)?)\s*[,、 ]*\s*東経\s*(?P<lon>-?\d+(?:\.\d+)?)"),
    re.compile(r"緯度\s*(?P<lat>-?\d+(?:\.\d+)?)\s*[,、 ]*\s*経度\s*(?P<lon>-?\d+(?:\.\d+)?)"),
    re.compile(r"(?P<lat>-?\d{1,2}(?:\.\d+)?)\s*[,、]\s*(?P<lon>-?\d{2,3}(?:\.\d+)?)"),
    re.compile(r"(?P<lat>-?\d{1,2}(?:\.\d+)?)\s*/\s*(?P<lon>-?\d{2,3}(?:\.\d+)?)"),
    re.compile(
        r"lat\s*=\s*(?P<lat>-?\d+(?:\.\d+)?)\s+lon\s*=\s*(?P<lon>-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)


def _clean(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).strip()
    value = _SPACE_RE.sub(" ", value)
    return value


def normalize_japanese_surface(text: str) -> str:
    """Normalize notation only; do not infer intent or semantic slots."""
    return _clean(text)


def _text_coordinate_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for pattern in _COORD_PATTERNS:
        for match in pattern.finditer(text):
            pair = (match.group("lat"), match.group("lon"))
            if pair not in pairs:
                pairs.append(pair)
    return pairs


_COORDINATE_SNAP_TOLERANCE = 0.05
_TIME_PAD_RE = re.compile(r"(\d):(\d{2})(?::(\d{2}))?")


def repair_tool_call_values(
    call: ToolCall, text: str, reference_datetime: str | None = None
) -> ToolCall:
    """Restore exact user-written values that a small model cannot copy reliably.

    Value fidelity only: coordinate digits, explicit IDs, relative-date arithmetic,
    and time zero-padding. The tool name is never changed and no slot is inferred
    that the model did not already choose (except a date derived from an explicit
    relative-day word, which is calendar arithmetic rather than intent parsing).
    """
    original = _clean(text)
    arguments = dict(call.arguments)

    # Coordinates: snap a near-miss lat/lon to the single pair written in the text.
    if isinstance(arguments.get("lat"), (int, float)) and isinstance(
        arguments.get("lon"), (int, float)
    ):
        pairs = _text_coordinate_pairs(original)
        if len(pairs) == 1:
            lat_value, lon_value = float(pairs[0][0]), float(pairs[0][1])
            if (
                abs(float(arguments["lat"]) - lat_value) <= _COORDINATE_SNAP_TOLERANCE
                and abs(float(arguments["lon"]) - lon_value) <= _COORDINATE_SNAP_TOLERANCE
            ):
                arguments["lat"] = lat_value
                arguments["lon"] = lon_value

    # IDs: snap to the single explicit ID written in the text.
    if isinstance(arguments.get("id"), str):
        ids = {match.rstrip(":") for match in _ID_RE.findall(original)}
        if len(ids) == 1:
            arguments["id"] = next(iter(ids))

    # Relative dates: calendar arithmetic belongs to the runtime, not the model.
    # Only station_departures is repaired; its YYYYMMDD format is schema-verified.
    if call.name == "station_departures":
        date_value = normalize_date(original, reference_datetime)
        if date_value:
            arguments["date"] = date_value

    # Time notation: zero-pad H:MM so MCP always receives HH:MM.
    time_value = arguments.get("time")
    if isinstance(time_value, str):
        match = _TIME_PAD_RE.fullmatch(time_value)
        if match:
            arguments["time"] = (
                f"{int(match.group(1)):02d}:{match.group(2)}"
                + (f":{match.group(3)}" if match.group(3) else "")
            )

    if arguments == call.arguments:
        return call
    return ToolCall(call.name, arguments)


def _reference_datetime(value: str | None) -> datetime:
    if value:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2}))?", value)
        if match:
            return datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4) or 0),
                int(match.group(5) or 0),
            )
    return datetime.now()


def _date_hint(text: str, reference_datetime: str | None) -> str | None:
    return normalize_date(text, reference_datetime)


def _time_hint(text: str) -> str | None:
    value = normalize_time(text)
    return f"{int(value[:2])}{value[2:]}" if value else None


def _route_endpoints(text: str) -> tuple[str, str] | None:
    natural = extract_route_endpoints(text)
    if natural:
        return natural
    patterns = (
        re.compile(r"(?P<origin>[^、。\s]+?)(?:駅)?から(?P<destination>[^、。\s]+?)(?:駅)?(?:まで|へ)"),
        re.compile(r"(?P<origin>[^、。\s]+?)(?:駅)?\s*(?:→|⇒|->)\s*(?P<destination>[^、。\s]+?)(?:駅)?(?=まで|へ|着|、|。|\s|$)"),
        re.compile(r"(?P<origin>[^、。\s]+?)(?:駅)?発(?:で|の)?(?P<destination>[^、。\s]+?)(?:駅)?着"),
        re.compile(r"(?P<origin>[^、。\s]+?)(?:駅)?発[、,\s]+(?P<destination>[^、。\s]+?)(?:駅)?着"),
        re.compile(r"(?P<origin>[^、。\s]+?)(?:駅)?を出て(?P<destination>[^、。\s]+?)(?:駅)?へ(?:向か|行)")
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group("origin"), match.group("destination")
    return None


def _station_query(text: str) -> str | None:
    quoted_station = re.search(r"[「\"]([^」\"]+)[」\"]を駅として", text)
    if quoted_station:
        return quoted_station.group(1).removesuffix("駅")
    quoted = re.search(r"駅名[「\"]([^」\"]+)[」\"]", text)
    if quoted:
        return quoted.group(1).removesuffix("駅")
    operator_qualified = re.search(r"[^、。\s]+の([^、。\s]+?)(?:駅)(?:って|は|を|の|$)", text)
    if operator_qualified:
        return operator_qualified.group(1)
    patterns = (
        r"乗る駅として([^、。\s]+?)を検索",
        r"降りる駅候補\s*[:：]\s*([^、。\s]+)",
        r"([^、。\s]+?)(?:の)?駅IDを",
        r"([^、。\s]+?)周辺の駅じゃなくて駅名として",
        r"([^、。\s]+?)(?:駅)?っぽいやつ",
        r"([^、。\s]+?)(?:駅)?で合ってるか候補",
        r"[「\"]([^」\"]+)[」\"]の駅候補",
        r"([^、。\s]+?)っていう駅",
        r"(?:東急|JR|地下鉄|鉄道)?(?:の)?([^、。\s「」]+?)(?:駅)(?:って|は|を|の|$)",
        r"([^、。\s「」]+?)を(?:鉄道)?駅として",
        r"([^、。\s「」]+?)(?:駅)?(?:の)?(?:ID|駅候補)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).removesuffix("駅")
            if value and value not in {"この", "その", "最寄り", "出発", "目的地"}:
                return value
    return None


def _place_query(text: str) -> str | None:
    quoted = re.search(r"(?:施設名|場所)[「\"]([^」\"]+)[」\"]", text)
    if quoted:
        return quoted.group(1)
    patterns = (
        r"([^、。\s]+?)を目的地として探",
        r"([^、。\s]+?)の最寄りを",
        r"([^、。\s]+?)に行きたいから場所候補",
        r"([^、。\s]+?)周辺の乗車地点",
        r"住所/施設として([^、。\s]+?)を検索",
        r"目的地が([^、。\s]+?)なんだけど",
        r"([^、。\s]+?)ってスポット",
        r"([^、。\s]+?)に行きたい。まずスポット候補",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    colloquial = re.search(
        r"([^、。\s]+?)(?:って|は)?(?:場所|施設)(?:の)?(?:候補|検索|情報)", text
    )
    if colloquial:
        return colloquial.group(1).removesuffix("を")
    match = re.search(r"([^、。\s]+?)を(?:駅じゃなく|駅ではなく)?(?:場所|施設)(?:として|で)?(?:探|検索)", text)
    if match:
        return match.group(1)
    match = re.search(r"(?:駅ではなく)?場所(?:の|として)?([^、。\s]+)", text)
    return match.group(1).removesuffix("を") if match else None


def _line_hints(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"([^\s、。]+線)(?=で|を|経由|利用|使)", text):
        value = match.group(1)
        for separator in ("から", "まで", "なら", "は"):
            value = value.split(separator)[-1]
        if value and value not in values:
            values.append(value)
    return values


def _canonical(intent: str, original: str, **slots: Any) -> str:
    lines = ["[normalized_ja]", f"intent={intent}"]
    for key, value in slots.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = "|".join(str(item) for item in value)
        else:
            rendered = str(value)
        lines.append(f"{key}={rendered}")
    lines.append(f"original={original}")
    return "\n".join(lines)


def _normalize_japanese_prompt_legacy(
    text: str, reference_datetime: str | None = None
) -> str:
    """Legacy semantic normalizer retained behind an explicit opt-in flag."""
    original = _clean(text)
    if not original:
        return original

    for pattern in _COORD_PATTERNS:
        match = pattern.search(original)
        if match:
            radius = re.search(r"(?:周囲|半径)\s*(\d+)\s*m", original, re.IGNORECASE)
            return _canonical(
                "reverse_geocode",
                original,
                lat=match.group("lat"),
                lon=match.group("lon"),
                radius_meters=radius.group(1) if radius else None,
            )

    station_id = (_ID_RE.search(original) or [None])[0]
    if station_id:
        station_id = station_id.rstrip(":")
    if station_id and re.search(r"発車|出発便|出る便|時刻表|発車標|始発", original):
        requested_limit = re.search(r"(\d+)\s*件", original)
        return _canonical(
            "station_departures",
            original,
            station_id=station_id,
            date=_date_hint(original, reference_datetime),
            time=_time_hint(original),
            route_type=(
                "last" if "終電" in original else "first" if "始発" in original else None
            ),
            limit=requested_limit.group(1) if requested_limit else None,
        )
    if station_id and re.search(r"詳細|どの駅|何駅|表す駅|駅情報|正式名称|詳しく", original):
        return _canonical("get_station", original, station_id=station_id)

    if re.search(
        r"(?:フィード|feeds?|GTFS|交通データ|データ(?:ソース|出典|提供元)|提供元データ|"
        r"収録|対応エリア|ライセンス|出典一覧|事業者(?:の)?一覧|どこの事業者|対応事業者)",
        original,
        re.IGNORECASE,
    ):
        return _canonical("list_feeds", original)

    place = _place_query(original)
    if place:
        return _canonical("suggest_places", original, query=place)

    if re.search(
        r"(?:出発地|目的地).*(?:まだ|未定|あとで)|ここから|この辺から|"
        r"から(?:出たい|どこか)|駅IDがわからない|どこ行くかは未定",
        original,
    ):
        return _canonical("clarification", original, missing="origin|destination")

    endpoints = _route_endpoints(original)
    if endpoints:
        origin, destination = endpoints
        route_intent = extract_route_intent(original, reference_datetime)
        route_type = mcp_time_type(route_intent["time_mode"])
        priority = {
            "fast": "fastest",
            "cheap": "lowest_fare",
            "few_transfers": "fewest_transfers",
            "less_walk": "shortest_walk",
        }.get(route_intent["priority"])
        return _canonical(
            "route",
            original,
            origin=origin.removesuffix("駅"),
            destination=destination.removesuffix("駅"),
            map=(
                bool(re.search(r"地図|マップ|経路図|グラフィカル", original))
                and not bool(re.search(r"地図(?:は)?(?:なし|不要|いらない)|マップ(?:は)?(?:なし|不要|いらない)", original))
            ),
            date=_date_hint(original, reference_datetime),
            time=_time_hint(original),
            route_type=route_type,
            priority=priority,
            preferred_lines=_line_hints(original),
            avoid_stations=route_intent["avoid_station_texts"],
            via_stations=route_intent["via_station_texts"],
            avoid_lines=route_intent["avoid_line_texts"],
        )

    station = _station_query(original)
    if station:
        return _canonical("suggest_stations", original, query=station)

    has_origin = "から" in original or bool(re.search(r"[^\s]+発(?:で)?", original))
    has_destination = "まで" in original or bool(
        re.search(r"へ(?:行|向|乗)|[^、。\s]+に(?:着き|到着)", original)
    )
    if has_origin != has_destination:
        return _canonical(
            "clarification",
            original,
            missing="destination" if has_origin else "origin",
        )
    if re.search(
        r"終電|最終列車|始発|経路|行きたい|着きたい|帰りたい|乗換|最寄り駅|"
        r"発車案内|料金|早いやつ|安い経路|安く行|混まない|徒歩少な|歩きたく|"
        r"JR使わない|線は嫌|避けたい|地図で|駅IDがわからない|ここから|この辺から|"
        r"方面で|今から行け|どこに着けば|目的地だけ|何時に出れば",
        original,
    ):
        return _canonical("clarification", original, missing="origin|destination")
    return original


def normalize_japanese_prompt(
    text: str,
    reference_datetime: str | None = None,
    *,
    semantic_fallback: bool = False,
) -> str:
    """Normalize Japanese notation without doing intent understanding.

    ``semantic_fallback=True`` exists only for compatibility experiments and
    offline migration. Production leaves it disabled so FunctionGemma owns
    priority/avoid/via/mode/time-mode extraction.
    """
    if semantic_fallback:
        return _normalize_japanese_prompt_legacy(text, reference_datetime)
    return normalize_japanese_surface(text)


def normalized_japanese_hints(
    text: str,
    reference_datetime: str | None = None,
    *,
    semantic_fallback: bool = False,
) -> dict[str, Any]:
    rendered = normalize_japanese_prompt(
        text, reference_datetime, semantic_fallback=semantic_fallback
    )
    if not rendered.startswith("[normalized_ja]\n"):
        return {"intent": "unknown", "original": rendered}
    hints: dict[str, Any] = {}
    for line in rendered.splitlines()[1:]:
        key, separator, value = line.partition("=")
        if separator:
            hints[key] = value
    return hints


def bind_normalized_tool_call(
    call: ToolCall,
    text: str,
    reference_datetime: str | None = None,
    route_stage: int = 0,
    history: list[dict[str, Any]] | None = None,
    *,
    semantic_fallback: bool = False,
) -> ToolCall | None:
    """Bind high-confidence normalized slots; return None for safe clarification."""
    hints = normalized_japanese_hints(
        text, reference_datetime, semantic_fallback=semantic_fallback
    )
    intent = hints.get("intent")
    if intent in {"unknown", None}:
        return repair_tool_call_values(call, text, reference_datetime)
    if intent == "clarification":
        return None

    direct_tools = {
        "suggest_stations",
        "suggest_places",
        "reverse_geocode",
        "station_departures",
        "get_station",
        "list_feeds",
    }
    name = str(intent) if intent in direct_tools else call.name
    arguments = dict(call.arguments)

    if intent == "suggest_stations":
        arguments = {"q": hints["query"], "limit": 5}
    elif intent == "suggest_places":
        arguments = {"q": hints["query"], "limit": 10}
    elif intent == "reverse_geocode":
        arguments = {
            "lat": float(hints["lat"]),
            "lon": float(hints["lon"]),
            "limit": 3,
            "radiusMeters": int(hints.get("radius_meters", 200)),
        }
    elif intent == "station_departures":
        arguments = {"id": hints["station_id"]}
        if hints.get("limit"):
            arguments["limit"] = int(hints["limit"])
        for key in ("date", "time"):
            if hints.get(key):
                arguments[key] = hints[key]
    elif intent == "get_station":
        arguments = {"id": hints["station_id"]}
    elif intent == "list_feeds":
        arguments = {}
    elif intent == "route":
        via_stations = [
            value for value in str(hints.get("via_stations") or "").split("|") if value
        ]
        resolution_queries = [hints["origin"], *via_stations, hints["destination"]]
        if route_stage < len(resolution_queries):
            name = "suggest_stations"
            arguments = {"q": resolution_queries[route_stage], "limit": 5}
        else:
            wants_map = hints.get("map") == "true"
            name = "plan_route_map" if wants_map else "plan_journey"
            arguments = {}
            resolved: list[tuple[str, str]] = []
            for message in history or []:
                if message.get("role") != "tool":
                    continue
                content = message.get("content") or {}
                response = content.get("response") if isinstance(content, dict) else None
                if not isinstance(response, dict):
                    continue
                candidates = response.get("stations") or response.get("places") or []
                if candidates and isinstance(candidates[0], dict):
                    candidate = candidates[0]
                    if candidate.get("id"):
                        resolved.append((str(candidate["id"]), str(candidate.get("name") or "")))
            if len(resolved) >= 2:
                arguments.update(
                    {
                        "from": resolved[0][0],
                        "to": resolved[-1][0],
                        "fromLabel": resolved[0][1],
                        "toLabel": resolved[-1][1],
                    }
                )
                if len(resolved) > 2:
                    arguments["via"] = [item[0] for item in resolved[1:-1]]
                    arguments["viaLabel"] = [item[1] for item in resolved[1:-1]]
            if name == "plan_route_map":
                strategies = {
                    "fastest": "fastest",
                    "lowest_fare": "lowestFare",
                    "fewest_transfers": "fewestTransfers",
                    "shortest_walk": "shortestWalk",
                }
                arguments["strategy"] = strategies.get(hints.get("priority"), "balanced")
            else:
                arguments.pop("strategy", None)
            if hints.get("date"):
                arguments["date"] = hints["date"]
            if hints.get("time"):
                arguments["time"] = hints["time"]
            if hints.get("route_type"):
                arguments["type"] = hints["route_type"]
    return ToolCall(name, arguments)


def constrain_normalized_tool_call(
    call: ToolCall | None,
    text: str,
    reference_datetime: str | None = None,
    route_stage: int = 0,
    history: list[dict[str, Any]] | None = None,
    *,
    semantic_fallback: bool = False,
) -> tuple[bool, ToolCall | None]:
    """Return a schema-oriented call for recognized intent, or preserve unknown output.

    The boolean says whether deterministic normalization recognized the request. A
    recognized clarification intentionally returns ``None`` and must not contact MCP.
    """
    hints = normalized_japanese_hints(
        text, reference_datetime, semantic_fallback=semantic_fallback
    )
    intent = hints.get("intent")
    if intent in {None, "unknown"}:
        if call is not None:
            return False, repair_tool_call_values(call, text, reference_datetime)
        return False, call
    if intent == "clarification":
        return True, None
    seed_name = str(intent)
    if intent == "route":
        via_count = len(
            [value for value in str(hints.get("via_stations") or "").split("|") if value]
        )
        required_resolutions = 2 + via_count
        seed_name = (
            "suggest_stations"
            if route_stage < required_resolutions
            else ("plan_route_map" if hints.get("map") == "true" else "plan_journey")
        )
    seed = call or ToolCall(seed_name, {})
    return True, bind_normalized_tool_call(
        seed,
        text,
        reference_datetime,
        route_stage,
        history,
        semantic_fallback=semantic_fallback,
    )


def normalize_user_messages(
    messages: list[dict[str, Any]],
    reference_datetime: str | None = None,
    *,
    semantic_fallback: bool = False,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "user" and isinstance(copied.get("content"), str):
            copied["content"] = normalize_japanese_prompt(
                copied["content"],
                reference_datetime,
                semantic_fallback=semantic_fallback,
            )
        normalized.append(copied)
    return normalized


__all__ = [
    "bind_normalized_tool_call",
    "constrain_normalized_tool_call",
    "normalize_japanese_prompt",
    "normalize_japanese_surface",
    "normalized_japanese_hints",
    "normalize_user_messages",
    "repair_tool_call_values",
]
