#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transit_functiongemma.line_operator_rules import JR_LINES, TOEI_SUBWAY_LINES, TOKYO_METRO_LINES


_CLOCK_PATTERN = re.compile(r"(?<!\d)(-?\d{1,3}):([0-5]\d)(?!\d)")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not row.get("id") or not row.get("prompt"):
                raise ValueError(f"{path}:{line_number}: id and prompt are required")
            rows.append(row)
    return rows


def post_query(url: str, prompt: str, request_id: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"prompt": prompt}, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"ok": False, "error": f"http_{exc.code}"}
        payload["http_status"] = exc.code
        return payload


def find_behavior_event(directory: Path, request_id: str) -> dict[str, Any] | None:
    for _ in range(20):
        for path in sorted(directory.glob("*.jsonl"), reverse=True):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                if request_id not in line:
                    continue
                # The API appends behavior events while this evaluator reads the
                # same JSONL file.  A snapshot can therefore end with a partial
                # line; ignore it and retry instead of aborting the whole suite.
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("request_id") == request_id:
                    return event
        time.sleep(0.05)
    return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999))
    return round(ordered[index], 2)


def normalize_clock(value: Any) -> str | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    hour, minute, *_ = value.split(":")
    try:
        return f"{int(hour):02d}:{int(minute):02d}"
    except ValueError:
        return None


def displayed_time_satisfies(answer: str, mode: str, requested: str) -> bool:
    requested_clock = normalize_clock(requested)
    if requested_clock is None:
        return False
    requested_hour, requested_minute = (int(part) for part in requested_clock.split(":"))
    requested_total = requested_hour * 60 + requested_minute
    displayed: list[int] = []
    for match in _CLOCK_PATTERN.finditer(answer):
        hour_text = match.group(1)
        hour = abs(int(hour_text))
        minute = int(match.group(2))
        total = hour * 60 + minute
        displayed.append(-total if hour_text.startswith("-") else total)
    if not displayed:
        return False
    if mode == "arrival":
        return max(displayed) <= requested_total
    if mode == "departure":
        return min(displayed) >= requested_total
    return True


def evaluate_row(
    row: dict[str, Any], response: dict[str, Any], event: dict[str, Any] | None
) -> dict[str, Any]:
    trace = (event or {}).get("trace") or {}
    route_processing = trace.get("route_processing") or {}
    mcp_calls = trace.get("mcp_calls") or []
    route_calls = [
        call
        for call in mcp_calls
        if call.get("tool") in {"plan_journey", "plan_route_map"}
    ]
    final_call = route_calls[-1] if route_calls else None
    arguments = (final_call or {}).get("arguments") or {}
    effective_mcp_type = (
        arguments.get("type") or ("departure" if final_call is not None else None)
    )
    answer = str(response.get("answer") or "")
    model_intent = trace.get("model_route_intent") or {}
    checks: dict[str, bool] = {
        "http_ok": response.get("ok") is True,
        "behavior_logged": event is not None,
    }

    if row.get("expected_no_call"):
        checks["no_mcp_call"] = not mcp_calls
        checks["no_call_trace"] = trace.get("no_call") is True
        checks["answer_kind"] = response.get("kind") == "answer"
    else:
        checks["answer_kind"] = response.get("kind") == "answer"
        checks["final_tool"] = bool(
            final_call and final_call.get("tool") == row.get("expected_tool", "plan_journey")
        )

    if row.get("expected_mcp_type"):
        checks["mcp_type"] = effective_mcp_type == row["expected_mcp_type"]
    if row.get("expected_mcp_time"):
        checks["mcp_time"] = normalize_clock(arguments.get("time")) == normalize_clock(
            row["expected_mcp_time"]
        )
        if row.get("expected_mcp_type") in {"arrival", "departure"}:
            checks["displayed_time_constraint"] = displayed_time_satisfies(
                answer,
                row["expected_mcp_type"],
                row["expected_mcp_time"],
            )
    if row.get("expected_mcp_date"):
        checks["mcp_date"] = arguments.get("date") == row["expected_mcp_date"]
    if row.get("expected_model_intent") is not None:
        checks["model_intent_exact"] = model_intent == row["expected_model_intent"]
    if row.get("expected_priority"):
        priority = row["expected_priority"]
        checks["priority_extracted"] = model_intent.get("priority") == priority
        ranking = route_processing.get("ranking") or {}
        expected_message = {
            "few_transfers": "乗換が少ない順に並べました",
            "less_walk": "徒歩が少ない順に並べました",
            "cheap": "安さ順",
            "fast": "所要時間が短い順に並べました",
        }.get(priority, "")
        checks["priority_applied"] = ranking.get("priority") == priority and (
            ranking.get("applied") is True
            or priority == "cheap" and "保証できません" in str(ranking.get("message") or "")
        )
        checks["priority_reported"] = expected_message in answer
    if row.get("expected_avoid_station"):
        station = row["expected_avoid_station"]
        checks["avoid_extracted"] = station in model_intent.get("avoid_station_texts", [])
        checks["avoid_satisfied"] = f"{station}駅を通りません" in answer
    if row.get("expected_via_station"):
        station = row["expected_via_station"]
        checks["via_extracted"] = station in model_intent.get("via_station_texts", [])
        checks["via_satisfied"] = f"{station}を経由する候補です" in answer
    if row.get("expected_avoid_line"):
        line = row["expected_avoid_line"]
        checks["avoid_line_extracted"] = line in model_intent.get("avoid_line_texts", [])
        checks["avoid_line_satisfied"] = f"{line}を使わない候補だけを表示します" in answer
    if row.get("expected_mode_constraint") == "subway":
        if "JR" in model_intent.get("avoid_operator_groups", []):
            checks["mode_constraint_extracted"] = True
            checks["mode_constraint_applied"] = "JRを使わない候補" in answer
        else:
            checks["mode_constraint_extracted"] = (
                "subway" in model_intent.get("allowed_operator_groups", [])
            )
            checks["mode_constraint_applied"] = "地下鉄だけの候補" in answer
        checks["mode_constraint_strict"] = not any(line in answer for line in JR_LINES)
    if row.get("expected_mode_constraint") == "tokyo_metro":
        checks["mode_constraint_extracted"] = (
            "tokyo_metro" in model_intent.get("allowed_operator_groups", [])
        )
        checks["mode_constraint_applied"] = "東京メトロだけの候補" in answer
        checks["mode_constraint_strict"] = not any(
            line in answer for line in (*JR_LINES, *TOEI_SUBWAY_LINES)
        )
    if row.get("expected_mode_constraint") == "toei_subway":
        checks["mode_constraint_extracted"] = (
            "toei_subway" in model_intent.get("allowed_operator_groups", [])
        )
        checks["mode_constraint_applied"] = "都営地下鉄だけの候補" in answer
        checks["mode_constraint_strict"] = not any(
            line in answer for line in (*JR_LINES, *TOKYO_METRO_LINES)
        )
    if row.get("expected_mode_constraint") == "rail_only":
        checks["mode_constraint_extracted"] = "bus" in model_intent.get("avoid_modes", [])
        checks["mode_constraint_applied"] = "バスを使わない候補" in answer
    for text in row.get("expected_answer_contains") or []:
        checks[f"answer_contains:{text}"] = text in answer
    expected_any = row.get("expected_any_answer_contains") or []
    if expected_any:
        checks["answer_contains_any"] = any(text in answer for text in expected_any)
    for text in row.get("forbidden_answer_contains") or []:
        checks[f"answer_excludes:{text}"] = text not in answer

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "id": row["id"],
        "category": row.get("category", "uncategorized"),
        "prompt": row["prompt"],
        "success": not failed,
        "checks": checks,
        "failed_checks": failed,
        "response": {
            key: response.get(key)
            for key in ("ok", "kind", "error", "answer", "elapsed_ms", "request_id")
            if response.get(key) is not None
        },
        "model_route_intent": model_intent,
        "final_mcp_call": final_call,
        "effective_mcp_type": effective_mcp_type,
        "route_processing": route_processing,
        "timings": trace.get("timings") or {},
    }


def markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Tokyo operational route evaluation",
        "",
        "実Web APIからFunctionGemmaと実Transit MCPへ送った実運用表現テスト。",
        "",
        "## Metrics",
        "",
        f"- Total: {metrics['total']}",
        f"- Passed: {metrics['passed']}",
        f"- Success rate: {metrics['success_rate']}",
        f"- p50 latency: {metrics['latency_p50_ms']} ms",
        f"- p95 latency: {metrics['latency_p95_ms']} ms",
        "",
        "## By category",
        "",
        "| Category | Passed | Total | Rate |",
        "|---|---:|---:|---:|",
    ]
    for category, values in sorted(report["by_category"].items()):
        lines.append(
            f"| `{category}` | {values['passed']} | {values['total']} | {values['success_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Scenarios",
            "",
            "| ID | Category | Prompt | Result | MCP type | MCP time | Latency ms | Failed checks |",
            "|---|---|---|---:|---|---|---:|---|",
        ]
    )
    for item in report["results"]:
        arguments = (item.get("final_mcp_call") or {}).get("arguments") or {}
        lines.append(
            f"| `{item['id']}` | `{item['category']}` | {item['prompt']} | "
            f"{'PASS' if item['success'] else 'FAIL'} | `{item.get('effective_mcp_type') or ''}` | "
            f"`{arguments.get('time', '')}` | {item['response'].get('elapsed_ms', '')} | "
            f"{', '.join(item['failed_checks'])} |"
        )
    failures = [item for item in report["results"] if not item["success"]]
    lines.extend(["", "## Failure details", ""])
    if not failures:
        lines.append("None.")
    for item in failures:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- Prompt: {item['prompt']}",
                f"- Failed: {', '.join(item['failed_checks'])}",
                f"- Model route intent: `{json.dumps(item['model_route_intent'], ensure_ascii=False)}`",
                f"- Final MCP call: `{json.dumps(item['final_mcp_call'], ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run operational Japanese route samples through the live Web API."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/eval/operational_tokyo_routes.jsonl")
    )
    parser.add_argument("--url", default="http://127.0.0.1:8091/query")
    parser.add_argument(
        "--behavior-log-dir", type=Path, default=Path("artifacts/behavior_logs")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/operational_tokyo_routes.json")
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("artifacts/operational_tokyo_routes.md"),
    )
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument(
        "--ids",
        help="Optional comma-separated scenario IDs for a focused rerun.",
    )
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    if args.ids:
        selected = {value.strip() for value in args.ids.split(",") if value.strip()}
        rows = [row for row in rows if row["id"] in selected]
        missing = selected - {row["id"] for row in rows}
        if missing:
            raise ValueError(f"unknown scenario IDs: {sorted(missing)}")
    results: list[dict[str, Any]] = []
    run_id = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d%H%M%S")
    for index, row in enumerate(rows, 1):
        request_id = f"{row['id']}-{run_id}"
        started = time.monotonic()
        try:
            response = post_query(args.url, row["prompt"], request_id, args.timeout)
        except Exception as exc:  # preserve the row and continue the operational suite
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        response.setdefault("elapsed_ms", round((time.monotonic() - started) * 1000, 2))
        event = find_behavior_event(args.behavior_log_dir, request_id)
        result = evaluate_row(row, response, event)
        results.append(result)
        print(
            f"[{index:02d}/{len(rows):02d}] {row['id']} "
            f"{'PASS' if result['success'] else 'FAIL'} "
            f"{','.join(result['failed_checks'])}",
            flush=True,
        )
        if args.delay:
            time.sleep(args.delay)

    latencies = [
        float(item["response"]["elapsed_ms"])
        for item in results
        if isinstance(item["response"].get("elapsed_ms"), (int, float))
    ]
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in results:
        category_counts[item["category"]]["total"] += 1
        category_counts[item["category"]]["passed"] += int(item["success"])
    by_category = {
        category: {
            "total": counts["total"],
            "passed": counts["passed"],
            "success_rate": round(counts["passed"] / counts["total"], 4),
        }
        for category, counts in category_counts.items()
    }
    metrics = {
        "total": len(results),
        "passed": sum(item["success"] for item in results),
        "success_rate": round(sum(item["success"] for item in results) / len(results), 4),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_ms": percentile(latencies, 0.95),
    }
    report = {
        "run_id": run_id,
        "dataset": str(args.dataset),
        "url": args.url,
        "metrics": metrics,
        "by_category": by_category,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
