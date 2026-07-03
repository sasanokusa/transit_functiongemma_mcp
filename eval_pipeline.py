#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from answer_pipeline import USER_ERROR, run_pipeline
from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, MCP_ENDPOINT, MODEL_ID
from transit_functiongemma.mcp import MCPError, MCPTimeoutError


FIXED_SCENARIOS = [
    {
        "id": "demo-01",
        "prompt": "東京駅を検索して",
        "expected_tool": "suggest_stations",
        "expected_arguments": {"q": "東京"},
    },
    {
        "id": "demo-02",
        "prompt": "東京タワーを場所として探して",
        "expected_tool": "suggest_places",
        "expected_arguments": {"q": "東京タワー"},
    },
    {
        "id": "demo-03",
        "prompt": "町田から池袋まで、渋谷を避けて",
        "expected_tool": "plan_journey",
        "expected_answer_contains": "渋谷駅を通りません",
    },
    {"id": "demo-04", "prompt": "横浜から上野まで、乗換少なめ", "expected_tool": "plan_journey"},
    {
        "id": "demo-05",
        "prompt": "明日9時に品川に着きたい",
        "expected_no_call": True,
        "expected_answer_contains": "出発地が不足",
    },
    {
        "id": "demo-06",
        "prompt": "終電で新宿から大宮まで帰れる？",
        "expected_tool": "plan_journey",
        "expected_arguments": {"type": "last"},
    },
    {
        "id": "demo-07",
        "prompt": "東京駅まで行きたい",
        "expected_no_call": True,
        "expected_answer_contains": "出発地が不足",
    },
]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999)))
    return round(ordered[index], 2)


def renderer_uses_present_facts(trace: dict[str, Any]) -> bool:
    answer = str(trace.get("rendered_answer") or "")
    normalized = json.dumps(trace.get("normalized_result") or {}, ensure_ascii=False)
    facts = re.findall(r"\b\d{1,2}:\d{2}\b|\b\d[\d,]*円\b", answer)
    return all(fact in normalized for fact in facts)


def contains_expected_arguments(
    actual: dict[str, Any], expected: dict[str, Any] | None
) -> bool:
    return not expected or all(actual.get(key) == value for key, value in expected.items())


def markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# E2E pipeline evaluation",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| `{key}` | {value} |" for key, value in metrics.items())
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Route candidates requested per scenario: {report['configuration']['max_routes']}",
        ]
    )
    lines.extend(
        [
            "",
            "## Scenarios",
            "",
            "| ID | Prompt | Success | Final tool | MCP calls | Latency ms | Error |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    for row in report["scenarios"]:
        lines.append(
            f"| `{row['id']}` | {row['prompt']} | {row['success']} | "
            f"`{row.get('final_tool') or 'no-call'}` | {row['mcp_call_count']} | "
            f"{row['latency_ms']} | {row.get('error') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate router -> real MCP -> renderer E2E.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--mcp-url", default=MCP_ENDPOINT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output", type=Path, default=Path("artifacts/e2e_eval_report.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("artifacts/e2e_eval_report.md"))
    parser.add_argument("--failures-output", type=Path, default=Path("artifacts/failures_e2e.jsonl"))
    parser.add_argument("--latency-output", type=Path, default=Path("artifacts/latency_report.json"))
    parser.add_argument("--trace-dir", type=Path, default=Path("artifacts/e2e_traces"))
    parser.add_argument(
        "--max-routes",
        type=int,
        default=1,
        choices=range(1, 7),
        help="Route candidates requested from MCP; use 1 for the latency gate.",
    )
    args = parser.parse_args()

    from infer import ToolRouter

    router = ToolRouter(
        args.base_model,
        args.adapter,
        args.schema,
        "baked",
        False,
        True,
    )
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    mcp_latencies: list[float] = []
    status_counts: Counter[str] = Counter()
    timeout_count = 0
    retry_calls = 0
    retry_success = 0
    args.trace_dir.mkdir(parents=True, exist_ok=True)

    for scenario in FIXED_SCENARIOS:
        trace: dict[str, Any] = {"scenario_id": scenario["id"]}
        started = time.monotonic()
        error = None
        try:
            answer = run_pipeline(
                scenario["prompt"],
                adapter=args.adapter,
                mcp_url=args.mcp_url,
                schema_path=args.schema,
                router_instance=router,
                normalize_ja=True,
                save_raw=args.trace_dir / "raw",
                save_normalized=args.trace_dir / "normalized",
                trace=trace,
                max_routes=args.max_routes,
            )
        except MCPTimeoutError as exc:
            timeout_count += 1
            status_counts["timeout"] += 1
            answer, error = "", str(exc)
        except MCPError as exc:
            status_counts["mcp_error"] += 1
            answer, error = "", str(exc)
        except Exception as exc:  # report the scenario instead of aborting the suite
            status_counts["pipeline_error"] += 1
            answer, error = "", f"{type(exc).__name__}: {exc}"
        elapsed = round((time.monotonic() - started) * 1000, 2)
        latencies.append(elapsed)
        calls = trace.get("mcp_calls") or []
        for call in calls:
            status_counts[str(call.get("status") or "unknown")] += 1
            if isinstance(call.get("latency_ms"), (int, float)):
                mcp_latencies.append(float(call["latency_ms"]))
            if int(call.get("attempts") or 0) > 1:
                retry_calls += 1
                if call.get("status") == "ok":
                    retry_success += 1
        final_tool = calls[-1]["tool"] if calls else None
        if scenario.get("expected_no_call"):
            success = not calls and bool(answer) and error is None
        else:
            success = (
                final_tool == scenario.get("expected_tool")
                and bool(answer)
                and answer != USER_ERROR
                and error is None
            )
        if success and scenario.get("expected_arguments"):
            success = contains_expected_arguments(
                calls[-1].get("arguments") or {}, scenario["expected_arguments"]
            )
        if success and scenario.get("expected_answer_contains"):
            success = scenario["expected_answer_contains"] in answer
        source_only = renderer_uses_present_facts(trace)
        success = success and source_only
        trace.update(
            {
                "answer": answer,
                "latency_ms": elapsed,
                "success": success,
                "error": error,
                "renderer_source_only": source_only,
            }
        )
        (args.trace_dir / f"{scenario['id']}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                **scenario,
                "success": success,
                "final_tool": final_tool,
                "mcp_call_count": len(calls),
                "latency_ms": elapsed,
                "error": error,
                "renderer_source_only": source_only,
                "answer": answer,
            }
        )

    metrics = {
        "e2e_success_rate": round(sum(row["success"] for row in rows) / len(rows), 4),
        "timeout_rate": round(timeout_count / len(rows), 4),
        "retry_success_rate": round(retry_success / retry_calls, 4) if retry_calls else None,
        "no_call_without_mcp_rate": round(
            sum(row["success"] for row in rows if row.get("expected_no_call"))
            / sum(bool(row.get("expected_no_call")) for row in rows),
            4,
        ),
        "renderer_source_only_rate": round(
            sum(row["renderer_source_only"] for row in rows) / len(rows), 4
        ),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": percentile(latencies, 0.95),
        "mcp_latency_p50_ms": round(statistics.median(mcp_latencies), 2) if mcp_latencies else None,
        "mcp_latency_p95_ms": percentile(mcp_latencies, 0.95),
    }
    report = {
        "configuration": {"max_routes": args.max_routes},
        "metrics": metrics,
        "mcp_status": dict(status_counts),
        "scenarios": rows,
    }
    latency = {
        "request_ms": latencies,
        "mcp_call_ms": mcp_latencies,
        "p50_ms": metrics["latency_p50_ms"],
        "p95_ms": metrics["latency_p95_ms"],
        "mcp_p50_ms": metrics["mcp_latency_p50_ms"],
        "mcp_p95_ms": metrics["mcp_latency_p95_ms"],
        "max_routes": args.max_routes,
    }
    for path in (args.output, args.markdown_output, args.failures_output, args.latency_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    args.latency_output.write_text(json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.failures_output.open("w", encoding="utf-8") as stream:
        for row in rows:
            if not row["success"]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
