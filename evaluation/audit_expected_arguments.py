#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH
from transit_functiongemma.local_tools import LOCAL_TOOLS
from transit_functiongemma.schemas import load_mcp_tools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "artifacts" / "eval_label_audit.json"
DEFAULT_OUTPUT_MARKDOWN = PROJECT_ROOT / "artifacts" / "EVAL_LABEL_AUDIT.md"
DEFAULT_MIGRATION_REPORT = PROJECT_ROOT / "artifacts" / "eval_label_migration.json"
AUDITED_FIELDS = ("expected_arguments", "expected_normalized")
Q_TOOLS = ("suggest_stations", "suggest_places")
DELTA_METRICS = (
    ("semantic_success_rate", "Semantic success"),
    ("expected_arguments_match_rate", "Expected arguments"),
    ("expected_arguments_exact_match_rate", "Expected arguments exact"),
    ("datetime_normalization_success_rate", "Datetime"),
    ("tool_name_accuracy", "Tool name"),
    ("no_call_when_missing_info_rate", "No-call"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        row["_audit_file"] = str(path)
        row["_audit_line"] = line_number
        rows.append(row)
    return rows


def load_tool_schemas(schema_path: Path) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    tools: list[dict[str, Any]] = []
    sources: list[str] = []
    warnings: list[str] = []
    if schema_path.exists():
        tools.extend(load_mcp_tools(schema_path))
        sources.append(str(schema_path))
    else:
        warnings.append(
            f"MCP schema not found at {schema_path}; using evaluator-local tools only."
        )
        local_schema = PROJECT_ROOT / "tools" / "local_tools_schema.json"
        if local_schema.exists():
            sources.append(str(local_schema))
    tools.extend(LOCAL_TOOLS)
    sources.append("transit_functiongemma.local_tools.LOCAL_TOOLS")
    return {str(tool["name"]): tool for tool in tools}, sources, warnings


def _kind_for_validator(validator: str) -> str:
    if validator == "type":
        return "type"
    if validator == "enum":
        return "enum"
    if validator == "pattern":
        return "pattern"
    if validator in {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "multipleOf",
    }:
        return "range"
    return "schema"


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_audit_")}


def audit_schema_violations(
    rows: list[dict[str, Any]], tools: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        tool_name = row.get("expected_tool")
        for field in AUDITED_FIELDS:
            arguments = row.get(field)
            if not isinstance(arguments, dict):
                continue
            if not isinstance(tool_name, str) or tool_name not in tools:
                violations.append(
                    {
                        "file": row["_audit_file"],
                        "line": row["_audit_line"],
                        "id": row.get("id"),
                        "tool": tool_name,
                        "field": field,
                        "key": "__tool__",
                        "kind": "unknown_tool",
                        "schema_rule": "tool",
                        "message": f"expected_tool is not in the loaded schema: {tool_name}",
                        "value": tool_name,
                        "user": row.get("user"),
                    }
                )
                continue
            properties = (
                tools[tool_name].get("inputSchema", {}).get("properties", {})
                if isinstance(tools[tool_name].get("inputSchema"), dict)
                else {}
            )
            for key, value in arguments.items():
                if key not in properties:
                    violations.append(
                        {
                            "file": row["_audit_file"],
                            "line": row["_audit_line"],
                            "id": row.get("id"),
                            "tool": tool_name,
                            "field": field,
                            "key": key,
                            "kind": "unknown_key",
                            "schema_rule": "properties",
                            "message": f"{tool_name}.{key} is not present in inputSchema",
                            "value": value,
                            "user": row.get("user"),
                        }
                    )
                    continue
                validator = Draft202012Validator(properties[key])
                for error in sorted(validator.iter_errors(value), key=str):
                    violations.append(
                        {
                            "file": row["_audit_file"],
                            "line": row["_audit_line"],
                            "id": row.get("id"),
                            "tool": tool_name,
                            "field": field,
                            "key": key,
                            "kind": _kind_for_validator(str(error.validator)),
                            "schema_rule": str(error.validator),
                            "message": error.message,
                            "value": value,
                            "user": row.get("user"),
                        }
                    )
    return violations


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def _strip_station_suffix(text: str) -> str:
    return re.sub(r"駅$", "", _norm(text))


def audit_q_labels(
    rows: list[dict[str, Any]], tools: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    suffix_by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    limit_by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    user_station_suffix_by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    default_limits = {
        tool: (
            tools.get(tool, {})
            .get("inputSchema", {})
            .get("properties", {})
            .get("limit", {})
            .get("default")
        )
        for tool in Q_TOOLS
    }

    for row in rows:
        tool_name = row.get("expected_tool")
        arguments = row.get("expected_arguments")
        if tool_name not in Q_TOOLS or not isinstance(arguments, dict) or "q" not in arguments:
            continue
        q = _norm(arguments["q"])
        user = _norm(row.get("user") or "")
        has_station_suffix = q.endswith("駅")
        base = _strip_station_suffix(q)
        user_has_station_suffix = bool(base and f"{base}駅" in user)
        if "limit" in arguments:
            limit_value = arguments["limit"]
            limit_label = str(limit_value)
        else:
            limit_value = None
            limit_label = f"omitted(default={default_limits.get(tool_name)})"
        suffix_label = "with_eki" if has_station_suffix else "without_eki"
        suffix_by_tool[str(tool_name)][suffix_label] += 1
        limit_by_tool[str(tool_name)][limit_label] += 1
        if user_has_station_suffix:
            user_station_suffix_by_tool[str(tool_name)][suffix_label] += 1
        records.append(
            {
                "file": row["_audit_file"],
                "line": row["_audit_line"],
                "id": row.get("id"),
                "tool": tool_name,
                "user": row.get("user"),
                "q": arguments["q"],
                "q_has_station_suffix": has_station_suffix,
                "user_has_station_suffix_form": user_has_station_suffix,
                "limit": limit_value,
                "limit_bucket": limit_label,
            }
        )

    combined_user_station = Counter()
    for counter in user_station_suffix_by_tool.values():
        combined_user_station.update(counter)
    majority = "tie"
    if combined_user_station["with_eki"] > combined_user_station["without_eki"]:
        majority = "with_eki"
    elif combined_user_station["without_eki"] > combined_user_station["with_eki"]:
        majority = "without_eki"

    return {
        "total_q_rows": len(records),
        "default_limits": default_limits,
        "station_suffix_distribution_by_tool": {
            tool: dict(counter) for tool, counter in sorted(suffix_by_tool.items())
        },
        "limit_distribution_by_tool": {
            tool: dict(counter) for tool, counter in sorted(limit_by_tool.items())
        },
        "user_station_suffix_distribution_by_tool": {
            tool: dict(counter)
            for tool, counter in sorted(user_station_suffix_by_tool.items())
        },
        "user_station_suffix_distribution": dict(combined_user_station),
        "user_station_suffix_majority": majority,
        "records": records,
        "minority_user_station_suffix_records": [
            item
            for item in records
            if item["user_has_station_suffix_form"]
            and (
                (majority == "with_eki" and not item["q_has_station_suffix"])
                or (majority == "without_eki" and item["q_has_station_suffix"])
            )
        ],
    }


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _load_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_metric_delta(
    before_path: Path | None, after_path: Path | None
) -> list[dict[str, Any]]:
    before = _load_report(before_path)
    after = _load_report(after_path)
    if not before or not after:
        return []
    rows: list[dict[str, Any]] = []
    for key, label in DELTA_METRICS:
        before_value = before.get("metrics", {}).get(key)
        after_value = after.get("metrics", {}).get(key)
        delta = (
            round(after_value - before_value, 4)
            if isinstance(before_value, (int, float))
            and isinstance(after_value, (int, float))
            else None
        )
        rows.append(
            {
                "metric": key,
                "label": label,
                "before": before_value,
                "after": after_value,
                "delta": delta,
            }
        )
    return rows


def _format_rate(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return str(value)


def _format_delta(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        sign = "+" if value >= 0 else ""
        return f"{sign}{value * 100:.2f} pp"
    return str(value)


def build_audit_report(
    eval_dir: Path = DEFAULT_EVAL_DIR,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    before_eval_report: Path | None = None,
    after_eval_report: Path | None = None,
    migration_report_path: Path | None = DEFAULT_MIGRATION_REPORT,
) -> dict[str, Any]:
    tools, schema_sources, warnings = load_tool_schemas(schema_path)
    paths = sorted(path for path in eval_dir.glob("*.jsonl") if path.is_file())
    rows = [row for path in paths for row in read_jsonl(path)]
    violations = audit_schema_violations(rows, tools)
    expected_argument_violations = [
        item for item in violations if item["field"] == "expected_arguments"
    ]
    expected_normalized_violations = [
        item for item in violations if item["field"] == "expected_normalized"
    ]
    q_audit = audit_q_labels(rows, tools)
    migration_report = _load_report(migration_report_path)
    report = {
        "schema_sources": schema_sources,
        "warnings": warnings,
        "eval_dir": str(eval_dir),
        "dataset_files": [str(path) for path in paths],
        "dataset_file_count": len(paths),
        "row_count": len(rows),
        "expected_arguments_case_count": sum(
            1 for row in rows if isinstance(row.get("expected_arguments"), dict)
        ),
        "expected_normalized_case_count": sum(
            1 for row in rows if isinstance(row.get("expected_normalized"), dict)
        ),
        "violations": violations,
        "violation_counts": {
            "total": len(violations),
            "expected_arguments": len(expected_argument_violations),
            "expected_normalized": len(expected_normalized_violations),
            "by_field": _counter_dict(Counter(item["field"] for item in violations)),
            "by_kind": _counter_dict(Counter(item["kind"] for item in violations)),
            "by_tool": _counter_dict(Counter(item["tool"] for item in violations)),
            "by_file": _counter_dict(
                Counter(Path(item["file"]).name for item in violations)
            ),
        },
        "q_label_audit": q_audit,
        "metric_delta": build_metric_delta(before_eval_report, after_eval_report),
        "migration": migration_report,
    }
    return report


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Eval label audit", ""]
    if report.get("metric_delta"):
        lines.extend(["## Re-audit metric delta", ""])
        lines.extend(
            _markdown_table(
                ["Metric", "Before r7 bound", "After re-audit", "Delta"],
                [
                    [
                        item["label"],
                        _format_rate(item["before"]),
                        _format_rate(item["after"]),
                        _format_delta(item["delta"]),
                    ]
                    for item in report["metric_delta"]
                ],
            )
        )
        lines.append("")
    else:
        lines.extend(
            [
                "## Re-audit metric delta",
                "",
                "Re-audit scoring has not been attached to this report yet.",
                "",
            ]
        )

    lines.extend(
        [
            "## Scope",
            "",
            f"- Dataset files scanned: {report['dataset_file_count']}",
            f"- Rows scanned: {report['row_count']}",
            f"- `expected_arguments` cases: {report['expected_arguments_case_count']}",
            f"- `expected_normalized` cases: {report['expected_normalized_case_count']}",
            f"- Schema sources: {', '.join(report['schema_sources'])}",
        ]
    )
    for warning in report.get("warnings") or []:
        lines.append(f"- Warning: {warning}")
    lines.append("")

    counts = report["violation_counts"]
    lines.extend(
        [
            "## Schema violations",
            "",
            f"- Total violations: {counts['total']}",
            f"- `expected_arguments`: {counts['expected_arguments']}",
            f"- `expected_normalized`: {counts['expected_normalized']}",
            "",
        ]
    )
    if counts["total"]:
        lines.extend(
            _markdown_table(
                ["File", "Line", "ID", "Field", "Tool", "Key", "Kind", "Value"],
                [
                    [
                        Path(item["file"]).name,
                        item["line"],
                        f"`{item['id']}`",
                        f"`{item['field']}`",
                        f"`{item['tool']}`",
                        f"`{item['key']}`",
                        item["kind"],
                        f"`{json.dumps(item['value'], ensure_ascii=False)}`",
                    ]
                    for item in report["violations"][:100]
                ],
            )
        )
        if counts["total"] > 100:
            lines.append(f"\nOnly first 100 violations shown; JSON contains all {counts['total']}.")
    else:
        lines.append("No schema violations remain in the audited fields.")
    lines.append("")

    q_audit = report["q_label_audit"]
    majority = q_audit["user_station_suffix_majority"]
    majority_text = {
        "with_eki": "q keeps the trailing `駅`",
        "without_eki": "q drops the trailing `駅`",
        "tie": "no majority",
    }[majority]
    lines.extend(
        [
            "## q label convention audit",
            "",
            f"- `suggest_stations` / `suggest_places` q rows: {q_audit['total_q_rows']}",
            f"- Rows where the user used an explicit `○○駅` form: "
            f"{sum(q_audit['user_station_suffix_distribution'].values())}",
            f"- Existing majority for explicit `○○駅` rows: {majority_text}",
            "- Proposed convention: keep q faithful to the user's surface form. "
            "When the user writes `○○駅` and the station-name span is clear, follow "
            "the existing majority and keep `駅` in q.",
            "",
            "### Station suffix distribution",
            "",
        ]
    )
    suffix_rows = []
    for tool, values in q_audit["station_suffix_distribution_by_tool"].items():
        suffix_rows.append(
            [
                f"`{tool}`",
                values.get("with_eki", 0),
                values.get("without_eki", 0),
            ]
        )
    lines.extend(_markdown_table(["Tool", "q with `駅`", "q without `駅`"], suffix_rows))
    lines.extend(["", "### Limit distribution", ""])
    limit_rows = []
    for tool, values in q_audit["limit_distribution_by_tool"].items():
        for limit, count in values.items():
            limit_rows.append([f"`{tool}`", f"`{limit}`", count])
    lines.extend(_markdown_table(["Tool", "Limit label", "Count"], limit_rows))
    if q_audit["minority_user_station_suffix_records"]:
        lines.extend(["", "### Convention minority rows", ""])
        lines.extend(
            _markdown_table(
                ["File", "Line", "ID", "q", "User"],
                [
                    [
                        Path(item["file"]).name,
                        item["line"],
                        f"`{item['id']}`",
                        f"`{item['q']}`",
                        item.get("user") or "",
                    ]
                    for item in q_audit["minority_user_station_suffix_records"][:50]
                ],
            )
        )
    lines.append("")

    migration = report.get("migration")
    lines.extend(["## Applied migration", ""])
    if migration:
        action_counts = Counter(item["action"] for item in migration.get("changes", []))
        lines.extend(
            [
                f"- Backup directory: `{migration.get('backup_dir')}`",
                f"- Changed files: {migration.get('changed_file_count', 0)}",
                f"- Changed rows: {migration.get('changed_row_count', 0)}",
                f"- Field-level changes: {len(migration.get('changes', []))}",
                f"- Needs human judgment: {len(migration.get('needs_human_judgment', []))}",
                "",
            ]
        )
        if action_counts:
            lines.extend(
                _markdown_table(
                    ["Action", "Count"],
                    [[f"`{action}`", count] for action, count in sorted(action_counts.items())],
                )
            )
            lines.append("")
        if migration.get("changes"):
            lines.extend(
                _markdown_table(
                    ["File", "Line", "ID", "Field", "Key", "Action", "Old", "New"],
                    [
                        [
                            Path(item["file"]).name,
                            item["line"],
                            f"`{item['id']}`",
                            f"`{item['field']}`",
                            f"`{item['key']}`",
                            item["action"],
                            f"`{json.dumps(item.get('old_value'), ensure_ascii=False)}`",
                            f"`{json.dumps(item.get('new_value'), ensure_ascii=False)}`",
                        ]
                        for item in migration["changes"][:100]
                    ],
                )
            )
            if len(migration["changes"]) > 100:
                lines.append(
                    f"\nOnly first 100 changes shown; JSON contains all {len(migration['changes'])}."
                )
        if migration.get("needs_human_judgment"):
            lines.extend(["", "### Needs human judgment", ""])
            lines.extend(
                _markdown_table(
                    ["File", "Line", "ID", "Reason"],
                    [
                        [
                            Path(item["file"]).name,
                            item["line"],
                            f"`{item['id']}`",
                            item["reason"],
                        ]
                        for item in migration["needs_human_judgment"]
                    ],
                )
            )
    else:
        lines.append("No migration report was found.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: dict[str, Any], output_json: Path, output_markdown: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_markdown.write_text(markdown_report(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit eval expected_arguments against the saved tool schemas."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--before-eval-report", type=Path)
    parser.add_argument("--after-eval-report", type=Path)
    parser.add_argument("--migration-report", type=Path, default=DEFAULT_MIGRATION_REPORT)
    args = parser.parse_args()

    report = build_audit_report(
        args.eval_dir,
        args.schema,
        args.before_eval_report,
        args.after_eval_report,
        args.migration_report,
    )
    write_outputs(report, args.output_json, args.markdown_output)
    print(json.dumps(report["violation_counts"], ensure_ascii=False, indent=2))
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.markdown_output}")


if __name__ == "__main__":
    main()
