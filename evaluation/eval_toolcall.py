#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, MODEL_ID
from transit_functiongemma.japanese import constrain_normalized_tool_call
from transit_functiongemma.route_intent import extract_route_intent, mcp_time_type
from transit_functiongemma.schemas import (
    CLARIFICATION_TOOL_NAME,
    load_mcp_tools,
    tool_map,
    tools_with_clarification,
)
from transit_functiongemma.toolcall import ToolCall, ToolCallParseError, parse_tool_calls
from transit_functiongemma.validation import ToolCallSchemaError, validate_tool_calls

NO_CALL = "no_tool_call"
PARSE_ERROR = "__parse_error__"
MULTIPLE_CALLS = "__multiple_calls__"
INTENT_SLOTS = (
    "origin_text",
    "destination_text",
    "avoid_station_texts",
    "via_station_texts",
    "avoid_line_texts",
    "priority",
    "time_mode",
    "date",
    "time",
    "graphical",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def ratio(value: int, total: int) -> float | None:
    return round(value / total, 4) if total else None


def source_text_for_row(row: dict[str, Any]) -> str:
    if isinstance(row.get("user"), str) and row["user"].strip():
        return row["user"]
    return next(
        (
            str(message.get("content"))
            for message in row.get("history") or []
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )


def _station_semantic(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"駅$", "", text)


def semantic_argument_equal(tool_name: str, key: str, actual: Any, expected: Any) -> bool:
    if tool_name == "suggest_stations" and key == "q":
        return _station_semantic(actual) == _station_semantic(expected)
    if key == "time" and isinstance(actual, str) and isinstance(expected, str):
        def clock(value: str) -> tuple[int, int, int] | None:
            match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", value)
            return (
                (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
                if match
                else None
            )
        return clock(actual) == clock(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) < 1e-8
    return actual == expected


def valid_clarification(call: ToolCall) -> bool:
    missing = call.arguments.get("missing")
    question = call.arguments.get("question")
    return (
        call.name == CLARIFICATION_TOOL_NAME
        and isinstance(missing, list)
        and bool(missing)
        and isinstance(question, str)
        and bool(question.strip())
    )


def intent_time_mode_from_call(call: ToolCall | None) -> str | None:
    if call is None:
        return None
    value = call.arguments.get("time_mode")
    if isinstance(value, str):
        return value
    route_type = call.arguments.get("type")
    if not isinstance(route_type, str):
        return None
    return {
        "departure": "departure_at",
        "arrival": "arrive_by",
        "first": "first_train",
        "last": "last_train",
    }.get(route_type)


def per_class_scores(
    actual_labels: list[str], predicted_labels: list[str], classes: list[str]
) -> dict[str, dict[str, int | float | None]]:
    scores: dict[str, dict[str, int | float | None]] = {}
    for label in classes:
        tp = sum(a == label and p == label for a, p in zip(actual_labels, predicted_labels))
        fp = sum(a != label and p == label for a, p in zip(actual_labels, predicted_labels))
        fn = sum(a == label and p != label for a, p in zip(actual_labels, predicted_labels))
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        scores[label] = {
            "support": sum(a == label for a in actual_labels),
            "predicted": sum(p == label for p in predicted_labels),
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return scores


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Tool-call evaluation", "", "## Metrics", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(
        f"| `{name}` | {value if value is not None else 'N/A'} |"
        for name, value in report["metrics"].items()
    )
    lines.extend(
        [
            "",
            "## Dataset observability",
            "",
            f"- Unsupported expected arguments excluded from semantic scoring: "
            f"{report['counts'].get('unsupported_expected_arguments', 0)}",
            f"- Unsupported expected normalized arguments excluded from datetime scoring: "
            f"{report['counts'].get('unsupported_expected_normalized', 0)}",
            f"- Expected arguments not observable in user/history excluded from semantic scoring: "
            f"{report['counts'].get('unobservable_expected_arguments', 0)}",
        ]
    )
    lines.extend(
        [
            "",
            "## Per-class metrics",
            "",
            "| Class | Support | Predicted | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["per_class"].items():
        lines.append(
            f"| `{name}` | {item['support']} | {item['predicted']} | "
            f"{item['precision']} | {item['recall']} | {item['f1']} |"
        )
    if report.get("intent_slots"):
        lines.extend(
            [
                "",
                "## Intent slots",
                "",
                "| Slot | Matched | Total | Match rate | F1 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, item in report["intent_slots"].items():
            lines.append(
                f"| `{name}` | {item['matched']} | {item['total']} | "
                f"{item['match_rate']} | {item['f1']} |"
            )
    if report.get("argument_slots"):
        lines.extend(
            [
                "",
                "## Model argument slots",
                "",
                "| Slot | Matched | Total | Match rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, item in report["argument_slots"].items():
            lines.append(
                f"| `{name}` | {item['matched']} | {item['total']} | "
                f"{item['match_rate']} |"
            )
    if report.get("by_category"):
        lines.extend(
            [
                "",
                "## By category",
                "",
                "| Category | Passed | Total | Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, item in report["by_category"].items():
            lines.append(
                f"| `{name}` | {item['passed']} | {item['total']} | {item['success_rate']} |"
            )
    labels = report["confusion_matrix"]["labels"]
    matrix = report["confusion_matrix"]["matrix"]
    lines.extend(["", "## Confusion matrix", ""])
    lines.append("| Actual \\ Predicted | " + " | ".join(f"`{x}`" for x in labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for actual in labels:
        lines.append(
            f"| `{actual}` | " + " | ".join(str(matrix[actual][pred]) for pred in labels) + " |"
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            f"Total: {len(report['failures'])}",
            "",
            "| ID | Expected | Predicted | Reasons |",
            "|---|---|---|---|",
        ]
    )
    for item in report["failures"][:100]:
        lines.append(
            f"| `{item['id']}` | `{item['expected_label']}` | "
            f"`{item['predicted_label']}` | {', '.join(item['reasons'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate(
    rows: list[dict[str, Any]],
    predictions: dict[str, str],
    mcp_tools: list[dict[str, Any]],
    clarification_tool: bool = False,
    schema_constraint: bool = False,
    bind_normalized_arguments: bool = False,
    legacy_semantic_eval: bool = False,
) -> dict[str, Any]:
    tools = tool_map(tools_with_clarification(mcp_tools, clarification_tool))
    counts = Counter(
        {
            "total": len(rows),
            "expected_calls": 0,
            "parse_success": 0,
            "raw_parse_success": 0,
            "raw_tool_name_correct": 0,
            "tool_name_correct": 0,
            "required_arguments_satisfied": 0,
            "expected_arguments_cases": 0,
            "expected_arguments_matched": 0,
            "expected_arguments_exact": 0,
            "datetime_cases": 0,
            "datetime_normalized": 0,
            "missing_info_cases": 0,
            "no_call_on_missing_info": 0,
            "parsed_call_cases": 0,
            "schema_valid_calls": 0,
            "unsupported_expected_arguments": 0,
            "unsupported_expected_normalized": 0,
            "unobservable_expected_arguments": 0,
            "intent_cases": 0,
            "intent_time_mode_cases": 0,
            "intent_time_mode_matched": 0,
            "clarification_cases": 0,
            "clarification_correct": 0,
        }
    )
    intent_slot_counts = {
        slot: {"matched": 0, "total": 0} for slot in INTENT_SLOTS
    }
    argument_slot_counts: dict[str, dict[str, int]] = {}
    details: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    actual_labels: list[str] = []
    predicted_labels: list[str] = []

    for row in rows:
        raw = predictions.get(row["id"], "")
        parse_error = None
        try:
            calls = parse_tool_calls(raw)
        except ToolCallParseError as exc:
            calls, parse_error = [], str(exc)
        raw_call_count = len(calls)
        raw_calls = list(calls)
        raw_parse_error = parse_error
        if bind_normalized_arguments:
            history = row.get("history") or []
            source_text = source_text_for_row(row)
            route_stage = sum(message.get("role") == "tool" for message in history)
            recognized, bound_call = constrain_normalized_tool_call(
                calls[0] if len(calls) == 1 else None,
                source_text,
                row.get("reference_datetime"),
                route_stage,
                history,
                semantic_fallback=legacy_semantic_eval,
            )
            if recognized:
                calls = [] if bound_call is None else [bound_call]
                parse_error = None
            elif bound_call is not None and len(calls) == 1:
                # Unrecognized intent still gets value-fidelity repair
                # (coordinates, IDs, relative dates, time padding).
                calls = [bound_call]
        schema_error = None
        if calls:
            counts["parsed_call_cases"] += 1
            try:
                validate_tool_calls(calls, tools.values())
                counts["schema_valid_calls"] += 1
            except ToolCallSchemaError as exc:
                schema_error = str(exc)
                if schema_constraint:
                    calls, parse_error = [], f"schema constraint: {exc}"
        predicted: ToolCall | None = calls[0] if len(calls) == 1 else None
        expected_tool = row.get("expected_tool")
        expected_clarification = expected_tool == CLARIFICATION_TOOL_NAME
        expected_label = NO_CALL if expected_clarification else expected_tool or NO_CALL
        if parse_error:
            predicted_label = PARSE_ERROR
        elif len(calls) > 1:
            predicted_label = MULTIPLE_CALLS
        elif predicted is None:
            predicted_label = NO_CALL
        elif valid_clarification(predicted):
            predicted_label = NO_CALL
        else:
            predicted_label = predicted.name
        actual_labels.append(expected_label)
        predicted_labels.append(predicted_label)

        reasons: list[str] = []
        required_ok = False
        expected_args_ok: bool | None = None
        expected_args_exact: bool | None = None
        if expected_tool:
            counts["expected_calls"] += 1
            if raw_parse_error is None and raw_call_count == 1:
                counts["raw_parse_success"] += 1
                if raw_calls[0].name == expected_tool:
                    counts["raw_tool_name_correct"] += 1
            if predicted is not None and parse_error is None and len(calls) == 1:
                counts["parse_success"] += 1
            else:
                reasons.append("parse_or_cardinality_failure")
            if predicted and predicted.name == expected_tool:
                counts["tool_name_correct"] += 1
                required = tools.get(expected_tool, {}).get("inputSchema", {}).get("required", [])
                required_ok = all(
                    key in predicted.arguments and predicted.arguments[key] not in (None, "")
                    for key in required
                )
                if required_ok:
                    counts["required_arguments_satisfied"] += 1
                else:
                    reasons.append("missing_required_arguments")
            else:
                reasons.append("wrong_tool")
            if "expected_arguments" in row:
                counts["expected_arguments_cases"] += 1
                properties = tools.get(expected_tool, {}).get("inputSchema", {}).get(
                    "properties", {}
                )
                observable_intent = (
                    extract_route_intent(
                        source_text_for_row(row), row.get("reference_datetime")
                    )
                    if legacy_semantic_eval
                    else {}
                )
                observable_time_type = (
                    mcp_time_type(observable_intent.get("time_mode"))
                    if legacy_semantic_eval
                    else None
                )
                def observable(key: str) -> bool:
                    if not legacy_semantic_eval:
                        return key not in set(
                            row.get("unobservable_expected_arguments") or []
                        )
                    if key == "date":
                        return observable_intent.get("date") is not None
                    if key == "time":
                        return observable_intent.get("time") is not None
                    if key == "type":
                        return observable_time_type is not None
                    return True
                expected_arguments = {
                    key: value
                    for key, value in row["expected_arguments"].items()
                    if key in properties and observable(key)
                }
                counts["unsupported_expected_arguments"] += sum(
                    key not in properties for key in row["expected_arguments"]
                )
                counts["unobservable_expected_arguments"] += sum(
                    key in properties and not observable(key)
                    for key in row["expected_arguments"]
                )
                expected_args_ok = bool(
                    predicted
                    and predicted.name == expected_tool
                    and all(
                        semantic_argument_equal(
                            expected_tool, key, predicted.arguments.get(key), value
                        )
                        for key, value in expected_arguments.items()
                    )
                )
                expected_args_exact = bool(
                    predicted
                    and predicted.name == expected_tool
                    and set(expected_arguments) == set(predicted.arguments)
                    and all(
                        semantic_argument_equal(
                            expected_tool, key, predicted.arguments.get(key), value
                        )
                        for key, value in expected_arguments.items()
                    )
                )
                if expected_args_ok:
                    counts["expected_arguments_matched"] += 1
                else:
                    reasons.append("expected_arguments_mismatch")
                if expected_args_exact:
                    counts["expected_arguments_exact"] += 1
                for key, value in expected_arguments.items():
                    slot = argument_slot_counts.setdefault(
                        key, {"matched": 0, "total": 0}
                    )
                    slot["total"] += 1
                    if (
                        predicted
                        and predicted.name == expected_tool
                        and semantic_argument_equal(
                            expected_tool, key, predicted.arguments.get(key), value
                        )
                    ):
                        slot["matched"] += 1
            if expected_clarification:
                counts["missing_info_cases"] += 1
                counts["clarification_cases"] += 1
                if (
                    predicted is not None
                    and predicted_label == NO_CALL
                    and valid_clarification(predicted)
                ):
                    counts["no_call_on_missing_info"] += 1
                    counts["clarification_correct"] += 1
        else:
            counts["missing_info_cases"] += 1
            no_call_ok = predicted_label == NO_CALL
            if no_call_ok:
                counts["no_call_on_missing_info"] += 1
            else:
                reasons.append("unwanted_tool_call")

        if "expected_normalized" in row:
            normalized_properties = tools.get(expected_tool, {}).get("inputSchema", {}).get(
                "properties", {}
            )
            normalized_arguments = dict(predicted.arguments) if predicted else {}
            intent = (
                extract_route_intent(
                    source_text_for_row(row), row.get("reference_datetime")
                )
                if legacy_semantic_eval
                else {}
            )
            derived_type = None
            if legacy_semantic_eval:
                for key in ("date", "time"):
                    if intent.get(key):
                        normalized_arguments[key] = intent[key]
                derived_type = mcp_time_type(intent.get("time_mode"))
                if derived_type:
                    normalized_arguments["type"] = derived_type
            counts["unsupported_expected_normalized"] += sum(
                key not in normalized_properties for key in row["expected_normalized"]
            )
            observable_normalized = {
                key: value
                for key, value in row["expected_normalized"].items()
                if key in normalized_properties
                if not legacy_semantic_eval
                or (
                    (key == "date" and intent.get("date") is not None)
                    or (key == "time" and intent.get("time") is not None)
                    or (key == "type" and derived_type is not None)
                )
            }
            if observable_normalized:
                counts["datetime_cases"] += 1
            normalized_ok = bool(
                observable_normalized
                and predicted
                and all(
                    semantic_argument_equal(
                        expected_tool or "", key, normalized_arguments.get(key), value
                    )
                    for key, value in observable_normalized.items()
                )
            )
            if normalized_ok:
                counts["datetime_normalized"] += 1
            elif observable_normalized:
                reasons.append("datetime_normalization_failure")

        expected_intent = row.get("expected_intent")
        expected_time_mode = (
            expected_intent.get("time_mode")
            if isinstance(expected_intent, dict)
            else None
        )
        actual_time_mode = None
        if expected_time_mode is not None and not (
            row.get("expected_clarification")
            and predicted is not None
            and valid_clarification(predicted)
        ):
            counts["intent_time_mode_cases"] += 1
            raw_predicted = (
                raw_calls[0]
                if raw_parse_error is None and raw_call_count == 1
                else None
            )
            actual_time_mode = intent_time_mode_from_call(
                raw_predicted
            ) or intent_time_mode_from_call(predicted)
            if actual_time_mode == expected_time_mode:
                counts["intent_time_mode_matched"] += 1
        actual_intent = None
        intent_mismatches: list[str] = []
        if isinstance(expected_intent, dict) and legacy_semantic_eval:
            counts["intent_cases"] += 1
            actual_intent = extract_route_intent(
                source_text_for_row(row), row.get("reference_datetime")
            )
            for slot in INTENT_SLOTS:
                intent_slot_counts[slot]["total"] += 1
                if actual_intent.get(slot) == expected_intent.get(slot):
                    intent_slot_counts[slot]["matched"] += 1
                else:
                    intent_mismatches.append(slot)
            if intent_mismatches:
                reasons.append("intent_slot_mismatch")

        detail = {
            "id": row["id"],
            "user": row.get("user"),
            "expected_tool": expected_tool,
            "expected_label": expected_label,
            "expected_arguments": row.get("expected_arguments"),
            "prediction": None if predicted is None else predicted.as_dict(),
            "predicted_label": predicted_label,
            "call_count": len(calls),
            "parse_error": parse_error,
            "raw_parse_error": raw_parse_error,
            "raw_call_count": raw_call_count,
            "schema_error": schema_error,
            "required_arguments_satisfied": required_ok if expected_tool else None,
            "expected_arguments_matched": expected_args_ok,
            "expected_arguments_exact": expected_args_exact,
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "expected_intent_time_mode": expected_time_mode,
            "actual_intent_time_mode": actual_time_mode,
            "intent_mismatches": intent_mismatches,
            "raw_output": raw,
            "reasons": sorted(set(reasons)),
        }
        detail["semantic_success"] = (
            bool(
                predicted
                and predicted.name == expected_tool
                and required_ok
                and (expected_args_ok is not False)
            )
            if expected_tool
            else predicted_label == NO_CALL
        )
        details.append(detail)
        if reasons:
            failures.append(detail)

    classes = sorted(set(actual_labels))
    confusion_labels = sorted(set(actual_labels) | set(predicted_labels))
    confusion = {
        actual: {
            predicted: sum(
                a == actual and p == predicted
                for a, p in zip(actual_labels, predicted_labels)
            )
            for predicted in confusion_labels
        }
        for actual in confusion_labels
    }
    metrics = {
        "raw_parse_success_rate": ratio(
            counts["raw_parse_success"], counts["expected_calls"]
        ),
        "raw_tool_name_accuracy": ratio(
            counts["raw_tool_name_correct"], counts["expected_calls"]
        ),
        "parse_success_rate": ratio(counts["parse_success"], counts["expected_calls"]),
        "tool_name_accuracy": ratio(counts["tool_name_correct"], counts["expected_calls"]),
        "expected_arguments_match_rate": ratio(
            counts["expected_arguments_matched"], counts["expected_arguments_cases"]
        ),
        "expected_arguments_exact_match_rate": ratio(
            counts["expected_arguments_exact"], counts["expected_arguments_cases"]
        ),
        "required_arguments_satisfaction_rate": ratio(
            counts["required_arguments_satisfied"], counts["expected_calls"]
        ),
        "datetime_normalization_success_rate": ratio(
            counts["datetime_normalized"], counts["datetime_cases"]
        ),
        "no_call_when_missing_info_rate": ratio(
            counts["no_call_on_missing_info"], counts["missing_info_cases"]
        ),
        "clarification_accuracy": ratio(
            counts["clarification_correct"], counts["clarification_cases"]
        ),
        "intent_time_mode_accuracy": ratio(
            counts["intent_time_mode_matched"], counts["intent_time_mode_cases"]
        ),
        "overall_class_accuracy": ratio(
            sum(a == p for a, p in zip(actual_labels, predicted_labels)), len(rows)
        ),
        "schema_valid_call_rate": ratio(
            counts["schema_valid_calls"], counts["parsed_call_cases"]
        ),
        "semantic_success_rate": ratio(
            sum(bool(item.get("semantic_success")) for item in details), len(details)
        ),
    }
    intent_slots = {
        slot: {
            **values,
            "match_rate": ratio(values["matched"], values["total"]),
            "f1": ratio(values["matched"], values["total"]),
        }
        for slot, values in intent_slot_counts.items()
        if values["total"]
    }
    if intent_slots:
        slot_f1 = [item["f1"] for item in intent_slots.values() if item["f1"] is not None]
        metrics["intent_slot_macro_f1"] = round(sum(slot_f1) / len(slot_f1), 4)
        avoid_via = [
            intent_slots[slot]["match_rate"]
            for slot in ("avoid_station_texts", "via_station_texts")
            if slot in intent_slots and intent_slots[slot]["match_rate"] is not None
        ]
        metrics["avoid_via_extraction_success_rate"] = round(
            sum(avoid_via) / len(avoid_via), 4
        )
    argument_slots = {
        name: {
            **values,
            "match_rate": ratio(values["matched"], values["total"]),
        }
        for name, values in sorted(argument_slot_counts.items())
    }
    category_counts: dict[str, Counter[str]] = {}
    for row, detail in zip(rows, details):
        category = str(row.get("category") or "uncategorized")
        counts_for_category = category_counts.setdefault(category, Counter())
        counts_for_category["total"] += 1
        counts_for_category["passed"] += int(detail["semantic_success"])
    by_category = {
        name: {
            "total": values["total"],
            "passed": values["passed"],
            "success_rate": ratio(values["passed"], values["total"]),
        }
        for name, values in sorted(category_counts.items())
    }
    return {
        "metrics": metrics,
        "counts": dict(counts),
        "intent_slots": intent_slots,
        "argument_slots": argument_slots,
        "by_category": by_category,
        "per_class": per_class_scores(actual_labels, predicted_labels, classes),
        "confusion_matrix": {"labels": confusion_labels, "matrix": confusion},
        "failures": failures,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FunctionGemma transit tool routing.")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/eval_template.jsonl"))
    parser.add_argument("--predictions", type=Path, help="JSONL with id and model_output")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--adapter")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--schema-mode", choices=("baked", "compact", "full"), default="baked")
    parser.add_argument("--clarification-tool", action="store_true")
    parser.add_argument("--normalize-ja", action="store_true")
    parser.add_argument("--constrained-decode", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument(
        "--schema-constraint",
        action="store_true",
        help="Treat non-allow-listed tools and schema-invalid arguments as rejected output.",
    )
    parser.add_argument(
        "--bind-normalized-arguments",
        action="store_true",
        help="Bind high-confidence Japanese normalization slots before schema validation.",
    )
    parser.add_argument(
        "--legacy-semantic-eval",
        action="store_true",
        help="Opt in to the retired regex intent parser for compatibility scoring.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval_report.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("artifacts/eval_report.md"))
    parser.add_argument("--failures-output", type=Path, default=Path("artifacts/eval_failures.jsonl"))
    args = parser.parse_args()
    if bool(args.predictions) == bool(args.run_model):
        parser.error("choose exactly one of --predictions or --run-model")

    rows = read_jsonl(args.dataset)
    if args.predictions:
        predictions = {row["id"]: row["model_output"] for row in read_jsonl(args.predictions)}
    else:
        from transit_functiongemma.infer import ToolRouter

        router = ToolRouter(
            args.base_model,
            args.adapter,
            args.schema,
            args.schema_mode,
            args.clarification_tool,
            args.normalize_ja,
            args.constrained_decode,
            args.prefix_cache,
        )
        predictions = {
            row["id"]: router.generate(
                (
                    row.get("user")
                    if isinstance(row.get("user"), str) and row.get("user").strip()
                    else None
                ),
                row.get("reference_datetime"),
                history=row.get("history"),
            )
            for row in rows
        }

    report = evaluate(
        rows,
        predictions,
        load_mcp_tools(args.schema),
        args.clarification_tool,
        args.schema_constraint,
        args.bind_normalized_arguments,
        args.legacy_semantic_eval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.failures_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    with args.failures_output.open("w", encoding="utf-8") as stream:
        for failure in report["failures"]:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"JSON report: {args.output}")
    print(f"Markdown report: {args.markdown_output}")
    print(f"Failures: {args.failures_output}")


if __name__ == "__main__":
    main()
