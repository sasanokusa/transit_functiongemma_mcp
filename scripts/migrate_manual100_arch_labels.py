#!/usr/bin/env python3
"""Migrate legacy route-first labels in manual_practical_100.jsonl.

The migration is deliberately conservative.  Only rows whose current label is
``suggest_stations`` and whose user utterance contains a high-confidence route
request are changed.  Explicit station-search requests and ambiguous route
phrases are reported, but left untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from transit_functiongemma.line_operator_rules import extract_operator_constraints
from transit_functiongemma.route_intent import extract_route_intent


DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "manual_practical_100.jsonl"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "eval" / "pre_arch_migration_backup"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "MANUAL100_ARCH_MIGRATION.md"
DEFAULT_SCHEMA = PROJECT_ROOT / "tools" / "local_tools_schema.json"

ROUTE_TOOL = "resolve_route_request"
STATION_TOOL = "suggest_stations"

_EXPLICIT_STATION_SEARCH_PATTERNS = (
    re.compile(r"駅として(?:拾|探|検索)"),
    re.compile(r"駅(?:名)?の候補"),
    re.compile(r"駅(?:名)?を(?:検索|探)"),
    re.compile(r"駅候補"),
)

_DESTINATION_BOUNDARY = re.compile(
    r"(?:まで|行きたい|行ける|に向か(?:う|いたい)?|"
    r"へ(?=[行向着、。,]|$)|[、,。])"
)
_ORIGIN_PREFIX = re.compile(
    r"^(?:(?:今日|明日|本日)の?)?(?:終電|始発)(?:で|に)?"
    r"|^(?:(?:今日|明日|本日)\s*)?"
    r"\d{1,2}(?::\d{2}|時(?:\d{1,2}分|半)?)?\s*(?:出発|発)で"
)


@dataclass(frozen=True)
class DocumentLine:
    number: int
    raw: bytes
    row: dict[str, Any]


@dataclass(frozen=True)
class Decision:
    kind: str
    reason: str
    endpoints: tuple[str, str] | None = None
    pattern: str | None = None


def load_schema(path: Path = DEFAULT_SCHEMA) -> tuple[tuple[str, ...], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    for tool in document.get("tools", []):
        if tool.get("name") == ROUTE_TOOL:
            schema = tool["inputSchema"]
            required = tuple(schema["required"])
            if len(required) != 14 or len(set(required)) != 14:
                raise AssertionError(f"{path}: resolve_route_request must have 14 fields")
            if set(required) != set(schema["properties"]):
                raise AssertionError(f"{path}: required/properties mismatch")
            return required, schema
    raise AssertionError(f"{path}: {ROUTE_TOOL} not found")


def read_jsonl_document(path: Path) -> list[DocumentLine]:
    lines: list[DocumentLine] = []
    for number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        payload = raw.rstrip(b"\r\n")
        if not payload.strip():
            raise AssertionError(f"{path}:{number}: blank JSONL line")
        row = json.loads(payload.decode("utf-8"))
        if not isinstance(row, dict):
            raise AssertionError(f"{path}:{number}: JSONL row is not an object")
        lines.append(DocumentLine(number, raw, row))
    return lines


def _line_ending(raw: bytes) -> bytes:
    if raw.endswith(b"\r\n"):
        return b"\r\n"
    if raw.endswith(b"\n"):
        return b"\n"
    if raw.endswith(b"\r"):
        return b"\r"
    return b""


def serialize_row(row: dict[str, Any], raw_template: bytes) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
        + _line_ending(raw_template)
    )


def _clean_surface_endpoint(value: str, *, origin: bool) -> str:
    value = value.strip(" \t\u3000、,。．.!！?？")
    if origin:
        value = _ORIGIN_PREFIX.sub("", value, count=1)
    return value.strip(" \t\u3000、,。．.!！?？")


def _surface_from_to(text: str) -> tuple[str, str] | None:
    from_markers = [
        match
        for match in re.finditer("から", text)
        if match.start() == 0 or text[match.start() - 1] != "だ"
    ]
    if len(from_markers) != 1:
        return None
    marker = from_markers[0]
    before, after = text[: marker.start()], text[marker.end() :]
    origin = _clean_surface_endpoint(before, origin=True)
    destination = _DESTINATION_BOUNDARY.split(after, maxsplit=1)[0]
    destination = _clean_surface_endpoint(destination, origin=False)
    if origin and destination:
        return origin, destination
    return None


def _surface_marked_endpoints(text: str) -> tuple[tuple[str, str], str] | None:
    patterns = (
        (
            re.compile(r"出発地は(?P<origin>[^、,。]+)[、,。]目的地は(?P<destination>[^、,。]+)"),
            "出発地／目的地",
        ),
        (
            re.compile(r"(?P<origin>[^、,。]+?)(?:発で|出発で)(?P<destination>[^、,。]+?)(?:着|行き|まで|$)"),
            "発／着",
        ),
        (
            re.compile(r"(?P<origin>[^、,。]+?)発(?P<destination>[^、,。]+?)(?:着|行き|まで|$)"),
            "発／行き",
        ),
        (
            re.compile(r"(?P<origin>[^、,。→⟶]+?)[→⟶](?P<destination>[^、,。]+)"),
            "矢印",
        ),
    )
    matches: list[tuple[tuple[str, str], str]] = []
    for pattern, name in patterns:
        match = pattern.search(text)
        if not match:
            continue
        origin = _clean_surface_endpoint(match.group("origin"), origin=True)
        destination = _clean_surface_endpoint(match.group("destination"), origin=False)
        if origin and destination:
            matches.append(((origin, destination), name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and len({item[0] for item in matches}) == 1:
        return matches[0]
    return None


def extract_surface_endpoints(text: str) -> tuple[tuple[str, str] | None, str | None]:
    """Return literal endpoint text and the route syntax that supplied it."""
    from_to = _surface_from_to(text)
    marked = _surface_marked_endpoints(text)
    candidates: list[tuple[tuple[str, str], str]] = []
    if from_to:
        candidates.append((from_to, "から"))
    if marked:
        candidates.append(marked)
    if not candidates:
        return None, None
    if len(candidates) > 1 and len({item[0] for item in candidates}) > 1:
        return None, None
    return candidates[0]


def is_explicit_station_search(text: str) -> bool:
    return any(pattern.search(text) for pattern in _EXPLICIT_STATION_SEARCH_PATTERNS)


def classify_row(row: dict[str, Any]) -> Decision:
    if row.get("expected_tool") != STATION_TOOL:
        return Decision(
            "excluded",
            f"expected_tool が {row.get('expected_tool')!r} であり、{STATION_TOOL!r} ではない",
        )
    text = row.get("user")
    if not isinstance(text, str) or not text.strip():
        return Decision("human", "発話本文が空または文字列ではないため、起点・終点を確認できない")
    if is_explicit_station_search(text):
        return Decision("excluded", "明示的な駅検索指示があるため、決定表上 suggest_stations のまま")
    endpoints, pattern = extract_surface_endpoints(text)
    if endpoints is None:
        return Decision(
            "human",
            "起点と終点を高い確信度で同時に抽出できないため、決定表に従い移行しない",
        )
    return Decision("target", "起点・終点を含む単発の経路要求で、明示的な駅検索指示がない", endpoints, pattern)


def _extract_preferred_lines(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"([^\s、,。]+線)で", text):
        value = match.group(1)
        if value not in values:
            values.append(value)
    return values


def derive_arguments(
    row: dict[str, Any], required_fields: Iterable[str]
) -> dict[str, Any]:
    text = row["user"]
    decision = classify_row(row)
    if decision.kind != "target" or decision.endpoints is None:
        raise ValueError(f"row {row.get('id')!r} is not a migration target")

    # This parser is used only for conservative offline annotation.  Endpoint
    # values are overwritten with literal surface text below because the
    # current manual-data policy explicitly requires that behavior.
    parsed = extract_route_intent(text, row.get("reference_datetime"))
    operators = extract_operator_constraints(text)
    avoid_lines = list(parsed.get("avoid_line_texts") or [])
    avoid_operator_groups = list(operators.get("avoid_operator_groups") or [])
    if "JR" in avoid_lines:
        avoid_lines = [value for value in avoid_lines if value != "JR"]
        if "JR" not in avoid_operator_groups:
            avoid_operator_groups.append("JR")

    values: dict[str, Any] = {
        "origin_text": decision.endpoints[0],
        "destination_text": decision.endpoints[1],
        "via_station_texts": list(parsed.get("via_station_texts") or []),
        "avoid_station_texts": list(parsed.get("avoid_station_texts") or []),
        "avoid_line_texts": avoid_lines,
        "preferred_line_texts": _extract_preferred_lines(text),
        "allowed_operator_groups": list(operators.get("allowed_operator_groups") or []),
        "avoid_operator_groups": avoid_operator_groups,
        "avoid_modes": list(operators.get("avoid_modes") or []),
        "priority": parsed.get("priority"),
        "time_mode": parsed.get("time_mode"),
        "date": parsed.get("date"),
        "time": parsed.get("time"),
        "graphical": bool(parsed.get("graphical")),
    }
    ordered = {field: values[field] for field in required_fields}
    validate_arguments(ordered, required_fields)
    return ordered


def validate_arguments(arguments: dict[str, Any], required_fields: Iterable[str]) -> None:
    required = tuple(required_fields)
    if tuple(arguments) != required:
        raise AssertionError(f"resolve_route_request argument order/fields mismatch: {tuple(arguments)}")
    if not isinstance(arguments["origin_text"], str) or not arguments["origin_text"]:
        raise AssertionError("origin_text must be a non-empty string")
    if not isinstance(arguments["destination_text"], str) or not arguments["destination_text"]:
        raise AssertionError("destination_text must be a non-empty string")
    for field in (
        "via_station_texts",
        "avoid_station_texts",
        "avoid_line_texts",
        "preferred_line_texts",
        "allowed_operator_groups",
        "avoid_operator_groups",
        "avoid_modes",
    ):
        if not isinstance(arguments[field], list) or not all(
            isinstance(value, str) for value in arguments[field]
        ):
            raise AssertionError(f"{field} must be a string list")
    if arguments["priority"] not in {None, "fast", "cheap", "few_transfers", "less_walk"}:
        raise AssertionError(f"invalid priority: {arguments['priority']!r}")
    if arguments["time_mode"] not in {
        None,
        "departure_at",
        "arrive_by",
        "first_train",
        "last_train",
    }:
        raise AssertionError(f"invalid time_mode: {arguments['time_mode']!r}")
    if arguments["date"] is not None and not re.fullmatch(r"\d{8}", arguments["date"]):
        raise AssertionError(f"invalid date: {arguments['date']!r}")
    if arguments["time"] is not None and not re.fullmatch(r"\d{2}:\d{2}", arguments["time"]):
        raise AssertionError(f"invalid time: {arguments['time']!r}")
    if not isinstance(arguments["graphical"], bool):
        raise AssertionError("graphical must be boolean")


def make_after_row(row: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    after = copy.deepcopy(row)
    after["expected_tool"] = ROUTE_TOOL
    after["expected_arguments"] = arguments
    return after


def _decision_item(document_line: DocumentLine, decision: Decision) -> dict[str, Any]:
    return {
        "line": document_line.number,
        "id": document_line.row.get("id"),
        "user": document_line.row.get("user"),
        "reason": decision.reason,
        "pattern": decision.pattern,
        "endpoints": list(decision.endpoints) if decision.endpoints else None,
    }


def plan_migration(path: Path, schema_path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    required_fields, schema = load_schema(schema_path)
    document = read_jsonl_document(path)
    targets: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    human: list[dict[str, Any]] = []
    after_rows: dict[int, dict[str, Any]] = {}
    for document_line in document:
        decision = classify_row(document_line.row)
        if decision.kind == "target":
            arguments = derive_arguments(document_line.row, required_fields)
            after = make_after_row(document_line.row, arguments)
            item = _decision_item(document_line, decision)
            item.update(
                {
                    "before": document_line.row,
                    "after": after,
                    "arguments": arguments,
                }
            )
            targets.append(item)
            after_rows[document_line.number] = after
        elif decision.kind == "human":
            item = _decision_item(document_line, decision)
            item.update({"before": document_line.row, "after": document_line.row})
            human.append(item)
        else:
            excluded.append(_decision_item(document_line, decision))

    before_bytes = path.read_bytes()
    after_bytes = b"".join(
        serialize_row(after_rows.get(line.number, line.row), line.raw) for line in document
    )
    target_numbers = {item["line"] for item in targets}
    assert_non_target_bytes(before_bytes, after_bytes, target_numbers, path)
    return {
        "path": path,
        "schema_path": schema_path,
        "schema": schema,
        "required_fields": required_fields,
        "document": document,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "targets": targets,
        "excluded": excluded,
        "needs_human_judgment": human,
        "target_numbers": target_numbers,
    }


def assert_non_target_bytes(
    before_bytes: bytes, after_bytes: bytes, target_numbers: set[int], path: Path | None = None
) -> None:
    before_lines = before_bytes.splitlines(keepends=True)
    after_lines = after_bytes.splitlines(keepends=True)
    if len(before_lines) != len(after_lines):
        raise AssertionError(f"{path or '<document>'}: line count changed")
    for number, (before, after) in enumerate(zip(before_lines, after_lines), 1):
        if number not in target_numbers and before != after:
            raise AssertionError(f"{path or '<document>'}:{number}: non-target bytes changed")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        mode = (path.stat().st_mode & 0o7777) if path.exists() else 0o644
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _backup_path(input_path: Path, backup_dir: Path) -> Path:
    return backup_dir / input_path.name


def _ensure_backup(input_path: Path, backup_dir: Path, before_bytes: bytes) -> tuple[Path, bool]:
    backup_path = _backup_path(input_path, backup_dir)
    if backup_path.exists():
        existing = backup_path.read_bytes()
        if existing != before_bytes:
            raise AssertionError(f"{backup_path}: existing backup differs from migration input")
        return backup_path, False
    backup_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(backup_path, before_bytes)
    return backup_path, True


def _json_for_report(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def build_report(plan: dict[str, Any], *, dry_run: bool, backup_path: Path) -> str:
    lines = [
        "# MANUAL100 アーキテクチャ・ラベル移行監査",
        "",
        f"- 入力: `{plan['path']}`",
        f"- バックアップ: `{backup_path}`",
        f"- スキーマ: `{plan['schema_path']}`",
        f"- モード: `{'dry-run' if dry_run else 'apply'}`",
        f"- 総行数: {len(plan['document'])}",
        f"- 移行対象: {len(plan['targets'])} 行",
        f"- 対象外: {len(plan['excluded'])} 行",
        f"- 要人手判断: {len(plan['needs_human_judgment'])} 行",
        "",
        "## 判定基準",
        "",
        "`expected_tool == suggest_stations` かつ、発話が起点・終点を含む経路要求で、明示的な駅検索指示がない行だけを `resolve_route_request` に移行した。",
        "起点・終点は発話中の表記を保持し、その他のスロットは発話から決定的に導出した。対象外・要人手判断の行は変更していない。",
        "",
        "## 移行対象（before / after 全文）",
        "",
    ]
    if not plan["targets"]:
        lines.append("移行対象なし。")
    for item in plan["targets"]:
        lines.extend(
            [
                f"### `{item['id']}`（line {item['line']}）",
                "",
                f"- 判定: {item['reason']}（構文: `{item['pattern']}`）",
                f"- 発話全文: {item['user']}",
                "",
                "#### before（JSONL 1行の内容）",
                "",
                "```json",
                _json_for_report(item["before"]),
                "```",
                "",
                "#### after（JSONL 1行の内容）",
                "",
                "```json",
                _json_for_report(item["after"]),
                "```",
                "",
            ]
        )

    lines.extend(["## 対象外（理由付き）", ""])
    if not plan["excluded"]:
        lines.append("対象外なし。")
    for item in plan["excluded"]:
        lines.append(
            f"- line {item['line']} `{item['id']}`: {item['reason']}。発話全文: {item['user']!r}"
        )

    lines.extend(["", "## 要人手判断（移行せず）", ""])
    if not plan["needs_human_judgment"]:
        lines.append("なし。")
    for item in plan["needs_human_judgment"]:
        lines.extend(
            [
                f"### `{item['id']}`（line {item['line']}）",
                "",
                f"- 理由: {item['reason']}",
                f"- 発話全文: {item['user']!r}",
                "- before/after: 変更なし",
                "",
            ]
        )

    lines.extend(
        [
            "## 検証",
            "",
            "- `resolve_route_request` のスキーマ必須14フィールドを検証した。",
            "- 変更対象以外のJSONL行は、適用前後でバイト単位一致をassertした。",
            "- 採点ロジックとモデル実行はこの移行では変更・実行していない。",
            "",
        ]
    )
    return "\n".join(lines)


def migrate(
    input_path: Path = DEFAULT_INPUT,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    report_path: Path = DEFAULT_REPORT,
    schema_path: Path = DEFAULT_SCHEMA,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    plan = plan_migration(input_path, schema_path)
    backup_path = _backup_path(input_path, backup_dir)
    report_written = False
    backup_written = False
    changed = bool(plan["targets"])

    if not dry_run and changed:
        backup_path, backup_written = _ensure_backup(input_path, backup_dir, plan["before_bytes"])
        _atomic_write(input_path, plan["after_bytes"])
        # Re-read the result and assert that the written document still has the
        # planned target rows and the exact original bytes elsewhere.
        after_on_disk = input_path.read_bytes()
        assert after_on_disk == plan["after_bytes"], f"{input_path}: written bytes differ from plan"
        assert_non_target_bytes(plan["before_bytes"], after_on_disk, plan["target_numbers"], input_path)

    if not dry_run and (changed or plan["needs_human_judgment"]):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            build_report(plan, dry_run=False, backup_path=backup_path), encoding="utf-8"
        )
        report_written = True

    return {
        "dry_run": dry_run,
        "changed_row_count": len(plan["targets"]),
        "changed_file_count": int(not dry_run and changed),
        "backup_path": str(backup_path),
        "backup_written": backup_written,
        "report_path": str(report_path),
        "report_written": report_written,
        "targets": plan["targets"],
        "excluded": plan["excluded"],
        "needs_human_judgment": plan["needs_human_judgment"],
    }


def _print_summary(result: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "dry_run": result["dry_run"],
                "changed_row_count": result["changed_row_count"],
                "changed_file_count": result["changed_file_count"],
                "excluded_count": len(result["excluded"]),
                "needs_human_judgment_count": len(result["needs_human_judgment"]),
                "backup_written": result["backup_written"],
                "report_written": result["report_written"],
                "backup_path": result["backup_path"],
                "report_path": result["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result["targets"]:
        print("移行対象:")
        for item in result["targets"]:
            print(f"  line {item['line']} {item['id']}: {item['user']}")
    if result["needs_human_judgment"]:
        print("要人手判断:")
        for item in result["needs_human_judgment"]:
            print(f"  line {item['line']} {item['id']}: {item['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="判定だけ行い、ファイルを書き換えない（既定）")
    mode.add_argument("--apply", action="store_true", help="バックアップ後にラベル移行を適用する")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    result = migrate(
        args.input,
        args.backup_dir,
        args.report,
        args.schema,
        dry_run=not args.apply,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()
