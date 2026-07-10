#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.audit_expected_arguments import (
    DEFAULT_EVAL_DIR,
    DEFAULT_MIGRATION_REPORT,
    build_audit_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_DIR = DEFAULT_EVAL_DIR / "pre_audit_backup"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = line_number
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            public = {key: value for key, value in row.items() if key != "_line"}
            stream.write(
                json.dumps(public, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _norm(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def _remove_station_departures_type(
    path: Path, row: dict[str, Any], changes: list[dict[str, Any]]
) -> bool:
    changed = False
    if row.get("expected_tool") != "station_departures":
        return False
    for field in ("expected_arguments", "expected_normalized"):
        arguments = row.get(field)
        if not isinstance(arguments, dict) or "type" not in arguments:
            continue
        old_value = arguments.pop("type")
        changes.append(
            {
                "file": str(path),
                "line": row["_line"],
                "id": row.get("id"),
                "field": field,
                "key": "type",
                "action": "remove_unsupported_key",
                "old_value": old_value,
                "new_value": None,
                "reason": "station_departures inputSchema has id/date/time/limit only",
            }
        )
        changed = True
    return changed


def _fix_station_q_suffix(
    path: Path, row: dict[str, Any], changes: list[dict[str, Any]]
) -> bool:
    if row.get("expected_tool") != "suggest_stations":
        return False
    arguments = row.get("expected_arguments")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("q"), str):
        return False
    q = arguments["q"]
    q_norm = _norm(q)
    if not q_norm or q_norm.endswith("駅"):
        return False
    user_norm = _norm(row.get("user") or "")
    if f"{q_norm}駅" not in user_norm:
        return False
    old_value = q
    arguments["q"] = f"{q}駅"
    changes.append(
        {
            "file": str(path),
            "line": row["_line"],
            "id": row.get("id"),
            "field": "expected_arguments",
            "key": "q",
            "action": "append_station_suffix_by_majority_convention",
            "old_value": old_value,
            "new_value": arguments["q"],
            "reason": "user wrote an explicit station suffix and audited majority keeps it",
        }
    )
    return True


def _schema_violation_needs_human_judgment(violation: dict[str, Any]) -> bool:
    return not (
        violation.get("tool") == "station_departures"
        and violation.get("key") == "type"
        and violation.get("kind") == "unknown_key"
        and violation.get("field") in {"expected_arguments", "expected_normalized"}
    )


def plan_and_apply(
    eval_dir: Path = DEFAULT_EVAL_DIR,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    output: Path = DEFAULT_MIGRATION_REPORT,
    dry_run: bool = False,
) -> dict[str, Any]:
    pre_audit = build_audit_report(eval_dir=eval_dir, migration_report_path=None)
    needs_human_judgment = [
        {
            "file": item["file"],
            "line": item["line"],
            "id": item["id"],
            "reason": item["message"],
        }
        for item in pre_audit["violations"]
        if _schema_violation_needs_human_judgment(item)
    ]

    changes: list[dict[str, Any]] = []
    rows_by_path: dict[Path, list[dict[str, Any]]] = {}
    changed_paths: set[Path] = set()
    changed_row_ids_by_path: dict[Path, set[str]] = defaultdict(set)

    for path in sorted(eval_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        file_changed = False
        for row in rows:
            row_changed = False
            row_changed |= _remove_station_departures_type(path, row, changes)
            row_changed |= _fix_station_q_suffix(path, row, changes)
            if row_changed:
                file_changed = True
                changed_row_ids_by_path[path].add(str(row.get("id")))
        if file_changed:
            changed_paths.add(path)
            rows_by_path[path] = rows

    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up_files: list[str] = []
    backup_skipped_existing: list[str] = []
    if not dry_run:
        for path in sorted(changed_paths):
            backup_path = backup_dir / path.name
            if backup_path.exists():
                backup_skipped_existing.append(str(backup_path))
            else:
                shutil.copy2(path, backup_path)
                backed_up_files.append(str(backup_path))
        for path in sorted(changed_paths):
            write_jsonl(path, rows_by_path[path])

    changed_row_count = sum(len(ids) for ids in changed_row_ids_by_path.values())
    report = {
        "dry_run": dry_run,
        "backup_dir": str(backup_dir),
        "backed_up_files": backed_up_files,
        "backup_skipped_existing": backup_skipped_existing,
        "changed_files": [str(path) for path in sorted(changed_paths)],
        "changed_file_count": len(changed_paths),
        "changed_row_count": changed_row_count,
        "changes": changes,
        "needs_human_judgment": needs_human_judgment,
    }
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply confirmed eval-label audit fixes.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MIGRATION_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = plan_and_apply(args.eval_dir, args.backup_dir, args.output, args.dry_run)
    print(json.dumps({
        "changed_file_count": report["changed_file_count"],
        "changed_row_count": report["changed_row_count"],
        "field_level_changes": len(report["changes"]),
        "needs_human_judgment": len(report["needs_human_judgment"]),
        "backup_dir": report["backup_dir"],
    }, ensure_ascii=False, indent=2))
    if not args.dry_run:
        print(f"Migration report: {args.output}")


if __name__ == "__main__":
    main()
