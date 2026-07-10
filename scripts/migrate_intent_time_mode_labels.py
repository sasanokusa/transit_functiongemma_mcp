#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_BACKUP_DIR = DEFAULT_EVAL_DIR / "pre_audit_backup"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "intent_time_mode_label_migration.json"
TYPE_TO_TIME_MODE = {"first": "first_train", "last": "last_train"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def old_time_mode(row: dict[str, Any]) -> str | None:
    if row.get("expected_tool") != "station_departures":
        return None
    for field in ("expected_normalized", "expected_arguments"):
        arguments = row.get(field)
        if isinstance(arguments, dict):
            value = arguments.get("type")
            if value in TYPE_TO_TIME_MODE:
                return TYPE_TO_TIME_MODE[value]
    return None


def migrate(
    eval_dir: Path = DEFAULT_EVAL_DIR,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    changed_files: set[str] = set()
    old_by_file: dict[str, dict[str, str]] = {}
    for backup_path in sorted(backup_dir.glob("*.jsonl")):
        mapping: dict[str, str] = {}
        for row in read_jsonl(backup_path):
            mode = old_time_mode(row)
            if mode:
                mapping[str(row["id"])] = mode
        if mapping:
            old_by_file[backup_path.name] = mapping

    for file_name, mapping in old_by_file.items():
        path = eval_dir / file_name
        if not path.exists():
            continue
        rows = read_jsonl(path)
        file_changed = False
        for line_number, row in enumerate(rows, 1):
            mode = mapping.get(str(row.get("id")))
            if not mode:
                continue
            intent = row.setdefault("expected_intent", {})
            existing = intent.get("time_mode")
            if existing not in (None, mode):
                conflicts.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "id": row.get("id"),
                        "existing": existing,
                        "migrated": mode,
                    }
                )
                continue
            if existing == mode:
                continue
            intent["time_mode"] = mode
            changes.append(
                {
                    "file": str(path),
                    "line": line_number,
                    "id": row.get("id"),
                    "old_value": existing,
                    "new_value": mode,
                }
            )
            file_changed = True
        if file_changed:
            write_jsonl(path, rows)
            changed_files.add(str(path))

    report = {
        "backup_dir": str(backup_dir),
        "changed_files": sorted(changed_files),
        "changed_file_count": len(changed_files),
        "changed_row_count": len(changes),
        "changes": changes,
        "conflicts": conflicts,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Move old station_departures type labels to expected_intent.time_mode.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = migrate(args.eval_dir, args.backup_dir, args.report)
    print(
        json.dumps(
            {
                "changed_file_count": report["changed_file_count"],
                "changed_row_count": report["changed_row_count"],
                "conflicts": len(report["conflicts"]),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
