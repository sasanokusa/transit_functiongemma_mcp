#!/usr/bin/env python3
"""Migrate station-only first/last-train evaluation labels to A+C policy."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEPENDENT = PROJECT_ROOT / "data" / "eval" / "independent_holdout_300.jsonl"
DEFAULT_MANUAL = PROJECT_ROOT / "data" / "eval" / "manual_practical_100.jsonl"
DEFAULT_DERIVED = PROJECT_ROOT / "artifacts" / "eval_nonroute_215_reaudited_dataset.jsonl"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "eval" / "pre_ac_policy_backup"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "AC_POLICY_LABEL_MIGRATION.md"

INDEPENDENT_TARGET_MODES = {
    "ind-113": "last_train",
    "ind-114": "first_train",
    "ind-118": "last_train",
    "ind-119": "first_train",
    "ind-123": "last_train",
    "ind-124": "first_train",
    "ind-128": "last_train",
    "ind-129": "first_train",
    "ind-133": "last_train",
    "ind-134": "first_train",
    "ind-138": "last_train",
    "ind-139": "first_train",
}
MANUAL_TARGET_MODES = {
    "man-030": "last_train",
    "man-032": "last_train",
    "man-034": "last_train",
    "man-036": "last_train",
    "man-038": "last_train",
}
ALL_TARGET_IDS = frozenset(INDEPENDENT_TARGET_MODES) | frozenset(MANUAL_TARGET_MODES)

CLARIFICATION = {
    "missing": ["destination"],
    "question": "目的地を教えてください。",
}

STATION_ID_RE = re.compile(
    r"(?<![A-Za-z0-9._:-])[A-Za-z0-9][A-Za-z0-9._-]*:"
    r"[A-Za-z0-9][A-Za-z0-9._:-]*(?![A-Za-z0-9._:-])"
)
ROUTE_CONTEXT_PATTERNS = (
    re.compile(r"(?:から|より).+?(?:まで|へ|行きたい|行く|着きたい|到着)"),
    re.compile(r"(?:出発|\b発\b|発で).+?(?:到着|\b着\b|着で|まで|へ)"),
    re.compile(r"(?:出発地|起点).+?(?:目的地|行き先|終点)"),
    re.compile(r"\b(?:from|origin)\b.+?\b(?:to|destination)\b", re.IGNORECASE),
    re.compile(r"\S+\s*(?:→|⇒|->)\s*\S+"),
)


@dataclass
class JsonlDocument:
    path: Path
    raw_lines: list[str]
    rows: list[dict[str, Any] | None]
    line_endings: list[str]

    @classmethod
    def read(cls, path: Path) -> "JsonlDocument":
        text = path.read_text(encoding="utf-8")
        raw_lines = text.splitlines(keepends=True)
        rows: list[dict[str, Any] | None] = []
        endings: list[str] = []
        for line_number, raw_line in enumerate(raw_lines, 1):
            if raw_line.endswith("\r\n"):
                body, ending = raw_line[:-2], "\r\n"
            elif raw_line.endswith("\n") or raw_line.endswith("\r"):
                body, ending = raw_line[:-1], raw_line[-1]
            else:
                body, ending = raw_line, ""
            endings.append(ending)
            if not body.strip():
                rows.append(None)
                continue
            row = json.loads(body)
            if not isinstance(row, dict):
                raise AssertionError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
        return cls(path=path, raw_lines=raw_lines, rows=rows, line_endings=endings)

    def row_locations(self) -> dict[str, tuple[int, dict[str, Any]]]:
        locations: dict[str, tuple[int, dict[str, Any]]] = {}
        for index, row in enumerate(self.rows):
            if row is None:
                continue
            row_id = row.get("id")
            if not isinstance(row_id, str):
                raise AssertionError(f"{self.path}:{index + 1}: missing string id")
            if row_id in locations:
                raise AssertionError(f"{self.path}: duplicate id {row_id}")
            locations[row_id] = (index, row)
        return locations

    def render_with(self, replacements: dict[int, dict[str, Any]]) -> bytes:
        rendered = list(self.raw_lines)
        for index, row in replacements.items():
            rendered[index] = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + self.line_endings[index]
            )
        replacement_indexes = set(replacements)
        assert all(
            rendered[index] == raw_line
            for index, raw_line in enumerate(self.raw_lines)
            if index not in replacement_indexes
        ), f"{self.path}: non-target raw line changed"
        return "".join(rendered).encode("utf-8")


def _machine_check(user: Any) -> dict[str, Any]:
    text = user if isinstance(user, str) else ""
    station_ids = STATION_ID_RE.findall(text)
    route_context_matches = [
        match.group(0) for pattern in ROUTE_CONTEXT_PATTERNS if (match := pattern.search(text))
    ]
    return {
        "station_ids": station_ids,
        "station_id_count": len(station_ids),
        "route_context_matches": route_context_matches,
        "eligible": len(station_ids) == 1 and not route_context_matches,
    }


def _row_state(row: dict[str, Any], expected_mode: str, station_id: str) -> str:
    intent = row.get("expected_intent")
    assert isinstance(intent, dict), f"{row.get('id')}: expected_intent must be an object"
    assert intent.get("time_mode") == expected_mode, (
        f"{row.get('id')}: expected time_mode {expected_mode!r}, got {intent.get('time_mode')!r}"
    )

    if row.get("expected_tool") == "station_departures":
        arguments = row.get("expected_arguments")
        assert isinstance(arguments, dict), f"{row.get('id')}: pre-migration arguments missing"
        assert arguments.get("id") == station_id, (
            f"{row.get('id')}: station id in user and expected_arguments disagree"
        )
        assert isinstance(row.get("expected_normalized"), dict), (
            f"{row.get('id')}: pre-migration expected_normalized missing"
        )
        assert "missing_info" not in row and "expected_clarification" not in row, (
            f"{row.get('id')}: mixed pre/post migration shape"
        )
        return "pre"

    if row.get("expected_tool") is None:
        assert row.get("missing_info") is True, f"{row.get('id')}: post missing_info is not true"
        assert row.get("expected_clarification") == CLARIFICATION, (
            f"{row.get('id')}: post clarification differs from policy"
        )
        assert "expected_arguments" not in row and "expected_normalized" not in row, (
            f"{row.get('id')}: obsolete argument fields remain"
        )
        return "post"

    raise AssertionError(f"{row.get('id')}: unexpected expected_tool {row.get('expected_tool')!r}")


def _after_row(row: dict[str, Any]) -> dict[str, Any]:
    after: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"expected_arguments", "expected_normalized", "missing_info", "expected_clarification"}:
            continue
        if key == "expected_tool":
            after[key] = None
            after["missing_info"] = True
            after["expected_clarification"] = dict(CLARIFICATION)
        else:
            after[key] = value
    return after


def _validate_target_set(
    document: JsonlDocument,
    expected_modes: dict[str, str],
) -> dict[str, tuple[int, dict[str, Any]]]:
    locations = document.row_locations()
    present_targets = set(locations) & ALL_TARGET_IDS
    assert present_targets == set(expected_modes), (
        f"{document.path}: target id set mismatch; "
        f"missing={sorted(set(expected_modes) - present_targets)}, "
        f"unexpected={sorted(present_targets - set(expected_modes))}"
    )
    return locations


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


def _validate_existing_backup(
    backup_path: Path,
    current: JsonlDocument,
    expected_modes: dict[str, str],
) -> None:
    backup = JsonlDocument.read(backup_path)
    current_locations = current.row_locations()
    backup_locations = _validate_target_set(backup, expected_modes)
    assert len(backup.raw_lines) == len(current.raw_lines), f"{backup_path}: line count mismatch"

    target_indexes = {index for index, _ in current_locations.values() if _.get("id") in expected_modes}
    for index, (backup_raw, current_raw) in enumerate(zip(backup.raw_lines, current.raw_lines)):
        if index not in target_indexes:
            assert backup_raw == current_raw, f"{backup_path}:{index + 1}: non-target content mismatch"

    for row_id, mode in expected_modes.items():
        current_index, current_row = current_locations[row_id]
        backup_index, backup_row = backup_locations[row_id]
        assert current_index == backup_index, f"{backup_path}: row order changed for {row_id}"
        check = _machine_check(backup_row.get("user"))
        assert check["station_id_count"] == 1, f"{backup_path}: invalid station id count for {row_id}"
        assert _row_state(backup_row, mode, check["station_ids"][0]) == "pre", (
            f"{backup_path}: target row is not a pre-migration backup"
        )
        current_check = _machine_check(current_row.get("user"))
        current_state = _row_state(current_row, mode, current_check["station_ids"][0])
        if current_state == "pre":
            assert backup_row == current_row, f"{backup_path}: pre row differs for {row_id}"
        else:
            assert _after_row(backup_row) == current_row, f"{backup_path}: post row differs for {row_id}"


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# A+C Policy Label Migration",
        "",
        f"- Dry run: `{str(result['dry_run']).lower()}`",
        f"- Changed physical rows: {result['changed_row_count']}",
        f"- Logical target rows: {len(ALL_TARGET_IDS)}",
        f"- Derived mirror rows: {len(INDEPENDENT_TARGET_MODES)}",
        f"- Needs human judgment: {len(result['needs_human_judgment'])}",
        "",
        "## Needs human judgment",
        "",
    ]
    if result["needs_human_judgment"]:
        for item in result["needs_human_judgment"]:
            lines.append(
                f"- `{item['file']}:{item['line']}` `{item['id']}`: "
                f"route context `{json.dumps(item['machine_check']['route_context_matches'], ensure_ascii=False)}`"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Row changes", ""])
    if not result["changes"]:
        lines.append("No rows changed; all targets already use the A+C policy.")
    for item in result["changes"]:
        check = item["machine_check"]
        lines.extend(
            [
                f"### `{item['file']}:{item['line']}` — `{item['id']}`",
                "",
                f"- Station IDs: `{json.dumps(check['station_ids'], ensure_ascii=False)}`",
                f"- Route-context matches: `{json.dumps(check['route_context_matches'], ensure_ascii=False)}`",
                "- Machine decision: eligible",
                "",
                "Before:",
                "",
                "```json",
                json.dumps(item["before"], ensure_ascii=False, separators=(",", ":")),
                "```",
                "",
                "After:",
                "",
                "```json",
                json.dumps(item["after"], ensure_ascii=False, separators=(",", ":")),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def migrate(
    independent_path: Path = DEFAULT_INDEPENDENT,
    manual_path: Path = DEFAULT_MANUAL,
    derived_path: Path = DEFAULT_DERIVED,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    report_path: Path = DEFAULT_REPORT,
    dry_run: bool = False,
) -> dict[str, Any]:
    assert len(INDEPENDENT_TARGET_MODES) == 12
    assert len(MANUAL_TARGET_MODES) == 5
    assert len(ALL_TARGET_IDS) == 17
    specifications = (
        ("independent", independent_path, INDEPENDENT_TARGET_MODES),
        ("manual", manual_path, MANUAL_TARGET_MODES),
        ("derived", derived_path, INDEPENDENT_TARGET_MODES),
    )
    documents = {role: JsonlDocument.read(path) for role, path, _ in specifications}
    locations = {
        role: _validate_target_set(documents[role], modes)
        for role, _, modes in specifications
    }

    for row_id in INDEPENDENT_TARGET_MODES:
        independent_row = locations["independent"][row_id][1]
        derived_row = locations["derived"][row_id][1]
        assert independent_row == derived_row, f"derived before row differs from independent: {row_id}"

    replacements_by_role: dict[str, dict[int, dict[str, Any]]] = {
        role: {} for role, _, _ in specifications
    }
    changes: list[dict[str, Any]] = []
    needs_human_judgment: list[dict[str, Any]] = []
    for role, path, modes in specifications:
        for row_id, expected_mode in modes.items():
            index, row = locations[role][row_id]
            check = _machine_check(row.get("user"))
            assert check["station_id_count"] == 1, (
                f"{path}:{index + 1} {row_id}: expected exactly one station id, "
                f"got {check['station_ids']}"
            )
            state = _row_state(row, expected_mode, check["station_ids"][0])
            if check["route_context_matches"]:
                assert state == "pre", f"{path}:{index + 1} {row_id}: routed row was already migrated"
                needs_human_judgment.append(
                    {
                        "file": str(path),
                        "line": index + 1,
                        "id": row_id,
                        "machine_check": check,
                    }
                )
                continue
            if state == "post":
                continue
            after = _after_row(row)
            replacements_by_role[role][index] = after
            changes.append(
                {
                    "file": str(path),
                    "line": index + 1,
                    "id": row_id,
                    "before": row,
                    "after": after,
                    "machine_check": check,
                }
            )

    rendered = {
        role: documents[role].render_with(replacements_by_role[role])
        for role, _, _ in specifications
    }
    changed_roles = [role for role, _, _ in specifications if replacements_by_role[role]]
    assert len(changes) == sum(len(replacements_by_role[role]) for role in changed_roles)
    result: dict[str, Any] = {
        "dry_run": dry_run,
        "changed_files": [str(documents[role].path) for role in changed_roles],
        "changed_file_count": len(changed_roles),
        "changed_row_count": len(changes),
        "changes": changes,
        "needs_human_judgment": needs_human_judgment,
        "backup_dir": str(backup_dir),
        "report": str(report_path),
        "report_written": False,
    }

    if dry_run:
        return result

    for role, _, modes in specifications:
        if role not in changed_roles:
            continue
        document = documents[role]
        backup_path = backup_dir / document.path.name
        if backup_path.exists():
            _validate_existing_backup(backup_path, document, modes)

    backup_dir.mkdir(parents=True, exist_ok=True)
    for role, _, _ in specifications:
        if role not in changed_roles:
            continue
        document = documents[role]
        backup_path = backup_dir / document.path.name
        if not backup_path.exists():
            shutil.copy2(document.path, backup_path)

    for role in changed_roles:
        _atomic_write(documents[role].path, rendered[role])

    if changes or needs_human_judgment:
        _atomic_write(report_path, _markdown_report(result).encode("utf-8"))
        result["report_written"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=Path, default=DEFAULT_INDEPENDENT)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = migrate(
        independent_path=args.independent,
        manual_path=args.manual,
        derived_path=args.derived,
        backup_dir=args.backup_dir,
        report_path=args.report,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "dry_run": result["dry_run"],
                "changed_file_count": result["changed_file_count"],
                "changed_row_count": result["changed_row_count"],
                "needs_human_judgment": len(result["needs_human_judgment"]),
                "report_written": result["report_written"],
                "report": result["report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
