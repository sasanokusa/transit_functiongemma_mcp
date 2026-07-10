from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_manual100_arch_labels import (
    DEFAULT_SCHEMA,
    classify_row,
    derive_arguments,
    extract_surface_endpoints,
    load_schema,
    migrate,
)


def compact(row: dict[str, object]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def route_row(row_id: str, user: str, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": row_id,
        "reference_datetime": "2026-06-29 10:00 Asia/Tokyo",
        "user": user,
        "expected_tool": "suggest_stations",
        "expected_arguments": {"q": "legacy", "limit": 5},
        "tags": ["manual", "route_first_step"],
    }
    row.update(extra)
    return row


class Manual100ArchitectureMigrationTest(unittest.TestCase):
    def test_station_search_is_excluded_but_route_is_target(self) -> None:
        station = route_row("station", "東京を駅として拾って")
        self.assertEqual(classify_row(station).kind, "excluded")
        explicit = route_row("explicit", "町田から池袋まで、まず駅を検索して")
        self.assertEqual(classify_row(explicit).kind, "excluded")
        target = route_row("route", "町田から池袋まで、渋谷を避けて早めで")
        decision = classify_row(target)
        self.assertEqual(decision.kind, "target")
        self.assertEqual(decision.endpoints, ("町田", "池袋"))

        marked = route_row("marked", "東京発大阪行き")
        self.assertEqual(classify_row(marked).endpoints, ("東京", "大阪"))

    def test_ambiguous_route_is_sent_to_human_judgment(self) -> None:
        row = route_row("ambiguous", "駅をいくつか候補出して")
        self.assertEqual(classify_row(row).kind, "human")

    def test_endpoints_keep_surface_aliases_and_slots_use_schema(self) -> None:
        row = route_row(
            "surface",
            "明日8:30出発で品川から成田空港、早いやつ",
        )
        self.assertEqual(extract_surface_endpoints(row["user"])[0], ("品川", "成田空港"))
        required_fields, _ = load_schema(DEFAULT_SCHEMA)
        arguments = derive_arguments(row, required_fields)
        self.assertEqual(arguments["origin_text"], "品川")
        self.assertEqual(arguments["destination_text"], "成田空港")
        self.assertEqual(arguments["priority"], "fast")
        self.assertEqual(arguments["time_mode"], "departure_at")
        self.assertEqual(arguments["date"], "20260630")
        self.assertEqual(arguments["time"], "08:30")
        self.assertFalse(arguments["graphical"])
        self.assertEqual(len(arguments), 14)

    def test_apply_backups_and_preserves_non_target_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "manual_practical_100.jsonl"
            backup_dir = root / "pre_arch_migration_backup"
            report = root / "artifacts" / "MANUAL100_ARCH_MIGRATION.md"
            unchanged = (
                '{"id":"unchanged","user":"東京を駅として拾って",'
                '"expected_tool":"suggest_stations"}\n'
            ).encode("utf-8")
            target = compact(route_row("target", "町田から池袋まで、安く"))
            source.write_bytes(unchanged + target)
            before = source.read_bytes()

            dry = migrate(source, backup_dir, report, DEFAULT_SCHEMA, dry_run=True)
            self.assertEqual(dry["changed_row_count"], 1)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(backup_dir.exists())
            self.assertFalse(report.exists())

            applied = migrate(source, backup_dir, report, DEFAULT_SCHEMA, dry_run=False)
            self.assertEqual(applied["changed_row_count"], 1)
            self.assertEqual((backup_dir / source.name).read_bytes(), before)
            self.assertEqual(source.read_bytes().splitlines(keepends=True)[0], unchanged)
            self.assertTrue(report.exists())
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("町田から池袋まで、安く", report_text)
            self.assertIn("before", report_text)
            self.assertIn("after", report_text)

            migrated = migrate(source, backup_dir, report, DEFAULT_SCHEMA, dry_run=False)
            self.assertEqual(migrated["changed_row_count"], 0)
            self.assertFalse(migrated["report_written"])

    def test_non_station_tool_is_never_a_target(self) -> None:
        row = route_row("other", "町田から池袋まで")
        row["expected_tool"] = "plan_journey"
        self.assertEqual(classify_row(row).kind, "excluded")


if __name__ == "__main__":
    unittest.main()
