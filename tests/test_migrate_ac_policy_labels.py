from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_ac_policy_labels import (
    CLARIFICATION,
    INDEPENDENT_TARGET_MODES,
    MANUAL_TARGET_MODES,
    _machine_check,
    migrate,
)


def _station_id(row_id: str) -> str:
    return f"demo-feed:{row_id.replace('-', '_')}"


def _pre_row(row_id: str, mode: str) -> dict[str, object]:
    station_id = _station_id(row_id)
    phrase = "始発を確認して" if mode == "first_train" else "今日の終電発車を見たい"
    return {
        "id": row_id,
        "reference_datetime": "2026-06-29 10:00 Asia/Tokyo",
        "user": f"{station_id} の{phrase}",
        "expected_tool": "station_departures",
        "expected_arguments": {"id": station_id, "date": "20260629"},
        "expected_normalized": {"date": "20260629"},
        "tags": ["departures", mode],
        "expected_intent": {"time_mode": mode},
    }


def _compact(row: dict[str, object]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"


class AcPolicyMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.independent = self.root / "independent_holdout_300.jsonl"
        self.manual = self.root / "manual_practical_100.jsonl"
        self.derived = self.root / "eval_nonroute_215_reaudited_dataset.jsonl"
        self.backup_dir = self.root / "backups"
        self.report = self.root / "AC_POLICY_LABEL_MIGRATION.md"
        self._write_fixtures()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_fixtures(self) -> None:
        sentinel = '{ "id": "sentinel", "expected_tool": null }\n'
        independent_rows = [
            _pre_row(row_id, mode) for row_id, mode in INDEPENDENT_TARGET_MODES.items()
        ]
        manual_rows = [_pre_row(row_id, mode) for row_id, mode in MANUAL_TARGET_MODES.items()]
        self.independent.write_text(
            sentinel + "".join(_compact(row) for row in independent_rows), encoding="utf-8"
        )
        self.manual.write_text(
            sentinel + "".join(_compact(row) for row in manual_rows), encoding="utf-8"
        )
        self.derived.write_text(
            sentinel + "".join(_compact(row) for row in independent_rows), encoding="utf-8"
        )

    def _migrate(self, dry_run: bool = False) -> dict[str, object]:
        return migrate(
            independent_path=self.independent,
            manual_path=self.manual,
            derived_path=self.derived,
            backup_dir=self.backup_dir,
            report_path=self.report,
            dry_run=dry_run,
        )

    @staticmethod
    def _rows(path: Path) -> dict[str, dict[str, object]]:
        return {
            row["id"]: row
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }

    def test_dry_run_reports_all_physical_changes_without_writing(self) -> None:
        before = {path: path.read_bytes() for path in (self.independent, self.manual, self.derived)}

        result = self._migrate(dry_run=True)

        self.assertEqual(result["changed_row_count"], 29)
        self.assertEqual(result["changed_file_count"], 3)
        self.assertEqual(result["needs_human_judgment"], [])
        self.assertFalse(result["report_written"])
        self.assertFalse(self.backup_dir.exists())
        self.assertFalse(self.report.exists())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_apply_preserves_non_target_bytes_and_second_run_is_idempotent(self) -> None:
        originals = {path: path.read_bytes() for path in (self.independent, self.manual, self.derived)}

        result = self._migrate()

        self.assertEqual(result["changed_row_count"], 29)
        self.assertTrue(result["report_written"])
        report_text = self.report.read_text(encoding="utf-8")
        self.assertIn("Logical target rows: 17", report_text)
        self.assertIn("Derived mirror rows: 12", report_text)
        for path, original in originals.items():
            self.assertEqual((self.backup_dir / path.name).read_bytes(), original)
            self.assertTrue(path.read_text(encoding="utf-8").startswith(
                '{ "id": "sentinel", "expected_tool": null }\n'
            ))
        for path, modes in (
            (self.independent, INDEPENDENT_TARGET_MODES),
            (self.manual, MANUAL_TARGET_MODES),
            (self.derived, INDEPENDENT_TARGET_MODES),
        ):
            rows = self._rows(path)
            for row_id, mode in modes.items():
                row = rows[row_id]
                self.assertIsNone(row["expected_tool"])
                self.assertIs(row["missing_info"], True)
                self.assertEqual(row["expected_clarification"], CLARIFICATION)
                self.assertNotIn("expected_arguments", row)
                self.assertNotIn("expected_normalized", row)
                self.assertEqual(row["expected_intent"]["time_mode"], mode)
        independent_rows = self._rows(self.independent)
        derived_rows = self._rows(self.derived)
        for row_id in INDEPENDENT_TARGET_MODES:
            self.assertEqual(independent_rows[row_id], derived_rows[row_id])

        migrated_bytes = {
            path: path.read_bytes() for path in (self.independent, self.manual, self.derived)
        }
        report_bytes = self.report.read_bytes()
        second = self._migrate()

        self.assertEqual(second["changed_row_count"], 0)
        self.assertFalse(second["report_written"])
        self.assertEqual(self.report.read_bytes(), report_bytes)
        for path, content in migrated_bytes.items():
            self.assertEqual(path.read_bytes(), content)

    def test_route_context_is_not_migrated_and_is_reported(self) -> None:
        for path in (self.independent, self.derived):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                if row["id"] == "ind-113":
                    row["user"] += "、ここから東京まで"
            path.write_text("".join(_compact(row) for row in rows), encoding="utf-8")

        result = self._migrate()

        self.assertEqual(result["changed_row_count"], 27)
        self.assertEqual(len(result["needs_human_judgment"]), 2)
        self.assertEqual(
            {item["id"] for item in result["needs_human_judgment"]}, {"ind-113"}
        )
        self.assertEqual(
            self._rows(self.independent)["ind-113"]["expected_tool"], "station_departures"
        )
        self.assertEqual(
            self._rows(self.derived)["ind-113"]["expected_tool"], "station_departures"
        )
        self.assertIn("ind-113", self.report.read_text(encoding="utf-8"))

    def test_missing_fixed_target_fails_before_any_write(self) -> None:
        lines = self.manual.read_text(encoding="utf-8").splitlines(keepends=True)
        self.manual.write_text(
            "".join(line for line in lines if '"man-038"' not in line), encoding="utf-8"
        )
        before = self.independent.read_bytes()

        with self.assertRaisesRegex(AssertionError, "target id set mismatch"):
            self._migrate()

        self.assertEqual(self.independent.read_bytes(), before)
        self.assertFalse(self.backup_dir.exists())
        self.assertFalse(self.report.exists())

    def test_derived_before_must_match_independent(self) -> None:
        text = self.derived.read_text(encoding="utf-8").replace(
            "今日の終電発車を見たい", "今日の最終便を見たい", 1
        )
        self.derived.write_text(text, encoding="utf-8")
        before = self.independent.read_bytes()

        with self.assertRaisesRegex(AssertionError, "derived before row differs"):
            self._migrate()

        self.assertEqual(self.independent.read_bytes(), before)
        self.assertFalse(self.backup_dir.exists())

    def test_station_id_count_must_be_exactly_one(self) -> None:
        for path in (self.independent, self.derived):
            text = path.read_text(encoding="utf-8").replace(
                "demo-feed:ind_113 の", "demo-feed:ind_113 と demo-feed:tokyo の", 1
            )
            path.write_text(text, encoding="utf-8")
        before = self.independent.read_bytes()

        with self.assertRaisesRegex(AssertionError, "expected exactly one station id"):
            self._migrate()

        self.assertEqual(self.independent.read_bytes(), before)
        self.assertFalse(self.backup_dir.exists())

    def test_existing_backup_mismatch_fails_without_overwrite(self) -> None:
        self.backup_dir.mkdir()
        backup_path = self.backup_dir / self.independent.name
        backup_path.write_bytes(
            self.independent.read_bytes().replace(b'"sentinel"', b'"different"', 1)
        )
        backup_before = backup_path.read_bytes()
        source_before = self.independent.read_bytes()

        with self.assertRaisesRegex(AssertionError, "non-target content mismatch"):
            self._migrate()

        self.assertEqual(backup_path.read_bytes(), backup_before)
        self.assertEqual(self.independent.read_bytes(), source_before)
        self.assertFalse(self.report.exists())

    def test_route_pair_variants_are_detected_without_overmatching_station_only_text(self) -> None:
        route_users = (
            "demo-feed:test より東京まで終電で行きたい",
            "demo-feed:test 出発地は東京、目的地は大阪",
            "demo-feed:test 起点は東京、行き先は大阪",
            "demo-feed:test FROM Tokyo TO Osaka",
            "demo-feed:test origin Tokyo destination Osaka",
        )
        station_only_users = (
            "demo-feed:test の今日の終電発車を見たい",
            "この駅ID demo-feed:test の始発を確認して",
        )

        for user in route_users:
            with self.subTest(user=user):
                check = _machine_check(user)
                self.assertEqual(check["station_id_count"], 1)
                self.assertTrue(check["route_context_matches"])
                self.assertFalse(check["eligible"])
        for user in station_only_users:
            with self.subTest(user=user):
                check = _machine_check(user)
                self.assertEqual(check["station_id_count"], 1)
                self.assertEqual(check["route_context_matches"], [])
                self.assertTrue(check["eligible"])


if __name__ == "__main__":
    unittest.main()
