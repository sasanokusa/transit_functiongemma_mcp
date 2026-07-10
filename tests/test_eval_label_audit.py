import unittest

from evaluation.audit_expected_arguments import audit_q_labels, audit_schema_violations


class EvalLabelAuditTest(unittest.TestCase):
    def test_schema_audit_classifies_key_type_range_and_pattern(self) -> None:
        tools = {
            "suggest_stations": {
                "name": "suggest_stations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                    },
                },
            },
            "plan_journey": {
                "name": "plan_journey",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "pattern": "^\\d{8}$"},
                    },
                },
            },
        }
        rows = [
            {
                "_audit_file": "tmp.jsonl",
                "_audit_line": 1,
                "id": "bad-station",
                "expected_tool": "suggest_stations",
                "expected_arguments": {"q": 123, "limit": 31, "extra": True},
            },
            {
                "_audit_file": "tmp.jsonl",
                "_audit_line": 2,
                "id": "bad-date",
                "expected_tool": "plan_journey",
                "expected_arguments": {"date": "2026-07-01"},
            },
        ]

        violations = audit_schema_violations(rows, tools)

        self.assertEqual(
            {item["kind"] for item in violations},
            {"unknown_key", "type", "range", "pattern"},
        )
        self.assertEqual(
            {(item["id"], item["key"]) for item in violations if item["kind"] == "unknown_key"},
            {("bad-station", "extra")},
        )

    def test_q_audit_finds_station_suffix_majority_and_minority_rows(self) -> None:
        tools = {
            "suggest_stations": {
                "name": "suggest_stations",
                "inputSchema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "default": 10}},
                },
            },
            "suggest_places": {
                "name": "suggest_places",
                "inputSchema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "default": 10}},
                },
            },
        }
        rows = [
            {
                "_audit_file": "tmp.jsonl",
                "_audit_line": 1,
                "id": "keeps-suffix",
                "user": "東京駅を検索して",
                "expected_tool": "suggest_stations",
                "expected_arguments": {"q": "東京駅", "limit": 5},
            },
            {
                "_audit_file": "tmp.jsonl",
                "_audit_line": 2,
                "id": "drops-suffix",
                "user": "横浜駅を検索して",
                "expected_tool": "suggest_stations",
                "expected_arguments": {"q": "横浜", "limit": 5},
            },
            {
                "_audit_file": "tmp.jsonl",
                "_audit_line": 3,
                "id": "keeps-suffix-2",
                "user": "上野駅の候補",
                "expected_tool": "suggest_stations",
                "expected_arguments": {"q": "上野駅"},
            },
        ]

        audit = audit_q_labels(rows, tools)

        self.assertEqual(audit["user_station_suffix_majority"], "with_eki")
        self.assertEqual(
            [item["id"] for item in audit["minority_user_station_suffix_records"]],
            ["drops-suffix"],
        )
        self.assertEqual(
            audit["limit_distribution_by_tool"]["suggest_stations"],
            {"5": 2, "omitted(default=10)": 1},
        )


if __name__ == "__main__":
    unittest.main()
