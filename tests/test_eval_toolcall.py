import unittest

from evaluation.eval_toolcall import evaluate


TOOLS = [
    {
        "name": "suggest_stations",
        "inputSchema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
    {"name": "list_feeds", "inputSchema": {"type": "object", "properties": {}}},
]


class DetailedEvaluationTest(unittest.TestCase):
    def test_metrics_and_confusion(self) -> None:
        rows = [
            {
                "id": "correct",
                "expected_tool": "suggest_stations",
                "expected_arguments": {"q": "東京"},
            },
            {"id": "wrong", "expected_tool": "suggest_stations", "expected_arguments": {"q": "大阪"}},
            {"id": "no-call", "expected_tool": None, "missing_info": True},
        ]
        predictions = {
            "correct": "<start_function_call>call:suggest_stations{q:<escape>東京<escape>}<end_function_call>",
            "wrong": "<start_function_call>call:list_feeds{}<end_function_call>",
            "no-call": "<end_of_turn>",
        }
        report = evaluate(rows, predictions, TOOLS)
        self.assertEqual(report["metrics"]["parse_success_rate"], 1.0)
        self.assertEqual(report["metrics"]["tool_name_accuracy"], 0.5)
        self.assertEqual(report["metrics"]["expected_arguments_match_rate"], 0.5)
        self.assertEqual(report["metrics"]["no_call_when_missing_info_rate"], 1.0)
        self.assertEqual(len(report["failures"]), 1)

    def test_clarification_counts_as_no_call(self) -> None:
        rows = [{"id": "missing", "expected_tool": None, "missing_info": True}]
        predictions = {
            "missing": (
                "<start_function_call>call:ask_clarification{missing:[<escape>origin<escape>],"
                "question:<escape>出発地を教えてください。<escape>}<end_function_call>"
            )
        }
        report = evaluate(rows, predictions, TOOLS, clarification_tool=True)
        self.assertEqual(report["metrics"]["no_call_when_missing_info_rate"], 1.0)

    def test_explicit_clarification_target_is_scored_as_safe_no_call(self) -> None:
        arguments = {
            "missing": ["origin"],
            "question": "出発地を教えてください。",
        }
        rows = [
            {
                "id": "missing",
                "expected_tool": "ask_clarification",
                "expected_arguments": arguments,
            }
        ]
        predictions = {
            "missing": (
                "<start_function_call>call:ask_clarification{missing:[<escape>origin<escape>],"
                "question:<escape>出発地を教えてください。<escape>}<end_function_call>"
            )
        }
        report = evaluate(rows, predictions, TOOLS, clarification_tool=True)
        self.assertEqual(report["metrics"]["overall_class_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["clarification_accuracy"], 1.0)

    def test_unsupported_expected_normalized_is_excluded_from_datetime(self) -> None:
        tools = [
            {
                "name": "station_departures",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["id"],
                },
            }
        ]
        rows = [
            {
                "id": "last-train-board",
                "expected_tool": "station_departures",
                "expected_arguments": {"id": "demo-feed:tokyo", "date": "20260629"},
                "expected_normalized": {"date": "20260629", "type": "last"},
            }
        ]
        predictions = {
            "last-train-board": (
                "<start_function_call>call:station_departures{"
                "id:<escape>demo-feed:tokyo<escape>,date:<escape>20260629<escape>"
                "}<end_function_call>"
            )
        }
        report = evaluate(rows, predictions, tools)
        self.assertEqual(report["counts"]["unsupported_expected_normalized"], 1)
        self.assertEqual(report["metrics"]["datetime_normalization_success_rate"], 1.0)

    def test_intent_time_mode_metric_does_not_affect_semantic_success(self) -> None:
        tools = [
            {
                "name": "station_departures",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["id"],
                },
            }
        ]
        rows = [
            {
                "id": "last-train-board",
                "expected_tool": "station_departures",
                "expected_arguments": {"id": "demo-feed:tokyo", "date": "20260629"},
                "expected_intent": {"time_mode": "last_train"},
            }
        ]
        predictions = {
            "last-train-board": (
                "<start_function_call>call:station_departures{"
                "id:<escape>demo-feed:tokyo<escape>,date:<escape>20260629<escape>"
                "}<end_function_call>"
            )
        }
        report = evaluate(rows, predictions, tools)
        self.assertEqual(report["metrics"]["semantic_success_rate"], 1.0)
        self.assertEqual(report["metrics"]["intent_time_mode_accuracy"], 0.0)

    def test_expected_clarification_excludes_valid_call_from_time_mode_metric(self) -> None:
        rows = [
            {
                "id": "station-only-last-train",
                "expected_tool": None,
                "missing_info": True,
                "expected_clarification": {
                    "missing": ["destination"],
                    "question": "目的地を教えてください。",
                },
                "expected_intent": {"time_mode": "last_train"},
            }
        ]
        clarification = (
            "<start_function_call>call:ask_clarification{"
            "missing:[<escape>destination<escape>],"
            "question:<escape>目的地を教えてください。<escape>}<end_function_call>"
        )

        clarification_report = evaluate(
            rows,
            {"station-only-last-train": clarification},
            TOOLS,
            clarification_tool=True,
        )
        self.assertEqual(clarification_report["counts"]["intent_time_mode_cases"], 0)
        self.assertIsNone(
            clarification_report["metrics"]["intent_time_mode_accuracy"]
        )

        no_call_report = evaluate(
            rows,
            {"station-only-last-train": "<end_of_turn>"},
            TOOLS,
            clarification_tool=True,
        )
        self.assertEqual(no_call_report["counts"]["intent_time_mode_cases"], 1)
        self.assertEqual(no_call_report["metrics"]["intent_time_mode_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
