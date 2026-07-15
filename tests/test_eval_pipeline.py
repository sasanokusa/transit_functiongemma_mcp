import unittest

from evaluation.eval_pipeline import (
    FIXED_SCENARIOS,
    scenario_expectations_met,
    valid_expected_clarification,
)


class EvalPipelineTest(unittest.TestCase):
    def test_station_query_expectation_preserves_written_suffix(self) -> None:
        scenario = next(row for row in FIXED_SCENARIOS if row["id"] == "demo-01")
        self.assertEqual(scenario["expected_arguments"], {"q": "東京駅"})

    def test_clarification_requires_zero_mcp_and_exact_missing_slot(self) -> None:
        trace = {
            "mcp_calls": [],
            "tool_calls": [
                {
                    "name": "ask_clarification",
                    "arguments": {
                        "missing": ["origin"],
                        "question": "出発地を教えてください。",
                    },
                }
            ],
        }
        self.assertTrue(
            valid_expected_clarification(trace, "出発地を教えてください。", ["origin"])
        )
        trace["mcp_calls"] = [{"tool": "suggest_stations", "arguments": {}}]
        self.assertFalse(
            valid_expected_clarification(trace, "出発地を教えてください。", ["origin"])
        )

    def test_generic_no_call_does_not_pass_explicit_clarification_case(self) -> None:
        scenario = {
            "expected_no_call": True,
            "expected_clarification_missing": ["origin"],
        }
        self.assertFalse(
            scenario_expectations_met(
                scenario,
                {"mcp_calls": [], "tool_calls": []},
                "出発地と目的地が不足しています。どこからどこまで行きますか？",
                None,
            )
        )

    def test_wrong_or_irrelevant_clarification_fails(self) -> None:
        scenario = {
            "expected_no_call": True,
            "expected_clarification_missing": ["origin"],
        }
        for arguments in (
            {"missing": ["destination"], "question": "目的地を教えてください。"},
            {"missing": ["origin"], "question": "もう一度入力してください。"},
        ):
            with self.subTest(arguments=arguments):
                answer = arguments["question"]
                self.assertFalse(
                    scenario_expectations_met(
                        scenario,
                        {
                            "mcp_calls": [],
                            "tool_calls": [
                                {"name": "ask_clarification", "arguments": arguments}
                            ],
                        },
                        answer,
                        None,
                    )
                )


if __name__ == "__main__":
    unittest.main()
