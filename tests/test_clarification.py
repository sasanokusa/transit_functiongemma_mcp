import unittest

from prepare_sft import convert_record
from transit_functiongemma.schemas import CLARIFICATION_TOOL_NAME


class ClarificationConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "id": "no-call",
            "reference_datetime": "2026-06-28 09:00 Asia/Tokyo",
            "user": "東京駅まで行きたい",
            "assistant": {
                "no_tool_call": True,
                "clarification": {
                    "missing": ["origin"],
                    "question": "出発地を教えてください。",
                },
            },
        }

    def test_default_stays_empty_assistant_turn(self) -> None:
        converted = convert_record(self.row, [], "baked")
        self.assertEqual(converted["messages"][-1]["content"], "")

    def test_optional_local_tool_target(self) -> None:
        converted = convert_record(self.row, [], "baked", clarification_tool=True)
        call = converted["messages"][-1]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], CLARIFICATION_TOOL_NAME)
        self.assertEqual(call["arguments"]["missing"], ["origin"])


if __name__ == "__main__":
    unittest.main()

