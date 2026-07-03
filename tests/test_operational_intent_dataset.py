import json
import unittest
from pathlib import Path

from local_tools import RESOLVE_ROUTE_REQUEST_TOOL
from transit_functiongemma.validation import validate_tool_call
from transit_functiongemma.toolcall import ToolCall


ROOT = Path(__file__).resolve().parents[1]


def rows(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class OperationalIntentDatasetTest(unittest.TestCase):
    def test_train_and_raw_eval_are_disjoint_and_complete(self) -> None:
        train = rows("data/raw/operational_intent_train.jsonl")
        evaluate = rows("data/eval/operational_intent_raw_100.jsonl")
        self.assertEqual(len(train), 100)
        self.assertEqual(len(evaluate), 100)
        self.assertFalse({item["user"] for item in train} & {item["user"] for item in evaluate})

    def test_route_intent_targets_are_schema_valid(self) -> None:
        for item in rows("data/raw/operational_intent_train.jsonl"):
            target = item["assistant"]
            if target.get("no_tool_call"):
                continue
            call = ToolCall(target["tool_name"], target["arguments"])
            validate_tool_call(call, [RESOLVE_ROUTE_REQUEST_TOOL])

    def test_eval_never_uses_legacy_expected_intent(self) -> None:
        for item in rows("data/eval/operational_intent_raw_100.jsonl"):
            self.assertNotIn("expected_intent", item)
            if item.get("expected_tool"):
                self.assertEqual(item["expected_tool"], "resolve_route_request")


if __name__ == "__main__":
    unittest.main()
