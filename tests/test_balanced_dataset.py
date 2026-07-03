import unittest

from datagen.generate_balanced_synthetic_dataset import CLASSES, build_class, input_key


class BalancedDatasetTest(unittest.TestCase):
    def test_exact_class_size_and_unique_inputs(self) -> None:
        for index, class_name in enumerate(CLASSES):
            rows = build_class(class_name, 12, 100 + index, False)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({input_key(row) for row in rows}), 12)
            for row in rows:
                target = row["assistant"]
                actual = "no_tool_call" if target.get("no_tool_call") else target["tool_name"]
                self.assertEqual(actual, class_name)

    def test_route_targets_only_after_both_ids_are_resolved(self) -> None:
        for class_name in ("plan_journey", "plan_route_map"):
            for row in build_class(class_name, 5, 200, False):
                history = row["history"]
                responses = [message for message in history if message["role"] == "tool"]
                self.assertEqual(len(responses), 2)
                arguments = row["assistant"]["arguments"]
                self.assertTrue(arguments["from"].startswith("demo-feed:"))
                self.assertTrue(arguments["to"].startswith("demo-feed:"))

    def test_no_call_has_clarification_metadata(self) -> None:
        for row in build_class("no_tool_call", 8, 300, False):
            clarification = row["assistant"]["clarification"]
            self.assertTrue(clarification["missing"])
            self.assertTrue(clarification["question"])


if __name__ == "__main__":
    unittest.main()

