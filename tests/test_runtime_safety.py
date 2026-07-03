import unittest
import json
import tempfile
from pathlib import Path

from transit_functiongemma.mcp import schema_hash
from web_api import BehaviorLogger, SlidingWindowRateLimiter


class RuntimeSafetyTest(unittest.TestCase):
    def test_schema_hash_is_order_insensitive_for_object_keys(self) -> None:
        first = [{"name": "x", "inputSchema": {"type": "object", "properties": {}}}]
        second = [{"inputSchema": {"properties": {}, "type": "object"}, "name": "x"}]
        self.assertEqual(schema_hash(first), schema_hash(second))

    def test_rate_limiter_rejects_after_limit(self) -> None:
        limiter = SlidingWindowRateLimiter(2, 60)
        self.assertTrue(limiter.allow("client"))
        self.assertTrue(limiter.allow("client"))
        self.assertFalse(limiter.allow("client"))

    def test_behavior_log_redacts_coordinates_and_excludes_raw_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = BehaviorLogger(
                enabled=True,
                directory=directory,
                retention_days=45,
                save_query=True,
                save_answer=True,
            )
            logger.write(
                "request-123",
                200,
                "35.6812,139.7671から16:00着",
                {"ok": True, "kind": "answer", "answer": "16:00 東京駅", "elapsed_ms": 10},
                {
                    "effective_prompt": "35.6812,139.7671から16:00着",
                    "trace": {
                        "tool_calls": [
                            {
                                "name": "plan_journey",
                                "arguments": {
                                    "from": "geo:35.6812,139.7671",
                                    "to": "geo:35.7,139.8",
                                    "time": "16:00",
                                    "type": "arrival",
                                },
                            }
                        ],
                        "normalized_result": {"must_not_be_logged": True},
                        "rendered_answer": "16:00 東京駅",
                    },
                },
            )
            path = next(Path(directory).glob("*.jsonl"))
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("35.6812", json.dumps(event, ensure_ascii=False))
            self.assertEqual(event["trace"]["tool_calls"][0]["arguments"]["type"], "arrival")
            self.assertNotIn("normalized_result", event["trace"])

    def test_behavior_log_deletes_expired_daily_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "2000-01-01.jsonl"
            old.write_text("{}\n", encoding="utf-8")
            BehaviorLogger(enabled=True, directory=directory, retention_days=45)
            self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
