import unittest

from transit_functiongemma.route_reranker import rerank_routes


class PriorityRerankerTest(unittest.TestCase):
    def test_few_transfers_then_duration(self) -> None:
        data = {
            "query": {"priority": "few_transfers"},
            "routes": [
                {"rank": 1, "summary": {"transfers": 2, "duration_min": 20}},
                {"rank": 2, "summary": {"transfers": 0, "duration_min": 30}},
                {"rank": 3, "summary": {"transfers": 0, "duration_min": 25}},
            ],
        }
        result = rerank_routes(data)
        self.assertEqual([r["source_rank"] for r in result["routes"]], [3, 2, 1])
        self.assertTrue(result["ranking"]["applied"])

    def test_less_walk_uses_walk_duration(self) -> None:
        data = {
            "query": {"priority": "less_walk"},
            "routes": [
                {"rank": 1, "summary": {"walk_duration_min": 8, "transfers": 0}},
                {"rank": 2, "summary": {"walk_duration_min": 2, "transfers": 1}},
            ],
        }
        self.assertEqual(rerank_routes(data)["routes"][0]["source_rank"], 2)

    def test_missing_fare_is_explicit(self) -> None:
        result = rerank_routes(
            {"query": {"priority": "cheap"}, "routes": [{"rank": 1, "summary": {}}]}
        )
        self.assertFalse(result["ranking"]["applied"])
        self.assertIn("保証できません", result["ranking"]["message"])
