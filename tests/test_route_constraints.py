import unittest

from route_constraints import (
    apply_route_constraints,
    evaluate_route_constraints,
    normalize_station_name,
)
from route_renderer import render_answer


class RouteConstraintsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.route = {
            "from": {"id": "feed:machida", "name": "町田駅"},
            "to": {"id": "feed:ikebukuro", "name": "池袋"},
            "legs": [
                {"from": "町田", "to": "新宿", "stops": [{"name": "渋谷駅"}]},
                {"from": "新宿", "to": "池袋"},
            ],
        }

    def test_station_suffix_and_width_normalization(self) -> None:
        self.assertEqual(normalize_station_name(" 渋谷駅 "), normalize_station_name("渋谷"))
        self.assertEqual(normalize_station_name("ＴＯＫＹＯ駅"), "tokyo")

    def test_no_substring_fuzzy_match(self) -> None:
        self.assertNotEqual(normalize_station_name("新宿"), normalize_station_name("新宿三丁目"))
        result = evaluate_route_constraints(self.route, avoid_station_texts=["新宿三丁目"])
        self.assertTrue(result["avoid_satisfied"])

    def test_avoid_violation(self) -> None:
        result = evaluate_route_constraints(self.route, avoid_station_texts=["渋谷"])
        self.assertFalse(result["avoid_satisfied"])
        self.assertEqual(result["violated_avoid_station_texts"], ["渋谷"])

    def test_avoid_satisfied_and_via_satisfied(self) -> None:
        result = evaluate_route_constraints(
            self.route, avoid_station_texts=["東京"], via_station_texts=["新宿駅"]
        )
        self.assertTrue(result["avoid_satisfied"])
        self.assertTrue(result["via_satisfied"])

    def test_id_takes_exact_match(self) -> None:
        result = evaluate_route_constraints(self.route, via_station_ids=["feed:ikebukuro"])
        self.assertTrue(result["via_satisfied"])

    def test_normalized_leg_endpoint_id(self) -> None:
        route = {"legs": [{"from": "町田", "from_id": "feed:machida", "to": "池袋"}]}
        result = evaluate_route_constraints(route, avoid_station_ids=["feed:machida"])
        self.assertFalse(result["avoid_satisfied"])

    def test_apply_returns_copy(self) -> None:
        data = {"query": {"avoid_station_texts": ["渋谷"]}, "routes": [self.route]}
        output = apply_route_constraints(data)
        self.assertNotIn("constraint_check", data["routes"][0])
        self.assertFalse(output["routes"][0]["constraint_check"]["avoid_satisfied"])

    def test_preferred_line_is_checked_against_leg(self) -> None:
        route = {"legs": [{"line": "京浜東北線（北行（大宮方面））"}]}
        matched = evaluate_route_constraints(route, preferred_line_texts=["京浜東北線"])
        missed = evaluate_route_constraints(route, preferred_line_texts=["東急東横線"])
        self.assertTrue(matched["line_satisfied"])
        self.assertFalse(missed["line_satisfied"])

    def test_avoided_line_is_rejected(self) -> None:
        route = {"legs": [{"line": "JR山手線", "from": "東京", "to": "新宿"}]}
        result = evaluate_route_constraints(
            route, avoid_line_texts=["山手線"]
        )
        self.assertFalse(result["avoid_line_satisfied"])
        self.assertEqual(result["violated_avoid_line_texts"], ["山手線"])

    def test_arrival_filter_rejects_walk_route_departing_at_requested_time(self) -> None:
        data = apply_route_constraints(
            {
                "status": "ok",
                "query": {"time_mode": "arrive_by", "time": "16:00"},
                "routes": [
                    {
                        "summary": {
                            "departure_time": "16:00",
                            "arrival_time": "17:36",
                        },
                        "legs": [
                            {
                                "type": "walk",
                                "from": "東京駅",
                                "to": "新宿駅",
                                "departure_time": "16:00",
                                "arrival_time": "17:36",
                            }
                        ],
                    },
                    {
                        "summary": {
                            "departure_time": "15:33",
                            "arrival_time": "15:56",
                        },
                        "legs": [
                            {
                                "type": "train",
                                "from": "東京",
                                "to": "新宿",
                                "line": "中央線快速",
                                "departure_time": "15:38",
                                "arrival_time": "15:52",
                            }
                        ],
                    },
                ],
            }
        )
        self.assertFalse(data["routes"][0]["constraint_check"]["time_satisfied"])
        self.assertTrue(data["routes"][1]["constraint_check"]["time_satisfied"])
        answer = render_answer(data, max_routes=1)
        self.assertIn("15:38 東京", answer)
        self.assertNotIn("16:00 東京駅", answer)

    def test_departure_filter_rejects_route_before_requested_time(self) -> None:
        data = apply_route_constraints(
            {
                "status": "ok",
                "query": {"time_mode": "departure_at", "time": "16:00"},
                "routes": [
                    {"summary": {"departure_time": "15:50", "arrival_time": "16:10"}},
                    {"summary": {"departure_time": "16:05", "arrival_time": "16:25"}},
                ],
            }
        )
        self.assertFalse(data["routes"][0]["constraint_check"]["time_satisfied"])
        self.assertTrue(data["routes"][1]["constraint_check"]["time_satisfied"])


if __name__ == "__main__":
    unittest.main()
