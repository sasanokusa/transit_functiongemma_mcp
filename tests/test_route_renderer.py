import unittest

from transit_functiongemma.route_renderer import EMPTY_MESSAGE, render_answer, render_clarification


class RouteRendererTest(unittest.TestCase):
    def test_route_rendering_from_present_values(self) -> None:
        data = {
            "status": "ok",
            "raw_tool_name": "plan_journey",
            "query": {"avoid_station_texts": ["渋谷"], "via_station_texts": []},
            "routes": [{
                "summary": {"duration_min": 42, "fare_yen": 510, "transfers": 1},
                "constraint_check": {"avoid_satisfied": True},
                "legs": [{
                    "from": "町田", "to": "新宿", "line": "小田急線",
                    "departure_time": "08:10", "arrival_time": "08:47"
                }],
            }],
        }
        text = render_answer(data)
        self.assertIn("渋谷を避ける候補", text)
        self.assertIn("所要時間は42分", text)
        self.assertIn("08:10 町田 → 08:47 新宿（小田急線）", text)

    def test_missing_facts_are_not_invented(self) -> None:
        data = {
            "status": "ok", "raw_tool_name": "plan_journey", "query": {},
            "routes": [{"summary": {"duration_min": 12}, "legs": [{"from": "A", "to": "B"}]}],
        }
        text = render_answer(data)
        self.assertIn("所要時間は12分", text)
        self.assertNotIn("円", text)
        self.assertNotRegex(text, r"\d{2}:\d{2}")
        self.assertNotIn("乗換", text)

    def test_violated_avoid_message(self) -> None:
        data = {
            "status": "ok", "raw_tool_name": "plan_journey",
            "query": {"avoid_station_texts": ["渋谷"]},
            "routes": [{"summary": {}, "legs": [{"from": "渋谷", "to": "新宿"}],
                        "constraint_check": {"avoid_satisfied": False}}],
        }
        self.assertIn("完全に避ける候補は見つかりません", render_answer(data))

    def test_multiple_avoid_routes_use_plural_wording(self) -> None:
        route = {"summary": {}, "legs": [{"from": "A", "to": "B"}],
                 "constraint_check": {"avoid_satisfied": True}}
        data = {"status": "ok", "raw_tool_name": "plan_journey",
                "query": {"avoid_station_texts": ["渋谷"]}, "routes": [route, route]}
        self.assertIn("これらの候補では渋谷駅を通りません", render_answer(data))

    def test_station_suggestions(self) -> None:
        data = {"status": "ok", "raw_tool_name": "suggest_stations",
                "suggestion_type": "station", "suggestions": [{"name": "東京"}]}
        text = render_answer(data)
        self.assertIn("1. 東京駅", text)
        self.assertIn("どの駅を使いますか", text)

    def test_station_suffix_is_not_duplicated(self) -> None:
        data = {"status": "ok", "raw_tool_name": "station_departures",
                "station": {"name": "東京駅"}, "departures": [{"time": "08:10"}]}
        text = render_answer(data)
        self.assertIn("東京駅の発車情報", text)
        self.assertNotIn("東京駅駅", text)

    def test_duplicate_suggestion_uses_existing_source_label(self) -> None:
        data = {"status": "ok", "raw_tool_name": "suggest_stations",
                "suggestion_type": "station", "suggestions": [
                    {"name": "東京", "source_label": "中央線"},
                    {"name": "東京", "source_label": "山手線"},
                ]}
        text = render_answer(data)
        self.assertIn("東京駅（中央線）", text)
        self.assertIn("東京駅（山手線）", text)

    def test_preferred_line_filters_returned_routes(self) -> None:
        data = {
            "status": "ok", "raw_tool_name": "plan_journey",
            "query": {"preferred_line_texts": ["京浜東北線"]},
            "routes": [
                {"summary": {}, "legs": [{"from": "横浜", "to": "上野", "line": "東海道線"}],
                 "constraint_check": {"line_satisfied": False}},
                {"summary": {}, "legs": [{"from": "横浜", "to": "上野", "line": "京浜東北線"}],
                 "constraint_check": {"line_satisfied": True}},
            ],
        }
        text = render_answer(data)
        self.assertIn("京浜東北線を使う候補だけ", text)
        self.assertIn("（京浜東北線）", text)
        self.assertNotIn("（東海道線）", text)

    def test_empty_and_clarification(self) -> None:
        self.assertEqual(render_answer({"status": "empty", "raw_tool_name": "plan_journey"}), EMPTY_MESSAGE)
        self.assertEqual(render_clarification(["origin"]), "出発地が不足しています。どこから出発しますか？")

    def test_consecutive_walk_legs_are_merged(self) -> None:
        data = {
            "status": "ok",
            "query": {},
            "routes": [{
                "summary": {"duration_min": 10},
                "legs": [
                    {"type": "walk", "from": "東京駅", "to": "丸の内口", "departure_time": "10:00", "arrival_time": "10:02", "duration_min": 2},
                    {"type": "walk", "from": "丸の内口", "to": "東京", "departure_time": "10:02", "arrival_time": "10:04", "duration_min": 2},
                ],
            }],
        }
        answer = render_answer(data)
        self.assertIn("10:00 東京駅 → 10:04 東京", answer)
        self.assertNotIn("丸の内口", answer)


if __name__ == "__main__":
    unittest.main()
