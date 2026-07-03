import unittest

from transit_functiongemma.japanese import (
    bind_normalized_tool_call,
    normalize_japanese_prompt,
)
from transit_functiongemma.toolcall import ToolCall


class JapaneseNormalizerTest(unittest.TestCase):
    def test_route_keeps_station_names_and_extracts_constraints(self) -> None:
        result = normalize_japanese_prompt(
            "品川シーサイドから渋谷まで、りんかい線で早めに。地図はいらない",
            "2026-06-29 09:00 Asia/Tokyo",
            semantic_fallback=True,
        )
        self.assertIn("intent=route", result)
        self.assertIn("origin=品川シーサイド", result)
        self.assertIn("destination=渋谷", result)
        self.assertIn("preferred_lines=りんかい線", result)
        self.assertIn("priority=fastest", result)
        self.assertIn("map=false", result)

    def test_coordinate_wording_is_canonicalized(self) -> None:
        result = normalize_japanese_prompt(
            "緯度35.6586 経度139.7454の近くの駅", semantic_fallback=True
        )
        self.assertIn("intent=reverse_geocode", result)
        self.assertIn("lat=35.6586", result)
        self.assertIn("lon=139.7454", result)

    def test_north_east_coordinate_wording(self) -> None:
        result = normalize_japanese_prompt(
            "北緯35.6812 東経139.7671の最寄り", semantic_fallback=True
        )
        self.assertIn("intent=reverse_geocode", result)

    def test_station_id_departures_and_datetime(self) -> None:
        result = normalize_japanese_prompt(
            "jp:tokyo:station:123 の明日9時の発車標",
            "2026-06-29 08:00 Asia/Tokyo",
            semantic_fallback=True,
        )
        self.assertIn("intent=station_departures", result)
        self.assertIn("station_id=jp:tokyo:station:123", result)
        self.assertIn("date=20260630", result)
        self.assertIn("time=9:00", result)

    def test_incomplete_route_requests_clarification(self) -> None:
        result = normalize_japanese_prompt("東京駅まで行きたい", semantic_fallback=True)
        self.assertIn("intent=clarification", result)
        self.assertIn("missing=origin", result)

    def test_unknown_text_is_not_rewritten(self) -> None:
        self.assertEqual(normalize_japanese_prompt("こんにちは"), "こんにちは")

    def test_default_is_notation_only(self) -> None:
        result = normalize_japanese_prompt("東京から新宿まで、１６：００着　")
        self.assertEqual(result, "東京から新宿まで、16:00着")
        self.assertNotIn("intent=", result)

    def test_colloquial_place_candidate_keeps_the_proper_noun(self) -> None:
        result = normalize_japanese_prompt(
            "日本武道館って場所の候補ある？", semantic_fallback=True
        )
        self.assertIn("intent=suggest_places", result)
        self.assertIn("query=日本武道館", result)

    def test_place_as_search_keeps_the_object_before_place(self) -> None:
        result = normalize_japanese_prompt(
            "東京タワーを場所として探して", semantic_fallback=True
        )
        self.assertIn("intent=suggest_places", result)
        self.assertIn("query=東京タワー", result)

    def test_extracted_coordinate_values_override_model_rounding(self) -> None:
        bound = bind_normalized_tool_call(
            ToolCall("suggest_places", {"q": "35.5"}),
            "35.5075, 139.6175から駅を逆引き",
            semantic_fallback=True,
        )
        self.assertEqual(bound.name, "reverse_geocode")
        self.assertEqual(bound.arguments["lat"], 35.5075)
        self.assertEqual(bound.arguments["radiusMeters"], 200)

    def test_incomplete_route_suppresses_call(self) -> None:
        bound = bind_normalized_tool_call(
            ToolCall("suggest_stations", {"q": "上野"}),
            "上野までお願い",
            semantic_fallback=True,
        )
        self.assertIsNone(bound)

    def test_arrival_destination_only_needs_origin(self) -> None:
        result = normalize_japanese_prompt(
            "明日9時に品川に着きたい", semantic_fallback=True
        )
        self.assertIn("intent=clarification", result)
        self.assertIn("missing=origin", result)

    def test_route_map_is_bound_after_both_station_resolutions(self) -> None:
        bound = bind_normalized_tool_call(
            ToolCall("plan_journey", {"from": "a", "to": "b"}),
            "横浜から上野まで地図で",
            route_stage=2,
            semantic_fallback=True,
        )
        self.assertEqual(bound.name, "plan_route_map")
        self.assertEqual(bound.arguments["strategy"], "balanced")


if __name__ == "__main__":
    unittest.main()
