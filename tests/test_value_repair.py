"""Value-fidelity repair: restore user-written values the model cannot copy.

Cases mirror real r7 nonroute-215 failures (artifacts/eval_nonroute_215_r7_final.json):
coordinate digit truncation, relative-date arithmetic, and time zero-padding.
"""
import unittest

from transit_functiongemma.japanese import repair_tool_call_values
from transit_functiongemma.toolcall import ToolCall


REFERENCE = "2026-06-29 10:00 Asia/Tokyo"


class CoordinateRepairTest(unittest.TestCase):
    def test_truncated_coordinates_snap_to_text(self):
        # ind-095: model rounded 35.443708 -> 35.4437 and 139.638026 -> 139.638
        call = ToolCall(
            "reverse_geocode",
            {"lat": 35.4437, "lon": 139.638, "limit": 3, "radiusMeters": 50},
        )
        repaired = repair_tool_call_values(
            call, "横浜付近の座標 lat=35.443708 lon=139.638026 付近", REFERENCE
        )
        self.assertEqual(repaired.arguments["lat"], 35.443708)
        self.assertEqual(repaired.arguments["lon"], 139.638026)
        self.assertEqual(repaired.arguments["limit"], 3)

    def test_japanese_coordinate_notation(self):
        # ind-099: 緯度/経度 words
        call = ToolCall("reverse_geocode", {"lat": 35.011636, "lon": 135.768, "limit": 3})
        repaired = repair_tool_call_values(
            call, "緯度35.011636、経度135.768029の近くの駅を探して", REFERENCE
        )
        self.assertEqual(repaired.arguments["lon"], 135.768029)

    def test_comma_pair_notation(self):
        # ind-103: model corrupted a digit (139.810063 vs 139.8107)
        call = ToolCall("reverse_geocode", {"lat": 35.710063, "lon": 139.810063})
        repaired = repair_tool_call_values(
            call, "現在地 35.710063,139.8107 から乗れそうな場所", REFERENCE
        )
        self.assertEqual(repaired.arguments["lon"], 139.8107)

    def test_far_model_value_is_not_snapped(self):
        call = ToolCall("reverse_geocode", {"lat": 34.7, "lon": 135.5})
        repaired = repair_tool_call_values(
            call, "現在地 35.710063,139.8107 から乗れそうな場所", REFERENCE
        )
        self.assertEqual(repaired.arguments["lat"], 34.7)
        self.assertEqual(repaired.arguments["lon"], 135.5)

    def test_multiple_pairs_are_ambiguous(self):
        call = ToolCall("reverse_geocode", {"lat": 35.7295, "lon": 139.7109})
        repaired = repair_tool_call_values(
            call, "35.729503,139.7109 と 35.443708,139.638026 のどちらか", REFERENCE
        )
        self.assertEqual(repaired.arguments["lat"], 35.7295)


class IdRepairTest(unittest.TestCase):
    def test_single_text_id_is_snapped(self):
        call = ToolCall("station_departures", {"id": "demo-feed:shinjuk"})
        repaired = repair_tool_call_values(
            call, "demo-feed:shinjuku で明日8時半ごろの発車案内", REFERENCE
        )
        self.assertEqual(repaired.arguments["id"], "demo-feed:shinjuku")

    def test_matching_id_is_kept(self):
        call = ToolCall("get_station", {"id": "demo-feed:shibuya"})
        repaired = repair_tool_call_values(
            call, "demo-feed:shibuya の詳細", REFERENCE
        )
        self.assertEqual(repaired.arguments["id"], "demo-feed:shibuya")


class RelativeDateRepairTest(unittest.TestCase):
    def test_tomorrow_off_by_one_is_fixed(self):
        # ind-112: 明日 resolved to the reference day instead of the next day
        call = ToolCall(
            "station_departures",
            {"id": "demo-feed:shinjuku", "date": "20260629", "time": "08:30"},
        )
        repaired = repair_tool_call_values(
            call, "demo-feed:shinjuku で明日8時半ごろの発車案内", REFERENCE
        )
        self.assertEqual(repaired.arguments["date"], "20260630")

    def test_missing_today_date_is_filled_for_last_train(self):
        # ind-113: 今日の終電 with no date argument
        call = ToolCall("station_departures", {"id": "demo-feed:shibuya"})
        repaired = repair_tool_call_values(
            call, "demo-feed:shibuya の今日の終電発車を見たい", REFERENCE
        )
        self.assertEqual(repaired.arguments["date"], "20260629")

    def test_no_date_cue_leaves_arguments_untouched(self):
        call = ToolCall("station_departures", {"id": "demo-feed:omiya"})
        repaired = repair_tool_call_values(
            call, "demo-feed:omiya の8時の発車案内", REFERENCE
        )
        self.assertNotIn("date", repaired.arguments)

    def test_other_tools_do_not_gain_a_date(self):
        call = ToolCall("suggest_stations", {"q": "新宿", "limit": 5})
        repaired = repair_tool_call_values(call, "明日使う新宿の駅候補", REFERENCE)
        self.assertNotIn("date", repaired.arguments)

    def test_plan_journey_tomorrow_date_is_fixed(self):
        call = ToolCall(
            "plan_journey",
            {"from": "demo-feed:tokyo", "to": "demo-feed:ueno", "date": "20260629"},
        )
        repaired = repair_tool_call_values(
            call, "明日、東京から上野まで行きたい", REFERENCE
        )
        self.assertEqual(repaired.arguments["date"], "20260630")

    def test_plan_route_map_today_date_is_filled(self):
        call = ToolCall(
            "plan_route_map",
            {"from": "demo-feed:tokyo", "to": "demo-feed:ueno"},
        )
        repaired = repair_tool_call_values(
            call, "今日、東京から上野まで地図で", REFERENCE
        )
        self.assertEqual(repaired.arguments["date"], "20260629")

    def test_route_request_tomorrow_repairs_only_value_slots(self):
        call = ToolCall(
            "resolve_route_request",
            {
                "origin_text": "東京",
                "destination_text": "上野",
                "graphical": True,
                "priority": "cheap",
                "time_mode": "arrive_by",
                "date": "20260629",
                "time": "08:00",
            },
        )
        repaired = repair_tool_call_values(
            call, "明日9時に東京から上野へ着きたい。安い順を地図で", REFERENCE
        )
        self.assertEqual(repaired.name, "resolve_route_request")
        self.assertEqual(repaired.arguments["date"], "20260630")
        self.assertEqual(repaired.arguments["time"], "09:00")
        self.assertEqual(repaired.arguments["time_mode"], "arrive_by")
        self.assertEqual(repaired.arguments["priority"], "cheap")
        self.assertTrue(repaired.arguments["graphical"])


class TimePaddingTest(unittest.TestCase):
    def test_python_ios_time_parity_fixture(self):
        # Mirrors JapaneseRuntimeCompatibility.verifyFixtures in LlamaState.swift.
        fixtures = {
            "夜9時": "21:00",
            "午後9時": "21:00",
            "朝9時": "09:00",
            "午前12時": "00:00",
            "午後12時": "12:00",
            "午後3時半": "15:30",
            "夜中1時": "01:00",
            "深夜1時": "01:00",
        }
        for text, expected in fixtures.items():
            with self.subTest(text=text):
                repaired = repair_tool_call_values(
                    ToolCall("resolve_route_request", {"time": "model-value"}),
                    f"東京から新宿まで{text}に出たい",
                    REFERENCE,
                )
                self.assertEqual(repaired.arguments["time"], expected)

    def test_single_digit_hour_is_padded(self):
        call = ToolCall(
            "station_departures", {"id": "demo-feed:shinjuku", "time": "8:30"}
        )
        repaired = repair_tool_call_values(
            call, "demo-feed:shinjuku で8:30の発車案内", REFERENCE
        )
        self.assertEqual(repaired.arguments["time"], "08:30")

    def test_padded_time_is_unchanged(self):
        call = ToolCall(
            "station_departures", {"id": "demo-feed:shinjuku", "time": "08:30"}
        )
        repaired = repair_tool_call_values(
            call, "demo-feed:shinjuku で8時半の発車案内", REFERENCE
        )
        self.assertEqual(repaired.arguments["time"], "08:30")

    def test_plan_journey_copies_explicit_japanese_clock(self):
        call = ToolCall(
            "plan_journey",
            {"from": "demo-feed:tokyo", "to": "demo-feed:ueno", "time": "16:00"},
        )
        repaired = repair_tool_call_values(
            call, "東京から上野まで16時30分に出たい", REFERENCE
        )
        self.assertEqual(repaired.arguments["time"], "16:30")

    def test_plan_route_map_copies_explicit_short_clock(self):
        call = ToolCall(
            "plan_route_map",
            {"from": "demo-feed:tokyo", "to": "demo-feed:ueno"},
        )
        repaired = repair_tool_call_values(
            call, "東京から上野まで9時に地図で", REFERENCE
        )
        self.assertEqual(repaired.arguments["time"], "09:00")

    def test_time_is_not_added_without_user_clock(self):
        call = ToolCall(
            "plan_journey",
            {"from": "demo-feed:tokyo", "to": "demo-feed:ueno"},
        )
        repaired = repair_tool_call_values(
            call, "東京から上野まで早めに行きたい", REFERENCE
        )
        self.assertNotIn("time", repaired.arguments)


class SafetyTest(unittest.TestCase):
    def test_tool_name_is_never_changed(self):
        call = ToolCall("reverse_geocode", {"lat": 35.4437, "lon": 139.638})
        repaired = repair_tool_call_values(
            call, "lat=35.443708 lon=139.638026", REFERENCE
        )
        self.assertEqual(repaired.name, "reverse_geocode")

    def test_unrelated_call_is_returned_as_is(self):
        call = ToolCall("suggest_places", {"q": "東京タワー", "limit": 10})
        repaired = repair_tool_call_values(call, "東京タワーを場所として探して", REFERENCE)
        self.assertIs(repaired, call)


if __name__ == "__main__":
    unittest.main()
