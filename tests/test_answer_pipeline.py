import unittest

from answer_pipeline import (
    _display_suggestions,
    _planned_call_from_model_intent,
    _route_arguments,
    _route_intent_with_defaults,
    handle_local_call,
    handle_no_call,
)
from local_tools import extract_route_hints


class AnswerPipelineOfflineTest(unittest.TestCase):
    def test_no_call_returns_question_without_mcp(self) -> None:
        self.assertEqual(
            handle_no_call("東京駅まで行きたい"),
            "出発地と目的地が不足しています。どこからどこまで行きますか？",
        )

    def test_arrival_with_destination_only_asks_for_origin(self) -> None:
        self.assertEqual(
            handle_no_call("明日9時に品川に着きたい"),
            "出発地と目的地が不足しています。どこからどこまで行きますか？",
        )

    def test_local_clarification(self) -> None:
        text = handle_local_call(
            "ask_clarification",
            {"missing": ["destination"], "question": "目的地を教えてください。"},
        )
        self.assertEqual(text, "目的地を教えてください。")

    def test_complete_but_unrouted_request_does_not_claim_fields_are_missing(self) -> None:
        self.assertEqual(
            handle_no_call("町田から池袋まで行きたい"),
            "出発地と目的地が不足しています。どこからどこまで行きますか？",
        )

    def test_avoid_hint_is_extracted_once(self) -> None:
        hints = extract_route_hints("町田から池袋まで、渋谷を避けて")
        self.assertEqual(hints["avoid_station_texts"], ["渋谷"])

    def test_preferred_line_hint(self) -> None:
        hints = extract_route_hints("横浜から上野まで、京浜東北線で行きたい")
        self.assertEqual(hints["preferred_line_texts"], ["京浜東北線"])

    def test_resolved_geo_endpoints_replace_model_copy_errors(self) -> None:
        result = _route_arguments(
            "plan_journey",
            {"from": "geo:35.46", "to": "geo:35.71", "type": "station"},
            [
                {"name": "横浜駅", "endpoint": "geo:35.4655,139.6231"},
                {"name": "上野駅", "endpoint": "geo:35.7124,139.7767"},
            ],
        )
        self.assertEqual(result["from"], "geo:35.4655,139.6231")
        self.assertEqual(result["to"], "geo:35.7124,139.7767")
        self.assertNotIn("type", result)

    def test_time_mode_requires_explicit_user_wording(self) -> None:
        stations = [
            {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
            {"name": "上野駅", "endpoint": "geo:35.71,139.77"},
        ]
        ordinary = _route_arguments("plan_journey", {"type": "last"}, stations, {})
        last = _route_arguments(
            "plan_journey", {}, stations, {"time_mode": "last_train"}
        )
        self.assertNotIn("type", ordinary)
        self.assertEqual(last["type"], "last")

    def test_requested_route_count_is_sent_to_mcp(self) -> None:
        stations = [
            {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
            {"name": "上野駅", "endpoint": "geo:35.71,139.77"},
        ]
        result = _route_arguments(
            "plan_journey", {}, stations, {}, num_itineraries=1
        )
        self.assertEqual(result["numItineraries"], 1)

    def test_ui_suggestions_prefer_exact_name_and_remove_artwork(self) -> None:
        candidates = [
            {"name": "東京(羽田)空港", "description": "駅"},
            {"name": "東京", "description": "東北新幹線"},
            {"name": "東京", "description": "artwork"},
        ]
        result = _display_suggestions("東京駅", candidates)
        self.assertEqual(result, [{"name": "東京", "description": "東北新幹線"}])

    def test_default_graphical_changes_effective_presentation_only(self) -> None:
        model_intent = {
            "origin_text": "東京",
            "destination_text": "上野",
            "graphical": False,
        }
        effective = _route_intent_with_defaults(
            model_intent, default_graphical=True
        )
        self.assertFalse(model_intent["graphical"])
        self.assertTrue(effective["graphical"])
        call = _planned_call_from_model_intent(effective, [])
        self.assertEqual(call.name, "suggest_stations")
        final_call = _planned_call_from_model_intent(
            effective,
            [
                {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
                {"name": "上野駅", "endpoint": "geo:35.71,139.77"},
            ],
        )
        self.assertEqual(final_call.name, "plan_route_map")


if __name__ == "__main__":
    unittest.main()
