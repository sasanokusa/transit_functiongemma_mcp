import unittest

from answer_pipeline import _planned_call_from_model_intent, _route_arguments


class ViaPlanningTest(unittest.TestCase):
    def test_model_intent_resolves_via_before_destination(self) -> None:
        intent = {
            "origin_text": "新宿",
            "destination_text": "東京",
            "via_station_texts": ["六本木"],
        }
        first = _planned_call_from_model_intent(intent, [])
        via = _planned_call_from_model_intent(intent, [{}])
        destination = _planned_call_from_model_intent(intent, [{}, {}])
        self.assertEqual(first.arguments["q"], "新宿")
        self.assertEqual(via.arguments["q"], "六本木")
        self.assertEqual(destination.arguments["q"], "東京")

    def test_route_arguments_include_schema_via(self) -> None:
        stations = [
            {"name": "新宿駅", "endpoint": "geo:1,1"},
            {"name": "六本木駅", "endpoint": "geo:2,2"},
            {"name": "東京駅", "endpoint": "geo:3,3"},
        ]
        result = _route_arguments(
            "plan_journey",
            {},
            stations,
            {
                "origin_text": "新宿",
                "destination_text": "東京",
                "via_station_texts": ["六本木"],
            },
        )
        self.assertEqual(result["from"], "geo:1,1")
        self.assertEqual(result["to"], "geo:3,3")
        self.assertEqual(result["via"], ["geo:2,2"])
        self.assertEqual(result["viaLabel"], ["六本木駅"])
