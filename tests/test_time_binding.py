import unittest

from transit_functiongemma.answer_pipeline import _route_arguments


STATIONS = [
    {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
    {"name": "新宿駅", "endpoint": "geo:35.69,139.70"},
]


class TimeBindingTest(unittest.TestCase):
    def test_preserved_arrival_intent_binds_arrival(self) -> None:
        result = _route_arguments(
            "plan_journey",
            {"time": "model-copy-must-not-win"},
            STATIONS,
            {"time_mode": "arrive_by", "time": "09:00"},
        )
        self.assertEqual(result["type"], "arrival")
        self.assertEqual(result["time"], "09:00")

    def test_preserved_departure_intent_binds_departure(self) -> None:
        result = _route_arguments(
            "plan_journey",
            {"time": "model-copy-must-not-win"},
            STATIONS,
            {"time_mode": "departure_at", "time": "09:00"},
        )
        self.assertEqual(result["type"], "departure")
        self.assertEqual(result["time"], "09:00")
