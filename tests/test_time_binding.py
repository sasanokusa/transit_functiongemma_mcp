import unittest

from answer_pipeline import _route_arguments


STATIONS = [
    {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
    {"name": "新宿駅", "endpoint": "geo:35.69,139.70"},
]


class TimeBindingTest(unittest.TestCase):
    def test_model_arrival_slot_binds_arrival(self) -> None:
        result = _route_arguments(
            "plan_journey",
            {"time": "9:00"},
            STATIONS,
            {"time_mode": "arrive_by"},
        )
        self.assertEqual(result["type"], "arrival")

    def test_model_departure_slot_binds_departure(self) -> None:
        result = _route_arguments(
            "plan_journey",
            {"time": "9:00"},
            STATIONS,
            {"time_mode": "departure_at"},
        )
        self.assertEqual(result["type"], "departure")
