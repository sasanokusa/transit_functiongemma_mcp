import unittest

from scripts.run_operational_samples import displayed_time_satisfies


class OperationalRunnerTimeTest(unittest.TestCase):
    def test_arrival_rejects_route_that_only_departs_at_requested_time(self) -> None:
        self.assertFalse(
            displayed_time_satisfies(
                "16:00 東京駅 → 17:36 新宿駅（徒歩）", "arrival", "16:00"
            )
        )

    def test_arrival_accepts_route_arriving_before_requested_time(self) -> None:
        self.assertTrue(
            displayed_time_satisfies(
                "15:33 東京駅 → 15:56 新宿駅", "arrival", "16:00"
            )
        )

    def test_departure_rejects_route_before_requested_time(self) -> None:
        self.assertFalse(
            displayed_time_satisfies(
                "15:50 東京駅 → 16:10 新宿駅", "departure", "16:00"
            )
        )

    def test_departure_accepts_route_at_requested_time(self) -> None:
        self.assertTrue(
            displayed_time_satisfies(
                "16:00 東京駅 → 16:24 新宿駅", "departure", "16:00"
            )
        )


if __name__ == "__main__":
    unittest.main()
