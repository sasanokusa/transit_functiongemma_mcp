import unittest

from transit_functiongemma.route_intent import extract_route_intent


class RouteIntentTest(unittest.TestCase):
    def test_operational_100_surface_variants(self) -> None:
        afternoon = extract_route_intent("品川から上野まで午後3時着で")
        self.assertEqual(afternoon["time"], "15:00")
        self.assertEqual(
            extract_route_intent("目黒から秋葉原まで、品川を通らないルートで")[
                "avoid_station_texts"
            ],
            ["品川"],
        )
        self.assertEqual(
            extract_route_intent("中野から上野まで、中央線を使わないで")[
                "avoid_line_texts"
            ],
            ["中央線"],
        )
        self.assertEqual(
            extract_route_intent("品川から池袋まで、なるべく早く")["priority"],
            "fast",
        )
        self.assertEqual(
            extract_route_intent("押上から池袋まで、歩く距離短めで")["priority"],
            "less_walk",
        )

    def test_constraints_and_datetime(self) -> None:
        intent = extract_route_intent(
            "明日8:30出発で品川から成田空港、渋谷を避けて乗換少なめ",
            "2026-06-29 10:00 Asia/Tokyo",
        )
        self.assertEqual(intent["origin_text"], "品川")
        self.assertEqual(intent["destination_text"], "成田空港")
        self.assertEqual(intent["avoid_station_texts"], ["渋谷"])
        self.assertEqual(intent["priority"], "few_transfers")
        self.assertEqual(intent["date"], "20260630")
        self.assertEqual(intent["time"], "08:30")
        self.assertEqual(intent["time_mode"], "departure_at")

    def test_via_line_avoid_and_place_alias(self) -> None:
        intent = extract_route_intent("押上から赤レンガ倉庫、東京駅経由で。山手線なしで")
        self.assertEqual(intent["destination_text"], "横浜赤レンガ倉庫")
        self.assertEqual(intent["via_station_texts"], ["東京"])
        self.assertEqual(intent["avoid_line_texts"], ["山手線"])

    def test_incomplete_deictic_origin_is_not_an_endpoint(self) -> None:
        intent = extract_route_intent("ここから新宿まで行きたい")
        self.assertIsNone(intent["origin_text"])
        self.assertIsNone(intent["destination_text"])

    def test_negative_via_is_only_an_avoid_constraint(self) -> None:
        intent = extract_route_intent("上野から渋谷まで、秋葉原経由は嫌")
        self.assertEqual(intent["avoid_station_texts"], ["秋葉原"])
        self.assertEqual(intent["via_station_texts"], [])


if __name__ == "__main__":
    unittest.main()
