import unittest

from transit_functiongemma.route_intent import extract_route_intent


class SpecialTimeNormalizerTest(unittest.TestCase):
    def test_first_train_prefix_does_not_pollute_endpoints(self) -> None:
        result = extract_route_intent("始発で池袋から品川まで行きたい", "2026-07-01 10:00")
        self.assertEqual(result["origin_text"], "池袋")
        self.assertEqual(result["destination_text"], "品川")
        self.assertEqual(result["time_mode"], "first_train")

    def test_special_time_variants(self) -> None:
        cases = {
            "始発に池袋から品川まで": "first_train",
            "始発の電車で池袋から品川まで": "first_train",
            "朝イチで池袋から品川まで": "first_train",
            "終電で新宿から立川まで": "last_train",
            "終電に新宿から立川まで": "last_train",
            "最終電車で新宿から立川まで": "last_train",
            "今日の終電で新宿から立川まで": "last_train",
            "明日の始発で池袋から品川まで": "first_train",
        }
        for text, mode in cases.items():
            with self.subTest(text=text):
                result = extract_route_intent(text, "2026-07-01 10:00")
                self.assertEqual(result["origin_text"], "池袋" if mode == "first_train" else "新宿")
                self.assertEqual(result["time_mode"], mode)

    def test_last_train_deadline_phrase_does_not_pollute_origin(self) -> None:
        result = extract_route_intent("終電に間に合うように新橋から中野まで")
        self.assertEqual(result["time_mode"], "last_train")
        self.assertEqual(result["origin_text"], "新橋")
        self.assertEqual(result["destination_text"], "中野")
