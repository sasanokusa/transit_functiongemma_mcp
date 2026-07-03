import unittest

from line_operator_rules import evaluate_operator_constraints, extract_operator_constraints
from route_constraints import apply_route_constraints
from route_renderer import render_answer


class ModeConstraintTest(unittest.TestCase):
    def test_extracts_operator_groups(self) -> None:
        self.assertEqual(
            extract_operator_constraints("東京から新宿まで地下鉄だけで")["allowed_operator_groups"],
            ["subway"],
        )
        self.assertEqual(
            extract_operator_constraints("JR使わないで")["avoid_operator_groups"], ["JR"]
        )
        self.assertEqual(extract_operator_constraints("バスなしで")["avoid_modes"], ["bus"])
        self.assertEqual(
            extract_operator_constraints("JRを使わないルートで")["avoid_operator_groups"],
            ["JR"],
        )

    def test_subway_rejects_jr_and_accepts_metro(self) -> None:
        jr = {"legs": [{"type": "train", "line": "中央線快速"}]}
        metro = {"legs": [{"type": "train", "line": "丸ノ内線"}]}
        self.assertFalse(evaluate_operator_constraints(jr, ["subway"])["satisfied"])
        self.assertTrue(evaluate_operator_constraints(metro, ["subway"])["satisfied"])

    def test_tokyo_metro_and_toei_subway_are_distinguished(self) -> None:
        metro_route = {"legs": [{"type": "train", "line": "丸ノ内線"}]}
        toei_route = {"legs": [{"type": "train", "line": "大江戸線"}]}
        self.assertTrue(
            evaluate_operator_constraints(metro_route, ["tokyo_metro"])["satisfied"]
        )
        self.assertFalse(
            evaluate_operator_constraints(toei_route, ["tokyo_metro"])["satisfied"]
        )
        self.assertTrue(
            evaluate_operator_constraints(toei_route, ["toei_subway"])["satisfied"]
        )
        self.assertFalse(
            evaluate_operator_constraints(metro_route, ["toei_subway"])["satisfied"]
        )
        self.assertFalse(
            evaluate_operator_constraints(metro_route, avoid_operator_groups=["tokyo_metro"])[
                "satisfied"
            ]
        )
        self.assertTrue(
            evaluate_operator_constraints(toei_route, avoid_operator_groups=["tokyo_metro"])[
                "satisfied"
            ]
        )

    def test_renderer_never_shows_condition_outside_route(self) -> None:
        data = apply_route_constraints(
            {
                "status": "ok",
                "query": {"allowed_operator_groups": ["subway"]},
                "routes": [{"summary": {}, "legs": [{"type": "train", "line": "中央線"}]}],
            }
        )
        answer = render_answer(data)
        self.assertIn("地下鉄だけの候補は見つかりません", answer)
        self.assertNotIn("中央線", answer)

    def test_group_specific_empty_messages(self) -> None:
        route = [{"summary": {}, "legs": [{"type": "train", "line": "中央線"}]}]
        metro = render_answer(
            apply_route_constraints(
                {"status": "ok", "query": {"allowed_operator_groups": ["tokyo_metro"]}, "routes": route}
            )
        )
        toei = render_answer(
            apply_route_constraints(
                {"status": "ok", "query": {"allowed_operator_groups": ["toei_subway"]}, "routes": route}
            )
        )
        self.assertIn("東京メトロだけの候補は見つかりません", metro)
        self.assertIn("都営地下鉄だけの候補は見つかりません", toei)

    def test_filter_runs_before_display_limit(self) -> None:
        data = apply_route_constraints(
            {
                "status": "ok",
                "query": {"allowed_operator_groups": ["subway"]},
                "routes": [
                    {"summary": {}, "legs": [{"type": "train", "from": "東京", "to": "新宿", "line": "中央線"}]},
                    {"summary": {}, "legs": [{"type": "train", "from": "東京", "to": "新宿", "line": "丸ノ内線"}]},
                ],
            }
        )
        answer = render_answer(data, max_routes=1)
        self.assertIn("丸ノ内線", answer)
        self.assertNotIn("中央線", answer)
