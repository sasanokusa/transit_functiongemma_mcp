import threading
import time
import unittest
from unittest.mock import patch

from answer_pipeline import StationSelectionRequired, SuggestionSelectionRequired
from web_api import TransitAPI, _compose_route_prompt


class WebAPISessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TransitAPI.__new__(TransitAPI)
        self.api.adapter = "adapter"
        self.api.schema_mode = "baked"
        self.api.lock = threading.Lock()
        self.api.router = object()
        self.api.sessions = {}

    def test_selection_resumes_original_prompt(self) -> None:
        candidates = [
            {"name": "東京駅", "endpoint": "geo:35.68,139.76", "description": "駅 6地点"},
            {"name": "東京駅", "endpoint": "geo:35.70,139.70", "description": "別地点"},
        ]
        with patch(
            "web_api.run_pipeline",
            side_effect=[StationSelectionRequired("東京", candidates), "経路候補です。"],
        ) as mocked:
            first = self.api.query("東京から上野まで")
            self.assertEqual(first["kind"], "selection")
            second = self.api.query(None, first["conversation_id"], 0)
            self.assertEqual(second["kind"], "answer")
            self.assertEqual(second["answer"], "経路候補です。")
            overrides = mocked.call_args.kwargs["station_overrides"]
            self.assertEqual(overrides["東京"]["endpoint"], "geo:35.68,139.76")
            self.assertTrue(mocked.call_args.kwargs["clarification_tool"])

    def test_expired_session_is_rejected(self) -> None:
        self.api.sessions["old"] = {
            "prompt": "x", "overrides": {}, "updated_at": time.monotonic() - 3600,
            "pending_query": "東京", "candidates": [],
        }
        with self.assertRaisesRegex(ValueError, "conversation_not_found"):
            self.api.query(None, "old", 0)

    def test_origin_and_destination_can_be_selected_in_sequence(self) -> None:
        origin = [{"name": "高田", "endpoint": "geo:35.55,139.62"}]
        destination = [{"name": "大久保", "endpoint": "geo:35.70,139.69"}]
        with patch(
            "web_api.run_pipeline",
            side_effect=[
                StationSelectionRequired("高田", origin),
                StationSelectionRequired("大久保", destination),
                "経路候補です。",
            ],
        ) as mocked:
            first = self.api.query("高田から大久保まで")
            second = self.api.query(None, first["conversation_id"], 0)
            self.assertEqual(second["kind"], "selection")
            third = self.api.query(None, second["conversation_id"], 0)
            self.assertEqual(third["kind"], "answer")
            overrides = mocked.call_args.kwargs["station_overrides"]
            self.assertEqual(set(overrides), {"高田", "大久保"})

    def test_direct_station_suggestion_can_be_selected(self) -> None:
        candidates = [
            {"id": "feed:tokyo", "name": "東京", "source_label": "東北新幹線"},
            {"id": "feed:tokyo-2", "name": "東京", "source_label": "京浜東北線"},
        ]
        with patch(
            "web_api.run_pipeline",
            side_effect=SuggestionSelectionRequired("suggest_stations", "東京駅", candidates),
        ):
            first = self.api.query("東京駅を検索して")
            self.assertEqual(first["kind"], "selection")
            second = self.api.query(None, first["conversation_id"], 0)
            self.assertEqual(second["kind"], "selected")
            self.assertIn("東北新幹線", second["answer"])
            self.assertEqual(second["selected"]["id"], "feed:tokyo")

    def test_selected_station_is_kept_for_followup_route(self) -> None:
        candidates = [
            {
                "id": "feed:tokyo",
                "name": "東京",
                "source_label": "京浜東北線",
                "lat": 35.68124,
                "lon": 139.76712,
            }
        ]
        with patch(
            "web_api.run_pipeline",
            side_effect=[
                SuggestionSelectionRequired("suggest_stations", "東京駅", candidates),
                "経路候補です。",
            ],
        ) as mocked:
            first = self.api.query("東京駅を検索して")
            selected = self.api.query(None, first["conversation_id"], 0)
            awaiting = self.api.query(None, first["conversation_id"], role="origin")
            result = self.api.query("上野", first["conversation_id"])

            self.assertEqual(selected["kind"], "selected")
            self.assertEqual(awaiting["kind"], "awaiting_route")
            self.assertEqual(result["kind"], "answer")
            self.assertEqual(result["answer"], "経路候補です。")
            self.assertEqual(mocked.call_args.args[0], "東京駅から上野まで 京浜東北線を利用")
            override = mocked.call_args.kwargs["station_overrides"]["東京"]
            self.assertEqual(override["endpoint"], "geo:35.68124,139.76712")

    def test_selected_destination_is_composed_without_loop(self) -> None:
        candidate = {"name": "上野", "source_label": "京浜東北線"}
        self.assertEqual(
            _compose_route_prompt("destination", candidate, "横浜から"),
            "横浜から上野駅まで 京浜東北線を利用",
        )

    def test_route_map_result_returns_map_kind_with_mcp_text(self) -> None:
        structured = {"options": [{"id": "route-1", "map": {"points": []}}]}
        content = [{"type": "text", "text": "東京駅 → 上野駅 候補1件"}]

        def fake_pipeline(prompt, **kwargs):
            kwargs["ui_payload"].update(
                {
                    "tool": "plan_route_map",
                    "content": content,
                    "structuredContent": structured,
                    "isError": False,
                }
            )
            return "renderer text"

        with patch("web_api.run_pipeline", side_effect=fake_pipeline):
            result = self.api.query("東京から上野まで、地図で")
        self.assertEqual(result["kind"], "map")
        self.assertEqual(result["answer"], "東京駅 → 上野駅 候補1件")
        self.assertEqual(result["map_result"]["structuredContent"], structured)
        self.assertFalse(result["map_result"]["isError"])

    def test_text_route_result_stays_answer_kind(self) -> None:
        with patch("web_api.run_pipeline", return_value="経路候補です。"):
            result = self.api.query("東京から上野まで")
        self.assertEqual(result["kind"], "answer")
        self.assertNotIn("map_result", result)

    def test_default_graphical_is_forwarded_to_pipeline(self) -> None:
        self.api.default_graphical = True
        with patch("web_api.run_pipeline", return_value="経路候補です。") as mocked:
            self.api.query("東京から上野まで")
        self.assertTrue(mocked.call_args.kwargs["default_graphical"])


class RouteMapResourceCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TransitAPI.__new__(TransitAPI)
        self.api.ui_lock = threading.Lock()
        self.api.ui_resource_cache = {}

    def _fake_client(self, reads: list[int], html: str = "<html>map</html>"):
        outer = self

        class FakeClient:
            def __init__(self, endpoint: str) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def initialize(self):
                return {}

            def request(self, method: str, params: dict) -> dict:
                assert method == "resources/read"
                assert params == {"uri": "ui://transit/route-map"}
                reads.append(1)
                return {
                    "result": {
                        "contents": [
                            {
                                "uri": params["uri"],
                                "mimeType": "text/html;profile=mcp-app",
                                "text": html,
                            }
                        ]
                    }
                }

        return FakeClient

    def test_resource_is_fetched_once_and_cached(self) -> None:
        reads: list[int] = []
        with patch("web_api.MCPClient", self._fake_client(reads)):
            first = self.api.route_map_html()
            second = self.api.route_map_html()
        self.assertEqual(first, "<html>map</html>")
        self.assertEqual(second, "<html>map</html>")
        self.assertEqual(len(reads), 1)

    def test_stale_copy_survives_mcp_outage(self) -> None:
        self.api.ui_resource_cache["ui://transit/route-map"] = (
            time.monotonic() - 1,
            "<html>stale</html>",
        )

        class FailingClient:
            def __init__(self, endpoint: str) -> None:
                pass

            def __enter__(self):
                from transit_functiongemma.mcp import MCPError

                raise MCPError("down")

            def __exit__(self, *args: object) -> None:
                pass

        with patch("web_api.MCPClient", FailingClient):
            self.assertEqual(self.api.route_map_html(), "<html>stale</html>")


if __name__ == "__main__":
    unittest.main()
