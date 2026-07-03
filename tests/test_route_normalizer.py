import json
import unittest

from transit_functiongemma.route_normalizer import normalize_mcp_result


class RouteNormalizerTest(unittest.TestCase):
    def test_saved_artifact_and_real_mcp_seconds_shape(self) -> None:
        payload = {
            "date": "20260629", "type": "arrival", "timezone": "Asia/Tokyo",
            "from": {"id": "feed:a", "name": "町田"},
            "to": {"id": "feed:b", "name": "池袋"},
            "journeys": [{
                "departureSecs": 29400, "arrivalSecs": 32460,
                "durationSecs": 3060, "transferCount": 1,
                "legs": [{
                    "kind": "train", "from": {"id": "feed:a", "name": "町田"},
                    "to": {"id": "feed:b", "name": "池袋"},
                    "departureSecs": 29400, "arrivalSecs": 32460,
                    "routeName": "テスト線"
                }]
            }]
        }
        artifact = {"envelope": {"jsonrpc": "2.0", "result": {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        }}}
        result = normalize_mcp_result(
            artifact, "plan_journey", {"fromLabel": "町田", "toLabel": "池袋"}
        )
        self.assertEqual(result["query"]["origin_text"], "町田")
        self.assertEqual(result["query"]["time_mode"], "arrive_by")
        self.assertEqual(result["routes"][0]["summary"]["duration_min"], 51)
        self.assertEqual(result["routes"][0]["legs"][0]["departure_time"], "08:10")
        self.assertEqual(result["routes"][0]["legs"][0]["line"], "テスト線")

    def test_structured_content_preferred(self) -> None:
        raw = {"result": {"structuredContent": {
            "stations": [{"id": "a", "name": "東京", "feedName": "中央線"}]
        }}}
        result = normalize_mcp_result(raw, "suggest_stations")
        self.assertEqual(result["suggestions"][0]["name"], "東京")
        self.assertEqual(result["suggestions"][0]["source_label"], "中央線")

    def test_unknown_or_missing_shape_does_not_crash(self) -> None:
        result = normalize_mcp_result({"result": {"content": []}}, "plan_journey")
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["routes"], [])

    def test_absent_fare_remains_absent(self) -> None:
        raw = {"journeys": [{"durationSecs": 600, "legs": []}]}
        result = normalize_mcp_result(raw, "plan_journey")
        self.assertNotIn("fare_yen", result["routes"][0]["summary"])

    def test_date_without_time_does_not_invent_midnight(self) -> None:
        raw = {"date": "20260629", "timezone": "Asia/Tokyo", "journeys": []}
        result = normalize_mcp_result(raw, "plan_journey")
        self.assertEqual(result["query"]["datetime"], "2026-06-29")
        self.assertNotIn("00:00", result["query"]["datetime"])

    def test_access_and_egress_walks_are_included(self) -> None:
        raw = {
            "from": {"id": "geo:1,1", "name": "出発地点"},
            "to": {"id": "geo:2,2", "name": "目的地点"},
            "journeys": [{
                "departureSecs": 3600,
                "arrivalSecs": 7200,
                "durationSecs": 3600,
                "accessWalkSecs": 600,
                "egressWalkSecs": 300,
                "legs": [{
                    "kind": "transit", "routeName": "テスト線",
                    "from": {"id": "a", "name": "A駅"},
                    "to": {"id": "b", "name": "B駅"},
                    "departureSecs": 3600, "arrivalSecs": 6900,
                }],
            }],
        }
        result = normalize_mcp_result(raw, "plan_journey")
        route = result["routes"][0]
        self.assertEqual(route["summary"]["duration_min"], 70)
        self.assertEqual(route["summary"]["departure_time"], "00:50")
        self.assertEqual(route["legs"][0]["from"], "出発地点")
        self.assertEqual(route["legs"][0]["to"], "A駅")
        self.assertEqual(route["legs"][-1]["from"], "B駅")
        self.assertEqual(route["legs"][-1]["to"], "目的地点")
        self.assertEqual(route["legs"][-1]["arrival_time"], "02:00")
        self.assertEqual(route["summary"]["walk_duration_min"], 15)


if __name__ == "__main__":
    unittest.main()
