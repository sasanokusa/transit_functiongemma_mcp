import unittest

from transit_functiongemma.station_resolver import resolve_physical_station, station_query_text


class StationResolverTest(unittest.TestCase):
    def test_exact_name_excludes_shin_yokohama(self) -> None:
        suggestions = [
            {"name": "横浜駅", "kind": "station", "endpoint": "geo:35.46550,139.62310", "source": "transit"},
            {"name": "新横浜駅", "kind": "station", "endpoint": "geo:35.50750,139.61750", "source": "transit"},
        ]
        result = resolve_physical_station("横浜", suggestions)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["station"]["name"], "横浜駅")

    def test_nearby_duplicate_sources_form_one_physical_station(self) -> None:
        suggestions = [
            {"id": "transit:query-landmark:y", "name": "横浜駅", "kind": "station",
             "endpoint": "geo:35.465508,139.623115", "source": "transit"},
            {"id": "osm:cluster:y", "name": "横浜駅", "kind": "station",
             "endpoint": "geo:35.465570,139.621650", "source": "osm"},
        ]
        result = resolve_physical_station("横浜", suggestions)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["station"]["member_count"], 2)
        self.assertTrue(result["station"]["endpoint"].startswith("geo:"))

    def test_same_name_far_apart_stays_ambiguous(self) -> None:
        suggestions = [
            {"name": "町田駅", "kind": "station", "endpoint": "geo:35.54,139.45"},
            {"name": "町田駅", "kind": "station", "endpoint": "geo:34.40,132.45"},
        ]
        self.assertEqual(resolve_physical_station("町田", suggestions)["status"], "ambiguous")

    def test_route_context_resolves_a_distant_same_name_station(self) -> None:
        suggestions = [
            {"name": "大宮駅", "kind": "station", "endpoint": "geo:35.9063,139.6238"},
            {"name": "大宮駅", "kind": "station", "endpoint": "geo:35.0037,135.7485"},
        ]
        result = resolve_physical_station(
            "大宮",
            suggestions,
            near={"endpoint": "geo:35.6902,139.6987"},
        )
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["station"]["endpoint"], "geo:35.9063,139.6238")
        self.assertEqual(result["resolution"], "nearest_to_route_context")

    def test_route_context_keeps_nearby_same_name_candidates_ambiguous(self) -> None:
        suggestions = [
            {"name": "高田駅", "kind": "station", "endpoint": "geo:35.55,139.62"},
            {"name": "高田駅", "kind": "station", "endpoint": "geo:35.70,139.69"},
        ]
        result = resolve_physical_station(
            "高田",
            suggestions,
            near={"endpoint": "geo:35.60,139.65"},
        )
        self.assertEqual(result["status"], "ambiguous")

    def test_nearby_same_name_clusters_merge_for_city_station(self) -> None:
        suggestions = [
            {"id": "transit:query-landmark:asakusa", "name": "浅草駅", "kind": "station",
             "endpoint": "geo:35.7118,139.7977", "source": "transit"},
            {"id": "osm:cluster:asakusa", "name": "浅草駅", "kind": "station",
             "endpoint": "geo:35.7107,139.8016", "source": "osm"},
        ]
        result = resolve_physical_station("浅草", suggestions)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["resolution"], "station_cluster")
        self.assertEqual(result["station"]["member_count"], 2)

    def test_no_fuzzy_fallback(self) -> None:
        suggestions = [{"name": "新横浜駅", "kind": "station", "endpoint": "geo:35.50,139.61"}]
        self.assertEqual(resolve_physical_station("横浜", suggestions)["status"], "not_found")

    def test_same_name_non_station_is_excluded(self) -> None:
        suggestions = [
            {"name": "横浜", "kind": "restaurant", "endpoint": "geo:34.87,136.91"},
            {"name": "横浜駅", "kind": "station", "endpoint": "geo:35.46,139.62"},
        ]
        result = resolve_physical_station("横浜", suggestions)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["station"]["name"], "横浜駅")

    def test_misclassified_artwork_is_excluded(self) -> None:
        suggestions = [
            {"name": "東京駅", "kind": "station", "description": "駅 6地点",
             "source": "transit", "endpoint": "geo:35.68,139.76"},
            {"name": "東京駅", "kind": "station", "description": "artwork",
             "source": "osm", "endpoint": "geo:36.80,139.71"},
        ]
        result = resolve_physical_station("東京", suggestions)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["station"]["description"], "駅 6地点")

    def test_station_entrance_is_not_a_separate_physical_station(self) -> None:
        suggestions = [
            {"name": "池袋駅", "kind": "station", "description": "駅・停留所 18地点",
             "source": "transit", "endpoint": "geo:35.73001,139.71162"},
            {"name": "池袋駅", "kind": "station", "description": "出入口 2地点",
             "source": "osm", "endpoint": "geo:35.73159,139.70752"},
        ]
        result = resolve_physical_station("池袋", suggestions)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["station"]["description"], "駅・停留所 18地点")

    def test_line_specific_stop_does_not_create_false_physical_ambiguity(self) -> None:
        suggestions = [
            {"id": "transit:query-landmark:shinjuku", "name": "新宿駅", "kind": "station",
             "endpoint": "geo:35.690196,139.698711", "source": "transit"},
            {"id": "scrape-jreast-saikyo:Shinjuku", "name": "新宿", "kind": "station",
             "endpoint": "scrape-jreast-saikyo:Shinjuku", "source": "transit",
             "lat": 35.6874862, "lon": 139.6967483},
        ]
        result = resolve_physical_station("新宿", suggestions)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["station"]["endpoint"], "geo:35.690196,139.698711")

    def test_station_query_suffix(self) -> None:
        self.assertEqual(station_query_text("横浜"), "横浜駅")
        self.assertEqual(station_query_text("横浜駅"), "横浜駅")


if __name__ == "__main__":
    unittest.main()
