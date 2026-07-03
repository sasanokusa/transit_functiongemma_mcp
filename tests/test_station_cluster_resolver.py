import unittest

from transit_functiongemma.station_resolver import resolve_physical_station


class StationClusterResolverTest(unittest.TestCase):
    def test_far_same_name_stations_still_require_confirmation(self) -> None:
        suggestions = [
            {"name": "大宮駅", "kind": "station", "endpoint": "geo:35.90,139.62"},
            {"name": "大宮駅", "kind": "station", "endpoint": "geo:35.00,135.75"},
        ]
        self.assertEqual(resolve_physical_station("大宮", suggestions)["status"], "ambiguous")

    def test_extended_city_cluster_has_auditable_annotation(self) -> None:
        suggestions = [
            {"id": "transit:query-landmark:a", "name": "浅草駅", "kind": "station",
             "endpoint": "geo:35.7118,139.7977", "source": "transit"},
            {"id": "osm:cluster:a", "name": "浅草駅", "kind": "station",
             "endpoint": "geo:35.7107,139.8016", "source": "osm"},
        ]
        result = resolve_physical_station("浅草", suggestions)
        self.assertEqual(result["resolution"], "station_cluster")
        self.assertGreater(result["station"]["cluster_span_meters"], 300)
