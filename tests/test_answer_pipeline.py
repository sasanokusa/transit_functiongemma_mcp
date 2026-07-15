import unittest
from datetime import datetime as RealDateTime
from unittest.mock import patch

from transit_functiongemma.answer_pipeline import (
    _display_suggestions,
    _planned_call_from_model_intent,
    _refine_clarification_call,
    _route_arguments,
    _route_intent_with_defaults,
    handle_local_call,
    handle_no_call,
    run_pipeline,
)
from transit_functiongemma.local_tools import extract_route_hints
from transit_functiongemma.toolcall import ToolCall


class AnswerPipelineOfflineTest(unittest.TestCase):
    def test_no_call_returns_question_without_mcp(self) -> None:
        self.assertEqual(
            handle_no_call("東京駅まで行きたい"),
            "出発地と目的地が不足しています。どこからどこまで行きますか？",
        )

    def test_arrival_with_destination_only_asks_for_origin(self) -> None:
        self.assertEqual(
            handle_no_call("明日9時に品川に着きたい"),
            "出発地と目的地が不足しています。どこからどこまで行きますか？",
        )

    def test_local_clarification(self) -> None:
        text = handle_local_call(
            "ask_clarification",
            {"missing": ["destination"], "question": "目的地を教えてください。"},
        )
        self.assertEqual(text, "目的地を教えてください。")

    def test_clarification_removes_destination_already_present_in_text(self) -> None:
        model_call = ToolCall(
            "ask_clarification",
            {
                "missing": ["origin", "destination"],
                "question": "出発地と目的地を教えてください。",
            },
        )
        refined = _refine_clarification_call(
            model_call, "明日9時に品川に着きたい"
        )
        self.assertEqual(refined.arguments["missing"], ["origin"])
        self.assertEqual(refined.arguments["question"], "出発地が不足しています。どこから出発しますか？")

    def test_clarification_is_not_relaxed_when_runtime_cannot_narrow_it(self) -> None:
        model_call = ToolCall(
            "ask_clarification",
            {
                "missing": ["origin", "destination"],
                "question": "出発地と目的地を教えてください。",
            },
        )
        self.assertIs(_refine_clarification_call(model_call, "乗換少なめで"), model_call)

    def test_complete_but_unrouted_request_does_not_claim_fields_are_missing(self) -> None:
        self.assertEqual(
            handle_no_call("町田から池袋まで行きたい"),
            "出発地と目的地が不足しています。どこからどこまで行きますか？",
        )

    def test_avoid_hint_is_extracted_once(self) -> None:
        hints = extract_route_hints("町田から池袋まで、渋谷を避けて")
        self.assertEqual(hints["avoid_station_texts"], ["渋谷"])

    def test_preferred_line_hint(self) -> None:
        hints = extract_route_hints("横浜から上野まで、京浜東北線で行きたい")
        self.assertEqual(hints["preferred_line_texts"], ["京浜東北線"])

    def test_resolved_geo_endpoints_replace_model_copy_errors(self) -> None:
        result = _route_arguments(
            "plan_journey",
            {"from": "geo:35.46", "to": "geo:35.71", "type": "station"},
            [
                {"name": "横浜駅", "endpoint": "geo:35.4655,139.6231"},
                {"name": "上野駅", "endpoint": "geo:35.7124,139.7767"},
            ],
        )
        self.assertEqual(result["from"], "geo:35.4655,139.6231")
        self.assertEqual(result["to"], "geo:35.7124,139.7767")
        self.assertNotIn("type", result)

    def test_time_mode_requires_explicit_user_wording(self) -> None:
        stations = [
            {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
            {"name": "上野駅", "endpoint": "geo:35.71,139.77"},
        ]
        ordinary = _route_arguments("plan_journey", {"type": "last"}, stations, {})
        last = _route_arguments(
            "plan_journey", {}, stations, {"time_mode": "last_train"}
        )
        self.assertNotIn("type", ordinary)
        self.assertEqual(last["type"], "last")

    def test_requested_route_count_is_sent_to_mcp(self) -> None:
        stations = [
            {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
            {"name": "上野駅", "endpoint": "geo:35.71,139.77"},
        ]
        result = _route_arguments(
            "plan_journey", {}, stations, {}, num_itineraries=1
        )
        self.assertEqual(result["numItineraries"], 1)

    def test_ui_suggestions_prefer_exact_name_and_remove_artwork(self) -> None:
        candidates = [
            {"name": "東京(羽田)空港", "description": "駅"},
            {"name": "東京", "description": "東北新幹線"},
            {"name": "東京", "description": "artwork"},
        ]
        result = _display_suggestions("東京駅", candidates)
        self.assertEqual(result, [{"name": "東京", "description": "東北新幹線"}])

    def test_default_graphical_changes_effective_presentation_only(self) -> None:
        model_intent = {
            "origin_text": "東京",
            "destination_text": "上野",
            "graphical": False,
        }
        effective = _route_intent_with_defaults(
            model_intent, default_graphical=True
        )
        self.assertFalse(model_intent["graphical"])
        self.assertTrue(effective["graphical"])
        call = _planned_call_from_model_intent(effective, [])
        self.assertEqual(call.name, "suggest_stations")
        final_call = _planned_call_from_model_intent(
            effective,
            [
                {"name": "東京駅", "endpoint": "geo:35.68,139.76"},
                {"name": "上野駅", "endpoint": "geo:35.71,139.77"},
            ],
        )
        self.assertEqual(final_call.name, "plan_route_map")

    def test_final_planner_call_is_fully_derived_from_preserved_state(self) -> None:
        intent = {
            "origin_text": "新宿",
            "destination_text": "東京",
            "via_station_texts": ["六本木"],
            "graphical": True,
            "priority": "few_transfers",
            "time_mode": "arrive_by",
            "date": "20260712",
            "time": "09:30",
            "avoid_modes": ["bus"],
        }
        call = _planned_call_from_model_intent(
            intent,
            [
                {"name": "新宿駅", "endpoint": "geo:1,1"},
                {"name": "六本木駅", "endpoint": "geo:2,2"},
                {"name": "東京駅", "endpoint": "geo:3,3"},
            ],
            num_itineraries=6,
        )
        self.assertEqual(call.name, "plan_route_map")
        self.assertEqual(
            call.arguments,
            {
                "from": "geo:1,1",
                "to": "geo:3,3",
                "fromLabel": "新宿駅",
                "toLabel": "東京駅",
                "via": ["geo:2,2"],
                "viaLabel": ["六本木駅"],
                "date": "20260712",
                "time": "09:30",
                "type": "arrival",
                "numItineraries": 6,
                "avoidModes": "bus",
                "strategy": "fewestTransfers",
            },
        )

    def test_route_request_calls_router_once_and_executes_preserved_intent(self) -> None:
        class FakeRouter:
            def __init__(self) -> None:
                self.calls = []

            def generate(self, user_text=None, history=None):
                self.calls.append((user_text, history))
                return (
                    "<start_function_call>call:resolve_route_request{"
                    "origin_text:<escape>新宿<escape>,"
                    "destination_text:<escape>新宿<escape>,"
                    "via_station_texts:[<escape>六本木<escape>],"
                    "graphical:true,priority:<escape>cheap<escape>,"
                    "time_mode:<escape>arrive_by<escape>,"
                    "date:<escape>20260715<escape>,time:<escape>08:10<escape>}"
                    "<end_function_call>"
                )

        class FixedDateTime(RealDateTime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 7, 11, 10, 0, tzinfo=tz)

        class FakeMCPClient:
            instance = None

            def __init__(self, _url) -> None:
                self.calls = []
                self.last_attempts = 1
                FakeMCPClient.instance = self

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def call_tool(self, name, arguments, *, tools):
                self.calls.append((name, arguments))
                return {"result": {"content": [], "structuredContent": {}}}

        stations = {
            "新宿": {"name": "新宿駅", "endpoint": "geo:1,1"},
            "六本木": {"name": "六本木駅", "endpoint": "geo:2,2"},
        }
        resolution_queries = []

        def resolved(query, _suggestions, *, near=None):
            resolution_queries.append(query)
            return {"status": "resolved", "query": query, "station": stations[query]}

        router = FakeRouter()
        with (
            patch("transit_functiongemma.answer_pipeline.MCPClient", FakeMCPClient),
            patch("transit_functiongemma.answer_pipeline.datetime", FixedDateTime),
            patch(
                "transit_functiongemma.answer_pipeline.normalize_mcp_result",
                side_effect=lambda _envelope, tool, _arguments, **_kwargs: (
                    {"suggestions": []}
                    if tool == "suggest_places"
                    else {"status": "ok", "routes": []}
                ),
            ),
            patch(
                "transit_functiongemma.answer_pipeline.resolve_physical_station",
                side_effect=resolved,
            ),
            patch(
                "transit_functiongemma.answer_pipeline.apply_route_constraints",
                side_effect=lambda value: value,
            ),
            patch(
                "transit_functiongemma.answer_pipeline.rerank_routes",
                side_effect=lambda value: value,
            ),
            patch("transit_functiongemma.answer_pipeline.render_answer", return_value="ok"),
        ):
            answer = run_pipeline(
                "明日9時に新宿から六本木経由で新宿へ着きたい。地図で安いルート",
                router_instance=router,
                save_raw=None,
            )

        self.assertEqual(answer, "ok")
        self.assertEqual(len(router.calls), 1)
        self.assertIsNone(router.calls[0][1])
        self.assertEqual(resolution_queries, ["新宿", "六本木", "新宿"])
        final_name, final_arguments = FakeMCPClient.instance.calls[-1]
        self.assertEqual(final_name, "plan_route_map")
        self.assertEqual(
            final_arguments,
            {
                "from": "geo:1,1",
                "to": "geo:1,1",
                "fromLabel": "新宿駅",
                "toLabel": "新宿駅",
                "via": ["geo:2,2"],
                "viaLabel": ["六本木駅"],
                "date": "20260712",
                "time": "09:00",
                "type": "arrival",
                "numItineraries": 6,
                "strategy": "lowestFare",
            },
        )


if __name__ == "__main__":
    unittest.main()
