import unittest

from transit_functiongemma.toolcall import ToolCallParseError, parse_tool_calls


class ToolCallParserTest(unittest.TestCase):
    def test_nested_values(self) -> None:
        raw = (
            "<start_function_call>call:plan_journey{from:<escape>a:1<escape>,"
            "to:<escape>b:2<escape>,via:[<escape>c:3<escape>],maxTransfers:2}"
            "<end_function_call><start_function_response>"
        )
        call = parse_tool_calls(raw)[0]
        self.assertEqual(call.name, "plan_journey")
        self.assertEqual(call.arguments["via"], ["c:3"])
        self.assertEqual(call.arguments["maxTransfers"], 2)

    def test_no_call(self) -> None:
        self.assertEqual(parse_tool_calls("<end_of_turn>"), [])

    def test_bare_none_token_parses_as_null(self) -> None:
        raw = (
            "<start_function_call>call:resolve_route_request{"
            "origin_text:<escape>東京<escape>,destination_text:<escape>新宿<escape>,"
            "priority:None,time_mode:None,date:null,time:None}"
            "<end_function_call>"
        )
        call = parse_tool_calls(raw)[0]
        self.assertIsNone(call.arguments["priority"])
        self.assertIsNone(call.arguments["time_mode"])
        self.assertIsNone(call.arguments["date"])
        self.assertIsNone(call.arguments["time"])

    def test_rejects_mixed_prose(self) -> None:
        raw = "了解です。<start_function_call>call:list_feeds{}<end_function_call>"
        with self.assertRaises(ToolCallParseError):
            parse_tool_calls(raw)


if __name__ == "__main__":
    unittest.main()

