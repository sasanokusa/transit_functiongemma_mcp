import unittest

from transit_functiongemma.toolcall import ToolCall
from transit_functiongemma.validation import (
    ToolCallSchemaError,
    validate_tool_call,
    validate_tool_name,
)


TOOLS = [
    {
        "name": "suggest_stations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["q"],
            "additionalProperties": False,
        },
    }
]


class ToolCallValidationTest(unittest.TestCase):
    def test_accepts_allow_listed_schema_valid_call(self) -> None:
        validate_tool_call(ToolCall("suggest_stations", {"q": "東京", "limit": 5}), TOOLS)

    def test_rejects_hallucinated_tool(self) -> None:
        with self.assertRaisesRegex(ToolCallSchemaError, "not allow-listed"):
            validate_tool_call(ToolCall("suggest_feeds", {}), TOOLS)

    def test_rejects_missing_required_argument(self) -> None:
        with self.assertRaisesRegex(ToolCallSchemaError, "required property"):
            validate_tool_call(ToolCall("suggest_stations", {"limit": 5}), TOOLS)

    def test_rejects_unknown_argument(self) -> None:
        with self.assertRaisesRegex(ToolCallSchemaError, "Additional properties"):
            validate_tool_call(ToolCall("suggest_stations", {"q": "東京", "foo": 1}), TOOLS)

    def test_name_only_check_allows_deferred_argument_cleanup(self) -> None:
        validate_tool_name("suggest_stations", TOOLS)


if __name__ == "__main__":
    unittest.main()
