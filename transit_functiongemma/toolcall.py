from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

START = "<start_function_call>"
END = "<end_function_call>"


class ToolCallParseError(ValueError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


class _ValueParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse_object(self) -> dict[str, Any]:
        self._expect("{")
        result: dict[str, Any] = {}
        self._space()
        if self._take("}"):
            return result
        while True:
            key = self._bare({":"}).strip()
            if not key:
                raise ToolCallParseError(f"Missing argument name at offset {self.pos}")
            self._expect(":")
            result[key] = self.parse_value()
            self._space()
            if self._take("}"):
                return result
            self._expect(",")

    def parse_array(self) -> list[Any]:
        self._expect("[")
        values: list[Any] = []
        self._space()
        if self._take("]"):
            return values
        while True:
            values.append(self.parse_value())
            self._space()
            if self._take("]"):
                return values
            self._expect(",")

    def parse_value(self) -> Any:
        self._space()
        if self.text.startswith("<escape>", self.pos):
            self.pos += len("<escape>")
            end = self.text.find("<escape>", self.pos)
            if end < 0:
                raise ToolCallParseError("Unterminated <escape> string")
            value = self.text[self.pos:end]
            self.pos = end + len("<escape>")
            return value
        if self._peek() == "{":
            return self.parse_object()
        if self._peek() == "[":
            return self.parse_array()
        token = self._bare({",", "}", "]"}).strip()
        if token == "true":
            return True
        if token == "false":
            return False
        if token in ("null", "None"):
            # FunctionGemma's own chat_template.jinja renders a Python None
            # argument via Jinja's bare {{ value }} fallback, which stringifies
            # it as "None" (not JSON "null"). Models trained on such targets
            # emit the literal token None, so both spellings must parse to null.
            return None
        if re.fullmatch(r"-?\d+", token):
            return int(token)
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", token):
            return float(token)
        return token.strip("\"'")

    def _bare(self, stops: set[str]) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in stops:
            self.pos += 1
        return self.text[start:self.pos]

    def _space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _take(self, value: str) -> bool:
        self._space()
        if self.text.startswith(value, self.pos):
            self.pos += len(value)
            return True
        return False

    def _expect(self, value: str) -> None:
        if not self._take(value):
            raise ToolCallParseError(f"Expected {value!r} at offset {self.pos}")


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Strictly parse FunctionGemma calls; natural-language output yields no calls."""
    calls: list[ToolCall] = []
    cursor = 0
    outside: list[str] = []
    while True:
        start = text.find(START, cursor)
        if start < 0:
            break
        outside.append(text[cursor:start])
        body_start = start + len(START)
        end = text.find(END, body_start)
        if end < 0:
            raise ToolCallParseError("Function call has no end marker")
        body = text[body_start:end].strip()
        match = re.fullmatch(r"call:([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})", body, re.DOTALL)
        if not match:
            raise ToolCallParseError(f"Invalid function-call body: {body[:120]!r}")
        parser = _ValueParser(match.group(2))
        arguments = parser.parse_object()
        parser._space()
        if parser.pos != len(parser.text):
            raise ToolCallParseError(f"Trailing call data at offset {parser.pos}")
        calls.append(ToolCall(match.group(1), arguments))
        cursor = end + len(END)
    outside.append(text[cursor:])
    if calls:
        remainder = "".join(outside)
        # These are control tokens FunctionGemma may emit around a valid call.
        remainder = re.sub(
            r"<(?:bos|eos|end_of_turn|start_function_response|end_function_response)>",
            "",
            remainder,
        )
        if remainder.strip():
            raise ToolCallParseError("Refusing mixed natural-language and function-call output")
    return calls
