from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass, field
from os import environ
from typing import Any, Callable, Iterable, Mapping, Sequence

START_FUNCTION_CALL = "<start_function_call>"
END_FUNCTION_CALL = "<end_function_call>"
END_OF_TURN = "<end_of_turn>"
START_FUNCTION_RESPONSE = "<start_function_response>"
END_FUNCTION_RESPONSE = "<end_function_response>"
DEBUG_ASSERT_ENV = "FUNCTIONGEMMA_CONSTRAINED_DECODE_DEBUG_ASSERT"


@dataclass
class TokenPrefixTree:
    children: dict[int, "TokenPrefixTree"] = field(default_factory=dict)
    terminal: bool = False

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[int]]) -> "TokenPrefixTree":
        root = cls()
        for sequence in sequences:
            node = root
            for token_id in sequence:
                node = node.children.setdefault(int(token_id), cls())
            node.terminal = True
        return root

    def node_for(self, prefix: Sequence[int]) -> "TokenPrefixTree | None":
        node: TokenPrefixTree = self
        for token_id in prefix:
            node = node.children.get(int(token_id))
            if node is None:
                return None
        return node

    def next_tokens(self, prefix: Sequence[int]) -> list[int]:
        node = self.node_for(prefix)
        return sorted(node.children) if node is not None else []

    def contains(self, sequence: Sequence[int]) -> bool:
        node = self.node_for(sequence)
        return bool(node and node.terminal)


def constrained_decode_enabled(
    cli_flag: bool, env: Mapping[str, str] | None = None
) -> bool:
    values = environ if env is None else env
    return bool(cli_flag and values.get("FUNCTIONGEMMA_CONSTRAINED_DECODE") == "1")


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [
        int(token_id)
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    ]


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def _resolve_vocab_size(tokenizer: Any, vocab_size: int | None = None) -> int:
    size = (
        vocab_size
        if vocab_size is not None
        else getattr(tokenizer, "vocab_size", None)
    )
    if size is None:
        try:
            size = len(tokenizer)
        except TypeError as exc:
            raise ValueError("tokenizer must expose vocab_size or __len__") from exc
    size = int(size)
    if size <= 0:
        raise ValueError("vocab_size must be positive")
    return size


def _iter_token_ids(value: Any) -> Iterable[int]:
    if value is None:
        return
    if isinstance(value, (str, bytes)):
        return
    if isinstance(value, int):
        yield int(value)
        return
    if isinstance(value, IterableABC):
        for item in value:
            yield from _iter_token_ids(item)
        return
    try:
        yield int(value)
    except (TypeError, ValueError):
        return


def _valid_token_ids(
    value: Any,
    *,
    vocab_size: int | None,
    debug_assert: bool = False,
    context: str = "token ids",
) -> set[int]:
    token_ids = [int(token_id) for token_id in _iter_token_ids(value)]
    invalid = [
        token_id
        for token_id in token_ids
        if token_id < 0 or (vocab_size is not None and token_id >= vocab_size)
    ]
    if debug_assert and invalid:
        raise AssertionError(
            f"{context} contains ids outside [0, {vocab_size}): {invalid[:8]}"
        )
    return {
        token_id
        for token_id in token_ids
        if token_id >= 0 and (vocab_size is None or token_id < vocab_size)
    }


def _vocab_ids(
    tokenizer: Any,
    required: Iterable[int],
    *,
    vocab_size: int | None = None,
    debug_assert: bool = False,
) -> set[int]:
    size = _resolve_vocab_size(tokenizer, vocab_size)
    return set(range(size)) | _valid_token_ids(
        list(required),
        vocab_size=size,
        debug_assert=debug_assert,
        context="required token ids",
    )


def first_valid_token_id(value: Any, vocab_size: int | None = None) -> int | None:
    for token_id in _iter_token_ids(value):
        token_id = int(token_id)
        if token_id >= 0 and (vocab_size is None or token_id < vocab_size):
            return token_id
    return None


def stop_token_ids_for_tokenizer(
    tokenizer: Any,
    extra_stop_token_ids: Iterable[int] | None = None,
    *,
    vocab_size: int | None = None,
    debug_assert: bool = False,
) -> set[int]:
    size = _resolve_vocab_size(tokenizer, vocab_size)
    stop_ids = _valid_token_ids(
        extra_stop_token_ids,
        vocab_size=size,
        debug_assert=debug_assert,
        context="extra stop token ids",
    )
    stop_ids.update(
        _valid_token_ids(
            getattr(tokenizer, "eos_token_id", None),
            vocab_size=size,
            debug_assert=debug_assert,
            context="tokenizer eos token ids",
        )
    )
    for literal in (END_OF_TURN,):
        encoded = _encode(tokenizer, literal)
        if len(encoded) == 1:
            stop_ids.update(
                _valid_token_ids(
                    encoded[0],
                    vocab_size=size,
                    debug_assert=debug_assert,
                    context=f"{literal} token id",
                )
            )
    return stop_ids


@dataclass
class FunctionCallConstraint:
    start_tokens: list[int]
    call_prefix_tree: TokenPrefixTree
    call_prefixes: list[list[int]]
    end_tokens: list[int]
    stop_token_ids: set[int]
    argument_token_ids: set[int]
    tokenizer: Any
    vocab_size: int
    debug_assert: bool = False

    def allowed_next_token_ids(self, generated_token_ids: Sequence[int]) -> list[int]:
        generated = [int(token_id) for token_id in generated_token_ids]
        if not generated:
            return self._allowed(
                {self.start_tokens[0], *self.stop_token_ids}, "initial"
            )
        if len(generated) == 1 and generated[0] in self.stop_token_ids:
            return self._stop_only("already stopped")
        if len(generated) < len(self.start_tokens):
            if generated == self.start_tokens[: len(generated)]:
                return self._allowed(
                    [self.start_tokens[len(generated)]], "function-call start"
                )
            return self._stop_only("invalid function-call start")
        if generated[: len(self.start_tokens)] != self.start_tokens:
            return self._stop_only("invalid generated prefix")

        tail = generated[len(self.start_tokens) :]
        call_prefix = self._matched_call_prefix(tail)
        if call_prefix is None:
            return self._allowed(
                self.call_prefix_tree.next_tokens(tail), "tool name prefix"
            )

        arguments_and_end = tail[len(call_prefix) :]
        return self._allowed_in_arguments(arguments_and_end)

    def _allowed(self, token_ids: Iterable[int], context: str) -> list[int]:
        allowed = _valid_token_ids(
            list(token_ids),
            vocab_size=self.vocab_size,
            debug_assert=self.debug_assert,
            context=context,
        )
        if allowed:
            return sorted(allowed)
        return self._stop_only(f"{context} empty")

    def _stop_only(self, context: str) -> list[int]:
        stop_ids = _valid_token_ids(
            self.stop_token_ids,
            vocab_size=self.vocab_size,
            debug_assert=self.debug_assert,
            context=f"{context} stop token ids",
        )
        if self.debug_assert:
            assert stop_ids, f"{context} produced no valid stop token ids"
        return sorted(stop_ids)

    def _matched_call_prefix(self, tail: Sequence[int]) -> list[int] | None:
        matches = [
            prefix
            for prefix in self.call_prefixes
            if len(tail) >= len(prefix) and list(tail[: len(prefix)]) == prefix
        ]
        return max(matches, key=len) if matches else None

    def _allowed_in_arguments(self, token_ids: Sequence[int]) -> list[int]:
        token_ids = list(token_ids)
        if (
            len(token_ids) >= len(self.end_tokens)
            and token_ids[-len(self.end_tokens) :] == self.end_tokens
            and _decode(self.tokenizer, token_ids[: -len(self.end_tokens)]).endswith("}")
        ):
            return self._stop_only("completed function call")
        for prefix_len in range(len(self.end_tokens) - 1, 0, -1):
            if token_ids[-prefix_len:] == self.end_tokens[:prefix_len] and _decode(
                self.tokenizer, token_ids[:-prefix_len]
            ).endswith("}"):
                return self._allowed(
                    [self.end_tokens[prefix_len]], "function-call end"
                )
        if _decode(self.tokenizer, token_ids).endswith("}"):
            return self._allowed([self.end_tokens[0]], "function-call end start")
        return self._allowed(self.argument_token_ids, "argument token ids")


def build_function_call_constraint(
    tokenizer: Any,
    tool_names: Iterable[str],
    *,
    extra_stop_token_ids: Iterable[int] | None = None,
    vocab_size: int | None = None,
    debug_assert: bool | None = None,
) -> FunctionCallConstraint:
    size = _resolve_vocab_size(tokenizer, vocab_size)
    debug = (
        environ.get(DEBUG_ASSERT_ENV) == "1" if debug_assert is None else debug_assert
    )
    start_tokens = _encode(tokenizer, START_FUNCTION_CALL)
    end_tokens = _encode(tokenizer, END_FUNCTION_CALL)
    if not start_tokens or not end_tokens:
        raise ValueError("function-call delimiters must tokenize to non-empty sequences")
    names = sorted({str(name) for name in tool_names if name})
    if not names:
        raise ValueError("at least one tool name is required for constrained decode")
    call_prefixes = [_encode(tokenizer, f"call:{name}{{") for name in names]
    stop_ids = stop_token_ids_for_tokenizer(
        tokenizer,
        extra_stop_token_ids,
        vocab_size=size,
        debug_assert=debug,
    )
    if not stop_ids:
        raise ValueError("at least one stop token id is required")

    required_tokens = [*start_tokens, *end_tokens, *stop_ids]
    for sequence in call_prefixes:
        required_tokens.extend(sequence)
    all_tokens = _vocab_ids(
        tokenizer,
        required_tokens,
        vocab_size=size,
        debug_assert=debug,
    )
    blocked = set(stop_ids)
    for literal in (
        START_FUNCTION_CALL,
        END_FUNCTION_CALL,
        START_FUNCTION_RESPONSE,
        END_FUNCTION_RESPONSE,
    ):
        encoded = _encode(tokenizer, literal)
        if len(encoded) == 1:
            blocked.update(
                _valid_token_ids(
                    encoded[0],
                    vocab_size=size,
                    debug_assert=debug,
                    context=f"{literal} token id",
                )
            )

    return FunctionCallConstraint(
        start_tokens=start_tokens,
        call_prefix_tree=TokenPrefixTree.from_sequences(call_prefixes),
        call_prefixes=call_prefixes,
        end_tokens=end_tokens,
        stop_token_ids=stop_ids,
        argument_token_ids=all_tokens - blocked,
        tokenizer=tokenizer,
        vocab_size=size,
        debug_assert=debug,
    )


def build_prefix_allowed_tokens_fn(
    constraint: FunctionCallConstraint, prompt_length: int
) -> Callable[[int, Any], list[int]]:
    def prefix_allowed_tokens_fn(_batch_id: int, input_ids: Any) -> list[int]:
        token_ids = (
            input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        )
        return constraint.allowed_next_token_ids(token_ids[prompt_length:])

    return prefix_allowed_tokens_fn


def build_constrained_generate_kwargs(
    tokenizer: Any,
    tool_names: Iterable[str],
    *,
    cli_flag: bool,
    env: Mapping[str, str] | None = None,
    prompt_length: int = 0,
    extra_stop_token_ids: Iterable[int] | None = None,
    vocab_size: int | None = None,
    debug_assert: bool | None = None,
) -> dict[str, Any]:
    if not constrained_decode_enabled(cli_flag, env):
        return {}
    constraint = build_function_call_constraint(
        tokenizer,
        tool_names,
        extra_stop_token_ids=extra_stop_token_ids,
        vocab_size=vocab_size,
        debug_assert=debug_assert,
    )
    return {
        "prefix_allowed_tokens_fn": build_prefix_allowed_tokens_fn(
            constraint, prompt_length
        )
    }


__all__ = [
    "FunctionCallConstraint",
    "TokenPrefixTree",
    "build_constrained_generate_kwargs",
    "build_function_call_constraint",
    "build_prefix_allowed_tokens_fn",
    "constrained_decode_enabled",
    "first_valid_token_id",
    "stop_token_ids_for_tokenizer",
]
