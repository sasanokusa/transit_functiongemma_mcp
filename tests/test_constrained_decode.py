import unittest

from transit_functiongemma.constrained_decode import (
    END_FUNCTION_CALL,
    START_FUNCTION_CALL,
    build_constrained_generate_kwargs,
    build_function_call_constraint,
    first_valid_token_id,
    stop_token_ids_for_tokenizer,
)


class CharTokenizer:
    eos_token_id = 0

    def __init__(self):
        alphabet = (
            START_FUNCTION_CALL
            + END_FUNCTION_CALL
            + "<start_function_response><end_function_response>"
            + "call:suggest_stations{q:<escape>東京<escape>,limit:5}list_feeds{}"
        )
        chars = sorted(set(alphabet))
        self.id_to_char = {0: "<eos>"}
        self.char_to_id = {}
        for index, char in enumerate(chars, 1):
            self.char_to_id[char] = index
            self.id_to_char[index] = char
        self.vocab_size = len(self.id_to_char)

    def __len__(self):
        return self.vocab_size + 16

    def encode(self, text, add_special_tokens=False):
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.id_to_char[token_id] for token_id in token_ids if token_id != 0)


class ConstrainedDecodeTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer()
        self.constraint = build_function_call_constraint(
            self.tokenizer, ["suggest_stations", "list_feeds"]
        )

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def test_first_token_is_no_call_or_function_start_only(self):
        allowed = self.constraint.allowed_next_token_ids([])
        self.assertEqual(
            set(allowed),
            {self.tokenizer.eos_token_id, self.ids(START_FUNCTION_CALL)[0]},
        )

    def test_tool_name_is_limited_by_runtime_prefix_tree(self):
        prefix = self.ids(f"{START_FUNCTION_CALL}call:")
        allowed_chars = {
            self.tokenizer.decode([token_id])
            for token_id in self.constraint.allowed_next_token_ids(prefix)
        }
        self.assertEqual(allowed_chars, {"l", "s"})
        self.assertNotIn(
            self.tokenizer.char_to_id["q"],
            self.constraint.allowed_next_token_ids(prefix),
        )

    def test_end_function_call_must_be_followed_by_stop(self):
        complete = self.ids(
            f"{START_FUNCTION_CALL}call:list_feeds{{}}{END_FUNCTION_CALL}"
        )
        self.assertEqual(
            self.constraint.allowed_next_token_ids(complete),
            [self.tokenizer.eos_token_id],
        )

    def test_response_token_cannot_follow_completed_call(self):
        complete = self.ids(
            f"{START_FUNCTION_CALL}call:suggest_stations{{q:<escape>東京<escape>}}{END_FUNCTION_CALL}"
        )
        allowed = self.constraint.allowed_next_token_ids(complete)
        self.assertNotIn(self.ids("<start_function_response>")[0], allowed)

    def test_generate_kwargs_are_empty_unless_flag_and_env_are_enabled(self):
        self.assertEqual(
            build_constrained_generate_kwargs(
                self.tokenizer,
                ["suggest_stations"],
                cli_flag=False,
                env={"FUNCTIONGEMMA_CONSTRAINED_DECODE": "1"},
            ),
            {},
        )
        self.assertEqual(
            build_constrained_generate_kwargs(
                self.tokenizer,
                ["suggest_stations"],
                cli_flag=True,
                env={},
            ),
            {},
        )
        enabled = build_constrained_generate_kwargs(
            self.tokenizer,
            ["suggest_stations"],
            cli_flag=True,
            env={"FUNCTIONGEMMA_CONSTRAINED_DECODE": "1"},
        )
        self.assertIn("prefix_allowed_tokens_fn", enabled)

    def test_allowed_ids_are_clamped_to_model_vocab_size(self):
        constraint = build_function_call_constraint(
            self.tokenizer,
            ["suggest_stations"],
            extra_stop_token_ids=[None, -1, self.tokenizer.vocab_size + 3, 0],
            vocab_size=self.tokenizer.vocab_size,
        )
        generated = self.ids(f"{START_FUNCTION_CALL}call:suggest_stations{{")
        allowed = constraint.allowed_next_token_ids(generated)
        self.assertTrue(allowed)
        self.assertTrue(
            all(0 <= token_id < self.tokenizer.vocab_size for token_id in allowed)
        )
        self.assertLess(max(allowed), self.tokenizer.vocab_size)

    def test_invalid_prefix_falls_back_to_stop_tokens_not_empty(self):
        constraint = build_function_call_constraint(
            self.tokenizer,
            ["suggest_stations"],
            vocab_size=self.tokenizer.vocab_size,
        )
        self.assertEqual(
            constraint.allowed_next_token_ids([self.tokenizer.vocab_size + 9]),
            [self.tokenizer.eos_token_id],
        )

    def test_stop_tokens_ignore_none_negative_and_out_of_range_ids(self):
        self.tokenizer.eos_token_id = [0, None, -1, self.tokenizer.vocab_size]
        stop_ids = stop_token_ids_for_tokenizer(
            self.tokenizer,
            extra_stop_token_ids=[None, -2, self.tokenizer.vocab_size + 1],
            vocab_size=self.tokenizer.vocab_size,
        )
        self.assertEqual(stop_ids, {0})
        self.assertEqual(first_valid_token_id(self.tokenizer.eos_token_id, 1), 0)

    def test_debug_assert_reports_out_of_range_ids(self):
        with self.assertRaises(AssertionError):
            build_function_call_constraint(
                self.tokenizer,
                ["suggest_stations"],
                extra_stop_token_ids=[self.tokenizer.vocab_size],
                vocab_size=self.tokenizer.vocab_size,
                debug_assert=True,
            )


if __name__ == "__main__":
    unittest.main()
