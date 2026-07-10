import random
from pathlib import Path

import pytest

from transit_functiongemma.config import MODEL_ID
from transit_functiongemma.constrained_decode import (
    START_FUNCTION_CALL,
    build_function_call_constraint,
)


class TokenizersWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            pytest.skip("cached FunctionGemma tokenizer does not expose <eos>")
        self.eos_token_id = eos_id
        self.vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)

    def __len__(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )


def load_cached_functiongemma_tokenizer():
    pytest.importorskip("huggingface_hub")
    tokenizers = pytest.importorskip("tokenizers")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            MODEL_ID,
            allow_patterns=["tokenizer.json"],
            local_files_only=True,
        )
    except Exception as exc:
        pytest.skip(f"FunctionGemma tokenizer is not cached locally: {exc}")
    tokenizer_path = Path(snapshot) / "tokenizer.json"
    if not tokenizer_path.exists():
        pytest.skip("cached FunctionGemma tokenizer has no tokenizer.json")
    return TokenizersWrapper(tokenizers.Tokenizer.from_file(str(tokenizer_path)))


def assert_indexes_mock_logits(allowed_ids, vocab_size):
    assert allowed_ids
    assert all(0 <= token_id < vocab_size for token_id in allowed_ids)
    logits = [random.random() for _ in range(vocab_size)]
    for token_id in allowed_ids:
        _ = logits[token_id]


def test_real_tokenizer_allowed_ids_are_bounded_for_model_vocab_size():
    tokenizer = load_cached_functiongemma_tokenizer()
    model_vocab_size = tokenizer.vocab_size
    constraint = build_function_call_constraint(
        tokenizer,
        ["suggest_stations", "station_departures"],
        extra_stop_token_ids=[tokenizer.eos_token_id, None, -1, model_vocab_size],
        vocab_size=model_vocab_size,
    )

    states = [
        [],
        constraint.start_tokens[:1],
        tokenizer.encode(f"{START_FUNCTION_CALL}call:", add_special_tokens=False),
        constraint.start_tokens + constraint.call_prefixes[0],
    ]
    for state in states:
        assert_indexes_mock_logits(
            constraint.allowed_next_token_ids(state), model_vocab_size
        )
