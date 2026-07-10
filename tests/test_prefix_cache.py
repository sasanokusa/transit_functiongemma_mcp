import logging

import pytest

torch = pytest.importorskip("torch")

from transit_functiongemma.infer import (  # noqa: E402
    PREFIX_CACHE_ENV,
    PrefixCacheState,
    ToolRouter,
    prefix_cache_enabled,
)


class FakeCache:
    def get_seq_length(self):
        return 3


def make_router(prefix_ids):
    router = object.__new__(ToolRouter)
    router.prefix_cache_enabled = True
    router.prefix_cache_state = PrefixCacheState(
        input_ids=torch.tensor([prefix_ids], dtype=torch.long),
        attention_mask=torch.ones((1, len(prefix_ids)), dtype=torch.long),
        past_key_values=FakeCache(),
    )
    router.last_prefix_cache_hit = False
    return router


def test_prefix_cache_env_and_cli_must_both_be_enabled(caplog):
    caplog.set_level(logging.WARNING)
    assert prefix_cache_enabled(True, {PREFIX_CACHE_ENV: "1"})
    assert not prefix_cache_enabled(False, {PREFIX_CACHE_ENV: "1"})
    assert not prefix_cache_enabled(True, {})
    warnings = [record.message for record in caplog.records]
    assert sum("Prefix cache disabled" in message for message in warnings) == 2


def test_prefix_cache_mismatch_falls_back_to_full_prompt():
    router = make_router([1, 2, 3])
    inputs = {
        "input_ids": torch.tensor([[1, 9, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    generate_inputs, prompt_len, hit = ToolRouter._inputs_with_prefix_cache(
        router, inputs
    )
    assert not hit
    assert not router.last_prefix_cache_hit
    assert prompt_len == 4
    assert generate_inputs is inputs
    assert "past_key_values" not in generate_inputs


def test_prefix_cache_hit_passes_suffix_and_copied_cache():
    router = make_router([1, 2, 3])
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
    }
    generate_inputs, prompt_len, hit = ToolRouter._inputs_with_prefix_cache(
        router, inputs
    )
    assert hit
    assert router.last_prefix_cache_hit
    assert prompt_len == 2
    assert generate_inputs["input_ids"].tolist() == [[4, 5]]
    assert generate_inputs["attention_mask"].shape == (1, 5)
    assert (
        generate_inputs["past_key_values"]
        is not router.prefix_cache_state.past_key_values
    )


def test_prefix_cache_default_off_leaves_inputs_unchanged():
    router = make_router([1, 2, 3])
    router.prefix_cache_enabled = False
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    generate_inputs, prompt_len, hit = ToolRouter._inputs_with_prefix_cache(
        router, inputs
    )
    assert not hit
    assert prompt_len == 4
    assert generate_inputs is inputs
