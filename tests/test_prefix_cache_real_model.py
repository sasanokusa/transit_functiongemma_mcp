import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoConfig  # noqa: E402

from transit_functiongemma.config import MODEL_ID  # noqa: E402
from transit_functiongemma.infer import ToolRouter  # noqa: E402


def real_model_path() -> str:
    candidates = [
        os.environ.get("FUNCTIONGEMMA_PREFIX_CACHE_TEST_MODEL"),
        "gguf_work/merged_r8b",
        MODEL_ID,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            AutoConfig.from_pretrained(candidate, local_files_only=True)
        except Exception:
            continue
        return candidate
    pytest.skip("FunctionGemma-compatible base model is not cached locally")


def test_prefix_cache_matches_uncached_greedy_outputs(monkeypatch):
    base_model = real_model_path()
    monkeypatch.setenv("FUNCTIONGEMMA_PREFIX_CACHE", "1")
    adapter_env = None
    for name in (
        "FUNCTIONGEMMA_PREFIX_CACHE_TEST_ADAPTER",
        "FUNCTIONGEMMA_ADAPTER",
    ):
        value = os.environ.get(name)
        if value and Path(value).exists():
            adapter_env = value
            break
    router = ToolRouter(
        base_model=base_model,
        adapter=adapter_env,
        schema_mode="baked",
        clarification_tool=True,
        normalize_ja=False,
        prefix_cache=True,
    )
    if router.prefix_cache_state is None:
        pytest.skip("prefix cache was not initialized")

    cases = [
        {"prompt": "東京駅を検索して"},
        {"prompt": "これは雑談です"},
        {
            "prompt": None,
            "history": [
                {"role": "user", "content": "東京駅を検索して"},
                {
                    "role": "assistant",
                    "content": (
                        "<start_function_call>call:suggest_stations"
                        "{q:<escape>東京駅<escape>,limit:5}<end_function_call>"
                    ),
                },
                {"role": "user", "content": "横浜も見たい"},
            ],
        },
    ]
    for case in cases:
        router.prefix_cache_enabled = False
        uncached = router.generate(
            case.get("prompt"),
            reference_datetime="2026-07-09 12:00 Asia/Tokyo",
            max_new_tokens=24,
            history=case.get("history"),
        )
        router.prefix_cache_enabled = True
        cached = router.generate(
            case.get("prompt"),
            reference_datetime="2026-07-09 12:00 Asia/Tokyo",
            max_new_tokens=24,
            history=case.get("history"),
        )
        assert router.last_prefix_cache_hit
        assert cached == uncached
