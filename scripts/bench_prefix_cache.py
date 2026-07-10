#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transit_functiongemma.config import DEVELOPER_PROMPT, MODEL_ID
from transit_functiongemma.infer import PREFIX_CACHE_ENV, ToolRouter


def measure(
    router: ToolRouter,
    query: str,
    reference_datetime: str,
    args: argparse.Namespace,
    enabled: bool,
):
    router.prefix_cache_enabled = enabled
    latencies: list[float] = []
    outputs: list[str] = []
    hits: list[bool] = []
    for _ in range(args.runs):
        started = time.perf_counter()
        outputs.append(
            router.generate(
                query,
                reference_datetime=reference_datetime,
                max_new_tokens=args.max_new_tokens,
            )
        )
        latencies.append(time.perf_counter() - started)
        hits.append(bool(router.last_prefix_cache_hit))
    return latencies, outputs, hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ToolRouter prefix KV cache.")
    parser.add_argument(
        "--base-model",
        default=os.getenv("FUNCTIONGEMMA_PREFIX_CACHE_TEST_MODEL", MODEL_ID),
    )
    parser.add_argument("--adapter", default=os.getenv("FUNCTIONGEMMA_ADAPTER"))
    parser.add_argument("--query", default="東京駅を検索して")
    parser.add_argument("--reference-datetime", default="2026-07-09 12:00 Asia/Tokyo")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prefix_cache_bench_local.json"),
    )
    args = parser.parse_args()

    if os.getenv(PREFIX_CACHE_ENV) != "1":
        raise SystemExit(f"Set {PREFIX_CACHE_ENV}=1 before running this benchmark.")

    router = ToolRouter(
        base_model=args.base_model,
        adapter=args.adapter,
        schema_mode="baked",
        clarification_tool=True,
        normalize_ja=False,
        prefix_cache=True,
    )
    messages = [
        {
            "role": "developer",
            "content": DEVELOPER_PROMPT.format(now=args.reference_datetime),
        },
        {"role": "user", "content": args.query},
    ]
    full_inputs = router._chat_inputs(messages, add_generation_prompt=True)
    full_tokens = int(full_inputs["input_ids"].shape[1])
    prefix_tokens = int(router.prefix_cache_tokens)

    router.prefix_cache_enabled = False
    uncached_warm = router.generate(
        args.query,
        reference_datetime=args.reference_datetime,
        max_new_tokens=args.max_new_tokens,
    )
    router.prefix_cache_enabled = True
    cached_warm = router.generate(
        args.query,
        reference_datetime=args.reference_datetime,
        max_new_tokens=args.max_new_tokens,
    )
    if uncached_warm != cached_warm:
        raise SystemExit("cached output differed from uncached output during warmup")

    uncached, uncached_outputs, uncached_hits = measure(
        router, args.query, args.reference_datetime, args, False
    )
    cached, cached_outputs, cached_hits = measure(
        router, args.query, args.reference_datetime, args, True
    )
    if uncached_outputs != cached_outputs:
        raise SystemExit("cached outputs differed from uncached outputs")

    result = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "query": args.query,
        "reference_datetime": args.reference_datetime,
        "runs": args.runs,
        "max_new_tokens": args.max_new_tokens,
        "full_prompt_tokens": full_tokens,
        "prefix_cache_tokens": prefix_tokens,
        "suffix_prompt_tokens": full_tokens - prefix_tokens,
        "uncached_seconds": uncached,
        "cached_seconds": cached,
        "uncached_avg_seconds": statistics.mean(uncached),
        "cached_avg_seconds": statistics.mean(cached),
        "speedup": statistics.mean(uncached) / statistics.mean(cached),
        "cached_hits": cached_hits,
        "uncached_hits": uncached_hits,
        "output": cached_outputs[0] if cached_outputs else "",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
