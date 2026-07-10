#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, DEVELOPER_PROMPT, MODEL_ID
from transit_functiongemma.constrained_decode import (
    build_constrained_generate_kwargs,
    constrained_decode_enabled,
    first_valid_token_id,
)
from transit_functiongemma.japanese import (
    bind_normalized_tool_call,
    normalize_japanese_prompt,
    normalize_user_messages,
)
from transit_functiongemma.schemas import (
    CLARIFICATION_TOOL_NAME,
    compact_functiongemma_tools,
    load_mcp_tools,
    tools_with_clarification,
    to_functiongemma_tools,
)
from transit_functiongemma.toolcall import ToolCallParseError, parse_tool_calls
from transit_functiongemma.validation import ToolCallSchemaError, validate_tool_calls

PREFIX_CACHE_ENV = "FUNCTIONGEMMA_PREFIX_CACHE"
PREFIX_CACHE_EXAMPLE_NOW_A = "2000-01-01 00:00 Asia/Tokyo"
PREFIX_CACHE_EXAMPLE_NOW_B = "2099-12-31 23:59 Asia/Tokyo"
logger = logging.getLogger(__name__)


@dataclass
class PrefixCacheState:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Any


def prefix_cache_enabled(
    cli_flag: bool, env: Mapping[str, str] | None = None
) -> bool:
    values = os.environ if env is None else env
    env_enabled = values.get(PREFIX_CACHE_ENV) == "1"
    if bool(cli_flag) != env_enabled:
        logger.warning(
            "Prefix cache disabled: set both %s=1 and --prefix-cache to enable it.",
            PREFIX_CACHE_ENV,
        )
    return bool(cli_flag and env_enabled)


def build_router_messages(
    prompt: str | None,
    reference_datetime: str | None = None,
    history: list[dict[str, Any]] | None = None,
    *,
    normalize_ja: bool = False,
) -> list[dict[str, Any]]:
    now = reference_datetime or datetime.now(ZoneInfo("Asia/Tokyo")).strftime(
        "%Y-%m-%d %H:%M Asia/Tokyo"
    )
    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": DEVELOPER_PROMPT.format(now=now)}
    ]
    rendered_history = history or []
    if normalize_ja:
        rendered_history = normalize_user_messages(rendered_history, now)
    messages.extend(rendered_history)
    if prompt is not None:
        if normalize_ja:
            prompt = normalize_japanese_prompt(prompt, now)
        messages.append({"role": "user", "content": prompt})
    if len(messages) == 1:
        raise ValueError("prompt or history is required")
    return messages


def apply_router_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    tokenize: bool = True,
    return_tensors: str | None = None,
    return_dict: bool = False,
) -> Any:
    kwargs = {"tools": tools} if tools else {}
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
        return_dict=return_dict,
        return_tensors=return_tensors,
        **kwargs,
    )


def render_router_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    return str(
        apply_router_chat_template(
            processor,
            messages,
            tools,
            add_generation_prompt=True,
            tokenize=False,
        )
    )


class ToolRouter:
    def __init__(
        self,
        base_model: str = MODEL_ID,
        adapter: str | None = None,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        schema_mode: str = "baked",
        clarification_tool: bool = False,
        normalize_ja: bool = False,
        constrained_decode: bool = False,
        prefix_cache: bool = False,
    ):
        self.processor = AutoProcessor.from_pretrained(adapter or base_model)
        if torch.cuda.is_available():
            # GTX 1650: directly storing FunctionGemma's large RMSNorm weights as
            # fp16 produces NaN logits. The 270M fp32 base still fits in 4GB;
            # training uses fp16 autocast, while inference stays stable. eager
            # attention avoids the same overflow path.
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.float32,
                device_map="auto",
                attn_implementation="eager",
            )
            if adapter:
                self.model = PeftModel.from_pretrained(self.model, adapter)
        else:
            # CPU (cluster): skip device_map="auto" so accelerate does not wrap
            # every submodule in dispatch hooks, and use sdpa attention, which is
            # faster than eager and numerically equivalent for fp32 inference.
            torch.set_num_threads(
                int(os.getenv("FUNCTIONGEMMA_THREADS", str(os.cpu_count() or 4)))
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            if adapter:
                self.model = PeftModel.from_pretrained(self.model, adapter)
            if os.getenv("FUNCTIONGEMMA_QUANTIZE") == "1":
                # Opt-in int8 dynamic quantization (ARM qnnpack backend). Roughly
                # halves latency and memory, but shifts some tool-call outputs on
                # this small model, so it is off by default. Merge the LoRA first
                # because quantized Linear layers cannot host PEFT adapters.
                torch.backends.quantized.engine = "qnnpack"
                merged = self.model.merge_and_unload() if adapter else self.model
                self.model = torch.ao.quantization.quantize_dynamic(
                    merged, {torch.nn.Linear}, dtype=torch.qint8
                )
        self.model.eval()
        self.normalize_ja = normalize_ja
        mcp_tools = tools_with_clarification(
            load_mcp_tools(schema_path), clarification_tool
        )
        self.validation_tools = mcp_tools
        self.constrained_decode = constrained_decode_enabled(
            constrained_decode, os.environ
        )
        self.constrained_tool_names = [str(tool["name"]) for tool in mcp_tools]
        self.prefix_cache_enabled = prefix_cache_enabled(prefix_cache, os.environ)
        self.prefix_cache_state: PrefixCacheState | None = None
        self.prefix_cache_tokens = 0
        self.last_prefix_cache_hit = False
        if schema_mode == "full":
            self.tools = to_functiongemma_tools(mcp_tools)
        elif schema_mode == "compact":
            self.tools = compact_functiongemma_tools(mcp_tools)
        else:
            self.tools = []
        if self.prefix_cache_enabled:
            if schema_mode != "baked":
                logger.warning(
                    "Prefix cache disabled: only baked schema mode is supported."
                )
                self.prefix_cache_enabled = False
            else:
                self.prefix_cache_state = self._build_prefix_cache()
                self.prefix_cache_tokens = int(
                    self.prefix_cache_state.input_ids.shape[1]
                )

    def _device(self) -> torch.device | str:
        return getattr(self.model, "device", "cpu")

    def _chat_inputs(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> Any:
        return apply_router_chat_template(
            self.processor,
            messages,
            self.tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device())

    def _prefix_messages(self, now: str) -> list[dict[str, Any]]:
        return [{"role": "developer", "content": DEVELOPER_PROMPT.format(now=now)}]

    def _common_prefix_length(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> int:
        left_ids = left[0]
        right_ids = right[0]
        limit = min(int(left_ids.shape[0]), int(right_ids.shape[0]))
        if limit == 0:
            return 0
        mismatch = (left_ids[:limit] != right_ids[:limit]).nonzero(as_tuple=False)
        return limit if mismatch.numel() == 0 else int(mismatch[0].item())

    def _build_prefix_cache(self) -> PrefixCacheState:
        first = self._chat_inputs(
            self._prefix_messages(PREFIX_CACHE_EXAMPLE_NOW_A),
            add_generation_prompt=False,
        )
        second = self._chat_inputs(
            self._prefix_messages(PREFIX_CACHE_EXAMPLE_NOW_B),
            add_generation_prompt=False,
        )
        prefix_len = self._common_prefix_length(first["input_ids"], second["input_ids"])
        if prefix_len <= 0:
            raise RuntimeError("prefix cache could not find a stable token prefix")

        input_ids = first["input_ids"][:, :prefix_len].contiguous()
        attention_mask = first.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask[:, :prefix_len].contiguous()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None or not hasattr(past_key_values, "get_seq_length"):
            raise RuntimeError("model did not return a Transformers Cache object")
        return PrefixCacheState(
            input_ids=input_ids.detach().clone(),
            attention_mask=attention_mask.detach().clone(),
            past_key_values=past_key_values,
        )

    def _prefix_cache_matches(self, input_ids: torch.Tensor) -> bool:
        state = self.prefix_cache_state
        if state is None:
            return False
        prefix_len = int(state.input_ids.shape[1])
        if int(input_ids.shape[1]) <= prefix_len:
            return False
        prefix_ids = state.input_ids.to(device=input_ids.device)
        return torch.equal(input_ids[:, :prefix_len], prefix_ids)

    def _inputs_with_prefix_cache(self, inputs: Any) -> tuple[Any, int, bool]:
        full_prompt_len = int(inputs["input_ids"].shape[1])
        self.last_prefix_cache_hit = False
        if not self.prefix_cache_enabled or not self._prefix_cache_matches(
            inputs["input_ids"]
        ):
            return inputs, full_prompt_len, False

        state = self.prefix_cache_state
        assert state is not None
        prefix_len = int(state.input_ids.shape[1])
        generate_inputs = dict(inputs)
        generate_inputs["input_ids"] = inputs["input_ids"][:, prefix_len:].contiguous()
        if "attention_mask" not in generate_inputs:
            generate_inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        generate_inputs["past_key_values"] = copy.deepcopy(state.past_key_values)
        self.last_prefix_cache_hit = True
        return generate_inputs, int(generate_inputs["input_ids"].shape[1]), True

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None,
        reference_datetime: str | None = None,
        max_new_tokens: int = 128,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        messages = build_router_messages(
            prompt,
            reference_datetime,
            history,
            normalize_ja=self.normalize_ja,
        )
        inputs = self._chat_inputs(messages, add_generation_prompt=True)
        generate_inputs, generated_prompt_len, _cache_hit = self._inputs_with_prefix_cache(
            inputs
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        model_vocab_size = getattr(
            getattr(self.model, "config", None), "vocab_size", None
        )
        eos_token_id = self.processor.eos_token_id
        constrained_kwargs = build_constrained_generate_kwargs(
            tokenizer,
            self.constrained_tool_names,
            cli_flag=self.constrained_decode,
            env=os.environ,
            prompt_length=generated_prompt_len,
            extra_stop_token_ids=eos_token_id,
            vocab_size=model_vocab_size,
        )
        pad_token_id = self.processor.eos_token_id
        if isinstance(pad_token_id, (list, tuple, set)):
            pad_token_id = first_valid_token_id(pad_token_id, model_vocab_size)
        if pad_token_id is None:
            pad_token_id = first_valid_token_id(
                getattr(self.processor, "pad_token_id", None), model_vocab_size
            )
        generated = self.model.generate(
            **generate_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            **constrained_kwargs,
        )
        new_tokens = generated[0][generated_prompt_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a transit MCP tool call.")
    parser.add_argument("prompt")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--adapter")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--schema-mode", choices=("baked", "compact", "full"), default="baked")
    parser.add_argument("--reference-datetime")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--clarification-tool", action="store_true")
    parser.add_argument("--normalize-ja", action="store_true")
    parser.add_argument("--constrained-decode", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    args = parser.parse_args()

    router = ToolRouter(
        args.base_model,
        args.adapter,
        args.schema,
        args.schema_mode,
        args.clarification_tool,
        args.normalize_ja,
        args.constrained_decode,
        args.prefix_cache,
    )
    raw = router.generate(args.prompt, args.reference_datetime, args.max_new_tokens)
    try:
        parsed_calls = parse_tool_calls(raw)
        if args.normalize_ja and len(parsed_calls) == 1:
            bound_call = bind_normalized_tool_call(
                parsed_calls[0], args.prompt, args.reference_datetime
            )
            parsed_calls = [] if bound_call is None else [bound_call]
        validate_tool_calls(parsed_calls, router.validation_tools)
        calls = [call.as_dict() for call in parsed_calls]
        error = None
    except (ToolCallParseError, ToolCallSchemaError) as exc:
        calls, error = [], str(exc)
    clarification = next(
        (call["arguments"] for call in calls if call["name"] == CLARIFICATION_TOOL_NAME),
        None,
    )
    print(
        json.dumps(
            {
                "raw_output": raw,
                "tool_calls": calls,
                "clarification": clarification,
                "parse_error": error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
