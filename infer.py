#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, DEVELOPER_PROMPT, MODEL_ID
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


class ToolRouter:
    def __init__(
        self,
        base_model: str = MODEL_ID,
        adapter: str | None = None,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        schema_mode: str = "baked",
        clarification_tool: bool = False,
        normalize_ja: bool = False,
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
        if schema_mode == "full":
            self.tools = to_functiongemma_tools(mcp_tools)
        elif schema_mode == "compact":
            self.tools = compact_functiongemma_tools(mcp_tools)
        else:
            self.tools = []

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None,
        reference_datetime: str | None = None,
        max_new_tokens: int = 128,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        now = reference_datetime or datetime.now(ZoneInfo("Asia/Tokyo")).strftime(
            "%Y-%m-%d %H:%M Asia/Tokyo"
        )
        messages: list[dict[str, Any]] = [
            {"role": "developer", "content": DEVELOPER_PROMPT.format(now=now)}
        ]
        rendered_history = history or []
        if self.normalize_ja:
            rendered_history = normalize_user_messages(rendered_history, now)
        messages.extend(rendered_history)
        if prompt is not None:
            if self.normalize_ja:
                prompt = normalize_japanese_prompt(prompt, now)
            messages.append({"role": "user", "content": prompt})
        if len(messages) == 1:
            raise ValueError("prompt or history is required")
        kwargs = {"tools": self.tools} if self.tools else {}
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        ).to(getattr(self.model, "device", "cpu"))
        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.eos_token_id,
        )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
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
    args = parser.parse_args()

    router = ToolRouter(
        args.base_model,
        args.adapter,
        args.schema,
        args.schema_mode,
        args.clarification_tool,
        args.normalize_ja,
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
