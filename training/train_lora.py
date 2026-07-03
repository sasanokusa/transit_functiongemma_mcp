#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from transit_functiongemma.config import MODEL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for the transit tool router.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="Continue training an existing LoRA adapter instead of creating a new one.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/sft_generated.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/functiongemma-transit-lora"),
    )
    parser.add_argument("--qlora", action="store_true", help="Load the base model in NF4 4-bit.")
    parser.add_argument("--max-seq-length", type=int, choices=(256, 512), default=512)
    parser.add_argument("--lora-rank", type=int, choices=(4, 8), default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this fp16 training recipe")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    processor = AutoProcessor.from_pretrained(args.init_adapter or args.model)
    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # FunctionGemma's pretrained RMSNorm weights can overflow when every
        # frozen base parameter is forcibly stored as fp16. Standard LoRA keeps
        # the frozen base in fp32 while Trainer uses fp16 autocast; QLoRA keeps
        # quantized linears plus fp32 norms and fp16 compute.
        dtype=torch.float16 if args.qlora else torch.float32,
        quantization_config=quantization_config,
        device_map={"": 0},
        attn_implementation="eager",
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.init_adapter:
        model = PeftModel.from_pretrained(
            model, str(args.init_adapter), is_trainable=True
        )
    model.config.use_cache = False

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise ValueError("At least two SFT records are required")

    # Arrow cannot represent messages where `content` is a string for normal
    # turns and an object for FunctionGemma tool-result turns. Render/tokenize
    # each Python record first, then construct an integer-only Dataset.
    tokenized_rows: list[list[int]] = []
    completion_masks: list[list[int]] = []
    dropped = 0
    for row in rows:
        kwargs = {"tools": row["tools"]} if row.get("tools") else {}
        token_ids = processor.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False, **kwargs
        )
        prompt_ids = processor.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True, **kwargs
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if token_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{row.get('id', '<unknown>')}: assistant prompt prefix mismatch")
        if len(token_ids) <= args.max_seq_length:
            tokenized_rows.append(token_ids)
            completion_masks.append(
                [0] * len(prompt_ids) + [1] * (len(token_ids) - len(prompt_ids))
            )
        else:
            dropped += 1
    if not tokenized_rows:
        raise ValueError(
            "All records exceed max_seq_length. Use --schema-mode baked in prepare_sft.py."
        )
    dataset = Dataset.from_dict(
        {"input_ids": tokenized_rows, "completion_mask": completion_masks}
    )
    print(f"usable records: {len(dataset)}; dropped over-length: {dropped}")
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    train_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        completion_only_loss=True,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        lr_scheduler_type="constant",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer_kwargs = dict(
        model=model,
        args=train_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=processor,
    )
    if not args.init_adapter:
        trainer_kwargs["peft_config"] = peft_config
    trainer = SFTTrainer(**trainer_kwargs)
    if trainer.accelerator.scaler is not None:
        # PyTorch's default fp16 loss scale (65536) overflows FunctionGemma's
        # gradients on the GTX 1650 before the scaler can adapt. Start at 1;
        # dynamic scaling remains enabled and all matrix compute stays fp16.
        trainer.accelerator.scaler = torch.amp.GradScaler(
            "cuda", init_scale=1.0, growth_interval=2000
        )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
