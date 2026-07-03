#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from transit_functiongemma.config import MODEL_ID

ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-conscious extended LoRA/QLoRA experiment for GTX 1650."
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed/sft_balanced.jsonl")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/functiongemma-transit-lora-plus"),
    )
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--max-seq-length", type=int, choices=(256, 512), default=512)
    parser.add_argument("--lora-rank", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument(
        "--target-modules",
        choices=("attention", "all"),
        default="attention",
        help="all adds gate_proj/up_proj/down_proj to the attention projections.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume optimizer, scheduler, RNG, and adapter state from a Trainer checkpoint.",
    )
    return parser.parse_args()


def tokenize_rows(
    rows: list[dict], processor: object, max_length: int
) -> tuple[Dataset, int]:
    input_rows: list[list[int]] = []
    masks: list[list[int]] = []
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
        if len(token_ids) > max_length:
            dropped += 1
            continue
        input_rows.append(token_ids)
        masks.append([0] * len(prompt_ids) + [1] * (len(token_ids) - len(prompt_ids)))
    if not input_rows:
        raise ValueError("All records exceed max sequence length")
    return Dataset.from_dict({"input_ids": input_rows, "completion_mask": masks}), dropped


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
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
        dtype=torch.float16 if args.qlora else torch.float32,
        quantization_config=quantization_config,
        device_map={"": 0},
        attn_implementation="eager",
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset, dropped = tokenize_rows(rows, processor, args.max_seq_length)
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)
    targets = ATTENTION_MODULES + (MLP_MODULES if args.target_modules == "all" else [])
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    config = SFTConfig(
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
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=processor,
        peft_config=peft_config,
    )
    if trainer.accelerator.scaler is not None:
        trainer.accelerator.scaler = torch.amp.GradScaler(
            "cuda", init_scale=1.0, growth_interval=2000
        )
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    print(
        f"records={len(dataset)} dropped={dropped} targets={targets} "
        f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.3f}%)"
    )
    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
