# Next experiment: balanced tool routing

この手順では、9 classを均等化し、train/evalの入力重複を検査してからLoRAを学習します。モデルはtool router専用です。経路、料金、所要時間、路線名を生成させず、必ずMCP結果を利用してください。

## 0. Server environment

```bash
ssh <server>
cd /path/to/transit_functiongemma_mcp
source /path/to/venvs/transit-functiongemma/bin/activate
export HF_HOME=/path/to/huggingface
export XDG_CACHE_HOME=/path/to/cache
export PIP_CACHE_DIR=/path/to/pip-cache
```

## 1. Generate balanced data

既定ではtrainを各class 64件（合計576件）、evalを各class 8件（合計72件）生成します。

```bash
python datagen/generate_balanced_synthetic_dataset.py \
  --per-class 64 \
  --eval-per-class 8
```

対象classは `suggest_stations`、`suggest_places`、`reverse_geocode`、`station_departures`、`get_station`、`list_feeds`、`plan_journey`、`plan_route_map`、`no_tool_call` です。

## 2. Analyze distribution and overlap

```bash
python evaluation/analyze_dataset.py \
  --train data/raw/synthetic_balanced.jsonl \
  --eval data/eval/eval_balanced.jsonl
```

結果は `artifacts/dataset_analysis.json` と `artifacts/dataset_analysis.md` に保存されます。class数、no-call数、argumentsの有無、入力重複、train/eval重複、token長を確認します。

## 3. Convert to FunctionGemma SFT

空assistant turnをno-call教師として使う互換モード:

```bash
python training/prepare_sft.py \
  --input data/raw/synthetic_balanced.jsonl \
  --output data/processed/sft_balanced.jsonl \
  --schema-mode baked
```

ローカル疑似toolを使う比較実験:

```bash
python training/prepare_sft.py \
  --input data/raw/synthetic_balanced.jsonl \
  --output data/processed/sft_balanced_clarification.jsonl \
  --schema-mode baked \
  --clarification-tool
```

`ask_clarification` はMCP toolではありません。`scripts/mcp_client.py` がローカルで質問文へ変換し、MCPへは送信しません。

## 4. Baseline LoRA

```bash
python training/train_lora.py \
  --dataset data/processed/sft_balanced.jsonl \
  --output-dir outputs/functiongemma-transit-balanced-r8 \
  --max-seq-length 512 \
  --lora-rank 8 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --gradient-accumulation-steps 16
```

固定条件はbatch size 1、gradient checkpointing有効、fp16、bf16無効です。

## 5. Evaluate

```bash
python evaluation/eval_toolcall.py \
  --dataset data/eval/eval_balanced.jsonl \
  --run-model \
  --adapter outputs/functiongemma-transit-balanced-r8 \
  --schema-mode baked
```

出力:

- `artifacts/eval_report.json`
- `artifacts/eval_report.md`
- `artifacts/eval_failures.jsonl`

clarification版adapterを評価する場合は `--clarification-tool` を追加します。既存のno-call率は、空出力と有効な `ask_clarification` の両方を安全なno-callとして数えます。

## 6. Plus LoRA experiment

attentionに加えてMLP projectionもLoRA対象にします。4GB VRAM向けにrank 4が既定です。

```bash
python training/train_lora_plus.py \
  --dataset data/processed/sft_balanced.jsonl \
  --output-dir outputs/functiongemma-transit-plus-r4 \
  --target-modules all \
  --lora-rank 4 \
  --max-seq-length 512 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --gradient-accumulation-steps 16
```

`all` は `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` を対象にします。比較用の `attention` は最初の4 projectionだけです。

## 7. OOM fallback

まずsequence lengthとrankを縮小します。

```bash
python training/train_lora_plus.py \
  --dataset data/processed/sft_balanced.jsonl \
  --output-dir outputs/functiongemma-transit-plus-oom-safe \
  --target-modules all \
  --max-seq-length 256 \
  --lora-rank 2 \
  --gradient-accumulation-steps 16 \
  --qlora
```

それでもOOMする場合は `--target-modules attention` に戻します。batch sizeは常に1のままにします。

## Success criteria

短期: parse 90%以上、tool名70%以上、no-call 66%以上。中期: parse 95%以上、tool名85%以上、expected arguments 80%以上。

