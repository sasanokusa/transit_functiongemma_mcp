#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/transit_work/eval_deterministic_v11}"
VENV="${VENV:-$HOME/transit_work/venv}"
HF_HOME="${HF_HOME:-$HOME/transit_work/hf_cache}"
LLAMA_ROOT="${LLAMA_ROOT:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_ROOT/build-server-noui/bin/llama-server}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-$LLAMA_ROOT/build/bin/llama-quantize}"
GGUF_DIR="${GGUF_DIR:-$HOME/transit_work/gguf}"
RUN="${RUN:-$ROOT/outputs/functiongemma-transit-rc11}"
OUT="${OUT:-$ROOT/artifacts/rc11_release_eval}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-$HF_HOME/hub/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c}"

source "$VENV/bin/activate"
export HF_HOME HF_HUB_OFFLINE=1
export PATH="$(dirname "$LLAMA_SERVER"):$(dirname "$LLAMA_QUANTIZE"):$PATH"
cd "$ROOT"
mkdir -p "$OUT"

datasets=(
  "mixeddev:data/eval/mixed_dev_selection.jsonl:data/eval/mixed_dev_schema.json"
  "independent300:data/eval/independent_holdout_300.jsonl:data/tool_schema.json"
  "manual100:data/eval/manual_practical_100.jsonl:data/eval/mixed_dev_schema.json"
  "route300:data/eval/operational_semantic_holdout_300_eval.jsonl:tools/local_tools_schema.json"
)

evaluate_predictions() {
  local name="$1" dataset="$2" schema="$3" predictions="$4" prefix="$5"
  python evaluation/eval_toolcall.py \
    --dataset "$dataset" --predictions "$predictions" --schema "$schema" \
    --clarification-tool --normalize-ja --schema-constraint --bind-normalized-arguments \
    --output "$OUT/${prefix}_${name}.json" \
    --markdown-output "$OUT/${prefix}_${name}.md" \
    --failures-output "$OUT/${prefix}_${name}_failures.jsonl"
}

echo "[1/6] fp32 epoch battery $(date --iso-8601=seconds)"
for epoch in epoch-1 epoch-2 epoch-3; do
  for spec in "${datasets[@]}"; do
    IFS=: read -r name dataset schema <<<"$spec"
    if [[ -s "$OUT/fp32_${epoch}_${name}.json" ]]; then
      echo "skip existing fp32 $epoch $name"
      continue
    fi
    echo "fp32 $epoch $name $(date --iso-8601=seconds)"
    python evaluation/eval_toolcall.py \
      --dataset "$dataset" --run-model --adapter "$RUN/$epoch" --schema "$schema" \
      --clarification-tool --normalize-ja --schema-constraint --bind-normalized-arguments \
      --output "$OUT/fp32_${epoch}_${name}.json" \
      --markdown-output "$OUT/fp32_${epoch}_${name}.md" \
      --failures-output "$OUT/fp32_${epoch}_${name}_failures.jsonl"
  done
done

python - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
datasets = ("mixeddev", "independent300", "manual100", "route300")
rows = []
for epoch in ("epoch-1", "epoch-2", "epoch-3"):
    reports = {name: json.loads((out / f"fp32_{epoch}_{name}.json").read_text()) for name in datasets}
    semantics = [reports[name]["metrics"]["semantic_success_rate"] for name in datasets]
    args = [reports[name]["metrics"]["expected_arguments_match_rate"] for name in datasets]
    tools = [reports[name]["metrics"]["tool_name_accuracy"] for name in datasets]
    rows.append({
        "epoch": epoch,
        "mean_semantic": statistics.fmean(semantics),
        "min_semantic": min(semantics),
        "mean_expected_arguments": statistics.fmean(args),
        "mean_tool": statistics.fmean(tools),
        "metrics": {name: reports[name]["metrics"] for name in datasets},
    })
best = max(rows, key=lambda row: (row["mean_semantic"], row["min_semantic"], row["mean_expected_arguments"], row["mean_tool"]))
(out / "best_epoch.txt").write_text(best["epoch"] + "\n")
(out / "fp32_selection.json").write_text(json.dumps({"policy": "max mean semantic; tie-break min semantic, mean args, mean tool", "best_epoch": best["epoch"], "epochs": rows}, ensure_ascii=False, indent=2))
print("best", best["epoch"], best["mean_semantic"])
PY

BEST="$(tr -d '\n' < "$OUT/best_epoch.txt")"
MERGED="$GGUF_DIR/merged_rc11_${BEST}"
F16="$GGUF_DIR/rc11_${BEST}_f16.gguf"
Q6="$GGUF_DIR/rc11_${BEST}_Q6_K.gguf"
Q8="$GGUF_DIR/rc11_${BEST}_Q8_0.gguf"

echo "[2/6] merge $BEST $(date --iso-8601=seconds)"
rm -rf "$MERGED"
python - "$BASE_SNAPSHOT" "$RUN/$BEST" "$MERGED" <<'PY'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path, adapter_path, output_path = sys.argv[1:]
tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)
PY

python - "$MERGED" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
bad_tokens = {"<image_soft_token>", "<end_of_image>"}
def bad(value):
    return (isinstance(value, str) and value in bad_tokens) or (isinstance(value, dict) and value.get("content") in bad_tokens)
tokenizer = json.loads((root / "tokenizer.json").read_text())
tokenizer["added_tokens"] = [item for item in tokenizer["added_tokens"] if item["id"] < 262144]
(root / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False))
for filename in ("added_tokens.json", "special_tokens_map.json"):
    path = root / filename
    if path.exists():
        data = json.loads(path.read_text())
        data = {key: value for key, value in data.items() if not bad(value) and not (isinstance(value, int) and value >= 262144)}
        for key, value in list(data.items()):
            if isinstance(value, list):
                data[key] = [item for item in value if not bad(item)]
        path.write_text(json.dumps(data, ensure_ascii=False))
config_path = root / "tokenizer_config.json"
config = json.loads(config_path.read_text())
for key in ("image_token", "eoi_token", "model_specific_special_tokens"):
    config.pop(key, None)
decoder = config.get("added_tokens_decoder", {})
for key in list(decoder):
    if int(key) >= 262144:
        del decoder[key]
extra = config.get("extra_special_tokens")
if isinstance(extra, dict):
    config["extra_special_tokens"] = {key: value for key, value in extra.items() if not bad(value)}
config_path.write_text(json.dumps(config, ensure_ascii=False))
PY

if [[ ! -s "$F16" ]]; then
  python "$LLAMA_ROOT/convert_hf_to_gguf.py" "$MERGED" --outfile "$F16" --outtype f16
fi
if [[ ! -s "$Q6" ]]; then
  "$LLAMA_QUANTIZE" "$F16" "$Q6" Q6_K
fi
if [[ ! -s "$Q8" ]]; then
  "$LLAMA_QUANTIZE" "$F16" "$Q8" Q8_0
fi
sha256sum "$RUN/$BEST/adapter_model.safetensors" "$F16" "$Q6" "$Q8" > "$OUT/model_hashes.sha256"

echo "[3/6] rc11 GGUF battery $(date --iso-8601=seconds)"
for quant in q6 q8; do
  if [[ "$quant" == q6 ]]; then model="$Q6"; else model="$Q8"; fi
  for spec in "${datasets[@]}"; do
    IFS=: read -r name dataset schema <<<"$spec"
    predictions="$OUT/rc11_${quant}_${name}_predictions.jsonl"
    echo "rc11 $quant $name $(date --iso-8601=seconds)"
    if [[ ! -s "$predictions" ]]; then
      python scripts/generate_gguf_predictions.py \
        --dataset "$dataset" --gguf "$model" --tokenizer "$MERGED" \
        --llama-server "$LLAMA_SERVER" --output "$predictions"
    fi
    evaluate_predictions "$name" "$dataset" "$schema" "$predictions" "rc11_${quant}"
  done
done

echo "[4/6] v1.0.0 Q6_K on current evaluator $(date --iso-8601=seconds)"
V1_Q6="$GGUF_DIR/v100_Q6_K.gguf"
V1_TOKENIZER="$GGUF_DIR/merged_v100"
for spec in "${datasets[@]}"; do
  IFS=: read -r name dataset schema <<<"$spec"
  predictions="$OUT/v100_q6_${name}_predictions.jsonl"
  existing="$ROOT/../repo/artifacts/preds_${name}_v100q6k.jsonl"
  if [[ -s "$existing" ]]; then
    cp "$existing" "$predictions"
  else
    python scripts/generate_gguf_predictions.py \
      --dataset "$dataset" --gguf "$V1_Q6" --tokenizer "$V1_TOKENIZER" \
      --llama-server "$LLAMA_SERVER" --output "$predictions"
  fi
  evaluate_predictions "$name" "$dataset" "$schema" "$predictions" "v100_q6"
done

echo "[5/6] comparison summary $(date --iso-8601=seconds)"
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
datasets = ("mixeddev", "independent300", "manual100", "route300")
best = (out / "best_epoch.txt").read_text().strip()
variants = {"rc11_fp32": f"fp32_{best}", "rc11_q6": "rc11_q6", "rc11_q8": "rc11_q8", "v100_q6": "v100_q6"}
summary = {"best_epoch": best, "variants": {}}
for label, prefix in variants.items():
    summary["variants"][label] = {}
    for name in datasets:
        path = out / f"{prefix}_{name}.json"
        summary["variants"][label][name] = json.loads(path.read_text())["metrics"]
(out / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[6/6] complete $(date --iso-8601=seconds)"
