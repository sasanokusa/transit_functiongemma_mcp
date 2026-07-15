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
V1_Q6="${V1_Q6:-$GGUF_DIR/v100_Q6_K.gguf}"
V1_TOKENIZER="${V1_TOKENIZER:-$GGUF_DIR/merged_v100}"
RESUME="${RESUME:-0}"
PROVENANCE_ONLY="${PROVENANCE_ONLY:-0}"
RUNNER_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"

case "$RESUME" in
  0|1) ;;
  *) echo "RESUME must be 0 or 1" >&2; exit 2 ;;
esac

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

# An evaluation directory is safe to resume only when every source that can
# change its meaning is byte-identical.  These are full content hashes on
# purpose: model_hashes.sha256 is produced only after conversion and therefore
# cannot prove the identity of adapters/base/v1 inputs at run start.  The extra
# I/O is preferable to mixing results from different releases in one OUT.
PROVENANCE_CANDIDATE="$OUT/.provenance.candidate.$$"
cleanup_provenance_candidate() {
  rm -f "$PROVENANCE_CANDIDATE"
}
trap cleanup_provenance_candidate EXIT

python - "$PROVENANCE_CANDIDATE" \
  "$RUNNER_SOURCE" \
  "evaluation/eval_toolcall.py" \
  "scripts/generate_gguf_predictions.py" \
  "transit_functiongemma" \
  "data/eval/mixed_dev_selection.jsonl" \
  "data/eval/independent_holdout_300.jsonl" \
  "data/eval/manual_practical_100.jsonl" \
  "data/eval/operational_semantic_holdout_300_eval.jsonl" \
  "data/eval/mixed_dev_schema.json" \
  "data/tool_schema.json" \
  "tools/local_tools_schema.json" \
  "$BASE_SNAPSHOT" \
  "$RUN/epoch-1" \
  "$RUN/epoch-2" \
  "$RUN/epoch-3" \
  "$V1_Q6" \
  "$V1_TOKENIZER" \
  "$LLAMA_ROOT/convert_hf_to_gguf.py" \
  "$LLAMA_SERVER" \
  "$LLAMA_QUANTIZE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
labels = (
    "evaluation_runner",
    "evaluator",
    "gguf_prediction_generator",
    "evaluation_runtime_sources",
    "dataset_mixeddev",
    "dataset_independent300",
    "dataset_manual100",
    "dataset_route300",
    "schema_mixeddev_manual100",
    "schema_independent300",
    "schema_route300",
    "base_model",
    "adapter_epoch_1",
    "adapter_epoch_2",
    "adapter_epoch_3",
    "v1_q6_model",
    "v1_tokenizer",
    "hf_to_gguf_converter",
    "llama_server",
    "llama_quantize",
)
paths = [Path(value).expanduser().resolve() for value in sys.argv[2:]]
if len(paths) != len(labels):
    raise SystemExit(f"internal error: expected {len(labels)} provenance inputs, got {len(paths)}")


def file_hash(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def fingerprint(path):
    if not path.exists():
        raise SystemExit(f"provenance input does not exist: {path}")
    if path.is_file():
        sha256, size = file_hash(path)
        return {"kind": "file", "sha256": sha256, "bytes": size}
    if not path.is_dir():
        raise SystemExit(f"unsupported provenance input type: {path}")

    digest = hashlib.sha256()
    digest.update(b"tree-sha256-v1\0")
    file_count = 0
    byte_count = 0
    for child in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if child.is_dir():
            continue
        if not child.is_file():
            raise SystemExit(f"unsupported item in provenance tree: {child}")
        relative = child.relative_to(path).as_posix().encode("utf-8")
        sha256, size = file_hash(child)
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        byte_count += size
    return {
        "kind": "directory",
        "tree_sha256": digest.hexdigest(),
        "files": file_count,
        "bytes": byte_count,
    }


def python_source_tree_fingerprint(path):
    if not path.is_dir():
        raise SystemExit(f"Python source tree does not exist: {path}")

    digest = hashlib.sha256()
    digest.update(b"python-source-tree-sha256-v1\0")
    file_count = 0
    byte_count = 0
    for child in sorted(path.rglob("*.py"), key=lambda value: value.relative_to(path).as_posix()):
        if not child.is_file():
            raise SystemExit(f"unsupported item in Python source tree: {child}")
        relative = child.relative_to(path).as_posix().encode("utf-8")
        sha256, size = file_hash(child)
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        byte_count += size
    if not file_count:
        raise SystemExit(f"Python source tree is empty: {path}")
    return {
        "kind": "python_source_tree",
        "tree_sha256": digest.hexdigest(),
        "files": file_count,
        "bytes": byte_count,
    }


manifest = {
    "format_version": 1,
    "hash_policy": "full-content-sha256",
    "inputs": [
        {
            "name": label,
            "path": os.fspath(path),
            "fingerprint": (
                python_source_tree_fingerprint(path)
                if label == "evaluation_runtime_sources"
                else fingerprint(path)
            ),
        }
        for label, path in zip(labels, paths)
    ],
}
output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
PY

PROVENANCE="$OUT/provenance.json"
if [[ "$RESUME" == 1 ]]; then
  if [[ ! -s "$PROVENANCE" ]]; then
    echo "refusing resume: $PROVENANCE is missing" >&2
    exit 1
  fi
  if ! cmp -s "$PROVENANCE" "$PROVENANCE_CANDIDATE"; then
    python - "$PROVENANCE" "$PROVENANCE_CANDIDATE" <<'PY'
import json
import sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text())
new = json.loads(Path(sys.argv[2]).read_text())
old_inputs = {item["name"]: item for item in old.get("inputs", [])}
new_inputs = {item["name"]: item for item in new.get("inputs", [])}
changed = [name for name in sorted(old_inputs.keys() | new_inputs.keys()) if old_inputs.get(name) != new_inputs.get(name)]
print("refusing resume: provenance mismatch", file=sys.stderr)
for name in changed:
    print(f"  changed: {name}", file=sys.stderr)
if not changed:
    print("  manifest metadata or format changed", file=sys.stderr)
PY
    exit 1
  fi
  rm -f "$PROVENANCE_CANDIDATE"
else
  existing="$(find "$OUT" -mindepth 1 -maxdepth 1 ! -name "$(basename "$PROVENANCE_CANDIDATE")" -print -quit)"
  if [[ -n "$existing" ]]; then
    echo "refusing fresh run: OUT is not empty ($existing)" >&2
    echo "use a new OUT, empty it explicitly, or set RESUME=1 to resume matching inputs" >&2
    exit 1
  fi
  mv "$PROVENANCE_CANDIDATE" "$PROVENANCE"
fi
PROVENANCE_CANDIDATE=""

if [[ "$PROVENANCE_ONLY" == 1 ]]; then
  echo "provenance verified: $PROVENANCE"
  exit 0
fi

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

if [[ "$RESUME" == 0 ]]; then
  rm -f "$F16" "$Q6" "$Q8"
elif [[ -e "$F16" || -e "$Q6" || -e "$Q8" ]]; then
  if [[ ! -s "$OUT/model_hashes.sha256" ]]; then
    echo "refusing derived-model reuse: $OUT/model_hashes.sha256 is missing" >&2
    exit 1
  fi
  if ! sha256sum --check --status "$OUT/model_hashes.sha256"; then
    echo "refusing derived-model reuse: model_hashes.sha256 verification failed" >&2
    exit 1
  fi
fi

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
for spec in "${datasets[@]}"; do
  IFS=: read -r name dataset schema <<<"$spec"
  predictions="$OUT/v100_q6_${name}_predictions.jsonl"
  if [[ ! -s "$predictions" ]]; then
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
