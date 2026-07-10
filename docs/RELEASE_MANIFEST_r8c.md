# Release manifest: r8c candidate (v2, frozen 2026-07-10)

このmanifestはリリース候補を不変に固定する。`best` symlinkではなく実epochとSHA-256で参照すること。

> **v2改訂**: 候補をepoch-1→**epoch-3**へ変更。route専用dev(dev950)のみの選択が公式300+100に
> 盲目で、epoch-3が300でsemantic +6pp / tool +15.5pp、route300は99.0%で回帰なしと判明したため。
> 以後のcheckpoint選択はroute+nonroute混合devを必須とする。
> v1のGGUF/Q6_Kハッシュはepoch-1由来のため、epoch-3の再変換後に更新すること。

## Model

| Item | Value |
|---|---|
| Adapter | `functiongemma-transit-r8c/epoch-3`（公式300+100とroute300ガードで選択） |
| adapter_model.safetensors SHA-256 | `babec9a45518c4ba52855f38c94e17a42ed00d4799add511d367a8aaec5bf927` |
| (v1: epoch-1 adapter SHA-256) | `87825e094ceb31df840921be3f2279d07337bd04831fb909305a6d2a009d771f` |
| Base model | `google/functiongemma-270m-it` snapshot `39eccb091651513a5dfb56892d3714c1b5b8276c` |
| GGUF f16 SHA-256 | `a85a16312b7a47f6c5fd717d2bd771f07fa006fb96136e93a26920b15f5fe255` |
| GGUF Q6_K SHA-256 | `4d356bb60ef43314b8cc9778913755cf4a381080824bb85e3b095792d517af90` |
| 正規化済みtokenizer.json SHA-256 | `77273a7d8fc85f6457d218e239d559e2d3f9b42faa6ed7fbb9f1e6103533f0e4`（embedding外visionトークン2個を除去済み） |

## Code / Data

| Item | Value |
|---|---|
| Git commit | `616a236`（master、A+Cボード要求ラベル復元後） |
| 学習データ `sft_r8c_replay.jsonl` | `2f024588b29e2dc3fa89ea8dae36c9e8c7225127b0fdfe6c10e60b675624dab5`（11,013件） |
| `independent_holdout_300.jsonl` | `879971bb22599e1ed1cb4f2aace33729f3482c673acc2ca5cac39f5ff5c3eeb0` |
| `manual_practical_100.jsonl` | `2b3b1adbfab22535e1144603b6174e7cd6d212b7b84bf182f2f7358ad8cd159a` |
| MCP schema `data/tool_schema.json` | `49d1c38ab8286b35342667ed1bdb5b86ce319c4df95358553fe53ccef080ec86` |
| `tools/local_tools_schema.json` | `481af66fe088d7f66d30ba5929e76a97463de828f65f12e5a1a6656ecc58b791` |

## Toolchain

| Item | Value |
|---|---|
| llama.cpp (PGX, aarch64 CUDA GB10) | version 9925 (`ed8c26150`) |
| llama.cpp (Mac, brew) | build 9860 (`fdb1db877`) |
| PyTorch (PGX) | 2.13.0+cu130 / transformers 4.57.6 / peft 0.19.1 / trl 0.24.0 |

## Training command (reproducible)

```bash
python training/prepare_sft.py \
  --input data/raw/intent_router_train_8000.jsonl \
  --extra-input data/raw/synthetic_balanced.jsonl \
  --extra-input data/raw/real_user_japanese.jsonl \
  --extra-input data/raw/sonnet5_hard_negatives.jsonl \
  --extra-input data/raw/nonroute_replay_r7.jsonl \
  --output data/processed/sft_r8c_replay.jsonl \
  --schema-mode baked --clarification-tool

python training/train_lora_plus.py \
  --dataset data/processed/sft_r8c_replay.jsonl \
  --output-dir outputs/functiongemma-transit-r8c \
  --target-modules all --lora-rank 4 --lora-alpha 8 \
  --max-seq-length 512 --learning-rate 2e-4 \
  --epochs 3 --gradient-accumulation-steps 16

python scripts/select_checkpoint.py \
  --run outputs/functiongemma-transit-r8c \
  --dev data/eval/intent_router_dev_950.jsonl \
  --dev-sample 200 --seed 42 \
  --schema tools/local_tools_schema.json
```

## 既知の測定値（このmanifest時点）

- route intent (operational_semantic_holdout_300_eval): semantic 99.0% / tool 100%
- nonroute 215 (independent_300の非経路部分): semantic 80.0% / parse 93.5% / no-call 90%
- Q6_K vs fp32 (575行): 合計 −0.7pp、route完全一致
- 公式ゲート（independent_300全件 / manual_100 / E2E）は**未再検証** — 診断実行中

## Gate criteria for this candidate

1. fp32が300+100の全絶対基準を満たすこと（現時点で未達見込み — 診断で失敗分布を確定する）
2. Q6_Kも同一基準を満たし、かつfp32からの主要指標低下 ≤1.0pp
3. GGUF版E2E 7/7（GGUF backend実装後）、no-call時MCP通信ゼロ
4. Pi実機p95がSLO内、r7へのロールバック試験成功
5. salt / edge rate limit / 外形監視 / ログ保持方針が本番設定で検証済み
