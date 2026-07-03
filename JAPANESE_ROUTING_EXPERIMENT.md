# Japanese routing experiment (historical)

> This document records the retired semantic-normalizer experiment. Production
> now limits the front normalizer to NFKC/whitespace and makes FunctionGemma emit
> `resolve_route_request`. See `artifacts/PARSER_RULE_AUDIT.md` and the three-stage
> evaluation in `README.md`. Do not copy the semantic-normalizer flags here into
> the current deployment.

FunctionGemmaへ全文英訳を渡さず、日本語の固有名詞を保持したままintentとslotを正規化し、口語表現を追加学習してからtool schema制約を適用する実験です。

## 1. Generate colloquial Japanese data

```bash
python generate_real_user_dataset.py
```

- train: `data/raw/real_user_japanese.jsonl`（86件）
- holdout eval: `data/eval/eval_real_user_japanese.jsonl`（17件）
- train/eval完全重複: 0件

## 2. Normalize Japanese and prepare SFT

```bash
python prepare_sft.py \
  --input data/raw/synthetic_balanced.jsonl \
  --extra-input data/raw/real_user_japanese.jsonl \
  --output data/processed/sft_ja_real.jsonl \
  --schema-mode baked \
  --normalize-ja
```

正規化例:

```text
[normalized_ja]
intent=route
origin=横浜
destination=上野
preferred_lines=京浜東北線
original=横浜から上野まで、京浜東北線で
```

駅名、施設名、路線名、駅IDは翻訳しません。未知の文章は書き換えず、そのままモデルへ渡します。

## 3. Train

```bash
python train_lora_plus.py \
  --dataset data/processed/sft_ja_real.jsonl \
  --output-dir outputs/functiongemma-transit-ja-real-r4 \
  --target-modules all \
  --lora-rank 4 \
  --lora-alpha 8 \
  --max-seq-length 512 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --gradient-accumulation-steps 16
```

## 4. Evaluate with schema constraints

```bash
python eval_toolcall.py \
  --dataset data/eval/eval_real_user_japanese.jsonl \
  --run-model \
  --adapter outputs/functiongemma-transit-ja-real-r4 \
  --schema-mode baked \
  --normalize-ja \
  --bind-normalized-arguments \
  --schema-constraint \
  --output artifacts/eval_ja_real_r4_bound.json \
  --markdown-output artifacts/eval_ja_real_r4_bound.md \
  --failures-output artifacts/eval_ja_real_r4_bound_failures.jsonl
```

`--schema-constraint`はMCPの`tools/list`にないtool名、required欠落、型・enum・追加プロパティ違反を実行前に拒否します。Web実行経路でも同じ検証を行います。

## Baseline

旧`functiongemma-transit-plus-r4`を17件の口語holdoutで評価:

| Input | Parse | Tool name | Exact arguments | No-call | Schema valid |
|---|---:|---:|---:|---:|---:|
| Raw Japanese | 100.00% | 71.43% | 42.86% | 100.00% | 92.86% |
| Normalized, before retraining | 100.00% | 78.57% | 28.57% | 66.67% | 53.33% |

未学習の正規化形式ではintent選択のみ改善し、引数とschema適合が悪化しました。このため正規化と追加学習を常にセットで評価します。

## Final result

GTX 1650 4GBで5 epochの本学習を完了しました。OOMはなく、validation lossは
epoch 1の`0.1476`からepoch 5の`0.01453`へ低下し、validation token accuracyは
`0.9951`でした。学習対象は949,248 parameter（全体の0.353%）だけで、full
fine-tuneとbf16は使用していません。

| Evaluation | Parse | Tool name | Arguments | Required | Datetime | No-call | Schema |
|---|---:|---:|---:|---:|---:|---:|---:|
| 口語holdout（17件） | 100.00% | 100.00% | 100.00% | 100.00% | N/A | 100.00% | 100.00% |
| balanced corrected（72件） | 100.00% | 100.00% | 81.25% | 100.00% | 100.00% | 100.00% | 100.00% |

balancedの`Arguments`は正解引数をすべて含む率です。完全一致率は68.75%で、主な差は
`limit`や`strategy`などschema上有効な既定値の付加です。tool分類、required、日時、
no-call、schema適合は9 classすべて100%でした。中期目標のparse 95%、tool 85%、
expected arguments 80%をすべて上回っています。

評価レポート:

- `artifacts/eval_ja_real_r4_bound.md`
- `artifacts/eval_ja_balanced_corrected_r4.md`

## Practical Web verification

スマホ相当の390×844 viewportで、配備済みWeb UIと実Transit MCPを確認しました。

1. 「あれ、京急の横浜駅ってどれ？」から横浜の路線別候補をボタン表示
2. 候補を選び「出発地にする」を押すと、目的地入力へ状態遷移
3. 「上野」を入力すると駅検索へ戻るループを起こさず、横浜→上野の京浜東北線候補を表示
4. 「最終列車を検索してほしい」はMCP経路callを行わず、出発地・目的地の確認質問を表示

配備先は`outputs/functiongemma-transit-ja-real-r4`で、Web serviceは
`FUNCTIONGEMMA_NORMALIZE_JA=1`とschema検証を有効にしています。日本語正規化は
翻訳器ではなく決定的なintent/slot抽出です。モデルの出力はallowlistとMCP由来JSON
Schemaを通過したときだけMCPへ送信されます。

この評価規模では実用試験へ進める結果です。ただし17件の口語holdoutは小さいため、
実ログ由来の匿名化データを継続的にholdoutへ追加し、駅名・事業者名・地方・日時表現の
分布ずれを監視します。
