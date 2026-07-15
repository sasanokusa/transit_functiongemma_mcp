# Release manifest v3: Tentetsu-270M v1.0.0 (frozen 2026-07-11)

**Tentetsu（転轍）-270M** — 日本語乗換案内向けMCP tool router。based on
`google/functiongemma-270m-it`（Gemma Terms of Use配下の派生モデル）。
**重みは非公開**（内部利用。配布する場合はGemma ToU §3.1のNotice・使用制限同梱が必要。
規約に派生モデルの命名制限はなし・§4.2によりGoogle推奨を示唆しないこと）。

## 系譜（改名表）

| 新名 | 旧名 | 位置付け |
|---|---|---|
| beta4〜beta7 | r4〜r7 | 旧アーキテクチャ〜intent router試験世代 |
| rc8 | r8c | replay配合の確立・監査済み評価の初適用 |
| rc9 | r9 | 対照ペア導入（反clarify/座標/map-cue） |
| rc10 | r10/r10b | **データ整理**（矛盾除去・テンプレcap）で綱引き解消 |
| **v1.0.0** | **r10b epoch-2** | 本manifest |

## Model

| Item | Value |
|---|---|
| Canonical path | `outputs/tentetsu-270m-v1.0.0`（= `functiongemma-transit-r10b/epoch-2` のコピー） |
| adapter_model.safetensors SHA-256 | `be4c41575667ce03a75adcebfe19c7e471c2b82e5381f336b52c9a8edda26164` |
| Base | `google/functiongemma-270m-it` snapshot `39eccb091651513a5dfb56892d3714c1b5b8276c` |
| 学習データ `r10b_curated.jsonl` SHA-256 | `119c2ad259bf97e743a339a8770c01863ae08c1edf7b4761765d00820c677e2f`（8,739行、`docs/DATA_POLICY.md`準拠） |
| 学習 | LoRA rank4/alpha8 全projection、512tok、3 epochs、外部4セット選択でepoch-2採用 |
| GGUF Q6_K SHA-256 | `89eeb9d467995a32e9935b26f8543fd7c758bce32a0f5b03391873e05d4aaabb` |
| iPhone bundle | `ios/Tentetsu/Tentetsu/Resources/models/tentetsu-q6_k.gguf`（gitignore対象、署名bundle内でhash検証） |

## 実測（fp32、2026-07-11バッテリー）

| セット | semantic | tool | parse | no-call |
|---|---:|---:|---:|---:|
| route300 | **98.7%** | 99.3% | 99.3% | — |
| independent300 | 75.3% | 92.2% | 95.6% | 86.7% |
| manual100（現行ラベル） | 73.0% | **97.7%** | 98.8% | 93.3% |
| mixed_dev | 90.1% | 93.9% | **100%** | **100%** |

残差74/300の内訳: tool正解・引数ミス48、parse系12、真のtool混同10 —
「tool選択」は解決済み、残る前線は**multi-turn経路callの引数忠実度**（strategy/type/time）。

## 2026-07-15 有線デモ採用判断

同一の現行final-bound評価器でv1.0.0 Q6_Kとrc11 fp32/Q6/Q8を比較した結果、rc11は
manualセットを改善する一方、independentセットを悪化させ、量子化候補として一貫した
昇格根拠になりませんでした。既に実機スモークを通過し、量子化候補中でindependentの
semanticが最も高い本Q6_Kを、入力範囲を明示した有線デモ用として維持します。

これは不特定多数向けの公開Go判定ではありません。比較値、rc11 hash、評価器の既知の
binder影響は`docs/RC11_RELEASE_EVALUATION.md`に分離して記録します。公開前にはGGUF
backendの広範なE2E、salt/WAF/監視/保持、App Store用アセットと配布規約を別途満たします。
