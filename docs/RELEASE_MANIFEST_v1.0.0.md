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
| GGUF Q6_K | 変換中（完了後にSHA-256追記） |

## 実測（fp32、2026-07-11バッテリー）

| セット | semantic | tool | parse | no-call |
|---|---:|---:|---:|---:|
| route300 | **98.7%** | 99.3% | 99.3% | — |
| independent300 | 75.3% | 92.2% | 95.6% | 86.7% |
| manual100（現行ラベル） | 73.0% | **97.7%** | 98.8% | 93.3% |
| mixed_dev | 90.1% | 93.9% | **100%** | **100%** |

残差74/300の内訳: tool正解・引数ミス48、parse系12、真のtool混同10 —
「tool選択」は解決済み、残る前線は**multi-turn経路callの引数忠実度**（strategy/type/time）。

## v1.0.0のGo条件（残）

1. Q6_K量子化ゲート: fp32比 主要指標 ≤1.0pp低下
2. multi-turn引数対照データ（rc11候補）はv1.0.x系の改善として並走
3. GGUF backend E2E・Pi配備・運用4項目（salt/WAF/監視/保持）は v1.0.0公開判定の前提
