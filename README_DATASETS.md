# Transit FunctionGemma 評価強化データセット

このzipは、`transit_functiongemma_mcp` の評価を 1〜2段階上げるための追加holdoutです。

## ファイル

- `data/eval/independent_holdout_300.jsonl`
  - 300件。既存の9 class評価を拡大した独立holdout。
  - `expected_tool`, `expected_arguments`, `expected_normalized`, `missing_info`, `history`, `tags` を含みます。
- `data/eval/manual_practical_100.jsonl`
  - 100件。実用寄りの手書き評価。
  - 「渋谷を避けて」「経由指定」「路線回避」「opaque ID」「座標表現」「日時」「終電/始発」などを含みます。
  - `expected_intent` は将来の `resolve_route_request` / constraints評価用です。
- `PROMPT_RAISE_ALL_EVALUATIONS.md`
  - Codexに渡す改善プロンプト。
- `validate_eval_datasets.py`
  - 件数、ID重複、class分布を確認する軽量スクリプト。

## 注意

`plan_journey` / `plan_route_map` の行は、駅解決済み状態を表すため `history` を含みます。
既存の `eval_toolcall.py` が `history` を未対応の場合は、先に評価器を `history` 対応にしてください。

生成日時の基準は `2026-06-29 10:00 Asia/Tokyo` です。
