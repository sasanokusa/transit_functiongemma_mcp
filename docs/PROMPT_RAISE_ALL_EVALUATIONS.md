# Codexプロンプト: Transit FunctionGemmaの評価を全体的に2段階上げる

現在の `transit_functiongemma_mcp` を、以下の評価目標に向けて改善してください。

## 現状

- FunctionGemma 270M LoRAをTransit MCP専用tool routerとして使用している
- 経路、料金、時刻、路線名はLLMに生成させず、Transit MCPの応答と決定的なPython処理で表示する
- 最終adapterは小規模balanced評価と口語holdoutでは高精度
- ただし、評価規模・E2E耐性・実用条件が不足しており、一般公開品質ではない

## 目標

評価を以下のように1〜2段階引き上げる。

- デモ: A- → A / A+
- 実用試験: B+ → A- / A
- 一般公開サービス: C+〜B- → B / B+

## 追加データセット

このzipに含まれる以下をプロジェクトへ取り込んでください。

- `data/eval/independent_holdout_300.jsonl`
- `data/eval/manual_practical_100.jsonl`

## 必須対応 1: 評価器の強化

`evaluation/eval_toolcall.py` または新規 `evaluation/eval_pipeline.py` を改善し、以下に対応してください。

1. `history` 付き評価
   - `plan_journey` / `plan_route_map` は駅解決済みの会話履歴を持つ
   - `history` がある場合、routerへ履歴込みで渡す
   - 既存の `user` 単独評価も壊さない

2. tool-call評価
   - parse成功率
   - tool名正解率
   - expected_arguments包含一致率
   - expected_arguments完全一致率
   - required arguments充足率
   - datetime normalization成功率
   - no-call成功率
   - schema適合率
   - tool別precision/recall/F1
   - confusion matrix

3. intent評価
   - `expected_intent` がある行では、以下のslot一致率を測る
     - origin_text
     - destination_text
     - avoid_station_texts
     - via_station_texts
     - avoid_line_texts
     - priority
     - time_mode
     - date
     - time
     - graphical

4. E2E評価
   - 実MCPを叩く評価を別モードで実装する
   - timeout率、retry成功率、MCP status、p50/p95レイテンシを記録
   - raw result、normalized result、rendered answerを保存

5. レポート
   - JSON: `artifacts/eval_*.json`
   - Markdown: `artifacts/eval_*.md`
   - 失敗例: `artifacts/eval_*_failures.jsonl`

## 必須対応 2: `expected_arguments`の意味的改善

以下を重点的に直してください。

1. 検索語クリーニング
   - `東京タワーを場所として検索して` → `q: 東京タワー`
   - `駅として拾って` や `候補を出して` を検索語に残さない
   - ただし施設名・駅名の一部は削らない

2. `strategy`保持
   - 早いやつ → `fastest`
   - 安いやつ → `lowestFare`
   - 乗換少なめ → `fewestTransfers`
   - 歩き少なめ → `shortestWalk`
   - バランスよく → `balanced`

3. `radiusMeters`保持
   - 半径50m/100m/500mなどが入力にある場合は保持
   - 指定がなければ既定値でよいが、評価では包含一致と完全一致を分ける

4. 日時正規化
   - 今日、明日、7/1、2026年7月1日
   - 朝8時半、9時着、18:00まで
   - 始発、終電
   - `reference_datetime` を必ず使う

## 必須対応 3: constraints対応

次の自然文制約を扱えるようにしてください。

- 渋谷を避けて
- 新宿を通らないで
- 池袋経由は嫌
- 新宿経由で
- 秋葉原に寄って
- 山手線なしで
- JR使わないで
- 地下鉄だけで
- 歩きたくない
- 乗換少なめ

実装方針:

1. FunctionGemmaはtool routerとして維持
2. 追加でローカル疑似tool `resolve_route_request` を導入可能にする
3. `resolve_route_request` はMCPへ送らない
4. local plannerが以下を行う
   - 駅名解決
   - 場所解決
   - avoid/via駅解決
   - MCP plan_journey / plan_route_map呼び出し
   - route候補filter
   - 日本語テンプレート表示

## 必須対応 4: deterministic renderer

後段LLMは使わないでください。

- MCP raw result → route_normalizer
- route_normalizer → route_constraints
- route_constraints → route_renderer
- route_rendererが日本語を返す

禁止:

- LLMに料金、時刻、路線名を生成させる
- LLMに回避条件の成否を判断させる
- JSONにない情報を補う

## 必須対応 5: デモA+化

Web UIまたはCLIデモで、以下の固定シナリオを安定して通してください。

1. 東京駅を検索して
2. 東京タワーを場所として探して
3. 町田から池袋まで、渋谷を避けて
4. 横浜から上野まで、乗換少なめ
5. 明日9時に品川に着きたい
6. 終電で新宿から大宮まで帰れる？
7. 東京駅まで行きたい → 出発地確認

UIでは可能なら以下を表示してください。

- user input
- FunctionGemma output
- parsed tool call
- schema validation
- MCP result
- normalized route JSON
- rendered Japanese answer
- latency

## 必須対応 6: 一般公開B化の最低ライン

以下を実装または設計してください。

- rate limit
- MCP timeout
- retry with exponential backoff
- short-term cache
- schema hash / schema変更検知
- 匿名化ログ
- request_id単位のtrace
- user query保存ON/OFF
- raw位置情報を保存しない設定
- unknown tool拒否
- malformed output拒否

## 評価目標

### independent_holdout_300

- parse成功率: 98%以上
- tool名正解率: 95%以上
- expected_arguments包含一致率: 90%以上
- required arguments充足率: 95%以上
- no-call正解率: 95%以上
- schema適合率: 100%

### manual_practical_100

- class/tool正解率: 90%以上
- no-call正解率: 95%以上
- intent slot macro F1: 85%以上
- avoid/via抽出成功率: 85%以上
- 日時正規化成功率: 90%以上

### E2E

- E2E成功率: 90%以上
- p95レイテンシ: 5秒以内を目標
- MCP timeout時にユーザー向けエラーへ落とす
- no-call時にMCPを呼ばない
- route rendererがJSONにない事実を生成しない

## 実装後に出してほしい成果物

- `artifacts/eval_independent_holdout_300.md`
- `artifacts/eval_manual_practical_100.md`
- `artifacts/e2e_eval_report.md`
- `artifacts/failures_*.jsonl`
- `artifacts/latency_report.json`
- `NEXT_RELEASE_CHECKLIST.md`
- README追記

## 最終判定基準

以下を満たしたら、評価を引き上げてよい。

- デモ: A- → A/A+
  - 固定7シナリオ成功
  - Web/CLIで内部状態が見える
  - 渋谷回避が実演できる

- 実用試験: B+ → A-/A
  - 300件holdoutと100件手書き評価を通過
  - E2E評価が自動化済み
  - 日時、終電、始発、回避、経由を網羅

- 一般公開サービス: C+〜B- → B/B+
  - retry/cache/rate limit/schema変更検知/匿名ログあり
  - timeoutや空結果でも破綻しない
  - MCP送信前の安全検証が常時有効
