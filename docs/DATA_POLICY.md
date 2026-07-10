# 学習データ方針（r10以降）

小型モデル（270M / LoRA rank4）は矛盾する教師に敏感なため、全学習データは以下の決定表と
受け入れ条件に従う。違反データの追加は禁止（矛盾＝epoch間トレードオフの直接原因、r9で実証）。

## 発話 → 期待挙動の決定表

| 発話パターン | 例 | 正解 |
|---|---|---|
| 断片（意図のみ・指示なし） | 「◯◯に行きたい」「◯◯まで」「乗換少なめで」 | `no_tool_call`（clarification） |
| 複合文（意図＋明示的検索指示） | 「◯◯に行きたい。候補出して」 | `suggest_stations` / `suggest_places` |
| 経路要求（起点＋終点、単発文） | 「AからBまで、Cを避けて安く」 | `resolve_route_request`（slot吸収） |
| 解決済みID＋経路（history内） | suggest_stations解決後 | `plan_journey`、地図cueありなら`plan_route_map` |
| ボード要求（ID＋発車案内） | 「demo-feed:x の終電発車を見たい」 | `station_departures`（行き先を聞き返さない） |
| 「終電いつ？」型（方向依存・終点なし） | 「◯◯駅の終電っていつ？」 | clarification（missing destination） |
| 起点終点＋終電/始発 | 「AからBまで今日の終電で」 | `plan_journey` type=last/first |
| 明示ID＋「詳細/ホーム/路線」 | 「demo-feed:x の詳細を引いて」 | `get_station` |
| 座標 | 「lat=.. lon=..の最寄り」 | `reverse_geocode`（数値は逐語転記） |
| データソース一覧 | 「対応フィードを見せて」 | `list_feeds` |

## 表記・引数規約

- `q`は発話中の表記に忠実（「◯◯駅」と書かれたら`q`も駅を保持）
- `limit`: suggest_stations=5 / suggest_places=10（発話に明示があればそれに従う）
- 相対日付（今日/明日）は`reference_datetime`から演算した`YYYYMMDD`をラベルに含める
- `demo-feed:*`は学習内の架空ID。実MCP IDを学習データに書かない

## データ受け入れ条件（追加バッチはすべて）

1. マスク文型ベースの**テンプレートcap ≤2〜3**（intent系の量産テンプレ再利用235回の轍を踏まない）
2. **ソース間ラベル矛盾検査**をパスすること（同一マスク文型に複数クラスが付く場合、
   決定表で説明できる文脈差があるか、なければ除去）
3. user文字列の全既存データとの重複ゼロ
4. train/eval/選択devのエンティティ・文型分離

## 現行の正準学習セット

`data/raw/r10_curated.jsonl`（8,300行、route比率50%）。来歴:
intent_8000(cap≤2)＋balanced＋real_user＋sonnet5_hard_negatives＋nonroute_replay_r7＋
sonnet5_r9_contrast − 曖昧短文41行。個別ソースへの追加ではなく、このファイルの
再生成（`scratchpad/gen_r9.py`系の圧縮スクリプト）で更新する。
