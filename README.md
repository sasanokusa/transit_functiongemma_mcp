# transit_functiongemma_mcp

`google/functiongemma-270m-it` を乗換案内MCP専用のtool routerとしてLoRA/QLoRA SFTする最小実験環境です。モデルは経路や料金を回答せず、許可されたMCP tool callだけを生成します。経路・料金・所要時間・路線情報は常に `https://api.transit.ls8h.com/mcp` が返す値を使用します。

## 構成

- `scripts/fetch_tools.py`: MCP `tools/list` を取得し、MCP形式とFunctionGemma形式で保存
- `training/prepare_sft.py`: synthetic JSONLをFunctionGemmaの `messages` / `tool_calls` 形式へ変換
- `training/train_lora.py`: fp16、batch size 1、gradient accumulation付きLoRA/QLoRA学習
- `transit_functiongemma/infer.py`: 決定的デコードでfunction callを生成
- `scripts/mcp_client.py`: callを厳格parse・schema検証後、MCP `tools/call` を実行
- `evaluation/eval_toolcall.py`: 指定された5指標を集計
- `data/raw/synthetic_template.jsonl`: 学習データの小さな雛形（本学習前に拡張必須）
- `datagen/generate_synthetic_dataset.py`: schema適合済みの拡張train/eval JSONLを決定的に生成
- `datagen/generate_balanced_synthetic_dataset.py`: 9 classを均等化したhard-negative付きデータを生成
- `evaluation/analyze_dataset.py`: 分布、重複、arguments、token長をJSON/Markdownで分析
- `training/train_lora_plus.py`: attention＋MLP projectionを選択できる4GB向け追加実験
- `transit_functiongemma/route_normalizer.py`: MCP raw resultを事実のみの共通JSONへ正規化
- `transit_functiongemma/route_constraints.py`: 経路内の駅を抽出し、avoid/via条件を決定的に検査
- `transit_functiongemma/route_renderer.py`: 正規化済みJSONを追加LLMなしで日本語化
- `transit_functiongemma/answer_pipeline.py`: router、MCP、正規化、制約検査、rendererを接続
- `transit_functiongemma/local_tools.py`: MCPへ送らないローカル疑似tool
- `transit_functiongemma/station_resolver.py`: 路線別IDを物理駅の`geo:`地点へ安全に解決

## 1. サーバーへ配置

モデル、venv、checkpoint、キャッシュは学習サーバー側の作業ディスクに置きます。`<server>` と `/path/to` は環境に合わせて読み替えてください。

```bash
rsync -av --exclude .venv --exclude outputs ./ <server>:/path/to/transit_functiongemma_mcp/
ssh <server>
cd /path/to/transit_functiongemma_mcp
bash scripts/setup_server.sh
source /path/to/venvs/transit-functiongemma/bin/activate
export HF_HOME=/path/to/huggingface
export XDG_CACHE_HOME=/path/to/cache
export PIP_CACHE_DIR=/path/to/pip-cache
```

Gemmaの利用規約をHugging Faceで承認し、`hf auth login` でアクセストークンを設定してください。PyTorch wheelはサーバーのNVIDIA driver/CUDAに合うものを先に入れるのが安全です。

## 2. schema取得とデータ変換

```bash
python scripts/fetch_tools.py
python training/prepare_sft.py --schema-mode baked
```

拡張データセットを再生成する場合:

```bash
python datagen/generate_synthetic_dataset.py --route-groups 80
python training/prepare_sft.py \
  --input data/raw/synthetic_generated.jsonl \
  --output data/processed/sft_generated.jsonl \
  --schema-mode baked
```

8ツールの完全な宣言は512 tokenを大幅に圧迫するため、既定はschemaを重みに覚えさせる `baked` です。`compact` / `full` も比較用に選べます。学習スクリプトはassistant targetまで512 tokenに収まらないrecordを除外し、全件超過なら停止します。

学習lossはdeveloper/user/tool responseをmaskし、最後のassistant function call（不足情報例では空のassistant turn）だけに適用します。これにより長いpromptの模倣よりtool call構文を優先します。512 token学習とprompt分布を一致させるため、推論も `baked` が既定です。

raw JSONLの1行は次の形です。駅名は直接 `plan_journey` に入れず、最初に `suggest_stations` で解決します。

```json
{"id":"example","reference_datetime":"2026-06-28 09:00 Asia/Tokyo","user":"東京駅を検索","assistant":{"tool_name":"suggest_stations","arguments":{"q":"東京駅","limit":5}}}
```

不足情報のnegative exampleは `"assistant":{"no_tool_call":true}` とします。変換後は空のassistant turnとなり、実行側も自然文をMCP callとして扱いません。

## 3. LoRA学習

GTX 1650 4GB向けの開始設定です。full fine-tune、bf16、batch size増加は実装していません。

```bash
python training/train_lora.py \
  --dataset data/processed/sft_generated.jsonl \
  --output-dir outputs/functiongemma-transit-lora \
  --max-seq-length 512 \
  --lora-rank 8 \
  --epochs 3 \
  --gradient-accumulation-steps 16
```

通常のLoRAでも270Mモデルは小さいため、まず上記を試します。OOM時は指定どおり次へ落とします。

FunctionGemmaの一部RMSNorm重みはGTX 1650上でbase全体をfp16保存するとoverflowするため、通常LoRAは凍結baseをfp32保持し、学習演算をfp16 autocastで行います。fp16 GradScalerは初期scaleを1にしてNaNを防止します。bf16は使用しません。`--qlora` ではNF4量子化、fp32 norm、fp16 matrix computeを使用します。

```bash
python training/train_lora.py --max-seq-length 256 --lora-rank 4 \
  --output-dir outputs/functiongemma-transit-lora-r4
```

さらにbase modelの常駐量を減らす比較実験は `--qlora` を追加します。GTX 1650とbitsandbytesの組み合わせでは環境依存があるため、LoRAを既定にしています。

## 4. 推論とMCP実行

```bash
python -m transit_functiongemma.infer '東京駅を検索して' \
  --adapter outputs/functiongemma-transit-lora

python scripts/mcp_client.py \
  --model-output '<start_function_call>call:suggest_stations{q:<escape>東京駅<escape>,limit:5}<end_function_call>'
```

`FUNCTIONGEMMA_CONSTRAINED_DECODE=1` と `--constrained-decode` を同時に指定すると、推論時に1件のtool callまたはno-callだけを許す最小文法制約をかけます。
`FUNCTIONGEMMA_PREFIX_CACHE=1` と `--prefix-cache` を同時に指定すると、baked modeの固定developer prefixをKV cache化してPyTorch推論のprefillを削減します。
prefixが1 tokenでも一致しない入力では自動的に従来のfull prompt生成へフォールバックします。

`scripts/mcp_client.py` は次を満たさない出力を実行しません。

- FunctionGemma形式で厳密にparseできる
- 自然文とcallが混在しない
- callがちょうど1件
- tool名が取得済みschemaの許可リスト内
- argumentsがMCPのJSON Schemaに適合

完全なMCP envelopeは `artifacts/mcp/` に保存されます。MCP Appsが返す `structuredContent`、埋め込みresource / resource link、`_meta` も欠落させず保存します。標準出力の自然文整形はMCP `content[type=text]` を表示するだけの決定的テンプレートです。

## 5. 評価

まず付属の正解形式predictionで評価器自体を確認できます。

```bash
python evaluation/eval_toolcall.py --predictions data/eval/predictions_example.jsonl
```

学習済みadapterを直接評価する場合:

```bash
python evaluation/eval_toolcall.py --run-model \
  --adapter outputs/functiongemma-transit-lora
```

生成時の余分なresponse tokenを抑える実験は、環境変数とCLIの両方を指定したときだけ有効です:
`FUNCTIONGEMMA_CONSTRAINED_DECODE=1 python evaluation/eval_toolcall.py --run-model --constrained-decode ...`
PyTorch推論の固定prefix cacheも同様に、`FUNCTIONGEMMA_PREFIX_CACHE=1` と `--prefix-cache` を両方指定したときだけ有効です。

GGUFモデルを既存採点器へ通すには `python scripts/generate_gguf_predictions.py --dataset data/eval/sonnet5_holdout_60.jsonl --gguf gguf_work/r8b_Q6_K.gguf --output artifacts/predictions.jsonl` でpredictionを作り、
`python evaluation/eval_toolcall.py --dataset data/eval/sonnet5_holdout_60.jsonl --predictions artifacts/predictions.jsonl` を実行します。

出力指標は `parse_success_rate`、`tool_name_accuracy`、`required_arguments_satisfaction_rate`、`datetime_normalization_success_rate`、`no_call_when_missing_info_rate` です。詳細は `artifacts/eval_report.json` に保存されます。

実運用経路はraw model、表記正規化+schema、実MCP final pipelineを必ず分離して評価します。

```bash
python scripts/run_three_stage_evaluation.py \
  --adapter outputs/functiongemma-transit-intent-r5 \
  --intent-dataset data/eval/operational_intent_raw_100.jsonl \
  --final-dataset data/eval/operational_tokyo_routes_100.jsonl
```

各段階の意味は次の通りです。

1. raw model: 元の日本語をFunctionGemmaへ直接入力し、`resolve_route_request`とslotを採点
2. normalized: NFKCと空白だけを正規化し、schema-invalid callを拒否して採点
3. final pipeline: model slotを駅解決・実Transit MCP・filter/reranker/rendererまで通して採点

追加SFTの`data/raw/operational_intent_train.jsonl`とraw holdout
`data/eval/operational_intent_raw_100.jsonl`は入力文重複ゼロです。前者は駅名を固定
ローテーションした同等表現、後者はユーザー作成の元100件です。

従来protocolの大規模holdoutと手書き実用評価は互換回帰として残しています。
旧regex意味評価を再現するときだけ`--legacy-semantic-eval`を明示してください。
以下のr4コマンドと2026-06-29実測値は旧beta4回帰記録であり、現行Web配備の
v1.0.0評価値ではありません。

```bash
python evaluation/validate_eval_datasets.py
python evaluation/eval_toolcall.py \
  --dataset data/eval/independent_holdout_300.jsonl \
  --run-model --adapter outputs/functiongemma-transit-ja-real-r4 \
  --normalize-ja --schema-constraint \
  --output artifacts/eval_independent_holdout_300.json \
  --markdown-output artifacts/eval_independent_holdout_300.md \
  --failures-output artifacts/failures_independent_holdout_300.jsonl
python evaluation/eval_toolcall.py \
  --dataset data/eval/manual_practical_100.jsonl \
  --run-model --adapter outputs/functiongemma-transit-ja-real-r4 \
  --normalize-ja --schema-constraint \
  --output artifacts/eval_manual_practical_100.json \
  --markdown-output artifacts/eval_manual_practical_100.md \
  --failures-output artifacts/failures_manual_practical_100.jsonl
```

`history`付き行は解決済み駅IDを含む会話全体をrouterへ渡します。rawモデルのparse/tool
精度と、表記正規化・schema制約後の精度は分けて記録します。前段はユーザー文から
priority / avoid / via / mode / time_modeを抽出・補完しません。

実Transit MCPを使う固定7シナリオのE2E評価:

```bash
python evaluation/eval_pipeline.py \
  --adapter outputs/functiongemma-transit-ja-real-r4 \
  --clarification-tool \
  --max-routes 1
```

raw envelope、normalized JSON、renderer回答、request/MCP latency、timeout、retryを
`artifacts/e2e_*`と`artifacts/latency_report.json`へ保存します。
固定7シナリオでは、不足情報をローカルで質問しMCPへ送らないことも評価するため、
`--clarification-tool`を明示します。
`--max-routes 1`はレイテンシゲート用にMCPの探索候補数も1件へ制限します。Web UIは
3候補表示を維持します。

2026-06-29の最終実測では、`independent_holdout_300`はparse/tool/arguments包含/
required/no-call/schemaがすべて100%でした。`manual_practical_100`はclass/toolとno-callが
100%、intent slot macro F1が97.84%、avoid/viaと日時正規化が100%でした。固定7シナリオの
実MCP E2Eは7/7、p50 1.50秒、p95 4.06秒、timeout 0%、no-call無通信100%、rendererの
source-only検査100%です。詳細は `artifacts/eval_independent_holdout_300.md`、
`artifacts/eval_manual_practical_100.md`、`artifacts/e2e_eval_report.md` を参照してください。

評価器は、現行MCP schemaに存在しない期待引数と、user/historyから観測できない期待値を
件数付きで意味評価から除外します。これはモデルに入力外の値を推測させないためで、raw
model指標、schema制約後の指標、除外件数を別々に保存します。

epoch別adapterから外部dev指標でbestを選ぶには `python scripts/select_checkpoint.py --run outputs/<run>` を使います。GPUなしの配線確認は `--dry-run --dev-sample N` で行えます。

## 注意

`synthetic_template.jsonl` の20件は配線確認用です。通常学習にはschema検証済み470件の `synthetic_generated.jsonl` を使います。さらなる実験では各tool、言い換え、相対日時、曖昧駅名、誤誘導、required欠落、多段tool responseをtrain/evalで重複しないよう増やしてください。`demo-feed:*` は学習例内のtool responseから受け取った架空IDであり、実MCPへ送る値ではありません。

tool選択改善用のbalanced実験は [NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) を参照してください。`ask_clarification` は任意のローカル疑似toolであり、`scripts/mcp_client.py` が必ずMCP送信前に処理します。

旧口語日本語実験の履歴は
[JAPANESE_ROUTING_EXPERIMENT.md](docs/JAPANESE_ROUTING_EXPERIMENT.md)を参照してください。
現在の`--normalize-ja`はNFKCと空白整理だけです。旧intent/slot regexは
`semantic_fallback=True`または`--legacy-semantic-eval`を明示した互換試験と、offlineの
annotation migrationに限定され、productionでは使いません。棚卸しは
`artifacts/PARSER_RULE_AUDIT.md`にあります。

以下は旧beta4（旧名r4）実験の記録です。現行Web配備の評価値ではありません。GTX 1650
4GBで口語日本語86件を加えた5 epochのLoRA本学習を完了しました。外部評価は
口語holdout 17件でparse/tool/arguments/no-call/schemaがすべて100%、9 class均等の
corrected eval 72件でparse/tool/no-call/schemaが100%、expected argumentsが81.25%です。
この旧実験のadapterは`outputs/functiongemma-transit-ja-real-r4`です。現行v1.0.0の系譜と
評価は[リリースmanifest](docs/RELEASE_MANIFEST_v1.0.0.md)を参照してください。

## 6. LLMを使わない日本語回答パイプライン

FunctionGemmaは自然文回答モデルではなく、MCP tool callを選ぶrouterとしてのみ使います。
経路、時刻、運賃、路線などの事実はTransit MCPが取得し、`transit_functiongemma/route_normalizer.py`が
MCPのraw resultを共通JSONへ変換します。`transit_functiongemma/route_constraints.py`は駅IDまたは保守的に
正規化した駅名で回避・経由条件を検査し、`transit_functiongemma/route_renderer.py`がそのJSONに存在する値
だけをPythonテンプレートで日本語化します。

後段LLMを置かないのは、MCP結果にない経路・料金・時刻・路線名や、回避条件の成否を
生成モデルが補完してしまう余地をなくすためです。normalizer、constraints、rendererは
ネットワークを使用せず、オフラインで単体テストできます。ネットワークアクセスは
`scripts/mcp_client.py`と`transit_functiongemma/answer_pipeline.py`に限定しています。

rendererだけをサンプルJSONで確認:

```bash
python -m transit_functiongemma.route_renderer --input data/examples/route_normalized_example.json
python -m transit_functiongemma.route_renderer --input data/examples/station_suggestions_example.json
python -m transit_functiongemma.route_renderer --input data/examples/place_suggestions_example.json
```

学習済みadapterを使った一連の実行:

```bash
export FUNCTIONGEMMA_ADAPTER=outputs/tentetsu-270m-v1.0.0

python -m transit_functiongemma.answer_pipeline "東京駅を検索して"
python -m transit_functiongemma.answer_pipeline "東京タワーを場所として探して"
python -m transit_functiongemma.answer_pipeline "町田から池袋まで、渋谷を避けて" \
  --normalize-ja --save-normalized artifacts/normalized --debug
```

現在のrouterは駅名を直接経路toolへ渡さず、まず`suggest_stations`で解決する段階式です。
経路要求では`transit_functiongemma/answer_pipeline.py`がそのcallを物理駅解決へ切り替え、`suggest_places`の
完全一致駅だけを残し、300m以内の同名Transit/OSM地点を1駅にまとめます。その代表
`geo:` endpointを多段tool履歴へ戻すため、東海道線・京浜東北線などのfeed IDを経路探索
前に選ぶ必要がありません。「横浜」と「新横浜」、同名の店舗などは一致扱いにしません。
同名の物理駅が離れた場所に複数ある場合だけ利用者へ確認します。

```bash
python -m transit_functiongemma.answer_pipeline "横浜から上野までの経路を探して"
python -m transit_functiongemma.answer_pipeline "横浜から上野まで、京浜東北線で行きたい"
```

「京浜東北線で」のような明示路線は、MCPが返した経路の各legをクライアント側で検査し、
該当候補を優先表示します。サードパーティーMCPの検索条件自体は変更できないため、返却
候補に指定路線がなければ、その範囲では確認できなかったと表示し、経路を捏造しません。

`transit_functiongemma/answer_pipeline.py`はrouter出力を厳格parseし、取得済みJSON Schemaで検証してから
MCPを呼びます。空callまたは`ask_clarification`の場合はMCPを呼ばず確認文を返します。
raw envelopeは既定で`artifacts/mcp/`へ保存され、`--save-normalized`を指定すると正規化
JSONも保存されます。主なオプションは`--adapter`、`--mcp-url`、`--schema-mode`、
`--save-raw`、`--save-normalized`、`--max-routes`、`--debug`です。

実MCP保存物をオフラインで正規化する場合:

```bash
python -m transit_functiongemma.route_normalizer \
  --input artifacts/mcp/example_plan_journey.json \
  --tool-name plan_journey \
  --output artifacts/normalized/example.json
python -m transit_functiongemma.route_renderer --input artifacts/normalized/example.json
```

## 7. Web UI

静的UIはサーバーの`/var/www/html/transit/`に配置し、Apacheの内部プロキシから
`127.0.0.1:8091`の常駐APIへ接続します。APIは学習済みadapterを一度だけGPUへロードし、
各リクエストで同じ`ToolRouter`を再利用します。APIポート自体は外部公開しません。
現行Web APIは`outputs/tentetsu-270m-v1.0.0`（Tentetsu-270M v1.0.0）をサーバー上の
PyTorch/PEFT routerとして実行します。iPhoneアプリが端末内で実行するQ6_K GGUFとは
画面上の安全境界を共有しますが、Web版はサーバー推論でありサーバレスではありません。
`/api/health`は従来の`model`/`ready`に加えて、`router_release`、`adapter`、
`inference_backend=server`を返します。

```text
https://<your-domain>/transit/
```

サービス確認・再起動:

```bash
systemctl --user status transit-functiongemma-web.service
systemctl --user restart transit-functiongemma-web.service
journalctl --user -u transit-functiongemma-web.service -f
```

### 実運用behavior log

公開デモ用systemd templateは匿名監査ログだけを有効にし、query hash・文字数・status
などの運用指標だけを保存します。ユーザーの検索文、回答本文、raw座標は保存せず、詳細な
behavior logも既定で無効です。

同意を得た閉じた検証で数週間の表現揺れ・時刻解釈・tool選択を調べる場合に限り、systemd
unitの`TRANSIT_BEHAVIOR_LOG=1`を明示的に有効にします。このflagだけではtiming、MCPの
tool名・status・latency・attempt回数などの非本文指標だけを保存します。検索文と入力由来の
router出力・tool引数・intentまで必要な場合は`TRANSIT_BEHAVIOR_LOG_USER_QUERY=1`を、
回答本文も必要な場合は`TRANSIT_BEHAVIOR_LOG_ANSWER=1`を個別に指定します。
これらは公開デモの既定値にしません。
有効にしたbehavior logはJSTの日付ごとに次へ保存されます。

```text
artifacts/behavior_logs/YYYY-MM-DD.jsonl
```

各行はrequest ID、status、応答種別、文字数、MCP tool名とstatus/attempt/latency、全体
latencyを持ちます。`TRANSIT_BEHAVIOR_LOG_USER_QUERY=1`の場合だけ、入力、router出力、
`model_route_intent`、deterministic planner step、parsed tool call、schema検証、MCPへ送った
最終引数も追加します。`TRANSIT_BEHAVIOR_LOG_ANSWER=1`の場合だけ回答本文も追加します。
詳細モードでも`lat`、`lon`、`geo:` endpointと緯度経度文字列は常にマスクし、MCP raw
resultとnormalized route本体はbehavior logへ保存しません。既定保持期間は45日で、
期限切れの日次ファイルは起動時と日付変更後の最初の記録時に削除されます。

```bash
# 今日のログを追う
tail -f artifacts/behavior_logs/$(date +%F).jsonl

# サーバーから回収する
rsync -av <server>:/path/to/transit_functiongemma_mcp/artifacts/behavior_logs/ \
  artifacts/behavior_logs/
```

詳細ログはアクセスを制限した検証期間だけ有効にし、終了後は
`TRANSIT_BEHAVIOR_LOG_USER_QUERY=0`、`TRANSIT_BEHAVIOR_LOG_ANSWER=0`、
`TRANSIT_BEHAVIOR_LOG=0`へ戻してください。

都内経路の実運用表現サンプルをまとめて実Web APIへ流す場合:

```bash
python scripts/run_operational_samples.py \
  --dataset data/eval/operational_tokyo_routes_100.jsonl \
  --url http://127.0.0.1:8091/query
```

結果は `artifacts/operational_tokyo_routes.json` と
`artifacts/operational_tokyo_routes.md` に保存されます。HTTP応答だけでなく、behavior logの
モデルの`resolve_route_request`と最終MCP引数を参照し、到着/出発、時刻、回避、経由、
優先条件がmodel extractionとexecutionの両段階を通ったかを採点します。元の日本語を
regexで再抽出して正解扱いにはしません。

同名の物理駅が複数ある場合、Web APIは15分間の対話セッションを作り、UIに候補ボタンを
表示します。選択した物理駅の`geo:` endpointを保持して元の検索を再開し、目的地も曖昧
なら続けて候補を表示します。候補名・事業者に加えて座標も表示するため、同名駅を区別
できます。セッション情報はメモリ内だけに保持され、完了または期限切れで削除されます。

単独の`suggest_stations` / `suggest_places`結果もボタン表示されます。候補を選ぶと駅IDを
確定して表示し、「出発地にする」「目的地にする」から次の検索文へつなげられます。経路
検索中の駅解決は路線別IDではなく物理駅`geo:`を使うため、候補選択が利用路線を不必要に
固定することはありません。

構成ファイルは`web/`、APIは`web_api.py`、systemd unitの原本は
`deploy/transit-functiongemma-web.service`です。

公開運用向けにclient単位rate limit、MCP timeout、指数backoff retry、短期cache、schema
hash変更検知、request ID、匿名監査ログを実装しています。user queryとraw座標の保存は
既定でOFFです。`TRANSIT_WEB_SHOW_TRACE=1`ではモデル出力、parse済みcall、schema検証、
MCP call、normalized JSON、renderer回答とlatencyを折り畳み表示できます。公開前の残作業は
[NEXT_RELEASE_CHECKLIST.md](docs/NEXT_RELEASE_CHECKLIST.md)を参照してください。
