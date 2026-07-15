# Tentetsu for iPhone

FunctionGemma v1.0.0 Q6_K を iPhone 上の llama.cpp で実行し、生成されたツール呼び出しだけを Transit MCP に送る試作アプリです。推論用サーバーは使いません。

## 現在の機能

- バンドル済み Q6_K GGUF の端末内推論
- 日本語のテキスト入力と音声入力
- FunctionGemma のツール呼び出しを厳格に解析
- `suggest_places` を使った駅名解決と Transit MCP の経路検索
- 候補切替、路線色、乗換タイムライン、経路ポリラインを備えたネイティブ地図表示
- 経路概要を先に表示し、時間のかかる地図をバックグラウンドで後着更新
- 未対応言語・曖昧入力・MCP障害を内部出力なしで案内する安全境界
- App Shortcut「乗換を音声で調べる」（アクションボタンから選択可能）

Transit MCP の接続先は `https://api.transit.ls8h.com/mcp` です。経路結果は端末内の決定論的ラッパーで、発着時刻・所要時間・乗換・徒歩・路線ごとの日本語表示へ整形します。

## 実機への導入

1. `Tentetsu.xcodeproj` を Xcode で開く。
2. Xcode の Settings > Accounts で Apple ID を追加する。
3. Tentetsu ターゲットの Signing & Capabilities で自分の Team を選ぶ。
4. 接続した iPhone を実行先に選び、Run を押す。
5. 初回起動時に音声認識とマイクへのアクセスを許可する。

無料の Personal Team でも自分の端末での試用は可能ですが、プロビジョニングには有効期限があります。

## アクションボタン

アプリを一度起動した後、iPhone の Settings > Action Button > Shortcut から「Tentetsuで乗換を調べる」を選びます。呼び出すとアプリを開き、音声入力を開始します。

## モデル

- ファイル: `Tentetsu/Resources/models/tentetsu-q6_k.gguf`
- 量子化: Q6_K
- SHA-256: `89eeb9d467995a32e9935b26f8543fd7c758bce32a0f5b03391873e05d4aaabb`

GGUF は `.gitignore` 対象です。別の端末やクローンでは同じ場所に配置してください。

## llama.cpp

アプリは GGUF 変換時と同じ llama.cpp `b9925`（commit `ed8c26150e6b0ed6e2635cab75ace5ed121482ca`）から生成した `Frameworks/llama.xcframework` を利用します。XCFramework もサイズが大きいため `.gitignore` 対象です。

## 確認済み環境

- Xcode 26.3
- iOS deployment target 17.0
- iPhone 向け arm64 署名ビルド成功
- 実機インストール、Q6_K端末内推論、音声入力開始を確認
- 「東京駅から自由が丘」で概要2候補の先行表示後、地図6区間への更新を確認

## デモ時の既知事項

Transit MCPの経路計算は、接続時間とは別に10〜25秒程度かかる場合があります。
地図付き表示では経路概要と地図を並行取得し、概要を先に操作可能にします。地図取得に
失敗しても、取得済みの時刻・乗換・徒歩案内は保持されます。
