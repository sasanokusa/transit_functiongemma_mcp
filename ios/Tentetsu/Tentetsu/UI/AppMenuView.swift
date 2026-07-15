import SwiftUI
import UIKit

struct AppMenuView: View {
    @Binding var prefersMap: Bool
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("検索") {
                    Toggle(isOn: $prefersMap) {
                        Label("地図を表示", systemImage: "map.fill")
                    }
                    Text("オンの場合も、経路概要を先に表示してから地図を追加します。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("プライバシーとデータ") {
                    NavigationLink {
                        PrivacyView()
                    } label: {
                        Label("データの取り扱い", systemImage: "hand.raised.fill")
                    }
                    NavigationLink {
                        DataSourcesView()
                    } label: {
                        Label("交通・地図データの提供元", systemImage: "building.columns.fill")
                    }
                    Button {
                        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
                        UIApplication.shared.open(url)
                    } label: {
                        Label("マイクなどの権限を開く", systemImage: "gear")
                    }
                }

                Section("アプリ情報") {
                    NavigationLink {
                        AboutView()
                    } label: {
                        Label("このアプリについて", systemImage: "info.circle.fill")
                    }
                    NavigationLink {
                        LicensesView()
                    } label: {
                        Label("ライセンスと利用条件", systemImage: "doc.text.fill")
                    }
                    LabeledContent("バージョン", value: AppBuildInfo.versionDescription)
                }

                Section {
                    Label("試作版", systemImage: "wrench.and.screwdriver.fill")
                        .foregroundStyle(.orange)
                    Text("経路や時刻は変更される場合があります。移動前に交通事業者の公式情報も確認してください。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("メニュー")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("完了") { dismiss() }
                }
            }
        }
    }
}

private struct AboutView: View {
    var body: some View {
        List {
            Section {
                VStack(spacing: 10) {
                    Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
                        .font(.system(size: 44, weight: .semibold))
                        .foregroundStyle(.blue)
                    Text("転轍").font(.title2.bold())
                    Text("端末内AI × Transit MCP")
                        .foregroundStyle(.secondary)
                    Text("バージョン \(AppBuildInfo.versionDescription)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }

            Section("仕組み") {
                LabeledContent("モデル", value: "FunctionGemma v1.0.0")
                LabeledContent("量子化", value: "Q6_K")
                LabeledContent("推論", value: "iPhone内")
                LabeledContent("経路検索", value: "Transit MCP")
                Text("AIは入力から検索条件だけを抽出します。経路、時刻、運賃、路線名はAIに生成させず、Transit MCPの応答から表示します。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("注意") {
                Text("本アプリは個人開発中の試作版であり、Google、Apple、交通事業者による公式アプリではありません。")
                Text("案内の完全性、正確性、最新性を保証するものではありません。重要な移動では交通事業者の公式案内を確認してください。")
            }
        }
        .navigationTitle("このアプリについて")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct PrivacyView: View {
    var body: some View {
        List {
            Section("端末内で処理するもの") {
                PrivacyRow(
                    icon: "iphone",
                    title: "入力の理解",
                    detail: "入力文はiPhone内のFunctionGemmaで解析します。AI推論のための外部サーバーへ入力全文を送信しません。"
                )
                PrivacyRow(
                    icon: "externaldrive.fill",
                    title: "保存",
                    detail: "検索履歴、音声録音、モデル出力、MCP応答をアプリが永続保存する機能はありません。地図表示の設定だけを端末に保存します。"
                )
            }

            Section("外部サービスへ送るもの") {
                PrivacyRow(
                    icon: "network",
                    title: "Transit MCP",
                    detail: "駅名、解決済み地点、検索日時、経由地、優先条件など、経路計算に必要な条件を送信します。入力全文や音声は送信しません。"
                )
                PrivacyRow(
                    icon: "waveform",
                    title: "音声認識",
                    detail: "AppleのSpeechフレームワークを使用します。端末が対応する場合は端末内認識を要求し、非対応時はAppleのサービスで処理される場合があります。アプリは録音を保存しません。"
                )
                PrivacyRow(
                    icon: "map.fill",
                    title: "地図",
                    detail: "背景地図の表示にはApple Mapsを使用します。本アプリは現在地の権限を要求しません。"
                )
            }

            Section("収集しないもの") {
                Label("アカウント・広告ID", systemImage: "person.crop.circle.badge.xmark")
                Label("端末の現在地", systemImage: "location.slash.fill")
                Label("独自のアクセス解析・広告SDK", systemImage: "chart.bar.xaxis")
            }
        }
        .navigationTitle("データの取り扱い")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct PrivacyRow: View {
    let icon: String
    let title: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: icon).font(.headline)
            Text(detail).font(.subheadline).foregroundStyle(.secondary)
        }
        .padding(.vertical, 3)
    }
}

private struct DataSourcesView: View {
    var body: some View {
        List {
            Section("経路・時刻表") {
                Text("経路、時刻、路線、駅情報はTransit MCPが統合する交通フィードから取得します。フィードごとの出典・ライセンスはTransit APIのカタログで確認できます。")
                Link("Transit API フィードカタログ", destination: URL(string: "https://api.transit.ls8h.com/api/docs")!)
                Link("Transit MCP 接続先", destination: URL(string: "https://api.transit.ls8h.com/mcp")!)
            }

            Section("地点情報") {
                Text("駅・施設の候補検索には、Transit MCPを通じてOpenStreetMap由来の地点情報が含まれる場合があります。")
                Link("© OpenStreetMap contributors", destination: URL(string: "https://www.openstreetmap.org/copyright")!)
            }

            Section("背景地図") {
                Text("地図の背景表示にはApple MapKitを使用します。地図上に表示されるAppleおよびデータ提供者の法的表示が適用されます。")
                Link("Appleマップ利用規約", destination: URL(string: "https://www.apple.com/legal/internet-services/maps/terms-jp.html")!)
            }
        }
        .navigationTitle("データ提供元")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct LicensesView: View {
    var body: some View {
        List {
            Section("モデル") {
                NavigationLink("FunctionGemma / Gemma 利用条件") {
                    GemmaLicenseView()
                }
            }

            Section("推論ランタイム") {
                NavigationLink("llama.cpp（MIT License）") {
                    MITLicenseView()
                }
            }

            Section("必須Notice") {
                Text("Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms")
                    .font(.footnote.monospaced())
                    .textSelection(.enabled)
            }
        }
        .navigationTitle("ライセンスと利用条件")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct GemmaLicenseView: View {
    var body: some View {
        List {
            Section("FunctionGemma") {
                Text("本アプリは google/functiongemma-270m-it を乗換検索用に追加学習し、Q6_Kへ量子化した派生モデルを同梱しています。変更は本プロジェクトによるもので、Googleによる承認・後援を意味しません。")
            }

            Section("利用条件") {
                Text("同梱モデルはGemma Terms of Useの対象です。違法な用途、およびGemma Prohibited Use Policyで禁止される用途には使用できません。利用者はこれらの条件に従う必要があります。")
                Link("Gemma Terms of Use", destination: URL(string: "https://ai.google.dev/gemma/terms")!)
                Link("Gemma Prohibited Use Policy", destination: URL(string: "https://ai.google.dev/gemma/prohibited_use_policy")!)
                Link("FunctionGemma model card", destination: URL(string: "https://huggingface.co/google/functiongemma-270m-it")!)
            }

            Section("Notice") {
                Text("Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms")
                    .font(.footnote.monospaced())
                    .textSelection(.enabled)
            }
        }
        .navigationTitle("FunctionGemma")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct MITLicenseView: View {
    var body: some View {
        ScrollView {
            Text(Self.license)
                .font(.footnote.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .navigationTitle("llama.cpp")
        .navigationBarTitleDisplayMode(.inline)
    }

    private static let license = """
    MIT License

    Copyright (c) 2023-2026 The ggml authors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
}

private enum AppBuildInfo {
    static var versionDescription: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "—"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "—"
        return "\(version) (\(build))"
    }
}

#Preview {
    AppMenuView(prefersMap: .constant(true))
}
