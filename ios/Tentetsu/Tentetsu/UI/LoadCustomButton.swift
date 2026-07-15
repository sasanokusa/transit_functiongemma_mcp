import AppIntents
import Foundation

extension Notification.Name {
    static let startVoiceSearchRequested = Notification.Name("startVoiceSearchRequested")
}

struct StartVoiceSearchIntent: AppIntent {
    static let title: LocalizedStringResource = "音声で乗換検索"
    static let description = IntentDescription("転轍を開いて音声入力を開始します。")
    static let openAppWhenRun = true

    @MainActor
    func perform() async throws -> some IntentResult {
        UserDefaults.standard.set(true, forKey: "startVoiceSearch")
        NotificationCenter.default.post(name: .startVoiceSearchRequested, object: nil)
        return .result()
    }
}

struct TentetsuShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: StartVoiceSearchIntent(),
            phrases: [
                "\(.applicationName)で乗換を調べる",
                "\(.applicationName)で音声検索"
            ],
            shortTitle: "音声で乗換検索",
            systemImageName: "tram.fill"
        )
    }
}
