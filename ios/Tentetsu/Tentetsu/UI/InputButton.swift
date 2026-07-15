import AVFoundation
import Speech

@MainActor
final class SpeechInput: ObservableObject {
    @Published var transcript = ""
    @Published var isRecording = false
    @Published var errorMessage: String?

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var isStarting = false

    func toggle() {
        if isRecording { stop() }
        else { Task { await start() } }
    }

    func start() async {
        guard !isRecording, !isStarting else { return }
        guard let recognizer else {
            errorMessage = "現在、音声認識を利用できません。"
            return
        }
        isStarting = true
        defer { isStarting = false }
        let speech = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard speech == .authorized else {
            errorMessage = "音声認識の許可が必要です。"
            return
        }
        guard recognizer.isAvailable else {
            errorMessage = "現在、音声認識を利用できません。通信状態を確認してください。"
            return
        }
        let microphone = await AVAudioApplication.requestRecordPermission()
        guard microphone else {
            errorMessage = "マイクの許可が必要です。"
            return
        }
        do {
            task?.cancel()
            task = nil
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

            let request = SFSpeechAudioBufferRecognitionRequest()
            request.shouldReportPartialResults = true
            request.taskHint = .dictation
            request.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
            self.request = request

            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.sampleRate > 0, format.channelCount > 0 else {
                throw SpeechInputError.audioInputUnavailable
            }
            input.removeTap(onBus: 0)
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
                request.append(buffer)
            }
            engine.prepare()
            try engine.start()
            isRecording = true
            errorMessage = nil
            task = recognizer.recognitionTask(with: request) { [weak self] result, error in
                Task { @MainActor in
                    if let result { self?.transcript = result.bestTranscription.formattedString }
                    if error != nil || result?.isFinal == true { self?.stop() }
                }
            }
        } catch {
            errorMessage = error.localizedDescription
            stop()
        }
    }

    func stop() {
        guard isRecording || request != nil else { return }
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        request = nil
        task?.cancel()
        task = nil
        isRecording = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}

private enum SpeechInputError: LocalizedError {
    case audioInputUnavailable

    var errorDescription: String? {
        "マイク入力を開始できませんでした。接続中の音声機器を確認してください。"
    }
}
