import Foundation

@MainActor
final class AppModel: ObservableObject {
    @Published var query = ""
    @Published var answer = ""
    @Published var modelOutput = ""
    @Published var status = "モデルを準備しています…"
    @Published var isBusy = false
    @Published var modelReady = false
    @Published var elapsedMilliseconds: Double?
    @Published var prefersMap = true
    @Published var routePresentation: TransitRoutePresentation?
    @Published private(set) var isMapLoading = false
    @Published private(set) var mapLoadMessage: String?

    let speech = SpeechInput()
    let showsDebugOutput = ProcessInfo.processInfo.environment["TENTETSU_SHOW_DEBUG_OUTPUT"] == "1"
    private let mcp = TransitMCPClient()
    private var context: LlamaContext?
    private var activeSearchID = UUID()
    private var mapLoadTask: Task<Void, Never>?

    init() {
#if DEBUG
        JapaneseRuntimeCompatibility.verifyFixtures()
        TransitInputPolicy.verifyFixtures()
#endif
        Task {
            await loadModel()
#if DEBUG
            if let smokeQuery = ProcessInfo.processInfo.environment["TENTETSU_SMOKE_QUERY"] {
                query = smokeQuery
                await search()
                print("TENTETSU_SMOKE_FIRST_RESULT_MS=\(elapsedMilliseconds ?? -1)")
                print("TENTETSU_SMOKE_MAP_LOADING_AFTER_SUMMARY=\(isMapLoading)")
                print("TENTETSU_SMOKE_SUMMARY_OPTIONS=\(routePresentation?.options.count ?? 0)")
                await mapLoadTask?.value
                print("TENTETSU_SMOKE_MODEL_OUTPUT=\(modelOutput)")
                print("TENTETSU_SMOKE_STATUS=\(status)")
                print("TENTETSU_SMOKE_MAP_MESSAGE=\(mapLoadMessage ?? "none")")
                print("TENTETSU_SMOKE_ANSWER=\(answer)")
                print("TENTETSU_SMOKE_ROUTE_OPTIONS=\(routePresentation?.options.count ?? 0)")
                print("TENTETSU_SMOKE_MAP_SEGMENTS=\(routePresentation?.options.first?.mapSegments.count ?? 0)")
            }
            if ProcessInfo.processInfo.environment["TENTETSU_SMOKE_SPEECH"] == "1" {
                await speech.start()
                print("TENTETSU_SMOKE_SPEECH_RECORDING=\(speech.isRecording)")
                print("TENTETSU_SMOKE_SPEECH_ERROR=\(speech.errorMessage ?? "none")")
            }
#endif
        }
    }

    func loadModel() async {
        guard context == nil else { return }
        let bundled = Bundle.main.url(
            forResource: "tentetsu-q6_k", withExtension: "gguf", subdirectory: "models"
        )
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let imported = (try? FileManager.default.contentsOfDirectory(
            at: documents, includingPropertiesForKeys: nil
        ))?.first(where: { $0.pathExtension.lowercased() == "gguf" })
        guard let modelURL = bundled ?? imported else {
            status = "GGUFモデルがありません。Resources/modelsへ追加してください。"
            return
        }
        do {
            status = "\(modelURL.lastPathComponent)を読み込み中…"
            context = try await Task.detached {
                try LlamaContext.create_context(path: modelURL.path)
            }.value
            modelReady = true
            status = "端末内モデル準備完了"
        } catch {
            status = "モデル読込エラー: \(error.localizedDescription)"
        }
    }

    func search() async {
        guard !isBusy else { return }
        guard let context else {
            await loadModel()
            return
        }
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let searchID = UUID()
        activeSearchID = searchID
        mapLoadTask?.cancel()
        mapLoadTask = nil
        isMapLoading = false
        mapLoadMessage = nil
        if let rejection = TransitInputPolicy.rejectionMessage(for: text) {
            answer = rejection
            modelOutput = ""
            routePresentation = nil
            elapsedMilliseconds = nil
            status = "入力を確認してください"
            return
        }
        isBusy = true
        answer = ""
        modelOutput = ""
        routePresentation = nil
        status = "端末内モデルで解析中…"
        let started = ContinuousClock.now
        do {
            let prompt = routerPrompt(user: JapaneseRuntimeCompatibility.normalizeSurface(text))
            let output = try await context.generate(prompt: prompt, maxTokens: 128)
            modelOutput = output
            status = "MCPで経路を検索中…"
            answer = try await execute(output: output, userText: text, searchID: searchID)
            let duration = started.duration(to: .now).components
            elapsedMilliseconds = Double(duration.seconds) * 1_000
                + Double(duration.attoseconds) / 1e15
            status = isMapLoading ? "経路を表示しました。地図を読み込み中…" : "完了"
        } catch {
            mapLoadTask?.cancel()
            isMapLoading = false
            mapLoadMessage = nil
            answer = userFacingErrorMessage(for: error)
            status = "エラー"
#if DEBUG
            print("TENTETSU_SEARCH_ERROR=\(String(reflecting: error))")
#endif
        }
        isBusy = false
    }

    func consumeSpeechTranscript() {
        if !speech.transcript.isEmpty { query = speech.transcript }
    }

    func startVoiceFromShortcutIfNeeded() {
        guard UserDefaults.standard.bool(forKey: "startVoiceSearch") else { return }
        UserDefaults.standard.removeObject(forKey: "startVoiceSearch")
        Task { await speech.start() }
    }

    private func execute(output: String, userText: String, searchID: UUID) async throws -> String {
        guard let parsedCall = try ToolCallParser.parse(output) ?? fallbackRouteCall(from: userText) else {
            return "出発地と目的地が不足しています。どこからどこまで行きますか？"
        }
        let call = JapaneseRuntimeCompatibility.repair(
            parsedCall, from: userText, referenceDate: Date()
        )
        if call.name == "ask_clarification" {
            return clarificationMessage(for: missingFields(in: userText))
        }
        if call.name == "resolve_route_request" {
            guard routeArgumentsAreGrounded(call.arguments, in: userText) else {
                if let fallback = fallbackRouteCall(from: userText) {
                    return try await executeRoute(
                        intent: fallback.arguments, userText: userText, searchID: searchID
                    )
                }
                return clarificationMessage(for: missingFields(in: userText))
            }
            return try await executeRoute(
                intent: call.arguments, userText: userText, searchID: searchID
            )
        }
        let allowed = [
            "suggest_stations", "suggest_places", "reverse_geocode",
            "station_departures", "get_station", "list_feeds"
        ]
        guard allowed.contains(call.name) else { throw ToolCallParserError.invalidOutput }
        let result = try await mcp.callTool(name: call.name, arguments: call.arguments)
        return TransitAnswerFormatter.render(result, toolName: call.name)
    }

    /// A conservative fallback for the explicit, unambiguous "AからB" form.
    /// It only copies user text into the local route-intent carrier and never
    /// invents station identifiers or transit facts.
    private func fallbackRouteCall(from text: String) -> LocalToolCall? {
        let normalized = JapaneseRuntimeCompatibility.normalizeSurface(text)
        guard let separator = normalized.range(of: "から") else { return nil }
        let origin = normalized[..<separator.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var destination = normalized[separator.upperBound...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        for suffix in ["まで行きたい", "まで", "へ行きたい", "へ行く", "に行きたい", "に行く"] {
            if destination.hasSuffix(suffix) {
                destination.removeLast(suffix.count)
                destination = destination.trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
        }
        guard !origin.isEmpty, !destination.isEmpty else { return nil }
        return LocalToolCall(name: "resolve_route_request", arguments: [
            "origin_text": .string(origin),
            "destination_text": .string(destination),
            "avoid_station_texts": .array([]),
            "via_station_texts": .array([]),
            "graphical": .bool(false),
            "priority": .null,
            "time_mode": .null,
            "date": .null,
            "time": .null
        ])
    }

    private func executeRoute(
        intent: [String: JSONValue], userText: String, searchID: UUID
    ) async throws -> String {
        guard let origin = intent["origin_text"]?.string,
              let destination = intent["destination_text"]?.string else {
            return "出発地と目的地が不足しています。どこからどこまで行きますか？"
        }
        let viaNames = intent["via_station_texts"]?.array?.compactMap(\.string) ?? []
        var resolved: [(name: String, endpoint: String)] = []
        for name in [origin] + viaNames + [destination] {
            resolved.append(try await resolveStation(name))
        }
        var arguments: [String: JSONValue] = [
            "from": .string(resolved.first!.endpoint),
            "to": .string(resolved.last!.endpoint),
            "fromLabel": .string(resolved.first!.name),
            "toLabel": .string(resolved.last!.name)
        ]
        if resolved.count > 2 {
            arguments["via"] = .array(resolved.dropFirst().dropLast().map { .string($0.endpoint) })
            arguments["viaLabel"] = .array(resolved.dropFirst().dropLast().map { .string($0.name) })
        }
        if let date = normalizedDate(from: userText) ?? intent["date"]?.string { arguments["date"] = .string(date) }
        if let time = normalizedTime(from: userText) ?? intent["time"]?.string { arguments["time"] = .string(time) }
        if let mode = intent["time_mode"]?.string {
            let type = ["last_train": "last", "first_train": "first", "arrive_by": "arrival", "departure_at": "departure"][mode]
            if let type { arguments["type"] = .string(type) }
        }
        let priority = intent["priority"]?.string
        let constrained = priority != nil || !(intent["avoid_station_texts"]?.array ?? []).isEmpty
        arguments["numItineraries"] = .number(constrained ? 6 : 1)
        let graphical = prefersMap || intent["graphical"] == .bool(true)
        guard graphical else {
            let result = try await mcp.callTool(name: "plan_journey", arguments: arguments)
            routePresentation = TransitRoutePresentation(result: result)
            return TransitAnswerFormatter.render(result, toolName: "plan_journey")
        }

        var mapArguments = arguments
        let strategy = [
            "fast": "fastest", "cheap": "lowestFare",
            "few_transfers": "fewestTransfers", "less_walk": "shortestWalk"
        ][priority ?? ""] ?? "balanced"
        mapArguments["strategy"] = .string(strategy)

        // Start the richer map request just after the summary request has had a
        // chance to reach the server. Users see the smaller journey response
        // first while both server operations overlap.
        isMapLoading = true
        let mapRequest = Task { [mcp] in
            try await Task.sleep(nanoseconds: 150_000_000)
            try Task.checkCancellation()
#if DEBUG
            if ProcessInfo.processInfo.environment["TENTETSU_SMOKE_FORCE_MAP_FAILURE"] == "1" {
                throw URLError(.badServerResponse)
            }
#endif
            return try await mcp.callTool(name: "plan_route_map", arguments: mapArguments)
        }

        let summaryResult: MCPToolResult
        do {
            summaryResult = try await mcp.callTool(name: "plan_journey", arguments: arguments)
        } catch {
            mapRequest.cancel()
            isMapLoading = false
            throw error
        }
        routePresentation = TransitRoutePresentation(result: summaryResult)
        let summary = TransitAnswerFormatter.render(summaryResult, toolName: "plan_journey")

        mapLoadTask = Task { [weak self] in
            guard let self else {
                mapRequest.cancel()
                return
            }
            defer {
                if self.activeSearchID == searchID { self.isMapLoading = false }
            }
            do {
                let mapResult = try await mapRequest.value
                guard !Task.isCancelled, self.activeSearchID == searchID else { return }
                guard let presentation = TransitRoutePresentation(result: mapResult) else {
                    self.mapLoadMessage = "地図を表示できませんでした。経路案内はそのまま利用できます。"
                    self.status = "経路を表示しました"
                    return
                }
                self.routePresentation = presentation
                self.answer = TransitAnswerFormatter.render(mapResult, toolName: "plan_route_map")
                self.mapLoadMessage = nil
                self.status = "完了"
            } catch is CancellationError {
                return
            } catch {
                guard self.activeSearchID == searchID else { return }
                self.mapLoadMessage = "地図を取得できませんでした。経路案内はそのまま利用できます。"
                self.status = "経路を表示しました"
#if DEBUG
                print("TENTETSU_MAP_ERROR=\(String(reflecting: error))")
#endif
            }
        }
        return summary
    }

    private func resolveStation(_ query: String) async throws -> (name: String, endpoint: String) {
        let result = try await mcp.callTool(name: "suggest_places", arguments: [
            "q": .string(query.hasSuffix("駅") ? query : query + "駅"), "limit": .number(30)
        ])
        let places = result.decodedText?.object?["places"]?.array ?? []
        let normalized = query.replacingOccurrences(of: "駅", with: "")
        let candidates = places.compactMap { value -> (String, String, String)? in
            guard let item = value.object,
                  let name = item["name"]?.string,
                  let endpoint = item["endpoint"]?.string,
                  let kind = item["kind"]?.string else { return nil }
            return (name, endpoint, kind)
        }
        let selected = candidates.first {
            $0.2 == "station" && $0.0.replacingOccurrences(of: "駅", with: "") == normalized
        } ?? candidates.first(where: { $0.2 == "station" })
        guard let selected else { throw RouteSearchError.stationNotFound(query) }
        return (selected.0, selected.1)
    }

    private func userFacingErrorMessage(for error: Error) -> String {
        if case RouteSearchError.stationNotFound(let query) = error {
            return "「\(query)」に一致する駅が見つかりませんでした。駅名を確認してもう一度お試しください。"
        }
        if error is ToolCallParserError {
            return "入力を乗換案内として解釈できませんでした。\n「〇〇駅から〇〇駅」のように入力してください。"
        }
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut:
                return "経路検索がタイムアウトしました。しばらく待ってからもう一度お試しください。"
            case .notConnectedToInternet, .networkConnectionLost:
                return "ネットワークに接続できません。通信状態を確認してもう一度お試しください。"
            default:
                return "Transit MCPから経路情報を取得できませんでした。しばらく待ってからもう一度お試しください。"
            }
        }
        return "処理を完了できませんでした。入力を確認してもう一度お試しください。"
    }

    private func routerPrompt(user: String) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        formatter.dateFormat = "yyyy-MM-dd HH:mm 'Asia/Tokyo'"
        let now = formatter.string(from: Date())
        let developer = "You are a model that can do function calling with the following functions. Return only a function call. Never invent stations, routes, fares, durations, lines, station IDs, or coordinates. For a complete natural-language route request, call resolve_route_request and extract only what the user stated. The deterministic planner will resolve stations and query transit facts. If required user information is missing, return no function call. Current local datetime: \(now)."
        return "<bos><start_of_turn>developer\n\(developer)<end_of_turn>\n<start_of_turn>user\n\(user.precomposedStringWithCompatibilityMapping)<end_of_turn>\n<start_of_turn>model\n"
    }

    private func missingFields(in text: String) -> [String] {
        var result: [String] = []
        if !text.contains("から") && text.range(of: #"\S+発"#, options: .regularExpression) == nil { result.append("origin") }
        if !text.contains("まで") && !text.contains("へ") && text.range(of: #"\S+に(?:着き|到着)"#, options: .regularExpression) == nil { result.append("destination") }
        return result
    }

    private func clarificationMessage(for missing: [String]) -> String {
        if missing == ["origin"] {
            return "出発地が不足しています。どこから出発しますか？"
        }
        if missing == ["destination"] {
            return "目的地が不足しています。どこまで行きますか？"
        }
        return "乗換案内には出発地と目的地が必要です。\n「〇〇駅から〇〇駅」のように入力してください。"
    }

    private func routeArgumentsAreGrounded(
        _ arguments: [String: JSONValue], in userText: String
    ) -> Bool {
        guard let origin = arguments["origin_text"]?.string,
              let destination = arguments["destination_text"]?.string else { return false }
        let source = groundedText(userText)
        let originValue = groundedText(origin)
        let destinationValue = groundedText(destination)
        return !originValue.isEmpty && !destinationValue.isEmpty
            && source.contains(originValue) && source.contains(destinationValue)
    }

    private func groundedText(_ text: String) -> String {
        JapaneseRuntimeCompatibility.normalizeSurface(text)
            .lowercased()
            .replacingOccurrences(of: #"[\s駅]"#, with: "", options: .regularExpression)
    }

    private func normalizedDate(from text: String) -> String? {
        JapaneseRuntimeCompatibility.normalizedDate(from: text, referenceDate: Date())
    }

    private func normalizedTime(from text: String) -> String? {
        JapaneseRuntimeCompatibility.normalizedTime(from: text)
    }
}

private enum RouteSearchError: Error {
    case stationNotFound(String)
}

/// Product boundary for text that the current Japanese route UI does not support.
/// Unsupported scripts are rejected before inference; ambiguous supported input
/// is handled by deterministic clarification messages after inference.
private enum TransitInputPolicy {
    static func rejectionMessage(for text: String) -> String? {
        guard text.unicodeScalars.contains(where: isUnsupportedScript) else { return nil }
        return "現在は日本語の乗換案内に対応しています。\n「〇〇駅から〇〇駅」のように、出発地と目的地を日本語で入力してください。"
    }

    private static func isUnsupportedScript(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x0400...0x052F, 0x1C80...0x1C8F, 0x2DE0...0x2DFF, 0xA640...0xA69F:
            return true // Cyrillic and its extensions.
        default:
            return false
        }
    }

#if DEBUG
    static func verifyFixtures() {
        assert(rejectionMessage(for: "Какой у тебя") != nil)
        assert(rejectionMessage(for: "〇〇駅から〇〇駅") == nil)
        assert(rejectionMessage(for: "TokyoからShibuya") == nil)
    }
#endif
}

/// Deterministic Japanese renderer for Transit MCP responses. It formats only
/// values contained in the response and never asks the language model to write
/// the final answer.
enum TransitAnswerFormatter {
    private static let emptyMessage = "条件に合う候補が見つかりませんでした。\n出発地・目的地・時刻を変えて再検索してください。"

    static func render(_ result: MCPToolResult, toolName: String) -> String {
        guard let decoded = result.decodedText, let root = decoded.object else {
            return result.text
        }
        if toolName == "plan_journey" || toolName == "plan_route_map" || root["journeys"] != nil {
            return renderJourneys(root)
        }
        if toolName == "station_departures", let departures = root["departures"]?.array {
            let station = placeName(root["station"]) ?? "駅"
            let lines = departures.compactMap { item -> String? in
                guard let value = item.object else { return nil }
                return [value["time"]?.string, value["line"]?.string, value["direction"]?.string]
                    .compactMap { $0 }.joined(separator: " ")
            }
            return (["\(withStationSuffix(station))の発車情報です。", ""] + lines).joined(separator: "\n")
        }
        if toolName == "get_station", let station = placeName(root["station"]) {
            return "\(withStationSuffix(station))の情報です。"
        }
        return pretty(decoded) ?? result.text
    }

    private static func renderJourneys(_ root: [String: JSONValue]) -> String {
        let journeys = root["journeys"]?.array?.compactMap(\.object)
            ?? root["options"]?.array?.compactMap { $0.object?["journey"]?.object }
            ?? []
        guard !journeys.isEmpty else { return emptyMessage }
        let origin = placeName(root["from"]) ?? "出発地"
        let destination = placeName(root["to"]) ?? "目的地"
        var sections = ["\(origin) → \(destination)"]
        if let date = root["date"]?.string, date.count == 8 {
            sections.append("\(date.prefix(4))/\(date.dropFirst(4).prefix(2))/\(date.suffix(2))")
        }

        for (index, journey) in journeys.prefix(3).enumerated() {
            var lines: [String] = []
            let departure = seconds(journey["departureSecs"]).map(time)
            let arrival = seconds(journey["arrivalSecs"]).map(time)
            let duration = seconds(journey["durationSecs"]).map(minutes)
            let transfers = journey["transferCount"]?.number.map { Int($0) }
            var heading = journeys.count > 1 ? "【候補\(index + 1)】" : "経路候補です。"
            if let departure, let arrival { heading += " \(departure)発 → \(arrival)着" }
            lines.append(heading)
            var summary: [String] = []
            if let duration { summary.append("所要\(duration)分") }
            if let fare = journey["fareYen"]?.number.map({ Int($0) }) { summary.append("\(fare)円") }
            if let transfers { summary.append("乗換\(transfers)回") }
            if !summary.isEmpty { lines.append(summary.joined(separator: " / ")) }

            for leg in journey["legs"]?.array?.compactMap(\.object) ?? [] {
                if let rendered = renderLeg(leg) { lines.append(rendered) }
            }
            sections.append(lines.joined(separator: "\n"))
        }
        return sections.joined(separator: "\n\n")
    }

    private static func renderLeg(_ leg: [String: JSONValue]) -> String? {
        let origin = placeName(leg["from"])
        let destination = placeName(leg["to"])
        guard origin != nil || destination != nil else { return nil }
        let departure = seconds(leg["departureSecs"]).map(time)
        let arrival = seconds(leg["arrivalSecs"]).map(time)
        let duration = {
            guard let start = seconds(leg["departureSecs"]), let end = seconds(leg["arrivalSecs"]), end >= start else { return nil as Int? }
            return minutes(end - start)
        }()
        let kind = leg["kind"]?.string?.lowercased()
        if kind == "walk" || kind == "walking" || kind == "foot" {
            let route = [origin, destination].compactMap { $0 }.joined(separator: " → ")
            return "徒歩 \(route)" + (duration.map { "（約\($0)分）" } ?? "")
        }
        var route = ""
        if let departure, let origin { route += "\(departure) \(origin)" }
        if !route.isEmpty { route += " → " }
        if let arrival { route += "\(arrival) " }
        if let destination { route += destination }
        var details: [String] = []
        if let line = leg["routeName"]?.string, !line.isEmpty { details.append(line) }
        if let headsign = leg["headsign"]?.string, !headsign.isEmpty { details.append("\(headsign)方面") }
        if !details.isEmpty { route += "（\(details.joined(separator: "・"))）" }
        return route
    }

    private static func placeName(_ value: JSONValue?) -> String? {
        value?.object?["name"]?.string
    }

    private static func seconds(_ value: JSONValue?) -> Int? {
        value?.number.map { Int($0) }
    }

    private static func time(_ seconds: Int) -> String {
        let value = ((seconds % 86_400) + 86_400) % 86_400
        return String(format: "%02d:%02d", value / 3_600, (value % 3_600) / 60)
    }

    private static func minutes(_ seconds: Int) -> Int {
        max(1, Int(ceil(Double(seconds) / 60.0)))
    }

    private static func withStationSuffix(_ value: String) -> String {
        value.hasSuffix("駅") ? value : value + "駅"
    }

    private static func pretty(_ value: JSONValue) -> String? {
        guard let data = try? JSONEncoder().encode(value),
              let object = try? JSONSerialization.jsonObject(with: data),
              let rendered = try? JSONSerialization.data(
                withJSONObject: object, options: [.prettyPrinted, .sortedKeys]
              ) else { return nil }
        return String(decoding: rendered, as: UTF8.self)
    }
}

struct TransitRoutePresentation {
    let origin: String
    let destination: String
    let date: String?
    let options: [TransitRouteOption]

    init?(result: MCPToolResult) {
        guard let root = result.decodedText?.object else { return nil }
        origin = root["from"]?.object?["name"]?.string ?? "出発地"
        destination = root["to"]?.object?["name"]?.string ?? "目的地"
        date = root["date"]?.string

        if let values = root["options"]?.array {
            options = values.enumerated().compactMap { index, value in
                TransitRouteOption(structured: value.object, fallbackRank: index + 1)
            }
        } else {
            options = (root["journeys"]?.array ?? []).enumerated().compactMap { index, value in
                TransitRouteOption(journey: value.object, rank: index + 1)
            }
        }
        guard !options.isEmpty else { return nil }
    }
}

struct TransitRouteOption: Identifiable {
    let id: String
    let rank: Int
    let recommended: Bool
    let selectedFor: String?
    let departureSeconds: Int?
    let arrivalSeconds: Int?
    let durationMinutes: Int?
    let transferCount: Int?
    let walkMinutes: Int?
    let legs: [TransitRouteLeg]
    let mapPoints: [TransitMapPoint]
    let mapSegments: [TransitMapSegment]

    init?(structured: [String: JSONValue]?, fallbackRank: Int) {
        guard let structured, let journey = structured["journey"]?.object else { return nil }
        let metrics = structured["metrics"]?.object ?? [:]
        let map = structured["map"]?.object ?? [:]
        id = structured["id"]?.string ?? "route-\(fallbackRank)"
        rank = Int(structured["rank"]?.number ?? Double(fallbackRank))
        recommended = structured["recommended"]?.bool ?? false
        selectedFor = structured["selectedFor"]?.string
        departureSeconds = Self.int(journey["departureSecs"])
        arrivalSeconds = Self.int(journey["arrivalSecs"])
        durationMinutes = Self.minutes(Self.int(metrics["durationSecs"]) ?? Self.int(journey["durationSecs"]))
        transferCount = Self.int(metrics["transferCount"]) ?? Self.int(journey["transferCount"])
        walkMinutes = Self.minutes(Self.int(metrics["walkSecs"]))
        legs = (journey["legs"]?.array ?? []).compactMap { TransitRouteLeg($0.object) }
        mapPoints = (map["points"]?.array ?? []).compactMap { TransitMapPoint($0.object) }
        mapSegments = (map["segments"]?.array ?? []).enumerated().compactMap {
            TransitMapSegment($0.element.object, index: $0.offset)
        }
    }

    init?(journey: [String: JSONValue]?, rank: Int) {
        guard let journey else { return nil }
        id = "journey-\(rank)"
        self.rank = rank
        recommended = rank == 1
        selectedFor = nil
        departureSeconds = Self.int(journey["departureSecs"])
        arrivalSeconds = Self.int(journey["arrivalSecs"])
        durationMinutes = Self.minutes(Self.int(journey["durationSecs"]))
        transferCount = Self.int(journey["transferCount"])
        let access = Self.int(journey["accessWalkSecs"]) ?? 0
        let egress = Self.int(journey["egressWalkSecs"]) ?? 0
        walkMinutes = Self.minutes(access + egress)
        legs = (journey["legs"]?.array ?? []).compactMap { TransitRouteLeg($0.object) }
        mapPoints = []
        mapSegments = []
    }

    private static func int(_ value: JSONValue?) -> Int? { value?.number.map(Int.init) }
    private static func minutes(_ seconds: Int?) -> Int? {
        seconds.map { max(1, Int(ceil(Double($0) / 60))) }
    }
}

struct TransitRouteLeg: Identifiable {
    let id = UUID()
    let kind: String
    let routeName: String?
    let colorHex: String?
    let headsign: String?
    let origin: String
    let destination: String
    let departureSeconds: Int?
    let arrivalSeconds: Int?
    let durationMinutes: Int?

    init?(_ value: [String: JSONValue]?) {
        guard let value else { return nil }
        kind = value["kind"]?.string ?? "transit"
        routeName = value["routeName"]?.string
        colorHex = value["color"]?.string
        headsign = value["headsign"]?.string
        origin = value["from"]?.object?["name"]?.string ?? ""
        destination = value["to"]?.object?["name"]?.string ?? ""
        departureSeconds = value["departureSecs"]?.number.map(Int.init)
        arrivalSeconds = value["arrivalSecs"]?.number.map(Int.init)
        if let departureSeconds, let arrivalSeconds, arrivalSeconds >= departureSeconds {
            durationMinutes = max(1, Int(ceil(Double(arrivalSeconds - departureSeconds) / 60)))
        } else {
            durationMinutes = nil
        }
    }
}

struct TransitCoordinate {
    let latitude: Double
    let longitude: Double

    init?(_ value: [String: JSONValue]?) {
        guard let latitude = value?["lat"]?.number,
              let longitude = value?["lon"]?.number else { return nil }
        self.latitude = latitude
        self.longitude = longitude
    }
}

struct TransitMapPoint: Identifiable {
    let id: String
    let name: String
    let role: String
    let coordinate: TransitCoordinate

    init?(_ value: [String: JSONValue]?) {
        guard let value, let coordinate = TransitCoordinate(value) else { return nil }
        id = value["id"]?.string ?? UUID().uuidString
        name = value["name"]?.string ?? ""
        role = value["role"]?.string ?? "stop"
        self.coordinate = coordinate
    }
}

struct TransitMapSegment: Identifiable {
    let id: String
    let kind: String
    let routeName: String?
    let colorHex: String?
    let coordinates: [TransitCoordinate]

    init?(_ value: [String: JSONValue]?, index: Int) {
        guard let value else { return nil }
        coordinates = (value["polyline"]?.array ?? []).compactMap {
            TransitCoordinate($0.object)
        }
        guard coordinates.count >= 2 else { return nil }
        kind = value["kind"]?.string ?? "transit"
        routeName = value["routeName"]?.string
        colorHex = value["color"]?.string
        id = "\(index)-\(value["fromPointId"]?.string ?? "from")-\(value["toPointId"]?.string ?? "to")"
    }
}

/// Runtime-only normalization and value binding kept in lockstep with
/// transit_functiongemma.japanese's production (semantic_fallback=false) path.
enum JapaneseRuntimeCompatibility {
    private static let tokyo = TimeZone(identifier: "Asia/Tokyo")!
    private static let repairableDateTimeTools: Set<String> = [
        "station_departures", "plan_journey", "plan_route_map", "resolve_route_request"
    ]

    static func normalizeSurface(_ text: String) -> String {
        let normalized = text.precomposedStringWithCompatibilityMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized.replacingOccurrences(
            of: #"\s+"#, with: " ", options: .regularExpression
        )
    }

    static func repair(
        _ call: LocalToolCall, from text: String, referenceDate: Date
    ) -> LocalToolCall {
        let original = normalizeSurface(text)
        var arguments = call.arguments

        if case .number(let currentLat) = arguments["lat"],
           case .number(let currentLon) = arguments["lon"],
           let pair = singleCoordinatePair(in: original),
           abs(currentLat - pair.lat) <= 0.05,
           abs(currentLon - pair.lon) <= 0.05 {
            arguments["lat"] = .number(pair.lat)
            arguments["lon"] = .number(pair.lon)
        }

        if case .string = arguments["id"] {
            let ids = matches(
                #"[A-Za-z0-9][A-Za-z0-9._-]*(?::[A-Za-z0-9._:-]+)+"#, in: original
            ).map { $0.hasSuffix(":") ? String($0.dropLast()) : $0 }
            let unique = Set(ids)
            if unique.count == 1, let id = unique.first { arguments["id"] = .string(id) }
        }

        if repairableDateTimeTools.contains(call.name) {
            if let date = normalizedDate(from: original, referenceDate: referenceDate) {
                arguments["date"] = .string(date)
            }
            if let time = normalizedTime(from: original) {
                arguments["time"] = .string(time)
            }
        }
        return LocalToolCall(name: call.name, arguments: arguments)
    }

    static func normalizedDate(from text: String, referenceDate: Date) -> String? {
        let value = normalizeSurface(text)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = tokyo
        let base = calendar.dateComponents([.year], from: referenceDate)
        if let captures = firstCaptures(
            #"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日"#, in: value, count: 3
        ), let month = Int(captures[1]), let day = Int(captures[2]) {
            let year = Int(captures[0]) ?? base.year ?? 1970
            return String(format: "%04d%02d%02d", year, month, day)
        }
        if let captures = firstCaptures(
            #"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)"#, in: value, count: 2
        ), let month = Int(captures[0]), let day = Int(captures[1]) {
            return String(format: "%04d%02d%02d", base.year ?? 1970, month, day)
        }
        let offset: Int?
        if value.contains("明日") { offset = 1 }
        else if ["今日", "本日", "終電", "最終列車", "始発"].contains(where: value.contains) { offset = 0 }
        else { offset = nil }
        guard let offset, let date = calendar.date(byAdding: .day, value: offset, to: referenceDate) else {
            return nil
        }
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = tokyo
        formatter.dateFormat = "yyyyMMdd"
        return formatter.string(from: date)
    }

    static func normalizedTime(from text: String) -> String? {
        let value = normalizeSurface(text)
        let regex = try! NSRegularExpression(
            pattern: #"(?<!\d)(\d{1,2})(?::(\d{2})|時(?:(\d{1,2})分|半)?)(?!\d)"#
        )
        guard let match = regex.firstMatch(
            in: value, range: NSRange(value.startIndex..., in: value)
        ), let hourRange = Range(match.range(at: 1), in: value) else { return nil }
        var hour = Int(value[hourRange]) ?? 0
        var minute = 0
        if let range = Range(match.range(at: 2), in: value) { minute = Int(value[range]) ?? 0 }
        else if let range = Range(match.range(at: 3), in: value) { minute = Int(value[range]) ?? 0 }
        else if let whole = Range(match.range, in: value), value[whole].contains("半") { minute = 30 }

        let matchStart = Range(match.range, in: value)!.lowerBound
        let prefixStart = value.index(matchStart, offsetBy: -4, limitedBy: value.startIndex) ?? value.startIndex
        let prefix = value[prefixStart..<matchStart]
        if prefix.contains("午後"), hour < 12 { hour += 12 }
        else if prefix.contains("午前"), hour == 12 { hour = 0 }
        guard hour < 24, minute < 60 else { return nil }
        return String(format: "%02d:%02d", hour, minute)
    }

    private static func singleCoordinatePair(in text: String) -> (lat: Double, lon: Double)? {
        let patterns = [
            #"北緯\s*(-?\d+(?:\.\d+)?)\s*[,、 ]*\s*東経\s*(-?\d+(?:\.\d+)?)"#,
            #"緯度\s*(-?\d+(?:\.\d+)?)\s*[,、 ]*\s*経度\s*(-?\d+(?:\.\d+)?)"#,
            #"(-?\d{1,2}(?:\.\d+)?)\s*[,、/]\s*(-?\d{2,3}(?:\.\d+)?)"#,
            #"(?i)lat\s*=\s*(-?\d+(?:\.\d+)?)\s+lon\s*=\s*(-?\d+(?:\.\d+)?)"#
        ]
        var pairs: [(Double, Double)] = []
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
                guard match.numberOfRanges >= 3,
                      let latRange = Range(match.range(at: 1), in: text),
                      let lonRange = Range(match.range(at: 2), in: text),
                      let lat = Double(text[latRange]), let lon = Double(text[lonRange]) else { continue }
                if !pairs.contains(where: { $0.0 == lat && $0.1 == lon }) { pairs.append((lat, lon)) }
            }
        }
        guard pairs.count == 1 else { return nil }
        return pairs[0]
    }

    private static func matches(_ pattern: String, in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        }
    }

    private static func firstCaptures(
        _ pattern: String, in text: String, count: Int
    ) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else {
            return nil
        }
        return (1...count).map { index in
            Range(match.range(at: index), in: text).map { String(text[$0]) } ?? ""
        }
    }

#if DEBUG
    static func verifyFixtures() {
        let formatter = ISO8601DateFormatter()
        let base = formatter.date(from: "2026-07-11T10:00:00Z")!
        assert(normalizeSurface("  東京\u{3000} 駅  ") == "東京 駅")
        assert(normalizedDate(from: "明日の始発", referenceDate: base) == "20260712")
        assert(normalizedDate(from: "8月3日", referenceDate: base) == "20260803")
        assert(normalizedTime(from: "午後3時半") == "15:30")
        let repaired = repair(
            LocalToolCall(name: "get_station", arguments: ["id": .string("wrong:id")]),
            from: "odpt.Station:JR-East.Yamanote.Tokyo を取得", referenceDate: base
        )
        assert(repaired.arguments["id"] == .string("odpt.Station:JR-East.Yamanote.Tokyo"))
    }
#endif
}
