import Foundation

enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let decoded = try? value.decode(Bool.self) { self = .bool(decoded) }
        else if let decoded = try? value.decode(Double.self) { self = .number(decoded) }
        else if let decoded = try? value.decode(String.self) { self = .string(decoded) }
        else if let decoded = try? value.decode([String: JSONValue].self) { self = .object(decoded) }
        else { self = .array(try value.decode([JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .string(let item): try value.encode(item)
        case .number(let item): try value.encode(item)
        case .bool(let item): try value.encode(item)
        case .object(let item): try value.encode(item)
        case .array(let item): try value.encode(item)
        case .null: try value.encodeNil()
        }
    }

    var string: String? { if case .string(let value) = self { value } else { nil } }
    var number: Double? { if case .number(let value) = self { value } else { nil } }
    var bool: Bool? { if case .bool(let value) = self { value } else { nil } }
    var array: [JSONValue]? { if case .array(let value) = self { value } else { nil } }
    var object: [String: JSONValue]? { if case .object(let value) = self { value } else { nil } }
}

struct LocalToolCall: Sendable, Equatable {
    let name: String
    let arguments: [String: JSONValue]
}

enum ToolCallParserError: Error { case invalidOutput }

struct ToolCallParser {
    static func parse(_ output: String) throws -> LocalToolCall? {
        let start = "<start_function_call>"
        let end = "<end_function_call>"
        guard let startRange = output.range(of: start) else { return nil }
        guard let endRange = output.range(of: end, range: startRange.upperBound..<output.endIndex) else {
            throw ToolCallParserError.invalidOutput
        }
        let outside = output[..<startRange.lowerBound] + output[endRange.upperBound...]
        guard outside.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ToolCallParserError.invalidOutput
        }
        var body = String(output[startRange.upperBound..<endRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard body.hasPrefix("call:"), let brace = body.firstIndex(of: "{") else {
            throw ToolCallParserError.invalidOutput
        }
        let name = String(body[body.index(body.startIndex, offsetBy: 5)..<brace])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard name.range(of: #"^[A-Za-z_][A-Za-z0-9_]*$"#, options: .regularExpression) != nil else {
            throw ToolCallParserError.invalidOutput
        }
        body = String(body[brace...])
        var parser = ValueParser(body)
        guard case .object(let arguments) = try parser.value(), parser.isAtEnd, !name.isEmpty else {
            throw ToolCallParserError.invalidOutput
        }
        return LocalToolCall(name: name, arguments: arguments)
    }

    private struct ValueParser {
        let characters: [Character]
        var index = 0
        init(_ text: String) { characters = Array(text) }
        var isAtEnd: Bool { skipIndex(index) == characters.count }

        mutating func value() throws -> JSONValue {
            skipWhitespace()
            guard index < characters.count else { throw ToolCallParserError.invalidOutput }
            if peek("<escape>") { return .string(try escaped()) }
            if characters[index] == "{" { return .object(try object()) }
            if characters[index] == "[" { return .array(try array()) }
            let token = bare(stoppingAt: [",", "}", "]"]).trimmingCharacters(in: .whitespaces)
            if token == "true" { return .bool(true) }
            if token == "false" { return .bool(false) }
            if token == "null" || token == "None" { return .null }
            if let number = Double(token) { return .number(number) }
            return .string(token.trimmingCharacters(in: CharacterSet(charactersIn: "\"'")))
        }

        mutating func object() throws -> [String: JSONValue] {
            try expect("{")
            var result: [String: JSONValue] = [:]
            skipWhitespace()
            if take("}") { return result }
            while true {
                let key = bare(stoppingAt: [":"]).trimmingCharacters(in: .whitespaces)
                guard !key.isEmpty else { throw ToolCallParserError.invalidOutput }
                try expect(":")
                result[key] = try value()
                skipWhitespace()
                if take("}") { return result }
                try expect(",")
            }
        }

        mutating func array() throws -> [JSONValue] {
            try expect("[")
            var result: [JSONValue] = []
            skipWhitespace()
            if take("]") { return result }
            while true {
                result.append(try value())
                skipWhitespace()
                if take("]") { return result }
                try expect(",")
            }
        }

        mutating func escaped() throws -> String {
            try expect("<escape>")
            let start = index
            while index < characters.count, !peek("<escape>") { index += 1 }
            guard index < characters.count else { throw ToolCallParserError.invalidOutput }
            let result = String(characters[start..<index])
            try expect("<escape>")
            return result
        }

        mutating func bare(stoppingAt stops: Set<Character>) -> String {
            let start = index
            while index < characters.count, !stops.contains(characters[index]) { index += 1 }
            return String(characters[start..<index])
        }

        mutating func expect(_ text: String) throws {
            guard take(text) else { throw ToolCallParserError.invalidOutput }
        }
        mutating func take(_ text: String) -> Bool {
            skipWhitespace()
            guard peek(text) else { return false }
            index += text.count
            return true
        }
        func peek(_ text: String) -> Bool {
            let token = Array(text)
            guard index + token.count <= characters.count else { return false }
            return Array(characters[index..<(index + token.count)]) == token
        }
        mutating func skipWhitespace() {
            while index < characters.count, characters[index].isWhitespace { index += 1 }
        }
        func skipIndex(_ original: Int) -> Int {
            var next = original
            while next < characters.count, characters[next].isWhitespace { next += 1 }
            return next
        }
    }
}

struct MCPToolResult: Sendable {
    let envelope: [String: JSONValue]
    let text: String
    let decodedText: JSONValue?
}

actor TransitMCPClient {
    private let endpoint = URL(string: "https://api.transit.ls8h.com/mcp")!
    private var requestID = 0
    private var sessionID: String?
    private var initialized = false

    func callTool(name: String, arguments: [String: JSONValue]) async throws -> MCPToolResult {
        if !initialized {
            _ = try await request(method: "initialize", params: .object([
                "protocolVersion": .string("2025-03-26"),
                "capabilities": .object([:]),
                "clientInfo": .object(["name": .string("tentetsu-ios"), "version": .string("0.1.0")])
            ]))
            try await notify(method: "notifications/initialized", params: .object([:]))
            initialized = true
        }
        let envelope = try await request(method: "tools/call", params: .object([
            "name": .string(name), "arguments": .object(arguments)
        ]))
        let contents = envelope["result"]?.object?["content"]?.array ?? []
        let text = contents.compactMap { item -> String? in
            guard let object = item.object, object["type"]?.string == "text" else { return nil }
            return object["text"]?.string
        }.joined(separator: "\n")
        let structured = envelope["result"]?.object?["structuredContent"]
        let decoded = structured ?? text.data(using: .utf8).flatMap {
            try? JSONDecoder().decode(JSONValue.self, from: $0)
        }
        return MCPToolResult(envelope: envelope, text: text, decodedText: decoded)
    }

    private func notify(method: String, params: JSONValue) async throws {
        let payload: JSONValue = .object([
            "jsonrpc": .string("2.0"), "method": .string(method), "params": params
        ])
        var request = configuredRequest()
        request.httpBody = try JSONEncoder().encode(payload)
        debugLog("MCP -> \(method)")
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        if let value = http.value(forHTTPHeaderField: "mcp-session-id") { sessionID = value }
        debugLog("MCP <- \(method) HTTP \(http.statusCode)")
    }

    private func request(method: String, params: JSONValue) async throws -> [String: JSONValue] {
        requestID += 1
        let payload: JSONValue = .object([
            "jsonrpc": .string("2.0"), "id": .number(Double(requestID)),
            "method": .string(method), "params": params
        ])
        var request = configuredRequest()
        request.httpBody = try JSONEncoder().encode(payload)
        debugLog("MCP -> \(method)")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        if let value = http.value(forHTTPHeaderField: "mcp-session-id") { sessionID = value }
        debugLog("MCP <- \(method) HTTP \(http.statusCode)")
        let contentType = http.value(forHTTPHeaderField: "content-type") ?? ""
        let jsonData: Data
        if contentType.contains("text/event-stream") {
            let event = String(decoding: data, as: UTF8.self).split(separator: "\n")
                .last(where: { $0.hasPrefix("data:") })?.dropFirst(5)
                .trimmingCharacters(in: .whitespaces) ?? ""
            jsonData = Data(event.utf8)
        } else { jsonData = data }
        guard case .object(let result) = try JSONDecoder().decode(JSONValue.self, from: jsonData), result["error"] == nil else {
            throw URLError(.cannotParseResponse)
        }
        return result
    }

    private func configuredRequest() -> URLRequest {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        if let sessionID { request.setValue(sessionID, forHTTPHeaderField: "Mcp-Session-Id") }
        return request
    }

    private func debugLog(_ message: String) {
#if DEBUG
        print(message)
#endif
    }
}
