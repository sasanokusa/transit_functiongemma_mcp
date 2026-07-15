import SwiftUI
import MapKit

struct ContentView: View {
    @StateObject private var model = AppModel()
    @Environment(\.scenePhase) private var scenePhase
    @FocusState private var queryIsFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    queryCard
                    statusCard
                    if let route = model.routePresentation {
                        RouteResultsView(
                            presentation: route,
                            isMapLoading: model.isMapLoading,
                            mapLoadMessage: model.mapLoadMessage
                        )
                            .id("\(route.origin)-\(route.destination)-\(route.date ?? "")")
                        DisclosureGroup("テキスト表示") {
                            Text(model.answer)
                                .font(.system(.body, design: .rounded))
                                .textSelection(.enabled)
                                .padding(.top, 8)
                        }
                        .padding()
                        .background(.background, in: RoundedRectangle(cornerRadius: 18))
                    } else if !model.answer.isEmpty {
                        answerCard
                    }
                    if model.showsDebugOutput, !model.modelOutput.isEmpty { debugCard }
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .background {
                Color(.systemGroupedBackground)
                    .onTapGesture { queryIsFocused = false }
            }
            .navigationTitle("転轍")
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("完了") { queryIsFocused = false }
                }
            }
            .onAppear { model.startVoiceFromShortcutIfNeeded() }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { model.startVoiceFromShortcutIfNeeded() }
            }
            .onReceive(NotificationCenter.default.publisher(for: .startVoiceSearchRequested)) { _ in
                model.startVoiceFromShortcutIfNeeded()
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("端末内AI × Transit MCP", systemImage: "tram.fill")
                .font(.headline)
            Text("入力の理解はiPhone内、経路の事実だけをTransit MCPから取得します。")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var queryCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextEditor(text: $model.query)
                .focused($queryIsFocused)
                .frame(minHeight: 88)
                .scrollContentBackground(.hidden)
                .padding(8)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                .overlay(alignment: .topLeading) {
                    if model.query.isEmpty {
                        Text("〇〇駅から〇〇駅")
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 16)
                            .allowsHitTesting(false)
                    }
                }
            HStack {
                VoiceButton(speech: model.speech) { transcript in
                    model.query = transcript
                }
                .simultaneousGesture(
                    TapGesture().onEnded { queryIsFocused = false }
                )
                Spacer()
                Button {
                    queryIsFocused = false
                    Task { await model.search() }
                } label: {
                    if model.isBusy { ProgressView().controlSize(.small) }
                    else { Label("検索", systemImage: "arrow.right.circle.fill") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(
                    model.isBusy || !model.modelReady
                        || model.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                )
            }
            Toggle(isOn: $model.prefersMap) {
                Label("地図付きで表示", systemImage: "map.fill")
                    .font(.subheadline)
            }
            .tint(.blue)
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 18))
    }

    private var statusCard: some View {
        HStack(spacing: 10) {
            Circle().fill(statusColor).frame(width: 8, height: 8)
            Text(model.status).font(.footnote)
            Spacer()
            if let ms = model.elapsedMilliseconds { Text(String(format: "%.0f ms", ms)).font(.caption.monospacedDigit()) }
        }
        .foregroundStyle(.secondary)
    }

    private var statusColor: Color {
        if !model.modelReady { return .orange }
        if model.isBusy || model.isMapLoading { return .blue }
        if model.status == "エラー" { return .red }
        if model.status.contains("確認") { return .orange }
        return .green
    }

    private var answerCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("結果", systemImage: "map")
                .font(.headline)
            Text(model.answer)
                .font(.system(.body, design: .rounded))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 18))
    }

    private var debugCard: some View {
        DisclosureGroup("モデル出力") {
            Text(model.modelOutput)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 8)
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 18))
    }
}

private struct RouteResultsView: View {
    let presentation: TransitRoutePresentation
    let isMapLoading: Bool
    let mapLoadMessage: String?
    @State private var selectedID: String

    init(
        presentation: TransitRoutePresentation,
        isMapLoading: Bool,
        mapLoadMessage: String?
    ) {
        self.presentation = presentation
        self.isMapLoading = isMapLoading
        self.mapLoadMessage = mapLoadMessage
        let initial = presentation.options.first(where: \.recommended) ?? presentation.options[0]
        _selectedID = State(initialValue: initial.id)
    }

    private var selected: TransitRouteOption {
        presentation.options.first(where: { $0.id == selectedID }) ?? presentation.options[0]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(presentation.origin) → \(presentation.destination)")
                        .font(.title3.bold())
                    if let date = presentation.date {
                        Text(displayDate(date)).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
                    .font(.title2).foregroundStyle(.blue)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(presentation.options) { option in
                        RouteOptionButton(option: option, selected: option.id == selectedID) {
                            withAnimation(.snappy) { selectedID = option.id }
                        }
                    }
                }
            }

            if !selected.mapSegments.isEmpty {
                TransitMapView(option: selected)
                    .frame(height: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay(alignment: .bottomTrailing) {
                        Text("© OpenStreetMap contributors")
                            .font(.system(size: 8)).padding(4)
                            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 4))
                            .padding(6)
                    }
            } else if isMapLoading {
                HStack(spacing: 12) {
                    ProgressView()
                    VStack(alignment: .leading, spacing: 3) {
                        Text("地図を読み込み中…").font(.subheadline.bold())
                        Text("経路案内は先に確認できます。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, minHeight: 76, alignment: .leading)
                .padding(.horizontal, 14)
                .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
            } else if let mapLoadMessage {
                Label(mapLoadMessage, systemImage: "map.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
            }

            HStack(spacing: 14) {
                MetricPill(icon: "clock", text: selected.durationMinutes.map { "\($0)分" } ?? "—")
                MetricPill(icon: "arrow.triangle.swap", text: selected.transferCount.map { "乗換\($0)回" } ?? "—")
                MetricPill(icon: "figure.walk", text: selected.walkMinutes.map { "徒歩\($0)分" } ?? "—")
            }

            RouteTimelineView(legs: selected.legs)
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 18))
    }

    private func displayDate(_ date: String) -> String {
        guard date.count == 8 else { return date }
        return "\(date.prefix(4))年\(date.dropFirst(4).prefix(2))月\(date.suffix(2))日"
    }
}

private struct RouteOptionButton: View {
    let option: TransitRouteOption
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 4) {
                    Text(option.recommended ? "おすすめ" : "候補\(option.rank)")
                        .font(.caption.bold())
                    if option.recommended { Image(systemName: "star.fill").font(.caption2) }
                }
                Text(timeRange).font(.subheadline.monospacedDigit().bold())
                Text(option.durationMinutes.map { "所要\($0)分" } ?? "所要時間—")
                    .font(.caption).foregroundStyle(selected ? .white.opacity(0.85) : .secondary)
            }
            .frame(minWidth: 116, alignment: .leading)
            .padding(10)
            .foregroundStyle(selected ? .white : .primary)
            .background(selected ? Color.blue : Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 13))
        }
        .buttonStyle(.plain)
    }

    private var timeRange: String {
        guard let departure = option.departureSeconds, let arrival = option.arrivalSeconds else { return "時刻—" }
        return "\(clock(departure)) → \(clock(arrival))"
    }
}

private struct MetricPill: View {
    let icon: String
    let text: String
    var body: some View {
        Label(text, systemImage: icon)
            .font(.caption.bold())
            .padding(.horizontal, 9).padding(.vertical, 6)
            .background(Color(.secondarySystemGroupedBackground), in: Capsule())
    }
}

private struct RouteTimelineView: View {
    let legs: [TransitRouteLeg]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(legs.enumerated()), id: \.element.id) { index, leg in
                if index == 0 {
                    TimelineStationRow(
                        time: leg.departureSeconds,
                        name: leg.origin,
                        color: legColor(leg),
                        emphasized: true
                    )
                } else if index > 0, !isContinuous(legs[index - 1], leg) {
                    let previous = legs[index - 1]
                    TimelineWaitRow(previous: previous, next: leg)
                    TimelineStationRow(
                        time: leg.departureSeconds,
                        name: leg.origin,
                        color: legColor(leg),
                        emphasized: true
                    )
                }

                TimelineSegmentRow(leg: leg, color: legColor(leg))

                let next: TransitRouteLeg? = index + 1 < legs.count ? legs[index + 1] : nil
                TimelineStationRow(
                    time: leg.arrivalSeconds,
                    name: leg.destination,
                    color: next.map(legColor) ?? legColor(leg),
                    emphasized: next == nil
                )
            }
        }
    }

    private func isContinuous(_ previous: TransitRouteLeg, _ next: TransitRouteLeg) -> Bool {
        previous.destination == next.origin && previous.arrivalSeconds == next.departureSeconds
    }

    private func legColor(_ leg: TransitRouteLeg) -> Color {
        leg.kind == "walk" ? .secondary : Color(hex: leg.colorHex) ?? .blue
    }
}

private struct TimelineStationRow: View {
    let time: Int?
    let name: String
    let color: Color
    let emphasized: Bool

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                // Rails intentionally overlap nodes. This opaque knockout always
                // erases the rail first, so no connector can show through a node.
                Circle()
                    .fill(Color(.systemBackground))
                    .frame(width: 26, height: 26)
                Circle()
                    .fill(Color(.systemBackground))
                    .overlay {
                        Circle()
                            .strokeBorder(color, lineWidth: emphasized ? 5 : 4)
                    }
                    .frame(width: 20, height: 20)
            }
            .frame(width: 20, height: 20)
            .compositingGroup()
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(time.map(clock) ?? "--:--")
                    .font(.body.monospacedDigit().weight(emphasized ? .bold : .regular))
                    .frame(width: 52, alignment: .leading)
                Text(name)
                    .font(.body.weight(emphasized ? .bold : .regular))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .zIndex(2)
    }
}

private struct TimelineSegmentRow: View {
    let leg: TransitRouteLeg
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Color.clear.frame(width: 20)
            VStack(alignment: .leading, spacing: 5) {
                if leg.kind == "walk" {
                    Label(
                        "徒歩" + (leg.durationMinutes.map { " 約\($0)分" } ?? ""),
                        systemImage: "figure.walk"
                    )
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                } else {
                    Text(leg.routeName ?? "鉄道")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 9).padding(.vertical, 4)
                        .background(color, in: Capsule())
                    if let headsign = leg.headsign, !headsign.isEmpty {
                        Text("\(headsign)方面")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .overlay(alignment: .leading) {
            GeometryReader { proxy in
                Rectangle()
                    .fill(color)
                    .frame(width: leg.kind == "walk" ? 4 : 7, height: proxy.size.height + 2)
                    .offset(x: leg.kind == "walk" ? 8 : 6.5, y: -1)
            }
            .allowsHitTesting(false)
        }
        .zIndex(0)
    }
}

private struct TimelineWaitRow: View {
    let previous: TransitRouteLeg
    let next: TransitRouteLeg

    private var waitMinutes: Int? {
        guard let arrival = previous.arrivalSeconds,
              let departure = next.departureSeconds,
              departure > arrival else { return nil }
        return max(1, Int(ceil(Double(departure - arrival) / 60)))
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Color.clear.frame(width: 20)
            Label(
                previous.destination == next.origin
                    ? "待ち時間" + (waitMinutes.map { " 約\($0)分" } ?? "")
                    : "乗換移動",
                systemImage: "hourglass"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.vertical, 7)
        }
        .overlay(alignment: .leading) {
            GeometryReader { proxy in
                Rectangle()
                    .fill(Color.secondary.opacity(0.65))
                    .frame(width: 4, height: proxy.size.height + 2)
                    .offset(x: 8, y: -1)
            }
            .allowsHitTesting(false)
        }
        .zIndex(0)
    }
}

private struct TransitMapView: View {
    let option: TransitRouteOption

    var body: some View {
        Map(initialPosition: .region(region)) {
            ForEach(option.mapSegments) { segment in
                MapPolyline(coordinates: segment.coordinates.map(\.clLocation))
                    .stroke(
                        segment.kind == "walk" ? Color.secondary : Color(hex: segment.colorHex) ?? .blue,
                        style: StrokeStyle(
                            lineWidth: segment.kind == "walk" ? 4 : 7,
                            lineCap: .round,
                            lineJoin: .round,
                            dash: segment.kind == "walk" ? [5, 5] : []
                        )
                    )
            }
            ForEach(option.mapPoints) { point in
                Annotation(point.name, coordinate: point.coordinate.clLocation, anchor: .bottom) {
                    VStack(spacing: 2) {
                        Image(systemName: point.role == "origin" ? "play.circle.fill" : point.role == "destination" ? "flag.circle.fill" : "circle.fill")
                            .font(point.role == "transfer" ? .caption : .title3)
                            .foregroundStyle(point.role == "destination" ? .red : .blue)
                        if point.role == "origin" || point.role == "destination" {
                            Text(point.name).font(.caption2.bold()).padding(3)
                                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }
            }
        }
        .mapStyle(.standard(elevation: .flat, pointsOfInterest: .excludingAll, showsTraffic: false))
    }

    private var region: MKCoordinateRegion {
        let coordinates = option.mapSegments.flatMap(\.coordinates)
        guard let first = coordinates.first else {
            return MKCoordinateRegion(center: .init(latitude: 35.6812, longitude: 139.7671), span: .init(latitudeDelta: 0.1, longitudeDelta: 0.1))
        }
        let minLat = coordinates.map(\.latitude).min() ?? first.latitude
        let maxLat = coordinates.map(\.latitude).max() ?? first.latitude
        let minLon = coordinates.map(\.longitude).min() ?? first.longitude
        let maxLon = coordinates.map(\.longitude).max() ?? first.longitude
        return MKCoordinateRegion(
            center: .init(latitude: (minLat + maxLat) / 2, longitude: (minLon + maxLon) / 2),
            span: .init(
                latitudeDelta: max(0.01, (maxLat - minLat) * 1.35),
                longitudeDelta: max(0.01, (maxLon - minLon) * 1.35)
            )
        )
    }
}

private extension TransitCoordinate {
    var clLocation: CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }
}

private extension Color {
    init?(hex: String?) {
        guard let hex else { return nil }
        let cleaned = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        guard cleaned.count == 6, let value = UInt64(cleaned, radix: 16) else { return nil }
        self.init(
            red: Double((value >> 16) & 0xff) / 255,
            green: Double((value >> 8) & 0xff) / 255,
            blue: Double(value & 0xff) / 255
        )
    }
}

private func clock(_ seconds: Int) -> String {
    let value = ((seconds % 86_400) + 86_400) % 86_400
    return String(format: "%02d:%02d", value / 3_600, (value % 3_600) / 60)
}

private struct VoiceButton: View {
    @ObservedObject var speech: SpeechInput
    let onTranscript: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button {
                speech.toggle()
            } label: {
                Label(
                    speech.isRecording ? "聞き取り中…" : "音声入力",
                    systemImage: speech.isRecording ? "stop.circle.fill" : "mic.circle.fill"
                )
            }
            .buttonStyle(.bordered)
            .tint(speech.isRecording ? .red : .accentColor)
            if let error = speech.errorMessage {
                Text(error).font(.caption).foregroundStyle(.red)
            }
        }
        .onChange(of: speech.transcript) { _, value in
            if !value.isEmpty { onTranscript(value) }
        }
    }
}

#Preview { ContentView() }
