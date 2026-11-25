//
//  CSVImportView.swift
//  Core Motion Data Collector iOS
//
//  Lets the user pick a local CSV file from the Files app, read it safely
//  through a security-scoped URL, and IMMEDIATELY analyze it: if the file
//  is a Sensor Logger motion export, the SenseWhenNeeded policy is replayed
//  offline (same pipeline as process_recording.py) and the activity
//  timeline, segments and per-activity breakdown are shown. Reading and
//  analysis are one flow — no separate "analyze" step.
//
//  Non-motion CSVs fall back to a plain table preview.
//

import SwiftUI
import UniformTypeIdentifiers

// MARK: - Model

struct CSVTable {
    let fileName: String
    let headers: [String]
    let rows: [[String]]
}

enum CSVImportError: LocalizedError {
    case emptyFile
    case unsupportedEncoding
    case malformedCSV

    var errorDescription: String? {
        switch self {
        case .emptyFile:
            return "The selected file is empty."
        case .unsupportedEncoding:
            return "Could not read the file. Please use UTF-8 or UTF-16 text CSV."
        case .malformedCSV:
            return "Could not parse the CSV. Check for a valid header row."
        }
    }
}

// MARK: - Parser

/// Small RFC-4180-ish parser: handles quoted fields, escaped quotes (""),
/// commas and newlines inside quotes, plus LF / CRLF line endings.
enum CSVParser {
    static func parse(text: String) throws -> (headers: [String], rows: [[String]]) {
        var cleaned = text
        if cleaned.first == "\u{FEFF}" {
            cleaned.removeFirst()
        }

        let records = parseRecords(cleaned)
        guard let headerRecord = records.first, !headerRecord.isEmpty else {
            throw CSVImportError.emptyFile
        }

        let headers = headerRecord.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard headers.contains(where: { !$0.isEmpty }) else {
            throw CSVImportError.malformedCSV
        }

        let rows = Array(records.dropFirst()).filter { record in
            !record.allSatisfy { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        }

        return (headers, rows)
    }

    private static func parseRecords(_ text: String) -> [[String]] {
        let characters = Array(text)
        var records: [[String]] = []
        var record: [String] = []
        var field = ""
        var inQuotes = false
        var index = 0

        while index < characters.count {
            let character = characters[index]

            if inQuotes {
                if character == "\"" {
                    if index + 1 < characters.count && characters[index + 1] == "\"" {
                        field.append("\"")
                        index += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(character)
                }
                index += 1
                continue
            }

            switch character {
            case "\"":
                inQuotes = true
                index += 1
            case ",":
                record.append(field)
                field = ""
                index += 1
            case "\n":
                record.append(field)
                records.append(record)
                record = []
                field = ""
                index += 1
            case "\r":
                if index + 1 < characters.count && characters[index + 1] == "\n" {
                    index += 1
                }
                record.append(field)
                records.append(record)
                record = []
                field = ""
                index += 1
            default:
                field.append(character)
                index += 1
            }
        }

        if !field.isEmpty || !record.isEmpty {
            record.append(field)
            records.append(record)
        }

        return records
    }
}

// MARK: - Analysis state

/// One "read + analyze" flow.
struct CSVAnalysisState {
    var table: CSVTable?
    var readError: String?
    var isAnalyzing = false
    var analysisError: String?
    var result: SWNOfflineResult?
    var modelsAvailable = false
}

// MARK: - View

struct CSVImportView: View {
    private static let previewRowLimit = 100

    @State private var showingImporter = false
    @State private var state = CSVAnalysisState()
    @State private var showRawTable = false

    var body: some View {
        NavigationStack {
            Group {
                if state.table == nil && state.readError == nil {
                    ContentUnavailableView {
                        Label("选择 CSV 文件", systemImage: "doc.text.magnifyingglass")
                    } description: {
                        Text("从“文件”App 选择一段运动数据（Sensor Logger 导出的 Motion.csv），\n读取后会立即自动完成活动识别分析。")
                    } actions: {
                        Button("选择文件") { showingImporter = true }
                            .buttonStyle(.borderedProminent)
                    }
                } else if let readError = state.readError, state.table == nil {
                    ContentUnavailableView {
                        Label("无法读取 CSV", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(readError)
                    } actions: {
                        Button("选择其他文件") { showingImporter = true }
                            .buttonStyle(.borderedProminent)
                    }
                } else if let table = state.table {
                    analysisView(table)
                }
            }
            .navigationTitle("CSV 活动分析")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showingImporter = true
                    } label: {
                        Label("打开", systemImage: "folder")
                    }
                }
            }
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.commaSeparatedText, .plainText],
                allowsMultipleSelection: false
            ) { result in
                handleImport(result)
            }
        }
    }

    // MARK: main view

    @ViewBuilder
    private func analysisView(_ table: CSVTable) -> some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 4) {
                    Text(table.fileName)
                        .font(.headline)
                    Text("\(table.rows.count) 行数据 · \(table.headers.count) 列")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
            }

            if state.isAnalyzing {
                Section {
                    HStack(spacing: 12) {
                        ProgressView()
                        Text("正在回放事件驱动策略（分类 + 事件检测）…")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }
            } else if let analysisError = state.analysisError {
                Section {
                    HStack(spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(analysisError)
                            .font(.callout)
                    }
                }
            } else if let result = state.result {
                diagnosticsSection(result)
                timelineSection(result)
                segmentsSection(result)
                breakdownSection(result)
            }

            Section {
                DisclosureGroup(isOn: $showRawTable) {
                    rawTablePreview(table)
                } label: {
                    Label("原始数据表", systemImage: "tablecells")
                }
            }
        }
    }

    // MARK: analysis sections

    /// Surfaces silent prediction failures so "all sitting" results are never
    /// mistaken for real ones.
    @ViewBuilder
    private func diagnosticsSection(_ result: SWNOfflineResult) -> some View {
        if !result.modelsLoaded {
            Section {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.octagon.fill")
                        .foregroundStyle(.red)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("未加载 Core ML 模型")
                            .font(.callout.weight(.semibold))
                        Text("下面的结果是默认占位（全部显示为静坐），不是真实识别。请先在 Mac 上运行 swn/coreml_export.py，把生成的 3 个 .mlmodel 加入工程后重新构建。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        } else if result.predictionFailures > 0 {
            Section {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(result.predictionFailures) 次预测失败（共 \(result.predictionAttempts) 次）")
                            .font(.callout.weight(.semibold))
                        if let detail = result.failureDetail {
                            Text(detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        } else {
            Section {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                    Text("模型已加载，\(result.predictionAttempts) 次预测全部成功")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func timelineSection(_ result: SWNOfflineResult) -> some View {
        Section("活动时间线") {
            // Color band: one segment per row, width proportional to duration.
            GeometryReader { geo in
                let w = geo.size.width
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(.systemGray6))
                    ForEach(Array(result.segments.enumerated()), id: \.offset) { _, seg in
                        let x0 = seg.start / max(result.duration, 1) * w
                        let x1 = seg.end / max(result.duration, 1) * w
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Self.activityColor(seg.label))
                            .frame(width: max(x1 - x0, 2), height: 30)
                            .offset(x: x0)
                            .overlay(alignment: .center) {
                                if x1 - x0 > 48 {
                                    Text(Self.activityName(seg.label))
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(.white)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.6)
                                }
                            }
                    }
                    ForEach(Array(result.eventTimes.enumerated()), id: \.offset) { _, t in
                        let x = t / max(result.duration, 1) * w
                        Rectangle()
                            .fill(Color.orange)
                            .frame(width: 2, height: 42)
                            .offset(x: x - 1)
                    }
                }
            }
            .frame(height: 42)
            .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))

            HStack {
                Label("\(Int(result.duration))s", systemImage: "clock")
                Spacer()
                Label("\(result.segments.count) 个活动段", systemImage: "rectangle.split.3x1")
                Spacer()
                Label("\(result.eventTimes.count) 次事件", systemImage: "bolt.fill")
                    .foregroundStyle(result.eventTimes.isEmpty ? Color.secondary : Color.orange)
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            // legend
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 88), alignment: .leading)], spacing: 6) {
                ForEach(0..<5, id: \.self) { i in
                    if result.perActivitySeconds[i] > 0 {
                        HStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Self.activityColor(i))
                                .frame(width: 10, height: 10)
                            Text(Self.activityName(i))
                                .font(.caption2)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func segmentsSection(_ result: SWNOfflineResult) -> some View {
        Section("分段结果") {
            ForEach(Array(result.segments.enumerated()), id: \.offset) { _, seg in
                HStack {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Self.activityColor(seg.label))
                        .frame(width: 10, height: 24)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(Self.activityName(seg.label))
                            .font(.callout.weight(.medium))
                        Text(String(format: "%.1f – %.1f s", seg.start, seg.end))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(String(format: "%.0f s", seg.duration))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func breakdownSection(_ result: SWNOfflineResult) -> some View {
        Section("各活动占比") {
            ForEach(0..<5, id: \.self) { i in
                if result.perActivitySeconds[i] > 0 {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(Self.activityName(i))
                                .font(.callout)
                            Spacer()
                            Text(String(format: "%.1f%%（%.0f s）",
                                        result.perActivityPercent[i],
                                        result.perActivitySeconds[i]))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Color(.systemGray6))
                                Capsule()
                                    .fill(Self.activityColor(i))
                                    .frame(width: geo.size.width * result.perActivityPercent[i] / 100)
                            }
                        }
                        .frame(height: 8)
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    @ViewBuilder
    private func rawTablePreview(_ table: CSVTable) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ScrollView([.horizontal, .vertical]) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                    GridRow {
                        ForEach(Array(table.headers.enumerated()), id: \.offset) { _, header in
                            Text(header.isEmpty ? "(empty)" : header)
                                .font(.caption.weight(.semibold))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    ForEach(Array(table.rows.prefix(Self.previewRowLimit).enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, value in
                                Text(value)
                                    .font(.caption2.monospaced())
                                    .lineLimit(1)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(height: 220)
            .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 10))
            if table.rows.count > Self.previewRowLimit {
                Text("仅显示前 \(Self.previewRowLimit) 行。")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: actions

    /// One flow: read -> parse -> (if motion CSV) analyze -> show results.
    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            state.readError = error.localizedDescription
        case .success(let urls):
            guard let url = urls.first else { return }
            state = CSVAnalysisState()
            state.isAnalyzing = true

            DispatchQueue.global(qos: .userInitiated).async {
                var newState = CSVAnalysisState()

                // 1) read + parse
                let readOutcome = readCSV(at: url)
                switch readOutcome {
                case .failure(let error):
                    newState.readError = error.localizedDescription
                case .success(let parsed):
                    newState.table = parsed

                    // 2) analyze immediately if it is a motion CSV
                    if SWNOfflineAnalyzer.isMotionCSV(headers: parsed.headers) {
                        let main = SWNPredictor(resource: "ActivityClassifier",
                                                inputNames: SWNFeatureExtractor.featureNames, classCount: 5)
                        let burst = SWNPredictor(resource: "BurstClassifier",
                                                 inputNames: SWNFeatureExtractor.featureNames, classCount: 5)
                        let event = SWNPredictor(resource: "EventDetector",
                                                 inputNames: SWNFeatureExtractor.eventFeatureNames, classCount: 2)
                        newState.modelsAvailable = main != nil && burst != nil
                        do {
                            let outcome = try SWNOfflineAnalyzer.analyze(
                                headers: parsed.headers, rows: parsed.rows,
                                main: main, burst: burst, event: event)
                            newState.result = outcome
                        } catch {
                            newState.analysisError = error.localizedDescription
                        }
                    } else {
                        newState.analysisError = "普通 CSV 文件：仅提供表格预览（运动分析需要 Sensor Logger 的 Motion.csv）。"
                    }
                }

                DispatchQueue.main.async {
                    state = newState
                }
            }
        }
    }

    private func readCSV(at url: URL) -> Result<CSVTable, Error> {
        let didStartAccessing = url.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                url.stopAccessingSecurityScopedResource()
            }
        }

        do {
            let data = try Data(contentsOf: url)
            guard let text = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .utf16) else {
                return .failure(CSVImportError.unsupportedEncoding)
            }
            let parsed = try CSVParser.parse(text: text)
            return .success(CSVTable(
                fileName: url.lastPathComponent,
                headers: parsed.headers,
                rows: parsed.rows
            ))
        } catch let error as CSVImportError {
            return .failure(error)
        } catch {
            return .failure(error)
        }
    }

    // MARK: activity metadata (matches swn/config.py ACTIVITIES order)

    static func activityName(_ label: Int) -> String {
        let names = ["行走", "慢跑", "上下楼", "静坐", "站立"]
        return label >= 0 && label < names.count ? names[label] : "未知"
    }

    static func activityColor(_ label: Int) -> Color {
        // walking/jogging/stairs/sitting/standing — same palette as the offline pipeline
        let colors: [Color] = [.blue, .red, .orange, .purple, .green]
        return label >= 0 && label < colors.count ? colors[label] : .gray
    }
}

#Preview {
    CSVImportView()
}
