//
//  WeeklyReportView.swift
//  Core Motion Data Collector iOS
//
//  Generates an AI weekly report from the last 7 days of journal entries
//  via an OpenAI-compatible LLM API. The report can be copied or saved
//  back into the journal as a regular entry.
//

import SwiftUI

struct WeeklyReportView: View {
    let store: LifeLogStore

    @State private var report: String?
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var showingConfigSheet = false
    @State private var savedToJournal = false
    @State private var copied = false

    private var recentEntryCount: Int {
        let calendar = Calendar.current
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        return store.entries.filter { $0.timestamp >= weekAgo }.count
    }

    var body: some View {
        VStack(spacing: 16) {
            header

            if isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Analyzing the last 7 days…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let errorMessage {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                        .foregroundStyle(.orange)
                    Text(errorMessage)
                        .font(.callout)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                    generateButton
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let report {
                reportBody(report)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "sparkles")
                        .font(.largeTitle)
                        .foregroundStyle(.tint)
                    Text("Generate a weekly report from your journal entries: life summary and health suggestions.")
                        .font(.callout)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)
                    generateButton
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding()
        .navigationTitle("Weekly Report")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showingConfigSheet = true
                } label: {
                    Image(systemName: "gearshape")
                }
            }
        }
        .sheet(isPresented: $showingConfigSheet) {
            LLMConfigSheet()
        }
    }

    // MARK: subviews

    private var header: some View {
        VStack(spacing: 4) {
            Text("\(recentEntryCount) entries in the last 7 days")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text("Photos are not uploaded - only your notes and timestamps are sent.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private var generateButton: some View {
        Button {
            generate()
        } label: {
            Label("Generate Report", systemImage: "sparkles")
                .font(.headline)
        }
        .buttonStyle(.borderedProminent)
        .disabled(recentEntryCount == 0)
    }

    private func reportBody(_ text: String) -> some View {
        VStack(spacing: 12) {
            ScrollView {
                // Render simple Markdown (bold, lists); fall back to plain text.
                if let attributed = try? AttributedString(
                    markdown: text,
                    options: AttributedString.MarkdownParsingOptions(
                        interpretedSyntax: .inlineOnlyPreservingWhitespace
                    )
                ) {
                    Text(attributed)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Text(text)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 12))

            HStack {
                Button {
                    UIPasteboard.general.string = text
                    copied = true
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                }
                .buttonStyle(.bordered)

                Button {
                    store.addEntry(note: "AI Weekly Report\n\n\(text)")
                    savedToJournal = true
                } label: {
                    Label(savedToJournal ? "Saved" : "Save to Journal", systemImage: savedToJournal ? "checkmark" : "square.and.arrow.down")
                }
                .buttonStyle(.borderedProminent)
                .disabled(savedToJournal)
            }
            .font(.callout)
        }
    }

    // MARK: action

    private func generate() {
        errorMessage = nil
        report = nil
        savedToJournal = false
        copied = false
        isLoading = true

        let prompt = LLMService.weeklyPrompt(entries: store.entries)
        let service = LLMService(config: LLMConfigStore.shared.config)

        Task {
            do {
                let answer = try await service.complete(system: prompt.system, user: prompt.user)
                await MainActor.run {
                    report = answer
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isLoading = false
                }
            }
        }
    }
}

// MARK: - API config sheet

struct LLMConfigSheet: View {
    @Environment(\.dismiss) private var dismiss

    @State private var baseURL: String = LLMConfigStore.shared.config.baseURL
    @State private var apiKey: String = LLMConfigStore.shared.config.apiKey
    @State private var model: String = LLMConfigStore.shared.config.model

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("https://api.openai.com/v1/chat/completions", text: $baseURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Base URL (OpenAI-compatible)")
                } footer: {
                    Text("Works with any OpenAI-style chat-completions endpoint: OpenAI, DeepSeek, Moonshot, local LLM servers, etc.")
                }

                Section {
                    SecureField("sk-…", text: $apiKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("API Key")
                } footer: {
                    Text("Stored on this device only (UserDefaults). Never uploaded anywhere except to the endpoint above.")
                }

                Section {
                    TextField("gpt-4o-mini", text: $model)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Model")
                }
            }
            .navigationTitle("LLM Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        LLMConfigStore.shared.config = LLMConfig(
                            baseURL: baseURL.trimmingCharacters(in: .whitespacesAndNewlines),
                            apiKey: apiKey.trimmingCharacters(in: .whitespacesAndNewlines),
                            model: model.trimmingCharacters(in: .whitespacesAndNewlines)
                        )
                        dismiss()
                    }
                }
            }
        }
    }
}
