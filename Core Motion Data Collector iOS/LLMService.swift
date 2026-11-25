//
//  LLMService.swift
//  Core Motion Data Collector iOS
//
//  Minimal OpenAI-compatible chat-completions client used for the weekly
//  life-log report.
//
//  >>> HOW TO CONFIGURE <<<
//  The app reads its LLM settings from the "Weekly Report" screen
//  (sparkles icon in the journal toolbar). Fill in:
//    - Base URL : any OpenAI-compatible chat-completions endpoint, e.g.
//                 OpenAI   -> https://api.openai.com/v1/chat/completions
//                 DeepSeek -> https://api.deepseek.com/chat/completions
//                 Moonshot -> https://api.moonshot.cn/v1/chat/completions
//    - API key  : your key (stored in UserDefaults on this device only)
//    - Model    : e.g. "gpt-4o-mini", "deepseek-chat", "moonshot-v1-8k"
//
//  The request format is the standard OpenAI schema:
//  POST {baseURL} {"model": ..., "messages": [{role, content}, ...]}
//

import Foundation

struct LLMConfig: Codable, Equatable {
    var baseURL: String = "https://api.openai.com/v1/chat/completions"
    var apiKey: String = ""
    var model: String = "gpt-4o-mini"
}

enum LLMError: LocalizedError {
    case missingAPIKey
    case badURL
    case http(status: Int, body: String)
    case emptyResponse

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "No API key configured. Tap the gear icon and paste your key."
        case .badURL:
            return "Invalid base URL."
        case .http(let status, let body):
            return "API returned HTTP \(status). \(body.prefix(300))"
        case .emptyResponse:
            return "API returned an empty response."
        }
    }
}

/// Stores the LLM configuration in UserDefaults (device-local only).
final class LLMConfigStore {
    static let shared = LLMConfigStore()

    private let storageKey = "llm.weeklyReport.config"

    var config: LLMConfig {
        get {
            guard let data = UserDefaults.standard.data(forKey: storageKey),
                  let decoded = try? JSONDecoder().decode(LLMConfig.self, from: data) else {
                return LLMConfig()
            }
            return decoded
        }
        set {
            if let data = try? JSONEncoder().encode(newValue) {
                UserDefaults.standard.set(data, forKey: storageKey)
            }
        }
    }
}

final class LLMService {
    private let config: LLMConfig

    init(config: LLMConfig) {
        self.config = config
    }

    /// Sends a two-message (system + user) chat completion and returns the
    /// assistant's text content.
    func complete(system: String, user: String) async throws -> String {
        guard !config.apiKey.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw LLMError.missingAPIKey
        }
        guard let url = URL(string: config.baseURL.trimmingCharacters(in: .whitespaces)) else {
            throw LLMError.badURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")

        let payload: [String: Any] = [
            "model": config.model,
            "temperature": 0.7,
            "messages": [
                ["role": "system", "content": system],
                ["role": "user", "content": user]
            ]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, urlResponse) = try await URLSession.shared.data(for: request)
        if let http = urlResponse as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw LLMError.http(status: http.statusCode,
                                body: String(data: data, encoding: .utf8) ?? "")
        }

        struct ChatResponse: Decodable {
            struct Choice: Decodable {
                struct Message: Decodable {
                    let content: String?
                }
                let message: Message
            }
            let choices: [Choice]
        }

        let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
        guard let content = decoded.choices.first?.message.content,
              !content.isEmpty else {
            throw LLMError.emptyResponse
        }
        return content
    }

    // MARK: - Weekly report

    /// Builds the analysis prompt from the last 7 days of journal entries.
    /// Photos are not uploaded (text-only, cheap and privacy-friendly); the
    /// hook for extending to multimodal input is `entriesPayload`.
    static func weeklyPrompt(entries: [LifeLogEntry], now: Date = Date()) -> (system: String, user: String) {
        let system = """
        你是一个友善的生活方式分析助手。用户会给你他们过去一周手动记录的生活日志（时间 + 简短备注，可能附带照片数量信息）。
        请基于这些记录生成一份简明的中文周报，使用 Markdown，包含以下两部分：
        1. **生活状况总结**：作息、活动、日常节奏的客观概括；
        2. **健康生活建议**：3-5 条务实、具体的建议（如活动量、休息、规律性等）。
        注意事项：
        - 只依据日志内容，不要编造未记录的事实；
        - 如果记录很少或信息不足，请如实说明并相应弱化结论；
        - 建议是通用健康提示，不构成医疗建议。
        """

        let calendar = Calendar.current
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: now) ?? now
        let recent = entries.filter { $0.timestamp >= weekAgo }

        let formatter = DateFormatter()
        formatter.dateFormat = "MM-dd HH:mm"

        let lines = recent.map { entry -> String in
            var line = "[\(formatter.string(from: entry.timestamp))]"
            if !entry.note.isEmpty { line += " \(entry.note)" }
            if entry.photoFilename != nil { line += " （附照片一张）" }
            if entry.motionRecordingActive { line += " （运动数据采集中）" }
            return line
        }

        let entriesPayload = lines.isEmpty
            ? "（本周没有记录）"
            : lines.joined(separator: "\n")

        let user = """
        以下是过去 7 天的生活日志（共 \(recent.count) 条）：

        \(entriesPayload)

        请据此生成周报。
        """

        return (system, user)
    }
}
