//
//  SWNActivityCard.swift
//  Core Motion Data Collector
//
//  Main-screen card for the SenseWhenNeeded activity-recognition module.
//  Toggling it on starts a dedicated 20 Hz DeviceMotion stream and the
//  event-driven inference policy (see SWNRecognizer). Committed activity
//  changes can be auto-written into the LifeLog journal, so the weekly
//  LLM report also gets an activity timeline to analyse.
//

import SwiftUI

struct SWNActivityCard: View {
    let lifeLogStore: LifeLogStore?

    @State private var recognizer = SWNRecognizer()
    @State private var enabled = false

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Toggle(isOn: $enabled) {
                    Label("活动识别（Sense When Needed）", systemImage: "brain.head.profile")
                }
                .onChange(of: enabled) { _, isOn in
                    if isOn {
                        recognizer.start(lifeLogStore: lifeLogStore)
                    } else {
                        recognizer.stop()
                    }
                }

                if enabled {
                    Divider()

                    // Model status
                    HStack(spacing: 6) {
                        Image(systemName: recognizer.modelsReady ? "checkmark.seal" : "exclamationmark.triangle")
                            .foregroundStyle(recognizer.modelsReady ? .green : .orange)
                            .font(.footnote)
                        Text(recognizer.modelStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    // Current activity
                    HStack(alignment: .firstTextBaseline, spacing: 12) {
                        if let label = recognizer.committedLabel {
                            Text(SWNRecognizer.activityNamesCN[label])
                                .font(.title)
                                .bold()
                            Text(String(format: "%.0f%%", recognizer.committedConfidence * 100))
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        } else {
                            Text("识别中…")
                                .font(.title3)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if let cheap = recognizer.cheapLabel {
                            VStack(alignment: .trailing, spacing: 2) {
                                Text("1s 快速估计")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                Text(SWNRecognizer.activityNamesCN[cheap])
                                    .font(.footnote)
                            }
                        }
                    }

                    // Event detection status
                    VStack(alignment: .leading, spacing: 3) {
                        Text("事件检测（每 6 s）")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        HStack(spacing: 12) {
                            Text(String(format: "P(change) = %.2f", recognizer.lastEventProbability))
                            Text(String(format: "rule z = %.1f", recognizer.lastRuleZ))
                            if let date = recognizer.lastEventDate {
                                Text("最近触发 \(date.formatted(date: .omitted, time: .standard)) (\(recognizer.lastEventDescription))")
                                    .foregroundStyle(.orange)
                            }
                        }
                        .font(.caption)
                    }

                    Toggle("活动变化自动写入日记", isOn: $recognizer.autoLogToJournal)
                        .font(.footnote)

                    Text(String(format: "已采 %d 个样本 · %d 个突发（突发 1s/6s 占空比）",
                                recognizer.sampleCount, recognizer.burstCount))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onDisappear {
            if enabled {
                enabled = false
                recognizer.stop()
            }
        }
    }
}
