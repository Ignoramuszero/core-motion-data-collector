//
//  SWNRecognizer.swift
//  Core Motion Data Collector
//
//  On-device activity recognition for the SenseWhenNeeded module.
//
//  Runs its own 20 Hz DeviceMotion stream (independent of the data
//  collector, so recognition quality does not depend on the user's
//  collection frequency setting) and follows the event-driven policy:
//
//    every 1 s  : 1 s burst -> BurstClassifier + burstStats history
//    every 3 s  : 3 s window -> ActivityClassifier (committed label)
//    every 6 s  : burst history -> EventDetector (RF >= 0.5 OR rule z >= 4.0)
//    on event   : force an immediate 3 s re-classification, reset history
//
//  Models are the Core ML conversions of the WISDM-trained Random Forests
//  (see swn-models-for-macos/MACOS_INTEGRATION.md). If the .mlmodel files
//  are not in the app bundle the module stays functional but inert and
//  reports what is missing in `modelStatus`.
//
//  When an activity change is committed it can automatically append a note
//  to the LifeLog journal (timestamps then align with any recorded CSV).
//

import CoreML
import CoreMotion
import Foundation

// MARK: - Core ML predictor wrapper

/// Thin wrapper around a converted sklearn tree classifier bundled as
/// <name>.mlmodelc. Expects named Double inputs and returns class label +
/// per-class probabilities.
final class SWNPredictor {
    let inputNames: [String]
    private let model: MLModel
    private let classCount: Int

    /// Diagnostics: how many predictions succeeded/failed since init, and
    /// why the last one failed. Used by the offline analyzer and the live
    /// card to surface silent failures (name mismatches, missing model…).
    private(set) var successCount = 0
    private(set) var failCount = 0
    private(set) var lastErrorMessage: String?

    init?(resource name: String, inputNames: [String], classCount: Int) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url) else { return nil }
        self.model = model
        self.inputNames = inputNames
        self.classCount = classCount
    }

    /// Returns (argmax class index, per-class probabilities).
    func predict(_ values: [Double]) -> (label: Int, probs: [Double])? {
        guard values.count == inputNames.count else {
            failCount += 1
            lastErrorMessage = "特征维度不匹配：模型需要 \(inputNames.count) 个，得到 \(values.count) 个"
            return nil
        }

        do {
            var features: [String: MLFeatureValue] = [:]
            for i in inputNames.indices {
                features[inputNames[i]] = MLFeatureValue(double: values[i])
            }
            let input = try MLDictionaryFeatureProvider(dictionary: features)
            let out = try model.prediction(from: input)

            var probs = [Double](repeating: 0, count: classCount)
            if let dict = out.featureValue(for: "output")?.dictionaryValue {
                for (key, value) in dict {
                    let index: Int?
                    if let number = key as? NSNumber {
                        index = number.intValue
                    } else if let string = key as? String {
                        index = Int(string)
                    } else {
                        index = nil
                    }
                    if let i = index, i >= 0, i < classCount {
                        probs[i] = value.doubleValue
                    }
                }
            }

            var label = -1
            if let classLabel = out.featureValue(for: "classLabel") {
                if classLabel.type == .int64 {
                    label = Int(classLabel.int64Value)
                } else if classLabel.type == .string {
                    label = Int(classLabel.stringValue) ?? -1
                }
            }
            if label < 0 {
                var best = 0
                for i in 1..<classCount where probs[i] > probs[best] { best = i }
                label = best
            }
            guard label >= 0, label < classCount else {
                failCount += 1
                lastErrorMessage = "模型输出的类别标签无法解析"
                return nil
            }
            successCount += 1
            return (label, probs)
        } catch {
            failCount += 1
            lastErrorMessage = error.localizedDescription
            return nil
        }
    }
}

// MARK: - Recognizer

struct SWNSample {
    let ax, ay, az: Double   // total acceleration (userAcceleration + gravity), m/s^2
    let gx, gy, gz: Double   // rotation rate, rad/s
}

@Observable final class SWNRecognizer {

    // Policy constants (match swn/config.py PolicyConfig / detector defaults).
    static let fs: Double = 20.0
    static let mainWindowSamples = 60      // 3 s @ 20 Hz
    static let burstSamples = 20           // 1 s @ 20 Hz
    static let mainPeriodTicks = 3         // high-fidelity window every 3 s
    static let eventPeriodTicks = 6        // change detection every 6 s
    static let maxBurstHistory = 4         // 24 s of context
    static let eventThreshold = 0.50       // P(change) of the RF branch
    static let ruleZThreshold = 4.0        // robust z of the rule branch
    static let confHigh = 0.85             // confident enough to commit
    static let confLow = 0.60              // keep/adopt a consistent label
    static let logCooldownSeconds: TimeInterval = 8

    static let activityNamesCN = ["行走", "慢跑", "上下楼", "静坐", "站立"]

    // MARK: Observable state

    private(set) var isRunning = false
    private(set) var modelStatus = "尚未启动"
    var modelsReady: Bool { mainPredictor != nil && burstPredictor != nil }

    private(set) var committedLabel: Int?
    private(set) var committedConfidence: Double = 0
    private(set) var cheapLabel: Int?
    private(set) var cheapConfidence: Double = 0
    private(set) var lastEventProbability: Double = 0
    private(set) var lastRuleZ: Double = 0
    private(set) var lastEventDate: Date?
    private(set) var lastEventDescription: String = "—"
    private(set) var burstCount = 0
    private(set) var sampleCount = 0

    /// When on, every committed activity change is written to the LifeLog journal.
    var autoLogToJournal = true

    // MARK: Internals

    private let motionManager = CMMotionManager()
    private var secondTimer: Timer?
    private var buffer: [SWNSample] = []
    private var burstHistory: [[Double]] = []
    private var tickCount = 0
    private var forceMainNextTick = false
    private var lastLogDate: Date?
    private weak var lifeLogStore: LifeLogStore?

    private var mainPredictor: SWNPredictor?
    private var burstPredictor: SWNPredictor?
    private var eventPredictor: SWNPredictor?

    // MARK: Lifecycle

    func start(lifeLogStore: LifeLogStore?) {
        guard !isRunning else { return }
        guard motionManager.isDeviceMotionAvailable else {
            modelStatus = "设备不支持 Device Motion"
            return
        }
        self.lifeLogStore = lifeLogStore
        loadModels()

        buffer.removeAll()
        burstHistory.removeAll()
        tickCount = 0
        burstCount = 0
        sampleCount = 0
        committedLabel = nil
        committedConfidence = 0
        cheapLabel = nil
        lastEventProbability = 0
        lastRuleZ = 0
        lastEventDate = nil
        lastEventDescription = "—"
        forceMainNextTick = false
        isRunning = true

        motionManager.deviceMotionUpdateInterval = 1.0 / Self.fs
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] data, _ in
            guard let self, let motion = data else { return }
            self.appendSample(
                ax: motion.userAcceleration.x + motion.gravity.x,
                ay: motion.userAcceleration.y + motion.gravity.y,
                az: motion.userAcceleration.z + motion.gravity.z,
                gx: motion.rotationRate.x,
                gy: motion.rotationRate.y,
                gz: motion.rotationRate.z
            )
        }

        secondTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    func stop() {
        secondTimer?.invalidate()
        secondTimer = nil
        motionManager.stopDeviceMotionUpdates()
        isRunning = false
        modelStatus = modelsReady ? "已停止（模型就绪）" : "已停止（" + missingModelHint + "）"
    }

    deinit {
        secondTimer?.invalidate()
        motionManager.stopDeviceMotionUpdates()
    }

    private var missingModelHint: String {
        mainPredictor == nil || burstPredictor == nil
            ? "缺 Activity/Burst 模型"
            : "缺 EventDetector 模型"
    }

    private func loadModels() {
        mainPredictor = SWNPredictor(resource: "ActivityClassifier",
                                     inputNames: SWNFeatureExtractor.featureNames, classCount: 5)
        burstPredictor = SWNPredictor(resource: "BurstClassifier",
                                      inputNames: SWNFeatureExtractor.featureNames, classCount: 5)
        eventPredictor = SWNPredictor(resource: "EventDetector",
                                      inputNames: SWNFeatureExtractor.eventFeatureNames, classCount: 2)

        if mainPredictor != nil && burstPredictor != nil {
            modelStatus = eventPredictor != nil
                ? "模型已加载（活动分类 + 事件检测）"
                : "仅活动分类（缺 EventDetector.mlmodel）"
        } else {
            modelStatus = "未找到 Core ML 模型：在 Mac 上运行 swn/coreml_export.py，"
                + "把生成的 3 个 .mlmodel 拖进 \"Core Motion Data Collector iOS\" 目录后重新构建"
        }
    }

    // MARK: Data flow

    private func appendSample(ax: Double, ay: Double, az: Double, gx: Double, gy: Double, gz: Double) {
        buffer.append(SWNSample(ax: ax, ay: ay, az: az, gx: gx, gy: gy, gz: gz))
        let maxBuffer = Int(SWNRecognizer.fs) * 6   // 6 s is enough for the largest window
        if buffer.count > maxBuffer {
            buffer.removeFirst(buffer.count - maxBuffer)
        }
        sampleCount += 1
    }

    private func lastSamples(_ count: Int) -> [SWNSample]? {
        guard buffer.count >= count else { return nil }
        return Array(buffer.suffix(count))
    }

    /// One decision tick per second.
    private func tick() {
        tickCount += 1
        guard let burstSamples = lastSamples(Self.burstSamples), burstSamples.count == Int(Self.fs) else { return }

        let acc = burstSamples.map { SWNVector3($0.ax, $0.ay, $0.az) }
        let gyro = burstSamples.map { SWNVector3($0.gx, $0.gy, $0.gz) }

        // 1) Cheap 1 s classification (the low-power stage's steady state).
        if let bp = burstPredictor {
            let feat = SWNFeatureExtractor.windowFeatures(acc: acc, gyro: gyro, fs: Self.fs)
            if let r = bp.predict(feat) {
                cheapLabel = r.label
                cheapConfidence = r.probs[r.label]
            }
        }

        // 2) High-fidelity 3 s classification (periodic, or forced by an event).
        if tickCount % Self.mainPeriodTicks == 0 || forceMainNextTick {
            forceMainNextTick = false
            classifyMain(lastSamples(Self.mainWindowSamples))
        }

        // 3) Event detection every 6 s: bursts spaced 6 s apart, matching the
        //    detector's training distribution (IDLE_PERIOD_S). The 1 s cheap
        //    classification above runs independently and much more often.
        if tickCount % Self.eventPeriodTicks == 0 {
            let stats = SWNFeatureExtractor.burstStats(acc: acc, fs: Self.fs)
            burstHistory.append(stats)
            if burstHistory.count > Self.maxBurstHistory {
                burstHistory.removeFirst(burstHistory.count - Self.maxBurstHistory)
            }
            burstCount += 1
            detectEvent()
        }
    }

    private func classifyMain(_ samples: [SWNSample]?) {
        guard let samples, samples.count == Self.mainWindowSamples, let mp = mainPredictor else { return }
        let acc = samples.map { SWNVector3($0.ax, $0.ay, $0.az) }
        let gyro = samples.map { SWNVector3($0.gx, $0.gy, $0.gz) }
        let feat = SWNFeatureExtractor.windowFeatures(acc: acc, gyro: gyro, fs: Self.fs)
        guard let r = mp.predict(feat) else { return }

        let conf = r.probs[r.label]
        let previous = committedLabel
        var shouldCommit = false
        if conf >= Self.confHigh {
            shouldCommit = true
        } else if conf >= Self.confLow && (previous == nil || previous == r.label) {
            shouldCommit = true
        }
        guard shouldCommit else { return }

        committedLabel = r.label
        committedConfidence = conf
        if r.label != previous {
            logActivityChange(newLabel: r.label, confidence: conf)
        }
    }

    private func detectEvent() {
        guard burstHistory.count >= 2 else { return }
        let feat = SWNFeatureExtractor.burstEventFeatures(history: burstHistory)
        let z = SWNFeatureExtractor.burstZ(history: burstHistory)
        lastRuleZ = z

        // RF branch probability of class 1 (change). Without the event model
        // the rule branch alone still works (hybrid degrades gracefully).
        var pRF = 0.0
        if let ep = eventPredictor, let r = ep.predict(feat), r.probs.count > 1 {
            pRF = r.probs[1]
        }
        lastEventProbability = pRF

        // Two independent operating points, exactly like the trained policy.
        let fired = pRF >= Self.eventThreshold || z >= Self.ruleZThreshold
        if fired {
            lastEventDate = Date()
            lastEventDescription = String(format: "P=%.2f, z=%.1f", pRF, z)
            forceMainNextTick = true
            // The regime changed; rebuild the reference from scratch
            // (same bootstrap behaviour as the Python runtime policy).
            burstHistory.removeAll()
        }
    }

    // MARK: Journal integration

    private func logActivityChange(newLabel: Int, confidence: Double) {
        guard newLabel >= 0, newLabel < Self.activityNamesCN.count else { return }
        let now = Date()
        if let last = lastLogDate, now.timeIntervalSince(last) < Self.logCooldownSeconds {
            return
        }
        lastLogDate = now
        guard autoLogToJournal, let store = lifeLogStore else { return }
        let note = String(format: "自动记录：检测到「%@」（置信度 %.0f%%）",
                          Self.activityNamesCN[newLabel], confidence * 100)
        store.addEntry(note: note)
    }
}
