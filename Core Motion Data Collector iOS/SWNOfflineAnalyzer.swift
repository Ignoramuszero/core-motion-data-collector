//
//  SWNOfflineAnalyzer.swift
//  Core Motion Data Collector iOS
//
//  Offline activity analysis for an imported motion CSV (Sensor Logger
//  export). Mirrors scripts/process_recording.py exactly:
//
//    parse columns (time, accelerationXYZ [g], gravityXYZ [g], rotationRateXYZ [rad/s])
//    -> total accel = (accel + gravity) * 9.80665 m/s^2
//    -> resample onto a uniform 20 Hz grid (linear interpolation)
//    -> replay the SAME event-driven policy as the live recognizer:
//         every 1 s: 1 s burst -> BurstClassifier
//         every 3 s: 3 s window -> ActivityClassifier (commit)
//         every 6 s: burst history -> EventDetector (RF >= 0.5 OR z >= 4.0)
//    -> per-second decisions -> contiguous segments + per-activity totals
//
//  Runs on any recording; requires the three Core ML models in the bundle
//  (the rule branch still works without EventDetector, degrading gracefully).
//

import Foundation

// MARK: - Result types

struct SWNOfflineDecision {
    let t: Double
    let cheapLabel: Int
    let cheapConf: Double
    let committedLabel: Int?
    let committedConf: Double
    let pChange: Double
    let ruleZ: Double
    let eventFired: Bool
    /// committed if available, otherwise the cheap estimate
    var labelUsed: Int { committedLabel ?? cheapLabel }
}

struct SWNOfflineSegment {
    let start: Double
    let end: Double
    let label: Int
    var duration: Double { end - start }
}

struct SWNOfflineResult {
    let duration: Double
    let decisions: [SWNOfflineDecision]
    let segments: [SWNOfflineSegment]
    /// seconds of each activity (index = class label)
    let perActivitySeconds: [Double]
    /// percentage of each activity (index = class label)
    let perActivityPercent: [Double]
    let eventTimes: [Double]

    // MARK: Diagnostics (filled by analyze(); defaults for direct buildResult)
    /// true when ActivityClassifier + BurstClassifier were available
    var modelsLoaded = false
    /// number of per-second decision points that used at least one model
    var predictionAttempts = 0
    /// number of failed Core ML predictions
    var predictionFailures = 0
    /// reason of the last failure, if any
    var failureDetail: String?
}

enum SWNOfflineError: LocalizedError {
    case notMotionCSV
    case tooShort

    var errorDescription: String? {
        switch self {
        case .notMotionCSV:
            return "这不是运动数据 CSV：需要包含 time、accelerationX/Y/Z、gravityX/Y/Z、rotationRateX/Y/Z 列（Sensor Logger 导出格式）。"
        case .tooShort:
            return "数据太短：至少需要 3 秒的运动样本才能做活动识别。"
        }
    }
}

// MARK: - Analyzer

enum SWNOfflineAnalyzer {

    static let fs: Double = 20.0
    static let gToMS2 = 9.80665

    /// Heuristic: does this CSV look like a Sensor Logger motion export?
    static func isMotionCSV(headers: [String]) -> Bool {
        let lower = headers.map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
        let hasTime = lower.contains("time")
        let hasAccel = lower.contains { $0.contains("acceleration") || $0.contains("accel") }
        let hasGyro = lower.contains { $0.contains("rotationrate") || $0.contains("gyro") }
        return hasTime && hasAccel && hasGyro
    }

    // MARK: CSV -> resampled 20 Hz streams

    /// Parses raw CSV rows into (t, totalAccel[3], gyro[3]) resampled to 20 Hz.
    static func loadMotion(headers: [String], rows: [[String]]) throws
        -> (t: [Double], acc: [[Double]], gyro: [[Double]]) {

        // Column resolution (case-insensitive, whitespace-tolerant).
        let lower = headers.map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
        func indexOf(_ candidates: [String]) -> Int? {
            for c in candidates {
                if let i = lower.firstIndex(of: c) { return i }
            }
            // fallback: contains-match
            for c in candidates {
                if let i = lower.firstIndex(where: { $0.contains(c) }) { return i }
            }
            return nil
        }

        guard let tCol = indexOf(["time"]),
              lower.indices.contains(where: { lower[$0].contains("acceleration") || lower[$0].contains("accel") }),
              lower.contains(where: { $0.contains("rotationrate") || $0.contains("gyro") }) else {
            throw SWNOfflineError.notMotionCSV
        }

        // Prefer dedicated accel/gravity/gyro column triples.
        func triple(_ prefix: String) -> [Int?]? {
            let x = indexOf(["\(prefix)x"])
            let y = indexOf(["\(prefix)y"])
            let z = indexOf(["\(prefix)z"])
            guard x != nil, y != nil, z != nil else { return nil }
            return [x, y, z]
        }

        let accelCols = triple("acceleration") ?? triple("accel")
        let gravityCols = triple("gravity")
        let gyroCols = triple("rotationrate") ?? triple("gyro")
        guard let accelCols, let gyroCols else { throw SWNOfflineError.notMotionCSV }

        var t: [Double] = []
        var acc: [[Double]] = [[], [], []]
        var gyro: [[Double]] = [[], [], []]
        var t0: Double?

        for row in rows {
            guard tCol < row.count else { continue }
            func value(_ col: Int?) -> Double? {
                guard let col, col < row.count else { return nil }
                return Double(row[col].trimmingCharacters(in: .whitespaces))
            }
            guard let time = value(tCol) else { continue }
            if t0 == nil { t0 = time }
            var a = [0.0, 0.0, 0.0]
            for axis in 0..<3 {
                a[axis] = value(accelCols[axis]) ?? 0
            }
            // Sensor Logger: acceleration is user acceleration in g, gravity
            // is separate. Total acceleration = (accel + gravity) * 9.80665.
            // Without gravity columns assume the acceleration is already total.
            if let g = gravityCols {
                for axis in 0..<3 {
                    a[axis] = (a[axis] + (value(g[axis]) ?? 0)) * gToMS2
                }
            } else {
                for axis in 0..<3 {
                    a[axis] *= gToMS2
                }
            }
            var w = [0.0, 0.0, 0.0]
            for axis in 0..<3 {
                w[axis] = value(gyroCols[axis]) ?? 0
            }

            t.append(time)
            for axis in 0..<3 {
                acc[axis].append(a[axis])
                gyro[axis].append(w[axis])
            }
        }

        guard t.count >= 60 else { throw SWNOfflineError.tooShort }  // >= 3 s

        // Normalize time to start at 0 and resample onto a uniform 20 Hz grid
        // with linear interpolation (handles irregular sampling / gaps).
        let base = t0!
        let tMax = t[t.count - 1] - base
        let step = 1.0 / fs
        var grid: [Double] = []
        var g = 0.0
        while g < tMax {
            grid.append(g)
            g += step
        }

        func interp(_ channel: [Double]) -> [Double] {
            var out = [Double](repeating: 0, count: grid.count)
            var j = 0
            for i in 0..<grid.count {
                let x = grid[i]
                while j + 1 < t.count && t[j + 1] - base < x { j += 1 }
                let x0 = t[j] - base
                let x1 = (j + 1 < t.count) ? t[j + 1] - base : x0
                let y0 = channel[j]
                let y1 = (j + 1 < channel.count) ? channel[j + 1] : y0
                let f = (x1 - x0) > 0 ? min(max((x - x0) / (x1 - x0), 0), 1) : 0
                out[i] = y0 + (y1 - y0) * f
            }
            return out
        }

        let resampledAcc = acc.map { interp($0) }
        let resampledGyro = gyro.map { interp($0) }
        return (grid, resampledAcc, resampledGyro)
    }

    // MARK: Policy replay (identical to process_recording.py / SWNRecognizer)

    static func analyze(headers: [String], rows: [[String]],
                        main: SWNPredictor?, burst: SWNPredictor?, event: SWNPredictor?) throws -> SWNOfflineResult {
        let (t, acc, gyro) = try loadMotion(headers: headers, rows: rows)
        let n = t.count
        let burstSamples = Int(fs)          // 20 samples = 1 s
        let mainSamples = 60                // 60 samples = 3 s
        let mainPeriod = 3                  // every 3 s
        let eventPeriod = 6                 // every 6 s
        let maxHistory = 4
        let eventThreshold = 0.50
        let ruleZThreshold = 4.0
        let confHigh = 0.85
        let confLow = 0.60

        var decisions: [SWNOfflineDecision] = []
        var history: [[Double]] = []
        var committed: (label: Int, conf: Double)?
        var forceMain = false

        let seconds = n / burstSamples
        for s in 0..<seconds {
            let b0 = s * burstSamples
            let burstAcc = (0..<burstSamples).map { SWNVector3(acc[0][b0 + $0], acc[1][b0 + $0], acc[2][b0 + $0]) }
            let burstGyro = (0..<burstSamples).map { SWNVector3(gyro[0][b0 + $0], gyro[1][b0 + $0], gyro[2][b0 + $0]) }

            // 1) cheap 1 s classification
            var cheapLabel = 3   // default to sitting if no model
            var cheapConf = 0.0
            if let bp = burst {
                let feat = SWNFeatureExtractor.windowFeatures(acc: burstAcc, gyro: burstGyro, fs: fs)
                if let r = bp.predict(feat) {
                    cheapLabel = r.label
                    cheapConf = r.probs[r.label]
                }
            }

            // 2) high-fidelity 3 s classification (periodic or event-forced)
            var changed = false
            if s % mainPeriod == 0 || forceMain {
                forceMain = false
                if b0 + mainSamples <= n {
                    let accW = (0..<mainSamples).map { SWNVector3(acc[0][b0 + $0], acc[1][b0 + $0], acc[2][b0 + $0]) }
                    let gyroW = (0..<mainSamples).map { SWNVector3(gyro[0][b0 + $0], gyro[1][b0 + $0], gyro[2][b0 + $0]) }
                    if let mp = main {
                        let feat = SWNFeatureExtractor.windowFeatures(acc: accW, gyro: gyroW, fs: fs)
                        if let r = mp.predict(feat) {
                            let conf = r.probs[r.label]
                            let prev = committed?.label
                            let shouldCommit = conf >= confHigh ||
                                (conf >= confLow && (prev == nil || prev == r.label))
                            if shouldCommit {
                                committed = (r.label, conf)
                                if prev == nil || r.label != prev! {
                                    changed = true
                                }
                            }
                        }
                    }
                }
            }

            // 3) event detection every 6 s (bursts spaced 6 s apart, matching
            //    the detector's training distribution)
            var pChange = 0.0
            var z = 0.0
            var fired = false
            if s % eventPeriod == 0 {
                let stats = SWNFeatureExtractor.burstStats(acc: burstAcc, fs: fs)
                history.append(stats)
                if history.count > maxHistory {
                    history.removeFirst(history.count - maxHistory)
                }
                if history.count >= 2 {
                    z = SWNFeatureExtractor.burstZ(history: history)
                    if let ep = event {
                        let feat = SWNFeatureExtractor.burstEventFeatures(history: history)
                        if let r = ep.predict(feat), r.probs.count > 1 {
                            pChange = r.probs[1]
                        }
                    }
                    fired = pChange >= eventThreshold || z >= ruleZThreshold
                    if fired {
                        forceMain = true
                        history.removeAll()
                    }
                }
            }

            decisions.append(SWNOfflineDecision(
                t: t[b0], cheapLabel: cheapLabel, cheapConf: cheapConf,
                committedLabel: committed?.label, committedConf: committed?.conf ?? 0,
                pChange: pChange, ruleZ: z, eventFired: fired))
            _ = changed
        }

        var result = buildResult(decisions: decisions)
        result.modelsLoaded = main != nil && burst != nil
        let predictors = [main, burst, event].compactMap { $0 }
        result.predictionAttempts = predictors.map { $0.successCount + $0.failCount }.max() ?? 0
        result.predictionFailures = predictors.map { $0.failCount }.max() ?? 0
        result.failureDetail = predictors.compactMap { $0.lastErrorMessage }.first
        return result
    }

    // MARK: Segments + totals (same logic as build_segments in Python)

    static func buildResult(decisions: [SWNOfflineDecision]) -> SWNOfflineResult {
        var segments: [SWNOfflineSegment] = []
        for d in decisions {
            let label = d.labelUsed
            if let last = segments.last, last.label == label, d.t - last.end <= 1.5 {
                segments[segments.count - 1] = SWNOfflineSegment(
                    start: last.start, end: d.t + 1.0, label: last.label)
            } else {
                segments.append(SWNOfflineSegment(start: d.t, end: d.t + 1.0, label: label))
            }
        }
        // drop slivers < 2 s (single-window misfires)
        segments = segments.filter { $0.duration >= 2.0 }

        var totals = [Double](repeating: 0, count: 5)
        for s in segments where s.label >= 0 && s.label < totals.count {
            totals[s.label] += s.duration
        }
        let duration = decisions.last.map { $0.t + 1.0 } ?? 0
        let percent = duration > 0 ? totals.map { 100.0 * $0 / duration } : totals
        let eventTimes = decisions.filter { $0.eventFired }.map { $0.t }

        return SWNOfflineResult(
            duration: duration,
            decisions: decisions,
            segments: segments,
            perActivitySeconds: totals,
            perActivityPercent: percent,
            eventTimes: eventTimes)
    }
}
