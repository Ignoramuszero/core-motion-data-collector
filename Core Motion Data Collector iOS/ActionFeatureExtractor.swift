//
//  ActionFeatureExtractor.swift
//  Core Motion Data Collector iOS
//
//  Swift port of the feature extraction used to train the SenseWhenNeeded
//  activity classifier (swn/features.py, window_features). The feature order
//  MUST stay identical to the Python version.
//

import Foundation

struct Vec3 {
    let x: Double
    let y: Double
    let z: Double
}

struct IMUSample {
    let t: Double
    let acc: Vec3
    let gyro: Vec3
}

struct ActionWindowResult {
    let start: Double
    let end: Double
    let label: Int
    let confidence: Double
    let probabilities: [Double]
}

struct ActionSegment {
    let label: Int
    let start: Double
    let end: Double
    let averageConfidence: Double
    let windowCount: Int
}

// MARK: - CSV -> IMU samples

enum ActionCSVImporter {
    static func samples(from table: CSVTable, targetFS: Double = 20.0) throws -> [IMUSample] {
        let timestampIndex = columnIndex(table, ["Timestamp"])
        let accX = columnIndex(table, ["Acceleration x"])
        let accY = columnIndex(table, ["Acceleration y"])
        let accZ = columnIndex(table, ["Acceleration z"])
        let userX = columnIndex(table, ["User Acceleration x"])
        let userY = columnIndex(table, ["User Acceleration y"])
        let userZ = columnIndex(table, ["User Acceleration z"])
        let gravityX = columnIndex(table, ["Gravity x"])
        let gravityY = columnIndex(table, ["Gravity y"])
        let gravityZ = columnIndex(table, ["Gravity z"])
        let gyroX = columnIndex(table, ["Rotation Rate x"])
        let gyroY = columnIndex(table, ["Rotation Rate y"])
        let gyroZ = columnIndex(table, ["Rotation Rate z"])

        let hasDirectAccel = accX != nil && accY != nil && accZ != nil
        let hasUserAccel = userX != nil && userY != nil && userZ != nil
        guard hasDirectAccel || hasUserAccel else {
            throw ActionRecognitionError.missingColumns("Need 'Acceleration x/y/z' or 'User Acceleration x/y/z'.")
        }
        guard let gx = gyroX, let gy = gyroY, let gz = gyroZ else {
            throw ActionRecognitionError.missingColumns("Need 'Rotation Rate x/y/z' for gyroscope.")
        }

        var rows: [IMUSample] = []
        rows.reserveCapacity(table.rows.count)

        for (index, row) in table.rows.enumerated() {
            func value(_ i: Int?) -> Double? {
                guard let i, i < row.count else { return nil }
                return Double(row[i].trimmingCharacters(in: .whitespacesAndNewlines))
            }

            let t: Double
            if let timestampIndex, let parsed = value(timestampIndex) {
                t = parsed
            } else {
                t = Double(index) / targetFS
            }

            let ax: Double
            let ay: Double
            let az: Double
            if hasDirectAccel, let x = value(accX), let y = value(accY), let z = value(accZ) {
                ax = x; ay = y; az = z
            } else if let x = value(userX), let y = value(userY), let z = value(userZ) {
                if let gxI = gravityX, let gyI = gravityY, let gzI = gravityZ,
                   let gxV = value(gxI), let gyV = value(gyI), let gzV = value(gzI) {
                    ax = x + gxV; ay = y + gyV; az = z + gzV
                } else {
                    ax = x; ay = y; az = z
                }
            } else {
                continue
            }

            guard let gxV = value(gx), let gyV = value(gy), let gzV = value(gz) else { continue }
            rows.append(IMUSample(t: t,
                                  acc: Vec3(x: ax, y: ay, z: az),
                                  gyro: Vec3(x: gxV, y: gyV, z: gzV)))
        }

        guard rows.count >= Int(3 * targetFS) else {
            throw ActionRecognitionError.tooFewSamples(rows.count, required: Int(3 * targetFS))
        }

        let sorted = rows.sorted { $0.t < $1.t }
        if needsResampling(sorted, targetFS: targetFS) {
            return resample(sorted, fs: targetFS)
        }
        return sorted
    }

    private static func columnIndex(_ table: CSVTable, _ candidates: [String]) -> Int? {
        let normalized = table.headers.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
        for candidate in candidates {
            if let index = normalized.firstIndex(of: candidate.lowercased()) { return index }
        }
        return nil
    }

    private static func needsResampling(_ rows: [IMUSample], targetFS: Double) -> Bool {
        guard rows.count > 2 else { return false }
        let span = rows.last!.t - rows.first!.t
        guard span > 0 else { return false }
        let averageDt = span / Double(rows.count - 1)
        return abs(averageDt - 1.0 / targetFS) > 0.002
    }

    private static func resample(_ rows: [IMUSample], fs: Double) -> [IMUSample] {
        let t0 = rows.first!.t
        let t1 = rows.last!.t
        let step = 1.0 / fs
        let count = Int(floor((t1 - t0) / step)) + 1
        guard count >= 2 else { return rows }

        var output: [IMUSample] = []
        output.reserveCapacity(count)
        var i = 0

        for k in 0..<count {
            let t = t0 + Double(k) * step
            while i < rows.count - 2 && rows[i + 1].t < t { i += 1 }
            let a = rows[i]
            let b = rows[min(i + 1, rows.count - 1)]
            let span = b.t - a.t
            let f = span > 0 ? min(max((t - a.t) / span, 0.0), 1.0) : 0.0
            output.append(IMUSample(t: t,
                                    acc: lerp(a.acc, b.acc, f),
                                    gyro: lerp(a.gyro, b.gyro, f)))
        }
        return output
    }

    private static func lerp(_ a: Vec3, _ b: Vec3, _ f: Double) -> Vec3 {
        Vec3(x: a.x + (b.x - a.x) * f,
             y: a.y + (b.y - a.y) * f,
             z: a.z + (b.z - a.z) * f)
    }

    static func windows(from samples: [IMUSample], windowSeconds: Double = 3.0,
                        hopSeconds: Double = 1.0, fs: Double = 20.0)
        -> [(start: Double, end: Double, acc: [Vec3], gyro: [Vec3])] {
        let windowSize = Int(round(windowSeconds * fs))
        let hopSize = Int(round(hopSeconds * fs))
        guard windowSize >= 2, samples.count >= windowSize else { return [] }

        var result: [(start: Double, end: Double, acc: [Vec3], gyro: [Vec3])] = []
        var end = windowSize
        while end <= samples.count {
            let slice = samples[(end - windowSize)..<end]
            result.append((start: slice.first!.t,
                           end: slice.last!.t + 1.0 / fs,
                           acc: slice.map(\.acc),
                           gyro: slice.map(\.gyro)))
            end += hopSize
        }
        return result
    }
}

// MARK: - Feature extraction

enum ActionFeatureExtractor {
    static let featureCount = 59

    /// Exact order from swn/features.py::window_features.
    static let featureNames: [String] = {
        let accStats = ["mean", "std", "min", "max", "rms", "median", "iqr", "skew", "kurt", "energy"]
        let axes = ["x", "y", "z"]
        let gyroStats = ["mean", "std", "rms", "max"]
        var names: [String] = []
        for stat in accStats {
            for axis in axes { names.append("acc_\(axis)_\(stat)") }
        }
        names += ["mag_mean", "mag_std", "mag_max", "mag_min", "mag_p90", "mag_p10"]
        names += ["mag_dom_freq", "mag_e_low", "mag_e_mid", "mag_e_high", "mag_entropy"]
        for stat in gyroStats {
            for axis in axes { names.append("gyro_\(axis)_\(stat)") }
        }
        names += ["gyromag_mean", "gyromag_std", "gyromag_max"]
        names += ["acc_corr_xy", "acc_corr_xz", "acc_corr_yz"]
        return names
    }()

    static func vector(acc: [Vec3], gyro: [Vec3], fs: Double = 20.0) -> [Double] {
        let accX = acc.map(\.x), accY = acc.map(\.y), accZ = acc.map(\.z)
        let gyroX = gyro.map(\.x), gyroY = gyro.map(\.y), gyroZ = gyro.map(\.z)

        let accAxes = [accX, accY, accZ]
        var values: [Double] = []

        let accMeans = accAxes.map { mean($0) }
        let accStds = accAxes.map { std($0) }
        for statIndex in 0..<10 {
            for axisIndex in 0..<3 {
                switch statIndex {
                case 0: values.append(accMeans[axisIndex])
                case 1: values.append(accStds[axisIndex])
                case 2: values.append(accAxes[axisIndex].min() ?? 0)
                case 3: values.append(accAxes[axisIndex].max() ?? 0)
                case 4: values.append(rms(accAxes[axisIndex]))
                case 5: values.append(median(accAxes[axisIndex]))
                case 6: values.append(percentile(accAxes[axisIndex], 75) - percentile(accAxes[axisIndex], 25))
                case 7: values.append(skew(accAxes[axisIndex], mean: accMeans[axisIndex], std: accStds[axisIndex]))
                case 8: values.append(kurt(accAxes[axisIndex], mean: accMeans[axisIndex], std: accStds[axisIndex]))
                default: values.append(accAxes[axisIndex].reduce(0) { $0 + $1 * $1 })
                }
            }
        }
        let magnitudes = acc.indices.map { i in
            sqrt(acc[i].x * acc[i].x + acc[i].y * acc[i].y + acc[i].z * acc[i].z)
        }
        let sortedMag = magnitudes.sorted()
        values += [mean(magnitudes), std(magnitudes), magnitudes.max() ?? 0,
                   magnitudes.min() ?? 0, percentile(sortedMag, 90), percentile(sortedMag, 10)]

        let spectral = spectralFeatures(magnitudes, fs: fs)
        values += spectral

        let gyroAxes = [gyroX, gyroY, gyroZ]
        let gyroMeans = gyroAxes.map { mean($0) }
        let gyroStds = gyroAxes.map { std($0) }
        for axisIndex in 0..<3 {
            values.append(gyroMeans[axisIndex])
            values.append(gyroStds[axisIndex])
            values.append(rms(gyroAxes[axisIndex]))
            values.append(gyroAxes[axisIndex].map(abs).max() ?? 0)
        }

        let gyroMagnitudes = gyro.indices.map { i in
            sqrt(gyro[i].x * gyro[i].x + gyro[i].y * gyro[i].y + gyro[i].z * gyro[i].z)
        }
        values += [mean(gyroMagnitudes), std(gyroMagnitudes), gyroMagnitudes.max() ?? 0]

        values += [correlation(accX, accY), correlation(accX, accZ), correlation(accY, accZ)]

        return values
    }

    // MARK: math helpers

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func std(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let m = mean(values)
        let variance = values.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(values.count)
        return sqrt(variance)
    }

    private static func rms(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return sqrt(values.reduce(0) { $0 + $1 * $1 } / Double(values.count))
    }

    private static func median(_ values: [Double]) -> Double {
        percentile(values.sorted(), 50)
    }

    /// NumPy's default linear percentile on an already-sorted array.
    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        if sorted.count == 1 { return sorted[0] }
        let position = (Double(sorted.count - 1)) * p / 100.0
        let lower = Int(floor(position))
        let upper = Int(ceil(position))
        if lower == upper { return sorted[lower] }
        let fraction = position - Double(lower)
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }

    private static func skew(_ values: [Double], mean m: Double, std s: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let third = values.reduce(0) { $0 + pow($1 - m, 3) } / Double(values.count)
        return third / (pow(s, 3) + 1e-9)
    }

    private static func kurt(_ values: [Double], mean m: Double, std s: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let fourth = values.reduce(0) { $0 + pow($1 - m, 4) } / Double(values.count)
        return fourth / (pow(s, 4) + 1e-9)
    }

    private static func moments(_ values: [Double], mean m: Double, std s: Double) -> (skew: Double, kurt: Double) {
        (skew(values, mean: m, std: s), kurt(values, mean: m, std: s))
    }

    private static func correlation(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        let ma = mean(a), mb = mean(b)
        var numerator = 0.0
        var sumA2 = 0.0
        var sumB2 = 0.0
        for i in a.indices {
            let da = a[i] - ma
            let db = b[i] - mb
            numerator += da * db
            sumA2 += da * da
            sumB2 += db * db
        }
        let denominator = sqrt(sumA2 * sumB2)
        guard denominator > 1e-12 else { return 0 }
        let value = numerator / denominator
        return value.isFinite ? value : 0
    }

    private static func spectralFeatures(_ magnitude: [Double], fs: Double) -> [Double] {
        let n = magnitude.count
        guard n >= 8 else { return [0, 0, 0, 0, 0] }

        let m = mean(magnitude)
        let window = (0..<n).map { i in
            0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(n - 1))
        }
        let centered = zip(magnitude, window).map { ($0 - m) * $1 }

        let spectrum = rfftMagnitudes(centered)
        let frequencies = (0..<spectrum.count).map { Double($0) * fs / Double(n) }
        let total = spectrum.reduce(0, +)
        guard total > 1e-9 else { return [0, 0, 0, 0, 0] }

        var dominant = 0.0
        if spectrum.count > 2 {
            var best = 1
            for i in 2..<spectrum.count where spectrum[i] > spectrum[best] { best = i }
            dominant = frequencies[best]
        }

        let probabilities = spectrum.map { $0 / total }
        var low = 0.0, mid = 0.0, high = 0.0, entropy = 0.0
        for i in frequencies.indices {
            let f = frequencies[i]
            let p = probabilities[i]
            if f >= 0.4 && f < 3.0 { low += p }
            if f >= 3.0 && f < 8.0 { mid += p }
            if f >= 8.0 { high += p }
            entropy += -p * log(p + 1e-12)
        }
        return [dominant, low, mid, high, entropy]
    }

    /// Unnormalised DFT magnitudes for k = 0 ... n/2 (equivalent to numpy.fft.rfft abs).
    private static func rfftMagnitudes(_ values: [Double]) -> [Double] {
        let n = values.count
        let bins = n / 2 + 1
        var output = [Double](repeating: 0, count: bins)
        for k in 0..<bins {
            var real = 0.0
            var imaginary = 0.0
            for j in 0..<n {
                let angle = -2.0 * Double.pi * Double(k * j) / Double(n)
                real += values[j] * cos(angle)
                imaginary += values[j] * sin(angle)
            }
            output[k] = sqrt(real * real + imaginary * imaginary)
        }
        return output
    }
}

// MARK: - Segment merging

enum ActionSegmenter {
    static func merge(_ results: [ActionWindowResult], minSegmentWindows: Int = 2) -> [ActionSegment] {
        guard !results.isEmpty else { return [] }
        var segments: [ActionSegment] = []
        var currentLabel = results[0].label
        var currentStart = results[0].start
        var currentEnd = results[0].end
        var confidenceSum = results[0].confidence
        var count = 1

        func flush() {
            segments.append(ActionSegment(label: currentLabel,
                                          start: currentStart,
                                          end: currentEnd,
                                          averageConfidence: confidenceSum / Double(count),
                                          windowCount: count))
        }

        for result in results.dropFirst() {
            if result.label == currentLabel {
                currentEnd = result.end
                confidenceSum += result.confidence
                count += 1
            } else {
                flush()
                currentLabel = result.label
                currentStart = result.start
                currentEnd = result.end
                confidenceSum = result.confidence
                count = 1
            }
        }
        flush()

        if minSegmentWindows > 1 {
            // Keep short segments but mark them via windowCount in the UI instead of
            // silently dropping transitions.
            return segments
        }
        return segments
    }
}

enum ActionRecognitionError: LocalizedError {
    case missingColumns(String)
    case tooFewSamples(Int, required: Int)
    case modelUnavailable
    case modelFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingColumns(let detail):
            return "CSV is missing required columns: \(detail)"
        case .tooFewSamples(let count, let required):
            return "CSV has only \(count) usable samples; need at least \(required) (3 s @ 20 Hz)."
        case .modelUnavailable:
            return "The bundled activity model is unavailable."
        case .modelFailed(let detail):
            return "Model inference failed: \(detail)"
        }
    }
}
