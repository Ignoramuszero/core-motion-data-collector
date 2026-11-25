//
//  SWNFeatureExtractor.swift
//  Core Motion Data Collector
//
//  Swift replication of the feature pipeline in the SenseWhenNeeded
//  project (swn/features.py). The Core ML models were trained on feature
//  vectors computed by that Python code, so this implementation MUST stay
//  numerically identical, in the same order:
//
//    windowFeatures(acc:gyro:fs:)      -> 59 features  (Activity/Burst classifiers)
//    burstEventFeatures(history:)      -> 30 features  (EventDetector)
//    burstZ(history:)                  -> 1 value      (rule branch of the event detector)
//
//  Conventions (identical to Python):
//    - fs = 20 Hz, accelerometer in m/s^2 (userAcceleration + gravity),
//      gyroscope in rad/s
//    - activity label order: ["walking", "jogging", "stairs", "sitting", "standing"]
//
//  A sanity test is recommended: run the Python snippet
//      from swn.features import window_features
//      print(window_features(acc, gyro, 20.0)[0])
//  on the same 20 Hz window and compare with SWNFeatureExtractor.selfTest().
//

import Foundation

typealias SWNVector3 = SIMD3<Double>

enum SWNFeatureExtractor {

    // MARK: - Names (define the Core ML input parameter order)

    static let activities = ["walking", "jogging", "stairs", "sitting", "standing"]

    private static let axes = ["x", "y", "z"]
    private static let accStats = ["mean", "std", "min", "max", "rms", "median", "iqr", "skew", "kurt", "energy"]
    private static let gyroStats = ["mean", "std", "rms", "max"]
    static let burstStatNames = ["mean", "std", "range", "mad", "dom", "diff"]

    /// Ordered input names of ActivityClassifier / BurstClassifier (59 features).
    static var featureNames: [String] {
        var names: [String] = []
        for stat in accStats { for ax in axes { names.append("acc_\(ax)_\(stat)") } }
        names += ["mag_mean", "mag_std", "mag_max", "mag_min", "mag_p90", "mag_p10"]
        names += ["mag_dom_freq", "mag_e_low", "mag_e_mid", "mag_e_high", "mag_entropy"]
        for stat in gyroStats { for ax in axes { names.append("gyro_\(ax)_\(stat)") } }
        names += ["gyromag_mean", "gyromag_std", "gyromag_max"]
        names += ["acc_corr_xy", "acc_corr_xz", "acc_corr_yz"]
        return names
    }

    /// Ordered input names of EventDetector (30 features).
    static var eventFeatureNames: [String] {
        var names: [String] = []
        for prefix in ["cur", "prev", "z", "ratio", "spread"] {
            for stat in burstStatNames { names.append("\(prefix)_\(stat)") }
        }
        return names
    }

    // Absolute floors per burst statistic (see swn/features.py) so that a
    // perfectly still subject does not produce huge z-scores from jitter.
    static let burstFloors: [Double] = [0.80, 0.60, 1.20, 0.40, 0.35, 0.30]
    static let burstRefScale: [Double] = [1.0, 0.10, 0.10, 0.10, 0.10, 0.10]

    // MARK: - Elementary statistics (numpy-compatible)

    static func mean(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        return x.reduce(0, +) / Double(x.count)
    }

    /// Population standard deviation (numpy `std`, ddof = 0).
    static func std(_ x: [Double], mean mu: Double) -> Double {
        guard !x.isEmpty else { return 0 }
        let ss = x.reduce(0) { $0 + ($1 - mu) * ($1 - mu) }
        return (ss / Double(x.count)).squareRoot()
    }

    /// numpy percentile with the default 'linear' interpolation.
    /// `q` is a fraction in [0, 1].
    static func percentile(_ x: [Double], _ q: Double) -> Double {
        guard !x.isEmpty else { return 0 }
        let s = x.sorted()
        let h = q * Double(s.count - 1)
        let lo = Int(h.rounded(.down))
        let hi = Int(h.rounded(.up))
        if lo == hi { return s[min(lo, s.count - 1)] }
        let frac = h - Double(lo)
        return s[lo] + (s[hi] - s[lo]) * frac
    }

    static func median(_ x: [Double]) -> Double {
        percentile(x, 0.5)
    }

    /// Pearson correlation (numpy corrcoef); zero variance -> 0 (numpy
    /// yields NaN, which the Python pipeline maps to 0 via nan_to_num).
    static func pearson(_ a: [Double], _ b: [Double]) -> Double {
        let n = Double(a.count)
        guard n > 1 else { return 0 }
        let ma = mean(a), mb = mean(b)
        let sa = std(a, mean: ma), sb = std(b, mean: mb)
        if sa < 1e-12 || sb < 1e-12 { return 0 }
        let cov = zip(a, b).map { ($0.0 - ma) * ($0.1 - mb) }.reduce(0, +) / n
        return cov / (sa * sb)
    }

    /// numpy hanning(n): 0.5 - 0.5 * cos(2*pi*k / (n-1)).
    static func hanning(_ n: Int) -> [Double] {
        guard n > 1 else { return [1.0] }
        return (0..<n).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n - 1)) }
    }

    /// One-sided magnitude spectrum via direct DFT (identical values to
    /// numpy.fft.rfft up to floating-point error; windows here are <= 60
    /// samples so O(n^2) is negligible).
    /// Returns (dominantFreq, eLow, eMid, eHigh, spectralEntropy), matching
    /// the Python `_spectral` (all zeros when the spectrum is flat-silent).
    static func spectral(_ mag: [Double], fs: Double) -> (dom: Double, eLow: Double, eMid: Double, eHigh: Double, entropy: Double) {
        let n = mag.count
        guard n >= 8 else { return (0, 0, 0, 0, 0) }
        let mu = mean(mag)
        let win = hanning(n)
        let x = (0..<n).map { (mag[$0] - mu) * win[$0] }

        let half = n / 2
        var spec = [Double](repeating: 0, count: half + 1)
        for k in 0...half {
            var re = 0.0, im = 0.0
            let step = -2 * Double.pi * Double(k) / Double(n)
            for j in 0..<n {
                let a = step * Double(j)
                re += x[j] * cos(a)
                im += x[j] * sin(a)
            }
            spec[k] = (re * re + im * im).squareRoot()
        }
        let total = spec.reduce(0, +)
        if total < 1e-9 { return (0, 0, 0, 0, 0) }
        let p = spec.map { $0 / total }

        var dom = 0.0
        if spec.count > 2 {
            var best = 1
            for k in 1..<spec.count where spec[k] > spec[best] { best = k }
            dom = Double(best) * fs / Double(n)
        }
        var eLow = 0.0, eMid = 0.0, eHigh = 0.0
        for k in 0...half {
            let f = Double(k) * fs / Double(n)
            if f >= 0.4 && f < 3.0 { eLow += p[k] }
            else if f >= 3.0 && f < 8.0 { eMid += p[k] }
            else if f >= 8.0 { eHigh += p[k] }
        }
        let entropy = -p.reduce(0) { $0 + $1 * log($1 + 1e-12) }
        return (dom, eLow, eMid, eHigh, entropy)
    }

    // MARK: - Window features (ActivityClassifier / BurstClassifier)

    /// 59-feature vector for one IMU window, identical to
    /// `swn/features.py::window_features`.
    static func windowFeatures(acc: [SWNVector3], gyro: [SWNVector3], fs: Double) -> [Double] {
        var f: [Double] = []

        let ax = acc.map { $0.x }, ay = acc.map { $0.y }, az = acc.map { $0.z }
        for stat in accStats {
            for channel in [ax, ay, az] {
                f.append(accStat(channel, stat))
            }
        }

        let mag = acc.map { ($0.x * $0.x + $0.y * $0.y + $0.z * $0.z).squareRoot() }
        f += [mean(mag), std(mag, mean: mean(mag)), mag.max() ?? 0, mag.min() ?? 0,
              percentile(mag, 0.90), percentile(mag, 0.10)]

        let sp = spectral(mag, fs: fs)
        f += [sp.dom, sp.eLow, sp.eMid, sp.eHigh, sp.entropy]

        let gx = gyro.map { $0.x }, gy = gyro.map { $0.y }, gz = gyro.map { $0.z }
        for stat in gyroStats {
            for channel in [gx, gy, gz] {
                f.append(gyroStat(channel, stat))
            }
        }

        let gmag = gyro.map { ($0.x * $0.x + $0.y * $0.y + $0.z * $0.z).squareRoot() }
        f += [mean(gmag), std(gmag, mean: mean(gmag)), gmag.map(abs).max() ?? 0]

        f += [pearson(ax, ay), pearson(ax, az), pearson(ay, az)]

        return f
    }

    private static func accStat(_ x: [Double], _ stat: String) -> Double {
        let n = x.count
        guard n > 0 else { return 0 }
        let mu = mean(x)
        switch stat {
        case "mean": return mu
        case "std": return std(x, mean: mu)
        case "min": return x.min() ?? 0
        case "max": return x.max() ?? 0
        case "rms": return (x.reduce(0) { $0 + $1 * $1 } / Double(n)).squareRoot()
        case "median": return median(x)
        case "iqr": return percentile(x, 0.75) - percentile(x, 0.25)
        case "skew":
            let s = std(x, mean: mu)
            let m3 = x.reduce(0) { $0 + pow($1 - mu, 3) } / Double(n)
            return m3 / (s * s * s + 1e-9)
        case "kurt":
            let s = std(x, mean: mu)
            let m4 = x.reduce(0) { $0 + pow($1 - mu, 4) } / Double(n)
            return m4 / (s * s * s * s + 1e-9)
        case "energy": return x.reduce(0) { $0 + $1 * $1 }
        default: return 0
        }
    }

    private static func gyroStat(_ x: [Double], _ stat: String) -> Double {
        let mu = mean(x)
        switch stat {
        case "mean": return mu
        case "std": return std(x, mean: mu)
        case "rms": return (x.reduce(0) { $0 + $1 * $1 } / Double(x.count)).squareRoot()
        case "max": return x.map(abs).max() ?? 0
        default: return 0
        }
    }

    // MARK: - Burst statistics (EventDetector)

    /// 6-value compact descriptor of one 1 s @ 20 Hz acceleration burst,
    /// identical to `swn/features.py::burst_stats`.
    static func burstStats(acc: [SWNVector3], fs: Double) -> [Double] {
        let mag = acc.map { ($0.x * $0.x + $0.y * $0.y + $0.z * $0.z).squareRoot() }
        let n = mag.count
        guard n > 0 else { return [0, 0, 0, 0, 0, 0] }
        let mu = mean(mag)
        let dom = spectral(mag, fs: fs).dom
        let diff = n > 1 ? zip(mag, mag.dropFirst()).map { abs($0.1 - $0.0) }.reduce(0, +) / Double(n - 1) : 0
        let mad = mag.map { abs($0 - mu) }.reduce(0, +) / Double(n)
        return [mu, std(mag, mean: mu), (mag.max() ?? 0) - (mag.min() ?? 0), mad, dom, diff]
    }

    /// Change-detection features from the recent burst history
    /// (identical to `swn/features.py::burst_event_features`).
    /// `history` rows are burstStats vectors; the last row is the newest.
    static func burstEventFeatures(history: [[Double]]) -> [Double] {
        let statCount = burstStatNames.count
        let hist = history.filter { $0.count == statCount }
        guard !hist.isEmpty else { return [Double](repeating: 0, count: 5 * statCount) }

        let cur = hist[hist.count - 1]
        let prev = hist.count > 1 ? hist[hist.count - 2] : cur
        let refHist = hist.count > 1 ? Array(hist.dropLast()) : hist

        var ref = [Double](repeating: 0, count: statCount)
        var mad = [Double](repeating: 0, count: statCount)
        var z = [Double](repeating: 0, count: statCount)
        var ratio = [Double](repeating: 0, count: statCount)
        var spread = [Double](repeating: 0, count: statCount)

        for j in 0..<statCount {
            let col = refHist.map { $0[j] }
            let r = median(col)
            ref[j] = r
            mad[j] = 1.4826 * median(col.map { abs($0 - r) })
            z[j] = (cur[j] - r) / (mad[j] + burstFloors[j])
            ratio[j] = cur[j] / (r + burstRefScale[j])
            spread[j] = std(col, mean: mean(col))
        }
        return cur + prev + z + ratio + spread
    }

    /// Rule branch of the hybrid event detector: largest robust z-score on
    /// level/energy (stats 0, 1, 2) — identical to `swn/features.py::burst_z`.
    static func burstZ(history: [[Double]]) -> Double {
        let feat = burstEventFeatures(history: history)
        let n = burstStatNames.count
        return max(abs(feat[2 * n]), abs(feat[2 * n + 1]), abs(feat[2 * n + 2]))
    }

    // MARK: - Optional numeric self-test

    /// Deterministic 20-sample window; useful to verify against the Python
    /// pipeline when validating the port (see header comment).
    static func selfTestWindow() -> (acc: [SWNVector3], gyro: [SWNVector3], fs: Double) {
        var acc: [SWNVector3] = [], gyro: [SWNVector3] = []
        for i in 0..<20 {
            let t = Double(i) / 20.0
            acc.append(SIMD3(0.3 * sin(2 * .pi * 1.5 * t), 9.81 + 0.2 * cos(2 * .pi * 1.5 * t), 0.1 * sin(2 * .pi * 3.0 * t)))
            gyro.append(SIMD3(0.05 * sin(2 * .pi * 1.0 * t), 0.02, 0.04 * cos(2 * .pi * 1.0 * t)))
        }
        return (acc, gyro, 20.0)
    }
}
