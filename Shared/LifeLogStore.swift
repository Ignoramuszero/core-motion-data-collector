//
//  LifeLogStore.swift
//  Core Motion Data Collector
//
//  A tiny local "daily life journal": timestamped notes + photos,
//  persisted inside the app's Documents directory as
//  LifeLog/entries.json + LifeLog/Photos/*.jpg
//
//  Pure Foundation (no UIKit) so it compiles on both iOS and watchOS.
//  Timestamps use the wall clock, which matches the timestamps printed
//  by MotionDataCollector — so journal entries can be aligned with a
//  recorded sensor CSV after export.
//

import Foundation

struct LifeLogEntry: Codable, Identifiable, Equatable {
    let id: UUID
    let timestamp: Date
    var note: String
    var photoFilename: String?
    /// True if a motion-recording session was running when the entry was created.
    var motionRecordingActive: Bool
}

@Observable final class LifeLogStore {
    private static let photoFileExtension = "jpg"

    private(set) var entries: [LifeLogEntry] = []

    private let fileManager = FileManager.default

    private var rootDirectoryURL: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LifeLog", isDirectory: true)
    }

    private var photosDirectoryURL: URL {
        rootDirectoryURL.appendingPathComponent("Photos", isDirectory: true)
    }

    private var archiveURL: URL {
        rootDirectoryURL.appendingPathComponent("entries.json")
    }

    init() {
        try? fileManager.createDirectory(at: photosDirectoryURL, withIntermediateDirectories: true)
        loadEntries()
    }

    // MARK: - Access

    func photoURL(for entry: LifeLogEntry) -> URL? {
        guard let name = entry.photoFilename else { return nil }
        return photosDirectoryURL.appendingPathComponent(name)
    }

    func photoData(for entry: LifeLogEntry) -> Data? {
        guard let url = photoURL(for: entry) else { return nil }
        return try? Data(contentsOf: url)
    }

    /// Newest entries first.
    var sortedEntries: [LifeLogEntry] {
        entries.sorted { $0.timestamp > $1.timestamp }
    }

    // MARK: - Mutation

    @discardableResult
    func addEntry(note: String, photoJPEGData: Data? = nil, motionRecordingActive: Bool = false) -> LifeLogEntry {
        var photoFilename: String?
        if let data = photoJPEGData, !data.isEmpty {
            let name = "\(UUID().uuidString).\(Self.photoFileExtension)"
            let url = photosDirectoryURL.appendingPathComponent(name)
            do {
                try data.write(to: url, options: .atomic)
                photoFilename = name
            } catch {
                logger.error("LifeLog: failed to save photo - \(String(describing: error))")
            }
        }

        let entry = LifeLogEntry(
            id: UUID(),
            timestamp: Date(),
            note: note,
            photoFilename: photoFilename,
            motionRecordingActive: motionRecordingActive
        )
        entries.append(entry)
        saveEntries()
        return entry
    }

    func deleteEntry(_ entry: LifeLogEntry) {
        if let name = entry.photoFilename {
            try? fileManager.removeItem(at: photosDirectoryURL.appendingPathComponent(name))
        }
        entries.removeAll { $0.id == entry.id }
        saveEntries()
    }

    // MARK: - Persistence

    private func loadEntries() {
        guard let data = try? Data(contentsOf: archiveURL) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        entries = (try? decoder.decode([LifeLogEntry].self, from: data)) ?? []
        logger.info("LifeLog: loaded \(self.entries.count) entries")
    }

    private func saveEntries() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(entries) else { return }
        try? data.write(to: archiveURL, options: .atomic)
    }

    // MARK: - Export

    /// Human-readable export that pairs every entry with its unix timestamp,
    /// suitable for aligning journal moments against a recorded motion CSV.
    func exportJSON() -> String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .custom { date, enc in
            var c = enc.singleValueContainer()
            try c.encode(date.timeIntervalSince1970)
        }
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(sortedEntries) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
