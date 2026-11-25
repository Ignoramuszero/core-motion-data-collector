//
//  JournalView.swift
//  Core Motion Data Collector iOS
//
//  Daily life journal UI: a day-grouped list of timestamped notes +
//  photos, plus a "log this moment" sheet with camera / library capture.
//

import SwiftUI
import PhotosUI

// MARK: - Camera wrapper

struct CameraPicker: UIViewControllerRepresentable {
    let onImage: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    static var isCameraAvailable: Bool {
        UIImagePickerController.isSourceTypeAvailable(.camera)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker
        init(_ parent: CameraPicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.onImage(image)
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Journal list

struct JournalView: View {
    let store: LifeLogStore

    @State private var showingAddSheet = false

    var body: some View {
        Group {
            if store.entries.isEmpty {
                ContentUnavailableView(
                    "No entries yet",
                    systemImage: "book.closed",
                    description: Text("Tap '+ Log' to capture a moment of your day.")
                )
            } else {
                List {
                    ForEach(dayGroups, id: \.day) { group in
                        Section(Self.dayText(group.day)) {
                            ForEach(group.entries) { entry in
                                NavigationLink {
                                    EntryDetailView(store: store, entry: entry)
                                } label: {
                                    JournalRow(store: store, entry: entry)
                                }
                            }
                            .onDelete { offsets in
                                for index in offsets {
                                    store.deleteEntry(group.entries[index])
                                }
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Daily Log")
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                NavigationLink {
                    WeeklyReportView(store: store)
                } label: {
                    Label("Weekly Report", systemImage: "sparkles")
                }
            }
            ToolbarItemGroup(placement: .topBarTrailing) {
                if let json = store.exportJSON() {
                    ShareLink(item: json, subject: Text("Life Log")) {
                        Label("Export", systemImage: "square.and.arrow.up")
                    }
                }
                Button {
                    showingAddSheet = true
                } label: {
                    Label("Log", systemImage: "plus")
                }
            }
        }
        .sheet(isPresented: $showingAddSheet) {
            AddEntrySheet(store: store, motionRecordingActive: isMotionRecordingActive)
        }
    }

    // The journal does not know about the motion collector directly; the
    // sheet reads liveness from MainView, so entries opened from here just
    // record the flag as false unless the caller says otherwise.
    private var isMotionRecordingActive: Bool { false }

    // MARK: grouping

    private var dayGroups: [(day: Date, entries: [LifeLogEntry])] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: store.entries) { calendar.startOfDay(for: $0.timestamp) }
        return grouped
            .map { (day: $0.key, entries: $0.value.sorted { $0.timestamp > $1.timestamp }) }
            .sorted { $0.day > $1.day }
    }

    private static func dayText(_ date: Date) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(date) { return "Today" }
        if calendar.isDateInYesterday(date) { return "Yesterday" }
        let formatter = DateFormatter()
        formatter.dateStyle = .full
        return formatter.string(from: date)
    }
}

// MARK: - Row

private struct JournalRow: View {
    let store: LifeLogStore
    let entry: LifeLogEntry

    var body: some View {
        HStack(spacing: 12) {
            if let data = store.photoData(for: entry),
               let image = UIImage(data: data) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 54, height: 54)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                Image(systemName: "text.bubble")
                    .font(.title2)
                    .foregroundStyle(.secondary)
                    .frame(width: 54, height: 54)
                    .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 8))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(Self.timeText(entry.timestamp))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(entry.note.isEmpty ? "(no note)" : entry.note)
                    .font(.body)
                    .lineLimit(2)
                if entry.motionRecordingActive {
                    Label("REC", systemImage: "record.circle")
                        .font(.caption2)
                        .foregroundStyle(.red)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private static func timeText(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Detail

struct EntryDetailView: View {
    let store: LifeLogStore
    let entry: LifeLogEntry

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let data = store.photoData(for: entry),
                   let image = UIImage(data: data) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                Text(entry.timestamp, format: .dateTime.year().month().day().hour().minute().second())
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if entry.motionRecordingActive {
                    Label("Captured while motion recording was active - this moment can be aligned with the sensor CSV by timestamp.",
                          systemImage: "record.circle")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                if !entry.note.isEmpty {
                    Text(entry.note)
                        .font(.body)
                }
            }
            .padding()
        }
        .navigationTitle("Entry")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Add sheet

struct AddEntrySheet: View {
    let store: LifeLogStore
    let motionRecordingActive: Bool

    @Environment(\.dismiss) private var dismiss
    @State private var note = ""
    @State private var pendingImage: UIImage?
    @State private var showingCamera = false
    @State private var libraryItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                photoPreview

                HStack(spacing: 24) {
                    Button {
                        showingCamera = true
                    } label: {
                        Label("Camera", systemImage: "camera")
                    }
                    .disabled(!CameraPicker.isCameraAvailable)

                    PhotosPicker(selection: $libraryItem, matching: .images) {
                        Label("Library", systemImage: "photo.on.rectangle")
                    }

                    if pendingImage != nil {
                        Button(role: .destructive) {
                            pendingImage = nil
                        } label: {
                            Label("Remove", systemImage: "trash")
                        }
                    }
                }
                .font(.callout)

                TextEditor(text: $note)
                    .frame(minHeight: 120)
                    .scrollContentBackground(.hidden)
                    .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(alignment: .topLeading) {
                        if note.isEmpty {
                            Text("What is happening right now?")
                                .foregroundStyle(.secondary)
                                .padding(.top, 8)
                                .padding(.leading, 5)
                                .allowsHitTesting(false)
                        }
                    }
                    .padding(.horizontal, 4)

                if motionRecordingActive {
                    Label("Motion recording is active. This entry will be time-aligned with the sensor CSV.",
                          systemImage: "record.circle")
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Log this moment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        store.addEntry(
                            note: note.trimmingCharacters(in: .whitespacesAndNewlines),
                            photoJPEGData: pendingImage?.jpegData(compressionQuality: 0.85),
                            motionRecordingActive: motionRecordingActive
                        )
                        dismiss()
                    }
                }
            }
            .fullScreenCover(isPresented: $showingCamera) {
                CameraPicker { image in
                    pendingImage = image
                }
                .ignoresSafeArea()
            }
            .onChange(of: libraryItem) { _, newValue in
                guard let item = newValue else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        pendingImage = image
                    }
                    libraryItem = nil
                }
            }
        }
    }

    @ViewBuilder
    private var photoPreview: some View {
        if let image = pendingImage {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 260)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        } else {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 44))
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity)
                .frame(height: 120)
                .background(.fill.quaternary, in: RoundedRectangle(cornerRadius: 12))
        }
    }
}
