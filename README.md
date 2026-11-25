# Core Motion Data Collector

Core Motion Data Collector makes it easy to obtain Core Motion data from an iPhone or Apple Watch in an accessible CSV format, ideal for analyzing user activity or training ML models. The types of data to include in the CSV output as well as the frequency at which they're collected can be configured within both the iOS and watchOS apps (separately from one another).

**New, November 2026:** iPhone Duo optimization - the settings screen now displays next to the main screen when the device is unfolded. 

**New, September 2025:** iOS support! Core Motion Data Collector can now be run on both watchOS and iOS to collect motion data from either platform.

## Requirements

* Xcode 16.4+ (recommended, though the app might also be compatible with other recent versions) and a compatible iPhone
* iOS 15.6+ (theoretically - only tested on iOS 26 and 27)
* watchOS 10+ (theoretically - only tested on watchOS 11.5, 26, and 27)

## Limitations

* The app only prints Core Motion data to Xcode's debug window when run through Xcode, so data can't be collected when the app is run outside of Xcode; CSV export/saving is not supported.
* Magnetometer data is not included in the collected Core Motion data types.
* Collection settings in the iOS and watchOS apps are separate from each other, so preferences from the iOS app won't automatically be reflected in the watchOS app, and vice versa.

## Setup

1. Open this project in Xcode.
2. Ensure that a supported iPhone and/or Apple Watch is registered with Xcode so that Core Motion Data Collector can be run on it through Xcode.
3. Select the desired app (iOS or watchOS) in Xcode's top bar, then click the Run button in Xcode's left sidebar to install and run the app on your device (the first installation may take some time, especially with wireless iPhone connections and Apple Watch).
4. (Optional) In the app, tap the settings button to configure the types of motion data to include in the CSV output and/or the frequency of data collection.
5. On the app's main screen, tap the "Start" button (or double tap on Apple Watch Series 9 or newer running watchOS 10.1 or newer) to begin collecting motion data as soon as you press the button.
6. When finished, tap the "Stop" button (or double tap, if supported) to stop collecting motion data.
7. Copy the emitted data from Xcode's debug window into an empty text file, and save it as a CSV.


---

## Life Log & Camera (local fork)

This fork adds a simple daily-life journal to the iOS app:

- **"Log a moment" button** on the main screen: create a timestamped entry with an optional photo (camera or photo library) and a note.
- **Daily Log** (book icon, top-left): entries grouped by day, tap for full photo + note, swipe to delete.
- **Export**: the share icon in the Daily Log toolbar shares all entries as JSON (unix timestamps included, so moments can be aligned with a recorded motion CSV).
- Entries where motion recording was active are marked with a red `REC` badge.

Storage: everything is saved locally in the app sandbox under `Documents/LifeLog/` (`entries.json` + `Photos/*.jpg`). Nothing is uploaded anywhere.

New files: `Shared/LifeLogStore.swift` (model + persistence), `Core Motion Data Collector iOS/JournalView.swift` (UI), plus a `NSCameraUsageDescription` entry in the iOS target build settings. The watchOS target is unaffected (no camera there).

### AI Weekly Report

The journal toolbar has a sparkles icon: it opens **Weekly Report**, which sends the last 7 days of journal entries (notes + timestamps only, never photos) to an OpenAI-compatible LLM API and renders a Chinese report with a life summary and 3-5 practical health suggestions.

Configure the endpoint via the gear icon on that screen:
- **Base URL**: any OpenAI-style `chat/completions` endpoint (OpenAI / DeepSeek / Moonshot / local servers)
- **API Key**: stored in UserDefaults on the device only
- **Model**: e.g. `gpt-4o-mini`, `deepseek-chat`

The generated report can be copied or saved back into the journal as an entry. Files: `iOS/LLMService.swift`, `iOS/WeeklyReportView.swift`.

### CSV File Reader

The iOS main screen has a **Read CSV file** button. It opens the Files picker and reads a local CSV (UTF-8/UTF-16, comma-separated, optional quoted fields). The reader shows the file name, total data-row count, column count, and a preview of the first 100 rows with the header row. This is useful for reviewing motion CSV files copied out of Xcode's debug console. Nothing is uploaded; the file is read in place through a security-scoped URL. File: `iOS/CSVImportView.swift`.
