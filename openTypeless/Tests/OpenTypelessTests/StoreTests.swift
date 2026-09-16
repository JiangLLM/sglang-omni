// SPDX-License-Identifier: Apache-2.0
import Testing
import Foundation
@testable import OpenTypeless

struct StoreTests {
    @Test func csvHandlesQuotedCommasNewlinesAndRejectsCorruption() throws {
        let parsed = try DictionaryCSV.parse("spoken,written\n\"a,b\",\"line 1\nline 2\"\n\"a\"\"b\",c\n")
        #expect(parsed == [["spoken", "written"], ["a,b", "line 1\nline 2"], ["a\"b", "c"]])
        #expect(throws: (any Error).self) { try DictionaryCSV.parse("\"unfinished") }
        #expect(throws: (any Error).self) { try DictionaryCSV.parse("\"quoted\"junk,word") }
    }

    @Test @MainActor func persistenceRetentionAndPrivacyDeleteAudio() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        store.preferences.keepAudio = true
        let audio = directory.appendingPathComponent("temp.wav")
        try Data([1, 2, 3]).write(to: audio)
        let entry = HistoryEntry(mode: .dictate, appName: "Test", rawText: "hello", text: "Hello.", duration: 1)
        store.add(entry, recording: audio)
        #expect(!FileManager.default.fileExists(atPath: audio.path))
        let saved = try #require(store.audioURL(for: store.history[0]))
        #expect(FileManager.default.fileExists(atPath: saved.path))
        let reloaded = AppStore(directory: directory)
        #expect(reloaded.history[0].text == "Hello.")
        reloaded.preferences.keepAudio = false
        #expect(!FileManager.default.fileExists(atPath: saved.path))
        #expect(reloaded.history[0].audioFile == nil)
        reloaded.history[0].date = Date(timeIntervalSinceNow: -86400 * 40)
        reloaded.prune()
        #expect(reloaded.history.isEmpty)
        reloaded.add(entry, recording: nil)
        reloaded.preferences.saveHistory = false
        #expect(reloaded.history.isEmpty)
    }

    @Test @MainActor func corruptLibraryIsNotOverwritten() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("library.json")
        let badData = Data("not json".utf8)
        try badData.write(to: file)
        let store = AppStore(directory: directory)
        #expect(!store.storageError.isEmpty)
        store.addWord(spoken: "hello", written: "Hello")
        #expect(try Data(contentsOf: file) == badData)
    }

    @Test @MainActor func dictionaryImportDeduplicatesAndIgnoresHeader() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        _ = try store.importWords("spoken,written\nS G Lang,SGLang\nCUDA\n")
        store.addWord(spoken: "s g lang", written: "SGLang", learned: true)
        #expect(store.dictionary.count == 2)
        #expect(store.dictionary[0].written == "SGLang")
        #expect(store.dictionary[1].written == "CUDA")
    }

    @Test func wordCountIncludesCJK() {
        #expect(HistoryEntry.countUnits("你好 Swift world") == 4)
        #expect(HistoryEntry.countUnits("") == 0)
    }

    @Test @MainActor func invalidDictionaryEditsDoNotReplaceSavedWords() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        #expect(store.addWord(spoken: "S G Lang", written: "SGLang", learned: true))
        let entry = try #require(store.dictionary.first)
        for invalid in ["", "   ", "a\0b", String(repeating: "e\u{301}", count: 61)] {
            #expect(!store.addWord(spoken: invalid, written: "SGLang", replacing: entry.id))
            #expect(store.dictionary == [entry])
        }
        #expect(store.addWord(spoken: "s g lang", written: "SGLang-Omni", replacing: entry.id))
        #expect(store.dictionary[0].id == entry.id)
        #expect(store.dictionary[0].learned)
        #expect(AppStore(directory: directory).dictionary[0].written == "SGLang-Omni")
        #expect(try store.importWords("bad,\"\"\ninvalid,a\0b") == 1)
    }

    @Test func writingInstructionsHonorUnicodeAndCombinedLimit() throws {
        let thousand = String(repeating: "a", count: 1000)
        let accepted = try Preferences.combinedInstructions(thousand, String(repeating: "b", count: 999))
        #expect(accepted.unicodeScalars.count == 2000)
        for (defaults, app) in [(thousand, thousand), (thousand + "a", ""), ("x\0y", ""), (String(repeating: "e\u{301}", count: 501), "")] {
            #expect(throws: (any Error).self) { try Preferences.combinedInstructions(defaults, app) }
        }
    }

    @Test @MainActor func failedAudioRetentionPreservesRecordingAndWarning() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        store.preferences.keepAudio = true
        try Data().write(to: store.audioDirectory) // A regular file prevents directory creation.
        let audio = directory.appendingPathComponent("recording.wav")
        try Data([1, 2, 3]).write(to: audio)
        let entry = HistoryEntry(mode: .dictate, appName: "Test", rawText: "hello", text: "Hello.", duration: 1)
        let warning = try #require(store.add(entry, recording: audio))
        #expect(FileManager.default.fileExists(atPath: audio.path))
        #expect(store.storageError.contains(warning))
        #expect(store.history[0].warning == warning)
        #expect(store.history[0].audioFile == nil)
        #expect(AppStore(directory: directory).history[0].warning == warning)
    }

    @Test @MainActor func retryUsesUpdatedRuntimeAndOriginalMode() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.keepAudio = true
        let originalPython = directory.appendingPathComponent("missing-original-python").path
        let correctedPython = directory.appendingPathComponent("missing-corrected-python").path
        store.preferences.pythonExecutable = originalPython
        let audio = directory.appendingPathComponent("recording.wav")
        try Data([1, 2, 3]).write(to: audio)
        store.add(HistoryEntry(mode: .dictate, appName: "Original app", rawText: "hello", text: "Hello.", duration: 42), recording: audio)
        let model = AppModel(store: store)
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.retry(try #require(store.history.first))
        for _ in 0..<100 where model.isBusy { try await Task.sleep(nanoseconds: 10_000_000) }
        try #require(model.canRetry)
        #expect(model.error.contains(originalPython))
        store.preferences.pythonExecutable = correctedPython
        model.mode = .ask
        model.retryLast()
        for _ in 0..<100 where model.isBusy { try await Task.sleep(nanoseconds: 10_000_000) }
        #expect(model.canRetry)
        #expect(model.error.contains(correctedPython))
        #expect(model.mode == .dictate)
        #expect(model.lastApp == "Original app")
    }
}
