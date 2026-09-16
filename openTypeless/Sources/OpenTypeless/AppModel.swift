// SPDX-License-Identifier: Apache-2.0
import AppKit
import AVFoundation
import Combine
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    enum Phase: String { case idle, starting, recording, processing, preparing }
    let store: AppStore
    let recorder = AudioRecorder()
    let worker = WorkerClient()
    private let shortcut = GlobalShortcut()
    @Published var phase: Phase = .idle
    @Published var mode: VoiceMode = .dictate
    @Published var resultText = ""
    @Published var rawText = ""
    @Published var notice = ""
    @Published var error = ""
    @Published var lastApp = ""
    @Published var microphoneAllowed = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
    @Published var accessibilityAllowed = false
    var showMainWindow: (() -> Void)?
    var showVoicePanel: (() -> Void)?
    var hideVoicePanel: (() -> Void)?
    private var target: InsertionTarget?
    private var task: Task<Void, Never>?
    private var generation = UUID()
    private var timer: Timer?
    private var preferencesSubscription: AnyCancellable?
    private struct FailedRecording {
        let url: URL
        let duration: Double
        let mode: VoiceMode
        let target: InsertionTarget?
        let appName: String
    }
    private var retryRecording: FailedRecording?
    private var sessionPreferences = Preferences()

    init(store: AppStore? = nil) {
        let store = store ?? AppStore()
        self.store = store
        if store.preferences.pythonExecutable.isEmpty {
            store.preferences.pythonExecutable = Bundle.main.object(forInfoDictionaryKey: "OpenTypelessPython") as? String
                ?? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("Library/Application Support/OpenTypeless/runtime/bin/python").path
        }
        refreshPermissions()
        preferencesSubscription = store.$preferences.dropFirst().removeDuplicates().sink { [weak self] preferences in
            // Published emits before storage changes; use the supplied value for shortcut settings.
            Task { @MainActor in self?.configureShortcut(preferences) }
        }
        configureShortcut(store.preferences)
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if self.phase == .recording && self.recorder.elapsed >= 300 { self.finish() }
                self.refreshPermissions()
                self.store.prune()
            }
        }
    }

    var isBusy: Bool { phase != .idle }
    var shortcutLabel: String {
        let p = store.preferences
        var label = ""
        if p.shortcutModifiers & CGEventFlags.maskControl.rawValue != 0 { label += "⌃" }
        if p.shortcutModifiers & CGEventFlags.maskAlternate.rawValue != 0 { label += "⌥" }
        if p.shortcutModifiers & CGEventFlags.maskShift.rawValue != 0 { label += "⇧" }
        if p.shortcutModifiers & CGEventFlags.maskCommand.rawValue != 0 { label += "⌘" }
        return label + ([UInt16(49): "Space", 63: "Fn", 96: "F5", 97: "F6", 100: "F8", 101: "F9"][p.shortcutKeyCode] ?? "Key \(p.shortcutKeyCode)")
    }

    private func configureShortcut(_ preferences: Preferences) {
        shortcut.start(keyCode: preferences.shortcutKeyCode, modifiers: preferences.shortcutModifiers,
                       hold: preferences.holdToTalk,
                       onStart: { [weak self] in self?.toggle() },
                       onStop: { [weak self] in
                           if self?.phase == .recording { self?.finish() }
                           else if self?.phase == .starting { self?.cancel() }
                       },
                       onCancel: { [weak self] in if self?.isBusy == true { self?.cancel() } })
    }

    func refreshPermissions() {
        let previous = accessibilityAllowed
        accessibilityAllowed = TextInsertion.isTrusted
        microphoneAllowed = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        if accessibilityAllowed && !previous { configureShortcut(store.preferences) }
    }

    func requestMicrophone() {
        Task { _ = await AVCaptureDevice.requestAccess(for: .audio); refreshPermissions() }
    }

    func requestAccessibility() { TextInsertion.requestPermission(); refreshPermissions() }

    func toggle(_ requestedMode: VoiceMode? = nil) {
        if phase == .recording { finish(); return }
        guard phase == .idle else { return }
        if let requestedMode { mode = requestedMode }
        start()
    }

    private func start() {
        error = ""; notice = ""; target = nil
        sessionPreferences = store.preferences
        do {
            let captured = try TextInsertion.capture()
            if captured.bundleID != Bundle.main.bundleIdentifier { target = captured }
        } catch TextInsertionError.secureField {
            error = "Password and secure fields cannot be used for voice typing."
            showMainWindow?()
            return
        } catch {
            // Unsupported controls still allow a copyable transcript.
            notice = "The current field is not accessible. Your result will be ready to copy."
        }
        if mode == .edit && (target?.selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true) {
            error = "Select the text you want to change in another app, then use the shortcut. Accessibility access is required."
            showMainWindow?()
            return
        }
        lastApp = target?.applicationName ?? "OpenTypeless"
        do { _ = try payload(audio: nil) }
        catch { self.error = error.localizedDescription; showMainWindow?(); return }
        phase = .starting
        let token = UUID(); generation = token
        task = Task {
            do {
                try await recorder.start(deviceUID: sessionPreferences.microphoneUID)
                guard generation == token, !Task.isCancelled else { recorder.cancel(); return }
                discardRetryRecording()
                phase = .recording; showVoicePanel?()
                if sessionPreferences.sounds { NSSound(named: "Tink")?.play() }
            } catch {
                guard generation == token else { return }
                target = nil; phase = .idle; self.error = error.localizedDescription; refreshPermissions(); showMainWindow?()
            }
        }
    }

    func finish() {
        guard phase == .recording else { return }
        do {
            let duration = recorder.elapsed
            let audio = try recorder.stop()
            if sessionPreferences.sounds { NSSound(named: "Pop")?.play() }
            run(audio: audio, duration: duration, allowInsertion: true)
        } catch { target = nil; phase = .idle; self.error = error.localizedDescription; hideVoicePanel?(); showMainWindow?() }
    }

    private func payload(audio: URL?, text: String? = nil) throws -> [String: Any] {
        let preferences = sessionPreferences
        let rule = store.rules.first { $0.bundleID == target?.bundleID }
        let instructions = try Preferences.combinedInstructions(preferences.instructions, rule?.instructions ?? "")
        guard store.dictionary.count <= 200, store.dictionary.allSatisfy(\.isValid) else {
            throw AppError.message("Fix or delete invalid Dictionary entries before recording. Each phrase must contain 1–120 characters without NUL, with at most 200 entries.")
        }
        let selectedText = (mode == .edit || mode == .ask) ? (target?.selectedText ?? "") : ""
        guard selectedText.unicodeScalars.count <= 12_000, !selectedText.contains("\0") else {
            throw AppError.message("Select at most 12,000 characters without NUL before using Voice edit or Ask.")
        }
        var request: [String: Any] = [
            "op": audio == nil ? "process" : "transcribe",
            "asr_model": preferences.asrModel, "text_model": preferences.textModel,
            "mode": mode.rawValue, "language": preferences.language,
            "target_language": preferences.targetLanguage, "style": rule?.style ?? preferences.style,
            "instructions": instructions,
            "dictionary": store.dictionary.map { ["spoken": $0.spoken, "written": $0.written] },
            "selected_text": selectedText, "app_name": lastApp
        ]
        if let audio { request["audio_path"] = audio.path }
        if let text { request["text"] = text }
        return request
    }

    private func run(audio: URL, duration: Double, allowInsertion: Bool) {
        phase = .processing; error = ""; notice = ""
        let token = UUID(); generation = token
        let request = Result { try payload(audio: audio) }
        let requestMode = mode
        let capturedTarget = target
        let preferences = sessionPreferences
        let recording = FailedRecording(url: audio, duration: duration, mode: requestMode,
                                        target: capturedTarget, appName: lastApp)
        task = Task {
            do {
                let response = try await worker.request(request.get(), python: preferences.pythonExecutable)
                guard generation == token, !Task.isCancelled else { try? FileManager.default.removeItem(at: audio); return }
                let text = response["text"] as? String ?? ""
                let raw = response["raw_text"] as? String ?? text
                let warning = response["warning"] as? String ?? ""
                resultText = text; rawText = raw
                if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    notice = "No speech detected. Try speaking closer to the microphone."
                    try? FileManager.default.removeItem(at: audio)
                } else {
                    let entry = HistoryEntry(mode: requestMode, appName: lastApp, rawText: raw, text: text,
                                             duration: duration, warning: warning.isEmpty ? nil : warning)
                    if let retentionError = store.add(entry, recording: audio) {
                        retryRecording = recording
                        self.error = retentionError
                    }
                    notice = warning
                    if allowInsertion, preferences.autoPaste, requestMode != .ask, let capturedTarget {
                        do {
                            try await TextInsertion.insert(text, into: capturedTarget)
                            guard generation == token else { return }
                            if notice.isEmpty { notice = "Inserted into \(capturedTarget.applicationName)." }
                        } catch {
                            notice = "Ready to copy. \(error.localizedDescription)"
                            showMainWindow?()
                        }
                    } else {
                        if notice.isEmpty { notice = requestMode == .ask ? "Your answer is ready." : "Your text is ready to copy." }
                        showMainWindow?()
                    }
                }
                guard generation == token else { return }
                target = nil; phase = .idle; hideVoicePanel?()
                if !self.error.isEmpty { showMainWindow?() }
            } catch {
                guard generation == token else { try? FileManager.default.removeItem(at: audio); return }
                retryRecording = recording
                target = nil; phase = .idle; hideVoicePanel?()
                self.error = error.localizedDescription
                if let raw = (error as? WorkerFailure)?.rawText, !raw.isEmpty {
                    rawText = raw; resultText = raw
                    notice = "Text processing failed. The original transcript is ready to copy."
                }
                showMainWindow?()
            }
        }
    }

    var canRetry: Bool { retryRecording != nil && phase == .idle }
    func retryLast() {
        guard phase == .idle, let recording = retryRecording else { return }
        retryRecording = nil
        target = recording.target; mode = recording.mode; lastApp = recording.appName
        sessionPreferences = store.preferences
        run(audio: recording.url, duration: recording.duration, allowInsertion: false)
    }

    func retry(_ entry: HistoryEntry) {
        guard phase == .idle, let audio = store.audioURL(for: entry) else { return }
        if entry.mode == .edit || entry.mode == .ask {
            error = "Record a new request for this mode. Selected text is not retained in history."
            return
        }
        do {
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("OpenTypeless-\(UUID()).wav")
            try FileManager.default.copyItem(at: audio, to: copy)
            discardRetryRecording()
            target = nil; mode = entry.mode; lastApp = entry.appName; sessionPreferences = store.preferences
            run(audio: copy, duration: entry.duration, allowInsertion: false)
        } catch { self.error = error.localizedDescription }
    }

    func prepareModels() {
        guard phase == .idle else { return }
        phase = .preparing; error = ""; notice = ""
        let preferences = store.preferences
        let token = UUID(); generation = token
        task = Task {
            do {
                _ = try await worker.request(["op": "prepare", "asr_model": preferences.asrModel,
                                              "text_model": preferences.textModel], python: preferences.pythonExecutable)
                guard generation == token else { return }
                notice = "Local models are ready. You can start dictating."
            } catch {
                guard generation == token else { return }
                self.error = error.localizedDescription
            }
            phase = .idle
        }
    }

    func releaseModels() { if phase == .idle { worker.stop(); notice = "Models unloaded. They will load on your next dictation." } }

    func cancel() {
        generation = UUID(); task?.cancel(); task = nil
        recorder.cancel(); worker.stop(); target = nil; phase = .idle; hideVoicePanel?()
        notice = "Cancelled."
    }

    func shutdown() {
        cancel(); shortcut.stop(); timer?.invalidate()
        discardRetryRecording()
    }

    private func discardRetryRecording() {
        if let retryRecording { try? FileManager.default.removeItem(at: retryRecording.url) }
        retryRecording = nil
    }

    func copyResult() { TextInsertion.copy(resultText); notice = "Copied to clipboard." }
}
