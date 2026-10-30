package com.orion.companion

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.audiofx.AcousticEchoCanceler
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.util.ArrayDeque
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.random.Random

/** User-started continuous listening, Gemini transcription and phone actions without a PC. */
class PhoneListeningService : Service() {
    companion object {
        private const val START = "com.orion.companion.PHONE_LISTEN_START"
        private const val STOP = "com.orion.companion.PHONE_LISTEN_STOP"
        private const val CHANNEL = "orion_phone_listening"
        private const val PREFS = "orion_phone_listening"
        private const val NOTIFICATION_ID = 407
        private val events = ArrayDeque<String>()
        @Volatile private var instance: PhoneListeningService? = null
        @Volatile var running = false
            private set
        @Volatile var captureActive = false
            private set
        @Volatile var echoCancellation = false
            private set
        @Volatile var level = 0.0
            private set
        @Volatile var status = "{\"state\":\"off\",\"text\":\"Phone listening is off\",\"level\":0}"
            private set
        fun drainEvents(): List<String> = synchronized(events) { events.toList().also { events.clear() } }
        fun requested(context: Context): Boolean =
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("enabled", false)
        fun requiresWake(context: Context): Boolean =
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("require_wake", false)
        fun setRequiresWake(context: Context, required: Boolean) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean("require_wake", required).apply()
        }
        /** Call from the visible phone Activity after microphone/notification permission checks. */
        fun start(context: Context) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean("enabled", true).apply()
            try {
                ContextCompat.startForegroundService(context,
                    Intent(context, PhoneListeningService::class.java).setAction(START))
            } catch (failure: RuntimeException) {
                context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                    .putBoolean("enabled", false).commit()
                throw failure
            }
        }
        fun stop(context: Context) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean("enabled", false).apply()
            context.stopService(Intent(context, PhoneListeningService::class.java))
        }
        /** Called on the Activity thread; the microphone stays owned by the existing service. */
        fun updateAiConfiguration(change: () -> Unit) {
            val service = instance
            service?.pauseForConfigurationChange()
            try { change() } finally { service?.resumeAfterConfigurationChange() }
        }
    }

    private val main = Handler(Looper.getMainLooper())
    private val captureWorker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-phone-listening").apply { isDaemon = true }
    }
    private val requestWorker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-phone-wake-reply").apply { isDaemon = true }
    }
    private val interruptWorker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-phone-stop-recognition").apply { isDaemon = true }
    }
    private val processing = AtomicBoolean()
    private val interruptionProcessing = AtomicBoolean()
    private val acceptedTurn = AtomicLong()
    private val requestGate = Any()
    private val settingsRevision = AtomicLong()
    @Volatile private var configurationChanging = false
    @Volatile private var active = false
    @Volatile private var recorder: AudioRecord? = null
    @Volatile private var cooldownUntil = 0L
    @Volatile private var pendingWakeUntil = 0L
    @Volatile private var currentState = "off"
    private var failures = 0
    private var wakeLock: PowerManager.WakeLock? = null
    private lateinit var brain: PhoneBrain
    private lateinit var interruptBrain: PhoneBrain
    private lateinit var voice: PhoneVoice
    private val clipTimes = ArrayDeque<Long>()
    private val interruptionTimes = ArrayDeque<Long>()

    override fun onBind(intent: Intent?): IBinder? = null

    private fun stopAfterFatal(message: String) {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("enabled", false).commit()
        publish("error", message)
        stopSelf()
    }

    override fun onCreate() {
        super.onCreate()
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(
            NotificationChannel(CHANNEL, "ORION phone listening", NotificationManager.IMPORTANCE_LOW))
        brain = PhoneBrain(this)
        interruptBrain = PhoneBrain(this)
        voice = PhoneVoice(this, onSpeaking = { speaking ->
            if (active) {
                if (!speaking) {
                    // Bluetooth and loudspeaker tails can reach the microphone
                    // after AudioTrack reports its final frame. Do not turn an
                    // echo of ORION's reply into another command.
                    cooldownUntil = maxOf(cooldownUntil, SystemClock.elapsedRealtime() + 3_000L)
                    if (requiresWake(this) && pendingWakeUntil > 0L)
                        pendingWakeUntil = SystemClock.elapsedRealtime() + 13_000L
                }
                if (currentState !in setOf("error", "voice_error")) publish(if (speaking) "speaking" else "listening",
                    if (speaking) "Speaking. Say ‘stop’ or tap Stop speaking."
                    else if (requiresWake(this) && pendingWakeUntil > 0L) "Tell me your request."
                    else if (requiresWake(this)) "Say ‘ORION’ followed by your request."
                    else "Listening")
            }
        }, onStatus = { text ->
            if (active && currentState != "error" && text != "Charon")
                publish(if (text.startsWith("Voice unavailable") || text.startsWith("Voice connection lost"))
                    "voice_error" else "status", text)
        })
        instance = this
    }

    private fun pauseForConfigurationChange() {
        configurationChanging = true
        settingsRevision.incrementAndGet()
        brain.cancel()
        interruptBrain.cancel()
        voice.stop()
        pendingWakeUntil = 0L
    }

    private fun resumeAfterConfigurationChange() {
        failures = 0
        cooldownUntil = SystemClock.elapsedRealtime() + 1_000L
        configurationChanging = false
        if (active) publish("listening", if (requiresWake(this))
            "Say ‘ORION’ followed by your request." else "Listening")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == STOP) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("enabled", false).apply()
            stopSelf(); return START_NOT_STICKY
        }
        if (active) return START_NOT_STICKY
        if (intent?.action != START) {
            stopSelf(); return START_NOT_STICKY
        }
        if (!PhoneAiProfile.isPhoneMode(this)) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("enabled", false).commit()
            stopSelf(); return START_NOT_STICKY
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            stopAfterFatal("Allow ORION's microphone permission in Android settings.")
            return START_NOT_STICKY
        }
        if (PhoneAiProfile.key(this) == null) {
            stopAfterFatal("Set up the encrypted Gemini key in phone Options before listening.")
            return START_NOT_STICKY
        }
        val stopIntent = PendingIntent.getService(this, 0,
            Intent(this, PhoneListeningService::class.java).setAction(STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val openIntent = PendingIntent.getActivity(this, 1,
            Intent(this, PhoneConsoleActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val notification = Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_orion).setContentTitle("ORION is listening on this phone")
            .setContentText(if (requiresWake(this)) "Say ‘ORION’ followed by a request."
                else "Speak naturally. Tap Stop to end listening.")
            .setContentIntent(openIntent).setOngoing(true)
            .addAction(android.R.drawable.ic_media_pause, "Stop listening", stopIntent).build()
        try {
            if (Build.VERSION.SDK_INT >= 30) startForeground(NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            else startForeground(NOTIFICATION_ID, notification)
            wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ORION:PhoneListening").apply {
                    setReferenceCounted(false); acquire(10 * 60_000L)
                }
        } catch (_: RuntimeException) {
            stopAfterFatal("Android could not start background microphone access. Open ORION and start listening again.")
            return START_NOT_STICKY
        }
        active = true
        running = true
        publish("listening", if (requiresWake(this)) "Say ‘ORION’ followed by your request."
            else "Listening")
        captureWorker.execute(::capture)
        return START_NOT_STICKY
    }

    private fun capture() {
        if (!active) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            stopAfterFatal("The microphone permission was removed. Allow it in Android settings.")
            return
        }
        val minimum = AudioRecord.getMinBufferSize(16_000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (minimum <= 0) { stopAfterFatal("The phone microphone is unavailable."); return }
        val record = try {
            // Match the working tap-to-dictate path. VOICE_COMMUNICATION can
            // suppress quiet wake words on phones with aggressive echo control.
            AudioRecord(android.media.MediaRecorder.AudioSource.VOICE_RECOGNITION, 16_000,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minimum * 2, 4096))
        } catch (_: RuntimeException) {
            stopAfterFatal("The microphone could not open. Close other recording apps and try again.")
            return
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release(); stopAfterFatal("The phone microphone did not initialise."); return
        }
        recorder = record
        val echo = runCatching {
            if (AcousticEchoCanceler.isAvailable()) AcousticEchoCanceler.create(record.audioSessionId)
                ?.apply { enabled = true } else null
        }.getOrNull()
        echoCancellation = runCatching { echo?.enabled == true }.getOrDefault(false)
        try {
            if (!active) return
            record.startRecording()
            check(record.recordingState == AudioRecord.RECORDSTATE_RECORDING)
            captureActive = true
            val segmenter = PcmSpeechSegmenter()
            val frame = ByteArray(640) // 20 ms, 16 kHz mono PCM16.
            var wasSpeaking = false
            var previousMode = BackgroundCapturePolicy.Mode.PAUSED
            var capturedPlayback = SpeechPlaybackEpoch.current()
            var capturedRevision = settingsRevision.get()
            var renewed = SystemClock.elapsedRealtime()
            var lastLevel = renewed
            while (active) {
                val count = record.read(frame, 0, frame.size)
                check(count >= 0 && count % 2 == 0) { "Microphone capture stopped" }
                if (count == 0) continue
                val now = SystemClock.elapsedRealtime()
                if (now - renewed >= 5 * 60_000L) { wakeLock?.acquire(10 * 60_000L); renewed = now }
                val speaking = PhoneVoice.anySpeaking() || RemoteSpeech.anySpeaking()
                val playback = SpeechPlaybackEpoch.current()
                if (playback != capturedPlayback) { segmenter.reset(); capturedPlayback = playback }
                val revision = settingsRevision.get()
                if (revision != capturedRevision) { segmenter.reset(); capturedRevision = revision }
                if (speaking != wasSpeaking) {
                    segmenter.reset()
                    if (wasSpeaking && !speaking) cooldownUntil = maxOf(cooldownUntil, now + 3_000L)
                    wasSpeaking = speaking
                }
                val mode = BackgroundCapturePolicy.mode(processing.get(), speaking, echoCancellation,
                    configurationChanging, now < cooldownUntil, NavigationAudioState.captureBlocked(now))
                if (mode != previousMode) { segmenter.reset(); previousMode = mode }
                if (mode == BackgroundCapturePolicy.Mode.PAUSED) {
                    segmenter.reset(); level = 0.0; continue
                }
                val wasDiscarding = segmenter.discarding
                val pcm = segmenter.accept(frame, count)
                level = segmenter.level
                if (now - lastLevel >= 150L) { publish("level", "", level); lastLevel = now }
                if (segmenter.discarding && !wasDiscarding)
                    publish("status", "That request ran past thirty seconds. Pausing until you finish; please repeat a shorter request.")
                if (pcm == null) continue
                if (mode == BackgroundCapturePolicy.Mode.STOP_ONLY) {
                    while (interruptionTimes.isNotEmpty() && now - interruptionTimes.first() >= 60_000L)
                        interruptionTimes.removeFirst()
                    if (interruptionTimes.size < 5 && interruptionProcessing.compareAndSet(false, true)) {
                        interruptionTimes.addLast(now)
                        val turn = acceptedTurn.get()
                        val clipPlayback = capturedPlayback
                        interruptWorker.execute { recogniseStop(pcm, revision, turn, clipPlayback) }
                    }
                    continue
                }
                while (clipTimes.isNotEmpty() && now - clipTimes.first() >= 60_000L) clipTimes.removeFirst()
                if (clipTimes.size >= 30) {
                    cooldownUntil = now + 10_000L
                    publish("status", "Too many speech clips. Pausing transcription for ten seconds.")
                    continue
                }
                val accepted = synchronized(requestGate) {
                    if (settingsRevision.get() == revision && !configurationChanging &&
                        processing.compareAndSet(false, true)) {
                        acceptedTurn.incrementAndGet()
                        true
                    } else false
                }
                if (accepted) {
                    clipTimes.addLast(now)
                    requestWorker.execute { process(pcm, speaking, revision) }
                }
            }
        } catch (failure: Exception) {
            if (active) stopAfterFatal("Microphone capture stopped. Check permission and start listening again.")
        } finally {
            runCatching { record.stop() }; record.release()
            runCatching { echo?.release() }
            echoCancellation = false; captureActive = false; level = 0.0
            recorder = null
            if (active) stopSelf()
        }
    }

    /** A separate recogniser cannot replace or cancel the main reply's HTTP call. */
    private fun recogniseStop(pcm: ByteArray, revision: Long, turn: Long, playback: Long) {
        fun current() = active && !configurationChanging && settingsRevision.get() == revision &&
            acceptedTurn.get() == turn && SpeechPlaybackEpoch.isCurrent(playback) &&
            !NavigationAudioState.captureBlocked(SystemClock.elapsedRealtime())
        try {
            if (!current()) return
            val transcript = interruptBrain.transcribe(PhoneWakeWords.wav(pcm)).trim()
            if (!current()) return
            val stop = if (requiresWake(this)) PhoneWakeWords.isStop(transcript)
                else PhoneWakeWords.isConversationStop(transcript)
            if (stop) {
                synchronized(requestGate) {
                    if (!current()) return
                    // Invalidate streamed deltas and pending actions before stopping their output.
                    settingsRevision.incrementAndGet()
                    brain.cancel()
                    PhoneVoice.stopAll(playback); RemoteSpeech.stopAll(playback)
                    cooldownUntil = maxOf(cooldownUntil, SystemClock.elapsedRealtime() + 3_000L)
                    publish("listening", "Speech stopped. Listening")
                }
            }
        } catch (_: Exception) {
            // An interruption check must not replace the running request's result or error.
        } finally { interruptionProcessing.set(false) }
    }

    private fun process(pcm: ByteArray, interruptOnly: Boolean, revision: Long) {
        fun isCurrent() = active && !configurationChanging && settingsRevision.get() == revision
        var spokenTurn: PhoneVoice.IncrementalTurn? = null
        var replying = false
        try {
            if (!isCurrent()) return
            if (!interruptOnly) PhonePerformance.route(this, "Background")
            if (!interruptOnly) voice.prepare()
            publish("transcribing", "Turning your speech into words…")
            val transcriptionStarted = PhonePerformance.now()
            val transcript = try { brain.transcribe(PhoneWakeWords.wav(pcm)).trim() }
                finally { if (!interruptOnly) PhonePerformance.record(this, "transcription", transcriptionStarted) }
            if (!isCurrent()) return
            if (transcript.isEmpty()) {
                if (!interruptOnly) publish("listening", "I did not hear any words. Please try again.")
                return
            }
            if (if (requiresWake(this)) PhoneWakeWords.isStop(transcript)
                else PhoneWakeWords.isConversationStop(transcript)) {
                PhoneVoice.stopAll(); RemoteSpeech.stopAll()
                publish("listening", "Speech stopped. Listening")
                return
            }
            if (interruptOnly) {
                // The echo canceller may still pass ORION's voice. Ignore any
                // clip that was captured during playback unless it said stop.
                return // Echo-cancelled interruptions can only stop voice, never run actions.
            }
            publish("transcript", transcript)
            val wakeRequired = requiresWake(this)
            val prefixed = PhoneWakeWords.request(transcript)
            val request = if (wakeRequired)
                prefixed ?: if (SystemClock.elapsedRealtime() < pendingWakeUntil) transcript else null
                else PhoneWakeWords.conversationRequest(transcript)
            if (request == null) {
                publish("wake_required", "I heard you. Start the request with ‘ORION’, or say ‘ORION’ and give the request after the prompt.")
                return
            }
            pendingWakeUntil = 0L
            if (PhoneWakeWords.stopListening(request)) {
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("enabled", false).apply()
                publish("off", "Phone listening is off.")
                main.post { stopSelf() }
                return
            }
            if (request.isEmpty()) {
                if (wakeRequired) pendingWakeUntil = SystemClock.elapsedRealtime() + 10_000L
                val greeting = "I'm listening. What would you like to do on your phone?"
                PhoneConversationStore.append(this, "user", transcript)
                PhoneConversationStore.append(this, "model", greeting)
                failures = 0
                publish("wake_ready", if (wakeRequired) "Tell me your request within ten seconds."
                    else "Listening")
                cooldownUntil = maxOf(cooldownUntil, SystemClock.elapsedRealtime() + 1_000L)
                publish("reply", greeting); voice.speak(greeting)
                return
            }
            PhoneConversationStore.append(this, "user", request)
            publish("thinking", "ORION is working on your phone request…")
            replying = true
            val streamSpeech = SpeechText.mayStreamAnswer(request)
            val speechDraft = SpeechText.Stream()
            val streamed = StringBuilder()
            val answer = brain.reply(request, onDelta = { delta ->
                streamed.append(delta)
                if (streamSpeech && isCurrent()) for (phrase in speechDraft.add(delta)) {
                    val turn = spokenTurn ?: voice.beginIncremental().also { spokenTurn = it }
                    turn.enqueue(phrase)
                }
            }, execute = { action, report ->
                if (isCurrent()) PhoneActionCoordinator.request(this, action, report)
                else report(JSONObject().put("ok", false).put("error", "Phone listening was stopped"))
            })
            if (!isCurrent()) return
            if (streamSpeech && spokenTurn != null) {
                if (answer.startsWith(streamed.toString()))
                    speechDraft.add(answer.substring(streamed.length)).forEach { spokenTurn?.enqueue(it) }
                speechDraft.finish().takeIf { it.isNotEmpty() }?.let { spokenTurn?.enqueue(it) }
                spokenTurn?.finish()
            }
            PhoneConversationStore.append(this, "model", answer)
            failures = 0
            publish("reply", answer)
            cooldownUntil = maxOf(cooldownUntil, SystemClock.elapsedRealtime() + 1_000L)
            if (spokenTurn == null) voice.speak(answer)
        } catch (failure: Exception) {
            spokenTurn?.cancel()
            if (isCurrent() && !interruptOnly) {
                failures = (failures + 1).coerceAtMost(6)
                val delay = PhoneWakeWords.retryDelay(failures)
                val providerPause = if (replying) brain.replyRetryDelayMillis()
                    else brain.transcriptionRetryDelayMillis()
                val pause = maxOf(providerPause, delay + Random.nextLong(0, delay / 4 + 1))
                cooldownUntil = SystemClock.elapsedRealtime() + pause
                publish("error", PhoneBrain.failureText(failure) + " " + if (providerPause > 0L)
                    "Listening will retry when the provider limit resets."
                    else "Listening resumes in ${delay / 1000} seconds; please repeat the request.")
                main.postDelayed({
                    if (isCurrent() && captureActive && currentState == "error" &&
                        SystemClock.elapsedRealtime() >= cooldownUntil) {
                        publish("recovered", if (requiresWake(this)) "Say ‘ORION’ followed by your request."
                            else "Listening")
                    }
                }, pause)
            }
        } finally {
            if (!isCurrent()) spokenTurn?.cancel()
            processing.set(false)
            if (isCurrent() && currentState in setOf("transcribing", "thinking") &&
                !PhoneVoice.anySpeaking() && SystemClock.elapsedRealtime() >= cooldownUntil)
                publish("listening", if (requiresWake(this)) "Say ‘ORION’ followed by your request."
                    else "Listening")
        }
    }

    private fun publish(state: String, text: String, meter: Double = level) {
        val event = JSONObject().put("state", state).put("text", text).put("level", meter).toString()
        if (state !in setOf("level", "transcript")) {
            currentState = state
            status = event
        }
        synchronized(events) {
            // Keep meaningful states whilst the app is closed; history is encrypted separately.
            if (state == "level" && events.size > 12) return@synchronized
            events.addLast(event)
            while (events.size > 64 || events.sumOf { it.length } > 2_097_152) events.removeFirst()
        }
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        active = false; running = false
        brain.cancel()
        interruptBrain.cancel()
        runCatching { recorder?.stop() }
        captureWorker.shutdownNow(); requestWorker.shutdownNow(); interruptWorker.shutdownNow()
        voice.close(); brain.close(); interruptBrain.close()
        wakeLock?.let { runCatching { if (it.isHeld) it.release() } }
        wakeLock = null
        level = 0.0
        // Keep a real failure visible after Android stops the service. A
        // generic "off" event would otherwise hide the cause in the app.
        if (currentState != "error") publish("off", "Phone listening is off. Tap the microphone to dictate.")
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }
}
